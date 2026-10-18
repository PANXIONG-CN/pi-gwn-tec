from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FIRST_TRANSITION_POLICY = "forecast_internal_only"
INACTIVE_TRANSITION_POLICY = "not_applicable"
PPINN_PROTOCOL_SCHEMA_VERSION = "cac_r1.ppinn_runtime.v1"
PPINN_COMPONENT_DIGEST = "325dcd2eac8faaaf740e9310d16166c3a73f93d5ac9c4638182e4764c7ce3573"

TRUE_TERM_FLAG_KEYS = (
    "use_nonneg",
    "use_rot",
    "use_roti",
    "use_diffusion",
    "use_chemistry",
    "use_night",
    "use_dynamic_rot_cap",
    "use_advection",
)
LEGACY_PPinn_ALIASES = ("nochem", "use_adv", "use_drivers")

_TERM_ORDER = (
    "nonneg",
    "rot",
    "roti",
    "diffusion",
    "chemistry",
    "night",
    "dynamic_rot_cap",
    "advection",
)
_FLAG_TO_TERM = dict(zip(TRUE_TERM_FLAG_KEYS, _TERM_ORDER))
_TERM_DEPENDENCIES = {
    "nonneg": ("finite_prediction", "dataset_normalizer"),
    "rot": ("finite_prediction", "forecast_step_seconds"),
    "roti": ("finite_prediction", "forecast_step_seconds"),
    "diffusion": ("finite_prediction", "sha256_verified_fixed_adjacency"),
    "chemistry": ("F107", "AE", "cosSZA"),
    "night": ("cosSZA",),
    "dynamic_rot_cap": ("Dst",),
    "advection": ("Kp", "grid_shape"),
}
_LOSS_TERMS = frozenset(_TERM_ORDER) - {"dynamic_rot_cap"}

_FROZEN_COMMON_CONSTANTS = {
    "dt_minutes": 120,
    "tec_ref": 50.0,
    "t_ref_sec": 7200.0,
    "roti_cap_scale": 0.7,
    "kappa_nd": 0.05,
    "alpha_nd": 0.01,
    "beta_nd": 0.01,
    "ae_factor": 0.001,
    "rot_cap_k": 0.01,
}
_FROZEN_DATASET_CONSTANTS = {
    "gim": {
        "normalizer_id": "gim_tec01_147p1_v1",
        "rot_cap_tecu_per_min": 0.5,
        "rot_cap_nd": 1.2,
        "roti_cap_nd": 0.84,
    },
    "china": {
        "normalizer_id": "china_tec01_300_v1",
        "rot_cap_tecu_per_min": 0.25,
        "rot_cap_nd": 0.6,
        "roti_cap_nd": 0.42,
    },
}
_P0_RECIPES = {
    "pi_static": {
        "use_nonneg": True,
        "use_rot": True,
        "use_roti": True,
        "use_diffusion": True,
        "use_chemistry": False,
        "use_night": False,
        "use_dynamic_rot_cap": False,
        "use_advection": False,
    },
    "pi_driver": {
        "use_nonneg": True,
        "use_rot": True,
        "use_roti": True,
        "use_diffusion": True,
        "use_chemistry": True,
        "use_night": True,
        "use_dynamic_rot_cap": True,
        "use_advection": False,
    },
}
_NON_PI_MODEL_KEYS = frozenset({"do", "de_raw", "de_zscore", "de_film"})


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataset_key(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"gim", "gimtec", "tec"}:
        return "gim"
    if normalized in {"china", "china_hires"} or normalized.startswith("china_hires"):
        return "china"
    raise ValueError(f"unsupported strict PPINN dataset {value!r}")


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field} must be an explicit boolean, got {value!r}")


def resolve_ppinn_term_config(
    config: Mapping[str, Any],
    *,
    strict_paper_run: bool = False,
    explicit_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, bool | str]:
    """Resolve true PPINN flags without conflating predictor and physics drivers.

    CAC-R1 strict runs must explicitly provide every true flag and the frozen
    first-transition policy.  Legacy aliases remain available only to old
    smoke/debug configurations.
    """

    values = dict(config)
    supplied = set(values) if explicit_keys is None else set(explicit_keys)
    legacy = sorted(set(LEGACY_PPinn_ALIASES) & supplied)
    if strict_paper_run and legacy:
        raise ValueError(f"legacy PPINN aliases are forbidden in strict paper runs: {', '.join(legacy)}")

    required = set(TRUE_TERM_FLAG_KEYS) | {"first_transition_policy"}
    if strict_paper_run:
        missing = sorted(required - supplied)
        if missing:
            raise ValueError(f"missing explicit PPINN fields for strict paper run: {', '.join(missing)}")

    # Non-strict legacy behavior: the old aggregate switch enabled chemistry,
    # night, and the dynamic cap; use_adv and nochem then refined that recipe.
    legacy_drivers = _as_bool(values.get("use_drivers", False), field="use_drivers")
    legacy_nochem = _as_bool(values.get("nochem", False), field="nochem")
    defaults = {
        "use_nonneg": True,
        "use_rot": True,
        "use_roti": True,
        "use_diffusion": True,
        "use_chemistry": legacy_drivers and not legacy_nochem,
        "use_night": legacy_drivers,
        "use_dynamic_rot_cap": legacy_drivers,
        "use_advection": _as_bool(values.get("use_adv", False), field="use_adv"),
    }
    resolved: dict[str, bool | str] = {}
    for key in TRUE_TERM_FLAG_KEYS:
        resolved[key] = _as_bool(values.get(key, defaults[key]), field=key)

    policy = str(values.get("first_transition_policy", FIRST_TRANSITION_POLICY)).strip()
    if policy == INACTIVE_TRANSITION_POLICY:
        active_flags = [key for key in TRUE_TERM_FLAG_KEYS if bool(resolved[key])]
        if active_flags:
            raise ValueError(
                "first_transition_policy='not_applicable' requires every PPINN term disabled; "
                f"active={active_flags}"
            )
    elif policy != FIRST_TRANSITION_POLICY:
        raise ValueError(
            "unsupported first_transition_policy: "
            f"{policy!r}; expected {FIRST_TRANSITION_POLICY!r} or "
            f"{INACTIVE_TRANSITION_POLICY!r}"
        )
    resolved["first_transition_policy"] = policy
    return resolved


def resolve_ppinn_profile(
    config: Mapping[str, Any],
    *,
    strict_paper_run: bool = False,
    explicit_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Resolve a complete P0 PPINN or explicit inactive/no-PPINN profile.

    ``not_applicable`` is a scientific state, not a fallback: it is accepted
    only for a declared non-PI model with ``use_pinn=false`` and all eight true
    flags disabled.  Active PI profiles must match one of the two frozen P0
    recipes exactly.
    """

    values = dict(config)
    supplied = set(values) if explicit_keys is None else set(explicit_keys)
    if strict_paper_run:
        missing_profile = sorted({"use_pinn", "model_key"} - supplied)
        if missing_profile:
            raise ValueError(
                "missing explicit PPINN profile fields for strict paper run: "
                + ", ".join(missing_profile)
            )
    resolved = resolve_ppinn_term_config(
        values,
        strict_paper_run=strict_paper_run,
        explicit_keys=supplied,
    )
    flags = {key: bool(resolved[key]) for key in TRUE_TERM_FLAG_KEYS}
    use_pinn = _as_bool(values.get("use_pinn", any(flags.values())), field="use_pinn")
    model_key = str(values.get("model_key", "")).strip().lower()
    policy = str(resolved["first_transition_policy"])
    active_term_ids = [_FLAG_TO_TERM[key] for key in TRUE_TERM_FLAG_KEYS if flags[key]]

    if policy == INACTIVE_TRANSITION_POLICY:
        if use_pinn or active_term_ids:
            raise ValueError(
                "not_applicable is legal only when use_pinn=false and every term is disabled"
            )
        if model_key not in _NON_PI_MODEL_KEYS:
            raise ValueError(
                f"inactive/no-PPINN profile requires a frozen non-PI model_key, got {model_key!r}"
            )
        status = "inactive_no_ppinn"
    else:
        if not use_pinn or not active_term_ids:
            raise ValueError(
                "forecast_internal_only requires use_pinn=true and an active PI recipe"
            )
        expected_recipe = _P0_RECIPES.get(model_key)
        if expected_recipe is None:
            raise ValueError(f"active PPINN profile requires pi_static or pi_driver, got {model_key!r}")
        mismatched = {
            key: (expected_recipe[key], flags[key])
            for key in TRUE_TERM_FLAG_KEYS
            if flags[key] != expected_recipe[key]
        }
        if mismatched:
            raise ValueError(f"frozen PPINN recipe mismatch for {model_key}: {mismatched}")
        status = "active_ppinn"

    digest_payload = {
        "schema_version": PPINN_PROTOCOL_SCHEMA_VERSION,
        "ppinn_component_digest": PPINN_COMPONENT_DIGEST,
        "status": status,
        "model_key": model_key,
        "use_pinn": use_pinn,
        "flags": flags,
        "first_transition_policy": policy,
        "active_term_ids": active_term_ids,
    }
    return {
        **digest_payload,
        "candidate_id": str(values.get("candidate_id", "")),
        "strict_paper_run": bool(strict_paper_run),
        "active_terms": [],
        "trainable_parameter_count": sum(
            1 for term_id in active_term_ids if term_id in _LOSS_TERMS
        ),
        "active_term_digest": _canonical_digest(digest_payload),
        "adjacency_sha256": None,
        "expected_adjacency_sha256": None,
        "transition_count_last_forward": None,
    }


def build_norm_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Build the symmetric normalized Laplacian ``I-D^-1/2 A D^-1/2``."""

    if not isinstance(adjacency, torch.Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape={tuple(adjacency.shape)}")
    if not torch.isfinite(adjacency).all().item():
        raise ValueError("adjacency contains nonfinite values")
    if (adjacency < 0).any().item():
        raise ValueError("adjacency contains negative values")
    degree = adjacency.sum(dim=1).clamp(min=1e-6)
    degree_m12 = torch.diag(torch.pow(degree, -0.5))
    identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
    return identity - degree_m12 @ adjacency @ degree_m12


class UncertaintyAggregator(nn.Module):
    """Fixed homoscedastic weights for the active loss terms."""

    def __init__(self, names: tuple[str, ...]):
        super().__init__()
        # Assign sequentially: ParameterDict sorts a plain dict in some torch
        # versions, while the protocol requires the declared term order.
        self.log_vars = nn.ParameterDict()
        for name in names:
            self.log_vars[name] = nn.Parameter(torch.zeros(1))

    def add(self, name: str, residual: torch.Tensor) -> torch.Tensor:
        if name not in self.log_vars:
            raise KeyError(f"PPINN term {name!r} has no registered uncertainty parameter")
        if residual.numel() == 0:
            raise ValueError(f"PPINN term {name!r} received an empty residual")
        if not torch.isfinite(residual).all().item():
            raise ValueError(f"PPINN term {name!r} residual is nonfinite")
        log_variance = self.log_vars[name]
        return (
            0.5 * torch.exp(-log_variance) * torch.mean(residual.square())
            + 0.5 * log_variance
        )


class PPinnLoss(nn.Module):
    """CAC-R1 PPINN loss with independent, auditable term flags.

    Temporal terms use only transitions between forecast steps.  In
    particular, no residual connects the final history value to forecast h=1.
    """

    def __init__(
        self,
        scaler_data,
        A: torch.Tensor | None,
        dataset: str | None = None,
        normalizer_id: str | None = None,
        model_key: str | None = None,
        dt_minutes: int = 120,
        tec_ref: float = 50.0,
        t_ref_sec: float = 7200.0,
        rot_cap_tecu_per_min: float = 0.5,
        roti_cap_scale: float = 0.7,
        kappa_nd: float = 0.05,
        roti_w: int = 3,
        use_nonneg: bool = True,
        use_rot: bool = True,
        use_roti: bool = True,
        use_diffusion: bool = True,
        use_chemistry: bool | None = None,
        use_night: bool | None = None,
        use_dynamic_rot_cap: bool | None = None,
        use_advection: bool | None = None,
        first_transition_policy: str = FIRST_TRANSITION_POLICY,
        alpha_nd: float = 0.01,
        beta_nd: float = 0.01,
        ae_factor: float = 0.001,
        rot_cap_k: float = 0.01,
        strict_paper_run: bool = False,
        adjacency_sha256: str | None = None,
        expected_adjacency_sha256: str | None = None,
        # Legacy aggregate aliases.  They are rejected in strict mode.
        use_drivers: bool | None = None,
        use_adv: bool | None = None,
        nochem: bool | None = None,
    ):
        super().__init__()
        self.scaler = scaler_data
        self.strict_paper_run = bool(strict_paper_run)

        direct = {
            "use_nonneg": use_nonneg,
            "use_rot": use_rot,
            "use_roti": use_roti,
            "use_diffusion": use_diffusion,
            "first_transition_policy": first_transition_policy,
        }
        explicit = set(direct)
        for key, value in (
            ("use_chemistry", use_chemistry),
            ("use_night", use_night),
            ("use_dynamic_rot_cap", use_dynamic_rot_cap),
            ("use_advection", use_advection),
        ):
            if value is not None:
                direct[key] = value
                explicit.add(key)
        for key, value in (("use_drivers", use_drivers), ("use_adv", use_adv), ("nochem", nochem)):
            if value is not None:
                direct[key] = value
                explicit.add(key)

        # Constructor defaults count as explicit true flags.  This makes the
        # production object usable directly while the config resolver still
        # enforces serialized explicitness before construction.
        for key, fallback in (
            ("use_chemistry", False),
            ("use_night", False),
            ("use_dynamic_rot_cap", False),
            ("use_advection", False),
        ):
            if key not in direct:
                direct[key] = fallback
                explicit.add(key)
        resolved = resolve_ppinn_term_config(
            direct,
            strict_paper_run=self.strict_paper_run,
            explicit_keys=explicit,
        )
        self.term_flags = {key: bool(resolved[key]) for key in TRUE_TERM_FLAG_KEYS}
        self.first_transition_policy = str(resolved["first_transition_policy"])
        for key, enabled in self.term_flags.items():
            setattr(self, key, enabled)
        # Backward-compatible read-only aggregate indicator; it is not a true
        # flag and is never serialized as one.
        self.use_drivers = any(
            self.term_flags[key]
            for key in ("use_chemistry", "use_night", "use_dynamic_rot_cap", "use_advection")
        )

        dataset_was_explicit = dataset is not None
        if dataset is None:
            dataset = "china" if float(rot_cap_tecu_per_min) == 0.25 else "gim"
        self.dataset = _dataset_key(dataset)
        expected_dataset_constants = _FROZEN_DATASET_CONSTANTS[self.dataset]
        self.normalizer_id = str(
            normalizer_id or expected_dataset_constants["normalizer_id"]
        )
        inferred_model_key = (
            "pi_driver"
            if any(
                self.term_flags[key]
                for key in ("use_chemistry", "use_night", "use_dynamic_rot_cap")
            )
            else "pi_static"
        )
        self.model_key = str(model_key or inferred_model_key).strip().lower()
        if self.strict_paper_run and dataset_was_explicit:
            if self.normalizer_id != expected_dataset_constants["normalizer_id"]:
                raise ValueError(
                    "frozen PPINN normalizer mismatch: "
                    f"dataset={self.dataset}, expected={expected_dataset_constants['normalizer_id']!r}, "
                    f"got={self.normalizer_id!r}"
                )
            if model_key is not None:
                resolve_ppinn_profile(
                    {
                        "use_pinn": True,
                        "model_key": self.model_key,
                        **self.term_flags,
                        "first_transition_policy": self.first_transition_policy,
                    },
                    strict_paper_run=True,
                    explicit_keys={"use_pinn", "model_key", *TRUE_TERM_FLAG_KEYS, "first_transition_policy"},
                )

        self.dt_sec = float(dt_minutes) * 60.0
        self.t_ref_sec = float(t_ref_sec)
        self.tec_ref = float(tec_ref)
        if self.dt_sec <= 0 or self.t_ref_sec <= 0 or self.tec_ref <= 0:
            raise ValueError("dt_minutes, t_ref_sec, and tec_ref must be positive")
        self.dt_nd = self.dt_sec / self.t_ref_sec
        self.rot_cap_base_tecu_per_min = float(rot_cap_tecu_per_min)
        self.roti_cap_scale = float(roti_cap_scale)
        self.kappa_nd = float(kappa_nd)
        self.alpha_nd = float(alpha_nd)
        self.beta_nd = float(beta_nd)
        self.ae_factor = float(ae_factor)
        self.rot_cap_k = float(rot_cap_k)
        self.roti_w = max(2, int(roti_w))
        fixed_values = (
            self.rot_cap_base_tecu_per_min,
            self.roti_cap_scale,
            self.kappa_nd,
            self.alpha_nd,
            self.beta_nd,
            self.ae_factor,
            self.rot_cap_k,
        )
        if not all(torch.isfinite(torch.tensor(value)).item() for value in fixed_values):
            raise ValueError("PPINN constants must be finite")

        cap_per_second = self.rot_cap_base_tecu_per_min / 60.0
        self.rot_cap_nd = (self.t_ref_sec / self.tec_ref) * cap_per_second
        self.roti_cap_nd = self.roti_cap_scale * self.rot_cap_nd

        self.constants = {
            "dt_minutes": int(dt_minutes),
            "dt_sec": self.dt_sec,
            "dt_nd": self.dt_nd,
            "tec_ref": self.tec_ref,
            "t_ref_sec": self.t_ref_sec,
            "rot_cap_tecu_per_min": self.rot_cap_base_tecu_per_min,
            "rot_cap_nd": self.rot_cap_nd,
            "roti_cap_scale": self.roti_cap_scale,
            "roti_cap_nd": self.roti_cap_nd,
            "kappa_nd": self.kappa_nd,
            "alpha_nd": self.alpha_nd,
            "beta_nd": self.beta_nd,
            "ae_factor": self.ae_factor,
            "rot_cap_k": self.rot_cap_k,
        }
        if self.strict_paper_run:
            expected = {
                **_FROZEN_COMMON_CONSTANTS,
                **expected_dataset_constants,
            }
            comparisons = {
                "dt_minutes": self.constants["dt_minutes"],
                "tec_ref": self.tec_ref,
                "t_ref_sec": self.t_ref_sec,
                "roti_cap_scale": self.roti_cap_scale,
                "kappa_nd": self.kappa_nd,
                "alpha_nd": self.alpha_nd,
                "beta_nd": self.beta_nd,
                "ae_factor": self.ae_factor,
                "rot_cap_k": self.rot_cap_k,
                "rot_cap_tecu_per_min": self.rot_cap_base_tecu_per_min,
                "rot_cap_nd": self.rot_cap_nd,
                "roti_cap_nd": self.roti_cap_nd,
            }
            mismatched_constants = {
                key: (expected[key], actual)
                for key, actual in comparisons.items()
                if not math.isclose(float(actual), float(expected[key]), rel_tol=0.0, abs_tol=1e-12)
            }
            if mismatched_constants:
                raise ValueError(f"frozen PPINN constants mismatch: {mismatched_constants}")

        actual_digest = str(adjacency_sha256 or "").strip().lower() or None
        expected_digest = str(expected_adjacency_sha256 or "").strip().lower() or None
        if actual_digest is not None and (len(actual_digest) != 64 or any(c not in "0123456789abcdef" for c in actual_digest)):
            raise ValueError("adjacency_sha256 must be a lowercase SHA-256 hex digest")
        if expected_digest is not None and actual_digest != expected_digest:
            raise ValueError(
                "diffusion adjacency digest mismatch: "
                f"actual={actual_digest!r}, expected={expected_digest!r}"
            )
        self.adjacency_sha256 = actual_digest
        self.expected_adjacency_sha256 = expected_digest

        if self.use_diffusion and A is None and self.strict_paper_run:
            raise ValueError("diffusion requires a fixed adjacency in strict paper runs")
        if self.use_diffusion and A is not None:
            laplacian = build_norm_laplacian(A)
        else:
            laplacian = torch.empty((0, 0), dtype=torch.float32)
            if A is None:
                self.use_diffusion = False
                self.term_flags["use_diffusion"] = False
        self.register_buffer("L", laplacian)

        self.active_term_ids = tuple(
            _FLAG_TO_TERM[key] for key in TRUE_TERM_FLAG_KEYS if self.term_flags[key]
        )
        uncertainty_names = tuple(
            term_id for term_id in self.active_term_ids if term_id in _LOSS_TERMS
        )
        self.uncert = UncertaintyAggregator(uncertainty_names)
        self._loss_contribution_counts = {term_id: 0 for term_id in self.active_term_ids}
        self._nonfinite_counts = {term_id: 0 for term_id in self.active_term_ids}
        self._transition_count_last_forward: int | None = None
        self._active_term_digest = self._compute_active_term_digest()

    @staticmethod
    def transition_count(horizon: int) -> int:
        return max(int(horizon) - 1, 0)

    @staticmethod
    def _as_BTN1(value: torch.Tensor, *, field: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field} must be a torch.Tensor")
        if value.ndim == 3:
            return value.unsqueeze(-1)
        if value.ndim == 4 and value.shape[-1] == 1:
            return value
        raise ValueError(f"{field} shape must be [B,T,N] or [B,T,N,1], got {tuple(value.shape)}")

    @staticmethod
    def _require_finite(value: torch.Tensor, *, field: str) -> None:
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{field} contains nonfinite values")

    def _to_nd(self, normalized: torch.Tensor) -> torch.Tensor:
        physical = self.scaler.inverse_transform(normalized)
        if not isinstance(physical, torch.Tensor):
            physical = torch.as_tensor(physical, device=normalized.device, dtype=normalized.dtype)
        self._require_finite(physical, field="inverse-transformed prediction")
        return physical / self.tec_ref

    def _rot_and_roti(self, y_nd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rot = (y_nd[:, 1:] - y_nd[:, :-1]) / self.dt_nd
        if rot.shape[1] >= self.roti_w:
            windows = rot.unfold(dimension=1, size=self.roti_w, step=1).contiguous()
            core = windows.std(dim=-1, unbiased=False)
            pad_left = (rot.shape[1] - core.shape[1]) // 2
            pad_right = rot.shape[1] - core.shape[1] - pad_left
            pieces = []
            if pad_left:
                pieces.append(core[:, :1].expand(-1, pad_left, -1, -1))
            pieces.append(core)
            if pad_right:
                pieces.append(core[:, -1:].expand(-1, pad_right, -1, -1))
            roti = torch.cat(pieces, dim=1)
        else:
            roti = torch.zeros_like(rot)
        return rot, roti

    @staticmethod
    def _normalize_indices(indices, *, device: torch.device, field: str) -> torch.Tensor | None:
        if indices is None:
            return None
        value = torch.as_tensor(indices, device=device, dtype=torch.long).reshape(-1)
        if value.numel() == 0:
            return None
        if (value < 0).any().item():
            raise ValueError(f"{field} contains negative node indices")
        return value

    @staticmethod
    def _select_nodes(value: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return value
        if value.ndim == 4:
            return torch.index_select(value, dim=2, index=indices)
        if value.ndim == 3:
            return torch.index_select(value, dim=2, index=indices)
        raise ValueError(f"cannot select nodes from shape={tuple(value.shape)}")

    def _driver_tensor(
        self,
        drivers: Mapping[str, Any] | None,
        field: str,
        *,
        batch: int,
        horizon: int,
        nodes: int,
        nodewise: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if drivers is None or field not in drivers or drivers[field] is None:
            raise ValueError(f"enabled PPINN term dependency missing: {field}")
        value = drivers[field]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        else:
            value = value.to(device=device, dtype=dtype)
        expected = (batch, horizon, nodes) if nodewise else (batch, horizon)
        if tuple(value.shape) != expected:
            raise ValueError(f"{field} driver shape mismatch: got {tuple(value.shape)}, expected {expected}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{field} driver contains nonfinite values")
        return value

    def _validate_driver_dependencies(
        self,
        drivers: Mapping[str, Any] | None,
        *,
        batch: int,
        horizon: int,
        nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}

        def require_global(field: str, term_id: str) -> None:
            try:
                validated[field] = self._driver_tensor(
                    drivers,
                    field,
                    batch=batch,
                    horizon=horizon,
                    nodes=nodes,
                    nodewise=False,
                    device=device,
                    dtype=dtype,
                )
            except ValueError as exc:
                raise ValueError(f"{term_id} dependency {exc}") from exc

        def require_nodewise(field: str, term_id: str) -> None:
            try:
                validated[field] = self._driver_tensor(
                    drivers,
                    field,
                    batch=batch,
                    horizon=horizon,
                    nodes=nodes,
                    nodewise=True,
                    device=device,
                    dtype=dtype,
                )
            except ValueError as exc:
                raise ValueError(f"{term_id} dependency {exc}") from exc

        if self.use_chemistry:
            require_global("F107", "chemistry")
            require_global("AE", "chemistry")
            require_nodewise("cosSZA", "chemistry")
        if self.use_night and "cosSZA" not in validated:
            require_nodewise("cosSZA", "night")
        if self.use_dynamic_rot_cap:
            require_global("Dst", "dynamic_rot_cap")
        if self.use_advection:
            require_global("Kp", "advection")
            if drivers is None or "grid_shape" not in drivers:
                raise ValueError("advection dependency missing: grid_shape")
            grid_shape = drivers["grid_shape"]
            if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
                raise ValueError(f"advection grid_shape is invalid: {grid_shape!r}")
            try:
                rows, columns = int(grid_shape[0]), int(grid_shape[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"advection grid_shape is invalid: {grid_shape!r}") from exc
            if rows <= 0 or columns <= 0 or rows * columns != nodes:
                raise ValueError(
                    f"advection grid_shape mismatch: got {(rows, columns)}, expected product {nodes}"
                )
            validated["grid_shape"] = (rows, columns)
        return validated

    def _record_contribution(self, term_id: str, residual: torch.Tensor) -> None:
        if not torch.isfinite(residual).all().item():
            self._nonfinite_counts[term_id] += 1
            raise ValueError(f"{term_id} residual contains nonfinite values")
        self._loss_contribution_counts[term_id] += 1

    def _aggregate(
        self,
        total: torch.Tensor,
        terms: dict[str, torch.Tensor],
        term_id: str,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        self._record_contribution(term_id, residual)
        total = total + self.uncert.add(term_id, residual)
        terms[term_id] = residual.square().mean()
        return total

    def _compute_active_term_digest(self) -> str:
        payload = {
            "schema_version": PPINN_PROTOCOL_SCHEMA_VERSION,
            "component_digest": PPINN_COMPONENT_DIGEST,
            "dataset": self.dataset,
            "normalizer_id": self.normalizer_id,
            "model_key": self.model_key,
            "first_transition_policy": self.first_transition_policy,
            "flags": self.term_flags,
            "active_term_ids": list(self.active_term_ids),
            "constants": self.constants,
            "adjacency_sha256": self.adjacency_sha256,
        }
        return _canonical_digest(payload)

    def protocol_payload(self) -> dict[str, Any]:
        active_terms = []
        for term_id in self.active_term_ids:
            parameter_ids = [f"uncert.log_vars.{term_id}"] if term_id in self.uncert.log_vars else []
            active_terms.append(
                {
                    "term_id": term_id,
                    "enabled": True,
                    "required_inputs": list(_TERM_DEPENDENCIES[term_id]),
                    "parameter_ids": parameter_ids,
                    "log_var_ids": parameter_ids,
                    "loss_contribution_count": self._loss_contribution_counts[term_id],
                    "nonfinite_count": self._nonfinite_counts[term_id],
                }
            )
        return {
            "schema_version": PPINN_PROTOCOL_SCHEMA_VERSION,
            "ppinn_component_digest": PPINN_COMPONENT_DIGEST,
            "status": "active_ppinn",
            "strict_paper_run": self.strict_paper_run,
            "dataset": self.dataset,
            "normalizer_id": self.normalizer_id,
            "model_key": self.model_key,
            "flags": dict(self.term_flags),
            "first_transition_policy": self.first_transition_policy,
            "active_term_ids": list(self.active_term_ids),
            "active_terms": active_terms,
            "active_term_digest": self._active_term_digest,
            "constants": dict(self.constants),
            "adjacency_sha256": self.adjacency_sha256,
            "expected_adjacency_sha256": self.expected_adjacency_sha256,
            "transition_count_last_forward": self._transition_count_last_forward,
        }

    def forward(
        self,
        y_pred01: torch.Tensor,
        y_true01: torch.Tensor | None = None,
        drivers: Mapping[str, Any] | None = None,
        *,
        node_keep_idx: torch.Tensor | None = None,
        node_holdout_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self._as_BTN1(y_pred01, field="prediction")
        self._require_finite(prediction, field="prediction")
        y_nd = self._to_nd(prediction)
        batch, horizon, nodes, _ = y_nd.shape
        transition_count = self.transition_count(horizon)
        self._transition_count_last_forward = transition_count

        if self.use_diffusion:
            if self.L.numel() == 0:
                raise ValueError("diffusion adjacency is unavailable")
            if tuple(self.L.shape) != (nodes, nodes):
                raise ValueError(
                    f"diffusion adjacency shape mismatch: got {tuple(self.L.shape)}, expected {(nodes, nodes)}"
                )

        validated = self._validate_driver_dependencies(
            drivers,
            batch=batch,
            horizon=horizon,
            nodes=nodes,
            device=y_nd.device,
            dtype=y_nd.dtype,
        )

        keep_idx = self._normalize_indices(
            node_keep_idx,
            device=y_nd.device,
            field="node_keep_idx",
        )
        if keep_idx is not None and (keep_idx >= nodes).any().item():
            raise ValueError("node_keep_idx is out of range")
        holdout_idx = self._normalize_indices(
            node_holdout_idx,
            device=y_nd.device,
            field="node_holdout_idx",
        )
        if holdout_idx is not None and (holdout_idx >= nodes).any().item():
            raise ValueError("node_holdout_idx is out of range")
        holdout_mask = None
        if holdout_idx is not None:
            holdout_mask = torch.zeros(nodes, device=y_nd.device, dtype=torch.bool)
            holdout_mask[holdout_idx] = True
            holdout_mask = holdout_mask.view(1, 1, nodes)

        def detach_holdout(value: torch.Tensor) -> torch.Tensor:
            if holdout_mask is None:
                return value
            return torch.where(holdout_mask, value.detach(), value)

        # A graph-connected zero keeps all-disabled and H=1 recipes safe for
        # backward() without inventing placeholder metrics.
        total = y_nd.sum() * 0.0
        terms: dict[str, torch.Tensor] = {}

        if self.use_nonneg:
            nonneg = F.relu(-self._select_nodes(y_nd, keep_idx))
            total = self._aggregate(total, terms, "nonneg", nonneg)

        if transition_count > 0:
            rot = roti = None
            if self.use_rot or self.use_roti:
                rot, roti = self._rot_and_roti(y_nd)

            if self.use_dynamic_rot_cap:
                dst = validated["Dst"]
                storm = torch.clamp(-dst, min=0.0)
                cap_tecu_per_min = self.rot_cap_base_tecu_per_min * (
                    1.0 + self.rot_cap_k * storm / 100.0
                )
                dynamic_rot_cap = (
                    self.t_ref_sec / self.tec_ref
                ) * (cap_tecu_per_min / 60.0)
                rot_cap = dynamic_rot_cap[:, 1:].view(batch, transition_count, 1, 1)
                roti_cap = self.roti_cap_scale * rot_cap
            else:
                rot_cap = self.rot_cap_nd
                roti_cap = self.roti_cap_nd

            if self.use_rot:
                assert rot is not None
                rot_residual = F.relu(rot.abs() - rot_cap)
                rot_residual = self._select_nodes(rot_residual, keep_idx)
                total = self._aggregate(total, terms, "rot", rot_residual)
            if self.use_roti:
                assert roti is not None
                roti_residual = F.relu(roti - roti_cap)
                roti_residual = self._select_nodes(roti_residual, keep_idx)
                total = self._aggregate(total, terms, "roti", roti_residual)

            y_t = detach_holdout(y_nd[:, :-1, :, 0])
            y_next = y_nd[:, 1:, :, 0]
            if self.use_diffusion:
                laplacian_y = torch.einsum("ij,btj->bti", self.L, y_t)
                diffusion_residual = (y_next - y_t) - self.dt_nd * self.kappa_nd * (-laplacian_y)
                diffusion_residual = self._select_nodes(diffusion_residual, keep_idx)
                total = self._aggregate(total, terms, "diffusion", diffusion_residual)

            cos_sza = validated.get("cosSZA")
            if self.use_chemistry:
                f107 = validated["F107"].unsqueeze(-1).expand(-1, -1, nodes)
                ae = validated["AE"].unsqueeze(-1).expand(-1, -1, nodes)
                production = self.alpha_nd * f107[:, :-1].clamp_min(0.0) * cos_sza[:, :-1].clamp_min(0.0)
                loss_rate = self.beta_nd * y_t * (1.0 + self.ae_factor * ae[:, :-1])
                chemistry_residual = (y_next - y_t) - self.dt_nd * (production - loss_rate)
                chemistry_residual = self._select_nodes(chemistry_residual, keep_idx)
                total = self._aggregate(total, terms, "chemistry", chemistry_residual)

            if self.use_night:
                night_mask = (cos_sza[:, :-1] <= 1e-6).to(y_nd.dtype)
                night_residual = night_mask * F.relu(y_next - y_t)
                night_residual = self._select_nodes(night_residual, keep_idx)
                total = self._aggregate(total, terms, "night", night_residual)

            if self.use_advection:
                rows, columns = validated["grid_shape"]
                y_grid = y_t.reshape(batch, transition_count, rows, columns)
                padded = F.pad(y_grid, (0, 0, 1, 1), mode="replicate")
                gradient_y = padded[:, :, 2:, :] - padded[:, :, :-2, :]
                gradient_y = gradient_y.reshape(batch, transition_count, nodes)
                kp = validated["Kp"][:, :-1].unsqueeze(-1).expand(-1, -1, nodes)
                velocity_y = 1e-3 * kp
                advection_residual = (y_next - y_t) + self.dt_nd * velocity_y * gradient_y
                advection_residual = self._select_nodes(advection_residual, keep_idx)
                total = self._aggregate(total, terms, "advection", advection_residual)

        if y_true01 is not None:
            truth = self._as_BTN1(y_true01, field="truth")
            if tuple(truth.shape) != tuple(prediction.shape):
                raise ValueError(
                    f"truth shape mismatch: got {tuple(truth.shape)}, expected {tuple(prediction.shape)}"
                )
            self._require_finite(truth, field="truth")
            if self.strict_paper_run:
                raise ValueError("strict PPINN does not register a lazy supervised data term")
            # Legacy-only compatibility without mutating the parameter set.
            data_residual = prediction - truth
            total = total + data_residual.square().mean()
            terms["data"] = data_residual.square().mean()

        if not torch.isfinite(total).all().item():
            raise ValueError("PPINN total loss is nonfinite")
        return total, terms
