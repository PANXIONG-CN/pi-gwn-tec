"""Explicit-mask regression losses, metrics, and sufficient statistics.

This is the sole CAC-R1 masked-regression module.  It intentionally coexists
with the legacy :mod:`lib.metrics` file; ``lib/metrics/`` must not be created.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


METRIC_PROTOCOL_ID = "cac_r1_explicit_mask_float64_stats_v1"
VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION = "cac_r1.validation_metric_runtime.v1"
VALIDATION_METRIC_RUNTIME_PROFILE_SCHEMA_VERSION = (
    "cac_r1.validation_metric_runtime_profile.v1"
)
DATASET_NEUTRAL_VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION = (
    "cac_r1.dataset_neutral_validation_metric_runtime.v1"
)
DATASET_NEUTRAL_EVALUATION_METRIC_RUNTIME_SCHEMA_VERSION = (
    "cac_r1.dataset_neutral_evaluation_metric_runtime.v1"
)
STAT_FIELDS = (
    "n_valid",
    "sse",
    "sae",
    "sum_y",
    "sum_y2",
    "sum_pred",
    "sum_pred2",
    "sum_pred_y",
)

SUPERVISED_LOSS_POLICY_SCHEMA_VERSION = "cac_r1.supervised_loss_policy.v1"
TRAINING_OBJECTIVE_SCHEMA_VERSION = "cac_r1.training_objective.v1"
SUPERVISED_LOSS_RUNTIME_TELEMETRY_SCHEMA_VERSION = (
    "cac_r1.supervised_loss_runtime_telemetry.v1"
)
_SUPERVISED_LOSS_POLICY_FIELDS = (
    "schema_version",
    "strict_explicit_mask_only",
    "loss_func",
    "admissibility",
    "effective_mask_value",
    "loss_domain",
    "cap_policy",
    "cap_enabled",
    "cap_value_tecu",
    "cap_weight",
    "cap_atol_tecu",
    "huber_delta_tecu",
    "target_offset_policy",
    "normalizer_id",
    "compute_dtype",
)
_GIM_NORMALIZER_ID = "gim_tec01_147p1_v1"
_CHINA_NORMALIZER_ID = "china_tec01_300_v1"


def _require_exact_float(value: Any, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"supervised loss {field_name} must be a finite JSON float")
    return value


@dataclass(frozen=True)
class SupervisedLossPolicy:
    """Immutable, digest-bound CAC-R1 supervised-training objective.

    The policy contains only JSON scalar values so its canonical digest cannot
    depend on process state, a mutable ``args`` object, or the scaler instance.
    ``effective_mask_value=None`` is deliberate: admissibility is resolved by
    :func:`masked_loss_from_callable` before this policy's loss is invoked.
    """

    schema_version: str
    strict_explicit_mask_only: bool
    loss_func: str
    admissibility: str
    effective_mask_value: Optional[float]
    loss_domain: str
    cap_policy: str
    cap_enabled: bool
    cap_value_tecu: Optional[float]
    cap_weight: Optional[float]
    cap_atol_tecu: Optional[float]
    huber_delta_tecu: Optional[float]
    target_offset_policy: str
    normalizer_id: str
    compute_dtype: str

    def __post_init__(self) -> None:
        if self.schema_version != SUPERVISED_LOSS_POLICY_SCHEMA_VERSION:
            raise ValueError("supervised loss policy schema mismatch")
        if type(self.strict_explicit_mask_only) is not bool or (
            self.strict_explicit_mask_only is not True
        ):
            raise ValueError("CAC-R1 supervised loss requires strict explicit masks")
        if self.loss_func not in {"mask_mae", "mask_huber"}:
            raise ValueError(f"unsupported CAC-R1 supervised loss: {self.loss_func!r}")
        if self.admissibility != "explicit_target_valid_only":
            raise ValueError("supervised loss admissibility policy mismatch")
        if self.effective_mask_value is not None:
            raise ValueError("strict supervised loss forbids an effective mask value")
        if self.loss_domain != "tecu_after_inverse_transform":
            raise ValueError("supervised loss must be computed in physical TECU")
        if self.target_offset_policy != "none":
            raise ValueError("CAC-R1 supervised loss target offset must be none")
        if self.compute_dtype != "float32":
            raise ValueError("CAC-R1 supervised loss arithmetic must be float32")
        if self.loss_func == "mask_mae":
            if self.huber_delta_tecu is not None:
                raise ValueError("CAC-R1 MAE requires a null Huber delta")
        elif _require_exact_float(
            self.huber_delta_tecu, "huber_delta_tecu"
        ) != 1.0:
            raise ValueError("CAC-R1 Huber delta must be 1.0 TECU")
        if type(self.cap_enabled) is not bool:
            raise TypeError("supervised loss cap_enabled must be bool")

        if self.normalizer_id == _GIM_NORMALIZER_ID:
            if self.cap_enabled or self.cap_policy != "disabled":
                raise ValueError("GIM supervised loss cap must be disabled")
            if any(
                value is not None
                for value in (
                    self.cap_value_tecu,
                    self.cap_weight,
                    self.cap_atol_tecu,
                )
            ):
                raise ValueError("disabled supervised loss cap parameters must be null")
        elif self.normalizer_id == _CHINA_NORMALIZER_ID:
            if not self.cap_enabled or self.cap_policy != "saturation_downweight_v1":
                raise ValueError("China supervised loss cap policy mismatch")
            cap_value = _require_exact_float(self.cap_value_tecu, "cap_value_tecu")
            cap_weight = _require_exact_float(self.cap_weight, "cap_weight")
            cap_atol = _require_exact_float(self.cap_atol_tecu, "cap_atol_tecu")
            if (cap_value, cap_weight, cap_atol) != (300.0, 0.3, 0.001):
                raise ValueError("China supervised loss cap constants are not frozen")
        else:
            raise ValueError(
                f"unsupported CAC-R1 supervised loss normalizer: {self.normalizer_id!r}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in _SUPERVISED_LOSS_POLICY_FIELDS}

    @property
    def training_objective_sha256(self) -> str:
        return training_objective_sha256(self)

    def summary(self) -> dict[str, Any]:
        """Return only the canonical nested ``supervised_loss`` document."""

        return self.canonical_payload()

    def objective_identity(self) -> dict[str, str]:
        """Return the two machine-resolved sibling objective identity fields."""

        return {
            "training_objective_schema_version": TRAINING_OBJECTIVE_SCHEMA_VERSION,
            "training_objective_sha256": self.training_objective_sha256,
        }


def _training_objective_payload(
    policy_or_payload: SupervisedLossPolicy | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(policy_or_payload, SupervisedLossPolicy):
        return policy_or_payload.canonical_payload()
    if not isinstance(policy_or_payload, Mapping):
        raise TypeError("training objective must be a policy or mapping")
    document = dict(policy_or_payload)
    expected = set(_SUPERVISED_LOSS_POLICY_FIELDS)
    missing = sorted(expected - set(document))
    extra = sorted(set(document) - expected)
    if missing or extra:
        raise ValueError(
            "supervised loss policy fields mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return {field_name: document[field_name] for field_name in _SUPERVISED_LOSS_POLICY_FIELDS}


def canonical_training_objective_bytes(
    policy_or_payload: SupervisedLossPolicy | Mapping[str, Any],
) -> bytes:
    """Return the public canonical serialization used for objective identity."""

    return _canonical_json_bytes(
        {
            "schema_version": TRAINING_OBJECTIVE_SCHEMA_VERSION,
            "supervised_loss": _training_objective_payload(policy_or_payload),
        }
    )


def training_objective_sha256(
    policy_or_payload: SupervisedLossPolicy | Mapping[str, Any],
) -> str:
    """SHA-256 of UTF-8 RFC-8259-like canonical JSON (sorted keys, no NaN)."""

    return hashlib.sha256(
        canonical_training_objective_bytes(policy_or_payload)
    ).hexdigest()


def build_supervised_loss_policy(
    *, dataset: str, loss_func: str = "mask_mae"
) -> SupervisedLossPolicy:
    """Build one of the two author-frozen CAC-R1 loss policies."""

    normalized_dataset = str(dataset).strip().lower()
    common: dict[str, Any] = {
        "schema_version": SUPERVISED_LOSS_POLICY_SCHEMA_VERSION,
        "strict_explicit_mask_only": True,
        "loss_func": str(loss_func),
        "admissibility": "explicit_target_valid_only",
        "effective_mask_value": None,
        "loss_domain": "tecu_after_inverse_transform",
        "huber_delta_tecu": None if loss_func == "mask_mae" else 1.0,
        "target_offset_policy": "none",
        "compute_dtype": "float32",
    }
    if normalized_dataset in {"gim", "gimtec", "tec"}:
        return SupervisedLossPolicy(
            **common,
            cap_policy="disabled",
            cap_enabled=False,
            cap_value_tecu=None,
            cap_weight=None,
            cap_atol_tecu=None,
            normalizer_id=_GIM_NORMALIZER_ID,
        )
    if normalized_dataset in {"china", "china_hires", "st8_xy1"}:
        return SupervisedLossPolicy(
            **common,
            cap_policy="saturation_downweight_v1",
            cap_enabled=True,
            cap_value_tecu=300.0,
            cap_weight=0.3,
            cap_atol_tecu=0.001,
            normalizer_id=_CHINA_NORMALIZER_ID,
        )
    raise ValueError(f"unsupported CAC-R1 supervised loss dataset: {dataset!r}")


def supervised_loss_policy_from_mapping(
    value: Mapping[str, Any],
) -> SupervisedLossPolicy:
    """Parse the exact nested resolved-config ``supervised_loss`` mapping."""

    if not isinstance(value, Mapping):
        raise TypeError("resolved supervised loss policy must be a mapping")
    expected = set(_SUPERVISED_LOSS_POLICY_FIELDS)
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            "supervised loss policy fields mismatch: "
            f"missing={missing}, extra={extra}"
        )
    payload = _training_objective_payload(value)
    return SupervisedLossPolicy(**payload)


@dataclass(frozen=True)
class StrictSupervisedLoss:
    """Immutable CAC-R1 callable with thread-safe aggregate runtime telemetry."""

    policy: SupervisedLossPolicy
    scaler: Any = field(repr=False, compare=False)
    _telemetry_lock: threading.Lock = field(init=False, repr=False, compare=False)
    _telemetry: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SupervisedLossPolicy):
            raise TypeError("strict supervised loss requires SupervisedLossPolicy")
        if not callable(getattr(self.scaler, "inverse_transform", None)):
            raise TypeError("strict supervised loss requires an inverse-transform scaler")
        object.__setattr__(self, "_telemetry_lock", threading.Lock())
        object.__setattr__(
            self,
            "_telemetry",
            {"calls": 0, "total_valid": 0, "cap_count": 0},
        )

    @property
    def training_objective_sha256(self) -> str:
        return self.policy.training_objective_sha256

    def _inverse_float32(self, value: torch.Tensor, label: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            raise TypeError(f"strict supervised loss {label} must be a floating tensor")
        converted = value.to(dtype=torch.float32)
        transformed = self.scaler.inverse_transform(converted)
        if not isinstance(transformed, torch.Tensor):
            raise TypeError(
                f"strict supervised loss inverse-transformed {label} must be a tensor"
            )
        transformed = transformed.to(dtype=torch.float32)
        if transformed.shape != value.shape:
            raise ValueError(
                f"strict supervised loss inverse transform changed {label} shape"
            )
        if not bool(torch.isfinite(transformed).all().item()):
            raise FloatingPointError(
                f"strict supervised loss inverse-transformed {label} is non-finite"
            )
        return transformed

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            raise ValueError(
                "strict supervised loss receives admissible elements, not an internal mask"
            )
        if prediction.shape != target.shape or prediction.numel() == 0:
            raise ValueError("strict supervised loss inputs are empty or shape-mismatched")
        prediction_tecu = self._inverse_float32(prediction, "prediction")
        target_tecu = self._inverse_float32(target, "target")
        residual = torch.abs(prediction_tecu - target_tecu)
        if self.policy.loss_func == "mask_mae":
            elementwise = residual
        elif self.policy.loss_func == "mask_huber":
            delta = torch.tensor(
                self.policy.huber_delta_tecu,
                device=residual.device,
                dtype=torch.float32,
            )
            elementwise = torch.where(
                residual <= delta,
                0.5 * torch.square(residual),
                delta * residual - 0.5 * torch.square(delta),
            )
        else:  # pragma: no cover - policy validation makes this unreachable.
            raise AssertionError("unreachable supervised loss type")

        cap_count = 0
        if self.policy.cap_enabled:
            cap_value = float(self.policy.cap_value_tecu)
            cap_atol = float(self.policy.cap_atol_tecu)
            # Thresholds are rounded once into the frozen float32 compute
            # domain. This makes both +/- atol boundaries inclusive without
            # accidentally widening the interval by an arbitrary tolerance.
            lower = torch.tensor(
                cap_value - cap_atol,
                device=target_tecu.device,
                dtype=torch.float32,
            )
            upper = torch.tensor(
                cap_value + cap_atol,
                device=target_tecu.device,
                dtype=torch.float32,
            )
            cap_mask = (target_tecu >= lower) & (target_tecu <= upper)
            cap_count = int(cap_mask.sum().item())
            cap_weight = torch.tensor(
                float(self.policy.cap_weight),
                device=elementwise.device,
                dtype=torch.float32,
            )
            elementwise = torch.where(
                cap_mask, elementwise * cap_weight, elementwise
            )
        if elementwise.numel() != prediction.numel():
            raise RuntimeError("strict supervised loss elementwise cardinality drift")
        if not bool(torch.isfinite(elementwise).all().item()):
            raise FloatingPointError("strict supervised elementwise loss is non-finite")
        mean_loss = torch.mean(elementwise, dtype=torch.float32)
        if mean_loss.ndim != 0 or not bool(torch.isfinite(mean_loss).item()):
            raise FloatingPointError("strict supervised scalar loss is non-finite")

        with self._telemetry_lock:
            self._telemetry["calls"] += 1
            self._telemetry["total_valid"] += int(elementwise.numel())
            self._telemetry["cap_count"] += cap_count
        return mean_loss, elementwise

    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._telemetry_lock:
            calls = int(self._telemetry["calls"])
            total_valid = int(self._telemetry["total_valid"])
            cap_count = int(self._telemetry["cap_count"])
        return {
            "schema_version": SUPERVISED_LOSS_RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "scope": "total_across_train_and_validation",
            "training_objective_sha256": self.policy.training_objective_sha256,
            "calls": calls,
            "total_valid": total_valid,
            "cap_count": cap_count,
            "cap_fraction": (
                float(cap_count) / float(total_valid) if total_valid else 0.0
            ),
        }

    @property
    def supervised_loss_runtime_telemetry(self) -> dict[str, Any]:
        return self.telemetry_snapshot()


def build_strict_supervised_loss(
    policy: SupervisedLossPolicy, scaler: Any
) -> StrictSupervisedLoss:
    """Build a strict callable whose behavior is independent of global args."""

    return StrictSupervisedLoss(policy=policy, scaler=scaler)


def _validate_torch_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool,
) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("torch masked losses require torch prediction and target")
    if not isinstance(target_valid, torch.Tensor):
        raise TypeError("torch masked losses require a torch bool target_valid")
    if prediction.shape != target.shape or target_valid.shape != target.shape:
        raise ValueError(
            f"prediction/target/valid shape mismatch: {prediction.shape}, {target.shape}, "
            f"{target_valid.shape}"
        )
    if target_valid.dtype != torch.bool:
        raise TypeError(f"target_valid must be bool, got {target_valid.dtype}")
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("prediction contains non-finite values")
    valid = target_valid & torch.isfinite(target)
    if strict_china_st8 and not bool(valid.all().item()):
        raise RuntimeError("st8 contract requires all-valid targets")
    if int(valid.sum().item()) == 0:
        raise RuntimeError("masked regression has zero valid elements")
    return valid


def _torch_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool,
) -> torch.Tensor:
    valid = _validate_torch_inputs(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    # Float16/bfloat16 inputs are promoted before subtract/square/reduction.
    compute_dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    return prediction[valid].to(compute_dtype) - target[valid].to(compute_dtype)


def masked_loss_from_callable(
    loss_callable: Callable[[torch.Tensor, torch.Tensor], Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool = False,
    target_offset: Optional[torch.Tensor] = None,
    strict_explicit_mask_only: bool = False,
) -> Any:
    """Call a configured supervised loss on explicit valid elements only.

    Selection happens before an optional residual/background subtraction, so a
    masked non-finite target can never enter arithmetic inside the configured
    loss. Predictions remain fail-fast globally, including masked positions.
    """

    if not callable(loss_callable):
        raise TypeError("configured loss must be callable")
    if type(strict_explicit_mask_only) is not bool:
        raise TypeError("strict_explicit_mask_only must be bool")
    if strict_explicit_mask_only and target_offset is not None:
        raise ValueError("strict explicit-mask supervised loss forbids target_offset")
    valid = _validate_torch_inputs(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    if target_offset is not None:
        if not isinstance(target_offset, torch.Tensor):
            raise TypeError("target_offset must be a torch.Tensor")
        if target_offset.shape != target.shape:
            raise ValueError(
                f"target/offset shape mismatch: {target.shape}, {target_offset.shape}"
            )
        selected_offset = target_offset[valid]
        if not bool(torch.isfinite(selected_offset).all().item()):
            raise FloatingPointError("selected target_offset contains non-finite values")
        selected_target = selected_target - selected_offset

    # Strict governed arithmetic is frozen to float32. Legacy behavior keeps
    # float64 inputs and promotes only low-precision tensors as before.
    if strict_explicit_mask_only:
        selected_prediction = selected_prediction.to(torch.float32)
        selected_target = selected_target.to(torch.float32)
    elif selected_prediction.dtype in (torch.float16, torch.bfloat16):
        selected_prediction = selected_prediction.to(torch.float32)
    if not strict_explicit_mask_only and selected_target.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        selected_target = selected_target.to(torch.float32)
    result = loss_callable(selected_prediction, selected_target)
    if not strict_explicit_mask_only:
        return result

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "strict explicit-mask callable must return scalar and elementwise loss"
        )
    scalar_loss, elementwise = result
    if not isinstance(scalar_loss, torch.Tensor) or scalar_loss.ndim != 0:
        raise TypeError("strict explicit-mask callable loss must be a scalar tensor")
    if not bool(torch.isfinite(scalar_loss).item()):
        raise FloatingPointError("strict explicit-mask callable loss must be finite")
    if not isinstance(elementwise, torch.Tensor):
        raise TypeError("strict explicit-mask callable elementwise loss must be a tensor")
    expected_cardinality = int(valid.sum().item())
    if int(elementwise.numel()) != expected_cardinality:
        raise ValueError(
            "strict explicit-mask callable elementwise cardinality mismatch: "
            f"observed={elementwise.numel()}, expected={expected_cardinality}"
        )
    if not bool(torch.isfinite(elementwise).all().item()):
        raise FloatingPointError(
            "strict explicit-mask callable elementwise loss must be finite"
        )
    return scalar_loss, elementwise


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool = False,
) -> torch.Tensor:
    difference = _torch_difference(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    return torch.mean(torch.square(difference))


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool = False,
) -> torch.Tensor:
    difference = _torch_difference(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    return torch.mean(torch.abs(difference))


def masked_rmse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    strict_china_st8: bool = False,
) -> torch.Tensor:
    return torch.sqrt(
        masked_mse_loss(
            prediction,
            target,
            target_valid,
            strict_china_st8=strict_china_st8,
        )
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validated_numpy_inputs(
    prediction: Any,
    target: Any,
    target_valid: Any,
    *,
    strict_china_st8: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_array = _to_numpy(prediction)
    target_array = _to_numpy(target)
    valid_array = _to_numpy(target_valid)
    if prediction_array.shape != target_array.shape or valid_array.shape != target_array.shape:
        raise ValueError(
            "prediction/target/valid shape mismatch: "
            f"{prediction_array.shape}, {target_array.shape}, {valid_array.shape}"
        )
    if valid_array.dtype != np.bool_:
        raise TypeError(f"target_valid must be bool, got {valid_array.dtype}")
    if not np.issubdtype(prediction_array.dtype, np.number) or not np.issubdtype(
        target_array.dtype, np.number
    ):
        raise TypeError("prediction and target must be numeric")
    if not bool(np.isfinite(prediction_array).all()):
        raise FloatingPointError("prediction contains non-finite values")
    effective_valid = valid_array & np.isfinite(target_array)
    if strict_china_st8 and not bool(effective_valid.all()):
        raise RuntimeError("st8 contract requires all-valid targets")
    if int(effective_valid.sum(dtype=np.int64)) == 0:
        raise RuntimeError("masked regression has zero valid elements")
    return prediction_array, target_array, effective_valid


@dataclass(frozen=True)
class RegressionSufficientStatistics:
    n_valid: int
    sse: float
    sae: float
    sum_y: float
    sum_y2: float
    sum_pred: float
    sum_pred2: float
    sum_pred_y: float

    FLOAT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "sse",
        "sae",
        "sum_y",
        "sum_y2",
        "sum_pred",
        "sum_pred2",
        "sum_pred_y",
    )

    def __post_init__(self) -> None:
        if int(self.n_valid) < 0:
            raise ValueError("n_valid must be nonnegative")
        for field in self.FLOAT_FIELDS:
            if not math.isfinite(float(getattr(self, field))):
                raise FloatingPointError(f"non-finite sufficient statistic: {field}")
        if self.sse < 0.0 or self.sae < 0.0:
            raise ValueError("SSE/SAE must be nonnegative")

    def _require_nonempty(self) -> None:
        if int(self.n_valid) <= 0:
            raise RuntimeError("metrics require nonzero valid elements")

    @property
    def rmse(self) -> float:
        self._require_nonempty()
        return math.sqrt(self.sse / float(self.n_valid))

    @property
    def mae(self) -> float:
        self._require_nonempty()
        return self.sae / float(self.n_valid)

    @property
    def r2(self) -> float:
        self._require_nonempty()
        centered_y = self.sum_y2 - self.sum_y * self.sum_y / float(self.n_valid)
        if not math.isfinite(centered_y) or centered_y <= 0.0:
            raise ValueError("R2 requires positive finite target variance")
        value = 1.0 - self.sse / centered_y
        if not math.isfinite(value):
            raise FloatingPointError("R2 is non-finite")
        return value

    @property
    def corr(self) -> float:
        self._require_nonempty()
        count = float(self.n_valid)
        covariance = self.sum_pred_y - self.sum_pred * self.sum_y / count
        centered_pred = self.sum_pred2 - self.sum_pred * self.sum_pred / count
        centered_y = self.sum_y2 - self.sum_y * self.sum_y / count
        if centered_pred <= 0.0 or centered_y <= 0.0:
            raise ValueError("CORR requires positive finite prediction and target variance")
        denominator = math.sqrt(centered_pred * centered_y)
        value = covariance / denominator
        if not math.isfinite(value):
            raise FloatingPointError("CORR is non-finite")
        # Roundoff can put an otherwise valid pooled correlation just outside [-1,1].
        if value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
            raise ValueError(f"CORR is outside [-1,1]: {value}")
        return min(1.0, max(-1.0, value))

    def metrics(self) -> dict[str, float]:
        return {"RMSE": self.rmse, "MAE": self.mae, "R2": self.r2, "CORR": self.corr}

    def to_dict(self, *, include_metrics: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = asdict(self)
        if include_metrics:
            value.update(self.metrics())
        return value


def masked_regression_statistics(
    prediction: Any,
    target: Any,
    target_valid: Any,
    *,
    strict_china_st8: bool = False,
) -> RegressionSufficientStatistics:
    prediction_array, target_array, valid = _validated_numpy_inputs(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    pred = prediction_array[valid].astype(np.float64, copy=False)
    truth = target_array[valid].astype(np.float64, copy=False)
    difference = pred - truth
    return RegressionSufficientStatistics(
        n_valid=int(valid.sum(dtype=np.int64)),
        sse=float(np.sum(np.square(difference), dtype=np.float64)),
        sae=float(np.sum(np.abs(difference), dtype=np.float64)),
        sum_y=float(np.sum(truth, dtype=np.float64)),
        sum_y2=float(np.sum(np.square(truth), dtype=np.float64)),
        sum_pred=float(np.sum(pred, dtype=np.float64)),
        sum_pred2=float(np.sum(np.square(pred), dtype=np.float64)),
        sum_pred_y=float(np.sum(pred * truth, dtype=np.float64)),
    )


def merge_statistics(
    statistics: Iterable[RegressionSufficientStatistics],
) -> RegressionSufficientStatistics:
    values = tuple(statistics)
    if not values:
        raise ValueError("cannot merge an empty sufficient-statistics collection")
    return RegressionSufficientStatistics(
        n_valid=sum(int(value.n_valid) for value in values),
        **{
            field: math.fsum(float(getattr(value, field)) for value in values)
            for field in RegressionSufficientStatistics.FLOAT_FIELDS
        },
    )


_RUNTIME_MAPPING_FIELDS = (
    "window_start_idx",
    "forecast_origin_utc",
    "history_time_utc",
    "target_time_utc",
    "target_year",
    "role",
    "source_file_hash",
    "activity_label",
    "event_id",
    "event_canonical_id",
    "activity_event_label_sha256",
)
_RUNTIME_METADATA_FIELDS = (
    "target_year",
    "activity_label",
    "event_id",
    "event_canonical_id",
    "activity_event_label_sha256",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(_to_numpy(value), dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError("runtime payload arrays cannot use object dtype")
    header = {
        "dtype": str(array.dtype),
        "shape": [int(dimension) for dimension in array.shape],
    }
    if array.dtype.kind in {"U", "S"}:
        content = _canonical_json_bytes(array.tolist())
    else:
        content = np.ascontiguousarray(array).tobytes(order="C")
    digest = hashlib.sha256(_canonical_json_bytes(header) + b"\0" + content).hexdigest()
    return {**header, "sha256": digest}


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


_DATASET_NEUTRAL_MAPPING_FIELDS = (
    "window_start_idx", "forecast_origin_utc", "history_time_utc",
    "target_time_utc", "target_year", "role", "source_file_hash",
)
_CHINA_ACTIVITY_EVENT_MAPPING_FIELDS = frozenset({
    "activity_label", "event_id", "event_canonical_id",
    "activity_event_label_sha256",
})
_EVALUATION_ACCESS_FIELDS = (
    "path_resolutions", "files_statted", "files_hashed", "files_opened",
)


def _require_contract_fields(
    document: Mapping[str, Any], required: Sequence[str], label: str
) -> None:
    if not isinstance(document, Mapping):
        raise TypeError(f"validation profile {label} must be an object")
    missing = sorted(set(required) - set(document))
    if missing:
        raise ValueError(f"validation profile {label} is missing fields: {missing}")
    _canonical_json_bytes(document)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if not _is_sha256(text):
        raise ValueError(f"validation profile {label} must be a SHA-256 digest")
    return text


def _document_digest_without(document: Mapping[str, Any], digest_field: str) -> str:
    payload = dict(document)
    payload.pop(digest_field, None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sorted_years(value: Any, label: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"validation profile {label} must be a year sequence")
    years = tuple(int(year) for year in value)
    if (nonempty and not years) or years != tuple(sorted(set(years))):
        raise ValueError(f"validation profile {label} must be unique and sorted")
    return years


@dataclass(frozen=True)
class ValidationMetricRuntimeProfile:
    """Digest-bound dataset contract; dataset extensions are explicit."""

    dataset: str
    role: str
    role_years: Tuple[int, ...]
    data_contract: Mapping[str, Any]
    role_split_contract: Mapping[str, Any]
    normalizer_contract: Mapping[str, Any]
    mask_contract: Mapping[str, Any]
    window_contract: Mapping[str, Any]
    evaluation_access_audit: Mapping[str, Any]
    dataset_extension: str
    profile_digest_sha256: str
    schema_version: str = VALIDATION_METRIC_RUNTIME_PROFILE_SCHEMA_VERSION
    physical_unit: str = "TECU"

    def _summary_without_digest(self) -> dict[str, Any]:
        document = {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "role": self.role,
            "role_years": list(self.role_years),
            "physical_unit": self.physical_unit,
            "dataset_extension": self.dataset_extension,
            "evaluation_access_audit": copy.deepcopy(dict(self.evaluation_access_audit)),
        }
        for name in ("data", "role_split", "normalizer", "mask", "window"):
            document[f"{name}_contract"] = copy.deepcopy(
                dict(getattr(self, f"{name}_contract"))
            )
        return document

    def summary(self) -> dict[str, Any]:
        return _json_clone({
            **self._summary_without_digest(),
            "profile_digest_sha256": self.profile_digest_sha256,
        })

    def verify(self) -> dict[str, Any]:
        _validate_runtime_profile(self)
        actual = hashlib.sha256(
            _canonical_json_bytes(self._summary_without_digest())
        ).hexdigest()
        if not _is_sha256(self.profile_digest_sha256) or self.profile_digest_sha256 != actual:
            raise ValueError("validation metric runtime profile digest mismatch")
        return self.summary()


def _validate_runtime_profile(profile: ValidationMetricRuntimeProfile) -> None:
    if profile.schema_version != VALIDATION_METRIC_RUNTIME_PROFILE_SCHEMA_VERSION:
        raise ValueError("validation metric runtime profile schema mismatch")
    if not profile.dataset or profile.role not in {"validation", "evaluation"}:
        raise ValueError("metric runtime profile dataset/role mismatch")
    years = _sorted_years(profile.role_years, "role years", nonempty=True)
    allowed_extensions = {
        "gim": {"not_applicable"},
        "china": {"china_activity_event_v1"},
    }
    if (
        profile.physical_unit != "TECU"
        or profile.dataset_extension
        not in allowed_extensions.get(profile.dataset, {"not_applicable"})
    ):
        raise ValueError("dataset-neutral metric profile unit/extension mismatch")

    data = profile.data_contract
    _require_contract_fields(data, (
        "data_protocol_id", "data_build_sha256", "source_bundle_sha256",
        "alignment_schema_version", "alignment_source",
    ), "data contract")
    if not all(str(data[key]).strip() for key in (
        "data_protocol_id", "alignment_schema_version", "alignment_source",
    )):
        raise ValueError("validation profile data contract identity is empty")
    for field in ("data_build_sha256", "source_bundle_sha256"):
        _require_sha256(data[field], field)

    split = profile.role_split_contract
    _require_contract_fields(split, (
        "protocol_id", "protocol_sha256", "train_years", "validation_years",
        "evaluation_years",
    ), "role split contract")
    _require_sha256(split["protocol_sha256"], "role split protocol digest")
    partitions = {
        name: _sorted_years(split[f"{name}_years"], f"{name} years")
        for name in ("train", "validation", "evaluation")
    }
    if years != partitions[profile.role] or any(
        set(partitions[left]) & set(partitions[right])
        for left, right in (("train", "validation"), ("train", "evaluation"),
                            ("validation", "evaluation"))
    ):
        raise ValueError("validation profile role split mismatch or overlap")

    normalizer = profile.normalizer_contract
    _require_contract_fields(normalizer, (
        "normalizer_id", "normalizer_digest_sha256", "physical_min",
        "physical_max", "network_min", "network_max", "physical_unit",
    ), "normalizer contract")
    _require_sha256(normalizer["normalizer_digest_sha256"], "normalizer digest")
    if not normalizer["normalizer_id"] or normalizer["physical_unit"] != "TECU":
        raise ValueError("validation profile normalizer identity/unit mismatch")
    for low, high in (("physical_min", "physical_max"),
                      ("network_min", "network_max")):
        bounds = (float(normalizer[low]), float(normalizer[high]))
        if not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("validation profile normalizer range is invalid")

    mask = profile.mask_contract
    _require_contract_fields(mask, (
        "mask_contract_id", "target_valid_dtype", "effective_valid_rule",
        "requires_all_valid", "zero_valid_policy",
    ), "mask contract")
    if (not mask["mask_contract_id"] or mask["target_valid_dtype"] != "bool"
            or mask["effective_valid_rule"] != "target_valid & isfinite(target)"
            or not isinstance(mask["requires_all_valid"], bool)
            or mask["zero_valid_policy"] != "fail"):
        raise ValueError("validation profile mask contract mismatch")

    window = profile.window_contract
    _require_contract_fields(window, (
        "window_contract_id", "cadence_seconds", "history_steps",
        "horizon_steps", "stride_steps", "prefix_boundary",
    ), "window contract")
    if not window["window_contract_id"] or not isinstance(window["prefix_boundary"], bool):
        raise ValueError("validation profile window identity/type mismatch")
    for field in ("cadence_seconds", "history_steps", "horizon_steps", "stride_steps"):
        if isinstance(window[field], bool) or not isinstance(window[field], int) or window[field] <= 0:
            raise ValueError(f"validation profile window {field} must be positive int")

    audit = profile.evaluation_access_audit
    _require_contract_fields(audit, (
        "schema_version", "dataset", "run_stage", "validation_only",
        "validation_years", "evaluation_access", "evaluation_loader_created",
        "artifact_digest_sha256",
    ), "evaluation-access audit")
    counters = audit["evaluation_access"]
    if not isinstance(counters, Mapping) or set(counters) != set(_EVALUATION_ACCESS_FIELDS):
        raise ValueError("validation profile evaluation access requires four counters")
    if audit["dataset"] != profile.dataset:
        raise PermissionError("metric profile evaluation-access dataset mismatch")
    run_stage = str(audit["run_stage"])
    if run_stage in {"pilot", "train_only"}:
        if profile.role != "validation":
            raise PermissionError("pilot metric profile role must be validation")
        if any(
            type(counters[field]) is not int or counters[field] != 0
            for field in _EVALUATION_ACCESS_FIELDS
        ):
            raise PermissionError("validation profile evaluation access is not four-zero")
        if (
            audit["validation_only"] is not True
            or audit["evaluation_loader_created"] is not False
            or _sorted_years(
                audit["validation_years"], "audit validation years"
            )
            != years
        ):
            raise PermissionError("validation profile evaluation-access audit mismatch")
    elif run_stage == "final":
        if any(
            type(counters[field]) is not int or counters[field] < 0
            for field in _EVALUATION_ACCESS_FIELDS
        ):
            raise ValueError("final metric profile evaluation-access counters are invalid")
        required_final = (
            "evaluation_years",
            "evaluation_access_count",
            "first_evaluation_access_at_utc",
            "selection_lock_digest_sha256",
            "selection_lock_created_at_utc",
        )
        _require_contract_fields(audit, required_final, "final evaluation-access audit")
        role_field = f"{profile.role}_years"
        if (
            audit["validation_only"] is not False
            or audit["evaluation_loader_created"] is not True
            or _sorted_years(audit[role_field], f"audit {profile.role} years") != years
        ):
            raise PermissionError("final metric profile role/access audit mismatch")
        access_count = audit["evaluation_access_count"]
        if (
            type(access_count) is not int
            or access_count <= 0
            or access_count != int(counters["files_opened"])
        ):
            raise PermissionError("final metric profile evaluation access count mismatch")
        _require_sha256(
            audit["selection_lock_digest_sha256"], "selection lock digest"
        )

        def parse_utc(value: Any, label: str) -> datetime:
            text = str(value or "")
            if not text.endswith("Z"):
                raise ValueError(f"final metric profile {label} must be UTC")
            try:
                parsed = datetime.fromisoformat(text[:-1] + "+00:00")
            except ValueError as exc:
                raise ValueError(
                    f"final metric profile {label} is invalid"
                ) from exc
            if parsed.tzinfo != timezone.utc:
                raise ValueError(f"final metric profile {label} must be UTC")
            return parsed

        first_access = parse_utc(
            audit["first_evaluation_access_at_utc"],
            "first evaluation access timestamp",
        )
        lock_created = parse_utc(
            audit["selection_lock_created_at_utc"],
            "selection lock timestamp",
        )
        if not lock_created < first_access:
            raise PermissionError(
                "selection lock must strictly predate first evaluation access"
            )
    else:
        raise ValueError("metric profile evaluation-access run stage is invalid")
    recorded = _require_sha256(audit["artifact_digest_sha256"], "audit digest")
    if recorded != _document_digest_without(audit, "artifact_digest_sha256"):
        raise ValueError("validation profile evaluation-access audit digest mismatch")


def build_validation_metric_runtime_profile(
    *, dataset: str, role: str, role_years: Sequence[int],
    data_contract: Mapping[str, Any], role_split_contract: Mapping[str, Any],
    normalizer_contract: Mapping[str, Any], mask_contract: Mapping[str, Any],
    window_contract: Mapping[str, Any], evaluation_access_audit: Mapping[str, Any],
    dataset_extension: str = "not_applicable",
) -> ValidationMetricRuntimeProfile:
    values = {
        name: _json_clone(value) for name, value in {
            "data_contract": data_contract, "role_split_contract": role_split_contract,
            "normalizer_contract": normalizer_contract, "mask_contract": mask_contract,
            "window_contract": window_contract,
            "evaluation_access_audit": evaluation_access_audit,
        }.items()
    }
    provisional = ValidationMetricRuntimeProfile(
        dataset=str(dataset), role=str(role),
        role_years=tuple(int(year) for year in role_years),
        dataset_extension=str(dataset_extension), profile_digest_sha256="", **values,
    )
    profile = replace(provisional, profile_digest_sha256=hashlib.sha256(
        _canonical_json_bytes(provisional._summary_without_digest())
    ).hexdigest())
    profile.verify()
    return profile


def _mapping_arrays(
    documents: Sequence[Mapping[str, Any]],
    digests: Sequence[str],
) -> dict[str, np.ndarray]:
    if not documents or len(documents) != len(digests):
        raise ValueError("validation runtime requires one digest per non-empty mapping document")
    collected = {field: [] for field in _RUNTIME_MAPPING_FIELDS}
    for batch_index, (document, digest) in enumerate(zip(documents, digests)):
        if not isinstance(document, Mapping):
            raise TypeError("validation mapping document must be an object")
        missing = sorted(set(_RUNTIME_MAPPING_FIELDS) - set(document))
        extra = sorted(set(document) - set(_RUNTIME_MAPPING_FIELDS))
        if missing or extra:
            raise ValueError(
                f"validation mapping document fields mismatch at batch {batch_index}: "
                f"missing={missing}, extra={extra}"
            )
        actual_digest = hashlib.sha256(_canonical_json_bytes(document)).hexdigest()
        if not _is_sha256(digest) or str(digest) != actual_digest:
            raise ValueError(
                f"validation mapping digest mismatch at batch {batch_index}: "
                f"recorded={digest}, actual={actual_digest}"
            )
        batch_size = len(document["window_start_idx"])
        if batch_size <= 0:
            raise ValueError("validation mapping batch cannot be empty")
        for field in _RUNTIME_MAPPING_FIELDS:
            values = document[field]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise TypeError(f"validation mapping field {field} must be a batch sequence")
            if len(values) != batch_size:
                raise ValueError(
                    f"validation mapping field {field} has {len(values)} rows; "
                    f"expected {batch_size}"
                )
            collected[field].extend(values)

    arrays = {
        "window_start_idx": _readonly_array(collected["window_start_idx"], dtype=np.int64),
        "forecast_origin_utc": _readonly_array(
            collected["forecast_origin_utc"], dtype=np.int64
        ),
        "history_time_utc": _readonly_array(collected["history_time_utc"], dtype=np.int64),
        "target_time_utc": _readonly_array(collected["target_time_utc"], dtype=np.int64),
        "target_year": _readonly_array(collected["target_year"], dtype=np.int16),
        "role": _readonly_array(collected["role"], dtype=np.str_),
        "source_file_hash": _readonly_array(collected["source_file_hash"], dtype=np.str_),
        "activity_label": _readonly_array(collected["activity_label"], dtype=np.uint8),
        "event_id": _readonly_array(collected["event_id"], dtype=np.int32),
        "event_canonical_id": _readonly_array(
            collected["event_canonical_id"], dtype=np.str_
        ),
        "activity_event_label_sha256": _readonly_array(
            collected["activity_event_label_sha256"], dtype=np.str_
        ),
    }
    window_count = int(arrays["window_start_idx"].shape[0])
    if arrays["history_time_utc"].ndim != 2 or arrays["target_time_utc"].ndim != 2:
        raise ValueError("validation history/target timestamps must be rank-2")
    if any(array.shape[0] != window_count for array in arrays.values()):
        raise ValueError("validation mapping arrays do not share a window dimension")
    if any(not _is_sha256(value) for value in arrays["source_file_hash"].tolist()):
        raise ValueError("validation mapping contains an invalid source-file digest")
    if any(
        not _is_sha256(value)
        for value in arrays["activity_event_label_sha256"].tolist()
    ):
        raise ValueError("validation mapping contains an invalid activity-label digest")
    cadence_seconds = 7200
    if not np.array_equal(
        arrays["history_time_utc"][:, -1] + cadence_seconds,
        arrays["forecast_origin_utc"],
    ) or not np.array_equal(
        arrays["target_time_utc"][:, 0],
        arrays["forecast_origin_utc"],
    ):
        raise ValueError("validation mapping forecast-origin timestamps do not close")
    if arrays["history_time_utc"].shape[1] > 1 and not np.all(
        np.diff(arrays["history_time_utc"], axis=1) == cadence_seconds
    ):
        raise ValueError("validation history timestamps violate the frozen cadence")
    if arrays["target_time_utc"].shape[1] > 1 and not np.all(
        np.diff(arrays["target_time_utc"], axis=1) == cadence_seconds
    ):
        raise ValueError("validation target timestamps violate the frozen cadence")

    from lib.truth.china_activity_events import label_identity_sha256

    for index in range(window_count):
        sequence = int(arrays["event_id"][index])
        canonical_id = str(arrays["event_canonical_id"][index])
        if (sequence == -1) != (canonical_id == ""):
            raise ValueError("validation event sequence/canonical ID binding mismatch")
        runtime_role = str(arrays["role"][index])
        target_year = int(arrays["target_year"][index])
        shard_id = (
            f"robustness_{target_year}"
            if runtime_role == "robustness"
            else runtime_role
        )
        expected_label = label_identity_sha256(
            shard_id=shard_id,
            runtime_role=runtime_role,
            target_time_utc=arrays["target_time_utc"][index],
            activity_label=int(arrays["activity_label"][index]),
            event_sequence=sequence,
            event_canonical_id=canonical_id,
        )
        if str(arrays["activity_event_label_sha256"][index]) != expected_label:
            raise ValueError("validation activity/event label identity mismatch")
    return arrays


def _statistics_summary(statistics: RegressionSufficientStatistics) -> dict[str, Any]:
    document = statistics.to_dict(include_metrics=False)
    document["RMSE"] = statistics.rmse
    document["MAE"] = statistics.mae
    return document


def _statistics_equal(
    expected: RegressionSufficientStatistics,
    observed: RegressionSufficientStatistics,
) -> bool:
    if int(expected.n_valid) != int(observed.n_valid):
        return False
    return all(
        math.isclose(
            float(getattr(expected, field)),
            float(getattr(observed, field)),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for field in RegressionSufficientStatistics.FLOAT_FIELDS
    )


def _expected_alignment_tracker(
    *,
    role: str,
    mapping: Mapping[str, np.ndarray],
    batch_mapping_digests: Sequence[str],
) -> dict[str, Any]:
    origins = [int(value) for value in mapping["forecast_origin_utc"].tolist()]
    if not origins:
        raise ValueError("validation alignment cannot be empty")
    if len(origins) != len(set(origins)):
        raise ValueError("validation alignment contains duplicate origins")
    if any(right <= left for left, right in zip(origins, origins[1:])):
        raise ValueError("validation alignment origins are not strictly ordered")
    return {
        "schema_version": "cac_r1.china_runtime_alignment.v1",
        "role": str(role),
        "n_windows": len(origins),
        "n_batches": len(batch_mapping_digests),
        "first_forecast_origin_utc": origins[0],
        "last_forecast_origin_utc": origins[-1],
        "batch_mapping_digest_sha256": list(batch_mapping_digests),
        "alignment_digest_sha256": hashlib.sha256(
            _canonical_json_bytes(list(batch_mapping_digests))
        ).hexdigest(),
    }


def _metric_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    *,
    require_all_valid: bool,
) -> tuple[RegressionSufficientStatistics, Tuple[RegressionSufficientStatistics, ...]]:
    overall = masked_regression_statistics(
        prediction, truth, valid, strict_china_st8=require_all_valid
    )
    horizons = tuple(
        masked_regression_statistics(
            prediction[:, lead : lead + 1, ...],
            truth[:, lead : lead + 1, ...],
            valid[:, lead : lead + 1, ...],
            strict_china_st8=require_all_valid,
        )
        for lead in range(prediction.shape[1])
    )
    return overall, horizons


def _verify_metric_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    overall: RegressionSufficientStatistics,
    horizons: Sequence[RegressionSufficientStatistics],
    *,
    require_all_valid: bool,
    label: str,
) -> None:
    expected_overall, expected_horizons = _metric_statistics(
        prediction, truth, valid, require_all_valid=require_all_valid
    )
    if not _statistics_equal(expected_overall, overall):
        raise ValueError(f"{label} overall sufficient statistics mismatch")
    if len(horizons) != prediction.shape[1] or any(
        not _statistics_equal(expected, observed)
        for expected, observed in zip(expected_horizons, horizons)
    ):
        raise ValueError(f"{label} stepwise sufficient statistics mismatch")
    if not _statistics_equal(merge_statistics(horizons), overall):
        raise ValueError(f"{label} stepwise/overall sufficient statistics do not close")


def _metric_array_identities(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    mapping: Mapping[str, np.ndarray],
    metadata: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, Mapping[str, Any]]:
    identities = {
        "prediction_tecu": _array_identity(prediction),
        "truth_tecu": _array_identity(truth),
        "target_valid": _array_identity(valid),
        **{f"mapping.{name}": _array_identity(value) for name, value in mapping.items()},
    }
    if metadata is not None:
        identities.update(
            {f"metadata.{name}": _array_identity(value) for name, value in metadata.items()}
        )
    return identities


def _metric_summary_fields(
    *,
    schema_version: str,
    epoch: int,
    role: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    quantization_policy: str,
    overall: RegressionSufficientStatistics,
    horizons: Sequence[RegressionSufficientStatistics],
    identities: Mapping[str, Mapping[str, Any]],
    mapping_digests: Sequence[str],
    provenance: Mapping[str, Any],
    alignment: Mapping[str, Any],
    profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    document = {
        "schema_version": schema_version,
        "metric_protocol_id": METRIC_PROTOCOL_ID,
        "epoch": int(epoch),
        "role": str(role),
        "prediction_dtype": str(prediction.dtype),
        "truth_dtype": str(truth.dtype),
        "target_valid_dtype": str(valid.dtype),
        "prediction_quantization_policy": quantization_policy,
        "overall": _statistics_summary(overall),
        "horizons": [
            {"lead": index + 1, **_statistics_summary(statistics)}
            for index, statistics in enumerate(horizons)
        ],
        "array_identities": copy.deepcopy(dict(identities)),
        "batch_mapping_digest_sha256": list(mapping_digests),
        "provenance": copy.deepcopy(dict(provenance)),
        "alignment_verification": copy.deepcopy(dict(alignment)),
    }
    if profile is not None:
        document["profile"] = copy.deepcopy(dict(profile))
    return document


@dataclass(frozen=True)
class ValidationMetricRuntimePayload:
    """In-memory strict-validation closure built from verified T153 batches."""

    epoch: int
    role: str
    prediction_tecu: np.ndarray
    truth_tecu: np.ndarray
    target_valid: np.ndarray
    metadata: Mapping[str, np.ndarray]
    mapping: Mapping[str, np.ndarray]
    batch_mapping_documents: Tuple[Mapping[str, Any], ...]
    batch_mapping_digests: Tuple[str, ...]
    provenance: Mapping[str, Any]
    alignment_verification: Mapping[str, Any]
    overall: RegressionSufficientStatistics
    horizons: Tuple[RegressionSufficientStatistics, ...]
    array_identities: Mapping[str, Mapping[str, Any]]
    payload_digest_sha256: str
    schema_version: str = VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION
    metric_protocol_id: str = METRIC_PROTOCOL_ID
    prediction_quantization_policy: str = "float32_tecu_cast_before_statistics"
    profile: Optional[ValidationMetricRuntimeProfile] = None

    def _summary_without_digest(self) -> dict[str, Any]:
        return _metric_summary_fields(
            schema_version=self.schema_version,
            epoch=self.epoch,
            role=self.role,
            prediction=self.prediction_tecu,
            truth=self.truth_tecu,
            valid=self.target_valid,
            quantization_policy=self.prediction_quantization_policy,
            overall=self.overall,
            horizons=self.horizons,
            identities=self.array_identities,
            mapping_digests=self.batch_mapping_digests,
            provenance=self.provenance,
            alignment=self.alignment_verification,
            profile=self.profile.summary() if self.profile is not None else None,
        )

    def summary(self) -> dict[str, Any]:
        document = self._summary_without_digest()
        document["payload_digest_sha256"] = self.payload_digest_sha256
        return _json_clone(document)

    def verify(self) -> dict[str, Any]:
        neutral = self.profile is not None
        if neutral:
            if not isinstance(self.profile, ValidationMetricRuntimeProfile):
                raise TypeError("dataset-neutral metric runtime profile is invalid")
            expected_schema = (
                DATASET_NEUTRAL_EVALUATION_METRIC_RUNTIME_SCHEMA_VERSION
                if self.profile.role == "evaluation"
                else DATASET_NEUTRAL_VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION
            )
            label = f"dataset-neutral {self.profile.role}"
        else:
            expected_schema = VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION
            label = "validation"
        if self.schema_version != expected_schema:
            raise ValueError(f"{label} metric runtime schema mismatch")
        if self.metric_protocol_id != METRIC_PROTOCOL_ID:
            raise ValueError(f"{label} metric protocol mismatch")
        if neutral:
            self.profile.verify()
            expected_role = self.profile.role
        else:
            expected_role = "validation"
        if int(self.epoch) < 0 or self.role != expected_role:
            raise ValueError(f"{label} metric runtime epoch/role mismatch")
        if self.prediction_tecu.dtype != np.float32 or self.truth_tecu.dtype != np.float32:
            raise TypeError(f"{label} prediction/truth must be float32 TECU")
        if self.target_valid.dtype != np.bool_:
            raise TypeError(f"{label} target_valid must be bool")
        if (
            self.prediction_tecu.shape != self.truth_tecu.shape
            or self.target_valid.shape != self.truth_tecu.shape
            or self.prediction_tecu.ndim < 2
        ):
            raise ValueError(f"{label} prediction/truth/mask shape mismatch")

        documents = tuple(_json_clone(document) for document in self.batch_mapping_documents)
        if neutral:
            recomputed_mapping = _profile_mapping_arrays(
                documents, self.batch_mapping_digests, profile=self.profile
            )
            expected_horizon = int(self.profile.window_contract["horizon_steps"])
        else:
            recomputed_mapping = _mapping_arrays(documents, self.batch_mapping_digests)
            expected_horizon = 12
        if set(self.mapping) != set(recomputed_mapping):
            raise ValueError(f"{label} mapping field set mismatch")
        for name, expected in recomputed_mapping.items():
            if not np.array_equal(expected, self.mapping[name]):
                raise ValueError(f"{label} mapping array mismatch: {name}")
        if self.prediction_tecu.shape[0] != recomputed_mapping["window_start_idx"].shape[0]:
            raise ValueError(f"{label} arrays/mapping window count mismatch")
        if (self.prediction_tecu.shape[1] != recomputed_mapping["target_time_utc"].shape[1]
                or self.prediction_tecu.shape[1] != expected_horizon):
            raise ValueError(f"{label} horizon contract mismatch")

        if neutral:
            if self.metadata:
                raise ValueError("dataset-neutral metric metadata must be empty")
            bindings = _verify_dataset_neutral_provenance(self.profile, self.provenance)
            _verify_mapping_source_bindings(recomputed_mapping, bindings)
            tracker = _expected_dataset_neutral_alignment_tracker(
                profile=self.profile, role=self.role, mapping=recomputed_mapping,
                batch_mapping_digests=self.batch_mapping_digests,
            )
            expected_alignment = {
                "source": self.profile.data_contract["alignment_source"],
                "tracker": tracker,
            }
            require_all = bool(self.profile.mask_contract["requires_all_valid"])
            if require_all and not bool(
                (self.target_valid & np.isfinite(self.truth_tecu)).all()
            ):
                raise RuntimeError("dataset-neutral metric profile requires all-valid targets")
        else:
            for name in _RUNTIME_METADATA_FIELDS:
                if name not in self.metadata or not np.array_equal(
                    recomputed_mapping[name], self.metadata[name]
                ):
                    raise ValueError(f"validation metadata array mismatch: {name}")
            from lib.datasets.china_hires_runtime import validate_runtime_payload
            validate_runtime_payload(self.provenance)
            expected_alignment = {
                "source": "T153.verify_prediction_truth_alignment",
                "tracker": _expected_alignment_tracker(
                    role=self.role, mapping=recomputed_mapping,
                    batch_mapping_digests=self.batch_mapping_digests,
                ),
            }
            require_all = True
        if dict(self.alignment_verification) != expected_alignment:
            raise ValueError(f"{label} alignment verification is not reproducible")

        _verify_metric_statistics(
            self.prediction_tecu,
            self.truth_tecu,
            self.target_valid,
            self.overall,
            self.horizons,
            require_all_valid=require_all,
            label=label,
        )
        current_identities = _metric_array_identities(
            self.prediction_tecu,
            self.truth_tecu,
            self.target_valid,
            self.mapping,
            None if neutral else self.metadata,
        )
        if current_identities != dict(self.array_identities):
            raise ValueError(f"{label} payload array identities mismatch")
        actual_payload_digest = hashlib.sha256(
            _canonical_json_bytes(self._summary_without_digest())
        ).hexdigest()
        if not _is_sha256(self.payload_digest_sha256) or (
            self.payload_digest_sha256 != actual_payload_digest
        ):
            raise ValueError(f"{label} metric runtime payload digest mismatch")
        return self.summary()


def build_validation_metric_runtime_payload(
    *,
    epoch: int,
    prediction_tecu: Any,
    truth_tecu: Any,
    target_valid: Any,
    batch_mapping_documents: Sequence[Mapping[str, Any]],
    batch_mapping_digests: Sequence[str],
    runtime_provenance: Mapping[str, Any],
    alignment_tracker: Mapping[str, Any],
) -> ValidationMetricRuntimePayload:
    prediction = _readonly_array(prediction_tecu, dtype=np.float32)
    truth = _readonly_array(truth_tecu, dtype=np.float32)
    valid = _readonly_array(target_valid, dtype=np.bool_)
    documents = tuple(_json_clone(document) for document in batch_mapping_documents)
    digests = tuple(str(digest) for digest in batch_mapping_digests)
    mapping = _mapping_arrays(documents, digests)
    if prediction.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError("validation prediction/truth/mask shape mismatch")
    if prediction.ndim < 2:
        raise ValueError("validation arrays require an explicit horizon dimension")
    if prediction.shape[0] != mapping["window_start_idx"].shape[0]:
        raise ValueError("validation arrays/mapping window count mismatch")
    if prediction.shape[1] != mapping["target_time_utc"].shape[1]:
        raise ValueError("validation arrays/target timestamp horizon mismatch")
    if prediction.shape[1] != 12:
        raise ValueError("strict China validation payload requires 12 forecast leads")
    metadata = {name: mapping[name] for name in _RUNTIME_METADATA_FIELDS}
    overall, horizons = _metric_statistics(
        prediction, truth, valid, require_all_valid=True
    )
    provenance = _json_clone(runtime_provenance)
    alignment = {
        "source": "T153.verify_prediction_truth_alignment",
        "tracker": _json_clone(alignment_tracker),
    }
    identities = _metric_array_identities(prediction, truth, valid, mapping, metadata)
    provisional = ValidationMetricRuntimePayload(
        epoch=int(epoch),
        role="validation",
        prediction_tecu=prediction,
        truth_tecu=truth,
        target_valid=valid,
        metadata=metadata,
        mapping=mapping,
        batch_mapping_documents=documents,
        batch_mapping_digests=digests,
        provenance=provenance,
        alignment_verification=alignment,
        overall=overall,
        horizons=horizons,
        array_identities=identities,
        payload_digest_sha256="",
    )
    payload = replace(
        provisional,
        payload_digest_sha256=hashlib.sha256(
            _canonical_json_bytes(provisional._summary_without_digest())
        ).hexdigest(),
    )
    payload.verify()
    return payload


def _dataset_neutral_mapping_arrays(
    documents: Sequence[Mapping[str, Any]],
    digests: Sequence[str],
    *,
    profile: ValidationMetricRuntimeProfile,
) -> dict[str, np.ndarray]:
    if not documents or len(documents) != len(digests):
        raise ValueError(
            "dataset-neutral validation runtime requires one digest per "
            "non-empty mapping document"
        )
    collected = {field: [] for field in _DATASET_NEUTRAL_MAPPING_FIELDS}
    for batch_index, (document, digest) in enumerate(zip(documents, digests)):
        if not isinstance(document, Mapping):
            raise TypeError("dataset-neutral validation mapping document must be an object")
        missing = sorted(set(_DATASET_NEUTRAL_MAPPING_FIELDS) - set(document))
        extra = sorted(set(document) - set(_DATASET_NEUTRAL_MAPPING_FIELDS))
        if missing or extra:
            china_extra = sorted(set(extra) & _CHINA_ACTIVITY_EVENT_MAPPING_FIELDS)
            suffix = (
                f"; dataset extension forbids China activity/event fields={china_extra}"
                if china_extra
                else ""
            )
            raise ValueError(
                "dataset-neutral validation mapping document fields mismatch at "
                f"batch {batch_index}: missing={missing}, extra={extra}{suffix}"
            )
        actual_digest = hashlib.sha256(_canonical_json_bytes(document)).hexdigest()
        if not _is_sha256(digest) or str(digest) != actual_digest:
            raise ValueError(
                "dataset-neutral validation mapping digest mismatch at batch "
                f"{batch_index}: recorded={digest}, actual={actual_digest}"
            )
        batch_size = len(document["window_start_idx"])
        if batch_size <= 0:
            raise ValueError("dataset-neutral validation mapping batch cannot be empty")
        for field in _DATASET_NEUTRAL_MAPPING_FIELDS:
            values = document[field]
            if not isinstance(values, Sequence) or isinstance(
                values, (str, bytes, bytearray)
            ):
                raise TypeError(
                    f"dataset-neutral validation mapping field {field} "
                    "must be a batch sequence"
                )
            if len(values) != batch_size:
                raise ValueError(
                    f"dataset-neutral validation mapping field {field} has "
                    f"{len(values)} rows; expected {batch_size}"
                )
            collected[field].extend(values)

    raw_years = np.asarray(collected["target_year"], dtype=np.int64)
    if raw_years.ndim != 1 or np.any((raw_years < 1900) | (raw_years > 3000)):
        raise ValueError("dataset-neutral validation mapping target year is invalid")
    arrays = {
        "window_start_idx": _readonly_array(
            collected["window_start_idx"], dtype=np.int64
        ),
        "forecast_origin_utc": _readonly_array(
            collected["forecast_origin_utc"], dtype=np.int64
        ),
        "history_time_utc": _readonly_array(
            collected["history_time_utc"], dtype=np.int64
        ),
        "target_time_utc": _readonly_array(
            collected["target_time_utc"], dtype=np.int64
        ),
        "target_year": _readonly_array(raw_years, dtype=np.int16),
        "role": _readonly_array(collected["role"], dtype=np.str_),
        "source_file_hash": _readonly_array(
            collected["source_file_hash"], dtype=np.str_
        ),
    }
    window_count = int(arrays["window_start_idx"].shape[0])
    if any(array.shape[0] != window_count for array in arrays.values()):
        raise ValueError(
            "dataset-neutral validation mapping arrays do not share a window dimension"
        )
    if arrays["history_time_utc"].ndim != 2 or arrays["target_time_utc"].ndim != 2:
        raise ValueError(
            "dataset-neutral validation history/target timestamps must be rank-2"
        )
    history_steps = int(profile.window_contract["history_steps"])
    horizon_steps = int(profile.window_contract["horizon_steps"])
    if arrays["history_time_utc"].shape[1] != history_steps:
        raise ValueError("dataset-neutral validation history step count mismatch")
    if arrays["target_time_utc"].shape[1] != horizon_steps:
        raise ValueError("dataset-neutral validation horizon step count mismatch")
    if any(not _is_sha256(value) for value in arrays["source_file_hash"].tolist()):
        raise ValueError(
            "dataset-neutral validation mapping contains an invalid source-file digest"
        )
    starts = arrays["window_start_idx"]
    if bool(np.any(starts < 0)) or (
        starts.size > 1 and not bool(np.all(np.diff(starts) > 0))
    ):
        raise ValueError(
            "dataset-neutral validation window starts are not strictly ordered"
        )

    origins = arrays["forecast_origin_utc"]
    if len(origins) != len(set(int(value) for value in origins.tolist())):
        raise ValueError("dataset-neutral validation alignment contains duplicate origins")
    if origins.size > 1 and not bool(np.all(np.diff(origins) > 0)):
        raise ValueError(
            "dataset-neutral validation alignment origins are not strictly ordered"
        )
    cadence_seconds = int(profile.window_contract["cadence_seconds"])
    if not np.array_equal(
        arrays["history_time_utc"][:, -1] + cadence_seconds,
        origins,
    ) or not np.array_equal(arrays["target_time_utc"][:, 0], origins):
        raise ValueError(
            "dataset-neutral validation mapping forecast-origin timestamps do not close"
        )
    if history_steps > 1 and not bool(
        np.all(np.diff(arrays["history_time_utc"], axis=1) == cadence_seconds)
    ):
        raise ValueError(
            "dataset-neutral validation history timestamps violate the frozen cadence"
        )
    if horizon_steps > 1 and not bool(
        np.all(np.diff(arrays["target_time_utc"], axis=1) == cadence_seconds)
    ):
        raise ValueError(
            "dataset-neutral validation target timestamps violate the frozen cadence"
        )
    declared_mapping_roles = profile.data_contract.get("mapping_roles")
    if declared_mapping_roles is None:
        expected_mapping_roles = {profile.role}
    else:
        if not isinstance(declared_mapping_roles, Sequence) or isinstance(
            declared_mapping_roles, (str, bytes, bytearray)
        ):
            raise TypeError("dataset-neutral metric mapping roles must be a sequence")
        expected_mapping_roles = {
            str(value) for value in declared_mapping_roles
        }
        if not expected_mapping_roles:
            raise ValueError("dataset-neutral metric mapping roles cannot be empty")
    observed_mapping_roles = {str(value) for value in arrays["role"].tolist()}
    if observed_mapping_roles != expected_mapping_roles:
        raise ValueError("dataset-neutral metric mapping role mismatch")
    allowed_years = set(int(year) for year in profile.role_years)
    observed_years = {int(year) for year in arrays["target_year"].tolist()}
    if any(year not in allowed_years for year in observed_years):
        raise ValueError("dataset-neutral validation mapping role/year mismatch")
    if (
        profile.evaluation_access_audit.get("run_stage") == "final"
        and observed_years != allowed_years
    ):
        raise ValueError("final metric mapping does not cover every frozen role year")
    for index in range(window_count):
        declared_year = int(arrays["target_year"][index])
        try:
            timestamp_years = {
                datetime.fromtimestamp(int(value), tz=timezone.utc).year
                for value in arrays["target_time_utc"][index].tolist()
            }
        except (OverflowError, OSError, ValueError) as error:
            raise ValueError(
                "dataset-neutral validation target timestamp is invalid"
            ) from error
        if timestamp_years != {declared_year}:
            raise ValueError(
                "dataset-neutral validation target timestamp/year mismatch"
            )
    return arrays


def _profile_mapping_arrays(
    documents: Sequence[Mapping[str, Any]],
    digests: Sequence[str],
    *,
    profile: ValidationMetricRuntimeProfile,
) -> dict[str, np.ndarray]:
    """Resolve the explicit mapping schema selected by the frozen profile."""

    if profile.dataset_extension == "china_activity_event_v1":
        arrays = _mapping_arrays(documents, digests)
        declared = profile.data_contract.get("mapping_roles")
        if not isinstance(declared, Sequence) or isinstance(
            declared, (str, bytes, bytearray)
        ):
            raise TypeError("China metric mapping roles must be a sequence")
        expected_roles = {str(value) for value in declared}
        observed_roles = {str(value) for value in arrays["role"].tolist()}
        if not expected_roles or observed_roles != expected_roles:
            raise ValueError("China metric mapping role mismatch")
        expected_years = set(int(year) for year in profile.role_years)
        observed_years = {int(year) for year in arrays["target_year"].tolist()}
        if not observed_years <= expected_years:
            raise ValueError("China metric mapping role/year mismatch")
        if (
            profile.evaluation_access_audit.get("run_stage") == "final"
            and observed_years != expected_years
        ):
            raise ValueError("final China metric mapping lacks a frozen role year")
        return arrays
    return _dataset_neutral_mapping_arrays(documents, digests, profile=profile)


def _expected_dataset_neutral_alignment_tracker(
    *,
    profile: ValidationMetricRuntimeProfile,
    role: str,
    mapping: Mapping[str, np.ndarray],
    batch_mapping_digests: Sequence[str],
) -> dict[str, Any]:
    origins = [int(value) for value in mapping["forecast_origin_utc"].tolist()]
    if not origins:
        raise ValueError("dataset-neutral validation alignment cannot be empty")
    if len(origins) != len(set(origins)):
        raise ValueError("dataset-neutral validation alignment contains duplicate origins")
    if any(right <= left for left, right in zip(origins, origins[1:])):
        raise ValueError(
            "dataset-neutral validation alignment origins are not strictly ordered"
        )
    return {
        "schema_version": str(
            profile.data_contract["alignment_schema_version"]
        ),
        "dataset": str(profile.dataset),
        "role": str(role),
        "n_windows": len(origins),
        "n_batches": len(batch_mapping_digests),
        "first_forecast_origin_utc": origins[0],
        "last_forecast_origin_utc": origins[-1],
        "batch_mapping_digest_sha256": list(batch_mapping_digests),
        "alignment_digest_sha256": hashlib.sha256(
            _canonical_json_bytes(list(batch_mapping_digests))
        ).hexdigest(),
    }


def _verify_embedded_document_digest(
    document: Mapping[str, Any],
    *,
    digest_field: str,
    label: str,
) -> None:
    if not isinstance(document, Mapping):
        raise TypeError(f"dataset-neutral validation {label} must be an object")
    recorded = document.get(digest_field)
    if not _is_sha256(recorded):
        raise ValueError(
            f"dataset-neutral validation {label} has invalid {digest_field}"
        )
    actual = _document_digest_without(document, digest_field)
    if str(recorded) != actual:
        raise ValueError(
            f"dataset-neutral validation {label} digest mismatch: "
            f"recorded={recorded}, actual={actual}"
        )


def _verify_dataset_neutral_provenance(
    profile: ValidationMetricRuntimeProfile,
    provenance: Mapping[str, Any],
) -> frozenset[tuple[int, str, str]]:
    _require_contract_fields(
        provenance,
        required=(
            "schema_version",
            "dataset",
            "data_runtime",
            "resolved_runtime_identity",
        ),
        label="runtime provenance",
    )
    allowed_schemas = {
        "cac_r1.dataset_neutral_validation_provenance.v1",
        "cac_r1.dataset_neutral_metric_provenance.v1",
    }
    if provenance["schema_version"] not in allowed_schemas:
        raise ValueError("dataset-neutral metric provenance schema mismatch")
    if str(provenance["dataset"]) != profile.dataset:
        raise ValueError("dataset-neutral validation provenance dataset mismatch")

    data_runtime = provenance["data_runtime"]
    _require_contract_fields(
        data_runtime,
        required=(
            "dataset",
            "data_protocol_id",
            "data_build_sha256",
            "source_bundle_sha256",
            "source_files",
            "runtime_digest_sha256",
        ),
        label="data runtime provenance",
    )
    _verify_embedded_document_digest(
        data_runtime,
        digest_field="runtime_digest_sha256",
        label="data runtime provenance",
    )
    if str(data_runtime["dataset"]) != profile.dataset:
        raise ValueError("dataset-neutral validation data runtime dataset mismatch")
    if data_runtime["data_protocol_id"] != profile.data_contract["data_protocol_id"]:
        raise ValueError("dataset-neutral validation data protocol mismatch")
    if data_runtime["data_build_sha256"] != profile.data_contract["data_build_sha256"]:
        raise ValueError("dataset-neutral validation data build digest mismatch")
    if data_runtime["source_bundle_sha256"] != profile.data_contract[
        "source_bundle_sha256"
    ]:
        raise ValueError("dataset-neutral validation source bundle digest mismatch")
    source_files = data_runtime["source_files"]
    if not isinstance(source_files, Sequence) or isinstance(
        source_files, (str, bytes, bytearray)
    ) or not source_files:
        raise ValueError("dataset-neutral validation source-files provenance is empty")
    source_bindings: list[tuple[int, str, str]] = []
    for source in source_files:
        _require_contract_fields(
            source,
            ("target_year", "role", "source_file_hash"),
            "source-file provenance",
        )
        binding = (
            int(source["target_year"]),
            str(source["role"]),
            _require_sha256(source["source_file_hash"], "source-file digest"),
        )
        source_bindings.append(binding)
    if len(source_bindings) != len(set(source_bindings)):
        raise ValueError("dataset-neutral validation source-file provenance is duplicated")

    identity = provenance["resolved_runtime_identity"]
    _require_contract_fields(
        identity,
        required=(
            "dataset",
            "normalizer",
            "graph",
            "driver",
            "window",
            "runtime_identity_sha256",
        ),
        label="resolved runtime identity",
    )
    _verify_embedded_document_digest(
        identity,
        digest_field="runtime_identity_sha256",
        label="resolved runtime identity",
    )
    if str(identity["dataset"]) != profile.dataset:
        raise ValueError("dataset-neutral validation runtime identity dataset mismatch")
    normalizer = identity["normalizer"]
    if not isinstance(normalizer, Mapping) or normalizer.get(
        "normalizer_id"
    ) != profile.normalizer_contract["normalizer_id"]:
        raise ValueError("dataset-neutral validation normalizer identity mismatch")
    if "digest_sha256" in normalizer and normalizer[
        "digest_sha256"
    ] != profile.normalizer_contract["normalizer_digest_sha256"]:
        raise ValueError("dataset-neutral validation normalizer digest mismatch")

    graph = identity["graph"]
    driver = identity["driver"]
    if not isinstance(graph, Mapping):
        raise TypeError("dataset-neutral validation graph provenance must be an object")
    if not isinstance(driver, Mapping):
        raise TypeError("dataset-neutral validation driver provenance must be an object")
    _require_sha256(
        graph.get("adjacency_sha256"), label="graph adjacency digest"
    )
    _require_sha256(driver.get("source_sha256"), label="driver source digest")
    _require_sha256(driver.get("scaler_sha256"), label="driver scaler digest")

    runtime_window = identity["window"]
    if not isinstance(runtime_window, Mapping):
        raise TypeError("dataset-neutral validation runtime window must be an object")
    expected_window = profile.window_contract
    expected_values = {
        "lag_steps": int(expected_window["history_steps"]),
        "horizon_steps": int(expected_window["horizon_steps"]),
        "stride_steps": int(expected_window["stride_steps"]),
        "prefix_boundary": bool(expected_window["prefix_boundary"]),
    }
    for field, expected in expected_values.items():
        if runtime_window.get(field) != expected:
            raise ValueError(
                f"dataset-neutral validation runtime window mismatch: {field}"
            )
    expected_step_hours = float(expected_window["cadence_seconds"]) / 3600.0
    if not math.isclose(
        float(runtime_window.get("step_hours", float("nan"))),
        expected_step_hours,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("dataset-neutral validation runtime cadence mismatch")
    return frozenset(source_bindings)


def _verify_mapping_source_bindings(
    mapping: Mapping[str, np.ndarray],
    source_bindings: frozenset[tuple[int, str, str]],
) -> None:
    for year, role, source_hash in zip(
        mapping["target_year"].tolist(),
        mapping["role"].tolist(),
        mapping["source_file_hash"].tolist(),
    ):
        binding = (int(year), str(role), str(source_hash))
        if binding not in source_bindings:
            raise ValueError(
                "dataset-neutral validation mapping source-file provenance mismatch"
            )


@dataclass(frozen=True)
class DatasetNeutralValidationMetricPayload(ValidationMetricRuntimePayload):
    """Marker type using the shared validation payload verification stack."""

    schema_version: str = DATASET_NEUTRAL_VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION


@dataclass(frozen=True)
class DatasetNeutralEvaluationMetricPayload(ValidationMetricRuntimePayload):
    """Immutable native final-evaluation payload using the shared closure."""

    schema_version: str = DATASET_NEUTRAL_EVALUATION_METRIC_RUNTIME_SCHEMA_VERSION


def build_dataset_neutral_metric_runtime_payload(
    *,
    profile: ValidationMetricRuntimeProfile,
    epoch: int,
    role: str,
    prediction_tecu: Any,
    truth_tecu: Any,
    target_valid: Any,
    batch_mapping_documents: Sequence[Mapping[str, Any]],
    batch_mapping_digests: Sequence[str],
    runtime_provenance: Mapping[str, Any],
    alignment_tracker: Mapping[str, Any],
) -> ValidationMetricRuntimePayload:
    if not isinstance(profile, ValidationMetricRuntimeProfile):
        raise TypeError("dataset-neutral metric payload requires a runtime profile")
    profile.verify()
    if str(role) != profile.role:
        raise ValueError("dataset-neutral metric payload role/profile mismatch")
    prediction = _readonly_array(prediction_tecu, dtype=np.float32)
    truth = _readonly_array(truth_tecu, dtype=np.float32)
    valid = _readonly_array(target_valid, dtype=np.bool_)
    documents = tuple(_json_clone(document) for document in batch_mapping_documents)
    digests = tuple(str(digest) for digest in batch_mapping_digests)
    mapping = _profile_mapping_arrays(
        documents,
        digests,
        profile=profile,
    )
    if prediction.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError(
            "dataset-neutral metric prediction/truth/mask shape mismatch"
        )
    if prediction.ndim < 2:
        raise ValueError(
            "dataset-neutral metric arrays require an explicit horizon dimension"
        )
    if prediction.shape[0] != mapping["window_start_idx"].shape[0]:
        raise ValueError(
            "dataset-neutral metric arrays/mapping window count mismatch"
        )
    if prediction.shape[1] != mapping["target_time_utc"].shape[1]:
        raise ValueError(
            "dataset-neutral metric arrays/target timestamp horizon mismatch"
        )
    if prediction.shape[1] != int(profile.window_contract["horizon_steps"]):
        raise ValueError("dataset-neutral metric arrays/profile horizon mismatch")
    if profile.mask_contract["requires_all_valid"] and not bool(
        (valid & np.isfinite(truth)).all()
    ):
        raise RuntimeError(
            "dataset-neutral metric profile requires all-valid targets"
        )

    provenance = _json_clone(runtime_provenance)
    source_bindings = _verify_dataset_neutral_provenance(profile, provenance)
    _verify_mapping_source_bindings(mapping, source_bindings)
    require_all = bool(profile.mask_contract["requires_all_valid"])
    overall, horizons = _metric_statistics(
        prediction, truth, valid, require_all_valid=require_all
    )
    alignment = {
        "source": str(profile.data_contract["alignment_source"]),
        "tracker": _json_clone(alignment_tracker),
    }
    identities = _metric_array_identities(prediction, truth, valid, mapping)
    payload_type = (
        DatasetNeutralEvaluationMetricPayload
        if profile.role == "evaluation"
        else DatasetNeutralValidationMetricPayload
    )
    provisional = payload_type(
        profile=profile,
        epoch=int(epoch),
        role=str(role),
        prediction_tecu=prediction,
        truth_tecu=truth,
        target_valid=valid,
        metadata={},
        mapping=mapping,
        batch_mapping_documents=documents,
        batch_mapping_digests=digests,
        provenance=provenance,
        alignment_verification=alignment,
        overall=overall,
        horizons=horizons,
        array_identities=identities,
        payload_digest_sha256="",
    )
    payload = replace(
        provisional,
        payload_digest_sha256=hashlib.sha256(
            _canonical_json_bytes(provisional._summary_without_digest())
        ).hexdigest(),
    )
    payload.verify()
    return payload


def build_dataset_neutral_validation_metric_payload(
    *,
    profile: ValidationMetricRuntimeProfile,
    epoch: int,
    role: str,
    prediction_tecu: Any,
    truth_tecu: Any,
    target_valid: Any,
    batch_mapping_documents: Sequence[Mapping[str, Any]],
    batch_mapping_digests: Sequence[str],
    runtime_provenance: Mapping[str, Any],
    alignment_tracker: Mapping[str, Any],
) -> DatasetNeutralValidationMetricPayload:
    """Backward-compatible pilot validation builder."""

    if str(role) != "validation" or profile.role != "validation":
        raise ValueError("validation payload builder requires role=validation")
    payload = build_dataset_neutral_metric_runtime_payload(
        profile=profile,
        epoch=epoch,
        role=role,
        prediction_tecu=prediction_tecu,
        truth_tecu=truth_tecu,
        target_valid=target_valid,
        batch_mapping_documents=batch_mapping_documents,
        batch_mapping_digests=batch_mapping_digests,
        runtime_provenance=runtime_provenance,
        alignment_tracker=alignment_tracker,
    )
    if not isinstance(payload, DatasetNeutralValidationMetricPayload):
        raise TypeError("validation payload builder returned the wrong payload type")
    return payload


def partition_regression_statistics(
    prediction: Any,
    target: Any,
    target_valid: Any,
    groups: Mapping[str, Any],
    *,
    strict_china_st8: bool = False,
) -> tuple[RegressionSufficientStatistics, dict[str, RegressionSufficientStatistics]]:
    if not groups:
        raise ValueError("partition groups cannot be empty")
    prediction_array, target_array, valid = _validated_numpy_inputs(
        prediction,
        target,
        target_valid,
        strict_china_st8=strict_china_st8,
    )
    normalized: dict[str, np.ndarray] = {}
    for name, mask in groups.items():
        mask_array = _to_numpy(mask)
        if mask_array.dtype != np.bool_:
            raise TypeError(f"partition mask must be bool: {name}")
        if mask_array.shape != valid.shape:
            raise ValueError(
                f"partition mask shape mismatch for {name}: {mask_array.shape} != {valid.shape}"
            )
        normalized[str(name)] = mask_array
    membership = np.zeros(valid.shape, dtype=np.uint16)
    for mask_array in normalized.values():
        membership += mask_array.astype(np.uint16, copy=False)
    if not bool(np.all(membership[valid] == 1)):
        raise ValueError("partition masks must be mutually exclusive and exhaustive over valid elements")

    overall = masked_regression_statistics(
        prediction_array,
        target_array,
        valid,
        strict_china_st8=False,
    )
    partitions = {
        name: masked_regression_statistics(
            prediction_array,
            target_array,
            valid & mask_array,
            strict_china_st8=False,
        )
        for name, mask_array in normalized.items()
    }
    return overall, partitions


def assert_partition_closure(
    overall: RegressionSufficientStatistics,
    partitions: Mapping[str, RegressionSufficientStatistics],
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-12,
) -> dict[str, Any]:
    if not partitions:
        raise ValueError("partition closure requires at least one partition")
    pooled = merge_statistics(partitions.values())
    checks = []
    for field in STAT_FIELDS:
        expected = getattr(overall, field)
        observed = getattr(pooled, field)
        if field == "n_valid":
            tolerance = 0.0
            passed = int(expected) == int(observed)
            delta = float(int(observed) - int(expected))
        else:
            tolerance = max(
                float(absolute_tolerance),
                float(relative_tolerance) * max(1.0, abs(float(expected))),
            )
            delta = float(observed) - float(expected)
            passed = math.isfinite(delta) and abs(delta) <= tolerance
        checks.append(
            {
                "field": field,
                "expected": expected,
                "observed": observed,
                "delta": delta,
                "tolerance": tolerance,
                "pass": bool(passed),
            }
        )
    passed = all(check["pass"] for check in checks)
    if not passed:
        raise ValueError(f"metric partition closure failed: {checks}")
    return {
        "schema_version": "cac_r1.metric_partition_closure.v1",
        "metric_protocol_id": METRIC_PROTOCOL_ID,
        "partition_count": len(partitions),
        "checks": checks,
        "pass": True,
    }


__all__ = [
    "DATASET_NEUTRAL_EVALUATION_METRIC_RUNTIME_SCHEMA_VERSION",
    "DATASET_NEUTRAL_VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION",
    "METRIC_PROTOCOL_ID",
    "SUPERVISED_LOSS_POLICY_SCHEMA_VERSION",
    "SUPERVISED_LOSS_RUNTIME_TELEMETRY_SCHEMA_VERSION",
    "TRAINING_OBJECTIVE_SCHEMA_VERSION",
    "DatasetNeutralEvaluationMetricPayload",
    "DatasetNeutralValidationMetricPayload",
    "RegressionSufficientStatistics",
    "StrictSupervisedLoss",
    "SupervisedLossPolicy",
    "VALIDATION_METRIC_RUNTIME_SCHEMA_VERSION",
    "VALIDATION_METRIC_RUNTIME_PROFILE_SCHEMA_VERSION",
    "ValidationMetricRuntimePayload",
    "ValidationMetricRuntimeProfile",
    "assert_partition_closure",
    "build_strict_supervised_loss",
    "build_supervised_loss_policy",
    "build_dataset_neutral_metric_runtime_payload",
    "build_dataset_neutral_validation_metric_payload",
    "build_validation_metric_runtime_payload",
    "build_validation_metric_runtime_profile",
    "canonical_training_objective_bytes",
    "masked_loss_from_callable",
    "masked_mae_loss",
    "masked_mse_loss",
    "masked_regression_statistics",
    "masked_rmse_loss",
    "merge_statistics",
    "partition_regression_statistics",
    "supervised_loss_policy_from_mapping",
    "training_objective_sha256",
]
