"""Strict CAC-R1 predictor-driver scaling and history visibility contracts.

The physics path continues to consume raw driver values.  This module owns a
separate, immutable metadata envelope for predictor history and applies only a
precomputed train-only scaler.  It never refits during validation/evaluation,
clips indices, pads missing observations, or substitutes synthetic values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


DRIVER_CHANNEL_ORDER = ("AE", "Dst", "Kp", "F107", "cosSZA")
SCALER_SCHEMA_VERSION = "cac_r1.driver_scaler.v1"
SCALER_STATUS = "FROZEN_MACHINE_DERIVED"
TRAIN_ZSCORE = "train_zscore"
RAW_SCALING = "raw"
_VALID_SCALINGS = frozenset({TRAIN_ZSCORE, RAW_SCALING})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _channel_order(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("driver channel order must be an explicit sequence")
    order = tuple(str(item) for item in value)
    if order != DRIVER_CHANNEL_ORDER:
        raise ValueError(
            f"driver channel order mismatch: expected {DRIVER_CHANNEL_ORDER}, got {order}"
        )
    return order


def _integer_tensor(value: Any, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if torch.is_floating_point(tensor):
        if not bool(torch.isfinite(tensor).all()) or not bool(torch.equal(tensor, tensor.round())):
            raise ValueError(f"{label} must contain finite integer Unix seconds")
    elif tensor.is_complex():
        raise ValueError(f"{label} must contain integer Unix seconds")
    return tensor.to(dtype=torch.int64)


def validate_history_visibility(
    history_timestamps: Any,
    forecast_origins: Any,
    *,
    expected_history_steps: int,
    cadence_seconds: int = 7200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate exact ``[origin-lag, origin)`` timestamps without fallback."""

    steps = int(expected_history_steps)
    cadence = int(cadence_seconds)
    if steps <= 0 or cadence <= 0:
        raise ValueError("history steps and cadence must be positive")
    timestamps = _integer_tensor(history_timestamps, "history timestamps")
    origins = _integer_tensor(forecast_origins, "forecast origins")
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    if timestamps.ndim != 2:
        raise ValueError("history timestamps must have shape [L] or [B,L]")
    if origins.ndim == 0:
        origins = origins.reshape(1)
    if origins.ndim != 1:
        raise ValueError("forecast origins must be scalar or shape [B]")
    if int(timestamps.size(1)) != steps:
        raise ValueError(f"expected {steps} historical timestamps")
    if int(timestamps.size(0)) != int(origins.numel()):
        raise ValueError("history timestamp/origin batch mismatch")
    if bool((timestamps >= origins[:, None]).any()):
        raise ValueError("future driver access is forbidden; history must end before origin")
    offsets = torch.arange(steps, 0, -1, dtype=torch.int64) * cadence
    expected = origins[:, None] - offsets[None, :]
    if not bool(torch.equal(timestamps, expected)):
        raise ValueError("driver history is not the exact gap-free [origin-lag,origin) grid")
    return timestamps, origins


@dataclass(frozen=True)
class TrainOnlyDriverStatistics:
    """Machine-derived statistics after unique-timestamp train filtering."""

    train_timestamps: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray
    counts: np.ndarray
    channel_order: tuple[str, ...] = DRIVER_CHANNEL_ORDER


def fit_train_only_driver_statistics(
    timestamps: Any,
    values: Any,
    roles: Any,
    *,
    channel_order: Sequence[str],
) -> TrainOnlyDriverStatistics:
    """Fit population statistics after deterministic timestamp de-duplication.

    ``timestamps`` and ``roles`` may carry overlapping-window leading axes.
    ``values`` must have the same leading shape followed by ``[N,5]``.  Copies
    of one timestamp must be byte-for-byte equal; conflicting duplicates are a
    protocol error rather than an invitation to choose one occurrence.
    """

    order = _channel_order(channel_order)
    raw_timestamps = np.asarray(timestamps)
    if raw_timestamps.dtype.kind in "fc":
        if not np.isfinite(raw_timestamps).all() or not np.equal(
            raw_timestamps, np.round(raw_timestamps)
        ).all():
            raise ValueError("timestamps must contain finite integer Unix seconds")
    timestamp_array = raw_timestamps.astype(np.int64, copy=False)
    role_array = np.asarray(roles)
    value_array = np.asarray(values)
    if timestamp_array.shape != role_array.shape:
        raise ValueError("roles must have the same shape as timestamps")
    if value_array.ndim != timestamp_array.ndim + 2:
        raise ValueError("driver values must have shape timestamps+[N,5]")
    if tuple(value_array.shape[: timestamp_array.ndim]) != tuple(timestamp_array.shape):
        raise ValueError("driver value leading axes must match timestamps")
    if int(value_array.shape[-1]) != len(order):
        raise ValueError("driver channel count mismatch")
    if int(value_array.shape[-2]) <= 0 or timestamp_array.size == 0:
        raise ValueError("driver fit input must be non-empty")
    if value_array.dtype.kind not in "fc":
        raise ValueError("driver fit values must be floating")

    flat_timestamps = timestamp_array.reshape(-1)
    flat_roles = role_array.astype(str).reshape(-1)
    flat_values = value_array.reshape(flat_timestamps.size, value_array.shape[-2], len(order))
    train_occurrences = flat_roles == "train"
    if not bool(train_occurrences.any()):
        raise ValueError("no train timestamps are available for driver scaler fitting")
    # Do not inspect validation/evaluation driver values at all.  Filtering by
    # role metadata happens before finite checks, duplicate comparison, or any
    # numerical reduction.
    flat_timestamps = flat_timestamps[train_occurrences]
    flat_values = flat_values[train_occurrences]
    if not np.isfinite(flat_values).all():
        raise ValueError("train driver fit values contain non-finite data")
    permutation = np.argsort(flat_timestamps, kind="mergesort")
    flat_timestamps = flat_timestamps[permutation]
    flat_values = flat_values[permutation]

    unique_timestamps: list[int] = []
    unique_values: list[np.ndarray] = []
    cursor = 0
    while cursor < flat_timestamps.size:
        end = cursor + 1
        while end < flat_timestamps.size and flat_timestamps[end] == flat_timestamps[cursor]:
            end += 1
        group_values = flat_values[cursor:end]
        reference = group_values[0]
        if any(not np.array_equal(candidate, reference) for candidate in group_values[1:]):
            raise ValueError(
                f"conflicting duplicate timestamp {int(flat_timestamps[cursor])}: driver values differ"
            )
        unique_timestamps.append(int(flat_timestamps[cursor]))
        unique_values.append(reference.copy())
        cursor = end

    train_timestamps = np.asarray(unique_timestamps, dtype=np.int64)
    train_values = np.stack(unique_values, axis=0).astype(np.float64, copy=False)

    channel_vectors: list[np.ndarray] = []
    for channel_index, channel in enumerate(order):
        values_at_channel = train_values[..., channel_index]
        if channel != "cosSZA":
            reference = values_at_channel[:, :1]
            if not np.array_equal(values_at_channel, np.broadcast_to(reference, values_at_channel.shape)):
                raise ValueError(f"global driver channel {channel} is not identical across nodes")
            values_at_channel = reference[:, 0]
        else:
            values_at_channel = values_at_channel.reshape(-1)
        channel_vectors.append(np.asarray(values_at_channel, dtype=np.float64))

    means = np.asarray([vector.mean(dtype=np.float64) for vector in channel_vectors])
    standard_deviations = np.asarray(
        [vector.std(dtype=np.float64, ddof=0) for vector in channel_vectors]
    )
    counts = np.asarray([vector.size for vector in channel_vectors], dtype=np.int64)
    if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
        raise ValueError("driver scaler statistics are non-finite")
    if bool((standard_deviations <= 0.0).any()):
        raise ValueError("driver scaler requires positive finite standard deviations")
    return TrainOnlyDriverStatistics(
        train_timestamps=train_timestamps,
        means=means,
        standard_deviations=standard_deviations,
        counts=counts,
    )


@dataclass(frozen=True)
class FrozenDriverScaler:
    """Verified, train-only, dataset-specific driver z-score scaler."""

    dataset: str
    scaler_id: str
    file_sha256: str
    source_driver_relative_path: str
    source_driver_sha256: str
    channel_order: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    counts: tuple[int, ...]
    timestamp_count: int
    unique_timestamp_count: int
    num_nodes: int
    grid_shape: tuple[int, int]
    fit_train_years: tuple[int, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_file_sha256: str,
        expected_source_sha256: str,
        expected_dataset: str | None = None,
    ) -> "FrozenDriverScaler":
        scaler_path = Path(path)
        expected_file = _validated_sha256(expected_file_sha256, "expected scaler file SHA-256")
        actual_file = _sha256_file(scaler_path)
        if actual_file != expected_file:
            raise ValueError(
                f"scaler file SHA-256 mismatch: expected {expected_file}, got {actual_file}"
            )
        with scaler_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, Mapping):
            raise ValueError("driver scaler document must be a JSON object")
        if document.get("schema_version") != SCALER_SCHEMA_VERSION:
            raise ValueError("unsupported driver scaler schema version")
        if document.get("status") != SCALER_STATUS:
            raise ValueError("driver scaler is not frozen")
        dataset = str(document.get("dataset", ""))
        if not dataset or (expected_dataset is not None and dataset != str(expected_dataset)):
            raise ValueError(
                f"driver scaler dataset mismatch: expected {expected_dataset}, got {dataset}"
            )
        order = _channel_order(document.get("channel_order", ()))
        source_sha = _validated_sha256(
            document.get("source_driver_sha256"), "scaler source driver SHA-256"
        )
        expected_source = _validated_sha256(
            expected_source_sha256, "expected source driver SHA-256"
        )
        if source_sha != expected_source:
            raise ValueError(
                f"source driver SHA-256 mismatch: expected {expected_source}, got {source_sha}"
            )

        if document.get("fit_roles") != ["train"]:
            raise ValueError("driver scaler must be fit from the train role only")
        if document.get("imputation_policy") != "none":
            raise ValueError("driver scaler imputation is forbidden")
        role_access = document.get("role_access")
        if not isinstance(role_access, Mapping) or role_access.get("included_roles") != ["train"]:
            raise ValueError("driver scaler role-access audit is malformed")
        for field in (
            "validation_values_included",
            "evaluation_values_included",
            "non_train_values_included",
        ):
            if role_access.get(field) != 0:
                raise ValueError(f"driver scaler contains forbidden non-train values: {field}")
        algorithm = document.get("algorithm")
        required_algorithm = {
            "timestamp_weighting": "each_unique_train_timestamp_once",
            "global_channel_weighting": "one_value_per_timestamp",
            "cosSZA_weighting": "one_value_per_timestamp_per_node",
            "zscore": "population_mean_and_standard_deviation",
            "ddof": 0,
        }
        if not isinstance(algorithm, Mapping) or any(
            algorithm.get(key) != value for key, value in required_algorithm.items()
        ):
            raise ValueError("driver scaler algorithm contract mismatch")

        timestamp_count = int(document.get("timestamp_count", -1))
        unique_timestamp_count = int(document.get("unique_timestamp_count", -1))
        if timestamp_count <= 0 or unique_timestamp_count != timestamp_count:
            raise ValueError("driver scaler timestamps are not unique and non-empty")
        num_nodes = int(document.get("num_nodes", -1))
        grid_shape_value = document.get("grid_shape")
        if not isinstance(grid_shape_value, list) or len(grid_shape_value) != 2:
            raise ValueError("driver scaler grid shape is malformed")
        grid_shape = (int(grid_shape_value[0]), int(grid_shape_value[1]))
        if num_nodes <= 0 or math.prod(grid_shape) != num_nodes:
            raise ValueError("driver scaler node count/grid shape mismatch")

        channels = document.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != set(order):
            raise ValueError("driver scaler channel definitions mismatch")
        means: list[float] = []
        standard_deviations: list[float] = []
        counts: list[int] = []
        for channel in order:
            entry = channels[channel]
            if not isinstance(entry, Mapping):
                raise ValueError(f"driver scaler channel {channel} is malformed")
            mean = float(entry.get("mean", float("nan")))
            standard_deviation = float(entry.get("standard_deviation", float("nan")))
            count = int(entry.get("count", -1))
            if not math.isfinite(mean):
                raise ValueError(f"driver scaler channel {channel} has a non-finite mean")
            if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
                raise ValueError(
                    f"driver scaler channel {channel} requires a positive finite standard deviation"
                )
            expected_count = timestamp_count * num_nodes if channel == "cosSZA" else timestamp_count
            if count != expected_count:
                raise ValueError(
                    f"driver scaler channel {channel} count mismatch: expected {expected_count}, got {count}"
                )
            _validated_sha256(
                entry.get("input_values_sha256"), f"driver scaler channel {channel} input SHA-256"
            )
            means.append(mean)
            standard_deviations.append(standard_deviation)
            counts.append(count)

        relative_path = str(document.get("source_driver_relative_path", ""))
        relative = Path(relative_path)
        if not relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source driver relative path is not portable")
        train_years = tuple(int(year) for year in document.get("fit_train_years", ()))
        if not train_years or len(set(train_years)) != len(train_years):
            raise ValueError("driver scaler train years are missing or duplicated")
        return cls(
            dataset=dataset,
            scaler_id=str(document.get("scaler_id", "")),
            file_sha256=actual_file,
            source_driver_relative_path=relative_path,
            source_driver_sha256=source_sha,
            channel_order=order,
            means=tuple(means),
            standard_deviations=tuple(standard_deviations),
            counts=tuple(counts),
            timestamp_count=timestamp_count,
            unique_timestamp_count=unique_timestamp_count,
            num_nodes=num_nodes,
            grid_shape=grid_shape,
            fit_train_years=train_years,
        )

    def verify_source_file(self, path: str | Path) -> str:
        actual = _sha256_file(Path(path))
        if actual != self.source_driver_sha256:
            raise ValueError(
                "source driver SHA-256 mismatch: "
                f"expected {self.source_driver_sha256}, got {actual}"
            )
        return actual

    def transform_history(
        self,
        values: torch.Tensor,
        *,
        channel_order: Sequence[str],
        source_driver_sha256: str,
    ) -> torch.Tensor:
        _channel_order(channel_order)
        source_sha = _validated_sha256(source_driver_sha256, "history source driver SHA-256")
        if source_sha != self.source_driver_sha256:
            raise ValueError(
                f"source driver SHA-256 mismatch: expected {self.source_driver_sha256}, got {source_sha}"
            )
        if not isinstance(values, torch.Tensor) or values.ndim != 4:
            raise ValueError("driver history must be a torch tensor with shape [B,L,N,5]")
        if any(int(size) <= 0 for size in values.shape):
            raise ValueError("driver history dimensions must all be positive")
        if int(values.size(-1)) != len(self.channel_order):
            raise ValueError("driver channel count mismatch")
        if int(values.size(-2)) != self.num_nodes:
            raise ValueError(
                f"driver node count mismatch: expected {self.num_nodes}, got {int(values.size(-2))}"
            )
        if not torch.is_floating_point(values):
            raise ValueError("driver history must have a floating dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("driver history contains non-finite or missing values")
        means = torch.as_tensor(self.means, dtype=values.dtype, device=values.device)
        standard_deviations = torch.as_tensor(
            self.standard_deviations, dtype=values.dtype, device=values.device
        )
        transformed = (values - means) / standard_deviations
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError("standardized driver history contains non-finite values")
        return transformed


@dataclass(frozen=True)
class DriverHistoryBatch:
    """Strict predictor-only driver history with complete provenance metadata."""

    values: torch.Tensor
    timestamps: torch.Tensor
    forecast_origins: torch.Tensor
    channel_order: tuple[str, ...]
    source_driver_sha256: str
    driver_scaler_sha256: str
    scaling: str
    cadence_seconds: int = 7200

    def validate(
        self,
        *,
        expected_history_steps: int,
        expected_num_nodes: int,
        expected_source_driver_sha256: str,
        expected_driver_scaler_sha256: str,
        expected_scaling: str,
        expected_batch_size: int | None = None,
    ) -> None:
        _channel_order(self.channel_order)
        if self.scaling not in _VALID_SCALINGS or self.scaling != expected_scaling:
            raise ValueError(
                f"driver scaling mismatch: expected {expected_scaling}, got {self.scaling}"
            )
        source_sha = _validated_sha256(self.source_driver_sha256, "history source SHA-256")
        expected_source = _validated_sha256(
            expected_source_driver_sha256, "expected history source SHA-256"
        )
        if source_sha != expected_source:
            raise ValueError(
                f"source driver SHA-256 mismatch: expected {expected_source}, got {source_sha}"
            )
        scaler_sha = _validated_sha256(self.driver_scaler_sha256, "history scaler SHA-256")
        expected_scaler = _validated_sha256(
            expected_driver_scaler_sha256, "expected history scaler SHA-256"
        )
        if scaler_sha != expected_scaler:
            raise ValueError(
                f"driver scaler SHA-256 mismatch: expected {expected_scaler}, got {scaler_sha}"
            )
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 4:
            raise ValueError("driver history values must have shape [B,L,N,5]")
        if tuple(self.values.shape[1:]) != (
            int(expected_history_steps),
            int(expected_num_nodes),
            len(DRIVER_CHANNEL_ORDER),
        ):
            if int(self.values.size(-2)) != int(expected_num_nodes):
                raise ValueError(
                    "driver node count mismatch: "
                    f"expected {expected_num_nodes}, got {int(self.values.size(-2))}"
                )
            raise ValueError(
                "driver history shape mismatch: expected "
                f"[B,{expected_history_steps},{expected_num_nodes},5], got {tuple(self.values.shape)}"
            )
        if expected_batch_size is not None and int(self.values.size(0)) != int(expected_batch_size):
            raise ValueError("driver history batch-size mismatch")
        if not torch.is_floating_point(self.values) or not bool(torch.isfinite(self.values).all()):
            raise ValueError("driver history contains non-finite or non-floating values")
        timestamps, origins = validate_history_visibility(
            self.timestamps,
            self.forecast_origins,
            expected_history_steps=expected_history_steps,
            cadence_seconds=self.cadence_seconds,
        )
        if int(timestamps.size(0)) != int(self.values.size(0)) or int(origins.numel()) != int(
            self.values.size(0)
        ):
            raise ValueError("driver history metadata batch-size mismatch")

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DriverHistoryBatch":
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("driver history target dtype must be floating")
        return replace(self, values=self.values.to(device=device, dtype=dtype))


def extract_strict_driver_history(
    timeline_values: torch.Tensor,
    timeline_timestamps: Any,
    forecast_origins: Any,
    *,
    history_steps: int,
    cadence_seconds: int,
    channel_order: Sequence[str],
    source_driver_sha256: str,
    scaling: str,
    scaler: FrozenDriverScaler | None = None,
    bound_driver_scaler_sha256: str | None = None,
) -> DriverHistoryBatch:
    """Extract exact history frames from a gap-free timeline and optionally scale.

    Timeline values at the forecast origin or later are never gathered or
    inspected.  Only timestamp metadata is used to resolve exact historical
    indices, making future-value mutation irrelevant by construction.
    """

    order = _channel_order(channel_order)
    if scaling not in _VALID_SCALINGS:
        raise ValueError(f"unsupported driver scaling {scaling!r}")
    source_sha = _validated_sha256(source_driver_sha256, "source driver SHA-256")
    timestamps = _integer_tensor(timeline_timestamps, "timeline timestamps")
    if timestamps.ndim != 1 or timestamps.numel() == 0:
        raise ValueError("timeline timestamps must be a non-empty rank-1 array")
    cadence = int(cadence_seconds)
    steps = int(history_steps)
    if cadence <= 0 or steps <= 0:
        raise ValueError("history steps and cadence must be positive")
    if timestamps.numel() > 1 and not bool(torch.equal(
        torch.diff(timestamps),
        torch.full((timestamps.numel() - 1,), cadence, dtype=torch.int64),
    )):
        raise ValueError("driver timeline contains a timestamp gap or duplicate")
    if not isinstance(timeline_values, torch.Tensor) or timeline_values.ndim != 3:
        raise ValueError("timeline driver values must have shape [T,N,5]")
    if int(timeline_values.size(0)) != int(timestamps.numel()):
        raise ValueError("driver timeline value/timestamp length mismatch")
    if int(timeline_values.size(-1)) != len(order):
        raise ValueError("driver channel count mismatch")
    if not torch.is_floating_point(timeline_values):
        raise ValueError("timeline driver values must have a floating dtype")

    origins = _integer_tensor(forecast_origins, "forecast origins")
    if origins.ndim == 0:
        origins = origins.reshape(1)
    if origins.ndim != 1 or origins.numel() == 0:
        raise ValueError("forecast origins must be scalar or a non-empty rank-1 array")
    offsets = torch.arange(steps, 0, -1, dtype=torch.int64) * cadence
    expected_timestamps = origins[:, None] - offsets[None, :]
    indices = torch.searchsorted(timestamps, expected_timestamps)
    if bool((indices >= timestamps.numel()).any()):
        raise ValueError("requested driver history is out of range")
    resolved = timestamps[indices.clamp(max=timestamps.numel() - 1)]
    if not bool(torch.equal(resolved, expected_timestamps)):
        raise ValueError("requested driver history is out of range or has a timestamp gap")

    gather_indices = indices.to(device=timeline_values.device)
    raw_history = timeline_values[gather_indices].clone()
    if not bool(torch.isfinite(raw_history).all()):
        raise ValueError("selected driver history contains non-finite or missing values")
    if scaling == TRAIN_ZSCORE:
        if scaler is None:
            raise ValueError("train_zscore history requires a frozen driver scaler")
        if int(timeline_values.size(1)) != scaler.num_nodes:
            raise ValueError(
                f"driver node count mismatch: expected {scaler.num_nodes}, got {int(timeline_values.size(1))}"
            )
        if bound_driver_scaler_sha256 is not None:
            bound = _validated_sha256(
                bound_driver_scaler_sha256, "bound driver scaler SHA-256"
            )
            if bound != scaler.file_sha256:
                raise ValueError("bound driver scaler SHA-256 mismatch")
        transformed = scaler.transform_history(
            raw_history,
            channel_order=order,
            source_driver_sha256=source_sha,
        )
        scaler_sha = scaler.file_sha256
    else:
        if scaler is not None:
            raise ValueError("raw predictor history must not apply a z-score scaler")
        scaler_sha = _validated_sha256(
            bound_driver_scaler_sha256, "protocol-bound driver scaler SHA-256"
        )
        transformed = raw_history

    batch = DriverHistoryBatch(
        values=transformed,
        timestamps=expected_timestamps,
        forecast_origins=origins,
        channel_order=order,
        source_driver_sha256=source_sha,
        driver_scaler_sha256=scaler_sha,
        scaling=scaling,
        cadence_seconds=cadence,
    )
    batch.validate(
        expected_history_steps=steps,
        expected_num_nodes=int(timeline_values.size(1)),
        expected_source_driver_sha256=source_sha,
        expected_driver_scaler_sha256=scaler_sha,
        expected_scaling=scaling,
        expected_batch_size=int(origins.numel()),
    )
    return batch


__all__ = [
    "DRIVER_CHANNEL_ORDER",
    "DriverHistoryBatch",
    "FrozenDriverScaler",
    "RAW_SCALING",
    "SCALER_SCHEMA_VERSION",
    "TRAIN_ZSCORE",
    "TrainOnlyDriverStatistics",
    "extract_strict_driver_history",
    "fit_train_only_driver_statistics",
    "validate_history_visibility",
]
