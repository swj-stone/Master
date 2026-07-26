"""Strict, redacted progress state and append-only run logging."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_DATASET_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
)
_ALGORITHMS = frozenset({"exact", "random", "guided", "complete"})
_PROGRESS_FIELDS = frozenset(
    {
        "request_digest",
        "schedule_digest",
        "dataset_id",
        "seed",
        "algorithm",
        "feature_index",
        "feature_count",
        "budget",
        "max_budget",
        "physical_model_fit_count",
        "cache_hit_count",
        "elapsed_s",
        "eta_s",
        "last_completed_trial_index",
    }
)


def _exact_nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _exact_positive_int(value: object, context: str) -> int:
    result = _exact_nonnegative_int(value, context)
    if result == 0:
        raise ValueError(f"{context} must be a positive integer")
    return result


def _nonnegative_finite_float(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{context} must be finite and non-negative")
    return float(value)


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class EstimatorProgressEvent:
    """One redacted, request-bound estimator progress snapshot."""

    request_digest: str
    schedule_digest: str
    dataset_id: str
    seed: int
    algorithm: str
    feature_index: int
    feature_count: int
    budget: int
    max_budget: int
    physical_model_fit_count: int
    cache_hit_count: int
    elapsed_s: float
    eta_s: float | None

    def __post_init__(self) -> None:
        request_digest = _sha256(
            self.request_digest, "progress request_digest"
        )
        schedule_digest = _sha256(
            self.schedule_digest, "progress schedule_digest"
        )
        if (
            not isinstance(self.dataset_id, str)
            or _SAFE_DATASET_RE.fullmatch(self.dataset_id) is None
            or self.dataset_id in {".", ".."}
        ):
            raise ValueError("progress dataset_id is invalid")
        self.dataset_id.encode("utf-8", errors="strict")
        seed = _exact_nonnegative_int(self.seed, "progress seed")
        if self.algorithm not in _ALGORITHMS:
            raise ValueError("progress algorithm is invalid")
        feature_count = _exact_positive_int(
            self.feature_count, "progress feature_count"
        )
        feature_index = _exact_nonnegative_int(
            self.feature_index, "progress feature_index"
        )
        if feature_index > feature_count:
            raise ValueError("progress feature_index exceeds feature_count")
        max_budget = _exact_positive_int(
            self.max_budget, "progress max_budget"
        )
        budget = _exact_nonnegative_int(self.budget, "progress budget")
        if budget > max_budget:
            raise ValueError("progress budget exceeds max_budget")
        physical_fits = _exact_nonnegative_int(
            self.physical_model_fit_count,
            "progress physical_model_fit_count",
        )
        cache_hits = _exact_nonnegative_int(
            self.cache_hit_count, "progress cache_hit_count"
        )
        elapsed = _nonnegative_finite_float(
            self.elapsed_s, "progress elapsed_s"
        )
        eta = (
            None
            if self.eta_s is None
            else _nonnegative_finite_float(self.eta_s, "progress eta_s")
        )

        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "schedule_digest", schedule_digest)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "feature_index", feature_index)
        object.__setattr__(self, "feature_count", feature_count)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "max_budget", max_budget)
        object.__setattr__(
            self, "physical_model_fit_count", physical_fits
        )
        object.__setattr__(self, "cache_hit_count", cache_hits)
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "eta_s", eta)


@dataclass(frozen=True)
class EstimatorProgressState:
    """Canonical durable progress payload."""

    event: EstimatorProgressEvent
    last_completed_trial_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.event, EstimatorProgressEvent):
            raise TypeError("progress state event is invalid")
        trial_index = _exact_nonnegative_int(
            self.last_completed_trial_index,
            "progress last_completed_trial_index",
        )
        object.__setattr__(
            self, "last_completed_trial_index", trial_index
        )


def progress_to_document(
    event: EstimatorProgressEvent,
    *,
    last_completed_trial_index: int,
) -> dict[str, object]:
    """Return the exact canonical JSON-compatible progress schema."""
    state = EstimatorProgressState(event, last_completed_trial_index)
    return {
        "request_digest": state.event.request_digest,
        "schedule_digest": state.event.schedule_digest,
        "dataset_id": state.event.dataset_id,
        "seed": state.event.seed,
        "algorithm": state.event.algorithm,
        "feature_index": state.event.feature_index,
        "feature_count": state.event.feature_count,
        "budget": state.event.budget,
        "max_budget": state.event.max_budget,
        "physical_model_fit_count": (
            state.event.physical_model_fit_count
        ),
        "cache_hit_count": state.event.cache_hit_count,
        "elapsed_s": state.event.elapsed_s,
        "eta_s": state.event.eta_s,
        "last_completed_trial_index": (
            state.last_completed_trial_index
        ),
    }


def progress_from_document(value: object) -> EstimatorProgressState:
    """Strictly validate one canonical progress document payload."""
    if not isinstance(value, dict) or set(value) != _PROGRESS_FIELDS:
        raise ValueError("estimator progress schema is invalid")
    event = EstimatorProgressEvent(
        request_digest=value["request_digest"],
        schedule_digest=value["schedule_digest"],
        dataset_id=value["dataset_id"],
        seed=value["seed"],
        algorithm=value["algorithm"],
        feature_index=value["feature_index"],
        feature_count=value["feature_count"],
        budget=value["budget"],
        max_budget=value["max_budget"],
        physical_model_fit_count=value["physical_model_fit_count"],
        cache_hit_count=value["cache_hit_count"],
        elapsed_s=value["elapsed_s"],
        eta_s=value["eta_s"],
    )
    return EstimatorProgressState(
        event=event,
        last_completed_trial_index=value["last_completed_trial_index"],
    )


def estimate_eta_s(
    completion_timestamps: Sequence[float],
    remaining_trial_upper_bound: int,
) -> float | None:
    """Estimate remaining seconds from the exact recent-fit throughput."""
    remaining = _exact_nonnegative_int(
        remaining_trial_upper_bound,
        "remaining trial upper bound",
    )
    timestamps = tuple(
        _nonnegative_finite_float(value, "completion timestamp")
        for value in completion_timestamps
    )
    if any(
        following < current
        for current, following in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("completion timestamps must be monotone")
    if len(timestamps) < 2:
        return None
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0.0:
        return None
    throughput = (len(timestamps) - 1) / elapsed
    eta = remaining / throughput
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("estimated ETA must be finite and non-negative")
    return eta


def _format_duration(seconds: float) -> str:
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{rounded}s"
    if rounded < 3600:
        return f"{int(round(rounded / 60))}m"
    hours, remainder = divmod(rounded, 3600)
    minutes = int(round(remainder / 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}h{minutes:02d}m"


def format_progress(event: EstimatorProgressEvent) -> str:
    """Format only explicitly redacted progress metadata."""
    if not isinstance(event, EstimatorProgressEvent):
        raise TypeError("event must be an EstimatorProgressEvent")
    eta = "?" if event.eta_s is None else _format_duration(event.eta_s)
    return (
        f"[{event.dataset_id} seed={event.seed} {event.algorithm}] "
        f"feature {event.feature_index}/{event.feature_count}, "
        f"K={event.budget}/{event.max_budget} "
        f"fits={event.physical_model_fit_count} "
        f"cache_hits={event.cache_hit_count} "
        f"elapsed={_format_duration(event.elapsed_s)} ETA={eta}"
    )


def append_run_log(path: Path, message: str) -> None:
    """Append one UTC-timestamped lifecycle line without replacing the log."""
    path = Path(path)
    if not isinstance(message, str):
        raise TypeError("log message must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")


def emit_progress(path: Path, event: EstimatorProgressEvent) -> None:
    """Print and append one redacted formatted progress line."""
    line = format_progress(event)
    print(line, flush=True)
    append_run_log(path, f"progress {line}")


__all__ = [
    "EstimatorProgressEvent",
    "EstimatorProgressState",
    "append_run_log",
    "emit_progress",
    "estimate_eta_s",
    "format_progress",
    "progress_from_document",
    "progress_to_document",
]
