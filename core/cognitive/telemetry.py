"""
Telemetry dataclasses for biofeedback self-monitoring.

Provides JSON-serializable summaries of model dynamics during generation,
enabling self-monitoring through confidence and strain signals.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Epsilon for division safety
EPS = 1e-8


@dataclass
class AggregateStats:
    """Rolling statistics for a single metric."""

    mean: float
    std: float
    min: float
    max: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_values(cls, values: list[float]) -> AggregateStats:
        """Create stats from a list of values."""
        if not values:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0, n_samples=0)

        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            n_samples=len(values),
        )


@dataclass
class TelemetrySummary:
    """Per-generation telemetry aggregates. JSON-serializable, no tensors."""

    # Metadata
    cycle_id: str
    timestamp: datetime
    model_id: str
    prompt_tokens: int
    generated_tokens: int
    sample_every_n_generated_tokens: int  # e.g., 4

    # Layer configuration (not baked into field names)
    residual_layers: list[int]  # e.g., [4, 16, 28]
    attention_layer: Optional[int]  # e.g., 16, or None if not captured

    # Logit dynamics (confidence proxy)
    logit_entropy: AggregateStats  # lower = more confident
    top1_top2_margin: AggregateStats  # higher = more confident

    # Residual stream (strain proxy)
    # Keyed by layer index for flexibility
    residual_norms: dict[int, AggregateStats]  # {4: stats, 16: stats, 28: stats}
    residual_slope: float  # (late_mean - early_mean) / max(early_mean, EPS)

    # Attention (optional v0, can be None)
    attention_entropy: Optional[AggregateStats]  # avg across heads

    # Derived scores (computed from z-scores, normalized 0-1)
    confidence_score: float = 0.0
    strain_score: float = 0.0

    # Raw z-scores (for debugging/tuning)
    margin_z: float = 0.0
    entropy_z: float = 0.0
    slope_z: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        d = {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "sample_every_n_generated_tokens": self.sample_every_n_generated_tokens,
            "residual_layers": self.residual_layers,
            "attention_layer": self.attention_layer,
            "logit_entropy": self.logit_entropy.to_dict(),
            "top1_top2_margin": self.top1_top2_margin.to_dict(),
            "residual_norms": {k: v.to_dict() for k, v in self.residual_norms.items()},
            "residual_slope": self.residual_slope,
            "attention_entropy": self.attention_entropy.to_dict() if self.attention_entropy else None,
            "confidence_score": self.confidence_score,
            "strain_score": self.strain_score,
            "margin_z": self.margin_z,
            "entropy_z": self.entropy_z,
            "slope_z": self.slope_z,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TelemetrySummary:
        """Reconstruct from JSON dict."""
        return cls(
            cycle_id=d["cycle_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            model_id=d["model_id"],
            prompt_tokens=d["prompt_tokens"],
            generated_tokens=d["generated_tokens"],
            sample_every_n_generated_tokens=d["sample_every_n_generated_tokens"],
            residual_layers=d["residual_layers"],
            attention_layer=d.get("attention_layer"),
            logit_entropy=AggregateStats(**d["logit_entropy"]),
            top1_top2_margin=AggregateStats(**d["top1_top2_margin"]),
            residual_norms={int(k): AggregateStats(**v) for k, v in d["residual_norms"].items()},
            residual_slope=d["residual_slope"],
            attention_entropy=AggregateStats(**d["attention_entropy"]) if d.get("attention_entropy") else None,
            confidence_score=d.get("confidence_score", 0.0),
            strain_score=d.get("strain_score", 0.0),
            margin_z=d.get("margin_z", 0.0),
            entropy_z=d.get("entropy_z", 0.0),
            slope_z=d.get("slope_z", 0.0),
        )


@dataclass
class TriggerState:
    """Computed by Reflexion from TelemetrySummary + rolling baselines."""

    # Raw z-scores
    margin_z: float
    entropy_z: float
    slope_z: float

    # Trigger predicates
    confidence_high: bool  # margin_z > k AND entropy_z < -k
    strain_high: bool  # slope_z > k

    # Combined trigger
    should_verify: bool  # confidence_high AND strain_high

    # Metadata
    baseline_samples: int  # How many cycles in baseline
    triggered_reason: Optional[str] = None  # Human-readable if triggered

    # Shadow mode tracking
    would_trigger_if_active: bool = False  # For calibration when disabled

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricBaseline:
    """Rolling baseline for a single metric."""

    mean: float = 0.0
    std: float = 0.0
    n_samples: int = 0
    values: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def update(self, new_value: float) -> None:
        """Add new value, maintain rolling window, recompute stats."""
        self.values.append(new_value)
        self.n_samples = len(self.values)

        if self.n_samples > 0:
            self.mean = statistics.mean(self.values)
            self.std = statistics.stdev(self.values) if self.n_samples > 1 else 0.0

    def compute_z_score(self, value: float) -> float:
        """Compute z-score with epsilon guard for small std."""
        safe_std = max(self.std, EPS)
        return (value - self.mean) / safe_std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "n_samples": self.n_samples,
            # Don't serialize the full deque for compactness
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetricBaseline:
        return cls(
            mean=d.get("mean", 0.0),
            std=d.get("std", 0.0),
            n_samples=d.get("n_samples", 0),
        )


def compute_residual_slope(early_mean: float, late_mean: float) -> float:
    """Compute relative slope: how much late layers amplify vs early."""
    return (late_mean - early_mean) / max(early_mean, EPS)


def normalize_score(z_score: float, scale: float = 3.0) -> float:
    """Normalize z-score to 0-1 range using sigmoid-like transform."""
    # Clamp to prevent overflow (exp(~700) overflows float64)
    clamped = max(-100.0, min(100.0, z_score / scale))

    # Map z-score through sigmoid, then scale to 0-1
    # z=0 -> 0.5, z=+3 -> ~0.73, z=-3 -> ~0.27
    return 1.0 / (1.0 + math.exp(-clamped))
