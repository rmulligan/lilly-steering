"""
Telemetry evaluator for biofeedback self-monitoring.

Maintains rolling baselines, computes z-scores, and evaluates trigger conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.cognitive.telemetry import (
    EPS,
    MetricBaseline,
    TelemetrySummary,
    TriggerState,
    normalize_score,
)

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TelemetryBaselines:
    """Rolling baselines for all tracked metrics."""

    # Logit dynamics
    logit_entropy: MetricBaseline = field(default_factory=MetricBaseline)
    top1_top2_margin: MetricBaseline = field(default_factory=MetricBaseline)

    # Residual dynamics
    residual_slope: MetricBaseline = field(default_factory=MetricBaseline)

    # Optional attention
    attention_entropy: MetricBaseline = field(default_factory=MetricBaseline)

    def to_dict(self) -> dict:
        return {
            "logit_entropy": self.logit_entropy.to_dict(),
            "top1_top2_margin": self.top1_top2_margin.to_dict(),
            "residual_slope": self.residual_slope.to_dict(),
            "attention_entropy": self.attention_entropy.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> TelemetryBaselines:
        return cls(
            logit_entropy=MetricBaseline.from_dict(d.get("logit_entropy", {})),
            top1_top2_margin=MetricBaseline.from_dict(d.get("top1_top2_margin", {})),
            residual_slope=MetricBaseline.from_dict(d.get("residual_slope", {})),
            attention_entropy=MetricBaseline.from_dict(d.get("attention_entropy", {})),
        )


class TelemetryEvaluator:
    """Evaluates telemetry against rolling baselines to detect anomalies."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.baselines = TelemetryBaselines()

        # Configuration from settings (lowercase attribute names per pydantic-settings)
        self.trigger_k = getattr(settings, "telemetry_trigger_k", 2.0)
        self.min_baseline_samples = getattr(settings, "telemetry_min_baseline", 10)
        self.biofeedback_enabled = getattr(settings, "telemetry_biofeedback_v0", False)
        self.shadow_mode = getattr(settings, "telemetry_biofeedback_shadow", True)

    def evaluate(self, summary: TelemetrySummary) -> TriggerState:
        """
        Evaluate telemetry against baselines and determine trigger state.

        Updates baselines with current values after evaluation.
        """
        # Compute z-scores from current baselines
        margin_z = self.baselines.top1_top2_margin.compute_z_score(summary.top1_top2_margin.mean)
        entropy_z = self.baselines.logit_entropy.compute_z_score(summary.logit_entropy.mean)
        slope_z = self.baselines.residual_slope.compute_z_score(summary.residual_slope)

        # Update summary with z-scores
        summary.margin_z = margin_z
        summary.entropy_z = entropy_z
        summary.slope_z = slope_z

        # Compute normalized scores
        # Confidence: high margin (positive z) AND low entropy (negative z)
        # Average the absolute contributions
        confidence_signal = (margin_z - entropy_z) / 2.0  # High margin, low entropy -> positive
        summary.confidence_score = normalize_score(confidence_signal)

        # Strain: high slope (positive z)
        summary.strain_score = normalize_score(slope_z)

        # Evaluate trigger predicates
        # confidence_high: margin above baseline AND entropy below baseline
        confidence_high = (margin_z > self.trigger_k) and (entropy_z < -self.trigger_k)

        # strain_high: slope above baseline
        strain_high = slope_z > self.trigger_k

        # Combined trigger
        should_verify_raw = confidence_high and strain_high

        # Check if we have enough baseline samples
        has_sufficient_baseline = self.baselines.top1_top2_margin.n_samples >= self.min_baseline_samples

        # Only trigger if baseline is stable and biofeedback is enabled
        should_verify = should_verify_raw and has_sufficient_baseline and self.biofeedback_enabled

        # Build trigger reason
        triggered_reason = None
        if should_verify_raw:
            reasons = []
            if confidence_high:
                reasons.append(f"confidence_high (margin_z={margin_z:.2f}, entropy_z={entropy_z:.2f})")
            if strain_high:
                reasons.append(f"strain_high (slope_z={slope_z:.2f})")
            triggered_reason = "; ".join(reasons)

        trigger_state = TriggerState(
            margin_z=margin_z,
            entropy_z=entropy_z,
            slope_z=slope_z,
            confidence_high=confidence_high,
            strain_high=strain_high,
            should_verify=should_verify,
            baseline_samples=self.baselines.top1_top2_margin.n_samples,
            triggered_reason=triggered_reason,
            would_trigger_if_active=should_verify_raw and has_sufficient_baseline,
        )

        # Log in shadow mode
        if self.shadow_mode and should_verify_raw and not should_verify:
            if not has_sufficient_baseline:
                logger.info(
                    f"[Telemetry Shadow] Would trigger but baseline warming up "
                    f"({self.baselines.top1_top2_margin.n_samples}/{self.min_baseline_samples}): "
                    f"{triggered_reason}"
                )
            elif not self.biofeedback_enabled:
                logger.info(f"[Telemetry Shadow] Would trigger (biofeedback disabled): {triggered_reason}")

        if should_verify:
            logger.info(f"[Telemetry] Trigger activated: {triggered_reason}")

        # Update baselines with current values (after evaluation)
        self._update_baselines(summary)

        return trigger_state

    def _update_baselines(self, summary: TelemetrySummary) -> None:
        """Update rolling baselines with current summary values."""
        self.baselines.logit_entropy.update(summary.logit_entropy.mean)
        self.baselines.top1_top2_margin.update(summary.top1_top2_margin.mean)
        self.baselines.residual_slope.update(summary.residual_slope)

        if summary.attention_entropy:
            self.baselines.attention_entropy.update(summary.attention_entropy.mean)

    def get_baselines_dict(self) -> dict:
        """Return baselines as dict for persistence."""
        return self.baselines.to_dict()

    def load_baselines(self, d: dict) -> None:
        """Load baselines from persisted dict."""
        self.baselines = TelemetryBaselines.from_dict(d)
