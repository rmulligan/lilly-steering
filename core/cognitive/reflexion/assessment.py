"""Health assessment for Reflexion phase.

Categorizes health signals from snapshots into THRIVING/STABLE/STRESSED/CRITICAL
categories based on configurable thresholds.

This module provides the HealthAssessor class that:
- Compares current metrics against rolling baselines
- Categorizes health with threshold-based classification
- Identifies trends (improving, stable, declining)
- Produces complete HealthAssessment profiles
"""

from __future__ import annotations

from typing import Any

from core.cognitive.reflexion.schemas import (
    HealthAssessment,
    HealthCategory,
    HealthSignal,
)


class HealthAssessor:
    """Categorizes health signals into severity levels.

    Uses configurable thresholds to classify metrics relative to baselines:
    - THRIVING: value/baseline >= thriving_threshold (default 1.2, 20% above)
    - STABLE: stressed_threshold < value/baseline < thriving_threshold
    - STRESSED: critical_threshold <= value/baseline <= stressed_threshold
    - CRITICAL: value/baseline < critical_threshold (default 0.3, 70% below)

    The assessor also determines trend direction:
    - "improving": ratio >= thriving_threshold
    - "stable": stressed_threshold < ratio < thriving_threshold
    - "declining": ratio <= stressed_threshold

    Attributes:
        thriving_threshold: Minimum ratio for THRIVING (default 1.2)
        stressed_threshold: Maximum ratio for STRESSED (default 0.7)
        critical_threshold: Maximum ratio for CRITICAL (default 0.3)
    """

    def __init__(
        self,
        thriving_threshold: float = 1.2,
        stressed_threshold: float = 0.7,
        critical_threshold: float = 0.3,
    ):
        """Initialize assessor with configurable thresholds.

        Args:
            thriving_threshold: Ratio above which THRIVING is assigned (default 1.2)
            stressed_threshold: Ratio below which STRESSED is assigned (default 0.7)
            critical_threshold: Ratio below which CRITICAL is assigned (default 0.3)
        """
        self.thriving_threshold = thriving_threshold
        self.stressed_threshold = stressed_threshold
        self.critical_threshold = critical_threshold

    def _categorize(
        self,
        value: float,
        baseline: float,
    ) -> tuple[HealthCategory, str]:
        """Categorize a value relative to its baseline.

        Computes value/baseline ratio and classifies:
        - THRIVING: ratio > thriving_threshold
        - STABLE: stressed_threshold < ratio <= thriving_threshold
        - STRESSED: critical_threshold <= ratio <= stressed_threshold
        - CRITICAL: ratio < critical_threshold

        Special cases:
        - Zero baseline with positive value: THRIVING (improving from nothing)
        - Zero baseline with zero value: STABLE (no change from nothing)

        Args:
            value: Current measured value
            baseline: Reference value for comparison

        Returns:
            Tuple of (HealthCategory, trend_string)
        """
        # Handle zero baseline edge cases
        if baseline == 0.0:
            if value > 0.0:
                return HealthCategory.THRIVING, "improving"
            else:
                return HealthCategory.STABLE, "stable"

        ratio = value / baseline

        # Determine category based on thresholds
        if ratio > self.thriving_threshold:
            return HealthCategory.THRIVING, "improving"
        elif ratio <= self.critical_threshold:
            return HealthCategory.CRITICAL, "declining"
        elif ratio <= self.stressed_threshold:
            return HealthCategory.STRESSED, "declining"
        else:
            return HealthCategory.STABLE, "stable"

    def assess_prediction(self, snapshot: dict[str, Any]) -> HealthSignal:
        """Assess prediction health from snapshot.

        Evaluates confirmation_rate against prediction baseline.
        Overrides to CRITICAL if confirmation_rate is 0 with sample_size >= 10.

        Args:
            snapshot: Complete health snapshot from HealthSignalCollector

        Returns:
            HealthSignal with prediction health assessment
        """
        prediction_data = snapshot.get("prediction", {})
        baselines = snapshot.get("baselines", {})

        value = prediction_data.get("confirmation_rate", 0.0)
        baseline = baselines.get("prediction", 0.35)
        sample_size = prediction_data.get("sample_size", 0)

        category, trend = self._categorize(value, baseline)

        # Override to CRITICAL if zero confirmation with sufficient samples
        context = None
        if value == 0.0 and sample_size >= 10:
            category = HealthCategory.CRITICAL
            trend = "declining"
            context = f"Zero confirmation rate with sufficient samples ({sample_size})"

        return HealthSignal(
            category=category,
            value=value,
            baseline=baseline,
            trend=trend,
            context=context,
        )

    def assess_integration(self, snapshot: dict[str, Any]) -> HealthSignal:
        """Assess integration health from snapshot.

        Evaluates success_rate against integration baseline.

        Args:
            snapshot: Complete health snapshot from HealthSignalCollector

        Returns:
            HealthSignal with integration health assessment
        """
        integration_data = snapshot.get("integration", {})
        baselines = snapshot.get("baselines", {})

        value = integration_data.get("success_rate", 1.0)
        baseline = baselines.get("integration", 0.92)

        category, trend = self._categorize(value, baseline)

        return HealthSignal(
            category=category,
            value=value,
            baseline=baseline,
            trend=trend,
        )

    def assess_coherence(self, snapshot: dict[str, Any]) -> HealthSignal:
        """Assess coherence health from snapshot.

        Computes coherence as average of thought_diversity and sae_diversity
        from phenomenological signals, then compares against coherence baseline.

        Args:
            snapshot: Complete health snapshot from HealthSignalCollector

        Returns:
            HealthSignal with coherence health assessment
        """
        phenomenological = snapshot.get("phenomenological", {})
        baselines = snapshot.get("baselines", {})

        thought_diversity = phenomenological.get("thought_diversity", 1.0)
        sae_diversity = phenomenological.get("sae_diversity", 1.0)

        # Coherence is average of diversity metrics
        value = (thought_diversity + sae_diversity) / 2.0
        baseline = baselines.get("coherence", 0.70)

        category, trend = self._categorize(value, baseline)

        return HealthSignal(
            category=category,
            value=value,
            baseline=baseline,
            trend=trend,
        )

    def assess(self, snapshot: dict[str, Any]) -> HealthAssessment:
        """Perform complete health assessment from snapshot.

        Calls all assess_* methods and constructs a full HealthAssessment.

        Args:
            snapshot: Complete health snapshot from HealthSignalCollector

        Returns:
            HealthAssessment with all health signals
        """
        prediction_signal = self.assess_prediction(snapshot)
        integration_signal = self.assess_integration(snapshot)
        coherence_signal = self.assess_coherence(snapshot)

        return HealthAssessment(
            prediction=prediction_signal,
            integration=integration_signal,
            coherence=coherence_signal,
        )
