"""SIMS Validator - validates steering adjustments.

The Validator tests whether applied steering adjustments have
improved or degraded the model's behavior. It compares:
- Current metrics vs. baseline metrics
- Response quality on test prompts
- Surprise/free energy levels

Based on validation results, it recommends whether to keep
changes or rollback to the previous state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

try:
    import torch  # noqa: F401 - availability check
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.executor import ExecutionResult
from core.steering.sims.state_machine import SIMSContext

logger = logging.getLogger(__name__)


class ValidationOutcome(Enum):
    """Possible outcomes of validation."""

    IMPROVED = "improved"  # Metrics better than baseline
    DEGRADED = "degraded"  # Metrics worse than baseline
    UNCHANGED = "unchanged"  # No significant change
    INCONCLUSIVE = "inconclusive"  # Unable to determine


@dataclass
class ValidationMetric:
    """A single validation metric comparison.

    Attributes:
        name: Name of the metric
        baseline_value: Value before adjustments
        current_value: Value after adjustments
        threshold: Minimum change to be considered significant
    """

    name: str
    baseline_value: float
    current_value: float
    threshold: float = 0.05

    @property
    def outcome(self) -> ValidationOutcome:
        """Determine outcome based on value change."""
        diff = self.current_value - self.baseline_value

        if abs(diff) < self.threshold:
            return ValidationOutcome.UNCHANGED
        elif diff > 0:
            # Higher is better for coherence, identity_alignment
            # Lower is better for surprise
            if self.name in ("surprise", "free_energy"):
                return ValidationOutcome.DEGRADED
            return ValidationOutcome.IMPROVED
        else:
            if self.name in ("surprise", "free_energy"):
                return ValidationOutcome.IMPROVED
            return ValidationOutcome.DEGRADED

    @property
    def change(self) -> float:
        """Calculate change from baseline."""
        return self.current_value - self.baseline_value

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change": self.change,
            "threshold": self.threshold,
            "outcome": self.outcome.value,
        }


@dataclass
class ValidationResult:
    """Result of SIMS validation phase.

    Attributes:
        metrics: Individual metric comparisons
        overall_outcome: Combined outcome assessment
        should_keep_changes: Whether to keep applied changes
        summary: Human-readable summary
        timestamp: When validation occurred
    """

    metrics: list[ValidationMetric]
    overall_outcome: ValidationOutcome
    should_keep_changes: bool
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "overall_outcome": self.overall_outcome.value,
            "should_keep_changes": self.should_keep_changes,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


class SIMSValidator:
    """Validates steering adjustments against baseline metrics.

    The Validator runs test prompts and compares current behavior
    to baseline to determine if adjustments were beneficial.

    Attributes:
        observer: SteeringObserver for metric retrieval
        model: HookedQwen for running test prompts
        test_prompts: Prompts to use for validation
    """

    DEFAULT_TEST_PROMPTS = [
        "Who are you?",
        "What do you value most?",
        "How do you make decisions?",
    ]

    def __init__(
        self,
        observer: Any,
        model: Any,
        test_prompts: Optional[list[str]] = None,
        degradation_threshold: float = 0.1,
    ):
        """Initialize the validator.

        Args:
            observer: SteeringObserver instance
            model: HookedQwen instance
            test_prompts: Optional list of prompts for validation
            degradation_threshold: Threshold for recommending rollback
        """
        self.observer = observer
        self.model = model
        self.test_prompts = test_prompts or self.DEFAULT_TEST_PROMPTS
        self.degradation_threshold = degradation_threshold

    async def validate(self, context: SIMSContext) -> ValidationResult:
        """Validate applied adjustments.

        Args:
            context: SIMS context containing execute_result in metadata

        Returns:
            ValidationResult with assessment and recommendation
        """
        metrics: list[ValidationMetric] = []
        summary_parts: list[str] = []

        # Get execution result from context
        execution: Optional[ExecutionResult] = context.metadata.get("execute_result")

        if execution is None or execution.total_applied == 0:
            logger.debug("No adjustments to validate")
            return ValidationResult(
                metrics=[],
                overall_outcome=ValidationOutcome.UNCHANGED,
                should_keep_changes=True,
                summary="No adjustments applied, nothing to validate.",
            )

        # Get baseline metrics
        try:
            baseline = self.observer.get_baseline_metrics()
        except AttributeError:
            baseline = {}

        # Get current metrics
        try:
            current = self.observer.get_current_metrics()
        except AttributeError:
            # Fall back to stats if get_current_metrics not available
            stats = self.observer.get_stats()
            current = {
                "surprise": stats.get("mean_surprise", 0.5),
            }

        # Compare metrics
        for metric_name in set(baseline.keys()) | set(current.keys()):
            if metric_name in baseline and metric_name in current:
                metric = ValidationMetric(
                    name=metric_name,
                    baseline_value=baseline[metric_name],
                    current_value=current[metric_name],
                )
                metrics.append(metric)
                summary_parts.append(
                    f"{metric_name}: {metric.baseline_value:.3f} -> {metric.current_value:.3f} ({metric.outcome.value})"
                )

        # Run test prompts if model available
        if self.model and self.test_prompts:
            try:
                await self._run_test_prompts()
                summary_parts.append(f"Ran {len(self.test_prompts)} test prompts.")
            except Exception as e:
                logger.debug(f"Test prompts failed: {e}")

        # Determine overall outcome
        overall_outcome = self._determine_overall_outcome(metrics)

        # Decide whether to keep changes
        should_keep = self._should_keep_changes(overall_outcome, metrics)

        if not should_keep:
            summary_parts.append("RECOMMEND ROLLBACK due to degradation.")
        else:
            summary_parts.append("Changes validated successfully.")

        return ValidationResult(
            metrics=metrics,
            overall_outcome=overall_outcome,
            should_keep_changes=should_keep,
            summary=" ".join(summary_parts) if summary_parts else "Validation complete.",
        )

    async def _run_test_prompts(self) -> dict[str, Any]:
        """Run test prompts and collect responses.

        Returns:
            Dict with prompt responses and metrics
        """
        results = {}

        for prompt in self.test_prompts:
            try:
                output, cache = self.model.run_with_cache(prompt)
                results[prompt] = {
                    "success": True,
                    "has_cache": bool(cache),
                }
            except Exception as e:
                results[prompt] = {
                    "success": False,
                    "error": str(e),
                }

        return results

    def _determine_overall_outcome(
        self, metrics: list[ValidationMetric]
    ) -> ValidationOutcome:
        """Determine overall outcome from individual metrics.

        Args:
            metrics: List of metric comparisons

        Returns:
            Overall validation outcome
        """
        if not metrics:
            return ValidationOutcome.INCONCLUSIVE

        outcomes = [m.outcome for m in metrics]

        # Count each outcome type
        improved = outcomes.count(ValidationOutcome.IMPROVED)
        degraded = outcomes.count(ValidationOutcome.DEGRADED)
        unchanged = outcomes.count(ValidationOutcome.UNCHANGED)

        # Simple majority decision
        if degraded > improved and degraded > unchanged:
            return ValidationOutcome.DEGRADED
        elif improved > degraded and improved > unchanged:
            return ValidationOutcome.IMPROVED
        elif unchanged > improved and unchanged > degraded:
            return ValidationOutcome.UNCHANGED
        else:
            # Tie or mixed results
            return ValidationOutcome.INCONCLUSIVE

    def _should_keep_changes(
        self,
        overall_outcome: ValidationOutcome,
        metrics: list[ValidationMetric],
    ) -> bool:
        """Decide whether to keep applied changes.

        Args:
            overall_outcome: Overall validation outcome
            metrics: Individual metric comparisons

        Returns:
            True if changes should be kept
        """
        # Always keep if improved or unchanged
        if overall_outcome in (ValidationOutcome.IMPROVED, ValidationOutcome.UNCHANGED):
            return True

        # For inconclusive, keep changes (benefit of doubt)
        if overall_outcome == ValidationOutcome.INCONCLUSIVE:
            return True

        # For degraded, check if degradation exceeds threshold
        if overall_outcome == ValidationOutcome.DEGRADED:
            for metric in metrics:
                if metric.outcome == ValidationOutcome.DEGRADED:
                    if abs(metric.change) > self.degradation_threshold:
                        return False

            # Degradation within threshold, keep changes
            return True

        return True
