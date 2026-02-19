"""Hard problem definitions for SOAR curriculum learning.

"Hard problems" are cognitive challenges where skills should demonstrably help.
In Lilly's context, these are prediction types with historically low verification rates.

SOAR principle: Teacher reward = student improvement on HARD problems (not easy ones).
This grounds the reward signal and avoids reward hacking on trivial tasks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient


class HardProblemType(Enum):
    """Types of cognitive challenges where skills should help most.

    These correspond to PredictionConditionType values that historically
    have low verification rates, indicating genuine cognitive difficulty.
    """

    METRIC_THRESHOLD = "metric_threshold"  # Quantitative predictions about metrics
    BELIEF_CHANGE = "belief_change"  # Predictions about belief evolution
    SAE_FEATURE_PATTERN = "sae_feature"  # Feature activation predictions
    GOAL_PROGRESS = "goal_progress"  # Predictions about goal advancement

    @classmethod
    def from_condition_type(cls, condition_type: str) -> "HardProblemType | None":
        """Map PredictionConditionType string to HardProblemType."""
        mapping = {
            "METRIC_THRESHOLD": cls.METRIC_THRESHOLD,
            "BELIEF_CHANGE": cls.BELIEF_CHANGE,
            "SAE_FEATURE_PATTERN": cls.SAE_FEATURE_PATTERN,
            "GOAL_PROGRESS": cls.GOAL_PROGRESS,
        }
        return mapping.get(condition_type)

    def is_hard_problem(self) -> bool:
        """All HardProblemTypes are considered hard by definition."""
        return True


@dataclass
class HardProblem:
    """A cognitive challenge for measuring skill effectiveness.

    Tracks baseline vs current accuracy to compute grounded improvement signal.
    """

    problem_type: HardProblemType
    condition_value: str  # The specific condition being predicted
    baseline_accuracy: float = 0.3  # Historical accuracy without skill (default 30%)
    current_accuracy: float = 0.0  # Accuracy after skill injection
    sample_count: int = 0  # Number of observations

    @property
    def improvement(self) -> float:
        """Grounded reward signal: improvement over baseline.

        This is the core SOAR metric - teacher is rewarded based on
        how much the student improves on hard problems.
        """
        return self.current_accuracy - self.baseline_accuracy

    @property
    def is_significant(self) -> bool:
        """Whether improvement is statistically meaningful."""
        # Require at least 5 samples and >5% improvement
        return self.sample_count >= 5 and self.improvement > 0.05

    def record_outcome(self, verified: bool) -> None:
        """Record a verification outcome for this hard problem."""
        self.sample_count += 1
        # EMA update for current accuracy
        alpha = min(0.3, 1.0 / self.sample_count)  # Adaptive learning rate
        outcome = 1.0 if verified else 0.0
        self.current_accuracy = alpha * outcome + (1 - alpha) * self.current_accuracy


@dataclass
class HardProblemRegistry:
    """Registry of hard problems for tracking skill effectiveness.

    Maintains per-type baselines and tracks improvement over time.
    """

    _problems: dict[HardProblemType, dict[str, HardProblem]] = field(
        default_factory=dict
    )
    _baselines: dict[HardProblemType, float] = field(default_factory=dict)

    async def initialize_baselines(self, psyche_client: "PsycheClient") -> None:
        """Load historical baselines from graph.

        Queries verification rates by condition_type before any skills existed.
        """
        for problem_type in HardProblemType:
            condition_type = problem_type.value.upper()

            result = await psyche_client.query(
                """
                MATCH (p:Prediction {condition_type: $type})
                WHERE p.outcome IS NOT NULL
                WITH p,
                     CASE WHEN p.outcome = 'VERIFIED' THEN 1.0 ELSE 0.0 END as success
                RETURN avg(success) as baseline_accuracy, count(p) as sample_count
                """,
                {"type": condition_type},
            )

            if result and result[0]["baseline_accuracy"] is not None:
                self._baselines[problem_type] = result[0]["baseline_accuracy"]
            else:
                # Default baseline if no historical data
                self._baselines[problem_type] = 0.3

    def get_or_create_problem(
        self,
        problem_type: HardProblemType,
        condition_value: str,
    ) -> HardProblem:
        """Get or create a hard problem tracker."""
        if problem_type not in self._problems:
            self._problems[problem_type] = {}

        if condition_value not in self._problems[problem_type]:
            self._problems[problem_type][condition_value] = HardProblem(
                problem_type=problem_type,
                condition_value=condition_value,
                baseline_accuracy=self._baselines.get(problem_type, 0.3),
            )

        return self._problems[problem_type][condition_value]

    def get_aggregate_improvement(self, problem_type: HardProblemType) -> float:
        """Get average improvement across all problems of a type."""
        if problem_type not in self._problems:
            return 0.0

        problems = self._problems[problem_type].values()
        significant = [p for p in problems if p.is_significant]

        if not significant:
            return 0.0

        return sum(p.improvement for p in significant) / len(significant)

    def get_all_improvements(self) -> dict[HardProblemType, float]:
        """Get improvement for all problem types."""
        return {pt: self.get_aggregate_improvement(pt) for pt in HardProblemType}


def is_hard_problem_condition(condition_type: str) -> bool:
    """Check if a prediction condition type qualifies as a hard problem.

    Args:
        condition_type: PredictionConditionType string value

    Returns:
        True if this is a hard problem type, False otherwise
    """
    return HardProblemType.from_condition_type(condition_type) is not None
