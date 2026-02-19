"""Skill effectiveness tracking for SOAR curriculum learning.

Implements grounded effectiveness tracking where skill effectiveness_score
is tied to actual cognitive improvement (verification rate on hard problems),
not intrinsic proxy metrics.

SOAR principle: Teacher reward = student improvement on hard problems.
This avoids reward hacking and diversity collapse from intrinsic rewards.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from core.cognitive.curriculum.hard_problems import (
    HardProblemRegistry,
    HardProblemType,
    is_hard_problem_condition,
)

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class SkillInfluencedCycle:
    """Record of a cognitive cycle that used skill injection.

    Tracks which skills were injected and what predictions resulted,
    enabling attribution of verification outcomes back to skills.
    """

    cycle_id: int
    skill_uids: list[str]  # Skills injected into this cycle
    predictions_made: list[str] = field(default_factory=list)  # Prediction UIDs
    verification_outcomes: dict[str, bool] = field(default_factory=dict)
    hard_problem_outcomes: dict[str, bool] = field(default_factory=dict)
    hard_problem_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_complete(self) -> bool:
        """Whether all predictions have been verified."""
        return len(self.verification_outcomes) >= len(self.predictions_made)

    @property
    def overall_accuracy(self) -> float:
        """Overall verification accuracy for this cycle."""
        if not self.verification_outcomes:
            return 0.0
        return sum(self.verification_outcomes.values()) / len(self.verification_outcomes)


class SkillEffectivenessTracker:
    """Tracks skill effectiveness via grounded student improvement.

    Core SOAR component that:
    1. Records when skills are injected into cognitive cycles
    2. Tracks verification outcomes for predictions made in those cycles
    3. Attributes improvements to the injected skills
    4. Updates skill effectiveness_score based on hard problem performance

    The key insight: effectiveness is measured by improvement on HARD problems
    (METRIC_THRESHOLD, BELIEF_CHANGE, etc.), not easy predictions.
    """

    def __init__(self, psyche_client: "PsycheClient"):
        self._client = psyche_client
        self._influenced_cycles: dict[int, SkillInfluencedCycle] = {}
        self._skill_contributions: dict[str, list[float]] = {}  # skill_uid -> improvements
        self._hard_problem_registry = HardProblemRegistry()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize baselines from historical data."""
        if self._initialized:
            return

        await self._hard_problem_registry.initialize_baselines(self._client)
        self._initialized = True
        logger.info("[SOAR] SkillEffectivenessTracker initialized with historical baselines")

    async def record_skill_injection(
        self,
        cycle_id: int,
        skill_uids: list[str],
        predictions: Optional[list[str]] = None,
    ) -> None:
        """Record that skills were injected into a cognitive cycle.

        Args:
            cycle_id: The cycle number
            skill_uids: UIDs of skills that were injected
            predictions: Optional list of prediction UIDs made in this cycle
        """
        if not skill_uids:
            return

        self._influenced_cycles[cycle_id] = SkillInfluencedCycle(
            cycle_id=cycle_id,
            skill_uids=skill_uids,
            predictions_made=predictions or [],
        )

        logger.info(
            f"[SOAR] Recorded skill injection: cycle={cycle_id}, "
            f"skills={len(skill_uids)}, predictions={len(predictions or [])}"
        )

    def add_prediction_to_cycle(self, cycle_id: int, prediction_uid: str) -> None:
        """Add a prediction to a tracked cycle after initial recording."""
        if cycle_id in self._influenced_cycles:
            self._influenced_cycles[cycle_id].predictions_made.append(prediction_uid)

    async def record_verification_outcome(
        self,
        cycle_id: int,
        prediction_uid: str,
        verified: bool,
        condition_type: str,
    ) -> None:
        """Record verification outcome for skill effectiveness tracking.

        Args:
            cycle_id: The cycle where prediction was made
            prediction_uid: UID of the prediction
            verified: Whether the prediction was verified
            condition_type: PredictionConditionType string
        """
        if cycle_id not in self._influenced_cycles:
            return

        cycle = self._influenced_cycles[cycle_id]
        cycle.verification_outcomes[prediction_uid] = verified

        # Track hard problem outcomes separately
        is_hard = is_hard_problem_condition(condition_type)
        if is_hard:
            cycle.hard_problem_outcomes[prediction_uid] = verified

            # Update hard problem registry
            problem_type = HardProblemType.from_condition_type(condition_type)
            if problem_type:
                # Get condition value from prediction
                condition_value = await self._get_prediction_condition_value(prediction_uid)
                problem = self._hard_problem_registry.get_or_create_problem(
                    problem_type, condition_value
                )
                problem.record_outcome(verified)

        # Update skill contributions when we have hard problem data
        if cycle.hard_problem_outcomes:
            await self._update_skill_contributions(cycle)

        logger.debug(
            f"[SOAR] Recorded outcome: pred={prediction_uid[:12]}, "
            f"verified={verified}, is_hard={is_hard}"
        )

    async def _get_prediction_condition_value(self, prediction_uid: str) -> str:
        """Get condition_value for a prediction."""
        result = await self._client.query(
            """
            MATCH (p:Prediction {uid: $uid})
            RETURN p.condition_value as value
            """,
            {"uid": prediction_uid},
        )
        return result[0]["value"] if result else "unknown"

    async def _update_skill_contributions(
        self,
        cycle: SkillInfluencedCycle,
    ) -> None:
        """Update effectiveness scores based on hard problem performance.

        This is the core SOAR attribution logic:
        1. Calculate accuracy on hard problems for this cycle
        2. Compare to baseline (what accuracy would be without skills)
        3. Attribute improvement (or degradation) to injected skills
        4. Update skill effectiveness_score in graph
        """
        if not cycle.hard_problem_outcomes:
            return

        # Calculate hard problem accuracy for this cycle
        hard_accuracy = (
            sum(cycle.hard_problem_outcomes.values())
            / len(cycle.hard_problem_outcomes)
        )
        cycle.hard_problem_accuracy = hard_accuracy

        # Attribute improvement to each injected skill
        for skill_uid in cycle.skill_uids:
            baseline = await self._get_skill_baseline(skill_uid)
            improvement = hard_accuracy - baseline

            if skill_uid not in self._skill_contributions:
                self._skill_contributions[skill_uid] = []
            self._skill_contributions[skill_uid].append(improvement)

            # Update skill effectiveness in graph
            await self._update_skill_effectiveness(skill_uid, cycle.cycle_id)

            logger.info(
                f"[SOAR] Skill contribution: skill={skill_uid[:12]}, "
                f"improvement={improvement:+.2%} (hard_acc={hard_accuracy:.2%}, baseline={baseline:.2%})"
            )

    async def _get_skill_baseline(self, skill_uid: str) -> float:
        """Get baseline accuracy before this skill was used.

        Baseline is the historical verification rate for this skill's
        cognitive_operation BEFORE the skill was created.
        """
        result = await self._client.query(
            """
            MATCH (s:LearnedSkill {uid: $uid})
            OPTIONAL MATCH (h:Hypothesis {cognitive_operation: s.cognitive_operation})
            WHERE h.created_cycle < s.created_cycle AND h.verification_rate IS NOT NULL
            RETURN avg(h.verification_rate) as baseline
            """,
            {"uid": skill_uid},
        )

        if result and result[0]["baseline"] is not None:
            return result[0]["baseline"]

        # Default baseline if no historical data
        return 0.3

    async def _update_skill_effectiveness(self, skill_uid: str, cycle_id: int) -> None:
        """Update skill's effectiveness_score based on grounded improvements.

        Uses EMA of recent improvements to smooth updates and avoid instability.
        """
        contributions = self._skill_contributions.get(skill_uid, [])
        if not contributions:
            return

        # EMA of recent improvements (last 10)
        recent = contributions[-10:]
        avg_improvement = sum(recent) / len(recent)

        # Convert improvement to effectiveness score (0-1 range)
        # improvement of +0.5 maps to effectiveness of 1.0
        # improvement of -0.5 maps to effectiveness of 0.0
        # improvement of 0.0 maps to effectiveness of 0.5
        raw_effectiveness = 0.5 + avg_improvement

        # Clamp to [0, 1]
        effectiveness = max(0.0, min(1.0, raw_effectiveness))

        # EMA update in graph
        alpha = 0.3
        await self._client.execute(
            """
            MATCH (s:LearnedSkill {uid: $uid})
            SET s.effectiveness_score = $alpha * $effectiveness + (1 - $alpha) * COALESCE(s.effectiveness_score, 0.5),
                s.usage_count = COALESCE(s.usage_count, 0) + 1,
                s.hard_problem_accuracy = $hard_acc,
                s.improvement_over_baseline = $improvement,
                s.last_used_cycle = $cycle
            """,
            {
                "uid": skill_uid,
                "alpha": alpha,
                "effectiveness": effectiveness,
                "hard_acc": avg_improvement + 0.3,  # Approximate
                "improvement": avg_improvement,
                "cycle": cycle_id,
            },
        )

        logger.info(
            f"[SOAR] Updated skill effectiveness: skill={skill_uid[:12]}, "
            f"effectiveness={effectiveness:.2f}, avg_improvement={avg_improvement:+.2%}"
        )

    async def get_skill_improvement(self, skill_uid: str) -> Optional[float]:
        """Get the average improvement attributed to a skill.

        Returns None if skill has no recorded contributions.
        """
        contributions = self._skill_contributions.get(skill_uid)
        if not contributions:
            return None
        return sum(contributions) / len(contributions)

    def get_influenced_cycle(self, cycle_id: int) -> Optional[SkillInfluencedCycle]:
        """Get a tracked cycle by ID."""
        return self._influenced_cycles.get(cycle_id)

    def get_all_improvements(self) -> dict[str, float]:
        """Get improvement for all tracked skills."""
        return {
            uid: sum(contribs) / len(contribs)
            for uid, contribs in self._skill_contributions.items()
            if contribs
        }

    def get_hard_problem_improvements(self) -> dict[HardProblemType, float]:
        """Get improvement by hard problem type."""
        return self._hard_problem_registry.get_all_improvements()

    async def cleanup_old_cycles(self, current_cycle: int, max_age: int = 100) -> int:
        """Remove old tracked cycles to prevent memory growth.

        Args:
            current_cycle: Current cycle number
            max_age: Maximum cycle age to keep

        Returns:
            Number of cycles removed
        """
        cutoff = current_cycle - max_age
        old_cycles = [cid for cid in self._influenced_cycles if cid < cutoff]

        for cid in old_cycles:
            del self._influenced_cycles[cid]

        if old_cycles:
            logger.debug(f"[SOAR] Cleaned up {len(old_cycles)} old tracked cycles")

        return len(old_cycles)
