"""Promotion mechanism for SOAR curriculum learning.

Implements the promotion queue that accumulates proven skills into the
curriculum (𝒟ᵦₑₛₜ). Skills that demonstrably improve student performance
on hard problems get promoted and prioritized for future retrieval.

SOAR principle: Teacher is rewarded when student improves on hard problems.
Promotion mechanism tracks this and accumulates effective stepping stones.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class PromotionConfig:
    """Configuration for skill promotion mechanism.

    Attributes:
        tau: Improvement threshold for promotion (default 5%)
        window_size: Cycles to average for baseline calculation
        min_usage: Minimum uses before skill is promotion-eligible
        max_promoted: Maximum size of promoted skill set
        demotion_threshold: Threshold for demoting underperforming skills
    """

    tau: float = 0.05  # 5% improvement required for promotion
    window_size: int = 20  # Cycles for baseline moving average
    min_usage: int = 3  # Minimum uses before eligible
    max_promoted: int = 50  # Maximum promoted skills (curriculum size)
    demotion_threshold: float = -0.05  # Demote if hurting performance


@dataclass
class SkillPerformanceRecord:
    """Track performance history for a skill candidate."""

    skill_uid: str
    accuracies: list[float] = field(default_factory=list)
    usage_count: int = 0
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None

    @property
    def average_accuracy(self) -> float:
        """Average accuracy across all recorded uses."""
        if not self.accuracies:
            return 0.0
        return sum(self.accuracies) / len(self.accuracies)

    @property
    def recent_accuracy(self) -> float:
        """Average of last 5 uses (more responsive to changes)."""
        if not self.accuracies:
            return 0.0
        recent = self.accuracies[-5:]
        return sum(recent) / len(recent)

    @property
    def is_eligible(self) -> bool:
        """Whether skill has enough data for promotion decision."""
        return self.usage_count >= 3  # Minimum samples


class PromotionQueue:
    """Manages skill promotion based on demonstrated improvement.

    Core SOAR component that:
    1. Tracks moving average baseline accuracy
    2. Monitors skill-influenced cycle performance
    3. Promotes skills that exceed baseline + τ
    4. Demotes skills that hurt performance
    5. Maintains bounded curriculum size

    The promoted set (𝒟ᵦₑₛₜ) represents proven stepping stones that
    have demonstrably helped cognitive improvement.
    """

    def __init__(self, config: Optional[PromotionConfig] = None):
        self.config = config or PromotionConfig()
        self._baseline_window: deque[float] = deque(maxlen=self.config.window_size)
        self._promoted_skills: set[str] = set()  # 𝒟ᵦₑₛₜ
        self._candidates: dict[str, SkillPerformanceRecord] = {}
        self._promotion_history: list[tuple[str, datetime, str]] = []  # (uid, time, action)

    @property
    def baseline(self) -> float:
        """Moving average baseline accuracy (without skill influence)."""
        if not self._baseline_window:
            return 0.3  # Default baseline
        return sum(self._baseline_window) / len(self._baseline_window)

    @property
    def promoted_count(self) -> int:
        """Number of skills currently promoted."""
        return len(self._promoted_skills)

    @property
    def is_full(self) -> bool:
        """Whether curriculum is at capacity."""
        return self.promoted_count >= self.config.max_promoted

    def record_baseline_accuracy(self, accuracy: float) -> None:
        """Update baseline moving average with a cycle's accuracy.

        Call this for cycles that did NOT have skill injection,
        to maintain an accurate baseline.

        Args:
            accuracy: Verification accuracy for a non-skill-influenced cycle
        """
        self._baseline_window.append(accuracy)
        logger.debug(f"[SOAR] Baseline updated: {accuracy:.2%} (avg: {self.baseline:.2%})")

    def record_skill_performance(
        self,
        skill_uid: str,
        accuracy: float,
    ) -> Optional[str]:
        """Record skill performance and check for promotion/demotion.

        Args:
            skill_uid: UID of the skill used
            accuracy: Hard problem accuracy when skill was injected

        Returns:
            - skill_uid if promoted
            - f"demoted:{skill_uid}" if demoted
            - None if no status change
        """
        # Get or create candidate record
        if skill_uid not in self._candidates:
            self._candidates[skill_uid] = SkillPerformanceRecord(skill_uid=skill_uid)

        record = self._candidates[skill_uid]
        record.accuracies.append(accuracy)
        record.usage_count += 1

        # Check eligibility
        if not record.is_eligible:
            return None

        improvement = record.average_accuracy - self.baseline

        # Check for promotion
        if skill_uid not in self._promoted_skills:
            if improvement >= self.config.tau:
                return self._promote_skill(skill_uid, record, improvement)
        else:
            # Check for demotion
            if improvement <= self.config.demotion_threshold:
                return self._demote_skill(skill_uid, record, improvement)

        return None

    def _promote_skill(
        self,
        skill_uid: str,
        record: SkillPerformanceRecord,
        improvement: float,
    ) -> str:
        """Promote skill to curriculum."""
        # Make room if at capacity
        if self.is_full:
            self._evict_weakest()

        self._promoted_skills.add(skill_uid)
        record.promoted_at = datetime.now(timezone.utc)
        self._promotion_history.append((skill_uid, record.promoted_at, "promoted"))

        logger.info(
            f"[SOAR] PROMOTED skill {skill_uid[:12]} to curriculum "
            f"(improvement: {improvement:+.2%}, baseline: {self.baseline:.2%})"
        )

        return skill_uid

    def _demote_skill(
        self,
        skill_uid: str,
        record: SkillPerformanceRecord,
        improvement: float,
    ) -> str:
        """Demote skill from curriculum."""
        self._promoted_skills.discard(skill_uid)
        record.demoted_at = datetime.now(timezone.utc)
        self._promotion_history.append((skill_uid, record.demoted_at, "demoted"))

        logger.info(
            f"[SOAR] DEMOTED skill {skill_uid[:12]} from curriculum "
            f"(improvement: {improvement:+.2%}, baseline: {self.baseline:.2%})"
        )

        return f"demoted:{skill_uid}"

    def _evict_weakest(self) -> Optional[str]:
        """Remove weakest performer from promoted set.

        Called when curriculum is at capacity and a new skill needs promotion.
        """
        if not self._promoted_skills:
            return None

        # Find skill with lowest average accuracy
        weakest_uid = None
        lowest_avg = float("inf")

        for uid in self._promoted_skills:
            if uid in self._candidates:
                avg = self._candidates[uid].average_accuracy
                if avg < lowest_avg:
                    lowest_avg = avg
                    weakest_uid = uid

        if weakest_uid:
            self._promoted_skills.discard(weakest_uid)
            self._promotion_history.append(
                (weakest_uid, datetime.now(timezone.utc), "evicted")
            )
            logger.info(
                f"[SOAR] EVICTED weakest skill {weakest_uid[:12]} "
                f"(avg: {lowest_avg:.2%})"
            )

        return weakest_uid

    def get_promoted_skills(self) -> set[str]:
        """Get current promoted skill set (𝒟ᵦₑₛₜ)."""
        return self._promoted_skills.copy()

    def is_promoted(self, skill_uid: str) -> bool:
        """Check if skill is in promoted set."""
        return skill_uid in self._promoted_skills

    def get_skill_improvement(self, skill_uid: str) -> Optional[float]:
        """Get improvement over baseline for a skill."""
        if skill_uid not in self._candidates:
            return None
        return self._candidates[skill_uid].average_accuracy - self.baseline

    def get_promotion_stats(self) -> dict:
        """Get statistics about promotion activity."""
        promoted_count = len([h for h in self._promotion_history if h[2] == "promoted"])
        demoted_count = len([h for h in self._promotion_history if h[2] == "demoted"])
        evicted_count = len([h for h in self._promotion_history if h[2] == "evicted"])

        return {
            "baseline": self.baseline,
            "promoted_skills": self.promoted_count,
            "max_capacity": self.config.max_promoted,
            "candidates_tracked": len(self._candidates),
            "total_promotions": promoted_count,
            "total_demotions": demoted_count,
            "total_evictions": evicted_count,
            "tau_threshold": self.config.tau,
        }

    async def sync_to_graph(self, psyche_client: "PsycheClient") -> int:
        """Sync promotion status to FalkorDB.

        Updates is_promoted field on LearnedSkill nodes.

        Returns:
            Number of skills updated
        """
        if not self._promoted_skills:
            return 0

        # Update promoted skills
        updated = 0
        for skill_uid in self._promoted_skills:
            try:
                await psyche_client.execute(
                    """
                    MATCH (s:LearnedSkill {uid: $uid})
                    SET s.is_promoted = true,
                        s.promotion_cycle = $cycle
                    """,
                    {"uid": skill_uid, "cycle": len(self._baseline_window)},
                )
                updated += 1
            except Exception as e:
                logger.warning(f"Failed to sync promotion for {skill_uid}: {e}")

        # Mark demoted skills
        for uid, _, action in self._promotion_history:
            if action in ("demoted", "evicted") and uid not in self._promoted_skills:
                try:
                    await psyche_client.execute(
                        """
                        MATCH (s:LearnedSkill {uid: $uid})
                        SET s.is_promoted = false
                        """,
                        {"uid": uid},
                    )
                except Exception:
                    pass  # Best effort

        logger.info(f"[SOAR] Synced {updated} promoted skills to graph")
        return updated

    async def load_from_graph(self, psyche_client: "PsycheClient") -> int:
        """Load promoted skills from FalkorDB on startup.

        Returns:
            Number of skills loaded
        """
        result = await psyche_client.query(
            """
            MATCH (s:LearnedSkill)
            WHERE s.is_promoted = true
            RETURN s.uid as uid, s.effectiveness_score as effectiveness
            """
        )

        for record in result:
            uid = record["uid"]
            self._promoted_skills.add(uid)

            # Create candidate record with existing data
            if uid not in self._candidates:
                self._candidates[uid] = SkillPerformanceRecord(
                    skill_uid=uid,
                    accuracies=[record["effectiveness"] or 0.5],
                    usage_count=1,
                )

        logger.info(f"[SOAR] Loaded {len(result)} promoted skills from graph")
        return len(result)


def calculate_retrieval_boost(
    is_promoted: bool,
    effectiveness_score: float,
    promotion_boost: float = 0.2,
    effectiveness_weight: float = 0.1,
) -> float:
    """Calculate retrieval priority boost for a skill.

    Promoted skills and high-effectiveness skills get retrieval priority.

    Args:
        is_promoted: Whether skill is in 𝒟ᵦₑₛₜ
        effectiveness_score: Skill's effectiveness (0-1)
        promotion_boost: Boost for promoted skills
        effectiveness_weight: Weight for effectiveness contribution

    Returns:
        Total boost to add to similarity score
    """
    boost = 0.0

    if is_promoted:
        boost += promotion_boost

    boost += effectiveness_weight * effectiveness_score

    return boost
