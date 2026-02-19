"""Teacher policy for SOAR curriculum learning.

Implements the outer loop of SOAR: a teacher policy that generates
skill curricula optimized for student (cognitive cycle) improvement.

SOAR Principle: Teacher generates stepping stones optimized for
student improvement, not teacher performance on the same problems.

Reference: https://arxiv.org/html/2601.18778v1
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.curriculum.skill_tracker import SkillEffectivenessTracker
    from core.cognitive.simulation.schemas import Hypothesis
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

STRUCTURAL_KEYWORDS = [
    "when",
    "if",
    "pattern",
    "structure",
    "approach",
    "method",
    "strategy",
    "trigger",
    "condition",
    "context",
    "involves",
    "requires",
    "depends",
    "before",
    "after",
]


@dataclass
class TeacherConfig:
    """Configuration for teacher skill generation policy.

    Attributes:
        exploration_rate: Probability of generating exploratory skills
        focus_on_weak_spots: Whether to prioritize low-performance areas
        structural_emphasis: Weight for structural patterns vs conclusions
        batch_size: Number of skills to generate per teaching episode
        policy_update_alpha: EMA alpha for policy weight updates
        min_verification_rate: Threshold below which an operation is a "weak spot"
    """

    exploration_rate: float = 0.2
    focus_on_weak_spots: bool = True
    structural_emphasis: float = 0.7
    batch_size: int = 3
    policy_update_alpha: float = 0.2
    min_verification_rate: float = 0.5


@dataclass
class WeakSpot:
    """A cognitive operation with low verification rate."""

    cognitive_operation: str
    avg_verification_rate: float
    hypothesis_count: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TeacherPolicy:
    """Teacher policy for generating skill curricula.

    SOAR principle: Teacher generates stepping stones optimized for
    student improvement, not teacher performance on the same problems.

    The teacher:
    1. Identifies weak spots (low verification rate cognitive operations)
    2. Selects hypotheses to generate skills from, prioritizing weak spots
    3. Updates policy weights based on skill effectiveness (RLOO-style)

    Attributes:
        config: Teacher configuration
        _tracker: Skill effectiveness tracker for improvement data
        _client: PsycheClient for graph queries
        _policy_weights: Learned weights per cognitive operation
        _weak_spots_cache: Cached weak spots (refreshed periodically)
    """

    def __init__(
        self,
        config: Optional[TeacherConfig] = None,
        effectiveness_tracker: Optional["SkillEffectivenessTracker"] = None,
        psyche_client: Optional["PsycheClient"] = None,
    ):
        """Initialize teacher policy.

        Args:
            config: Teacher configuration (uses defaults if None)
            effectiveness_tracker: Tracker for skill effectiveness data
            psyche_client: Client for graph queries
        """
        self.config = config or TeacherConfig()
        self._tracker = effectiveness_tracker
        self._client = psyche_client
        self._policy_weights: dict[str, float] = {}
        self._weak_spots_cache: list[WeakSpot] = []
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds: int = 300  # 5 minute cache

    async def identify_weak_spots(self, force_refresh: bool = False) -> list[WeakSpot]:
        """Identify cognitive operations with low verification rates.

        These are "hard problems" where skills could help most.

        Args:
            force_refresh: Force refresh of cached weak spots

        Returns:
            List of WeakSpot instances ordered by verification rate (lowest first)
        """
        if not self._client:
            return []

        # Check cache validity
        now = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._weak_spots_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            return self._weak_spots_cache

        try:
            results = await self._client.query(
                """
                MATCH (h:Hypothesis)
                WHERE h.verification_rate IS NOT NULL
                  AND h.cognitive_operation IS NOT NULL
                WITH h.cognitive_operation as op,
                     avg(h.verification_rate) as avg_rate,
                     count(h) as hyp_count
                WHERE avg_rate < $min_rate
                RETURN op, avg_rate, hyp_count
                ORDER BY avg_rate ASC
                LIMIT 10
                """,
                {"min_rate": self.config.min_verification_rate},
            )

            self._weak_spots_cache = [
                WeakSpot(
                    cognitive_operation=r["op"],
                    avg_verification_rate=r["avg_rate"],
                    hypothesis_count=r["hyp_count"],
                )
                for r in results
                if r["op"]
            ]
            self._cache_timestamp = now

            if self._weak_spots_cache:
                logger.info(
                    f"[TEACHER] Identified {len(self._weak_spots_cache)} weak spots: "
                    f"{[ws.cognitive_operation for ws in self._weak_spots_cache[:3]]}"
                )

        except Exception as e:
            logger.warning(f"[TEACHER] Failed to identify weak spots: {e}")
            return []

        return self._weak_spots_cache

    def get_weak_spot_operations(self) -> list[str]:
        """Get list of cognitive operation names that are weak spots.

        Returns cached list without async query - use identify_weak_spots()
        to refresh the cache first.

        Returns:
            List of cognitive operation names
        """
        return [ws.cognitive_operation for ws in self._weak_spots_cache]

    async def select_generation_targets(
        self,
        verified_hypotheses: list["Hypothesis"],
    ) -> list["Hypothesis"]:
        """Select hypotheses for skill generation based on teacher policy.

        Prioritizes:
        1. Hypotheses in weak cognitive operation areas
        2. Hypotheses with strong structural patterns
        3. Hypotheses with clear positive/negative examples
        4. Hypotheses in cognitive operations with high policy weights

        Args:
            verified_hypotheses: List of recently verified hypotheses

        Returns:
            Selected hypotheses (up to batch_size) ordered by priority
        """
        if not verified_hypotheses:
            return []

        # Refresh weak spots if focusing on them
        weak_spot_ops: list[str] = []
        if self.config.focus_on_weak_spots:
            weak_spots = await self.identify_weak_spots()
            weak_spot_ops = [ws.cognitive_operation for ws in weak_spots]

        scored_hypotheses: list[tuple["Hypothesis", float]] = []
        for h in verified_hypotheses:
            score = self._score_hypothesis(h, weak_spot_ops)
            scored_hypotheses.append((h, score))

        # Sort by score descending and take batch_size
        scored_hypotheses.sort(key=lambda x: x[1], reverse=True)
        selected = [h for h, _ in scored_hypotheses[: self.config.batch_size]]

        if selected:
            logger.debug(
                f"[TEACHER] Selected {len(selected)} hypotheses for skill generation "
                f"(from {len(verified_hypotheses)} candidates)"
            )

        return selected

    def _score_hypothesis(
        self,
        hypothesis: "Hypothesis",
        weak_spot_ops: list[str],
    ) -> float:
        """Score a hypothesis for skill generation priority.

        Args:
            hypothesis: Hypothesis to score
            weak_spot_ops: List of weak spot cognitive operations

        Returns:
            Priority score (higher = more likely to generate skill)
        """
        score = 0.0

        # Weak spot boost (highest priority)
        if hypothesis.cognitive_operation in weak_spot_ops:
            # Higher boost for worse weak spots (earlier in list)
            try:
                idx = weak_spot_ops.index(hypothesis.cognitive_operation)
                boost = 0.5 * (1.0 - idx / max(len(weak_spot_ops), 1))
                score += boost
            except ValueError:
                pass

        # Structural pattern quality
        pattern_count = len(hypothesis.patterns_extracted) if hypothesis.patterns_extracted else 0
        structural_count = self._count_structural_patterns(
            hypothesis.patterns_extracted or []
        )
        score += min(0.3, structural_count * 0.1) * self.config.structural_emphasis

        # Example quality (both positive and negative present)
        if hypothesis.positive_example and hypothesis.negative_example:
            score += 0.2

        # Policy weight from past effectiveness
        if hypothesis.cognitive_operation in self._policy_weights:
            weight = self._policy_weights[hypothesis.cognitive_operation]
            score += max(-0.2, min(0.2, weight))  # Clamp to [-0.2, 0.2]

        return score

    def _count_structural_patterns(self, patterns: list[str]) -> int:
        """Count patterns that emphasize structure over conclusions.

        SOAR finding: Question structure matters more than answer correctness.

        Args:
            patterns: List of pattern strings

        Returns:
            Count of structural patterns
        """
        count = 0
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if any(kw in pattern_lower for kw in STRUCTURAL_KEYWORDS):
                count += 1

        return count

    async def update_policy_weights(
        self,
        skill_uid: str,
        improvement: float,
    ) -> Optional[tuple[str, float, float]]:
        """Update teacher policy based on skill effectiveness.

        SOAR RLOO-style update: reward teacher based on student improvement.
        Positive improvements increase the weight for that cognitive operation,
        negative improvements decrease it.

        Args:
            skill_uid: UID of the skill
            improvement: Improvement over baseline (can be negative)

        Returns:
            Tuple of (cognitive_operation, old_weight, new_weight) if updated,
            None if skill not found or client unavailable
        """
        if not self._client:
            return None

        try:
            # Get skill's cognitive operation
            result = await self._client.query(
                """
                MATCH (s:LearnedSkill {uid: $uid})
                RETURN s.cognitive_operation as op
                """,
                {"uid": skill_uid},
            )

            if not result or not result[0].get("op"):
                return None

            op = result[0]["op"]

            # Update policy weight with EMA
            alpha = self.config.policy_update_alpha
            old_weight = self._policy_weights.get(op, 0.0)
            new_weight = alpha * improvement + (1 - alpha) * old_weight
            self._policy_weights[op] = new_weight

            logger.info(
                f"[TEACHER] Updated policy weight for '{op}': "
                f"{old_weight:.3f} -> {new_weight:.3f} "
                f"(improvement={improvement:+.3f})"
            )

            return (op, old_weight, new_weight)

        except Exception as e:
            logger.warning(f"[TEACHER] Failed to update policy weights: {e}")
            return None

    def get_policy_weights(self) -> dict[str, float]:
        """Get current policy weights for all cognitive operations.

        Returns:
            Dict mapping cognitive operation to weight
        """
        return self._policy_weights.copy()

    def get_policy_stats(self) -> dict:
        """Get statistics about teacher policy state.

        Returns:
            Dict with policy statistics
        """
        weights = list(self._policy_weights.values())
        return {
            "tracked_operations": len(self._policy_weights),
            "weak_spots_cached": len(self._weak_spots_cache),
            "avg_weight": sum(weights) / len(weights) if weights else 0.0,
            "max_weight": max(weights) if weights else 0.0,
            "min_weight": min(weights) if weights else 0.0,
            "config": {
                "exploration_rate": self.config.exploration_rate,
                "focus_on_weak_spots": self.config.focus_on_weak_spots,
                "structural_emphasis": self.config.structural_emphasis,
                "batch_size": self.config.batch_size,
            },
        }


def filter_structural_patterns(patterns: list[str]) -> list[str]:
    """Filter patterns to emphasize structural quality.

    SOAR finding: Question structure matters more than answer correctness.

    Args:
        patterns: List of pattern strings

    Returns:
        Patterns reordered with structural patterns first
    """
    structural = []
    other = []

    for pattern in patterns:
        pattern_lower = pattern.lower()
        if any(kw in pattern_lower for kw in STRUCTURAL_KEYWORDS):
            structural.append(pattern)
        else:
            other.append(pattern)

    return structural + other
