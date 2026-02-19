"""SAE Feature Tension Tracking for learned dialectical exploration.

This module tracks which SAE feature pairs create productive cognitive tension,
enabling the system to learn which feature combinations lead to high-surprise,
novel outputs over time.

The tension tracker observes feature transitions between thoughts:
- Features that "flip" (active → inactive or vice versa) are candidates for tension
- High surprise during a flip indicates productive tension
- Tension relationships are persisted to Psyche for long-term learning
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

# Minimum observations before using tension for suggestions
MIN_OBSERVATIONS_FOR_SUGGESTION = 3

# Minimum surprise to record as productive tension
MIN_SURPRISE_FOR_TENSION = 40.0

# Cache size for in-memory tension lookup
MAX_CACHE_SIZE = 1000


@dataclass
class FeatureTension:
    """Record of tension between two SAE features.

    Tracks how often two features co-occur in high-surprise transitions,
    indicating they represent complementary or opposing concepts that
    create productive cognitive tension.

    Attributes:
        feature_a: First feature index (always smaller)
        feature_b: Second feature index (always larger)
        tension_type: Classification of the tension relationship
        surprise_sum: Sum of surprises when both features involved in transition
        observation_count: Number of times this tension was observed
        last_observed: When this tension was last seen
    """

    feature_a: int
    feature_b: int
    tension_type: str = "anti_correlated"  # anti_correlated, complementary, contradictory
    surprise_sum: float = 0.0
    observation_count: int = 0
    last_observed: Optional[datetime] = None

    @property
    def surprise_avg(self) -> float:
        """Average surprise when this tension is observed."""
        if self.observation_count == 0:
            return 0.0
        return self.surprise_sum / self.observation_count

    def record_observation(self, surprise: float) -> None:
        """Record a new observation of this tension."""
        self.surprise_sum += surprise
        self.observation_count += 1
        self.last_observed = datetime.now(timezone.utc)


class TensionTracker:
    """Learns which SAE feature combinations create productive exploration.

    The tracker observes feature transitions between consecutive thoughts:
    1. Features that disappear (were active, now inactive) = "departed"
    2. Features that appear (were inactive, now active) = "arrived"
    3. High surprise during such transitions = productive tension

    Over time, the tracker learns which feature pairs consistently produce
    high-surprise outputs, enabling proactive suggestion of opposing features
    when the cognitive loop stagnates.

    Attributes:
        psyche: Client for persisting tension relationships
        _cache: In-memory cache of tension relationships
        _dirty: Set of cache keys that need persistence
    """

    def __init__(self, psyche_client: Optional["PsycheClient"] = None):
        """Initialize tension tracker.

        Args:
            psyche_client: Client for persistence (optional, can be set later)
        """
        self.psyche = psyche_client
        self._cache: dict[tuple[int, int], FeatureTension] = {}
        self._dirty: set[tuple[int, int]] = set()
        self._persist_lock = asyncio.Lock()

    def set_psyche_client(self, client: "PsycheClient") -> None:
        """Set the Psyche client for persistence."""
        self.psyche = client

    async def load_from_psyche(self, limit: int = 500) -> int:
        """Load existing tension relationships from Psyche.

        Args:
            limit: Maximum relationships to load

        Returns:
            Number of relationships loaded
        """
        if self.psyche is None:
            logger.warning("No Psyche client - cannot load tension relationships")
            return 0

        try:
            results = await self.psyche.query(
                """
                MATCH (t:SAEFeatureTension)
                RETURN t.feature_a as a, t.feature_b as b,
                       t.surprise_sum as surprise_sum,
                       t.observation_count as count,
                       t.tension_type as type
                ORDER BY t.observation_count DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            count = 0
            for record in results:
                key = (record["a"], record["b"])
                self._cache[key] = FeatureTension(
                    feature_a=record["a"],
                    feature_b=record["b"],
                    tension_type=record.get("type", "anti_correlated"),
                    surprise_sum=record.get("surprise_sum", 0.0),
                    observation_count=record.get("count", 0),
                )
                count += 1

            logger.info(f"Loaded {count} tension relationships from Psyche")
            return count

        except Exception as e:
            logger.warning(f"Failed to load tension relationships: {e}")
            return 0

    async def record_observation(
        self,
        features_before: list[tuple[int, float]],
        features_after: list[tuple[int, float]],
        surprise: float,
        min_activation: float = 0.1,
    ) -> list[tuple[int, int]]:
        """Record feature transition and resulting surprise.

        Identifies features that "flipped" between thoughts and records
        tension relationships when surprise is high enough.

        Args:
            features_before: SAE features from previous thought [(idx, activation), ...]
            features_after: SAE features from current thought
            surprise: Measured surprise for this transition
            min_activation: Minimum activation to consider a feature "active"

        Returns:
            List of (feature_a, feature_b) pairs that were recorded as tensions
        """
        # Only record high-surprise transitions
        if surprise < MIN_SURPRISE_FOR_TENSION:
            return []

        # Extract active feature sets
        before_set = {f for f, a in features_before if a >= min_activation}
        after_set = {f for f, a in features_after if a >= min_activation}

        # Features that flipped
        departed = before_set - after_set  # Were active, now inactive
        arrived = after_set - before_set    # Were inactive, now active

        if not departed or not arrived:
            return []

        # Record tension between departed and arrived features
        # Limit to top features to avoid combinatorial explosion
        recorded = []
        for d in list(departed)[:5]:
            for a in list(arrived)[:5]:
                key = self._normalize_key(d, a)
                await self._record_tension(key, surprise)
                recorded.append(key)

        if recorded:
            logger.debug(
                f"Recorded {len(recorded)} tension observations "
                f"(surprise={surprise:.1f}, departed={len(departed)}, arrived={len(arrived)})"
            )

        return recorded

    def _normalize_key(self, feature_a: int, feature_b: int) -> tuple[int, int]:
        """Normalize feature pair to consistent ordering."""
        return (min(feature_a, feature_b), max(feature_a, feature_b))

    async def _record_tension(
        self,
        key: tuple[int, int],
        surprise: float,
    ) -> None:
        """Record a tension observation in cache."""
        if key in self._cache:
            self._cache[key].record_observation(surprise)
        else:
            # Create new tension record
            if len(self._cache) >= MAX_CACHE_SIZE:
                # Evict least-observed entry
                min_key = min(self._cache.keys(), key=lambda k: self._cache[k].observation_count)
                del self._cache[min_key]
                self._dirty.discard(min_key)

            self._cache[key] = FeatureTension(
                feature_a=key[0],
                feature_b=key[1],
                surprise_sum=surprise,
                observation_count=1,
                last_observed=datetime.now(timezone.utc),
            )

        self._dirty.add(key)

    async def persist_dirty(self) -> int:
        """Persist dirty cache entries to Psyche.

        Returns:
            Number of entries persisted
        """
        if self.psyche is None or not self._dirty:
            return 0

        async with self._persist_lock:
            to_persist = list(self._dirty)
            self._dirty.clear()

            persisted = 0
            for key in to_persist:
                if key not in self._cache:
                    continue

                tension = self._cache[key]
                try:
                    await self.psyche.execute(
                        """
                        MERGE (t:SAEFeatureTension {
                            feature_a: $a,
                            feature_b: $b
                        })
                        SET t.surprise_sum = $surprise_sum,
                            t.observation_count = $count,
                            t.tension_type = $type,
                            t.updated_at = $updated_at
                        """,
                        {
                            "a": tension.feature_a,
                            "b": tension.feature_b,
                            "surprise_sum": tension.surprise_sum,
                            "count": tension.observation_count,
                            "type": tension.tension_type,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"Failed to persist tension {key}: {e}")
                    self._dirty.add(key)  # Re-add for retry

            if persisted:
                logger.debug(f"Persisted {persisted} tension relationships to Psyche")

            return persisted

    def get_opposing_features(
        self,
        current_features: list[tuple[int, float]],
        top_k: int = 5,
        min_activation: float = 0.1,
    ) -> list[tuple[int, float]]:
        """Find features that historically create tension with current features.

        Args:
            current_features: Currently active SAE features
            top_k: Maximum opposing features to return
            min_activation: Minimum activation to consider a feature "active"

        Returns:
            List of (feature_idx, expected_surprise) tuples, sorted by expected surprise
        """
        current_set = {f for f, a in current_features if a >= min_activation}

        if not current_set:
            return []

        candidates: dict[int, float] = {}  # feature -> max surprise

        for key, tension in self._cache.items():
            # Skip if not enough observations
            if tension.observation_count < MIN_OBSERVATIONS_FOR_SUGGESTION:
                continue

            # If one feature is active, the other is a candidate
            if key[0] in current_set and key[1] not in current_set:
                candidate = key[1]
            elif key[1] in current_set and key[0] not in current_set:
                candidate = key[0]
            else:
                continue

            # Track highest expected surprise for this candidate
            if candidate not in candidates or tension.surprise_avg > candidates[candidate]:
                candidates[candidate] = tension.surprise_avg

        # Sort by expected surprise (higher = more productive)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        return sorted_candidates[:top_k]

    def get_tension_stats(self) -> dict:
        """Get statistics about tracked tensions.

        Returns:
            Dictionary with cache size, observation counts, etc.
        """
        if not self._cache:
            return {
                "cache_size": 0,
                "total_observations": 0,
                "avg_surprise": 0.0,
                "dirty_count": len(self._dirty),
            }

        total_obs = sum(t.observation_count for t in self._cache.values())
        total_surprise = sum(t.surprise_sum for t in self._cache.values())

        return {
            "cache_size": len(self._cache),
            "total_observations": total_obs,
            "avg_surprise": total_surprise / total_obs if total_obs > 0 else 0.0,
            "dirty_count": len(self._dirty),
            "top_tensions": [
                {
                    "features": (t.feature_a, t.feature_b),
                    "observations": t.observation_count,
                    "avg_surprise": t.surprise_avg,
                }
                for t in sorted(
                    self._cache.values(),
                    key=lambda x: x.observation_count,
                    reverse=True,
                )[:5]
            ],
        }


# Singleton instance
_tension_tracker: Optional[TensionTracker] = None


def get_tension_tracker(psyche_client: Optional["PsycheClient"] = None) -> TensionTracker:
    """Get or create the singleton TensionTracker.

    Args:
        psyche_client: Psyche client for persistence

    Returns:
        The singleton TensionTracker instance
    """
    global _tension_tracker

    if _tension_tracker is None:
        _tension_tracker = TensionTracker(psyche_client)

    elif psyche_client is not None and _tension_tracker.psyche is None:
        _tension_tracker.set_psyche_client(psyche_client)

    return _tension_tracker
