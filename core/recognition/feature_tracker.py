"""Track SAE features correlated with Ryan's approval.

Learns which internal activation patterns Ryan consistently approves of,
providing a fitness signal for emergence detection and steering refinement.

The tracker uses EMA (Exponential Moving Average) to slowly update approval
rates, preventing overfit to single signals while building stable patterns.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

from core.recognition.schema import RecognitionSignal, SignalType

logger = logging.getLogger(__name__)

# Configuration
APPROVAL_EMA_ALPHA = 0.1  # Slow learning - ~10 observations to shift significantly
MIN_ACTIVATION_FOR_ATTRIBUTION = 0.1  # Skip weak feature activations
MIN_OBSERVATIONS_FOR_SIGNAL = 5  # Need this many observations before trusting
MAX_CACHE_SIZE = 5000  # Maximum tracked features


@dataclass
class ApprovedFeaturePattern:
    """Tracks approval statistics for a single SAE feature.

    Attributes:
        feature_idx: The SAE feature index
        approval_count: Total APPROVE signals when this feature was active
        disapproval_count: Total DISAPPROVE signals
        curious_count: Total CURIOUS signals
        approval_rate: EMA-updated approval rate (0.0 to 1.0)
        last_updated: When this pattern was last updated
        total_observations: Sum of all signal counts
    """

    feature_idx: int
    approval_count: int = 0
    disapproval_count: int = 0
    curious_count: int = 0
    approval_rate: float = 0.5  # Start neutral
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_observations(self) -> int:
        """Total number of signals observed for this feature."""
        return self.approval_count + self.disapproval_count + self.curious_count

    @property
    def is_significant(self) -> bool:
        """Whether we have enough observations to trust this pattern."""
        return self.total_observations >= MIN_OBSERVATIONS_FOR_SIGNAL

    @property
    def approval_strength(self) -> float:
        """Approval strength centered at 0 (-0.5 to +0.5 range).

        Positive = tends toward approval, negative = tends toward disapproval.
        """
        return self.approval_rate - 0.5


class ApprovedFeatureTracker:
    """Learn which SAE features Ryan approves of.

    Maintains a cache of feature patterns, updating approval rates
    as signals come in. Can compute approval bonuses for current
    feature activations to inform emergence scoring.

    Attributes:
        psyche: Client for optional persistence
        stats: Overall tracking statistics
    """

    def __init__(self, psyche: Optional["PsycheClient"] = None):
        """Initialize the tracker.

        Args:
            psyche: Optional client for persistence to graph
        """
        self.psyche = psyche
        self._cache: dict[int, ApprovedFeaturePattern] = {}
        self._dirty: set[int] = set()
        self._persist_lock = asyncio.Lock()
        self._total_signals_processed = 0

    def set_psyche_client(self, client: "PsycheClient") -> None:
        """Set the Psyche client for persistence."""
        self.psyche = client

    async def record_signal(
        self,
        signal: RecognitionSignal,
        sae_features: list[tuple[int, float]],
    ) -> int:
        """Update feature patterns based on a recognition signal.

        For each feature that was active during the recognized thought,
        update its approval statistics via EMA.

        Args:
            signal: Recognition signal from Ryan
            sae_features: SAE features active during the thought [(idx, activation), ...]

        Returns:
            Number of features updated
        """
        if not sae_features:
            return 0

        # Determine target approval rate based on signal type
        if signal.signal_type == SignalType.APPROVE:
            target = 1.0
        elif signal.signal_type == SignalType.DISAPPROVE:
            target = 0.0
        else:  # CURIOUS
            target = 0.6  # Slight positive - engagement is good

        updated = 0
        for feature_idx, activation in sae_features:
            if activation < MIN_ACTIVATION_FOR_ATTRIBUTION:
                continue

            pattern = self._get_or_create_pattern(feature_idx)

            # Update counts
            if signal.signal_type == SignalType.APPROVE:
                pattern.approval_count += 1
            elif signal.signal_type == SignalType.DISAPPROVE:
                pattern.disapproval_count += 1
            else:
                pattern.curious_count += 1

            # EMA update of approval rate
            # Weight by activation strength - stronger activations have more influence
            effective_alpha = APPROVAL_EMA_ALPHA * min(1.0, activation)
            pattern.approval_rate = (
                (1 - effective_alpha) * pattern.approval_rate + effective_alpha * target
            )
            pattern.last_updated = datetime.now(timezone.utc)

            self._cache[feature_idx] = pattern
            self._dirty.add(feature_idx)
            updated += 1

        self._total_signals_processed += 1
        logger.debug(
            f"Recorded {signal.signal_type.value} signal across {updated} features"
        )

        return updated

    def _get_or_create_pattern(self, feature_idx: int) -> ApprovedFeaturePattern:
        """Get existing pattern or create new one.

        Handles cache size limits by evicting least-observed patterns.
        """
        if feature_idx in self._cache:
            return self._cache[feature_idx]

        # Evict if at capacity
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Find least-observed pattern
            min_idx = min(
                self._cache.keys(), key=lambda k: self._cache[k].total_observations
            )
            del self._cache[min_idx]
            self._dirty.discard(min_idx)
            logger.debug(f"Evicted feature pattern {min_idx} from cache")

        return ApprovedFeaturePattern(feature_idx=feature_idx)

    def get_approval_bonus(self, sae_features: list[tuple[int, float]]) -> float:
        """Compute approval bonus for current feature activations.

        Returns a weighted average of approval strengths for active features,
        indicating whether the current activation pattern tends toward
        Ryan-approved or Ryan-disapproved behaviors.

        Args:
            sae_features: Currently active features [(idx, activation), ...]

        Returns:
            Approval bonus in range [-1, 1]:
            - Positive: Features tend toward approval
            - Negative: Features tend toward disapproval
            - Zero: Neutral or unknown
        """
        if not sae_features:
            return 0.0

        total_weighted = 0.0
        total_activation = 0.0

        for feature_idx, activation in sae_features:
            if activation < MIN_ACTIVATION_FOR_ATTRIBUTION:
                continue

            if feature_idx in self._cache:
                pattern = self._cache[feature_idx]

                # Only use patterns with sufficient observations
                if pattern.is_significant:
                    # Weight by activation AND observation count (more observations = more trust)
                    confidence = min(1.0, pattern.total_observations / 20.0)
                    weight = activation * confidence

                    # approval_strength is -0.5 to +0.5, scale to -1 to +1
                    total_weighted += weight * (pattern.approval_strength * 2)
                    total_activation += weight

        if total_activation == 0:
            return 0.0

        return total_weighted / total_activation

    def get_approved_features(
        self, min_approval_rate: float = 0.7, min_observations: int = 5
    ) -> list[tuple[int, float]]:
        """Get features with high approval rates.

        Args:
            min_approval_rate: Minimum approval rate threshold
            min_observations: Minimum observation count

        Returns:
            List of (feature_idx, approval_rate) tuples, sorted by rate
        """
        approved = []
        for feature_idx, pattern in self._cache.items():
            if (
                pattern.approval_rate >= min_approval_rate
                and pattern.total_observations >= min_observations
            ):
                approved.append((feature_idx, pattern.approval_rate))

        approved.sort(key=lambda x: x[1], reverse=True)
        return approved

    def get_disapproved_features(
        self, max_approval_rate: float = 0.3, min_observations: int = 5
    ) -> list[tuple[int, float]]:
        """Get features with low approval rates.

        Args:
            max_approval_rate: Maximum approval rate threshold
            min_observations: Minimum observation count

        Returns:
            List of (feature_idx, approval_rate) tuples, sorted by rate (ascending)
        """
        disapproved = []
        for feature_idx, pattern in self._cache.items():
            if (
                pattern.approval_rate <= max_approval_rate
                and pattern.total_observations >= min_observations
            ):
                disapproved.append((feature_idx, pattern.approval_rate))

        disapproved.sort(key=lambda x: x[1])  # Ascending - most disapproved first
        return disapproved

    async def persist_dirty(self) -> int:
        """Persist dirty patterns to Psyche.

        Returns:
            Number of patterns persisted
        """
        if self.psyche is None or not self._dirty:
            return 0

        async with self._persist_lock:
            to_persist = list(self._dirty)
            self._dirty.clear()

            persisted = 0
            for feature_idx in to_persist:
                if feature_idx not in self._cache:
                    continue

                pattern = self._cache[feature_idx]
                try:
                    # Use a custom query to upsert approval patterns
                    await self.psyche.execute(
                        """
                        MERGE (f:SAEFeature {index: $feature_idx})
                        SET f.approval_rate = $approval_rate,
                            f.approval_count = $approval_count,
                            f.disapproval_count = $disapproval_count,
                            f.curious_count = $curious_count,
                            f.approval_updated_at = $updated_at
                        """,
                        {
                            "feature_idx": feature_idx,
                            "approval_rate": pattern.approval_rate,
                            "approval_count": pattern.approval_count,
                            "disapproval_count": pattern.disapproval_count,
                            "curious_count": pattern.curious_count,
                            "updated_at": pattern.last_updated.isoformat(),
                        },
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"Failed to persist approval pattern {feature_idx}: {e}")
                    self._dirty.add(feature_idx)  # Re-add for retry

            if persisted:
                logger.debug(f"Persisted {persisted} approval patterns to Psyche")

            return persisted

    async def load_from_psyche(self, limit: int = 1000) -> int:
        """Load existing approval patterns from Psyche.

        Args:
            limit: Maximum patterns to load

        Returns:
            Number of patterns loaded
        """
        if self.psyche is None:
            logger.warning("No Psyche client - cannot load approval patterns")
            return 0

        try:
            results = await self.psyche.query(
                """
                MATCH (f:SAEFeature)
                WHERE f.approval_rate IS NOT NULL
                RETURN f.index as feature_idx,
                       f.approval_rate as approval_rate,
                       f.approval_count as approval_count,
                       f.disapproval_count as disapproval_count,
                       f.curious_count as curious_count,
                       f.approval_updated_at as updated_at
                ORDER BY f.approval_count + f.disapproval_count + f.curious_count DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            count = 0
            for record in results:
                feature_idx = record["feature_idx"]

                # Parse timestamp
                updated_at = datetime.now(timezone.utc)
                if record.get("updated_at"):
                    try:
                        updated_at = datetime.fromisoformat(record["updated_at"])
                    except (ValueError, TypeError):
                        pass

                self._cache[feature_idx] = ApprovedFeaturePattern(
                    feature_idx=feature_idx,
                    approval_rate=record.get("approval_rate", 0.5),
                    approval_count=record.get("approval_count", 0),
                    disapproval_count=record.get("disapproval_count", 0),
                    curious_count=record.get("curious_count", 0),
                    last_updated=updated_at,
                )
                count += 1

            logger.info(f"Loaded {count} approval patterns from Psyche")
            return count

        except Exception as e:
            logger.warning(f"Failed to load approval patterns: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get tracking statistics.

        Returns:
            Dictionary with cache stats, pattern distributions, etc.
        """
        if not self._cache:
            return {
                "cache_size": 0,
                "total_signals_processed": self._total_signals_processed,
                "avg_approval_rate": 0.5,
                "significant_features": 0,
                "dirty_count": len(self._dirty),
            }

        significant = [p for p in self._cache.values() if p.is_significant]
        avg_rate = (
            sum(p.approval_rate for p in self._cache.values()) / len(self._cache)
        )

        return {
            "cache_size": len(self._cache),
            "total_signals_processed": self._total_signals_processed,
            "avg_approval_rate": avg_rate,
            "significant_features": len(significant),
            "dirty_count": len(self._dirty),
            "top_approved": [
                {"feature": p.feature_idx, "rate": p.approval_rate, "obs": p.total_observations}
                for p in sorted(
                    significant, key=lambda x: x.approval_rate, reverse=True
                )[:5]
            ],
            "top_disapproved": [
                {"feature": p.feature_idx, "rate": p.approval_rate, "obs": p.total_observations}
                for p in sorted(significant, key=lambda x: x.approval_rate)[:5]
            ],
        }
