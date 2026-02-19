"""Novelty metric: diversity driver via inverse similarity.

Measures how different a crystal's steering direction is from recently
selected vectors. This drives selection diversity - crystals that explore
new directions score higher.
"""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from core.steering.qd.metrics.base import BaseMetric
from core.steering.utils import cosine_similarity

if TYPE_CHECKING:
    from core.steering.crystal import CrystalEntry


class NoveltyMetric(BaseMetric):
    """Novelty metric based on inverse similarity to recent selections.

    Computes how different a crystal is from recently selected vectors.
    Uses recency-weighted maximum similarity - vectors selected more
    recently contribute more to the novelty calculation.

    High novelty = different from recent selections = encourages diversity
    Low novelty = similar to recent selections = discourages repetition

    Attributes:
        window_size: Number of recent selections to track
        decay: Recency decay factor (0.9 = 10% less weight per step back)
    """

    def __init__(self, window_size: int = 20, decay: float = 0.9):
        """Initialize novelty metric.

        Args:
            window_size: Maximum number of recent selections to track
            decay: Recency decay factor (applied per step back in history)
        """
        self.window_size = window_size
        self.decay = decay
        self._recent_selections: deque[np.ndarray] = deque(maxlen=window_size)

    def compute(
        self,
        crystal: "CrystalEntry",
        recent_selections: deque[np.ndarray] | None = None,
    ) -> float:
        """Compute novelty score for a crystal.

        Args:
            crystal: The crystal entry to score
            recent_selections: Optional override for recent selections.
                If None, uses internal tracking.

        Returns:
            Novelty score in [0, 1] range. Higher = more novel.
        """
        selections = recent_selections if recent_selections is not None else self._recent_selections

        if not selections:
            # No history - maximum novelty
            return 1.0

        # Compute weighted maximum similarity
        # More recent selections have higher weight
        max_weighted_similarity = 0.0

        for i, recent_vec in enumerate(reversed(list(selections))):
            # i=0 is most recent
            recency_weight = self.decay**i
            similarity = cosine_similarity(crystal.vector, recent_vec)

            # We care about absolute similarity (anti-aligned is still similar in behavior)
            weighted_sim = recency_weight * abs(similarity)
            max_weighted_similarity = max(max_weighted_similarity, weighted_sim)

        # Novelty is inverse of similarity
        # 1.0 = completely novel, 0.0 = identical to recent
        novelty = 1.0 - max_weighted_similarity

        return self.clamp(novelty)

    def record_selection(self, vector: np.ndarray) -> None:
        """Record a selected vector for novelty tracking.

        Args:
            vector: The selected steering vector
        """
        self._recent_selections.append(vector.copy())

    def clear_history(self) -> None:
        """Clear the selection history."""
        self._recent_selections.clear()
