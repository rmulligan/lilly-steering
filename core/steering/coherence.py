"""Cross-zone coherence tracking for timescale integration.

Brain research shows cross-timescale coherence correlates with cognitive
performance. This module tracks alignment between fast-adapting steering
zones (e.g., exploration) and slow-adapting zones (e.g., identity).

High coherence indicates the fast exploration is aligned with long-term
identity goals. Low coherence suggests divergence that may need attention.
"""

from collections import deque
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


@dataclass
class CoherenceRecord:
    """Single coherence measurement between steering zones.

    Attributes:
        cycle: The cognitive cycle number when measured.
        score: Coherence score in [0, 1]. 1.0 = perfectly aligned, 0.0 = opposite.
        fast_zone: Name of the fast-adapting zone (e.g., "exploration").
        slow_zone: Name of the slow-adapting zone (e.g., "identity").
    """
    cycle: int
    score: float
    fast_zone: str = ""
    slow_zone: str = ""


class CrossZoneCoherence:
    """Track alignment between fast and slow steering zones.

    Computes coherence as cosine similarity mapped to [0, 1]:
    - 1.0 = vectors point in same direction (perfect alignment)
    - 0.5 = vectors are orthogonal (independent)
    - 0.0 = vectors point in opposite directions (conflict)

    Maintains a bounded history for trend analysis, allowing detection
    of improving or declining cross-timescale integration.

    Example:
        coherence = CrossZoneCoherence(history_size=50)

        # Record measurements each cycle
        score = coherence.record(fast_vector, slow_vector, cycle=42)

        # Analyze trends
        if coherence.trend() < 0:
            print("Cross-zone coherence declining")

        # Get recent health
        avg = coherence.recent_average(n=10)
    """

    def __init__(self, history_size: int = 50):
        """Initialize coherence tracker.

        Args:
            history_size: Maximum number of records to retain. Older records
                are dropped as new ones are added.
        """
        self.history_size = history_size
        self.history: deque[CoherenceRecord] = deque(maxlen=history_size)

    def compute(self, fast_vector: "np.ndarray", slow_vector: "np.ndarray") -> float:
        """Compute coherence as cosine similarity mapped to [0, 1].

        Args:
            fast_vector: Steering vector from fast-adapting zone.
            slow_vector: Steering vector from slow-adapting zone.

        Returns:
            Coherence score in [0, 1]. Returns 0.0 if either vector is zero.
        """
        if not NUMPY_AVAILABLE:
            return 0.5

        norm_fast = np.linalg.norm(fast_vector)
        norm_slow = np.linalg.norm(slow_vector)

        if norm_fast == 0 or norm_slow == 0:
            return 0.0

        cosine = np.dot(fast_vector, slow_vector) / (norm_fast * norm_slow)
        # Map cosine similarity from [-1, 1] to [0, 1]
        return float((cosine + 1.0) / 2.0)

    def record(
        self,
        fast_vector: "np.ndarray",
        slow_vector: "np.ndarray",
        cycle: int,
        fast_zone: str = "exploration",
        slow_zone: str = "identity",
    ) -> float:
        """Record coherence measurement and return score.

        Args:
            fast_vector: Steering vector from fast-adapting zone.
            slow_vector: Steering vector from slow-adapting zone.
            cycle: Current cognitive cycle number.
            fast_zone: Name of the fast zone (default: "exploration").
            slow_zone: Name of the slow zone (default: "identity").

        Returns:
            Computed coherence score.
        """
        score = self.compute(fast_vector, slow_vector)
        self.history.append(
            CoherenceRecord(
                cycle=cycle,
                score=score,
                fast_zone=fast_zone,
                slow_zone=slow_zone,
            )
        )
        return score

    def trend(self) -> float | None:
        """Compute trend in coherence (positive = improving alignment).

        Uses simple linear regression slope to determine if coherence
        is improving (positive), declining (negative), or stable (near zero).

        Returns:
            Slope of coherence over time, or None if insufficient history
            (requires at least 3 records).
        """
        if len(self.history) < 3:
            return None

        scores = [r.score for r in self.history]
        x = np.arange(len(scores))
        x_mean = x.mean()
        y_mean = np.mean(scores)

        numerator = np.sum((x - x_mean) * (scores - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def recent_average(self, n: int = 10) -> float:
        """Get average coherence over last n cycles.

        Args:
            n: Number of recent records to average.

        Returns:
            Average coherence score. Returns 0.5 (neutral) if no history.
        """
        if not self.history:
            return 0.5
        recent = list(self.history)[-n:]
        return sum(r.score for r in recent) / len(recent)
