"""Coefficient Optimizer for steering vectors.

Adjusts vector coefficients based on experience valence,
implementing a simple reinforcement learning approach.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from .vector_library import VectorLibrary

logger = logging.getLogger(__name__)


@dataclass
class ValenceFeedback:
    """Feedback from an experience.

    Attributes:
        valence: Affective valence (-1 to 1)
        arousal: Affective arousal (0 to 1)
        active_vectors: Which vectors were active
        timestamp: When feedback received
    """
    valence: float
    arousal: float
    active_vectors: list[str]
    timestamp: datetime


class CoefficientOptimizer:
    """Optimizes steering coefficients from valence feedback.

    Implements a simple reinforcement approach:
    - Positive valence -> reinforce active vectors
    - Negative valence -> weaken active vectors
    - Neutral -> no change

    Also tracks patterns to identify problematic vectors.

    Attributes:
        library: The vector library to optimize
        learning_rate: How much to adjust per feedback
        positive_threshold: Valence above this reinforces
        negative_threshold: Valence below this weakens
    """

    def __init__(
        self,
        library: VectorLibrary,
        learning_rate: float = 0.05,
        positive_threshold: float = 0.3,
        negative_threshold: float = -0.2,
    ):
        """Initialize CoefficientOptimizer.

        Args:
            library: Vector library to optimize
            learning_rate: Coefficient adjustment per feedback
            positive_threshold: Valence to trigger reinforcement
            negative_threshold: Valence to trigger weakening
        """
        self.library = library
        self.learning_rate = learning_rate
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self._max_history = 1000
        self._feedback_history: deque[ValenceFeedback] = deque(maxlen=self._max_history)

    def record_feedback(
        self,
        valence: float,
        arousal: float,
        active_vectors: list[str],
    ) -> dict[str, str]:
        """Record valence feedback and adjust coefficients.

        Args:
            valence: Experience valence (-1 to 1)
            arousal: Experience arousal (0 to 1)
            active_vectors: Which vectors were active

        Returns:
            Dict mapping vector name to action taken
        """
        feedback = ValenceFeedback(
            valence=valence,
            arousal=arousal,
            active_vectors=active_vectors,
            timestamp=datetime.now(timezone.utc),
        )

        self._feedback_history.append(feedback)
        # deque with maxlen automatically discards oldest entries

        actions = {}

        # Scale adjustment by arousal (stronger feelings = larger adjustment)
        adjustment = self.learning_rate * (0.5 + arousal * 0.5)

        if valence >= self.positive_threshold:
            # Reinforce active vectors
            for name in active_vectors:
                self.library.reinforce(name, adjustment)
                actions[name] = "reinforced"

        elif valence <= self.negative_threshold:
            # Weaken active vectors
            for name in active_vectors:
                self.library.weaken(name, adjustment)
                actions[name] = "weakened"

        else:
            # Neutral - no adjustment
            for name in active_vectors:
                actions[name] = "unchanged"

        if actions:
            logger.debug(
                f"Coefficient feedback (valence={valence:.2f}): "
                f"{', '.join(f'{k}={v}' for k, v in actions.items())}"
            )

        return actions

    def get_problematic_vectors(self, min_negative_ratio: float = 0.6) -> list[str]:
        """Identify vectors associated with negative experiences.

        Args:
            min_negative_ratio: Threshold for flagging as problematic

        Returns:
            List of vector names with high negative association
        """
        # Count positive/negative associations per vector
        # Use consistent thresholds with record_feedback logic
        vector_counts: dict[str, dict[str, int]] = {}

        for feedback in self._feedback_history:
            # Use same thresholds as record_feedback for consistency
            if feedback.valence >= self.positive_threshold:
                label = "positive"
            elif feedback.valence <= self.negative_threshold:
                label = "negative"
            else:
                continue  # Skip neutral feedback

            for name in feedback.active_vectors:
                if name not in vector_counts:
                    vector_counts[name] = {"positive": 0, "negative": 0}
                vector_counts[name][label] += 1

        # Find problematic vectors
        problematic = []
        for name, counts in vector_counts.items():
            total = counts["positive"] + counts["negative"]
            if total >= 10:  # Need enough data
                negative_ratio = counts["negative"] / total
                if negative_ratio >= min_negative_ratio:
                    problematic.append(name)

        return problematic

    def get_statistics(self) -> dict:
        """Get optimization statistics."""
        if not self._feedback_history:
            return {"feedback_count": 0}

        valences = [f.valence for f in self._feedback_history]

        return {
            "feedback_count": len(self._feedback_history),
            "mean_valence": sum(valences) / len(valences),
            "positive_count": sum(1 for v in valences if v >= self.positive_threshold),
            "negative_count": sum(1 for v in valences if v <= self.negative_threshold),
            "neutral_count": sum(
                1 for v in valences
                if self.negative_threshold < v < self.positive_threshold
            ),
        }
