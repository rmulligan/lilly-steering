"""Base class for QD metrics."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.steering.crystal import CrystalEntry


class BaseMetric(ABC):
    """Abstract base class for QD metrics.

    Each metric computes a score in [0, 1] range for a crystal candidate.
    Higher scores indicate better performance on that metric dimension.
    """

    @abstractmethod
    def compute(self, crystal: "CrystalEntry", context: Any) -> float:
        """Compute the metric score for a crystal.

        Args:
            crystal: The crystal entry to score
            context: Metric-specific context (varies by implementation)

        Returns:
            Score in [0, 1] range where higher is better
        """
        pass

    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp a value to the specified range.

        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))
