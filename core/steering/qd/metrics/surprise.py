"""Surprise metric: emergence signal from activation surprise.

Wraps the existing avg_surprise from CrystalEntry, normalizing it to
[0, 1] range for consistent QD scoring.
"""

from typing import TYPE_CHECKING

from core.steering.qd.metrics.base import BaseMetric

if TYPE_CHECKING:
    from core.steering.crystal import CrystalEntry


class SurpriseMetric(BaseMetric):
    """Surprise metric based on crystal's average surprise score.

    Uses the existing avg_surprise property from CrystalEntry, which
    tracks the mean surprise value when the crystal was selected.
    Higher surprise indicates the crystal produced more unexpected
    (and potentially interesting) activations.

    Attributes:
        normalize_max: Maximum surprise value for normalization
    """

    def __init__(self, normalize_max: float = 100.0):
        """Initialize surprise metric.

        Args:
            normalize_max: Maximum surprise value for normalization.
                Values above this are clamped to 1.0.
        """
        self.normalize_max = normalize_max

    def compute(self, crystal: "CrystalEntry", context: None = None) -> float:
        """Compute surprise score for a crystal.

        Args:
            crystal: The crystal entry to score
            context: Unused, present for interface consistency

        Returns:
            Surprise score in [0, 1] range
        """
        # Get avg_surprise from crystal (defaults to birth_surprise if never selected)
        surprise = crystal.avg_surprise

        # Normalize to [0, 1]
        normalized = surprise / self.normalize_max

        return self.clamp(normalized)
