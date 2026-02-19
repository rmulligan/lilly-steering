"""Presence metric: authenticity from human approval patterns.

Tracks Ryan's approval signals for crystals via SAE feature associations.
Crystals that consistently produce approved features score higher,
indicating they steer toward authentic, valued directions.
"""

from typing import TYPE_CHECKING, Optional

from core.steering.qd.metrics.base import BaseMetric

if TYPE_CHECKING:
    from core.recognition.feature_tracker import ApprovedFeatureTracker
    from core.steering.crystal import CrystalEntry


class PresenceMetric(BaseMetric):
    """Presence metric based on approval pattern tracking.

    Integrates with ApprovedFeatureTracker to compute how aligned
    a crystal is with Ryan's approval patterns. Uses EMA to track
    approval history for each crystal, providing stable presence scores.

    Attributes:
        feature_tracker: Tracker for SAE feature approval patterns
        ema_alpha: EMA alpha for presence score updates
        presence_scores: Cached presence scores by crystal name
    """

    def __init__(
        self,
        feature_tracker: Optional["ApprovedFeatureTracker"] = None,
        ema_alpha: float = 0.1,
    ):
        """Initialize presence metric.

        Args:
            feature_tracker: Tracker for approval patterns.
                If None, presence defaults to 0.5 (neutral).
            ema_alpha: EMA alpha for presence score updates.
                Lower = slower updates, more stable scores.
        """
        self.feature_tracker = feature_tracker
        self.ema_alpha = ema_alpha
        self._presence_scores: dict[str, float] = {}
        self._observation_counts: dict[str, int] = {}

    def compute(
        self,
        crystal: "CrystalEntry",
        sae_features: list[tuple[int, float]] | None = None,
    ) -> float:
        """Compute presence score for a crystal.

        Args:
            crystal: The crystal entry to score
            sae_features: Optional current SAE features for live update.
                If None, returns cached score.

        Returns:
            Presence score in [0, 1] range
        """
        crystal_name = crystal.name

        # Get cached score or initialize to neutral
        current_score = self._presence_scores.get(crystal_name, 0.5)
        obs_count = self._observation_counts.get(crystal_name, 0)

        # If no tracker or no features, return cached/neutral score
        if self.feature_tracker is None or sae_features is None:
            return current_score

        # Get approval bonus from tracker (-1 to 1 range)
        approval_bonus = self.feature_tracker.get_approval_bonus(sae_features)

        # Map approval_bonus from [-1, 1] to [0, 1] for presence
        # -1 = strongly disapproved = 0.0 presence
        # +1 = strongly approved = 1.0 presence
        new_presence = (approval_bonus + 1.0) / 2.0

        # EMA update
        updated_score = (1 - self.ema_alpha) * current_score + self.ema_alpha * new_presence

        # Cache the updated score
        self._presence_scores[crystal_name] = updated_score
        self._observation_counts[crystal_name] = obs_count + 1

        return self.clamp(updated_score)

    def get_observation_count(self, crystal_name: str) -> int:
        """Get the number of observations for a crystal.

        Args:
            crystal_name: Name of the crystal

        Returns:
            Number of observations recorded
        """
        return self._observation_counts.get(crystal_name, 0)

    def has_sufficient_observations(
        self, crystal_name: str, min_observations: int = 5
    ) -> bool:
        """Check if a crystal has enough observations for reliable scoring.

        Args:
            crystal_name: Name of the crystal
            min_observations: Minimum required observations

        Returns:
            True if crystal has at least min_observations
        """
        return self.get_observation_count(crystal_name) >= min_observations

    def reset_crystal(self, crystal_name: str) -> None:
        """Reset presence tracking for a crystal.

        Args:
            crystal_name: Name of the crystal to reset
        """
        self._presence_scores.pop(crystal_name, None)
        self._observation_counts.pop(crystal_name, None)

    def get_stats(self) -> dict:
        """Get presence metric statistics.

        Returns:
            Dictionary with tracking statistics
        """
        if not self._presence_scores:
            return {
                "crystals_tracked": 0,
                "avg_presence": 0.5,
                "avg_observations": 0,
            }

        return {
            "crystals_tracked": len(self._presence_scores),
            "avg_presence": sum(self._presence_scores.values()) / len(self._presence_scores),
            "avg_observations": sum(self._observation_counts.values()) / len(self._observation_counts),
            "high_presence_crystals": [
                name for name, score in self._presence_scores.items()
                if score > 0.7
            ],
            "low_presence_crystals": [
                name for name, score in self._presence_scores.items()
                if score < 0.3
            ],
        }
