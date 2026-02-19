"""Hypothesis-driven steering vectors with effectiveness tracking.

This module provides dataclasses for steering vectors that are derived from
cognitive hypotheses. These vectors implement the outcome-based self-steering
loop where simulation proposes experiments, generation executes them via
learned steering vectors, and verification feedback reinforces what works.

Key concepts:
- HypothesisSteeringVector: A steering vector linked to a specific hypothesis
  and cognitive operation, with effectiveness tracking via EMA updates
- CapacityState: Tracks the estimated "capacity" for steering magnitude
  based on observed effect history, enabling dynamic budget allocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """Create UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class HypothesisSteeringVector:
    """A steering vector derived from a cognitive hypothesis.

    Links a steering vector to the hypothesis that motivated it and the
    cognitive operation it aims to induce. Tracks effectiveness through
    verification feedback, using EMA updates to adjust the score.

    The vector can be applied during generation to steer cognition toward
    the desired operation. Verification results (from predictions) update
    the effectiveness score, allowing the system to learn which steering
    approaches work.

    Attributes:
        uid: Unique identifier for this steering vector
        hypothesis_uid: UID of the hypothesis that generated this vector
        cognitive_operation: The cognitive operation this vector induces
            (e.g., "explore_emergence", "deepen_understanding", "seek_contradiction")
        vector_data: The steering vector values (list of floats)
        layer: Transformer layer where this vector is applied
        effectiveness_score: EMA-tracked effectiveness (0.0 to 1.0)
        application_count: Number of times this vector has been applied
        verified_count: Number of verified predictions when this vector was active
        falsified_count: Number of falsified predictions when this vector was active
        measured_capacity: Estimated maximum magnitude before saturation
        created_at: Timestamp of creation
        last_applied: Timestamp of last application (None if never applied)
    """

    uid: str
    hypothesis_uid: str
    cognitive_operation: str
    vector_data: list[float]
    layer: int
    effectiveness_score: float = 0.5
    application_count: int = 0
    verified_count: int = 0
    falsified_count: int = 0
    measured_capacity: float = 2.0
    created_at: datetime = field(default_factory=utc_now)
    last_applied: Optional[datetime] = None

    def update_effectiveness(self, verified: bool, alpha: float = 0.2) -> None:
        """Update effectiveness score using exponential moving average.

        Uses EMA to smoothly adjust effectiveness based on verification
        outcomes. Verified predictions increase the score, falsified ones
        decrease it.

        Args:
            verified: True if the prediction was verified, False if falsified
            alpha: Learning rate for EMA (0.0 to 1.0). Higher = faster adaptation.
        """
        target = 1.0 if verified else 0.0
        self.effectiveness_score = self.effectiveness_score + alpha * (
            target - self.effectiveness_score
        )

        # Clamp to valid range
        self.effectiveness_score = max(0.0, min(1.0, self.effectiveness_score))

        # Update verification counts
        if verified:
            self.verified_count += 1
        else:
            self.falsified_count += 1

    def record_application(self) -> None:
        """Record that this vector was applied during generation.

        Updates application count and last_applied timestamp.
        """
        self.application_count += 1
        self.last_applied = utc_now()

    def should_prune(
        self,
        min_applications: int = 10,
        min_effectiveness: float = 0.2,
        max_falsified_ratio: float = 0.7,
    ) -> bool:
        """Determine if this vector should be pruned from the population.

        Pruning criteria:
        1. Must have enough applications to evaluate (min_applications)
        2. Effectiveness below minimum threshold (min_effectiveness), OR
        3. Falsification ratio above maximum threshold (max_falsified_ratio)

        Args:
            min_applications: Minimum applications before pruning eligible.
                Defaults to 10 (matches settings.steering_min_applications).
            min_effectiveness: Effectiveness threshold below which to prune.
                Defaults to 0.2 (matches settings.steering_prune_threshold).
            max_falsified_ratio: Falsification ratio above which to prune.
                Defaults to 0.7.

        Returns:
            True if this vector should be pruned
        """
        # Not enough data to evaluate
        if self.application_count < min_applications:
            return False

        # Low effectiveness
        if self.effectiveness_score < min_effectiveness:
            return True

        # High falsification ratio
        total_verified = self.verified_count + self.falsified_count
        if total_verified > 0:
            falsified_ratio = self.falsified_count / total_verified
            if falsified_ratio > max_falsified_ratio:
                return True

        return False

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "uid": self.uid,
            "hypothesis_uid": self.hypothesis_uid,
            "cognitive_operation": self.cognitive_operation,
            "vector_data": self.vector_data,
            "layer": self.layer,
            "effectiveness_score": self.effectiveness_score,
            "application_count": self.application_count,
            "verified_count": self.verified_count,
            "falsified_count": self.falsified_count,
            "measured_capacity": self.measured_capacity,
            "created_at": self.created_at.isoformat(),
            "last_applied": self.last_applied.isoformat()
            if self.last_applied
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HypothesisSteeringVector":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation from to_dict or persistence.

        Returns:
            Reconstructed HypothesisSteeringVector instance.
        """
        # Parse timestamps
        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except (ValueError, KeyError, TypeError):
            created_at = utc_now()

        last_applied = None
        if data.get("last_applied"):
            try:
                last_applied = datetime.fromisoformat(data["last_applied"])
            except (ValueError, TypeError):
                pass

        return cls(
            uid=data["uid"],
            hypothesis_uid=data["hypothesis_uid"],
            cognitive_operation=data["cognitive_operation"],
            vector_data=data["vector_data"],
            layer=data["layer"],
            effectiveness_score=data.get("effectiveness_score", 0.5),
            application_count=data.get("application_count", 0),
            verified_count=data.get("verified_count", 0),
            falsified_count=data.get("falsified_count", 0),
            measured_capacity=data.get("measured_capacity", 2.0),
            created_at=created_at,
            last_applied=last_applied,
        )


@dataclass
class CapacityState:
    """Tracks estimated steering capacity based on observed effects.

    Maintains a history of (magnitude, effect) pairs to estimate the
    optimal steering budget. The capacity estimate represents the
    magnitude at which effects start to saturate or diminish.

    This enables dynamic budget allocation where the system learns
    how much steering is "too much" based on observed outcomes.

    Attributes:
        current_magnitude: Most recently applied magnitude
        estimated_capacity: Estimated maximum effective magnitude
        effect_history: List of (magnitude, effect) tuples for learning
    """

    current_magnitude: float = 0.0
    estimated_capacity: float = 2.0
    effect_history: list[tuple[float, float]] = field(default_factory=list)

    # Configuration
    _MAX_HISTORY_LENGTH: int = field(default=20, repr=False)
    _CAPACITY_ALPHA: float = field(default=0.1, repr=False)
    _HEADROOM_FACTOR: float = field(default=0.8, repr=False)

    def update(self, magnitude: float, effect: float) -> None:
        """Record an observation of magnitude and its effect.

        Updates the current magnitude, adds to history, and adjusts
        the capacity estimate based on observed effects.

        Args:
            magnitude: The steering magnitude that was applied
            effect: The observed effect (0.0 to 1.0, higher = better)
        """
        self.current_magnitude = magnitude

        # Add to history
        self.effect_history.append((magnitude, effect))

        # Cap history length
        if len(self.effect_history) > self._MAX_HISTORY_LENGTH:
            self.effect_history = self.effect_history[-self._MAX_HISTORY_LENGTH :]

        # Adjust capacity estimate based on observation
        self._adjust_capacity(magnitude, effect)

    def _adjust_capacity(self, magnitude: float, effect: float) -> None:
        """Adjust capacity estimate based on observed magnitude-effect pair.

        If we observe good effects at magnitudes above current capacity,
        we increase the estimate. If effects are poor at or below capacity,
        we decrease it.

        Args:
            magnitude: The observed magnitude
            effect: The observed effect (0.0 to 1.0)
        """
        # Good effect at high magnitude -> increase capacity
        if magnitude > self.estimated_capacity * 0.8 and effect > 0.5:
            # Increase capacity toward this magnitude
            self.estimated_capacity = self.estimated_capacity + self._CAPACITY_ALPHA * (
                magnitude * 1.2 - self.estimated_capacity
            )

        # Poor effect at moderate magnitude -> decrease capacity
        elif magnitude < self.estimated_capacity and effect < 0.4:
            # Decrease capacity toward this magnitude
            self.estimated_capacity = self.estimated_capacity + self._CAPACITY_ALPHA * (
                magnitude * 0.9 - self.estimated_capacity
            )

        # Ensure capacity stays positive
        self.estimated_capacity = max(0.5, self.estimated_capacity)

    def get_optimal_budget(self) -> float:
        """Calculate the optimal steering budget based on history.

        Uses effect history to find the magnitude that maximizes effect
        without going past the point of diminishing returns.

        Returns:
            Recommended steering magnitude budget.
        """
        if not self.effect_history:
            # No history, use conservative estimate
            return self.estimated_capacity * self._HEADROOM_FACTOR

        # Find the magnitude with best effect in history
        best_magnitude = 0.0
        best_effect = 0.0

        for magnitude, effect in self.effect_history:
            if effect > best_effect:
                best_effect = effect
                best_magnitude = magnitude

        # Look for diminishing returns
        diminishing_point = self.estimated_capacity
        sorted_history = sorted(self.effect_history, key=lambda x: x[0])

        prev_effect = 0.0
        for magnitude, effect in sorted_history:
            # If effect drops significantly compared to previous, note this point
            if prev_effect > 0.5 and effect < prev_effect * 0.7:
                diminishing_point = min(diminishing_point, magnitude)
            prev_effect = effect

        # Choose the smaller of best magnitude and diminishing point
        optimal = min(best_magnitude, diminishing_point)

        # Apply headroom and cap at capacity
        budget = optimal * self._HEADROOM_FACTOR
        budget = min(budget, self.estimated_capacity)

        # Ensure positive budget
        return max(0.1, budget)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "current_magnitude": self.current_magnitude,
            "estimated_capacity": self.estimated_capacity,
            "effect_history": self.effect_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapacityState":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation from to_dict or persistence.

        Returns:
            Reconstructed CapacityState instance.
        """
        # Convert history list items to tuples if needed
        effect_history = []
        for item in data.get("effect_history", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                effect_history.append((float(item[0]), float(item[1])))

        return cls(
            current_magnitude=data.get("current_magnitude", 0.0),
            estimated_capacity=data.get("estimated_capacity", 2.0),
            effect_history=effect_history,
        )
