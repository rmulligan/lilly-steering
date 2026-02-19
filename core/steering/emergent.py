"""Emergent slot for live vector evolution in EvalatisSteerer.

The emergent slot is the "hot" part of hybrid steering - a vector that
continuously evolves via EMA from observed activations. When it accumulates
sufficient cumulative surprise, it can crystallize into a permanent population
member.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EmergentSlot:
    """Live emergence slot that tracks a vector evolving from activations.

    The emergent slot competes with crystallized vectors in the selection
    arena. When its surprise_ema is high, it wins selection and continues
    evolving. When surprise sustains above thresholds, it crystallizes.

    Attributes:
        vector: Current emergent vector, updated via EMA from activations.
        surprise_ema: Exponential moving average of surprise values.
        cycles_since_crystallize: Cycles since last crystallization event.
        peak_surprise: Highest surprise observed during this emergence period.
        peak_vector: The vector state at peak surprise (crystallization candidate).
        cumulative_surprise: Total surprise accumulated since last crystallization.
    """

    vector: np.ndarray
    surprise_ema: float = 0.0
    cycles_since_crystallize: int = 0
    peak_surprise: float = 0.0
    peak_vector: Optional[np.ndarray] = None
    cumulative_surprise: float = 0.0

    # EMA alpha for surprise tracking (controls smoothing)
    surprise_ema_alpha: float = field(default=0.15, repr=False)

    def update(
        self,
        new_direction: np.ndarray,
        surprise: float,
        ema_alpha: float = 0.1,
    ) -> None:
        """Update the emergent vector with new activation direction.

        Args:
            new_direction: Direction from activations (will be normalized).
            surprise: Current cycle's surprise value.
            ema_alpha: Alpha for vector EMA update.
        """
        # Update surprise EMA
        self.surprise_ema = (
            1 - self.surprise_ema_alpha
        ) * self.surprise_ema + self.surprise_ema_alpha * surprise

        # Accumulate total surprise
        self.cumulative_surprise += surprise
        self.cycles_since_crystallize += 1

        # Track peak for crystallization
        if surprise > self.peak_surprise:
            self.peak_surprise = surprise
            self.peak_vector = self.vector.copy()

        # Normalize new direction
        norm = np.linalg.norm(new_direction)
        if norm == 0:
            return

        normalized = new_direction / norm

        # EMA blend vector
        self.vector = (1 - ema_alpha) * self.vector + ema_alpha * normalized

    def reset_for_new_emergence(self, d_model: int) -> None:
        """Reset slot after crystallization for fresh emergence.

        Args:
            d_model: Model dimension for vector sizing.
        """
        self.vector = np.zeros(d_model, dtype=np.float32)
        self.surprise_ema = 0.0
        self.cycles_since_crystallize = 0
        self.peak_surprise = 0.0
        self.peak_vector = None
        self.cumulative_surprise = 0.0


def create_emergent_slot(d_model: int) -> EmergentSlot:
    """Factory to create a fresh emergent slot.

    Args:
        d_model: Model hidden dimension for vector sizing.

    Returns:
        New EmergentSlot with zero-initialized vector.
    """
    return EmergentSlot(
        vector=np.zeros(d_model, dtype=np.float32),
        peak_vector=None,
    )
