"""Minimal affect steering library - vectors from lived emotional transitions."""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel


class AffectVector(BaseModel):
    """A steering vector extracted from an emotional transition."""

    destination: List[float]  # 8D Plutchik affect state we transitioned to
    direction: List[float]  # Steering vector (stored as list, converted to numpy on use)

    class Config:
        arbitrary_types_allowed = True


class AffectLibrary:
    """In-memory library of affect steering vectors."""

    def __init__(self, min_samples: int = 3):
        self.vectors: List[AffectVector] = []
        self.min_samples = min_samples

    def add(self, destination: List[float], direction: np.ndarray) -> None:
        """Add a vector from an emotional transition."""
        self.vectors.append(
            AffectVector(
                destination=destination,
                direction=direction.tolist(),
            )
        )

    def is_ready(self) -> bool:
        """Check if we have enough samples to steer."""
        return len(self.vectors) >= self.min_samples

    def get_nearest(
        self, current: List[float], valence: float
    ) -> Optional[np.ndarray]:
        """Get steering vector for current affect state, modulated by valence.

        Args:
            current: 8D Plutchik affect state
            valence: Current valence (0 to 1). Values < 0.5 invert direction.

        Returns:
            Steering vector or None if not ready.
        """
        if not self.is_ready():
            return None

        # Find nearest by destination affect state
        current_arr = np.array(current)
        distances = [
            np.linalg.norm(current_arr - np.array(v.destination))
            for v in self.vectors
        ]
        nearest = self.vectors[int(np.argmin(distances))]
        direction_vec = np.array(nearest.direction)

        # Modulate by valence: valence < 0.5 inverts, > 0.5 reinforces.
        # Map valence from [0, 1] to [-1, 1] where 0.5 is neutral.
        steering_valence = (valence - 0.5) * 2.0

        return steering_valence * direction_vec
