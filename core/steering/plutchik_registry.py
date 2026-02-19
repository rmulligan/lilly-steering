"""Plutchik emotion steering vector registry.

Manages 8D Plutchik steering vectors at runtime, providing
composite vector blending based on current emotional state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

# Handle optional torch dependency
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from core.self_model.affective_system import PLUTCHIK_PRIMARIES

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.self_model.affective_system import AffectiveState

logger = logging.getLogger(__name__)

# Minimum intensity threshold for a vector to contribute to steering
ACTIVATION_THRESHOLD = 0.1

# Maximum magnitude for the composite vector (exploration zone cap)
MAX_COMPOSITE_MAGNITUDE = 3.0

# Conflict resolution: reduce magnitude for emotions in conflicting pairs
# This implements "ambivalence blending" - both emotions still contribute
# but at reduced intensity to reflect the psychological tension
CONFLICT_MAGNITUDE_FACTOR = 0.5


class PlutchikRegistry:
    """Manages 8D Plutchik steering vectors for emotional steering.

    Loads pre-extracted steering vectors from Psyche and provides
    intensity-weighted blending based on current emotional state.

    Attributes:
        vectors: Dict mapping emotion name to steering vector (numpy array)
        _loaded: Whether vectors have been loaded from Psyche
    """

    def __init__(self):
        """Initialize empty registry."""
        self.vectors: dict[str, np.ndarray] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether vectors have been loaded from Psyche."""
        return self._loaded

    @property
    def vector_count(self) -> int:
        """Number of loaded vectors."""
        return len(self.vectors)

    async def load(self, psyche: "PsycheClient") -> bool:
        """Load vectors from Psyche.

        Args:
            psyche: PsycheClient for graph operations

        Returns:
            True if at least one vector was loaded
        """
        loaded_count = 0

        for emotion in PLUTCHIK_PRIMARIES:
            name = f"plutchik_{emotion}"
            try:
                result = await psyche.get_steering_vector(name)
                if result and result.get("vector_data"):
                    vector_data = result["vector_data"]
                    # Handle both list and already-parsed formats
                    if isinstance(vector_data, list):
                        self.vectors[emotion] = np.array(vector_data, dtype=np.float32)
                        loaded_count += 1
                        logger.debug(
                            f"[PLUTCHIK] Loaded {emotion} vector "
                            f"(dim={len(vector_data)})"
                        )
            except Exception as e:
                logger.warning(f"[PLUTCHIK] Failed to load {emotion} vector: {e}")

        self._loaded = True

        if loaded_count > 0:
            logger.info(f"[PLUTCHIK] Loaded {loaded_count}/8 vectors from Psyche")
        else:
            logger.warning("[PLUTCHIK] No vectors found in Psyche")

        return loaded_count > 0

    def get_composite(
        self, affect_state: "AffectiveState"
    ) -> Optional[np.ndarray]:
        """Blend vectors based on current emotional intensities.

        Creates a weighted combination of steering vectors where each
        emotion's contribution is proportional to its intensity in the
        current affective state.

        When opposite emotions are both elevated (ambivalence), both
        vectors are applied at reduced magnitude to reflect psychological
        tension without suppressing either emotion.

        Args:
            affect_state: Current 8D emotional state

        Returns:
            Composite steering vector as numpy array, or None if no
            vectors are loaded or all intensities are below threshold.
        """
        if not self._loaded or not self.vectors:
            return None

        state_vec = affect_state.to_vector()
        if len(state_vec) != 8:
            logger.warning(
                f"[PLUTCHIK] Expected 8D state, got {len(state_vec)}D"
            )
            return None

        # Detect emotional conflicts (opposite pairs both elevated)
        conflicts = affect_state.detect_conflicts()
        conflicted_emotions: set[str] = set()
        for emotion_a, emotion_b in conflicts:
            conflicted_emotions.update((emotion_a, emotion_b))
            logger.debug(
                f"[PLUTCHIK] Conflict detected: {emotion_a} ↔ {emotion_b}, "
                f"applying {CONFLICT_MAGNITUDE_FACTOR}x magnitude"
            )

        # Start with zero vector of correct dimension
        first_vec = next(iter(self.vectors.values()))
        combined = np.zeros_like(first_vec)
        total_weight = 0.0

        for i, emotion in enumerate(PLUTCHIK_PRIMARIES):
            intensity = state_vec[i]
            if intensity > ACTIVATION_THRESHOLD and emotion in self.vectors:
                # Apply reduced magnitude for conflicting emotions
                weight = intensity
                if emotion in conflicted_emotions:
                    weight *= CONFLICT_MAGNITUDE_FACTOR
                combined += weight * self.vectors[emotion]
                total_weight += weight

        if total_weight == 0:
            return None

        # Normalize to prevent magnitude explosion
        combined /= total_weight

        # Apply magnitude cap
        norm = np.linalg.norm(combined)
        if norm > MAX_COMPOSITE_MAGNITUDE:
            combined = combined * (MAX_COMPOSITE_MAGNITUDE / norm)
            logger.debug(
                f"[PLUTCHIK] Capped composite magnitude: {norm:.3f} -> "
                f"{MAX_COMPOSITE_MAGNITUDE:.3f}"
            )

        return combined

    def get_single(self, emotion: str) -> Optional[np.ndarray]:
        """Get a single emotion's steering vector.

        Args:
            emotion: Plutchik primary emotion name

        Returns:
            Steering vector or None if not loaded
        """
        return self.vectors.get(emotion)

    def describe_state(self, affect_state: "AffectiveState") -> str:
        """Describe which emotions are contributing to steering.

        Args:
            affect_state: Current 8D emotional state

        Returns:
            Human-readable description of active emotions
        """
        state_vec = affect_state.to_vector()
        active = []

        for i, emotion in enumerate(PLUTCHIK_PRIMARIES):
            intensity = state_vec[i]
            if intensity > ACTIVATION_THRESHOLD and emotion in self.vectors:
                active.append(f"{emotion}={intensity:.2f}")

        if not active:
            return "no active emotions above threshold"

        return ", ".join(active)
