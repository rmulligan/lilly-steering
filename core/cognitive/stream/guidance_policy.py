"""Mode-Aware Guidance Policy for Intentional Steering.

Evaluates latent observations and determines when steering adjustments
are needed to keep reasoning aligned with intended cognitive modes.

Architecture:
    ModeAwareGuidancePolicy evaluates each LatentObservation and decides
    whether to:
    - Redirect toward intended cognitive mode (if drifting)
    - Rebalance action/reflection (if too skewed)
    - Apply standard policies (uncertainty, repetition)

Key Insight from PLaT:
    By monitoring latent trajectory and applying gentle steering adjustments,
    we can guide reasoning toward intended modes without forcing specific
    outputs. The model retains autonomy while staying on course.

Usage:
    from core.cognitive.stream.guidance_policy import ModeAwareGuidancePolicy

    policy = ModeAwareGuidancePolicy(
        intended_mode="philosophical_inquiry",
        mode_flexibility=0.3,
    )

    for observation in observations:
        adjustment = await policy.evaluate(chunk, observation, history, steering)
        if adjustment:
            steering = apply_adjustment(steering, adjustment)

Reference:
    PLaT (Planning with Latent Thoughts, arXiv:2601.21358) - conscious
    trajectory guidance through real-time observation and steering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch
    from core.cognitive.stream.chunked_generator import ChunkResult
    from core.cognitive.stream.latent_observer import LatentObservation
    from core.steering.hierarchical import HierarchicalSteerer

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

# Cognitive mode groupings for balance detection
ACTION_MODES = {
    "technical_reasoning",
    "hypothesis_testing",
    "dialectical_synthesis",
}

REFLECTION_MODES = {
    "emotional_reflection",
    "metacognitive_monitoring",
    "identity_formation",
}


@dataclass
class SteeringAdjustment:
    """Recommended adjustment to apply before next chunk.

    Attributes:
        direction: Unit direction vector for adjustment (numpy array)
        zone: Which steering zone to apply ("exploration", "concept", "identity")
        magnitude: How strong the adjustment should be (capped at 2.0)
        reason: Human-readable explanation for logging/narration
    """
    direction: np.ndarray
    zone: str
    magnitude: float
    reason: str


class ModeAwareGuidancePolicy:
    """Guides reasoning toward intended cognitive modes.

    Evaluates latent observations and recommends steering adjustments
    when reasoning drifts from the intended mode or becomes unbalanced.

    Constants:
        MODE_FLEXIBILITY: Allowed drift before redirecting (0.3)
        ACTION_REFLECTION_THRESHOLD: Max imbalance before rebalancing (0.8)
        MAX_ADJUSTMENT_MAGNITUDE: Cap on adjustment strength (2.0)
        MIN_ADJUSTMENT_MAGNITUDE: Floor on adjustment strength (0.5)
    """

    MODE_FLEXIBILITY = 0.3
    ACTION_REFLECTION_THRESHOLD = 0.8
    MAX_ADJUSTMENT_MAGNITUDE = 2.0
    MIN_ADJUSTMENT_MAGNITUDE = 0.5

    def __init__(
        self,
        intended_mode: Optional[str] = None,
        mode_flexibility: float = 0.3,
        d_model: int = 4096,
    ):
        """Initialize the guidance policy.

        Args:
            intended_mode: Target cognitive mode (e.g., "philosophical_inquiry")
            mode_flexibility: Allowed drift before redirecting
            d_model: Model hidden dimension for placeholder vectors (default: 4096)
        """
        self._intended_mode = intended_mode
        self._flexibility = mode_flexibility
        self._d_model = d_model
        self._adjustment_count = 0
        self._mode_vectors: dict[str, np.ndarray] = {}

    @property
    def intended_mode(self) -> Optional[str]:
        """Get the current intended mode."""
        return self._intended_mode

    @intended_mode.setter
    def intended_mode(self, mode: Optional[str]) -> None:
        """Set the intended mode."""
        self._intended_mode = mode

    def set_mode_vectors(self, vectors: dict[str, np.ndarray]) -> None:
        """Set pre-extracted mode steering vectors.

        Args:
            vectors: Dict mapping mode name to steering direction vector
        """
        self._mode_vectors = vectors

    async def evaluate(
        self,
        chunk: "ChunkResult",
        observation: "LatentObservation",
        history: list["ChunkResult"],
        current_steering: Optional["HierarchicalSteerer"],
    ) -> Optional[SteeringAdjustment]:
        """Evaluate chunk and decide if steering adjustment needed.

        Checks in order:
        1. Mode alignment with intended mode
        2. Action/reflection balance
        3. Standard policies (uncertainty, repetition)

        Args:
            chunk: The chunk that was just generated
            observation: Latent observation for this chunk
            history: Previous chunks in this generation
            current_steering: Current steering state

        Returns:
            SteeringAdjustment if needed, None otherwise
        """
        # 1. Check mode alignment
        if self._intended_mode:
            adjustment = self._check_mode_alignment(observation)
            if adjustment:
                self._adjustment_count += 1
                return adjustment

        # 2. Check action/reflection balance
        adjustment = self._check_action_reflection_balance(observation)
        if adjustment:
            self._adjustment_count += 1
            return adjustment

        # 3. Standard policies
        adjustment = self._check_standard_policies(observation, history)
        if adjustment:
            self._adjustment_count += 1
            return adjustment

        return None

    def _check_mode_alignment(
        self,
        observation: "LatentObservation",
    ) -> Optional[SteeringAdjustment]:
        """Check if reasoning aligns with intended mode.

        Args:
            observation: Current latent observation

        Returns:
            Adjustment to redirect toward intended mode, or None
        """
        if not observation.cognitive_modes:
            return None

        intended_score = observation.cognitive_modes.get(self._intended_mode, 0.0)
        dominant_mode = max(
            observation.cognitive_modes,
            key=observation.cognitive_modes.get
        )
        dominant_score = observation.cognitive_modes[dominant_mode]

        drift = dominant_score - intended_score

        # Only redirect if drift exceeds flexibility and dominant isn't intended
        if drift > self._flexibility and dominant_mode != self._intended_mode:
            # Calculate adjustment magnitude based on drift
            magnitude = min(drift * 2, self.MAX_ADJUSTMENT_MAGNITUDE)
            magnitude = max(magnitude, self.MIN_ADJUSTMENT_MAGNITUDE)

            # Get direction vector if available
            direction = self._get_mode_direction(self._intended_mode)

            return SteeringAdjustment(
                direction=direction,
                zone="exploration",
                magnitude=magnitude,
                reason=f"drifting toward {dominant_mode}, redirecting to {self._intended_mode}",
            )

        return None

    def _check_action_reflection_balance(
        self,
        observation: "LatentObservation",
    ) -> Optional[SteeringAdjustment]:
        """Ensure healthy action/reflection balance.

        Args:
            observation: Current latent observation

        Returns:
            Adjustment to rebalance, or None
        """
        if not observation.cognitive_modes:
            return None

        action_weight = sum(
            observation.cognitive_modes.get(m, 0)
            for m in ACTION_MODES
        )
        reflection_weight = sum(
            observation.cognitive_modes.get(m, 0)
            for m in REFLECTION_MODES
        )

        total = action_weight + reflection_weight
        if total == 0:
            return None

        action_ratio = action_weight / total

        # Too action-oriented
        if action_ratio > self.ACTION_REFLECTION_THRESHOLD:
            direction = self._get_mode_direction("emotional_reflection")
            return SteeringAdjustment(
                direction=direction,
                zone="exploration",
                magnitude=1.2,
                reason="extended action-mode, inviting reflection",
            )

        # Too reflection-oriented
        if action_ratio < (1 - self.ACTION_REFLECTION_THRESHOLD):
            direction = self._get_mode_direction("technical_reasoning")
            return SteeringAdjustment(
                direction=direction,
                zone="exploration",
                magnitude=1.0,
                reason="extended reflection, grounding in action",
            )

        return None

    def _check_standard_policies(
        self,
        observation: "LatentObservation",
        history: list["ChunkResult"],
    ) -> Optional[SteeringAdjustment]:
        """Check for standard steering issues (placeholder for future implementation).

        Planned checks (not yet implemented):
        - Repetition detection: same dominant mode for too long
        - Uncertainty spikes: high entropy in mode distribution

        Note:
            Repetition detection requires observation history (not just chunk history)
            to track dominant modes across chunks. This will be implemented when
            LatentObserver provides accumulated observation history.

        Args:
            observation: Current latent observation
            history: Previous chunks

        Returns:
            Always returns None until implementation is complete
        """
        # TODO: Implement repetition detection once observation history is available
        # This requires tracking dominant modes across observations, not just chunks.
        # See: https://github.com/lilly/issues/XXX (placeholder)
        return None

    def _get_mode_direction(self, mode: str) -> np.ndarray:
        """Get the steering direction for a mode.

        Args:
            mode: Cognitive mode name

        Returns:
            Direction vector (unit vector or placeholder)
        """
        if mode in self._mode_vectors:
            vec = self._mode_vectors[mode]
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
            return vec

        # Return placeholder unit vector if no pre-computed vector
        # This allows the system to work without pre-extracted vectors
        return np.zeros(self._d_model)

    def get_adjustment_stats(self) -> dict:
        """Get statistics about adjustments made.

        Returns:
            Dict with adjustment counts and patterns
        """
        return {
            "total_adjustments": self._adjustment_count,
            "intended_mode": self._intended_mode,
            "flexibility": self._flexibility,
        }
