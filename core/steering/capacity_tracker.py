"""Dynamic capacity tracking for steering magnitude optimization.

This module provides the CapacityTracker class which empirically measures
steering capacity - the point at which increasing magnitude no longer has
an effect. Uses KL divergence between steered and unsteered outputs to
detect saturation and diminishing returns.

Key concepts:
- KL divergence: Measures how much the steered distribution differs from baseline
- Diminishing returns: When marginal effect per unit magnitude drops below threshold
- Capacity estimation: Learns the optimal magnitude range from observed effects
- Safety margin: Returns budget at 85% of estimated capacity to avoid oversteer

Integration:
- Uses CapacityState from hypothesis_vectors.py for state persistence
- Called by generation phase to determine optimal steering magnitude
- Updates based on observed effects to refine capacity estimates over time

Example:
    tracker = CapacityTracker(safety_margin=0.85)

    # Measure effect at current magnitude
    effect = tracker.measure_effect(model, prompt, vector, magnitude=1.5)

    # Update capacity estimate with observation
    is_saturated = tracker.update(magnitude=1.5, effect=effect)

    # Get optimal budget for next generation
    budget = tracker.get_optimal_budget()  # e.g., 1.7 if capacity is 2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.steering.hypothesis_vectors import CapacityState

if TYPE_CHECKING:
    import torch
    from core.model.hooked_qwen import HookedQwen


# Handle optional torch dependency
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore


def kl_divergence(p_logits: "torch.Tensor", q_logits: "torch.Tensor") -> float:
    """Compute KL divergence between two logit distributions.

    Measures how much distribution P (from p_logits) diverges from
    distribution Q (from q_logits). Used to measure steering effect:
    larger KL = more impact from steering.

    The formula is:
        KL(P || Q) = sum(P * (log(P) - log(Q)))

    Args:
        p_logits: Logits tensor for distribution P [batch, seq, vocab]
        q_logits: Logits tensor for distribution Q [batch, seq, vocab]

    Returns:
        KL divergence as a Python float. Non-negative, 0 when identical.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for KL divergence computation")

    # Convert logits to probability distributions
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps

    # Renormalize after adding epsilon
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    # Compute KL divergence: sum over vocab, mean over batch and sequence
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()

    return kl.item()


class CapacityTracker:
    """Tracks steering capacity using KL divergence measurements.

    Empirically measures the point at which increasing steering magnitude
    no longer produces proportional effects (diminishing returns). Uses
    this to estimate optimal steering budget.

    The tracker maintains a history of (magnitude, effect) observations
    and uses them to:
    1. Detect diminishing returns (marginal effect < threshold)
    2. Estimate capacity (magnitude where saturation begins)
    3. Return safe budget (capacity * safety_margin)

    Attributes:
        capacity_state: CapacityState instance for state management
        diminishing_threshold: Marginal effect threshold for saturation detection
        safety_margin: Fraction of capacity to use as budget (default 0.85)
    """

    def __init__(
        self,
        initial_state: Optional[CapacityState] = None,
        diminishing_threshold: float = 0.1,
        safety_margin: float = 0.85,
    ):
        """Initialize the capacity tracker.

        Args:
            initial_state: Optional existing CapacityState for continuity.
                If None, creates fresh state with default capacity of 2.0.
            diminishing_threshold: Minimum marginal effect (effect/magnitude)
                before considering it diminishing returns. Default 0.1.
            safety_margin: Fraction of estimated capacity to return as budget.
                Default 0.85 (85% of capacity).
        """
        self.capacity_state = initial_state or CapacityState()
        self.diminishing_threshold = diminishing_threshold
        self.safety_margin = safety_margin

        # Track last observation for marginal effect calculation
        self._last_magnitude: Optional[float] = None
        self._last_effect: Optional[float] = None

        # Initialize from existing history if present
        if self.capacity_state.effect_history:
            last = self.capacity_state.effect_history[-1]
            self._last_magnitude = last[0]
            self._last_effect = last[1]

    def measure_effect(
        self,
        model: "HookedQwen",
        prompt: str,
        steering_vector: np.ndarray,
        magnitude: float,
    ) -> float:
        """Measure the effect of steering at a given magnitude.

        Computes KL divergence between:
        1. Baseline: Model output without steering
        2. Steered: Model output with steering_vector * magnitude

        Args:
            model: HookedQwen model instance with run_with_cache method
            prompt: Text prompt to process
            steering_vector: Steering vector as numpy array (d_model,)
            magnitude: Magnitude to scale the steering vector by

        Returns:
            Effect as KL divergence (float). Higher = more steering effect.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for effect measurement")

        # Get baseline logits (no steering)
        baseline_logits, _ = model.run_with_cache(prompt)

        # Get steered logits
        # Scale vector by magnitude before passing to model
        scaled_vector = steering_vector * magnitude
        steered_logits, _ = model.run_with_cache(
            prompt,
            steering=scaled_vector,
            magnitude=magnitude,
        )

        # Compute KL divergence as effect measure
        effect = kl_divergence(baseline_logits, steered_logits)

        return effect

    def update(self, magnitude: float, effect: float) -> bool:
        """Record an observation and check for diminishing returns.

        Updates the capacity state with the new (magnitude, effect) pair
        and checks if marginal effect has dropped below threshold.

        Args:
            magnitude: Steering magnitude that was applied
            effect: Observed effect (e.g., KL divergence)

        Returns:
            True if diminishing returns detected (marginal effect < threshold),
            False otherwise. First observation always returns False.
        """
        # Check for diminishing returns before updating
        is_diminishing = False

        if self._last_magnitude is not None and self._last_effect is not None:
            # Calculate marginal effect: change in effect per unit magnitude
            magnitude_delta = magnitude - self._last_magnitude
            effect_delta = effect - self._last_effect

            # Avoid division by zero
            if abs(magnitude_delta) > 1e-10:
                marginal_effect = effect_delta / magnitude_delta

                # Diminishing returns if marginal effect is below threshold
                if marginal_effect < self.diminishing_threshold:
                    is_diminishing = True

        # Update capacity state (which handles history and capacity estimation)
        self.capacity_state.update(magnitude, effect)

        # Store for next comparison
        self._last_magnitude = magnitude
        self._last_effect = effect

        return is_diminishing

    def get_optimal_budget(self) -> float:
        """Get the optimal steering magnitude budget.

        Returns 85% (or configured safety_margin) of the estimated capacity.
        This provides a safety margin to avoid overshooting into saturation.

        If no observations exist, falls back to the CapacityState's
        get_optimal_budget method which uses default capacity.

        Returns:
            Recommended steering magnitude budget (positive float).
        """
        # Delegate to CapacityState's method, then apply our safety margin
        base_budget = self.capacity_state.get_optimal_budget()

        # CapacityState already applies its own headroom factor,
        # so we use our safety margin on the estimated capacity
        capacity = self.capacity_state.estimated_capacity
        safe_budget = capacity * self.safety_margin

        # Use the smaller of the two to be conservative
        return min(base_budget, safe_budget)
