"""Quality Diversity (QD) configuration for EvalatisSteerer.

Defines weights and thresholds for the five QD metrics:
- COHERENCE (15%): Quality floor, measures embedding alignment
- NOVELTY (30%): Diversity driver, inverse similarity to recent selections
- SURPRISE (20%): Emergence signal from activation surprise
- PRESENCE (15%): Authenticity from human approval patterns
- LATENT_COHERENCE (20%): Proactive diversity via embedding dynamics (ATP-Latent inspired)

Supports adaptive weight learning based on cycle outcomes (H_sem, D, verification_rate).
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

if TYPE_CHECKING:
    from core.steering.qd.curvature import CurvatureTracker

logger = logging.getLogger(__name__)


@dataclass
class QDConfig:
    """Configuration for Quality Diversity scoring with adaptive weights.

    Metric weights must sum to 1.0. The coherence threshold acts as a
    quality floor - crystals scoring below it are rejected regardless
    of other metric scores.

    Adaptive Learning:
        When adaptive=True and frozen_weights=False, weights adjust based on
        cycle outcomes. The system learns which weight configurations produce
        better H_sem (semantic entropy), D (discovery parameter), and
        verification_rate outcomes.

    Attributes:
        coherence_weight: Weight for coherence metric (embedding alignment)
        novelty_weight: Weight for novelty metric (diversity)
        surprise_weight: Weight for surprise metric (emergence signal)
        presence_weight: Weight for presence metric (approval patterns)
        latent_coherence_weight: Weight for latent coherence (proactive diversity)
        coherence_threshold: Minimum coherence score (quality floor)
        novelty_window: Number of recent selections for novelty comparison
        novelty_similarity_decay: Recency weighting for novelty (0.9 = 10% per step)
        surprise_normalize_max: Max surprise value for normalization
        presence_ema_alpha: EMA alpha for presence tracking
        presence_min_observations: Minimum observations before trusting presence
        latent_window_size: Window for latent coherence embedding history
        latent_collapse_threshold: Similarity above this = mode collapse
        latent_coherence_floor: Similarity below this = incoherent
        latent_optimal_similarity: Target similarity for ideal balance
        adaptive: Whether to enable adaptive weight learning
        frozen_weights: Safety toggle to disable adaptation without changing adaptive flag
        weight_learning_rate: How fast weights adapt (0.01 = conservative)
        weight_min: Minimum allowed weight for any metric
        weight_max: Maximum allowed weight for any metric
        target_H_sem: Desired semantic entropy (adaptation target)
        target_D: Desired discovery parameter (adaptation target)
        target_verification: Desired verification rate (adaptation target)
        niche_count: Number of MAP-Elites niches
        niche_names: Names for each niche (behavioral characterization)
    """

    # Health-based throttle map for adaptation strength
    _HEALTH_THROTTLE_MAP: ClassVar[dict[str, float]] = {
        "CRITICAL": 0.1,   # Mostly first-order, minimal risk
        "STRESSED": 0.3,   # Cautious adaptation
        "STABLE": 0.6,     # Moderate preconditioning
        "THRIVING": 1.0,   # Full adaptation strength
    }

    # Metric weights (must sum to 1.0)
    # Reduced from original to make room for latent_coherence
    coherence_weight: float = 0.15
    novelty_weight: float = 0.30
    surprise_weight: float = 0.20
    presence_weight: float = 0.15
    latent_coherence_weight: float = 0.20  # NEW: ATP-Latent inspired

    # Coherence settings
    coherence_threshold: float = 0.5  # Quality floor

    # Novelty settings
    novelty_window: int = 20  # Recent vectors to compare
    novelty_similarity_decay: float = 0.9  # Recency weighting

    # Surprise settings (uses existing EMA)
    surprise_normalize_max: float = 100.0

    # Presence settings
    presence_ema_alpha: float = 0.1
    presence_min_observations: int = 5

    # Latent coherence settings (ATP-Latent inspired)
    latent_window_size: int = 10
    latent_collapse_threshold: float = 0.9
    latent_coherence_floor: float = 0.3
    latent_optimal_similarity: float = 0.6

    # Adaptive weight learning
    adaptive: bool = True  # Enable adaptive learning
    frozen_weights: bool = False  # Allow weight adaptation by default
    weight_learning_rate: float = 0.01  # Conservative learning rate
    weight_min: float = 0.05  # Minimum weight for any metric
    weight_max: float = 0.50  # Maximum weight for any metric
    target_H_sem: float = 0.5  # Desired semantic entropy
    target_D: float = 0.0  # Desired discovery parameter (0 = balanced)
    target_verification: float = 0.3  # Desired verification rate

    # Natural gradient-inspired adaptation throttle (TNGD paper)
    # Interpolates between first-order (0.0) and preconditioned (1.0) updates
    adaptation_strength: float = 0.5  # Conservative default
    curvature_ema_alpha: float = 0.1  # EMA for diagonal Fisher tracking
    enable_preconditioning: bool = True  # Enable curvature-based scaling

    # MAP-Elites niches (for future behavioral diversity)
    niche_count: int = 5
    niche_names: tuple[str, ...] = field(
        default_factory=lambda: (
            "exploratory",
            "conceptual",
            "emotional",
            "analytical",
            "integrative",
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self._validate_weights()

        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError(
                f"Coherence threshold must be in [0, 1], got {self.coherence_threshold}"
            )

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total = self._weight_sum()
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Metric weights must sum to 1.0, got {total:.3f}")

    def _weight_sum(self) -> float:
        """Calculate sum of all weights."""
        return (
            self.coherence_weight
            + self.novelty_weight
            + self.surprise_weight
            + self.presence_weight
            + self.latent_coherence_weight
        )

    def get_weights(self) -> dict[str, float]:
        """Get current weights as a dictionary.

        Returns:
            Dictionary mapping metric names to weights.
        """
        return {
            "coherence": self.coherence_weight,
            "novelty": self.novelty_weight,
            "surprise": self.surprise_weight,
            "presence": self.presence_weight,
            "latent_coherence": self.latent_coherence_weight,
        }

    def adapt_weights(
        self, 
        outcomes: dict[str, float],
        curvature_tracker: "CurvatureTracker | None" = None,
        health_status: str | None = None,
    ) -> dict[str, float]:
        """Adjust weights based on cycle outcomes with optional preconditioning.

        Uses gradient-free optimization to shift weights toward configurations
        that improve target outcomes:
        - Low H_sem → increase novelty + latent_coherence (diversity pressure)
        - Negative D → increase latent_coherence (exploration pressure)
        - Low verification → increase coherence (reasoning quality)
        
        Natural gradient-inspired preconditioning (optional):
        - If curvature_tracker provided: scales deltas by inverse Fisher diagonal
        - If health_status provided: throttles adaptation strength based on health
        - Interpolates between first-order and second-order updates
        
        Args:
            outcomes: Dictionary with keys 'H_sem', 'D', 'verification_rate'.
                Missing keys use neutral values.
            curvature_tracker: Optional tracker for diagonal Fisher information
            health_status: Optional health status (CRITICAL/STRESSED/STABLE/THRIVING)

        Returns:
            Dictionary of weight deltas applied (for logging).
        """
        if not self.adaptive or self.frozen_weights:
            return {}

        H_sem = outcomes.get("H_sem", self.target_H_sem)
        D = outcomes.get("D", self.target_D)
        verification = outcomes.get("verification_rate", self.target_verification)

        lr = self.weight_learning_rate
        deltas: dict[str, float] = {
            "coherence": 0.0,
            "novelty": 0.0,
            "surprise": 0.0,
            "presence": 0.0,
            "latent_coherence": 0.0,
        }

        # === Semantic diversity pressure ===
        # Low H_sem → need more diversity
        if H_sem < self.target_H_sem:
            deficit = self.target_H_sem - H_sem
            delta = deficit * lr

            # Increase diversity-driving metrics
            deltas["novelty"] += delta * 0.4
            deltas["latent_coherence"] += delta * 0.4

            # Decrease quality gate slightly to allow more exploration
            deltas["coherence"] -= delta * 0.4
            deltas["presence"] -= delta * 0.4

        # === Discovery parameter pressure ===
        # Negative D → too much exploitation, need exploration
        if D < self.target_D:
            deficit = self.target_D - D
            delta = deficit * lr * 0.5  # Smaller influence

            # Increase exploration-driving metrics
            deltas["latent_coherence"] += delta * 0.5
            deltas["novelty"] += delta * 0.3

            # Decrease stability-seeking metrics
            deltas["surprise"] -= delta * 0.4
            deltas["presence"] -= delta * 0.4

        # === Verification rate pressure ===
        # Low verification → predictions aren't being validated
        # Need better reasoning quality
        if verification < self.target_verification:
            deficit = self.target_verification - verification
            delta = deficit * lr * 0.3  # Smaller influence

            # Increase quality metrics
            deltas["coherence"] += delta * 0.5
            deltas["surprise"] += delta * 0.3

            # Slightly decrease pure diversity
            deltas["novelty"] -= delta * 0.4
            deltas["latent_coherence"] -= delta * 0.4

        # Compute adaptation strength once (used in preconditioning and logging)
        strength: float | None = None
        if health_status is not None:
            strength = self.get_health_throttle(health_status)

        # === Natural gradient preconditioning (optional) ===
        if self.enable_preconditioning and curvature_tracker is not None:
            # Use precomputed strength or fallback to default
            effective_strength = strength if strength is not None else self.adaptation_strength
            
            # Update curvature estimates and precondition deltas
            preconditioned_deltas: dict[str, float] = {}
            for metric, delta in deltas.items():
                # Update Fisher diagonal with this gradient
                curvature_tracker.update(metric, delta)
                # Precondition: interpolate between g and g̃ = F^{-1}g
                preconditioned_deltas[metric] = curvature_tracker.precondition(
                    metric, delta, effective_strength
                )
            deltas = preconditioned_deltas

        # Apply deltas
        self.coherence_weight += deltas["coherence"]
        self.novelty_weight += deltas["novelty"]
        self.surprise_weight += deltas["surprise"]
        self.presence_weight += deltas["presence"]
        self.latent_coherence_weight += deltas["latent_coherence"]

        # Clamp to valid range
        self._clamp_weights()

        # Renormalize to sum to 1.0
        self._renormalize_weights()

        # Log significant adaptations
        total_delta = sum(abs(d) for d in deltas.values())
        if total_delta > 0.001:
            log_msg = (
                f"QD weights adapted: {self.get_weights()}, "
                f"outcomes={{H_sem={H_sem:.3f}, D={D:.3f}, verif={verification:.3f}}}"
            )
            if curvature_tracker is not None and strength is not None:
                log_msg += f", throttle={strength:.2f}, health={health_status}"
            logger.info(log_msg)

        return deltas

    def _clamp_weights(self) -> None:
        """Clamp all weights to [weight_min, weight_max]."""
        self.coherence_weight = float(np.clip(
            self.coherence_weight, self.weight_min, self.weight_max
        ))
        self.novelty_weight = float(np.clip(
            self.novelty_weight, self.weight_min, self.weight_max
        ))
        self.surprise_weight = float(np.clip(
            self.surprise_weight, self.weight_min, self.weight_max
        ))
        self.presence_weight = float(np.clip(
            self.presence_weight, self.weight_min, self.weight_max
        ))
        self.latent_coherence_weight = float(np.clip(
            self.latent_coherence_weight, self.weight_min, self.weight_max
        ))

    def _renormalize_weights(self) -> None:
        """Renormalize weights to sum to 1.0 while respecting min/max bounds.

        Uses iterative error distribution:
        1. Clamp all weights to [weight_min, weight_max]
        2. Calculate error (1.0 - sum)
        3. Distribute error among weights not at bounds
        4. Repeat until sum is close to 1.0
        """
        weight_names = [
            "coherence_weight",
            "novelty_weight",
            "surprise_weight",
            "presence_weight",
            "latent_coherence_weight",
        ]

        max_iterations = 10
        tolerance = 1e-6

        for _ in range(max_iterations):
            # Step 1: Clamp all weights to bounds
            self._clamp_weights()

            # Step 2: Calculate error
            current_sum = self._weight_sum()
            if current_sum < 1e-8:
                # Reset to defaults if something went wrong
                self.coherence_weight = 0.15
                self.novelty_weight = 0.30
                self.surprise_weight = 0.20
                self.presence_weight = 0.15
                self.latent_coherence_weight = 0.20
                return

            error = 1.0 - current_sum
            if abs(error) < tolerance:
                return  # Close enough to 1.0

            # Step 3: Find weights that can absorb error (not at bounds)
            adjustable = []
            for name in weight_names:
                weight = getattr(self, name)
                if error > 0 and weight < self.weight_max:
                    # Need to increase, weight has room to grow
                    adjustable.append(name)
                elif error < 0 and weight > self.weight_min:
                    # Need to decrease, weight has room to shrink
                    adjustable.append(name)

            if not adjustable:
                # No weights can absorb error - this shouldn't happen
                # with valid bounds, but handle gracefully
                break

            # Step 4: Distribute error equally among adjustable weights
            delta_per_weight = error / len(adjustable)
            for name in adjustable:
                current = getattr(self, name)
                setattr(self, name, current + delta_per_weight)

    def freeze(self) -> None:
        """Freeze weights to prevent adaptation."""
        self.frozen_weights = True
        logger.info("QD weights frozen")

    def unfreeze(self) -> None:
        """Unfreeze weights to allow adaptation."""
        self.frozen_weights = False
        logger.info("QD weights unfrozen")

    def get_health_throttle(self, health_status: str) -> float:
        """Map health status to adaptation strength throttle.

        Inspired by TNGD's continuous-time interpolation parameter:
        - t=0: pure first-order (SGD-like)
        - t=1: full second-order (NGD-like)

        For Lilly, we use health as a safety gate:
        - CRITICAL: minimal adaptation (let stability recover)
        - STRESSED: reduced adaptation
        - STABLE: moderate adaptation
        - THRIVING: full adaptation strength

        Args:
            health_status: One of CRITICAL, STRESSED, STABLE, THRIVING

        Returns:
            Throttle multiplier in [0, 1]
        """
        return self._HEALTH_THROTTLE_MAP.get(health_status, 0.5) * self.adaptation_strength

    def reset_weights(self) -> None:
        """Reset weights to defaults."""
        self.coherence_weight = 0.15
        self.novelty_weight = 0.30
        self.surprise_weight = 0.20
        self.presence_weight = 0.15
        self.latent_coherence_weight = 0.20
        logger.info("QD weights reset to defaults")
