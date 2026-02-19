"""Curvature tracking for natural gradient-inspired weight adaptation.

Inspired by Thermodynamic Natural Gradient Descent (TNGD) paper:
- Tracks diagonal Fisher information approximation
- Enables preconditioned weight updates (natural gradient scaling)
- Reduces oscillation under noisy verification signals

Reference: "Thermodynamic natural gradient descent"
Nature npj Unconventional Computing, 2026
https://www.nature.com/articles/s44335-025-00049-x
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CurvatureTracker:
    """Tracks diagonal Fisher information for QD metric weight adaptation.
    
    The diagonal Fisher matrix F_ii ≈ E[∂log p/∂θ_i]² provides curvature
    information about the loss landscape. For natural gradient descent,
    we scale updates by F^{-1}, which reduces oscillation in steep directions
    and accelerates in flat directions.
    
    For QD weight adaptation, we approximate this using:
    - Gradient signals from weight deltas
    - EMA smoothing for stability
    - Per-metric diagonal scaling
    
    Attributes:
        diag_fisher: Dictionary mapping metric names to Fisher diagonal estimates
        ema_alpha: Exponential moving average rate for Fisher updates
        epsilon: Numerical stability constant for division
    """
    
    def __init__(self, ema_alpha: float = 0.1, epsilon: float = 1e-8):
        """Initialize curvature tracker.
        
        Args:
            ema_alpha: EMA rate for Fisher updates (0.1 = 10% new, 90% old)
            epsilon: Small constant for numerical stability in division
        """
        self.diag_fisher: Dict[str, float] = {}
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon
        logger.info(f"CurvatureTracker initialized (α={ema_alpha}, ε={epsilon})")
    
    def update(self, metric_name: str, gradient: float) -> None:
        """Update diagonal Fisher estimate for a metric.
        
        Fisher diagonal approximation: F_ii ≈ E[(∂L/∂θ_i)²]
        We use EMA to track this over cycles:
        
        F_ii^new = α * g_i² + (1-α) * F_ii^old
        
        Args:
            metric_name: Name of the metric (e.g., "coherence", "novelty")
            gradient: Weight gradient (delta) for this metric this cycle
        """
        squared_grad = gradient ** 2
        
        if metric_name not in self.diag_fisher:
            # Initialize with squared gradient
            self.diag_fisher[metric_name] = squared_grad
        else:
            # EMA update
            self.diag_fisher[metric_name] = (
                self.ema_alpha * squared_grad +
                (1 - self.ema_alpha) * self.diag_fisher[metric_name]
            )
    
    def precondition(
        self, 
        metric_name: str, 
        gradient: float,
        strength: float = 1.0
    ) -> float:
        """Apply natural gradient scaling to a weight gradient.
        
        Natural gradient: g̃ = F^{-1} * g
        
        We interpolate between first-order (g) and second-order (g̃):
        g_adapted = (1-t) * g + t * g̃
                  = (1-t) * g + t * g / (F + ε)
        
        Where t ∈ [0,1] is the strength parameter (TNGD's runtime analog).
        
        Args:
            metric_name: Name of the metric
            gradient: Raw gradient (weight delta)
            strength: Interpolation strength (0=first-order, 1=second-order)
            
        Returns:
            Preconditioned gradient
        """
        if metric_name not in self.diag_fisher:
            # No curvature info yet - return raw gradient
            return gradient
        
        fisher_diag = self.diag_fisher[metric_name]
        
        # Natural gradient: scale by inverse Fisher
        natural_grad = gradient / (fisher_diag + self.epsilon)
        
        # Interpolate: (1-t)*g + t*g̃
        preconditioned = (1 - strength) * gradient + strength * natural_grad
        
        return preconditioned
    
    def get_stats(self) -> Dict[str, float]:
        """Get current Fisher diagonal estimates.
        
        Returns:
            Dictionary mapping metric names to Fisher diagonal values
        """
        return dict(self.diag_fisher)
    
    def reset(self) -> None:
        """Reset all curvature estimates."""
        self.diag_fisher.clear()
        logger.info("CurvatureTracker reset")
