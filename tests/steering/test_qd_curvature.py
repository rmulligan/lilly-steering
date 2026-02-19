"""Tests for QD curvature tracking and natural gradient-inspired adaptation.

Tests the implementation of TNGD-inspired preconditioning for QD weight adaptation.
"""

import pytest

from core.steering.qd.config import QDConfig
from core.steering.qd.curvature import CurvatureTracker


class TestCurvatureTracker:
    """Test curvature tracker for diagonal Fisher estimation."""

    def test_initialization(self):
        """Test tracker initializes with correct parameters."""
        tracker = CurvatureTracker(ema_alpha=0.2, epsilon=1e-6)
        assert tracker.ema_alpha == 0.2
        assert tracker.epsilon == 1e-6
        assert len(tracker.diag_fisher) == 0

    def test_update_initializes_fisher(self):
        """Test first update initializes Fisher diagonal."""
        tracker = CurvatureTracker()
        tracker.update("coherence", 0.5)
        
        assert "coherence" in tracker.diag_fisher
        assert tracker.diag_fisher["coherence"] == 0.5 ** 2  # Squared gradient

    def test_update_applies_ema(self):
        """Test subsequent updates apply EMA smoothing."""
        tracker = CurvatureTracker(ema_alpha=0.5)  # 50/50 mix for easy math
        
        # First update: F = g1^2 = 4.0
        tracker.update("novelty", 2.0)
        assert tracker.diag_fisher["novelty"] == 4.0
        
        # Second update: F = 0.5 * g2^2 + 0.5 * F_old
        #                  = 0.5 * 9.0 + 0.5 * 4.0 = 6.5
        tracker.update("novelty", 3.0)
        assert tracker.diag_fisher["novelty"] == 6.5

    def test_precondition_without_curvature_returns_gradient(self):
        """Test preconditioning without Fisher info returns raw gradient."""
        tracker = CurvatureTracker()
        result = tracker.precondition("unknown", 1.5, strength=1.0)
        assert result == 1.5  # No Fisher info, return raw gradient

    def test_precondition_with_zero_strength_returns_gradient(self):
        """Test strength=0 (first-order) returns raw gradient."""
        tracker = CurvatureTracker()
        tracker.update("coherence", 0.5)
        
        result = tracker.precondition("coherence", 2.0, strength=0.0)
        assert result == 2.0  # (1-0)*g + 0*g̃ = g

    def test_precondition_with_full_strength_applies_natural_gradient(self):
        """Test strength=1 (second-order) applies full Fisher scaling."""
        tracker = CurvatureTracker(epsilon=0.0)  # No epsilon for clean math
        
        # Set Fisher diagonal to 4.0
        tracker.diag_fisher["novelty"] = 4.0
        
        # Natural gradient: g̃ = g / F = 8.0 / 4.0 = 2.0
        # Full strength: (1-1)*g + 1*g̃ = g̃ = 2.0
        result = tracker.precondition("novelty", 8.0, strength=1.0)
        assert result == 2.0

    def test_precondition_interpolates_between_first_and_second_order(self):
        """Test intermediate strength interpolates between g and g̃."""
        tracker = CurvatureTracker(epsilon=0.0)
        tracker.diag_fisher["surprise"] = 4.0
        
        # g = 8.0, g̃ = 8.0 / 4.0 = 2.0
        # strength=0.5: (1-0.5)*8 + 0.5*2 = 4 + 1 = 5.0
        result = tracker.precondition("surprise", 8.0, strength=0.5)
        assert result == 5.0

    def test_precondition_handles_division_by_zero(self):
        """Test epsilon prevents division by zero."""
        tracker = CurvatureTracker(epsilon=1e-8)
        tracker.diag_fisher["presence"] = 0.0  # Zero Fisher
        
        # Should not raise, uses epsilon
        result = tracker.precondition("presence", 1.0, strength=1.0)
        assert result == 1.0 / 1e-8  # g / (F + ε) = 1.0 / 1e-8

    def test_get_stats_returns_all_fisher_diagonals(self):
        """Test get_stats returns Fisher estimates."""
        tracker = CurvatureTracker()
        tracker.update("coherence", 1.0)
        tracker.update("novelty", 2.0)
        
        stats = tracker.get_stats()
        assert stats == {
            "coherence": 1.0,
            "novelty": 4.0,
        }

    def test_reset_clears_all_estimates(self):
        """Test reset clears all Fisher diagonals."""
        tracker = CurvatureTracker()
        tracker.update("coherence", 1.0)
        tracker.update("novelty", 2.0)
        
        tracker.reset()
        assert len(tracker.diag_fisher) == 0


class TestQDConfigWithCurvature:
    """Test QDConfig adaptation with curvature preconditioning."""

    def test_adapt_weights_without_curvature_behaves_as_before(self):
        """Test backward compatibility: no tracker = old behavior."""
        config = QDConfig(adaptive=True, frozen_weights=False)
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        
        # Without tracker - should work as before
        deltas = config.adapt_weights(outcomes)
        
        assert deltas  # Some adaptation happened
        assert "coherence" in deltas
        assert "novelty" in deltas

    def test_adapt_weights_with_curvature_applies_preconditioning(self):
        """Test that curvature tracker modifies deltas."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            enable_preconditioning=True,
            adaptation_strength=1.0,  # Full second-order
        )
        tracker = CurvatureTracker()
        
        # Prime tracker with some curvature
        tracker.update("novelty", 2.0)  # F_novelty = 4.0
        
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        deltas = config.adapt_weights(outcomes, curvature_tracker=tracker)
        
        # Deltas should be preconditioned (scaled by inverse Fisher)
        assert deltas
        # Can't assert exact values since adaptation logic is complex,
        # but we can verify the tracker was updated
        stats = tracker.get_stats()
        assert "novelty" in stats
        assert "latent_coherence" in stats  # Should be updated from adaptation

    def test_adapt_weights_with_preconditioning_disabled_ignores_tracker(self):
        """Test enable_preconditioning=False ignores tracker."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            enable_preconditioning=False,  # Disabled
        )
        tracker = CurvatureTracker()
        
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        deltas_without = config.adapt_weights(outcomes)
        
        # With tracker but disabled - should behave the same
        deltas_with = config.adapt_weights(
            outcomes, curvature_tracker=tracker
        )
        
        # Should be identical (or very close due to FP)
        for key in deltas_without:
            assert abs(deltas_without[key] - deltas_with[key]) < 1e-9

    def test_health_throttle_scales_adaptation_strength(self):
        """Test health status throttles adaptation via strength parameter."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            enable_preconditioning=True,
            adaptation_strength=1.0,
        )
        tracker = CurvatureTracker()
        
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        
        # CRITICAL health → strength = 0.1 * 1.0 = 0.1
        throttle = config.get_health_throttle("CRITICAL")
        assert throttle == 0.1
        
        # THRIVING health → strength = 1.0 * 1.0 = 1.0
        throttle = config.get_health_throttle("THRIVING")
        assert throttle == 1.0
        
        # Adapt with CRITICAL - should produce smaller deltas
        deltas_critical = config.adapt_weights(
            outcomes, curvature_tracker=tracker, health_status="CRITICAL"
        )
        
        # Reset and adapt with THRIVING - should produce larger deltas
        config.reset_weights()
        tracker.reset()
        deltas_thriving = config.adapt_weights(
            outcomes, curvature_tracker=tracker, health_status="THRIVING"
        )
        
        # Thriving deltas should be larger (in absolute value)
        # Since both start from same position, higher throttle = more adaptation
        # Compare total variation (sum of absolute deltas, not sum of deltas)
        total_var_thriving = sum(abs(d) for d in deltas_thriving.values())
        total_var_critical = sum(abs(d) for d in deltas_critical.values())
        assert total_var_thriving > total_var_critical

    def test_get_health_throttle_returns_correct_values(self):
        """Test health throttle mapping."""
        config = QDConfig(adaptation_strength=0.8)
        
        assert config.get_health_throttle("CRITICAL") == 0.1 * 0.8
        assert config.get_health_throttle("STRESSED") == 0.3 * 0.8
        assert config.get_health_throttle("STABLE") == 0.6 * 0.8
        assert config.get_health_throttle("THRIVING") == 1.0 * 0.8
        assert config.get_health_throttle("UNKNOWN") == 0.5 * 0.8  # Default

    def test_adaptation_with_frozen_weights_returns_empty(self):
        """Test frozen weights prevents adaptation even with tracker."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=True,  # Frozen
            enable_preconditioning=True,
        )
        tracker = CurvatureTracker()
        
        outcomes = {"H_sem": 0.0, "D": -1.0, "verification_rate": 0.0}
        deltas = config.adapt_weights(
            outcomes, curvature_tracker=tracker, health_status="THRIVING"
        )
        
        assert deltas == {}  # No adaptation when frozen

    def test_curvature_tracker_accumulates_across_cycles(self):
        """Test tracker accumulates Fisher info across multiple adaptations."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            enable_preconditioning=True,
            adaptation_strength=1.0,
        )
        tracker = CurvatureTracker(ema_alpha=0.1)
        
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        
        # First adaptation
        config.adapt_weights(outcomes, curvature_tracker=tracker)
        stats_1 = tracker.get_stats()
        
        # Second adaptation - Fisher should update via EMA
        config.adapt_weights(outcomes, curvature_tracker=tracker)
        stats_2 = tracker.get_stats()
        
        # Fisher values should change (EMA update)
        assert stats_2 != stats_1
        # Should have Fisher estimates for all adapted metrics
        assert len(stats_2) >= 3  # At least coherence, novelty, latent_coherence


class TestQDScorerWithCurvature:
    """Test QDScorer integration with curvature tracker."""

    def test_scorer_initializes_curvature_tracker(self):
        """Test scorer creates tracker on init."""
        from core.steering.qd.scorer import QDScorer
        
        config = QDConfig(curvature_ema_alpha=0.15)
        scorer = QDScorer(config=config)
        
        assert hasattr(scorer, "curvature_tracker")
        assert scorer.curvature_tracker is not None
        assert scorer.curvature_tracker.ema_alpha == 0.15

    def test_scorer_adapt_weights_passes_tracker(self):
        """Test scorer passes tracker to config.adapt_weights."""
        from core.steering.qd.scorer import QDScorer
        
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            enable_preconditioning=True,
        )
        scorer = QDScorer(config=config)
        
        outcomes = {"H_sem": 0.2, "D": 0.0, "verification_rate": 0.3}
        deltas = scorer.adapt_weights(outcomes, health_status="STABLE")
        
        assert deltas  # Adaptation happened
        # Verify tracker was used (has Fisher estimates)
        stats = scorer.curvature_tracker.get_stats()
        assert len(stats) > 0

    def test_scorer_adapt_weights_backward_compatible(self):
        """Test scorer.adapt_weights works without health_status."""
        from core.steering.qd.scorer import QDScorer
        
        scorer = QDScorer()
        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        
        # Old call signature should still work
        deltas = scorer.adapt_weights(outcomes)
        assert isinstance(deltas, dict)
