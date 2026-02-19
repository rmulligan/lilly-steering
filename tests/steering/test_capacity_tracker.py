"""Tests for dynamic capacity tracking using KL divergence measurements.

Tests the CapacityTracker which empirically measures steering capacity -
the point at which increasing magnitude no longer has an effect. Uses
KL divergence between steered and unsteered outputs to detect saturation.

Uses mocks to avoid requiring actual GPU/model loading.
Skips tests if torch is not available in the environment.
"""

import pytest
from unittest.mock import MagicMock

# Handle optional torch dependency
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not NUMPY_AVAILABLE,
    reason="torch or numpy not installed"
)


# =============================================================================
# Test Constants
# =============================================================================

VOCAB_SIZE = 1000
SEQUENCE_LENGTH = 10


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_model():
    """Create mock HookedQwen model for testing.

    The mock model returns logits with predictable steering effects:
    - Baseline: uniform-ish logits
    - Low magnitude: small KL divergence (steering has effect)
    - High magnitude: larger KL divergence (but at some point saturates)
    """
    model = MagicMock()
    model.d_model = 768

    def create_logits(steering=None, magnitude=0.0):
        """Generate logits with steering effect proportional to magnitude.

        Simulates a model where:
        - No steering: uniform-ish logits
        - Steering present: shifts distribution based on magnitude
        - Saturation: effect plateaus at high magnitudes (simulated)
        """
        base_logits = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)

        if steering is not None and magnitude > 0:
            # Apply steering effect with diminishing returns
            # Effect grows with sqrt of magnitude to simulate saturation
            effective_magnitude = min(magnitude, 3.0) ** 0.5
            shift = torch.ones(1, SEQUENCE_LENGTH, VOCAB_SIZE) * effective_magnitude * 0.5
            base_logits = base_logits + shift

        return base_logits

    def run_with_cache_impl(prompt, steering=None, magnitude=0.0):
        """Mock run_with_cache that returns logits based on steering."""
        logits = create_logits(steering, magnitude)
        cache = {}  # Empty cache for this test
        return logits, cache

    # Store the implementation for easy access in tests
    model._create_logits = create_logits
    model.run_with_cache = MagicMock(side_effect=run_with_cache_impl)

    return model


@pytest.fixture
def sample_steering_vector():
    """Create a sample steering vector for testing."""
    vector = np.random.randn(768).astype(np.float32)
    # Normalize to unit magnitude
    vector = vector / np.linalg.norm(vector)
    return vector


# =============================================================================
# CapacityTracker Initialization Tests
# =============================================================================

class TestCapacityTrackerInitialization:
    """Tests for CapacityTracker initialization."""

    def test_tracker_initialization_defaults(self):
        """CapacityTracker should initialize with default CapacityState."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        assert tracker.capacity_state is not None
        assert tracker.capacity_state.estimated_capacity == 2.0
        assert tracker.capacity_state.effect_history == []

    def test_tracker_initialization_with_custom_threshold(self):
        """CapacityTracker should accept custom diminishing returns threshold."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(diminishing_threshold=0.05)

        assert tracker.diminishing_threshold == 0.05

    def test_tracker_initialization_with_safety_margin(self):
        """CapacityTracker should accept custom safety margin."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(safety_margin=0.9)

        assert tracker.safety_margin == 0.9


# =============================================================================
# KL Divergence Calculation Tests
# =============================================================================

class TestKLDivergenceCalculation:
    """Tests for KL divergence computation."""

    def test_kl_divergence_identical_distributions(self):
        """KL divergence should be near zero for identical distributions."""
        from core.steering.capacity_tracker import kl_divergence

        logits = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        kl = kl_divergence(logits, logits)

        assert kl < 1e-5

    def test_kl_divergence_different_distributions(self):
        """KL divergence should be positive for different distributions."""
        from core.steering.capacity_tracker import kl_divergence

        logits_p = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        logits_q = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        kl = kl_divergence(logits_p, logits_q)

        assert kl > 0.0

    def test_kl_divergence_scaled_distribution(self):
        """KL divergence should increase with distribution scaling.

        Note: Adding a constant to logits doesn't change the softmax output
        (translation invariance), so we scale by different factors instead.
        """
        from core.steering.capacity_tracker import kl_divergence

        base_logits = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        # Scaling logits changes the temperature of the distribution
        small_scale = base_logits * 1.1  # Slightly sharper distribution
        large_scale = base_logits * 2.0  # Much sharper distribution

        kl_small = kl_divergence(base_logits, small_scale)
        kl_large = kl_divergence(base_logits, large_scale)

        # Both should be positive (distributions are different)
        assert kl_small > 0
        assert kl_large > 0
        # Larger scaling creates more divergence
        assert kl_large > kl_small

    def test_kl_divergence_returns_float(self):
        """KL divergence should return a Python float."""
        from core.steering.capacity_tracker import kl_divergence

        logits_p = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        logits_q = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)
        kl = kl_divergence(logits_p, logits_q)

        assert isinstance(kl, float)


# =============================================================================
# Effect Measurement Tests
# =============================================================================

class TestMeasureEffect:
    """Tests for measure_effect method."""

    def test_measure_effect_returns_float(self, mock_model, sample_steering_vector):
        """measure_effect should return a float effect value."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        effect = tracker.measure_effect(
            model=mock_model,
            prompt="Test prompt",
            steering_vector=sample_steering_vector,
            magnitude=1.0,
        )

        assert isinstance(effect, float)

    def test_measure_effect_zero_magnitude_near_zero(
        self, mock_model, sample_steering_vector
    ):
        """measure_effect with zero magnitude should return near-zero effect."""
        from core.steering.capacity_tracker import CapacityTracker

        # Generate deterministic logits that will be returned for both calls
        # This simulates a model that produces identical output with no steering
        fixed_logits = torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE)

        def run_with_cache_identical(*args, **kwargs):
            # Return the same logits for both baseline and steered
            return fixed_logits.clone(), {}

        mock_model.run_with_cache = MagicMock(side_effect=run_with_cache_identical)

        tracker = CapacityTracker()

        effect = tracker.measure_effect(
            model=mock_model,
            prompt="Test prompt",
            steering_vector=sample_steering_vector,
            magnitude=0.0,
        )

        # Effect should be very small (near zero) when outputs are identical
        assert effect < 1e-5

    def test_measure_effect_calls_model_twice(
        self, mock_model, sample_steering_vector
    ):
        """measure_effect should call model.run_with_cache twice (baseline + steered)."""
        from core.steering.capacity_tracker import CapacityTracker

        # Use a simpler mock that just tracks calls
        call_count = 0
        def counting_run_with_cache(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return torch.randn(1, SEQUENCE_LENGTH, VOCAB_SIZE), {}

        mock_model.run_with_cache = MagicMock(side_effect=counting_run_with_cache)

        tracker = CapacityTracker()

        tracker.measure_effect(
            model=mock_model,
            prompt="Test prompt",
            steering_vector=sample_steering_vector,
            magnitude=1.0,
        )

        assert call_count == 2

    def test_measure_effect_non_negative(self, mock_model, sample_steering_vector):
        """measure_effect should always return non-negative values."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        for magnitude in [0.0, 0.5, 1.0, 2.0, 3.0]:
            effect = tracker.measure_effect(
                model=mock_model,
                prompt="Test prompt",
                steering_vector=sample_steering_vector,
                magnitude=magnitude,
            )
            assert effect >= 0.0


# =============================================================================
# Update Method Tests
# =============================================================================

class TestUpdateMethod:
    """Tests for update method."""

    def test_update_records_observation(self):
        """update should record (magnitude, effect) observation."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        tracker.update(magnitude=1.0, effect=0.5)

        assert len(tracker.capacity_state.effect_history) == 1
        assert tracker.capacity_state.effect_history[0] == (1.0, 0.5)

    def test_update_multiple_observations(self):
        """update should accumulate multiple observations."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        tracker.update(magnitude=0.5, effect=0.3)
        tracker.update(magnitude=1.0, effect=0.5)
        tracker.update(magnitude=1.5, effect=0.6)

        assert len(tracker.capacity_state.effect_history) == 3

    def test_update_adjusts_capacity_estimate(self):
        """update should adjust capacity estimate based on observations."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()
        initial_capacity = tracker.capacity_state.estimated_capacity

        # Good effect at high magnitude should increase capacity
        tracker.update(magnitude=2.5, effect=0.7)

        # Capacity should have changed
        assert tracker.capacity_state.estimated_capacity != initial_capacity


# =============================================================================
# Diminishing Returns Detection Tests
# =============================================================================

class TestDiminishingReturnsDetection:
    """Tests for detecting diminishing returns."""

    def test_detects_diminishing_returns(self):
        """Should detect when marginal effect falls below threshold."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(diminishing_threshold=0.1)

        # First observation: good effect at low magnitude
        tracker.update(magnitude=1.0, effect=0.5)

        # Second observation: only slightly more effect at higher magnitude
        is_diminishing = tracker.update(magnitude=2.0, effect=0.52)

        # Marginal effect = (0.52 - 0.5) / (2.0 - 1.0) = 0.02 < 0.1
        assert is_diminishing is True

    def test_not_diminishing_with_good_marginal_effect(self):
        """Should return False when marginal effect is above threshold."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(diminishing_threshold=0.1)

        # First observation
        tracker.update(magnitude=1.0, effect=0.5)

        # Second observation with good marginal effect
        is_diminishing = tracker.update(magnitude=1.5, effect=0.8)

        # Marginal effect = (0.8 - 0.5) / (1.5 - 1.0) = 0.6 > 0.1
        assert is_diminishing is False

    def test_first_observation_not_diminishing(self):
        """First observation should not be marked as diminishing returns."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        is_diminishing = tracker.update(magnitude=1.0, effect=0.1)

        # No previous observation to compare against
        assert is_diminishing is False


# =============================================================================
# Optimal Budget Tests
# =============================================================================

class TestGetOptimalBudget:
    """Tests for get_optimal_budget method."""

    def test_get_optimal_budget_returns_float(self):
        """get_optimal_budget should return a float."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        budget = tracker.get_optimal_budget()

        assert isinstance(budget, float)

    def test_get_optimal_budget_respects_safety_margin(self):
        """get_optimal_budget should apply safety margin to capacity.

        Note: get_optimal_budget uses min of CapacityState's budget (which
        applies 0.8 headroom) and our safety margin (0.85). With empty
        history, CapacityState returns capacity * 0.8 = 1.6, which is less
        than capacity * 0.85 = 1.7, so we get 1.6.
        """
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(safety_margin=0.85)
        # Set known capacity
        tracker.capacity_state.estimated_capacity = 2.0

        budget = tracker.get_optimal_budget()

        # CapacityState's headroom (0.8) takes precedence: 2.0 * 0.8 = 1.6
        # Our safety margin (0.85) would give 1.7, but we take min
        assert abs(budget - 1.6) < 0.01

    def test_get_optimal_budget_uses_history(self):
        """get_optimal_budget should leverage effect history."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        # Add history with clear diminishing returns
        tracker.update(magnitude=0.5, effect=0.4)
        tracker.update(magnitude=1.0, effect=0.7)
        tracker.update(magnitude=1.5, effect=0.75)  # Diminishing
        tracker.update(magnitude=2.0, effect=0.76)  # Saturated

        budget = tracker.get_optimal_budget()

        # Budget should be below the saturation point
        assert budget < 2.0

    def test_get_optimal_budget_positive(self):
        """get_optimal_budget should always return positive value."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker()

        budget = tracker.get_optimal_budget()

        assert budget > 0.0


# =============================================================================
# Integration with CapacityState Tests
# =============================================================================

class TestCapacityStateIntegration:
    """Tests for integration with CapacityState from hypothesis_vectors.py."""

    def test_uses_capacity_state(self):
        """CapacityTracker should use CapacityState for state management."""
        from core.steering.capacity_tracker import CapacityTracker
        from core.steering.hypothesis_vectors import CapacityState

        tracker = CapacityTracker()

        assert isinstance(tracker.capacity_state, CapacityState)

    def test_can_inject_capacity_state(self):
        """Should accept existing CapacityState for continuity across sessions."""
        from core.steering.capacity_tracker import CapacityTracker
        from core.steering.hypothesis_vectors import CapacityState

        # Create pre-existing state with history
        existing_state = CapacityState(
            current_magnitude=1.5,
            estimated_capacity=2.5,
            effect_history=[(1.0, 0.6), (1.5, 0.7)],
        )

        tracker = CapacityTracker(initial_state=existing_state)

        assert tracker.capacity_state.estimated_capacity == 2.5
        assert len(tracker.capacity_state.effect_history) == 2

    def test_state_serialization_roundtrip(self):
        """CapacityTracker state should survive serialization."""
        from core.steering.capacity_tracker import CapacityTracker
        from core.steering.hypothesis_vectors import CapacityState

        tracker = CapacityTracker()
        tracker.update(magnitude=1.0, effect=0.5)
        tracker.update(magnitude=1.5, effect=0.6)

        # Serialize and restore
        state_dict = tracker.capacity_state.to_dict()
        restored_state = CapacityState.from_dict(state_dict)
        new_tracker = CapacityTracker(initial_state=restored_state)

        assert new_tracker.capacity_state.estimated_capacity == tracker.capacity_state.estimated_capacity
        assert len(new_tracker.capacity_state.effect_history) == 2


# =============================================================================
# Simulated Saturation Tests
# =============================================================================

class TestSimulatedSaturation:
    """Tests with simulated diminishing returns for capacity detection."""

    def test_capacity_detection_with_simulated_saturation(self):
        """Should correctly estimate capacity with simulated saturation curve."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(diminishing_threshold=0.05)

        # Simulate a saturation curve: effect = 1 - exp(-magnitude)
        # This naturally saturates around magnitude 2-3
        magnitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        for mag in magnitudes:
            effect = 1.0 - np.exp(-mag)
            tracker.update(magnitude=mag, effect=effect)

        budget = tracker.get_optimal_budget()

        # Budget should be reasonable (not too high, not too low)
        assert 0.5 < budget < 3.0

    def test_early_saturation_detection(self):
        """Should detect early saturation when effects plateau quickly."""
        from core.steering.capacity_tracker import CapacityTracker

        tracker = CapacityTracker(diminishing_threshold=0.1)

        # Quick saturation: effect plateaus at magnitude 1.0
        tracker.update(magnitude=0.5, effect=0.4)
        tracker.update(magnitude=1.0, effect=0.5)
        is_diminishing = tracker.update(magnitude=1.5, effect=0.51)

        assert is_diminishing is True
        # Budget should reflect early saturation
        budget = tracker.get_optimal_budget()
        assert budget < 1.5
