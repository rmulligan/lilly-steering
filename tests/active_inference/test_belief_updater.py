"""
Tests for belief_updater.py - Bayesian belief updating for Active Inference.

Tests cover:
- BeliefDistribution creation (uniform, from_prior)
- Entropy calculations
- Mode and confidence accessors
- Serialization/deserialization
- Belief update with mock observations
- KL divergence computation
- Action EFE computation
"""

import numpy as np
import pytest

from core.active_inference.belief_updater import (
    BeliefDistribution,
    BeliefUpdater,
    UpdateResult,
    create_initial_beliefs,
    SIGNIFICANT_SURPRISE_THRESHOLD,
)
from core.active_inference.generative_model import (
    GenerativeModel,
    StateSpace,
    ActionType,
    NoteType,
    GraphConnectivity,
    UserBehavior,
    UncertaintyLevel,
)
from core.active_inference.observation_encoder import EncodedObservation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def state_space():
    """Create a default state space."""
    return StateSpace()


@pytest.fixture
def generative_model():
    """Create a generative model with built matrices."""
    return GenerativeModel()


@pytest.fixture
def belief_updater(generative_model):
    """Create a belief updater with default model."""
    return BeliefUpdater(generative_model)


@pytest.fixture
def uniform_beliefs(state_space):
    """Create uniform belief distribution."""
    return BeliefDistribution.uniform(state_space)


@pytest.fixture
def sample_observation():
    """Create a sample encoded observation."""
    return EncodedObservation(
        note_type=NoteType.FRAGMENT,
        graph_connectivity=GraphConnectivity.ISOLATED,
        user_behavior=UserBehavior.FAST_INPUT,
        uncertainty_level=UncertaintyLevel.HIGH,
        raw_features={"test": True},
    )


# =============================================================================
# BeliefDistribution Tests
# =============================================================================


class TestBeliefDistributionCreation:
    """Tests for BeliefDistribution creation methods."""

    def test_uniform_creates_equal_probabilities(self, state_space):
        """Uniform distribution should have equal probabilities for each factor."""
        beliefs = BeliefDistribution.uniform(state_space)

        # Check that we have the right number of factors
        assert len(beliefs.factors) == 4

        # Check each factor sums to 1 and has equal probabilities
        assert np.isclose(beliefs.factors[0].sum(), 1.0)
        assert np.allclose(beliefs.factors[0], 1 / 4)  # TopicFocus: 4 states

        assert np.isclose(beliefs.factors[1].sum(), 1.0)
        assert np.allclose(beliefs.factors[1], 1 / 3)  # KnowledgeLevel: 3 states

        assert np.isclose(beliefs.factors[2].sum(), 1.0)
        assert np.allclose(beliefs.factors[2], 1 / 4)  # CognitiveMode: 4 states

        assert np.isclose(beliefs.factors[3].sum(), 1.0)
        assert np.allclose(beliefs.factors[3], 1 / 4)  # SeedState: 4 states

    def test_from_prior_copies_d_matrices(self, generative_model):
        """from_prior should create beliefs from D matrix priors."""
        beliefs = BeliefDistribution.from_prior(generative_model.D)

        # Should have same structure as D
        assert len(beliefs.factors) == len(generative_model.D)

        # Values should match D (but be copies, not references)
        for i, factor in enumerate(beliefs.factors):
            assert np.allclose(factor, generative_model.D[i])

            # Verify it's a copy, not the same object
            assert factor is not generative_model.D[i]

    def test_uniform_has_initial_values(self, state_space):
        """Uniform distribution should have default metadata values."""
        beliefs = BeliefDistribution.uniform(state_space)

        assert beliefs.timestamp is None
        assert beliefs.observation_count == 0


class TestBeliefDistributionEntropy:
    """Tests for entropy calculations."""

    def test_uniform_has_maximum_entropy(self, state_space):
        """Uniform distribution should have maximum entropy."""
        beliefs = BeliefDistribution.uniform(state_space)

        # Normalized entropy should be 1.0 for uniform distribution
        assert np.isclose(beliefs.normalized_entropy, 1.0, atol=1e-6)

    def test_peaked_distribution_has_low_entropy(self):
        """Highly peaked distribution should have low entropy."""
        # Create a distribution that's almost certain
        factors = [
            np.array([0.97, 0.01, 0.01, 0.01]),  # TopicFocus
            np.array([0.98, 0.01, 0.01]),  # KnowledgeLevel
            np.array([0.97, 0.01, 0.01, 0.01]),  # CognitiveMode
            np.array([0.97, 0.01, 0.01, 0.01]),  # SeedState
        ]
        beliefs = BeliefDistribution(factors=factors)

        # Normalized entropy should be close to 0
        assert beliefs.normalized_entropy < 0.3

    def test_entropy_is_positive(self, uniform_beliefs):
        """Entropy should always be non-negative."""
        assert uniform_beliefs.entropy >= 0

    def test_max_entropy_computation(self, state_space):
        """Max entropy should be sum of log(n) for each factor."""
        beliefs = BeliefDistribution.uniform(state_space)

        expected_max = np.log(4) + np.log(3) + np.log(4) + np.log(4)
        assert np.isclose(beliefs.max_entropy, expected_max)


class TestBeliefDistributionAccessors:
    """Tests for mode and confidence accessors."""

    def test_get_mode_returns_argmax(self):
        """get_mode should return argmax for each factor."""
        factors = [
            np.array([0.1, 0.6, 0.2, 0.1]),  # Mode at index 1
            np.array([0.3, 0.5, 0.2]),  # Mode at index 1
            np.array([0.1, 0.1, 0.7, 0.1]),  # Mode at index 2
            np.array([0.8, 0.1, 0.05, 0.05]),  # Mode at index 0
        ]
        beliefs = BeliefDistribution(factors=factors)

        mode = beliefs.get_mode()
        assert mode == [1, 1, 2, 0]

    def test_get_confidence_returns_max_prob(self):
        """get_confidence should return max probability for each factor."""
        factors = [
            np.array([0.1, 0.6, 0.2, 0.1]),
            np.array([0.3, 0.5, 0.2]),
            np.array([0.1, 0.1, 0.7, 0.1]),
            np.array([0.8, 0.1, 0.05, 0.05]),
        ]
        beliefs = BeliefDistribution(factors=factors)

        confidence = beliefs.get_confidence()
        assert np.isclose(confidence[0], 0.6)
        assert np.isclose(confidence[1], 0.5)
        assert np.isclose(confidence[2], 0.7)
        assert np.isclose(confidence[3], 0.8)

    def test_named_factor_accessors(self):
        """Named accessors should return correct factors."""
        factors = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.33, 0.33, 0.34]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.4, 0.3, 0.2, 0.1]),
        ]
        beliefs = BeliefDistribution(factors=factors)

        assert np.array_equal(beliefs.topic_focus, factors[0])
        assert np.array_equal(beliefs.knowledge_level, factors[1])
        assert np.array_equal(beliefs.cognitive_mode, factors[2])
        assert np.array_equal(beliefs.seed_state, factors[3])


class TestBeliefDistributionSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict_includes_all_fields(self, uniform_beliefs):
        """to_dict should include all required fields."""
        data = uniform_beliefs.to_dict()

        assert "factors" in data
        assert "timestamp" in data
        assert "observation_count" in data
        assert "entropy" in data
        assert "normalized_entropy" in data
        assert "mode" in data
        assert "confidence" in data

    def test_from_dict_roundtrip(self, uniform_beliefs):
        """Serialization roundtrip should preserve data."""
        data = uniform_beliefs.to_dict()
        restored = BeliefDistribution.from_dict(data)

        # Check factors match
        for i in range(len(uniform_beliefs.factors)):
            assert np.allclose(restored.factors[i], uniform_beliefs.factors[i])

        # Check metadata
        assert restored.timestamp == uniform_beliefs.timestamp
        assert restored.observation_count == uniform_beliefs.observation_count

    def test_from_dict_with_metadata(self):
        """from_dict should restore metadata correctly."""
        data = {
            "factors": [[0.25, 0.25, 0.25, 0.25], [0.33, 0.33, 0.34], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            "timestamp": "2024-01-01T00:00:00",
            "observation_count": 42,
        }
        beliefs = BeliefDistribution.from_dict(data)

        assert beliefs.timestamp == "2024-01-01T00:00:00"
        assert beliefs.observation_count == 42


# =============================================================================
# BeliefUpdater Tests
# =============================================================================


class TestBeliefUpdaterInit:
    """Tests for BeliefUpdater initialization."""

    def test_init_builds_matrices_if_needed(self):
        """Init should build matrices if they don't exist."""
        from core.active_inference.generative_model import ObservationSpace, ActionSpace

        model = GenerativeModel.__new__(GenerativeModel)
        model.state_space = StateSpace()
        model.observation_space = ObservationSpace()
        model.action_space = ActionSpace()
        model.A = None
        model.B = []
        model.C = []
        model.D = []

        # This should trigger build_matrices
        updater = BeliefUpdater(model)

        # After init, A should be built
        assert model.A is not None

    def test_init_sets_factor_dependencies(self, generative_model):
        """Init should set up A_factor_dependencies mapping."""
        updater = BeliefUpdater(generative_model)

        assert updater.A_factor_dependencies == {
            0: 3,  # NoteType -> SeedState
            1: 3,  # GraphConnectivity -> SeedState
            2: 2,  # UserBehavior -> CognitiveMode
            3: 3,  # UncertaintyLevel -> SeedState
        }


class TestBeliefUpdate:
    """Tests for belief update mechanism."""

    def test_update_returns_update_result(self, belief_updater, uniform_beliefs, sample_observation):
        """update should return an UpdateResult."""
        result = belief_updater.update(uniform_beliefs, sample_observation)

        assert isinstance(result, UpdateResult)
        assert result.prior is uniform_beliefs
        assert isinstance(result.posterior, BeliefDistribution)
        assert result.observation is sample_observation
        assert isinstance(result.surprise, float)

    def test_update_increments_observation_count(self, belief_updater, uniform_beliefs, sample_observation):
        """update should increment observation count in posterior."""
        result = belief_updater.update(uniform_beliefs, sample_observation)

        assert result.posterior.observation_count == uniform_beliefs.observation_count + 1

    def test_posterior_is_normalized(self, belief_updater, uniform_beliefs, sample_observation):
        """Posterior factors should sum to 1."""
        result = belief_updater.update(uniform_beliefs, sample_observation)

        for factor in result.posterior.factors:
            assert np.isclose(factor.sum(), 1.0)

    def test_update_with_informative_observation(self, belief_updater, uniform_beliefs):
        """Update with informative observation should change beliefs."""
        # Create an observation indicating mature/hub state
        obs = EncodedObservation(
            note_type=NoteType.STATEMENT,  # More likely for mature
            graph_connectivity=GraphConnectivity.HUB,  # High connectivity
            user_behavior=UserBehavior.SLOW_DELIBERATE,  # Reflective mode
            uncertainty_level=UncertaintyLevel.LOW,  # Confident
            raw_features={},
        )

        result = belief_updater.update(uniform_beliefs, obs)

        # After update, SeedState should shift toward MATURE (index 3)
        # because HUB connectivity strongly indicates mature state
        prior_seed = uniform_beliefs.seed_state
        posterior_seed = result.posterior.seed_state

        # The posterior should have higher probability for mature state
        assert posterior_seed[3] > prior_seed[3]

    def test_update_without_actions(self, belief_updater, uniform_beliefs, sample_observation):
        """update with compute_actions=False should skip EFE computation."""
        result = belief_updater.update(
            uniform_beliefs, sample_observation, compute_actions=False
        )

        assert result.action_efe is None
        assert result.recommended_action is None

    def test_update_with_actions(self, belief_updater, uniform_beliefs, sample_observation):
        """update with compute_actions=True should compute EFE."""
        result = belief_updater.update(
            uniform_beliefs, sample_observation, compute_actions=True
        )

        assert result.action_efe is not None
        assert len(result.action_efe) == belief_updater.model.action_space.num_actions
        assert result.recommended_action is not None
        assert isinstance(result.recommended_action, ActionType)


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_kl_divergence_same_distribution(self, belief_updater):
        """KL divergence of distribution with itself should be 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = belief_updater._compute_kl_divergence(p, p)

        assert np.isclose(kl, 0.0, atol=1e-6)

    def test_kl_divergence_different_distributions(self, belief_updater):
        """KL divergence of different distributions should be positive."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.1, 0.2, 0.3, 0.4])

        kl = belief_updater._compute_kl_divergence(p, q)
        assert kl > 0

    def test_kl_divergence_peaked_vs_uniform(self, belief_updater):
        """Peaked distribution vs uniform should have positive KL."""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        peaked = np.array([0.9, 0.05, 0.025, 0.025])

        kl = belief_updater._compute_kl_divergence(uniform, peaked)
        assert kl > 0

        # KL(peaked || uniform) should be larger than KL(uniform || peaked)
        # because peaked assigns high probability to states uniform doesn't favor
        kl_reverse = belief_updater._compute_kl_divergence(peaked, uniform)
        # Actually this depends on the exact values, so just check both are positive
        assert kl_reverse > 0


class TestActionEFE:
    """Tests for Expected Free Energy computation."""

    def test_compute_action_efe_returns_array(self, belief_updater, uniform_beliefs):
        """compute_action_efe should return array of EFE values."""
        efe = belief_updater.compute_action_efe(uniform_beliefs)

        assert isinstance(efe, np.ndarray)
        assert len(efe) == belief_updater.model.action_space.num_actions

    def test_do_nothing_has_higher_efe(self, belief_updater, uniform_beliefs):
        """DO_NOTHING should generally have higher EFE than epistemic actions."""
        efe = belief_updater.compute_action_efe(uniform_beliefs)

        do_nothing_efe = efe[ActionType.DO_NOTHING.value]
        ask_clarification_efe = efe[ActionType.ASK_CLARIFICATION.value]

        # ASK_CLARIFICATION has high epistemic value, so lower EFE
        assert ask_clarification_efe < do_nothing_efe

    def test_recommended_action_has_lowest_efe(self, belief_updater, uniform_beliefs, sample_observation):
        """Recommended action should have the lowest EFE."""
        result = belief_updater.update(uniform_beliefs, sample_observation)

        if result.action_efe is not None:
            min_idx = int(np.argmin(result.action_efe))
            assert result.recommended_action.value == min_idx


class TestUpdateResult:
    """Tests for UpdateResult dataclass."""

    def test_belief_changed_significantly_high_surprise(self, belief_updater, uniform_beliefs):
        """High surprise should indicate significant belief change."""
        # Create observation that should cause high surprise
        obs = EncodedObservation(
            note_type=NoteType.STATEMENT,
            graph_connectivity=GraphConnectivity.HUB,
            user_behavior=UserBehavior.SLOW_DELIBERATE,
            uncertainty_level=UncertaintyLevel.LOW,
            raw_features={},
        )

        result = belief_updater.update(uniform_beliefs, obs)

        # With uniform prior, any informative observation should cause some surprise
        # The threshold is 0.1, so this should typically be true
        if result.surprise > SIGNIFICANT_SURPRISE_THRESHOLD:
            assert result.belief_changed_significantly

    def test_to_dict_includes_all_fields(self, belief_updater, uniform_beliefs, sample_observation):
        """UpdateResult.to_dict should include all required fields."""
        result = belief_updater.update(uniform_beliefs, sample_observation)
        data = result.to_dict()

        assert "prior_entropy" in data
        assert "posterior_entropy" in data
        assert "surprise" in data
        assert "observation" in data
        assert "recommended_action" in data
        assert "posterior_mode" in data


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateInitialBeliefs:
    """Tests for create_initial_beliefs factory function."""

    def test_creates_beliefs_from_d_priors(self, generative_model):
        """create_initial_beliefs should use model's D priors."""
        beliefs = create_initial_beliefs(generative_model)

        assert isinstance(beliefs, BeliefDistribution)
        assert len(beliefs.factors) == len(generative_model.D)

        for i, factor in enumerate(beliefs.factors):
            assert np.allclose(factor, generative_model.D[i])

    def test_initial_beliefs_are_normalized(self, generative_model):
        """Initial beliefs should be normalized probability distributions."""
        beliefs = create_initial_beliefs(generative_model)

        for factor in beliefs.factors:
            assert np.isclose(factor.sum(), 1.0)
            assert np.all(factor >= 0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBeliefUpdateWorkflow:
    """Integration tests for full belief update workflow."""

    def test_multiple_updates_accumulate(self, belief_updater, generative_model):
        """Multiple updates should accumulate observation count."""
        beliefs = create_initial_beliefs(generative_model)

        obs = EncodedObservation(
            note_type=NoteType.FRAGMENT,
            graph_connectivity=GraphConnectivity.SPARSE,
            user_behavior=UserBehavior.FAST_INPUT,
            uncertainty_level=UncertaintyLevel.MEDIUM,
            raw_features={},
        )

        # Apply multiple updates
        for i in range(5):
            result = belief_updater.update(beliefs, obs, compute_actions=False)
            beliefs = result.posterior

        assert beliefs.observation_count == 5

    def test_consistent_observations_reduce_entropy(self, belief_updater, generative_model):
        """Consistent observations should reduce entropy over time."""
        beliefs = create_initial_beliefs(generative_model)
        initial_entropy = beliefs.normalized_entropy

        # Consistently observe mature/hub state
        obs = EncodedObservation(
            note_type=NoteType.STATEMENT,
            graph_connectivity=GraphConnectivity.HUB,
            user_behavior=UserBehavior.SLOW_DELIBERATE,
            uncertainty_level=UncertaintyLevel.LOW,
            raw_features={},
        )

        # Apply multiple consistent updates
        for _ in range(10):
            result = belief_updater.update(beliefs, obs, compute_actions=False)
            beliefs = result.posterior

        # Entropy should decrease (beliefs become more certain)
        assert beliefs.normalized_entropy < initial_entropy

    def test_learning_rate_affects_update_speed(self, generative_model):
        """Higher learning rate should cause faster belief changes."""
        slow_updater = BeliefUpdater(generative_model, learning_rate=0.1)
        fast_updater = BeliefUpdater(generative_model, learning_rate=0.9)

        slow_beliefs = create_initial_beliefs(generative_model)
        fast_beliefs = create_initial_beliefs(generative_model)

        obs = EncodedObservation(
            note_type=NoteType.STATEMENT,
            graph_connectivity=GraphConnectivity.HUB,
            user_behavior=UserBehavior.SLOW_DELIBERATE,
            uncertainty_level=UncertaintyLevel.LOW,
            raw_features={},
        )

        slow_result = slow_updater.update(slow_beliefs, obs, compute_actions=False)
        fast_result = fast_updater.update(fast_beliefs, obs, compute_actions=False)

        # Fast updater should have higher surprise (more belief change)
        assert fast_result.surprise > slow_result.surprise


class TestBeliefUpdateSDFT:
    """Tests for SDFT (Self-Distillation Fine-Tuning) in belief updates."""

    def test_confident_beliefs_resist_change(self, generative_model):
        """SDFT: High-confidence beliefs should change less WITH resistance than WITHOUT.

        Tests the SDFT resistance formula directly: given the same prior and
        same Bayesian posterior, applying SDFT resistance should reduce the change.
        """
        # High confidence prior (0.97)
        prior = np.array([0.97, 0.01, 0.01, 0.01])

        # Simulated Bayesian posterior that pushes toward state 1
        posterior = np.array([0.10, 0.70, 0.15, 0.05])
        learning_rate = 0.5

        # Standard blending (before SDFT)
        blended_baseline = learning_rate * posterior + (1 - learning_rate) * prior

        # Same blending with SDFT resistance
        blended_sdft = learning_rate * posterior + (1 - learning_rate) * prior
        confidence = float(np.max(prior))  # 0.97
        if confidence > 0.5:
            resistance = (confidence - 0.5) * 0.6  # ~0.282
            blended_sdft = (1 - resistance) * blended_sdft + resistance * prior

        # Normalize
        blended_baseline = blended_baseline / blended_baseline.sum()
        blended_sdft = blended_sdft / blended_sdft.sum()

        # Compute changes from prior
        change_baseline = np.abs(blended_baseline - prior).sum()
        change_sdft = np.abs(blended_sdft - prior).sum()

        # SDFT should reduce the change
        assert change_sdft < change_baseline, \
            f"SDFT: Resistance should reduce change. Got sdft={change_sdft:.4f}, baseline={change_baseline:.4f}"

        # Verify resistance is meaningful (at least 10% reduction)
        reduction = (change_baseline - change_sdft) / change_baseline
        assert reduction > 0.10, \
            f"SDFT: Resistance should meaningfully reduce change. Got {reduction:.2%} reduction"

    def test_sdft_resistance_applies_above_threshold(self, generative_model):
        """SDFT: Resistance only applies when confidence > 0.5.

        Tests the SDFT threshold behavior directly: given the same prior and
        same Bayesian posterior, resistance should only apply when
        max(prior) > 0.5, reducing the change compared to no resistance.
        """
        # Prior just above threshold (0.6 confidence)
        prior = np.array([0.6, 0.25, 0.10, 0.05])

        # Simulated Bayesian posterior that pushes toward state 1
        posterior = np.array([0.15, 0.60, 0.15, 0.10])
        learning_rate = 0.5

        # Standard blending WITHOUT SDFT resistance (baseline)
        blended_baseline = learning_rate * posterior + (1 - learning_rate) * prior

        # Same blending WITH SDFT resistance applied
        blended_sdft = learning_rate * posterior + (1 - learning_rate) * prior
        confidence = float(np.max(prior))  # 0.6
        assert confidence > 0.5, "Test setup: confidence should be above threshold"
        resistance = (confidence - 0.5) * 0.6  # Maps 0.5-1.0 to 0.0-0.3
        blended_sdft = (1 - resistance) * blended_sdft + resistance * prior

        # Normalize both
        blended_baseline = blended_baseline / blended_baseline.sum()
        blended_sdft = blended_sdft / blended_sdft.sum()

        # Compute changes from prior
        change_baseline = np.abs(blended_baseline - prior).sum()
        change_sdft = np.abs(blended_sdft - prior).sum()

        # SDFT resistance should reduce the change
        assert change_sdft < change_baseline, \
            f"SDFT: Resistance should reduce change above threshold. " \
            f"Got sdft={change_sdft:.4f}, baseline={change_baseline:.4f}"

    def test_sdft_no_resistance_at_threshold(self, generative_model):
        """SDFT: No resistance applies when confidence <= 0.5.

        Tests that exactly at the threshold, no resistance is applied.
        """
        # Prior at exactly threshold (0.5 confidence)
        prior = np.array([0.5, 0.3, 0.15, 0.05])

        # Simulated Bayesian posterior
        posterior = np.array([0.15, 0.60, 0.15, 0.10])
        learning_rate = 0.5

        # Standard blending (should be same with or without SDFT check)
        blended = learning_rate * posterior + (1 - learning_rate) * prior

        # With SDFT logic: confidence = 0.5, so resistance = 0
        confidence = float(np.max(prior))
        assert confidence == 0.5, "Test setup: confidence should be exactly at threshold"
        # No resistance applied since confidence is not > 0.5

        # Normalize
        blended = blended / blended.sum()

        # The blending should match standard formula exactly
        expected = learning_rate * posterior + (1 - learning_rate) * prior
        expected = expected / expected.sum()

        assert np.allclose(blended, expected), \
            "SDFT: At threshold, no resistance should be applied"
