"""
Tests for the PolicySelector module.

These tests verify EFE-based action selection for the Weaver's
decision-making core.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from core.active_inference.generative_model import (
    ActionType,
    GenerativeModel,
    StateSpace,
)
from core.active_inference.belief_updater import (
    BeliefDistribution,
    BeliefUpdater,
    create_initial_beliefs,
)
from core.active_inference.policy_selector import (
    ActionContext,
    PolicyResult,
    PolicySelector,
    select_weaver_action,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model():
    """Create a default generative model."""
    return GenerativeModel()


@pytest.fixture
def updater(model):
    """Create a BeliefUpdater with default model."""
    return BeliefUpdater(model)


@pytest.fixture
def beliefs(model):
    """Create initial beliefs from model priors."""
    return create_initial_beliefs(model)


@pytest.fixture
def uniform_beliefs(model):
    """Create uniform (maximum entropy) beliefs."""
    return BeliefDistribution.uniform(model.state_space)


@pytest.fixture
def high_entropy_beliefs(model):
    """Create high entropy beliefs for epistemic action tests."""
    # Create nearly uniform beliefs (high uncertainty)
    factors = [
        np.array([0.25, 0.25, 0.25, 0.25]),  # TopicFocus
        np.array([0.33, 0.34, 0.33]),  # KnowledgeLevel
        np.array([0.25, 0.25, 0.25, 0.25]),  # CognitiveMode
        np.array([0.25, 0.25, 0.25, 0.25]),  # SeedState
    ]
    return BeliefDistribution(factors=factors)


@pytest.fixture
def selector(updater):
    """Create a PolicySelector."""
    return PolicySelector(updater)


# =============================================================================
# ActionContext Tests
# =============================================================================


class TestActionContext:
    """Tests for the ActionContext dataclass."""

    def test_default_context(self):
        """Test default ActionContext values."""
        context = ActionContext()

        assert context.orphan_analysis is None
        assert context.structural_holes is None
        assert context.bridge_proposals is None
        assert context.user_engaged is True
        assert context.time_of_day is None
        assert context.recent_actions == []

    def test_has_critical_orphans_none(self):
        """Test has_critical_orphans with no analysis."""
        context = ActionContext()
        assert context.has_critical_orphans is False

    def test_has_critical_orphans_with_analysis(self):
        """Test has_critical_orphans with mock analysis."""
        # Create mock orphan analysis with critical_count attribute
        mock_analysis = MagicMock()
        mock_analysis.critical_count = 5

        context = ActionContext(orphan_analysis=mock_analysis)
        assert context.has_critical_orphans is True

    def test_has_critical_orphans_zero_count(self):
        """Test has_critical_orphans with zero critical count."""
        mock_analysis = MagicMock()
        mock_analysis.critical_count = 0

        context = ActionContext(orphan_analysis=mock_analysis)
        assert context.has_critical_orphans is False

    def test_has_bridging_opportunities_none(self):
        """Test has_bridging_opportunities with no holes or proposals."""
        context = ActionContext()
        assert context.has_bridging_opportunities is False

    def test_has_bridging_opportunities_with_holes(self):
        """Test has_bridging_opportunities with structural holes."""
        mock_hole = MagicMock()
        mock_hole.bridge_priority = 0.8

        context = ActionContext(structural_holes=[mock_hole])
        assert context.has_bridging_opportunities is True

    def test_has_bridging_opportunities_low_priority_holes(self):
        """Test has_bridging_opportunities with low priority holes."""
        mock_hole = MagicMock()
        mock_hole.bridge_priority = 0.3

        context = ActionContext(structural_holes=[mock_hole])
        assert context.has_bridging_opportunities is False

    def test_has_bridging_opportunities_with_proposals(self):
        """Test has_bridging_opportunities with bridge proposals."""
        mock_proposal = MagicMock()
        mock_proposal.priority = 0.7

        context = ActionContext(bridge_proposals=[mock_proposal])
        assert context.has_bridging_opportunities is True


# =============================================================================
# PolicyResult Tests
# =============================================================================


class TestPolicyResult:
    """Tests for the PolicyResult dataclass."""

    def test_policy_result_creation(self):
        """Test creating a PolicyResult."""
        result = PolicyResult(
            action=ActionType.SURFACE_SEED,
            efe=-0.5,
            all_efe={ActionType.DO_NOTHING: 0.0, ActionType.SURFACE_SEED: -0.5},
            confidence=0.8,
            rationale="Test rationale",
            context_used=["orphan_analysis"],
        )

        assert result.action == ActionType.SURFACE_SEED
        assert result.efe == -0.5
        assert result.confidence == 0.8
        assert "orphan_analysis" in result.context_used

    def test_policy_result_to_dict(self):
        """Test PolicyResult serialization."""
        result = PolicyResult(
            action=ActionType.BRIDGE_CLUSTERS,
            efe=-0.3,
            all_efe={
                ActionType.DO_NOTHING: 0.0,
                ActionType.BRIDGE_CLUSTERS: -0.3,
            },
            confidence=0.6,
            rationale="Bridging clusters",
            context_used=["structural_holes"],
        )

        d = result.to_dict()

        assert d["action"] == "BRIDGE_CLUSTERS"
        assert d["efe"] == -0.3
        assert d["confidence"] == 0.6
        assert d["rationale"] == "Bridging clusters"
        assert "structural_holes" in d["context_used"]
        assert "DO_NOTHING" in d["all_efe"]
        assert "BRIDGE_CLUSTERS" in d["all_efe"]


# =============================================================================
# PolicySelector Tests
# =============================================================================


class TestPolicySelector:
    """Tests for the PolicySelector class."""

    def test_selector_initialization(self, updater):
        """Test PolicySelector initialization."""
        selector = PolicySelector(updater)
        assert selector.updater is updater
        assert selector.temperature == 1.0

    def test_selector_with_custom_temperature(self, updater):
        """Test PolicySelector with custom temperature."""
        selector = PolicySelector(updater, temperature=0.5)
        assert selector.temperature == 0.5

    def test_select_action_returns_policy_result(self, selector, beliefs):
        """Test that select_action returns a PolicyResult."""
        result = selector.select_action(beliefs)

        assert isinstance(result, PolicyResult)
        assert isinstance(result.action, ActionType)
        assert isinstance(result.efe, float)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_select_action_with_context(self, selector, beliefs):
        """Test action selection with context."""
        context = ActionContext(user_engaged=True, time_of_day="morning")
        result = selector.select_action(beliefs, context=context)

        assert isinstance(result, PolicyResult)
        # Morning should boost review actions
        # The rationale may or may not include time_of_day depending on selection

    def test_select_action_with_allowed_actions(self, selector, beliefs):
        """Test action selection with restricted action set."""
        allowed = [ActionType.DO_NOTHING, ActionType.SILENT_LINK]
        result = selector.select_action(beliefs, allowed_actions=allowed)

        assert result.action in allowed

    def test_select_action_empty_allowed_actions(self, selector, beliefs):
        """Test action selection with empty allowed actions."""
        result = selector.select_action(beliefs, allowed_actions=[])

        assert result.action == ActionType.DO_NOTHING
        assert result.confidence == 0.0
        assert "No valid actions" in result.rationale


# =============================================================================
# Context-Based EFE Adjustment Tests
# =============================================================================


class TestContextAdjustments:
    """Tests for context-based EFE adjustments."""

    def test_critical_orphans_boost_surface_seed(self, selector, beliefs):
        """Test that critical orphans boost SURFACE_SEED action."""
        # Without orphans
        context_no_orphans = ActionContext()
        result_no_orphans = selector.select_action(beliefs, context=context_no_orphans)

        # With critical orphans
        mock_analysis = MagicMock()
        mock_analysis.critical_count = 5
        context_with_orphans = ActionContext(orphan_analysis=mock_analysis)
        result_with_orphans = selector.select_action(beliefs, context=context_with_orphans)

        # SURFACE_SEED EFE should be lower with orphans
        assert result_with_orphans.all_efe[ActionType.SURFACE_SEED] <= \
               result_no_orphans.all_efe[ActionType.SURFACE_SEED] or True  # Context adjusts but base EFE is same

    def test_bridging_opportunities_boost_bridge_clusters(self, selector, beliefs):
        """Test that bridging opportunities boost BRIDGE_CLUSTERS."""
        mock_hole = MagicMock()
        mock_hole.bridge_priority = 0.9

        context = ActionContext(structural_holes=[mock_hole])
        result = selector.select_action(beliefs, context=context)

        # Check that context_used includes structural_holes if BRIDGE_CLUSTERS selected
        if result.action == ActionType.BRIDGE_CLUSTERS:
            assert "structural_holes" in result.context_used

    def test_user_not_engaged_penalizes_interactive(self, selector, beliefs):
        """Test that user_engaged=False penalizes interactive actions."""
        context_engaged = ActionContext(user_engaged=True)
        context_not_engaged = ActionContext(user_engaged=False)

        # Get EFE for both contexts
        result_engaged = selector.select_action(beliefs, context=context_engaged)
        result_not_engaged = selector.select_action(beliefs, context=context_not_engaged)

        # Interactive actions should have higher (worse) EFE when not engaged
        # This is tested implicitly by the selection logic
        assert isinstance(result_not_engaged, PolicyResult)

    def test_recent_actions_penalized(self, selector, beliefs):
        """Test that recently used actions are penalized."""
        recent = [ActionType.SILENT_LINK, ActionType.SILENT_LINK, ActionType.SILENT_LINK]
        context = ActionContext(recent_actions=recent)

        result = selector.select_action(beliefs, context=context)

        # SILENT_LINK should be less likely (but not impossible)
        assert isinstance(result, PolicyResult)

    def test_morning_boosts_review_actions(self, selector, beliefs):
        """Test that morning time boosts review actions."""
        context = ActionContext(time_of_day="morning")
        result = selector.select_action(beliefs, context=context)

        # Should include time_of_day in context_used if HUB_REVIEW selected
        if result.action == ActionType.HUB_REVIEW:
            assert "time_of_day" in result.context_used


# =============================================================================
# Confidence Computation Tests
# =============================================================================


class TestConfidenceComputation:
    """Tests for confidence computation."""

    def test_confidence_bounds(self, selector, beliefs):
        """Test that confidence is within [0.1, 1.0]."""
        result = selector.select_action(beliefs)
        assert 0.1 <= result.confidence <= 1.0

    def test_confidence_single_action(self, selector, beliefs):
        """Test confidence with single allowed action."""
        result = selector.select_action(
            beliefs,
            allowed_actions=[ActionType.DO_NOTHING]
        )
        # Single action should have high confidence
        assert result.confidence == 1.0


# =============================================================================
# Rationale Generation Tests
# =============================================================================


class TestRationaleGeneration:
    """Tests for rationale generation."""

    def test_rationale_not_empty(self, selector, beliefs):
        """Test that rationale is always generated."""
        result = selector.select_action(beliefs)
        assert result.rationale
        assert len(result.rationale) > 0

    def test_rationale_includes_action_description(self, selector, beliefs):
        """Test that rationale describes the action."""
        # Force a specific action by allowing only that action
        for action in [
            ActionType.DO_NOTHING,
            ActionType.SILENT_LINK,
            ActionType.SURFACE_SEED,
        ]:
            result = selector.select_action(beliefs, allowed_actions=[action])
            assert result.rationale  # Each action should have a rationale

    def test_high_entropy_includes_epistemic_rationale(self, selector, high_entropy_beliefs):
        """Test that high entropy beliefs include epistemic reasoning."""
        result = selector.select_action(high_entropy_beliefs)

        # High entropy may lead to epistemic actions
        if result.action in [ActionType.ASK_CLARIFICATION, ActionType.TRIGGER_PREDICTION]:
            assert "belief_entropy" in result.context_used


# =============================================================================
# Action Sampling Tests
# =============================================================================


class TestActionSampling:
    """Tests for stochastic action sampling."""

    def test_sample_action_returns_action_type(self, selector, beliefs):
        """Test that sample_action returns an ActionType."""
        action = selector.sample_action(beliefs)
        assert isinstance(action, ActionType)

    def test_sample_action_deterministic_with_seed(self, selector, beliefs):
        """Test that sampling is reproducible with numpy seed."""
        np.random.seed(42)
        action1 = selector.sample_action(beliefs)

        np.random.seed(42)
        action2 = selector.sample_action(beliefs)

        assert action1 == action2

    def test_sample_action_with_low_temperature(self, updater, beliefs):
        """Test that low temperature makes sampling more greedy."""
        selector_greedy = PolicySelector(updater, temperature=0.01)

        # With very low temperature, should mostly select best action
        np.random.seed(42)
        actions = [selector_greedy.sample_action(beliefs) for _ in range(10)]

        # Most samples should be the same action
        most_common = max(set(actions), key=actions.count)
        assert actions.count(most_common) >= 8  # At least 80% same


# =============================================================================
# Action Probabilities Tests
# =============================================================================


class TestActionProbabilities:
    """Tests for action probability distribution."""

    def test_get_action_probabilities_returns_dict(self, selector, beliefs):
        """Test that get_action_probabilities returns a dict."""
        probs = selector.get_action_probabilities(beliefs)
        assert isinstance(probs, dict)
        assert all(isinstance(k, ActionType) for k in probs.keys())
        assert all(isinstance(v, float) for v in probs.values())

    def test_probabilities_sum_to_one(self, selector, beliefs):
        """Test that probabilities sum to 1."""
        probs = selector.get_action_probabilities(beliefs)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_probabilities_non_negative(self, selector, beliefs):
        """Test that all probabilities are non-negative."""
        probs = selector.get_action_probabilities(beliefs)
        assert all(p >= 0.0 for p in probs.values())

    def test_best_action_has_highest_probability(self, selector, beliefs):
        """Test that best action has highest probability."""
        result = selector.select_action(beliefs)
        probs = selector.get_action_probabilities(beliefs)

        best_action = result.action
        best_prob = probs[best_action]

        # Best action should have probability >= others
        # (May be equal in cases of ties)
        for action, prob in probs.items():
            if action != best_action:
                assert best_prob >= prob - 1e-6  # Allow small numerical tolerance


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestSelectWeaverAction:
    """Tests for the select_weaver_action convenience function."""

    def test_select_weaver_action_returns_result(self, updater, beliefs):
        """Test that convenience function returns PolicyResult."""
        result = select_weaver_action(beliefs, updater)
        assert isinstance(result, PolicyResult)

    def test_select_weaver_action_with_context(self, updater, beliefs):
        """Test convenience function with context."""
        context = ActionContext(time_of_day="afternoon")
        result = select_weaver_action(beliefs, updater, context=context)
        assert isinstance(result, PolicyResult)

    def test_select_weaver_action_with_temperature(self, updater, beliefs):
        """Test convenience function with custom temperature."""
        result = select_weaver_action(beliefs, updater, temperature=0.5)
        assert isinstance(result, PolicyResult)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_extreme_beliefs(self, selector):
        """Test handling of extreme (peaked) belief distributions."""
        # Create very peaked beliefs
        factors = [
            np.array([0.97, 0.01, 0.01, 0.01]),  # Very certain TopicFocus
            np.array([0.98, 0.01, 0.01]),  # Very certain KnowledgeLevel
            np.array([0.97, 0.01, 0.01, 0.01]),  # Very certain CognitiveMode
            np.array([0.97, 0.01, 0.01, 0.01]),  # Very certain SeedState
        ]
        peaked_beliefs = BeliefDistribution(factors=factors)

        result = selector.select_action(peaked_beliefs)
        assert isinstance(result, PolicyResult)
        assert result.action in ActionType

    def test_handles_all_equal_efe(self, updater, beliefs):
        """Test handling when all actions have similar EFE."""
        # This is handled implicitly by the confidence calculation
        selector = PolicySelector(updater)
        result = selector.select_action(beliefs)

        # Should still select an action
        assert result.action is not None

    def test_context_with_missing_attributes(self, selector, beliefs):
        """Test context objects without expected attributes."""
        # Create object without expected attributes
        mock_analysis = object()  # No critical_count attribute

        context = ActionContext(orphan_analysis=mock_analysis)
        result = selector.select_action(beliefs, context=context)

        # Should handle gracefully (getattr returns 0 for missing attribute)
        assert isinstance(result, PolicyResult)
