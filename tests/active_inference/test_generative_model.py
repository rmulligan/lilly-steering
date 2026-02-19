"""Tests for the generative model: pymdp state/observation/action spaces."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.active_inference.generative_model import (
    # Hidden state enums
    TopicFocus,
    KnowledgeLevel,
    CognitiveMode,
    SeedState,
    # Observation enums
    NoteType,
    GraphConnectivity,
    UserBehavior,
    UncertaintyLevel,
    # Action enum
    ActionType,
    # Dataclasses
    StateSpace,
    ObservationSpace,
    ActionSpace,
    GenerativeModel,
)


# =============================================================================
# Hidden State Enum Tests
# =============================================================================

class TestTopicFocus:
    """Tests for TopicFocus hidden state enum."""

    def test_enum_values(self):
        """Verify TopicFocus enum values."""
        assert TopicFocus.WORK.value == 0
        assert TopicFocus.PERSONAL.value == 1
        assert TopicFocus.LEARNING.value == 2
        assert TopicFocus.EXPLORING.value == 3
        assert TopicFocus.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [t for t in TopicFocus if t != TopicFocus.NUM_STATES]
        assert len(actual_states) == TopicFocus.NUM_STATES.value


class TestKnowledgeLevel:
    """Tests for KnowledgeLevel hidden state enum."""

    def test_enum_values(self):
        """Verify KnowledgeLevel enum values."""
        assert KnowledgeLevel.NOVICE.value == 0
        assert KnowledgeLevel.INTERMEDIATE.value == 1
        assert KnowledgeLevel.EXPERT.value == 2
        assert KnowledgeLevel.NUM_STATES.value == 3

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [k for k in KnowledgeLevel if k != KnowledgeLevel.NUM_STATES]
        assert len(actual_states) == KnowledgeLevel.NUM_STATES.value


class TestCognitiveMode:
    """Tests for CognitiveMode hidden state enum."""

    def test_enum_values(self):
        """Verify CognitiveMode enum values."""
        assert CognitiveMode.TASKING.value == 0
        assert CognitiveMode.REFLECTIVE.value == 1
        assert CognitiveMode.LEARNING.value == 2
        assert CognitiveMode.EXPLORING.value == 3
        assert CognitiveMode.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [c for c in CognitiveMode if c != CognitiveMode.NUM_STATES]
        assert len(actual_states) == CognitiveMode.NUM_STATES.value


class TestSeedState:
    """Tests for SeedState hidden state enum."""

    def test_enum_values(self):
        """Verify SeedState enum values."""
        assert SeedState.ORPHAN.value == 0
        assert SeedState.SPROUTING.value == 1
        assert SeedState.CONNECTED.value == 2
        assert SeedState.MATURE.value == 3
        assert SeedState.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [s for s in SeedState if s != SeedState.NUM_STATES]
        assert len(actual_states) == SeedState.NUM_STATES.value


# =============================================================================
# Observation Enum Tests
# =============================================================================

class TestNoteType:
    """Tests for NoteType observation enum."""

    def test_enum_values(self):
        """Verify NoteType enum values."""
        assert NoteType.FRAGMENT.value == 0
        assert NoteType.QUESTION.value == 1
        assert NoteType.STATEMENT.value == 2
        assert NoteType.DIAGRAM.value == 3
        assert NoteType.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [n for n in NoteType if n != NoteType.NUM_STATES]
        assert len(actual_states) == NoteType.NUM_STATES.value


class TestGraphConnectivity:
    """Tests for GraphConnectivity observation enum."""

    def test_enum_values(self):
        """Verify GraphConnectivity enum values."""
        assert GraphConnectivity.ISOLATED.value == 0
        assert GraphConnectivity.SPARSE.value == 1
        assert GraphConnectivity.CONNECTED.value == 2
        assert GraphConnectivity.HUB.value == 3
        assert GraphConnectivity.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [g for g in GraphConnectivity if g != GraphConnectivity.NUM_STATES]
        assert len(actual_states) == GraphConnectivity.NUM_STATES.value


class TestUserBehavior:
    """Tests for UserBehavior observation enum."""

    def test_enum_values(self):
        """Verify UserBehavior enum values."""
        assert UserBehavior.FAST_INPUT.value == 0
        assert UserBehavior.SLOW_DELIBERATE.value == 1
        assert UserBehavior.REVIEWING.value == 2
        assert UserBehavior.IDLE.value == 3
        assert UserBehavior.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [u for u in UserBehavior if u != UserBehavior.NUM_STATES]
        assert len(actual_states) == UserBehavior.NUM_STATES.value


class TestUncertaintyLevel:
    """Tests for UncertaintyLevel observation enum."""

    def test_enum_values(self):
        """Verify UncertaintyLevel enum values."""
        assert UncertaintyLevel.LOW.value == 0
        assert UncertaintyLevel.MEDIUM.value == 1
        assert UncertaintyLevel.HIGH.value == 2
        assert UncertaintyLevel.VERY_HIGH.value == 3
        assert UncertaintyLevel.NUM_STATES.value == 4

    def test_num_states_correct(self):
        """NUM_STATES should equal the count of actual states."""
        actual_states = [u for u in UncertaintyLevel if u != UncertaintyLevel.NUM_STATES]
        assert len(actual_states) == UncertaintyLevel.NUM_STATES.value


# =============================================================================
# Action Enum Tests
# =============================================================================

class TestActionType:
    """Tests for ActionType enum with all 15 action types."""

    def test_basic_actions(self):
        """Verify basic action enum values."""
        assert ActionType.DO_NOTHING.value == 0
        assert ActionType.SILENT_LINK.value == 1
        assert ActionType.SUGGEST_LINK.value == 2
        assert ActionType.ASK_CLARIFICATION.value == 3
        assert ActionType.SURFACE_SEED.value == 4
        assert ActionType.TRIGGER_PREDICTION.value == 5
        assert ActionType.BRIDGE_CLUSTERS.value == 6
        assert ActionType.HUB_REVIEW.value == 7

    def test_hypothesis_actions(self):
        """Verify hypothesis testing action enum values."""
        assert ActionType.FORM_HYPOTHESIS.value == 8
        assert ActionType.TEST_HYPOTHESIS.value == 9
        assert ActionType.COMMIT_BELIEF.value == 10

    def test_dream_actions(self):
        """Verify dream/consolidation action enum values."""
        assert ActionType.DREAM_CONSOLIDATE.value == 11
        assert ActionType.DETECT_CONTRADICTIONS.value == 12

    def test_conflict_actions(self):
        """Verify conflict resolution action enum values."""
        assert ActionType.RESOLVE_CONFLICT.value == 13

    def test_meta_actions(self):
        """Verify meta-improvement action enum values."""
        assert ActionType.PROPOSE_IMPROVEMENT.value == 14

    def test_num_actions(self):
        """NUM_ACTIONS should equal 15."""
        assert ActionType.NUM_ACTIONS.value == 15

    def test_num_actions_correct(self):
        """NUM_ACTIONS should equal the count of actual actions."""
        actual_actions = [a for a in ActionType if a != ActionType.NUM_ACTIONS]
        assert len(actual_actions) == ActionType.NUM_ACTIONS.value


# =============================================================================
# StateSpace Tests
# =============================================================================

class TestStateSpace:
    """Tests for StateSpace dataclass."""

    def test_default_dimensions(self):
        """Default dimensions should match enum NUM_STATES."""
        space = StateSpace()
        assert space.topic_focus_dim == 4
        assert space.knowledge_level_dim == 3
        assert space.cognitive_mode_dim == 4
        assert space.seed_state_dim == 4

    def test_num_factors(self):
        """num_factors should be 4."""
        space = StateSpace()
        assert space.num_factors == 4

    def test_num_states(self):
        """num_states should return list of dimensions."""
        space = StateSpace()
        assert space.num_states == [4, 3, 4, 4]

    def test_total_states(self):
        """total_states should be 4 x 3 x 4 x 4 = 192."""
        space = StateSpace()
        assert space.total_states == 192

    def test_custom_dimensions(self):
        """Custom dimensions should be respected."""
        space = StateSpace(
            topic_focus_dim=2,
            knowledge_level_dim=2,
            cognitive_mode_dim=2,
            seed_state_dim=2,
        )
        assert space.total_states == 16

    def test_state_index_to_tuple(self):
        """state_index_to_tuple should convert enums to indices."""
        space = StateSpace()
        result = space.state_index_to_tuple(
            topic=TopicFocus.LEARNING,
            knowledge=KnowledgeLevel.EXPERT,
            mode=CognitiveMode.REFLECTIVE,
            seed=SeedState.MATURE,
        )
        assert result == (2, 2, 1, 3)


# =============================================================================
# ObservationSpace Tests
# =============================================================================

class TestObservationSpace:
    """Tests for ObservationSpace dataclass."""

    def test_default_dimensions(self):
        """Default dimensions should match enum NUM_STATES."""
        space = ObservationSpace()
        assert space.note_type_dim == 4
        assert space.graph_connectivity_dim == 4
        assert space.user_behavior_dim == 4
        assert space.uncertainty_level_dim == 4

    def test_num_modalities(self):
        """num_modalities should be 4."""
        space = ObservationSpace()
        assert space.num_modalities == 4

    def test_num_observations(self):
        """num_observations should return list of dimensions."""
        space = ObservationSpace()
        assert space.num_observations == [4, 4, 4, 4]

    def test_custom_dimensions(self):
        """Custom dimensions should be respected."""
        space = ObservationSpace(
            note_type_dim=3,
            graph_connectivity_dim=5,
            user_behavior_dim=2,
            uncertainty_level_dim=6,
        )
        assert space.num_observations == [3, 5, 2, 6]


# =============================================================================
# ActionSpace Tests
# =============================================================================

class TestActionSpace:
    """Tests for ActionSpace dataclass."""

    def test_default_num_actions(self):
        """Default num_actions should be 15."""
        space = ActionSpace()
        assert space.num_actions == 15

    def test_action_to_type(self):
        """action_to_type should convert index to ActionType."""
        space = ActionSpace()
        assert space.action_to_type(0) == ActionType.DO_NOTHING
        assert space.action_to_type(6) == ActionType.BRIDGE_CLUSTERS
        assert space.action_to_type(11) == ActionType.DREAM_CONSOLIDATE
        assert space.action_to_type(14) == ActionType.PROPOSE_IMPROVEMENT

    def test_action_to_type_invalid(self):
        """action_to_type should raise ValueError for invalid index."""
        space = ActionSpace()
        # ActionType(15) raises ValueError, but ActionType(-1) may not
        # depending on Python version, so we test what we can
        with pytest.raises(ValueError):
            space.action_to_type(100)  # Definitely out of range

    def test_action_names(self):
        """action_names should return lowercase action names."""
        space = ActionSpace()
        names = space.action_names
        assert len(names) == 15
        assert "do_nothing" in names
        assert "silent_link" in names
        assert "bridge_clusters" in names
        assert "dream_consolidate" in names
        assert "propose_improvement" in names

    def test_custom_num_actions(self):
        """Custom num_actions should be respected."""
        space = ActionSpace(num_actions=10)
        assert space.num_actions == 10


# =============================================================================
# GenerativeModel Tests
# =============================================================================

class TestGenerativeModel:
    """Tests for GenerativeModel class."""

    def test_default_initialization(self):
        """Default initialization should build all matrices."""
        model = GenerativeModel()
        assert len(model.A) == 4  # 4 modalities
        assert len(model.B) == 4  # 4 factors
        assert len(model.C) == 4  # 4 modalities
        assert len(model.D) == 4  # 4 factors

    def test_state_space_default(self):
        """state_space should have default values."""
        model = GenerativeModel()
        assert model.state_space.num_factors == 4
        assert model.state_space.total_states == 192

    def test_observation_space_default(self):
        """observation_space should have default values."""
        model = GenerativeModel()
        assert model.observation_space.num_modalities == 4

    def test_action_space_default(self):
        """action_space should have default values."""
        model = GenerativeModel()
        assert model.action_space.num_actions == 15


class TestGenerativeModelAMatrices:
    """Tests for A (observation likelihood) matrices."""

    def test_a_matrix_count(self):
        """Should have one A matrix per modality."""
        model = GenerativeModel()
        assert len(model.A) == model.observation_space.num_modalities

    def test_a_matrix_shapes(self):
        """A matrices should have shape (num_obs, num_states_factor)."""
        model = GenerativeModel()
        # A[0]: NoteType given SeedState
        assert model.A[0].shape == (4, 4)
        # A[1]: GraphConnectivity given SeedState
        assert model.A[1].shape == (4, 4)
        # A[2]: UserBehavior given CognitiveMode
        assert model.A[2].shape == (4, 4)
        # A[3]: UncertaintyLevel given SeedState
        assert model.A[3].shape == (4, 4)

    def test_a_matrices_are_probability_distributions(self):
        """A matrix columns should sum to 1 (probability distributions)."""
        model = GenerativeModel()
        for i, A_m in enumerate(model.A):
            col_sums = A_m.sum(axis=0)
            np.testing.assert_array_almost_equal(
                col_sums, np.ones_like(col_sums),
                err_msg=f"A[{i}] columns don't sum to 1"
            )

    def test_a_matrices_non_negative(self):
        """A matrices should have non-negative entries."""
        model = GenerativeModel()
        for i, A_m in enumerate(model.A):
            assert np.all(A_m >= 0), f"A[{i}] has negative entries"


class TestGenerativeModelBMatrices:
    """Tests for B (transition probability) matrices."""

    def test_b_matrix_count(self):
        """Should have one B matrix per state factor."""
        model = GenerativeModel()
        assert len(model.B) == model.state_space.num_factors

    def test_b_matrix_shapes(self):
        """B matrices should have shape (num_states, num_states, num_actions)."""
        model = GenerativeModel()
        num_actions = model.action_space.num_actions
        # B[0]: TopicFocus transitions
        assert model.B[0].shape == (4, 4, num_actions)
        # B[1]: KnowledgeLevel transitions
        assert model.B[1].shape == (3, 3, num_actions)
        # B[2]: CognitiveMode transitions
        assert model.B[2].shape == (4, 4, num_actions)
        # B[3]: SeedState transitions
        assert model.B[3].shape == (4, 4, num_actions)

    def test_b_matrices_are_transition_matrices(self):
        """B matrix columns should sum to 1 for each action."""
        model = GenerativeModel()
        for f, B_f in enumerate(model.B):
            for a in range(model.action_space.num_actions):
                col_sums = B_f[:, :, a].sum(axis=0)
                np.testing.assert_array_almost_equal(
                    col_sums, np.ones_like(col_sums),
                    err_msg=f"B[{f}][:,:,{a}] columns don't sum to 1"
                )

    def test_b_matrices_non_negative(self):
        """B matrices should have non-negative entries."""
        model = GenerativeModel()
        for f, B_f in enumerate(model.B):
            assert np.all(B_f >= 0), f"B[{f}] has negative entries"

    def test_seed_state_transitions_for_silent_link(self):
        """SILENT_LINK action should increase connectivity."""
        model = GenerativeModel()
        B_seed = model.B[3]
        a = ActionType.SILENT_LINK.value
        # orphan -> sprouting probability > 0
        assert B_seed[1, 0, a] > 0
        # sprouting -> connected probability > 0
        assert B_seed[2, 1, a] > 0

    def test_seed_state_transitions_for_bridge_clusters(self):
        """BRIDGE_CLUSTERS action should have major connectivity boost."""
        model = GenerativeModel()
        B_seed = model.B[3]
        a = ActionType.BRIDGE_CLUSTERS.value
        # orphan -> connected probability > 0
        assert B_seed[2, 0, a] > 0
        # sprouting -> connected probability should be higher than SILENT_LINK
        a_silent = ActionType.SILENT_LINK.value
        assert B_seed[2, 1, a] > B_seed[2, 1, a_silent]


class TestGenerativeModelCVectors:
    """Tests for C (prior preference) vectors."""

    def test_c_vector_count(self):
        """Should have one C vector per modality."""
        model = GenerativeModel()
        assert len(model.C) == model.observation_space.num_modalities

    def test_c_vector_shapes(self):
        """C vectors should have length equal to observations per modality."""
        model = GenerativeModel()
        for i, C_m in enumerate(model.C):
            assert len(C_m) == model.observation_space.num_observations[i]

    def test_c_prefers_connectivity(self):
        """C[1] should prefer HUB over ISOLATED."""
        model = GenerativeModel()
        C_conn = model.C[1]
        # HUB preference > ISOLATED preference
        assert C_conn[GraphConnectivity.HUB.value] > C_conn[GraphConnectivity.ISOLATED.value]

    def test_c_prefers_low_uncertainty(self):
        """C[3] should prefer LOW over VERY_HIGH uncertainty."""
        model = GenerativeModel()
        C_uncert = model.C[3]
        # LOW preference > VERY_HIGH preference
        assert C_uncert[UncertaintyLevel.LOW.value] > C_uncert[UncertaintyLevel.VERY_HIGH.value]


class TestGenerativeModelDVectors:
    """Tests for D (prior belief) vectors."""

    def test_d_vector_count(self):
        """Should have one D vector per state factor."""
        model = GenerativeModel()
        assert len(model.D) == model.state_space.num_factors

    def test_d_vector_shapes(self):
        """D vectors should have length equal to states per factor."""
        model = GenerativeModel()
        for i, D_f in enumerate(model.D):
            assert len(D_f) == model.state_space.num_states[i]

    def test_d_vectors_are_probability_distributions(self):
        """D vectors should sum to 1."""
        model = GenerativeModel()
        for i, D_f in enumerate(model.D):
            np.testing.assert_almost_equal(
                D_f.sum(), 1.0,
                err_msg=f"D[{i}] doesn't sum to 1"
            )

    def test_d_vectors_non_negative(self):
        """D vectors should have non-negative entries."""
        model = GenerativeModel()
        for i, D_f in enumerate(model.D):
            assert np.all(D_f >= 0), f"D[{i}] has negative entries"


class TestGenerativeModelPyMDPParams:
    """Tests for pymdp parameter export."""

    def test_get_pymdp_params_keys(self):
        """get_pymdp_params should return A, B, C, D, policy_len."""
        model = GenerativeModel()
        params = model.get_pymdp_params()
        assert "A" in params
        assert "B" in params
        assert "C" in params
        assert "D" in params
        assert "policy_len" in params

    def test_get_pymdp_params_values(self):
        """get_pymdp_params should return the model's matrices."""
        model = GenerativeModel()
        params = model.get_pymdp_params()
        assert params["A"] is model.A
        assert params["B"] is model.B
        assert params["C"] is model.C
        assert params["D"] is model.D
        assert params["policy_len"] == 1


# =============================================================================
# Serialization Tests
# =============================================================================

class TestGenerativeModelSerialization:
    """Tests for model serialization/deserialization."""

    def test_to_dict_keys(self):
        """to_dict should include all required keys."""
        model = GenerativeModel()
        data = model.to_dict()
        assert "state_space" in data
        assert "observation_space" in data
        assert "action_space" in data
        assert "A" in data
        assert "B" in data
        assert "C" in data
        assert "D" in data

    def test_to_dict_state_space(self):
        """to_dict should serialize state_space correctly."""
        model = GenerativeModel()
        data = model.to_dict()
        assert data["state_space"]["topic_focus_dim"] == 4
        assert data["state_space"]["knowledge_level_dim"] == 3
        assert data["state_space"]["cognitive_mode_dim"] == 4
        assert data["state_space"]["seed_state_dim"] == 4

    def test_to_dict_matrices_are_lists(self):
        """to_dict should convert numpy arrays to lists."""
        model = GenerativeModel()
        data = model.to_dict()
        # Check that A matrices are lists
        for a in data["A"]:
            assert isinstance(a, list)
        # Check that B matrices are lists
        for b in data["B"]:
            assert isinstance(b, list)

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) should produce equivalent model."""
        model = GenerativeModel()
        data = model.to_dict()
        restored = GenerativeModel.from_dict(data)

        # Check state space
        assert restored.state_space.total_states == model.state_space.total_states

        # Check observation space
        assert restored.observation_space.num_modalities == model.observation_space.num_modalities

        # Check action space
        assert restored.action_space.num_actions == model.action_space.num_actions

        # Check A matrices
        for i in range(len(model.A)):
            np.testing.assert_array_almost_equal(restored.A[i], model.A[i])

        # Check B matrices
        for i in range(len(model.B)):
            np.testing.assert_array_almost_equal(restored.B[i], model.B[i])

        # Check C vectors
        for i in range(len(model.C)):
            np.testing.assert_array_almost_equal(restored.C[i], model.C[i])

        # Check D vectors
        for i in range(len(model.D)):
            np.testing.assert_array_almost_equal(restored.D[i], model.D[i])

    def test_from_dict_with_empty_matrices(self):
        """from_dict should handle missing matrices gracefully."""
        data = {
            "state_space": {},
            "observation_space": {},
            "action_space": {},
        }
        model = GenerativeModel.from_dict(data)
        # Should build matrices from defaults
        assert len(model.A) == 4
        assert len(model.B) == 4


# =============================================================================
# from_graph_topology Tests
# =============================================================================

class TestFromGraphTopology:
    """Tests for from_graph_topology class method with mocked PsycheClient."""

    @pytest.mark.asyncio
    async def test_from_graph_topology_mature_graph(self):
        """Mature graph (>5 hubs) should adjust D[3] priors."""
        # Mock PsycheClient with 6 hubs
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(return_value=[
            {"uid": f"hub_{i}", "degree": 10 - i}
            for i in range(6)
        ])

        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")

        # D[3] should be adjusted for mature graph
        expected = np.array([0.2, 0.3, 0.35, 0.15])
        np.testing.assert_array_almost_equal(model.D[3], expected)

    @pytest.mark.asyncio
    async def test_from_graph_topology_growing_graph(self):
        """Growing graph (3-5 hubs) should adjust D[3] priors."""
        # Mock PsycheClient with 3 hubs
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(return_value=[
            {"uid": f"hub_{i}", "degree": 10 - i}
            for i in range(3)
        ])

        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")

        # D[3] should be adjusted for growing graph
        expected = np.array([0.3, 0.35, 0.25, 0.1])
        np.testing.assert_array_almost_equal(model.D[3], expected)

    @pytest.mark.asyncio
    async def test_from_graph_topology_new_graph(self):
        """New graph (<3 hubs) should keep default D[3] priors."""
        # Mock PsycheClient with 1 hub
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(return_value=[
            {"uid": "hub_0", "degree": 5}
        ])

        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")

        # D[3] should be the default (biased toward orphan)
        # Default is [0.4, 0.3, 0.2, 0.1] normalized
        expected = np.array([0.4, 0.3, 0.2, 0.1])
        np.testing.assert_array_almost_equal(model.D[3], expected)

    @pytest.mark.asyncio
    async def test_from_graph_topology_empty_hubs(self):
        """Empty hub list should keep default D[3] priors."""
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(return_value=[])

        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")

        # D[3] should be the default
        expected = np.array([0.4, 0.3, 0.2, 0.1])
        np.testing.assert_array_almost_equal(model.D[3], expected)

    @pytest.mark.asyncio
    async def test_from_graph_topology_handles_exception(self):
        """Exception in get_top_hubs should be handled gracefully."""
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(side_effect=Exception("Connection failed"))

        # Should not raise, just return default model
        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")

        # Model should still be valid
        assert len(model.A) == 4
        assert len(model.D) == 4

    @pytest.mark.asyncio
    async def test_from_graph_topology_other_matrices_unchanged(self):
        """A, B, C matrices should not be affected by topology."""
        mock_graph = MagicMock()
        mock_graph.get_top_hubs = AsyncMock(return_value=[
            {"uid": f"hub_{i}", "degree": 10}
            for i in range(10)
        ])

        model = await GenerativeModel.from_graph_topology(mock_graph, "test_tenant")
        default_model = GenerativeModel()

        # A matrices should be the same
        for i in range(len(model.A)):
            np.testing.assert_array_almost_equal(model.A[i], default_model.A[i])

        # B matrices should be the same
        for i in range(len(model.B)):
            np.testing.assert_array_almost_equal(model.B[i], default_model.B[i])

        # C vectors should be the same
        for i in range(len(model.C)):
            np.testing.assert_array_almost_equal(model.C[i], default_model.C[i])

        # D[0], D[1], D[2] should be the same
        for i in range(3):
            np.testing.assert_array_almost_equal(model.D[i], default_model.D[i])
