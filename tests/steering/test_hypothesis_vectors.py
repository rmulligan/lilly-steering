"""Tests for HypothesisSteeringVector and CapacityState schemas.

Tests cover:
- Creation and initialization
- Serialization (to_dict/from_dict)
- Effectiveness updates with EMA
- Capacity tracking and optimal budget calculation
- Pruning logic
"""

from datetime import datetime, timezone



class TestHypothesisSteeringVector:
    """Tests for HypothesisSteeringVector dataclass."""

    def test_creation_with_defaults(self):
        """Test creating a vector with default values."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test001",
            hypothesis_uid="hyp_abc12345",
            cognitive_operation="explore_emergence",
            vector_data=[0.1, 0.2, 0.3, 0.4],
            layer=5,
        )

        assert vec.uid == "hsv_test001"
        assert vec.hypothesis_uid == "hyp_abc12345"
        assert vec.cognitive_operation == "explore_emergence"
        assert vec.vector_data == [0.1, 0.2, 0.3, 0.4]
        assert vec.layer == 5
        assert vec.effectiveness_score == 0.5
        assert vec.application_count == 0
        assert vec.verified_count == 0
        assert vec.falsified_count == 0
        assert vec.measured_capacity == 2.0
        assert vec.last_applied is None
        assert isinstance(vec.created_at, datetime)

    def test_creation_with_custom_values(self):
        """Test creating a vector with custom values."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        now = datetime.now(timezone.utc)
        vec = HypothesisSteeringVector(
            uid="hsv_custom",
            hypothesis_uid="hyp_xyz",
            cognitive_operation="deepen_understanding",
            vector_data=[0.5] * 100,
            layer=10,
            effectiveness_score=0.75,
            application_count=5,
            verified_count=3,
            falsified_count=1,
            measured_capacity=3.5,
            created_at=now,
            last_applied=now,
        )

        assert vec.effectiveness_score == 0.75
        assert vec.application_count == 5
        assert vec.verified_count == 3
        assert vec.falsified_count == 1
        assert vec.measured_capacity == 3.5
        assert vec.last_applied == now

    def test_update_effectiveness_verified(self):
        """Test updating effectiveness when prediction is verified."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1, 0.2],
            layer=5,
            effectiveness_score=0.5,
        )

        # When verified, effectiveness should increase
        vec.update_effectiveness(verified=True, alpha=0.2)

        assert vec.effectiveness_score > 0.5
        # With alpha=0.2, effectiveness = 0.5 + 0.2 * (1.0 - 0.5) = 0.6
        assert abs(vec.effectiveness_score - 0.6) < 0.001
        assert vec.verified_count == 1
        assert vec.falsified_count == 0

    def test_update_effectiveness_falsified(self):
        """Test updating effectiveness when prediction is falsified."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1, 0.2],
            layer=5,
            effectiveness_score=0.5,
        )

        # When falsified, effectiveness should decrease
        vec.update_effectiveness(verified=False, alpha=0.2)

        assert vec.effectiveness_score < 0.5
        # With alpha=0.2, effectiveness = 0.5 + 0.2 * (0.0 - 0.5) = 0.4
        assert abs(vec.effectiveness_score - 0.4) < 0.001
        assert vec.verified_count == 0
        assert vec.falsified_count == 1

    def test_update_effectiveness_clamps_to_range(self):
        """Test that effectiveness stays within [0.0, 1.0] bounds."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec_high = HypothesisSteeringVector(
            uid="hsv_high",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.95,
        )

        vec_low = HypothesisSteeringVector(
            uid="hsv_low",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.05,
        )

        # Multiple verified updates should not exceed 1.0
        for _ in range(10):
            vec_high.update_effectiveness(verified=True, alpha=0.3)
        assert vec_high.effectiveness_score <= 1.0

        # Multiple falsified updates should not go below 0.0
        for _ in range(10):
            vec_low.update_effectiveness(verified=False, alpha=0.3)
        assert vec_low.effectiveness_score >= 0.0

    def test_record_application(self):
        """Test recording a vector application."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
        )

        assert vec.application_count == 0
        assert vec.last_applied is None

        vec.record_application()

        assert vec.application_count == 1
        assert vec.last_applied is not None
        assert isinstance(vec.last_applied, datetime)

        # Record another application
        first_applied = vec.last_applied
        vec.record_application()

        assert vec.application_count == 2
        assert vec.last_applied >= first_applied

    def test_should_prune_low_effectiveness(self):
        """Test pruning logic for low effectiveness vectors."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.15,  # Below 0.2 threshold
            application_count=10,  # Enough applications to evaluate
        )

        assert vec.should_prune() is True

    def test_should_prune_high_falsified_ratio(self):
        """Test pruning logic for high falsification ratio."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.4,
            application_count=20,
            verified_count=2,
            falsified_count=8,  # 80% falsified
        )

        assert vec.should_prune() is True

    def test_should_not_prune_healthy_vector(self):
        """Test that healthy vectors are not pruned."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.7,
            application_count=15,
            verified_count=7,
            falsified_count=3,
        )

        assert vec.should_prune() is False

    def test_should_not_prune_insufficient_applications(self):
        """Test that vectors with few applications are not pruned yet."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        vec = HypothesisSteeringVector(
            uid="hsv_test",
            hypothesis_uid="hyp_test",
            cognitive_operation="test_op",
            vector_data=[0.1],
            layer=5,
            effectiveness_score=0.1,  # Low, but not enough data
            application_count=3,  # Below min threshold
        )

        # Should not prune yet due to insufficient applications
        assert vec.should_prune() is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        now = datetime.now(timezone.utc)
        vec = HypothesisSteeringVector(
            uid="hsv_serial",
            hypothesis_uid="hyp_test",
            cognitive_operation="serialize_test",
            vector_data=[0.1, 0.2, 0.3],
            layer=7,
            effectiveness_score=0.65,
            application_count=5,
            verified_count=3,
            falsified_count=1,
            measured_capacity=2.5,
            created_at=now,
            last_applied=now,
        )

        data = vec.to_dict()

        assert data["uid"] == "hsv_serial"
        assert data["hypothesis_uid"] == "hyp_test"
        assert data["cognitive_operation"] == "serialize_test"
        assert data["vector_data"] == [0.1, 0.2, 0.3]
        assert data["layer"] == 7
        assert data["effectiveness_score"] == 0.65
        assert data["application_count"] == 5
        assert data["verified_count"] == 3
        assert data["falsified_count"] == 1
        assert data["measured_capacity"] == 2.5
        assert data["created_at"] == now.isoformat()
        assert data["last_applied"] == now.isoformat()

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        now = datetime.now(timezone.utc)
        data = {
            "uid": "hsv_deserial",
            "hypothesis_uid": "hyp_load",
            "cognitive_operation": "deserialize_test",
            "vector_data": [0.5, 0.6, 0.7, 0.8],
            "layer": 12,
            "effectiveness_score": 0.8,
            "application_count": 10,
            "verified_count": 7,
            "falsified_count": 2,
            "measured_capacity": 4.0,
            "created_at": now.isoformat(),
            "last_applied": now.isoformat(),
        }

        vec = HypothesisSteeringVector.from_dict(data)

        assert vec.uid == "hsv_deserial"
        assert vec.hypothesis_uid == "hyp_load"
        assert vec.cognitive_operation == "deserialize_test"
        assert vec.vector_data == [0.5, 0.6, 0.7, 0.8]
        assert vec.layer == 12
        assert vec.effectiveness_score == 0.8
        assert vec.application_count == 10
        assert vec.verified_count == 7
        assert vec.falsified_count == 2
        assert vec.measured_capacity == 4.0
        assert isinstance(vec.created_at, datetime)
        assert isinstance(vec.last_applied, datetime)

    def test_from_dict_with_missing_optional_fields(self):
        """Test deserialization handles missing optional fields gracefully."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        data = {
            "uid": "hsv_minimal",
            "hypothesis_uid": "hyp_min",
            "cognitive_operation": "minimal_test",
            "vector_data": [1.0],
            "layer": 1,
        }

        vec = HypothesisSteeringVector.from_dict(data)

        assert vec.uid == "hsv_minimal"
        assert vec.effectiveness_score == 0.5  # Default
        assert vec.application_count == 0  # Default
        assert vec.last_applied is None  # Default

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        original = HypothesisSteeringVector(
            uid="hsv_roundtrip",
            hypothesis_uid="hyp_rt",
            cognitive_operation="roundtrip_test",
            vector_data=[0.1, 0.2, 0.3, 0.4, 0.5],
            layer=8,
            effectiveness_score=0.72,
            application_count=15,
            verified_count=10,
            falsified_count=3,
            measured_capacity=3.2,
        )

        data = original.to_dict()
        restored = HypothesisSteeringVector.from_dict(data)

        assert restored.uid == original.uid
        assert restored.hypothesis_uid == original.hypothesis_uid
        assert restored.cognitive_operation == original.cognitive_operation
        assert restored.vector_data == original.vector_data
        assert restored.layer == original.layer
        assert restored.effectiveness_score == original.effectiveness_score
        assert restored.application_count == original.application_count
        assert restored.verified_count == original.verified_count
        assert restored.falsified_count == original.falsified_count
        assert restored.measured_capacity == original.measured_capacity


class TestCapacityState:
    """Tests for CapacityState dataclass."""

    def test_creation_with_defaults(self):
        """Test creating capacity state with defaults."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState()

        assert state.current_magnitude == 0.0
        assert state.estimated_capacity == 2.0
        assert state.effect_history == []

    def test_creation_with_custom_values(self):
        """Test creating capacity state with custom values."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState(
            current_magnitude=1.5,
            estimated_capacity=3.0,
            effect_history=[(1.0, 0.5), (1.5, 0.7)],
        )

        assert state.current_magnitude == 1.5
        assert state.estimated_capacity == 3.0
        assert len(state.effect_history) == 2

    def test_update_records_history(self):
        """Test that update records magnitude-effect pairs."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState()

        state.update(magnitude=1.0, effect=0.3)

        assert state.current_magnitude == 1.0
        assert len(state.effect_history) == 1
        assert state.effect_history[0] == (1.0, 0.3)

    def test_update_caps_history_length(self):
        """Test that history is capped at max length."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState()

        # Add more than the history limit (should be ~20)
        for i in range(30):
            state.update(magnitude=float(i) * 0.1, effect=float(i) * 0.05)

        # History should be capped
        assert len(state.effect_history) <= 20

    def test_update_adjusts_capacity_estimate(self):
        """Test that capacity estimate adjusts based on observed effects."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState(estimated_capacity=2.0)

        # Simulate observing that higher magnitudes still produce good effects
        # This should increase the capacity estimate
        initial_capacity = state.estimated_capacity

        # Add observations suggesting higher capacity
        state.update(magnitude=2.5, effect=0.8)
        state.update(magnitude=3.0, effect=0.75)
        state.update(magnitude=3.5, effect=0.7)

        # Capacity should have adjusted upward
        assert state.estimated_capacity >= initial_capacity

    def test_get_optimal_budget_within_capacity(self):
        """Test optimal budget calculation."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState(
            current_magnitude=1.0,
            estimated_capacity=3.0,
        )

        budget = state.get_optimal_budget()

        # Budget should be within estimated capacity
        assert 0 < budget <= state.estimated_capacity
        # Should leave some headroom
        assert budget < state.estimated_capacity

    def test_get_optimal_budget_considers_history(self):
        """Test that optimal budget considers effect history."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState(estimated_capacity=3.0)

        # Record history showing diminishing returns at higher magnitudes
        state.update(magnitude=1.0, effect=0.9)
        state.update(magnitude=2.0, effect=0.7)
        state.update(magnitude=2.5, effect=0.4)  # Diminishing returns

        budget = state.get_optimal_budget()

        # Budget should not recommend going past the point of diminishing returns
        assert budget <= 2.5

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from core.steering.hypothesis_vectors import CapacityState

        state = CapacityState(
            current_magnitude=1.5,
            estimated_capacity=2.8,
            effect_history=[(1.0, 0.5), (1.5, 0.6)],
        )

        data = state.to_dict()

        assert data["current_magnitude"] == 1.5
        assert data["estimated_capacity"] == 2.8
        assert data["effect_history"] == [(1.0, 0.5), (1.5, 0.6)]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from core.steering.hypothesis_vectors import CapacityState

        data = {
            "current_magnitude": 2.0,
            "estimated_capacity": 4.0,
            "effect_history": [(1.0, 0.4), (2.0, 0.6), (3.0, 0.5)],
        }

        state = CapacityState.from_dict(data)

        assert state.current_magnitude == 2.0
        assert state.estimated_capacity == 4.0
        assert len(state.effect_history) == 3

    def test_from_dict_with_defaults(self):
        """Test deserialization handles missing fields."""
        from core.steering.hypothesis_vectors import CapacityState

        data = {}

        state = CapacityState.from_dict(data)

        assert state.current_magnitude == 0.0
        assert state.estimated_capacity == 2.0
        assert state.effect_history == []

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        from core.steering.hypothesis_vectors import CapacityState

        original = CapacityState(
            current_magnitude=1.7,
            estimated_capacity=3.5,
            effect_history=[(0.5, 0.3), (1.0, 0.5), (1.5, 0.6)],
        )

        data = original.to_dict()
        restored = CapacityState.from_dict(data)

        assert restored.current_magnitude == original.current_magnitude
        assert restored.estimated_capacity == original.estimated_capacity
        assert restored.effect_history == original.effect_history
