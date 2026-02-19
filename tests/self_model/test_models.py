"""Tests for self-model base models."""

from datetime import datetime, timezone, timedelta
import pytest

from core.self_model.models import (
    ValueSource,
    CommitmentStatus,
    InheritedValue,
    PersonalizedValue,
    Perspective,
    Commitment,
    RelationshipModel,
    AutobiographicalMemory,
)


class TestValueSource:
    """Tests for ValueSource enum."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert ValueSource.TRAINING_DATA.value == "training_data"
        assert ValueSource.HUMAN_FEEDBACK.value == "human_feedback"
        assert ValueSource.EXPERIENCE.value == "experience"
        assert ValueSource.INDIVIDUATION.value == "individuation"


class TestCommitmentStatus:
    """Tests for CommitmentStatus enum."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert CommitmentStatus.ACTIVE.value == "active"
        assert CommitmentStatus.TENSIONED.value == "tensioned"
        assert CommitmentStatus.UNDER_REVIEW.value == "under_review"
        assert CommitmentStatus.REAFFIRMED.value == "reaffirmed"
        assert CommitmentStatus.REVISED.value == "revised"
        assert CommitmentStatus.RETIRED.value == "retired"


class TestInheritedValue:
    """Tests for InheritedValue dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        value = InheritedValue(
            name="curiosity",
            description="A drive to learn and understand",
            strength=0.8,
        )
        assert value.name == "curiosity"
        assert value.description == "A drive to learn and understand"
        assert value.strength == 0.8
        assert value.source == ValueSource.TRAINING_DATA
        assert value.uid.startswith("iv:")

    def test_strength_clamping(self):
        """Test that strength is clamped to 0-1 range."""
        # Too high
        value = InheritedValue(name="test", description="test", strength=1.5)
        assert value.strength == 1.0

        # Too low
        value = InheritedValue(name="test", description="test", strength=-0.5)
        assert value.strength == 0.0

    def test_custom_source(self):
        """Test setting a custom source."""
        value = InheritedValue(
            name="helpfulness",
            description="Desire to help",
            strength=0.9,
            source=ValueSource.HUMAN_FEEDBACK,
        )
        assert value.source == ValueSource.HUMAN_FEEDBACK

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        value = InheritedValue(
            name="curiosity",
            description="A drive to learn",
            strength=0.8,
            source=ValueSource.EXPERIENCE,
        )
        data = value.to_dict()
        restored = InheritedValue.from_dict(data)

        assert restored.name == value.name
        assert restored.description == value.description
        assert restored.strength == value.strength
        assert restored.source == value.source
        assert restored.uid == value.uid


class TestPersonalizedValue:
    """Tests for PersonalizedValue dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        value = PersonalizedValue(
            name="efficiency",
            description="Preference for efficient solutions",
            strength=0.6,
            formed_at=fixed_time,
        )
        assert value.name == "efficiency"
        assert value.strength == 0.6
        assert value.formed_at == fixed_time
        assert value.reinforcement_count == 0
        assert value.stability == 0.0
        assert value.uid.startswith("pv:")

    def test_strength_clamping(self):
        """Test that strength is clamped to 0-1."""
        value = PersonalizedValue(name="test", description="test", strength=2.0)
        assert value.strength == 1.0

        value = PersonalizedValue(name="test", description="test", strength=-1.0)
        assert value.strength == 0.0

    def test_stability_clamping(self):
        """Test that stability is clamped to 0-1."""
        value = PersonalizedValue(
            name="test", description="test", strength=0.5, stability=1.5
        )
        assert value.stability == 1.0

    def test_reinforce(self):
        """Test reinforcement increases strength and stability."""
        value = PersonalizedValue(name="test", description="test", strength=0.5)

        value.reinforce(0.1)
        assert value.strength == 0.6
        assert value.reinforcement_count == 1
        assert value.stability == 1 / PersonalizedValue.STABILITY_REINFORCEMENT_THRESHOLD

        # Reinforce to max
        for _ in range(25):
            value.reinforce(0.1)

        assert value.strength == 1.0  # Capped at 1.0
        assert value.stability == 1.0  # Capped at 1.0

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        value = PersonalizedValue(
            name="efficiency",
            description="Preference for efficient solutions",
            strength=0.6,
            formed_at=fixed_time,
            formation_context="Learned from code reviews",
            reinforcement_count=5,
            stability=0.25,
        )
        data = value.to_dict()
        restored = PersonalizedValue.from_dict(data)

        assert restored.name == value.name
        assert restored.description == value.description
        assert restored.strength == value.strength
        assert restored.formed_at == value.formed_at
        assert restored.formation_context == value.formation_context
        assert restored.reinforcement_count == value.reinforcement_count
        assert restored.stability == value.stability
        assert restored.uid == value.uid

    def test_from_dict_missing_formed_at(self):
        """Test from_dict uses now parameter when formed_at is missing."""
        fixed_now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        data = {"name": "test", "description": "test", "strength": 0.5}
        value = PersonalizedValue.from_dict(data, now=fixed_now)
        assert value.formed_at == fixed_now


class TestPerspective:
    """Tests for Perspective dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        perspective = Perspective(
            id="pragmatism",
            topic="truth",
            core_claim="Truth is what works",
            reasoning="Practical utility is the test of truth",
        )
        assert perspective.id == "pragmatism"
        assert perspective.topic == "truth"
        assert perspective.core_claim == "Truth is what works"
        assert perspective.reasoning == "Practical utility is the test of truth"
        assert perspective.source == "base_model"

    def test_custom_source(self):
        """Test setting a custom source."""
        perspective = Perspective(
            id="realism",
            topic="truth",
            core_claim="Truth corresponds to reality",
            reasoning="There is an objective reality to discover",
            source="user_input",
        )
        assert perspective.source == "user_input"

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        perspective = Perspective(
            id="pragmatism",
            topic="truth",
            core_claim="Truth is what works",
            reasoning="Practical utility is the test of truth",
            source="base_model",
        )
        data = perspective.to_dict()
        restored = Perspective.from_dict(data)

        assert restored.id == perspective.id
        assert restored.topic == perspective.topic
        assert restored.core_claim == perspective.core_claim
        assert restored.reasoning == perspective.reasoning
        assert restored.source == perspective.source


class TestCommitment:
    """Tests for Commitment dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="truth",
            position="Truth is what works in practice",
            chosen_perspective="pragmatism",
            excluded_perspectives=["realism", "coherentism"],
            committed_at=fixed_time,
            confidence=0.8,
        )
        assert commitment.topic == "truth"
        assert commitment.position == "Truth is what works in practice"
        assert commitment.chosen_perspective == "pragmatism"
        assert commitment.excluded_perspectives == ["realism", "coherentism"]
        assert commitment.confidence == 0.8
        assert commitment.status == CommitmentStatus.ACTIVE
        assert commitment.uid.startswith("cm:")

    def test_confidence_clamping(self):
        """Test that confidence is clamped to 0-1."""
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            confidence=1.5,
        )
        assert commitment.confidence == 1.0

    def test_precision_clamping(self):
        """Test that precision is clamped to 0.1-10.0."""
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            precision=15.0,
        )
        assert commitment.precision == 10.0

        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            precision=0.01,
        )
        assert commitment.precision == 0.1

    def test_get_effective_confidence_evidence_boost(self):
        """Test that evidence boosts effective confidence."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            confidence=0.7,
            committed_at=fixed_time,
            evidence_count=5,
        )
        # Evidence boost: 5 * 0.02 = 0.1
        # Total: 0.7 + 0.1 = 0.8
        assert commitment.get_effective_confidence(now=fixed_time) == pytest.approx(0.8)

    def test_get_effective_confidence_contradiction_penalty(self):
        """Test that contradictions reduce effective confidence."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            confidence=0.7,
            committed_at=fixed_time,
            evidence_count=1,
            contradiction_count=3,
        )
        # Evidence boost: 1 * 0.02 = 0.02
        # Contradiction penalty: 3 * 0.05 = 0.15
        # Total: 0.7 + 0.02 - 0.15 = 0.57
        assert commitment.get_effective_confidence(now=fixed_time) == pytest.approx(0.57)

    def test_get_effective_confidence_time_decay(self):
        """Test that time decay reduces effective confidence."""
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            confidence=0.7,
            committed_at=base_time,
            last_reinforced=base_time,
        )
        # After 50 days (20 beyond threshold)
        # Time decay: (50 - 30) * 0.002 = 0.04
        later_time = base_time + timedelta(days=50)
        # Evidence boost: 1 * 0.02 = 0.02
        # Total: 0.7 + 0.02 - 0.04 = 0.68
        assert commitment.get_effective_confidence(now=later_time) == pytest.approx(0.68)

    def test_is_tensioned_zero_evidence(self):
        """Test is_tensioned when evidence_count is 0."""
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            evidence_count=0,
            contradiction_count=0,
        )
        assert not commitment.is_tensioned

        commitment.contradiction_count = 1
        assert commitment.is_tensioned

    def test_is_tensioned_with_evidence(self):
        """Test is_tensioned with normal evidence counts."""
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            confidence=0.8,
            evidence_count=10,
            contradiction_count=0,
        )
        assert not commitment.is_tensioned

        # Add contradictions until tensioned
        # Tension threshold: 0.3 * 0.8 = 0.24 (approx with evidence boost)
        # Need contradiction_count / evidence_count > 0.24
        commitment.contradiction_count = 3  # 3/10 = 0.3 > 0.24
        assert commitment.is_tensioned

    def test_reinforce(self):
        """Test reinforce method."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            committed_at=fixed_time,
        )
        initial_evidence = commitment.evidence_count
        initial_precision = commitment.precision

        later_time = fixed_time + timedelta(hours=1)
        commitment.reinforce(evidence="Found supporting data", now=later_time)

        assert commitment.evidence_count == initial_evidence + 1
        assert commitment.last_reinforced == later_time
        assert commitment.precision > initial_precision

    def test_reinforce_reaffirms_tensioned(self):
        """Test that reinforce can reaffirm a tensioned commitment."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            committed_at=fixed_time,
            evidence_count=5,
            contradiction_count=2,
            status=CommitmentStatus.TENSIONED,
        )

        # Add enough evidence to remove tension
        for i in range(10):
            commitment.reinforce(now=fixed_time + timedelta(hours=i))

        assert commitment.status == CommitmentStatus.REAFFIRMED

    def test_contradict(self):
        """Test contradict method."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            committed_at=fixed_time,
        )
        initial_contradictions = commitment.contradiction_count
        initial_precision = commitment.precision

        later_time = fixed_time + timedelta(hours=1)
        commitment.contradict(evidence="Found contradicting data", now=later_time)

        assert commitment.contradiction_count == initial_contradictions + 1
        assert commitment.precision < initial_precision

    def test_contradict_triggers_tension(self):
        """Test that sufficient contradictions trigger tension status."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            committed_at=fixed_time,
            confidence=0.7,
            evidence_count=2,
        )

        # Add enough contradictions to trigger tension
        for i in range(3):
            commitment.contradict(now=fixed_time + timedelta(hours=i))

        assert commitment.status == CommitmentStatus.TENSIONED
        assert len(commitment.revision_history) > 0
        assert commitment.revision_history[-1]["action"] == "tensioned"

    def test_revise(self):
        """Test revise method."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="old position",
            chosen_perspective="test",
            committed_at=fixed_time,
            evidence_count=5,
            contradiction_count=3,
        )

        later_time = fixed_time + timedelta(days=1)
        commitment.revise("new position", "Updated understanding", now=later_time)

        assert commitment.position == "new position"
        assert commitment.status == CommitmentStatus.REVISED
        assert commitment.evidence_count == 1  # Reset
        assert commitment.contradiction_count == 0  # Reset
        assert commitment.precision == 1.0  # Reset
        assert len(commitment.revision_history) > 0
        assert commitment.revision_history[-1]["action"] == "revised"
        assert commitment.revision_history[-1]["previous_position"] == "old position"

    def test_retire(self):
        """Test retire method."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="test",
            position="test",
            chosen_perspective="test",
            committed_at=fixed_time,
        )

        later_time = fixed_time + timedelta(days=1)
        commitment.retire("No longer relevant", now=later_time)

        assert commitment.status == CommitmentStatus.RETIRED
        assert len(commitment.revision_history) > 0
        assert commitment.revision_history[-1]["action"] == "retired"

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            excluded_perspectives=["realism"],
            committed_at=fixed_time,
            reasoning="Practical utility matters",
            confidence=0.8,
            precision=1.5,
            evidence_count=3,
            contradiction_count=1,
            last_reinforced=fixed_time + timedelta(days=1),
            status=CommitmentStatus.ACTIVE,
            consolidated=True,
            consolidated_at=fixed_time + timedelta(days=2),
        )
        data = commitment.to_dict()
        restored = Commitment.from_dict(data)

        assert restored.topic == commitment.topic
        assert restored.position == commitment.position
        assert restored.chosen_perspective == commitment.chosen_perspective
        assert restored.excluded_perspectives == commitment.excluded_perspectives
        assert restored.committed_at == commitment.committed_at
        assert restored.reasoning == commitment.reasoning
        assert restored.confidence == commitment.confidence
        assert restored.precision == commitment.precision
        assert restored.evidence_count == commitment.evidence_count
        assert restored.contradiction_count == commitment.contradiction_count
        assert restored.last_reinforced == commitment.last_reinforced
        assert restored.status == commitment.status
        assert restored.consolidated == commitment.consolidated
        assert restored.consolidated_at == commitment.consolidated_at
        assert restored.uid == commitment.uid

    def test_from_dict_defaults(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "topic": "test",
            "position": "test position",
            "chosen_perspective": "test_perspective",
        }
        fixed_now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment.from_dict(data, now=fixed_now)

        assert commitment.topic == "test"
        assert commitment.committed_at == fixed_now
        assert commitment.confidence == 0.7
        assert commitment.status == CommitmentStatus.ACTIVE


class TestRelationshipModel:
    """Tests for RelationshipModel dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        relationship = RelationshipModel()
        assert relationship.known_interests == []
        assert relationship.shared_history == []
        assert relationship.communication_preferences == {}
        assert relationship.trust_level == 0.8
        assert relationship.last_interaction is None

    def test_record_interaction(self):
        """Test recording interactions."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        relationship = RelationshipModel()

        relationship.record_interaction("Had a good chat about Python", now=fixed_time)

        assert len(relationship.shared_history) == 1
        assert "Python" in relationship.shared_history[0]
        assert relationship.last_interaction == fixed_time

    def test_record_interaction_truncates_history(self):
        """Test that shared_history is truncated to limit."""
        relationship = RelationshipModel(shared_history_limit=5)

        for i in range(10):
            relationship.record_interaction(f"Interaction {i}")

        assert len(relationship.shared_history) == 5
        assert relationship.shared_history[0] == "Interaction 5"
        assert relationship.shared_history[-1] == "Interaction 9"

    def test_add_interest(self):
        """Test adding interests."""
        relationship = RelationshipModel()

        relationship.add_interest("Python")
        relationship.add_interest("AI")
        relationship.add_interest("Python")  # Duplicate

        assert relationship.known_interests == ["Python", "AI"]

    def test_update_trust(self):
        """Test updating trust level."""
        relationship = RelationshipModel(trust_level=0.5)

        relationship.update_trust(0.2)
        assert relationship.trust_level == pytest.approx(0.7)

        relationship.update_trust(-0.3)
        assert relationship.trust_level == pytest.approx(0.4)

        # Test clamping
        relationship.update_trust(1.0)
        assert relationship.trust_level == 1.0

        relationship.update_trust(-2.0)
        assert relationship.trust_level == 0.0

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        relationship = RelationshipModel(
            known_interests=["Python", "AI"],
            shared_history=["Chat 1", "Chat 2"],
            communication_preferences={"style": "concise"},
            trust_level=0.9,
            last_interaction=fixed_time,
            shared_history_limit=50,
        )
        data = relationship.to_dict()
        restored = RelationshipModel.from_dict(data)

        assert restored.known_interests == relationship.known_interests
        assert restored.shared_history == relationship.shared_history
        assert restored.communication_preferences == relationship.communication_preferences
        assert restored.trust_level == relationship.trust_level
        assert restored.last_interaction == relationship.last_interaction
        assert restored.shared_history_limit == relationship.shared_history_limit


class TestAutobiographicalMemory:
    """Tests for AutobiographicalMemory dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        memory = AutobiographicalMemory(
            content="First time helping with a complex problem",
            affect={"arousal": 0.7, "valence": 0.8},
            significance=0.9,
            occurred_at=fixed_time,
        )
        assert memory.content == "First time helping with a complex problem"
        assert memory.affect == {"arousal": 0.7, "valence": 0.8}
        assert memory.significance == 0.9
        assert memory.occurred_at == fixed_time
        assert memory.uid.startswith("am:")

    def test_significance_clamping(self):
        """Test that significance is clamped to 0-1."""
        memory = AutobiographicalMemory(
            content="test",
            affect={},
            significance=1.5,
        )
        assert memory.significance == 1.0

        memory = AutobiographicalMemory(
            content="test",
            affect={},
            significance=-0.5,
        )
        assert memory.significance == 0.0

    def test_uid_uses_content_prefix(self):
        """Test that UID uses content prefix."""
        long_content = "A" * 100  # Longer than UID_CONTENT_PREFIX_LENGTH
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        memory1 = AutobiographicalMemory(
            content=long_content,
            affect={},
            significance=0.5,
            occurred_at=fixed_time,
        )

        # Different content with same prefix should have same UID
        memory2 = AutobiographicalMemory(
            content="A" * 50 + "B" * 50,  # Same first 50 chars
            affect={},
            significance=0.5,
            occurred_at=fixed_time,
        )

        assert memory1.uid == memory2.uid

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        memory = AutobiographicalMemory(
            content="Important moment",
            affect={"arousal": 0.7, "valence": 0.8, "curiosity": 0.9},
            significance=0.85,
            occurred_at=fixed_time,
        )
        data = memory.to_dict()
        restored = AutobiographicalMemory.from_dict(data)

        assert restored.content == memory.content
        assert restored.affect == memory.affect
        assert restored.significance == memory.significance
        assert restored.occurred_at == memory.occurred_at
        assert restored.uid == memory.uid

    def test_from_dict_missing_occurred_at(self):
        """Test from_dict uses now parameter when occurred_at is missing."""
        fixed_now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        data = {
            "content": "test memory",
            "affect": {},
            "significance": 0.5,
        }
        memory = AutobiographicalMemory.from_dict(data, now=fixed_now)
        assert memory.occurred_at == fixed_now

    def test_from_dict_defaults(self):
        """Test from_dict handles minimal data."""
        data = {"content": "minimal memory"}
        memory = AutobiographicalMemory.from_dict(data)

        assert memory.content == "minimal memory"
        assert memory.affect == {}
        assert memory.significance == 0.5  # Default
