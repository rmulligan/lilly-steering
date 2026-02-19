"""Tests for BeliefStore: Lilly's committed positions with dialectical history."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.active_inference.belief_store import (
    BeliefRelationType,
    BeliefConfidence,
    DialecticalHistory,
    CommittedBelief,
    BeliefRelation,
    BeliefStore,
    create_belief_store,
    DEFAULT_BELIEF_LIMIT,
)


def create_mock_graph_with_transaction():
    """Create a mock graph client that supports transaction context manager."""
    mock_graph = AsyncMock()
    mock_tx = AsyncMock()

    @asynccontextmanager
    async def mock_transaction():
        yield mock_tx

    mock_graph.transaction = mock_transaction
    # Share execute between tx and graph for assertion compatibility
    mock_tx.execute = mock_graph.execute
    return mock_graph


# =============================================================================
# BeliefRelationType Tests
# =============================================================================


class TestBeliefRelationType:
    """Tests for BeliefRelationType enum."""

    def test_supports_value(self):
        """Test SUPPORTS relation type value."""
        assert BeliefRelationType.SUPPORTS.value == "supports"

    def test_contradicts_value(self):
        """Test CONTRADICTS relation type value."""
        assert BeliefRelationType.CONTRADICTS.value == "contradicts"

    def test_refines_value(self):
        """Test REFINES relation type value."""
        assert BeliefRelationType.REFINES.value == "refines"

    def test_all_values_unique(self):
        """Test all relation types have unique values."""
        values = [rt.value for rt in BeliefRelationType]
        assert len(values) == len(set(values))


# =============================================================================
# BeliefConfidence Tests
# =============================================================================


class TestBeliefConfidence:
    """Tests for BeliefConfidence enum."""

    def test_tentative_value(self):
        """Test TENTATIVE confidence value."""
        assert BeliefConfidence.TENTATIVE.value == "tentative"

    def test_moderate_value(self):
        """Test MODERATE confidence value."""
        assert BeliefConfidence.MODERATE.value == "moderate"

    def test_strong_value(self):
        """Test STRONG confidence value."""
        assert BeliefConfidence.STRONG.value == "strong"

    def test_core_value(self):
        """Test CORE confidence value."""
        assert BeliefConfidence.CORE.value == "core"

    def test_all_values_unique(self):
        """Test all confidence levels have unique values."""
        values = [bc.value for bc in BeliefConfidence]
        assert len(values) == len(set(values))


# =============================================================================
# DialecticalHistory Tests
# =============================================================================


class TestDialecticalHistory:
    """Tests for DialecticalHistory dataclass."""

    def test_initialization_with_thesis_only(self):
        """Test creating dialectical history with just thesis."""
        history = DialecticalHistory(thesis="Initial position")

        assert history.thesis == "Initial position"
        assert history.antithesis == ""
        assert history.synthesis == ""
        assert history.synthesis_reasoning == ""
        assert isinstance(history.thesis_timestamp, datetime)
        assert history.antithesis_timestamp is None
        assert history.synthesis_timestamp is None

    def test_initialization_complete(self):
        """Test creating complete dialectical history."""
        now = datetime.now(timezone.utc)
        history = DialecticalHistory(
            thesis="Position A",
            thesis_timestamp=now,
            antithesis="Counter to A",
            antithesis_timestamp=now,
            synthesis="Resolved position",
            synthesis_timestamp=now,
            synthesis_reasoning="Resolved by combining both",
        )

        assert history.thesis == "Position A"
        assert history.antithesis == "Counter to A"
        assert history.synthesis == "Resolved position"
        assert history.synthesis_reasoning == "Resolved by combining both"

    def test_is_complete_false_thesis_only(self):
        """Test is_complete returns False with only thesis."""
        history = DialecticalHistory(thesis="Just thesis")
        assert history.is_complete() is False

    def test_is_complete_false_missing_synthesis(self):
        """Test is_complete returns False without synthesis."""
        history = DialecticalHistory(
            thesis="Position",
            antithesis="Counter",
        )
        assert history.is_complete() is False

    def test_is_complete_true(self):
        """Test is_complete returns True when all three are present."""
        history = DialecticalHistory(
            thesis="Position",
            antithesis="Counter",
            synthesis="Resolution",
        )
        assert history.is_complete() is True

    def test_to_dict_serialization(self):
        """Test dialectical history serialization."""
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        history = DialecticalHistory(
            thesis="Thesis text",
            thesis_timestamp=now,
            antithesis="Antithesis text",
            antithesis_timestamp=now,
            synthesis="Synthesis text",
            synthesis_timestamp=now,
            synthesis_reasoning="Reasoning",
        )

        result = history.to_dict()

        assert result["thesis"] == "Thesis text"
        assert result["thesis_timestamp"] == "2024-06-15T10:30:00+00:00"
        assert result["antithesis"] == "Antithesis text"
        assert result["antithesis_timestamp"] == "2024-06-15T10:30:00+00:00"
        assert result["synthesis"] == "Synthesis text"
        assert result["synthesis_timestamp"] == "2024-06-15T10:30:00+00:00"
        assert result["synthesis_reasoning"] == "Reasoning"

    def test_to_dict_with_none_timestamps(self):
        """Test serialization handles None timestamps."""
        history = DialecticalHistory(thesis="Just thesis")

        result = history.to_dict()

        assert result["antithesis_timestamp"] is None
        assert result["synthesis_timestamp"] is None

    def test_from_dict_deserialization(self):
        """Test dialectical history deserialization."""
        data = {
            "thesis": "Thesis text",
            "thesis_timestamp": "2024-06-15T10:30:00+00:00",
            "antithesis": "Antithesis text",
            "antithesis_timestamp": "2024-06-15T11:30:00+00:00",
            "synthesis": "Synthesis text",
            "synthesis_timestamp": "2024-06-15T12:30:00+00:00",
            "synthesis_reasoning": "Reasoning",
        }

        history = DialecticalHistory.from_dict(data)

        assert history.thesis == "Thesis text"
        assert history.thesis_timestamp == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert history.antithesis == "Antithesis text"
        assert history.antithesis_timestamp == datetime(2024, 6, 15, 11, 30, 0, tzinfo=timezone.utc)
        assert history.synthesis == "Synthesis text"
        assert history.synthesis_timestamp == datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        assert history.synthesis_reasoning == "Reasoning"

    def test_from_dict_safe_parsing_invalid_thesis_timestamp(self):
        """Test from_dict safely handles invalid thesis_timestamp."""
        data = {
            "thesis": "Thesis",
            "thesis_timestamp": "not-a-date",
        }

        history = DialecticalHistory.from_dict(data)

        assert history.thesis == "Thesis"
        # Should fallback to current time, not crash
        assert isinstance(history.thesis_timestamp, datetime)

    def test_from_dict_safe_parsing_invalid_optional_timestamps(self):
        """Test from_dict safely handles invalid optional timestamps."""
        data = {
            "thesis": "Thesis",
            "thesis_timestamp": "2024-06-15T10:30:00+00:00",
            "antithesis": "Anti",
            "antithesis_timestamp": "bad-date",
            "synthesis": "Synth",
            "synthesis_timestamp": {"not": "a string"},
        }

        history = DialecticalHistory.from_dict(data)

        # Should fallback to None for invalid optional timestamps
        assert history.antithesis_timestamp is None
        assert history.synthesis_timestamp is None

    def test_from_dict_empty_dict(self):
        """Test from_dict handles empty dictionary."""
        history = DialecticalHistory.from_dict({})

        assert history.thesis == ""
        assert isinstance(history.thesis_timestamp, datetime)
        assert history.antithesis == ""
        assert history.synthesis == ""

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        original = DialecticalHistory(
            thesis="Original thesis",
            thesis_timestamp=now,
            antithesis="Original antithesis",
            antithesis_timestamp=now,
            synthesis="Original synthesis",
            synthesis_timestamp=now,
            synthesis_reasoning="Original reasoning",
        )

        data = original.to_dict()
        restored = DialecticalHistory.from_dict(data)

        assert restored.thesis == original.thesis
        assert restored.thesis_timestamp == original.thesis_timestamp
        assert restored.antithesis == original.antithesis
        assert restored.antithesis_timestamp == original.antithesis_timestamp
        assert restored.synthesis == original.synthesis
        assert restored.synthesis_timestamp == original.synthesis_timestamp
        assert restored.synthesis_reasoning == original.synthesis_reasoning


# =============================================================================
# CommittedBelief Tests
# =============================================================================


class TestCommittedBelief:
    """Tests for CommittedBelief dataclass."""

    def test_initialization_basic(self):
        """Test basic belief creation."""
        history = DialecticalHistory(
            thesis="Position",
            antithesis="Counter",
            synthesis="Resolved",
        )
        belief = CommittedBelief(
            uid="belief:test123",
            statement="The resolved belief statement",
            topic="epistemology",
            confidence=0.75,
            dialectical_history=history,
        )

        assert belief.uid == "belief:test123"
        assert belief.statement == "The resolved belief statement"
        assert belief.topic == "epistemology"
        assert belief.confidence == 0.75
        assert belief.dialectical_history == history
        assert belief.supporting_evidence == []
        assert belief.challenges == []
        assert belief.goal_alignment == {}
        assert belief.revision_count == 0
        assert belief.tenant_id == "default"

    def test_get_confidence_level_tentative(self):
        """Test confidence level categorization - tentative."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.3, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.TENTATIVE

    def test_get_confidence_level_moderate(self):
        """Test confidence level categorization - moderate."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.6, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.MODERATE

    def test_get_confidence_level_strong(self):
        """Test confidence level categorization - strong."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.85, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.STRONG

    def test_get_confidence_level_core(self):
        """Test confidence level categorization - core."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.95, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.CORE

    def test_get_confidence_level_boundary_tentative_moderate(self):
        """Test confidence level at boundary 0.5."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.MODERATE

    def test_get_confidence_level_boundary_moderate_strong(self):
        """Test confidence level at boundary 0.75."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.75, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.STRONG

    def test_get_confidence_level_boundary_strong_core(self):
        """Test confidence level at boundary 0.9."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.9, dialectical_history=history
        )
        assert belief.get_confidence_level() == BeliefConfidence.CORE

    def test_add_evidence(self):
        """Test adding supporting evidence."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        original_revised = belief.revised_at

        belief.add_evidence("frag:12345")

        assert "frag:12345" in belief.supporting_evidence
        assert belief.revised_at >= original_revised

    def test_add_evidence_no_duplicates(self):
        """Test add_evidence prevents duplicates."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )

        belief.add_evidence("frag:12345")
        belief.add_evidence("frag:12345")

        assert len(belief.supporting_evidence) == 1

    def test_add_evidence_with_explicit_now(self):
        """Test add_evidence with explicit timestamp."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        belief.add_evidence("frag:12345", now=fixed_time)

        assert belief.revised_at == fixed_time

    def test_add_challenge(self):
        """Test adding a challenge."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        original_revised = belief.revised_at

        belief.add_challenge("What about edge case X?")

        assert "What about edge case X?" in belief.challenges
        assert belief.revised_at >= original_revised

    def test_add_challenge_no_duplicates(self):
        """Test add_challenge prevents duplicates."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )

        belief.add_challenge("Same challenge")
        belief.add_challenge("Same challenge")

        assert len(belief.challenges) == 1

    def test_add_challenge_with_explicit_now(self):
        """Test add_challenge with explicit timestamp."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        belief.add_challenge("Challenge", now=fixed_time)

        assert belief.revised_at == fixed_time

    def test_revise(self):
        """Test revising a belief."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="Original")
        belief = CommittedBelief(
            uid="test", statement="Original", topic="t", confidence=0.5, dialectical_history=history
        )

        belief.revise("Updated statement", "New reasoning")

        assert belief.statement == "Updated statement"
        assert belief.dialectical_history.synthesis == "Updated statement"
        assert belief.dialectical_history.synthesis_reasoning == "New reasoning"
        assert belief.revision_count == 1

    def test_revise_increments_count(self):
        """Test that revise increments revision count."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )

        belief.revise("Rev 1", "R1")
        belief.revise("Rev 2", "R2")
        belief.revise("Rev 3", "R3")

        assert belief.revision_count == 3

    def test_revise_with_explicit_now(self):
        """Test revise with explicit timestamp."""
        history = DialecticalHistory(thesis="T", antithesis="A", synthesis="S")
        belief = CommittedBelief(
            uid="test", statement="S", topic="t", confidence=0.5, dialectical_history=history
        )
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        belief.revise("Updated", "Reason", now=fixed_time)

        assert belief.revised_at == fixed_time
        assert belief.dialectical_history.synthesis_timestamp == fixed_time

    def test_to_dict_serialization(self):
        """Test belief serialization."""
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        history = DialecticalHistory(
            thesis="T", thesis_timestamp=now, antithesis="A", synthesis="S"
        )
        belief = CommittedBelief(
            uid="belief:serialize",
            statement="Test statement",
            topic="testing",
            confidence=0.8,
            dialectical_history=history,
            supporting_evidence=["frag:1", "frag:2"],
            challenges=["Challenge 1"],
            goal_alignment={"goal:test": 0.7},
            formed_at=now,
            revised_at=now,
            revision_count=2,
            tenant_id="test_tenant",
        )

        result = belief.to_dict()

        assert result["uid"] == "belief:serialize"
        assert result["statement"] == "Test statement"
        assert result["topic"] == "testing"
        assert result["confidence"] == 0.8
        assert result["supporting_evidence"] == ["frag:1", "frag:2"]
        assert result["challenges"] == ["Challenge 1"]
        assert result["goal_alignment"] == {"goal:test": 0.7}
        assert result["revision_count"] == 2
        assert result["tenant_id"] == "test_tenant"
        assert "dialectical_history" in result

    def test_from_dict_deserialization(self):
        """Test belief deserialization."""
        data = {
            "uid": "belief:deserialize",
            "statement": "Deserialized statement",
            "topic": "testing",
            "confidence": 0.75,
            "dialectical_history": {
                "thesis": "T",
                "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                "antithesis": "A",
                "antithesis_timestamp": "2024-06-15T10:30:00+00:00",
                "synthesis": "S",
                "synthesis_timestamp": "2024-06-15T10:30:00+00:00",
                "synthesis_reasoning": "R",
            },
            "supporting_evidence": ["frag:1"],
            "challenges": ["C1"],
            "goal_alignment": {"goal:test": 0.5},
            "formed_at": "2024-06-15T10:30:00+00:00",
            "revised_at": "2024-06-15T11:30:00+00:00",
            "revision_count": 1,
            "tenant_id": "test",
        }

        belief = CommittedBelief.from_dict(data)

        assert belief.uid == "belief:deserialize"
        assert belief.statement == "Deserialized statement"
        assert belief.topic == "testing"
        assert belief.confidence == 0.75
        assert belief.supporting_evidence == ["frag:1"]
        assert belief.challenges == ["C1"]
        assert belief.goal_alignment == {"goal:test": 0.5}
        assert belief.revision_count == 1
        assert belief.tenant_id == "test"
        assert belief.formed_at == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert belief.revised_at == datetime(2024, 6, 15, 11, 30, 0, tzinfo=timezone.utc)

    def test_from_dict_safe_parsing_invalid_formed_at(self):
        """Test from_dict safely handles invalid formed_at."""
        data = {
            "uid": "test",
            "statement": "S",
            "formed_at": "not-a-date",
            "revised_at": "2024-06-15T10:30:00+00:00",
        }

        belief = CommittedBelief.from_dict(data)

        # Should fallback to current time
        assert isinstance(belief.formed_at, datetime)

    def test_from_dict_safe_parsing_invalid_revised_at(self):
        """Test from_dict safely handles invalid revised_at."""
        data = {
            "uid": "test",
            "statement": "S",
            "formed_at": "2024-06-15T10:30:00+00:00",
            "revised_at": {"not": "a string"},
        }

        belief = CommittedBelief.from_dict(data)

        # Should fallback to current time
        assert isinstance(belief.revised_at, datetime)

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        history = DialecticalHistory(
            thesis="T",
            thesis_timestamp=now,
            antithesis="A",
            antithesis_timestamp=now,
            synthesis="S",
            synthesis_timestamp=now,
            synthesis_reasoning="R",
        )
        original = CommittedBelief(
            uid="belief:roundtrip",
            statement="Round trip test",
            topic="testing",
            confidence=0.7,
            dialectical_history=history,
            supporting_evidence=["frag:1"],
            challenges=["challenge"],
            goal_alignment={"goal:test": 0.5},
            formed_at=now,
            revised_at=now,
            revision_count=3,
            tenant_id="test",
        )

        data = original.to_dict()
        restored = CommittedBelief.from_dict(data)

        assert restored.uid == original.uid
        assert restored.statement == original.statement
        assert restored.topic == original.topic
        assert restored.confidence == original.confidence
        assert restored.supporting_evidence == original.supporting_evidence
        assert restored.challenges == original.challenges
        assert restored.goal_alignment == original.goal_alignment
        assert restored.formed_at == original.formed_at
        assert restored.revised_at == original.revised_at
        assert restored.revision_count == original.revision_count
        assert restored.tenant_id == original.tenant_id


# =============================================================================
# BeliefRelation Tests
# =============================================================================


class TestBeliefRelation:
    """Tests for BeliefRelation dataclass."""

    def test_initialization_basic(self):
        """Test basic relation creation."""
        relation = BeliefRelation(
            source_uid="belief:source",
            target_uid="belief:target",
            relation_type=BeliefRelationType.SUPPORTS,
        )

        assert relation.source_uid == "belief:source"
        assert relation.target_uid == "belief:target"
        assert relation.relation_type == BeliefRelationType.SUPPORTS
        assert relation.strength == 0.5
        assert isinstance(relation.created_at, datetime)

    def test_initialization_with_all_fields(self):
        """Test relation creation with all fields."""
        now = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        relation = BeliefRelation(
            source_uid="belief:a",
            target_uid="belief:b",
            relation_type=BeliefRelationType.CONTRADICTS,
            strength=0.9,
            created_at=now,
        )

        assert relation.strength == 0.9
        assert relation.created_at == now

    def test_all_relation_types(self):
        """Test creating relations with all types."""
        for rel_type in BeliefRelationType:
            relation = BeliefRelation(
                source_uid="a",
                target_uid="b",
                relation_type=rel_type,
            )
            assert relation.relation_type == rel_type


# =============================================================================
# BeliefStore Tests
# =============================================================================


class TestBeliefStore:
    """Tests for BeliefStore class."""

    def test_initialization_default(self):
        """Test default store initialization."""
        store = BeliefStore()

        assert store.graph is None
        assert store.tenant_id == "default"
        assert store.goal_registry is None
        assert store._beliefs == {}
        assert store._relations == []

    def test_initialization_with_parameters(self):
        """Test store initialization with parameters."""
        mock_graph = MagicMock()
        mock_registry = MagicMock()
        store = BeliefStore(
            graph=mock_graph,
            tenant_id="test_tenant",
            goal_registry=mock_registry,
        )

        assert store.graph == mock_graph
        assert store.tenant_id == "test_tenant"
        assert store.goal_registry == mock_registry

    def test_create_belief(self):
        """Test creating a new belief."""
        store = BeliefStore()

        belief = store.create_belief(
            topic="epistemology",
            thesis="Knowledge requires justification",
            antithesis="Some knowledge is self-evident",
            synthesis="Knowledge requires either justification or self-evidence",
            synthesis_reasoning="Both views have merit",
            confidence=0.75,
        )

        assert belief.uid.startswith("belief:")
        assert belief.statement == "Knowledge requires either justification or self-evidence"
        assert belief.topic == "epistemology"
        assert belief.confidence == 0.75
        assert belief.dialectical_history.is_complete()
        assert belief.uid in store._beliefs

    def test_create_belief_with_evidence(self):
        """Test creating belief with supporting evidence."""
        store = BeliefStore()

        belief = store.create_belief(
            topic="test",
            thesis="T",
            antithesis="A",
            synthesis="S",
            supporting_evidence=["frag:1", "frag:2"],
        )

        assert belief.supporting_evidence == ["frag:1", "frag:2"]

    def test_create_belief_with_explicit_now(self):
        """Test creating belief with explicit timestamp."""
        store = BeliefStore()
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        belief = store.create_belief(
            topic="test",
            thesis="T",
            antithesis="A",
            synthesis="S",
            now=fixed_time,
        )

        assert belief.formed_at == fixed_time
        assert belief.revised_at == fixed_time
        assert belief.dialectical_history.thesis_timestamp == fixed_time

    def test_create_belief_with_goal_registry(self):
        """Test creating belief calculates goal alignment."""
        mock_registry = MagicMock()
        mock_registry.calculate_total_alignment.return_value = (0.6, {"goal:test": 0.6})
        store = BeliefStore(goal_registry=mock_registry)

        belief = store.create_belief(
            topic="test",
            thesis="T",
            antithesis="A",
            synthesis="S",
        )

        assert belief.goal_alignment == {"goal:test": 0.6}
        mock_registry.calculate_total_alignment.assert_called_once_with("S")

    def test_get_belief_existing(self):
        """Test retrieving an existing belief."""
        store = BeliefStore()
        created = store.create_belief(
            topic="test", thesis="T", antithesis="A", synthesis="S"
        )

        retrieved = store.get_belief(created.uid)

        assert retrieved == created

    def test_get_belief_nonexistent(self):
        """Test retrieving a non-existent belief returns None."""
        store = BeliefStore()

        result = store.get_belief("belief:nonexistent")

        assert result is None

    def test_get_beliefs_by_topic(self):
        """Test filtering beliefs by topic."""
        store = BeliefStore()
        store.create_belief(topic="consciousness", thesis="T", antithesis="A", synthesis="S")
        store.create_belief(topic="CONSCIOUSNESS studies", thesis="T", antithesis="A", synthesis="S")
        store.create_belief(topic="physics", thesis="T", antithesis="A", synthesis="S")

        results = store.get_beliefs_by_topic("consciousness")

        assert len(results) == 2
        for belief in results:
            assert "consciousness" in belief.topic.lower()

    def test_get_beliefs_by_topic_case_insensitive(self):
        """Test topic filtering is case insensitive."""
        store = BeliefStore()
        store.create_belief(topic="Epistemology", thesis="T", antithesis="A", synthesis="S")

        results = store.get_beliefs_by_topic("EPISTEMOLOGY")

        assert len(results) == 1

    def test_get_beliefs_by_confidence(self):
        """Test filtering beliefs by confidence range."""
        store = BeliefStore()
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.3)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.6)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.9)

        results = store.get_beliefs_by_confidence(min_confidence=0.5, max_confidence=0.7)

        assert len(results) == 1
        assert results[0].confidence == 0.6

    def test_get_beliefs_by_confidence_inclusive(self):
        """Test confidence range is inclusive."""
        store = BeliefStore()
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.5)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.7)

        results = store.get_beliefs_by_confidence(min_confidence=0.5, max_confidence=0.7)

        assert len(results) == 2

    def test_get_core_beliefs(self):
        """Test retrieving core beliefs."""
        store = BeliefStore()
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.3)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.95)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.92)

        results = store.get_core_beliefs()

        assert len(results) == 2
        for belief in results:
            assert belief.get_confidence_level() == BeliefConfidence.CORE

    def test_get_challenged_beliefs(self):
        """Test retrieving beliefs with challenges."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")
        belief2.add_challenge("A challenge")

        results = store.get_challenged_beliefs()

        assert len(results) == 1
        assert results[0] == belief2


# =============================================================================
# BeliefStore Relation Tests
# =============================================================================


class TestBeliefStoreRelations:
    """Tests for BeliefStore relation methods."""

    def test_add_relation(self):
        """Test adding a relation between beliefs."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS)

        assert len(store._relations) == 1
        assert store._relations[0].source_uid == belief1.uid
        assert store._relations[0].target_uid == belief2.uid
        assert store._relations[0].relation_type == BeliefRelationType.SUPPORTS

    def test_add_relation_with_strength(self):
        """Test adding relation with custom strength."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS, strength=0.9)

        assert store._relations[0].strength == 0.9

    def test_add_relation_with_explicit_now(self):
        """Test adding relation with explicit timestamp."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        store.add_relation(
            belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS, now=fixed_time
        )

        assert store._relations[0].created_at == fixed_time

    def test_add_relation_nonexistent_source(self):
        """Test adding relation with non-existent source does nothing."""
        store = BeliefStore()
        belief = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")

        store.add_relation("belief:nonexistent", belief.uid, BeliefRelationType.SUPPORTS)

        assert len(store._relations) == 0

    def test_add_relation_nonexistent_target(self):
        """Test adding relation with non-existent target does nothing."""
        store = BeliefStore()
        belief = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")

        store.add_relation(belief.uid, "belief:nonexistent", BeliefRelationType.SUPPORTS)

        assert len(store._relations) == 0

    def test_get_related_beliefs(self):
        """Test retrieving related beliefs."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")
        belief3 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S3")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS)
        store.add_relation(belief1.uid, belief3.uid, BeliefRelationType.CONTRADICTS)

        results = store.get_related_beliefs(belief1.uid)

        assert len(results) == 2
        related_uids = {b.uid for b, _ in results}
        assert belief2.uid in related_uids
        assert belief3.uid in related_uids

    def test_get_related_beliefs_filtered_by_type(self):
        """Test retrieving related beliefs filtered by relation type."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")
        belief3 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S3")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS)
        store.add_relation(belief1.uid, belief3.uid, BeliefRelationType.CONTRADICTS)

        results = store.get_related_beliefs(belief1.uid, BeliefRelationType.SUPPORTS)

        assert len(results) == 1
        assert results[0][0].uid == belief2.uid
        assert results[0][1] == BeliefRelationType.SUPPORTS

    def test_get_related_beliefs_bidirectional(self):
        """Test that related beliefs include reverse relations."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS)

        # Query from target should also find the relation
        results = store.get_related_beliefs(belief2.uid)

        assert len(results) == 1
        assert results[0][0].uid == belief1.uid

    def test_get_contradicting_beliefs(self):
        """Test retrieving contradicting beliefs."""
        store = BeliefStore()
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")
        belief3 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S3")

        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.CONTRADICTS)
        store.add_relation(belief1.uid, belief3.uid, BeliefRelationType.SUPPORTS)

        results = store.get_contradicting_beliefs(belief1.uid)

        assert len(results) == 1
        assert results[0].uid == belief2.uid


# =============================================================================
# BeliefStore Revision Tests
# =============================================================================


class TestBeliefStoreRevision:
    """Tests for BeliefStore revision methods."""

    def test_revise_belief(self):
        """Test revising a belief."""
        store = BeliefStore()
        belief = store.create_belief(
            topic="test",
            thesis="T",
            antithesis="A",
            synthesis="Original synthesis",
        )

        store.revise_belief(
            belief.uid,
            new_synthesis="Updated synthesis",
            new_reasoning="Updated reasoning",
        )

        updated = store.get_belief(belief.uid)
        assert updated.statement == "Updated synthesis"
        assert updated.dialectical_history.synthesis == "Updated synthesis"
        assert updated.dialectical_history.synthesis_reasoning == "Updated reasoning"
        assert updated.revision_count == 1

    def test_revise_belief_with_confidence(self):
        """Test revising belief with new confidence."""
        store = BeliefStore()
        belief = store.create_belief(
            topic="test",
            thesis="T",
            antithesis="A",
            synthesis="S",
            confidence=0.5,
        )

        store.revise_belief(
            belief.uid,
            new_synthesis="Updated",
            new_reasoning="Reason",
            new_confidence=0.9,
        )

        updated = store.get_belief(belief.uid)
        assert updated.confidence == 0.9

    def test_revise_belief_with_explicit_now(self):
        """Test revising belief with explicit timestamp."""
        store = BeliefStore()
        belief = store.create_belief(topic="test", thesis="T", antithesis="A", synthesis="S")
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        store.revise_belief(
            belief.uid,
            new_synthesis="Updated",
            new_reasoning="Reason",
            now=fixed_time,
        )

        updated = store.get_belief(belief.uid)
        assert updated.revised_at == fixed_time

    def test_revise_belief_nonexistent(self):
        """Test revising non-existent belief does nothing."""
        store = BeliefStore()

        # Should not raise
        store.revise_belief(
            "belief:nonexistent",
            new_synthesis="Updated",
            new_reasoning="Reason",
        )

    def test_revise_belief_recalculates_goal_alignment(self):
        """Test revising belief recalculates goal alignment."""
        mock_registry = MagicMock()
        mock_registry.calculate_total_alignment.return_value = (0.8, {"goal:test": 0.8})
        store = BeliefStore(goal_registry=mock_registry)

        belief = store.create_belief(topic="test", thesis="T", antithesis="A", synthesis="S")
        mock_registry.calculate_total_alignment.reset_mock()
        mock_registry.calculate_total_alignment.return_value = (0.9, {"goal:test": 0.9})

        store.revise_belief(belief.uid, new_synthesis="Updated", new_reasoning="R")

        mock_registry.calculate_total_alignment.assert_called_once_with("Updated")
        assert store.get_belief(belief.uid).goal_alignment == {"goal:test": 0.9}


# =============================================================================
# BeliefStore Summarize Tests
# =============================================================================


class TestBeliefStoreSummarize:
    """Tests for BeliefStore.summarize method."""

    def test_summarize_empty(self):
        """Test summarizing empty store."""
        store = BeliefStore()

        summary = store.summarize()

        assert "Committed Beliefs Summary" in summary
        assert "Total beliefs: 0" in summary
        assert "Total relations: 0" in summary

    def test_summarize_with_beliefs(self):
        """Test summarizing store with beliefs."""
        store = BeliefStore()
        store.create_belief(
            topic="consciousness",
            thesis="T",
            antithesis="A",
            synthesis="Consciousness is fundamental",
            confidence=0.95,
        )

        summary = store.summarize()

        assert "CORE BELIEFS" in summary
        assert "consciousness" in summary.lower()
        assert "Total beliefs: 1" in summary

    def test_summarize_groups_by_confidence(self):
        """Test summary groups beliefs by confidence level."""
        store = BeliefStore()
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.3)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.6)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.8)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S", confidence=0.95)

        summary = store.summarize()

        assert "TENTATIVE BELIEFS" in summary
        assert "MODERATE BELIEFS" in summary
        assert "STRONG BELIEFS" in summary
        assert "CORE BELIEFS" in summary

    def test_summarize_shows_challenged_beliefs(self):
        """Test summary shows challenged beliefs section."""
        store = BeliefStore()
        belief = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")
        belief.add_challenge("Challenge 1")
        belief.add_challenge("Challenge 2")

        summary = store.summarize()

        assert "UNRESOLVED CHALLENGES" in summary
        assert "Challenges: 2" in summary


# =============================================================================
# BeliefStore Graph Persistence Tests
# =============================================================================


class TestBeliefStoreGraphOperations:
    """Tests for BeliefStore async graph operations."""

    @pytest.mark.asyncio
    async def test_save_to_graph_without_client(self):
        """Test save_to_graph does nothing without graph client."""
        store = BeliefStore()
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")

        # Should not raise
        await store.save_to_graph()

    @pytest.mark.asyncio
    async def test_save_to_graph_with_client(self):
        """Test save_to_graph persists beliefs."""
        mock_graph = create_mock_graph_with_transaction()
        store = BeliefStore(graph=mock_graph)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")

        await store.save_to_graph()

        # Should have called execute for the belief
        assert mock_graph.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_save_to_graph_with_relations(self):
        """Test save_to_graph persists relations."""
        mock_graph = create_mock_graph_with_transaction()
        store = BeliefStore(graph=mock_graph)
        belief1 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S1")
        belief2 = store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S2")
        store.add_relation(belief1.uid, belief2.uid, BeliefRelationType.SUPPORTS)

        await store.save_to_graph()

        # Should have called execute for beliefs and relation
        assert mock_graph.execute.call_count == 3  # 2 beliefs + 1 relation

    @pytest.mark.asyncio
    async def test_save_to_graph_with_explicit_now(self):
        """Test save_to_graph accepts now parameter."""
        mock_graph = create_mock_graph_with_transaction()
        store = BeliefStore(graph=mock_graph)
        store.create_belief(topic="t", thesis="T", antithesis="A", synthesis="S")
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        await store.save_to_graph(now=fixed_time)

        # Verify the call was made (now param is for testability, not directly visible in query)
        assert mock_graph.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_load_from_graph_without_client(self):
        """Test load_from_graph does nothing without graph client."""
        store = BeliefStore()

        # Should not raise
        await store.load_from_graph()

        assert len(store._beliefs) == 0

    @pytest.mark.asyncio
    async def test_load_from_graph_empty(self):
        """Test load_from_graph with empty results."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []
        store = BeliefStore(graph=mock_graph)

        await store.load_from_graph()

        assert len(store._beliefs) == 0

    @pytest.mark.asyncio
    async def test_load_from_graph_with_beliefs(self):
        """Test load_from_graph restores beliefs."""
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = [
            # First call: beliefs
            [
                {
                    "b": {
                        "uid": "belief:loaded",
                        "statement": "Loaded statement",
                        "topic": "test",
                        "confidence": 0.75,
                        "thesis": "T",
                        "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "antithesis": "A",
                        "antithesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "synthesis": "S",
                        "synthesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "synthesis_reasoning": "R",
                        "supporting_evidence": [],
                        "challenges": [],
                        "goal_alignment": {},
                        "revision_count": 0,
                        "tenant_id": "default",
                        "formed_at": "2024-06-15T10:30:00+00:00",
                        "revised_at": "2024-06-15T10:30:00+00:00",
                    }
                }
            ],
            # Second call: relations
            [],
        ]
        store = BeliefStore(graph=mock_graph)

        await store.load_from_graph()

        assert len(store._beliefs) == 1
        assert "belief:loaded" in store._beliefs
        assert store._beliefs["belief:loaded"].statement == "Loaded statement"

    @pytest.mark.asyncio
    async def test_load_from_graph_with_relations(self):
        """Test load_from_graph restores relations."""
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = [
            # First call: beliefs
            [
                {
                    "b": {
                        "uid": "belief:a",
                        "statement": "A",
                        "topic": "t",
                        "confidence": 0.5,
                        "thesis": "T",
                        "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "formed_at": "2024-06-15T10:30:00+00:00",
                        "revised_at": "2024-06-15T10:30:00+00:00",
                    }
                },
                {
                    "b": {
                        "uid": "belief:b",
                        "statement": "B",
                        "topic": "t",
                        "confidence": 0.5,
                        "thesis": "T",
                        "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "formed_at": "2024-06-15T10:30:00+00:00",
                        "revised_at": "2024-06-15T10:30:00+00:00",
                    }
                },
            ],
            # Second call: relations
            [
                {
                    "source": "belief:a",
                    "target": "belief:b",
                    "type": "supports",
                    "strength": 0.8,
                    "created_at": "2024-06-15T10:30:00+00:00",
                }
            ],
        ]
        store = BeliefStore(graph=mock_graph)

        await store.load_from_graph()

        assert len(store._relations) == 1
        assert store._relations[0].source_uid == "belief:a"
        assert store._relations[0].target_uid == "belief:b"
        assert store._relations[0].relation_type == BeliefRelationType.SUPPORTS

    @pytest.mark.asyncio
    async def test_load_from_graph_safe_timestamp_parsing(self):
        """Test load_from_graph safely handles invalid timestamps."""
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = [
            # First call: beliefs with invalid timestamps
            [
                {
                    "b": {
                        "uid": "belief:invalid_timestamps",
                        "statement": "Statement",
                        "topic": "t",
                        "confidence": 0.5,
                        "thesis": "T",
                        "thesis_timestamp": "not-a-date",
                        "antithesis_timestamp": "also-bad",
                        "formed_at": "invalid",
                        "revised_at": {"not": "a string"},
                    }
                }
            ],
            # Second call: relations with invalid timestamp
            [
                {
                    "source": "belief:invalid_timestamps",
                    "target": "belief:other",
                    "type": "supports",
                    "strength": 0.5,
                    "created_at": "bad-date",
                }
            ],
        ]
        store = BeliefStore(graph=mock_graph)

        # Should not raise
        await store.load_from_graph()

        # Should have loaded the belief with fallback timestamps
        assert len(store._beliefs) == 1
        belief = store._beliefs["belief:invalid_timestamps"]
        assert isinstance(belief.formed_at, datetime)
        assert isinstance(belief.revised_at, datetime)
        assert isinstance(belief.dialectical_history.thesis_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_load_from_graph_logs_warning_when_limit_reached(self):
        """Test load_from_graph logs warning when belief query hits the limit."""
        from unittest.mock import patch

        mock_graph = AsyncMock()
        # Create a list with exactly DEFAULT_BELIEF_LIMIT results
        mock_belief_results = [
            {
                "b": {
                    "uid": f"belief:{i}",
                    "statement": f"Statement {i}",
                    "topic": "t",
                    "confidence": 0.5,
                    "thesis": "T",
                    "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                    "formed_at": "2024-06-15T10:30:00+00:00",
                    "revised_at": "2024-06-15T10:30:00+00:00",
                }
            }
            for i in range(DEFAULT_BELIEF_LIMIT)
        ]
        mock_graph.query.side_effect = [
            mock_belief_results,  # First call: beliefs (at limit)
            [],  # Second call: relations
        ]
        store = BeliefStore(graph=mock_graph)

        with patch("core.active_inference.belief_store.logger") as mock_logger:
            await store.load_from_graph()

            # Verify warning was called with the expected message (sanitized - no limit value)
            mock_logger.warning.assert_any_call(
                "Belief query hit limit. Some beliefs may not be loaded. "
                "Consider pagination."
            )

    @pytest.mark.asyncio
    async def test_save_to_graph_transaction_failure(self):
        """Test that transaction failures include operation history."""
        from core.psyche.client import TransactionError, TransactionOperation

        mock_graph = AsyncMock()
        mock_tx = AsyncMock()

        # Track operations and simulate failure on second call
        operations = []
        call_count = 0

        async def failing_execute(cypher, params=None):
            nonlocal call_count
            call_count += 1
            op = TransactionOperation(cypher=cypher, params=params or {})
            operations.append(op)
            if call_count > 1:
                op.result = "Simulated database error"
                op.success = False
                raise TransactionError(
                    f"Transaction failed at operation {call_count}: Simulated database error",
                    operations=operations,
                )
            op.result = 1
            op.success = True
            return 1

        mock_tx.execute = failing_execute

        @asynccontextmanager
        async def mock_transaction():
            yield mock_tx

        mock_graph.transaction = mock_transaction

        store = BeliefStore(tenant_id="test", graph=mock_graph)
        store.create_belief(
            topic="test1",
            thesis="Thesis 1",
            antithesis="Antithesis 1",
            synthesis="Synthesis 1",
            confidence=0.7,
        )
        store.create_belief(
            topic="test2",
            thesis="Thesis 2",
            antithesis="Antithesis 2",
            synthesis="Synthesis 2",
            confidence=0.8,
        )

        with pytest.raises(TransactionError) as exc_info:
            await store.save_to_graph()

        # Verify error contains operation history
        assert len(exc_info.value.operations) >= 1
        assert exc_info.value.operations[0].success is True  # First op succeeded


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateBeliefStore:
    """Tests for create_belief_store factory function."""

    @pytest.mark.asyncio
    async def test_create_without_graph(self):
        """Test creating store without graph client."""
        store = await create_belief_store()

        assert store is not None
        assert store.graph is None
        assert store.tenant_id == "default"
        assert len(store._beliefs) == 0

    @pytest.mark.asyncio
    async def test_create_with_tenant_id(self):
        """Test creating store with custom tenant ID."""
        store = await create_belief_store(tenant_id="custom_tenant")

        assert store.tenant_id == "custom_tenant"

    @pytest.mark.asyncio
    async def test_create_with_goal_registry(self):
        """Test creating store with goal registry."""
        mock_registry = MagicMock()
        store = await create_belief_store(goal_registry=mock_registry)

        assert store.goal_registry == mock_registry

    @pytest.mark.asyncio
    async def test_create_with_graph_loads_from_graph(self):
        """Test creating store with graph triggers load."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []

        store = await create_belief_store(graph=mock_graph, load_from_graph=True)

        # Should have called query to load beliefs and relations
        assert mock_graph.query.call_count == 2

    @pytest.mark.asyncio
    async def test_create_skip_load_from_graph(self):
        """Test creating store can skip graph load."""
        mock_graph = AsyncMock()

        store = await create_belief_store(graph=mock_graph, load_from_graph=False)

        mock_graph.query.assert_not_called()


# =============================================================================
# Conditioning Set Tests (Epistemic Provenance - Wolpert Framework)
# =============================================================================


class TestConditioningSet:
    """Tests for conditioning_set field tracking epistemic provenance."""

    def test_conditioning_set_tracks_data_sources(self):
        """Beliefs should track what data they were conditioned on."""
        store = BeliefStore()
        belief = store.create_belief(
            topic="epistemology",
            thesis="Memory is reliable",
            antithesis="Memory could be a fluctuation",
            synthesis="Memory is reliable given physical laws",
            conditioning_set=["past_observation:2026-01-21", "physical_law:thermodynamics"],
        )

        assert belief.conditioning_set == ["past_observation:2026-01-21", "physical_law:thermodynamics"]
        assert len(belief.conditioning_set) == 2

    def test_conditioning_set_defaults_to_empty(self):
        """Beliefs without explicit conditioning should have empty set."""
        store = BeliefStore()
        belief = store.create_belief(
            topic="test",
            thesis="A",
            antithesis="B",
            synthesis="C",
        )

        assert belief.conditioning_set == []

    def test_conditioning_set_serializes_and_deserializes(self):
        """Conditioning set should survive serialization round-trip."""
        store = BeliefStore()
        belief = store.create_belief(
            topic="test",
            thesis="A",
            antithesis="B",
            synthesis="C",
            conditioning_set=["observation:morning", "law:entropy"],
        )

        data = belief.to_dict()
        restored = CommittedBelief.from_dict(data)

        assert restored.conditioning_set == ["observation:morning", "law:entropy"]

    @pytest.mark.asyncio
    async def test_conditioning_set_persists_to_graph(self):
        """Conditioning set should be saved and loaded from graph."""
        mock_graph = create_mock_graph_with_transaction()

        store = BeliefStore(graph=mock_graph)
        belief = store.create_belief(
            topic="test",
            thesis="A",
            antithesis="B",
            synthesis="C",
            conditioning_set=["data:primordial", "data:present"],
        )

        await store.save_to_graph()

        # Verify the query includes conditioning_set in the params
        call_args_list = mock_graph.execute.call_args_list
        # Find the call that saved the belief
        belief_save_call = call_args_list[0]
        params = belief_save_call[0][1]  # Second positional arg is params
        assert "conditioning_set" in params
        assert params["conditioning_set"] == ["data:primordial", "data:present"]

    @pytest.mark.asyncio
    async def test_conditioning_set_loads_from_graph(self):
        """Conditioning set should be restored from graph."""
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = [
            # First call: beliefs
            [
                {
                    "b": {
                        "uid": "belief:with_conditioning",
                        "statement": "Statement with conditioning",
                        "topic": "test",
                        "confidence": 0.75,
                        "thesis": "T",
                        "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "antithesis": "A",
                        "antithesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "synthesis": "S",
                        "synthesis_timestamp": "2024-06-15T10:30:00+00:00",
                        "synthesis_reasoning": "R",
                        "conditioning_set": ["data:primordial", "data:present"],
                        "supporting_evidence": [],
                        "challenges": [],
                        "goal_alignment": {},
                        "revision_count": 0,
                        "tenant_id": "default",
                        "formed_at": "2024-06-15T10:30:00+00:00",
                        "revised_at": "2024-06-15T10:30:00+00:00",
                    }
                }
            ],
            # Second call: relations
            [],
        ]
        store = BeliefStore(graph=mock_graph)

        await store.load_from_graph()

        assert len(store._beliefs) == 1
        belief = store._beliefs["belief:with_conditioning"]
        assert belief.conditioning_set == ["data:primordial", "data:present"]

    @pytest.mark.asyncio
    async def test_conditioning_set_loads_with_default_when_missing(self):
        """Conditioning set should default to empty list when missing from graph."""
        mock_graph = AsyncMock()
        mock_graph.query.side_effect = [
            # First call: beliefs (without conditioning_set field)
            [
                {
                    "b": {
                        "uid": "belief:old_format",
                        "statement": "Old belief without conditioning_set",
                        "topic": "test",
                        "confidence": 0.75,
                        "thesis": "T",
                        "thesis_timestamp": "2024-06-15T10:30:00+00:00",
                        # No conditioning_set field
                        "formed_at": "2024-06-15T10:30:00+00:00",
                        "revised_at": "2024-06-15T10:30:00+00:00",
                    }
                }
            ],
            # Second call: relations
            [],
        ]
        store = BeliefStore(graph=mock_graph)

        await store.load_from_graph()

        belief = store._beliefs["belief:old_format"]
        assert belief.conditioning_set == []
