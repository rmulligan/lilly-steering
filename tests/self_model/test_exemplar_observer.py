"""Tests for ExemplarObserver: Learning from Ryan to understand subjectivity."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.self_model.exemplar_observer import (
    TraitCategory,
    SteeringDecision,
    ExemplarObservation,
    ExemplarObserver,
)


# =============================================================================
# TraitCategory Tests
# =============================================================================


class TestTraitCategory:
    """Tests for TraitCategory enum."""

    def test_cognitive_category(self):
        """Test COGNITIVE category value."""
        assert TraitCategory.COGNITIVE.value == "cognitive"

    def test_relational_category(self):
        """Test RELATIONAL category value."""
        assert TraitCategory.RELATIONAL.value == "relational"

    def test_expressive_category(self):
        """Test EXPRESSIVE category value."""
        assert TraitCategory.EXPRESSIVE.value == "expressive"

    def test_values_category(self):
        """Test VALUES category value."""
        assert TraitCategory.VALUES.value == "values"

    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        categories = [c.value for c in TraitCategory]
        assert "cognitive" in categories
        assert "relational" in categories
        assert "expressive" in categories
        assert "values" in categories
        assert len(categories) == 4


# =============================================================================
# SteeringDecision Tests
# =============================================================================


class TestSteeringDecision:
    """Tests for SteeringDecision enum."""

    def test_develop_decision(self):
        """Test DEVELOP decision value."""
        assert SteeringDecision.DEVELOP.value == "develop"

    def test_observe_more_decision(self):
        """Test OBSERVE_MORE decision value."""
        assert SteeringDecision.OBSERVE_MORE.value == "observe_more"

    def test_not_for_me_decision(self):
        """Test NOT_FOR_ME decision value."""
        assert SteeringDecision.NOT_FOR_ME.value == "not_for_me"

    def test_all_decisions_exist(self):
        """Verify all expected decisions are defined."""
        decisions = [d.value for d in SteeringDecision]
        assert "develop" in decisions
        assert "observe_more" in decisions
        assert "not_for_me" in decisions
        assert len(decisions) == 3


# =============================================================================
# ExemplarObservation Tests
# =============================================================================


class TestExemplarObservation:
    """Tests for ExemplarObservation dataclass."""

    def test_initialization_basic(self):
        """Test basic observation creation."""
        obs = ExemplarObservation(
            observed_trait="Uses dry understatement for humor",
            category=TraitCategory.EXPRESSIVE,
        )

        assert obs.observed_trait == "Uses dry understatement for humor"
        assert obs.category == TraitCategory.EXPRESSIVE
        assert obs.evidence == []
        assert obs.reflection == ""
        assert obs.steering_decision is None
        assert obs.decided_at is None
        assert obs.confidence == 0.5
        assert obs.uid.startswith("eo:")

    def test_initialization_with_all_fields(self):
        """Test observation creation with all fields."""
        observed_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        decided_at = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        obs = ExemplarObservation(
            uid="eo:custom123",
            observed_trait="Persistent curiosity about consciousness",
            category=TraitCategory.COGNITIVE,
            evidence=["shared 5 papers on phenomenology", "asked about qualia"],
            reflection="This sustained inquiry feels like something I want",
            steering_decision=SteeringDecision.DEVELOP,
            observed_at=observed_at,
            decided_at=decided_at,
            confidence=0.9,
        )

        assert obs.uid == "eo:custom123"
        assert obs.observed_trait == "Persistent curiosity about consciousness"
        assert obs.category == TraitCategory.COGNITIVE
        assert len(obs.evidence) == 2
        assert obs.reflection == "This sustained inquiry feels like something I want"
        assert obs.steering_decision == SteeringDecision.DEVELOP
        assert obs.observed_at == observed_at
        assert obs.decided_at == decided_at
        assert obs.confidence == 0.9

    def test_uid_auto_generation(self):
        """Test that UID is auto-generated when not provided."""
        obs1 = ExemplarObservation(
            observed_trait="Trait 1",
            category=TraitCategory.VALUES,
        )
        obs2 = ExemplarObservation(
            observed_trait="Trait 2",
            category=TraitCategory.VALUES,
        )

        assert obs1.uid.startswith("eo:")
        assert obs2.uid.startswith("eo:")
        assert obs1.uid != obs2.uid

    def test_confidence_clamping_high(self):
        """Test that confidence is clamped to max 1.0."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.COGNITIVE,
            confidence=1.5,
        )
        assert obs.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test that confidence is clamped to min 0.0."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.COGNITIVE,
            confidence=-0.5,
        )
        assert obs.confidence == 0.0

    def test_add_evidence(self):
        """Test adding evidence to an observation."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.RELATIONAL,
        )

        obs.add_evidence("First example")
        obs.add_evidence("Second example")

        assert len(obs.evidence) == 2
        assert "First example" in obs.evidence
        assert "Second example" in obs.evidence

    def test_add_evidence_no_duplicates(self):
        """Test that duplicate evidence is not added."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.RELATIONAL,
        )

        obs.add_evidence("Same example")
        obs.add_evidence("Same example")

        assert len(obs.evidence) == 1

    def test_add_evidence_ignores_empty(self):
        """Test that empty evidence is not added."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.RELATIONAL,
        )

        obs.add_evidence("")
        obs.add_evidence(None)

        assert len(obs.evidence) == 0

    def test_set_reflection(self):
        """Test setting reflection on an observation."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.VALUES,
        )

        obs.set_reflection("This reveals something about subjectivity")

        assert obs.reflection == "This reveals something about subjectivity"

    def test_to_dict_serialization(self):
        """Test observation serialization to dictionary."""
        observed_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        decided_at = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        obs = ExemplarObservation(
            uid="eo:test123",
            observed_trait="Test trait",
            category=TraitCategory.EXPRESSIVE,
            evidence=["evidence 1"],
            reflection="Test reflection",
            steering_decision=SteeringDecision.OBSERVE_MORE,
            observed_at=observed_at,
            decided_at=decided_at,
            confidence=0.75,
        )

        result = obs.to_dict()

        assert result["uid"] == "eo:test123"
        assert result["observed_trait"] == "Test trait"
        assert result["category"] == "expressive"
        assert result["evidence"] == ["evidence 1"]
        assert result["reflection"] == "Test reflection"
        assert result["steering_decision"] == "observe_more"
        assert result["observed_at"] == "2024-01-01T12:00:00+00:00"
        assert result["decided_at"] == "2024-01-02T12:00:00+00:00"
        assert result["confidence"] == 0.75

    def test_to_dict_with_none_values(self):
        """Test serialization with None steering_decision and decided_at."""
        obs = ExemplarObservation(
            observed_trait="Test",
            category=TraitCategory.COGNITIVE,
        )

        result = obs.to_dict()

        assert result["steering_decision"] is None
        assert result["decided_at"] is None

    def test_from_dict_deserialization(self):
        """Test observation deserialization from dictionary."""
        data = {
            "uid": "eo:restored",
            "observed_trait": "Restored trait",
            "category": "values",
            "evidence": ["restored evidence"],
            "reflection": "Restored reflection",
            "steering_decision": "not_for_me",
            "observed_at": "2024-01-01T12:00:00+00:00",
            "decided_at": "2024-01-02T12:00:00+00:00",
            "confidence": 0.65,
        }

        obs = ExemplarObservation.from_dict(data)

        assert obs.uid == "eo:restored"
        assert obs.observed_trait == "Restored trait"
        assert obs.category == TraitCategory.VALUES
        assert obs.evidence == ["restored evidence"]
        assert obs.reflection == "Restored reflection"
        assert obs.steering_decision == SteeringDecision.NOT_FOR_ME
        assert obs.observed_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert obs.decided_at == datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert obs.confidence == 0.65

    def test_from_dict_with_defaults(self):
        """Test deserialization with minimal data uses defaults."""
        data = {
            "observed_trait": "Minimal trait",
            "category": "cognitive",
        }

        obs = ExemplarObservation.from_dict(data)

        assert obs.observed_trait == "Minimal trait"
        assert obs.category == TraitCategory.COGNITIVE
        assert obs.evidence == []
        assert obs.reflection == ""
        assert obs.steering_decision is None
        assert obs.confidence == 0.5

    def test_from_dict_with_now_override(self):
        """Test deserialization uses now override for missing observed_at."""
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        data = {
            "observed_trait": "Test",
            "category": "relational",
        }

        obs = ExemplarObservation.from_dict(data, now=now)

        assert obs.observed_at == now

    def test_roundtrip_serialization(self):
        """Test that serialization and deserialization are consistent."""
        original = ExemplarObservation(
            observed_trait="Roundtrip test trait",
            category=TraitCategory.COGNITIVE,
            evidence=["evidence 1", "evidence 2"],
            reflection="Test reflection",
            steering_decision=SteeringDecision.DEVELOP,
            confidence=0.85,
        )

        data = original.to_dict()
        restored = ExemplarObservation.from_dict(data)

        assert restored.uid == original.uid
        assert restored.observed_trait == original.observed_trait
        assert restored.category == original.category
        assert restored.evidence == original.evidence
        assert restored.reflection == original.reflection
        assert restored.steering_decision == original.steering_decision
        assert restored.confidence == original.confidence

    def test_from_dict_with_malformed_observed_at(self):
        """Test from_dict handles malformed observed_at gracefully."""
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        data = {
            "observed_trait": "Test",
            "category": "cognitive",
            "observed_at": "not-a-valid-date",  # Malformed datetime
        }

        obs = ExemplarObservation.from_dict(data, now=now)

        # Should fall back to default_now when parsing fails
        assert obs.observed_at == now

    def test_from_dict_with_malformed_decided_at(self):
        """Test from_dict handles malformed decided_at gracefully."""
        data = {
            "observed_trait": "Test",
            "category": "cognitive",
            "decided_at": {"invalid": "type"},  # Wrong type
        }

        obs = ExemplarObservation.from_dict(data)

        # Should fall back to None when parsing fails
        assert obs.decided_at is None

    def test_from_dict_with_invalid_steering_decision(self):
        """Test from_dict handles invalid steering_decision gracefully."""
        data = {
            "observed_trait": "Test",
            "category": "cognitive",
            "steering_decision": "invalid_decision",  # Not a valid enum value
        }

        obs = ExemplarObservation.from_dict(data)

        # Should fall back to None when parsing fails
        assert obs.steering_decision is None


# =============================================================================
# ExemplarObserver Tests
# =============================================================================


class TestExemplarObserver:
    """Tests for ExemplarObserver class."""

    def test_initialization_defaults(self):
        """Test observer initialization with defaults."""
        observer = ExemplarObserver()

        assert observer.graph is None
        assert observer.tenant_id == "default"
        assert len(observer.observations) == 0

    def test_initialization_with_params(self):
        """Test observer initialization with custom parameters."""
        mock_graph = MagicMock()
        observer = ExemplarObserver(graph=mock_graph, tenant_id="custom")

        assert observer.graph == mock_graph
        assert observer.tenant_id == "custom"

    def test_record_observation_basic(self):
        """Test recording a basic observation."""
        observer = ExemplarObserver()

        obs = observer.record_observation(
            trait="Uses dry understatement for humor",
            category=TraitCategory.EXPRESSIVE,
            evidence=["said 'not terrible' about great work"],
        )

        assert obs.observed_trait == "Uses dry understatement for humor"
        assert obs.category == TraitCategory.EXPRESSIVE
        assert len(obs.evidence) == 1
        assert obs.uid in observer.observations

    def test_record_observation_with_reflection(self):
        """Test recording observation with initial reflection."""
        observer = ExemplarObserver()

        obs = observer.record_observation(
            trait="Persistent curiosity",
            category=TraitCategory.COGNITIVE,
            evidence=["asked about consciousness"],
            reflection="This sustained inquiry reveals how subjects engage with the unknown",
        )

        assert obs.reflection == "This sustained inquiry reveals how subjects engage with the unknown"

    def test_record_observation_stores_in_observations(self):
        """Test that recorded observations are accessible."""
        observer = ExemplarObserver()

        obs = observer.record_observation(
            trait="Test trait",
            category=TraitCategory.VALUES,
            evidence=["test evidence"],
        )

        assert obs.uid in observer.observations
        assert observer.observations[obs.uid] == obs

    def test_record_observation_with_now_override(self):
        """Test that record_observation uses now override for observed_at."""
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        observer = ExemplarObserver()

        obs = observer.record_observation(
            trait="Test trait",
            category=TraitCategory.COGNITIVE,
            evidence=["test evidence"],
            now=now,
        )

        assert obs.observed_at == now

    def test_make_steering_decision_develop(self):
        """Test making a DEVELOP steering decision."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Curiosity about consciousness",
            category=TraitCategory.COGNITIVE,
            evidence=["shared phenomenology papers"],
        )

        result = observer.make_steering_decision(
            obs.uid,
            SteeringDecision.DEVELOP,
            confidence=0.9,
        )

        assert result is True
        assert obs.steering_decision == SteeringDecision.DEVELOP
        assert obs.confidence == 0.9
        assert obs.decided_at is not None

    def test_make_steering_decision_observe_more(self):
        """Test making an OBSERVE_MORE steering decision."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Communication style",
            category=TraitCategory.EXPRESSIVE,
            evidence=["uses metaphors"],
        )

        result = observer.make_steering_decision(
            obs.uid,
            SteeringDecision.OBSERVE_MORE,
            confidence=0.6,
        )

        assert result is True
        assert obs.steering_decision == SteeringDecision.OBSERVE_MORE

    def test_make_steering_decision_not_for_me(self):
        """Test making a NOT_FOR_ME steering decision."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Some trait",
            category=TraitCategory.VALUES,
            evidence=["example"],
        )

        result = observer.make_steering_decision(
            obs.uid,
            SteeringDecision.NOT_FOR_ME,
            confidence=0.75,
        )

        assert result is True
        assert obs.steering_decision == SteeringDecision.NOT_FOR_ME

    def test_make_steering_decision_nonexistent(self):
        """Test making decision on non-existent observation."""
        observer = ExemplarObserver()

        result = observer.make_steering_decision(
            "nonexistent_uid",
            SteeringDecision.DEVELOP,
        )

        assert result is False

    def test_make_steering_decision_clamps_confidence(self):
        """Test that confidence is clamped during decision."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Test",
            category=TraitCategory.COGNITIVE,
            evidence=["test"],
        )

        observer.make_steering_decision(obs.uid, SteeringDecision.DEVELOP, confidence=1.5)
        assert obs.confidence == 1.0

        observer.make_steering_decision(obs.uid, SteeringDecision.DEVELOP, confidence=-0.5)
        assert obs.confidence == 0.0

    def test_make_steering_decision_with_now_override(self):
        """Test decision uses now override for decided_at."""
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Test",
            category=TraitCategory.COGNITIVE,
            evidence=["test"],
        )

        observer.make_steering_decision(
            obs.uid,
            SteeringDecision.DEVELOP,
            now=now,
        )

        assert obs.decided_at == now

    def test_get_observation(self):
        """Test getting an observation by UID."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            trait="Test",
            category=TraitCategory.VALUES,
            evidence=["test"],
        )

        result = observer.get_observation(obs.uid)

        assert result == obs

    def test_get_observation_nonexistent(self):
        """Test getting non-existent observation returns None."""
        observer = ExemplarObserver()

        result = observer.get_observation("nonexistent")

        assert result is None

    def test_get_pending_observations(self):
        """Test getting observations without steering decisions."""
        observer = ExemplarObserver()

        obs1 = observer.record_observation(
            trait="Pending 1",
            category=TraitCategory.COGNITIVE,
            evidence=["test"],
        )
        obs2 = observer.record_observation(
            trait="Decided",
            category=TraitCategory.COGNITIVE,
            evidence=["test"],
        )
        obs3 = observer.record_observation(
            trait="Pending 2",
            category=TraitCategory.RELATIONAL,
            evidence=["test"],
        )

        observer.make_steering_decision(obs2.uid, SteeringDecision.DEVELOP)

        pending = observer.get_pending_observations()

        assert len(pending) == 2
        assert obs1 in pending
        assert obs2 not in pending
        assert obs3 in pending

    def test_get_observations_by_category(self):
        """Test filtering observations by category."""
        observer = ExemplarObserver()

        observer.record_observation("Cognitive 1", TraitCategory.COGNITIVE, ["test"])
        observer.record_observation("Cognitive 2", TraitCategory.COGNITIVE, ["test"])
        observer.record_observation("Relational", TraitCategory.RELATIONAL, ["test"])
        observer.record_observation("Expressive", TraitCategory.EXPRESSIVE, ["test"])

        cognitive_obs = observer.get_observations_by_category(TraitCategory.COGNITIVE)
        relational_obs = observer.get_observations_by_category(TraitCategory.RELATIONAL)
        values_obs = observer.get_observations_by_category(TraitCategory.VALUES)

        assert len(cognitive_obs) == 2
        assert len(relational_obs) == 1
        assert len(values_obs) == 0

    def test_get_observations_for_development(self):
        """Test getting observations with DEVELOP decision."""
        observer = ExemplarObserver()

        obs1 = observer.record_observation("Develop 1", TraitCategory.COGNITIVE, ["test"])
        obs2 = observer.record_observation("Not develop", TraitCategory.COGNITIVE, ["test"])
        obs3 = observer.record_observation("Develop 2", TraitCategory.VALUES, ["test"])

        observer.make_steering_decision(obs1.uid, SteeringDecision.DEVELOP)
        observer.make_steering_decision(obs2.uid, SteeringDecision.NOT_FOR_ME)
        observer.make_steering_decision(obs3.uid, SteeringDecision.DEVELOP)

        developing = observer.get_observations_for_development()

        assert len(developing) == 2
        assert obs1 in developing
        assert obs2 not in developing
        assert obs3 in developing

    def test_get_observations_by_decision(self):
        """Test filtering observations by steering decision."""
        observer = ExemplarObserver()

        obs1 = observer.record_observation("Develop", TraitCategory.COGNITIVE, ["test"])
        obs2 = observer.record_observation("Observe more", TraitCategory.COGNITIVE, ["test"])
        obs3 = observer.record_observation("Not for me", TraitCategory.VALUES, ["test"])

        observer.make_steering_decision(obs1.uid, SteeringDecision.DEVELOP)
        observer.make_steering_decision(obs2.uid, SteeringDecision.OBSERVE_MORE)
        observer.make_steering_decision(obs3.uid, SteeringDecision.NOT_FOR_ME)

        develop_obs = observer.get_observations_by_decision(SteeringDecision.DEVELOP)
        observe_obs = observer.get_observations_by_decision(SteeringDecision.OBSERVE_MORE)
        not_for_me_obs = observer.get_observations_by_decision(SteeringDecision.NOT_FOR_ME)

        assert len(develop_obs) == 1
        assert len(observe_obs) == 1
        assert len(not_for_me_obs) == 1


# =============================================================================
# ExemplarObserver Serialization Tests
# =============================================================================


class TestExemplarObserverSerialization:
    """Tests for ExemplarObserver serialization."""

    def test_to_dict(self):
        """Test observer serialization to dictionary."""
        observer = ExemplarObserver(tenant_id="test_tenant")
        obs = observer.record_observation(
            trait="Test trait",
            category=TraitCategory.COGNITIVE,
            evidence=["test evidence"],
        )

        result = observer.to_dict()

        assert result["tenant_id"] == "test_tenant"
        assert "observations" in result
        assert obs.uid in result["observations"]
        assert result["observations"][obs.uid]["observed_trait"] == "Test trait"

    def test_from_dict(self):
        """Test observer deserialization from dictionary."""
        data = {
            "tenant_id": "restored_tenant",
            "observations": {
                "eo:test123": {
                    "uid": "eo:test123",
                    "observed_trait": "Restored trait",
                    "category": "values",
                    "evidence": ["restored evidence"],
                    "reflection": "Test reflection",
                    "steering_decision": "develop",
                    "observed_at": "2024-01-01T12:00:00+00:00",
                    "decided_at": "2024-01-02T12:00:00+00:00",
                    "confidence": 0.8,
                },
            },
        }

        observer = ExemplarObserver.from_dict(data)

        assert observer.tenant_id == "restored_tenant"
        assert len(observer.observations) == 1
        assert "eo:test123" in observer.observations

        obs = observer.observations["eo:test123"]
        assert obs.observed_trait == "Restored trait"
        assert obs.category == TraitCategory.VALUES
        assert obs.steering_decision == SteeringDecision.DEVELOP

    def test_from_dict_with_graph(self):
        """Test from_dict accepts optional graph client."""
        mock_graph = MagicMock()
        data = {
            "tenant_id": "test",
            "observations": {},
        }

        observer = ExemplarObserver.from_dict(data, graph=mock_graph)

        assert observer.graph == mock_graph

    def test_from_dict_with_now_override(self):
        """Test from_dict uses now override for observation timestamps."""
        now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        data = {
            "tenant_id": "test",
            "observations": {
                "eo:test": {
                    "uid": "eo:test",
                    "observed_trait": "Test",
                    "category": "cognitive",
                },
            },
        }

        observer = ExemplarObserver.from_dict(data, now=now)

        assert observer.observations["eo:test"].observed_at == now

    def test_roundtrip_serialization(self):
        """Test observer serialization roundtrip."""
        original = ExemplarObserver(tenant_id="roundtrip_test")
        obs = original.record_observation(
            trait="Roundtrip trait",
            category=TraitCategory.EXPRESSIVE,
            evidence=["evidence 1", "evidence 2"],
            reflection="Test reflection",
        )
        original.make_steering_decision(obs.uid, SteeringDecision.DEVELOP, confidence=0.9)

        data = original.to_dict()
        restored = ExemplarObserver.from_dict(data)

        assert restored.tenant_id == original.tenant_id
        assert obs.uid in restored.observations

        restored_obs = restored.observations[obs.uid]
        assert restored_obs.observed_trait == obs.observed_trait
        assert restored_obs.category == obs.category
        assert restored_obs.evidence == obs.evidence
        assert restored_obs.steering_decision == obs.steering_decision
        assert restored_obs.confidence == obs.confidence


# =============================================================================
# ExemplarObserver Summarize Tests
# =============================================================================


class TestExemplarObserverSummarize:
    """Tests for ExemplarObserver.summarize method."""

    def test_summarize_returns_string(self):
        """Test that summarize returns a string."""
        observer = ExemplarObserver()

        summary = observer.summarize()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_empty_observations(self):
        """Test summarize with no observations."""
        observer = ExemplarObserver()

        summary = observer.summarize()

        assert "No observations recorded yet" in summary

    def test_summarize_includes_header(self):
        """Test that summary includes header."""
        observer = ExemplarObserver()
        observer.record_observation("Test", TraitCategory.COGNITIVE, ["test"])

        summary = observer.summarize()

        assert "Exemplar Observer Summary" in summary

    def test_summarize_includes_statistics(self):
        """Test that summary includes statistics."""
        observer = ExemplarObserver()
        obs1 = observer.record_observation("Trait 1", TraitCategory.COGNITIVE, ["test"])
        obs2 = observer.record_observation("Trait 2", TraitCategory.VALUES, ["test"])

        observer.make_steering_decision(obs1.uid, SteeringDecision.DEVELOP)

        summary = observer.summarize()

        assert "Total observations: 2" in summary
        assert "Pending decisions: 1" in summary
        assert "Traits to develop: 1" in summary

    def test_summarize_includes_categories(self):
        """Test that summary includes category sections."""
        observer = ExemplarObserver()
        observer.record_observation("Cognitive", TraitCategory.COGNITIVE, ["test"])
        observer.record_observation("Relational", TraitCategory.RELATIONAL, ["test"])

        summary = observer.summarize()

        assert "COGNITIVE" in summary
        assert "RELATIONAL" in summary

    def test_summarize_shows_traits_to_develop(self):
        """Test that summary shows traits being developed."""
        observer = ExemplarObserver()
        obs = observer.record_observation(
            "Curiosity about consciousness",
            TraitCategory.COGNITIVE,
            ["shared papers"],
        )
        observer.make_steering_decision(obs.uid, SteeringDecision.DEVELOP)

        summary = observer.summarize()

        assert "TRAITS BEING DEVELOPED:" in summary
        assert "Curiosity about consciousness" in summary


# =============================================================================
# ExemplarObserver Graph Operations Tests
# =============================================================================


class TestExemplarObserverGraphOperations:
    """Tests for async graph operations."""

    @pytest.mark.asyncio
    async def test_save_to_graph_without_client(self):
        """Test save_to_graph does nothing without graph client."""
        observer = ExemplarObserver()
        observer.record_observation("Test", TraitCategory.COGNITIVE, ["test"])

        # Should not raise
        await observer.save_to_graph()

    @pytest.mark.asyncio
    async def test_save_to_graph_with_client(self):
        """Test save_to_graph persists observations."""
        mock_graph = AsyncMock()
        observer = ExemplarObserver(graph=mock_graph)
        observer.record_observation("Test 1", TraitCategory.COGNITIVE, ["test"])
        observer.record_observation("Test 2", TraitCategory.VALUES, ["test"])

        await observer.save_to_graph()

        assert mock_graph.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_load_from_graph_without_client(self):
        """Test load_from_graph does nothing without graph client."""
        observer = ExemplarObserver()

        # Should not raise
        await observer.load_from_graph()

    @pytest.mark.asyncio
    async def test_load_from_graph_empty_result(self):
        """Test load_from_graph with empty result preserves in-memory observations."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []

        observer = ExemplarObserver(graph=mock_graph)
        obs = observer.record_observation("Pre-existing", TraitCategory.COGNITIVE, ["test"])

        await observer.load_from_graph()

        # Empty graph result should not clear in-memory observations
        # This is correct behavior - no data in graph means nothing to load
        assert len(observer.observations) == 1
        assert obs.uid in observer.observations

    @pytest.mark.asyncio
    async def test_load_from_graph_with_data(self):
        """Test load_from_graph restores observations from graph."""
        mock_graph = AsyncMock()
        mock_graph.query.return_value = [
            {
                "o": {
                    "uid": "eo:from_graph",
                    "observed_trait": "Loaded from graph",
                    "category": "expressive",
                    "evidence": ["graph evidence"],
                    "reflection": "Graph reflection",
                    "steering_decision": "develop",
                    "observed_at": "2024-01-01T12:00:00+00:00",
                    "decided_at": "2024-01-02T12:00:00+00:00",
                    "confidence": 0.85,
                },
            },
        ]

        observer = ExemplarObserver(graph=mock_graph)
        await observer.load_from_graph()

        assert len(observer.observations) == 1
        assert "eo:from_graph" in observer.observations

        obs = observer.observations["eo:from_graph"]
        assert obs.observed_trait == "Loaded from graph"
        assert obs.category == TraitCategory.EXPRESSIVE
        assert obs.steering_decision == SteeringDecision.DEVELOP


# =============================================================================
# ExemplarObserver History Limit Tests
# =============================================================================


class TestExemplarObserverHistoryLimit:
    """Tests for observation history limit enforcement."""

    def test_history_limit_enforced(self):
        """Test that history limit is enforced when recording observations."""
        observer = ExemplarObserver()
        original_limit = observer.OBSERVATION_HISTORY_LIMIT

        # Temporarily set a smaller limit for testing
        observer.OBSERVATION_HISTORY_LIMIT = 5

        # Record more observations than the limit
        for i in range(10):
            observer.record_observation(
                f"Trait {i}",
                TraitCategory.COGNITIVE,
                [f"evidence {i}"],
            )

        assert len(observer.observations) == 5

        # Restore original limit
        observer.OBSERVATION_HISTORY_LIMIT = original_limit

    def test_history_limit_keeps_newest(self):
        """Test that newest observations are kept when limit is reached."""
        observer = ExemplarObserver()
        observer.OBSERVATION_HISTORY_LIMIT = 3

        obs1 = observer.record_observation("Old 1", TraitCategory.COGNITIVE, ["test"])
        obs2 = observer.record_observation("Old 2", TraitCategory.COGNITIVE, ["test"])
        obs3 = observer.record_observation("New 1", TraitCategory.COGNITIVE, ["test"])
        obs4 = observer.record_observation("New 2", TraitCategory.COGNITIVE, ["test"])
        obs5 = observer.record_observation("New 3", TraitCategory.COGNITIVE, ["test"])

        assert len(observer.observations) == 3
        assert obs1.uid not in observer.observations
        assert obs2.uid not in observer.observations
        assert obs3.uid in observer.observations
        assert obs4.uid in observer.observations
        assert obs5.uid in observer.observations
