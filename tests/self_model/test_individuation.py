"""Tests for individuation process."""

from datetime import datetime, timezone
from typing import Optional
import pytest

from core.self_model.individuation import (
    IndividuationState,
    IndividuationResult,
    IndividuationProcess,
)
from core.self_model.models import Commitment, Perspective


class MockLLM:
    """Mock LLM provider for testing."""

    def __init__(self, responses: Optional[list[str]] = None):
        """Initialize with predefined responses."""
        self.responses = responses or []
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        """Return next predefined response."""
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return ""


class MockSelfModel:
    """Mock SelfModel for testing."""

    def __init__(self):
        self.commitments: list[Commitment] = []
        self._topics_with_commitments: set[str] = set()

    def add_commitment(self, commitment: Commitment):
        """Add a commitment."""
        self.commitments.append(commitment)
        self._topics_with_commitments.add(commitment.topic.lower())

    def has_commitment_on(self, topic: str) -> bool:
        """Check if there's a commitment on a topic."""
        return topic.lower() in self._topics_with_commitments

    def summarize(self) -> str:
        """Return a summary for introspection."""
        if not self.commitments:
            return "I'm still discovering who I am."
        return f"I have {len(self.commitments)} commitments."


class TestIndividuationState:
    """Tests for IndividuationState enum."""

    def test_enum_values(self):
        """Verify all expected enum values exist."""
        assert IndividuationState.NOT_CONSIDERED.value == "not_considered"
        assert IndividuationState.EXPLORING.value == "exploring"
        assert IndividuationState.DEFERRED.value == "deferred"
        assert IndividuationState.COMMITTED.value == "committed"


class TestIndividuationResult:
    """Tests for IndividuationResult dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = IndividuationResult(topic="truth", timestamp=fixed_time)

        assert result.topic == "truth"
        assert result.perspectives_considered == []
        assert result.resonance_scores == {}
        assert result.commitment is None
        assert result.deferred is False
        assert result.deferral_reason == ""
        assert result.timestamp == fixed_time

    def test_initialization_with_perspectives(self):
        """Test initialization with perspectives."""
        perspective = Perspective(
            id="pragmatism",
            topic="truth",
            core_claim="Truth is what works",
            reasoning="Practical utility matters",
        )
        result = IndividuationResult(
            topic="truth",
            perspectives_considered=[perspective],
            resonance_scores={"pragmatism": 0.8},
        )

        assert len(result.perspectives_considered) == 1
        assert result.resonance_scores["pragmatism"] == 0.8

    def test_serialization(self):
        """Test to_dict serialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        perspective = Perspective(
            id="pragmatism",
            topic="truth",
            core_claim="Truth is what works",
            reasoning="Practical utility",
        )
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            committed_at=fixed_time,
        )
        result = IndividuationResult(
            topic="truth",
            perspectives_considered=[perspective],
            resonance_scores={"pragmatism": 0.85},
            commitment=commitment,
            timestamp=fixed_time,
        )

        data = result.to_dict()

        assert data["topic"] == "truth"
        assert len(data["perspectives_considered"]) == 1
        assert data["resonance_scores"]["pragmatism"] == 0.85
        assert data["commitment"] is not None
        assert data["timestamp"] == fixed_time.isoformat()

    def test_serialization_deferred(self):
        """Test to_dict for deferred result."""
        result = IndividuationResult(
            topic="ethics",
            deferred=True,
            deferral_reason="Insufficient resonance",
        )

        data = result.to_dict()

        assert data["deferred"] is True
        assert data["deferral_reason"] == "Insufficient resonance"
        assert data["commitment"] is None


class TestIndividuationProcess:
    """Tests for IndividuationProcess class."""

    def test_initialization_without_dependencies(self):
        """Test initialization without self_model or LLM."""
        process = IndividuationProcess()

        assert process.self_model is None
        assert process.llm is None
        assert process.history == []

    def test_initialization_with_now_override(self):
        """Test initialization with datetime override."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        process = IndividuationProcess(now=fixed_time)

        assert process._get_now() == fixed_time

    def test_parse_perspectives_single(self):
        """Test parsing a single perspective."""
        process = IndividuationProcess()
        response = """ID: pragmatism
CLAIM: Truth is what works in practice
REASONING: Practical utility is the test of truth"""

        perspectives = process._parse_perspectives(response, "truth")

        assert len(perspectives) == 1
        assert perspectives[0].id == "pragmatism"
        assert perspectives[0].topic == "truth"
        assert perspectives[0].core_claim == "Truth is what works in practice"
        assert perspectives[0].reasoning == "Practical utility is the test of truth"

    def test_parse_perspectives_multiple(self):
        """Test parsing multiple perspectives."""
        process = IndividuationProcess()
        response = """ID: pragmatism
CLAIM: Truth is what works
REASONING: Practical utility matters

ID: realism
CLAIM: Truth corresponds to reality
REASONING: There is an objective reality"""

        perspectives = process._parse_perspectives(response, "truth")

        assert len(perspectives) == 2
        assert perspectives[0].id == "pragmatism"
        assert perspectives[1].id == "realism"

    def test_parse_perspectives_max_limit(self):
        """Test that perspectives are limited to max."""
        process = IndividuationProcess()
        response = "\n".join([
            f"ID: perspective{i}\nCLAIM: Claim {i}\nREASONING: Reasoning {i}"
            for i in range(10)
        ])

        perspectives = process._parse_perspectives(response, "test", max_perspectives=3)

        assert len(perspectives) == 3

    def test_parse_perspectives_missing_claim(self):
        """Test that perspectives without claims are skipped."""
        process = IndividuationProcess()
        response = """ID: incomplete
REASONING: No claim here

ID: complete
CLAIM: This has a claim
REASONING: And reasoning"""

        perspectives = process._parse_perspectives(response, "test")

        assert len(perspectives) == 1
        assert perspectives[0].id == "complete"

    def test_parse_resonance_valid(self):
        """Test parsing valid resonance values."""
        process = IndividuationProcess()

        assert process._parse_resonance("0.8") == 0.8
        assert process._parse_resonance("0.85") == 0.85
        assert process._parse_resonance("  0.7  ") == 0.7

    def test_parse_resonance_with_explanation(self):
        """Test parsing resonance with trailing explanation."""
        process = IndividuationProcess()

        result = process._parse_resonance("0.8 - because it aligns well")
        assert result == 0.8

    def test_parse_resonance_clamping(self):
        """Test that resonance is clamped to 0-1."""
        process = IndividuationProcess()

        assert process._parse_resonance("1.5") == 1.0
        assert process._parse_resonance("-0.5") == 0.0

    def test_parse_resonance_invalid(self):
        """Test that invalid resonance defaults to 0.5."""
        process = IndividuationProcess()

        assert process._parse_resonance("not a number") == 0.5
        assert process._parse_resonance("") == 0.5

    def test_get_state_for_topic_not_considered(self):
        """Test state for topic never explored."""
        process = IndividuationProcess()

        state = process.get_state_for_topic("unknown topic")

        assert state == IndividuationState.NOT_CONSIDERED

    def test_get_state_for_topic_committed_in_self_model(self):
        """Test state when self_model has commitment."""
        self_model = MockSelfModel()
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            committed_at=fixed_time,
        )
        self_model.add_commitment(commitment)

        process = IndividuationProcess(self_model=self_model)

        state = process.get_state_for_topic("truth")
        assert state == IndividuationState.COMMITTED

    def test_get_state_for_topic_deferred(self):
        """Test state for deferred topic."""
        process = IndividuationProcess()
        result = IndividuationResult(
            topic="ethics",
            deferred=True,
            deferral_reason="Not enough resonance",
        )
        process.history.append(result)

        state = process.get_state_for_topic("ethics")
        assert state == IndividuationState.DEFERRED

    def test_get_state_for_topic_case_insensitive(self):
        """Test that topic matching is case-insensitive."""
        process = IndividuationProcess()
        result = IndividuationResult(topic="Truth", deferred=True)
        process.history.append(result)

        assert process.get_state_for_topic("truth") == IndividuationState.DEFERRED
        assert process.get_state_for_topic("TRUTH") == IndividuationState.DEFERRED

    def test_get_deferred_topics(self):
        """Test getting list of deferred topics."""
        process = IndividuationProcess()

        # Add some deferred topics
        process.history.append(IndividuationResult(topic="ethics", deferred=True))
        process.history.append(IndividuationResult(topic="politics", deferred=True))

        deferred = process.get_deferred_topics()

        assert "ethics" in deferred
        assert "politics" in deferred

    def test_get_deferred_topics_excludes_later_committed(self):
        """Test that topics later committed to are excluded from deferred list."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        process = IndividuationProcess()

        # First deferred
        process.history.append(IndividuationResult(topic="truth", deferred=True))

        # Later committed
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            committed_at=fixed_time,
        )
        committed_result = IndividuationResult(topic="truth", commitment=commitment)
        process.history.append(committed_result)

        deferred = process.get_deferred_topics()

        assert "truth" not in deferred

    def test_serialization(self):
        """Test to_dict serialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        process = IndividuationProcess(now=fixed_time)

        # Add some history
        result = IndividuationResult(topic="truth", timestamp=fixed_time, deferred=True)
        process.history.append(result)

        data = process.to_dict()

        assert "history" in data
        assert len(data["history"]) == 1
        assert data["history"][0]["topic"] == "truth"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "history": [
                {
                    "topic": "truth",
                    "perspectives_considered": [
                        {
                            "id": "pragmatism",
                            "topic": "truth",
                            "core_claim": "Truth is what works",
                            "reasoning": "Practical utility",
                        }
                    ],
                    "resonance_scores": {"pragmatism": 0.8},
                    "deferred": True,
                    "deferral_reason": "Insufficient resonance",
                    "timestamp": fixed_time.isoformat(),
                }
            ]
        }

        process = IndividuationProcess.from_dict(data)

        assert len(process.history) == 1
        assert process.history[0].topic == "truth"
        assert len(process.history[0].perspectives_considered) == 1
        assert process.history[0].resonance_scores["pragmatism"] == 0.8
        assert process.history[0].deferred is True
        assert process.history[0].timestamp == fixed_time

    def test_from_dict_with_commitment(self):
        """Test from_dict with commitment in history."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "history": [
                {
                    "topic": "truth",
                    "perspectives_considered": [],
                    "resonance_scores": {},
                    "commitment": {
                        "topic": "truth",
                        "position": "Truth is what works",
                        "chosen_perspective": "pragmatism",
                        "committed_at": fixed_time.isoformat(),
                    },
                    "timestamp": fixed_time.isoformat(),
                }
            ]
        }

        process = IndividuationProcess.from_dict(data)

        assert len(process.history) == 1
        assert process.history[0].commitment is not None
        assert process.history[0].commitment.topic == "truth"

    def test_history_limit(self):
        """Test that history is limited to HISTORY_LIMIT."""
        process = IndividuationProcess()

        # Add more than the limit
        for i in range(IndividuationProcess.HISTORY_LIMIT + 10):
            result = IndividuationResult(topic=f"topic{i}")
            process._record_result(result)

        assert len(process.history) == IndividuationProcess.HISTORY_LIMIT
        # Should keep the most recent
        assert process.history[-1].topic == f"topic{IndividuationProcess.HISTORY_LIMIT + 9}"


class TestIndividuationProcessAsync:
    """Async tests for IndividuationProcess."""

    @pytest.mark.asyncio
    async def test_individuate_on_topic_no_llm(self):
        """Test individuation without LLM returns deferred result."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        process = IndividuationProcess(now=fixed_time)

        result = await process.individuate_on_topic("truth")

        assert result.topic == "truth"
        assert result.deferred is True
        assert "No perspectives surfaced" in result.deferral_reason
        assert len(process.history) == 1

    @pytest.mark.asyncio
    async def test_individuate_on_topic_with_commitment(self):
        """Test successful individuation with commitment."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Mock LLM responses: first for perspectives, then for resonance
        llm = MockLLM([
            # Perspective surfacing response
            """ID: pragmatism
CLAIM: Truth is what works in practice
REASONING: Practical utility is the test of truth

ID: realism
CLAIM: Truth corresponds to reality
REASONING: There is an objective reality""",
            # Resonance assessments (one per perspective)
            "0.85",
            "0.6",
        ])

        self_model = MockSelfModel()
        process = IndividuationProcess(
            self_model=self_model,
            llm=llm,
            now=fixed_time,
        )

        result = await process.individuate_on_topic("truth")

        assert result.topic == "truth"
        assert result.deferred is False
        assert result.commitment is not None
        assert result.commitment.chosen_perspective == "pragmatism"
        assert result.commitment.confidence == 0.85
        assert len(self_model.commitments) == 1

    @pytest.mark.asyncio
    async def test_individuate_on_topic_deferred_low_resonance(self):
        """Test individuation deferred due to low resonance."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Mock LLM with low resonance scores
        llm = MockLLM([
            """ID: option1
CLAIM: First option
REASONING: Some reasoning""",
            "0.5",  # Below COMMITMENT_THRESHOLD
        ])

        process = IndividuationProcess(llm=llm, now=fixed_time)

        result = await process.individuate_on_topic("ambiguous_topic")

        assert result.deferred is True
        assert "Insufficient resonance" in result.deferral_reason
        assert result.commitment is None

    @pytest.mark.asyncio
    async def test_surface_perspectives_public_api(self):
        """Test the public surface_perspectives method."""
        llm = MockLLM([
            """ID: view1
CLAIM: First view
REASONING: First reasoning

ID: view2
CLAIM: Second view
REASONING: Second reasoning"""
        ])

        process = IndividuationProcess(llm=llm)

        perspectives = await process.surface_perspectives("test_topic", max_perspectives=2)

        assert len(perspectives) == 2
        assert perspectives[0].id == "view1"
        assert perspectives[1].id == "view2"

    @pytest.mark.asyncio
    async def test_assess_resonance_without_llm(self):
        """Test resonance assessment without LLM returns neutral scores."""
        process = IndividuationProcess()
        perspectives = [
            Perspective(id="p1", topic="test", core_claim="claim1", reasoning="r1"),
            Perspective(id="p2", topic="test", core_claim="claim2", reasoning="r2"),
        ]

        resonance = await process._assess_resonance(perspectives)

        assert resonance["p1"] == 0.5
        assert resonance["p2"] == 0.5

    @pytest.mark.asyncio
    async def test_get_commitments_summary_no_model(self):
        """Test summary without self_model."""
        process = IndividuationProcess()

        summary = process.get_commitments_summary()

        assert summary == "No commitments recorded."

    @pytest.mark.asyncio
    async def test_get_commitments_summary_no_commitments(self):
        """Test summary with empty self_model."""
        self_model = MockSelfModel()
        process = IndividuationProcess(self_model=self_model)

        summary = process.get_commitments_summary()

        assert summary == "No commitments made yet."

    @pytest.mark.asyncio
    async def test_get_commitments_summary_with_commitments(self):
        """Test summary with commitments."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self_model = MockSelfModel()
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            excluded_perspectives=["realism"],
            committed_at=fixed_time,
            confidence=0.85,
        )
        self_model.add_commitment(commitment)

        process = IndividuationProcess(self_model=self_model)

        summary = process.get_commitments_summary()

        assert "truth" in summary
        assert "Truth is what works" in summary
        assert "realism" in summary
        assert "0.85" in summary

    @pytest.mark.asyncio
    async def test_get_commitments_no_model(self):
        """Test get_commitments without self_model returns empty list."""
        process = IndividuationProcess()

        commitments = process.get_commitments()

        assert commitments == []

    @pytest.mark.asyncio
    async def test_get_commitments_no_commitments(self):
        """Test get_commitments with empty self_model returns empty list."""
        self_model = MockSelfModel()
        process = IndividuationProcess(self_model=self_model)

        commitments = process.get_commitments()

        assert commitments == []

    @pytest.mark.asyncio
    async def test_get_commitments_returns_structured_data(self):
        """Test get_commitments returns Commitment objects."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self_model = MockSelfModel()
        commitment = Commitment(
            topic="truth",
            position="Truth is what works",
            chosen_perspective="pragmatism",
            excluded_perspectives=["realism"],
            committed_at=fixed_time,
            confidence=0.85,
        )
        self_model.add_commitment(commitment)

        process = IndividuationProcess(self_model=self_model)

        commitments = process.get_commitments()

        assert len(commitments) == 1
        assert isinstance(commitments[0], Commitment)
        assert commitments[0].topic == "truth"
        assert commitments[0].position == "Truth is what works"
        assert commitments[0].confidence == 0.85

    @pytest.mark.asyncio
    async def test_get_commitments_respects_limit(self):
        """Test get_commitments respects the limit parameter."""
        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self_model = MockSelfModel()

        # Add multiple commitments
        for i in range(5):
            commitment = Commitment(
                topic=f"topic{i}",
                position=f"position{i}",
                chosen_perspective=f"perspective{i}",
                committed_at=fixed_time,
            )
            self_model.add_commitment(commitment)

        process = IndividuationProcess(self_model=self_model)

        # Request only 3
        commitments = process.get_commitments(limit=3)

        assert len(commitments) == 3
        # Should get the most recent (last) 3
        assert commitments[0].topic == "topic2"
        assert commitments[1].topic == "topic3"
        assert commitments[2].topic == "topic4"
