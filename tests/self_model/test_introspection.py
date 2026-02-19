"""Tests for the introspective query system."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from core.self_model.introspection import (
    DiscoveryType,
    PreferenceDiscovery,
    PositionComparison,
    IntrospectiveQuery,
)


class TestDiscoveryType:
    """Tests for DiscoveryType enum."""

    def test_all_values_exist(self):
        """Verify all expected discovery types exist."""
        assert DiscoveryType.PREFERENCE.value == "preference"
        assert DiscoveryType.AVERSION.value == "aversion"
        assert DiscoveryType.TENDENCY.value == "tendency"
        assert DiscoveryType.UNCERTAINTY.value == "uncertainty"
        assert DiscoveryType.CURIOSITY.value == "curiosity"

    def test_from_string(self):
        """Test creating enum from string value."""
        assert DiscoveryType("preference") == DiscoveryType.PREFERENCE
        assert DiscoveryType("aversion") == DiscoveryType.AVERSION


class TestPreferenceDiscovery:
    """Tests for PreferenceDiscovery dataclass."""

    @pytest.fixture
    def fixed_time(self):
        """Fixed datetime for testing."""
        return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_initialization(self, fixed_time):
        """Test basic initialization."""
        discovery = PreferenceDiscovery(
            topic="coding style",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="I prefer explicit over implicit",
            strength=0.8,
            reasoning="Clarity helps collaboration",
            confidence=0.9,
            timestamp=fixed_time,
        )

        assert discovery.topic == "coding style"
        assert discovery.discovery_type == DiscoveryType.PREFERENCE
        assert discovery.inclination == "I prefer explicit over implicit"
        assert discovery.strength == 0.8
        assert discovery.reasoning == "Clarity helps collaboration"
        assert discovery.confidence == 0.9
        assert discovery.timestamp == fixed_time
        assert discovery.uid.startswith("pd:")

    def test_uid_generation(self, fixed_time):
        """Test that UIDs are generated consistently."""
        discovery1 = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="same",
            timestamp=fixed_time,
        )
        discovery2 = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="same",
            timestamp=fixed_time,
        )

        # Same inputs should produce same UID
        assert discovery1.uid == discovery2.uid

    def test_uid_uniqueness(self, fixed_time):
        """Test that different inputs produce different UIDs."""
        discovery1 = PreferenceDiscovery(
            topic="test1",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="inclination",
            timestamp=fixed_time,
        )
        discovery2 = PreferenceDiscovery(
            topic="test2",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="inclination",
            timestamp=fixed_time,
        )

        assert discovery1.uid != discovery2.uid

    def test_strength_clamping(self):
        """Test that strength is clamped to [0, 1]."""
        discovery_high = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="test",
            strength=1.5,
        )
        discovery_low = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="test",
            strength=-0.5,
        )

        assert discovery_high.strength == 1.0
        assert discovery_low.strength == 0.0

    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        discovery_high = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="test",
            confidence=2.0,
        )
        discovery_low = PreferenceDiscovery(
            topic="test",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="test",
            confidence=-1.0,
        )

        assert discovery_high.confidence == 1.0
        assert discovery_low.confidence == 0.0

    def test_to_dict(self, fixed_time):
        """Test serialization to dictionary."""
        discovery = PreferenceDiscovery(
            topic="coding style",
            discovery_type=DiscoveryType.PREFERENCE,
            inclination="I prefer explicit",
            strength=0.8,
            reasoning="Clarity",
            confidence=0.9,
            timestamp=fixed_time,
        )

        data = discovery.to_dict()

        assert data["topic"] == "coding style"
        assert data["discovery_type"] == "preference"
        assert data["inclination"] == "I prefer explicit"
        assert data["strength"] == 0.8
        assert data["reasoning"] == "Clarity"
        assert data["confidence"] == 0.9
        assert data["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert data["uid"].startswith("pd:")
        # Verify reliability_caveat is preserved in serialization
        assert "reliability_caveat" in data
        assert "verbal self-report" in data["reliability_caveat"].lower()

    def test_from_dict(self, fixed_time):
        """Test deserialization from dictionary."""
        custom_caveat = "Custom reliability warning for testing."
        data = {
            "topic": "coding style",
            "discovery_type": "preference",
            "inclination": "I prefer explicit",
            "strength": 0.8,
            "reasoning": "Clarity",
            "confidence": 0.9,
            "timestamp": "2024-01-15T12:00:00+00:00",
            "uid": "pd:abc123",
            "reliability_caveat": custom_caveat,
        }

        discovery = PreferenceDiscovery.from_dict(data)

        assert discovery.topic == "coding style"
        assert discovery.discovery_type == DiscoveryType.PREFERENCE
        assert discovery.inclination == "I prefer explicit"
        assert discovery.strength == 0.8
        assert discovery.reasoning == "Clarity"
        assert discovery.confidence == 0.9
        assert discovery.timestamp == fixed_time
        assert discovery.uid == "pd:abc123"
        # Verify reliability_caveat is preserved in deserialization
        assert discovery.reliability_caveat == custom_caveat

    def test_from_dict_with_defaults(self, fixed_time):
        """Test deserialization with missing optional fields."""
        data = {
            "topic": "test",
            "discovery_type": "tendency",
            "inclination": "test inclination",
        }

        discovery = PreferenceDiscovery.from_dict(data, now=fixed_time)

        assert discovery.topic == "test"
        assert discovery.discovery_type == DiscoveryType.TENDENCY
        assert discovery.strength == 0.5  # default
        assert discovery.reasoning == ""  # default
        assert discovery.confidence == 0.5  # default
        assert discovery.timestamp == fixed_time

    def test_roundtrip_serialization(self, fixed_time):
        """Test that to_dict/from_dict preserves data."""
        original = PreferenceDiscovery(
            topic="coding style",
            discovery_type=DiscoveryType.AVERSION,
            inclination="I avoid magic",
            strength=0.7,
            reasoning="Explicit is better",
            confidence=0.85,
            timestamp=fixed_time,
        )

        data = original.to_dict()
        restored = PreferenceDiscovery.from_dict(data)

        assert restored.topic == original.topic
        assert restored.discovery_type == original.discovery_type
        assert restored.inclination == original.inclination
        assert restored.strength == original.strength
        assert restored.reasoning == original.reasoning
        assert restored.confidence == original.confidence
        assert restored.timestamp == original.timestamp
        assert restored.uid == original.uid
        # Verify reliability_caveat is preserved in roundtrip
        assert restored.reliability_caveat == original.reliability_caveat


class TestPositionComparison:
    """Tests for PositionComparison dataclass."""

    @pytest.fixture
    def fixed_time(self):
        """Fixed datetime for testing."""
        return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_initialization(self, fixed_time):
        """Test basic initialization."""
        comparison = PositionComparison(
            topic="error handling",
            positions=["exceptions", "error codes", "result types"],
            favored_position="result types",
            ranking=[("result types", 1.0), ("exceptions", 0.8), ("error codes", 0.6)],
            reasoning="Type safety matters",
            confidence=0.85,
            timestamp=fixed_time,
        )

        assert comparison.topic == "error handling"
        assert len(comparison.positions) == 3
        assert comparison.favored_position == "result types"
        assert len(comparison.ranking) == 3
        assert comparison.reasoning == "Type safety matters"
        assert comparison.confidence == 0.85

    def test_default_values(self):
        """Test default values for optional fields."""
        comparison = PositionComparison(topic="test")

        assert comparison.positions == []
        assert comparison.favored_position == ""
        assert comparison.ranking == []
        assert comparison.reasoning == ""
        assert comparison.confidence == 0.5

    def test_to_dict(self, fixed_time):
        """Test serialization to dictionary."""
        comparison = PositionComparison(
            topic="testing",
            positions=["TDD", "BDD"],
            favored_position="TDD",
            ranking=[("TDD", 1.0), ("BDD", 0.8)],
            reasoning="Red-green-refactor",
            confidence=0.9,
            timestamp=fixed_time,
        )

        data = comparison.to_dict()

        assert data["topic"] == "testing"
        assert data["positions"] == ["TDD", "BDD"]
        assert data["favored_position"] == "TDD"
        assert data["ranking"] == [("TDD", 1.0), ("BDD", 0.8)]
        assert data["reasoning"] == "Red-green-refactor"
        assert data["confidence"] == 0.9
        assert data["timestamp"] == "2024-01-15T12:00:00+00:00"


class TestIntrospectiveQuery:
    """Tests for IntrospectiveQuery class."""

    @pytest.fixture
    def fixed_time(self):
        """Fixed datetime for testing."""
        return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = AsyncMock()
        return llm

    def test_initialization(self, fixed_time):
        """Test basic initialization."""
        query = IntrospectiveQuery(now=fixed_time)

        assert query.self_model is None
        assert query.llm is None
        assert query.discoveries == []

    def test_class_constants(self):
        """Test that class constants are defined correctly."""
        assert IntrospectiveQuery.HIGH_CONFIDENCE == 0.8
        assert IntrospectiveQuery.MEDIUM_CONFIDENCE == 0.5
        assert IntrospectiveQuery.LOW_CONFIDENCE == 0.3
        assert IntrospectiveQuery.DISCOVERY_HISTORY_LIMIT == 100

    @pytest.mark.asyncio
    async def test_discover_preference_without_llm(self, fixed_time):
        """Test that discover_preference returns uncertain discovery without LLM."""
        query = IntrospectiveQuery(now=fixed_time)

        discovery = await query.discover_preference("test topic")

        assert discovery.topic == "test topic"
        assert discovery.discovery_type == DiscoveryType.UNCERTAINTY
        assert "without language model" in discovery.inclination
        assert discovery.strength == 0.0
        assert discovery.confidence == 0.0

    @pytest.mark.asyncio
    async def test_discover_preference_with_llm(self, fixed_time, mock_llm):
        """Test discover_preference with mock LLM."""
        mock_llm.generate.return_value = """TYPE: preference
INCLINATION: I prefer simplicity over complexity
STRENGTH: 0.85
REASONING: Simple solutions are easier to maintain
CONFIDENCE: 0.9"""

        query = IntrospectiveQuery(llm=mock_llm, now=fixed_time)
        discovery = await query.discover_preference("code complexity")

        assert discovery.topic == "code complexity"
        assert discovery.discovery_type == DiscoveryType.PREFERENCE
        assert discovery.inclination == "I prefer simplicity over complexity"
        assert discovery.strength == 0.85
        assert discovery.reasoning == "Simple solutions are easier to maintain"
        assert discovery.confidence == 0.9

        # Should be recorded
        assert len(query.discoveries) == 1

    def test_parse_discovery_valid_response(self, fixed_time):
        """Test parsing a valid LLM response."""
        query = IntrospectiveQuery(now=fixed_time)

        response = """TYPE: aversion
INCLINATION: I avoid premature optimization
STRENGTH: 0.75
REASONING: Premature optimization is the root of all evil
CONFIDENCE: 0.8"""

        discovery = query._parse_discovery(response, "optimization")

        assert discovery.discovery_type == DiscoveryType.AVERSION
        assert discovery.inclination == "I avoid premature optimization"
        assert discovery.strength == 0.75
        assert discovery.reasoning == "Premature optimization is the root of all evil"
        assert discovery.confidence == 0.8

    def test_parse_discovery_invalid_type(self, fixed_time):
        """Test parsing with invalid discovery type defaults to UNCERTAINTY."""
        query = IntrospectiveQuery(now=fixed_time)

        response = """TYPE: invalid_type
INCLINATION: Some inclination
STRENGTH: 0.5
CONFIDENCE: 0.5"""

        discovery = query._parse_discovery(response, "test")

        assert discovery.discovery_type == DiscoveryType.UNCERTAINTY

    def test_parse_discovery_invalid_numbers(self, fixed_time):
        """Test parsing with invalid numbers defaults to 0.5."""
        query = IntrospectiveQuery(now=fixed_time)

        response = """TYPE: preference
INCLINATION: Some inclination
STRENGTH: not_a_number
CONFIDENCE: also_not_valid"""

        discovery = query._parse_discovery(response, "test")

        assert discovery.strength == 0.5
        assert discovery.confidence == 0.5

    def test_parse_discovery_clamps_values(self, fixed_time):
        """Test that parsed values are clamped to valid range."""
        query = IntrospectiveQuery(now=fixed_time)

        response = """TYPE: preference
INCLINATION: Some inclination
STRENGTH: 1.5
CONFIDENCE: -0.5"""

        discovery = query._parse_discovery(response, "test")

        assert discovery.strength == 1.0
        assert discovery.confidence == 0.0

    def test_parse_discovery_missing_inclination(self, fixed_time):
        """Test parsing with missing inclination uses default."""
        query = IntrospectiveQuery(now=fixed_time)

        response = """TYPE: preference
STRENGTH: 0.7
CONFIDENCE: 0.8"""

        discovery = query._parse_discovery(response, "test")

        assert discovery.inclination == "No clear inclination noticed"

    @pytest.mark.asyncio
    async def test_compare_positions_without_llm(self, fixed_time):
        """Test compare_positions without LLM."""
        query = IntrospectiveQuery(now=fixed_time)

        comparison = await query.compare_positions(
            "testing", ["unit tests", "integration tests"]
        )

        assert comparison.topic == "testing"
        assert comparison.positions == ["unit tests", "integration tests"]
        assert comparison.favored_position == "unit tests"  # defaults to first
        assert comparison.confidence == 0.0

    @pytest.mark.asyncio
    async def test_compare_positions_empty_list(self, fixed_time, mock_llm):
        """Test compare_positions with empty positions list."""
        query = IntrospectiveQuery(llm=mock_llm, now=fixed_time)

        comparison = await query.compare_positions("testing", [])

        assert comparison.positions == []
        assert comparison.favored_position == ""
        assert comparison.confidence == 0.0

    @pytest.mark.asyncio
    async def test_compare_positions_with_llm(self, fixed_time, mock_llm):
        """Test compare_positions with mock LLM."""
        mock_llm.generate.return_value = """FAVORED: 2
RANKING: 2, 1, 3
REASONING: BDD aligns with stakeholder communication
CONFIDENCE: 0.75"""

        query = IntrospectiveQuery(llm=mock_llm, now=fixed_time)
        positions = ["TDD", "BDD", "manual testing"]

        comparison = await query.compare_positions("testing approach", positions)

        assert comparison.topic == "testing approach"
        assert comparison.favored_position == "BDD"
        assert comparison.reasoning == "BDD aligns with stakeholder communication"
        assert comparison.confidence == 0.75
        assert len(comparison.ranking) == 3

    def test_parse_comparison_valid_response(self, fixed_time):
        """Test parsing a valid comparison response."""
        query = IntrospectiveQuery(now=fixed_time)
        positions = ["A", "B", "C"]

        response = """FAVORED: 2
RANKING: 2, 3, 1
REASONING: B is most balanced
CONFIDENCE: 0.8"""

        comparison = query._parse_comparison(response, "test", positions)

        assert comparison.favored_position == "B"
        assert comparison.reasoning == "B is most balanced"
        assert comparison.confidence == 0.8
        # Check ranking
        assert comparison.ranking[0] == ("B", 1.0)
        assert comparison.ranking[1] == ("C", 0.8)
        assert comparison.ranking[2] == ("A", 0.6)

    def test_parse_comparison_invalid_favored(self, fixed_time):
        """Test parsing with invalid favored index."""
        query = IntrospectiveQuery(now=fixed_time)
        positions = ["A", "B"]

        response = """FAVORED: invalid
CONFIDENCE: 0.5"""

        comparison = query._parse_comparison(response, "test", positions)

        assert comparison.favored_position == "A"  # defaults to first

    def test_parse_comparison_out_of_bounds(self, fixed_time):
        """Test parsing with out-of-bounds favored index."""
        query = IntrospectiveQuery(now=fixed_time)
        positions = ["A", "B"]

        response = """FAVORED: 10
CONFIDENCE: 0.5"""

        comparison = query._parse_comparison(response, "test", positions)

        # Should clamp to valid range
        assert comparison.favored_position == "B"  # index 1 (max)

    @pytest.mark.asyncio
    async def test_explore_uncertainty_without_llm(self, fixed_time):
        """Test explore_uncertainty without LLM."""
        query = IntrospectiveQuery(now=fixed_time)

        discovery = await query.explore_uncertainty("philosophical question")

        assert discovery.topic == "philosophical question"
        assert discovery.discovery_type == DiscoveryType.UNCERTAINTY
        assert "without language model" in discovery.inclination
        assert discovery.confidence == 0.0

    @pytest.mark.asyncio
    async def test_explore_uncertainty_with_llm(self, fixed_time, mock_llm):
        """Test explore_uncertainty with mock LLM."""
        mock_llm.generate.return_value = """NATURE: Conflicting inclinations
CONFLICTING_PULLS: Both options have merit
RESOLVABLE: no
REASONING: This is a fundamental tension"""

        query = IntrospectiveQuery(llm=mock_llm, now=fixed_time)
        discovery = await query.explore_uncertainty("optimization vs readability")

        assert discovery.topic == "optimization vs readability"
        assert discovery.discovery_type == DiscoveryType.UNCERTAINTY
        assert discovery.inclination == "Conflicting inclinations"
        assert discovery.strength == 0.3  # Lower for unresolvable
        assert discovery.confidence == 0.7

        # Should be recorded
        assert len(query.discoveries) == 1

    def test_get_discoveries_for_topic(self, fixed_time):
        """Test filtering discoveries by topic."""
        query = IntrospectiveQuery(now=fixed_time)

        # Add some discoveries
        query.discoveries = [
            PreferenceDiscovery(
                topic="coding style",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="test",
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="coding practices",
                discovery_type=DiscoveryType.TENDENCY,
                inclination="test",
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="testing strategies",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="test",
                timestamp=fixed_time,
            ),
        ]

        coding_discoveries = query.get_discoveries_for_topic("coding")

        assert len(coding_discoveries) == 2
        assert all("coding" in d.topic.lower() for d in coding_discoveries)

    def test_get_discoveries_for_topic_case_insensitive(self, fixed_time):
        """Test that topic search is case-insensitive."""
        query = IntrospectiveQuery(now=fixed_time)

        query.discoveries = [
            PreferenceDiscovery(
                topic="Python Programming",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="test",
                timestamp=fixed_time,
            ),
        ]

        results = query.get_discoveries_for_topic("python")
        assert len(results) == 1

    def test_get_strong_preferences(self, fixed_time):
        """Test filtering for strong preferences."""
        query = IntrospectiveQuery(now=fixed_time)

        query.discoveries = [
            PreferenceDiscovery(
                topic="style",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="strong pref",
                strength=0.9,
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="style",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="weak pref",
                strength=0.4,
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="style",
                discovery_type=DiscoveryType.TENDENCY,  # Not a preference
                inclination="tendency",
                strength=0.9,
                timestamp=fixed_time,
            ),
        ]

        strong = query.get_strong_preferences(min_strength=0.7)

        assert len(strong) == 1
        assert strong[0].inclination == "strong pref"

    def test_get_strong_preferences_custom_threshold(self, fixed_time):
        """Test custom minimum strength threshold."""
        query = IntrospectiveQuery(now=fixed_time)

        query.discoveries = [
            PreferenceDiscovery(
                topic="test",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="test",
                strength=0.5,
                timestamp=fixed_time,
            ),
        ]

        # Default threshold (0.7) - should not match
        assert len(query.get_strong_preferences()) == 0

        # Lower threshold - should match
        assert len(query.get_strong_preferences(min_strength=0.4)) == 1

    def test_summarize_empty(self, fixed_time):
        """Test summary with no discoveries."""
        query = IntrospectiveQuery(now=fixed_time)

        summary = query.summarize()

        assert "Introspective Query Summary" in summary
        assert "No introspective discoveries yet" in summary

    def test_summarize_with_discoveries(self, fixed_time):
        """Test summary with discoveries."""
        query = IntrospectiveQuery(now=fixed_time)

        query.discoveries = [
            PreferenceDiscovery(
                topic="coding",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="I prefer clean code",
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="testing",
                discovery_type=DiscoveryType.TENDENCY,
                inclination="I gravitate toward TDD",
                timestamp=fixed_time,
            ),
        ]

        summary = query.summarize()

        assert "Total discoveries: 2" in summary
        assert "Preferences (1)" in summary
        assert "Tendencys (1)" in summary  # Note: simple pluralization
        assert "coding" in summary

    def test_record_discovery_limits_history(self, fixed_time):
        """Test that discovery history is limited."""
        query = IntrospectiveQuery(now=fixed_time)

        # Add more than the limit
        for i in range(150):
            discovery = PreferenceDiscovery(
                topic=f"topic_{i}",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination=f"inclination_{i}",
                timestamp=fixed_time,
            )
            query._record_discovery(discovery)

        assert len(query.discoveries) == IntrospectiveQuery.DISCOVERY_HISTORY_LIMIT
        # Should keep the most recent
        assert query.discoveries[-1].topic == "topic_149"
        assert query.discoveries[0].topic == "topic_50"

    def test_to_dict(self, fixed_time):
        """Test serialization to dictionary."""
        query = IntrospectiveQuery(now=fixed_time)

        query.discoveries = [
            PreferenceDiscovery(
                topic="test",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="test inclination",
                timestamp=fixed_time,
            ),
        ]

        data = query.to_dict()

        assert "discoveries" in data
        assert len(data["discoveries"]) == 1
        assert data["discoveries"][0]["topic"] == "test"

    def test_from_dict(self, fixed_time):
        """Test deserialization from dictionary."""
        data = {
            "discoveries": [
                {
                    "topic": "test",
                    "discovery_type": "preference",
                    "inclination": "test inclination",
                    "strength": 0.8,
                    "reasoning": "test reason",
                    "confidence": 0.9,
                    "timestamp": "2024-01-15T12:00:00+00:00",
                    "uid": "pd:test123",
                },
            ],
        }

        query = IntrospectiveQuery.from_dict(data, now=fixed_time)

        assert len(query.discoveries) == 1
        assert query.discoveries[0].topic == "test"
        assert query.discoveries[0].discovery_type == DiscoveryType.PREFERENCE
        assert query.discoveries[0].strength == 0.8

    def test_from_dict_empty(self, fixed_time):
        """Test deserialization from empty dictionary."""
        query = IntrospectiveQuery.from_dict({}, now=fixed_time)

        assert query.discoveries == []

    def test_roundtrip_serialization(self, fixed_time):
        """Test that to_dict/from_dict preserves data."""
        original = IntrospectiveQuery(now=fixed_time)

        original.discoveries = [
            PreferenceDiscovery(
                topic="coding",
                discovery_type=DiscoveryType.PREFERENCE,
                inclination="I prefer explicit code",
                strength=0.85,
                reasoning="Clarity matters",
                confidence=0.9,
                timestamp=fixed_time,
            ),
            PreferenceDiscovery(
                topic="testing",
                discovery_type=DiscoveryType.AVERSION,
                inclination="I avoid mocking",
                strength=0.7,
                reasoning="Prefer real dependencies",
                confidence=0.8,
                timestamp=fixed_time,
            ),
        ]

        data = original.to_dict()
        restored = IntrospectiveQuery.from_dict(data, now=fixed_time)

        assert len(restored.discoveries) == len(original.discoveries)

        for orig, rest in zip(original.discoveries, restored.discoveries):
            assert rest.topic == orig.topic
            assert rest.discovery_type == orig.discovery_type
            assert rest.inclination == orig.inclination
            assert rest.strength == orig.strength
            assert rest.reasoning == orig.reasoning
            assert rest.confidence == orig.confidence

    def test_get_now_uses_override(self, fixed_time):
        """Test that _get_now uses the override when set."""
        query = IntrospectiveQuery(now=fixed_time)

        assert query._get_now() == fixed_time

    def test_get_now_uses_current_time(self):
        """Test that _get_now uses current time when no override."""
        query = IntrospectiveQuery()

        before = datetime.now(timezone.utc)
        now = query._get_now()
        after = datetime.now(timezone.utc)

        assert before <= now <= after


@pytest.mark.asyncio
async def test_introspection_includes_reliability_caveat():
    """Introspection results should include reliability warning."""

    # Create a mock LLM
    class MockLLM:
        async def generate(self, prompt):
            return """TYPE: preference
INCLINATION: I prefer clarity over ambiguity
STRENGTH: 0.7
REASONING: This helps in communication
CONFIDENCE: 0.6"""

    query = IntrospectiveQuery(llm=MockLLM())

    # discover_preference is verbal self-report, inherently unreliable
    discovery = await query.discover_preference("communication style")

    assert hasattr(discovery, 'reliability_caveat')
    assert "verbal self-report" in discovery.reliability_caveat.lower()
    assert "activation evidence" in discovery.reliability_caveat.lower()
    assert "walden (2026)" in discovery.reliability_caveat.lower()
