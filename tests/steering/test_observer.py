"""Tests for the steering observer.

Tests the SteeringObserver which tracks interactions and generates
contrastive pairs from observed successes and failures. The observer
is the first component in the SIMS loop:
    Observer -> Reflector -> Executor -> Validator

Uses mocks to avoid requiring actual GPU/model loading.
Skips torch-dependent tests if torch is not available.
"""

import pytest
from datetime import datetime, timezone

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


# =============================================================================
# ObservedInteraction Tests
# =============================================================================

class TestObservedInteraction:
    """Tests for ObservedInteraction dataclass."""

    def test_observed_interaction_creation(self):
        """ObservedInteraction should store prompt, response, and success."""
        from core.steering.observer import ObservedInteraction

        interaction = ObservedInteraction(
            prompt="What is 2+2?",
            response="The answer is 4.",
            success=True,
        )

        assert interaction.prompt == "What is 2+2?"
        assert interaction.response == "The answer is 4."
        assert interaction.success is True

    def test_observed_interaction_defaults(self):
        """ObservedInteraction should have sensible defaults."""
        from core.steering.observer import ObservedInteraction

        interaction = ObservedInteraction(
            prompt="test prompt",
            response="test response",
            success=False,
        )

        assert interaction.activations == {}
        assert interaction.surprise_score == 0.0
        assert isinstance(interaction.timestamp, datetime)
        assert interaction.metadata == {}

    def test_observed_interaction_with_surprise_score(self):
        """ObservedInteraction should store surprise_score."""
        from core.steering.observer import ObservedInteraction

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=True,
            surprise_score=0.85,
        )

        assert interaction.surprise_score == 0.85

    def test_observed_interaction_with_metadata(self):
        """ObservedInteraction should store arbitrary metadata."""
        from core.steering.observer import ObservedInteraction

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=True,
            metadata={"task_type": "math", "user_id": "user-123"},
        )

        assert interaction.metadata["task_type"] == "math"
        assert interaction.metadata["user_id"] == "user-123"

    def test_observed_interaction_timestamp_is_utc(self):
        """ObservedInteraction timestamp should be UTC."""
        from core.steering.observer import ObservedInteraction

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=True,
        )

        assert interaction.timestamp.tzinfo == timezone.utc

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
    def test_observed_interaction_with_activations(self):
        """ObservedInteraction should store layer activations."""
        from core.steering.observer import ObservedInteraction

        activations = {
            10: torch.randn(1, 5, 768),
            15: torch.randn(1, 5, 768),
        }

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=True,
            activations=activations,
        )

        assert 10 in interaction.activations
        assert 15 in interaction.activations
        assert interaction.activations[10].shape == (1, 5, 768)


# =============================================================================
# SteeringObserver Initialization Tests
# =============================================================================

class TestSteeringObserverInitialization:
    """Tests for SteeringObserver initialization."""

    def test_observer_initialization_default_buffer_size(self):
        """SteeringObserver should default to buffer_size=1000."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert observer._buffer_size == 1000
        assert observer.buffer.maxlen == 1000

    def test_observer_initialization_custom_buffer_size(self):
        """SteeringObserver should accept custom buffer_size."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver(buffer_size=500)

        assert observer._buffer_size == 500
        assert observer.buffer.maxlen == 500

    def test_observer_initialization_counters(self):
        """SteeringObserver should initialize counters to zero."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert observer.success_count == 0
        assert observer.failure_count == 0

    def test_observer_initialization_empty_buffer(self):
        """SteeringObserver should start with empty buffer."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert len(observer.buffer) == 0


# =============================================================================
# Recording Interactions Tests
# =============================================================================

class TestSteeringObserverRecord:
    """Tests for recording interactions."""

    @pytest.mark.asyncio
    async def test_observer_records_interaction(self):
        """record should add interaction to buffer."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        interaction = ObservedInteraction(
            prompt="Hello",
            response="Hi there!",
            success=True,
        )

        await observer.record(interaction)

        assert len(observer.buffer) == 1
        assert observer.buffer[0] is interaction

    @pytest.mark.asyncio
    async def test_observer_record_increments_success_count(self):
        """record should increment success_count for successful interactions."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=True,
        )

        await observer.record(interaction)

        assert observer.success_count == 1
        assert observer.failure_count == 0

    @pytest.mark.asyncio
    async def test_observer_record_increments_failure_count(self):
        """record should increment failure_count for failed interactions."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        interaction = ObservedInteraction(
            prompt="prompt",
            response="response",
            success=False,
        )

        await observer.record(interaction)

        assert observer.success_count == 0
        assert observer.failure_count == 1

    @pytest.mark.asyncio
    async def test_observer_record_multiple_interactions(self):
        """record should handle multiple interactions."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        for i in range(5):
            interaction = ObservedInteraction(
                prompt=f"prompt {i}",
                response=f"response {i}",
                success=(i % 2 == 0),
            )
            await observer.record(interaction)

        assert len(observer.buffer) == 5
        assert observer.success_count == 3  # 0, 2, 4 are successes
        assert observer.failure_count == 2  # 1, 3 are failures

    @pytest.mark.asyncio
    async def test_observer_buffer_evicts_old_interactions(self):
        """record should evict oldest when buffer is full."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver(buffer_size=3)

        for i in range(5):
            interaction = ObservedInteraction(
                prompt=f"prompt {i}",
                response=f"response {i}",
                success=True,
            )
            await observer.record(interaction)

        # Buffer should only contain last 3
        assert len(observer.buffer) == 3
        assert observer.buffer[0].prompt == "prompt 2"
        assert observer.buffer[1].prompt == "prompt 3"
        assert observer.buffer[2].prompt == "prompt 4"

        # But counters track all recorded interactions
        assert observer.success_count == 5


# =============================================================================
# Get Successes/Failures Tests
# =============================================================================

class TestSteeringObserverFiltering:
    """Tests for filtering interactions."""

    @pytest.mark.asyncio
    async def test_get_successes_returns_only_successes(self):
        """get_successes should return only successful interactions."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))
        await observer.record(ObservedInteraction("p3", "r3", success=True))

        successes = observer.get_successes()

        assert len(successes) == 2
        assert all(i.success for i in successes)
        assert successes[0].prompt == "p1"
        assert successes[1].prompt == "p3"

    @pytest.mark.asyncio
    async def test_get_failures_returns_only_failures(self):
        """get_failures should return only failed interactions."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))
        await observer.record(ObservedInteraction("p3", "r3", success=False))

        failures = observer.get_failures()

        assert len(failures) == 2
        assert all(not i.success for i in failures)
        assert failures[0].prompt == "p2"
        assert failures[1].prompt == "p3"

    def test_get_successes_empty_buffer(self):
        """get_successes should return empty list for empty buffer."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert observer.get_successes() == []

    def test_get_failures_empty_buffer(self):
        """get_failures should return empty list for empty buffer."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert observer.get_failures() == []


# =============================================================================
# High Surprise Tests
# =============================================================================

class TestSteeringObserverHighSurprise:
    """Tests for filtering by surprise score."""

    @pytest.mark.asyncio
    async def test_get_high_surprise_default_threshold(self):
        """get_high_surprise should use 0.7 threshold by default."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", True, surprise_score=0.5))
        await observer.record(ObservedInteraction("p2", "r2", True, surprise_score=0.8))
        await observer.record(ObservedInteraction("p3", "r3", False, surprise_score=0.9))

        high_surprise = observer.get_high_surprise()

        assert len(high_surprise) == 2
        assert high_surprise[0].surprise_score == 0.8
        assert high_surprise[1].surprise_score == 0.9

    @pytest.mark.asyncio
    async def test_get_high_surprise_custom_threshold(self):
        """get_high_surprise should accept custom threshold."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", True, surprise_score=0.3))
        await observer.record(ObservedInteraction("p2", "r2", True, surprise_score=0.5))
        await observer.record(ObservedInteraction("p3", "r3", True, surprise_score=0.7))

        high_surprise = observer.get_high_surprise(threshold=0.4)

        assert len(high_surprise) == 2
        assert high_surprise[0].surprise_score == 0.5
        assert high_surprise[1].surprise_score == 0.7

    @pytest.mark.asyncio
    async def test_get_high_surprise_includes_exact_threshold(self):
        """get_high_surprise should include interactions at exactly threshold."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", True, surprise_score=0.7))

        high_surprise = observer.get_high_surprise(threshold=0.7)

        assert len(high_surprise) == 1

    def test_get_high_surprise_empty_buffer(self):
        """get_high_surprise should return empty list for empty buffer."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        assert observer.get_high_surprise() == []


# =============================================================================
# Contrastive Pair Generation Tests
# =============================================================================

class TestSteeringObserverContrastivePairs:
    """Tests for contrastive pair generation."""

    @pytest.mark.asyncio
    async def test_observer_generates_contrastive_pairs(self):
        """generate_contrastive_pairs should create pairs from successes/failures."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("good prompt", "good response", success=True))
        await observer.record(ObservedInteraction("bad prompt", "bad response", success=False))

        pairs = await observer.generate_contrastive_pairs(behavior="test_behavior")

        assert len(pairs) == 1
        assert pairs[0].positive == "good prompt\ngood response"
        assert pairs[0].negative == "bad prompt\nbad response"
        assert pairs[0].behavior == "test_behavior"

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_default_behavior(self):
        """generate_contrastive_pairs should default to 'general' behavior."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))

        pairs = await observer.generate_contrastive_pairs()

        assert pairs[0].behavior == "general"

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_multiple(self):
        """generate_contrastive_pairs should pair multiple successes/failures."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        # 3 successes, 2 failures -> 2 pairs (limited by shorter list)
        await observer.record(ObservedInteraction("s1", "r1", success=True))
        await observer.record(ObservedInteraction("f1", "r2", success=False))
        await observer.record(ObservedInteraction("s2", "r3", success=True))
        await observer.record(ObservedInteraction("f2", "r4", success=False))
        await observer.record(ObservedInteraction("s3", "r5", success=True))

        pairs = await observer.generate_contrastive_pairs()

        assert len(pairs) == 2

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_returns_empty_without_successes(self):
        """generate_contrastive_pairs should return empty if no successes."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=False))
        await observer.record(ObservedInteraction("p2", "r2", success=False))

        pairs = await observer.generate_contrastive_pairs()

        assert pairs == []

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_returns_empty_without_failures(self):
        """generate_contrastive_pairs should return empty if no failures."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=True))

        pairs = await observer.generate_contrastive_pairs()

        assert pairs == []

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_respects_min_pairs(self):
        """generate_contrastive_pairs should return empty if below min_pairs."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        # Only 1 success, 1 failure - but we require min_pairs=2
        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))

        pairs = await observer.generate_contrastive_pairs(min_pairs=2)

        assert pairs == []

    @pytest.mark.asyncio
    async def test_generate_contrastive_pairs_satisfies_min_pairs(self):
        """generate_contrastive_pairs should return pairs when min_pairs satisfied."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("s1", "r1", success=True))
        await observer.record(ObservedInteraction("s2", "r2", success=True))
        await observer.record(ObservedInteraction("f1", "r3", success=False))
        await observer.record(ObservedInteraction("f2", "r4", success=False))

        pairs = await observer.generate_contrastive_pairs(min_pairs=2)

        assert len(pairs) == 2


# =============================================================================
# Clear and Stats Tests
# =============================================================================

class TestSteeringObserverClearAndStats:
    """Tests for clear and stats methods."""

    @pytest.mark.asyncio
    async def test_clear_empties_buffer(self):
        """clear should empty the buffer."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))

        observer.clear()

        assert len(observer.buffer) == 0

    @pytest.mark.asyncio
    async def test_clear_resets_counters(self):
        """clear should reset success and failure counters."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver()

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=False))

        observer.clear()

        assert observer.success_count == 0
        assert observer.failure_count == 0

    @pytest.mark.asyncio
    async def test_stats_returns_correct_values(self):
        """stats should return buffer and counter information."""
        from core.steering.observer import ObservedInteraction, SteeringObserver

        observer = SteeringObserver(buffer_size=100)

        await observer.record(ObservedInteraction("p1", "r1", success=True))
        await observer.record(ObservedInteraction("p2", "r2", success=True))
        await observer.record(ObservedInteraction("p3", "r3", success=False))

        stats = observer.stats()

        assert stats["buffer_size"] == 100
        assert stats["current_count"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert abs(stats["success_rate"] - 2/3) < 1e-5

    def test_stats_handles_empty_buffer(self):
        """stats should handle empty buffer without division by zero."""
        from core.steering.observer import SteeringObserver

        observer = SteeringObserver()

        stats = observer.stats()

        assert stats["current_count"] == 0
        assert stats["success_rate"] == 0.0
