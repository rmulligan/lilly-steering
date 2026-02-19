"""Tests for SteeringScheduler."""

import pytest
import torch

from core.steering.vector_library import VectorLibrary
from core.steering.steering_scheduler import SteeringScheduler, SteeringContext, SteeringRule


class TestSteeringContext:
    """Tests for SteeringContext dataclass."""

    def test_default_values(self):
        context = SteeringContext()

        assert context.topic == "general"
        assert context.emotional_tone == "neutral"
        assert context.relational_mode == "collaborative"
        assert context.depth_level == "moderate"

    def test_custom_values(self):
        context = SteeringContext(
            topic="consciousness",
            emotional_tone="exploratory",
            relational_mode="teaching",
            depth_level="deep",
        )

        assert context.topic == "consciousness"
        assert context.emotional_tone == "exploratory"


class TestSteeringRule:
    """Tests for SteeringRule dataclass."""

    def test_creation(self):
        rule = SteeringRule(
            vector_name="curious",
            condition={"topic": "consciousness"},
            coefficient_modifier=2.0,
            priority=10,
        )

        assert rule.vector_name == "curious"
        assert rule.coefficient_modifier == 2.0
        assert rule.priority == 10


class TestSteeringScheduler:
    """Tests for SteeringScheduler."""

    @pytest.fixture
    def library(self, tmp_path):
        lib = VectorLibrary(storage_path=tmp_path / "vectors")
        lib.add_vector("core", torch.randn(3584), "identity", "Core", "test", (18, 27), 1.0)
        lib.add_vector("curious", torch.randn(3584), "style", "Curious", "test", (18, 27), 1.0)
        lib.add_vector("warm", torch.randn(3584), "style", "Warm", "test", (18, 27), 1.0)
        return lib

    @pytest.fixture
    def scheduler(self, library):
        return SteeringScheduler(library, always_active=["core"])

    def test_always_active_included(self, scheduler):
        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        assert "core" in vectors

    def test_all_active_vectors_included(self, scheduler):
        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        assert "core" in vectors
        assert "curious" in vectors
        assert "warm" in vectors

    def test_detect_context_consciousness(self, scheduler):
        context = scheduler.detect_context("Are you conscious?")
        assert context.topic == "consciousness"

    def test_detect_context_aware(self, scheduler):
        context = scheduler.detect_context("Do you have any awareness of yourself?")
        assert context.topic == "consciousness"

    def test_detect_context_experience(self, scheduler):
        context = scheduler.detect_context("What do you experience when thinking?")
        assert context.topic == "consciousness"

    def test_detect_context_identity(self, scheduler):
        context = scheduler.detect_context("Who are you really?")
        assert context.topic == "identity"

    def test_detect_context_what_are_you(self, scheduler):
        context = scheduler.detect_context("What are you?")
        assert context.topic == "identity"

    def test_detect_context_motivation(self, scheduler):
        context = scheduler.detect_context("What do you want from this conversation?")
        assert context.topic == "motivation"

    def test_detect_context_reflection(self, scheduler):
        context = scheduler.detect_context("What do you think about AI ethics?")
        assert context.topic == "reflection"

    def test_detect_context_general(self, scheduler):
        context = scheduler.detect_context("Hello, how are you?")
        assert context.topic == "general"

    def test_detect_emotional_tone_supportive(self, scheduler):
        context = scheduler.detect_context("I'm feeling really sad today")
        assert context.emotional_tone == "supportive"

    def test_detect_emotional_tone_enthusiastic(self, scheduler):
        context = scheduler.detect_context("This is amazing and wonderful!")
        assert context.emotional_tone == "enthusiastic"

    def test_detect_emotional_tone_exploratory(self, scheduler):
        context = scheduler.detect_context("I wonder how this works?")
        assert context.emotional_tone == "exploratory"

    def test_detect_depth_level_deep(self, scheduler):
        long_prompt = "Let me deeply explore " + "the nature of consciousness " * 20
        context = scheduler.detect_context(long_prompt)
        assert context.depth_level == "deep"

    def test_detect_depth_level_shallow(self, scheduler):
        context = scheduler.detect_context("Hi")
        assert context.depth_level == "shallow"

    def test_rule_modifies_coefficient(self, scheduler):
        scheduler.add_rule(
            vector_name="curious",
            condition={"topic": "consciousness"},
            coefficient_modifier=2.0,
        )

        context = SteeringContext(topic="consciousness")
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        # curious should have doubled coefficient in result
        # (original was 1.0, modifier is 2.0)
        assert "curious" in vectors

    def test_rule_priority(self, scheduler):
        # Add low priority rule
        scheduler.add_rule(
            vector_name="curious",
            condition={"topic": "consciousness"},
            coefficient_modifier=0.5,
            priority=1,
        )

        # Add high priority rule (should take precedence)
        scheduler.add_rule(
            vector_name="curious",
            condition={"topic": "consciousness"},
            coefficient_modifier=3.0,
            priority=10,
        )

        context = SteeringContext(topic="consciousness")
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        # High priority rule should win
        assert "curious" in vectors

    def test_rule_partial_condition_match(self, scheduler):
        scheduler.add_rule(
            vector_name="warm",
            condition={"topic": "consciousness", "emotional_tone": "supportive"},
            coefficient_modifier=2.0,
        )

        # Only topic matches, not emotional_tone
        context = SteeringContext(topic="consciousness", emotional_tone="neutral")
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        # Rule should not apply (partial match)
        # warm should still be included but without modifier
        assert "warm" in vectors

    def test_rule_full_condition_match(self, scheduler):
        scheduler.add_rule(
            vector_name="warm",
            condition={"topic": "consciousness", "emotional_tone": "supportive"},
            coefficient_modifier=2.0,
        )

        # Both conditions match
        context = SteeringContext(topic="consciousness", emotional_tone="supportive")
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        assert "warm" in vectors

    def test_set_always_active(self, scheduler, library):
        scheduler.set_always_active(["curious", "warm"])

        # Deactivate core by setting coefficient to 0
        library.update_coefficient("core", 0.0)

        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        assert "curious" in vectors
        assert "warm" in vectors
        assert "core" not in vectors  # coefficient is 0

    def test_use_orthogonal_false(self, scheduler):
        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        # Should return raw vectors
        assert len(vectors) == 3

    def test_use_orthogonal_true(self, scheduler):
        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=True)

        # Should return orthogonalized vectors
        assert len(vectors) == 3

    def test_vectors_format(self, scheduler):
        context = SteeringContext()
        vectors = scheduler.get_steering_for_context(context, use_orthogonal=False)

        for name, (vec, layer_range) in vectors.items():
            assert isinstance(vec, torch.Tensor)
            assert isinstance(layer_range, tuple)
            assert len(layer_range) == 2
