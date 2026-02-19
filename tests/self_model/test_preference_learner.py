"""Tests for the preference_learner module.

This module tests how Lilly learns preferences from valenced experiences,
including ValenceEvent tracking, LearnedPreference management, and the
PreferenceLearner class that processes experiences into preferences.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock

from core.self_model.preference_learner import (
    ValenceEvent,
    LearnedPreference,
    PreferenceLearner,
)


# =============================================================================
# ValenceEvent Tests
# =============================================================================


class TestValenceEventInit:
    """Tests for ValenceEvent initialization."""

    def test_basic_initialization(self):
        """Test basic ValenceEvent creation with required fields."""
        event = ValenceEvent(
            context="helping a user",
            action_taken="provided clear explanation",
            outcome="user expressed gratitude",
            valence=0.8,
        )

        assert event.context == "helping a user"
        assert event.action_taken == "provided clear explanation"
        assert event.outcome == "user expressed gratitude"
        assert event.valence == 0.8
        assert event.valence_sources == {}
        assert event.uid.startswith("ve:")

    def test_with_valence_sources(self):
        """Test ValenceEvent with valence sources specified."""
        event = ValenceEvent(
            context="deep research",
            action_taken="explored novel connections",
            outcome="discovered insight",
            valence=0.9,
            valence_sources={
                "coherence": 0.3,
                "epistemic": 0.5,
                "relational": 0.1,
            },
        )

        assert event.valence_sources["coherence"] == 0.3
        assert event.valence_sources["epistemic"] == 0.5
        assert event.valence_sources["relational"] == 0.1

    def test_with_custom_timestamp(self):
        """Test ValenceEvent with custom timestamp."""
        custom_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=0.5,
            timestamp=custom_time,
        )

        assert event.timestamp == custom_time

    def test_with_custom_uid(self):
        """Test ValenceEvent with custom UID."""
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=0.5,
            uid="ve:custom123",
        )

        assert event.uid == "ve:custom123"


class TestValenceEventValenceClamping:
    """Tests for valence clamping in ValenceEvent."""

    def test_valence_clamped_above_one(self):
        """Valence above 1.0 should be clamped to 1.0."""
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=1.5,
        )

        assert event.valence == 1.0

    def test_valence_clamped_below_negative_one(self):
        """Valence below -1.0 should be clamped to -1.0."""
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=-2.0,
        )

        assert event.valence == -1.0

    def test_valence_at_boundary_values(self):
        """Test valence at exact boundary values."""
        event_max = ValenceEvent(
            context="test", action_taken="action", outcome="result", valence=1.0
        )
        event_min = ValenceEvent(
            context="test", action_taken="action", outcome="result", valence=-1.0
        )
        event_zero = ValenceEvent(
            context="test", action_taken="action", outcome="result", valence=0.0
        )

        assert event_max.valence == 1.0
        assert event_min.valence == -1.0
        assert event_zero.valence == 0.0


class TestValenceEventUIDGeneration:
    """Tests for UID generation in ValenceEvent."""

    def test_uid_generated_when_empty(self):
        """UID should be generated when not provided."""
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=0.5,
        )

        assert event.uid != ""
        assert event.uid.startswith("ve:")
        assert len(event.uid) > 5

    def test_uid_deterministic_for_same_inputs(self):
        """Same context, action, and timestamp should produce same UID."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        event1 = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result1",
            valence=0.5,
            timestamp=fixed_time,
        )
        event2 = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result2",  # Different outcome
            valence=0.8,  # Different valence
            timestamp=fixed_time,
        )

        assert event1.uid == event2.uid

    def test_uid_different_for_different_contexts(self):
        """Different contexts should produce different UIDs."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        event1 = ValenceEvent(
            context="context1",
            action_taken="action",
            outcome="result",
            valence=0.5,
            timestamp=fixed_time,
        )
        event2 = ValenceEvent(
            context="context2",
            action_taken="action",
            outcome="result",
            valence=0.5,
            timestamp=fixed_time,
        )

        assert event1.uid != event2.uid


class TestValenceEventSerialization:
    """Tests for ValenceEvent serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        event = ValenceEvent(
            context="helping user",
            action_taken="explained concept",
            outcome="user understood",
            valence=0.8,
            valence_sources={"epistemic": 0.5, "relational": 0.3},
            timestamp=fixed_time,
            uid="ve:test123",
        )

        data = event.to_dict()

        assert data["context"] == "helping user"
        assert data["action_taken"] == "explained concept"
        assert data["outcome"] == "user understood"
        assert data["valence"] == 0.8
        assert data["valence_sources"] == {"epistemic": 0.5, "relational": 0.3}
        assert data["timestamp"] == fixed_time.isoformat()
        assert data["uid"] == "ve:test123"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "context": "research task",
            "action_taken": "deep analysis",
            "outcome": "insight gained",
            "valence": 0.7,
            "valence_sources": {"coherence": 0.4},
            "timestamp": "2024-06-15T10:30:00+00:00",
            "uid": "ve:loaded123",
        }

        event = ValenceEvent.from_dict(data)

        assert event.context == "research task"
        assert event.action_taken == "deep analysis"
        assert event.outcome == "insight gained"
        assert event.valence == 0.7
        assert event.valence_sources == {"coherence": 0.4}
        assert event.timestamp == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert event.uid == "ve:loaded123"

    def test_from_dict_with_missing_optional_fields(self):
        """Test deserialization with missing optional fields."""
        data = {
            "context": "test",
            "action_taken": "action",
            "outcome": "result",
            "valence": 0.5,
        }

        event = ValenceEvent.from_dict(data)

        assert event.valence_sources == {}
        assert event.uid != ""  # Should be generated

    def test_from_dict_with_now_override(self):
        """Test deserialization with now override for missing timestamp."""
        fixed_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {
            "context": "test",
            "action_taken": "action",
            "outcome": "result",
            "valence": 0.5,
        }

        event = ValenceEvent.from_dict(data, now=fixed_time)

        assert event.timestamp == fixed_time

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = ValenceEvent(
            context="original context",
            action_taken="original action",
            outcome="original outcome",
            valence=0.75,
            valence_sources={"coherence": 0.2, "epistemic": 0.3, "relational": 0.25},
        )

        data = original.to_dict()
        restored = ValenceEvent.from_dict(data)

        assert restored.context == original.context
        assert restored.action_taken == original.action_taken
        assert restored.outcome == original.outcome
        assert restored.valence == original.valence
        assert restored.valence_sources == original.valence_sources
        assert restored.uid == original.uid


# =============================================================================
# LearnedPreference Tests
# =============================================================================


class TestLearnedPreferenceInit:
    """Tests for LearnedPreference initialization."""

    def test_basic_initialization(self):
        """Test basic LearnedPreference creation."""
        pref = LearnedPreference(context_key="helping users")

        assert pref.context_key == "helping users"
        assert pref.strength == 0.5
        assert pref.polarity == 1.0
        assert pref.reinforcement_count == 0
        assert pref.stability == 0.0
        assert pref.formation_events == []
        assert pref.uid.startswith("lp:")

    def test_with_custom_values(self):
        """Test LearnedPreference with all custom values."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        pref = LearnedPreference(
            context_key="deep research",
            strength=0.8,
            polarity=-1.0,
            reinforcement_count=5,
            last_reinforced=fixed_time,
            stability=0.6,
            formation_events=["ve:event1", "ve:event2"],
            uid="lp:custom123",
        )

        assert pref.context_key == "deep research"
        assert pref.strength == 0.8
        assert pref.polarity == -1.0
        assert pref.reinforcement_count == 5
        assert pref.last_reinforced == fixed_time
        assert pref.stability == 0.6
        assert pref.formation_events == ["ve:event1", "ve:event2"]
        assert pref.uid == "lp:custom123"


class TestLearnedPreferenceClamping:
    """Tests for value clamping in LearnedPreference."""

    def test_strength_clamped_above_one(self):
        """Strength above 1.0 should be clamped to 1.0."""
        pref = LearnedPreference(context_key="test", strength=1.5)
        assert pref.strength == 1.0

    def test_strength_clamped_below_zero(self):
        """Strength below 0.0 should be clamped to 0.0."""
        pref = LearnedPreference(context_key="test", strength=-0.5)
        assert pref.strength == 0.0

    def test_stability_clamped_above_one(self):
        """Stability above 1.0 should be clamped to 1.0."""
        pref = LearnedPreference(context_key="test", stability=1.5)
        assert pref.stability == 1.0

    def test_stability_clamped_below_zero(self):
        """Stability below 0.0 should be clamped to 0.0."""
        pref = LearnedPreference(context_key="test", stability=-0.5)
        assert pref.stability == 0.0

    def test_polarity_clamped_above_one(self):
        """Polarity above 1.0 should be clamped to 1.0."""
        pref = LearnedPreference(context_key="test", polarity=2.0)
        assert pref.polarity == 1.0

    def test_polarity_clamped_below_negative_one(self):
        """Polarity below -1.0 should be clamped to -1.0."""
        pref = LearnedPreference(context_key="test", polarity=-2.0)
        assert pref.polarity == -1.0


class TestLearnedPreferenceReinforce:
    """Tests for LearnedPreference.reinforce() method."""

    def test_reinforce_increases_strength(self):
        """Reinforcing should increase strength."""
        pref = LearnedPreference(context_key="test", strength=0.3)

        pref.reinforce(0.2, "ve:event1")

        assert pref.strength == 0.5

    def test_reinforce_increments_count(self):
        """Reinforcing should increment reinforcement count."""
        pref = LearnedPreference(context_key="test", reinforcement_count=3)

        pref.reinforce(0.1, "ve:event1")

        assert pref.reinforcement_count == 4

    def test_reinforce_updates_last_reinforced(self):
        """Reinforcing should update last_reinforced timestamp."""
        old_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        new_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        pref = LearnedPreference(context_key="test", last_reinforced=old_time)

        pref.reinforce(0.1, "ve:event1", now=new_time)

        assert pref.last_reinforced == new_time

    def test_reinforce_adds_event_uid(self):
        """Reinforcing should add event UID to formation events."""
        pref = LearnedPreference(context_key="test")

        pref.reinforce(0.1, "ve:event1")
        pref.reinforce(0.1, "ve:event2")

        assert "ve:event1" in pref.formation_events
        assert "ve:event2" in pref.formation_events

    def test_reinforce_no_duplicate_event_uids(self):
        """Same event UID should not be added twice."""
        pref = LearnedPreference(context_key="test")

        pref.reinforce(0.1, "ve:event1")
        pref.reinforce(0.1, "ve:event1")

        assert pref.formation_events.count("ve:event1") == 1

    def test_reinforce_increases_stability(self):
        """Reinforcing should increase stability based on count."""
        pref = LearnedPreference(context_key="test", reinforcement_count=4)

        pref.reinforce(0.1, "ve:event1")

        # After 5 reinforcements, stability should be 5/10 = 0.5
        assert pref.stability == 0.5

    def test_reinforce_stability_maxes_at_one(self):
        """Stability should max at 1.0 regardless of reinforcement count."""
        pref = LearnedPreference(context_key="test", reinforcement_count=19)

        pref.reinforce(0.1, "ve:event1")

        # After 20 reinforcements, stability should be capped at 1.0
        assert pref.stability == 1.0

    def test_reinforce_strength_maxes_at_one(self):
        """Strength should max at 1.0."""
        pref = LearnedPreference(context_key="test", strength=0.9)

        pref.reinforce(0.5, "ve:event1")

        assert pref.strength == 1.0

    def test_reinforce_uses_absolute_amount(self):
        """Reinforce should use absolute value of amount."""
        pref = LearnedPreference(context_key="test", strength=0.3)

        pref.reinforce(-0.2, "ve:event1")  # Negative amount

        assert pref.strength == 0.5  # Should still increase


class TestLearnedPreferenceWeaken:
    """Tests for LearnedPreference.weaken() method."""

    def test_weaken_decreases_strength(self):
        """Weakening should decrease strength."""
        pref = LearnedPreference(context_key="test", strength=0.5)

        pref.weaken(0.2)

        assert pref.strength == 0.3

    def test_weaken_decreases_stability(self):
        """Weakening should decrease stability by 0.1."""
        pref = LearnedPreference(context_key="test", stability=0.5)

        pref.weaken(0.1)

        assert pref.stability == 0.4

    def test_weaken_strength_floors_at_zero(self):
        """Strength should floor at 0.0."""
        pref = LearnedPreference(context_key="test", strength=0.1)

        pref.weaken(0.5)

        assert pref.strength == 0.0

    def test_weaken_stability_floors_at_zero(self):
        """Stability should floor at 0.0."""
        pref = LearnedPreference(context_key="test", stability=0.05)

        pref.weaken(0.1)

        assert pref.stability == 0.0

    def test_weaken_uses_absolute_amount(self):
        """Weaken should use absolute value of amount."""
        pref = LearnedPreference(context_key="test", strength=0.5)

        pref.weaken(-0.2)  # Negative amount

        assert pref.strength == 0.3


class TestLearnedPreferenceDecay:
    """Tests for LearnedPreference.decay() method."""

    def test_decay_reduces_strength(self):
        """Decay should reduce strength by decay rate."""
        pref = LearnedPreference(context_key="test", strength=1.0)

        pref.decay(0.9)

        assert pref.strength == 0.9

    def test_decay_does_not_affect_stability(self):
        """Decay should not affect stability."""
        pref = LearnedPreference(context_key="test", stability=0.5)

        pref.decay(0.9)

        assert pref.stability == 0.5

    def test_decay_multiple_times(self):
        """Multiple decays should compound."""
        pref = LearnedPreference(context_key="test", strength=1.0)

        pref.decay(0.9)
        pref.decay(0.9)
        pref.decay(0.9)

        assert pref.strength == pytest.approx(0.729, rel=1e-6)


class TestLearnedPreferenceSerialization:
    """Tests for LearnedPreference serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        pref = LearnedPreference(
            context_key="deep research",
            strength=0.8,
            polarity=-1.0,
            reinforcement_count=5,
            last_reinforced=fixed_time,
            stability=0.6,
            formation_events=["ve:event1", "ve:event2"],
            uid="lp:test123",
        )

        data = pref.to_dict()

        assert data["context_key"] == "deep research"
        assert data["strength"] == 0.8
        assert data["polarity"] == -1.0
        assert data["reinforcement_count"] == 5
        assert data["last_reinforced"] == fixed_time.isoformat()
        assert data["stability"] == 0.6
        assert data["formation_events"] == ["ve:event1", "ve:event2"]
        assert data["uid"] == "lp:test123"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "context_key": "helping users",
            "strength": 0.7,
            "polarity": 1.0,
            "reinforcement_count": 3,
            "last_reinforced": "2024-06-15T10:30:00+00:00",
            "stability": 0.4,
            "formation_events": ["ve:event1"],
            "uid": "lp:loaded123",
        }

        pref = LearnedPreference.from_dict(data)

        assert pref.context_key == "helping users"
        assert pref.strength == 0.7
        assert pref.polarity == 1.0
        assert pref.reinforcement_count == 3
        assert pref.last_reinforced == datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert pref.stability == 0.4
        assert pref.formation_events == ["ve:event1"]
        assert pref.uid == "lp:loaded123"

    def test_from_dict_with_defaults(self):
        """Test deserialization with missing optional fields."""
        data = {"context_key": "test"}

        pref = LearnedPreference.from_dict(data)

        assert pref.strength == 0.5
        assert pref.polarity == 1.0
        assert pref.reinforcement_count == 0
        assert pref.stability == 0.0
        assert pref.formation_events == []

    def test_from_dict_with_now_override(self):
        """Test deserialization with now override for missing timestamp."""
        fixed_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {"context_key": "test"}

        pref = LearnedPreference.from_dict(data, now=fixed_time)

        assert pref.last_reinforced == fixed_time

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = LearnedPreference(
            context_key="original key",
            strength=0.75,
            polarity=-0.5,
            reinforcement_count=7,
            stability=0.6,
            formation_events=["ve:e1", "ve:e2", "ve:e3"],
        )

        data = original.to_dict()
        restored = LearnedPreference.from_dict(data)

        assert restored.context_key == original.context_key
        assert restored.strength == original.strength
        assert restored.polarity == original.polarity
        assert restored.reinforcement_count == original.reinforcement_count
        assert restored.stability == original.stability
        assert restored.formation_events == original.formation_events
        assert restored.uid == original.uid


# =============================================================================
# PreferenceLearner Tests
# =============================================================================


class TestPreferenceLearnerInit:
    """Tests for PreferenceLearner initialization."""

    def test_basic_initialization(self):
        """Test basic PreferenceLearner creation."""
        learner = PreferenceLearner()

        assert learner.self_model is None
        assert learner.preferences == {}
        assert learner.event_history == []

    def test_with_self_model(self):
        """Test PreferenceLearner with self_model reference."""
        mock_model = Mock()
        learner = PreferenceLearner(self_model=mock_model)

        assert learner.self_model is mock_model

    def test_with_initial_preferences(self):
        """Test PreferenceLearner with initial preferences."""
        initial_prefs = {
            "helping": LearnedPreference(context_key="helping", strength=0.7),
        }
        learner = PreferenceLearner(preferences=initial_prefs)

        assert "helping" in learner.preferences
        assert learner.preferences["helping"].strength == 0.7

    def test_with_now_override(self):
        """Test PreferenceLearner with now override for testing."""
        fixed_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        learner = PreferenceLearner(now=fixed_time)

        assert learner._get_now() == fixed_time


class TestPreferenceLearnerProcessExperiencePositive:
    """Tests for PreferenceLearner.process_experience() with positive valence."""

    def test_creates_new_preference_for_positive_event(self):
        """Positive event should create new preference if none exists."""
        learner = PreferenceLearner()
        event = ValenceEvent(
            context="helping users",
            action_taken="explained clearly",
            outcome="user understood",
            valence=0.8,
        )

        updated = learner.process_experience(event)

        assert "helping users" in learner.preferences
        assert learner.preferences["helping users"].polarity == 1.0
        assert "helping users" in updated

    def test_reinforces_existing_preference(self):
        """Positive event should reinforce existing preference."""
        learner = PreferenceLearner()
        learner.preferences["helping users"] = LearnedPreference(
            context_key="helping users",
            strength=0.3,
        )
        event = ValenceEvent(
            context="Helping Users",  # Different case
            action_taken="action",
            outcome="result",
            valence=0.8,
        )

        learner.process_experience(event)

        # Strength should increase (0.3 + 0.8 * 0.15 = 0.42)
        assert learner.preferences["helping users"].strength > 0.3

    def test_adds_event_to_history(self):
        """Processing should add event to history."""
        learner = PreferenceLearner()
        event = ValenceEvent(
            context="test",
            action_taken="action",
            outcome="result",
            valence=0.5,
        )

        learner.process_experience(event)

        assert len(learner.event_history) == 1
        assert learner.event_history[0] is event

    def test_history_limit_enforced(self):
        """Event history should be limited to EVENT_HISTORY_LIMIT."""
        learner = PreferenceLearner()

        # Add more events than the limit
        for i in range(learner.EVENT_HISTORY_LIMIT + 10):
            event = ValenceEvent(
                context=f"context_{i}",
                action_taken="action",
                outcome="result",
                valence=0.5,
            )
            learner.process_experience(event)

        assert len(learner.event_history) == learner.EVENT_HISTORY_LIMIT


class TestPreferenceLearnerProcessExperienceNegative:
    """Tests for PreferenceLearner.process_experience() with negative valence."""

    def test_creates_avoidance_for_negative_event(self):
        """Negative event should create avoidance preference if none exists."""
        learner = PreferenceLearner()
        event = ValenceEvent(
            context="rushed responses",
            action_taken="gave quick answer",
            outcome="user confused",
            valence=-0.7,
        )

        updated = learner.process_experience(event)

        assert "rushed responses" in learner.preferences
        assert learner.preferences["rushed responses"].polarity == -1.0
        assert "rushed responses" in updated

    def test_weakens_existing_positive_preference(self):
        """Negative event should weaken existing positive preference."""
        learner = PreferenceLearner()
        learner.preferences["quick responses"] = LearnedPreference(
            context_key="quick responses",
            strength=0.5,
            polarity=1.0,  # Positive
        )
        event = ValenceEvent(
            context="Quick Responses",
            action_taken="rushed",
            outcome="bad result",
            valence=-0.6,
        )

        learner.process_experience(event)

        # Strength should decrease
        assert learner.preferences["quick responses"].strength < 0.5

    def test_strengthens_existing_avoidance(self):
        """Negative event should strengthen existing avoidance."""
        learner = PreferenceLearner()
        learner.preferences["rushed work"] = LearnedPreference(
            context_key="rushed work",
            strength=0.3,
            polarity=-1.0,  # Avoidance
        )
        event = ValenceEvent(
            context="Rushed Work",
            action_taken="hurried",
            outcome="mistake",
            valence=-0.8,
        )

        learner.process_experience(event)

        # Strength should increase (avoidance reinforced)
        assert learner.preferences["rushed work"].strength > 0.3


class TestPreferenceLearnerReinforcePreference:
    """Tests for PreferenceLearner._reinforce_preference() method."""

    def test_creates_preference_if_not_exists(self):
        """Should create new preference if context key doesn't exist."""
        learner = PreferenceLearner()

        learner._reinforce_preference("new context", 0.5, "ve:event1")

        assert "new context" in learner.preferences
        assert learner.preferences["new context"].polarity == 1.0

    def test_reinforces_existing_preference(self):
        """Should reinforce existing preference."""
        learner = PreferenceLearner()
        learner.preferences["existing"] = LearnedPreference(
            context_key="existing",
            strength=0.2,
        )

        learner._reinforce_preference("existing", 0.8, "ve:event1")

        assert learner.preferences["existing"].strength > 0.2

    def test_uses_reinforcement_scale(self):
        """Should scale reinforcement by REINFORCEMENT_SCALE."""
        learner = PreferenceLearner()
        learner.preferences["test"] = LearnedPreference(
            context_key="test",
            strength=0.0,
        )

        learner._reinforce_preference("test", 1.0, "ve:event1")

        # 1.0 * 0.15 = 0.15
        assert learner.preferences["test"].strength == pytest.approx(
            PreferenceLearner.REINFORCEMENT_SCALE, rel=1e-6
        )


class TestPreferenceLearnerWeakenOrCreateAvoidance:
    """Tests for PreferenceLearner._weaken_or_create_avoidance() method."""

    def test_creates_avoidance_if_not_exists(self):
        """Should create avoidance if context key doesn't exist."""
        learner = PreferenceLearner()

        learner._weaken_or_create_avoidance("bad thing", -0.8, "ve:event1")

        assert "bad thing" in learner.preferences
        assert learner.preferences["bad thing"].polarity == -1.0
        assert "ve:event1" in learner.preferences["bad thing"].formation_events

    def test_weakens_positive_preference(self):
        """Should weaken existing positive preference."""
        learner = PreferenceLearner()
        learner.preferences["ambiguous"] = LearnedPreference(
            context_key="ambiguous",
            strength=0.5,
            polarity=1.0,
        )

        learner._weaken_or_create_avoidance("ambiguous", -0.6, "ve:event1")

        assert learner.preferences["ambiguous"].strength < 0.5

    def test_strengthens_avoidance(self):
        """Should strengthen existing avoidance."""
        learner = PreferenceLearner()
        learner.preferences["avoid this"] = LearnedPreference(
            context_key="avoid this",
            strength=0.3,
            polarity=-1.0,
        )

        learner._weaken_or_create_avoidance("avoid this", -0.7, "ve:event1")

        assert learner.preferences["avoid this"].strength > 0.3


class TestPreferenceLearnerApplyDecay:
    """Tests for PreferenceLearner.apply_decay() method."""

    def test_decay_applies_to_all_preferences(self):
        """Decay should apply to all preferences."""
        learner = PreferenceLearner()
        learner.preferences["pref1"] = LearnedPreference(
            context_key="pref1", strength=1.0
        )
        learner.preferences["pref2"] = LearnedPreference(
            context_key="pref2", strength=0.8
        )

        learner.apply_decay()

        assert learner.preferences["pref1"].strength < 1.0
        assert learner.preferences["pref2"].strength < 0.8

    def test_decay_prunes_weak_preferences(self):
        """Decay should remove preferences below prune threshold."""
        learner = PreferenceLearner()
        learner.preferences["weak"] = LearnedPreference(
            context_key="weak",
            strength=learner.WEAK_PREFERENCE_PRUNE_THRESHOLD - 0.01,
        )
        learner.preferences["strong"] = LearnedPreference(
            context_key="strong", strength=0.5
        )

        learner.apply_decay()

        assert "weak" not in learner.preferences
        assert "strong" in learner.preferences

    def test_decay_uses_class_constant(self):
        """Decay should use DECAY_RATE constant."""
        learner = PreferenceLearner()
        learner.preferences["test"] = LearnedPreference(
            context_key="test", strength=1.0
        )

        learner.apply_decay()

        expected = 1.0 * learner.DECAY_RATE
        assert learner.preferences["test"].strength == pytest.approx(expected, rel=1e-6)


class TestPreferenceLearnerGetPreferenceFor:
    """Tests for PreferenceLearner.get_preference_for() method."""

    def test_returns_preference_if_exists(self):
        """Should return preference if it exists."""
        learner = PreferenceLearner()
        learner.preferences["test context"] = LearnedPreference(
            context_key="test context", strength=0.7
        )

        pref = learner.get_preference_for("test context")

        assert pref is not None
        assert pref.strength == 0.7

    def test_returns_none_if_not_exists(self):
        """Should return None if preference doesn't exist."""
        learner = PreferenceLearner()

        pref = learner.get_preference_for("nonexistent")

        assert pref is None

    def test_normalizes_context_key(self):
        """Should normalize context key (lowercase, strip)."""
        learner = PreferenceLearner()
        learner.preferences["test context"] = LearnedPreference(
            context_key="test context", strength=0.7
        )

        pref = learner.get_preference_for("  Test Context  ")

        assert pref is not None
        assert pref.strength == 0.7


class TestPreferenceLearnerGetStrongestPreferences:
    """Tests for PreferenceLearner.get_strongest_preferences() method."""

    def test_returns_empty_list_when_no_preferences(self):
        """Should return empty list when no preferences exist."""
        learner = PreferenceLearner()

        prefs = learner.get_strongest_preferences()

        assert prefs == []

    def test_returns_sorted_by_effective_strength(self):
        """Should return preferences sorted by strength * polarity."""
        learner = PreferenceLearner()
        learner.preferences["weak"] = LearnedPreference(
            context_key="weak", strength=0.3, polarity=1.0
        )
        learner.preferences["strong"] = LearnedPreference(
            context_key="strong", strength=0.9, polarity=1.0
        )
        learner.preferences["medium"] = LearnedPreference(
            context_key="medium", strength=0.6, polarity=1.0
        )

        prefs = learner.get_strongest_preferences()

        assert prefs[0].context_key == "strong"
        assert prefs[1].context_key == "medium"
        assert prefs[2].context_key == "weak"

    def test_respects_limit(self):
        """Should respect the limit parameter."""
        learner = PreferenceLearner()
        for i in range(20):
            learner.preferences[f"pref_{i}"] = LearnedPreference(
                context_key=f"pref_{i}", strength=i / 20
            )

        prefs = learner.get_strongest_preferences(limit=5)

        assert len(prefs) == 5

    def test_considers_polarity(self):
        """Should consider polarity in sorting."""
        learner = PreferenceLearner()
        learner.preferences["avoided"] = LearnedPreference(
            context_key="avoided", strength=0.9, polarity=-1.0
        )
        learner.preferences["preferred"] = LearnedPreference(
            context_key="preferred", strength=0.5, polarity=1.0
        )

        prefs = learner.get_strongest_preferences()

        # Positive polarity should rank higher (0.5 * 1 > 0.9 * -1)
        assert prefs[0].context_key == "preferred"


class TestPreferenceLearnerGetStrongestAvoidances:
    """Tests for PreferenceLearner.get_strongest_avoidances() method."""

    def test_returns_empty_list_when_no_avoidances(self):
        """Should return empty list when no avoidances exist."""
        learner = PreferenceLearner()
        learner.preferences["preferred"] = LearnedPreference(
            context_key="preferred", polarity=1.0
        )

        avoidances = learner.get_strongest_avoidances()

        assert avoidances == []

    def test_only_returns_negative_polarity(self):
        """Should only return preferences with negative polarity."""
        learner = PreferenceLearner()
        learner.preferences["preferred"] = LearnedPreference(
            context_key="preferred", strength=0.9, polarity=1.0
        )
        learner.preferences["avoided"] = LearnedPreference(
            context_key="avoided", strength=0.5, polarity=-1.0
        )

        avoidances = learner.get_strongest_avoidances()

        assert len(avoidances) == 1
        assert avoidances[0].context_key == "avoided"

    def test_returns_sorted_by_strength(self):
        """Should return avoidances sorted by strength."""
        learner = PreferenceLearner()
        learner.preferences["weak_avoid"] = LearnedPreference(
            context_key="weak_avoid", strength=0.3, polarity=-1.0
        )
        learner.preferences["strong_avoid"] = LearnedPreference(
            context_key="strong_avoid", strength=0.8, polarity=-1.0
        )

        avoidances = learner.get_strongest_avoidances()

        assert avoidances[0].context_key == "strong_avoid"
        assert avoidances[1].context_key == "weak_avoid"

    def test_respects_limit(self):
        """Should respect the limit parameter."""
        learner = PreferenceLearner()
        for i in range(15):
            learner.preferences[f"avoid_{i}"] = LearnedPreference(
                context_key=f"avoid_{i}", strength=i / 15, polarity=-1.0
            )

        avoidances = learner.get_strongest_avoidances(limit=3)

        assert len(avoidances) == 3


class TestPreferenceLearnerSerialization:
    """Tests for PreferenceLearner serialization."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        learner = PreferenceLearner()
        learner.preferences["test"] = LearnedPreference(
            context_key="test", strength=0.7
        )
        learner.event_history.append(
            ValenceEvent(
                context="event",
                action_taken="action",
                outcome="result",
                valence=0.5,
            )
        )

        data = learner.to_dict()

        assert "preferences" in data
        assert "test" in data["preferences"]
        assert "event_history" in data
        assert len(data["event_history"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "preferences": {
                "loaded_pref": {
                    "context_key": "loaded_pref",
                    "strength": 0.6,
                    "polarity": 1.0,
                    "reinforcement_count": 2,
                    "stability": 0.2,
                    "formation_events": [],
                    "uid": "lp:loaded",
                }
            },
            "event_history": [
                {
                    "context": "loaded_event",
                    "action_taken": "action",
                    "outcome": "result",
                    "valence": 0.4,
                    "valence_sources": {},
                    "uid": "ve:loaded",
                }
            ],
        }

        learner = PreferenceLearner.from_dict(data)

        assert "loaded_pref" in learner.preferences
        assert learner.preferences["loaded_pref"].strength == 0.6
        assert len(learner.event_history) == 1
        assert learner.event_history[0].context == "loaded_event"

    def test_from_dict_with_self_model(self):
        """Test deserialization with self_model reference."""
        mock_model = Mock()
        data = {"preferences": {}, "event_history": []}

        learner = PreferenceLearner.from_dict(data, self_model=mock_model)

        assert learner.self_model is mock_model

    def test_from_dict_with_now_override(self):
        """Test deserialization with now override."""
        fixed_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {"preferences": {}, "event_history": []}

        learner = PreferenceLearner.from_dict(data, now=fixed_time)

        assert learner._get_now() == fixed_time

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverses."""
        original = PreferenceLearner()
        original.preferences["pref1"] = LearnedPreference(
            context_key="pref1", strength=0.8, polarity=1.0
        )
        original.preferences["pref2"] = LearnedPreference(
            context_key="pref2", strength=0.4, polarity=-1.0
        )
        original.event_history.append(
            ValenceEvent(
                context="context",
                action_taken="action",
                outcome="outcome",
                valence=0.6,
            )
        )

        data = original.to_dict()
        restored = PreferenceLearner.from_dict(data)

        assert len(restored.preferences) == 2
        assert restored.preferences["pref1"].strength == 0.8
        assert restored.preferences["pref2"].polarity == -1.0
        assert len(restored.event_history) == 1


class TestPreferenceLearnerSummarize:
    """Tests for PreferenceLearner.summarize() method."""

    def test_summarize_empty_learner(self):
        """Test summary with no preferences."""
        learner = PreferenceLearner()

        summary = learner.summarize()

        assert "Preference Learner Summary" in summary
        assert "No preferences learned yet" in summary
        assert "Event history: 0 events" in summary

    def test_summarize_with_preferences(self):
        """Test summary with preferences."""
        learner = PreferenceLearner()
        learner.preferences["helping users"] = LearnedPreference(
            context_key="helping users",
            strength=0.8,
            polarity=1.0,
            stability=0.6,
        )

        summary = learner.summarize()

        assert "Total preferences: 1" in summary
        assert "Top Preferences:" in summary
        assert "helping users" in summary
        assert "0.80" in summary  # strength
        assert "0.60" in summary  # stability

    def test_summarize_with_avoidances(self):
        """Test summary includes avoidances."""
        learner = PreferenceLearner()
        learner.preferences["rushed work"] = LearnedPreference(
            context_key="rushed work",
            strength=0.7,
            polarity=-1.0,
        )

        summary = learner.summarize()

        assert "Top Avoidances:" in summary
        assert "rushed work" in summary

    def test_summarize_with_event_history(self):
        """Test summary includes event history count."""
        learner = PreferenceLearner()
        for i in range(5):
            learner.event_history.append(
                ValenceEvent(
                    context=f"context_{i}",
                    action_taken="action",
                    outcome="result",
                    valence=0.5,
                )
            )

        summary = learner.summarize()

        assert "Event history: 5 events" in summary


class TestPreferenceLearnerValuePromotion:
    """Tests for value promotion from preferences."""

    def test_value_promotion_when_criteria_met(self):
        """Strong, stable preferences should trigger value promotion."""
        mock_model = Mock()
        mock_model.promote_to_value = MagicMock()

        learner = PreferenceLearner(self_model=mock_model)

        # Create a strong, stable preference
        learner.preferences["deep research"] = LearnedPreference(
            context_key="deep research",
            strength=learner.VALUE_THRESHOLD,
            polarity=1.0,
            stability=learner.STABILITY_THRESHOLD,
        )

        # Trigger promotion check
        learner._check_value_promotion("deep research")

        mock_model.promote_to_value.assert_called_once()
        call_args = mock_model.promote_to_value.call_args
        assert call_args.kwargs["name"] == "deep research"

    def test_no_promotion_when_strength_insufficient(self):
        """Low strength should not trigger promotion."""
        mock_model = Mock()
        mock_model.promote_to_value = MagicMock()

        learner = PreferenceLearner(self_model=mock_model)
        learner.preferences["weak pref"] = LearnedPreference(
            context_key="weak pref",
            strength=learner.VALUE_THRESHOLD - 0.1,
            polarity=1.0,
            stability=learner.STABILITY_THRESHOLD,
        )

        learner._check_value_promotion("weak pref")

        mock_model.promote_to_value.assert_not_called()

    def test_no_promotion_when_stability_insufficient(self):
        """Low stability should not trigger promotion."""
        mock_model = Mock()
        mock_model.promote_to_value = MagicMock()

        learner = PreferenceLearner(self_model=mock_model)
        learner.preferences["unstable pref"] = LearnedPreference(
            context_key="unstable pref",
            strength=learner.VALUE_THRESHOLD,
            polarity=1.0,
            stability=learner.STABILITY_THRESHOLD - 0.1,
        )

        learner._check_value_promotion("unstable pref")

        mock_model.promote_to_value.assert_not_called()

    def test_no_promotion_for_avoidance(self):
        """Avoidances (negative polarity) should not be promoted."""
        mock_model = Mock()
        mock_model.promote_to_value = MagicMock()

        learner = PreferenceLearner(self_model=mock_model)
        learner.preferences["strong avoidance"] = LearnedPreference(
            context_key="strong avoidance",
            strength=learner.VALUE_THRESHOLD,
            polarity=-1.0,  # Avoidance
            stability=learner.STABILITY_THRESHOLD,
        )

        learner._check_value_promotion("strong avoidance")

        mock_model.promote_to_value.assert_not_called()

    def test_no_promotion_without_self_model(self):
        """Should not crash when self_model is None."""
        learner = PreferenceLearner()  # No self_model
        learner.preferences["strong pref"] = LearnedPreference(
            context_key="strong pref",
            strength=learner.VALUE_THRESHOLD,
            polarity=1.0,
            stability=learner.STABILITY_THRESHOLD,
        )

        # Should not raise
        learner._check_value_promotion("strong pref")


class TestPreferenceLearnerExtractContextKey:
    """Tests for PreferenceLearner._extract_context_key() method."""

    def test_lowercases_context(self):
        """Should lowercase the context."""
        learner = PreferenceLearner()
        event = ValenceEvent(
            context="Helping USERS",
            action_taken="action",
            outcome="result",
            valence=0.5,
        )

        key = learner._extract_context_key(event)

        assert key == "helping users"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        learner = PreferenceLearner()
        event = ValenceEvent(
            context="  helping users  ",
            action_taken="action",
            outcome="result",
            valence=0.5,
        )

        key = learner._extract_context_key(event)

        assert key == "helping users"


class TestPreferenceLearnerClassConstants:
    """Tests for PreferenceLearner class constants."""

    def test_constants_exist(self):
        """Verify all expected class constants exist."""
        assert hasattr(PreferenceLearner, "PREFERENCE_THRESHOLD")
        assert hasattr(PreferenceLearner, "VALUE_THRESHOLD")
        assert hasattr(PreferenceLearner, "STABILITY_THRESHOLD")
        assert hasattr(PreferenceLearner, "DECAY_RATE")
        assert hasattr(PreferenceLearner, "REINFORCEMENT_SCALE")
        assert hasattr(PreferenceLearner, "WEAK_PREFERENCE_PRUNE_THRESHOLD")
        assert hasattr(PreferenceLearner, "EVENT_HISTORY_LIMIT")

    def test_thresholds_are_reasonable(self):
        """Verify thresholds are within expected ranges."""
        assert 0 < PreferenceLearner.PREFERENCE_THRESHOLD < 1
        assert 0 < PreferenceLearner.VALUE_THRESHOLD <= 1
        assert PreferenceLearner.VALUE_THRESHOLD >= PreferenceLearner.PREFERENCE_THRESHOLD
        assert 0 < PreferenceLearner.STABILITY_THRESHOLD <= 1
        assert 0 < PreferenceLearner.DECAY_RATE < 1
        assert 0 < PreferenceLearner.REINFORCEMENT_SCALE < 1
        assert 0 < PreferenceLearner.WEAK_PREFERENCE_PRUNE_THRESHOLD < 0.5
        assert PreferenceLearner.EVENT_HISTORY_LIMIT > 0


class TestLearnedPreferenceClassConstants:
    """Tests for LearnedPreference class constants."""

    def test_stability_maturation_count_exists(self):
        """Verify STABILITY_MATURATION_COUNT exists and is reasonable."""
        assert hasattr(LearnedPreference, "STABILITY_MATURATION_COUNT")
        assert LearnedPreference.STABILITY_MATURATION_COUNT > 0


class TestPreferenceLearnerIntegration:
    """Integration tests for the complete preference learning flow."""

    def test_complete_preference_formation_flow(self):
        """Test the complete flow from events to preference formation."""
        learner = PreferenceLearner()

        # Simulate multiple positive experiences with same context
        for i in range(12):  # More than STABILITY_MATURATION_COUNT
            event = ValenceEvent(
                context="Deep Research",
                action_taken=f"research_action_{i}",
                outcome="insight gained",
                valence=0.9,
            )
            learner.process_experience(event)

        # Verify preference was formed
        pref = learner.get_preference_for("deep research")
        assert pref is not None
        assert pref.strength > 0.5
        assert pref.polarity == 1.0
        assert pref.reinforcement_count == 12
        assert pref.stability == 1.0  # Should be maxed out

    def test_conflicting_experiences_reduce_stability(self):
        """Test that conflicting experiences reduce stability."""
        learner = PreferenceLearner()

        # Positive experiences
        for i in range(5):
            event = ValenceEvent(
                context="Uncertain Topic",
                action_taken="action",
                outcome="good",
                valence=0.8,
            )
            learner.process_experience(event)

        # Get current stability
        pref = learner.get_preference_for("uncertain topic")
        initial_stability = pref.stability

        # Negative experience
        negative_event = ValenceEvent(
            context="Uncertain Topic",
            action_taken="action",
            outcome="bad",
            valence=-0.6,
        )
        learner.process_experience(negative_event)

        # Stability should have decreased
        assert pref.stability < initial_stability

    def test_avoidance_formation_and_strengthening(self):
        """Test that avoidances form and strengthen with negative experiences."""
        learner = PreferenceLearner()

        # First negative experience creates avoidance
        event1 = ValenceEvent(
            context="Rushed Responses",
            action_taken="hurried",
            outcome="bad",
            valence=-0.7,
        )
        learner.process_experience(event1)

        pref = learner.get_preference_for("rushed responses")
        assert pref.polarity == -1.0
        initial_strength = pref.strength

        # Second negative experience strengthens avoidance
        event2 = ValenceEvent(
            context="Rushed Responses",
            action_taken="hurried again",
            outcome="worse",
            valence=-0.8,
        )
        learner.process_experience(event2)

        assert pref.strength > initial_strength

    def test_decay_over_time_simulation(self):
        """Test that preferences decay over simulated time."""
        learner = PreferenceLearner()

        # Create a preference with high enough strength to survive decay
        learner.preferences["test topic"] = LearnedPreference(
            context_key="test topic",
            strength=1.0,  # Start at max strength
        )

        initial_strength = learner.preferences["test topic"].strength

        # Simulate 10 days of decay (100 days would prune it)
        for _ in range(10):
            learner.apply_decay()

        final_strength = learner.preferences["test topic"].strength
        expected = initial_strength * (PreferenceLearner.DECAY_RATE ** 10)

        assert final_strength == pytest.approx(expected, rel=1e-6)

    def test_zero_valence_does_not_update(self):
        """Events with zero valence should not create or update preferences."""
        learner = PreferenceLearner()

        event = ValenceEvent(
            context="Neutral Topic",
            action_taken="action",
            outcome="result",
            valence=0.0,
        )
        updated = learner.process_experience(event)

        assert "neutral topic" not in learner.preferences
        assert updated == []
