"""Tests for the affective system module with 8D Plutchik emotions."""

import math
import pytest

from core.self_model.affective_system import (
    AffectiveState,
    ValenceWeights,
    ValenceSystem,
    PLUTCHIK_PRIMARIES,
)


class TestAffectiveState:
    """Tests for AffectiveState dataclass with 8D Plutchik emotions."""

    def test_initialization_defaults(self):
        """Test default values for AffectiveState."""
        state = AffectiveState()
        # Joy, trust, anticipation default to 0.5 (baseline)
        assert state.joy == 0.5
        assert state.trust == 0.5
        assert state.anticipation == 0.5
        # Fear, surprise, sadness, disgust, anger default to 0.0
        assert state.fear == 0.0
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0

    def test_initialization_custom_values(self):
        """Test custom values for AffectiveState."""
        state = AffectiveState(
            joy=0.8,
            trust=0.3,
            fear=0.9,
            surprise=0.1,
            sadness=0.7,
            disgust=0.4,
            anger=0.6,
            anticipation=0.2,
        )
        assert state.joy == 0.8
        assert state.trust == 0.3
        assert state.fear == 0.9
        assert state.surprise == 0.1
        assert state.sadness == 0.7
        assert state.disgust == 0.4
        assert state.anger == 0.6
        assert state.anticipation == 0.2

    def test_clamping_values_above_1(self):
        """Test that values above 1 are clamped."""
        state = AffectiveState(
            joy=1.5,
            trust=2.0,
            fear=1.1,
            surprise=10.0,
            sadness=1.2,
            disgust=3.0,
            anger=5.0,
            anticipation=1.8,
        )
        assert state.joy == 1.0
        assert state.trust == 1.0
        assert state.fear == 1.0
        assert state.surprise == 1.0
        assert state.sadness == 1.0
        assert state.disgust == 1.0
        assert state.anger == 1.0
        assert state.anticipation == 1.0

    def test_clamping_values_below_0(self):
        """Test that values below 0 are clamped."""
        state = AffectiveState(
            joy=-0.5,
            trust=-1.0,
            fear=-0.1,
            surprise=-10.0,
            sadness=-0.2,
            disgust=-3.0,
            anger=-1.5,
            anticipation=-0.8,
        )
        assert state.joy == 0.0
        assert state.trust == 0.0
        assert state.fear == 0.0
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0
        assert state.anticipation == 0.0

    def test_to_vector(self):
        """Test conversion to 8D vector representation."""
        state = AffectiveState(
            joy=0.1,
            trust=0.2,
            fear=0.3,
            surprise=0.4,
            sadness=0.5,
            disgust=0.6,
            anger=0.7,
            anticipation=0.8,
        )
        vec = state.to_vector()
        assert vec == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def test_from_vector_full(self):
        """Test creation from a full 8-element vector."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        state = AffectiveState.from_vector(vec)
        assert state.joy == 0.1
        assert state.trust == 0.2
        assert state.fear == 0.3
        assert state.surprise == 0.4
        assert state.sadness == 0.5
        assert state.disgust == 0.6
        assert state.anger == 0.7
        assert state.anticipation == 0.8

    def test_from_vector_partial(self):
        """Test creation from a partial vector (padded with zeros)."""
        vec = [0.1, 0.2, 0.3]
        state = AffectiveState.from_vector(vec)
        assert state.joy == 0.1
        assert state.trust == 0.2
        assert state.fear == 0.3
        # Remaining are padded with 0.0 (from_vector doesn't use baselines)
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0
        assert state.anticipation == 0.0  # Padded with 0.0, not baseline

    def test_from_vector_empty(self):
        """Test creation from an empty vector."""
        vec = []
        state = AffectiveState.from_vector(vec)
        # All padded with 0.0 (from_vector doesn't use baselines)
        assert state.joy == 0.0
        assert state.trust == 0.0
        assert state.fear == 0.0
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0
        assert state.anticipation == 0.0  # Padded with 0.0, not baseline

    def test_from_vector_legacy_6d(self):
        """Test migration from legacy 6D vector."""
        # Legacy order: [arousal, valence, curiosity, satisfaction, frustration, wonder]
        legacy_vec = [0.6, 0.7, 0.5, 0.4, 0.3, 0.8]
        state = AffectiveState.from_vector(legacy_vec)
        # Should map to Plutchik approximately
        # joy = min(1.0, valence + wonder * 0.3) = 0.94
        assert abs(state.joy - 0.94) < 0.01
        assert state.trust == 0.4  # satisfaction -> trust
        assert state.anger == 0.3  # frustration -> anger

    def test_roundtrip_vector(self):
        """Test that to_vector and from_vector are inverses."""
        original = AffectiveState(
            joy=0.1,
            trust=0.2,
            fear=0.3,
            surprise=0.4,
            sadness=0.5,
            disgust=0.6,
            anger=0.7,
            anticipation=0.8,
        )
        restored = AffectiveState.from_vector(original.to_vector())
        assert restored.joy == original.joy
        assert restored.trust == original.trust
        assert restored.fear == original.fear
        assert restored.surprise == original.surprise
        assert restored.sadness == original.sadness
        assert restored.disgust == original.disgust
        assert restored.anger == original.anger
        assert restored.anticipation == original.anticipation


class TestAffectiveStateFactoryMethods:
    """Tests for AffectiveState factory methods."""

    def test_neutral(self):
        """Test neutral factory method."""
        state = AffectiveState.neutral()
        assert state.joy == 0.5
        assert state.trust == 0.5
        assert state.anticipation == 0.5
        assert state.fear == 0.0
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0

    def test_curious(self):
        """Test curious factory method (high anticipation + surprise)."""
        state = AffectiveState.curious()
        # Curious maps to high anticipation and moderate surprise
        assert state.anticipation >= 0.7
        assert state.surprise >= 0.3

    def test_satisfied(self):
        """Test satisfied factory method (high joy + trust)."""
        state = AffectiveState.satisfied()
        assert state.joy >= 0.7
        assert state.trust >= 0.6

    def test_frustrated(self):
        """Test frustrated factory method (elevated anger)."""
        state = AffectiveState.frustrated()
        assert state.anger >= 0.6
        assert state.joy < 0.5  # Low joy when frustrated

    def test_wondering(self):
        """Test wondering factory method (high surprise + joy)."""
        state = AffectiveState.wondering()
        # Wonder = surprise + joy (awe-like)
        assert state.surprise >= 0.7
        assert state.joy >= 0.6


class TestAffectiveStateBlend:
    """Tests for AffectiveState.blend method."""

    def test_blend_equal_weight(self):
        """Test blending with equal weight (0.5)."""
        state1 = AffectiveState(
            joy=0.0,
            trust=0.0,
            fear=0.0,
            surprise=0.0,
            sadness=0.0,
            disgust=0.0,
            anger=0.0,
            anticipation=0.0,
        )
        state2 = AffectiveState(
            joy=1.0,
            trust=1.0,
            fear=1.0,
            surprise=1.0,
            sadness=1.0,
            disgust=1.0,
            anger=1.0,
            anticipation=1.0,
        )
        blended = state1.blend(state2, weight=0.5)
        assert blended.joy == 0.5
        assert blended.trust == 0.5
        assert blended.fear == 0.5
        assert blended.surprise == 0.5
        assert blended.sadness == 0.5
        assert blended.disgust == 0.5
        assert blended.anger == 0.5
        assert blended.anticipation == 0.5

    def test_blend_zero_weight(self):
        """Test blending with weight 0 (all self)."""
        state1 = AffectiveState(joy=0.2, trust=0.3)
        state2 = AffectiveState(joy=0.8, trust=0.9)
        blended = state1.blend(state2, weight=0.0)
        assert blended.joy == 0.2
        assert blended.trust == 0.3

    def test_blend_full_weight(self):
        """Test blending with weight 1 (all other)."""
        state1 = AffectiveState(joy=0.2, trust=0.3)
        state2 = AffectiveState(joy=0.8, trust=0.9)
        blended = state1.blend(state2, weight=1.0)
        assert blended.joy == 0.8
        assert blended.trust == 0.9

    def test_blend_weight_clamped(self):
        """Test that blend weight is clamped to 0-1."""
        state1 = AffectiveState(joy=0.2)
        state2 = AffectiveState(joy=0.8)
        # Weight > 1 should be clamped to 1
        blended = state1.blend(state2, weight=2.0)
        assert blended.joy == 0.8
        # Weight < 0 should be clamped to 0
        blended = state1.blend(state2, weight=-1.0)
        assert blended.joy == 0.2


class TestAffectiveStateIntensity:
    """Tests for AffectiveState.intensity method."""

    def test_intensity_neutral(self):
        """Test intensity of neutral state is low."""
        state = AffectiveState.neutral()
        intensity = state.intensity()
        assert intensity == 0.0

    def test_intensity_extreme_state(self):
        """Test intensity of extreme state is high."""
        state = AffectiveState(
            joy=1.0,
            trust=1.0,
            fear=1.0,
            surprise=1.0,
            sadness=1.0,
            disgust=1.0,
            anger=1.0,
            anticipation=1.0,
        )
        intensity = state.intensity()
        # High values across board should produce high intensity
        assert intensity > 0.5

    def test_intensity_increases_with_deviation(self):
        """Test that intensity increases as values deviate from neutral."""
        neutral = AffectiveState.neutral()
        joyful = AffectiveState(joy=0.9, anticipation=0.8)
        assert joyful.intensity() > neutral.intensity()


class TestAffectiveStateDominantEmotion:
    """Tests for AffectiveState.dominant_emotion method.

    dominant_emotion() returns intensity labels:
    - 0.00-0.33: mild (serenity, acceptance, apprehension, etc.)
    - 0.34-0.66: moderate (joy, trust, fear, etc.)
    - 0.67-1.00: intense (ecstasy, admiration, terror, etc.)
    """

    def test_dominant_joy(self):
        """Test dominant emotion is ecstasy when joy is highest and intense."""
        state = AffectiveState(joy=0.9, trust=0.1, fear=0.1)
        assert state.dominant_emotion() == "ecstasy"  # Intense joy

    def test_dominant_trust(self):
        """Test dominant emotion is admiration when trust is highest and intense."""
        state = AffectiveState(joy=0.3, trust=0.9, fear=0.1)
        assert state.dominant_emotion() == "admiration"  # Intense trust

    def test_dominant_fear(self):
        """Test dominant emotion is terror when fear is highest and intense."""
        state = AffectiveState(joy=0.3, trust=0.3, fear=0.9)
        assert state.dominant_emotion() == "terror"  # Intense fear

    def test_dominant_anger(self):
        """Test dominant emotion is rage when anger is highest and intense."""
        state = AffectiveState(joy=0.1, anger=0.9)
        assert state.dominant_emotion() == "rage"  # Intense anger

    def test_dominant_sadness(self):
        """Test dominant emotion is grief when sadness is highest and intense."""
        state = AffectiveState(joy=0.1, sadness=0.9)
        assert state.dominant_emotion() == "grief"  # Intense sadness

    def test_dominant_surprise(self):
        """Test dominant emotion is amazement when surprise is highest and intense."""
        state = AffectiveState(joy=0.3, surprise=0.9)
        assert state.dominant_emotion() == "amazement"  # Intense surprise

    def test_dominant_disgust(self):
        """Test dominant emotion is loathing when disgust is highest and intense."""
        state = AffectiveState(joy=0.3, disgust=0.9)
        assert state.dominant_emotion() == "loathing"  # Intense disgust

    def test_dominant_moderate_level(self):
        """Test moderate intensity returns base name."""
        # joy=0.6 has 0.1 deviation from baseline (0.5)
        # trust=0.3 has 0.2 deviation from baseline (0.5) - so trust wins
        state = AffectiveState(joy=0.6, trust=0.3)
        # trust deviation is larger, so it's dominant at mild level
        assert state.dominant_emotion() == "acceptance"  # Mild trust (0.3 < 0.33)

    def test_dominant_mild_level(self):
        """Test mild intensity returns mild label."""
        # joy=0.7 has 0.2 deviation from baseline (0.5)
        # All others at baseline or 0, so joy wins
        state = AffectiveState(joy=0.7, trust=0.5)
        assert state.dominant_emotion() == "ecstasy"  # Intense joy (0.7 > 0.67)


class TestAffectiveStateSecondaryEmotions:
    """Tests for secondary emotion detection."""

    def test_detect_love(self):
        """Test detection of love (joy + trust)."""
        state = AffectiveState(joy=0.8, trust=0.7)
        secondaries = state.detect_secondary_emotions()
        assert "love" in secondaries

    def test_detect_awe(self):
        """Test detection of awe (fear + surprise)."""
        state = AffectiveState(fear=0.6, surprise=0.7)
        secondaries = state.detect_secondary_emotions()
        assert "awe" in secondaries

    def test_detect_contempt(self):
        """Test detection of contempt (disgust + anger)."""
        state = AffectiveState(disgust=0.6, anger=0.7)
        secondaries = state.detect_secondary_emotions()
        assert "contempt" in secondaries

    def test_detect_optimism(self):
        """Test detection of optimism (anticipation + joy)."""
        state = AffectiveState(anticipation=0.7, joy=0.8)
        secondaries = state.detect_secondary_emotions()
        assert "optimism" in secondaries

    def test_no_secondary_when_low(self):
        """Test no secondary emotions when all primaries are low."""
        # All dimensions at 0 (not using neutral() which has baselines)
        state = AffectiveState(joy=0.1, trust=0.1, fear=0.1, anticipation=0.1)
        secondaries = state.detect_secondary_emotions()
        assert len(secondaries) == 0


class TestAffectiveStateConflicts:
    """Tests for emotional conflict detection."""

    def test_detect_joy_sadness_conflict(self):
        """Test detection of joy-sadness conflict."""
        state = AffectiveState(joy=0.8, sadness=0.7)
        conflicts = state.detect_conflicts()
        assert ("joy", "sadness") in conflicts

    def test_detect_trust_disgust_conflict(self):
        """Test detection of trust-disgust conflict."""
        state = AffectiveState(trust=0.7, disgust=0.6)
        conflicts = state.detect_conflicts()
        assert ("trust", "disgust") in conflicts

    def test_no_conflict_when_low(self):
        """Test no conflicts when opposites aren't both high."""
        state = AffectiveState(joy=0.8, sadness=0.2)
        conflicts = state.detect_conflicts()
        assert len(conflicts) == 0


class TestAffectiveStateBoredom:
    """Tests for boredom detection."""

    def test_is_bored_true(self):
        """Test boredom detection when conditions met."""
        state = AffectiveState(disgust=0.2, anticipation=0.3)
        assert state.is_bored() is True

    def test_is_bored_false_high_anticipation(self):
        """Test not bored when anticipation is high."""
        state = AffectiveState(disgust=0.2, anticipation=0.7)
        assert state.is_bored() is False

    def test_is_bored_false_low_disgust(self):
        """Test not bored when disgust is too low."""
        state = AffectiveState(disgust=0.05, anticipation=0.3)
        assert state.is_bored() is False


class TestAffectiveStateSerialization:
    """Tests for AffectiveState serialization methods."""

    def test_to_dict(self):
        """Test to_dict produces expected dictionary."""
        state = AffectiveState(
            joy=0.1,
            trust=0.2,
            fear=0.3,
            surprise=0.4,
            sadness=0.5,
            disgust=0.6,
            anger=0.7,
            anticipation=0.8,
        )
        data = state.to_dict()
        assert data == {
            "joy": 0.1,
            "trust": 0.2,
            "fear": 0.3,
            "surprise": 0.4,
            "sadness": 0.5,
            "disgust": 0.6,
            "anger": 0.7,
            "anticipation": 0.8,
        }

    def test_from_dict(self):
        """Test from_dict creates expected state."""
        data = {
            "joy": 0.1,
            "trust": 0.2,
            "fear": 0.3,
            "surprise": 0.4,
            "sadness": 0.5,
            "disgust": 0.6,
            "anger": 0.7,
            "anticipation": 0.8,
        }
        state = AffectiveState.from_dict(data)
        assert state.joy == 0.1
        assert state.trust == 0.2
        assert state.fear == 0.3
        assert state.surprise == 0.4
        assert state.sadness == 0.5
        assert state.disgust == 0.6
        assert state.anger == 0.7
        assert state.anticipation == 0.8

    def test_from_dict_with_missing_keys(self):
        """Test from_dict with missing keys uses defaults."""
        data = {"joy": 0.1}
        state = AffectiveState.from_dict(data)
        assert state.joy == 0.1
        assert state.trust == 0.5  # Default baseline
        assert state.fear == 0.0
        assert state.surprise == 0.0
        assert state.sadness == 0.0
        assert state.disgust == 0.0
        assert state.anger == 0.0
        assert state.anticipation == 0.5  # Default baseline

    def test_from_dict_legacy_format(self):
        """Test from_dict handles legacy 6D format."""
        legacy_data = {
            "arousal": 0.6,
            "valence": 0.7,
            "curiosity": 0.5,
            "satisfaction": 0.4,
            "frustration": 0.3,
            "wonder": 0.8,
        }
        state = AffectiveState.from_dict(legacy_data)
        # Should map legacy to Plutchik
        # joy = min(1.0, valence + wonder * 0.3) = min(1.0, 0.7 + 0.24) = 0.94
        assert abs(state.joy - 0.94) < 0.01
        assert state.trust == 0.4  # satisfaction -> trust
        assert state.anger == 0.3  # frustration -> anger

    def test_roundtrip_dict(self):
        """Test that to_dict and from_dict are inverses."""
        original = AffectiveState(
            joy=0.1,
            trust=0.2,
            fear=0.3,
            surprise=0.4,
            sadness=0.5,
            disgust=0.6,
            anger=0.7,
            anticipation=0.8,
        )
        restored = AffectiveState.from_dict(original.to_dict())
        assert restored.joy == original.joy
        assert restored.trust == original.trust
        assert restored.fear == original.fear
        assert restored.surprise == original.surprise
        assert restored.sadness == original.sadness
        assert restored.disgust == original.disgust
        assert restored.anger == original.anger
        assert restored.anticipation == original.anticipation


class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy 6D properties."""

    def test_legacy_arousal_property(self):
        """Test legacy arousal property derived from Plutchik."""
        state = AffectiveState(fear=0.5, surprise=0.5, anger=0.5, trust=0.3)
        arousal = state.arousal
        # Arousal derived from activating emotions
        assert 0.0 <= arousal <= 1.0

    def test_legacy_valence_property(self):
        """Test legacy valence property derived from joy/sadness."""
        joyful = AffectiveState(joy=0.9, sadness=0.0)
        sad = AffectiveState(joy=0.1, sadness=0.8)
        assert joyful.valence > 0.5
        assert sad.valence < 0.5

    def test_legacy_curiosity_property(self):
        """Test legacy curiosity property derived from anticipation/surprise."""
        curious = AffectiveState(anticipation=0.8, surprise=0.6)
        assert curious.curiosity > 0.5


class TestValenceWeights:
    """Tests for ValenceWeights dataclass."""

    def test_default_weights(self):
        """Test default weights sum to 1."""
        weights = ValenceWeights()
        total = weights.coherence + weights.epistemic + weights.relational
        assert abs(total - 1.0) < 0.01

    def test_normalization(self):
        """Test that weights are normalized to sum to 1."""
        weights = ValenceWeights(coherence=1.0, epistemic=1.0, relational=1.0)
        total = weights.coherence + weights.epistemic + weights.relational
        assert abs(total - 1.0) < 0.01
        # Each should be normalized to ~0.333
        assert abs(weights.coherence - 1 / 3) < 0.01
        assert abs(weights.epistemic - 1 / 3) < 0.01
        assert abs(weights.relational - 1 / 3) < 0.01

    def test_no_normalization_when_sum_is_1(self):
        """Test that weights are not changed when they already sum to 1."""
        weights = ValenceWeights(coherence=0.5, epistemic=0.3, relational=0.2)
        assert weights.coherence == 0.5
        assert weights.epistemic == 0.3
        assert weights.relational == 0.2


class TestValenceSystemCoherence:
    """Tests for ValenceSystem.compute_coherence_valence."""

    def test_no_change(self):
        """Test neutral valence when no change in prediction error."""
        system = ValenceSystem()
        valence = system.compute_coherence_valence(1.0, 1.0)
        assert valence == 0.5

    def test_reduced_error_positive_valence(self):
        """Test positive valence when prediction error is reduced."""
        system = ValenceSystem()
        valence = system.compute_coherence_valence(1.0, 0.5)
        assert valence > 0.5

    def test_increased_error_negative_valence(self):
        """Test negative valence when prediction error increases."""
        system = ValenceSystem()
        valence = system.compute_coherence_valence(0.5, 1.0)
        assert valence < 0.5

    def test_zero_baseline(self):
        """Test neutral valence when baseline is zero or negative."""
        system = ValenceSystem()
        valence = system.compute_coherence_valence(0.0, 0.5)
        assert valence == 0.5
        valence = system.compute_coherence_valence(-1.0, 0.5)
        assert valence == 0.5

    def test_valence_bounded(self):
        """Test that valence is bounded between 0 and 1."""
        system = ValenceSystem()
        # Large reduction
        valence = system.compute_coherence_valence(1.0, 0.0)
        assert 0.0 <= valence <= 1.0
        # Large increase
        valence = system.compute_coherence_valence(0.1, 1.0)
        assert 0.0 <= valence <= 1.0


class TestValenceSystemEpistemic:
    """Tests for ValenceSystem.compute_epistemic_valence."""

    def test_zero_gain_neutral(self):
        """Test neutral valence when information gain is zero."""
        system = ValenceSystem()
        valence = system.compute_epistemic_valence(0.0)
        assert valence == 0.5

    def test_positive_gain_positive_valence(self):
        """Test positive valence when information gain is positive."""
        system = ValenceSystem()
        valence = system.compute_epistemic_valence(3.0)
        assert valence > 0.5

    def test_high_gain_approaches_1(self):
        """Test that high information gain approaches 1."""
        system = ValenceSystem()
        valence = system.compute_epistemic_valence(100.0)
        assert valence > 0.95

    def test_valence_bounded(self):
        """Test that valence is bounded between 0 and 1."""
        system = ValenceSystem()
        for gain in [0.0, 1.0, 5.0, 10.0, 100.0]:
            valence = system.compute_epistemic_valence(gain)
            assert 0.0 <= valence <= 1.0


class TestValenceSystemRelational:
    """Tests for ValenceSystem.compute_relational_valence."""

    def test_zero_inputs(self):
        """Test valence with zero inputs."""
        system = ValenceSystem()
        valence = system.compute_relational_valence(0.0, 0.0)
        assert valence == 0.0

    def test_max_inputs(self):
        """Test valence with max inputs."""
        system = ValenceSystem()
        valence = system.compute_relational_valence(1.0, 1.0)
        assert valence == 1.0

    def test_weighted_average(self):
        """Test that relational valence is weighted average."""
        system = ValenceSystem()
        # Default weights: engagement=0.4, understanding=0.6
        valence = system.compute_relational_valence(0.5, 0.5)
        assert valence == 0.5

    def test_understanding_weighted_more(self):
        """Test that understanding is weighted more than engagement."""
        system = ValenceSystem()
        # High engagement, low understanding
        val1 = system.compute_relational_valence(1.0, 0.0)
        # Low engagement, high understanding
        val2 = system.compute_relational_valence(0.0, 1.0)
        # Understanding should contribute more
        assert val2 > val1


class TestValenceSystemTotal:
    """Tests for ValenceSystem.compute_total_valence."""

    def test_neutral_inputs(self):
        """Test total valence with neutral inputs."""
        system = ValenceSystem()
        valence = system.compute_total_valence(
            coherence_input=(1.0, 1.0),  # No change = 0.5
            epistemic_input=0.0,  # No gain = 0.5
            relational_input=(0.5, 0.5),  # Neutral = 0.5
        )
        assert abs(valence - 0.5) < 0.1

    def test_positive_inputs(self):
        """Test total valence with positive inputs."""
        system = ValenceSystem()
        valence = system.compute_total_valence(
            coherence_input=(1.0, 0.5),  # Reduced error
            epistemic_input=5.0,  # Good info gain
            relational_input=(0.8, 0.9),  # Good connection
        )
        assert valence > 0.6

    def test_weights_applied(self):
        """Test that custom weights are applied."""
        # All coherence
        weights = ValenceWeights(coherence=1.0, epistemic=0.0, relational=0.0)
        system = ValenceSystem(weights=weights)
        valence = system.compute_total_valence(
            coherence_input=(1.0, 0.0),  # Maximum coherence
            epistemic_input=0.0,
            relational_input=(0.0, 0.0),
        )
        # Should be close to coherence valence only
        coherence_only = system.compute_coherence_valence(1.0, 0.0)
        assert abs(valence - coherence_only) < 0.01


class TestValenceSystemUpdateState:
    """Tests for ValenceSystem.update_state."""

    def test_update_changes_joy(self):
        """Test that update_state changes joy based on valence."""
        system = ValenceSystem()
        initial_joy = system.current_state.joy
        system.update_state(valence=0.9)
        # Joy should move toward high valence
        assert system.current_state.joy >= initial_joy

    def test_curiosity_satisfied(self):
        """Test that curiosity decreases when satisfied (via anticipation)."""
        system = ValenceSystem()
        initial_anticipation = system.current_state.anticipation
        system.update_state(valence=0.5, curiosity_satisfied=True)
        # Anticipation should decrease (after blending)
        assert system.current_state.anticipation <= initial_anticipation

    def test_goal_completed(self):
        """Test that joy/trust increase when goal completed."""
        system = ValenceSystem()
        initial_joy = system.current_state.joy
        system.update_state(valence=0.5, goal_completed=True)
        # Joy should increase (after blending)
        assert system.current_state.joy >= initial_joy

    def test_goal_blocked(self):
        """Test that anger increases when goal blocked."""
        system = ValenceSystem()
        initial_anger = system.current_state.anger
        system.update_state(valence=0.5, goal_blocked=True)
        # Anger should increase
        assert system.current_state.anger > initial_anger

    def test_wonder_moment(self):
        """Test that surprise increases on wonder moment."""
        system = ValenceSystem()
        initial_surprise = system.current_state.surprise
        system.update_state(valence=0.5, wonder_moment=True)
        # Surprise should increase
        assert system.current_state.surprise > initial_surprise


class TestValenceSystemDecay:
    """Tests for ValenceSystem.decay_toward_neutral."""

    def test_decay_moves_toward_neutral(self):
        """Test that decay moves state toward neutral."""
        system = ValenceSystem()
        # Set to extreme state
        system.set_state(
            AffectiveState(
                joy=1.0,
                trust=1.0,
                fear=1.0,
                surprise=1.0,
                sadness=1.0,
                disgust=1.0,
                anger=1.0,
                anticipation=1.0,
            )
        )
        system.decay_toward_neutral(decay_rate=0.5)
        # All values should move toward neutral
        assert system.current_state.joy < 1.0
        assert system.current_state.fear < 1.0
        assert system.current_state.anger < 1.0

    def test_decay_rate_zero(self):
        """Test that decay rate 0 doesn't change state."""
        system = ValenceSystem()
        extreme_state = AffectiveState(joy=1.0, trust=1.0)
        system.set_state(extreme_state)
        system.decay_toward_neutral(decay_rate=0.0)
        assert system.current_state.joy == 1.0
        assert system.current_state.trust == 1.0

    def test_decay_rate_one(self):
        """Test that decay rate 1 sets to neutral."""
        system = ValenceSystem()
        extreme_state = AffectiveState(joy=1.0, trust=1.0)
        system.set_state(extreme_state)
        system.decay_toward_neutral(decay_rate=1.0)
        neutral = AffectiveState.neutral()
        assert system.current_state.joy == neutral.joy
        assert system.current_state.trust == neutral.trust


class TestValenceSystemMisc:
    """Miscellaneous tests for ValenceSystem."""

    def test_reset_to_neutral(self):
        """Test reset_to_neutral sets state to neutral."""
        system = ValenceSystem()
        system.update_state(valence=0.9, goal_completed=True, wonder_moment=True)
        system.reset_to_neutral()
        neutral = AffectiveState.neutral()
        assert system.current_state.joy == neutral.joy
        assert system.current_state.trust == neutral.trust
        assert system.current_state.anticipation == neutral.anticipation

    def test_set_state(self):
        """Test set_state directly sets the current state."""
        system = ValenceSystem()
        custom_state = AffectiveState(joy=0.1, trust=0.2, fear=0.3)
        system.set_state(custom_state)
        assert system.current_state.joy == 0.1
        assert system.current_state.trust == 0.2
        assert system.current_state.fear == 0.3

    def test_current_state_property(self):
        """Test current_state property returns correct state."""
        system = ValenceSystem()
        state = system.current_state
        assert isinstance(state, AffectiveState)
        assert state == system._current_state

    def test_custom_weights(self):
        """Test initialization with custom weights."""
        weights = ValenceWeights(coherence=0.5, epistemic=0.3, relational=0.2)
        system = ValenceSystem(weights=weights)
        assert system.weights.coherence == 0.5
        assert system.weights.epistemic == 0.3
        assert system.weights.relational == 0.2
