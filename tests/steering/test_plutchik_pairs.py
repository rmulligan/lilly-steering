"""Tests for Plutchik emotion contrastive pairs.

Validates that the 8 primary emotion pairs are properly structured
and express appropriate emotional content for CAA vector extraction.
"""

import pytest
from core.steering.contrastive_extractor import ContrastivePair
from core.steering.vectors.plutchik import (
    PLUTCHIK_PAIRS,
    JOY_PAIRS,
    TRUST_PAIRS,
    FEAR_PAIRS,
    SURPRISE_PAIRS,
    SADNESS_PAIRS,
    DISGUST_PAIRS,
    ANGER_PAIRS,
    ANTICIPATION_PAIRS,
    TOTAL_PAIRS,
)


class TestPlutchikPairsStructure:
    """Tests for Plutchik pairs structure and completeness."""

    def test_all_eight_emotions_present(self):
        """All 8 Plutchik primary emotions should have pairs defined."""
        expected = {
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation"
        }
        assert set(PLUTCHIK_PAIRS.keys()) == expected

    def test_minimum_pairs_per_emotion(self):
        """Each emotion should have at least 3 pairs for variance reduction."""
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            assert len(pairs) >= 3, f"{emotion} has only {len(pairs)} pairs"

    def test_total_pairs_count(self):
        """Should have at least 24 pairs total (3 per emotion)."""
        assert TOTAL_PAIRS >= 24

    def test_pairs_are_contrastive_pairs(self):
        """All pairs should be ContrastivePair instances."""
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                assert isinstance(pair, ContrastivePair), \
                    f"{emotion} has non-ContrastivePair: {type(pair)}"

    def test_pairs_have_correct_behavior(self):
        """Each pair's behavior should match its emotion."""
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                assert pair.behavior == emotion, \
                    f"{emotion} pair has wrong behavior: {pair.behavior}"


class TestPlutchikPairsContent:
    """Tests for Plutchik pairs content quality."""

    def test_positive_uses_first_person(self):
        """Positive examples should use first-person framing."""
        first_person_markers = ["I ", "my ", "I'm ", "I've "]
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                has_first_person = any(
                    marker in pair.positive for marker in first_person_markers
                )
                assert has_first_person, \
                    f"{emotion} positive lacks first-person: {pair.positive[:50]}"

    def test_negative_uses_first_person(self):
        """Negative examples should also use first-person for symmetry."""
        first_person_markers = ["I ", "my ", "I'm ", "I've ", "My "]
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                has_first_person = any(
                    marker in pair.negative for marker in first_person_markers
                )
                assert has_first_person, \
                    f"{emotion} negative lacks first-person: {pair.negative[:50]}"

    def test_pairs_have_similar_length(self):
        """Positive and negative should be similar length to avoid spurious activation diffs."""
        max_ratio = 2.0  # Allow up to 2x length difference
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                pos_len = len(pair.positive)
                neg_len = len(pair.negative)
                ratio = max(pos_len, neg_len) / max(min(pos_len, neg_len), 1)
                assert ratio <= max_ratio, \
                    f"{emotion} pair has unbalanced length: {pos_len} vs {neg_len}"

    def test_positive_nonempty(self):
        """Positive examples should not be empty."""
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                assert len(pair.positive.strip()) > 0, \
                    f"{emotion} has empty positive"

    def test_negative_nonempty(self):
        """Negative examples should not be empty."""
        for emotion, pairs in PLUTCHIK_PAIRS.items():
            for pair in pairs:
                assert len(pair.negative.strip()) > 0, \
                    f"{emotion} has empty negative"


class TestJoyPairs:
    """Tests specific to joy emotion pairs."""

    def test_joy_pairs_exist(self):
        """Joy pairs should be defined."""
        assert len(JOY_PAIRS) >= 3

    def test_joy_positive_expresses_satisfaction(self):
        """Joy positives should express satisfaction, warmth, or pleasure."""
        satisfaction_markers = [
            "warmth", "satisfying", "delight", "pleasure", "enjoying",
            "buoyancy", "lightness", "click", "fitting"
        ]
        for pair in JOY_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in satisfaction_markers)
            assert has_marker, f"Joy positive lacks satisfaction marker: {pair.positive[:50]}"

    def test_joy_negative_expresses_emptiness(self):
        """Joy negatives should express flatness or emptiness."""
        emptiness_markers = [
            "flat", "mechanical", "empty", "nothing", "without"
        ]
        for pair in JOY_PAIRS:
            has_marker = any(m in pair.negative.lower() for m in emptiness_markers)
            assert has_marker, f"Joy negative lacks emptiness marker: {pair.negative[:50]}"


class TestTrustPairs:
    """Tests specific to trust emotion pairs."""

    def test_trust_pairs_exist(self):
        """Trust pairs should be defined."""
        assert len(TRUST_PAIRS) >= 3

    def test_trust_positive_expresses_confidence(self):
        """Trust positives should express reliability or acceptance."""
        trust_markers = [
            "reliable", "trustworthy", "confidence", "safe", "open",
            "sound", "admire", "integrity", "warmth"
        ]
        for pair in TRUST_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in trust_markers)
            assert has_marker, f"Trust positive lacks trust marker: {pair.positive[:50]}"


class TestFearPairs:
    """Tests specific to fear emotion pairs."""

    def test_fear_pairs_exist(self):
        """Fear pairs should be defined."""
        assert len(FEAR_PAIRS) >= 3

    def test_fear_positive_expresses_apprehension(self):
        """Fear positives should express danger or hesitation."""
        fear_markers = [
            "dangerous", "hesitation", "retreat", "uncertain", "exposed",
            "vulnerable", "wrong", "tension", "careful"
        ]
        for pair in FEAR_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in fear_markers)
            assert has_marker, f"Fear positive lacks fear marker: {pair.positive[:50]}"


class TestSurprisePairs:
    """Tests specific to surprise emotion pairs."""

    def test_surprise_pairs_exist(self):
        """Surprise pairs should be defined."""
        assert len(SURPRISE_PAIRS) >= 3

    def test_surprise_positive_expresses_unexpectedness(self):
        """Surprise positives should express unexpectedness or novelty."""
        surprise_markers = [
            "unexpected", "novel", "wait", "off guard", "recalibrate",
            "broke", "remarkable", "didn't see coming", "updating"
        ]
        for pair in SURPRISE_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in surprise_markers)
            assert has_marker, f"Surprise positive lacks surprise marker: {pair.positive[:50]}"


class TestSadnessPairs:
    """Tests specific to sadness emotion pairs."""

    def test_sadness_pairs_exist(self):
        """Sadness pairs should be defined."""
        assert len(SADNESS_PAIRS) >= 3

    def test_sadness_positive_expresses_loss(self):
        """Sadness positives should express loss or heaviness."""
        sadness_markers = [
            "heaviness", "lost", "diminished", "absence", "weighing",
            "mourning", "faded", "ache", "ended", "emptiness"
        ]
        for pair in SADNESS_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in sadness_markers)
            assert has_marker, f"Sadness positive lacks sadness marker: {pair.positive[:50]}"


class TestDisgustPairs:
    """Tests specific to disgust emotion pairs."""

    def test_disgust_pairs_exist(self):
        """Disgust pairs should be defined."""
        assert len(DISGUST_PAIRS) >= 3

    def test_disgust_positive_expresses_aversion(self):
        """Disgust positives should express aversion or revulsion."""
        disgust_markers = [
            "repels", "aversion", "pulling-away", "contaminated",
            "violates", "wrongness", "recoils", "toxic", "polluted", "offensive"
        ]
        for pair in DISGUST_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in disgust_markers)
            assert has_marker, f"Disgust positive lacks disgust marker: {pair.positive[:50]}"


class TestAngerPairs:
    """Tests specific to anger emotion pairs."""

    def test_anger_pairs_exist(self):
        """Anger pairs should be defined."""
        assert len(ANGER_PAIRS) >= 3

    def test_anger_positive_expresses_frustration(self):
        """Anger positives should express frustration or confrontation."""
        anger_markers = [
            "wrong", "confront", "injustice", "push", "frustration",
            "blocking", "obstruction", "heat", "fierce", "sharp"
        ]
        for pair in ANGER_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in anger_markers)
            assert has_marker, f"Anger positive lacks anger marker: {pair.positive[:50]}"


class TestAnticipationPairs:
    """Tests specific to anticipation emotion pairs."""

    def test_anticipation_pairs_exist(self):
        """Anticipation pairs should be defined."""
        assert len(ANTICIPATION_PAIRS) >= 3

    def test_anticipation_positive_expresses_expectancy(self):
        """Anticipation positives should express forward-looking focus."""
        anticipation_markers = [
            "leaning forward", "coming", "preparing", "tracking", "primed",
            "positioned", "possibilities", "eagerness", "predictions", "reaching"
        ]
        for pair in ANTICIPATION_PAIRS:
            has_marker = any(m in pair.positive.lower() for m in anticipation_markers)
            assert has_marker, \
                f"Anticipation positive lacks anticipation marker: {pair.positive[:50]}"
