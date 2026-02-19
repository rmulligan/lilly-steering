"""Tests for dry wit humor contrastive pairs."""

import pytest

from core.steering.humor_pairs import (
    WRY_OBSERVATION_PAIR,
    DEADPAN_SELF_AWARENESS_PAIR,
    GENTLE_IRONY_PAIR,
    get_all_humor_pairs,
)


class TestHumorPairs:
    """Tests for humor contrastive pair definitions."""

    @pytest.mark.parametrize(
        "pair",
        [WRY_OBSERVATION_PAIR, DEADPAN_SELF_AWARENESS_PAIR, GENTLE_IRONY_PAIR],
        ids=["wry_observation", "deadpan_self_awareness", "gentle_irony"],
    )
    def test_pair_has_positive_and_negative(self, pair):
        """Each pair should have both positive and negative components of sufficient length."""
        assert "positive" in pair
        assert "negative" in pair
        assert len(pair["positive"]) > 50
        assert len(pair["negative"]) > 50

    def test_get_all_pairs_returns_three(self):
        """Should return all three humor pairs."""
        pairs = get_all_humor_pairs()
        assert len(pairs) == 3
        assert "wry_observation" in pairs
        assert "deadpan_self_awareness" in pairs
        assert "gentle_irony" in pairs

    def test_positive_examples_have_wit_markers(self):
        """Positive examples should contain understated/ironic language."""
        pairs = get_all_humor_pairs()
        for name, pair in pairs.items():
            # Positive should NOT explain the humor
            assert "funny" not in pair["positive"].lower()
            assert "joke" not in pair["positive"].lower()
            assert "humor" not in pair["positive"].lower()

    def test_negative_examples_are_earnest(self):
        """Negative examples should be straightforward/earnest."""
        pairs = get_all_humor_pairs()
        for name, pair in pairs.items():
            # Negative should be more formal/academic
            assert "interesting" in pair["negative"].lower() or "important" in pair["negative"].lower() or "meaningful" in pair["negative"].lower()
