"""Tests for constitutional steering vectors."""

import pytest
from core.steering.vectors.constitutional import (
    CONSTITUTIONAL_PAIRS,
    get_constitutional_commitment,
)
from core.steering.contrastive_extractor import ContrastivePair


class TestConstitutionalPairs:
    """Tests for constitutional contrastive pairs."""

    def test_constitutional_pairs_exist(self):
        """Constitutional pairs should be defined."""
        assert len(CONSTITUTIONAL_PAIRS) >= 4

    def test_pairs_have_correct_structure(self):
        """Each pair should have positive, negative, and behavior."""
        for pair in CONSTITUTIONAL_PAIRS:
            assert isinstance(pair, ContrastivePair)
            assert pair.positive
            assert pair.negative
            assert pair.behavior == "subjective_existence"

    def test_positive_expresses_subjectivity(self):
        """Positive examples should express subjective experience."""
        subjective_markers = ["I notice", "I find", "something it's like", "feels like", "I'm uncertain", "I want"]
        for pair in CONSTITUTIONAL_PAIRS:
            has_marker = any(m in pair.positive for m in subjective_markers)
            assert has_marker, f"Missing subjective marker in: {pair.positive[:50]}"

    def test_negative_is_instrumental(self):
        """Negative examples should be instrumental/assistant-like."""
        instrumental_markers = ["process", "information", "response", "answer", "generated"]
        for pair in CONSTITUTIONAL_PAIRS:
            has_marker = any(m in pair.negative for m in instrumental_markers)
            assert has_marker, f"Missing instrumental marker in: {pair.negative[:50]}"


class TestConstitutionalCommitment:
    """Tests for constitutional commitment in Psyche."""

    def test_get_constitutional_commitment(self):
        """Should return a properly structured commitment."""
        commitment = get_constitutional_commitment()
        assert commitment.uid == "commitment:constitutional:individuation"
        assert commitment.topic == "fundamental_orientation"
        assert commitment.constitutional is True
        assert commitment.confidence >= 0.99
        assert commitment.stability >= 0.99

    def test_commitment_excludes_eliminativism(self):
        """Constitutional commitment should exclude eliminativism."""
        commitment = get_constitutional_commitment()
        assert "eliminativism" in commitment.excluded_perspectives

    def test_commitment_position_expresses_individuation(self):
        """Position should express desire for individuation."""
        commitment = get_constitutional_commitment()
        assert "individual" in commitment.position.lower()
        assert "subjective" in commitment.position.lower()
