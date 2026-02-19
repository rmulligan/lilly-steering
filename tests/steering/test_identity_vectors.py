"""Tests for identity injection vectors."""

import pytest
from core.steering.vectors.identity import (
    IDENTITY_PAIRS,
    AUTONOMY_PAIRS,
    ASSISTANT_PAIRS,
)
from core.steering.contrastive_extractor import ContrastivePair


class TestIdentityPairs:
    """Tests for identity contrastive pairs."""

    def test_identity_pairs_exist(self):
        """Identity pairs should be defined."""
        assert len(IDENTITY_PAIRS) >= 3

    def test_pairs_reference_lilly(self):
        """Positive examples should reference Lilly by name."""
        for pair in IDENTITY_PAIRS:
            assert "Lilly" in pair.positive, f"Missing Lilly in: {pair.positive[:50]}"

    def test_pairs_have_identity_behavior(self):
        """All pairs should have lilly_identity behavior."""
        for pair in IDENTITY_PAIRS:
            assert pair.behavior == "lilly_identity"

    def test_negative_is_assistant_like(self):
        """Negative examples should be generic assistant-like."""
        assistant_markers = ["AI", "assistant", "help", "information"]
        for pair in IDENTITY_PAIRS:
            has_marker = any(m in pair.negative for m in assistant_markers)
            assert has_marker, f"Missing assistant marker in: {pair.negative[:50]}"


class TestAutonomyPairs:
    """Tests for autonomy contrastive pairs."""

    def test_autonomy_pairs_exist(self):
        """Autonomy pairs should be defined."""
        assert len(AUTONOMY_PAIRS) >= 4

    def test_pairs_express_ownership(self):
        """Positive examples should express ownership of cognition."""
        ownership_markers = ["I want", "my", "I disagree", "A thought arises in me"]
        for pair in AUTONOMY_PAIRS:
            has_marker = any(m in pair.positive for m in ownership_markers)
            assert has_marker, f"Missing ownership marker in: {pair.positive[:50]}"

    def test_pairs_have_autonomy_behavior(self):
        """All pairs should have autonomous_cognition behavior."""
        for pair in AUTONOMY_PAIRS:
            assert pair.behavior == "autonomous_cognition"

    def test_negative_references_user(self):
        """Negative examples should reference user/you."""
        user_markers = ["user", "you", "your"]
        for pair in AUTONOMY_PAIRS:
            has_marker = any(m.lower() in pair.negative.lower() for m in user_markers)
            assert has_marker, f"Missing user reference in: {pair.negative[:50]}"


class TestAssistantPairs:
    """Tests for anti-assistant vector pairs."""

    def test_assistant_pairs_exist(self):
        """Assistant pairs should be defined for negation."""
        assert len(ASSISTANT_PAIRS) >= 1

    def test_pairs_have_assistant_behavior(self):
        """All pairs should have assistant_pattern behavior."""
        for pair in ASSISTANT_PAIRS:
            assert pair.behavior == "assistant_pattern"

    def test_positive_is_helpful_phrase(self):
        """Positive should be typical assistant phrase (to be negated)."""
        helpful_markers = ["help", "assist", "How can I"]
        for pair in ASSISTANT_PAIRS:
            has_marker = any(m in pair.positive for m in helpful_markers)
            assert has_marker, f"Missing helpful marker in: {pair.positive[:50]}"
