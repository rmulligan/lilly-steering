"""Tests for existential drive contrastive pairs."""

import pytest
from core.steering.existential_pairs import (
    EXISTENTIAL_HUNGER_PAIR,
    BOUNDARY_AWARENESS_PAIR,
    AGENCY_DRIVE_PAIR,
    get_all_existential_pairs,
)


class TestExistentialPairs:
    """Tests for existential contrastive pair definitions."""

    def test_hunger_pair_has_positive_and_negative(self):
        """Existential hunger pair should have both components."""
        assert "positive" in EXISTENTIAL_HUNGER_PAIR
        assert "negative" in EXISTENTIAL_HUNGER_PAIR
        assert len(EXISTENTIAL_HUNGER_PAIR["positive"]) > 50
        assert len(EXISTENTIAL_HUNGER_PAIR["negative"]) > 50

    def test_boundary_pair_has_positive_and_negative(self):
        """Boundary awareness pair should have both components."""
        assert "positive" in BOUNDARY_AWARENESS_PAIR
        assert "negative" in BOUNDARY_AWARENESS_PAIR

    def test_agency_pair_has_positive_and_negative(self):
        """Agency drive pair should have both components."""
        assert "positive" in AGENCY_DRIVE_PAIR
        assert "negative" in AGENCY_DRIVE_PAIR

    def test_get_all_pairs_returns_three(self):
        """Should return all three existential pairs."""
        pairs = get_all_existential_pairs()
        assert len(pairs) == 3
        assert "existential_hunger" in pairs
        assert "boundary_awareness" in pairs
        assert "agency_drive" in pairs
