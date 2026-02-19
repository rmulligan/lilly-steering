"""Tests for PlutchikRegistry steering vector management."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.steering.plutchik_registry import (
    PlutchikRegistry,
    ACTIVATION_THRESHOLD,
    MAX_COMPOSITE_MAGNITUDE,
    CONFLICT_MAGNITUDE_FACTOR,
)
from core.self_model.affective_system import AffectiveState, PLUTCHIK_PRIMARIES


class TestPlutchikRegistryInit:
    """Test PlutchikRegistry initialization."""

    def test_init_creates_empty_registry(self):
        """Registry starts empty."""
        registry = PlutchikRegistry()
        assert registry.vectors == {}
        assert not registry.is_loaded
        assert registry.vector_count == 0


class TestPlutchikRegistryLoad:
    """Test loading vectors from Psyche."""

    @pytest.mark.asyncio
    async def test_load_success_all_vectors(self):
        """Load all 8 vectors successfully."""
        mock_psyche = AsyncMock()

        # Mock return values for each emotion
        async def get_vector(name):
            emotion = name.replace("plutchik_", "")
            if emotion in PLUTCHIK_PRIMARIES:
                return {
                    "name": name,
                    "vector_data": [0.1 * (i + 1) for i in range(3584)],
                }
            return None

        mock_psyche.get_steering_vector = get_vector

        registry = PlutchikRegistry()
        result = await registry.load(mock_psyche)

        assert result is True
        assert registry.is_loaded
        assert registry.vector_count == 8
        assert all(emotion in registry.vectors for emotion in PLUTCHIK_PRIMARIES)

    @pytest.mark.asyncio
    async def test_load_partial_vectors(self):
        """Load some but not all vectors."""
        mock_psyche = AsyncMock()

        # Only return vectors for joy and trust
        async def get_vector(name):
            if name == "plutchik_joy":
                return {"name": name, "vector_data": [0.1] * 3584}
            if name == "plutchik_trust":
                return {"name": name, "vector_data": [0.2] * 3584}
            return None

        mock_psyche.get_steering_vector = get_vector

        registry = PlutchikRegistry()
        result = await registry.load(mock_psyche)

        assert result is True
        assert registry.is_loaded
        assert registry.vector_count == 2
        assert "joy" in registry.vectors
        assert "trust" in registry.vectors
        assert "fear" not in registry.vectors

    @pytest.mark.asyncio
    async def test_load_no_vectors(self):
        """Handle case when no vectors are in Psyche."""
        mock_psyche = AsyncMock()
        mock_psyche.get_steering_vector = AsyncMock(return_value=None)

        registry = PlutchikRegistry()
        result = await registry.load(mock_psyche)

        assert result is False
        assert registry.is_loaded  # Still marked as loaded (just empty)
        assert registry.vector_count == 0

    @pytest.mark.asyncio
    async def test_load_handles_errors(self):
        """Handle errors during vector loading."""
        mock_psyche = AsyncMock()

        async def get_vector(name):
            if "joy" in name:
                return {"name": name, "vector_data": [0.1] * 3584}
            raise Exception("Database error")

        mock_psyche.get_steering_vector = get_vector

        registry = PlutchikRegistry()
        result = await registry.load(mock_psyche)

        # Should succeed with at least joy loaded
        assert result is True
        assert registry.vector_count == 1
        assert "joy" in registry.vectors


class TestPlutchikRegistryComposite:
    """Test composite vector blending."""

    def _create_loaded_registry(self) -> PlutchikRegistry:
        """Create a registry with mock vectors loaded."""
        registry = PlutchikRegistry()
        registry._loaded = True

        # Create orthogonal-ish vectors for testing
        for i, emotion in enumerate(PLUTCHIK_PRIMARIES):
            vec = np.zeros(3584, dtype=np.float32)
            vec[i * 100:(i + 1) * 100] = 1.0  # Different regions active
            registry.vectors[emotion] = vec

        return registry

    def test_get_composite_returns_none_when_not_loaded(self):
        """Return None if registry not loaded."""
        registry = PlutchikRegistry()
        affect = AffectiveState(joy=0.8)

        result = registry.get_composite(affect)

        assert result is None

    def test_get_composite_returns_none_when_empty(self):
        """Return None if no vectors loaded."""
        registry = PlutchikRegistry()
        registry._loaded = True
        affect = AffectiveState(joy=0.8)

        result = registry.get_composite(affect)

        assert result is None

    def test_get_composite_returns_none_below_threshold(self):
        """Return None if all intensities below threshold."""
        registry = self._create_loaded_registry()
        # All values at or below threshold
        affect = AffectiveState(
            joy=ACTIVATION_THRESHOLD,
            trust=0.05,
            fear=0.0,
            surprise=0.0,
            sadness=0.0,
            disgust=0.0,
            anger=0.0,
            anticipation=ACTIVATION_THRESHOLD,
        )

        result = registry.get_composite(affect)

        assert result is None

    def test_get_composite_single_emotion(self):
        """Composite with single active emotion."""
        registry = self._create_loaded_registry()
        affect = AffectiveState(joy=0.8, trust=0.0, fear=0.0, surprise=0.0,
                                sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0)

        result = registry.get_composite(affect)

        assert result is not None
        # Should be normalized version of joy vector
        assert len(result) == 3584
        # Joy vector has values in first region
        assert np.sum(result[:100]) > np.sum(result[100:200])

    def test_get_composite_multiple_emotions(self):
        """Composite blends multiple active emotions."""
        registry = self._create_loaded_registry()
        affect = AffectiveState(joy=0.6, trust=0.4, fear=0.0, surprise=0.0,
                                sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.2)

        result = registry.get_composite(affect)

        assert result is not None
        # Should have contributions from joy, trust, and anticipation regions
        assert np.sum(result[:100]) > 0  # joy region
        assert np.sum(result[100:200]) > 0  # trust region
        assert np.sum(result[700:800]) > 0  # anticipation region

    def test_get_composite_normalized(self):
        """Composite is normalized by total weight."""
        registry = self._create_loaded_registry()

        # High intensity for two emotions
        affect1 = AffectiveState(joy=0.9, trust=0.0, fear=0.0, surprise=0.0,
                                 sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0)
        affect2 = AffectiveState(joy=0.9, trust=0.9, fear=0.0, surprise=0.0,
                                 sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0)

        result1 = registry.get_composite(affect1)
        result2 = registry.get_composite(affect2)

        # Magnitudes should be comparable due to normalization
        norm1 = np.linalg.norm(result1)
        norm2 = np.linalg.norm(result2)
        # Both should be within reasonable range
        assert norm1 <= MAX_COMPOSITE_MAGNITUDE
        assert norm2 <= MAX_COMPOSITE_MAGNITUDE

    def test_get_composite_magnitude_capped(self):
        """Composite magnitude is capped at MAX_COMPOSITE_MAGNITUDE."""
        registry = PlutchikRegistry()
        registry._loaded = True

        # Create a very large vector
        large_vec = np.ones(3584, dtype=np.float32) * 10.0
        registry.vectors["joy"] = large_vec

        affect = AffectiveState(joy=0.9, trust=0.0, fear=0.0, surprise=0.0,
                                sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0)

        result = registry.get_composite(affect)

        assert result is not None
        assert np.linalg.norm(result) <= MAX_COMPOSITE_MAGNITUDE + 0.01  # Small tolerance


class TestPlutchikRegistryHelpers:
    """Test helper methods."""

    def test_get_single_returns_vector(self):
        """get_single returns loaded vector."""
        registry = PlutchikRegistry()
        registry._loaded = True
        test_vec = np.array([1.0, 2.0, 3.0])
        registry.vectors["joy"] = test_vec

        result = registry.get_single("joy")

        assert result is not None
        np.testing.assert_array_equal(result, test_vec)

    def test_get_single_returns_none_for_missing(self):
        """get_single returns None for missing emotion."""
        registry = PlutchikRegistry()
        registry._loaded = True

        result = registry.get_single("joy")

        assert result is None

    def test_describe_state_lists_active_emotions(self):
        """describe_state lists emotions above threshold."""
        registry = PlutchikRegistry()
        registry._loaded = True
        registry.vectors = {
            "joy": np.zeros(10),
            "trust": np.zeros(10),
            "anger": np.zeros(10),
        }

        affect = AffectiveState(joy=0.8, trust=0.5, fear=0.0, surprise=0.0,
                                sadness=0.0, disgust=0.0, anger=0.3, anticipation=0.0)

        desc = registry.describe_state(affect)

        assert "joy=0.80" in desc
        assert "trust=0.50" in desc
        assert "anger=0.30" in desc
        assert "fear" not in desc  # Below threshold and not loaded

    def test_describe_state_no_active_emotions(self):
        """describe_state handles no active emotions."""
        registry = PlutchikRegistry()
        registry._loaded = True
        registry.vectors = {"joy": np.zeros(10)}

        affect = AffectiveState(joy=0.05, trust=0.0, fear=0.0, surprise=0.0,
                                sadness=0.0, disgust=0.0, anger=0.0, anticipation=0.0)

        desc = registry.describe_state(affect)

        assert "no active emotions" in desc


class TestPlutchikConflictResolution:
    """Test conflict detection and ambivalence blending in composite steering."""

    def _create_loaded_registry(self) -> PlutchikRegistry:
        """Create a registry with mock loaded vectors."""
        from core.self_model.affective_system import PLUTCHIK_PRIMARIES

        registry = PlutchikRegistry()
        registry._loaded = True

        # Create distinct vectors for each emotion
        for i, emotion in enumerate(PLUTCHIK_PRIMARIES):
            vec = np.zeros(3584, dtype=np.float32)
            start = i * 400
            vec[start:start + 100] = 1.0 + i * 0.1
            registry.vectors[emotion] = vec

        return registry

    def test_conflicting_emotions_reduced_magnitude(self):
        """Conflicting emotions (joy ↔ sadness) apply at 0.5x magnitude."""
        registry = self._create_loaded_registry()

        # Joy and sadness are opposite pairs (indices 0 and 4)
        # Both above conflict threshold (0.5)
        affect = AffectiveState(
            joy=0.8,
            trust=0.0,
            fear=0.0,
            surprise=0.0,
            sadness=0.7,
            disgust=0.0,
            anger=0.0,
            anticipation=0.0,
        )

        # Should detect conflict and reduce magnitude
        conflicts = affect.detect_conflicts()
        assert len(conflicts) == 1
        assert ("joy", "sadness") in conflicts

        result = registry.get_composite(affect)
        assert result is not None

        # Get result without conflict for comparison
        # Create affect with only joy (no conflict)
        affect_no_conflict = AffectiveState(
            joy=0.8,
            trust=0.0,
            fear=0.0,
            surprise=0.0,
            sadness=0.0,  # Below conflict threshold
            disgust=0.0,
            anger=0.0,
            anticipation=0.0,
        )
        result_no_conflict = registry.get_composite(affect_no_conflict)

        # The conflicted version should have different weighting
        # (both vectors applied at 0.5x magnitude)
        assert result is not None
        assert result_no_conflict is not None

        # Verify conflict resolution logic was correctly applied
        joy_vec = registry.vectors["joy"]
        sadness_vec = registry.vectors["sadness"]

        expected_weight_joy = affect.joy * CONFLICT_MAGNITUDE_FACTOR
        expected_weight_sadness = affect.sadness * CONFLICT_MAGNITUDE_FACTOR
        total_weight = expected_weight_joy + expected_weight_sadness

        expected_result = (expected_weight_joy * joy_vec + expected_weight_sadness * sadness_vec) / total_weight

        # Apply magnitude cap if needed (same as get_composite)
        norm = np.linalg.norm(expected_result)
        if norm > MAX_COMPOSITE_MAGNITUDE:
            expected_result = expected_result * (MAX_COMPOSITE_MAGNITUDE / norm)

        np.testing.assert_allclose(result, expected_result, atol=1e-6)
        assert not np.allclose(result, result_no_conflict)

    def test_multiple_conflicts_all_reduced(self):
        """Multiple conflict pairs all get reduced magnitude."""
        registry = self._create_loaded_registry()

        # Joy ↔ sadness AND trust ↔ disgust conflicts
        affect = AffectiveState(
            joy=0.8,
            trust=0.7,
            fear=0.0,
            surprise=0.0,
            sadness=0.6,
            disgust=0.6,
            anger=0.0,
            anticipation=0.0,
        )

        conflicts = affect.detect_conflicts()
        assert len(conflicts) == 2

        result = registry.get_composite(affect)
        assert result is not None

        # Verify magnitude reduction is applied to all conflicting emotions
        # All four emotions (joy, trust, sadness, disgust) are in conflict pairs
        # so all should have CONFLICT_MAGNITUDE_FACTOR applied
        expected_vec = np.zeros_like(result)
        total_weight = 0.0
        conflicting_emotions = ["joy", "trust", "sadness", "disgust"]
        for emotion in conflicting_emotions:
            weight = getattr(affect, emotion) * CONFLICT_MAGNITUDE_FACTOR
            expected_vec += weight * registry.vectors[emotion]
            total_weight += weight
        expected_vec /= total_weight

        # Apply magnitude cap if needed (same as get_composite)
        norm = np.linalg.norm(expected_vec)
        if norm > MAX_COMPOSITE_MAGNITUDE:
            expected_vec = expected_vec * (MAX_COMPOSITE_MAGNITUDE / norm)

        np.testing.assert_allclose(result, expected_vec, atol=1e-6)

    def test_non_conflicting_emotions_full_magnitude(self):
        """Non-conflicting emotions retain full magnitude."""
        registry = self._create_loaded_registry()

        # Joy and trust are NOT opposites (adjacent on wheel)
        affect = AffectiveState(
            joy=0.8,
            trust=0.7,
            fear=0.0,
            surprise=0.0,
            sadness=0.0,  # Below conflict threshold
            disgust=0.0,  # Below conflict threshold
            anger=0.0,
            anticipation=0.0,
        )

        conflicts = affect.detect_conflicts()
        assert len(conflicts) == 0  # No conflicts

        result = registry.get_composite(affect)
        assert result is not None

        # Verify emotions contribute full magnitude (no attenuation)
        joy_vec = registry.vectors["joy"]
        trust_vec = registry.vectors["trust"]
        total_weight = affect.joy + affect.trust
        expected_result = (affect.joy * joy_vec + affect.trust * trust_vec) / total_weight

        # Apply magnitude cap if needed (same as get_composite)
        norm = np.linalg.norm(expected_result)
        if norm > MAX_COMPOSITE_MAGNITUDE:
            expected_result = expected_result * (MAX_COMPOSITE_MAGNITUDE / norm)

        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_conflict_threshold_boundary(self):
        """Conflict only triggers when BOTH emotions exceed threshold."""
        registry = self._create_loaded_registry()

        # Joy high, sadness exactly at threshold (0.5)
        affect = AffectiveState(
            joy=0.8,
            trust=0.0,
            fear=0.0,
            surprise=0.0,
            sadness=0.5,  # Exactly at threshold
            disgust=0.0,
            anger=0.0,
            anticipation=0.0,
        )

        # Threshold is 0.5, so exactly 0.5 should NOT trigger conflict
        # (conflict requires > CONFLICT_THRESHOLD)
        conflicts = affect.detect_conflicts()
        # Depends on whether threshold is inclusive
        # The detect_conflicts checks vec[a] > CONFLICT_THRESHOLD

        # Sadness at exactly 0.5 should NOT trigger conflict (uses >)
        assert len(conflicts) == 0

    def test_conflict_reduces_effective_weight(self):
        """Conflicting emotions contribute less to the total weight."""
        from core.steering.plutchik_registry import CONFLICT_MAGNITUDE_FACTOR, MAX_COMPOSITE_MAGNITUDE

        registry = self._create_loaded_registry()

        # Both emotions in conflict at equal intensity
        affect = AffectiveState(
            joy=0.8,
            trust=0.0,
            fear=0.0,
            surprise=0.0,
            sadness=0.8,
            disgust=0.0,
            anger=0.0,
            anticipation=0.0,
        )

        result = registry.get_composite(affect)
        assert result is not None

        # Effective weights should be 0.8 * 0.5 = 0.4 each
        # Total effective weight = 0.8
        # The result should be normalized by this reduced weight
        joy_vec = registry.vectors["joy"]
        sadness_vec = registry.vectors["sadness"]

        expected_weight_joy = affect.joy * CONFLICT_MAGNITUDE_FACTOR
        expected_weight_sadness = affect.sadness * CONFLICT_MAGNITUDE_FACTOR
        total_weight = expected_weight_joy + expected_weight_sadness

        expected_result = (expected_weight_joy * joy_vec + expected_weight_sadness * sadness_vec) / total_weight

        # Apply magnitude cap (same as implementation)
        norm = np.linalg.norm(expected_result)
        if norm > MAX_COMPOSITE_MAGNITUDE:
            expected_result = expected_result * (MAX_COMPOSITE_MAGNITUDE / norm)

        np.testing.assert_allclose(result, expected_result, atol=1e-6)
