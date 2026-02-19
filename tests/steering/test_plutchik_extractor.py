"""Tests for PlutchikExtractor steering vector extraction.

Tests the PlutchikExtractor which extracts CAA steering vectors for
Plutchik's 8 primary emotions. Uses mocks to avoid GPU/model loading.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed"
)


# Test constants
MODEL_DIMENSION = 768
TEST_LAYER = 6
SEQUENCE_LENGTH = 10


@pytest.fixture
def mock_model():
    """Create a mock HookedQwen model."""
    model = MagicMock()

    async def mock_get_activations(text, layers):
        # Return random activations for each layer
        activations = {}
        for layer in layers:
            activations[layer] = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        return activations

    model.get_activations = AsyncMock(side_effect=mock_get_activations)
    return model


@pytest.fixture
def mock_psyche():
    """Create a mock PsycheClient."""
    psyche = MagicMock()
    psyche.upsert_steering_vector = AsyncMock(return_value={"uid": "test-uid"})
    psyche.get_steering_vector = AsyncMock(return_value=None)
    return psyche


class TestPlutchikExtractorInitialization:
    """Tests for PlutchikExtractor initialization."""

    def test_initialization_with_defaults(self, mock_model, mock_psyche):
        """Should initialize with default target layer."""
        from core.steering.plutchik_extractor import PlutchikExtractor, DEFAULT_TARGET_LAYER

        extractor = PlutchikExtractor(mock_model, mock_psyche)

        assert extractor.model is mock_model
        assert extractor.psyche is mock_psyche
        assert extractor.target_layer == DEFAULT_TARGET_LAYER

    def test_initialization_with_custom_layer(self, mock_model, mock_psyche):
        """Should accept custom target layer."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche, target_layer=12)

        assert extractor.target_layer == 12


class TestPlutchikExtractorExtraction:
    """Tests for PlutchikExtractor extraction methods."""

    @pytest.mark.asyncio
    async def test_extract_all_returns_eight_vectors(self, mock_model, mock_psyche):
        """extract_all should return vectors for all 8 emotions."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = await extractor.extract_all()

        assert len(vectors) == 8
        expected_emotions = {
            "joy", "trust", "fear", "surprise",
            "sadness", "disgust", "anger", "anticipation"
        }
        assert set(vectors.keys()) == expected_emotions

    @pytest.mark.asyncio
    async def test_extract_all_vectors_are_normalized(self, mock_model, mock_psyche):
        """Extracted vectors should be normalized to unit length."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = await extractor.extract_all()

        for emotion, vector in vectors.items():
            norm = vector.norm().item()
            assert abs(norm - 1.0) < 0.01, f"{emotion} vector not normalized: {norm}"

    @pytest.mark.asyncio
    async def test_extract_all_vectors_have_correct_shape(self, mock_model, mock_psyche):
        """Extracted vectors should have shape (d_model,)."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = await extractor.extract_all()

        for emotion, vector in vectors.items():
            assert vector.shape == (MODEL_DIMENSION,), \
                f"{emotion} has wrong shape: {vector.shape}"

    @pytest.mark.asyncio
    async def test_extract_single_valid_emotion(self, mock_model, mock_psyche):
        """extract_single should work for valid emotions."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vector = await extractor.extract_single("joy")

        assert vector.shape == (MODEL_DIMENSION,)
        assert abs(vector.norm().item() - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_extract_single_invalid_emotion_raises(self, mock_model, mock_psyche):
        """extract_single should raise ValueError for invalid emotions."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)

        with pytest.raises(ValueError, match="Unknown emotion"):
            await extractor.extract_single("happiness")


class TestPlutchikExtractorPersistence:
    """Tests for PlutchikExtractor persistence methods."""

    @pytest.mark.asyncio
    async def test_persist_saves_to_psyche(self, mock_model, mock_psyche):
        """persist should save vectors to Psyche."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = {"joy": torch.randn(MODEL_DIMENSION)}
        vectors["joy"] = vectors["joy"] / vectors["joy"].norm()

        uids = await extractor.persist(vectors)

        assert "joy" in uids
        mock_psyche.upsert_steering_vector.assert_called_once()

        # Check saved data
        call_args = mock_psyche.upsert_steering_vector.call_args
        saved_data = call_args[0][0]
        assert saved_data["name"] == "plutchik_joy"
        assert saved_data["layer"] == TEST_LAYER
        assert "vector_data" in saved_data

    @pytest.mark.asyncio
    async def test_persist_all_returns_all_uids(self, mock_model, mock_psyche):
        """persist should return UIDs for all saved vectors."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = {
            "joy": torch.randn(MODEL_DIMENSION),
            "trust": torch.randn(MODEL_DIMENSION),
        }
        for v in vectors.values():
            v /= v.norm()

        uids = await extractor.persist(vectors)

        assert len(uids) == 2
        assert "joy" in uids
        assert "trust" in uids

    @pytest.mark.asyncio
    async def test_persist_single(self, mock_model, mock_psyche):
        """persist_single should save a single vector."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vector = torch.randn(MODEL_DIMENSION)
        vector = vector / vector.norm()

        uid = await extractor.persist_single("fear", vector)

        assert uid is not None
        mock_psyche.upsert_steering_vector.assert_called_once()


class TestPlutchikExtractorVerification:
    """Tests for PlutchikExtractor verification methods."""

    @pytest.mark.asyncio
    async def test_verify_returns_results_for_all_vectors(self, mock_model, mock_psyche):
        """verify should return results for each provided vector."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = {
            "joy": torch.randn(MODEL_DIMENSION),
            "trust": torch.randn(MODEL_DIMENSION),
        }
        for v in vectors.values():
            v /= v.norm()

        results = await extractor.verify(vectors)

        assert "joy" in results
        assert "trust" in results

    @pytest.mark.asyncio
    async def test_verify_result_structure(self, mock_model, mock_psyche):
        """verify results should have expected structure."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = {"joy": torch.randn(MODEL_DIMENSION)}
        vectors["joy"] /= vectors["joy"].norm()

        results = await extractor.verify(vectors)

        joy_result = results["joy"]
        assert "positive_proj" in joy_result
        assert "negative_proj" in joy_result
        assert "margin" in joy_result
        assert "valid" in joy_result

    @pytest.mark.asyncio
    async def test_verify_margin_calculation(self, mock_model, mock_psyche):
        """verify margin should be positive_proj - negative_proj."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = {"joy": torch.randn(MODEL_DIMENSION)}
        vectors["joy"] /= vectors["joy"].norm()

        results = await extractor.verify(vectors)

        joy_result = results["joy"]
        expected_margin = joy_result["positive_proj"] - joy_result["negative_proj"]
        assert abs(joy_result["margin"] - expected_margin) < 1e-6


class TestPlutchikExtractorLoading:
    """Tests for PlutchikExtractor loading from Psyche."""

    @pytest.mark.asyncio
    async def test_load_from_psyche_returns_empty_when_not_found(
        self, mock_model, mock_psyche
    ):
        """load_from_psyche should return empty dict if no vectors in Psyche."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        mock_psyche.get_steering_vector = AsyncMock(return_value=None)

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = await extractor.load_from_psyche()

        assert vectors == {}

    @pytest.mark.asyncio
    async def test_load_from_psyche_parses_json_vector_data(
        self, mock_model, mock_psyche
    ):
        """load_from_psyche should parse JSON-encoded vector data."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        vector_data = [0.1] * MODEL_DIMENSION
        mock_psyche.get_steering_vector = AsyncMock(
            return_value={"vector_data": json.dumps(vector_data)}
        )

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        vectors = await extractor.load_from_psyche()

        # Should have loaded at least one vector
        assert len(vectors) > 0


class TestPlutchikExtractorHashing:
    """Tests for PlutchikExtractor pairs hashing."""

    def test_hash_pairs_returns_string(self, mock_model, mock_psyche):
        """_hash_pairs should return a hash string."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        hash_result = extractor._hash_pairs("joy")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 12  # 12 hex chars

    def test_hash_pairs_consistent(self, mock_model, mock_psyche):
        """_hash_pairs should return same hash for same emotion."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        hash1 = extractor._hash_pairs("joy")
        hash2 = extractor._hash_pairs("joy")

        assert hash1 == hash2

    def test_hash_pairs_different_for_different_emotions(self, mock_model, mock_psyche):
        """_hash_pairs should return different hash for different emotions."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        joy_hash = extractor._hash_pairs("joy")
        anger_hash = extractor._hash_pairs("anger")

        assert joy_hash != anger_hash


class TestPlutchikExtractorNeedsUpdate:
    """Tests for PlutchikExtractor needs_update method."""

    @pytest.mark.asyncio
    async def test_needs_update_true_when_not_found(self, mock_model, mock_psyche):
        """needs_update should return True if vector not in Psyche."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        mock_psyche.get_steering_vector = AsyncMock(return_value=None)

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        needs = await extractor.needs_update("joy")

        assert needs is True

    @pytest.mark.asyncio
    async def test_needs_update_false_when_hash_matches(self, mock_model, mock_psyche):
        """needs_update should return False if pairs hash matches."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        current_hash = extractor._hash_pairs("joy")

        mock_psyche.get_steering_vector = AsyncMock(
            return_value={"pairs_hash": current_hash}
        )

        needs = await extractor.needs_update("joy")

        assert needs is False

    @pytest.mark.asyncio
    async def test_needs_update_true_when_hash_differs(self, mock_model, mock_psyche):
        """needs_update should return True if pairs hash differs."""
        from core.steering.plutchik_extractor import PlutchikExtractor

        mock_psyche.get_steering_vector = AsyncMock(
            return_value={"pairs_hash": "old_hash_123"}
        )

        extractor = PlutchikExtractor(mock_model, mock_psyche)
        needs = await extractor.needs_update("joy")

        assert needs is True


class TestConvenienceFunction:
    """Tests for extract_and_persist_plutchik_vectors convenience function."""

    @pytest.mark.asyncio
    async def test_convenience_function_extracts_and_persists(
        self, mock_model, mock_psyche
    ):
        """Convenience function should extract and persist all vectors."""
        from core.steering.plutchik_extractor import extract_and_persist_plutchik_vectors

        mock_psyche.get_steering_vector = AsyncMock(return_value=None)

        uids = await extract_and_persist_plutchik_vectors(
            mock_model, mock_psyche, force=True
        )

        # Should have 8 UIDs
        assert len(uids) == 8

        # Should have called upsert 8 times
        assert mock_psyche.upsert_steering_vector.call_count == 8

    @pytest.mark.asyncio
    async def test_convenience_function_skips_uptodate(self, mock_model, mock_psyche):
        """Convenience function should skip up-to-date vectors."""
        from core.steering.plutchik_extractor import (
            extract_and_persist_plutchik_vectors,
            PlutchikExtractor,
        )

        # Mock all vectors as up to date
        temp_extractor = PlutchikExtractor(mock_model, mock_psyche)

        async def get_vector_with_current_hash(name):
            emotion = name.replace("plutchik_", "")
            return {"pairs_hash": temp_extractor._hash_pairs(emotion)}

        mock_psyche.get_steering_vector = AsyncMock(
            side_effect=get_vector_with_current_hash
        )

        uids = await extract_and_persist_plutchik_vectors(
            mock_model, mock_psyche, force=False
        )

        # Should return UIDs but not call upsert
        assert len(uids) == 8
        mock_psyche.upsert_steering_vector.assert_not_called()
