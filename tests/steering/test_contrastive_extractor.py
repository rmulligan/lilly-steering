"""Tests for contrastive pair extraction.

Tests the ContrastiveExtractor which computes steering vectors from
positive/negative example pairs using the CAA (Contrastive Activation Addition)
approach: vector = mean(positive_activations) - mean(negative_activations)

Uses mocks to avoid requiring actual GPU/model loading.
Skips tests if torch is not available in the environment.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed"
)


# =============================================================================
# Test Constants
# =============================================================================

MODEL_DIMENSION = 768
TEST_LAYER = 15
SEQUENCE_LENGTH = 10


# =============================================================================
# ContrastivePair Tests
# =============================================================================

class TestContrastivePair:
    """Tests for ContrastivePair dataclass."""

    def test_contrastive_pair_creation(self):
        """ContrastivePair should store positive and negative examples."""
        from core.steering.contrastive_extractor import ContrastivePair

        pair = ContrastivePair(
            positive="Let me think through this carefully...",
            negative="I'll just guess...",
            behavior="deliberate_reasoning",
        )
        assert pair.positive == "Let me think through this carefully..."
        assert pair.negative == "I'll just guess..."
        assert pair.behavior == "deliberate_reasoning"

    def test_contrastive_pair_with_uid(self):
        """ContrastivePair should accept optional uid."""
        from core.steering.contrastive_extractor import ContrastivePair

        pair = ContrastivePair(
            positive="positive example",
            negative="negative example",
            behavior="test_behavior",
            uid="pair-001",
        )
        assert pair.uid == "pair-001"

    def test_contrastive_pair_uid_defaults_to_none(self):
        """ContrastivePair uid should default to None."""
        from core.steering.contrastive_extractor import ContrastivePair

        pair = ContrastivePair(
            positive="positive",
            negative="negative",
            behavior="test",
        )
        assert pair.uid is None


# =============================================================================
# ContrastiveExtractor Initialization Tests
# =============================================================================

class TestContrastiveExtractorInitialization:
    """Tests for ContrastiveExtractor initialization."""

    def test_extractor_initialization(self):
        """ContrastiveExtractor should initialize with model reference."""
        from core.steering.contrastive_extractor import ContrastiveExtractor

        mock_model = MagicMock()
        extractor = ContrastiveExtractor(model=mock_model)

        assert extractor.model is mock_model


# =============================================================================
# Single Pair Extraction Tests
# =============================================================================

class TestContrastiveExtractorSinglePair:
    """Tests for single pair steering vector extraction."""

    @pytest.mark.asyncio
    async def test_extract_vector_returns_tensor(self):
        """extract_vector should return a 1D tensor."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        # Mock activations
        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="positive example",
            negative="negative example",
            behavior="test",
        )

        vector = await extractor.extract_vector(pair, layer=TEST_LAYER)

        assert isinstance(vector, torch.Tensor)
        assert vector.dim() == 1

    @pytest.mark.asyncio
    async def test_extract_vector_correct_shape(self):
        """extract_vector should return (d_model,) shape."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="positive example",
            negative="negative example",
            behavior="test",
        )

        vector = await extractor.extract_vector(pair, layer=TEST_LAYER)

        assert vector.shape == (MODEL_DIMENSION,)

    @pytest.mark.asyncio
    async def test_extract_vector_is_normalized(self):
        """extract_vector should return a normalized vector."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="positive example",
            negative="negative example",
            behavior="test",
        )

        vector = await extractor.extract_vector(pair, layer=TEST_LAYER)

        # Check normalization (L2 norm should be 1.0)
        norm = vector.norm().item()
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_extract_vector_calls_get_activations(self):
        """extract_vector should call model.get_activations for both examples."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="positive text",
            negative="negative text",
            behavior="test",
        )

        await extractor.extract_vector(pair, layer=TEST_LAYER)

        assert mock_model.get_activations.call_count == 2
        # Verify calls were made with correct arguments
        calls = mock_model.get_activations.call_args_list
        assert calls[0][0][0] == "positive text"
        assert calls[0][1]["layers"] == [TEST_LAYER]
        assert calls[1][0][0] == "negative text"
        assert calls[1][1]["layers"] == [TEST_LAYER]


# =============================================================================
# Multiple Pairs Extraction Tests
# =============================================================================

class TestContrastiveExtractorMultiplePairs:
    """Tests for multiple pair steering vector extraction."""

    @pytest.mark.asyncio
    async def test_extract_from_pairs_returns_tensor(self):
        """extract_from_pairs should return a 1D tensor."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        mock_model = MagicMock()
        # Two pairs = 4 activation calls
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pairs = [
            ContrastivePair(positive="pos1", negative="neg1", behavior="test"),
            ContrastivePair(positive="pos2", negative="neg2", behavior="test"),
        ]

        vector = await extractor.extract_from_pairs(pairs, layer=TEST_LAYER)

        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (MODEL_DIMENSION,)

    @pytest.mark.asyncio
    async def test_extract_from_pairs_is_normalized(self):
        """extract_from_pairs should return a normalized vector."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pairs = [
            ContrastivePair(positive="pos1", negative="neg1", behavior="test"),
            ContrastivePair(positive="pos2", negative="neg2", behavior="test"),
        ]

        vector = await extractor.extract_from_pairs(pairs, layer=TEST_LAYER)

        norm = vector.norm().item()
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_extract_from_pairs_empty_raises(self):
        """extract_from_pairs should raise ValueError for empty pairs list."""
        from core.steering.contrastive_extractor import ContrastiveExtractor

        mock_model = MagicMock()
        extractor = ContrastiveExtractor(model=mock_model)

        with pytest.raises(ValueError, match="At least one pair required"):
            await extractor.extract_from_pairs([], layer=TEST_LAYER)

    @pytest.mark.asyncio
    async def test_extract_from_pairs_processes_all_pairs(self):
        """extract_from_pairs should process all pairs in the list."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        mock_model = MagicMock()
        # Three pairs = 6 activation calls
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)}
            for _ in range(6)
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pairs = [
            ContrastivePair(positive=f"pos{i}", negative=f"neg{i}", behavior="test")
            for i in range(3)
        ]

        await extractor.extract_from_pairs(pairs, layer=TEST_LAYER)

        assert mock_model.get_activations.call_count == 6


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestExtractSteeringVectorFunction:
    """Tests for the extract_steering_vector convenience function."""

    @pytest.mark.asyncio
    async def test_extract_steering_vector_function(self):
        """extract_steering_vector should compute mean-of-differences steering vector."""
        from core.steering.contrastive_extractor import (
            ContrastivePair,
            extract_steering_vector,
        )

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        pairs = [
            ContrastivePair(
                positive="positive example",
                negative="negative example",
                behavior="test",
            )
        ]

        vector = await extract_steering_vector(mock_model, pairs, layer=TEST_LAYER)

        assert vector.shape == (MODEL_DIMENSION,)
        assert abs(vector.norm().item() - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_extract_steering_vector_delegates_to_extractor(self):
        """extract_steering_vector should use ContrastiveExtractor internally."""
        from core.steering.contrastive_extractor import (
            ContrastivePair,
            extract_steering_vector,
        )

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
        ])

        pairs = [
            ContrastivePair(positive="pos", negative="neg", behavior="test")
        ]

        await extract_steering_vector(mock_model, pairs, layer=TEST_LAYER)

        # Verify model's get_activations was called
        assert mock_model.get_activations.call_count == 2


# =============================================================================
# Mean-of-Differences Algorithm Tests
# =============================================================================

class TestMeanOfDifferencesAlgorithm:
    """Tests verifying correct CAA algorithm implementation."""

    @pytest.mark.asyncio
    async def test_vector_is_mean_difference(self):
        """Vector should be mean(pos) - mean(neg), then normalized."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        # Create deterministic activations for verification
        pos_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 2.0
        neg_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 1.0

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="positive",
            negative="negative",
            behavior="test",
        )

        vector = await extractor.extract_vector(pair, layer=TEST_LAYER)

        # Expected: (2.0 - 1.0) = 1.0 for all dimensions, then normalized
        expected_unnormalized = torch.ones(MODEL_DIMENSION)
        expected_normalized = expected_unnormalized / expected_unnormalized.norm()

        assert torch.allclose(vector, expected_normalized, atol=1e-5)

    @pytest.mark.asyncio
    async def test_multiple_pairs_averaging(self):
        """Multiple pairs should average their differences."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        # Pair 1: diff = 2.0 - 1.0 = 1.0
        pos1 = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 2.0
        neg1 = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 1.0
        # Pair 2: diff = 4.0 - 1.0 = 3.0
        pos2 = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 4.0
        neg2 = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 1.0

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos1},
            {TEST_LAYER: neg1},
            {TEST_LAYER: pos2},
            {TEST_LAYER: neg2},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pairs = [
            ContrastivePair(positive="pos1", negative="neg1", behavior="test"),
            ContrastivePair(positive="pos2", negative="neg2", behavior="test"),
        ]

        vector = await extractor.extract_from_pairs(pairs, layer=TEST_LAYER)

        # Expected: mean([1.0, 3.0]) = 2.0 for all dimensions, then normalized
        expected_unnormalized = torch.ones(MODEL_DIMENSION) * 2.0
        expected_normalized = expected_unnormalized / expected_unnormalized.norm()

        assert torch.allclose(vector, expected_normalized, atol=1e-5)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestZeroNormEdgeCases:
    """Tests for handling identical positive/negative activations."""

    @pytest.mark.asyncio
    async def test_extract_vector_handles_identical_activations(self):
        """extract_vector should handle identical positive/negative activations gracefully."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        # Both activations are identical - will produce zero difference
        same_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: same_acts.clone()},
            {TEST_LAYER: same_acts.clone()},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pair = ContrastivePair(
            positive="same text",
            negative="same text",
            behavior="identical_test",
        )

        # Should not raise - returns zero vector instead
        vector = await extractor.extract_vector(pair, layer=TEST_LAYER)

        assert isinstance(vector, torch.Tensor)
        assert vector.shape == (MODEL_DIMENSION,)
        # Vector should be all zeros (not normalized due to zero norm)
        assert torch.allclose(vector, torch.zeros(MODEL_DIMENSION), atol=1e-5)

    @pytest.mark.asyncio
    async def test_extract_from_pairs_handles_identical_activations(self):
        """extract_from_pairs should handle identical activations gracefully."""
        from core.steering.contrastive_extractor import (
            ContrastiveExtractor,
            ContrastivePair,
        )

        same_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model = MagicMock()
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: same_acts.clone()},
            {TEST_LAYER: same_acts.clone()},
        ])

        extractor = ContrastiveExtractor(model=mock_model)

        pairs = [
            ContrastivePair(positive="same", negative="same", behavior="identical"),
        ]

        # Should not raise
        vector = await extractor.extract_from_pairs(pairs, layer=TEST_LAYER)

        assert vector.shape == (MODEL_DIMENSION,)
        assert torch.allclose(vector, torch.zeros(MODEL_DIMENSION), atol=1e-5)
