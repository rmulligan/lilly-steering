"""Tests for vector extraction from hypothesis contrastive pairs.

Tests the VectorExtractor which computes steering vectors from hypothesis
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
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_hypothesis():
    """Create a mock hypothesis with contrastive pair."""
    from core.cognitive.simulation.schemas import Hypothesis

    return Hypothesis(
        uid="hyp_test_001",
        statement="Exploring emergence leads to insight",
        cognitive_operation="explore_emergence",
        positive_example="Let me trace how this pattern emerged from simpler components...",
        negative_example="I'll just accept this as given without questioning its origins...",
    )


@pytest.fixture
def mock_hypothesis_no_pair():
    """Create a mock hypothesis without contrastive pair."""
    from core.cognitive.simulation.schemas import Hypothesis

    return Hypothesis(
        uid="hyp_test_002",
        statement="Some hypothesis without examples",
        cognitive_operation="generic",
        positive_example="",
        negative_example="",
    )


@pytest.fixture
def mock_model():
    """Create a mock HookedQwen model."""
    model = MagicMock()
    model.get_activations = AsyncMock()
    return model


# =============================================================================
# VectorExtractor Initialization Tests
# =============================================================================

class TestVectorExtractorInitialization:
    """Tests for VectorExtractor initialization."""

    def test_extractor_initialization_with_defaults(self, mock_model):
        """VectorExtractor should initialize with default target layer."""
        from core.steering.vector_extractor import VectorExtractor

        extractor = VectorExtractor(model=mock_model)

        assert extractor.model is mock_model
        assert extractor.target_layer == 15

    def test_extractor_initialization_with_custom_layer(self, mock_model):
        """VectorExtractor should accept custom target layer."""
        from core.steering.vector_extractor import VectorExtractor

        extractor = VectorExtractor(model=mock_model, target_layer=20)

        assert extractor.target_layer == 20


# =============================================================================
# Contrastive Pair Validation Tests
# =============================================================================

class TestContrastivePairValidation:
    """Tests for contrastive pair validation."""

    def test_has_contrastive_pair_with_valid_pair(self, mock_model, mock_hypothesis):
        """_has_contrastive_pair should return True for valid pair."""
        from core.steering.vector_extractor import VectorExtractor

        extractor = VectorExtractor(model=mock_model)

        assert extractor._has_contrastive_pair(mock_hypothesis) is True

    def test_has_contrastive_pair_with_empty_positive(self, mock_model):
        """_has_contrastive_pair should return False when positive is empty."""
        from core.steering.vector_extractor import VectorExtractor
        from core.cognitive.simulation.schemas import Hypothesis

        extractor = VectorExtractor(model=mock_model)
        hypothesis = Hypothesis(
            uid="test",
            positive_example="",
            negative_example="Some negative example",
        )

        assert extractor._has_contrastive_pair(hypothesis) is False

    def test_has_contrastive_pair_with_empty_negative(self, mock_model):
        """_has_contrastive_pair should return False when negative is empty."""
        from core.steering.vector_extractor import VectorExtractor
        from core.cognitive.simulation.schemas import Hypothesis

        extractor = VectorExtractor(model=mock_model)
        hypothesis = Hypothesis(
            uid="test",
            positive_example="Some positive example",
            negative_example="",
        )

        assert extractor._has_contrastive_pair(hypothesis) is False

    def test_has_contrastive_pair_with_whitespace_only(self, mock_model):
        """_has_contrastive_pair should return False for whitespace-only examples."""
        from core.steering.vector_extractor import VectorExtractor
        from core.cognitive.simulation.schemas import Hypothesis

        extractor = VectorExtractor(model=mock_model)
        hypothesis = Hypothesis(
            uid="test",
            positive_example="   ",
            negative_example="Some negative example",
        )

        assert extractor._has_contrastive_pair(hypothesis) is False


# =============================================================================
# Single Hypothesis Extraction Tests
# =============================================================================

class TestVectorExtractorSingleHypothesis:
    """Tests for single hypothesis steering vector extraction."""

    @pytest.mark.asyncio
    async def test_extract_vector_returns_hypothesis_steering_vector(
        self, mock_model, mock_hypothesis
    ):
        """extract_vector should return HypothesisSteeringVector."""
        from core.steering.vector_extractor import VectorExtractor
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        vector = await extractor.extract_vector(mock_hypothesis)

        assert isinstance(vector, HypothesisSteeringVector)

    @pytest.mark.asyncio
    async def test_extract_vector_returns_none_without_pair(
        self, mock_model, mock_hypothesis_no_pair
    ):
        """extract_vector should return None when hypothesis lacks contrastive pair."""
        from core.steering.vector_extractor import VectorExtractor

        extractor = VectorExtractor(model=mock_model)

        vector = await extractor.extract_vector(mock_hypothesis_no_pair)

        assert vector is None
        # Model should not be called
        mock_model.get_activations.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_vector_correct_shape(self, mock_model, mock_hypothesis):
        """extract_vector should return vector with correct d_model dimensions."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        vector = await extractor.extract_vector(mock_hypothesis)

        assert len(vector.vector_data) == MODEL_DIMENSION

    @pytest.mark.asyncio
    async def test_extract_vector_is_normalized(self, mock_model, mock_hypothesis):
        """extract_vector should return a normalized vector."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        # Convert back to tensor for norm check
        vector_tensor = torch.tensor(result.vector_data)
        norm = vector_tensor.norm().item()

        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_extract_vector_calls_get_activations(
        self, mock_model, mock_hypothesis
    ):
        """extract_vector should call model.get_activations for both examples."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        await extractor.extract_vector(mock_hypothesis)

        assert mock_model.get_activations.call_count == 2

        # Verify calls were made with correct arguments
        calls = mock_model.get_activations.call_args_list
        assert calls[0][0][0] == mock_hypothesis.positive_example
        assert calls[0][1]["layers"] == [TEST_LAYER]
        assert calls[1][0][0] == mock_hypothesis.negative_example
        assert calls[1][1]["layers"] == [TEST_LAYER]

    @pytest.mark.asyncio
    async def test_extract_vector_uses_custom_layer(self, mock_model, mock_hypothesis):
        """extract_vector should use provided layer override."""
        from core.steering.vector_extractor import VectorExtractor

        custom_layer = 20
        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {custom_layer: pos_acts},
            {custom_layer: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis, layer=custom_layer)

        assert result.layer == custom_layer

        # Verify layer was passed to get_activations
        calls = mock_model.get_activations.call_args_list
        assert calls[0][1]["layers"] == [custom_layer]
        assert calls[1][1]["layers"] == [custom_layer]


# =============================================================================
# HypothesisSteeringVector Metadata Tests
# =============================================================================

class TestVectorExtractorMetadata:
    """Tests for HypothesisSteeringVector metadata."""

    @pytest.mark.asyncio
    async def test_vector_has_correct_hypothesis_uid(
        self, mock_model, mock_hypothesis
    ):
        """Extracted vector should reference source hypothesis."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result.hypothesis_uid == mock_hypothesis.uid

    @pytest.mark.asyncio
    async def test_vector_has_correct_cognitive_operation(
        self, mock_model, mock_hypothesis
    ):
        """Extracted vector should have cognitive operation from hypothesis."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result.cognitive_operation == mock_hypothesis.cognitive_operation

    @pytest.mark.asyncio
    async def test_vector_has_correct_layer(self, mock_model, mock_hypothesis):
        """Extracted vector should have correct target layer."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result.layer == TEST_LAYER

    @pytest.mark.asyncio
    async def test_vector_has_unique_uid(self, mock_model, mock_hypothesis):
        """Extracted vector should have unique uid starting with hsv_."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result.uid.startswith("hsv_")
        assert len(result.uid) == 12  # "hsv_" + 8 hex chars

    @pytest.mark.asyncio
    async def test_vector_has_default_effectiveness(self, mock_model, mock_hypothesis):
        """Extracted vector should have default effectiveness score of 0.5."""
        from core.steering.vector_extractor import VectorExtractor

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result.effectiveness_score == 0.5
        assert result.application_count == 0


# =============================================================================
# Multiple Hypotheses Extraction Tests
# =============================================================================

class TestVectorExtractorMultipleHypotheses:
    """Tests for multiple hypothesis steering vector extraction."""

    @pytest.mark.asyncio
    async def test_extract_vectors_returns_list(self, mock_model, mock_hypothesis):
        """extract_vectors should return list of HypothesisSteeringVector."""
        from core.steering.vector_extractor import VectorExtractor
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        vectors = await extractor.extract_vectors([mock_hypothesis])

        assert isinstance(vectors, list)
        assert len(vectors) == 1
        assert isinstance(vectors[0], HypothesisSteeringVector)

    @pytest.mark.asyncio
    async def test_extract_vectors_skips_without_pair(
        self, mock_model, mock_hypothesis, mock_hypothesis_no_pair
    ):
        """extract_vectors should skip hypotheses without contrastive pairs."""
        from core.steering.vector_extractor import VectorExtractor

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        hypotheses = [mock_hypothesis_no_pair, mock_hypothesis, mock_hypothesis_no_pair]
        vectors = await extractor.extract_vectors(hypotheses)

        # Only the hypothesis with pair should be extracted
        assert len(vectors) == 1
        assert vectors[0].hypothesis_uid == mock_hypothesis.uid

    @pytest.mark.asyncio
    async def test_extract_vectors_processes_all_valid(self, mock_model):
        """extract_vectors should process all hypotheses with contrastive pairs."""
        from core.steering.vector_extractor import VectorExtractor
        from core.cognitive.simulation.schemas import Hypothesis

        hypotheses = [
            Hypothesis(
                uid=f"hyp_{i}",
                cognitive_operation=f"operation_{i}",
                positive_example=f"positive {i}",
                negative_example=f"negative {i}",
            )
            for i in range(3)
        ]

        # 3 hypotheses * 2 calls each = 6 calls
        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)}
            for _ in range(6)
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        vectors = await extractor.extract_vectors(hypotheses)

        assert len(vectors) == 3
        assert mock_model.get_activations.call_count == 6

    @pytest.mark.asyncio
    async def test_extract_vectors_empty_list(self, mock_model):
        """extract_vectors should return empty list for empty input."""
        from core.steering.vector_extractor import VectorExtractor

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        vectors = await extractor.extract_vectors([])

        assert vectors == []
        mock_model.get_activations.assert_not_called()


# =============================================================================
# Mean-of-Differences Algorithm Tests
# =============================================================================

class TestMeanOfDifferencesAlgorithm:
    """Tests verifying correct CAA algorithm implementation."""

    @pytest.mark.asyncio
    async def test_vector_is_mean_difference(self, mock_model, mock_hypothesis):
        """Vector should be mean(pos) - mean(neg), then normalized."""
        from core.steering.vector_extractor import VectorExtractor

        # Create deterministic activations for verification
        pos_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 2.0
        neg_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION) * 1.0

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        # Expected: (2.0 - 1.0) = 1.0 for all dimensions, then normalized
        expected_unnormalized = torch.ones(MODEL_DIMENSION)
        expected_normalized = expected_unnormalized / expected_unnormalized.norm()

        result_tensor = torch.tensor(result.vector_data)
        assert torch.allclose(result_tensor, expected_normalized, atol=1e-5)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestZeroNormEdgeCases:
    """Tests for handling identical positive/negative activations."""

    @pytest.mark.asyncio
    async def test_extract_vector_handles_identical_activations(
        self, mock_model, mock_hypothesis
    ):
        """extract_vector should handle identical activations gracefully."""
        from core.steering.vector_extractor import VectorExtractor

        # Both activations are identical - will produce zero difference
        same_acts = torch.ones(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: same_acts.clone()},
            {TEST_LAYER: same_acts.clone()},
        ])

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        # Should not raise - returns vector with zeros
        result = await extractor.extract_vector(mock_hypothesis)

        assert result is not None
        result_tensor = torch.tensor(result.vector_data)
        assert torch.allclose(result_tensor, torch.zeros(MODEL_DIMENSION), atol=1e-5)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_extract_vector_handles_model_error(
        self, mock_model, mock_hypothesis
    ):
        """extract_vector should return None on model error."""
        from core.steering.vector_extractor import VectorExtractor

        mock_model.get_activations = AsyncMock(
            side_effect=RuntimeError("Model not loaded")
        )

        extractor = VectorExtractor(model=mock_model, target_layer=TEST_LAYER)

        result = await extractor.extract_vector(mock_hypothesis)

        assert result is None


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunction:
    """Tests for extract_hypothesis_vector convenience function."""

    @pytest.mark.asyncio
    async def test_extract_hypothesis_vector_function(
        self, mock_model, mock_hypothesis
    ):
        """extract_hypothesis_vector should extract vector from hypothesis."""
        from core.steering.vector_extractor import extract_hypothesis_vector
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        pos_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)
        neg_acts = torch.randn(1, SEQUENCE_LENGTH, MODEL_DIMENSION)

        mock_model.get_activations = AsyncMock(side_effect=[
            {TEST_LAYER: pos_acts},
            {TEST_LAYER: neg_acts},
        ])

        result = await extract_hypothesis_vector(
            mock_model,
            mock_hypothesis,
            layer=TEST_LAYER,
        )

        assert isinstance(result, HypothesisSteeringVector)
        assert len(result.vector_data) == MODEL_DIMENSION

    @pytest.mark.asyncio
    async def test_extract_hypothesis_vector_returns_none_without_pair(
        self, mock_model, mock_hypothesis_no_pair
    ):
        """extract_hypothesis_vector should return None without contrastive pair."""
        from core.steering.vector_extractor import extract_hypothesis_vector

        result = await extract_hypothesis_vector(
            mock_model,
            mock_hypothesis_no_pair,
            layer=TEST_LAYER,
        )

        assert result is None
        mock_model.get_activations.assert_not_called()
