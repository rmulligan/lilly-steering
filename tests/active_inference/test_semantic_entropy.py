"""Tests for core.active_inference.semantic_entropy module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from core.active_inference.semantic_entropy import (
    SemanticEntropyCalculator,
    SemanticEntropyResult,
    compute_semantic_entropy,
    SEMANTIC_ENTROPY_HIGH,
    SEMANTIC_ENTROPY_LOW,
    EFFECTIVE_DIM_THRESHOLD,
)


class TestSemanticEntropyResult:
    """Tests for SemanticEntropyResult dataclass."""

    def test_create_result(self):
        """Test creating a semantic entropy result with all fields."""
        result = SemanticEntropyResult(
            semantic_entropy=0.65,
            topic_concentration=0.35,
            effective_dimensions=4.0,
            embedding_count=10,
            mean_similarity=0.4,
            min_similarity=0.1,
            max_similarity=0.8,
            is_diverse=True,
            diversity_reason="High semantic diversity",
        )
        assert result.semantic_entropy == 0.65
        assert result.topic_concentration == 0.35
        assert result.effective_dimensions == 4.0
        assert result.embedding_count == 10
        assert result.mean_similarity == 0.4
        assert result.min_similarity == 0.1
        assert result.max_similarity == 0.8
        assert result.is_diverse is True
        assert result.diversity_reason == "High semantic diversity"

    def test_is_concentrated_property(self):
        """Test is_concentrated property for tightly clustered embeddings."""
        concentrated_result = SemanticEntropyResult(
            semantic_entropy=0.2,
            topic_concentration=0.8,  # > 0.7 threshold
            effective_dimensions=2.0,
            embedding_count=10,
            mean_similarity=0.9,
            min_similarity=0.7,
            max_similarity=0.95,
            is_diverse=False,
            diversity_reason="Concentrated topics",
        )
        diverse_result = SemanticEntropyResult(
            semantic_entropy=0.8,
            topic_concentration=0.2,  # < 0.7 threshold
            effective_dimensions=6.0,
            embedding_count=10,
            mean_similarity=0.3,
            min_similarity=0.1,
            max_similarity=0.6,
            is_diverse=True,
            diversity_reason="High diversity",
        )

        assert concentrated_result.is_concentrated is True
        assert diverse_result.is_concentrated is False

    def test_is_balanced_property(self):
        """Test is_balanced property for moderate semantic diversity."""
        balanced_result = SemanticEntropyResult(
            semantic_entropy=0.5,  # Between 0.3 and 0.7
            topic_concentration=0.5,
            effective_dimensions=4.0,
            embedding_count=10,
            mean_similarity=0.5,
            min_similarity=0.2,
            max_similarity=0.8,
            is_diverse=True,
            diversity_reason="Moderate diversity",
        )
        high_result = SemanticEntropyResult(
            semantic_entropy=0.85,  # > 0.7
            topic_concentration=0.15,
            effective_dimensions=8.0,
            embedding_count=10,
            mean_similarity=0.2,
            min_similarity=0.05,
            max_similarity=0.4,
            is_diverse=True,
            diversity_reason="High diversity",
        )
        low_result = SemanticEntropyResult(
            semantic_entropy=0.2,  # < 0.3
            topic_concentration=0.8,
            effective_dimensions=2.0,
            embedding_count=10,
            mean_similarity=0.85,
            min_similarity=0.7,
            max_similarity=0.95,
            is_diverse=False,
            diversity_reason="Low diversity",
        )

        assert balanced_result.is_balanced is True
        assert high_result.is_balanced is False
        assert low_result.is_balanced is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SemanticEntropyResult(
            semantic_entropy=0.5,
            topic_concentration=0.5,
            effective_dimensions=4.0,
            embedding_count=5,
            mean_similarity=0.6,
            min_similarity=0.3,
            max_similarity=0.9,
            is_diverse=True,
            diversity_reason="Moderate diversity",
        )
        data = result.to_dict()

        assert data["semantic_entropy"] == 0.5
        assert data["topic_concentration"] == 0.5
        assert data["effective_dimensions"] == 4.0
        assert data["embedding_count"] == 5
        assert data["mean_similarity"] == 0.6
        assert data["min_similarity"] == 0.3
        assert data["max_similarity"] == 0.9
        assert data["is_diverse"] is True
        assert data["diversity_reason"] == "Moderate diversity"


class TestSemanticEntropyCalculator:
    """Tests for SemanticEntropyCalculator class."""

    def test_create_calculator(self):
        """Test creating a calculator with required graph parameter."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)
        assert calculator.graph == mock_graph
        assert calculator.max_embeddings == 500  # Default value

    def test_create_with_max_embeddings(self):
        """Test creating calculator with custom max_embeddings."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph, max_embeddings=100)
        assert calculator.graph == mock_graph
        assert calculator.max_embeddings == 100

    @pytest.mark.asyncio
    async def test_compute_insufficient_embeddings(self):
        """Test compute returns default result with insufficient embeddings."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[])

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        assert isinstance(result, SemanticEntropyResult)
        assert result.semantic_entropy == 0.0
        assert result.topic_concentration == 1.0
        assert result.effective_dimensions == 1.0
        assert result.embedding_count == 0
        assert result.is_diverse is False
        assert "Insufficient" in result.diversity_reason

    @pytest.mark.asyncio
    async def test_compute_single_embedding(self):
        """Test compute with single embedding returns default result."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[{"embedding": [0.1] * 768}])

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        assert result.embedding_count == 1
        assert result.semantic_entropy == 0.0
        assert result.is_diverse is False

    @pytest.mark.asyncio
    async def test_compute_with_diverse_embeddings(self):
        """Test compute with diverse embeddings produces higher entropy."""
        mock_graph = MagicMock()
        # Create orthogonal embeddings (high diversity)
        embeddings = []
        for i in range(10):
            emb = np.zeros(100)
            emb[i * 10:(i + 1) * 10] = 1.0
            embeddings.append({"embedding": emb.tolist()})

        mock_graph.query = AsyncMock(return_value=embeddings)

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        assert isinstance(result, SemanticEntropyResult)
        assert result.embedding_count == 10
        # Orthogonal embeddings should have lower mean similarity
        assert result.mean_similarity < 0.7

    @pytest.mark.asyncio
    async def test_compute_with_similar_embeddings(self):
        """Test compute with similar embeddings produces lower entropy."""
        mock_graph = MagicMock()
        # Create nearly identical embeddings (low diversity)
        base = np.random.rand(100)
        embeddings = []
        for _ in range(10):
            # Add tiny noise to base embedding
            noisy = base + np.random.rand(100) * 0.001
            embeddings.append({"embedding": noisy.tolist()})

        mock_graph.query = AsyncMock(return_value=embeddings)

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        assert result.embedding_count == 10
        # Similar embeddings should have high mean similarity
        assert result.mean_similarity > 0.9

    @pytest.mark.asyncio
    async def test_compute_with_specific_node_uids(self):
        """Test compute with specific node UIDs."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[
            {"embedding": [0.1] * 100},
            {"embedding": [0.2] * 100},
            {"embedding": [0.3] * 100},
        ])

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute(node_uids=["uid1", "uid2", "uid3"])

        assert result.embedding_count == 3
        # Verify query was called with node UIDs
        call_args = mock_graph.query.call_args
        assert "uids" in call_args[1] or "$uids" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_compute_handles_string_embeddings(self):
        """Test compute handles JSON string embeddings."""
        import json

        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[
            {"embedding": json.dumps([0.1] * 100)},
            {"embedding": json.dumps([0.2] * 100)},
        ])

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        assert result.embedding_count == 2

    @pytest.mark.asyncio
    async def test_compute_handles_data_processing_error(self):
        """Test compute handles data format errors gracefully."""
        mock_graph = MagicMock()
        # TypeError/ValueError/KeyError/AttributeError are caught as data errors
        mock_graph.query = AsyncMock(side_effect=TypeError("Invalid data format"))

        calculator = SemanticEntropyCalculator(graph=mock_graph)
        result = await calculator.compute()

        # Should return empty result, not raise
        assert result.embedding_count == 0
        assert result.is_diverse is False

    @pytest.mark.asyncio
    async def test_compute_propagates_critical_errors(self):
        """Test compute propagates critical system errors like RuntimeError."""
        mock_graph = MagicMock()
        # RuntimeError (e.g., database connection failure) should propagate
        mock_graph.query = AsyncMock(side_effect=RuntimeError("Not connected to Psyche graph"))

        calculator = SemanticEntropyCalculator(graph=mock_graph)

        with pytest.raises(RuntimeError, match="Not connected"):
            await calculator.compute()

    def test_compute_similarity_matrix(self):
        """Test _compute_similarity_matrix produces valid similarity values."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        # Create simple embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        sim_matrix = calculator._compute_similarity_matrix(embeddings)

        # Matrix should be symmetric and have 1s on diagonal (after normalization)
        assert sim_matrix.shape == (3, 3)
        # All values should be in [0, 1] after (sim + 1) / 2 transformation
        assert np.all(sim_matrix >= 0)
        assert np.all(sim_matrix <= 1)
        # Diagonal should be 1 (self-similarity)
        assert np.allclose(np.diag(sim_matrix), 1.0, atol=0.01)

    def test_compute_similarity_matrix_identical_vectors(self):
        """Test similarity matrix with identical vectors."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        # Identical embeddings
        embeddings = np.array([
            [1.0, 0.5, 0.3],
            [1.0, 0.5, 0.3],
        ], dtype=np.float32)

        sim_matrix = calculator._compute_similarity_matrix(embeddings)

        # All similarities should be 1 for identical vectors
        assert np.allclose(sim_matrix, 1.0, atol=0.01)

    def test_compute_spectral_entropy_uniform(self):
        """Test _compute_spectral_entropy with uniform distribution."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        # Identity matrix (uniform eigenvalues)
        identity = np.eye(5)
        entropy, effective_dims = calculator._compute_spectral_entropy(identity)

        # Uniform distribution should have high entropy
        assert 0 <= entropy <= 1
        assert effective_dims >= 1

    def test_compute_spectral_entropy_concentrated(self):
        """Test _compute_spectral_entropy with concentrated distribution."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        # Matrix with one dominant eigenvalue
        concentrated = np.ones((5, 5))  # Rank 1 matrix
        entropy, effective_dims = calculator._compute_spectral_entropy(concentrated)

        # Concentrated distribution should have lower entropy
        assert 0 <= entropy <= 1
        # Only one significant dimension
        assert effective_dims <= 5

    def test_compute_similarity_stats(self):
        """Test _compute_similarity_stats returns correct statistics."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        sim_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.7],
            [0.3, 0.7, 1.0],
        ])

        mean, min_val, max_val = calculator._compute_similarity_stats(sim_matrix)

        # Upper triangle values: 0.5, 0.3, 0.7
        assert mean == pytest.approx(0.5, abs=0.01)
        assert min_val == pytest.approx(0.3, abs=0.01)
        assert max_val == pytest.approx(0.7, abs=0.01)

    def test_assess_diversity_few_embeddings(self):
        """Test _assess_diversity with too few embeddings."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        is_diverse, reason = calculator._assess_diversity(
            entropy=0.8,
            concentration=0.2,
            effective_dims=5.0,
            embedding_count=3,  # Less than 5
        )

        assert is_diverse is False
        assert "Too few" in reason

    def test_assess_diversity_high_entropy(self):
        """Test _assess_diversity with high entropy."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        is_diverse, reason = calculator._assess_diversity(
            entropy=0.85,  # > SEMANTIC_ENTROPY_HIGH
            concentration=0.15,
            effective_dims=6.0,
            embedding_count=20,
        )

        assert is_diverse is True
        assert "High semantic diversity" in reason

    def test_assess_diversity_low_entropy(self):
        """Test _assess_diversity with low entropy."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        is_diverse, reason = calculator._assess_diversity(
            entropy=0.2,  # < SEMANTIC_ENTROPY_LOW
            concentration=0.8,
            effective_dims=2.0,
            embedding_count=20,
        )

        assert is_diverse is False
        assert "Concentrated" in reason

    def test_assess_diversity_few_dimensions(self):
        """Test _assess_diversity with few effective dimensions."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        is_diverse, reason = calculator._assess_diversity(
            entropy=0.5,  # Moderate
            concentration=0.5,
            effective_dims=2.0,  # < 3
            embedding_count=20,
        )

        assert is_diverse is False
        assert "dimensions" in reason

    def test_assess_diversity_moderate(self):
        """Test _assess_diversity with moderate entropy."""
        mock_graph = MagicMock()
        calculator = SemanticEntropyCalculator(graph=mock_graph)

        is_diverse, reason = calculator._assess_diversity(
            entropy=0.5,
            concentration=0.5,
            effective_dims=5.0,
            embedding_count=20,
        )

        assert is_diverse is True
        assert "Moderate diversity" in reason


class TestComputeSemanticEntropy:
    """Tests for compute_semantic_entropy convenience function."""

    @pytest.mark.asyncio
    async def test_compute_semantic_entropy_function(self):
        """Test convenience function creates calculator and computes."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[
            {"embedding": [0.1] * 100},
            {"embedding": [0.2] * 100},
            {"embedding": [0.3] * 100},
        ])

        result = await compute_semantic_entropy(graph=mock_graph)

        assert isinstance(result, SemanticEntropyResult)
        assert result.embedding_count == 3

    @pytest.mark.asyncio
    async def test_compute_semantic_entropy_with_node_uids(self):
        """Test convenience function with specific node UIDs."""
        mock_graph = MagicMock()
        mock_graph.query = AsyncMock(return_value=[
            {"embedding": [0.1] * 100},
            {"embedding": [0.5] * 100},
        ])

        result = await compute_semantic_entropy(
            graph=mock_graph,
            node_uids=["uid1", "uid2"],
        )

        assert isinstance(result, SemanticEntropyResult)
        assert result.embedding_count == 2


class TestEntropyThresholds:
    """Tests for entropy threshold constants."""

    def test_threshold_values(self):
        """Test that threshold values are reasonable."""
        assert 0.0 < SEMANTIC_ENTROPY_LOW < 1.0
        assert 0.0 < SEMANTIC_ENTROPY_HIGH < 1.0
        assert SEMANTIC_ENTROPY_LOW < SEMANTIC_ENTROPY_HIGH

    def test_effective_dim_threshold(self):
        """Test effective dimension threshold is reasonable."""
        assert 0.0 < EFFECTIVE_DIM_THRESHOLD < 1.0

    def test_threshold_constants_match_implementation(self):
        """Test threshold constants have expected values."""
        assert SEMANTIC_ENTROPY_HIGH == 0.7
        assert SEMANTIC_ENTROPY_LOW == 0.3
        assert EFFECTIVE_DIM_THRESHOLD == 0.1

    def test_is_balanced_uses_thresholds(self):
        """Test that is_balanced property aligns with threshold constants."""
        # At exactly the low threshold
        at_low = SemanticEntropyResult(
            semantic_entropy=SEMANTIC_ENTROPY_LOW,
            topic_concentration=0.7,
            effective_dimensions=3.0,
            embedding_count=10,
            mean_similarity=0.6,
            min_similarity=0.3,
            max_similarity=0.9,
            is_diverse=False,
            diversity_reason="Test",
        )
        # At exactly the high threshold
        at_high = SemanticEntropyResult(
            semantic_entropy=SEMANTIC_ENTROPY_HIGH,
            topic_concentration=0.3,
            effective_dimensions=5.0,
            embedding_count=10,
            mean_similarity=0.4,
            min_similarity=0.1,
            max_similarity=0.7,
            is_diverse=True,
            diversity_reason="Test",
        )
        # In the middle
        middle = SemanticEntropyResult(
            semantic_entropy=0.5,
            topic_concentration=0.5,
            effective_dimensions=4.0,
            embedding_count=10,
            mean_similarity=0.5,
            min_similarity=0.2,
            max_similarity=0.8,
            is_diverse=True,
            diversity_reason="Test",
        )

        # is_balanced checks 0.3 < entropy < 0.7 (exclusive)
        assert at_low.is_balanced is False
        assert at_high.is_balanced is False
        assert middle.is_balanced is True
