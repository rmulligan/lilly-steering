"""Tests for layer calibration using TransformerLens activation analysis."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import asyncio

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.layer_calibrator import (
    LayerCalibrator,
    CalibrationResult,
    CachedCalibration,
)
from core.steering.contrastive_extractor import ContrastivePair


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestLayerCalibrator:
    """Tests for LayerCalibrator class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock HookedQwen model."""
        model = MagicMock()
        model.n_layers = 32
        model.model_name = "Qwen/Qwen2.5-7B"

        # Return different activation patterns per layer
        async def fake_activations(text, layers):
            result = {}
            for layer in layers:
                # Simulate stronger separation in middle layers
                if 12 <= layer <= 18:
                    result[layer] = torch.randn(1, 10, 768) * 2.0
                else:
                    result[layer] = torch.randn(1, 10, 768) * 0.5
            return result

        model.get_activations = fake_activations
        return model

    @pytest.fixture
    def sample_pairs(self):
        """Sample contrastive pairs for testing."""
        return [
            ContrastivePair(
                positive="I am Lilly, considering this...",
                negative="As an assistant, I will help...",
                behavior="identity",
            ),
        ]

    def test_calibrator_init(self, mock_model):
        """Should initialize with model reference."""
        calibrator = LayerCalibrator(model=mock_model)
        assert calibrator.model == mock_model

    @pytest.mark.asyncio
    async def test_compute_layer_separation(self, mock_model, sample_pairs):
        """Should compute separation score for a single layer."""
        calibrator = LayerCalibrator(model=mock_model)

        separation = await calibrator.compute_layer_separation(
            pair=sample_pairs[0],
            layer=15,
        )

        assert isinstance(separation, float)
        assert separation >= 0.0

    @pytest.mark.asyncio
    async def test_find_optimal_layer(self, mock_model, sample_pairs):
        """Should find the layer with strongest separation."""
        calibrator = LayerCalibrator(model=mock_model)

        result = await calibrator.find_optimal_layer(
            pairs=sample_pairs,
            layer_range=(10, 22),
        )

        assert isinstance(result, CalibrationResult)
        assert 10 <= result.optimal_layer <= 22
        assert result.separation_score > 0

    @pytest.mark.asyncio
    async def test_find_optimal_range(self, mock_model, sample_pairs):
        """Should find optimal layer range around peak."""
        calibrator = LayerCalibrator(model=mock_model)

        result = await calibrator.find_optimal_range(
            pairs=sample_pairs,
            search_range=(8, 24),
            range_width=4,
        )

        assert isinstance(result.layer_range, tuple)
        assert len(result.layer_range) == 2
        assert result.layer_range[1] - result.layer_range[0] == 4

    @pytest.mark.asyncio
    async def test_calibrate_all_vectors(self, mock_model):
        """Should calibrate all vector types and return config."""
        calibrator = LayerCalibrator(model=mock_model)

        result = await calibrator.calibrate_all()

        assert "identity" in result
        assert "autonomy" in result
        assert "anti_assistant" in result
        assert "constitutional" in result

        # Each should have layer_range
        for vector_type, calibration in result.items():
            assert hasattr(calibration, "layer_range")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_result_stores_metadata(self):
        """Should store layer, score, and metadata."""
        result = CalibrationResult(
            optimal_layer=15,
            layer_range=(13, 17),
            separation_score=0.85,
            vector_type="identity",
        )

        assert result.optimal_layer == 15
        assert result.layer_range == (13, 17)
        assert result.separation_score == 0.85


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestCachedCalibration:
    """Tests for calibration caching."""

    def test_cache_key_includes_model(self):
        """Cache key should include model identifier."""
        cache = CachedCalibration(
            model_name="Qwen/Qwen2.5-7B",
            model_hash="abc123",
            calibrations={},
        )

        assert cache.model_name == "Qwen/Qwen2.5-7B"
        assert cache.model_hash == "abc123"

    def test_is_valid_for_model(self):
        """Should validate cache against current model."""
        cache = CachedCalibration(
            model_name="Qwen/Qwen2.5-7B",
            model_hash="abc123",
            calibrations={},
        )

        assert cache.is_valid_for("Qwen/Qwen2.5-7B", "abc123")
        assert not cache.is_valid_for("Qwen/Qwen2.5-7B", "different_hash")
        assert not cache.is_valid_for("different_model", "abc123")
