"""Tests for synthesis layer vector blending."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.synthesis_blender import (
    SynthesisBlender,
    BlendConfig,
    BlendedVector,
    ArousalModulator,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestBlendConfig:
    """Tests for blend configuration."""

    def test_default_weights(self):
        """Should have sensible default weights."""
        config = BlendConfig()
        assert config.identity_weight == 1.0
        assert config.psyche_weight == 0.5
        assert config.constitutional_floor == 0.3

    def test_custom_weights(self):
        """Should accept custom weights."""
        config = BlendConfig(
            identity_weight=0.8,
            psyche_weight=0.7,
        )
        assert config.identity_weight == 0.8
        assert config.psyche_weight == 0.7

    def test_constitutional_floor_enforced(self):
        """Constitutional floor should prevent zero weighting."""
        config = BlendConfig(constitutional_floor=0.5)
        assert config.constitutional_floor == 0.5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestArousalModulator:
    """Tests for arousal-based modulation."""

    def test_neutral_arousal_no_change(self):
        """Neutral arousal (0.5) should not modify strength."""
        modulator = ArousalModulator()
        result = modulator.modulate(base_strength=1.0, arousal=0.5)
        assert result == pytest.approx(1.0)

    def test_high_arousal_increases_strength(self):
        """High arousal should increase strength."""
        modulator = ArousalModulator()
        result = modulator.modulate(base_strength=1.0, arousal=0.9)
        assert result > 1.0

    def test_low_arousal_decreases_strength(self):
        """Low arousal should decrease strength."""
        modulator = ArousalModulator()
        result = modulator.modulate(base_strength=1.0, arousal=0.1)
        assert result < 1.0

    def test_respects_min_max_bounds(self):
        """Should respect configured min/max bounds."""
        modulator = ArousalModulator(min_multiplier=0.5, max_multiplier=1.5)

        # Extreme low arousal
        low_result = modulator.modulate(base_strength=1.0, arousal=0.0)
        assert low_result == pytest.approx(0.5)

        # Extreme high arousal
        high_result = modulator.modulate(base_strength=1.0, arousal=1.0)
        assert high_result == pytest.approx(1.5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestBlendedVector:
    """Tests for blended vector output."""

    def test_blended_vector_structure(self):
        """Should have all required fields."""
        vector = torch.randn(768)
        blended = BlendedVector(
            vector=vector,
            sources=["identity", "constitutional"],
            weights={"identity": 0.6, "constitutional": 0.4},
            arousal_multiplier=1.1,
            timestamp=datetime.now(timezone.utc),
        )

        assert blended.vector is not None
        assert len(blended.sources) == 2
        assert "identity" in blended.weights
        assert blended.arousal_multiplier == 1.1

    def test_to_dict_serialization(self):
        """Should serialize to dict for logging."""
        vector = torch.randn(768)
        blended = BlendedVector(
            vector=vector,
            sources=["identity"],
            weights={"identity": 1.0},
            arousal_multiplier=1.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = blended.to_dict()
        assert "sources" in result
        assert "weights" in result
        assert "arousal_multiplier" in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSynthesisBlender:
    """Tests for SynthesisBlender class."""

    @pytest.fixture
    def mock_identity_hooks(self):
        """Create mock IdentityHooks."""
        hooks = MagicMock()
        hooks.identity_vector = torch.randn(768)
        hooks.autonomy_vector = torch.randn(768)
        hooks.anti_assistant_vector = torch.randn(768)
        hooks.constitutional_vector = torch.randn(768)

        # Mock get_vectors_for_layer to return vectors based on layer
        def get_vectors(layer):
            vectors = {}
            if 12 <= layer <= 16:
                vectors["identity"] = hooks.identity_vector
            if 14 <= layer <= 18:
                vectors["autonomy"] = hooks.autonomy_vector
            if 10 <= layer <= 14:
                vectors["anti_assistant"] = hooks.anti_assistant_vector
            if 16 <= layer <= 20:
                vectors["constitutional"] = hooks.constitutional_vector
            return vectors

        hooks.get_vectors_for_layer = get_vectors
        return hooks

    @pytest.fixture
    def mock_psyche_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock(return_value=[])
        return client

    def test_blender_init(self, mock_identity_hooks, mock_psyche_client):
        """Should initialize with required components."""
        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
        )

        assert blender.identity_hooks == mock_identity_hooks
        assert blender.psyche == mock_psyche_client
        assert blender.hidden_size == 768

    @pytest.mark.asyncio
    async def test_blend_identity_only(self, mock_identity_hooks, mock_psyche_client):
        """Should blend identity vectors when no Psyche vectors available."""
        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
        )

        result = await blender.blend(layer=15, arousal=0.5)

        assert isinstance(result, BlendedVector)
        assert result.vector.shape == (768,)
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_blend_respects_arousal(
        self, mock_identity_hooks, mock_psyche_client
    ):
        """Higher arousal should produce different blend than lower."""
        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
        )

        low_arousal = await blender.blend(layer=15, arousal=0.2)
        high_arousal = await blender.blend(layer=15, arousal=0.8)

        # Different arousal should produce different multipliers
        assert low_arousal.arousal_multiplier != high_arousal.arousal_multiplier

    @pytest.mark.asyncio
    async def test_constitutional_always_included(
        self, mock_identity_hooks, mock_psyche_client
    ):
        """Constitutional vector should always be included with floor weight.

        Uses low identity_weight and low arousal so the calculated weight
        would be below the floor, verifying the floor logic is triggered.
        """
        config = BlendConfig(
            identity_weight=0.1,  # Low weight so calculated weight < floor
            constitutional_floor=0.3,
        )
        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
            config=config,
        )

        # Test at layer within constitutional range with low arousal
        # With identity_weight=0.1 and arousal=0.1, the arousal_mult ~0.6
        # So calculated weight = 0.1 * 0.6 = 0.06, which is below floor of 0.3
        result = await blender.blend(layer=18, arousal=0.1)

        assert "constitutional" in result.sources
        # The weight should be boosted to at least the floor
        assert result.weights.get("constitutional", 0) >= 0.3

    @pytest.mark.asyncio
    async def test_load_psyche_vectors(
        self, mock_identity_hooks, mock_psyche_client
    ):
        """Should load and blend vectors from Psyche."""
        # Mock Psyche returning a steering vector
        mock_psyche_client.query = AsyncMock(return_value=[
            {
                "uid": "sv:test",
                "name": "curiosity",
                "embedding": [0.1] * 768,
                "strength": 0.8,
            }
        ])

        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
        )

        result = await blender.blend(layer=15, arousal=0.5)

        # Should have called psyche query
        mock_psyche_client.query.assert_called()

    @pytest.mark.asyncio
    async def test_blend_for_layer_range(
        self, mock_identity_hooks, mock_psyche_client
    ):
        """Should produce appropriate blend for each layer in range."""
        blender = SynthesisBlender(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche_client,
            hidden_size=768,
        )

        # Layers outside identity range should produce empty/zero blend
        result_low = await blender.blend(layer=5, arousal=0.5)
        result_mid = await blender.blend(layer=15, arousal=0.5)

        # Mid-range should have more sources than low
        assert len(result_mid.sources) >= len(result_low.sources)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSynthesisBlenderIntegration:
    """Integration-style tests for synthesis blender."""

    @pytest.fixture
    def full_blender(self):
        """Create blender with all mock dependencies."""
        hooks = MagicMock()
        hooks.identity_vector = torch.randn(768)
        hooks.autonomy_vector = torch.randn(768)
        hooks.anti_assistant_vector = torch.randn(768)
        hooks.constitutional_vector = torch.randn(768)
        hooks.layer_config = MagicMock()
        hooks.layer_config.identity_layers = (12, 16)
        hooks.layer_config.autonomy_layers = (14, 18)
        hooks.layer_config.anti_assistant_layers = (10, 14)
        hooks.layer_config.constitutional_layers = (16, 20)

        # Mock get_vectors_for_layer
        def get_vectors(layer):
            vectors = {}
            if 12 <= layer <= 16:
                vectors["identity"] = hooks.identity_vector
            if 14 <= layer <= 18:
                vectors["autonomy"] = hooks.autonomy_vector
            if 10 <= layer <= 14:
                vectors["anti_assistant"] = hooks.anti_assistant_vector
            if 16 <= layer <= 20:
                vectors["constitutional"] = hooks.constitutional_vector
            return vectors

        hooks.get_vectors_for_layer = get_vectors

        psyche = MagicMock()
        psyche.query = AsyncMock(return_value=[])

        return SynthesisBlender(
            identity_hooks=hooks,
            psyche=psyche,
            hidden_size=768,
        )

    @pytest.mark.asyncio
    async def test_create_hook_function(self, full_blender):
        """Should create callable hook function."""
        hook_fn = await full_blender.create_blend_hook(layer=15, arousal=0.5)

        assert callable(hook_fn)

        # Apply to fake activation
        activation = torch.zeros(1, 10, 768)
        result = hook_fn(activation, _hook=None)

        # Should modify activations
        assert not torch.allclose(result, activation)
