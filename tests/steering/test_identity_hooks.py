"""Tests for identity hooks multi-vector injection."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.identity_hooks import (
    IdentityHooks,
    LayerConfig,
    InjectionConfig,
)


class TestLayerConfig:
    """Tests for layer configuration."""

    def test_default_layer_ranges(self):
        """Should have sensible default layer ranges."""
        config = LayerConfig()
        assert config.identity_layers == (12, 16)
        assert config.autonomy_layers == (14, 18)
        assert config.anti_assistant_layers == (10, 14)
        assert config.constitutional_layers == (16, 20)

    def test_custom_layer_ranges(self):
        """Should accept custom layer ranges."""
        config = LayerConfig(
            identity_layers=(10, 14),
            autonomy_layers=(12, 16),
        )
        assert config.identity_layers == (10, 14)
        assert config.autonomy_layers == (12, 16)


class TestInjectionConfig:
    """Tests for injection configuration."""

    def test_default_strengths(self):
        """Should have sensible default strengths."""
        config = InjectionConfig()
        assert config.identity_strength == 1.0
        assert config.autonomy_strength == 1.0
        assert config.anti_assistant_strength == 0.5
        assert config.constitutional_strength == 1.0

    def test_anti_assistant_negated(self):
        """Anti-assistant should be marked for negation."""
        config = InjectionConfig()
        assert config.negate_anti_assistant is True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestIdentityHooks:
    """Tests for IdentityHooks class."""

    @pytest.fixture
    def mock_extractor(self):
        """Create mock contrastive extractor."""
        extractor = MagicMock()
        # Return fake 768-dim vectors - use AsyncMock for async method
        extractor.extract_from_pairs = AsyncMock(return_value=torch.randn(768))
        return extractor

    @pytest.mark.asyncio
    async def test_init_extracts_vectors(self, mock_extractor):
        """Should extract all vector types on initialize."""
        hooks = IdentityHooks(extractor=mock_extractor, hidden_size=768)
        await hooks.initialize()

        # Should have called extract for each vector type
        assert mock_extractor.extract_from_pairs.call_count >= 4

    @pytest.mark.asyncio
    async def test_vectors_stored_correctly(self, mock_extractor):
        """Should store extracted vectors."""
        hooks = IdentityHooks(extractor=mock_extractor, hidden_size=768)
        await hooks.initialize()

        assert hooks.identity_vector is not None
        assert hooks.autonomy_vector is not None
        assert hooks.anti_assistant_vector is not None
        assert hooks.constitutional_vector is not None

    @pytest.mark.asyncio
    async def test_get_hooks_for_layer(self, mock_extractor):
        """Should return appropriate vectors for each layer."""
        hooks = IdentityHooks(extractor=mock_extractor, hidden_size=768)
        await hooks.initialize()

        # Layer 12 should include identity
        vectors_12 = hooks.get_vectors_for_layer(12)
        assert "identity" in vectors_12

        # Layer 15 should include identity, autonomy
        vectors_15 = hooks.get_vectors_for_layer(15)
        assert "identity" in vectors_15
        assert "autonomy" in vectors_15

        # Layer 5 should include nothing (below all ranges)
        vectors_5 = hooks.get_vectors_for_layer(5)
        assert len(vectors_5) == 0

    @pytest.mark.asyncio
    async def test_create_hook_function(self, mock_extractor):
        """Should create valid hook function."""
        hooks = IdentityHooks(extractor=mock_extractor, hidden_size=768)
        await hooks.initialize()

        hook_fn = hooks.create_hook_function(layer=15)
        assert callable(hook_fn)

    @pytest.mark.asyncio
    async def test_hook_adds_vectors_to_activations(self, mock_extractor):
        """Hook should add vectors to activations."""
        hooks = IdentityHooks(extractor=mock_extractor, hidden_size=768)
        await hooks.initialize()
        hook_fn = hooks.create_hook_function(layer=15)

        # Create fake activation tensor [batch, seq, hidden]
        activation = torch.zeros(1, 10, 768)

        # Apply hook
        result = hook_fn(activation, hook=None)

        # Result should be modified (non-zero)
        assert not torch.allclose(result, torch.zeros_like(result))
