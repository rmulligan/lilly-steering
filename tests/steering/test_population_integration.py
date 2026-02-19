# tests/steering/test_population_integration.py
"""Integration tests for PopulationSteerer with HookedQwen generation."""
import pytest
import numpy as np


@pytest.fixture
def mock_model():
    """Create a mock HookedQwen for testing."""
    from unittest.mock import MagicMock, AsyncMock

    model = MagicMock()
    model.d_model = 4096

    # Mock generate_with_cache to return thought and cache
    async def mock_generate(*args, **kwargs):
        return "A thought about curiosity.", {"resid_post": MagicMock()}

    model.generate_with_cache = AsyncMock(side_effect=mock_generate)
    return model


@pytest.mark.asyncio
async def test_population_steerer_with_generation(mock_model):
    """PopulationSteerer can be used in generation flow."""
    from core.steering.config import HierarchicalSteeringConfig
    from core.steering.population_steerer import PopulationSteerer

    config = HierarchicalSteeringConfig()
    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    # Add vectors to each zone
    for zone in config.zones:
        vector = np.random.randn(4096).astype(np.float32)
        steerer.add_vector(zone.name, f"{zone.name}_default", vector)

    # Set prompt context
    steerer.set_prompt_context("What tensions do I feel?")

    # Simulate getting vectors for all steering layers
    vectors_applied = []
    for layer in range(config.observation_layer):
        vec = steerer.get_vector(layer)
        if vec is not None:
            vectors_applied.append(layer)

    # Should have vectors for layers in zone ranges
    assert len(vectors_applied) > 0

    # Record selections
    steerer.record_last_selections()

    # Check staleness was updated
    for zone in config.zones:
        entry = steerer.populations[zone.name].get(f"{zone.name}_default")
        assert entry.selection_count == 1


@pytest.mark.asyncio
async def test_affinity_learning_from_surprise():
    """High surprise should boost affinity for selected path."""
    from core.steering.config import HierarchicalSteeringConfig
    from core.steering.population_steerer import PopulationSteerer

    config = HierarchicalSteeringConfig()
    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    # Add a vector
    vector = np.random.randn(4096).astype(np.float32)
    steerer.add_vector("exploration", "curious", vector)

    prompt_key = "What tensions do I feel?"
    steerer.set_prompt_context(prompt_key)

    # Get initial affinity (default)
    initial_affinity = steerer.affinity_matrix.get(prompt_key, "curious")
    assert initial_affinity == 0.5  # Default

    # Simulate high surprise outcome -> boost affinity
    surprise_score = 0.8  # High surprise
    steerer.update_affinity(prompt_key, "curious", surprise_score)

    # Affinity should have increased (EMA toward 0.8)
    new_affinity = steerer.affinity_matrix.get(prompt_key, "curious")
    assert new_affinity > initial_affinity
