# tests/steering/test_population_steerer.py
"""Tests for PopulationSteerer multi-vector management."""
import pytest
import numpy as np
from core.steering.config import HierarchicalSteeringConfig


@pytest.fixture
def config():
    return HierarchicalSteeringConfig()


def test_population_steerer_initialization(config):
    """PopulationSteerer initializes with empty populations per zone."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    assert steerer.d_model == 4096
    assert len(steerer.populations) == len(config.zones)
    for zone in config.zones:
        assert zone.name in steerer.populations
        assert len(steerer.populations[zone.name]) == 0


def test_population_steerer_add_vector(config):
    """Can add vectors to zone populations."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)
    vector = np.random.randn(4096).astype(np.float32)

    steerer.add_vector("exploration", "curious", vector)

    assert len(steerer.populations["exploration"]) == 1
    assert "curious" in steerer.populations["exploration"]


def test_population_steerer_get_vector_single(config):
    """get_vector with single vector returns it (capped)."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)
    # Create vector with large magnitude
    vector = np.ones(4096, dtype=np.float32) * 10
    steerer.add_vector("exploration", "test", vector)

    # Layer 6 is in exploration zone (4-8)
    result = steerer.get_vector(6)

    assert result is not None
    magnitude = np.linalg.norm(result)
    # Should be capped to exploration max_magnitude (3.0)
    assert magnitude <= 3.0 + 0.01


def test_population_steerer_get_vector_selects_by_affinity(config):
    """get_vector selects vector with highest effective score."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    # Add two vectors
    v1 = np.zeros(4096, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(4096, dtype=np.float32)
    v2[1] = 1.0

    steerer.add_vector("exploration", "low_affinity", v1)
    steerer.add_vector("exploration", "high_affinity", v2)

    # Set affinities (high_affinity has higher)
    steerer.affinity_matrix.set("default", "low_affinity", 0.3)
    steerer.affinity_matrix.set("default", "high_affinity", 0.9)

    result = steerer.get_vector(6)

    # Should select high_affinity (v2 points in direction [0, 1, 0, ...])
    assert result is not None
    assert result[1] > result[0]  # v2's direction


def test_population_steerer_staleness_affects_selection(config):
    """Stale vectors are penalized in selection."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    v1 = np.zeros(4096, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(4096, dtype=np.float32)
    v2[1] = 1.0

    steerer.add_vector("exploration", "fresh", v1)
    steerer.add_vector("exploration", "stale", v2)

    # Same base affinity
    steerer.affinity_matrix.set("default", "fresh", 0.8)
    steerer.affinity_matrix.set("default", "stale", 0.8)

    # Make "stale" very stale
    steerer.populations["exploration"].get("stale").staleness = 100.0

    result = steerer.get_vector(6)

    # Should select "fresh" because "stale" has 0.5 staleness penalty
    assert result is not None
    assert result[0] > result[1]  # fresh's direction


def test_population_steerer_record_selection(config):
    """record_selection updates staleness."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)
    vector = np.zeros(4096, dtype=np.float32)
    steerer.add_vector("exploration", "test", vector)

    entry = steerer.populations["exploration"].get("test")
    assert entry.staleness == 0.0

    steerer.record_selection("exploration", "test")
    assert entry.staleness == 1.0
    assert entry.selection_count == 1


def test_population_steerer_apply_cycle_decay(config):
    """apply_cycle_decay decays all populations."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)
    steerer.add_vector("exploration", "test", np.zeros(4096, dtype=np.float32))

    entry = steerer.populations["exploration"].get("test")
    entry.staleness = 10.0

    steerer.apply_cycle_decay()

    assert entry.staleness < 10.0


def test_population_steerer_empty_population_returns_none(config):
    """get_vector returns None for empty population."""
    from core.steering.population_steerer import PopulationSteerer

    steerer = PopulationSteerer(config, d_model=4096, population_size=5)

    # Layer 6 is in exploration zone, but population is empty
    result = steerer.get_vector(6)
    assert result is None
