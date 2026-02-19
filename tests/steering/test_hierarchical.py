# tests/steering/test_hierarchical.py
"""Tests for HierarchicalSteerer multi-zone vector management."""
import pytest
import numpy as np
from core.steering.hierarchical import HierarchicalSteerer
from core.steering.config import HierarchicalSteeringConfig


@pytest.fixture
def steerer():
    return HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)


def test_steerer_initializes_zero_vectors(steerer):
    """All zone vectors start as zeros."""
    for zone_name, vector in steerer.vectors.items():
        assert vector.shape == (4096,)
        assert np.allclose(vector, 0)


def test_steerer_returns_none_for_observation_layer(steerer):
    """Layers >= observation_layer return None (no steering)."""
    # observation_layer is 20 in current config
    assert steerer.get_vector(20) is None
    assert steerer.get_vector(24) is None
    assert steerer.get_vector(30) is None


def test_steerer_returns_none_for_non_steered_layers(steerer):
    """Layers in gaps between zones return None."""
    # Layer 10 is between exploration (4-8) and concept (12-15)
    assert steerer.get_vector(10) is None
    # Layer 0 is before any zone
    assert steerer.get_vector(0) is None


def test_steerer_returns_vector_for_steering_layer(steerer):
    """Layers in zones return appropriate vectors."""
    # Set a non-zero vector for exploration zone
    steerer.vectors["exploration"] = np.ones(4096)

    vec = steerer.get_vector(6)  # Within exploration zone (4-8)
    assert vec is not None
    assert np.allclose(vec, np.ones(4096) * (3.0 / np.sqrt(4096)))  # Capped to max_magnitude


def test_steerer_caps_magnitude(steerer):
    """Vectors exceeding zone max_magnitude are scaled down."""
    # Create a vector with magnitude 10
    steerer.vectors["exploration"] = np.ones(4096) * 10 / np.sqrt(4096)

    vec = steerer.get_vector(6)
    magnitude = np.linalg.norm(vec)
    assert magnitude <= 3.0 + 0.01  # exploration max_magnitude = 3.0


def test_steerer_preserves_vectors_under_max_magnitude(steerer):
    """Vectors under max_magnitude are not modified."""
    # Create a vector with magnitude 1.0 (under exploration's max of 3.0)
    unit_vec = np.zeros(4096)
    unit_vec[0] = 1.0  # magnitude = 1.0
    steerer.vectors["exploration"] = unit_vec.copy()

    vec = steerer.get_vector(6)
    assert np.allclose(vec, unit_vec)


def test_steerer_update_vector_ema_blending(steerer):
    """update_vector uses EMA blending with zone's alpha."""
    # Start with zero vector
    assert np.allclose(steerer.vectors["exploration"], 0)

    # Update with a direction
    direction = np.zeros(4096)
    direction[0] = 1.0  # Unit vector pointing in one direction
    steerer.update_vector("exploration", direction, scale=1.0)

    # With ema_alpha=0.12 for exploration: (1-0.12)*0 + 0.12*1.0 = 0.12
    expected = direction * 0.12
    assert np.allclose(steerer.vectors["exploration"], expected)


def test_steerer_update_vector_normalizes_direction(steerer):
    """update_vector normalizes the new direction before blending."""
    # Create a non-unit direction
    direction = np.ones(4096) * 5.0  # magnitude = 5 * sqrt(4096) ~ 320

    steerer.update_vector("exploration", direction, scale=2.0)

    # Direction should be normalized to unit, then scaled by 2.0, then EMA blended
    # (1-0.12)*0 + 0.12 * (normalized * 2.0)
    normalized = direction / np.linalg.norm(direction)
    expected = normalized * 2.0 * 0.12
    assert np.allclose(steerer.vectors["exploration"], expected)


def test_steerer_update_vector_unknown_zone_no_op(steerer):
    """update_vector with unknown zone name does nothing."""
    original = steerer.vectors["exploration"].copy()
    steerer.update_vector("nonexistent_zone", np.ones(4096), scale=1.0)
    assert np.allclose(steerer.vectors["exploration"], original)


def test_steerer_update_vector_zero_direction_no_op(steerer):
    """update_vector with zero direction does nothing (avoids division by zero)."""
    # First, set a non-zero vector
    steerer.vectors["exploration"] = np.ones(4096) * 0.1

    original = steerer.vectors["exploration"].copy()
    steerer.update_vector("exploration", np.zeros(4096), scale=1.0)

    # Should remain unchanged since zero vector has norm 0
    assert np.allclose(steerer.vectors["exploration"], original)


def test_steerer_different_zones_different_vectors(steerer):
    """Each zone maintains its own independent vector."""
    steerer.vectors["exploration"] = np.ones(4096) * 0.1
    steerer.vectors["concept"] = np.ones(4096) * 0.2
    steerer.vectors["identity"] = np.ones(4096) * 0.3

    # Get vectors from each zone
    exploration_vec = steerer.get_vector(6)   # exploration (4-8)
    concept_vec = steerer.get_vector(14)      # concept (12-15)
    identity_vec = steerer.get_vector(17)     # identity (17-18)

    # Each should reflect its zone's vector (possibly capped)
    assert not np.allclose(exploration_vec, concept_vec)
    assert not np.allclose(concept_vec, identity_vec)


def test_steerer_adjust_max_magnitude():
    """Zone max_magnitude can be adjusted at runtime."""
    steerer = HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)

    # Set up a vector that exceeds the initial max_magnitude
    steerer.vectors["exploration"] = np.ones(4096) * 10 / np.sqrt(4096)  # magnitude ~10

    # Initial cap should be 3.0
    vec = steerer.get_vector(6)
    assert np.linalg.norm(vec) <= 3.0 + 0.01

    # Adjust the max_magnitude upward
    steerer.adjust_zone_parameter("exploration", "max_magnitude", 15.0)

    # Now the vector should not be capped
    vec = steerer.get_vector(6)
    assert np.linalg.norm(vec) > 9.0  # Should be close to original 10


def test_steerer_adjust_ema_alpha():
    """Zone ema_alpha can be adjusted at runtime."""
    steerer = HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)

    # Initial ema_alpha for exploration is 0.12
    direction = np.zeros(4096)
    direction[0] = 1.0
    steerer.update_vector("exploration", direction, scale=1.0)

    # Should get 0.12 of the direction
    assert np.allclose(steerer.vectors["exploration"], direction * 0.12)

    # Adjust ema_alpha to 0.5
    steerer.adjust_zone_parameter("exploration", "ema_alpha", 0.5)

    # Reset and update again
    steerer.vectors["exploration"] = np.zeros(4096)
    steerer.update_vector("exploration", direction, scale=1.0)

    # Should now get 0.5 of the direction
    assert np.allclose(steerer.vectors["exploration"], direction * 0.5)


def test_steerer_adjust_unknown_zone_raises():
    """Adjusting unknown zone raises KeyError."""
    steerer = HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)

    with pytest.raises(KeyError):
        steerer.adjust_zone_parameter("nonexistent", "max_magnitude", 5.0)


def test_steerer_adjust_unknown_parameter_raises():
    """Adjusting unknown parameter raises ValueError."""
    steerer = HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)

    with pytest.raises(ValueError):
        steerer.adjust_zone_parameter("exploration", "unknown_param", 5.0)


def test_steerer_get_zone_parameters():
    """Can retrieve current zone parameters."""
    steerer = HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=4096)

    params = steerer.get_zone_parameters("exploration")
    assert params["max_magnitude"] == 3.0
    assert params["ema_alpha"] == 0.12
    assert params["layers"] == (4, 8)

    # Adjust and verify
    steerer.adjust_zone_parameter("exploration", "max_magnitude", 5.0)
    params = steerer.get_zone_parameters("exploration")
    assert params["max_magnitude"] == 5.0
