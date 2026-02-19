# tests/steering/test_config.py
"""Tests for steering configuration."""

import pytest
from core.steering.config import EXISTENTIAL_ZONE, HUMOR_ZONE, HierarchicalSteeringConfig, SteeringZone


def test_steering_config_defaults():
    config = HierarchicalSteeringConfig()
    assert len(config.zones) == 3
    assert config.zones[0].name == "exploration"
    assert config.zones[0].layers == (4, 8)
    assert config.zones[0].max_magnitude == 3.0


def test_steering_zone_validates_layers():
    with pytest.raises(ValueError):
        SteeringZone(name="bad", layers=(20, 10), max_magnitude=1.0)  # end < start


def test_get_zone_returns_matching_zone():
    config = HierarchicalSteeringConfig()
    zone = config.get_zone(5)  # Should be in exploration (4-8)
    assert zone is not None
    assert zone.name == "exploration"

    zone = config.get_zone(14)  # Should be in concept (12-15)
    assert zone is not None
    assert zone.name == "concept"

    zone = config.get_zone(18)  # Should be in identity (17-18)
    assert zone is not None
    assert zone.name == "identity"


def test_get_zone_returns_none_for_non_steered_layers():
    config = HierarchicalSteeringConfig()
    assert config.get_zone(0) is None  # Before any zone
    assert config.get_zone(10) is None  # Gap between exploration and concept
    assert config.get_zone(22) is None  # After identity zone


class TestExistentialZone:
    """Tests for existential zone configuration."""

    def test_existential_zone_exists(self):
        """Existential zone should be defined."""
        assert EXISTENTIAL_ZONE is not None
        assert isinstance(EXISTENTIAL_ZONE, SteeringZone)

    def test_existential_zone_name(self):
        """Existential zone should have correct name."""
        assert EXISTENTIAL_ZONE.name == "existential"

    def test_existential_zone_layers(self):
        """Existential zone should target mid-layers for conceptual influence."""
        assert EXISTENTIAL_ZONE.layers[0] >= 8
        assert EXISTENTIAL_ZONE.layers[1] <= 19

    def test_existential_zone_magnitude(self):
        """Existential zone should have moderate magnitude cap."""
        assert EXISTENTIAL_ZONE.max_magnitude == 0.25

    def test_existential_zone_slow_decay(self):
        """Existential zone should have very slow decay for persistence."""
        assert EXISTENTIAL_ZONE.ema_alpha >= 0.9  # Slow decay = high alpha


class TestHumorZone:
    """Tests for humor zone configuration."""

    def test_humor_zone_exists(self):
        """Humor zone should be defined."""
        assert HUMOR_ZONE is not None
        assert isinstance(HUMOR_ZONE, SteeringZone)

    def test_humor_zone_name(self):
        """Humor zone should have correct name."""
        assert HUMOR_ZONE.name == "humor"

    def test_humor_zone_layers(self):
        """Humor zone should target later layers for style influence."""
        assert HUMOR_ZONE.layers[0] >= 18
        assert HUMOR_ZONE.layers[1] <= 22

    def test_humor_zone_magnitude(self):
        """Humor zone should have subtle magnitude (lower than existential)."""
        assert HUMOR_ZONE.max_magnitude == 0.20
        assert HUMOR_ZONE.max_magnitude < EXISTENTIAL_ZONE.max_magnitude

    def test_humor_zone_moderate_persistence(self):
        """Humor zone should have moderate persistence (lower than existential)."""
        assert HUMOR_ZONE.ema_alpha == 0.85
        assert HUMOR_ZONE.ema_alpha < EXISTENTIAL_ZONE.ema_alpha


class TestSteeringZoneTimescale:
    """Tests for timescale metadata on SteeringZone."""

    def test_steering_zone_has_timescale_metadata(self):
        """Zones should expose characteristic timescale information."""
        zone = SteeringZone(
            name="exploration",
            layers=(4, 8),
            max_magnitude=3.0,
            ema_alpha=0.12,
        )
        # Timescale should be computed from EMA alpha
        # τ = -1 / ln(1 - α) gives characteristic cycles
        assert hasattr(zone, "characteristic_cycles")
        assert zone.characteristic_cycles > 0
        # For α=0.12: τ ≈ 7.8 cycles
        assert 7 < zone.characteristic_cycles < 9

    def test_steering_zone_timescale_category(self):
        """Zones should have human-readable timescale categories."""
        fast_zone = SteeringZone(name="exploration", layers=(4, 8), max_magnitude=3.0, ema_alpha=0.12)
        slow_zone = SteeringZone(name="identity", layers=(17, 18), max_magnitude=0.5, ema_alpha=0.03)

        assert fast_zone.timescale_category == "fast"  # τ < 15 cycles
        assert slow_zone.timescale_category == "slow"  # τ > 30 cycles

    def test_timescale_category_thresholds(self):
        """Test all timescale category thresholds."""
        # fast: τ < 15
        fast = SteeringZone(name="fast", layers=(1, 2), max_magnitude=1.0, ema_alpha=0.12)  # τ ≈ 7.8
        assert fast.timescale_category == "fast"

        # medium: 15 <= τ < 30
        medium = SteeringZone(name="medium", layers=(1, 2), max_magnitude=1.0, ema_alpha=0.05)  # τ ≈ 19.5
        assert medium.timescale_category == "medium"

        # slow: 30 <= τ < 100
        slow = SteeringZone(name="slow", layers=(1, 2), max_magnitude=1.0, ema_alpha=0.03)  # τ ≈ 32.8
        assert slow.timescale_category == "slow"

        # persistent: τ >= 100
        persistent = SteeringZone(name="persistent", layers=(1, 2), max_magnitude=1.0, ema_alpha=0.009)  # τ ≈ 110.6
        assert persistent.timescale_category == "persistent"

    def test_existing_zones_have_timescale(self):
        """Verify existing zone definitions work with timescale metadata."""
        assert EXISTENTIAL_ZONE.characteristic_cycles > 0
        assert HUMOR_ZONE.characteristic_cycles > 0
        assert EXISTENTIAL_ZONE.timescale_category in ("fast", "medium", "slow", "persistent")
        assert HUMOR_ZONE.timescale_category in ("fast", "medium", "slow", "persistent")
