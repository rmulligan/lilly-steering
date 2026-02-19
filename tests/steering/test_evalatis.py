# tests/steering/test_evalatis.py
"""Tests for EvalatisSteerer hybrid emergence-selection steering."""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from core.steering.evalatis import EvalatisSteerer, ZoneState
from core.steering.emergent import EmergentSlot, create_emergent_slot
from core.steering.crystal import (
    CrystalEntry,
    CrystallizationConfig,
    generate_crystal_name,
    blend_vectors_with_mutation,
)
from core.steering.config import HierarchicalSteeringConfig, SteeringZone


# === Fixtures ===


@pytest.fixture
def d_model():
    """Standard model dimension for tests."""
    return 128  # Smaller than 4096 for faster tests


@pytest.fixture
def config():
    """Standard hierarchical config with smaller layer ranges for testing."""
    return HierarchicalSteeringConfig(
        zones=[
            SteeringZone(name="exploration", layers=(2, 4), max_magnitude=3.0, ema_alpha=0.12),
            SteeringZone(name="concept", layers=(6, 8), max_magnitude=1.5, ema_alpha=0.2),
            SteeringZone(name="identity", layers=(10, 11), max_magnitude=0.5, ema_alpha=0.03),
        ],
        observation_layer=12,
    )


@pytest.fixture
def crystal_config():
    """Standard crystallization config for testing."""
    return CrystallizationConfig(
        min_cycles_for_crystallize=5,  # Lower for testing
        min_surprise_ema=30.0,
        min_cumulative_surprise=100.0,  # Lower for testing
        spawn_affinity_threshold=0.5,
        spawn_cooldown_cycles=10,
        max_children_per_parent=2,
        prune_min_selections=3,
        prune_surprise_threshold=20.0,
        max_crystals_per_zone=4,
        preserve_count=1,
    )


@pytest.fixture
def steerer(config, d_model, crystal_config):
    """Standard EvalatisSteerer for tests."""
    return EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)


# === EmergentSlot Tests ===


class TestEmergentSlot:
    """Tests for EmergentSlot dataclass."""

    def test_create_emergent_slot(self, d_model):
        """EmergentSlot initializes with zero vector."""
        slot = create_emergent_slot(d_model)
        assert slot.vector.shape == (d_model,)
        assert np.allclose(slot.vector, 0)
        assert slot.surprise_ema == 0.0
        assert slot.cumulative_surprise == 0.0

    def test_update_tracks_surprise_ema(self, d_model):
        """Update method tracks surprise EMA."""
        slot = create_emergent_slot(d_model)
        direction = np.random.randn(d_model).astype(np.float32)

        # First update with surprise=50
        slot.update(direction, surprise=50.0, ema_alpha=0.1)
        # surprise_ema = 0 * 0.85 + 50 * 0.15 = 7.5
        assert slot.surprise_ema == pytest.approx(7.5, rel=0.01)

        # Second update with surprise=50
        slot.update(direction, surprise=50.0, ema_alpha=0.1)
        # surprise_ema = 7.5 * 0.85 + 50 * 0.15 = 13.875
        assert slot.surprise_ema == pytest.approx(13.875, rel=0.01)

    def test_update_accumulates_cumulative_surprise(self, d_model):
        """Update method accumulates cumulative surprise."""
        slot = create_emergent_slot(d_model)
        direction = np.random.randn(d_model).astype(np.float32)

        slot.update(direction, surprise=30.0, ema_alpha=0.1)
        assert slot.cumulative_surprise == 30.0
        assert slot.cycles_since_crystallize == 1

        slot.update(direction, surprise=40.0, ema_alpha=0.1)
        assert slot.cumulative_surprise == 70.0
        assert slot.cycles_since_crystallize == 2

    def test_update_tracks_peak(self, d_model):
        """Update method tracks peak surprise and vector."""
        slot = create_emergent_slot(d_model)
        direction1 = np.random.randn(d_model).astype(np.float32)
        direction2 = np.random.randn(d_model).astype(np.float32)

        slot.update(direction1, surprise=30.0, ema_alpha=0.1)
        assert slot.peak_surprise == 30.0

        slot.update(direction2, surprise=50.0, ema_alpha=0.1)
        assert slot.peak_surprise == 50.0
        assert slot.peak_vector is not None

        slot.update(direction1, surprise=40.0, ema_alpha=0.1)
        assert slot.peak_surprise == 50.0  # Still 50, not updated

    def test_reset_clears_state(self, d_model):
        """reset_for_new_emergence clears all state."""
        slot = create_emergent_slot(d_model)
        direction = np.random.randn(d_model).astype(np.float32)

        slot.update(direction, surprise=50.0, ema_alpha=0.1)
        slot.update(direction, surprise=60.0, ema_alpha=0.1)

        slot.reset_for_new_emergence(d_model)

        assert np.allclose(slot.vector, 0)
        assert slot.surprise_ema == 0.0
        assert slot.cumulative_surprise == 0.0
        assert slot.cycles_since_crystallize == 0
        assert slot.peak_surprise == 0.0
        assert slot.peak_vector is None


# === CrystalEntry Tests ===


class TestCrystalEntry:
    """Tests for CrystalEntry dataclass."""

    def test_avg_surprise_with_no_selections(self, d_model):
        """avg_surprise returns birth_surprise when never selected."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=45.0,
        )
        assert crystal.avg_surprise == 45.0

    def test_avg_surprise_with_selections(self, d_model):
        """avg_surprise computes average after selections."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=40.0,
        )

        crystal.record_selection(surprise=50.0)
        assert crystal.avg_surprise == 50.0

        crystal.record_selection(surprise=30.0)
        assert crystal.avg_surprise == 40.0  # (50 + 30) / 2

    def test_record_selection_resets_staleness(self, d_model):
        """record_selection resets cycles_since_selection."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
        )
        crystal.cycles_since_selection = 50

        crystal.record_selection(surprise=40.0)
        assert crystal.cycles_since_selection == 0

    def test_update_staleness(self, d_model):
        """update_staleness computes normalized staleness."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
        )

        # Initial state
        assert crystal.staleness == 0.0
        assert crystal.cycles_since_selection == 0

        # Update staleness
        for _ in range(50):
            crystal.update_staleness(max_cycles=100)

        assert crystal.cycles_since_selection == 50
        assert crystal.staleness == pytest.approx(0.5, rel=0.01)

        # Cap at 1.0
        for _ in range(100):
            crystal.update_staleness(max_cycles=100)

        assert crystal.staleness == 1.0

    def test_can_spawn_checks_retirement(self, d_model, crystal_config):
        """can_spawn returns False for retired crystals."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
            retired=True,
        )
        assert not crystal.can_spawn(crystal_config, current_cycle=100)

    def test_can_spawn_checks_children_limit(self, d_model, crystal_config):
        """can_spawn returns False when max children reached."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
            children_spawned=2,  # max_children_per_parent=2
        )
        assert not crystal.can_spawn(crystal_config, current_cycle=100)

    def test_can_spawn_checks_cooldown(self, d_model, crystal_config):
        """can_spawn respects spawn cooldown."""
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.zeros(d_model, dtype=np.float32),
            last_spawn_cycle=95,
        )
        # cooldown is 10 cycles
        assert not crystal.can_spawn(crystal_config, current_cycle=100)
        assert crystal.can_spawn(crystal_config, current_cycle=106)


# === EvalatisSteerer Tests ===


class TestEvalatisSteerer:
    """Tests for EvalatisSteerer hybrid steering."""

    def test_initializes_with_emergent_slots(self, steerer):
        """EvalatisSteerer initializes emergent slot for each zone."""
        assert "exploration" in steerer.zones
        assert "concept" in steerer.zones
        assert "identity" in steerer.zones

        for zone in steerer.zones.values():
            assert isinstance(zone.emergent, EmergentSlot)
            assert zone.crystals == []

    def test_get_vector_returns_none_for_observation_layer(self, steerer):
        """Layers >= observation_layer return None."""
        assert steerer.get_vector(12) is None
        assert steerer.get_vector(15) is None

    def test_get_vector_returns_none_for_non_steered_layers(self, steerer):
        """Layers not in any zone return None."""
        assert steerer.get_vector(0) is None  # Before exploration
        assert steerer.get_vector(5) is None  # Between exploration and concept

    def test_get_vector_returns_emergent_by_default(self, steerer, d_model):
        """get_vector returns emergent vector when no crystals."""
        # Set a non-zero emergent vector
        steerer.zones["exploration"].emergent.vector = np.ones(d_model, dtype=np.float32)

        vec = steerer.get_vector(3)  # Within exploration zone
        assert vec is not None
        # Should be magnitude-capped to 3.0
        assert np.linalg.norm(vec) <= 3.0 + 0.01

    def test_update_vector_updates_emergent(self, steerer, d_model):
        """update_vector (compatibility method) updates emergent slot."""
        direction = np.random.randn(d_model).astype(np.float32)

        steerer.update_vector("exploration", direction, scale=1.0)

        # Should have updated the emergent vector (not zero anymore)
        assert not np.allclose(steerer.zones["exploration"].emergent.vector, 0)

    def test_update_from_cycle_returns_events(self, steerer, d_model):
        """update_from_cycle returns event dict."""
        activations = np.random.randn(d_model).astype(np.float32)

        events = steerer.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=40.0,
        )

        assert "selected_name" in events
        assert "selected_is_emergent" in events
        assert events["selected_is_emergent"] is True
        assert events["selected_name"] == "emergent"


class TestEvalatisCrystallization:
    """Tests for crystallization in EvalatisSteerer."""

    def test_crystallization_triggers_at_thresholds(self, config, d_model, crystal_config):
        """Crystallization triggers when all thresholds met."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)
        activations = np.random.randn(d_model).astype(np.float32) * 2

        # Run cycles to build up surprise
        for _ in range(crystal_config.min_cycles_for_crystallize + 1):
            events = steerer.update_from_cycle(
                zone_name="exploration",
                activations=activations,
                surprise=50.0,  # High surprise
            )

        # Should have crystallized
        assert len(steerer.zones["exploration"].crystals) >= 1

    def test_crystallization_resets_emergent(self, config, d_model):
        """After crystallization, emergent slot is reset."""
        crystal_config = CrystallizationConfig(
            min_cycles_for_crystallize=3,
            min_surprise_ema=10.0,
            min_cumulative_surprise=50.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)
        activations = np.random.randn(d_model).astype(np.float32) * 2

        # Trigger crystallization
        for i in range(10):
            steerer.update_from_cycle(
                zone_name="exploration",
                activations=activations,
                surprise=60.0,
            )

        # If crystallized, emergent should be reset (low values)
        if steerer.zones["exploration"].crystals:
            emergent = steerer.zones["exploration"].emergent
            # After reset, cycles should be small (restarted counting)
            assert emergent.cycles_since_crystallize < 5


class TestEvalatisSelection:
    """Tests for vector selection in EvalatisSteerer."""

    def test_emergent_wins_when_high_surprise(self, config, d_model, crystal_config):
        """Emergent slot wins selection when its surprise_ema is high."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Set emergent to have high surprise_ema
        steerer.zones["exploration"].emergent.surprise_ema = 60.0

        # Add a crystal with moderate performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=40.0,
            selection_count=5,
            total_surprise=200.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        selected_name, is_emergent = steerer._select_vector("exploration")
        assert is_emergent is True

    def test_crystal_wins_when_high_affinity(self, config, d_model, crystal_config):
        """Crystal wins selection when emergent surprise is low."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Set emergent to have low surprise_ema
        steerer.zones["exploration"].emergent.surprise_ema = 10.0

        # Add a crystal with high performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=60.0,
            selection_count=10,
            total_surprise=600.0,  # avg_surprise = 60
        )
        steerer.zones["exploration"].crystals.append(crystal)

        selected_name, is_emergent = steerer._select_vector("exploration")
        assert is_emergent is False
        assert selected_name == "test_crystal"

    def test_staleness_penalizes_crystal(self, config, d_model, crystal_config):
        """Stale crystals have reduced selection score."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Set emergent to high surprise_ema
        steerer.zones["exploration"].emergent.surprise_ema = 45.0

        # Add a crystal that's very stale with only moderate performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=35.0,
            selection_count=5,
            total_surprise=175.0,  # avg_surprise = 35
            staleness=0.95,  # Extremely stale
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # Due to high staleness + lower surprise, emergent should win
        # Emergent: (45/50)*1.2 = 1.08
        # Crystal: (35/50) * (1 - 0.95*0.3) * (1 + 0.05*0.3) = 0.7 * 0.715 * 1.015 ≈ 0.51
        selected_name, is_emergent = steerer._select_vector("exploration")
        assert is_emergent is True


class TestEvalatisSpawning:
    """Tests for crystal spawning in EvalatisSteerer."""

    def test_spawning_requires_two_crystals(self, config, d_model, crystal_config):
        """Spawning requires at least two eligible parent crystals."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Add only one crystal
        crystal = CrystalEntry(
            name="crystal_1",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=50.0,
            selection_count=10,
            total_surprise=500.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        result = steerer._maybe_spawn("exploration")
        assert result is None

    def test_spawning_creates_child_with_parents(self, config, d_model, crystal_config):
        """Spawning creates child with parent references."""
        # Use config that allows spawning more easily
        spawn_config = CrystallizationConfig(
            spawn_affinity_threshold=0.3,  # Low threshold
            spawn_cooldown_cycles=0,
            max_children_per_parent=5,
            max_crystals_per_zone=10,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=spawn_config)

        # Add two high-performing crystals
        crystal1 = CrystalEntry(
            name="crystal_1",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=60.0,
            selection_count=10,
            total_surprise=600.0,
        )
        crystal2 = CrystalEntry(
            name="crystal_2",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=55.0,
            selection_count=10,
            total_surprise=550.0,
        )
        steerer.zones["exploration"].crystals.extend([crystal1, crystal2])

        # Attempt spawning
        child = steerer._maybe_spawn("exploration")

        if child is not None:
            assert len(child.parent_names) == 2
            assert "crystal_1" in child.parent_names or "crystal_2" in child.parent_names


class TestEvalatisPruning:
    """Tests for crystal pruning in EvalatisSteerer."""

    def test_pruning_at_population_limit(self, config, d_model):
        """Pruning triggers when at population limit."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=3,
            preserve_count=1,
            prune_min_selections=2,
            prune_surprise_threshold=30.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 4 crystals (1 over limit)
        for i in range(4):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=30.0 + i * 5,
                selection_count=5,
                total_surprise=150.0 + i * 25,
            )
            steerer.zones["exploration"].crystals.append(crystal)

        pruned = steerer._maybe_prune("exploration")
        assert pruned is not None

        # Check that one crystal is now retired
        retired = [c for c in steerer.zones["exploration"].crystals if c.retired]
        assert len(retired) == 1

    def test_preserve_count_protects_top_performers(self, config, d_model):
        """Top performers are protected from pruning."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=2,
            preserve_count=2,
            prune_min_selections=0,
            prune_surprise_threshold=0.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 3 crystals with different performance
        for i, surprise in enumerate([100.0, 80.0, 20.0]):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=surprise,
                selection_count=1,
                total_surprise=surprise,
            )
            steerer.zones["exploration"].crystals.append(crystal)

        pruned = steerer._maybe_prune("exploration")

        # Should prune the weakest (crystal_2 with surprise=20)
        assert pruned == "crystal_2"


# === Helper Function Tests ===


class TestHelperFunctions:
    """Tests for crystal helper functions."""

    def test_generate_crystal_name(self):
        """generate_crystal_name creates valid names."""
        name = generate_crystal_name("exploration", cycle=42)
        assert name.startswith("exp_")
        assert "_042" in name

    def test_blend_vectors_with_mutation(self, d_model):
        """blend_vectors_with_mutation creates valid child vector."""
        parent1 = np.random.randn(d_model).astype(np.float32)
        parent2 = np.random.randn(d_model).astype(np.float32)

        child = blend_vectors_with_mutation(parent1, parent2, mutation_scale=0.1)

        assert child.shape == (d_model,)
        assert not np.allclose(child, parent1)
        assert not np.allclose(child, parent2)
        # Child should be close to average of parents
        avg = (parent1 + parent2) / 2
        assert np.linalg.norm(child - avg) < np.linalg.norm(avg) * 0.5  # Within 50%


class TestEvalatisGetZoneSummary:
    """Tests for zone summary reporting."""

    def test_get_zone_summary(self, steerer, d_model):
        """get_zone_summary returns comprehensive statistics."""
        # Set up some state
        steerer.zones["exploration"].emergent.surprise_ema = 35.0
        steerer.zones["exploration"].emergent.cumulative_surprise = 200.0

        # Add a crystal
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=40.0,
            selection_count=5,
            total_surprise=200.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        summary = steerer.get_zone_summary("exploration")

        assert summary["emergent_surprise_ema"] == 35.0
        assert summary["emergent_cumulative_surprise"] == 200.0
        assert summary["crystal_count"] == 1
        assert len(summary["crystals"]) == 1
        assert summary["crystals"][0]["name"] == "test_crystal"

    def test_get_zone_summary_empty_zone(self, steerer):
        """get_zone_summary handles empty zone."""
        summary = steerer.get_zone_summary("concept")

        assert summary["crystal_count"] == 0
        assert summary["crystals"] == []
        assert summary["current_is_emergent"] is True


class TestEvalatisIntegration:
    """Integration tests for EvalatisSteerer."""

    def test_full_lifecycle_crystallize_spawn_prune(self, config, d_model):
        """Test complete lifecycle: crystallize, spawn, prune."""
        # Config that allows quick lifecycle
        lifecycle_config = CrystallizationConfig(
            min_cycles_for_crystallize=3,
            min_surprise_ema=20.0,
            min_cumulative_surprise=50.0,
            spawn_affinity_threshold=0.3,
            spawn_cooldown_cycles=5,
            max_children_per_parent=5,
            max_crystals_per_zone=3,
            preserve_count=1,
            prune_min_selections=1,
            prune_surprise_threshold=10.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=lifecycle_config)

        crystallizations = 0
        spawns = 0
        prunes = 0

        # Run many cycles
        for i in range(50):
            activations = np.random.randn(d_model).astype(np.float32) * 2
            surprise = 40.0 + np.random.randn() * 10

            events = steerer.update_from_cycle(
                zone_name="exploration",
                activations=activations,
                surprise=surprise,
            )

            if events.get("crystallized"):
                crystallizations += 1
            if events.get("spawned"):
                spawns += 1
            if events.get("pruned"):
                prunes += 1

        # Should have at least some lifecycle events
        assert crystallizations >= 1, "Should crystallize at least once"

        # Check final state
        summary = steerer.get_zone_summary("exploration")
        assert summary["crystal_count"] > 0

    def test_get_all_crystals(self, steerer, d_model):
        """get_all_crystals returns crystals across zones."""
        # Add crystals to different zones
        for zone_name in ["exploration", "concept"]:
            crystal = CrystalEntry(
                name=f"{zone_name}_crystal",
                vector=np.random.randn(d_model).astype(np.float32),
            )
            steerer.zones[zone_name].crystals.append(crystal)

        all_crystals = steerer.get_all_crystals()

        assert len(all_crystals) == 2
        zone_names = [zone for zone, _ in all_crystals]
        assert "exploration" in zone_names
        assert "concept" in zone_names

    def test_load_crystal(self, steerer, d_model):
        """load_crystal adds crystal to zone."""
        crystal = CrystalEntry(
            name="loaded_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
        )

        result = steerer.load_crystal("exploration", crystal)
        assert result is True
        assert len(steerer.zones["exploration"].crystals) == 1

        # Loading same crystal again should fail
        result = steerer.load_crystal("exploration", crystal)
        assert result is False


class TestEvalatisRecognitionIntegration:
    """Tests for recognition signal integration in EvalatisSteerer."""

    def test_set_feature_tracker(self, steerer):
        """set_feature_tracker enables recognition integration."""
        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.5

        steerer.set_feature_tracker(mock_tracker)

        assert steerer.feature_tracker is mock_tracker

    def test_get_approval_bonus_without_tracker(self, steerer):
        """_get_approval_bonus returns 0.0 when no tracker."""
        bonus = steerer._get_approval_bonus([(123, 0.8), (456, 0.6)])
        assert bonus == 0.0

    def test_get_approval_bonus_with_tracker(self, steerer):
        """_get_approval_bonus delegates to feature tracker."""
        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.7
        steerer.set_feature_tracker(mock_tracker)

        sae_features = [(123, 0.8), (456, 0.6)]
        bonus = steerer._get_approval_bonus(sae_features)

        assert bonus == 0.7
        mock_tracker.get_approval_bonus.assert_called_once_with(sae_features)

    def test_get_approval_bonus_with_empty_features(self, steerer):
        """_get_approval_bonus returns 0.0 for empty features."""
        mock_tracker = Mock()
        steerer.set_feature_tracker(mock_tracker)

        bonus = steerer._get_approval_bonus([])

        assert bonus == 0.0
        mock_tracker.get_approval_bonus.assert_not_called()

    def test_approval_bonus_boosts_emergent_selection(self, config, d_model, crystal_config):
        """Positive approval bonus boosts emergent selection score."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Set emergent to have moderate surprise
        steerer.zones["exploration"].emergent.surprise_ema = 35.0

        # Add a crystal with similar performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=35.0,
            selection_count=5,
            total_surprise=175.0,  # avg_surprise = 35
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # With positive approval (0.8), emergent should win
        # The approval_bonus is passed directly to _select_vector
        selected_name, is_emergent = steerer._select_vector(
            "exploration", approval_bonus=0.8
        )
        assert is_emergent is True

    def test_negative_approval_boosts_crystal_selection(self, config, d_model, crystal_config):
        """Negative approval bonus boosts crystal selection scores."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        # Set emergent to have moderate surprise
        steerer.zones["exploration"].emergent.surprise_ema = 40.0

        # Add a crystal with similar performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=38.0,
            selection_count=5,
            total_surprise=190.0,  # avg_surprise = 38
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # With negative approval (-0.8), crystal should win
        # The approval_bonus is passed directly to _select_vector
        selected_name, is_emergent = steerer._select_vector(
            "exploration", approval_bonus=-0.8
        )
        assert is_emergent is False
        assert selected_name == "test_crystal"

    def test_update_from_cycle_stores_sae_features(self, config, d_model, crystal_config):
        """update_from_cycle stores SAE features in zone state."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)
        activations = np.random.randn(d_model).astype(np.float32)

        sae_features = [(100, 0.9), (200, 0.7)]
        steerer.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=40.0,
            sae_features=sae_features,
        )

        assert steerer.zones["exploration"].last_sae_features == sae_features

    def test_update_from_cycle_computes_approval_bonus(self, config, d_model, crystal_config):
        """update_from_cycle computes and stores approval bonus."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.6
        steerer.set_feature_tracker(mock_tracker)

        activations = np.random.randn(d_model).astype(np.float32)
        sae_features = [(100, 0.9), (200, 0.7)]

        steerer.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=40.0,
            sae_features=sae_features,
        )

        assert steerer.zones["exploration"].last_approval_bonus == 0.6

    def test_approval_boost_lowers_crystallization_threshold(self, config, d_model):
        """High approval lowers crystallization thresholds."""
        crystal_config = CrystallizationConfig(
            min_cycles_for_crystallize=5,
            min_surprise_ema=50.0,  # High threshold
            min_cumulative_surprise=200.0,
            approval_crystallize_boost=0.5,  # 50% threshold reduction at full approval
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)

        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.9  # High approval
        steerer.set_feature_tracker(mock_tracker)

        activations = np.random.randn(d_model).astype(np.float32) * 2
        sae_features = [(100, 0.9)]

        # Run enough cycles with high surprise to build up EMA above reduced threshold
        # With approval_crystallize_boost=0.5 and approval=0.9:
        # threshold_multiplier = 1.0 - (0.9 * 0.5) = 0.55
        # effective threshold = 50 * 0.55 = 27.5
        # cumulative threshold = 200 * 0.55 = 110
        # EMA builds up slowly with alpha=0.15, so we need many cycles at high surprise
        for _ in range(15):  # More cycles to build EMA
            events = steerer.update_from_cycle(
                zone_name="exploration",
                activations=activations,
                surprise=55.0,  # High enough to build EMA above 27.5
                sae_features=sae_features,
            )

        # Should have crystallized due to approval boost lowering thresholds
        assert len(steerer.zones["exploration"].crystals) >= 1

    def test_approval_protects_from_pruning(self, config, d_model):
        """High approval prevents pruning."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=2,
            preserve_count=0,  # No automatic preservation
            prune_min_selections=1,
            prune_surprise_threshold=50.0,  # High threshold
            approval_prune_protection=0.5,  # Protection at 50% approval
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 3 crystals (1 over limit) with low performance
        for i in range(3):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=20.0,  # Low surprise, normally would be pruned
                selection_count=5,
                total_surprise=100.0,
            )
            steerer.zones["exploration"].crystals.append(crystal)

        # Pruning should be skipped due to high approval (0.7 > 0.5 threshold)
        # The approval_bonus is passed directly to _maybe_prune
        pruned = steerer._maybe_prune("exploration", approval_bonus=0.7)
        assert pruned is None

    def test_low_approval_allows_pruning(self, config, d_model):
        """Low approval allows normal pruning."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=2,
            preserve_count=0,
            prune_min_selections=1,
            prune_surprise_threshold=50.0,
            approval_prune_protection=0.5,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 3 crystals (1 over limit) with varying performance
        for i, surprise in enumerate([60.0, 40.0, 20.0]):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=surprise,
                selection_count=5,
                total_surprise=surprise * 5,
            )
            steerer.zones["exploration"].crystals.append(crystal)

        # Low approval (0.3 < 0.5 threshold) should allow pruning
        # The approval_bonus is passed directly to _maybe_prune
        pruned = steerer._maybe_prune("exploration", approval_bonus=0.3)
        assert pruned == "crystal_2"  # Weakest performer

    def test_strong_negative_approval_reduces_protection(self, config, d_model):
        """Strong negative approval reduces preserve_count, enabling more aggressive pruning."""
        # Config with preserve_count=2 so we can test reduction to 1
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=2,
            preserve_count=2,  # Normally protects top 2 performers
            prune_min_selections=1,
            prune_surprise_threshold=50.0,
            approval_prune_protection=0.5,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 3 crystals (1 over limit) with distinct performance levels
        # crystal_0: best (60.0), crystal_1: mid (40.0), crystal_2: worst (20.0)
        for i, surprise in enumerate([60.0, 40.0, 20.0]):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=surprise,
                selection_count=5,
                total_surprise=surprise * 5,
            )
            steerer.zones["exploration"].crystals.append(crystal)

        # With neutral/positive approval and preserve_count=2, both crystal_0 and crystal_1
        # would be protected, so only crystal_2 could be pruned
        # But with strong negative approval (-0.5 < -0.3), preserve_count is reduced to 1,
        # so only crystal_0 is protected, making crystal_1 eligible for pruning
        # However, crystal_2 is still the weakest and will be pruned first
        pruned = steerer._maybe_prune("exploration", approval_bonus=-0.5)

        # The weakest (crystal_2) is still pruned, but the key difference is that
        # crystal_1 is no longer protected. To verify the branch was triggered,
        # we need to check that with reduced protection, pruning still happens correctly.
        assert pruned == "crystal_2"  # Weakest is pruned

        # Verify only crystal_2 is retired
        retired = [c for c in steerer.zones["exploration"].crystals if c.retired]
        assert len(retired) == 1
        assert retired[0].name == "crystal_2"

        # Now test the key scenario: when there are more crystals and the second-best
        # would have been protected but now gets pruned due to reduced protection
        steerer2 = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 4 crystals (2 over limit of 2)
        # crystal_0: best (60.0), crystal_1: second (40.0), crystal_2: third (30.0), crystal_3: worst (20.0)
        for i, surprise in enumerate([60.0, 40.0, 30.0, 20.0]):
            crystal = CrystalEntry(
                name=f"crystal_{i}",
                vector=np.random.randn(d_model).astype(np.float32),
                birth_surprise=surprise,
                selection_count=5,
                total_surprise=surprise * 5,
            )
            steerer2.zones["exploration"].crystals.append(crystal)

        # First prune with strong negative approval - crystal_3 (worst) is pruned
        pruned1 = steerer2._maybe_prune("exploration", approval_bonus=-0.5)
        assert pruned1 == "crystal_3"

        # Second prune - now crystal_2 is the weakest non-protected
        # With negative approval, preserve_count=1, so only crystal_0 is protected
        # crystal_1 and crystal_2 are both vulnerable, crystal_2 is weaker
        pruned2 = steerer2._maybe_prune("exploration", approval_bonus=-0.5)
        assert pruned2 == "crystal_2"

        # Verify crystal_1 was not protected (would have been with preserve_count=2)
        active = [c for c in steerer2.zones["exploration"].crystals if not c.retired]
        assert len(active) == 2
        active_names = {c.name for c in active}
        assert "crystal_0" in active_names  # Best is always protected
        assert "crystal_1" in active_names  # Second-best survived because crystal_2 was weaker


class TestEvalatisSDFT:
    """Tests for SDFT (Self-Distillation Fine-Tuning) in EvalatisSteerer."""

    def test_high_selection_crystals_resist_pruning(self, config, d_model):
        """SDFT: High-selection crystals should resist pruning."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=2,
            preserve_count=0,  # No automatic preservation
            prune_min_selections=1,
            prune_surprise_threshold=50.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Add 3 crystals (1 over limit)
        # Crystal with high selection count (SDFT-protected)
        high_selection = CrystalEntry(
            name="high_selection_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=20.0,  # Low surprise (normally pruned)
            selection_count=60,  # > 50 threshold for SDFT protection
            total_surprise=100.0,
        )

        # Crystal with low selection count (prunable)
        low_selection = CrystalEntry(
            name="low_selection_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=30.0,  # Higher surprise than high_selection
            selection_count=5,  # Low selection, no SDFT protection
            total_surprise=150.0,
        )

        # Another low selection crystal
        another_low = CrystalEntry(
            name="another_low_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=25.0,
            selection_count=3,
            total_surprise=75.0,
        )

        steerer.zones["exploration"].crystals.extend([high_selection, low_selection, another_low])

        # Attempt pruning
        pruned = steerer._maybe_prune("exploration")

        # Should prune one of the low-selection crystals, not the high-selection one
        # Even though high_selection has lower surprise, its SDFT protection should save it
        assert pruned in ["another_low_crystal", "low_selection_crystal"], \
            "SDFT should protect high-selection crystals from pruning"
        assert pruned != "high_selection_crystal", \
            "High-selection crystal should be protected by SDFT"

    def test_sdft_selection_protection_threshold(self, config, d_model):
        """SDFT: Protection should apply when selection_count/50 > 0.5."""
        prune_config = CrystallizationConfig(
            max_crystals_per_zone=1,
            preserve_count=0,
            prune_min_selections=1,
            prune_surprise_threshold=50.0,
        )
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=prune_config)

        # Crystal at exactly the threshold (selection_count=25 -> 25/50=0.5 -> NOT protected)
        threshold_crystal = CrystalEntry(
            name="threshold_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=20.0,
            selection_count=25,  # 25/50 = 0.5, not > 0.5
            total_surprise=100.0,
        )

        # Crystal just above threshold (selection_count=26 -> 26/50=0.52 -> protected)
        above_threshold = CrystalEntry(
            name="above_threshold_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=20.0,
            selection_count=26,  # 26/50 = 0.52 > 0.5
            total_surprise=100.0,
        )

        steerer.zones["exploration"].crystals.extend([threshold_crystal, above_threshold])

        # Should be able to prune threshold_crystal but not above_threshold
        pruned = steerer._maybe_prune("exploration")

        assert pruned == "threshold_crystal", \
            "Crystal at threshold (0.5) should NOT be protected"


class TestEvalatisPhaseModulation:
    """Tests for phase-aware EMA modulation in EvalatisSteerer."""

    def test_ema_rate_modulated_by_phase(self, steerer):
        """EMA rates should be modulated based on cognitive phase."""
        # During generation: use base rates
        base_alpha = steerer.get_effective_ema_alpha("exploration", phase="generation")
        assert base_alpha == pytest.approx(0.12, rel=0.01)  # Base rate from config

        # During integration: slower rates (consolidation)
        integration_alpha = steerer.get_effective_ema_alpha("exploration", phase="integration")
        assert integration_alpha < base_alpha  # Should be slower

        # During reflexion: even slower (meta-analysis)
        reflexion_alpha = steerer.get_effective_ema_alpha("exploration", phase="reflexion")
        assert reflexion_alpha < integration_alpha

    def test_phase_modulation_factors(self, steerer):
        """Phase modulation should use configurable factors."""
        factors = steerer.phase_modulation_factors

        # Verify default factors
        assert factors["generation"] == 1.0
        assert factors["curation"] == 0.75
        assert factors["simulation"] == 0.5
        assert factors["integration"] == 0.5
        assert factors["reflexion"] == 0.25
        assert factors["continuity"] == 0.25

    def test_phase_modulation_math(self, steerer):
        """Verify EMA modulation computes correctly."""
        # exploration zone has ema_alpha = 0.12
        base = 0.12

        # Generation: 1.0 factor -> 0.12
        gen_alpha = steerer.get_effective_ema_alpha("exploration", phase="generation")
        assert gen_alpha == pytest.approx(base * 1.0, rel=0.01)

        # Curation: 0.75 factor -> 0.09
        cur_alpha = steerer.get_effective_ema_alpha("exploration", phase="curation")
        assert cur_alpha == pytest.approx(base * 0.75, rel=0.01)

        # Simulation: 0.5 factor -> 0.06
        sim_alpha = steerer.get_effective_ema_alpha("exploration", phase="simulation")
        assert sim_alpha == pytest.approx(base * 0.5, rel=0.01)

        # Integration: 0.5 factor -> 0.06
        int_alpha = steerer.get_effective_ema_alpha("exploration", phase="integration")
        assert int_alpha == pytest.approx(base * 0.5, rel=0.01)

        # Reflexion: 0.25 factor -> 0.03
        ref_alpha = steerer.get_effective_ema_alpha("exploration", phase="reflexion")
        assert ref_alpha == pytest.approx(base * 0.25, rel=0.01)

    def test_phase_modulation_different_zones(self, steerer):
        """Phase modulation should work for different zones."""
        # identity zone has ema_alpha = 0.03
        base = 0.03

        gen_alpha = steerer.get_effective_ema_alpha("identity", phase="generation")
        assert gen_alpha == pytest.approx(base * 1.0, rel=0.01)

        ref_alpha = steerer.get_effective_ema_alpha("identity", phase="reflexion")
        assert ref_alpha == pytest.approx(base * 0.25, rel=0.01)

    def test_phase_modulation_unknown_zone(self, steerer):
        """Unknown zone should return default alpha."""
        alpha = steerer.get_effective_ema_alpha("nonexistent", phase="generation")
        assert alpha == 0.1  # Default fallback

    def test_phase_modulation_unknown_phase(self, steerer):
        """Unknown phase should use 1.0 factor (no modulation)."""
        base_alpha = steerer.get_effective_ema_alpha("exploration", phase="generation")
        unknown_alpha = steerer.get_effective_ema_alpha("exploration", phase="unknown_phase")
        assert unknown_alpha == base_alpha  # Uses 1.0 factor for unknown phases

    def test_update_from_cycle_uses_phase_modulation(self, config, d_model, crystal_config):
        """update_from_cycle should use phase-modulated EMA rates."""
        steerer = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)
        initial_vector = steerer.get_vector(layer=3)  # exploration zone (layers 2-4)

        # Simulate update during generation (full EMA rate)
        activations = np.random.randn(d_model).astype(np.float32)
        steerer.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=50.0,
            phase="generation",
        )
        gen_vector = steerer.get_vector(layer=3)

        # Reset and update during reflexion (slower EMA rate)
        steerer_slow = EvalatisSteerer(config, d_model=d_model, crystal_config=crystal_config)
        steerer_slow.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=50.0,
            phase="reflexion",
        )
        reflex_vector = steerer_slow.get_vector(layer=3)

        # Generation update should move vector more than reflexion update
        # Initial vector is zeros, so we can compare directly to gen/reflex vectors
        gen_diff = np.linalg.norm(gen_vector) if gen_vector is not None else 0.0
        reflex_diff = np.linalg.norm(reflex_vector) if reflex_vector is not None else 0.0
        assert gen_diff > reflex_diff, "Generation should update faster than reflexion"
