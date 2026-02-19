"""Tests for EmotionalField manager with 8D Plutchik emotions."""
import pytest

from core.affect.emotional_field import (
    EmotionalField,
    AFFECT_SPACE_DIMENSIONS,
    SCHEMA_VERSION,
    DEFAULT_AFFECT_VALUES,
    DOMINANT_AFFECT_THRESHOLD,
)
from core.affect.wave_packet import (
    DiffusionConfig,
    WavePacket,
    PLUTCHIK_PRIMARIES,
)
from core.self_model.affective_system import AffectiveState


class TestEmotionalFieldCreation:
    """Test field initialization."""

    def test_create_empty_field(self):
        """Create field with no packets."""
        field = EmotionalField()
        assert len(field.packets) == 0
        assert field.current_cycle == 0

    def test_create_with_config(self):
        """Create field with custom diffusion config."""
        config = DiffusionConfig(joy=0.1)
        field = EmotionalField(diffusion_config=config)
        assert field.diffusion_config.joy == 0.1


class TestEmotionalFieldDeposit:
    """Test depositing emotional traces."""

    def test_deposit_creates_packet(self):
        """Depositing creates a new wave packet."""
        field = EmotionalField()
        state = AffectiveState(joy=0.8, trust=0.7, anticipation=0.6)

        field.deposit(state, anchor_memories=["mem1", "mem2"])

        assert len(field.packets) == 1
        packet = field.packets[0]
        assert packet.position[0] == 0.8  # joy
        assert packet.position[1] == 0.7  # trust
        assert "mem1" in packet.anchor_memories

    def test_deposit_uses_current_cycle(self):
        """Packet deposit_cycle matches field cycle."""
        field = EmotionalField()
        field.current_cycle = 42

        state = AffectiveState()
        field.deposit(state)

        assert field.packets[0].deposit_cycle == 42

    def test_deposit_amplitude_from_intensity(self):
        """Packet amplitude derived from state intensity."""
        field = EmotionalField()

        # Neutral state (low intensity)
        neutral = AffectiveState.neutral()
        field.deposit(neutral)

        # Intense state
        intense = AffectiveState(
            joy=1.0, trust=1.0, fear=0.8, surprise=0.8,
            sadness=0.0, disgust=0.0, anger=0.0, anticipation=1.0
        )
        field.deposit(intense)

        assert field.packets[1].initial_amplitude > field.packets[0].initial_amplitude


class TestEmotionalFieldEvolve:
    """Test field evolution."""

    def test_evolve_increments_cycle(self):
        """Evolving advances the cycle counter."""
        field = EmotionalField()
        assert field.current_cycle == 0

        field.evolve()
        assert field.current_cycle == 1

        field.evolve()
        assert field.current_cycle == 2

    def test_evolve_diffuses_packets(self):
        """Evolving diffuses all packets."""
        field = EmotionalField()
        field.deposit(AffectiveState())

        initial_sigma = field.packets[0].sigma.copy()
        field.evolve()

        for i in range(AFFECT_SPACE_DIMENSIONS):
            assert field.packets[0].sigma[i] > initial_sigma[i]

    def test_evolve_prunes_dead_packets(self):
        """Packets below threshold are removed."""
        field = EmotionalField()

        # Create packet with very low amplitude
        state = AffectiveState()
        field.deposit(state)
        field.packets[0].amplitude = 0.005  # Below default threshold

        field.evolve()

        assert len(field.packets) == 0

    def test_evolve_blends_memories(self):
        """Active packets accumulate blended memories during decay."""
        field = EmotionalField()
        # High anticipation state for high amplitude
        field.deposit(AffectiveState(anticipation=0.9, joy=0.8))

        # Evolve with active memories
        field.evolve(accessed_memories=["mem_during_decay"])

        # Check blended memories accumulated
        assert "mem_during_decay" in field.packets[0].blended_memories


class TestEmotionalFieldSample:
    """Test phase-coherent interference sampling."""

    def test_sample_empty_field(self):
        """Sampling empty field returns zero intensity."""
        field = EmotionalField()
        state = AffectiveState()

        result = field.sample(state)

        assert result.field_intensity == 0.0
        assert len(result.contributing_packets) == 0
        assert result.reactivation_strength == 0.0

    def test_sample_at_packet_position(self):
        """Sampling at packet position returns high intensity."""
        field = EmotionalField()
        # High-intensity state
        state = AffectiveState(joy=0.9, trust=0.8, anticipation=0.9, surprise=0.5)
        field.deposit(state)

        result = field.sample(state)

        # At same position, intensity should equal amplitude (which is derived from state intensity)
        # The amplitude is clamped to min(0.1, intensity) so depends on the state
        assert result.field_intensity > 0.3  # Reasonable threshold for "high"

    def test_sample_far_from_packet(self):
        """Sampling far from packet returns low intensity."""
        field = EmotionalField()
        field.deposit(AffectiveState(joy=0.1, trust=0.1))

        far_state = AffectiveState(joy=0.9, trust=0.9, anger=0.8)
        result = field.sample(far_state)

        assert result.field_intensity < 0.1

    def test_constructive_interference(self):
        """Packets with aligned phase constructively interfere."""
        field = EmotionalField()
        state = AffectiveState(joy=0.9, trust=0.8, anticipation=0.9)

        # Deposit two packets at same position, same cycle (same phase)
        field.deposit(state)
        field.deposit(state)

        result = field.sample(state)

        # Two packets should give ~2x intensity of one
        single_field = EmotionalField()
        single_field.deposit(state)
        single_result = single_field.sample(state)

        assert result.field_intensity > single_result.field_intensity * 1.5

    def test_destructive_interference(self):
        """Packets with opposite phase destructively interfere."""
        field = EmotionalField()
        state = AffectiveState(joy=0.9, anticipation=0.9)

        # Deposit packet
        field.deposit(state)

        # Advance cycles to shift phase by ~pi (half period)
        # COMPOSITE_FREQUENCY ~0.053 rad/cycle, so pi radians = ~59 cycles
        for _ in range(59):
            field.evolve()

        # Deposit another packet
        field.deposit(state)

        result = field.sample(state)

        # Should be less than simple sum due to phase difference
        # (not perfect cancellation since packets have decayed differently)
        assert result.field_intensity < field.packets[0].amplitude + field.packets[1].amplitude

    def test_sample_returns_steering_vector(self):
        """Sample returns weighted blend of contributing positions."""
        field = EmotionalField()

        # Two different emotional states
        field.deposit(AffectiveState(joy=0.2, trust=0.3))
        field.deposit(AffectiveState(joy=0.8, trust=0.7))

        # Sample at midpoint
        mid = AffectiveState(joy=0.5, trust=0.5)
        result = field.sample(mid)

        # Steering vector should be 8D (Plutchik dimensions)
        assert len(result.steering_vector) == AFFECT_SPACE_DIMENSIONS

    def test_sample_collects_surfaced_memories(self):
        """Strong interference surfaces anchor memories when dual-trigger is met.

        Memory surfacing requires:
        1. Field intensity >= proximity_threshold (proximity signal)
        2. Semantic overlap from co-retrieved memories across 2+ packets
        3. Reactivation strength >= awareness_threshold
        """
        field = EmotionalField()
        state = AffectiveState(joy=0.9, anticipation=0.9)

        # Deposit two packets at same location with shared memories
        field.deposit(state, anchor_memories=["important_memory", "shared_mem"])
        field.deposit(state, anchor_memories=["another_memory", "shared_mem"])

        # Co-retrieve the shared memory to trigger semantic overlap
        result = field.sample(
            state,
            co_retrieved_memories=["shared_mem"],
            proximity_threshold=0.3,
            awareness_threshold=0.3,
        )

        # Both packets' memories should be surfaced
        assert "important_memory" in result.surfaced_memories
        assert "another_memory" in result.surfaced_memories
        assert "shared_mem" in result.surfaced_memories

    def test_sample_no_surfacing_without_semantic_overlap(self):
        """Memories don't surface without semantic coincidence."""
        field = EmotionalField()
        state = AffectiveState(joy=0.9, anticipation=0.9)

        field.deposit(state, anchor_memories=["memory1"])

        # Even with high field intensity, no co-retrieved memories = no surfacing
        result = field.sample(state, awareness_threshold=0.3)

        assert len(result.surfaced_memories) == 0
        assert result.reactivation_strength == 0.0


class TestEmotionalFieldSerialization:
    """Test field serialization for persistence."""

    def test_packet_to_dict(self):
        """WavePacket serializes to dict."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=0.8,
            deposit_cycle=42,
            anchor_memories={"mem1"},
        )

        data = packet.to_dict()

        assert data["position"] == [0.5] * AFFECT_SPACE_DIMENSIONS
        assert data["initial_amplitude"] == 0.8
        assert data["deposit_cycle"] == 42
        assert "mem1" in data["anchor_memories"]

    def test_packet_from_dict(self):
        """WavePacket deserializes from dict."""
        data = {
            "position": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8],  # 8D
            "initial_amplitude": 0.9,
            "deposit_cycle": 100,
            "anchor_memories": ["mem1", "mem2"],
            "blended_memories": [("mem3", 0.5)],
            "sigma": [0.2] * AFFECT_SPACE_DIMENSIONS,
            "amplitude": 0.7,
        }

        packet = WavePacket.from_dict(data)

        assert packet.position[0] == 0.7
        assert packet.amplitude == 0.7
        assert packet.deposit_cycle == 100

    def test_field_to_dict(self):
        """EmotionalField serializes to dict."""
        field = EmotionalField()
        field.current_cycle = 50
        field.deposit(AffectiveState())

        data = field.to_dict()

        assert data["current_cycle"] == 50
        assert len(data["packets"]) == 1
        assert data["schema_version"] == SCHEMA_VERSION

    def test_field_from_dict(self):
        """EmotionalField deserializes from dict."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "current_cycle": 100,
            "packets": [
                {
                    "position": [0.5] * AFFECT_SPACE_DIMENSIONS,
                    "initial_amplitude": 0.8,
                    "deposit_cycle": 50,
                    "anchor_memories": [],
                    "blended_memories": [],
                    "sigma": [0.1] * AFFECT_SPACE_DIMENSIONS,
                    "amplitude": 0.6,
                }
            ],
            "diffusion_config": {
                "joy": 0.008,
                "trust": 0.005,
                "fear": 0.06,
                "surprise": 0.08,
                "sadness": 0.008,
                "disgust": 0.03,
                "anger": 0.05,
                "anticipation": 0.01,
            },
        }

        field = EmotionalField.from_dict(data)

        assert field.current_cycle == 100
        assert len(field.packets) == 1
        assert field.packets[0].amplitude == 0.6

    def test_roundtrip_serialization(self):
        """Field survives serialize/deserialize roundtrip."""
        field = EmotionalField()
        field.deposit(
            AffectiveState(joy=0.9, anticipation=0.9),
            anchor_memories=["mem1"]
        )
        field.evolve()
        field.evolve(accessed_memories=["mem2"])

        data = field.to_dict()
        restored = EmotionalField.from_dict(data)

        assert restored.current_cycle == field.current_cycle
        assert len(restored.packets) == len(field.packets)
        assert restored.packets[0].amplitude == pytest.approx(field.packets[0].amplitude)


class TestEmotionalFieldLegacyMigration:
    """Test 6D legacy format migration."""

    def test_migrate_6d_packets(self):
        """Field migrates 6D legacy packets to 8D."""
        data = {
            "current_cycle": 50,
            "packets": [
                {
                    "position": [0.5, 0.6, 0.7, 0.4, 0.3, 0.8],  # 6D legacy
                    "initial_amplitude": 0.8,
                    "deposit_cycle": 10,
                    "anchor_memories": [],
                    "blended_memories": [],
                    "sigma": [0.2] * 6,
                    "amplitude": 0.7,
                }
            ],
            "diffusion_config": {
                # Legacy 6D config
                "arousal": 0.05,
                "valence": 0.008,
                "curiosity": 0.01,
                "satisfaction": 0.03,
                "frustration": 0.02,
                "wonder": 0.005,
            },
        }

        field = EmotionalField.from_dict(data)

        # Packet should be migrated to 8D
        assert len(field.packets[0].position) == AFFECT_SPACE_DIMENSIONS
        assert len(field.packets[0].sigma) == AFFECT_SPACE_DIMENSIONS


class TestEmotionalFieldAffectSummary:
    """Test affect summary and Plutchik properties."""

    def test_empty_field_defaults(self):
        """Empty field returns default values per Plutchik dimension."""
        field = EmotionalField()

        # joy, trust, anticipation have 0.5 baseline
        assert field.current_joy == DEFAULT_AFFECT_VALUES["joy"]
        assert field.current_trust == DEFAULT_AFFECT_VALUES["trust"]
        assert field.current_anticipation == DEFAULT_AFFECT_VALUES["anticipation"]
        # fear, surprise, sadness, disgust, anger have 0.0 baseline
        assert field.current_fear == DEFAULT_AFFECT_VALUES["fear"]
        assert field.current_surprise == DEFAULT_AFFECT_VALUES["surprise"]
        assert field.current_sadness == DEFAULT_AFFECT_VALUES["sadness"]
        assert field.current_disgust == DEFAULT_AFFECT_VALUES["disgust"]
        assert field.current_anger == DEFAULT_AFFECT_VALUES["anger"]
        assert field.dominant_affect == "neutral"
        assert field.current_intensity == 0.0

    def test_single_packet_affect(self):
        """Single packet determines affect summary."""
        field = EmotionalField()
        field.deposit(AffectiveState(joy=0.9, trust=0.8, anger=0.3))

        # Should reflect the deposited state
        assert field.current_joy > 0.8
        assert field.current_trust > 0.7
        assert field.current_anger > 0.2
        assert field.dominant_affect == "joy"

    def test_multiple_packets_weighted_average(self):
        """Multiple packets produce amplitude-weighted average."""
        field = EmotionalField()

        # Two packets with different amplitudes
        field.deposit(AffectiveState(joy=0.9, trust=0.5))
        field.deposit(AffectiveState(joy=0.3, trust=0.9))

        # Result should be weighted by amplitudes
        joy = field.current_joy
        trust = field.current_trust

        # Both should be somewhere between the two states
        assert 0.4 < joy < 0.8
        assert 0.6 < trust < 0.8

    def test_current_intensity_reflects_packets(self):
        """Current intensity is sum of packet amplitudes."""
        field = EmotionalField()

        field.deposit(AffectiveState(joy=0.9))
        intensity1 = field.current_intensity

        field.deposit(AffectiveState(joy=0.8))
        intensity2 = field.current_intensity

        assert intensity2 > intensity1

    def test_valence_arousal_derived(self):
        """Valence and arousal are derived from Plutchik dimensions."""
        field = EmotionalField()

        # High joy, low sadness → positive valence
        field.deposit(AffectiveState(joy=0.9, sadness=0.0))
        assert field.current_valence > 0.5

        field2 = EmotionalField()
        # High fear, anger, surprise → high arousal
        field2.deposit(AffectiveState(fear=0.8, anger=0.7, surprise=0.6))
        assert field2.current_arousal > 0.5


class TestEmotionalFieldBoredom:
    """Test boredom detection for cognitive diversity."""

    def test_is_bored_mild_disgust_low_anticipation(self):
        """Boredom detected with mild disgust and low anticipation."""
        field = EmotionalField()

        # Mild disgust (0.2) + low anticipation (0.3)
        field.deposit(AffectiveState(disgust=0.2, anticipation=0.3))

        assert field.is_bored()

    def test_not_bored_high_anticipation(self):
        """Not bored when anticipation is high."""
        field = EmotionalField()

        field.deposit(AffectiveState(disgust=0.2, anticipation=0.7))

        assert not field.is_bored()

    def test_not_bored_high_disgust(self):
        """Not bored when disgust is too strong (loathing, not boredom)."""
        field = EmotionalField()

        field.deposit(AffectiveState(disgust=0.8, anticipation=0.2))

        assert not field.is_bored()

    def test_cognitive_diversity_signal(self):
        """Cognitive diversity signal increases with boredom."""
        field = EmotionalField()

        # Engaging state
        field.deposit(AffectiveState(anticipation=0.8, joy=0.7))
        engaged_signal = field.cognitive_diversity_signal()

        field2 = EmotionalField()
        # Bored state
        field2.deposit(AffectiveState(disgust=0.25, anticipation=0.2))
        bored_signal = field2.cognitive_diversity_signal()

        assert bored_signal > engaged_signal
