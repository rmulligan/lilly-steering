"""Tests for WavePacket dataclass with 8D Plutchik emotions."""
import pytest
from core.affect.wave_packet import (
    WavePacket,
    DiffusionConfig,
    AFFECT_SPACE_DIMENSIONS,
    PLUTCHIK_PRIMARIES,
    INITIAL_WAVE_PACKET_SIGMA,
)


class TestDiffusionConfig:
    """Test diffusion rate configuration for 8D Plutchik emotions."""

    def test_default_rates(self):
        """Default rates match Plutchik emotion decay design."""
        config = DiffusionConfig()
        # Joy and trust linger (slow diffusion)
        assert config.joy == 0.008
        assert config.trust == 0.005
        # Fear and surprise fade quickly (fast diffusion)
        assert config.fear == 0.06
        assert config.surprise == 0.08
        # Sadness lingers
        assert config.sadness == 0.008
        # Disgust and anger are medium-fast
        assert config.disgust == 0.03
        assert config.anger == 0.05
        # Anticipation is slow (builds over time)
        assert config.anticipation == 0.01

    def test_to_vector(self):
        """Converts to 8D vector in correct Plutchik order."""
        config = DiffusionConfig()
        vec = config.to_vector()
        assert len(vec) == AFFECT_SPACE_DIMENSIONS
        assert vec[0] == 0.008  # joy
        assert vec[1] == 0.005  # trust
        assert vec[2] == 0.06   # fear
        assert vec[3] == 0.08   # surprise
        assert vec[4] == 0.008  # sadness
        assert vec[5] == 0.03   # disgust
        assert vec[6] == 0.05   # anger
        assert vec[7] == 0.01   # anticipation


class TestWavePacket:
    """Test WavePacket creation and properties with 8D Plutchik."""

    def test_creation_with_defaults(self):
        """Create packet with minimal args."""
        packet = WavePacket(
            position=[0.5, 0.6, 0.1, 0.2, 0.0, 0.0, 0.0, 0.7],
            initial_amplitude=0.8,
            deposit_cycle=100,
        )
        assert packet.amplitude == 0.8
        assert packet.deposit_cycle == 100
        assert len(packet.position) == AFFECT_SPACE_DIMENSIONS
        assert len(packet.sigma) == AFFECT_SPACE_DIMENSIONS
        assert packet.anchor_memories == set()
        assert packet.blended_memories == {}

    def test_phase_at_deposit(self):
        """Phase equals deposit cycle initially."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=0.5,
            deposit_cycle=42,
        )
        assert packet.current_phase(cycle=42) == 0.0

    def test_phase_advances_with_cycles(self):
        """Phase advances based on cycles elapsed."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=0.5,
            deposit_cycle=0,
        )
        # After 100 cycles, phase should advance
        phase = packet.current_phase(cycle=100)
        assert phase > 0

    def test_intensity_property(self):
        """Intensity computed from position deviation from Plutchik baselines."""
        # Neutral position (joy, trust, anticipation at 0.5; others at 0)
        neutral = WavePacket(
            position=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        # Extreme position
        extreme = WavePacket(
            position=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        assert extreme.intensity > neutral.intensity


class TestWavePacketMigration:
    """Test 6D to 8D migration."""

    def test_6d_to_8d_migration(self):
        """Packet with 6D position is automatically migrated to 8D."""
        packet = WavePacket(
            position=[0.6, 0.7, 0.5, 0.4, 0.3, 0.8],  # 6D legacy
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        # Should now be 8D
        assert len(packet.position) == AFFECT_SPACE_DIMENSIONS
        assert len(packet.sigma) == AFFECT_SPACE_DIMENSIONS
        # Mapping: valence (idx 1) -> joy (idx 0)
        assert packet.position[0] == 0.7  # joy = valence
        # Mapping: satisfaction (idx 3) -> trust (idx 1)
        assert packet.position[1] == 0.4  # trust = satisfaction
        # Mapping: frustration (idx 4) -> anger (idx 6)
        assert packet.position[6] == 0.3  # anger = frustration

    def test_6d_sigma_migration(self):
        """Sigma with 6D is automatically migrated to 8D."""
        packet = WavePacket(
            position=[0.5] * 6,  # 6D legacy
            initial_amplitude=1.0,
            deposit_cycle=0,
            sigma=[0.3] * 6,  # 6D sigma
        )
        assert len(packet.sigma) == AFFECT_SPACE_DIMENSIONS
        # New dimensions get default sigma
        assert packet.sigma[6] == INITIAL_WAVE_PACKET_SIGMA
        assert packet.sigma[7] == INITIAL_WAVE_PACKET_SIGMA


class TestWavePacketDynamics:
    """Test diffusion and decay behavior."""

    def test_evolve_increases_sigma(self):
        """Evolving packet increases sigma (spread)."""
        config = DiffusionConfig()
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        initial_sigma = packet.sigma.copy()
        packet.evolve(config)

        # All sigmas should increase
        for i in range(AFFECT_SPACE_DIMENSIONS):
            assert packet.sigma[i] > initial_sigma[i]

    def test_evolve_anisotropic(self):
        """Surprise diffuses faster than trust."""
        config = DiffusionConfig()
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        # Evolve 10 cycles
        for _ in range(10):
            packet.evolve(config)

        # Surprise (idx 3) should have spread more than trust (idx 1)
        assert packet.sigma[3] > packet.sigma[1]

    def test_evolve_decays_amplitude(self):
        """Evolving packet decreases amplitude."""
        config = DiffusionConfig()
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        initial_amp = packet.amplitude
        packet.evolve(config)

        assert packet.amplitude < initial_amp

    def test_intense_packet_decays_slower(self):
        """Higher intensity = slower decay (longer persistence)."""
        config = DiffusionConfig()

        # Mild emotion (near neutral)
        mild = WavePacket(
            position=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        # Intense emotion (high values across board)
        intense = WavePacket(
            position=[0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.8, 0.9],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        # Evolve both 50 cycles
        for _ in range(50):
            mild.evolve(config)
            intense.evolve(config)

        # Intense should have higher remaining amplitude
        assert intense.amplitude > mild.amplitude

    def test_is_alive_threshold(self):
        """Packet is_alive returns False when amplitude below threshold."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=0.02,
            deposit_cycle=0,
        )
        assert packet.is_alive(threshold=0.01)

        packet.amplitude = 0.005
        assert not packet.is_alive(threshold=0.01)


class TestWavePacketSDFT:
    """Tests for SDFT (Self-Distillation Fine-Tuning) in WavePacket."""

    def test_importance_property_no_anchors(self):
        """Importance should be 0 with no anchor memories."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        assert packet.importance == 0.0

    def test_importance_property_full_anchors(self):
        """Importance should be 1.0 at IMPORTANCE_MATURITY anchors."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
            anchor_memories={f"mem_{i}" for i in range(5)},  # IMPORTANCE_MATURITY = 5
        )
        assert packet.importance == 1.0

    def test_importance_property_partial(self):
        """Importance should be proportional to anchor count."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
            anchor_memories={"mem_1", "mem_2"},  # 2 out of 5
        )
        assert packet.importance == 2 / 5

    def test_importance_property_capped(self):
        """Importance should cap at 1.0 even with more anchors."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
            anchor_memories={f"mem_{i}" for i in range(10)},  # 10 > IMPORTANCE_MATURITY
        )
        assert packet.importance == 1.0

    def test_sdft_important_packets_decay_slower(self):
        """SDFT: Packets with more anchor memories should decay slower."""
        config = DiffusionConfig()

        # Packet with no anchors (unimportant)
        unimportant = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        # Packet with many anchors (important)
        important = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
            anchor_memories={f"mem_{i}" for i in range(5)},
        )

        # Evolve both 50 cycles
        for _ in range(50):
            unimportant.evolve(config)
            important.evolve(config)

        # Important packet should have higher remaining amplitude
        assert important.amplitude > unimportant.amplitude, \
            "Important packets should decay slower due to SDFT"


class TestWavePacketSpatialContribution:
    """Test Gaussian spatial falloff calculation."""

    def test_contribution_at_center(self):
        """Max contribution at packet center."""
        packet = WavePacket(
            position=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        # Sample at exact position
        contrib = packet.spatial_contribution([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        assert contrib == pytest.approx(1.0, rel=0.01)

    def test_contribution_decreases_with_distance(self):
        """Contribution falls off with distance."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        center = packet.spatial_contribution([0.5] * AFFECT_SPACE_DIMENSIONS)
        near = packet.spatial_contribution([0.6] * AFFECT_SPACE_DIMENSIONS)
        far = packet.spatial_contribution([0.9] * AFFECT_SPACE_DIMENSIONS)

        assert center > near > far

    def test_contribution_anisotropic(self):
        """Distance in fast-diffusing dimensions matters less over time."""
        config = DiffusionConfig()
        # Use neutral Plutchik position
        packet = WavePacket(
            position=[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            initial_amplitude=1.0,
            deposit_cycle=0,
        )

        # Evolve to spread sigma
        for _ in range(20):
            packet.evolve(config)

        # Same absolute deviation (0.2) in different dimensions
        # Deviation only in surprise (fast diffusion, idx 3)
        surprise_offset = [0.5, 0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 0.5]
        # Deviation only in trust (slow diffusion, idx 1)
        trust_offset = [0.5, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        surprise_contrib = packet.spatial_contribution(surprise_offset)
        trust_contrib = packet.spatial_contribution(trust_offset)

        # Surprise deviation should matter less (larger sigma)
        assert surprise_contrib > trust_contrib

    def test_contribution_scales_with_amplitude(self):
        """Contribution scales with current amplitude."""
        strong = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        weak = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=0.5,
            deposit_cycle=0,
        )

        sample = [0.6] * AFFECT_SPACE_DIMENSIONS
        assert strong.spatial_contribution(sample) > weak.spatial_contribution(sample)

    def test_contribution_handles_6d_sample(self):
        """Contribution handles 6D sample position gracefully."""
        packet = WavePacket(
            position=[0.5] * AFFECT_SPACE_DIMENSIONS,
            initial_amplitude=1.0,
            deposit_cycle=0,
        )
        # 6D sample should be padded automatically
        contrib = packet.spatial_contribution([0.5] * 6)
        assert contrib > 0
