"""Emotional field manager for wave packet dynamics.

Uses 8D Plutchik affect-space: joy, trust, fear, surprise, sadness, disgust, anger, anticipation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

from core.affect.wave_packet import (
    WavePacket,
    DiffusionConfig,
    AFFECT_SPACE_DIMENSIONS,
    PLUTCHIK_PRIMARIES,
)

if TYPE_CHECKING:
    from core.self_model.affective_system import AffectiveState


# Schema version for migration support
SCHEMA_VERSION = "8d-v1"

# Threshold for blending memories during decay
BLEND_AMPLITUDE_THRESHOLD = 0.1

# Default pruning threshold
DEFAULT_PRUNE_THRESHOLD = 0.01

# Default neutral affect values per dimension
# joy, trust, anticipation have 0.5 baseline; others have 0.0
DEFAULT_AFFECT_VALUES = {
    "joy": 0.5,
    "trust": 0.5,
    "fear": 0.0,
    "surprise": 0.0,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.5,
}

# Threshold for dominant affect detection (below this returns "neutral")
DOMINANT_AFFECT_THRESHOLD = 0.1

# Backward-compatibility factors for deriving legacy valence/arousal from Plutchik dimensions
# Valence derivation: joy contributes positive, sadness contributes negative
VALENCE_JOY_FACTOR = 0.5
VALENCE_SADNESS_FACTOR = 0.5

# Arousal derivation: fear, surprise, anger are activating; trust is calming
AROUSAL_ACTIVATING_FACTOR = 0.4
AROUSAL_CALMING_FACTOR = 0.2


@dataclass
class InterferenceResult:
    """Result of sampling the emotional field.

    Attributes:
        field_intensity: Sum of contributions at sample point
        contributing_packets: Packets with positive contribution
        steering_vector: 8D affect vector for steering (weighted blend)
        surfaced_memories: Memories to bring to explicit awareness
        reactivation_strength: Combined signal strength for awareness threshold
    """
    field_intensity: float
    contributing_packets: List[WavePacket]
    steering_vector: List[float]
    surfaced_memories: List[str]
    reactivation_strength: float


class EmotionalField:
    """Manages wave packets in 8D Plutchik affect-space.

    The field maintains a collection of emotional traces that diffuse,
    decay, and interfere over cognitive cycles.

    Dimensions:
        0: joy (serenity → joy → ecstasy)
        1: trust (acceptance → trust → admiration)
        2: fear (apprehension → fear → terror)
        3: surprise (distraction → surprise → amazement)
        4: sadness (pensiveness → sadness → grief)
        5: disgust (boredom → disgust → loathing)
        6: anger (annoyance → anger → rage)
        7: anticipation (interest → anticipation → vigilance)
    """

    def __init__(
        self,
        diffusion_config: Optional[DiffusionConfig] = None,
        prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    ):
        """Initialize empty emotional field.

        Args:
            diffusion_config: Anisotropic diffusion rates per emotion
            prune_threshold: Amplitude below which packets are removed
        """
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.prune_threshold = prune_threshold
        self.packets: List[WavePacket] = []
        self.current_cycle: int = 0
        # Cache for _compute_affect_summary() to avoid recomputation per-property
        self._summary_cache: Optional[dict] = None
        self._summary_cache_cycle: int = -1

    def deposit(
        self,
        affect_state: "AffectiveState",
        anchor_memories: Optional[List[str]] = None,
    ) -> WavePacket:
        """Deposit a new emotional trace into the field.

        Args:
            affect_state: Current affective state to encode
            anchor_memories: Memory IDs explicitly linked to this emotion

        Returns:
            The created WavePacket
        """
        # Convert AffectiveState to 8D position vector
        position = affect_state.to_vector()

        # Amplitude from intensity (minimum 0.1 to ensure some persistence)
        amplitude = max(0.1, affect_state.intensity())

        packet = WavePacket(
            position=position,
            initial_amplitude=amplitude,
            deposit_cycle=self.current_cycle,
            anchor_memories=set(anchor_memories) if anchor_memories else set(),
        )

        self.packets.append(packet)
        # Invalidate affect summary cache since packets changed
        self._summary_cache = None
        return packet

    def evolve(self, accessed_memories: Optional[List[str]] = None) -> None:
        """Advance one cognitive cycle.

        Diffuses all packets, decays amplitudes, prunes dead packets,
        and optionally blends accessed memories into active packets.

        Args:
            accessed_memories: Memory IDs accessed this cycle (for blending)
        """
        self.current_cycle += 1
        # Invalidate affect summary cache since packets will change
        self._summary_cache = None

        surviving = []
        for packet in self.packets:
            # Apply diffusion and decay
            packet.evolve(self.diffusion_config)

            # Blend memories if packet is active
            if accessed_memories and packet.amplitude >= BLEND_AMPLITUDE_THRESHOLD:
                for mem_id in accessed_memories:
                    # Strength proportional to current amplitude
                    packet.add_blended_memory(mem_id, packet.amplitude)

            # Keep if alive
            if packet.is_alive(self.prune_threshold):
                surviving.append(packet)

        self.packets = surviving

    def packet_count(self) -> int:
        """Return number of active packets."""
        return len(self.packets)

    def sample(
        self,
        affect_state: "AffectiveState",
        co_retrieved_memories: Optional[List[str]] = None,
        proximity_threshold: float = 0.3,
        awareness_threshold: float = 0.6,
    ) -> InterferenceResult:
        """Sample the field at current affect position with phase-coherent interference.

        Args:
            affect_state: Current affective state (sample position)
            co_retrieved_memories: Memory IDs being retrieved this cycle
            proximity_threshold: Minimum field_intensity for reactivation
            awareness_threshold: Minimum reactivation_strength for surfacing

        Returns:
            InterferenceResult with intensity, steering, and memories
        """
        if not self.packets:
            return InterferenceResult(
                field_intensity=0.0,
                contributing_packets=[],
                steering_vector=[0.0] * AFFECT_SPACE_DIMENSIONS,
                surfaced_memories=[],
                reactivation_strength=0.0,
            )

        sample_position = affect_state.to_vector()

        # Calculate phase-coherent contributions
        contributions: List[Tuple[WavePacket, float, float]] = []
        for packet in self.packets:
            # Spatial contribution (Gaussian falloff)
            spatial = packet.spatial_contribution(sample_position)

            # Phase factor (coherent interference)
            current_phase = packet.current_phase(self.current_cycle)
            # Reference phase is 0 for current cycle deposits
            phase_factor = math.cos(current_phase)

            # Combined contribution (can be negative for destructive interference)
            contribution = spatial * phase_factor
            contributions.append((packet, contribution, spatial))

        # Sum contributions for field intensity
        field_intensity = sum(c[1] for c in contributions)
        field_intensity = max(0.0, field_intensity)  # Clamp to non-negative

        # Collect positively contributing packets
        contributing = [p for p, c, _ in contributions if c > 0]

        # Compute 8D steering vector (weighted blend of positions)
        steering_vector = self._compute_steering_vector(contributions)

        # Check semantic coincidence for reactivation
        semantic_overlap = self._compute_semantic_overlap(
            contributing, co_retrieved_memories or []
        )

        # Reactivation requires both signals
        reactivation_strength = 0.0
        if field_intensity >= proximity_threshold and semantic_overlap > 0:
            reactivation_strength = field_intensity * semantic_overlap

        # Surface memories if above awareness threshold
        surfaced_memories: List[str] = []
        if reactivation_strength >= awareness_threshold:
            surfaced_memories = self._collect_surfaced_memories(contributing)

        return InterferenceResult(
            field_intensity=field_intensity,
            contributing_packets=contributing,
            steering_vector=steering_vector,
            surfaced_memories=surfaced_memories,
            reactivation_strength=reactivation_strength,
        )

    def _compute_steering_vector(
        self, contributions: List[Tuple[WavePacket, float, float]]
    ) -> List[float]:
        """Compute weighted blend of contributing packet positions.

        Returns 8D Plutchik steering vector.

        Args:
            contributions: List of (packet, contribution, spatial_only)
        """
        # Use absolute spatial contribution for weighting (ignore phase sign)
        total_weight = sum(abs(spatial) for _, _, spatial in contributions)

        if total_weight == 0:
            return [0.0] * AFFECT_SPACE_DIMENSIONS

        steering = [0.0] * AFFECT_SPACE_DIMENSIONS
        for packet, _, spatial in contributions:
            weight = abs(spatial) / total_weight
            for i in range(min(len(packet.position), AFFECT_SPACE_DIMENSIONS)):
                steering[i] += packet.position[i] * weight

        return steering

    def _compute_semantic_overlap(
        self,
        contributing_packets: List[WavePacket],
        co_retrieved: List[str],
    ) -> float:
        """Check if memories from different packets are co-retrieved.

        Returns overlap score (0 if < 2 packets have co-retrieved memories).
        """
        if len(contributing_packets) < 2 or not co_retrieved:
            return 0.0

        # Convert to set once for O(1) membership checks
        co_retrieved_set = set(co_retrieved)

        packets_with_overlap = 0
        for packet in contributing_packets:
            all_memories = packet.anchor_memories.copy()
            all_memories.update(packet.blended_memories.keys())

            if not co_retrieved_set.isdisjoint(all_memories):
                packets_with_overlap += 1

        # Need at least 2 packets with overlapping memories
        if packets_with_overlap >= 2:
            return packets_with_overlap / len(contributing_packets)

        return 0.0

    def _collect_surfaced_memories(
        self, contributing_packets: List[WavePacket]
    ) -> List[str]:
        """Collect unique memories from contributing packets."""
        memories: set[str] = set()
        for packet in contributing_packets:
            memories.update(packet.anchor_memories)
            memories.update(packet.blended_memories.keys())
        return list(memories)

    def to_dict(self) -> dict:
        """Serialize field state for persistence."""
        return {
            "schema_version": SCHEMA_VERSION,
            "current_cycle": self.current_cycle,
            "packets": [p.to_dict() for p in self.packets],
            "diffusion_config": self.diffusion_config.to_dict(),
            "prune_threshold": self.prune_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalField":
        """Deserialize field from dictionary.

        Handles migration from 6D legacy format to 8D Plutchik.
        """
        schema_version = data.get("schema_version", "6d-legacy")

        config = DiffusionConfig.from_dict(data.get("diffusion_config", {}))
        field = cls(
            diffusion_config=config,
            prune_threshold=data.get("prune_threshold", DEFAULT_PRUNE_THRESHOLD),
        )
        field.current_cycle = data.get("current_cycle", 0)

        # Packets automatically migrate in WavePacket.from_dict and __post_init__
        field.packets = [WavePacket.from_dict(p) for p in data.get("packets", [])]

        return field

    def _compute_affect_summary(self) -> dict:
        """Compute weighted averages for all 8 Plutchik dimensions.

        Results are cached per-cycle to avoid recomputation when multiple
        properties (current_joy, current_trust, etc.) are accessed.

        Returns:
            Dictionary with:
                - Each Plutchik dimension weighted average (0-1)
                - total_amplitude: Sum of all packet amplitudes
                - dominant: Name of highest weighted dimension
        """
        # Return cached result if valid for current cycle
        if self._summary_cache is not None and self._summary_cache_cycle == self.current_cycle:
            return self._summary_cache

        if not self.packets:
            result = {name: DEFAULT_AFFECT_VALUES.get(name, 0.0) for name in PLUTCHIK_PRIMARIES}
            result["total_amplitude"] = 0.0
            result["dominant"] = "neutral"
            self._summary_cache = result
            self._summary_cache_cycle = self.current_cycle
            return result

        total_amplitude = 0.0
        weighted_sums = [0.0] * AFFECT_SPACE_DIMENSIONS

        for p in self.packets:
            total_amplitude += p.amplitude
            for i in range(min(len(p.position), AFFECT_SPACE_DIMENSIONS)):
                weighted_sums[i] += p.position[i] * p.amplitude

        if total_amplitude == 0:
            result = {name: DEFAULT_AFFECT_VALUES.get(name, 0.0) for name in PLUTCHIK_PRIMARIES}
            result["total_amplitude"] = 0.0
            result["dominant"] = "neutral"
            self._summary_cache = result
            self._summary_cache_cycle = self.current_cycle
            return result

        weighted_avgs = [ws / total_amplitude for ws in weighted_sums]

        # Build result dictionary
        result = {}
        for i, name in enumerate(PLUTCHIK_PRIMARIES):
            result[name] = weighted_avgs[i] if i < len(weighted_avgs) else DEFAULT_AFFECT_VALUES.get(name, 0.0)

        result["total_amplitude"] = total_amplitude

        # Find dominant dimension
        max_idx = 0
        max_val = weighted_avgs[0] if weighted_avgs else 0.0
        for i, val in enumerate(weighted_avgs):
            if val > max_val:
                max_val = val
                max_idx = i

        result["dominant"] = PLUTCHIK_PRIMARIES[max_idx] if max_val > DOMINANT_AFFECT_THRESHOLD else "neutral"

        # Cache result for this cycle
        self._summary_cache = result
        self._summary_cache_cycle = self.current_cycle
        return result

    def current_affect_summary(self) -> Tuple[float, float]:
        """Compute amplitude-weighted average valence and arousal from all packets.

        For backward compatibility - derives valence/arousal from Plutchik dimensions.

        Returns:
            Tuple of (valence, arousal) - defaults to (0.5, 0.5) if no packets.
            valence: Derived from joy - sadness
            arousal: Derived from fear + surprise + anger - trust
        """
        summary = self._compute_affect_summary()

        # Derive valence: joy contributes positive, sadness contributes negative
        # Centered at 0.5
        valence = (
            0.5
            + (summary.get("joy", 0.5) - 0.5) * VALENCE_JOY_FACTOR
            - summary.get("sadness", 0.0) * VALENCE_SADNESS_FACTOR
        )

        # Derive arousal: fear, surprise, anger are activating; trust is calming
        # Average activating emotions, subtract calming influence
        activating = (summary.get("fear", 0.0) + summary.get("surprise", 0.0) + summary.get("anger", 0.0)) / 3
        arousal = (
            0.5
            + activating * AROUSAL_ACTIVATING_FACTOR
            - (summary.get("trust", 0.5) - 0.5) * AROUSAL_CALMING_FACTOR
        )

        return (max(0.0, min(1.0, valence)), max(0.0, min(1.0, arousal)))

    @property
    def current_valence(self) -> float:
        """Get current valence (0-1, negative to positive).

        Derived from joy - sadness dimensions.
        """
        valence, _ = self.current_affect_summary()
        return valence

    @property
    def current_intensity(self) -> float:
        """Get current emotional intensity as sum of packet amplitudes.

        Returns the total amplitude across all active packets.
        Higher values indicate stronger emotional field presence.
        """
        return self._compute_affect_summary()["total_amplitude"]

    @property
    def current_arousal(self) -> float:
        """Get current arousal (0-1, calm to excited).

        Derived from fear, surprise, anger (activating) vs trust (calming).
        """
        _, arousal = self.current_affect_summary()
        return arousal

    # Aliases for backward compatibility with orchestrator
    @property
    def dominant_valence(self) -> float:
        """Alias for current_valence."""
        return self.current_valence

    @property
    def dominant_arousal(self) -> float:
        """Alias for current_arousal."""
        return self.current_arousal

    @property
    def dominant_affect(self) -> str:
        """Get the dominant Plutchik emotion name.

        Returns the name of the dimension with highest weighted value,
        or "neutral" if below threshold.
        """
        return self._compute_affect_summary()["dominant"]

    # Plutchik-specific properties for direct access
    @property
    def current_joy(self) -> float:
        """Get current joy level (0-1)."""
        return self._compute_affect_summary().get("joy", DEFAULT_AFFECT_VALUES["joy"])

    @property
    def current_trust(self) -> float:
        """Get current trust level (0-1)."""
        return self._compute_affect_summary().get("trust", DEFAULT_AFFECT_VALUES["trust"])

    @property
    def current_fear(self) -> float:
        """Get current fear level (0-1)."""
        return self._compute_affect_summary().get("fear", 0.0)

    @property
    def current_surprise(self) -> float:
        """Get current surprise level (0-1)."""
        return self._compute_affect_summary().get("surprise", 0.0)

    @property
    def current_sadness(self) -> float:
        """Get current sadness level (0-1)."""
        return self._compute_affect_summary().get("sadness", 0.0)

    @property
    def current_disgust(self) -> float:
        """Get current disgust level (0-1)."""
        return self._compute_affect_summary().get("disgust", 0.0)

    @property
    def current_anger(self) -> float:
        """Get current anger level (0-1)."""
        return self._compute_affect_summary().get("anger", 0.0)

    @property
    def current_anticipation(self) -> float:
        """Get current anticipation level (0-1)."""
        return self._compute_affect_summary().get("anticipation", DEFAULT_AFFECT_VALUES["anticipation"])

    def is_bored(self) -> bool:
        """Check if the field indicates boredom.

        Boredom = mild disgust (0.1-0.3) + low anticipation (<0.4).
        Useful for cognitive diversity signaling.
        """
        summary = self._compute_affect_summary()
        disgust = summary.get("disgust", 0.0)
        anticipation = summary.get("anticipation", 0.5)
        return 0.1 <= disgust <= 0.3 and anticipation < 0.4

    def cognitive_diversity_signal(self) -> float:
        """Compute cognitive diversity score from emotional state.

        Returns:
            0.0-1.0 score where higher values indicate need for novelty.
            Based on boredom (mild disgust + low anticipation).
        """
        summary = self._compute_affect_summary()
        disgust = summary.get("disgust", 0.0)
        anticipation = summary.get("anticipation", 0.5)

        # Boredom contributes to diversity signal
        boredom_signal = 0.0
        if 0.1 <= disgust <= 0.5:
            boredom_signal = min(1.0, disgust * 2)  # Scale up disgust

        # Low anticipation amplifies boredom
        anticipation_factor = max(0.0, 1.0 - anticipation * 2)

        return min(1.0, boredom_signal * (1.0 + anticipation_factor * 0.5))
