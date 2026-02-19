"""Wave packet representation for emotional traces.

Uses Plutchik's 8 primary emotions as the affect-space dimensions:
- Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# Plutchik primary emotion names (must match order in AffectiveState)
PLUTCHIK_PRIMARIES = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]


@dataclass
class DiffusionConfig:
    """Anisotropic diffusion rates per Plutchik emotion dimension.

    Higher values = faster diffusion (fades quickly).
    Values represent sigma expansion per cognitive cycle.

    Diffusion rates reflect emotional persistence:
    - Trust and joy linger (slow diffusion)
    - Fear and surprise fade quickly (fast diffusion)
    - Anger burns out moderately fast
    """
    joy: float = 0.008         # ~87 cycles half-life (slow - joy lingers)
    trust: float = 0.005       # ~139 cycles half-life (very slow - trust persists)
    fear: float = 0.06         # ~12 cycles half-life (fast - fear fades quickly)
    surprise: float = 0.08     # ~9 cycles half-life (very fast - surprise is transient)
    sadness: float = 0.008     # ~87 cycles half-life (slow - sadness lingers)
    disgust: float = 0.03      # ~23 cycles half-life (medium)
    anger: float = 0.05        # ~14 cycles half-life (fast - anger burns out)
    anticipation: float = 0.01 # ~69 cycles half-life (slow - anticipation builds)

    def to_vector(self) -> List[float]:
        """Return rates as 8D vector matching Plutchik order."""
        return [
            self.joy,
            self.trust,
            self.fear,
            self.surprise,
            self.sadness,
            self.disgust,
            self.anger,
            self.anticipation,
        ]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiffusionConfig":
        """Deserialize from dictionary.

        Handles both legacy 6D format and new 8D Plutchik format.
        """
        # Check for legacy format
        if "arousal" in data and "joy" not in data:
            # Map legacy to Plutchik (approximate)
            return cls(
                joy=data.get("valence", 0.008),
                trust=data.get("satisfaction", 0.005),
                fear=0.06,  # New dimension
                surprise=0.08,  # New dimension
                sadness=data.get("valence", 0.008),  # Inverse of valence
                disgust=data.get("frustration", 0.03),
                anger=data.get("frustration", 0.05),
                anticipation=data.get("arousal", 0.01),
            )
        # New format or defaults
        return cls(
            joy=data.get("joy", 0.008),
            trust=data.get("trust", 0.005),
            fear=data.get("fear", 0.06),
            surprise=data.get("surprise", 0.08),
            sadness=data.get("sadness", 0.008),
            disgust=data.get("disgust", 0.03),
            anger=data.get("anger", 0.05),
            anticipation=data.get("anticipation", 0.01),
        )


# Base frequencies for phase evolution (rad/cycle)
# Mapped to Plutchik emotions
BASE_FREQUENCIES = {
    "joy": 0.04,           # Slow oscillation
    "trust": 0.02,         # Very slow
    "fear": 0.10,          # Fast oscillation
    "surprise": 0.12,      # Very fast
    "sadness": 0.03,       # Slow
    "disgust": 0.05,       # Medium
    "anger": 0.08,         # Fast
    "anticipation": 0.04,  # Slow
}

# Weighted average for composite frequency
COMPOSITE_FREQUENCY = sum(BASE_FREQUENCIES.values()) / len(BASE_FREQUENCIES)

# Wave packet initialization constants
AFFECT_SPACE_DIMENSIONS = 8
INITIAL_WAVE_PACKET_SIGMA = 0.25

# 6D → 8D Migration constants
# These define how legacy 6D affect values map to 8D Plutchik dimensions
MIGRATION_VALENCE_BASELINE = 0.5      # Neutral point for valence in legacy format
MIGRATION_SADNESS_SCALE = 0.5         # Attenuation factor for sadness derivation from low valence
MIGRATION_AROUSAL_WEIGHT = 0.5        # Weight of arousal component in anticipation calculation
MIGRATION_CURIOSITY_WEIGHT = 0.5      # Weight of curiosity component in anticipation calculation
MIGRATION_ANTICIPATION_MAX = 1.0      # Maximum cap for derived anticipation value


@dataclass
class WavePacket:
    """A single emotional trace in 8D Plutchik affect-space.

    Represents an emotional moment that diffuses and decays over time,
    carrying phase information for interference calculations.

    Attributes:
        position: 8D affect-space position [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
        initial_amplitude: Starting amplitude (derived from emotional intensity)
        deposit_cycle: Cognitive cycle when this packet was created
        anchor_memories: Memory embeddings explicitly linked at deposit time
        blended_memories: Weaker associations accumulated during decay
        sigma: Current spread in each dimension (starts at 0.25, grows via diffusion)
    """
    position: List[float]
    initial_amplitude: float
    deposit_cycle: int
    anchor_memories: Set[str] = field(default_factory=set)
    blended_memories: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    sigma: List[float] = field(default_factory=lambda: [INITIAL_WAVE_PACKET_SIGMA] * AFFECT_SPACE_DIMENSIONS)
    _amplitude: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize amplitude and handle dimension migration."""
        if self._amplitude is None:
            self._amplitude = self.initial_amplitude

        # Handle 6D → 8D migration
        if len(self.position) == 6:
            # Zero-pad the new dimensions (fear, surprise at positions 2, 3)
            # Legacy order: [arousal, valence, curiosity, satisfaction, frustration, wonder]
            # Mapped to: joy, trust, fear(0), surprise(0), sadness, disgust, anger, anticipation
            old_pos = self.position
            self.position = [
                old_pos[1],  # valence → joy
                old_pos[3],  # satisfaction → trust
                0.0,         # fear (new)
                0.0,         # surprise (new, curiosity partially maps)
                max(0, MIGRATION_VALENCE_BASELINE - old_pos[1]) * MIGRATION_SADNESS_SCALE,  # low valence → sadness
                0.0,         # disgust (new)
                old_pos[4],  # frustration → anger
                min(MIGRATION_ANTICIPATION_MAX, old_pos[0] * MIGRATION_AROUSAL_WEIGHT + old_pos[2] * MIGRATION_CURIOSITY_WEIGHT),  # arousal + curiosity → anticipation
            ]

        if len(self.sigma) == 6:
            # Zero-pad sigma for new dimensions
            self.sigma = self.sigma + [INITIAL_WAVE_PACKET_SIGMA, INITIAL_WAVE_PACKET_SIGMA]

        # Ensure correct dimensions
        while len(self.position) < AFFECT_SPACE_DIMENSIONS:
            self.position.append(0.0)
        while len(self.sigma) < AFFECT_SPACE_DIMENSIONS:
            self.sigma.append(INITIAL_WAVE_PACKET_SIGMA)

    @property
    def amplitude(self) -> float:
        """Current amplitude (decays over time)."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        """Set current amplitude."""
        self._amplitude = max(0.0, value)

    @property
    def intensity(self) -> float:
        """Compute intensity from position deviation from neutral.

        Mirrors AffectiveState.intensity() calculation.
        """
        # Joy, trust, anticipation have 0.5 baseline; others have 0 baseline
        deviations = [
            abs(self.position[0] - 0.5),  # joy
            abs(self.position[1] - 0.5),  # trust
            self.position[2],              # fear (0 baseline)
            self.position[3],              # surprise (0 baseline)
            self.position[4],              # sadness (0 baseline)
            self.position[5],              # disgust (0 baseline)
            self.position[6],              # anger (0 baseline)
            abs(self.position[7] - 0.5),  # anticipation
        ]
        return sum(deviations) / len(deviations) * 2

    def current_phase(self, cycle: int) -> float:
        """Compute current phase based on cycles elapsed.

        Phase advances with cognitive cycles, not wall-clock time.
        """
        cycles_elapsed = cycle - self.deposit_cycle
        return (cycles_elapsed * COMPOSITE_FREQUENCY) % (2 * math.pi)

    def add_anchor_memory(self, memory_id: str) -> None:
        """Add an anchor memory (explicit link at deposit)."""
        self.anchor_memories.add(memory_id)

    def add_blended_memory(self, memory_id: str, strength: float) -> None:
        """Add a blended memory (accumulated during decay)."""
        self.blended_memories[memory_id] = strength

    # Decay parameters
    BASE_HALFLIFE = 50.0  # Base halflife in cycles
    # SDFT: Maximum anchor memories for full importance bonus
    IMPORTANCE_MATURITY = 5
    # SDFT: Maximum halflife bonus for fully-mature packets (50% = 1.5x halflife)
    SDFT_IMPORTANCE_BONUS_MAX = 0.5

    @property
    def importance(self) -> float:
        """Importance factor based on anchor memory count (SDFT).

        Packets with more anchored memories are more "mature" and decay slower.
        Returns 0.0 (no anchors) to 1.0 (fully mature at IMPORTANCE_MATURITY anchors).
        """
        return min(1.0, len(self.anchor_memories) / self.IMPORTANCE_MATURITY)

    def evolve(self, config: DiffusionConfig) -> None:
        """Advance one cognitive cycle: diffuse and decay.

        Implements SDFT principle: packets with more anchored memories
        (more connections/importance) decay slower.

        Args:
            config: Diffusion rates per dimension
        """
        # Anisotropic diffusion: sigma grows at different rates
        rates = config.to_vector()
        for i in range(AFFECT_SPACE_DIMENSIONS):
            self.sigma[i] += rates[i]

        # Intensity-proportional decay
        # intensity_multiplier = 1 + intensity means intense emotions persist longer
        intensity_multiplier = 1.0 + self.intensity

        # SDFT: Importance-proportional decay
        # Packets with more anchored memories decay slower (up to 50% bonus)
        importance_multiplier = 1.0 + self.importance * self.SDFT_IMPORTANCE_BONUS_MAX

        effective_halflife = self.BASE_HALFLIFE * intensity_multiplier * importance_multiplier

        # Exponential decay per cycle
        decay_factor = 0.5 ** (1.0 / effective_halflife)
        self.amplitude = self.amplitude * decay_factor

    def is_alive(self, threshold: float = 0.01) -> bool:
        """Check if packet is above pruning threshold."""
        return self.amplitude >= threshold

    def spatial_contribution(self, sample_position: List[float]) -> float:
        """Compute Gaussian contribution at a sample position.

        Uses anisotropic Gaussian with per-dimension sigma.

        Args:
            sample_position: 8D position to sample at

        Returns:
            Contribution value (0 to amplitude)
        """
        # Handle dimension mismatch gracefully
        sample_len = len(sample_position)
        if sample_len < AFFECT_SPACE_DIMENSIONS:
            # Pad with neutral values
            sample_position = list(sample_position) + [0.0] * (AFFECT_SPACE_DIMENSIONS - sample_len)
        elif sample_len > AFFECT_SPACE_DIMENSIONS:
            sample_position = sample_position[:AFFECT_SPACE_DIMENSIONS]

        # Compute squared Mahalanobis-like distance
        # Each dimension weighted by its sigma
        weighted_sq_dist = 0.0
        for i in range(AFFECT_SPACE_DIMENSIONS):
            diff = sample_position[i] - self.position[i]
            # Avoid division by zero
            sigma = max(self.sigma[i], 0.001)
            weighted_sq_dist += (diff / sigma) ** 2

        # Gaussian falloff
        spatial_factor = math.exp(-0.5 * weighted_sq_dist)

        return self.amplitude * spatial_factor

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "position": self.position,
            "initial_amplitude": self.initial_amplitude,
            "deposit_cycle": self.deposit_cycle,
            "anchor_memories": list(self.anchor_memories),
            "blended_memories": list(self.blended_memories.items()),
            "sigma": self.sigma,
            "amplitude": self.amplitude,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WavePacket":
        """Deserialize from dictionary.

        Handles 6D → 8D migration automatically via __post_init__.
        """
        packet = cls(
            position=data["position"],
            initial_amplitude=data["initial_amplitude"],
            deposit_cycle=data["deposit_cycle"],
            anchor_memories=set(data.get("anchor_memories", [])),
            blended_memories={m[0]: m[1] for m in data.get("blended_memories", [])},
            sigma=data.get("sigma", [INITIAL_WAVE_PACKET_SIGMA] * len(data["position"])),
        )
        if "amplitude" in data:
            packet.amplitude = data["amplitude"]
        return packet
