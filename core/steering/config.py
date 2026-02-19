"""Configuration for hierarchical steering."""
import math
from dataclasses import dataclass, field
from typing import Literal, Tuple, List

# Timescale categories based on characteristic cycles (τ)
TimescaleCategory = Literal["fast", "medium", "slow", "persistent"]


@dataclass
class SteeringZone:
    """A layer range with specific steering parameters.

    Phase 1 Full Operational Autonomy:
    Parameters (max_magnitude, ema_alpha) are mutable and can be adjusted
    at runtime via HierarchicalSteerer.adjust_zone_parameter(). This allows
    Lilly to modify steering constraints based on her assessment of needs.
    """

    name: str
    layers: Tuple[int, int]  # (start, end) inclusive
    max_magnitude: float
    ema_alpha: float = 0.1  # Update rate for vector learning

    def __post_init__(self):
        if self.layers[1] < self.layers[0]:
            raise ValueError(f"Zone {self.name}: end layer must be >= start layer")

    @property
    def characteristic_cycles(self) -> float:
        """Compute characteristic timescale in cycles.

        Uses τ = -1 / ln(1 - α) which gives the number of cycles
        for the EMA to reach ~63% of a step change. This is the
        "intrinsic neural timescale" analog for steering zones.
        """
        return -1.0 / math.log(1.0 - self.ema_alpha)

    @property
    def timescale_category(self) -> TimescaleCategory:
        """Human-readable timescale category based on characteristic cycles.

        Categories:
        - fast: τ < 15 cycles (quick adaptation, exploration)
        - medium: 15 <= τ < 30 cycles (balanced responsiveness)
        - slow: 30 <= τ < 100 cycles (stable, identity-like)
        - persistent: τ >= 100 cycles (near-permanent traits)
        """
        tau = self.characteristic_cycles
        if tau < 15:
            return "fast"
        elif tau < 30:
            return "medium"
        elif tau < 100:
            return "slow"
        else:
            return "persistent"


# Existential zone for persistent self-inquiry drive
EXISTENTIAL_ZONE = SteeringZone(
    name="existential",
    layers=(8, 19),  # Mid-layers for conceptual influence
    max_magnitude=0.25,  # Low magnitude for subtle persistent bias
    ema_alpha=0.95,  # Very slow decay - persistent baseline
)

# Humor zone for subtle dry wit personality trait
HUMOR_ZONE = SteeringZone(
    name="humor",
    layers=(18, 22),  # Later layers for style/tone influence
    max_magnitude=0.20,  # Subtle - lower than existential
    ema_alpha=0.85,  # Moderate persistence - humor shouldn't be constant
)


@dataclass
class HierarchicalSteeringConfig:
    """Configuration for hierarchical multi-layer steering."""

    zones: List[SteeringZone] = field(
        default_factory=lambda: [
            SteeringZone(
                name="exploration",
                layers=(4, 8),
                max_magnitude=3.0,
                # CHANGED: Fast adaptation for curiosity (was 0.05)
                # Semantic collapse analysis: exploration should be nimble,
                # quickly following new directions to prevent stagnation
                ema_alpha=0.12,
            ),
            SteeringZone(
                name="concept",
                layers=(12, 15),  # Extended to 15 based on targeted scan
                max_magnitude=1.5,
                ema_alpha=0.2,  # Faster updates for topic rotation
            ),
            SteeringZone(
                name="identity",
                layers=(17, 18),  # Moved from 16 to 17-18 based on subjectivity peak
                max_magnitude=0.5,
                # CHANGED: Very slow adaptation for stable identity (was 0.1)
                # Semantic collapse analysis: identity should be rock-solid,
                # preventing the "philosophical poetry" attractor from dominating
                ema_alpha=0.03,
            ),
        ]
    )

    observation_layer: int = 20  # Lowered from 24 - commitment starts at 22

    def get_zone(self, layer: int) -> SteeringZone | None:
        """Get the zone containing this layer, or None if no steering."""
        for zone in self.zones:
            if zone.layers[0] <= layer <= zone.layers[1]:
                return zone
        return None
