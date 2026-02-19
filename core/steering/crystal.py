"""Crystallized vectors and configuration for EvalatisSteerer.

Crystals are "frozen" steering vectors that have proven their worth through
sustained surprise performance. They compete with emergent vectors for
selection, and can spawn new vectors through blending with other successful
crystals.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass
class CrystalEntry:
    """A crystallized steering vector in the population.

    Crystals are born from emergent vectors that sustained high surprise.
    They compete for selection based on affinity to current context and
    their staleness (how long since selected). Strong crystals can spawn
    children with other high-performing crystals.

    Attributes:
        name: Unique identifier (e.g., "exp_01191432_042").
        vector: Frozen steering vector.
        parent_names: Names of parent crystals (empty if from emergence).
        birth_cycle: Cycle number when crystallized.
        birth_surprise: Surprise value at crystallization.
        selection_count: Times this crystal has been selected.
        total_surprise: Cumulative surprise when selected.
        staleness: Normalized staleness score (0=fresh, 1=very stale).
        cycles_since_selection: Cycles since last selected.
        last_spawn_cycle: Cycle when this crystal last participated in spawn.
        children_spawned: Number of children this crystal has parented.
        retired: Whether this crystal has been pruned.
        created_at: Timestamp of crystallization.
    """

    name: str
    vector: np.ndarray
    parent_names: list[str] = field(default_factory=list)
    birth_cycle: int = 0
    birth_surprise: float = 0.0
    selection_count: int = 0
    total_surprise: float = 0.0
    staleness: float = 0.0
    cycles_since_selection: int = 0
    last_spawn_cycle: Optional[int] = None
    children_spawned: int = 0
    retired: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def avg_surprise(self) -> float:
        """Average surprise when selected."""
        if self.selection_count == 0:
            return self.birth_surprise
        return self.total_surprise / self.selection_count

    def record_selection(self, surprise: float) -> None:
        """Record that this crystal was selected for steering.

        Args:
            surprise: Surprise value from this cycle.
        """
        self.selection_count += 1
        self.total_surprise += surprise
        self.cycles_since_selection = 0

    def update_staleness(self, max_cycles: int = 100) -> None:
        """Update staleness score based on cycles since selection.

        Args:
            max_cycles: Cycles at which staleness reaches 1.0.
        """
        self.cycles_since_selection += 1
        self.staleness = min(1.0, self.cycles_since_selection / max_cycles)

    def can_spawn(self, config: "CrystallizationConfig", current_cycle: int) -> bool:
        """Check if this crystal can participate in spawning.

        Args:
            config: Configuration with spawn constraints.
            current_cycle: Current cognitive cycle number.

        Returns:
            True if crystal can spawn.
        """
        if self.retired:
            return False
        if self.children_spawned >= config.max_children_per_parent:
            return False
        if self.last_spawn_cycle is not None:
            if current_cycle - self.last_spawn_cycle < config.spawn_cooldown_cycles:
                return False
        return True


@dataclass
class CrystallizationConfig:
    """Configuration for crystallization, spawning, and pruning.

    Controls the thresholds and limits that govern the lifecycle of
    crystallized vectors in the population.
    """

    # Crystallization triggers (emergence -> population)
    min_cycles_for_crystallize: int = 20
    min_surprise_ema: float = 40.0
    min_cumulative_surprise: float = 800.0

    # Spawning (parent pair -> child)
    spawn_affinity_threshold: float = 0.7
    spawn_cooldown_cycles: int = 50
    max_children_per_parent: int = 3
    spawn_mutation_scale: float = 0.1  # Orthogonal noise magnitude

    # Pruning (retire underperformers)
    prune_min_selections: int = 10
    prune_surprise_threshold: float = 30.0
    staleness_penalty_scale: float = 0.3  # How much staleness hurts score

    # Population limits
    max_crystals_per_zone: int = 8
    preserve_count: int = 2  # Always keep top N performers

    # Selection scoring
    emergent_bonus: float = 1.2  # Bonus multiplier for emergent slot
    freshness_bonus_max: float = 1.3  # Max bonus for recently selected crystals
    staleness_max_cycles: int = 100  # Cycles until staleness = 1.0

    # Recognition signal influence (Ryan's approval feedback)
    approval_bonus_weight: float = 0.3  # How much approval affects selection (0-1)
    approval_crystallize_boost: float = 0.8  # Lower threshold multiplier when approved
    approval_prune_protection: float = 0.5  # Protection from pruning when approved
    approval_logging_threshold: float = 0.1  # Min abs(approval_bonus) to log recognition signal
    approval_prune_aggression_threshold: float = -0.3  # Below this, prune more aggressively


def generate_crystal_name(zone_name: str, cycle: int) -> str:
    """Generate a unique crystal name.

    Args:
        zone_name: Name of the zone (e.g., "exploration").
        cycle: Current cycle number.

    Returns:
        Name like "exp_01191432_042".
    """
    timestamp = datetime.now(timezone.utc).strftime("%m%d%H%M")
    zone_prefix = zone_name[:3]
    return f"{zone_prefix}_{timestamp}_{cycle:03d}"


def blend_vectors_with_mutation(
    parent1: np.ndarray,
    parent2: np.ndarray,
    mutation_scale: float = 0.1,
) -> np.ndarray:
    """Blend two parent vectors and add orthogonal mutation.

    Creates a child vector by averaging parents, then adding a small
    random orthogonal perturbation to encourage exploration of nearby
    vector space.

    Args:
        parent1: First parent vector.
        parent2: Second parent vector.
        mutation_scale: Scale of orthogonal mutation (relative to vector norm).

    Returns:
        Blended and mutated child vector.
    """
    # Average parents
    child = (parent1 + parent2) / 2.0

    # Generate random perturbation
    noise = np.random.randn(len(child)).astype(np.float32)

    # Make noise orthogonal to child direction
    child_norm = np.linalg.norm(child)
    if child_norm > 0:
        child_unit = child / child_norm
        # Remove component parallel to child
        noise = noise - np.dot(noise, child_unit) * child_unit

    # Scale and add mutation
    noise_norm = np.linalg.norm(noise)
    if noise_norm > 0:
        noise = noise / noise_norm * child_norm * mutation_scale

    return child + noise
