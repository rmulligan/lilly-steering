# core/steering/population_steerer.py
"""Population-based hierarchical steering for Evalatis architecture.

Extends the HierarchicalSteerer concept to use populations of vectors
per zone, enabling multi-path evaluation and QD selection.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from core.steering.config import HierarchicalSteeringConfig
from core.steering.population import VectorPopulation, AffinityMatrix, SteeringVectorEntry


@dataclass
class PopulationSteerer:
    """Manages populations of steering vectors across zones.

    Each zone holds a population of alternative vectors. Selection considers
    both affinity scores and staleness penalties to balance exploitation
    and exploration.

    Attributes:
        config: Hierarchical steering configuration
        d_model: Model dimension (vector size)
        population_size: Maximum vectors per zone
        populations: Dict mapping zone names to VectorPopulation
        affinity_matrix: Tracks prompt x vector correlations
    """
    config: HierarchicalSteeringConfig
    d_model: int
    population_size: int = 10
    decay_rate: float = 0.95
    populations: Dict[str, VectorPopulation] = field(default_factory=dict)
    affinity_matrix: AffinityMatrix = field(default_factory=AffinityMatrix)
    _current_prompt_key: str = "default"
    _last_selected: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize empty populations for each zone."""
        if not self.populations:
            self.populations = {
                zone.name: VectorPopulation(
                    d_model=self.d_model,
                    max_size=self.population_size,
                )
                for zone in self.config.zones
            }

    def add_vector(
        self,
        zone_name: str,
        vector_name: str,
        vector: np.ndarray,
        description: Optional[str] = None,
        birth_cycle: int = 0,
    ) -> Optional[SteeringVectorEntry]:
        """Add a vector to a zone's population.

        Returns the created entry, or None if zone doesn't exist.
        """
        if zone_name not in self.populations:
            return None

        return self.populations[zone_name].add(
            name=vector_name,
            vector=vector,
            description=description,
            birth_cycle=birth_cycle,
        )

    def set_prompt_context(self, prompt_key: str) -> None:
        """Set current prompt context for affinity-based selection."""
        self._current_prompt_key = prompt_key

    def get_vector(self, layer: int) -> Optional[np.ndarray]:
        """Get steering vector for a layer using population selection.

        Selects the vector with highest effective score:
        effective_score = affinity * (1 - staleness_penalty)

        Args:
            layer: The transformer layer index

        Returns:
            Selected steering vector (capped to zone magnitude), or None
        """
        if layer >= self.config.observation_layer:
            return None

        zone = self.config.get_zone(layer)
        if zone is None:
            return None

        population = self.populations.get(zone.name)
        if population is None or len(population) == 0:
            return None

        # Select vector with highest effective score
        best_entry: Optional[SteeringVectorEntry] = None
        best_score = -1.0

        for entry in population:
            affinity = self.affinity_matrix.get(self._current_prompt_key, entry.name)
            effective_score = affinity * (1 - entry.staleness_penalty)

            if effective_score > best_score:
                best_score = effective_score
                best_entry = entry

        if best_entry is None:
            return None

        # Track selection for later recording
        self._last_selected[zone.name] = best_entry.name

        # Return vector capped to zone max magnitude
        vector = best_entry.vector.copy()
        magnitude = np.linalg.norm(vector)
        if magnitude > zone.max_magnitude:
            vector = vector * (zone.max_magnitude / magnitude)

        return vector

    def record_selection(self, zone_name: str, vector_name: str) -> None:
        """Record that a vector was selected (updates staleness)."""
        population = self.populations.get(zone_name)
        if population is None:
            return

        entry = population.get(vector_name)
        if entry is not None:
            entry.record_selection()

    def record_last_selections(self) -> None:
        """Record selections from the last get_vector calls."""
        for zone_name, vector_name in self._last_selected.items():
            self.record_selection(zone_name, vector_name)
        self._last_selected.clear()

    def apply_cycle_decay(self) -> None:
        """Apply staleness decay to all populations and affinity matrix."""
        for population in self.populations.values():
            population.apply_decay_all(self.decay_rate)
        self.affinity_matrix.decay_all()

    def update_affinity(
        self,
        prompt_key: str,
        vector_name: str,
        signal: float,
    ) -> None:
        """Update affinity based on outcome signal (e.g., QD score)."""
        self.affinity_matrix.update(prompt_key, vector_name, signal)
