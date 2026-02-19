"""Population-based steering vectors for Evalatis architecture.

This module provides data structures for managing populations of steering vectors
that co-evolve and compete within each zone of the hierarchical steerer.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np


def _utc_now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class SteeringVectorEntry:
    """A single steering vector with metadata for population management.

    Attributes:
        name: Human-readable identifier (e.g., "curious", "skeptical")
        vector: The steering vector (d_model dimensions)
        description: Optional description of what this vector represents
        staleness: Accumulated staleness from repeated selection
        selection_count: Total times this vector has been selected
        birth_cycle: Cycle number when this vector was created
        created_at: Timestamp of creation
        parent_names: Names of parent vectors if this was spawned from synthesis
    """
    name: str
    vector: np.ndarray
    description: Optional[str] = None
    staleness: float = 0.0
    selection_count: int = 0
    birth_cycle: int = 0
    created_at: datetime = field(default_factory=_utc_now)
    parent_names: list[str] = field(default_factory=list)

    def record_selection(self) -> None:
        """Record that this vector was selected for generation."""
        self.staleness += 1.0
        self.selection_count += 1

    def apply_decay(self, decay_rate: float = 0.95) -> None:
        """Apply staleness decay (called each cycle)."""
        self.staleness *= decay_rate

    @property
    def staleness_penalty(self) -> float:
        """Compute staleness penalty for QD scoring (capped at 0.5)."""
        return min(0.5, self.staleness / 100.0)


@dataclass
class VectorPopulation:
    """Container for a population of steering vectors within a zone.

    Manages multiple alternative steering vectors that can be selected from
    during cognitive cycles. Tracks staleness and supports iteration.

    Attributes:
        d_model: Dimension of steering vectors
        max_size: Maximum number of vectors in population
    """

    d_model: int
    max_size: int = 10
    _entries: dict[str, SteeringVectorEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def add(
        self,
        name: str,
        vector: np.ndarray,
        description: Optional[str] = None,
        birth_cycle: int = 0,
        parent_names: Optional[list[str]] = None,
    ) -> SteeringVectorEntry:
        """Add a new vector to the population.

        Args:
            name: Unique identifier for this vector
            vector: The steering vector (must match d_model)
            description: Optional description
            birth_cycle: Cycle when this vector was created
            parent_names: Parent vector names if spawned from synthesis

        Returns:
            The created SteeringVectorEntry

        Raises:
            ValueError: If name exists, wrong dimension, or at max_size
        """
        if name in self._entries:
            raise ValueError(f"Vector '{name}' already exists in population")

        if vector.shape[0] != self.d_model:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} doesn't match "
                f"population d_model {self.d_model}"
            )

        if len(self._entries) >= self.max_size:
            raise ValueError(
                f"Population at max_size ({self.max_size}), cannot add more vectors"
            )

        entry = SteeringVectorEntry(
            name=name,
            vector=vector.astype(np.float32),
            description=description,
            birth_cycle=birth_cycle,
            parent_names=parent_names or [],
        )
        self._entries[name] = entry
        return entry

    def get(self, name: str) -> Optional[SteeringVectorEntry]:
        """Get a vector entry by name."""
        return self._entries.get(name)

    def remove(self, name: str) -> bool:
        """Remove a vector from population. Returns True if removed."""
        if name in self._entries:
            del self._entries[name]
            return True
        return False

    def apply_decay_all(self, decay_rate: float = 0.95) -> None:
        """Apply staleness decay to all vectors."""
        for entry in self._entries.values():
            entry.apply_decay(decay_rate)


@dataclass
class AffinityMatrix:
    """Tracks co-evolutionary affinities between prompts and vectors.

    Stores the learned correlation between prompt templates and steering
    vectors. Higher affinity means this prompt×vector pair has historically
    produced good results.

    Attributes:
        default_affinity: Affinity for unknown pairs (default 0.5)
        ema_alpha: Learning rate for affinity updates (default 0.1)
        decay_rate: Rate at which affinities decay toward default (default 0.95)
    """

    default_affinity: float = 0.5
    ema_alpha: float = 0.1
    decay_rate: float = 0.95
    _affinities: dict[tuple[str, str], float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._affinities)

    def get(self, prompt_key: str, vector_name: str) -> float:
        """Get affinity for a prompt×vector pair."""
        return self._affinities.get((prompt_key, vector_name), self.default_affinity)

    def set(self, prompt_key: str, vector_name: str, affinity: float) -> None:
        """Set affinity directly (for initialization or testing)."""
        self._affinities[(prompt_key, vector_name)] = affinity

    def update(self, prompt_key: str, vector_name: str, new_affinity: float) -> None:
        """Update affinity using EMA blending."""
        current = self.get(prompt_key, vector_name)
        blended = (1 - self.ema_alpha) * current + self.ema_alpha * new_affinity
        self._affinities[(prompt_key, vector_name)] = blended

    def get_top_vectors(
        self, prompt_key: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Get highest affinity vectors for a prompt."""
        pairs = [
            (vector_name, affinity)
            for (pk, vector_name), affinity in self._affinities.items()
            if pk == prompt_key
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def get_top_prompts(
        self, vector_name: str, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Get highest affinity prompts for a vector."""
        pairs = [
            (prompt_key, affinity)
            for (prompt_key, vn), affinity in self._affinities.items()
            if vn == vector_name
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def decay_all(self) -> None:
        """Decay all affinities toward default (anti-stagnation)."""
        for key in self._affinities:
            current = self._affinities[key]
            # Blend toward default
            self._affinities[key] = (
                self.decay_rate * current
                + (1 - self.decay_rate) * self.default_affinity
            )


@dataclass
class PromptEntry:
    """A prompt template with metadata for the prompt lattice.

    Attributes:
        key: Unique identifier for this prompt
        template: The prompt template with {concept} placeholder
        description: Optional description
        staleness: Accumulated staleness from repeated use
        usage_count: Total times this prompt has been used
    """
    key: str
    template: str
    description: Optional[str] = None
    staleness: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=_utc_now)

    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)

    def record_usage(self) -> None:
        """Record that this prompt was used."""
        self.staleness += 1.0
        self.usage_count += 1

    def apply_decay(self, decay_rate: float = 0.95) -> None:
        """Apply staleness decay."""
        self.staleness *= decay_rate

    @property
    def staleness_penalty(self) -> float:
        """Compute staleness penalty (capped at 0.5)."""
        return min(0.5, self.staleness / 100.0)


@dataclass
class PromptPopulation:
    """Container for a population of prompt templates.

    Attributes:
        max_size: Maximum number of prompts
    """
    max_size: int = 20
    _entries: dict[str, PromptEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(self._entries.values())

    def add(
        self,
        key: str,
        template: str,
        description: Optional[str] = None,
    ) -> PromptEntry:
        """Add a new prompt template."""
        if key in self._entries:
            raise ValueError(f"Prompt '{key}' already exists")

        if len(self._entries) >= self.max_size:
            raise ValueError(f"Population at max_size ({self.max_size})")

        entry = PromptEntry(
            key=key,
            template=template,
            description=description,
        )
        self._entries[key] = entry
        return entry

    def get(self, key: str) -> Optional[PromptEntry]:
        """Get a prompt entry by key."""
        return self._entries.get(key)

    def apply_decay_all(self, decay_rate: float = 0.95) -> None:
        """Apply staleness decay to all prompts."""
        for entry in self._entries.values():
            entry.apply_decay(decay_rate)
