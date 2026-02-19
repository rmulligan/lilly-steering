"""Pydantic schemas for the feature substrate layer.

This module defines the core data types for the emergent cognition substrate:
- AttractorType: Categories of attractors in feature space
- SubstratePhase: Lifecycle phases (bootstrap -> weight_learning -> self_coherence)
- Attractor: Stable configurations in feature space
- FeatureActivation: Individual SAE feature activation
- EvokedContext: Memories surfaced by feature patterns
- SubstrateHealth: Diagnostic metrics
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


class AttractorType(str, Enum):
    """Types of attractors in feature space."""
    ENTITY = "entity"       # From knowledge graph entities
    ZETTEL = "zettel"       # From InsightZettels
    MOOD = "mood"           # From affective states
    IDENTITY = "identity"   # From core values/commitments
    EMERGENT = "emergent"   # From novel feature clusters


class SubstratePhase(str, Enum):
    """Lifecycle phases of the substrate."""
    BOOTSTRAP = "bootstrap"              # First ~1000 cycles
    WEIGHT_LEARNING = "weight_learning"  # Cycles ~1000-5000
    SELF_COHERENCE = "self_coherence"    # Ongoing


class DreamCycleType(str, Enum):
    """Types of consolidation cycles."""
    MICRO = "micro"    # Per-cycle: buffer -> trace
    NAP = "nap"        # ~2 hours: trace pruning, strong -> embedding
    FULL = "full"      # Daily: embedding clusters -> graph attractors
    DEEP = "deep"      # Weekly: attractor pruning, topology restructuring


class FeatureActivation(BaseModel):
    """A single SAE feature activation."""
    feature_idx: int
    activation: float


class Attractor(BaseModel):
    """A stable configuration in feature space with gravitational pull.

    Attractors represent concepts, memories, moods, or emergent patterns
    that pull nearby feature activations toward them.
    """
    uid: str
    attractor_type: AttractorType
    position: list[float]           # [embed_dim] vector
    source_uid: str                 # UID of source (Entity, Zettel, etc.)
    source_name: str = ""           # Human-readable name

    pull_strength: float = 1.0      # How strongly it attracts
    pull_radius: float = 0.5        # How far its influence extends
    value_weight: float = 1.0       # Contribution to value signal

    visit_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    last_visited: datetime = Field(default_factory=utc_now)


class EvokedEntity(BaseModel):
    """An entity evoked by feature proximity."""
    uid: str
    name: str
    entity_type: str
    activation: float               # Strength of evocation


class EvokedZettel(BaseModel):
    """A zettel evoked by feature proximity."""
    uid: str
    insight: str
    activation: float


class EvokedMood(BaseModel):
    """A mood evoked by feature proximity."""
    name: str
    valence: float
    arousal: float
    activation: float


class EvokedQuestion(BaseModel):
    """An open question evoked by feature proximity."""
    uid: str
    question: str
    activation: float


class EvokedContext(BaseModel):
    """Memories and moods surfaced by current feature state."""
    entities: list[EvokedEntity] = Field(default_factory=list)
    zettels: list[EvokedZettel] = Field(default_factory=list)
    moods: list[EvokedMood] = Field(default_factory=list)
    questions: list[EvokedQuestion] = Field(default_factory=list)


class SubstrateHealth(BaseModel):
    """Diagnostic metrics for substrate health."""
    attractor_count: int
    mean_attractor_strength: float
    feature_coverage: float         # Fraction of features near an attractor
    trace_sparsity: float           # Sparsity of trace matrix
    embedding_variance: float       # Variance in embedding space
    phase: SubstratePhase
    total_observations: int


class ValueSignalSnapshot(BaseModel):
    """Snapshot of value signal components for learning."""
    surprise: float = 0.0
    insight: float = 0.0
    narration: float = 0.0
    feedback: float = 0.0
    self_coherence: float = 0.0
    composite: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


class PhaseTransition(BaseModel):
    """Record of a phase transition."""
    from_phase: SubstratePhase
    to_phase: SubstratePhase
    timestamp: datetime = Field(default_factory=utc_now)
    weights_snapshot: dict[str, float] = Field(default_factory=dict)
    observation_count: int = 0
