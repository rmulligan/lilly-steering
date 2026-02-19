"""Feature substrate for emergent cognition.

This module provides a unified substrate enabling:
1. Associative memory surfacing
2. Self-organizing attention
3. Feature-based personality traits
4. Predictive feature anticipation
"""

from core.substrate.schemas import (
    AttractorType,
    SubstratePhase,
    DreamCycleType,
    FeatureActivation,
    Attractor,
    EvokedEntity,
    EvokedZettel,
    EvokedMood,
    EvokedQuestion,
    EvokedContext,
    SubstrateHealth,
    ValueSignalSnapshot,
    PhaseTransition,
)
from core.substrate.activation_buffer import ActivationBuffer
from core.substrate.trace_matrix import TraceMatrix
from core.substrate.embedding_space import EmbeddingSpace
from core.substrate.value_signal import ValueSignal
from core.substrate.consolidation import ConsolidationManager
from core.substrate.substrate import FeatureSubstrate

__all__ = [
    "AttractorType",
    "SubstratePhase",
    "DreamCycleType",
    "FeatureActivation",
    "Attractor",
    "EvokedEntity",
    "EvokedZettel",
    "EvokedMood",
    "EvokedQuestion",
    "EvokedContext",
    "SubstrateHealth",
    "ValueSignalSnapshot",
    "PhaseTransition",
    "ActivationBuffer",
    "TraceMatrix",
    "EmbeddingSpace",
    "ValueSignal",
    "ConsolidationManager",
    "FeatureSubstrate",
]
