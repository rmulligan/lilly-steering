"""Activation steering components for model behavior modification.

This package provides tools for computing and applying steering vectors
to influence model behavior during inference. The core approach is
Contrastive Activation Addition (CAA):

    steering_vector = mean(positive_activations) - mean(negative_activations)

Key components:
- ContrastivePair: Positive/negative example pairs for behavior targeting
- ContrastiveExtractor: Computes steering vectors from contrastive pairs
- extract_steering_vector: Convenience function for vector extraction
- ObservedInteraction: A single interaction with success/failure outcome
- SteeringObserver: Tracks interactions and generates contrastive pairs

Vector Personality System:
- VectorLibrary: Manages orthogonalized steering vectors with metadata
- VectorMetadata: Metadata for a steering vector
- CoefficientOptimizer: Adjusts coefficients based on valence feedback
- AutoExtractor: Discovers new personality dimensions from experiences
- SteeringScheduler: Context-dependent vector selection
- SteeringContext: Context for steering decisions
"""

from core.steering.contrastive_extractor import (
    ContrastiveExtractor,
    ContrastivePair,
    extract_steering_vector,
)
from core.steering.observer import (
    ObservedInteraction,
    SteeringObserver,
)
from core.steering.vector_library import VectorLibrary, VectorMetadata
from core.steering.coefficient_optimizer import CoefficientOptimizer
from core.steering.auto_extractor import AutoExtractor
from core.steering.steering_scheduler import SteeringScheduler, SteeringContext
from core.steering.plutchik_registry import PlutchikRegistry

__all__ = [
    "ContrastiveExtractor",
    "ContrastivePair",
    "extract_steering_vector",
    "ObservedInteraction",
    "SteeringObserver",
    "VectorLibrary",
    "VectorMetadata",
    "CoefficientOptimizer",
    "AutoExtractor",
    "SteeringScheduler",
    "SteeringContext",
    "PlutchikRegistry",
]
