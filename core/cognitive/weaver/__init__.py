"""Weaver control loop: Active Inference orchestration for cognitive cultivation.

This package implements the SENSE-THINK-ACT-LEARN control loop that
orchestrates Lilly's autonomous cognitive development.

Components:
- DiscoveryParameter: Computes D = H_semantic - H_structural
- CognitiveVelocityTracker: Tracks focus traversal speed
- FeedbackAggregator: Persists learnings to knowledge graph
- WeaverControlLoop: Main orchestrator

The Weaver operates on "Calm Technology" principles:
- Non-intrusive interventions
- Contextual relevance
- User-deferrable actions
- Continuous learning from outcomes
"""

from core.cognitive.weaver.control_loop import (
    ControlLoopState,
    InterventionCallback,
    WeaverControlLoop,
)
from core.cognitive.weaver.discovery import (
    DiscoveryParameter,
    DiscoveryResult,
    DiscoveryState,
    WeaverAction,
    compute_discovery_parameter,
    interpret_discovery_state,
)
from core.cognitive.weaver.feedback import (
    EmbeddingProvider,
    FeedbackAggregator,
    SimulationLearning,
    aggregate_findings,
)
from core.cognitive.weaver.velocity import (
    CognitiveVelocityTracker,
    FocusMode,
    FocusObservation,
    Momentum,
    VelocityState,
    create_velocity_tracker,
    interpret_velocity_for_weaver,
)

__all__ = [
    # Discovery
    "DiscoveryState",
    "WeaverAction",
    "DiscoveryResult",
    "DiscoveryParameter",
    "compute_discovery_parameter",
    "interpret_discovery_state",
    # Velocity
    "Momentum",
    "FocusMode",
    "FocusObservation",
    "VelocityState",
    "CognitiveVelocityTracker",
    "create_velocity_tracker",
    "interpret_velocity_for_weaver",
    # Feedback
    "SimulationLearning",
    "EmbeddingProvider",
    "FeedbackAggregator",
    "aggregate_findings",
    # Control Loop
    "ControlLoopState",
    "InterventionCallback",
    "WeaverControlLoop",
]
