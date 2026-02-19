"""Cognitive systems: weaver control loop, cognitive stream, and identity.

This package implements higher-order cognitive functions:
- Weaver Control Loop: Autonomous SENSE-THINK-ACT-LEARN cycle
- Cognitive Stream: Externalized cognition and narration pipeline
- Identity/Creed: Identity constitution and coherence scoring

All systems are designed for graceful degradation and can be
enabled/disabled independently via feature flags in settings.
"""

from core.cognitive.types import (
    InterventionType,
    WeaverContext,
    WeaverIntervention,
    WeaverPolicy,
)
from core.cognitive.weaver import (
    ControlLoopState,
    InterventionCallback,
    WeaverControlLoop,
    DiscoveryParameter,
    DiscoveryResult,
    DiscoveryState,
    WeaverAction,
    compute_discovery_parameter,
    interpret_discovery_state,
    CognitiveVelocityTracker,
    FocusMode,
    FocusObservation,
    Momentum,
    VelocityState,
    create_velocity_tracker,
    interpret_velocity_for_weaver,
    FeedbackAggregator,
    SimulationLearning,
    aggregate_findings,
)
from core.cognitive.polarity import (
    PolarityDetector,
    detect_repetition,
)
from core.cognitive.tension import (
    TensionTracker,
    FeatureTension,
    get_tension_tracker,
)
from core.cognitive.stage import (
    CognitiveStage,
    StageConfig,
    STAGE_CONFIGS,
    DEFAULT_EXPLORATION_CONFIG,
)
from core.cognitive.goal import (
    InquiryGoal,
    detect_emerging_goal,
    extract_goal_question,
)
from core.cognitive.saturation import (
    SaturationSignal,
    check_saturation,
    advance_stage,
)
from core.cognitive.stage_prompt import (
    build_stage_prompt,
    adjust_steerer_for_stage,
)
from core.cognitive.episode_selector import (
    EmotionalSignals,
    EpisodeSelector,
    SelectionSignals,
)

__all__ = [
    # Core types
    "InterventionType",
    "WeaverPolicy",
    "WeaverContext",
    "WeaverIntervention",
    # Weaver control loop
    "ControlLoopState",
    "InterventionCallback",
    "WeaverControlLoop",
    # Discovery
    "DiscoveryParameter",
    "DiscoveryResult",
    "DiscoveryState",
    "WeaverAction",
    "compute_discovery_parameter",
    "interpret_discovery_state",
    # Velocity
    "CognitiveVelocityTracker",
    "FocusMode",
    "FocusObservation",
    "Momentum",
    "VelocityState",
    "create_velocity_tracker",
    "interpret_velocity_for_weaver",
    # Feedback
    "FeedbackAggregator",
    "SimulationLearning",
    "aggregate_findings",
    # Polarity / Dialectical
    "PolarityDetector",
    "detect_repetition",
    # Tension Tracking
    "TensionTracker",
    "FeatureTension",
    "get_tension_tracker",
    # Progressive Thinking - Stages
    "CognitiveStage",
    "StageConfig",
    "STAGE_CONFIGS",
    "DEFAULT_EXPLORATION_CONFIG",
    # Progressive Thinking - Goals
    "InquiryGoal",
    "detect_emerging_goal",
    "extract_goal_question",
    # Progressive Thinking - Saturation
    "SaturationSignal",
    "check_saturation",
    "advance_stage",
    # Progressive Thinking - Prompts
    "build_stage_prompt",
    "adjust_steerer_for_stage",
    # Episode Selection
    "EmotionalSignals",
    "EpisodeSelector",
    "SelectionSignals",
]
