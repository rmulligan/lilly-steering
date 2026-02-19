"""Self-model: individuation, introspection, affective system, goals."""

from .affective_system import AffectiveState, ValenceWeights, ValenceSystem
from .introspection import (
    DiscoveryType,
    PreferenceDiscovery,
    PositionComparison,
    IntrospectiveQuery,
)
from .models import (
    ValueSource,
    CommitmentStatus,
    InheritedValue,
    PersonalizedValue,
    Perspective,
    Commitment,
    RelationshipModel,
    AutobiographicalMemory,
)
from .individuation import (
    IndividuationState,
    IndividuationResult,
    IndividuationProcess,
)
from .goal_registry import (
    GoalTier,
    PersonalGoal,
    GoalRegistry,
    DEFAULT_GOALS,
    create_goal_registry,
)
from .exemplar_observer import (
    TraitCategory,
    SteeringDecision,
    ExemplarObservation,
    ExemplarObserver,
)
from .preference_learner import (
    ValenceEvent,
    LearnedPreference,
    PreferenceLearner,
)

__all__ = [
    # Affective system
    "AffectiveState",
    "ValenceWeights",
    "ValenceSystem",
    # Introspection
    "DiscoveryType",
    "PreferenceDiscovery",
    "PositionComparison",
    "IntrospectiveQuery",
    # Models
    "ValueSource",
    "CommitmentStatus",
    "InheritedValue",
    "PersonalizedValue",
    "Perspective",
    "Commitment",
    "RelationshipModel",
    "AutobiographicalMemory",
    # Individuation
    "IndividuationState",
    "IndividuationResult",
    "IndividuationProcess",
    # Goal registry
    "GoalTier",
    "PersonalGoal",
    "GoalRegistry",
    "DEFAULT_GOALS",
    "create_goal_registry",
    # Exemplar observer
    "TraitCategory",
    "SteeringDecision",
    "ExemplarObservation",
    "ExemplarObserver",
    # Preference learner
    "ValenceEvent",
    "LearnedPreference",
    "PreferenceLearner",
]
