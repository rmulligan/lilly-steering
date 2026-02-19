"""Active inference: belief updating, surprise detection."""

from core.active_inference.belief_store import (
    BeliefRelationType,
    BeliefConfidence,
    DialecticalHistory,
    CommittedBelief,
    BeliefRelation,
    BeliefStore,
    create_belief_store,
)
from core.active_inference.graph_entropy import (
    EntropyResult,
    GraphEntropy,
    compute_graph_entropy,
    ENTROPY_THRESHOLD_CULTIVATE,
    ORPHAN_RATE_CONCERN,
    HUB_CONCENTRATION_CONCERN,
)
from core.active_inference.generative_model import (
    GenerativeModel,
    StateSpace,
    ActionSpace,
    ActionType,
    ObservationSpace,
    TopicFocus,
    KnowledgeLevel,
    CognitiveMode,
    SeedState,
    NoteType,
    GraphConnectivity,
    UserBehavior,
    UncertaintyLevel,
)
from core.active_inference.observation_encoder import (
    EncodedObservation,
    ObservationEncoder,
    encode_observation,
)
from core.active_inference.belief_updater import (
    BeliefDistribution,
    BeliefUpdater,
    UpdateResult,
    create_initial_beliefs,
)
from core.active_inference.policy_selector import (
    ActionContext,
    PolicyResult,
    PolicySelector,
    select_weaver_action,
)
from core.active_inference.semantic_entropy import (
    SemanticEntropyResult,
    SemanticEntropyCalculator,
    compute_semantic_entropy,
    SEMANTIC_ENTROPY_HIGH,
    SEMANTIC_ENTROPY_LOW,
)

__all__ = [
    # Belief store
    "BeliefRelationType",
    "BeliefConfidence",
    "DialecticalHistory",
    "CommittedBelief",
    "BeliefRelation",
    "BeliefStore",
    "create_belief_store",
    # Graph entropy
    "EntropyResult",
    "GraphEntropy",
    "compute_graph_entropy",
    "ENTROPY_THRESHOLD_CULTIVATE",
    "ORPHAN_RATE_CONCERN",
    "HUB_CONCENTRATION_CONCERN",
    # Generative Model
    "GenerativeModel",
    "StateSpace",
    "ActionSpace",
    "ActionType",
    "ObservationSpace",
    # State Enums
    "TopicFocus",
    "KnowledgeLevel",
    "CognitiveMode",
    "SeedState",
    # Observation Enums
    "NoteType",
    "GraphConnectivity",
    "UserBehavior",
    "UncertaintyLevel",
    # Observation Encoder
    "EncodedObservation",
    "ObservationEncoder",
    "encode_observation",
    # Belief Updater
    "BeliefDistribution",
    "BeliefUpdater",
    "UpdateResult",
    "create_initial_beliefs",
    # Policy Selector
    "ActionContext",
    "PolicyResult",
    "PolicySelector",
    "select_weaver_action",
    # Semantic Entropy
    "SemanticEntropyResult",
    "SemanticEntropyCalculator",
    "compute_semantic_entropy",
    "SEMANTIC_ENTROPY_HIGH",
    "SEMANTIC_ENTROPY_LOW",
]
