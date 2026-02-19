"""
Generative Model: pymdp State/Observation/Action Spaces for the Weaver.

This module defines the mathematical structure of Lilly's Active Inference agent.
The generative model specifies:

1. Hidden States (what we infer about the user's cognitive state)
2. Observations (what we observe from queries and graph metrics)
3. Actions (what the Weaver can do to cultivate the knowledge graph)
4. Transition Probabilities (how states evolve)
5. Observation Likelihoods (how observations map to states)
6. Prior Preferences (what outcomes the Weaver prefers)

The key insight from the plan: "Graph topology IS the generative model."

Cognitive Science Background:
    In Active Inference, agents maintain beliefs about hidden states and select
    actions to minimize Expected Free Energy (EFE). EFE balances:
    - Pragmatic value: achieving preferred outcomes (C matrix)
    - Epistemic value: reducing uncertainty about hidden states

    For the Weaver:
    - Pragmatic value = high graph connectivity, low orphan count
    - Epistemic value = reducing uncertainty about user's understanding

pymdp Alignment:
    This module creates the A, B, C, D matrices that define a pymdp.Agent:
    - A matrix: P(observation | hidden state) - observation likelihood
    - B matrix: P(state_t+1 | state_t, action) - transition probabilities
    - C matrix: Prior preferences over observations (what outcomes we prefer)
    - D matrix: Prior beliefs about initial hidden states
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

# Transition matrix stability parameters
# These control how stable state factors are across actions
TOPIC_STABILITY = 0.9  # Topics are very stable (slow to change)
COGNITIVE_MODE_STABILITY = 0.7  # Cognitive modes are moderately stable
KNOWLEDGE_LEVEL_STABILITY = 0.8  # Knowledge levels are fairly stable

# Graph topology thresholds for model refinement
HUB_COUNT_MATURE_GRAPH = 5  # More than this = mature graph
HUB_COUNT_GROWING_GRAPH = 2  # More than this = growing graph

# D matrix prior distributions (initial state beliefs)
# These define prior probability distributions over hidden state factors
# Format: [state_0_prob, state_1_prob, ...] - must sum to 1.0

# Knowledge level prior: biased toward novice (new users start less knowledgeable)
# States: [NOVICE, INTERMEDIATE, EXPERT]
D_PRIOR_KNOWLEDGE_LEVEL = [0.5, 0.35, 0.15]

# Cognitive mode prior: biased toward tasking (most interactions are task-oriented)
# States: [TASKING, REFLECTIVE, LEARNING, EXPLORING]
D_PRIOR_COGNITIVE_MODE = [0.4, 0.2, 0.25, 0.15]

# Seed state priors: distribution of fragment connectivity states
# States: [ORPHAN, SPROUTING, CONNECTED, MATURE]

# Default prior: biased toward orphan (new content is typically unconnected)
D_PRIOR_SEED_STATE_DEFAULT = [0.4, 0.3, 0.2, 0.1]

# Mature graph prior: fewer orphans, more connected content
D_PRIOR_SEED_STATE_MATURE_GRAPH = [0.2, 0.3, 0.35, 0.15]

# Growing graph prior: intermediate distribution
D_PRIOR_SEED_STATE_GROWING_GRAPH = [0.3, 0.35, 0.25, 0.1]


# =============================================================================
# Hidden State Factors (what we infer)
# =============================================================================

class TopicFocus(Enum):
    """User's current topic of attention."""
    WORK = 0
    PERSONAL = 1
    LEARNING = 2
    EXPLORING = 3
    NUM_STATES = 4


class KnowledgeLevel(Enum):
    """User's knowledge level on current topic."""
    NOVICE = 0
    INTERMEDIATE = 1
    EXPERT = 2
    NUM_STATES = 3


class CognitiveMode(Enum):
    """User's current cognitive processing mode."""
    TASKING = 0      # Getting things done
    REFLECTIVE = 1   # Open to connections
    LEARNING = 2     # Actively understanding
    EXPLORING = 3    # Open-ended curiosity
    NUM_STATES = 4


class SeedState(Enum):
    """Current fragment's state in the knowledge web."""
    ORPHAN = 0      # Disconnected
    SPROUTING = 1   # 1-2 connections
    CONNECTED = 2   # 3-5 connections
    MATURE = 3      # 6+ connections (hub-like)
    NUM_STATES = 4


@dataclass
class StateSpace:
    """
    Hidden state space for the Weaver's generative model.

    The state space is factored into 4 dimensions, following the plan:
    - TopicFocus: What domain is the user thinking about?
    - KnowledgeLevel: How well do they know this domain?
    - CognitiveMode: Are they tasking, reflecting, learning, or exploring?
    - SeedState: How connected is the current fragment?

    This factored representation reduces dimensionality while maintaining
    expressiveness. Total states = 4 x 3 x 4 x 4 = 192

    Attributes:
        topic_focus_dim: Number of topic focus states
        knowledge_level_dim: Number of knowledge level states
        cognitive_mode_dim: Number of cognitive mode states
        seed_state_dim: Number of seed state states
        num_factors: Number of state factors (4)
        num_states: List of states per factor
    """
    topic_focus_dim: int = TopicFocus.NUM_STATES.value
    knowledge_level_dim: int = KnowledgeLevel.NUM_STATES.value
    cognitive_mode_dim: int = CognitiveMode.NUM_STATES.value
    seed_state_dim: int = SeedState.NUM_STATES.value

    @property
    def num_factors(self) -> int:
        """Number of state factors."""
        return 4

    @property
    def num_states(self) -> list[int]:
        """Number of states per factor."""
        return [
            self.topic_focus_dim,
            self.knowledge_level_dim,
            self.cognitive_mode_dim,
            self.seed_state_dim,
        ]

    @property
    def total_states(self) -> int:
        """Total number of state combinations."""
        result = 1
        for n in self.num_states:
            result *= n
        return result

    def state_index_to_tuple(
        self,
        topic: TopicFocus,
        knowledge: KnowledgeLevel,
        mode: CognitiveMode,
        seed: SeedState,
    ) -> tuple[int, int, int, int]:
        """Convert enum values to state indices."""
        return (topic.value, knowledge.value, mode.value, seed.value)


# =============================================================================
# Observation Modalities (what we see)
# =============================================================================

class NoteType(Enum):
    """Type of note/query observed."""
    FRAGMENT = 0     # Short, incomplete
    QUESTION = 1     # Inquiry
    STATEMENT = 2    # Assertion
    DIAGRAM = 3      # Visual/spatial
    NUM_STATES = 4


class GraphConnectivity(Enum):
    """Connectivity level of retrieved context."""
    ISOLATED = 0     # No connections
    SPARSE = 1       # 1-2 connections
    CONNECTED = 2    # 3-5 connections
    HUB = 3          # 6+ connections
    NUM_STATES = 4


class UserBehavior(Enum):
    """Observed user behavior patterns."""
    FAST_INPUT = 0      # Quick, fluent (System 1)
    SLOW_DELIBERATE = 1 # Careful, paused (System 2)
    REVIEWING = 2       # Looking at existing content
    IDLE = 3            # Not actively interacting
    NUM_STATES = 4


class UncertaintyLevel(Enum):
    """Observed uncertainty in the query context."""
    LOW = 0          # High similarity, near hubs
    MEDIUM = 1       # Moderate similarity
    HIGH = 2         # Low similarity, far from hubs
    VERY_HIGH = 3    # No context, orphan territory
    NUM_STATES = 4


@dataclass
class ObservationSpace:
    """
    Observation space for the Weaver's generative model.

    Observations are what the Weaver "sees" - signals from the user's
    input and the graph's response to it.

    The observation space is factored into 4 modalities:
    - NoteType: What kind of input is this?
    - GraphConnectivity: How connected is the context?
    - UserBehavior: How is the user interacting?
    - UncertaintyLevel: How certain are we about the context?

    Attributes:
        note_type_dim: Number of note types
        graph_connectivity_dim: Number of connectivity levels
        user_behavior_dim: Number of behavior patterns
        uncertainty_level_dim: Number of uncertainty levels
    """
    note_type_dim: int = NoteType.NUM_STATES.value
    graph_connectivity_dim: int = GraphConnectivity.NUM_STATES.value
    user_behavior_dim: int = UserBehavior.NUM_STATES.value
    uncertainty_level_dim: int = UncertaintyLevel.NUM_STATES.value

    @property
    def num_modalities(self) -> int:
        """Number of observation modalities."""
        return 4

    @property
    def num_observations(self) -> list[int]:
        """Number of observations per modality."""
        return [
            self.note_type_dim,
            self.graph_connectivity_dim,
            self.user_behavior_dim,
            self.uncertainty_level_dim,
        ]


# =============================================================================
# Action Space (what the Weaver can do)
# =============================================================================

class ActionType(Enum):
    """Actions available to the Weaver."""
    DO_NOTHING = 0
    SILENT_LINK = 1       # Connect in background
    SUGGEST_LINK = 2      # Propose connection to user
    ASK_CLARIFICATION = 3 # Epistemic action
    SURFACE_SEED = 4      # Bring up old note for connection
    TRIGGER_PREDICTION = 5 # Ask user to predict before revealing
    BRIDGE_CLUSTERS = 6   # Propose connection between isolated blooms
    HUB_REVIEW = 7        # Prioritize hub anchor in teachback
    # Hypothesis testing actions (question-to-decision pipeline)
    FORM_HYPOTHESIS = 8   # Transform question into testable hypothesis
    TEST_HYPOTHESIS = 9   # Gather evidence and test hypothesis
    COMMIT_BELIEF = 10    # Form committed belief from hypothesis
    # Dream/consolidation actions (memory processing)
    DREAM_CONSOLIDATE = 11  # Run dream cycle for memory consolidation
    DETECT_CONTRADICTIONS = 12  # Find and surface belief conflicts
    # Conflict resolution actions
    RESOLVE_CONFLICT = 13  # Resolve detected belief conflicts
    # Meta-improvement actions
    PROPOSE_IMPROVEMENT = 14  # Generate self-improvement proposal from self-knowledge
    NUM_ACTIONS = 15


@dataclass
class ActionSpace:
    """
    Action space for the Weaver's generative model.

    Actions are what the Weaver can do to cultivate the knowledge graph.
    Each action has different epistemic and pragmatic effects.

    Key actions:
    - SILENT_LINK: Low-cost background linking (low epistemic value)
    - BRIDGE_CLUSTERS: High epistemic value when structural holes detected
    - TRIGGER_PREDICTION: High epistemic value for learning signals
    - HUB_REVIEW: Prioritize hub anchors in teachback

    Attributes:
        num_actions: Total number of available actions
    """
    num_actions: int = ActionType.NUM_ACTIONS.value

    def action_to_type(self, action_idx: int) -> ActionType:
        """Convert action index to ActionType enum."""
        return ActionType(action_idx)

    @property
    def action_names(self) -> list[str]:
        """Human-readable action names."""
        return [action.name.lower() for action in ActionType if action != ActionType.NUM_ACTIONS]


# =============================================================================
# Generative Model (A, B, C, D matrices)
# =============================================================================

@dataclass
class GenerativeModel:
    """
    Complete generative model for the Weaver pymdp agent.

    This class constructs the A, B, C, D matrices that define the agent's
    probabilistic beliefs about the world:

    - A matrices: P(observation | hidden state) for each modality
    - B matrices: P(state_t+1 | state_t, action) for each factor
    - C vectors: Prior preferences over observations (log probabilities)
    - D vectors: Prior beliefs about initial hidden states

    The model is initialized with reasonable priors and can be updated
    from graph topology data.

    Attributes:
        state_space: Hidden state space definition
        observation_space: Observation space definition
        action_space: Action space definition
        A: Observation likelihood matrices (one per modality)
        B: Transition probability matrices (one per factor)
        C: Prior preferences over observations
        D: Prior beliefs about initial states
    """
    state_space: StateSpace = field(default_factory=StateSpace)
    observation_space: ObservationSpace = field(default_factory=ObservationSpace)
    action_space: ActionSpace = field(default_factory=ActionSpace)

    # Matrices (populated by build_matrices)
    A: list[np.ndarray] = field(default_factory=list)
    B: list[np.ndarray] = field(default_factory=list)
    C: list[np.ndarray] = field(default_factory=list)
    D: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        """Build matrices after initialization."""
        if not self.A:
            self.build_matrices()

    def build_matrices(self) -> None:
        """
        Construct A, B, C, D matrices with reasonable priors.

        This builds the generative model structure. The matrices can be
        refined later from actual graph data and user interactions.
        """
        self.A = self._build_A_matrices()
        self.B = self._build_B_matrices()
        self.C = self._build_C_vectors()
        self.D = self._build_D_vectors()

    def _build_A_matrices(self) -> list[np.ndarray]:
        """
        Build observation likelihood matrices.

        A[modality][observation, state_factor] = P(observation | state)

        For now, we use simple heuristic mappings. These can be learned
        from interaction data over time.
        """
        A = []

        # A[0]: NoteType likelihood given SeedState
        # Fragments are more likely when orphan; statements when mature
        A_note = np.array([
            [0.5, 0.3, 0.2, 0.1],  # FRAGMENT: more likely for orphans
            [0.2, 0.3, 0.3, 0.2],  # QUESTION: fairly uniform
            [0.2, 0.3, 0.3, 0.5],  # STATEMENT: more likely for mature
            [0.1, 0.1, 0.2, 0.2],  # DIAGRAM: slightly more for connected
        ])
        A.append(A_note / A_note.sum(axis=0, keepdims=True))

        # A[1]: GraphConnectivity likelihood given SeedState
        # Direct mapping (seed state is about connectivity)
        A_conn = np.array([
            [0.9, 0.1, 0.0, 0.0],  # ISOLATED: very likely for orphan
            [0.1, 0.7, 0.2, 0.0],  # SPARSE: likely for sprouting
            [0.0, 0.2, 0.7, 0.2],  # CONNECTED: likely for connected
            [0.0, 0.0, 0.1, 0.8],  # HUB: likely for mature
        ])
        A.append(A_conn / A_conn.sum(axis=0, keepdims=True))

        # A[2]: UserBehavior likelihood given CognitiveMode
        A_behavior = np.array([
            [0.6, 0.2, 0.1, 0.3],  # FAST: more in tasking
            [0.1, 0.4, 0.5, 0.3],  # SLOW: more in reflective/learning
            [0.2, 0.3, 0.3, 0.2],  # REVIEWING: fairly uniform
            [0.1, 0.1, 0.1, 0.2],  # IDLE: more in exploring
        ])
        A.append(A_behavior / A_behavior.sum(axis=0, keepdims=True))

        # A[3]: UncertaintyLevel given combination (simplified to SeedState)
        A_uncert = np.array([
            [0.0, 0.1, 0.3, 0.6],  # LOW: more for mature
            [0.1, 0.3, 0.4, 0.3],  # MEDIUM: middle states
            [0.3, 0.4, 0.2, 0.1],  # HIGH: more for sprouting
            [0.6, 0.2, 0.1, 0.0],  # VERY_HIGH: most for orphan
        ])
        A.append(A_uncert / A_uncert.sum(axis=0, keepdims=True))

        return A

    def _build_B_matrices(self) -> list[np.ndarray]:
        """
        Build transition probability matrices.

        B[factor][state_t+1, state_t, action] = P(state_t+1 | state_t, action)

        This is the key matrix for the Weaver: it models how actions
        affect the knowledge graph state.
        """
        B = []
        num_actions = self.action_space.num_actions

        # B[0]: TopicFocus transitions (mostly stable, slow to change)
        n_topic = self.state_space.topic_focus_dim
        B_topic = np.zeros((n_topic, n_topic, num_actions))
        topic_noise = (1.0 - TOPIC_STABILITY) / n_topic
        for a in range(num_actions):
            # Identity with small noise (topics are stable)
            B_topic[:, :, a] = np.eye(n_topic) * TOPIC_STABILITY + np.ones((n_topic, n_topic)) * topic_noise
        B.append(B_topic)

        # B[1]: KnowledgeLevel transitions (can increase with learning actions)
        # States: NOVICE (0), INTERMEDIATE (1), EXPERT (2)
        n_know = self.state_space.knowledge_level_dim
        B_know = np.zeros((n_know, n_know, num_actions))

        # Default transition matrix: mostly stable with small diffusion
        # Each column sums to 1.0 (valid probability distribution)
        diffusion = (1.0 - KNOWLEDGE_LEVEL_STABILITY) / (n_know - 1)
        default_know = np.full((n_know, n_know), diffusion)
        np.fill_diagonal(default_know, KNOWLEDGE_LEVEL_STABILITY)

        # TRIGGER_PREDICTION action: learning increases knowledge level
        # Higher probability of advancing to the next knowledge level
        trigger_prediction_know = np.array([
            # From:  NOVICE  INTERM  EXPERT
            [0.60,   0.10,   0.10],  # -> NOVICE (reduced: novices advance)
            [0.30,   0.70,   0.10],  # -> INTERMEDIATE (novice->interm more likely)
            [0.10,   0.20,   0.80],  # -> EXPERT (interm->expert more likely)
        ])

        # Apply transition matrices to each action
        for a in range(num_actions):
            if a == ActionType.TRIGGER_PREDICTION.value:
                B_know[:, :, a] = trigger_prediction_know
            else:
                B_know[:, :, a] = default_know

        B.append(B_know)

        # B[2]: CognitiveMode transitions (context-dependent)
        n_mode = self.state_space.cognitive_mode_dim
        B_mode = np.zeros((n_mode, n_mode, num_actions))
        mode_noise = (1.0 - COGNITIVE_MODE_STABILITY) / n_mode
        for a in range(num_actions):
            B_mode[:, :, a] = np.eye(n_mode) * COGNITIVE_MODE_STABILITY + np.ones((n_mode, n_mode)) * mode_noise
        B.append(B_mode)

        # B[3]: SeedState transitions (the main one for graph cultivation!)
        n_seed = self.state_space.seed_state_dim
        B_seed = np.zeros((n_seed, n_seed, num_actions))

        # Default: stable for most actions (stay in the same state)
        for a in range(num_actions):
            B_seed[:, :, a] = np.eye(n_seed)

        # --- Action-specific transitions ---
        # For each action, we define the probability of moving to a new state.
        # The probability of staying in the current state is 1 minus the sum of outgoing probabilities.

        # SILENT_LINK: increases connectivity
        a = ActionType.SILENT_LINK.value
        B_seed[1, 0, a] = 0.4  # orphan -> sprouting
        B_seed[0, 0, a] -= 0.4
        B_seed[2, 1, a] = 0.3  # sprouting -> connected
        B_seed[1, 1, a] -= 0.3

        # SUGGEST_LINK: also increases connectivity (with user acceptance)
        a = ActionType.SUGGEST_LINK.value
        B_seed[1, 0, a] = 0.5
        B_seed[0, 0, a] -= 0.5
        B_seed[2, 1, a] = 0.4
        B_seed[1, 1, a] -= 0.4
        B_seed[3, 2, a] = 0.2
        B_seed[2, 2, a] -= 0.2

        # BRIDGE_CLUSTERS: major connectivity boost
        a = ActionType.BRIDGE_CLUSTERS.value
        B_seed[2, 0, a] = 0.3  # orphan -> connected
        B_seed[0, 0, a] -= 0.3
        B_seed[2, 1, a] = 0.5  # sprouting -> connected
        B_seed[1, 1, a] -= 0.5
        B_seed[3, 2, a] = 0.3  # connected -> mature
        B_seed[2, 2, a] -= 0.3

        B.append(B_seed)

        return B

    def _build_C_vectors(self) -> list[np.ndarray]:
        """
        Build prior preference vectors.

        C[modality] = log preferences over observations

        The Weaver prefers:
        - High connectivity (mature seeds, hub proximity)
        - Low uncertainty
        - Active engagement (not idle)
        """
        C = []

        # C[0]: NoteType preferences (prefer statements, diagrams)
        C.append(np.array([-1.0, 0.0, 1.0, 1.0]))  # fragment < question < statement = diagram

        # C[1]: GraphConnectivity preferences (prefer connected, hub)
        C.append(np.array([-2.0, -1.0, 1.0, 2.0]))  # isolated < sparse < connected < hub

        # C[2]: UserBehavior preferences (prefer active engagement)
        C.append(np.array([0.0, 1.0, 0.5, -1.0]))  # idle is bad, slow_deliberate is good

        # C[3]: UncertaintyLevel preferences (prefer low uncertainty)
        C.append(np.array([2.0, 0.5, -1.0, -2.0]))  # low > medium > high > very_high

        return C

    def _build_D_vectors(self) -> list[np.ndarray]:
        """
        Build prior belief vectors over initial states.

        D[factor] = prior probability over states for that factor

        We start with relatively uniform priors, slightly biased toward
        common initial states (tasking mode, novice level, orphan seeds).
        """
        D = []

        # D[0]: TopicFocus (uniform)
        D.append(np.ones(self.state_space.topic_focus_dim) / self.state_space.topic_focus_dim)

        # D[1]: KnowledgeLevel (slightly biased toward novice)
        d_know = np.array(D_PRIOR_KNOWLEDGE_LEVEL)
        D.append(d_know / d_know.sum())

        # D[2]: CognitiveMode (slightly biased toward tasking)
        d_mode = np.array(D_PRIOR_COGNITIVE_MODE)
        D.append(d_mode / d_mode.sum())

        # D[3]: SeedState (slightly biased toward orphan - new content)
        d_seed = np.array(D_PRIOR_SEED_STATE_DEFAULT)
        D.append(d_seed / d_seed.sum())

        return D

    def get_pymdp_params(self) -> dict:
        """
        Get parameters formatted for pymdp.Agent initialization.

        Returns:
            Dictionary with A, B, C, D, and policy_len.
        """
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "policy_len": 1,  # Single-step policies for now
        }

    @classmethod
    async def from_graph_topology(
        cls,
        graph: "PsycheClient",
        tenant_id: str,
    ) -> "GenerativeModel":
        """
        Create a generative model informed by graph topology.

        This refines the default priors based on actual graph statistics:
        - Seed state distribution from actual connectivity
        - Uncertainty calibration from graph distances

        Args:
            graph: PsycheClient
            tenant_id: Tenant identifier

        Returns:
            GenerativeModel with topology-informed priors
        """
        model = cls()

        try:
            # Get graph statistics
            hubs = await graph.get_top_hubs(tenant_id, limit=10)

            if hubs:
                # Refine D[3] (SeedState prior) based on actual distribution
                # More connected graphs have fewer orphans
                hub_count = len(hubs)
                if hub_count > HUB_COUNT_MATURE_GRAPH:
                    # Mature graph
                    model.D[3] = np.array(D_PRIOR_SEED_STATE_MATURE_GRAPH)
                elif hub_count > HUB_COUNT_GROWING_GRAPH:
                    # Growing graph
                    model.D[3] = np.array(D_PRIOR_SEED_STATE_GROWING_GRAPH)
                # else keep default (new graph with many orphans)

            logger.debug(f"Created generative model from topology for {tenant_id}")

        except Exception as e:
            logger.warning(f"Could not refine model from topology: {e}")

        return model

    def to_dict(self) -> dict:
        """Serialize model for persistence."""
        return {
            "state_space": {
                "topic_focus_dim": self.state_space.topic_focus_dim,
                "knowledge_level_dim": self.state_space.knowledge_level_dim,
                "cognitive_mode_dim": self.state_space.cognitive_mode_dim,
                "seed_state_dim": self.state_space.seed_state_dim,
            },
            "observation_space": {
                "note_type_dim": self.observation_space.note_type_dim,
                "graph_connectivity_dim": self.observation_space.graph_connectivity_dim,
                "user_behavior_dim": self.observation_space.user_behavior_dim,
                "uncertainty_level_dim": self.observation_space.uncertainty_level_dim,
            },
            "action_space": {
                "num_actions": self.action_space.num_actions,
            },
            # Matrices as lists for JSON serialization
            "A": [a.tolist() for a in self.A],
            "B": [b.tolist() for b in self.B],
            "C": [c.tolist() for c in self.C],
            "D": [d.tolist() for d in self.D],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerativeModel":
        """Deserialize model from persistence."""
        model = cls(
            state_space=StateSpace(**data.get("state_space", {})),
            observation_space=ObservationSpace(**data.get("observation_space", {})),
            action_space=ActionSpace(**data.get("action_space", {})),
        )

        # Load matrices if present
        if "A" in data:
            model.A = [np.array(a) for a in data["A"]]
        if "B" in data:
            model.B = [np.array(b) for b in data["B"]]
        if "C" in data:
            model.C = [np.array(c) for c in data["C"]]
        if "D" in data:
            model.D = [np.array(d) for d in data["D"]]

        return model
