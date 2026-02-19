"""
Belief Updater: Bayesian Inference for the Weaver's Mind.

This module implements pymdp-compatible belief updating for Lilly's Active
Inference engine. It maintains a probability distribution over hidden states
and updates this distribution based on new observations.

Cognitive Science Background:
    Active Inference agents maintain a "generative model" of the world-a
    probabilistic belief about hidden states that explain their observations.
    When new observations arrive, beliefs are updated via Bayes' rule:

        P(state | observation) ~ P(observation | state) x P(state)

    The key insight: the brain doesn't store raw data-it updates beliefs.
    Every new experience changes what we believe about the world.

pymdp Integration:
    This module wraps pymdp's inference functions to work with Lilly's
    observation encoder and generative model. The key functions:

    - `infer_states`: Update beliefs given new observations
    - `update_B`: Learn transition probabilities from experience
    - `compute_efe`: Calculate Expected Free Energy for action selection

Usage:
    from core.active_inference import BeliefUpdater

    updater = BeliefUpdater(generative_model)
    new_beliefs = updater.update(current_beliefs, observation)

    # Action selection via EFE
    efe_values = updater.compute_action_efe(new_beliefs)
    best_action = ActionType(efe_values.argmin())
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.active_inference.generative_model import (
    GenerativeModel,
    StateSpace,
    ActionType,
)
from core.active_inference.observation_encoder import EncodedObservation

logger = logging.getLogger(__name__)

# Belief update thresholds
SIGNIFICANT_SURPRISE_THRESHOLD = 0.1  # Threshold for determining significant belief changes
DEFAULT_LEARNING_RATE = 0.5  # Default weight for new observations in belief updates

# Risk computation multipliers
EPISTEMIC_ACTION_RISK_MULTIPLIER = 0.5  # Risk reduction for epistemic actions

# Ambiguity computation multipliers per action type
# Maps ActionType -> multiplier for base ambiguity
AMBIGUITY_MULTIPLIERS: dict[ActionType, float] = {
    ActionType.ASK_CLARIFICATION: 0.3,   # Ambiguity reduction for clarification
    ActionType.TRIGGER_PREDICTION: 0.5,  # Ambiguity reduction for prediction
    ActionType.DO_NOTHING: 1.2,          # Ambiguity increase for inaction
}
AMBIGUITY_DEFAULT_MULTIPLIER = 1.0  # Default multiplier (no change)

# Epistemic value multipliers per action type
# Maps ActionType -> multiplier for base epistemic value
EPISTEMIC_VALUE_MULTIPLIERS: dict[ActionType, float] = {
    ActionType.DO_NOTHING: 0.0,           # Learn nothing from inaction
    ActionType.SILENT_LINK: 0.1,          # Small learning from background linking
    ActionType.SUGGEST_LINK: 0.3,         # User response teaches us
    ActionType.ASK_CLARIFICATION: 0.8,    # High learning from clarification
    ActionType.SURFACE_SEED: 0.4,         # Connecting old ideas teaches context
    ActionType.TRIGGER_PREDICTION: 0.9,   # Maximum learning from prediction
    ActionType.BRIDGE_CLUSTERS: 0.6,      # Learning from bridging
    ActionType.HUB_REVIEW: 0.5,           # Learning from hub review
}
EPISTEMIC_VALUE_DEFAULT = 0.3  # Default for unspecified actions


@dataclass
class BeliefDistribution:
    """
    A probability distribution over hidden states.

    In pymdp, beliefs are represented as a list of vectors-one for each
    state factor. Each vector is a categorical distribution that sums to 1.

    Attributes:
        factors: List of probability vectors, one per state factor
            - factors[0]: TopicFocus distribution (4 values)
            - factors[1]: KnowledgeLevel distribution (3 values)
            - factors[2]: CognitiveMode distribution (4 values)
            - factors[3]: SeedState distribution (4 values)
        timestamp: When this belief was last updated
        observation_count: How many observations have been incorporated
    """

    factors: list[np.ndarray]
    timestamp: Optional[str] = None
    observation_count: int = 0

    @classmethod
    def uniform(cls, state_space: StateSpace) -> "BeliefDistribution":
        """Create a uniform (maximum entropy) belief distribution."""
        factors = [
            np.ones(state_space.topic_focus_dim) / state_space.topic_focus_dim,
            np.ones(state_space.knowledge_level_dim) / state_space.knowledge_level_dim,
            np.ones(state_space.cognitive_mode_dim) / state_space.cognitive_mode_dim,
            np.ones(state_space.seed_state_dim) / state_space.seed_state_dim,
        ]
        return cls(factors=factors)

    @classmethod
    def from_prior(cls, D: list[np.ndarray]) -> "BeliefDistribution":
        """Create a belief distribution from D matrix priors."""
        factors = [d.copy() for d in D]
        return cls(factors=factors)

    @property
    def entropy(self) -> float:
        """Compute total entropy across all factors."""
        total = 0.0
        for factor in self.factors:
            # Shannon entropy: -sum(p * log(p))
            p = np.clip(factor, 1e-10, 1.0)  # Avoid log(0)
            total -= np.sum(p * np.log(p))
        return total

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        total = 0.0
        for factor in self.factors:
            n = len(factor)
            # Uniform distribution entropy: log(n)
            total += np.log(n)
        return total

    @property
    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1] range."""
        max_ent = self.max_entropy
        if max_ent == 0:
            return 0.0
        return self.entropy / max_ent

    # Named accessors for individual factors (for compatibility with EpistemicPlanner)
    @property
    def topic_focus(self) -> np.ndarray:
        """TopicFocus distribution (factor 0)."""
        return self.factors[0]

    @property
    def knowledge_level(self) -> np.ndarray:
        """KnowledgeLevel distribution (factor 1)."""
        return self.factors[1]

    @property
    def cognitive_mode(self) -> np.ndarray:
        """CognitiveMode distribution (factor 2)."""
        return self.factors[2]

    @property
    def seed_state(self) -> np.ndarray:
        """SeedState distribution (factor 3)."""
        return self.factors[3]

    def get_mode(self) -> list[int]:
        """Get the most likely state (argmax for each factor)."""
        return [int(np.argmax(f)) for f in self.factors]

    def get_confidence(self) -> list[float]:
        """Get the confidence (max probability) for each factor."""
        return [float(np.max(f)) for f in self.factors]

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "factors": [f.tolist() for f in self.factors],
            "timestamp": self.timestamp,
            "observation_count": self.observation_count,
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "mode": self.get_mode(),
            "confidence": self.get_confidence(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BeliefDistribution":
        """Deserialize from persistence."""
        factors = [np.array(f) for f in data["factors"]]
        return cls(
            factors=factors,
            timestamp=data.get("timestamp"),
            observation_count=data.get("observation_count", 0),
        )


@dataclass
class UpdateResult:
    """
    Result of a belief update operation.

    Attributes:
        prior: Belief before the update
        posterior: Belief after the update
        observation: The observation that triggered the update
        surprise: Information gain from this observation
        action_efe: Expected Free Energy for each action
        recommended_action: Action with lowest EFE
    """

    prior: BeliefDistribution
    posterior: BeliefDistribution
    observation: EncodedObservation
    surprise: float
    action_efe: Optional[np.ndarray] = None
    recommended_action: Optional[ActionType] = None

    @property
    def belief_changed_significantly(self) -> bool:
        """Whether the update changed beliefs meaningfully."""
        return self.surprise > SIGNIFICANT_SURPRISE_THRESHOLD

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "prior_entropy": self.prior.entropy,
            "posterior_entropy": self.posterior.entropy,
            "surprise": self.surprise,
            "observation": self.observation.to_dict(),
            "recommended_action": (
                self.recommended_action.name if self.recommended_action else None
            ),
            "posterior_mode": self.posterior.get_mode(),
        }


class BeliefUpdater:
    """
    Updates beliefs using Bayesian inference with pymdp-compatible matrices.

    This is the inference engine of the Weaver-taking observations and
    updating the probability distribution over hidden states.

    The update process:
    1. Extract observation indices from EncodedObservation
    2. Use A matrix to compute likelihood: P(obs | state)
    3. Multiply prior by likelihood
    4. Normalize to get posterior

    Attributes:
        model: GenerativeModel containing A, B, C, D matrices
        learning_rate: How quickly to update beliefs (0-1)
    """

    def __init__(
        self,
        model: GenerativeModel,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ):
        """
        Initialize the belief updater.

        Args:
            model: GenerativeModel with A, B, C, D matrices
            learning_rate: Weight for new observations (0=ignore, 1=replace)
        """
        self.model = model
        self.learning_rate = learning_rate

        # Ensure model matrices are built
        if self.model.A is None:
            self.model.build_matrices()

        # Define which state factor each observation modality depends on
        # This mapping is derived from the generative model structure:
        # - NoteType depends on SeedState (factor 3)
        # - GraphConnectivity depends on SeedState (factor 3)
        # - UserBehavior depends on CognitiveMode (factor 2)
        # - UncertaintyLevel depends on SeedState (factor 3)
        self.A_factor_dependencies = {
            0: 3,  # NoteType -> SeedState
            1: 3,  # GraphConnectivity -> SeedState
            2: 2,  # UserBehavior -> CognitiveMode
            3: 3,  # UncertaintyLevel -> SeedState
        }

    def update(
        self,
        current_beliefs: BeliefDistribution,
        observation: EncodedObservation,
        compute_actions: bool = True,
    ) -> UpdateResult:
        """
        Update beliefs given a new observation.

        This implements Bayesian belief updating:
            posterior ~ likelihood x prior

        Args:
            current_beliefs: Current belief distribution
            observation: New observation to incorporate
            compute_actions: Whether to compute EFE for actions

        Returns:
            UpdateResult with prior, posterior, surprise, and recommendations
        """
        # Get observation indices
        obs_indices = observation.to_indices()

        # Compute posterior for each state factor
        posterior_factors = []
        total_surprise = 0.0

        for factor_idx, prior in enumerate(current_beliefs.factors):
            # Get likelihood from A matrix
            likelihood = self._get_likelihood(factor_idx, obs_indices)

            # Bayes update: posterior ~ likelihood x prior
            unnormalized = likelihood * prior

            # Normalize
            posterior = unnormalized / (unnormalized.sum() + 1e-10)

            # Blend with prior using learning rate
            blended = (
                self.learning_rate * posterior +
                (1 - self.learning_rate) * prior
            )

            # SDFT: High-confidence beliefs resist change more strongly
            # Confidence is the max probability in the prior (how certain we were)
            belief_confidence = float(np.max(prior))
            if belief_confidence > 0.5:  # Only apply resistance for confident beliefs
                # Max 30% reduction in learning for highly confident beliefs
                resistance = (belief_confidence - 0.5) * 0.6  # Maps 0.5-1.0 to 0.0-0.3
                blended = (1 - resistance) * blended + resistance * prior

            # Renormalize after blending
            blended = blended / blended.sum()

            posterior_factors.append(blended)

            # Compute surprise (KL divergence from prior to posterior)
            surprise = self._compute_kl_divergence(prior, blended)
            total_surprise += surprise

        # Create posterior distribution
        posterior = BeliefDistribution(
            factors=posterior_factors,
            observation_count=current_beliefs.observation_count + 1,
        )

        # Compute action EFE if requested
        action_efe = None
        recommended_action = None
        if compute_actions:
            action_efe = self.compute_action_efe(posterior)
            recommended_action = ActionType(int(np.argmin(action_efe)))

        return UpdateResult(
            prior=current_beliefs,
            posterior=posterior,
            observation=observation,
            surprise=total_surprise,
            action_efe=action_efe,
            recommended_action=recommended_action,
        )

    def _get_likelihood(
        self,
        factor_idx: int,
        obs_indices: list[int],
    ) -> np.ndarray:
        """
        Get observation likelihood from A matrices.

        For each state factor, we compute the likelihood of the observation
        given each possible state value. This is conditioned on the known
        dependencies between observation modalities and state factors.
        """
        # Get state factor dimension
        factor_dims = self.model.state_space.num_states
        state_dim = factor_dims[factor_idx]

        # Initialize uniform likelihood
        likelihood = np.ones(state_dim)

        # Combine likelihoods from each observation modality
        for modality_idx, obs_idx in enumerate(obs_indices):
            # Check if this modality depends on the current state factor
            dependency_factor = self.A_factor_dependencies.get(modality_idx)

            if factor_idx == dependency_factor:
                # This modality is informative for the current factor
                A = self.model.A[modality_idx]
                marginal = A[obs_idx, :]
            else:
                # This modality is not informative, assume uniform likelihood
                marginal = np.ones(state_dim)

            # Multiply likelihoods (assuming independence)
            likelihood *= marginal

        # Normalize to avoid numerical issues
        likelihood = likelihood / (likelihood.sum() + 1e-10)

        return likelihood

    def _compute_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """
        Compute KL divergence D_KL(q || p).

        This measures "surprise"-how much q differs from p.
        """
        p_safe = np.clip(p, 1e-10, 1.0)
        q_safe = np.clip(q, 1e-10, 1.0)
        return float(np.sum(q_safe * np.log(q_safe / p_safe)))

    def compute_action_efe(
        self,
        beliefs: BeliefDistribution,
    ) -> np.ndarray:
        """
        Compute Expected Free Energy for each action.

        EFE = Risk + Ambiguity - Epistemic Value

        Lower EFE = better action. The Weaver selects the action that
        minimizes expected surprise while maximizing information gain.

        Args:
            beliefs: Current belief distribution

        Returns:
            Array of EFE values, one per action
        """
        n_actions = self.model.action_space.num_actions
        efe = np.zeros(n_actions)

        for action_idx in range(n_actions):
            action = ActionType(action_idx)

            # Risk: Expected cost from preferences (C matrix)
            risk = self._compute_risk(beliefs, action)

            # Ambiguity: Expected uncertainty in observations
            ambiguity = self._compute_ambiguity(beliefs, action)

            # Epistemic value: Expected information gain
            epistemic = self._compute_epistemic_value(beliefs, action)

            efe[action_idx] = risk + ambiguity - epistemic

        return efe

    def _compute_risk(
        self,
        beliefs: BeliefDistribution,
        action: ActionType,
    ) -> float:
        """
        Compute risk (expected cost based on preferences).

        Risk = -E[C(o)] where expectation is over predicted observations
        based on current beliefs.

        Args:
            beliefs: Current belief distributions over each state factor.
            action: Action being evaluated (for action-specific risk modulation).

        Returns:
            Risk value (negative = good, positive = bad outcomes expected).
        """
        total_risk = 0.0

        for modality_idx, c_vector in enumerate(self.model.C):
            # Get the state factor this modality depends on
            factor_idx = self.A_factor_dependencies.get(modality_idx)

            if factor_idx is not None and factor_idx < len(beliefs.factors):
                # Get the A matrix for this modality
                A = self.model.A[modality_idx]
                # Compute predicted observation distribution: P(o) = A @ P(s)
                predicted_obs = A @ beliefs.factors[factor_idx]
                # Expected cost: E[C(o)] = sum(P(o) * C(o))
                expected_cost = np.dot(predicted_obs, c_vector)
            else:
                # Fallback: uniform expectation
                expected_cost = np.mean(c_vector)

            total_risk += -expected_cost

        # Normalize by number of modalities
        total_risk = total_risk / len(self.model.C) if self.model.C else 0.0

        # Actions with high epistemic value should reduce risk
        if action in [
            ActionType.ASK_CLARIFICATION,
            ActionType.TRIGGER_PREDICTION,
            ActionType.BRIDGE_CLUSTERS,
        ]:
            total_risk *= EPISTEMIC_ACTION_RISK_MULTIPLIER

        return total_risk

    def _compute_ambiguity(
        self,
        beliefs: BeliefDistribution,
        action: ActionType,
    ) -> float:
        """
        Compute ambiguity term: expected observation uncertainty.

        Ambiguity measures how uncertain observations will be given beliefs.
        High ambiguity means we can't predict what we'll observe.
        """
        # Use belief entropy as proxy for ambiguity
        base_ambiguity = beliefs.normalized_entropy

        # Apply action-specific multiplier from lookup table
        multiplier = AMBIGUITY_MULTIPLIERS.get(action, AMBIGUITY_DEFAULT_MULTIPLIER)
        return base_ambiguity * multiplier

    def _compute_epistemic_value(
        self,
        beliefs: BeliefDistribution,
        action: ActionType,
    ) -> float:
        """
        Compute epistemic value: expected information gain.

        Epistemic value measures how much we expect to learn from an action.
        High value = action reveals hidden state information.
        """
        # Base epistemic value from belief uncertainty
        base_value = beliefs.normalized_entropy

        # Apply action-specific multiplier from module-level lookup table
        multiplier = EPISTEMIC_VALUE_MULTIPLIERS.get(action, EPISTEMIC_VALUE_DEFAULT)
        return base_value * multiplier


def create_initial_beliefs(model: GenerativeModel) -> BeliefDistribution:
    """
    Create initial beliefs from the generative model's D priors.

    Usage:
        beliefs = create_initial_beliefs(model)
        updater = BeliefUpdater(model)
        result = updater.update(beliefs, observation)
    """
    return BeliefDistribution.from_prior(model.D)
