"""
Policy Selector: EFE-Based Action Selection for the Weaver.

This module implements the Weaver's decision-making core-selecting actions
by computing Expected Free Energy (EFE) and choosing the action that minimizes it.

Cognitive Science Background:
    In Active Inference, agents don't maximize reward-they minimize surprise.
    This is operationalized via Expected Free Energy:

    EFE = Risk + Ambiguity - Epistemic Value

    Where:
    - Risk: Expected cost of outcomes (from preferences C matrix)
    - Ambiguity: Uncertainty about observations given states
    - Epistemic Value: Expected information gain

    Key insight: Under high uncertainty, epistemic actions (questions,
    predictions) naturally dominate because they have high epistemic value.
    Under low uncertainty, pragmatic actions (linking, surfacing) dominate.

Usage:
    from core.active_inference import PolicySelector

    selector = PolicySelector(generative_model)
    result = selector.select_action(
        beliefs=current_beliefs,
        observation=encoded_obs,
        context=action_context,
    )

    print(f"Selected: {result.action.name} with EFE={result.efe:.3f}")
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
import numpy as np

from core.active_inference.generative_model import ActionType


if TYPE_CHECKING:
    from core.active_inference.belief_updater import BeliefDistribution, BeliefUpdater

# Type aliases for optional context types (not migrated)
# These are used in TYPE_CHECKING for documentation but won't exist at runtime
OrphanAnalysis = Any
StructuralHole = Any
BridgeProposal = Any

logger = logging.getLogger(__name__)

# Policy selection constants
# EFE adjustment magnitudes for context-based action boosting/penalizing
EFE_BOOST_CRITICAL_ORPHANS_SURFACE = 0.3  # Boost SURFACE_SEED for critical orphans
EFE_BOOST_CRITICAL_ORPHANS_LINK = 0.2  # Boost SUGGEST_LINK for critical orphans
EFE_BOOST_BRIDGING_CLUSTERS = 0.4  # Boost BRIDGE_CLUSTERS when holes exist
EFE_BOOST_BRIDGING_LINK = 0.2  # Boost SUGGEST_LINK for bridging opportunities
EFE_PENALTY_USER_NOT_ENGAGED = 0.5  # Penalty for interactive actions when user idle
EFE_PENALTY_RECENT_ACTION = 0.2  # Penalty for recently used actions
EFE_BOOST_MORNING_HUB_REVIEW = 0.2  # Morning boost for HUB_REVIEW
EFE_BOOST_MORNING_SURFACE = 0.1  # Morning boost for SURFACE_SEED
EFE_BOOST_HIGH_ENTROPY_CLARIFICATION = 0.3  # High entropy boost for ASK_CLARIFICATION
EFE_BOOST_HIGH_ENTROPY_PREDICTION = 0.2  # High entropy boost for TRIGGER_PREDICTION

# Thresholds
HIGH_ENTROPY_THRESHOLD = 0.7  # Normalized entropy above which epistemic actions are boosted
RECENT_ACTIONS_WINDOW = 3  # Number of recent actions to consider for repetition penalty
CONFIDENCE_GAP_THRESHOLD = 0.5  # Gap between best and second-best EFE for full confidence


@dataclass
class ActionContext:
    """
    Context that influences action selection.

    The context provides additional information beyond beliefs that should
    affect which action is chosen.

    Attributes:
        orphan_analysis: Current state of orphan seeds
        structural_holes: Detected gaps between clusters
        bridge_proposals: Pre-computed bridge proposals
        user_engaged: Whether user is actively interacting
        time_of_day: For circadian-aware suggestions
        recent_actions: Actions taken recently (avoid repetition)
    """

    orphan_analysis: Optional[Any] = None  # OrphanAnalysis not migrated
    structural_holes: Optional[list[Any]] = None  # StructuralHole not migrated
    bridge_proposals: Optional[list[Any]] = None  # BridgeProposal not migrated
    user_engaged: bool = True
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    recent_actions: list[ActionType] = field(default_factory=list)

    @property
    def has_critical_orphans(self) -> bool:
        """Whether there are orphans needing immediate attention."""
        if self.orphan_analysis is None:
            return False
        # Check for critical_count attribute if OrphanAnalysis-like object
        return getattr(self.orphan_analysis, 'critical_count', 0) > 0

    @property
    def has_bridging_opportunities(self) -> bool:
        """Whether there are structural holes to bridge."""
        holes_exist = bool(
            self.structural_holes and any(
                getattr(h, 'bridge_priority', 0) > 0.5 for h in self.structural_holes
            )
        )
        proposals_exist = bool(
            self.bridge_proposals and any(
                getattr(p, 'priority', 0) > 0.5 for p in self.bridge_proposals
            )
        )
        return holes_exist or proposals_exist


@dataclass
class PolicyResult:
    """
    Result of policy selection.

    Attributes:
        action: The selected action
        efe: Expected Free Energy of the selected action
        all_efe: EFE values for all actions (for debugging)
        confidence: How confident we are in this selection
        rationale: Human-readable explanation
        context_used: Which context factors influenced selection
    """

    action: ActionType
    efe: float
    all_efe: dict[ActionType, float]
    confidence: float
    rationale: str
    context_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "action": self.action.name,
            "efe": self.efe,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "context_used": self.context_used,
            "all_efe": {k.name: v for k, v in self.all_efe.items()},
        }


class PolicySelector:
    """
    Selects actions by minimizing Expected Free Energy.

    The PolicySelector is the Weaver's decision-making core. It evaluates
    all possible actions and selects the one that best balances:
    - Pragmatic value (achieving preferred outcomes)
    - Epistemic value (gaining information)
    - Context appropriateness (matching the situation)

    The key insight: the EFE math naturally produces "intelligent" behavior.
    Under uncertainty, the agent seeks information. Under confidence, it acts.

    Attributes:
        updater: BeliefUpdater for EFE computation
        temperature: Softmax temperature for action selection (lower = greedier)
    """

    def __init__(
        self,
        updater: "BeliefUpdater",
        temperature: float = 1.0,
    ):
        """
        Initialize the PolicySelector.

        Args:
            updater: BeliefUpdater for EFE computation
            temperature: Softmax temperature (1.0 = balanced, 0.1 = greedy)
        """
        self.updater = updater
        self.temperature = temperature

    def select_action(
        self,
        beliefs: "BeliefDistribution",
        context: Optional[ActionContext] = None,
        allowed_actions: Optional[list[ActionType]] = None,
    ) -> PolicyResult:
        """
        Select the best action given current beliefs and context.

        Computes EFE for each action and selects the one with lowest EFE.
        Context factors can modify EFE values to make certain actions
        more or less appropriate.

        Args:
            beliefs: Current belief distribution
            context: Optional context for situation-aware selection
            allowed_actions: Optional subset of actions to consider

        Returns:
            PolicyResult with selected action and rationale
        """
        context = context or ActionContext()

        # Compute base EFE for all actions
        efe_array = self.updater.compute_action_efe(beliefs)

        # Convert to dict for easier manipulation
        all_efe = {
            ActionType(i): efe_array[i]
            for i in range(len(efe_array))
        }

        # Apply context-based adjustments
        adjusted_efe = self._apply_context_adjustments(all_efe, context, beliefs)

        # Filter to allowed actions if specified
        if allowed_actions is not None:
            adjusted_efe = {
                k: v for k, v in adjusted_efe.items()
                if k in allowed_actions
            }

        # Select action with lowest EFE
        if not adjusted_efe:
            # Fallback to DO_NOTHING
            return PolicyResult(
                action=ActionType.DO_NOTHING,
                efe=0.0,
                all_efe=all_efe,
                confidence=0.0,
                rationale="No valid actions available",
            )

        best_action = min(adjusted_efe, key=adjusted_efe.get)
        best_efe = adjusted_efe[best_action]

        # Compute confidence (how much better is best vs. second best)
        confidence = self._compute_confidence(adjusted_efe, best_action)

        # Generate rationale
        rationale, context_used = self._generate_rationale(
            best_action, beliefs, context
        )

        result = PolicyResult(
            action=best_action,
            efe=best_efe,
            all_efe=all_efe,
            confidence=confidence,
            rationale=rationale,
            context_used=context_used,
        )

        logger.debug(
            f"Policy selected: {best_action.name} "
            f"(EFE={best_efe:.3f}, confidence={confidence:.2f})"
        )

        return result

    def _apply_context_adjustments(
        self,
        efe: dict[ActionType, float],
        context: ActionContext,
        beliefs: "BeliefDistribution",
    ) -> dict[ActionType, float]:
        """
        Adjust EFE values based on context.

        Context factors can make certain actions more or less appropriate:
        - Critical orphans -> boost SURFACE_SEED
        - Structural holes -> boost BRIDGE_CLUSTERS
        - User not engaged -> penalize interactive actions
        - Recent action repetition -> penalize same action
        """
        adjusted = efe.copy()

        # Boost actions for critical orphans
        if context.has_critical_orphans:
            adjusted[ActionType.SURFACE_SEED] -= EFE_BOOST_CRITICAL_ORPHANS_SURFACE
            adjusted[ActionType.SUGGEST_LINK] -= EFE_BOOST_CRITICAL_ORPHANS_LINK

        # Boost bridging when holes exist
        if context.has_bridging_opportunities:
            adjusted[ActionType.BRIDGE_CLUSTERS] -= EFE_BOOST_BRIDGING_CLUSTERS
            adjusted[ActionType.SUGGEST_LINK] -= EFE_BOOST_BRIDGING_LINK

        # Penalize interactive actions when user not engaged
        if not context.user_engaged:
            interactive_actions = [
                ActionType.ASK_CLARIFICATION,
                ActionType.SUGGEST_LINK,
                ActionType.TRIGGER_PREDICTION,
            ]
            for action in interactive_actions:
                if action in adjusted:
                    adjusted[action] += EFE_PENALTY_USER_NOT_ENGAGED

        # Penalize recently used actions (avoid repetition)
        for recent in context.recent_actions[-RECENT_ACTIONS_WINDOW:]:
            if recent in adjusted:
                adjusted[recent] += EFE_PENALTY_RECENT_ACTION

        # Morning boost for review actions
        if context.time_of_day == "morning":
            adjusted[ActionType.HUB_REVIEW] -= EFE_BOOST_MORNING_HUB_REVIEW
            adjusted[ActionType.SURFACE_SEED] -= EFE_BOOST_MORNING_SURFACE

        # High entropy boost for epistemic actions
        if beliefs.normalized_entropy > HIGH_ENTROPY_THRESHOLD:
            adjusted[ActionType.ASK_CLARIFICATION] -= EFE_BOOST_HIGH_ENTROPY_CLARIFICATION
            adjusted[ActionType.TRIGGER_PREDICTION] -= EFE_BOOST_HIGH_ENTROPY_PREDICTION

        return adjusted

    def _compute_confidence(
        self,
        efe: dict[ActionType, float],
        best_action: ActionType,
    ) -> float:
        """
        Compute confidence in the selected action.

        Confidence is based on how much better the best action is compared
        to alternatives. If all actions have similar EFE, confidence is low.
        """
        if len(efe) < 2:
            return 1.0

        efe_values = list(efe.values())
        best_efe = efe[best_action]

        # Find second-best EFE
        sorted_efe = sorted(efe_values)
        second_best = sorted_efe[1] if len(sorted_efe) > 1 else sorted_efe[0]

        # Confidence based on gap between best and second-best
        gap = second_best - best_efe
        confidence = min(1.0, gap / CONFIDENCE_GAP_THRESHOLD)

        return max(0.1, confidence)

    def _generate_rationale(
        self,
        action: ActionType,
        beliefs: "BeliefDistribution",
        context: ActionContext,
    ) -> tuple[str, list[str]]:
        """Generate human-readable rationale for action selection."""
        context_used = []
        reasons = []

        # Base rationale from action type
        action_rationales = {
            ActionType.DO_NOTHING: "No intervention needed at this time.",
            ActionType.SILENT_LINK: "Silently connecting related fragments in the background.",
            ActionType.SUGGEST_LINK: "Proposing a connection for your consideration.",
            ActionType.ASK_CLARIFICATION: "Seeking clarification to reduce uncertainty.",
            ActionType.SURFACE_SEED: "Bringing up a disconnected idea that may be relevant.",
            ActionType.TRIGGER_PREDICTION: "Inviting you to predict before revealing-this strengthens learning.",
            ActionType.BRIDGE_CLUSTERS: "Proposing a bridge between isolated knowledge clusters.",
            ActionType.HUB_REVIEW: "Recommending review of a central concept.",
        }

        base_rationale = action_rationales.get(action, "Taking action.")
        reasons.append(base_rationale)

        # Add context-specific reasoning
        if context.has_critical_orphans and action in [
            ActionType.SURFACE_SEED,
            ActionType.SUGGEST_LINK,
        ]:
            reasons.append("Critical orphan seeds need attention.")
            context_used.append("orphan_analysis")

        if context.has_bridging_opportunities and action == ActionType.BRIDGE_CLUSTERS:
            reasons.append("Structural holes detected between clusters.")
            context_used.append("structural_holes")

        if beliefs.normalized_entropy > HIGH_ENTROPY_THRESHOLD and action in [
            ActionType.ASK_CLARIFICATION,
            ActionType.TRIGGER_PREDICTION,
        ]:
            reasons.append("High uncertainty makes epistemic actions valuable.")
            context_used.append("belief_entropy")

        if context.time_of_day == "morning" and action == ActionType.HUB_REVIEW:
            reasons.append("Morning is optimal for reviewing key concepts.")
            context_used.append("time_of_day")

        return " ".join(reasons), context_used

    def sample_action(
        self,
        beliefs: "BeliefDistribution",
        context: Optional[ActionContext] = None,
    ) -> ActionType:
        """
        Sample an action using softmax over negative EFE.

        Instead of always choosing the best action, this samples from a
        distribution proportional to exp(-EFE/temperature). This allows
        for exploration and prevents getting stuck in local optima.

        Args:
            beliefs: Current belief distribution
            context: Optional context

        Returns:
            Sampled ActionType
        """
        context = context or ActionContext()

        # Compute EFE
        efe_array = self.updater.compute_action_efe(beliefs)

        # Convert to probabilities via softmax over negative EFE
        # (negative because lower EFE = better)
        neg_efe = -efe_array / self.temperature
        exp_neg_efe = np.exp(neg_efe - np.max(neg_efe))  # Numerical stability
        probs = exp_neg_efe / exp_neg_efe.sum()

        # Sample action
        action_idx = np.random.choice(len(probs), p=probs)
        return ActionType(action_idx)

    def get_action_probabilities(
        self,
        beliefs: "BeliefDistribution",
        context: Optional[ActionContext] = None,
    ) -> dict[ActionType, float]:
        """
        Get probability distribution over actions.

        Useful for visualization or debugging.

        Args:
            beliefs: Current belief distribution
            context: Optional context

        Returns:
            Dict mapping ActionType to probability
        """
        efe_array = self.updater.compute_action_efe(beliefs)

        # Softmax
        neg_efe = -efe_array / self.temperature
        exp_neg_efe = np.exp(neg_efe - np.max(neg_efe))
        probs = exp_neg_efe / exp_neg_efe.sum()

        return {ActionType(i): probs[i] for i in range(len(probs))}


def select_weaver_action(
    beliefs: "BeliefDistribution",
    updater: "BeliefUpdater",
    context: Optional[ActionContext] = None,
    temperature: float = 1.0,
) -> PolicyResult:
    """
    Convenience function to select an action.

    Usage:
        from core.active_inference import select_weaver_action

        result = select_weaver_action(
            beliefs=current_beliefs,
            updater=belief_updater,
            context=ActionContext(orphan_analysis=analysis),
        )

        print(f"Do: {result.action.name}")
    """
    selector = PolicySelector(updater, temperature)
    return selector.select_action(beliefs, context)
