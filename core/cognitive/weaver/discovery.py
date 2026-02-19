"""Discovery Parameter: The Homeostatic Setpoint of Cognitive Cultivation.

This module implements the Critical Discovery Parameter, a single scalar that
captures the tension between semantic diversity and structural connectivity.
It serves as the primary sensory input for the Weaver's policy selection.

Mathematical Definition:
    D = H_sem - H_struct

    Where:
    - H_sem: Semantic entropy (diversity of ideas/embeddings)
    - H_struct: Structural entropy (connectivity of the graph)

Cognitive Science Background:
    In Active Inference, agents maintain homeostasis by minimizing deviation
    from a "preferred state." For the knowledge graph, the preferred state is
    D ~ 0 (Isomorphism): structure matches meaning.

    Deviations from this setpoint trigger corrective actions:

    | State | D Value | Diagnosis | Weaver Action |
    |-------|---------|-----------|---------------|
    | Exploration | D > 0, small | Gathering ideas | WAIT |
    | Tension | D >> 0 | Ideas outpacing connections | BRIDGE |
    | Stagnation | D << 0 | Over-structuring | PROMPT |
    | Isomorphism | D ~ 0 | Balanced | MONITOR |

Reference:
    Based on "Self-Organizing Graph Reasoning Evolves into a Critical State
    for Continuous Discovery Through Structural-Semantic Dynamics" (arXiv:2503.18852)
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

from core.active_inference.graph_entropy import EntropyResult, GraphEntropy
from core.active_inference.semantic_entropy import (
    SemanticEntropyCalculator,
    SemanticEntropyResult,
)

logger = logging.getLogger(__name__)


class DiscoveryState(Enum):
    """Cognitive state of the knowledge graph based on D.

    Each state suggests a different Weaver policy:
    - EXPLORATION: Passive monitoring, let ideas accumulate
    - TENSION: Active bridging to release semantic pressure
    - STAGNATION: Prompt for new ideas to break structural rigidity
    - ISOMORPHISM: Balanced state, light maintenance
    - UNKNOWN: Insufficient data for assessment
    """

    EXPLORATION = "exploration"  # D > 0, small: gathering ideas
    TENSION = "tension"  # D >> 0: semantic pressure building
    STAGNATION = "stagnation"  # D << 0: over-structured bureaucracy
    ISOMORPHISM = "isomorphism"  # D ~ 0: balanced flow state
    UNKNOWN = "unknown"  # Insufficient data


class WeaverAction(Enum):
    """Recommended action based on Discovery Parameter state.

    Maps directly from DiscoveryState to action policy:
    - WAIT: Let ideas accumulate without intervention
    - BRIDGE: Propose connections to release semantic pressure
    - PROMPT: Inject new semantic variance through questions
    - MONITOR: Light-touch maintenance, observe
    - ASSESS: Need more data before acting
    """

    WAIT = "wait"
    BRIDGE = "bridge"
    PROMPT = "prompt"
    MONITOR = "monitor"
    ASSESS = "assess"


@dataclass
class DiscoveryResult:
    """Result of Discovery Parameter computation.

    Attributes:
        discovery_parameter: The D value (H_sem - H_struct)
        semantic_entropy: The H_sem component
        structural_entropy: The H_struct component (total_entropy from GraphEntropy)
        state: Current cognitive state of the graph
        recommended_action: Suggested Weaver action
        confidence: Confidence in the assessment (0-1)
        pressure_direction: "semantic" or "structural" or "balanced"
        tension_magnitude: Absolute value of D (strength of imbalance)
        semantic_result: Full semantic entropy result
        structural_result: Full structural entropy result
        explanation: Human-readable explanation of the state
        unresolved_conflicts: Count of unresolved belief contradictions
    """

    discovery_parameter: float
    semantic_entropy: float
    structural_entropy: float
    state: DiscoveryState
    recommended_action: WeaverAction
    confidence: float
    pressure_direction: str
    tension_magnitude: float
    semantic_result: SemanticEntropyResult
    structural_result: EntropyResult
    explanation: str
    unresolved_conflicts: int = 0

    @property
    def is_balanced(self) -> bool:
        """Graph is in isomorphic state."""
        return self.state == DiscoveryState.ISOMORPHISM

    @property
    def needs_bridging(self) -> bool:
        """Semantic pressure requires bridging action."""
        return self.state == DiscoveryState.TENSION

    @property
    def needs_prompting(self) -> bool:
        """Structural rigidity requires new ideas."""
        return self.state == DiscoveryState.STAGNATION

    @property
    def is_accumulating(self) -> bool:
        """Ideas are being gathered, allow accumulation."""
        return self.state == DiscoveryState.EXPLORATION

    def to_dict(self) -> dict:
        """Serialize for logging, persistence, and API responses."""
        return {
            "discovery_parameter": self.discovery_parameter,
            "semantic_entropy": self.semantic_entropy,
            "structural_entropy": self.structural_entropy,
            "state": self.state.value,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
            "pressure_direction": self.pressure_direction,
            "tension_magnitude": self.tension_magnitude,
            "explanation": self.explanation,
            "unresolved_conflicts": self.unresolved_conflicts,
            "semantic": self.semantic_result.to_dict(),
            "structural": self.structural_result.to_dict(),
        }

    @classmethod
    def for_error(cls, explanation: str = "Failed to compute discovery parameter") -> "DiscoveryResult":
        """Create a neutral/unknown state DiscoveryResult for error cases.

        Use this when discovery computation fails and a fallback result is needed.

        Args:
            explanation: Human-readable explanation of the error

        Returns:
            DiscoveryResult with neutral values and UNKNOWN state
        """
        return cls(
            discovery_parameter=0.0,
            semantic_entropy=0.0,
            structural_entropy=0.0,
            state=DiscoveryState.UNKNOWN,
            recommended_action=WeaverAction.ASSESS,
            confidence=0.3,
            pressure_direction="balanced",
            tension_magnitude=0.0,
            semantic_result=SemanticEntropyResult(
                semantic_entropy=0.0,
                topic_concentration=1.0,
                effective_dimensions=1.0,
                embedding_count=0,
                mean_similarity=1.0,
                min_similarity=1.0,
                max_similarity=1.0,
                is_diverse=False,
                diversity_reason="Insufficient data",
            ),
            structural_result=EntropyResult(
                structural_entropy=0.0,
                cluster_entropy=0.0,
                orphan_rate=0.0,
                hub_concentration=0.0,
                total_entropy=0.0,
                node_count=0,
                edge_count=0,
                cluster_count=1,
                orphan_count=0,
                should_cultivate=False,
                cultivation_reason="Insufficient data for assessment",
            ),
            explanation=explanation,
        )


# Thresholds for Discovery Parameter interpretation
TENSION_THRESHOLD = 0.2  # D > 0.2 -> TENSION (need bridges)
EXPLORATION_THRESHOLD = 0.1  # 0 < D < 0.2 -> EXPLORATION (wait)
BALANCE_THRESHOLD = 0.1  # |D| < 0.1 -> ISOMORPHISM (balanced)
STAGNATION_THRESHOLD = -0.2  # D < -0.2 -> STAGNATION (need prompts)

# Minimum data requirements for reliable assessment
MIN_NODES_FOR_ASSESSMENT = 5
MIN_EMBEDDINGS_FOR_ASSESSMENT = 3


class DiscoveryParameter:
    """Computes the Discovery Parameter (D) for the knowledge graph.

    The Discovery Parameter is the difference between semantic entropy
    (diversity of ideas) and structural entropy (connectivity of graph).
    It serves as the homeostatic setpoint for the Weaver.

    This class combines:
    - SemanticEntropyCalculator: Measures meaning diversity
    - GraphEntropy: Measures structural fragmentation

    And produces a single scalar (D) plus interpretive guidance.

    Attributes:
        graph: PsycheClient for queries
        semantic_calculator: Calculator for semantic entropy
        structural_calculator: Calculator for structural entropy
    """

    def __init__(
        self,
        graph: "PsycheClient",
        max_embeddings: int = 500,
    ):
        """Initialize the Discovery Parameter calculator.

        Args:
            graph: PsycheClient instance
            max_embeddings: Maximum embeddings for semantic analysis
        """
        self.graph = graph
        self.semantic_calculator = SemanticEntropyCalculator(
            graph, max_embeddings=max_embeddings
        )
        self.structural_calculator = GraphEntropy(graph)

    async def compute(
        self,
        node_uids: Optional[list[str]] = None,
        tenant_id: str = "default",
    ) -> DiscoveryResult:
        """Compute the Discovery Parameter for the knowledge graph.

        This runs both semantic and structural entropy calculations
        concurrently, then combines them into the Discovery Parameter.

        Args:
            node_uids: Optional specific nodes to analyze (for semantic)
            tenant_id: Tenant identifier for graph operations

        Returns:
            DiscoveryResult with D, state, and recommendations
        """
        # Run both entropy calculations concurrently
        semantic_task = self.semantic_calculator.compute(node_uids)
        structural_task = self.structural_calculator.compute(tenant_id)

        semantic_result, structural_result = await asyncio.gather(
            semantic_task, structural_task
        )

        # Compute the Discovery Parameter
        semantic_entropy = semantic_result.semantic_entropy
        structural_entropy = structural_result.total_entropy

        # D = H_sem - H_struct
        discovery_parameter = semantic_entropy - structural_entropy

        # Determine state and confidence
        state, confidence = self._determine_state(
            discovery_parameter,
            semantic_result,
            structural_result,
        )

        # Map state to recommended action
        recommended_action = self._get_recommended_action(state)

        # Determine pressure direction
        if discovery_parameter > BALANCE_THRESHOLD:
            pressure_direction = "semantic"
        elif discovery_parameter < -BALANCE_THRESHOLD:
            pressure_direction = "structural"
        else:
            pressure_direction = "balanced"

        # Generate explanation
        explanation = self._generate_explanation(
            discovery_parameter,
            state,
            semantic_result,
            structural_result,
        )

        return DiscoveryResult(
            discovery_parameter=discovery_parameter,
            semantic_entropy=semantic_entropy,
            structural_entropy=structural_entropy,
            state=state,
            recommended_action=recommended_action,
            confidence=confidence,
            pressure_direction=pressure_direction,
            tension_magnitude=abs(discovery_parameter),
            semantic_result=semantic_result,
            structural_result=structural_result,
            explanation=explanation,
        )

    def _determine_state(
        self,
        discovery_parameter: float,
        semantic_result: SemanticEntropyResult,
        structural_result: EntropyResult,
    ) -> tuple[DiscoveryState, float]:
        """Determine cognitive state from Discovery Parameter.

        Returns:
            Tuple of (state, confidence)
        """
        # Check if we have enough data
        node_count = structural_result.node_count
        embedding_count = semantic_result.embedding_count

        if node_count < MIN_NODES_FOR_ASSESSMENT:
            return DiscoveryState.UNKNOWN, 0.3

        if embedding_count < MIN_EMBEDDINGS_FOR_ASSESSMENT:
            # Can only assess structural, not semantic
            return DiscoveryState.UNKNOWN, 0.4

        # Determine state based on D thresholds
        if discovery_parameter > TENSION_THRESHOLD:
            # High semantic, low structural -> TENSION
            confidence = min(1.0, 0.6 + (discovery_parameter - TENSION_THRESHOLD))
            return DiscoveryState.TENSION, confidence

        if discovery_parameter > EXPLORATION_THRESHOLD:
            # Moderate semantic excess -> EXPLORATION
            confidence = 0.7
            return DiscoveryState.EXPLORATION, confidence

        if discovery_parameter < STAGNATION_THRESHOLD:
            # High structural, low semantic -> STAGNATION
            confidence = min(
                1.0, 0.6 + abs(discovery_parameter + STAGNATION_THRESHOLD)
            )
            return DiscoveryState.STAGNATION, confidence

        # Near zero -> ISOMORPHISM
        balance_quality = 1.0 - abs(discovery_parameter) / BALANCE_THRESHOLD
        confidence = 0.6 + 0.3 * balance_quality
        return DiscoveryState.ISOMORPHISM, confidence

    def _get_recommended_action(self, state: DiscoveryState) -> WeaverAction:
        """Map cognitive state to recommended Weaver action."""
        action_map = {
            DiscoveryState.EXPLORATION: WeaverAction.WAIT,
            DiscoveryState.TENSION: WeaverAction.BRIDGE,
            DiscoveryState.STAGNATION: WeaverAction.PROMPT,
            DiscoveryState.ISOMORPHISM: WeaverAction.MONITOR,
            DiscoveryState.UNKNOWN: WeaverAction.ASSESS,
        }
        return action_map.get(state, WeaverAction.ASSESS)

    def _generate_explanation(
        self,
        discovery_parameter: float,
        state: DiscoveryState,
        semantic_result: SemanticEntropyResult,
        structural_result: EntropyResult,
    ) -> str:
        """Generate human-readable explanation of the Discovery state."""
        d_sign = "+" if discovery_parameter >= 0 else ""

        base = (
            f"D = {d_sign}{discovery_parameter:.3f} "
            f"(H_sem={semantic_result.semantic_entropy:.2f}, "
            f"H_struct={structural_result.total_entropy:.2f})"
        )

        if state == DiscoveryState.TENSION:
            return (
                f"{base}. Semantic pressure is high - ideas are accumulating "
                "faster than connections. Consider proposing bridges between clusters."
            )

        if state == DiscoveryState.EXPLORATION:
            return f"{base}. Ideas are gathering. Allow accumulation before intervening."

        if state == DiscoveryState.STAGNATION:
            return (
                f"{base}. Graph is over-structured relative to semantic diversity. "
                "Inject new ideas through prompts or questions."
            )

        if state == DiscoveryState.ISOMORPHISM:
            return (
                f"{base}. Balanced flow state - structure matches meaning. "
                "Light maintenance only."
            )

        return f"{base}. Insufficient data for reliable assessment."


async def compute_discovery_parameter(
    graph: "PsycheClient",
    node_uids: Optional[list[str]] = None,
    tenant_id: str = "default",
) -> DiscoveryResult:
    """Convenience function to compute the Discovery Parameter.

    Usage:
        from core.cognitive.weaver import compute_discovery_parameter

        result = await compute_discovery_parameter(graph)

        if result.needs_bridging:
            await bridge_clusters(...)
    """
    calculator = DiscoveryParameter(graph)
    return await calculator.compute(node_uids, tenant_id=tenant_id)


def interpret_discovery_state(result: DiscoveryResult) -> str:
    """Generate a conversational interpretation of the Discovery state.

    This is useful for user-facing explanations or logging.
    """
    if result.state == DiscoveryState.TENSION:
        return (
            f"Your knowledge graph has {result.semantic_result.embedding_count} "
            f"diverse ideas but they're not yet well-connected. "
            f"Would you like me to suggest some bridges between related concepts?"
        )

    if result.state == DiscoveryState.EXPLORATION:
        return (
            "You're in exploration mode - gathering ideas freely. "
            "I'll let these accumulate before suggesting connections."
        )

    if result.state == DiscoveryState.STAGNATION:
        return (
            "Your graph is well-structured but could use more semantic variety. "
            "What new domains or perspectives would you like to explore?"
        )

    if result.state == DiscoveryState.ISOMORPHISM:
        return (
            "Your knowledge graph is in a balanced state - ideas and connections "
            "are in harmony. I'll just do light maintenance."
        )

    return "I need more data to assess your knowledge graph's state."
