"""
Observation Encoder: Transforming Sensory Data into Discrete Observations.

This module encodes raw notes, graph context, and user behavior into the discrete
observation vectors that pymdp expects. It bridges System 1 (fast, continuous)
and System 2 (slow, discrete).

Cognitive Science Background:
    In Active Inference, observations are the sensory evidence that drives belief
    updating. The brain doesn't process raw sensory data directly-it categorizes
    inputs into discrete "symbols" that can be compared against predictions.

    For Lilly, observations come from three sources:
    1. The note content (what was written)
    2. The graph context (how connected the note is)
    3. The motor trace (how it was written)

    The encoder maps these continuous signals to discrete observation indices
    that can be used with pymdp's categorical inference.

Usage:
    from core.active_inference import ObservationEncoder

    encoder = ObservationEncoder(graph_client)
    observation = await encoder.encode_fragment(fragment, tenant_id)

    # observation.to_indices() returns [note_type_idx, connectivity_idx, behavior_idx, uncertainty_idx]
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.active_inference.generative_model import (
    NoteType,
    GraphConnectivity,
    UserBehavior,
    UncertaintyLevel,
)

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.models.fragment import Fragment

logger = logging.getLogger(__name__)


@dataclass
class EncodedObservation:
    """
    A discrete observation vector for pymdp belief updating.

    This is the "sensory evidence" that the Weaver uses to update its beliefs
    about the hidden state of the user's knowledge graph.

    Attributes:
        note_type: What kind of note was captured (fragment, question, statement, diagram)
        graph_connectivity: How connected this note is in the graph
        user_behavior: How the note was written (fast, deliberate, reviewing)
        uncertainty_level: How confident we are about this note
        raw_features: Dictionary of underlying continuous features for debugging
    """

    note_type: NoteType
    graph_connectivity: GraphConnectivity
    user_behavior: UserBehavior
    uncertainty_level: UncertaintyLevel
    raw_features: dict

    def to_indices(self) -> list[int]:
        """
        Convert to pymdp observation indices.

        Returns list of indices for each observation modality.
        """
        return [
            self.note_type.value,
            self.graph_connectivity.value,
            self.user_behavior.value,
            self.uncertainty_level.value,
        ]

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "note_type": self.note_type.name,
            "graph_connectivity": self.graph_connectivity.name,
            "user_behavior": self.user_behavior.name,
            "uncertainty_level": self.uncertainty_level.name,
            "indices": self.to_indices(),
            "raw_features": self.raw_features,
        }


# Thresholds for note type classification
QUESTION_THRESHOLD = 0.3  # 30% of content is question-like
STATEMENT_WORD_THRESHOLD = 10  # Statements have at least 10 words
DIAGRAM_KEYWORDS = {"->", "<-", "->", "<-", "--", "|", "|-", "|_", "|-", "|-"}
DIAGRAM_SCORE_THRESHOLD = 2  # Score threshold to classify as diagram note
UNCERTAINTY_MARKER_PENALTY = 0.1  # Penalty per uncertainty marker (?, maybe, etc.)

# Thresholds for graph connectivity
SPARSE_EDGE_THRESHOLD = 2
CONNECTED_EDGE_THRESHOLD = 5
HUB_EDGE_THRESHOLD = 10

# Thresholds for user behavior from motor trace
SLOW_VELOCITY_THRESHOLD = 0.3  # Fluency score < 0.3 = slow deliberate
FAST_VELOCITY_THRESHOLD = 0.7  # Fluency score > 0.7 = fast/automatic

# Thresholds for uncertainty
CONFIDENT_THRESHOLD = 0.8
MODERATE_THRESHOLD = 0.5
UNCERTAIN_THRESHOLD = 0.3


class ObservationEncoder:
    """
    Encodes fragments and context into discrete observations for pymdp.

    This is the "perception layer" of the Weaver-transforming raw sensory
    data (notes, graph topology, motor traces) into the categorical observations
    that drive belief updating.

    The encoding is intentionally lossy-we compress rich continuous features
    into 4-level discrete categories. This matches how the brain compresses
    sensory input into categorical percepts for efficient inference.

    Attributes:
        graph: PsycheClient for graph connectivity queries
    """

    def __init__(self, graph: Optional["PsycheClient"] = None):
        """
        Initialize the observation encoder.

        Args:
            graph: PsycheClient for connectivity queries (optional for testing)
        """
        self.graph = graph

    async def encode_fragment(
        self,
        fragment: "Fragment",
        tenant_id: str,
        edge_count: Optional[int] = None,
        has_conflicts: bool = False,
    ) -> EncodedObservation:
        """
        Encode a fragment into a discrete observation vector.

        Args:
            fragment: The fragment to encode
            tenant_id: Tenant identifier for graph queries
            edge_count: Pre-computed edge count (optional, will query if not provided)
            has_conflicts: Whether this fragment has detected contradictions

        Returns:
            EncodedObservation with discrete indices for each modality
        """
        # Extract raw features
        raw_features = {}

        # 1. Encode note type from content analysis
        note_type, note_features = self._encode_note_type(fragment)
        raw_features.update(note_features)

        # 2. Encode graph connectivity
        if edge_count is None and self.graph:
            edge_count = await self._get_edge_count(fragment.uid, tenant_id)
        elif edge_count is None:
            edge_count = 0

        connectivity, conn_features = self._encode_connectivity(edge_count)
        raw_features.update(conn_features)

        # 3. Encode user behavior from motor trace
        behavior, behavior_features = self._encode_user_behavior(fragment)
        raw_features.update(behavior_features)

        # 4. Encode uncertainty level
        uncertainty, uncertainty_features = self._encode_uncertainty(
            fragment, has_conflicts
        )
        raw_features.update(uncertainty_features)

        return EncodedObservation(
            note_type=note_type,
            graph_connectivity=connectivity,
            user_behavior=behavior,
            uncertainty_level=uncertainty,
            raw_features=raw_features,
        )

    def _encode_note_type(self, fragment: "Fragment") -> tuple[NoteType, dict]:
        """
        Classify note type from content analysis.

        Note types reflect the user's intent:
        - FRAGMENT: Short, incomplete thought
        - QUESTION: Seeking information
        - STATEMENT: Complete assertion
        - DIAGRAM: Visual/structural content
        """
        content = fragment.content
        features = {}

        # Check for diagram-like content
        diagram_score = sum(1 for kw in DIAGRAM_KEYWORDS if kw in content)
        features["diagram_keywords"] = diagram_score

        if diagram_score >= DIAGRAM_SCORE_THRESHOLD:
            return NoteType.DIAGRAM, features

        # Check for questions
        question_markers = ["?", "what", "why", "how", "when", "where", "who"]
        question_count = sum(
            1 for marker in question_markers
            if marker in content.lower()
        )
        question_ratio = question_count / max(1, len(content.split()))
        features["question_ratio"] = question_ratio
        features["has_question_mark"] = fragment.episodic_context.has_question_mark

        if fragment.episodic_context.has_question_mark or question_ratio > QUESTION_THRESHOLD:
            return NoteType.QUESTION, features

        # Check for statement vs fragment
        word_count = fragment.episodic_context.word_count
        features["word_count"] = word_count

        # Check for complete sentence structure
        has_period = content.strip().endswith(".")
        has_capital_start = bool(content) and content[0].isupper()
        features["has_period"] = has_period
        features["has_capital_start"] = has_capital_start

        if word_count >= STATEMENT_WORD_THRESHOLD and has_period:
            return NoteType.STATEMENT, features

        return NoteType.FRAGMENT, features

    def _encode_connectivity(self, edge_count: int) -> tuple[GraphConnectivity, dict]:
        """
        Classify graph connectivity from edge count.

        Connectivity reflects how integrated this note is in the knowledge web:
        - ISOLATED: No connections (orphan seed)
        - SPARSE: Few connections (sprouting)
        - CONNECTED: Good connectivity
        - HUB: High connectivity (central concept)
        """
        features = {"edge_count": edge_count}

        if edge_count == 0:
            return GraphConnectivity.ISOLATED, features
        elif edge_count < SPARSE_EDGE_THRESHOLD:
            return GraphConnectivity.SPARSE, features
        elif edge_count < HUB_EDGE_THRESHOLD:
            return GraphConnectivity.CONNECTED, features
        else:
            return GraphConnectivity.HUB, features

    def _encode_user_behavior(self, fragment: "Fragment") -> tuple[UserBehavior, dict]:
        """
        Classify user behavior from motor trace.

        Behavior reflects cognitive mode:
        - FAST_INPUT: Automatic, System 1 capture
        - SLOW_DELIBERATE: Careful, System 2 reasoning
        - REVIEWING: Re-reading/editing existing content
        - IDLE: No active input (background processing)
        """
        features = {}

        motor_trace = fragment.motor_trace
        if motor_trace is None:
            # No motor trace-assume fast input (digital input)
            features["has_motor_trace"] = False
            features["fluency_score"] = None
            return UserBehavior.FAST_INPUT, features

        features["has_motor_trace"] = True

        # Use velocity metrics from motor trace
        velocity = motor_trace.velocity_metrics
        if velocity:
            fluency_score = velocity.fluency_score
            features["fluency_score"] = fluency_score

            if fluency_score < SLOW_VELOCITY_THRESHOLD:
                return UserBehavior.SLOW_DELIBERATE, features
            elif fluency_score > FAST_VELOCITY_THRESHOLD:
                return UserBehavior.FAST_INPUT, features
            else:
                # Moderate speed-could be reviewing
                return UserBehavior.REVIEWING, features

        # Default to slow deliberate for stylus input without velocity data
        features["fluency_score"] = None
        return UserBehavior.SLOW_DELIBERATE, features

    def _encode_uncertainty(
        self,
        fragment: "Fragment",
        has_conflicts: bool,
    ) -> tuple[UncertaintyLevel, dict]:
        """
        Classify uncertainty level from confidence and conflicts.

        Uncertainty drives epistemic behavior:
        - LOW: High confidence, low uncertainty
        - MEDIUM: Some uncertainty, may benefit from clarification
        - HIGH: Low confidence, needs verification
        - VERY_HIGH: Contradictory information detected (orphan territory)
        """
        features = {"has_conflicts": has_conflicts}

        # Check for explicit conflicts first (highest uncertainty)
        if has_conflicts:
            return UncertaintyLevel.VERY_HIGH, features

        # Use confidence at capture
        confidence = fragment.confidence_at_capture
        features["confidence_at_capture"] = confidence

        # Check for uncertainty markers in content
        uncertainty_markers = [
            "maybe", "perhaps", "might", "possibly", "not sure",
            "i think", "seems like", "could be", "unsure"
        ]
        content_lower = fragment.content.lower()
        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker in content_lower
        )
        features["uncertainty_markers"] = uncertainty_count

        # Adjust confidence based on markers
        adjusted_confidence = confidence - (uncertainty_count * UNCERTAINTY_MARKER_PENALTY)
        features["adjusted_confidence"] = adjusted_confidence

        if adjusted_confidence >= CONFIDENT_THRESHOLD:
            return UncertaintyLevel.LOW, features
        elif adjusted_confidence >= MODERATE_THRESHOLD:
            return UncertaintyLevel.MEDIUM, features
        else:
            return UncertaintyLevel.HIGH, features

    async def _get_edge_count(self, uid: str, tenant_id: str) -> int:
        """Query graph for fragment's edge count."""
        if not self.graph:
            return 0

        try:
            query = """
                MATCH (n {uid: $uid, tenant_id: $tenant_id})-[r]-()
                RETURN count(r) as count
            """
            result = await self.graph.query(
                query,
                params={"uid": uid, "tenant_id": tenant_id}
            )
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.debug(f"Edge count query failed for {uid}: {e}")
            return 0


async def encode_observation(
    fragment: "Fragment",
    graph: Optional["PsycheClient"] = None,
    tenant_id: str = "default",
    edge_count: Optional[int] = None,
    has_conflicts: bool = False,
) -> EncodedObservation:
    """
    Convenience function to encode a fragment observation.

    Usage:
        from core.active_inference import encode_observation

        observation = await encode_observation(fragment, graph, tenant_id)
        indices = observation.to_indices()
    """
    encoder = ObservationEncoder(graph)
    return await encoder.encode_fragment(
        fragment, tenant_id, edge_count, has_conflicts
    )
