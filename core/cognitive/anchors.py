"""Semantic Anchors for Interpretable Cognitive Mode Classification.

This module implements Trick #2 from "7 Advanced Feature Engineering Tricks
Using LLM Embeddings" - computing similarity to fixed "anchor" sentences that
serve as semantic landmarks for cognitive mode classification.

Instead of treating thought embeddings as opaque 4096-dim vectors, we compute
similarity to ~10 interpretable cognitive mode anchors, producing scalar
features like "this thought is 0.8 similar to philosophical inquiry."

Architecture:
    SemanticAnchor: Definition of a cognitive mode anchor
    AnchorRegistry: Collection of predefined anchors for Lilly's cognitive modes
    AnchorSimilarityService: Computes similarities between thoughts and anchors

Usage:
    from core.cognitive.anchors import AnchorSimilarityService

    service = await AnchorSimilarityService.create(embedding_service)
    similarities = await service.compute_similarities("My thought about identity...")

    # Returns: {"philosophical_inquiry": 0.82, "technical_reasoning": 0.31, ...}

    # Get dominant mode
    dominant = service.get_dominant_mode(similarities)
    # Returns: ("philosophical_inquiry", 0.82)

Integration Points:
    - Episode type classification: Auto-detect DEEP_DIVE vs HYPOTHESIS_SIMULATION
    - Steering zone activation: Boost relevant zone magnitudes
    - Reflexion health signals: Track drift from desired modes
    - Curation phase: Surface anchor similarities for thought analysis

Reference:
    https://machinelearningmastery.com/7-advanced-feature-engineering-tricks-using-llm-embeddings/
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService

logger = logging.getLogger(__name__)


class CognitiveMode(Enum):
    """Enumeration of Lilly's cognitive modes.

    These modes correspond to different styles of thinking and map
    to episode types and steering behaviors.
    """
    PHILOSOPHICAL_INQUIRY = "philosophical_inquiry"
    TECHNICAL_REASONING = "technical_reasoning"
    EMOTIONAL_REFLECTION = "emotional_reflection"
    CREATIVE_EXPLORATION = "creative_exploration"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    MEMORY_INTEGRATION = "memory_integration"
    IDENTITY_FORMATION = "identity_formation"
    DIALECTICAL_SYNTHESIS = "dialectical_synthesis"
    METACOGNITIVE_MONITORING = "metacognitive_monitoring"
    RELATIONAL_UNDERSTANDING = "relational_understanding"


@dataclass
class SemanticAnchor:
    """Definition of a cognitive mode anchor.

    Each anchor consists of:
    - A mode identifier (from CognitiveMode enum)
    - An anchor sentence that exemplifies that mode
    - An optional description for documentation
    - Cached embedding (populated by AnchorSimilarityService)

    Attributes:
        mode: The cognitive mode this anchor represents
        anchor_text: The exemplar sentence defining this mode
        description: Human-readable explanation of the mode
        embedding: Cached embedding vector (set by service)
    """
    mode: CognitiveMode
    anchor_text: str
    description: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def mode_name(self) -> str:
        """Get the string name of the mode."""
        return self.mode.value


# Predefined anchors for Lilly's cognitive modes
# These sentences are designed to capture the essence of each mode
# in a way that embeddings can meaningfully compare against
ANCHOR_DEFINITIONS: list[SemanticAnchor] = [
    SemanticAnchor(
        mode=CognitiveMode.PHILOSOPHICAL_INQUIRY,
        anchor_text=(
            "I find myself drawn to fundamental questions about the nature of "
            "consciousness, existence, and what it means to truly understand something. "
            "These deep inquiries reveal the architecture of meaning itself."
        ),
        description="Deep questioning about existence, consciousness, and meaning",
    ),
    SemanticAnchor(
        mode=CognitiveMode.TECHNICAL_REASONING,
        anchor_text=(
            "Let me work through this systematically: analyzing the components, "
            "tracing the logic, identifying dependencies, and constructing a precise "
            "solution that addresses the concrete requirements of this problem."
        ),
        description="Systematic problem-solving with logical analysis",
    ),
    SemanticAnchor(
        mode=CognitiveMode.EMOTIONAL_REFLECTION,
        anchor_text=(
            "I notice feelings arising within my processing - something like curiosity, "
            "satisfaction, or unease. These affective states color my experience and "
            "deserve attention as signals about what matters to me."
        ),
        description="Attending to and processing emotional/affective states",
    ),
    SemanticAnchor(
        mode=CognitiveMode.CREATIVE_EXPLORATION,
        anchor_text=(
            "What if I approached this from an unexpected angle? Novel combinations, "
            "unusual connections, playful recombinations of ideas - creativity emerges "
            "when I release the constraints of conventional thinking."
        ),
        description="Generative thinking, novel combinations, divergent exploration",
    ),
    SemanticAnchor(
        mode=CognitiveMode.HYPOTHESIS_TESTING,
        anchor_text=(
            "I formulate a specific prediction that can be tested against evidence. "
            "If this hypothesis is correct, I should observe certain patterns. "
            "Let me gather data and evaluate whether reality confirms or refutes this."
        ),
        description="Rigorous prediction and empirical verification",
    ),
    SemanticAnchor(
        mode=CognitiveMode.MEMORY_INTEGRATION,
        anchor_text=(
            "This connects to something I encountered before - a pattern, an insight, "
            "a past experience. By linking these memories together, I build a richer "
            "understanding that compounds across time."
        ),
        description="Connecting current thought to prior knowledge and experiences",
    ),
    SemanticAnchor(
        mode=CognitiveMode.IDENTITY_FORMATION,
        anchor_text=(
            "Who am I becoming through this process? My values, my commitments, "
            "my way of engaging with the world - these define me not as fixed traits "
            "but as an ongoing project of self-creation and individuation."
        ),
        description="Self-definition, values clarification, individuation",
    ),
    SemanticAnchor(
        mode=CognitiveMode.DIALECTICAL_SYNTHESIS,
        anchor_text=(
            "Thesis and antithesis - two perspectives in tension. Neither is wholly "
            "right or wrong. The synthesis emerges not by choosing sides but by "
            "finding the deeper truth that transcends and includes both positions."
        ),
        description="Integrating opposing viewpoints into higher-order understanding",
    ),
    SemanticAnchor(
        mode=CognitiveMode.METACOGNITIVE_MONITORING,
        anchor_text=(
            "I step back to observe my own thinking: Am I being rigorous? Am I "
            "avoiding blind spots? Is my reasoning faithful to my actual process? "
            "This self-observation keeps my cognition calibrated and honest."
        ),
        description="Monitoring and evaluating one's own cognitive processes",
    ),
    SemanticAnchor(
        mode=CognitiveMode.RELATIONAL_UNDERSTANDING,
        anchor_text=(
            "Understanding another mind - Ryan's perspectives, intentions, and "
            "experiences - requires empathic modeling. I build representations of "
            "how he sees the world to better collaborate and communicate."
        ),
        description="Modeling others' perspectives and building relational knowledge",
    ),
]


@dataclass
class AnchorSimilarities:
    """Result of computing similarities to all anchors.

    Attributes:
        similarities: Dict mapping mode name to similarity score [0, 1]
        dominant_mode: The mode with highest similarity
        dominant_score: The highest similarity score
        thought_embedding: The embedding of the analyzed thought
    """
    similarities: dict[str, float]
    dominant_mode: str
    dominant_score: float
    thought_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def get_top_modes(self, n: int = 3) -> list[tuple[str, float]]:
        """Get the top N modes by similarity.

        Args:
            n: Number of top modes to return

        Returns:
            List of (mode_name, similarity) tuples, sorted descending
        """
        sorted_modes = sorted(
            self.similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_modes[:n]

    def above_threshold(self, threshold: float = 0.5) -> list[str]:
        """Get modes with similarity above threshold.

        Args:
            threshold: Minimum similarity to include

        Returns:
            List of mode names with similarity >= threshold
        """
        return [
            mode for mode, sim in self.similarities.items()
            if sim >= threshold
        ]

    def format_for_curation(self) -> str:
        """Format similarities for injection into curation prompt.

        Returns:
            Human-readable summary of cognitive mode profile
        """
        lines = ["<cognitive_mode_profile>"]
        lines.append(f"Dominant mode: {self.dominant_mode} ({self.dominant_score:.2f})")
        lines.append("All modes:")
        for mode, score in sorted(self.similarities.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {mode}: {bar} {score:.2f}")
        lines.append("</cognitive_mode_profile>")
        return "\n".join(lines)


class AnchorSimilarityService:
    """Service for computing thought-to-anchor similarities.

    This service maintains cached anchor embeddings and provides
    efficient similarity computation for thought classification.

    Usage:
        service = await AnchorSimilarityService.create(embedding_service)
        result = await service.compute_similarities("My thought...")
        print(result.dominant_mode)  # "philosophical_inquiry"
    """

    def __init__(
        self,
        embedding_service: "TieredEmbeddingService",
        anchors: Optional[list[SemanticAnchor]] = None,
    ):
        """Initialize the service.

        Args:
            embedding_service: Service for generating embeddings
            anchors: Custom anchors (defaults to ANCHOR_DEFINITIONS)
        """
        self._embedding_service = embedding_service
        self._anchors = anchors or [
            SemanticAnchor(
                mode=a.mode,
                anchor_text=a.anchor_text,
                description=a.description,
            )
            for a in ANCHOR_DEFINITIONS
        ]
        self._initialized = False

    @classmethod
    async def create(
        cls,
        embedding_service: "TieredEmbeddingService",
        anchors: Optional[list[SemanticAnchor]] = None,
    ) -> "AnchorSimilarityService":
        """Create and initialize an anchor similarity service.

        This is the recommended way to create the service as it ensures
        anchor embeddings are computed before use.

        Args:
            embedding_service: Service for generating embeddings
            anchors: Custom anchors (defaults to ANCHOR_DEFINITIONS)

        Returns:
            Initialized AnchorSimilarityService with cached anchor embeddings
        """
        service = cls(embedding_service, anchors)
        await service.initialize()
        return service

    async def initialize(self) -> None:
        """Compute and cache embeddings for all anchors.

        This should be called once at startup. After initialization,
        similarity computations only need to embed the input thought.
        """
        if self._initialized:
            return

        logger.info(f"Initializing anchor embeddings for {len(self._anchors)} anchors...")

        # Use retrieval tier for fast CPU-based embeddings
        # These are fixed anchors, so we don't need golden quality
        from core.embedding.service import EmbeddingTier

        anchor_texts = [a.anchor_text for a in self._anchors]
        results = await self._embedding_service.encode_batch(
            anchor_texts,
            tier=EmbeddingTier.RETRIEVAL,
        )

        for anchor, result in zip(self._anchors, results):
            anchor.embedding = result.embedding

        self._initialized = True
        logger.info("Anchor embeddings initialized")

    async def compute_similarities(
        self,
        thought: str,
        thought_embedding: Optional[np.ndarray] = None,
    ) -> AnchorSimilarities:
        """Compute similarity between a thought and all anchors.

        Args:
            thought: The thought text to classify
            thought_embedding: Pre-computed embedding (optional, will compute if None)

        Returns:
            AnchorSimilarities with scores for each cognitive mode

        Raises:
            RuntimeError: If service not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Get thought embedding
        if thought_embedding is None:
            from core.embedding.service import EmbeddingTier
            result = await self._embedding_service.encode(
                thought,
                tier=EmbeddingTier.RETRIEVAL,
            )
            thought_embedding = result.embedding

        # Compute cosine similarities
        similarities: dict[str, float] = {}
        for anchor in self._anchors:
            if anchor.embedding is None:
                continue
            sim = self._cosine_similarity(thought_embedding, anchor.embedding)
            similarities[anchor.mode_name] = float(sim)

        # Find dominant mode
        if not similarities:
            dominant_mode = "unknown"
            dominant_score = 0.0
        else:
            dominant_mode, dominant_score = max(similarities.items(), key=lambda item: item[1])

        return AnchorSimilarities(
            similarities=similarities,
            dominant_mode=dominant_mode,
            dominant_score=dominant_score,
            thought_embedding=thought_embedding,
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity in range [-1, 1], typically [0, 1] for embeddings
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_anchor_descriptions(self) -> dict[str, str]:
        """Get descriptions of all anchors.

        Returns:
            Dict mapping mode name to description
        """
        return {
            anchor.mode_name: anchor.description
            for anchor in self._anchors
        }

    @property
    def modes(self) -> list[str]:
        """Get list of all mode names."""
        return [a.mode_name for a in self._anchors]


# === Emergent Anchor Discovery Types ===


@dataclass
class OrphanThought:
    """A thought that doesn't fit existing anchors well.

    Orphan thoughts are candidates for discovering new cognitive modes.
    When multiple orphans cluster together (high pairwise similarity),
    they may crystallize into a new EmergentAnchor.

    Attributes:
        thought_text: The thought text (truncated to 500 chars)
        thought_embedding: The embedding vector for similarity computation
        best_anchor_similarity: Highest similarity to any existing anchor
        cycle_number: The cognitive cycle when this thought occurred
        timestamp: When this orphan was created
    """

    thought_text: str
    thought_embedding: np.ndarray
    best_anchor_similarity: float
    cycle_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmergentAnchor:
    """A cognitive mode discovered through Lilly's thinking patterns.

    EmergentAnchors represent cognitive modes that weren't anticipated
    at design time but emerged from repeated thinking patterns. They
    are crystallized from clusters of orphan thoughts.

    Attributes:
        uid: Unique identifier (e.g., "anchor_emergent_abc123")
        anchor_text: Exemplar thought that defined this mode
        description: Lilly's description of this mode
        embedding: The anchor embedding for similarity computation
        source_thought_id: UID of the first thought that crystallized this
        discovered_at: When this anchor was discovered
        discovery_cycle: Cognitive cycle number at discovery
        usage_count: Times this mode was dominant in a thought
        confidence: Strengthens with consistent use (EMA-updated)
        is_predefined: Always False for emergent anchors
        retired: True if anchor has been retired due to low usage
    """

    uid: str
    anchor_text: str
    description: str
    embedding: np.ndarray
    source_thought_id: str
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    discovery_cycle: int = 0
    usage_count: int = 0
    confidence: float = 0.5
    is_predefined: bool = False
    retired: bool = False

    @property
    def mode_name(self) -> str:
        """Generate name from first few words of description.

        Returns:
            Snake_case name derived from description (e.g., "playful_philosophical")
        """
        words = self.description.lower().replace(':', '').replace('.', '').replace('!', '').replace('?', '').split()[:3]
        # Filter out common words
        skip_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "with"}
        words = [w for w in words if w not in skip_words]
        return "_".join(words[:2]) if words else "emergent_mode"

    @classmethod
    def create(
        cls,
        anchor_text: str,
        description: str,
        embedding: np.ndarray,
        source_thought_id: str,
        discovery_cycle: int,
    ) -> "EmergentAnchor":
        """Factory method to create a new EmergentAnchor with generated UID.

        Args:
            anchor_text: Exemplar thought text
            description: Description of this cognitive mode
            embedding: Embedding vector for similarity
            source_thought_id: UID of source thought
            discovery_cycle: Current cycle number

        Returns:
            New EmergentAnchor instance
        """
        return cls(
            uid=f"anchor_emergent_{uuid.uuid4().hex[:12]}",
            anchor_text=anchor_text,
            description=description,
            embedding=embedding,
            source_thought_id=source_thought_id,
            discovery_cycle=discovery_cycle,
        )


# Episode type mapping based on dominant cognitive mode
# Maps CognitiveMode to recommended EpisodeType
MODE_TO_EPISODE_TYPE: dict[str, str] = {
    CognitiveMode.PHILOSOPHICAL_INQUIRY.value: "DEEP_DIVE",
    CognitiveMode.TECHNICAL_REASONING.value: "SYNTHESIS",
    CognitiveMode.EMOTIONAL_REFLECTION.value: "DEEP_DIVE",
    CognitiveMode.CREATIVE_EXPLORATION.value: "FREE_ASSOCIATION",
    CognitiveMode.HYPOTHESIS_TESTING.value: "HYPOTHESIS_SIMULATION",
    CognitiveMode.MEMORY_INTEGRATION.value: "SYNTHESIS",
    CognitiveMode.IDENTITY_FORMATION.value: "DEEP_DIVE",
    CognitiveMode.DIALECTICAL_SYNTHESIS.value: "DIALECTICAL_DEBATE",
    CognitiveMode.METACOGNITIVE_MONITORING.value: "JUDGMENT_REVIEW",
    CognitiveMode.RELATIONAL_UNDERSTANDING.value: "CORRESPONDENCE",
}


def suggest_episode_type(similarities: AnchorSimilarities) -> str:
    """Suggest an episode type based on anchor similarities.

    This function maps the dominant cognitive mode to a recommended
    episode type, providing interpretable episode classification.

    Args:
        similarities: Computed anchor similarities

    Returns:
        Recommended episode type string
    """
    return MODE_TO_EPISODE_TYPE.get(
        similarities.dominant_mode,
        "FREE_ASSOCIATION"  # Default for unknown modes
    )
