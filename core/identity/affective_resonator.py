"""Affective Resonator - Phase 1 of Integrated Identity Layer.

Translates recent valenced experiences into real-time steering during inference.
Creates "felt" biases from experience - "this approach failed before" manifests
as reluctance at the activation level, not a retrieved fact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Protocol

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.psyche.schema import Fragment
    from core.self_model.affective_system import AffectiveState
    from core.embedding.service import TieredEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def encode(self, text: str) -> "EmbeddingResult":
        """Encode text to embedding."""
        ...


@dataclass
class EmbeddingResult:
    """Minimal embedding result for protocol compatibility."""
    embedding: list[float]


@dataclass
class ResonanceConfig:
    """Configuration for affective resonance computation.

    Attributes:
        recency_decay_hours: How quickly older experiences lose influence
        max_experiences: Maximum number of similar experiences to consider
        valence_weight: Weight of aggregated valence in final vector
        affect_weight: Weight of current affective state in final vector
        min_similarity: Minimum similarity score to include experience
        resonance_layers: Which transformer layers to target for injection
    """

    recency_decay_hours: float = 24.0
    max_experiences: int = 10
    valence_weight: float = 0.6
    affect_weight: float = 0.4
    min_similarity: float = 0.3
    resonance_layers: tuple[int, int] = (14, 18)


@dataclass
class ResonanceResult:
    """Result of affective resonance computation.

    Attributes:
        vector: The steering vector to apply (or None if not computed)
        aggregated_valence: Net valence from relevant experiences (-1 to +1)
        experience_count: Number of relevant experiences found
        mean_similarity: Average similarity of matched experiences
        dominant_affect: Primary affective dimension contributing
        computation_time_ms: How long the computation took
    """

    vector: Optional["torch.Tensor"]
    aggregated_valence: float
    experience_count: int
    mean_similarity: float
    dominant_affect: str
    computation_time_ms: float

    def to_dict(self) -> dict:
        """Serialize for logging/storage."""
        return {
            "has_vector": self.vector is not None,
            "aggregated_valence": self.aggregated_valence,
            "experience_count": self.experience_count,
            "mean_similarity": self.mean_similarity,
            "dominant_affect": self.dominant_affect,
            "computation_time_ms": self.computation_time_ms,
        }


@dataclass
class _WeightedExperience:
    """Internal: Experience with computed weights."""

    fragment: "Fragment"
    similarity: float
    recency_weight: float
    valence: float  # Inferred from resonance/confidence

    @property
    def combined_weight(self) -> float:
        """Combined weight considering both recency and similarity."""
        return self.similarity * self.recency_weight


class AffectiveResonator:
    """Translates valenced experiences into real-time steering.

    Before each generation, queries recent relevant experiences from psyche,
    computes resonance based on accumulated valence, blends with current
    affective state, and produces a steering vector.

    The resulting vector creates "felt" biases - reluctance or enthusiasm
    at the activation level rather than explicit retrieved facts.
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[ResonanceConfig] = None,
        embedding_service: Optional["TieredEmbeddingService"] = None,
    ):
        """Initialize the affective resonator.

        Args:
            hidden_size: Model's hidden dimension for steering vectors
            config: Resonance computation configuration
            embedding_service: Service for encoding context to embeddings
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for AffectiveResonator")

        self.hidden_size = hidden_size
        self.config = config or ResonanceConfig()
        self._embedding_service = embedding_service

        # Learned projection matrices (initialized to small random values)
        # These project from valence/affect space to model's hidden space
        self._valence_projection = torch.nn.Linear(1, hidden_size)
        self._affect_projection = torch.nn.Linear(6, hidden_size)

        # Initialize with small values to start with minimal intervention
        with torch.no_grad():
            self._valence_projection.weight.data *= 0.01
            self._affect_projection.weight.data *= 0.01

    def set_embedding_service(self, service: "TieredEmbeddingService") -> None:
        """Set the embedding service for context encoding."""
        self._embedding_service = service

    async def compute_resonance(
        self,
        context: str,
        psyche: "PsycheClient",
        affective_state: "AffectiveState",
    ) -> ResonanceResult:
        """Compute affective resonance for the given context.

        This is the main entry point. Given the current context and affective
        state, queries psyche for similar past experiences and computes a
        steering vector that encodes the accumulated valence.

        Args:
            context: Current generation context (prompt/situation)
            psyche: Client to query for similar experiences
            affective_state: Current affective state

        Returns:
            ResonanceResult with steering vector and metadata
        """
        start = datetime.now(timezone.utc)

        # Get embedding for context
        if self._embedding_service is None:
            logger.warning("No embedding service configured, returning neutral")
            return self._neutral_result(start)

        try:
            from core.embedding.service import EmbeddingTier

            embedding_result = await self._embedding_service.encode(
                context, tier=EmbeddingTier.RETRIEVAL
            )
            context_embedding = embedding_result.embedding
        except Exception as e:
            logger.warning(f"Failed to encode context: {e}")
            return self._neutral_result(start)

        # Query for similar experiences
        try:
            similar = await psyche.semantic_search(
                embedding=context_embedding,
                limit=self.config.max_experiences,
            )
        except Exception as e:
            logger.warning(f"Failed to search psyche: {e}")
            return self._neutral_result(start)

        # Filter by minimum similarity
        relevant = [
            (frag, score)
            for frag, score in similar
            if score >= self.config.min_similarity
        ]

        if not relevant:
            return self._neutral_result(start)

        # Compute weighted experiences
        weighted = self._weight_experiences(relevant)

        # Aggregate valence from experiences
        aggregated_valence = self._aggregate_valence(weighted)

        # Blend with current affective state
        vector = self._compute_steering_vector(
            aggregated_valence, affective_state
        )

        # Compute metadata
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        mean_sim = sum(w.similarity for w in weighted) / len(weighted)
        dominant = self._get_dominant_affect(affective_state)

        return ResonanceResult(
            vector=vector,
            aggregated_valence=aggregated_valence,
            experience_count=len(weighted),
            mean_similarity=mean_sim,
            dominant_affect=dominant,
            computation_time_ms=elapsed,
        )

    def _weight_experiences(
        self, experiences: list[tuple["Fragment", float]]
    ) -> list[_WeightedExperience]:
        """Compute weights for each experience based on recency and similarity."""
        now = datetime.now(timezone.utc)
        weighted = []

        for fragment, similarity in experiences:
            # Compute recency weight (exponential decay)
            age = now - fragment.last_accessed
            age_hours = age.total_seconds() / 3600
            recency_weight = 2 ** (-age_hours / self.config.recency_decay_hours)

            # Infer valence from fragment properties
            # resonance > 0.5 suggests positive, < 0.5 suggests negative
            # confidence affects strength of the signal
            valence = (fragment.resonance - 0.5) * 2 * fragment.confidence

            weighted.append(
                _WeightedExperience(
                    fragment=fragment,
                    similarity=similarity,
                    recency_weight=recency_weight,
                    valence=valence,
                )
            )

        return weighted

    def _aggregate_valence(
        self, experiences: list[_WeightedExperience]
    ) -> float:
        """Aggregate valence from weighted experiences.

        Uses weighted average where weight = similarity * recency_weight.
        """
        if not experiences:
            return 0.0

        total_weight = sum(e.combined_weight for e in experiences)
        if total_weight == 0:
            return 0.0

        weighted_valence = sum(
            e.valence * e.combined_weight for e in experiences
        )
        aggregated = weighted_valence / total_weight

        # Clamp to valid range
        return max(-1.0, min(1.0, aggregated))

    def _compute_steering_vector(
        self,
        aggregated_valence: float,
        affective_state: "AffectiveState",
    ) -> "torch.Tensor":
        """Compute final steering vector from valence and affect.

        Blends valence-derived and affect-derived components using
        configured weights.
        """
        # Project valence to hidden space
        valence_tensor = torch.tensor(
            [[aggregated_valence]], dtype=torch.float32
        )
        valence_component = self._valence_projection(valence_tensor).squeeze(0)

        # Project affective state to hidden space
        affect_vector = affective_state.to_vector()
        affect_tensor = torch.tensor([affect_vector], dtype=torch.float32)
        affect_component = self._affect_projection(affect_tensor).squeeze(0)

        # Blend components
        combined = (
            self.config.valence_weight * valence_component +
            self.config.affect_weight * affect_component
        )

        return combined

    def _get_dominant_affect(self, state: "AffectiveState") -> str:
        """Determine the dominant affective dimension by its intensity."""
        affects = {
            "arousal": state.arousal,
            "valence": abs(state.valence - 0.5),
            "curiosity": state.curiosity,
            "satisfaction": state.satisfaction,
            "frustration": state.frustration,
            "wonder": state.wonder,
        }
        return max(affects, key=affects.get)

    def _neutral_result(self, start: datetime) -> ResonanceResult:
        """Return a neutral result when computation fails or finds nothing."""
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return ResonanceResult(
            vector=None,
            aggregated_valence=0.0,
            experience_count=0,
            mean_similarity=0.0,
            dominant_affect="neutral",
            computation_time_ms=elapsed,
        )

    def get_target_layers(self) -> tuple[int, int]:
        """Get the layer range for resonance injection."""
        return self.config.resonance_layers

    def to_dict(self) -> dict:
        """Serialize configuration for logging."""
        return {
            "hidden_size": self.hidden_size,
            "config": {
                "recency_decay_hours": self.config.recency_decay_hours,
                "max_experiences": self.config.max_experiences,
                "valence_weight": self.config.valence_weight,
                "affect_weight": self.config.affect_weight,
                "min_similarity": self.config.min_similarity,
                "resonance_layers": self.config.resonance_layers,
            },
        }
