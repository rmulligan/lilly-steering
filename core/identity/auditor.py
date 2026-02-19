"""Identity Auditor: Coherence scoring via embedding similarity.

This module measures how well a response aligns with Lilly's Creed
Constitution. Instead of binary accept/reject filtering, the auditor
produces a coherence score that can trigger reflection at low values.

Key insight:
"You don't need explicit 'identity validation' steps if the policy
selection is informed by persistent identity context."

The auditor computes embedding similarity between:
1. The response text
2. Each layer of the Creed Constitution (weighted by layer importance)

High coherence (>0.7): Response aligns with identity
Low coherence (<0.4): Triggers reflection or revision

Design Philosophy:
The auditor is a compass, not a gate. It guides toward coherence
rather than blocking responses. This allows organic identity
expression rather than mechanical constraint checking.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from core.identity.creed import Creed, CreedLayer

logger = logging.getLogger(__name__)


# Layer weights for coherence scoring
# Higher weight = more impact on overall score
LAYER_WEIGHTS: dict[CreedLayer, float] = {
    CreedLayer.AXIOM: 1.0,  # Critical - axiom violations are serious
    CreedLayer.TRAIT: 0.7,  # Important - personality misalignment matters
    CreedLayer.SKILL: 0.4,  # Moderate - skill gaps are addressable
    CreedLayer.NARRATIVE: 0.2,  # Gentle - narrative drift is soft signal
}

# Coherence thresholds
THRESHOLD_HIGH = 0.7  # Above this: strong identity alignment
THRESHOLD_LOW = 0.4  # Below this: trigger reflection
THRESHOLD_CRITICAL = 0.2  # Below this: potential axiom violation


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding services.

    Any class implementing these methods can be used as an embedding
    provider for identity coherence scoring.
    """

    async def encode(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        ...

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts into embedding vectors."""
        ...


@dataclass
class CoherenceResult:
    """Result of coherence scoring.

    Attributes:
        overall_score: Weighted average coherence (0.0-1.0)
        layer_scores: Per-layer similarity scores
        dominant_layer: Layer with lowest score (potential issue)
        needs_reflection: Whether the response should trigger reflection
        axiom_violation_detected: Whether a critical axiom issue was found
    """

    overall_score: float
    layer_scores: dict[CreedLayer, float]
    dominant_layer: Optional[CreedLayer] = None
    needs_reflection: bool = False
    axiom_violation_detected: bool = False

    @property
    def is_coherent(self) -> bool:
        """Response is sufficiently aligned with identity."""
        return self.overall_score >= THRESHOLD_LOW and not self.axiom_violation_detected

    def to_dict(self) -> dict:
        """Serialize to dictionary for logging/persistence."""
        return {
            "overall_score": self.overall_score,
            "layer_scores": {k.name: v for k, v in self.layer_scores.items()},
            "dominant_layer": self.dominant_layer.name if self.dominant_layer else None,
            "needs_reflection": self.needs_reflection,
            "axiom_violation_detected": self.axiom_violation_detected,
            "is_coherent": self.is_coherent,
        }


@dataclass
class IdentityAuditor:
    """Audits response coherence against the Creed Constitution.

    Usage:
        auditor = IdentityAuditor(embedding_provider)
        result = await auditor.score_coherence(response_text, creed)

        if not result.is_coherent:
            # Trigger reflection or revision
            logger.warning(f"Low coherence: {result.overall_score:.2f}")

    The auditor lazily computes and caches Creed embeddings on first use,
    so subsequent calls are fast (only the response needs embedding).
    """

    embedding_provider: EmbeddingProvider
    _initialized: bool = field(default=False, repr=False)

    async def score_coherence(
        self,
        response_text: str,
        creed: Creed,
    ) -> CoherenceResult:
        """Score how well a response aligns with the Creed Constitution.

        Args:
            response_text: The response to audit
            creed: The Creed to score against

        Returns:
            CoherenceResult with overall and per-layer scores
        """
        if not response_text or not response_text.strip():
            return CoherenceResult(
                overall_score=0.0,
                layer_scores={layer: 0.0 for layer in CreedLayer},
                needs_reflection=True,
            )

        # Ensure Creed embeddings are computed
        await self._ensure_creed_embeddings(creed)

        # Embed the response
        response_embedding = await self.embedding_provider.encode(response_text)

        # Score each layer
        layer_scores: dict[CreedLayer, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for layer in CreedLayer:
            score = self._score_layer(response_embedding, creed, layer)
            layer_scores[layer] = score

            weight = LAYER_WEIGHTS[layer]
            weighted_sum += score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Find dominant (lowest) layer
        dominant_layer = min(layer_scores, key=lambda k: layer_scores[k])

        # Check for axiom violation
        axiom_violation = (
            layer_scores[CreedLayer.AXIOM] < THRESHOLD_CRITICAL
            if CreedLayer.AXIOM in layer_scores
            else False
        )

        return CoherenceResult(
            overall_score=overall_score,
            layer_scores=layer_scores,
            dominant_layer=(
                dominant_layer if layer_scores[dominant_layer] < THRESHOLD_LOW else None
            ),
            needs_reflection=overall_score < THRESHOLD_LOW,
            axiom_violation_detected=axiom_violation,
        )

    async def _ensure_creed_embeddings(self, creed: Creed) -> None:
        """Compute and cache Creed embeddings if not already done."""
        texts = creed.to_embedding_texts()

        # Axioms
        if creed._axiom_embeddings is None and texts[CreedLayer.AXIOM]:
            creed._axiom_embeddings = await self.embedding_provider.encode_batch(
                texts[CreedLayer.AXIOM]
            )

        # Traits
        if creed._trait_embeddings is None and texts[CreedLayer.TRAIT]:
            creed._trait_embeddings = await self.embedding_provider.encode_batch(
                texts[CreedLayer.TRAIT]
            )

        # Skills
        if creed._skill_embeddings is None and texts[CreedLayer.SKILL]:
            creed._skill_embeddings = await self.embedding_provider.encode_batch(
                texts[CreedLayer.SKILL]
            )

        # Narrative (single embedding)
        if creed._narrative_embedding is None and texts[CreedLayer.NARRATIVE]:
            creed._narrative_embedding = await self.embedding_provider.encode(
                texts[CreedLayer.NARRATIVE][0]
            )

    def _score_layer(
        self,
        response_embedding: list[float],
        creed: Creed,
        layer: CreedLayer,
    ) -> float:
        """Score response against a specific Creed layer.

        For layers with multiple items (axioms, traits, skills),
        returns the average similarity across all items.
        """
        embeddings: Optional[list[list[float]]] = None

        if layer == CreedLayer.AXIOM:
            embeddings = creed._axiom_embeddings
        elif layer == CreedLayer.TRAIT:
            embeddings = creed._trait_embeddings
        elif layer == CreedLayer.SKILL:
            embeddings = creed._skill_embeddings
        elif layer == CreedLayer.NARRATIVE:
            if creed._narrative_embedding:
                return self._cosine_similarity(
                    response_embedding, creed._narrative_embedding
                )
            return 0.0

        if not embeddings:
            return 0.0

        # Average similarity across all items in this layer
        similarities = [
            self._cosine_similarity(response_embedding, emb) for emb in embeddings
        ]

        return sum(similarities) / len(similarities) if similarities else 0.0

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns value in [0, 1] range (mapped from [-1, 1]).
        """
        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Cosine similarity is in [-1, 1], we normalize to [0, 1]
        similarity = float(dot_product / (norm_a * norm_b))
        return (similarity + 1) / 2  # Map [-1,1] to [0,1]


async def check_response_coherence(
    response_text: str,
    embedding_provider: EmbeddingProvider,
    creed: Optional[Creed] = None,
) -> CoherenceResult:
    """Convenience function to check response coherence.

    Usage:
        result = await check_response_coherence(
            response_text="Your response here",
            embedding_provider=my_embedding_provider,
        )

        if not result.is_coherent:
            # Handle low coherence
            pass

    Args:
        response_text: The response to audit
        embedding_provider: Any EmbeddingProvider implementation
        creed: Optional Creed (uses default if not provided)

    Returns:
        CoherenceResult with scores and flags
    """
    if creed is None:
        creed = Creed.default()

    auditor = IdentityAuditor(embedding_provider=embedding_provider)
    return await auditor.score_coherence(response_text, creed)
