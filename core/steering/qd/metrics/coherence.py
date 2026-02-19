"""Coherence metric: embedding alignment quality floor.

Measures how well a crystal's steering direction aligns with recent
thought embeddings. This acts as a quality floor - crystals that don't
maintain coherent semantic alignment are rejected.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.steering.qd.metrics.base import BaseMetric
from core.steering.utils import cosine_similarity

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.steering.crystal import CrystalEntry


class CoherenceMetric(BaseMetric):
    """Coherence metric based on embedding alignment.

    Computes cosine similarity between the crystal's steering vector
    direction and the current context embedding. Higher similarity
    indicates the crystal is steering toward semantically coherent
    directions.

    Attributes:
        embedding_service: Optional service for generating embeddings
    """

    def __init__(
        self,
        embedding_service: Optional["TieredEmbeddingService"] = None,
    ):
        """Initialize coherence metric.

        Args:
            embedding_service: Service for generating embeddings.
                If None, coherence defaults to 0.5 (neutral).
        """
        self.embedding_service = embedding_service

    def compute(
        self,
        crystal: "CrystalEntry",
        context_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute coherence score for a crystal.

        Args:
            crystal: The crystal entry to score
            context_embedding: Current context embedding for comparison.
                If None, returns 0.5 (neutral score).

        Returns:
            Coherence score in [0, 1] range
        """
        if context_embedding is None:
            # No context for comparison - return neutral
            return 0.5

        # Compute cosine similarity between crystal vector and context
        similarity = cosine_similarity(crystal.vector, context_embedding)

        # Map from [-1, 1] to [0, 1]
        # High positive similarity = high coherence
        # High negative similarity = low coherence (actively anti-aligned)
        score = (similarity + 1.0) / 2.0

        return self.clamp(score)
