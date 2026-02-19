"""Semantic entropy: Measuring meaning diversity in the knowledge graph.

This module computes entropy over the semantic content of the knowledge graph,
complementing structural entropy (topology-based) with meaning-based metrics.

Cognitive Science Background:
    While structural entropy measures "how connected is the graph?", semantic
    entropy measures "how diverse are the ideas?". A graph can be well-connected
    but contain only one topic (low semantic entropy) or poorly connected but
    span many domains (high semantic entropy).

    The interplay between these two creates the Discovery Parameter (D):
    - D > 0: High semantic diversity, low structural connectivity -> BRIDGE
    - D < 0: Low semantic diversity, high structural connectivity -> PROMPT
    - D ~ 0: Balanced -> MONITOR (isomorphic state)

Mathematical Foundation:
    Semantic entropy is computed from the spectral properties of the cosine
    similarity matrix of node embeddings. The eigenvalues capture the "modes"
    of semantic variation.

    H_sem = -sum(lambda_i * log(lambda_i)) / log(N)

    Where lambda_i are normalized eigenvalues (summing to 1).

    Low semantic entropy: embeddings cluster tightly (few topics)
    High semantic entropy: embeddings spread widely (many topics)
"""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class SemanticEntropyResult:
    """Result of semantic entropy computation.

    Attributes:
        semantic_entropy: Spectral entropy of embedding similarity (0-1)
        topic_concentration: How clustered are the embeddings (0-1, 1 = very clustered)
        effective_dimensions: Number of "significant" semantic dimensions
        embedding_count: Number of embeddings analyzed
        mean_similarity: Average pairwise cosine similarity
        min_similarity: Minimum pairwise similarity (most distant concepts)
        max_similarity: Maximum pairwise similarity (most similar concepts)
        is_diverse: Whether the graph has high semantic diversity
        diversity_reason: Human-readable explanation
    """

    semantic_entropy: float
    topic_concentration: float
    effective_dimensions: float
    embedding_count: int
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    is_diverse: bool
    diversity_reason: str

    @property
    def is_concentrated(self) -> bool:
        """Embeddings are tightly clustered (low diversity)."""
        return self.topic_concentration > 0.7

    @property
    def is_balanced(self) -> bool:
        """Moderate semantic diversity."""
        return 0.3 < self.semantic_entropy < 0.7

    def to_dict(self) -> dict:
        """Serialize for logging and persistence."""
        return {
            "semantic_entropy": self.semantic_entropy,
            "topic_concentration": self.topic_concentration,
            "effective_dimensions": self.effective_dimensions,
            "embedding_count": self.embedding_count,
            "mean_similarity": self.mean_similarity,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
            "is_diverse": self.is_diverse,
            "diversity_reason": self.diversity_reason,
        }


# Thresholds for semantic entropy decisions
SEMANTIC_ENTROPY_HIGH = 0.7  # Above this = highly diverse ideas
SEMANTIC_ENTROPY_LOW = 0.3  # Below this = concentrated topics
EFFECTIVE_DIM_THRESHOLD = 0.1  # Eigenvalue threshold for counting dimensions


class SemanticEntropyCalculator:
    """Computes semantic entropy from knowledge graph embeddings.

    Measures the diversity of *meaning* in the graph, independent of
    its topological structure. Answers the question: "How many distinct
    semantic themes exist in this knowledge base?"

    The computation uses spectral analysis of the embedding similarity matrix:
    1. Fetch all embeddings
    2. Compute pairwise cosine similarity matrix
    3. Normalize to create a valid probability distribution
    4. Compute eigenvalues (spectral decomposition)
    5. Derive entropy from eigenvalue distribution

    Attributes:
        graph: PsycheClient for embedding queries
        max_embeddings: Maximum embeddings to analyze (for performance)
    """

    def __init__(
        self,
        graph: "PsycheClient",
        max_embeddings: int = 500,
    ):
        """Initialize the semantic entropy calculator.

        Args:
            graph: PsycheClient instance
            max_embeddings: Maximum number of embeddings to analyze.
                           Larger values are more accurate but slower.
        """
        self.graph = graph
        self.max_embeddings = max_embeddings

    async def compute(
        self,
        node_uids: Optional[list[str]] = None,
    ) -> SemanticEntropyResult:
        """Compute semantic entropy for the knowledge graph.

        Args:
            node_uids: Optional list of specific node UIDs to analyze.
                      If None, samples from all nodes with embeddings.

        Returns:
            SemanticEntropyResult with entropy metrics
        """
        # Fetch embeddings
        embeddings = await self._fetch_embeddings(node_uids)

        if len(embeddings) < 2:
            return SemanticEntropyResult(
                semantic_entropy=0.0,
                topic_concentration=1.0,
                effective_dimensions=1.0,
                embedding_count=len(embeddings),
                mean_similarity=1.0,
                min_similarity=1.0,
                max_similarity=1.0,
                is_diverse=False,
                diversity_reason="Insufficient embeddings for entropy calculation",
            )

        # Build similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)

        # Compute spectral entropy
        entropy, effective_dims = self._compute_spectral_entropy(similarity_matrix)

        # Compute similarity statistics
        mean_sim, min_sim, max_sim = self._compute_similarity_stats(similarity_matrix)

        # Topic concentration is inverse of entropy
        topic_concentration = 1.0 - entropy

        # Determine diversity level
        is_diverse, reason = self._assess_diversity(
            entropy, topic_concentration, effective_dims, len(embeddings)
        )

        return SemanticEntropyResult(
            semantic_entropy=entropy,
            topic_concentration=topic_concentration,
            effective_dimensions=effective_dims,
            embedding_count=len(embeddings),
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            is_diverse=is_diverse,
            diversity_reason=reason,
        )

    async def _fetch_embeddings(
        self,
        node_uids: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Fetch embeddings from PsycheClient.

        Returns numpy array of shape (N, D) where N is number of nodes
        and D is embedding dimension.
        """
        try:
            if node_uids:
                # Fetch specific nodes
                query = """
                    MATCH (n)
                    WHERE n.uid IN $uids AND n.embedding IS NOT NULL
                    RETURN n.embedding as embedding
                    LIMIT $limit
                """
                params = {
                    "uids": node_uids[: self.max_embeddings],
                    "limit": self.max_embeddings,
                }
            else:
                # Sample from all nodes with embeddings (InsightZettels have embeddings)
                query = """
                    MATCH (n)
                    WHERE n.embedding IS NOT NULL
                    WITH n, rand() as r
                    RETURN n.embedding as embedding
                    ORDER BY r
                    LIMIT $limit
                """
                params = {"limit": self.max_embeddings}

            result = await self.graph.query(query, params)

            if not result:
                return np.array([])

            # Parse embeddings - handle both list and string formats
            embeddings = []
            for row in result:
                emb = row.get("embedding")
                if emb is not None:
                    if isinstance(emb, str):
                        try:
                            emb = json.loads(emb)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to decode embedding string, skipping: {e}")
                            continue
                    if isinstance(emb, (list, tuple)) and len(emb) > 0:
                        embeddings.append(emb)

            if not embeddings:
                return np.array([])

            # Group embeddings by dimension to handle mixed embedding sizes
            # (e.g., 1024-dim retrieval vs 4096-dim golden embeddings)
            dim_groups: dict[int, list] = {}
            for emb in embeddings:
                dim = len(emb)
                if dim not in dim_groups:
                    dim_groups[dim] = []
                dim_groups[dim].append(emb)

            if not dim_groups:
                return np.array([])

            # Use the largest group (most common dimension)
            largest_dim = max(dim_groups.keys(), key=lambda d: len(dim_groups[d]))
            filtered_embeddings = dim_groups[largest_dim]

            if len(dim_groups) > 1:
                logger.debug(
                    f"Filtered embeddings to {largest_dim}-dim (kept {len(filtered_embeddings)}, "
                    f"dropped {len(embeddings) - len(filtered_embeddings)} of other dimensions)"
                )

            return np.array(filtered_embeddings, dtype=np.float32)

        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # These exceptions indicate data format issues that can legitimately
            # result in no usable embeddings (e.g., malformed rows, missing keys)
            logger.warning(f"Failed to process embeddings due to data error: {e}")
            return np.array([])
        # Note: RuntimeError (database connection issues) and other critical
        # exceptions are intentionally not caught here - they should propagate
        # to signal system faults rather than silently returning empty results

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix from embeddings.

        Args:
            embeddings: (N, D) array of embeddings

        Returns:
            (N, N) similarity matrix with values in [0, 1]
        """
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(normalized, normalized.T)

        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2

        return similarity

    def _compute_spectral_entropy(
        self,
        similarity_matrix: np.ndarray,
    ) -> tuple[float, float]:
        """Compute spectral entropy from similarity matrix eigenvalues.

        The eigenvalues of the normalized similarity matrix represent
        the "variance explained" by each semantic dimension. Entropy
        over this distribution measures semantic diversity.

        Returns:
            Tuple of (normalized_entropy, effective_dimensions)
        """
        n = similarity_matrix.shape[0]

        if n < 2:
            return 0.0, 1.0

        try:
            # Normalize matrix to have unit trace
            trace = np.trace(similarity_matrix)
            if trace > 0:
                normalized_matrix = similarity_matrix / trace
            else:
                return 1.0, float(n)

            # Compute eigenvalues (symmetric matrix, use eigh for efficiency)
            eigenvalues = np.linalg.eigvalsh(normalized_matrix)

            # Filter to positive eigenvalues (numerical stability)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]

            if len(eigenvalues) == 0:
                return 1.0, 1.0

            # Normalize to create probability distribution
            total = np.sum(eigenvalues)
            if total > 0:
                probs = eigenvalues / total
            else:
                return 1.0, 1.0

            # Compute Shannon entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Normalize by maximum entropy (uniform distribution)
            max_entropy = np.log(len(eigenvalues))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Count effective dimensions (eigenvalues above threshold)
            effective_dims = np.sum(probs > EFFECTIVE_DIM_THRESHOLD)

            return float(np.clip(normalized_entropy, 0, 1)), float(effective_dims)

        except np.linalg.LinAlgError as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return 0.5, float(n) / 2  # Return neutral estimate

    def _compute_similarity_stats(
        self,
        similarity_matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute summary statistics of the similarity matrix.

        Returns:
            Tuple of (mean, min, max) similarity values
        """
        n = similarity_matrix.shape[0]
        if n < 2:
            return 1.0, 1.0, 1.0

        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]

        if len(upper_triangle) == 0:
            return 1.0, 1.0, 1.0

        return (
            float(np.mean(upper_triangle)),
            float(np.min(upper_triangle)),
            float(np.max(upper_triangle)),
        )

    def _assess_diversity(
        self,
        entropy: float,
        concentration: float,
        effective_dims: float,
        embedding_count: int,
    ) -> tuple[bool, str]:
        """Assess semantic diversity level.

        Returns:
            Tuple of (is_diverse, reason)
        """
        if embedding_count < 5:
            return (
                False,
                f"Too few embeddings ({embedding_count}) for reliable assessment",
            )

        if entropy > SEMANTIC_ENTROPY_HIGH:
            return (
                True,
                f"High semantic diversity ({entropy:.2f}) across {effective_dims:.0f} dimensions",
            )

        if entropy < SEMANTIC_ENTROPY_LOW:
            return (
                False,
                f"Concentrated topics ({concentration:.1%}) - consider broadening scope",
            )

        if effective_dims < 3:
            return (
                False,
                f"Only {effective_dims:.0f} semantic dimensions - limited variety",
            )

        return (
            True,
            f"Moderate diversity ({entropy:.2f}) with {effective_dims:.0f} dimensions",
        )


async def compute_semantic_entropy(
    graph: "PsycheClient",
    node_uids: Optional[list[str]] = None,
) -> SemanticEntropyResult:
    """Convenience function to compute semantic entropy.

    Usage:
        from core.active_inference.semantic_entropy import compute_semantic_entropy

        result = await compute_semantic_entropy(graph)
        print(f"Semantic diversity: {result.semantic_entropy:.2f}")
    """
    calculator = SemanticEntropyCalculator(graph)
    return await calculator.compute(node_uids)
