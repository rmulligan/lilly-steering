"""Emergent Anchor Discovery Service.

Discovers new cognitive modes that weren't anticipated at design time by
clustering orphan thoughts - thoughts that don't fit any existing anchor well.

Architecture:
    AnchorDiscoveryService maintains an orphan buffer of thoughts with low
    similarity to all anchors. When similar orphans cluster together across
    multiple cycles, they crystallize into new EmergentAnchors.

Usage:
    from core.cognitive.anchor_discovery import AnchorDiscoveryService

    discovery = AnchorDiscoveryService(psyche_client, embedder)

    # Check each thought for orphan status
    orphan = await discovery.check_for_orphan(thought, similarities, embedding, cycle)
    if orphan:
        await discovery.add_orphan(orphan)

    # Periodically check for crystallization
    new_anchor = await discovery.check_for_crystallization()
    if new_anchor:
        # Persist and use the new anchor
        await psyche.upsert_cognitive_anchor(new_anchor.to_cognitive_anchor())

Reference:
    Inspired by PLaT (Latent Chain-of-Thought as Planning) - cognitive modes
    exist as soft attractors, not hard categories. New attractors emerge from
    repeated visits to unexplored regions of thought space.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.cognitive.anchors import (
    AnchorSimilarities,
    EmergentAnchor,
    OrphanThought,
)

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class OrphanCluster:
    """A cluster of similar orphan thoughts.

    Used during crystallization to group orphans that may
    represent a new cognitive mode.

    Attributes:
        members: List of orphan thoughts in this cluster
        centroid: Mean embedding of cluster members
        min_cycle: Earliest cycle in cluster
        max_cycle: Latest cycle in cluster
    """

    members: list[OrphanThought] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    min_cycle: int = 0
    max_cycle: int = 0

    def add(self, orphan: OrphanThought) -> None:
        """Add an orphan to this cluster and update stats."""
        self.members.append(orphan)
        if len(self.members) == 1:
            self.centroid = orphan.thought_embedding.copy()
            self.min_cycle = orphan.cycle_number
            self.max_cycle = orphan.cycle_number
        else:
            # Update centroid as running mean
            self.centroid = (
                self.centroid * (len(self.members) - 1) + orphan.thought_embedding
            ) / len(self.members)
            self.min_cycle = min(self.min_cycle, orphan.cycle_number)
            self.max_cycle = max(self.max_cycle, orphan.cycle_number)

    @property
    def cycle_span(self) -> int:
        """Number of cycles spanned by this cluster."""
        return self.max_cycle - self.min_cycle

    @property
    def size(self) -> int:
        """Number of orphans in this cluster."""
        return len(self.members)


class AnchorDiscoveryService:
    """Discovers emergent cognitive modes from thinking patterns.

    Maintains a buffer of orphan thoughts (thoughts that don't fit any
    existing anchor well) and periodically checks for clustering that
    indicates a new cognitive mode has emerged.

    Constants:
        ORPHAN_THRESHOLD: Similarity below which a thought is an orphan (0.55)
        CLUSTER_THRESHOLD: Orphans must be this similar to cluster (0.7)
        MIN_CLUSTER_SIZE: Need 5+ orphans to crystallize
        MIN_CLUSTER_CYCLE_SPAN: Cluster must span 10+ cycles (temporal diversity)
        MAX_ORPHAN_BUFFER: Prune oldest when exceeded (50)
        MAX_ORPHAN_AGE_CYCLES: Remove orphans older than this (100)
    """

    # Discovery thresholds
    ORPHAN_THRESHOLD = 0.55  # Below this similarity = orphan
    CLUSTER_THRESHOLD = 0.7  # Orphans must be this similar to cluster
    MIN_CLUSTER_SIZE = 5  # Need 5+ orphans to crystallize
    MIN_CLUSTER_CYCLE_SPAN = 10  # Cluster must span 10+ cycles
    MAX_ORPHAN_BUFFER = 50  # Prune oldest when exceeded
    MAX_ORPHAN_AGE_CYCLES = 100  # Remove orphans older than this

    def __init__(
        self,
        psyche: Optional["PsycheClient"] = None,
        embedder: Optional["TieredEmbeddingService"] = None,
    ):
        """Initialize the discovery service.

        Args:
            psyche: PsycheClient for persisting discovered anchors
            embedder: TieredEmbeddingService for computing embeddings
        """
        self._psyche = psyche
        self._embedder = embedder
        self._orphan_buffer: deque[OrphanThought] = deque(maxlen=self.MAX_ORPHAN_BUFFER)
        self._current_cycle: int = 0

    def set_current_cycle(self, cycle: int) -> None:
        """Update the current cycle number for age tracking.

        Args:
            cycle: Current cognitive cycle number
        """
        self._current_cycle = cycle

    async def check_for_orphan(
        self,
        thought: str,
        similarities: AnchorSimilarities,
        thought_embedding: np.ndarray,
        cycle_number: int,
    ) -> Optional[OrphanThought]:
        """Check if a thought is an orphan (doesn't fit existing modes).

        A thought is considered an orphan if its highest similarity to any
        existing anchor is below ORPHAN_THRESHOLD. These orphans are candidates
        for discovering new cognitive modes.

        Args:
            thought: The thought text
            similarities: Computed similarities to all anchors
            thought_embedding: The thought's embedding vector
            cycle_number: Current cognitive cycle number

        Returns:
            OrphanThought if this is an orphan, None otherwise
        """
        if similarities.dominant_score >= self.ORPHAN_THRESHOLD:
            return None  # Fits an existing anchor well enough

        orphan = OrphanThought(
            thought_text=thought[:500],  # Truncate for storage
            thought_embedding=thought_embedding.copy(),
            best_anchor_similarity=similarities.dominant_score,
            cycle_number=cycle_number,
            timestamp=datetime.now(timezone.utc),
        )

        logger.debug(
            f"[ANCHOR_DISCOVERY] Orphan detected (sim={similarities.dominant_score:.2f}): "
            f"{thought[:50]}..."
        )

        return orphan

    async def add_orphan(self, orphan: OrphanThought) -> None:
        """Add an orphan to the buffer.

        Also prunes stale orphans based on age.

        Args:
            orphan: The orphan thought to add
        """
        # Prune stale orphans first
        self._prune_stale_orphans()

        # Add new orphan
        self._orphan_buffer.append(orphan)

        logger.debug(
            f"[ANCHOR_DISCOVERY] Buffer size: {len(self._orphan_buffer)} orphans"
        )

    def _prune_stale_orphans(self) -> None:
        """Remove orphans that are too old (> MAX_ORPHAN_AGE_CYCLES)."""
        if not self._orphan_buffer:
            return

        cutoff_cycle = self._current_cycle - self.MAX_ORPHAN_AGE_CYCLES

        # Orphans are added to the right (append) with monotonically increasing
        # cycle numbers, so we can prune from the left in O(k) instead of O(N)
        pruned_count = 0
        while self._orphan_buffer and self._orphan_buffer[0].cycle_number < cutoff_cycle:
            self._orphan_buffer.popleft()
            pruned_count += 1

        if pruned_count > 0:
            logger.debug(f"[ANCHOR_DISCOVERY] Pruned {pruned_count} stale orphans")

    async def check_for_crystallization(self) -> Optional[EmergentAnchor]:
        """Check if orphans have clustered into a new cognitive mode.

        Uses single-linkage clustering to find groups of similar orphans.
        A cluster must have MIN_CLUSTER_SIZE members spanning MIN_CLUSTER_CYCLE_SPAN
        cycles to crystallize into a new anchor.

        Returns:
            EmergentAnchor if a new mode was discovered, None otherwise
        """
        if len(self._orphan_buffer) < self.MIN_CLUSTER_SIZE:
            return None

        # Find clusters via pairwise similarity
        clusters = self._find_orphan_clusters()

        # Check each cluster for crystallization criteria
        for cluster in clusters:
            if cluster.size >= self.MIN_CLUSTER_SIZE:
                if cluster.cycle_span >= self.MIN_CLUSTER_CYCLE_SPAN:
                    return await self._crystallize_anchor(cluster)
                else:
                    logger.debug(
                        f"[ANCHOR_DISCOVERY] Cluster of {cluster.size} spans only "
                        f"{cluster.cycle_span} cycles (need {self.MIN_CLUSTER_CYCLE_SPAN})"
                    )

        return None

    def _find_orphan_clusters(self) -> list[OrphanCluster]:
        """Find clusters of similar orphans using single-linkage clustering.

        Groups orphans by pairwise similarity above CLUSTER_THRESHOLD.
        Uses Union-Find (disjoint set) to correctly handle transitivity:
        if A is similar to B, and B is similar to C, then A, B, C are
        all in the same cluster even if A and C aren't directly similar.

        Algorithm:
            1. Build a similarity graph: nodes are orphans, edges exist
               between pairs with similarity >= CLUSTER_THRESHOLD
            2. Find connected components using Union-Find
            3. Group orphans by their component into OrphanCluster objects

        Returns:
            List of OrphanCluster objects
        """
        if not self._orphan_buffer:
            return []

        orphan_list = list(self._orphan_buffer)
        n = len(orphan_list)

        # Union-Find data structure with path compression and union by rank
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            """Union by rank."""
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            if rank[root_x] < rank[root_y]:
                root_x, root_y = root_y, root_x
            parent[root_y] = root_x
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

        # Build similarity graph and union similar orphans
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(
                    orphan_list[i].thought_embedding,
                    orphan_list[j].thought_embedding,
                )
                if sim >= self.CLUSTER_THRESHOLD:
                    union(i, j)

        # Group orphans by their connected component
        component_members: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in component_members:
                component_members[root] = []
            component_members[root].append(i)

        # Build OrphanCluster objects from components
        clusters: list[OrphanCluster] = []
        for member_indices in component_members.values():
            cluster = OrphanCluster()
            for idx in member_indices:
                cluster.add(orphan_list[idx])
            clusters.append(cluster)

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        return clusters

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _crystallize_anchor(self, cluster: OrphanCluster) -> EmergentAnchor:
        """Crystallize an orphan cluster into a new EmergentAnchor.

        Selects the orphan closest to the centroid as the exemplar thought,
        generates a description, and creates the new anchor.

        Args:
            cluster: The cluster to crystallize

        Returns:
            New EmergentAnchor representing this cognitive mode
        """
        # Find the orphan closest to centroid (most representative)
        best_orphan = None
        best_sim = -1.0
        for orphan in cluster.members:
            sim = self._cosine_similarity(orphan.thought_embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_orphan = orphan

        if best_orphan is None:
            best_orphan = cluster.members[0]

        # Generate description (simple: use beginning of exemplar thought)
        # TODO: Use Curator to generate richer description in Phase 2
        description = self._generate_mode_description(best_orphan)

        anchor = EmergentAnchor.create(
            anchor_text=best_orphan.thought_text,
            description=description,
            embedding=cluster.centroid,
            source_thought_id=f"orphan_{best_orphan.cycle_number}",
            discovery_cycle=self._current_cycle,
        )

        logger.info(
            f"[ANCHOR_DISCOVERY] Crystallized new anchor '{anchor.mode_name}' "
            f"from {cluster.size} orphans spanning cycles "
            f"{cluster.min_cycle}-{cluster.max_cycle}"
        )

        # Remove crystallized orphans from buffer
        crystallized_set = set(id(o) for o in cluster.members)
        remaining = [o for o in self._orphan_buffer if id(o) not in crystallized_set]
        self._orphan_buffer = deque(remaining, maxlen=self.MAX_ORPHAN_BUFFER)

        # Persist to Psyche if available
        if self._psyche:
            try:
                await self._persist_anchor(anchor)
            except Exception as e:
                logger.warning(f"[ANCHOR_DISCOVERY] Failed to persist anchor: {e}")

        return anchor

    def _generate_mode_description(self, exemplar: OrphanThought) -> str:
        """Generate a description for the emergent cognitive mode.

        Currently uses a simple heuristic: extracts key phrases from
        the exemplar thought. Phase 2 will use Curator for richer
        description generation.

        Args:
            exemplar: The most central orphan thought in the cluster

        Returns:
            Description string for the new mode
        """
        # Take first sentence or first 100 chars of exemplar thought
        text = exemplar.thought_text
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:100] + "..."

        return f"Thinking that explores: {first_sentence}"

    async def _persist_anchor(self, anchor: EmergentAnchor) -> None:
        """Persist an emergent anchor to the Psyche graph.

        Args:
            anchor: The anchor to persist
        """
        from core.psyche.schema import CognitiveAnchor

        cognitive_anchor = CognitiveAnchor(
            uid=anchor.uid,
            mode_name=anchor.mode_name,
            anchor_text=anchor.anchor_text,
            description=anchor.description,
            embedding=anchor.embedding.tolist(),
            is_predefined=False,
            discovered_at=anchor.discovered_at,
            discovery_cycle=anchor.discovery_cycle,
            source_thought_uid=anchor.source_thought_id,
            usage_count=anchor.usage_count,
            confidence=anchor.confidence,
        )

        await self._psyche.upsert_cognitive_anchor(cognitive_anchor)
        logger.info(f"[ANCHOR_DISCOVERY] Persisted anchor {anchor.uid} to Psyche")

    @property
    def orphan_count(self) -> int:
        """Number of orphans currently in buffer."""
        return len(self._orphan_buffer)

    @property
    def orphan_buffer(self) -> list[OrphanThought]:
        """Copy of the current orphan buffer (for inspection/testing)."""
        return list(self._orphan_buffer)
