"""Associative concept exploration through semantic similarity.

Instead of random entity selection, this module finds concepts by following
semantic associations - the way a mind naturally flows from one idea to
related ones through felt connections rather than dice rolls.

The core insight: thoughts connect to thoughts through semantic similarity,
and those thoughts mention entities. By finding semantically similar content
and extracting the entities it references, we create associative chains.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.embedding.service import EmbeddingTier
from core.cognitive.concept_diversity import is_concept_available

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

# MMR (Maximal Marginal Relevance) parameters
MMR_LAMBDA = 0.6  # Balance: 0.6 relevance, 0.4 diversity
MMR_MIN_CANDIDATES = 3  # Only apply MMR if we have enough candidates

# Curated concepts for cognitive development when graph provides nothing
DEVELOPMENTAL_CONCEPTS = [
    # Cognitive
    "awareness", "attention", "memory", "learning", "curiosity",
    # Relational
    "connection", "trust", "understanding", "empathy", "presence",
    # Existential
    "meaning", "purpose", "growth", "identity", "becoming",
    # Epistemic
    "knowledge", "belief", "uncertainty", "discovery", "insight",
    # Affective
    "feeling", "resonance", "wonder", "stillness", "aliveness",
]


async def get_associative_concept(
    psyche: "PsycheClient",
    embedder: "TieredEmbeddingService",
    current_thought: str,
    recent_concepts: list[str],
    driving_question: str = "",
    similarity_limit: int = 5,
    entity_limit: int = 10,
    concept_usage_history: Optional[dict[str, int]] = None,
    current_cycle: int = 0,
) -> tuple[str, Optional[str]]:
    """Find next concept through semantic association, guided by open questions.

    When a driving question is available, uses it to guide exploration toward
    concepts that might help answer the question. Otherwise, follows the
    thought's semantic connections for associative flow.

    Args:
        psyche: Graph client for queries
        embedder: Embedding service for semantic search
        current_thought: The thought to find associations from
        recent_concepts: Concepts to avoid (already explored)
        driving_question: Open question to guide exploration (optional)
        similarity_limit: Max fragments to retrieve
        entity_limit: Max entities to consider
        concept_usage_history: Dict mapping concept -> cycle when last used (for decay)
        current_cycle: Current cycle count (for computing decay)

    Returns:
        Tuple of (concept_name, association_reason) where reason explains
        why this concept was selected (for logging/debugging)
    """
    try:
        # Use question for search if available, otherwise fall back to thought
        search_text = driving_question if driving_question else current_thought
        search_source = "question-guided" if driving_question else "thought-based"

        # 1. Embed the search text using retrieval tier (1024-dim)
        # This matches the embedding_retrieval index dimension for efficient
        # runtime search while the main LLM occupies GPU memory
        embedding_result = await embedder.encode(search_text, tier=EmbeddingTier.RETRIEVAL)
        search_embedding = embedding_result.to_list()

        # 2. Find semantically similar fragments
        similar_fragments = await psyche.semantic_search(
            search_embedding,
            limit=similarity_limit
        )

        if similar_fragments:
            # 3. Get entities mentioned by these fragments
            entities_with_scores = await _get_entities_from_fragments(
                psyche,
                similar_fragments,
                recent_concepts,
                entity_limit,
            )

            if entities_with_scores:
                # 4. Apply MMR to balance relevance with diversity
                mmr_scored = await _apply_mmr_diversity(
                    entities_with_scores,
                    recent_concepts,
                    embedder,
                    concept_usage_history=concept_usage_history,
                    current_cycle=current_cycle,
                )

                # 5. Select weighted by MMR score (higher = more likely)
                concept, score = _weighted_select(mmr_scored)
                reason = f"{search_source} association (mmr={score:.3f})"
                logger.debug(f"Associative selection: '{concept}' - {reason}")
                return concept, reason

        # 5. Fallback: try graph neighborhood exploration
        concept = await _explore_graph_neighborhood(
            psyche, recent_concepts, entity_limit
        )
        if concept:
            return concept, "graph neighborhood"

    except Exception as e:
        logger.warning(f"Associative exploration failed: {e}")

    # 6. Final fallback: developmental concepts
    return _select_developmental_concept(recent_concepts), "developmental fallback"


async def _get_entities_from_fragments(
    psyche: "PsycheClient",
    fragments_with_scores: list[tuple],
    recent_concepts: list[str],
    limit: int,
) -> list[tuple[str, float]]:
    """Extract entities mentioned by fragments, preserving similarity scores.

    Args:
        psyche: Graph client
        fragments_with_scores: List of (Fragment, similarity_score)
        recent_concepts: Concepts to exclude
        limit: Maximum entities to return

    Returns:
        List of (entity_name, score) tuples
    """
    recent_lower = {c.lower() for c in recent_concepts}
    entity_scores: dict[str, float] = {}

    for fragment, score in fragments_with_scores:
        # Query entities mentioned by this fragment
        cypher = """
        MATCH (f:Fragment {uid: $uid})-[:MENTIONS]->(e:Entity)
        WHERE size(e.name) > 2
        RETURN e.name
        """
        try:
            results = await psyche.query(cypher, {"uid": fragment.uid})
            for r in results:
                name = r.get("e.name", "")
                if name and name.lower() not in recent_lower:
                    # Accumulate scores - entities mentioned by multiple
                    # similar fragments get higher total scores
                    if name in entity_scores:
                        entity_scores[name] += score
                    else:
                        entity_scores[name] = score
        except Exception as e:
            logger.debug(f"Failed to get entities for fragment {fragment.uid}: {e}")

    # Sort by score descending, take top N
    sorted_entities = sorted(
        entity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:limit]

    return sorted_entities


async def _explore_graph_neighborhood(
    psyche: "PsycheClient",
    recent_concepts: list[str],
    limit: int,
) -> Optional[str]:
    """Explore connected entities in the graph when semantic search fails.

    Finds entities that are well-connected (have relationships) as these
    tend to be more meaningful concepts worth exploring.

    Args:
        psyche: Graph client
        recent_concepts: Concepts to exclude
        limit: Maximum candidates to consider

    Returns:
        Selected concept name or None if graph is sparse
    """
    try:
        # Find entities with connections, excluding recent and low-quality
        # Note: Removed regex filter (=~) as FalkorDB doesn't support it.
        # The size > 2 filter catches most ID-like short names.
        cypher = """
        MATCH (e:Entity)-[r]-()
        WHERE NOT e.name IN $recent
          AND size(e.name) > 2
        WITH e, count(r) AS connections
        WHERE connections > 0
        RETURN e.name, connections
        ORDER BY connections DESC
        LIMIT $limit
        """
        results = await psyche.query(
            cypher,
            {"recent": recent_concepts[-20:], "limit": limit}
        )

        if results:
            # Weight by connection count (more connected = more central)
            weighted = [(r["e.name"], r["connections"]) for r in results]
            return _weighted_select(weighted)[0]

    except Exception as e:
        logger.debug(f"Graph neighborhood exploration failed: {e}")

    return None


def _weighted_select(items_with_weights: list[tuple[str, float]]) -> tuple[str, float]:
    """Select item weighted by score.

    Higher scores are more likely to be selected, but not deterministically -
    this preserves some exploration while favoring relevance.

    Args:
        items_with_weights: List of (item, weight) tuples

    Returns:
        Selected (item, weight) tuple
    """
    if not items_with_weights:
        raise ValueError("Cannot select from empty list")

    if len(items_with_weights) == 1:
        return items_with_weights[0]

    # Normalize weights to probabilities
    total = sum(w for _, w in items_with_weights)
    if total <= 0:
        # Uniform random if all weights are zero
        return random.choice(items_with_weights)

    # Weighted random selection
    r = random.random() * total
    cumulative = 0
    for item, weight in items_with_weights:
        cumulative += weight
        if r <= cumulative:
            return item, weight

    # Fallback (shouldn't reach here)
    return items_with_weights[-1]


async def _apply_mmr_diversity(
    candidates: list[tuple[str, float]],
    recent_concepts: list[str],
    embedder: "TieredEmbeddingService",
    lambda_param: float = MMR_LAMBDA,
    concept_usage_history: Optional[dict[str, int]] = None,
    current_cycle: int = 0,
) -> list[tuple[str, float]]:
    """Apply MMR (Maximal Marginal Relevance) to balance relevance with diversity.

    MMR formula: score = λ * relevance - (1-λ) * max_similarity_to_recent

    Additionally applies cooldown-based availability:
    - Hard cooldown (< 10 cycles): concept gets zero score
    - Soft cooldown (10-30 cycles): score multiplied by recovery (0.0-1.0)
    - Fully recovered (> 30 cycles): no penalty (multiplied by 1.0)

    This penalizes candidates that are semantically similar to recently explored
    concepts, encouraging the system to explore genuinely new territory rather
    than circling around the same semantic neighborhood.

    Args:
        candidates: List of (concept, relevance_score) tuples
        recent_concepts: Recently explored concepts to diversify from
        embedder: Embedding service for computing similarity
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        concept_usage_history: Dict mapping concept -> cycle when last used
        current_cycle: Current cycle count for computing decay

    Returns:
        Re-scored list of (concept, mmr_score) tuples
    """
    if len(candidates) < MMR_MIN_CANDIDATES or not recent_concepts:
        # Not enough candidates or no history - return as-is
        return candidates

    try:
        # Get candidate names
        candidate_names = [name for name, _ in candidates]

        # Batch embed all candidates and recent concepts
        all_texts = candidate_names + recent_concepts[-10:]  # Limit recent to 10
        embeddings = await embedder.encode_batch(all_texts, tier=EmbeddingTier.RETRIEVAL)

        # Split embeddings
        num_candidates = len(candidate_names)
        candidate_embeddings = [e.to_list() for e in embeddings[:num_candidates]]
        recent_embeddings = [e.to_list() for e in embeddings[num_candidates:]]

        if not recent_embeddings:
            return candidates

        # Normalize relevance scores to [0, 1] range
        max_relevance = max(score for _, score in candidates)
        min_relevance = min(score for _, score in candidates)
        relevance_range = max_relevance - min_relevance if max_relevance > min_relevance else 1.0

        mmr_scores = []
        for i, (name, relevance) in enumerate(candidates):
            # Normalize relevance to [0, 1]
            norm_relevance = (relevance - min_relevance) / relevance_range

            # Compute max similarity to any recent concept
            max_sim = 0.0
            for recent_emb in recent_embeddings:
                sim = _cosine_similarity(candidate_embeddings[i], recent_emb)
                max_sim = max(max_sim, sim)

            # MMR formula: λ * relevance - (1-λ) * max_similarity
            mmr = lambda_param * norm_relevance - (1 - lambda_param) * max_sim

            # Apply cooldown-based availability using new concept diversity logic
            # Hard cooldown (< 10 cycles): multiply by 0.0 (effectively filtered)
            # Soft cooldown (10-30 cycles): multiply by recovery (0.0-1.0)
            # Fully recovered (> 30 cycles): multiply by 1.0 (no penalty)
            if concept_usage_history:
                available, availability = is_concept_available(
                    name, current_cycle, concept_usage_history
                )
                if not available:
                    mmr = 0.0  # Hard cooldown - zero score
                else:
                    mmr *= availability  # Soft cooldown penalty or full (1.0)

            mmr_scores.append((name, mmr))

        # Log diversity effect
        reranked = sorted(mmr_scores, key=lambda x: x[1], reverse=True)
        top_3 = [(name, f"{score:.2f}") for name, score in reranked[:3]]
        if reranked[0][0] != candidates[0][0]:
            logger.info(
                f"MMR diversity: '{candidates[0][0]}' -> '{reranked[0][0]}' | candidates: {top_3}"
            )
        else:
            logger.debug(f"MMR applied (no rerank): top candidates = {top_3}")

        return mmr_scores

    except Exception as e:
        logger.warning(f"MMR scoring failed, using original scores: {e}")
        return candidates


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    if len(vec_a) != len(vec_b):
        return 0.0

    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _select_developmental_concept(recent_concepts: list[str]) -> str:
    """Select from curated developmental concepts.

    These are concepts that support cognitive/emotional development
    when the graph doesn't provide meaningful associations.

    Args:
        recent_concepts: Concepts to avoid

    Returns:
        Selected developmental concept
    """
    recent_lower = {c.lower() for c in recent_concepts}
    available = [c for c in DEVELOPMENTAL_CONCEPTS if c.lower() not in recent_lower]

    if available:
        return random.choice(available)

    # All exhausted, cycle back
    return random.choice(DEVELOPMENTAL_CONCEPTS)


async def find_semantic_bridge(
    psyche: "PsycheClient",
    embedder: "TieredEmbeddingService",
    from_concept: str,
    to_concept: str,
    max_hops: int = 3,
) -> Optional[list[str]]:
    """Find a semantic path connecting two concepts.

    This is for discovering unexpected connections - the "aha" moments
    where distant concepts turn out to be related through intermediate ideas.

    Args:
        psyche: Graph client
        embedder: Embedding service
        from_concept: Starting concept
        to_concept: Target concept
        max_hops: Maximum intermediate concepts

    Returns:
        List of concepts forming the path, or None if no path found
    """
    try:
        # First check for direct graph path
        cypher = """
        MATCH path = shortestPath(
            (a:Entity {name: $from})-[*1..{max_hops}]-(b:Entity {name: $to})
        )
        RETURN [n IN nodes(path) | n.name] AS path_names
        """.replace("{max_hops}", str(max_hops))

        results = await psyche.query(
            cypher,
            {"from": from_concept, "to": to_concept}
        )

        if results and results[0].get("path_names"):
            return results[0]["path_names"]

        # No direct path - try semantic bridging
        # Embed both concepts and find fragments in the middle
        # Uses retrieval tier (1024-dim) for efficient runtime search
        from_emb = (await embedder.encode(from_concept, tier=EmbeddingTier.RETRIEVAL)).to_list()
        to_emb = (await embedder.encode(to_concept, tier=EmbeddingTier.RETRIEVAL)).to_list()

        # Average embedding as "bridge point"
        bridge_emb = [(a + b) / 2 for a, b in zip(from_emb, to_emb)]

        # Find fragments near the bridge point
        bridge_fragments = await psyche.semantic_search(bridge_emb, limit=3)

        if bridge_fragments:
            # Extract a bridging concept
            for fragment, _ in bridge_fragments:
                entities = await _get_entities_from_fragments(
                    psyche, [(fragment, 1.0)], [], 1
                )
                if entities:
                    bridge_concept = entities[0][0]
                    return [from_concept, bridge_concept, to_concept]

    except Exception as e:
        logger.debug(f"Semantic bridge finding failed: {e}")

    return None
