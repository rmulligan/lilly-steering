"""
Insight Zettel Library for knowledge compounding.

This module implements a Zettelkasten-style library of distilled insights
that can be retrieved via multiple pathways:
- Semantic similarity (embedding-based)
- SAE activation overlap (internal resonance)
- Open question matching (forward progress)

The library enables compounding knowledge where insights from past cycles
and external sources flow into future generations.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.client import PsycheClient
    from core.psyche.schema import InsightZettel
    from core.psyche.flow_score import InformationFlowScorer
    from core.cognitive.position_hints import AnnotatedContext, PositionHintComputer

logger = logging.getLogger(__name__)


# Phase-specific retrieval limits (mimics brain timescale adaptation)
# Different cognitive phases need different retrieval depths:
# - Generation: Quick, focused - needs fast context
# - Curation: Analysis phase - moderate depth
# - Simulation: Hypothesis testing - deeper exploration
# - Integration: Consolidation - deepest retrieval
# - Reflexion: Self-reflection - questions important
# - Continuity: Synthesis - moderate depth
PHASE_RETRIEVAL_LIMITS: dict[str, dict[str, int]] = {
    "generation": {
        "semantic_limit": 3,    # Quick, focused
        "activation_limit": 2,
        "question_limit": 1,
    },
    "curation": {
        "semantic_limit": 4,
        "activation_limit": 3,
        "question_limit": 2,
    },
    "simulation": {
        "semantic_limit": 5,
        "activation_limit": 4,
        "question_limit": 3,
    },
    "integration": {
        "semantic_limit": 6,    # Deep consolidation
        "activation_limit": 5,
        "question_limit": 3,
    },
    "reflexion": {
        "semantic_limit": 5,
        "activation_limit": 4,
        "question_limit": 4,    # Questions important for reflection
    },
    "continuity": {
        "semantic_limit": 4,
        "activation_limit": 3,
        "question_limit": 2,
    },
}

# Recency boost factors by phase timescale
# Fast phases (generation) boost recent zettels more strongly
# Slow phases (integration) value all timeframes equally (consolidation)
# This mimics brain cross-timescale integration
PHASE_RECENCY_BOOST: dict[str, float] = {
    "generation": 0.3,    # Strong recency preference (fast processing)
    "curation": 0.2,
    "simulation": 0.1,
    "integration": 0.05,  # Weak recency (consolidation values all time)
    "reflexion": 0.1,
    "continuity": 0.15,
}

# Flow score weight by phase
# How much connectivity/hub status influences retrieval ranking
# Integration phase cares most about high-connectivity concepts (consolidation)
# Generation phase cares least (fast, doesn't need hub info)
PHASE_FLOW_WEIGHT: dict[str, float] = {
    "generation": 0.1,    # Minimal flow influence (fast phase)
    "curation": 0.15,
    "simulation": 0.2,
    "integration": 0.3,   # Strong flow influence (consolidation values hubs)
    "reflexion": 0.25,  # Self-monitoring benefits from hub awareness
    "continuity": 0.2,
}


@dataclass
class RetrievedInsight:
    """An insight retrieved via one of the retrieval paths."""

    zettel: "InsightZettel"
    score: float  # Similarity or overlap score
    retrieval_path: str  # "semantic", "activation", or "question"


@dataclass
class RetrievedContext:
    """Context retrieved from the Zettel library.

    Combines results from all three retrieval paths along with
    immediate context (breadcrumb trail and thread summary).
    """

    semantic_insights: list[tuple[str, str, float]] = field(default_factory=list)
    """(uid, insight_text, similarity) tuples from semantic retrieval"""

    activated_insights: list[tuple[str, str, float]] = field(default_factory=list)
    """(uid, insight_text, activation) tuples from SAE feature overlap"""

    open_questions: list[tuple[str, str, float]] = field(default_factory=list)
    """(uid, question_text, similarity) tuples from open question matching"""

    breadcrumb_trail: str = ""
    """Recent concept progression: 'surface vs depth → holding shadows'"""

    thread_summary: str = ""
    """Summary of recent exploration thread"""

    # Position hints (computed after retrieval)
    _annotated: Optional["AnnotatedContext"] = field(default=None, repr=False)
    """Computed position hints for graph-aware prompt formatting"""

    async def compute_position_hints(
        self,
        hint_computer: "PositionHintComputer",
        current_concept: str,
        current_sae_features: Optional[list[int]] = None,
    ) -> None:
        """Compute graph-aware position hints for all retrieved insights.

        This should be called after retrieval to annotate insights with
        semantic proximity metadata for better LLM attention.

        Args:
            hint_computer: PositionHintComputer instance with psyche access
            current_concept: Current exploration concept
            current_sae_features: Active SAE feature indices (optional)
        """
        self._annotated = await hint_computer.compute_for_context(
            semantic_insights=self.semantic_insights,
            activated_insights=self.activated_insights,
            open_questions=self.open_questions,
            current_concept=current_concept,
            current_sae_features=current_sae_features,
        )

    def format_for_prompt(self) -> str:
        """Format all retrieved context for prompt injection.

        Returns formatted text ready for inclusion in the narrative prompt,
        organized by source type with appropriate framing.

        If position hints have been computed via compute_position_hints(),
        insights will include bracketed annotations like [concept: consciousness].
        """
        sections = []

        # Breadcrumb trail and thread summary (always present if available)
        if self.breadcrumb_trail:
            sections.append(f"Recent thread: {self.breadcrumb_trail}")

        if self.thread_summary:
            sections.append(self.thread_summary)

        # Semantic insights - "I recall thinking..."
        if self.semantic_insights:
            if self._annotated and self._annotated.semantic:
                recall_lines = [
                    f"- {ai.format_with_hints()}" for ai in self._annotated.semantic
                ]
            else:
                recall_lines = [f'- "{text}"' for _, text, _ in self.semantic_insights]
            sections.append("I recall thinking:\n" + "\n".join(recall_lines))

        # Activated insights - "Something stirs beneath thought..."
        if self.activated_insights:
            if self._annotated and self._annotated.activated:
                stir_lines = [
                    f"- {ai.format_with_hints()}" for ai in self._annotated.activated
                ]
            else:
                stir_lines = [f'- "{text}"' for _, text, _ in self.activated_insights]
            sections.append(
                "Something stirs beneath thought:\n" + "\n".join(stir_lines)
            )

        # Open questions - "An old question lingers..."
        if self.open_questions:
            if self._annotated and self._annotated.questions:
                question_lines = [
                    f"- {ai.format_with_hints()}" for ai in self._annotated.questions
                ]
            else:
                question_lines = [f'- "{text}"' for _, text, _ in self.open_questions]
            sections.append(
                "An old question lingers:\n" + "\n".join(question_lines)
            )

        return "\n\n".join(sections)

    def get_retrieved_zettel_uids(self) -> list[str]:
        """Get UIDs of all retrieved zettels for lineage tracking."""
        # Use set for O(1) deduplication instead of O(n) list lookup
        uids_set: set[str] = set()
        for uid, _, _ in self.semantic_insights:
            uids_set.add(uid)
        for uid, _, _ in self.activated_insights:
            uids_set.add(uid)
        for uid, _, _ in self.open_questions:
            uids_set.add(uid)
        return list(uids_set)

    def get_open_question_uids(self) -> list[str]:
        """Get UIDs of retrieved open questions for ADDRESSED_BY tracking."""
        return [uid for uid, _, _ in self.open_questions]


class ZettelLibrary:
    """Manages insight storage and multi-path retrieval.

    The ZettelLibrary is the central hub for Lilly's compounding knowledge system.
    It stores distilled insights from cognitive cycles and external sources,
    and retrieves relevant context via three parallel pathways.

    Attributes:
        psyche: PsycheClient for graph storage
        embedding_service: Service for computing embeddings
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        embedding_service: "TieredEmbeddingService",
    ):
        """Initialize the Zettel library.

        Args:
            psyche: PsycheClient for graph storage
            embedding_service: Service for computing embeddings
        """
        self.psyche = psyche
        self.embedding_service = embedding_service
        self._recent_zettels: deque["InsightZettel"] = deque(maxlen=10)
        self._recent_concepts: deque[str] = deque(maxlen=10)
        # Temporal cache to prevent duplicate retrievals within cooldown window
        # Bounded to max 100 entries to prevent unbounded memory growth
        self._retrieval_cache: dict[str, float] = {}  # uid -> timestamp
        self._retrieval_cache_maxlen: int = 100
        self._retrieval_cooldown_seconds: float = 60.0  # Skip zettels retrieved within 60s
        # Flow score integration for connectivity-weighted retrieval
        self._flow_scorer: Optional["InformationFlowScorer"] = None
        self._flow_cache: dict[str, float] = {}  # concept -> flow_score

    @property
    def recent_concepts(self) -> list[str]:
        """Get list of recently explored concepts."""
        return list(self._recent_concepts)

    def _is_recently_retrieved(self, uid: str) -> bool:
        """Check if a zettel was retrieved within the cooldown window.

        Args:
            uid: Zettel UID to check

        Returns:
            True if zettel was retrieved within cooldown window
        """
        if uid not in self._retrieval_cache:
            return False
        elapsed = time.time() - self._retrieval_cache[uid]
        return elapsed < self._retrieval_cooldown_seconds

    def _mark_retrieved(self, uid: str) -> None:
        """Mark a zettel as retrieved at current time.

        Args:
            uid: Zettel UID to mark
        """
        self._retrieval_cache[uid] = time.time()
        # Evict oldest entries if cache exceeds max size
        if len(self._retrieval_cache) > self._retrieval_cache_maxlen:
            # Sort by timestamp and remove oldest entries
            sorted_entries = sorted(self._retrieval_cache.items(), key=lambda x: x[1])
            entries_to_remove = len(self._retrieval_cache) - self._retrieval_cache_maxlen
            for uid_to_remove, _ in sorted_entries[:entries_to_remove]:
                del self._retrieval_cache[uid_to_remove]

    def _cleanup_cache(self) -> None:
        """Remove expired entries from the retrieval cache."""
        current_time = time.time()
        expired = [
            uid for uid, ts in self._retrieval_cache.items()
            if current_time - ts >= self._retrieval_cooldown_seconds
        ]
        for uid in expired:
            del self._retrieval_cache[uid]

    def _get_zettel_cycle(self, uid: str) -> Optional[int]:
        """Get the cycle number for a zettel from local cache.

        Args:
            uid: Zettel UID to look up

        Returns:
            Cycle number if found in cache, None otherwise
        """
        for zettel in self._recent_zettels:
            if zettel.uid == uid:
                return zettel.cycle
        return None

    def _apply_recency_boost(
        self,
        results: list[tuple[str, str, float]],
        phase: str,
        current_cycle: int,
    ) -> list[tuple[str, str, float]]:
        """Apply recency boost to retrieval results based on phase timescale.

        Fast phases (generation) boost recent insights more strongly.
        Slow phases (integration) have minimal recency bias, valuing
        all timeframes for consolidation.

        Args:
            results: List of (uid, text, score) tuples from retrieval
            phase: Cognitive phase name
            current_cycle: Current cognitive cycle number

        Returns:
            Re-sorted list of (uid, text, boosted_score) tuples
        """
        boost_factor = PHASE_RECENCY_BOOST.get(phase, 0.1)

        boosted = []
        for uid, text, score in results:
            # Get zettel cycle from metadata or default to None
            zettel_cycle = self._get_zettel_cycle(uid)

            if zettel_cycle is not None and current_cycle > 0:
                # Recency as fraction of current cycle (0.0 = oldest, 1.0 = most recent)
                recency = 1.0 - (current_cycle - zettel_cycle) / max(current_cycle, 1)
                recency = max(0.0, min(1.0, recency))  # Clamp to [0, 1]

                # Boost = base_score + (recency * boost_factor)
                boosted_score = score + (recency * boost_factor)
            else:
                boosted_score = score

            boosted.append((uid, text, boosted_score))

        # Re-sort by boosted score (descending)
        boosted.sort(key=lambda x: x[2], reverse=True)
        return boosted

    def set_flow_scorer(self, scorer: "InformationFlowScorer") -> None:
        """Set the flow scorer for connectivity-weighted retrieval.

        Args:
            scorer: InformationFlowScorer instance for computing flow scores.
        """
        self._flow_scorer = scorer

    async def _get_concept_flow_score(self, concept: str) -> float:
        """Get flow score for a concept, using cache if available.

        Args:
            concept: Concept name to score.

        Returns:
            Flow score (0.5 if no scorer available, cached value if present).
        """
        if self._flow_scorer is None:
            return 0.5  # Neutral score when no scorer

        if concept in self._flow_cache:
            return self._flow_cache[concept]

        score = await self._flow_scorer.compute_flow_score(concept)
        self._flow_cache[concept] = score
        return score

    def _get_zettel_concept(self, uid: str) -> Optional[str]:
        """Get the concept for a zettel from local cache.

        Args:
            uid: Zettel UID to look up

        Returns:
            Concept if found in cache, None otherwise
        """
        for zettel in self._recent_zettels:
            if zettel.uid == uid:
                return zettel.concept
        return None

    def _apply_flow_boost(
        self,
        results: list[tuple[str, str, float]],
        phase: str,
    ) -> list[tuple[str, str, float]]:
        """Apply flow score boost to retrieval results based on concept connectivity.

        High-connectivity concepts (hubs) get boosted during consolidation phases.
        The boost formula is: boosted_score = score * (1.0 + flow_weight * (flow_score - 0.5))

        This means:
        - flow_score > 0.5: boost increases score
        - flow_score = 0.5: no change (neutral)
        - flow_score < 0.5: reduces score (penalty for low-connectivity)

        Args:
            results: List of (uid, text, score) tuples from retrieval
            phase: Cognitive phase name

        Returns:
            Re-sorted list of (uid, text, boosted_score) tuples
        """
        flow_weight = PHASE_FLOW_WEIGHT.get(phase, 0.1)

        boosted = []
        for uid, text, score in results:
            # Get zettel concept from cache
            concept = self._get_zettel_concept(uid)

            if concept is not None and concept in self._flow_cache:
                flow_score = self._flow_cache[concept]
                # boost = 1.0 + flow_weight * (flow_score - 0.5)
                boost = 1.0 + flow_weight * (flow_score - 0.5)
                boosted_score = score * boost
            else:
                boosted_score = score

            boosted.append((uid, text, boosted_score))

        # Re-sort by boosted score (descending)
        boosted.sort(key=lambda x: x[2], reverse=True)
        return boosted

    async def store_zettel(
        self,
        insight_text: str,
        source_type: str,
        source_uid: str,
        concept: str,
        question_text: Optional[str] = None,
        cycle: Optional[int] = None,
        sae_features: Optional[list[tuple[int, float]]] = None,
        emerged_from: Optional[list[str]] = None,
    ) -> "InsightZettel":
        """Store a new insight in the Zettel library.

        Creates an InsightZettel node with embedding, links to source fragment
        and SAE features, and establishes lineage via EMERGED_FROM edges.

        Args:
            insight_text: The distilled insight (1-2 sentences)
            source_type: "cognitive", "letter", "research", or "reflection"
            source_uid: Fragment UID of the full source
            concept: Concept being explored when insight emerged
            question_text: Optional open question that emerged
            cycle: Cognitive cycle number (None for external sources)
            sae_features: Optional list of (feature_idx, activation) tuples
            emerged_from: Optional list of parent zettel UIDs for lineage

        Returns:
            The created InsightZettel
        """
        from core.embedding.service import EmbeddingTier
        from core.psyche.schema import InsightSourceType, InsightZettel, QuestionStatus

        # Generate UID
        uid = f"zettel_{uuid4().hex[:12]}"

        # Compute embedding for retrieval
        embedding_result = await self.embedding_service.encode(
            insight_text, tier=EmbeddingTier.RETRIEVAL
        )
        embedding = embedding_result.to_list()

        # Extract top SAE feature indices for fast lookup
        sae_feature_indices = []
        if sae_features:
            # Sort by activation and take top 10
            sorted_features = sorted(sae_features, key=lambda x: x[1], reverse=True)
            sae_feature_indices = [idx for idx, _ in sorted_features[:10]]

        # Create the zettel
        zettel = InsightZettel(
            uid=uid,
            insight_text=insight_text,
            question_text=question_text,
            question_status=QuestionStatus.OPEN if question_text else QuestionStatus.DISSOLVED,
            source_type=InsightSourceType(source_type),
            source_uid=source_uid,
            concept=concept,
            cycle=cycle,
            embedding=embedding,
            sae_feature_indices=sae_feature_indices,
            created_at=datetime.now(tz=timezone.utc),
        )

        # Store in graph
        created = await self.psyche.create_zettel(zettel)
        if not created:
            logger.warning(f"Failed to create zettel {uid}")
            return zettel

        # Link to source fragment
        await self.psyche.link_zettel_to_fragment(uid, source_uid)

        # Link to SAE snapshot if available
        if sae_features:
            await self.psyche.link_zettel_to_sae_snapshot(uid, source_uid)

        # Establish lineage
        if emerged_from:
            for parent_uid in emerged_from:
                await self.psyche.link_zettel_emerged_from(uid, parent_uid)

        # Update local cache
        self._recent_zettels.append(zettel)
        if concept not in self._recent_concepts:
            self._recent_concepts.append(concept)

        logger.debug(f"Stored zettel {uid}: {insight_text[:50]}...")
        return zettel

    async def retrieve_context(
        self,
        concept: str,
        current_embedding: Optional[list[float]] = None,
        sae_features: Optional[list[tuple[int, float]]] = None,
        phase: str = "generation",
        current_cycle: int = 0,
        semantic_limit: Optional[int] = None,
        activation_limit: Optional[int] = None,
        question_limit: Optional[int] = None,
        use_flow_scores: bool = False,
    ) -> RetrievedContext:
        """Retrieve relevant context from the Zettel library.

        Combines results from three parallel retrieval paths:
        1. Semantic: Embedding similarity to current context
        2. Activation: SAE feature overlap (internal resonance)
        3. Questions: Open questions semantically related to current context

        Retrieval limits adapt to cognitive phase timescale - quick for
        generation, deep for integration/consolidation. Results are also
        boosted by recency based on phase timescale - fast phases favor
        recent insights, slow phases value all timeframes equally.

        When use_flow_scores is True, results are additionally boosted by
        concept connectivity - zettels about high-connectivity concepts
        (information hubs) get higher scores during consolidation phases.

        Args:
            concept: Current exploration concept
            current_embedding: Embedding of current context for semantic retrieval
            sae_features: Active SAE features for activation-based retrieval
            phase: Cognitive phase ("generation", "curation", "simulation",
                   "integration", "reflexion", "continuity"). Determines default limits
                   and recency boost strength.
            current_cycle: Current cognitive cycle number for recency calculation
            semantic_limit: Override phase default for semantic results
            activation_limit: Override phase default for activation results
            question_limit: Override phase default for open questions
            use_flow_scores: If True, apply connectivity-weighted flow boost

        Returns:
            RetrievedContext with results from all paths and formatting
        """
        # Get phase-specific limits, falling back to generation defaults
        phase_limits = PHASE_RETRIEVAL_LIMITS.get(
            phase, PHASE_RETRIEVAL_LIMITS["generation"]
        )

        # Use explicit overrides if provided, otherwise use phase defaults
        effective_semantic_limit = (
            semantic_limit if semantic_limit is not None
            else phase_limits["semantic_limit"]
        )
        effective_activation_limit = (
            activation_limit if activation_limit is not None
            else phase_limits["activation_limit"]
        )
        effective_question_limit = (
            question_limit if question_limit is not None
            else phase_limits["question_limit"]
        )
        # Clean up expired cache entries at the start of each retrieval
        self._cleanup_cache()

        semantic_insights = []
        activated_insights = []
        open_questions = []

        # Path 1: Semantic retrieval
        if current_embedding:
            try:
                # Request extra results to account for filtering (temporal + empty)
                results = await self.psyche.query_zettels_by_embedding(
                    embedding=current_embedding,
                    limit=effective_semantic_limit * 3,  # Increased to account for temporal filtering
                )
                for z, score in results:
                    # Skip zettels with empty insight_text (legacy data)
                    if not z.insight_text:
                        continue
                    # Skip recently retrieved zettels to prevent repetition
                    if self._is_recently_retrieved(z.uid):
                        logger.debug(f"Skipping recently retrieved zettel {z.uid}")
                        continue
                    semantic_insights.append((z.uid, z.insight_text, score))
                    self._mark_retrieved(z.uid)
                    if len(semantic_insights) >= effective_semantic_limit:
                        break
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")

        # Path 2: Activation-based retrieval (SAE coactivation)
        if sae_features:
            try:
                feature_indices = [idx for idx, _ in sae_features]
                results = await self.psyche.query_zettels_by_sae_features(
                    feature_indices=feature_indices,
                    limit=effective_activation_limit * 3,  # Increased to account for filtering
                )
                count = 0
                for zettel, overlap in results:
                    # Skip empty insights
                    if not zettel.insight_text:
                        continue
                    # Dedupe against semantic results
                    if zettel.uid in [uid for uid, _, _ in semantic_insights]:
                        continue
                    # Skip recently retrieved zettels
                    if self._is_recently_retrieved(zettel.uid):
                        logger.debug(f"Skipping recently retrieved zettel {zettel.uid}")
                        continue
                    activated_insights.append((zettel.uid, zettel.insight_text, overlap))
                    self._mark_retrieved(zettel.uid)
                    count += 1
                    if count >= effective_activation_limit:
                        break
            except Exception as e:
                logger.warning(f"Activation retrieval failed: {e}")

        # Path 3: Open questions
        if current_embedding:
            try:
                results = await self.psyche.query_open_questions(
                    embedding=current_embedding,
                    limit=effective_question_limit * 2,  # Extra for filtering
                )
                count = 0
                for z, score in results:
                    if not z.question_text:
                        continue
                    # Dedupe against previous paths
                    if z.uid in [uid for uid, _, _ in semantic_insights]:
                        continue
                    if z.uid in [uid for uid, _, _ in activated_insights]:
                        continue
                    # Skip recently retrieved zettels
                    if self._is_recently_retrieved(z.uid):
                        continue
                    open_questions.append((z.uid, z.question_text, score))
                    self._mark_retrieved(z.uid)
                    count += 1
                    if count >= effective_question_limit:
                        break
            except Exception as e:
                logger.warning(f"Open questions retrieval failed: {e}")

        # Apply recency boost to semantic insights based on phase timescale
        if current_cycle > 0:
            semantic_insights = self._apply_recency_boost(
                semantic_insights, phase, current_cycle
            )

        # Apply flow score boost when enabled
        if use_flow_scores and self._flow_scorer is not None:
            # Pre-compute flow scores for concepts in results
            concepts_to_fetch = {self._get_zettel_concept(uid) for uid, _, _ in semantic_insights}
            tasks = [
                self._get_concept_flow_score(concept)
                for concept in concepts_to_fetch
                if concept and concept not in self._flow_cache
            ]
            if tasks:
                await asyncio.gather(*tasks)

            # Apply flow boost to semantic insights
            semantic_insights = self._apply_flow_boost(semantic_insights, phase)

        # Build breadcrumb trail from recent concepts
        breadcrumb_trail = self.get_breadcrumb_trail()

        # Generate thread summary from recent zettels
        thread_summary = await self.generate_thread_summary()

        return RetrievedContext(
            semantic_insights=semantic_insights,
            activated_insights=activated_insights,
            open_questions=open_questions,
            breadcrumb_trail=breadcrumb_trail,
            thread_summary=thread_summary,
        )

    async def mark_question_addressed(
        self,
        question_uid: str,
        answer_uid: str,
    ) -> bool:
        """Mark an open question as addressed by a new insight.

        Creates ADDRESSED_BY relationship and updates question status.

        Args:
            question_uid: UID of the InsightZettel with open question
            answer_uid: UID of the InsightZettel that addresses it

        Returns:
            True if successfully updated
        """
        return await self.psyche.link_zettel_addressed_by(question_uid, answer_uid)

    def get_breadcrumb_trail(self, limit: int = 5) -> str:
        """Generate breadcrumb trail from recent concepts.

        Returns a string like: 'surface vs depth → holding shadows → emergence'

        Args:
            limit: Maximum number of concepts to include

        Returns:
            Formatted breadcrumb trail string
        """
        if not self._recent_concepts:
            return ""

        concepts = list(self._recent_concepts)[-limit:]
        return " → ".join(concepts)

    async def generate_thread_summary(self) -> str:
        """Generate summary of recent exploration thread.

        Analyzes recent zettels to create a brief summary of the
        current intellectual thread.

        Returns:
            Summary string describing recent exploration
        """
        if not self._recent_zettels:
            return ""

        # For now, create a simple summary from recent insights
        recent = list(self._recent_zettels)[-3:]
        if not recent:
            return ""

        # Extract key themes
        insights = [z.insight_text for z in recent]

        if len(insights) == 1:
            return f"I've been exploring: {insights[0]}"

        # Simple concatenation for now - could be enhanced with LLM summarization
        return f"I've been circling: {insights[-1]}"

    async def load_recent_from_graph(self, limit: int = 10) -> int:
        """Load recent zettels from graph into local cache.

        Useful for warming up the cache after initialization.

        Args:
            limit: Maximum zettels to load

        Returns:
            Number of zettels loaded
        """
        try:
            zettels = await self.psyche.get_recent_zettels(limit=limit)
            for z in reversed(zettels):  # Add oldest first to maintain order
                self._recent_zettels.append(z)
                if z.concept and z.concept not in self._recent_concepts:
                    self._recent_concepts.append(z.concept)
            return len(zettels)
        except Exception as e:
            logger.warning(f"Failed to load recent zettels: {e}")
            return 0
