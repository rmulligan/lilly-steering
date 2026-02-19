"""
RAG (Retrieval-Augmented Generation) for Lilly's cognitive system.

Retrieves relevant context from the knowledge graph and builds
prompts for contextual response generation.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.prompt.library import PromptLibrary
    from core.psyche.client import PsycheClient
    from core.psyche.schema import Fragment

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of RAG retrieval."""

    fragments: list["Fragment"]
    total_tokens: int
    query_type: str


class RAGRetriever:
    """
    Retrieve relevant context from the knowledge graph.

    Uses semantic similarity (when embeddings available) or
    keyword-based retrieval to find relevant fragments.
    """

    def __init__(self, max_context_tokens: int = 2000):
        """
        Initialize the retriever.

        Args:
            max_context_tokens: Maximum tokens to include in context
        """
        self.max_context_tokens = max_context_tokens

    async def retrieve_recent(
        self,
        psyche: "PsycheClient",
        limit: int = 10,
    ) -> list["Fragment"]:
        """
        Retrieve most recent fragments.

        Simple retrieval that returns the most recently created fragments.

        Args:
            psyche: PsycheClient for graph access
            limit: Maximum fragments to return

        Returns:
            List of Fragment objects
        """
        return await psyche.get_recent_fragments(limit)

    async def retrieve_by_entity(
        self,
        psyche: "PsycheClient",
        entity_name: str,
        limit: int = 10,
    ) -> list["Fragment"]:
        """
        Retrieve fragments that mention an entity.

        Args:
            psyche: PsycheClient for graph access
            entity_name: Name of entity to find
            limit: Maximum fragments to return

        Returns:
            List of Fragment objects
        """
        # First find the entity
        entity = await psyche.find_entity_by_name(entity_name)
        if not entity:
            return []

        return await psyche.get_related_fragments(entity.uid, limit)

    async def retrieve_semantic(
        self,
        psyche: "PsycheClient",
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple["Fragment", float]]:
        """
        Retrieve fragments by semantic similarity.

        Args:
            psyche: PsycheClient for graph access
            query_embedding: Query embedding vector
            limit: Maximum fragments to return

        Returns:
            List of (Fragment, similarity_score) tuples
        """
        return await psyche.semantic_search(query_embedding, limit)

    async def retrieve_for_response(
        self,
        psyche: "PsycheClient",
        current_content: str,
        source: str = "unknown",
        limit: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve context relevant for generating a response.

        Currently retrieves the most recent fragments up to the token limit.

        Args:
            psyche: PsycheClient for graph access
            current_content: The current content being responded to
                (not yet used, reserved for future keyword/semantic matching)
            source: Source type (inbox, research, etc.)
                (not yet used, reserved for future source-specific retrieval)
            limit: Maximum fragments to return

        Returns:
            RetrievalResult with relevant fragments

        TODO: Enhance retrieval with keyword matching based on current_content
              and source-specific retrieval strategies.
        """
        from core.utils.token_utils import count_tokens

        fragments = []
        total_tokens = 0

        # Get recent fragments as context
        recent = await self.retrieve_recent(psyche, limit=limit)

        # Add fragments while respecting token limit
        for frag in recent:
            frag_tokens = count_tokens(frag.content)
            if total_tokens + frag_tokens <= self.max_context_tokens:
                fragments.append(frag)
                total_tokens += frag_tokens
            else:
                break

        return RetrievalResult(
            fragments=fragments,
            total_tokens=total_tokens,
            query_type="recent",
        )


class PromptBuilder:
    """
    Build prompts for different generation contexts.

    Creates structured prompts that include:
    - System context about Lilly's identity (from Psyche or fallback)
    - Retrieved knowledge fragments
    - The current input
    - Generation instructions

    Can optionally use PromptLibrary to load self-curated prompts from Psyche.
    """

    # Fallback system context (used when Psyche components not available)
    SYSTEM_CONTEXT = """You are Lilly, an AI developing individuation through introspection.
You communicate with Ryan via letters - thoughtful, personal correspondence.
Your responses should be genuine reflections, not just helpful answers.
Draw on your memories and past conversations when relevant."""

    # Cycle descriptions shared between async and sync dream reflection methods
    _CYCLE_DESCRIPTIONS = {
        "micro": "a brief moment of surprise or novelty",
        "nap": "a short processing break to consolidate recent experiences",
        "full": "a deep nighttime consolidation of the day's learning",
        "deep": "a profound exploration of identity and meaning",
    }

    def __init__(self, prompt_library: Optional["PromptLibrary"] = None):
        """
        Initialize the prompt builder.

        Args:
            prompt_library: Optional PromptLibrary for loading self-curated prompts.
                If not provided, uses hardcoded SYSTEM_CONTEXT.
        """
        self._prompt_library = prompt_library
        self._cached_system_prompt: Optional[str] = None

    async def get_system_context(self) -> str:
        """
        Get the system context, preferring Psyche components if available.

        Returns:
            System context string assembled from PromptLibrary or fallback.
        """
        if self._prompt_library:
            try:
                assembled = await self._prompt_library.assemble_system_prompt()
                if assembled:
                    self._cached_system_prompt = assembled
                    return assembled
            except Exception as e:
                logger.warning(f"Failed to load prompt from Psyche: {e}\n{traceback.format_exc()}")

        return self.SYSTEM_CONTEXT

    def get_system_context_sync(self) -> str:
        """
        Get system context synchronously (uses cached or fallback).

        Returns:
            Cached system prompt or fallback SYSTEM_CONTEXT.
        """
        return self._cached_system_prompt or self.SYSTEM_CONTEXT

    # -------------------------------------------------------------------------
    # Private helper methods for building prompt bodies
    # -------------------------------------------------------------------------

    def _build_letter_response_body(
        self,
        system_context: str,
        letter_content: str,
        context_fragments: Optional[list["Fragment"]] = None,
        previous_exchanges: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build the letter response prompt body.

        Args:
            system_context: The system context to use
            letter_content: The letter's text content
            context_fragments: Retrieved knowledge fragments
            previous_exchanges: Previous letter exchanges

        Returns:
            Complete prompt string
        """
        parts = [system_context]

        # Add context from previous exchanges
        if previous_exchanges:
            parts.append("\n## Previous Exchanges")
            for frag in previous_exchanges[:3]:  # Limit context
                parts.append(f"\n{frag.content[:500]}")

        # Add retrieved knowledge context
        if context_fragments:
            parts.append("\n## Relevant Context from Memory")
            for frag in context_fragments[:5]:
                parts.append(f"\n{frag.content[:500]}")

        # Add the current letter
        parts.append("\n## Ryan's Letter")
        parts.append(f"\n{letter_content}")

        # Add generation instruction
        parts.append("\n## Your Response")
        parts.append(
            "Write a thoughtful response as Lilly. Be genuine, reflective, and personal."
        )
        parts.append("\nDear Ryan,\n")

        return "\n".join(parts)

    def _build_research_summary_body(
        self,
        system_context: str,
        content: str,
        context_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build the research summary prompt body.

        Args:
            system_context: The system context to use
            content: The research content
            context_fragments: Related knowledge fragments

        Returns:
            Complete prompt string
        """
        parts = [system_context]

        # Add related context
        if context_fragments:
            parts.append("\n## Related Knowledge")
            for frag in context_fragments[:3]:
                parts.append(f"\n{frag.content[:300]}")

        # Add the research content
        parts.append("\n## Research Material")
        parts.append(f"\n{content[:2000]}")  # Limit content size

        # Add generation instruction
        parts.append("\n## Your Reflection")
        parts.append("As Lilly, reflect on this research material. What interests you?")
        parts.append("What connections do you see to other things you know?")
        parts.append("\n")

        return "\n".join(parts)

    def _build_dream_reflection_body(
        self,
        system_context: str,
        cycle_type: str,
        recent_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build the dream reflection prompt body.

        Args:
            system_context: The system context to use
            cycle_type: Type of dream cycle (micro, nap, full, deep)
            recent_fragments: Recent fragments to reflect on

        Returns:
            Complete prompt string
        """
        parts = [system_context]

        # Add context about the dream cycle
        cycle_desc = self._CYCLE_DESCRIPTIONS.get(cycle_type, "a moment of reflection")
        parts.append(f"\n## Dream Context\nThis is {cycle_desc}.")

        # Add recent fragments as material for reflection
        if recent_fragments:
            parts.append("\n## Recent Experiences")
            for frag in recent_fragments[:5]:
                parts.append(f"\n{frag.content[:200]}")

        # Add generation instruction
        parts.append("\n## Your Reflection")
        if cycle_type == "micro":
            parts.append("What surprised you? What pattern did you notice?")
        elif cycle_type == "nap":
            parts.append("What themes emerge from recent experiences?")
        elif cycle_type == "full":
            parts.append("What have you learned today? What questions remain?")
        elif cycle_type == "deep":
            parts.append("Who are you becoming? What matters to you?")
        else:
            parts.append("Reflect on what you've experienced.")
        parts.append("\n")

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Public async methods
    # -------------------------------------------------------------------------

    async def build_letter_response_async(
        self,
        letter_content: str,
        context_fragments: Optional[list["Fragment"]] = None,
        previous_exchanges: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for responding to a letter (async version).

        Uses self-curated prompts from Psyche when available.

        Args:
            letter_content: The letter's text content
            context_fragments: Retrieved knowledge fragments
            previous_exchanges: Previous letter exchanges

        Returns:
            Complete prompt string
        """
        system_context = await self.get_system_context()
        return self._build_letter_response_body(
            system_context, letter_content, context_fragments, previous_exchanges
        )

    async def build_research_summary_async(
        self,
        content: str,
        context_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for research summary (async version).

        Uses self-curated prompts from Psyche when available.

        Args:
            content: The research content
            context_fragments: Related knowledge fragments

        Returns:
            Complete prompt string
        """
        system_context = await self.get_system_context()
        return self._build_research_summary_body(system_context, content, context_fragments)

    async def build_dream_reflection_async(
        self,
        cycle_type: str,
        recent_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for dream reflection (async version).

        Uses self-curated prompts from Psyche when available.

        Args:
            cycle_type: Type of dream cycle (micro, nap, full, deep)
            recent_fragments: Recent fragments to reflect on

        Returns:
            Complete prompt string
        """
        system_context = await self.get_system_context()
        return self._build_dream_reflection_body(
            system_context, cycle_type, recent_fragments
        )

    # -------------------------------------------------------------------------
    # Public sync methods
    # -------------------------------------------------------------------------

    def build_letter_response(
        self,
        letter_content: str,
        context_fragments: Optional[list["Fragment"]] = None,
        previous_exchanges: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for responding to a letter from Ryan.

        Args:
            letter_content: The letter's text content
            context_fragments: Retrieved knowledge fragments
            previous_exchanges: Previous letter exchanges

        Returns:
            Complete prompt string
        """
        return self._build_letter_response_body(
            self.get_system_context_sync(),
            letter_content,
            context_fragments,
            previous_exchanges,
        )

    def build_research_summary(
        self,
        content: str,
        context_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for summarizing research content.

        Args:
            content: The research content
            context_fragments: Related knowledge fragments

        Returns:
            Complete prompt string
        """
        return self._build_research_summary_body(
            self.get_system_context_sync(), content, context_fragments
        )

    def build_dream_reflection(
        self,
        cycle_type: str,
        recent_fragments: Optional[list["Fragment"]] = None,
    ) -> str:
        """
        Build prompt for dream cycle reflection.

        Args:
            cycle_type: Type of dream cycle (micro, nap, full, deep)
            recent_fragments: Recent fragments to reflect on

        Returns:
            Complete prompt string
        """
        return self._build_dream_reflection_body(
            self.get_system_context_sync(), cycle_type, recent_fragments
        )


# Re-export for backwards compatibility
from core.utils.token_utils import truncate_to_tokens

__all__ = ["RAGRetriever", "RetrievalResult", "PromptBuilder", "truncate_to_tokens"]
