"""Entity resolution for HippoRAG-style knowledge graph linking.

This module provides intelligent resolution of text mentions to canonical
Entity nodes, enabling the transition from disconnected triples to a
connected knowledge graph.
"""

from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.psyche.schema import Entity

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolve text mentions to canonical Entity nodes.

    This class implements a multi-stage resolution strategy:
    1. Exact match (case-insensitive) via find_entity_by_name()
    2. Fuzzy match using difflib.SequenceMatcher with configurable threshold
    3. Create new entity if no match and create_if_missing=True

    The resolver maintains a per-session cache to avoid redundant database
    lookups when processing batches of triples.

    Example:
        resolver = EntityResolver(psyche_client, similarity_threshold=0.8)

        # Resolve a subject mention
        entity = await resolver.resolve("Machine Learning", entity_type="CONCEPT")

        # Clear cache between batches if needed
        resolver.clear_cache()
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        similarity_threshold: float = 0.8,
    ):
        """Initialize the entity resolver.

        Args:
            psyche: PsycheClient for database operations
            similarity_threshold: Minimum similarity ratio (0-1) for fuzzy matching.
                Higher values require closer matches. Default 0.8.
        """
        self._psyche = psyche
        self._threshold = similarity_threshold
        # Cache maps normalized name -> (Entity or None)
        self._cache: dict[str, Optional["Entity"]] = {}
        # Track cache stats for monitoring
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache hit/miss statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear the resolution cache.

        Call this between processing batches if entity definitions
        may have changed in the database.
        """
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def resolve(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        create_if_missing: bool = True,
    ) -> Optional["Entity"]:
        """Resolve a text mention to a canonical Entity.

        Resolution order:
        1. Check cache for previously resolved name
        2. Exact match (case-insensitive) via find_entity_by_name()
        3. Fuzzy match against existing entities
        4. Create new entity if no match and create_if_missing=True

        Args:
            name: The text mention to resolve (e.g., "machine learning")
            entity_type: Type for new entities (default "CONCEPT")
            create_if_missing: Whether to create a new entity if not found

        Returns:
            The resolved Entity, or None if not found and create_if_missing=False
        """
        normalized = self._normalize(name)

        # Skip empty or very short names
        if len(normalized) < 2:
            return None

        # Check cache first
        if normalized in self._cache:
            self._cache_hits += 1
            cached = self._cache[normalized]
            if cached is not None:
                return cached
            # None in cache means we tried before and found nothing
            if not create_if_missing:
                return None

        self._cache_misses += 1

        # Step 1: Try exact match
        entity = await self._psyche.find_entity_by_name(normalized)
        if entity:
            self._cache[normalized] = entity
            logger.debug(f"Resolved '{name}' to exact match: {entity.name}")
            return entity

        # Step 2: Try fuzzy match
        candidates = await self._psyche.search_entities_fuzzy(normalized, limit=10)
        best_match = self._find_best_fuzzy_match(normalized, candidates)

        if best_match:
            self._cache[normalized] = best_match
            logger.debug(
                f"Resolved '{name}' to fuzzy match: {best_match.name} "
                f"(similarity: {self._similarity(normalized, self._normalize(best_match.name)):.2f})"
            )
            return best_match

        # Step 3: Create new entity if requested
        if create_if_missing:
            entity, created = await self._psyche.get_or_create_entity(
                name=normalized,
                entity_type=entity_type,
            )
            if created:
                logger.debug(f"Created new entity for '{name}': {entity.uid}")
            self._cache[normalized] = entity
            return entity

        # Cache the miss
        self._cache[normalized] = None
        return None

    async def resolve_batch(
        self,
        names: list[str],
        entity_type: str = "CONCEPT",
        create_if_missing: bool = True,
    ) -> dict[str, Optional["Entity"]]:
        """Resolve multiple names efficiently using concurrent resolution.

        Args:
            names: List of text mentions to resolve
            entity_type: Type for new entities
            create_if_missing: Whether to create new entities

        Returns:
            Dict mapping original names to resolved entities
        """
        tasks = [
            self.resolve(name, entity_type=entity_type, create_if_missing=create_if_missing)
            for name in names
        ]
        resolved_entities = await asyncio.gather(*tasks)
        return dict(zip(names, resolved_entities))

    def _normalize(self, name: str) -> str:
        """Normalize a name for matching.

        - Converts to lowercase
        - Strips whitespace
        - Collapses multiple spaces

        Args:
            name: The raw name to normalize

        Returns:
            Normalized name string
        """
        if not name:
            return ""

        # Lowercase and strip
        result = name.lower().strip()

        # Collapse multiple whitespace
        result = re.sub(r"\s+", " ", result)

        return result

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings.

        Uses SequenceMatcher which provides a good balance between
        performance and match quality for entity names.

        Args:
            a: First string (already normalized)
            b: Second string (already normalized)

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _find_best_fuzzy_match(
        self,
        normalized_name: str,
        candidates: list["Entity"],
    ) -> Optional["Entity"]:
        """Find the best fuzzy match from candidates.

        Args:
            normalized_name: The normalized name to match
            candidates: List of candidate entities

        Returns:
            Best matching entity if above threshold, else None
        """
        if not candidates:
            return None

        best_entity = None
        best_score = 0.0

        for candidate in candidates:
            candidate_normalized = self._normalize(candidate.name)
            score = self._similarity(normalized_name, candidate_normalized)

            if score > best_score and score >= self._threshold:
                best_score = score
                best_entity = candidate

        return best_entity
