"""Persistence layer for ReflexionEntry records.

This module provides the ReflexionJournal class for persisting reflexion
entries to the Psyche knowledge graph, including FOLLOWS relationships
between consecutive entries and MODIFIED relationships to track parameter
changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

from core.cognitive.reflexion.schemas import (
    Modification,
    ModificationTier,
    ReflexionEntry,
    ReflexionResult,
)


import uuid


def _generate_uid() -> str:
    """Generate a unique identifier for a reflexion entry.

    Returns:
        A UID in the format "reflex_{timestamp}" where timestamp is
        YYYYMMDDHHMMSSfff (17 characters) plus a short unique suffix
        to ensure uniqueness even for rapid consecutive calls.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]
    # Add short uuid suffix to ensure uniqueness
    suffix = uuid.uuid4().hex[:4]
    return f"reflex_{timestamp}_{suffix}"


class ReflexionJournal:
    """Persists ReflexionEntry records to Psyche with relationships.

    Creates ReflexionEntry nodes in the graph with:
    - FOLLOWS relationships to previous entries (by cycle_number)
    - MODIFIED relationships to CognitiveParameter or PromptComponent nodes

    Attributes:
        _psyche: Client for graph operations
    """

    def __init__(self, psyche: "PsycheClient"):
        """Initialize journal with Psyche client.

        Args:
            psyche: Client for graph persistence operations
        """
        self._psyche = psyche

    async def create_entry(
        self,
        result: ReflexionResult,
        cycle_number: int,
        metrics_snapshot: dict[str, float],
        baseline_comparison: Optional[dict[str, float]] = None,
        phenomenological: Optional[dict[str, float]] = None,
    ) -> ReflexionEntry:
        """Create and persist a ReflexionEntry from a ReflexionResult.

        Args:
            result: The ReflexionResult containing health assessment and modifications
            cycle_number: The cognitive cycle number when reflexion occurred
            metrics_snapshot: Current metrics values at time of reflexion
            baseline_comparison: Optional comparison to baseline values
            phenomenological: Optional phenomenological signal values

        Returns:
            The created ReflexionEntry with UID and timestamp
        """
        # Extract overall coherence from health assessment
        overall_coherence = result.health_assessment.coherence.value

        # Convert skipped modifications from tuple format to Modification format
        # The ReflexionResult stores skipped as list[tuple[str, str]] (path, reason)
        # but ReflexionEntry expects list[Modification]
        # For now, we'll store an empty list since the schema expects Modification objects
        modifications_skipped: list[Modification] = []

        entry = ReflexionEntry(
            uid=_generate_uid(),
            cycle_number=cycle_number,
            timestamp=datetime.now(timezone.utc),
            baseline_comparison=baseline_comparison or {},
            phenomenological=phenomenological or {},
            modifications=result.modifications,
            modifications_skipped=modifications_skipped,
            overall_coherence=overall_coherence,
            narrative_summary=result.narrative_summary,
        )

        await self._persist_entry(entry)
        await self._link_to_previous(entry)
        await self._link_modifications(entry)

        return entry

    async def _persist_entry(self, entry: ReflexionEntry) -> None:
        """Persist ReflexionEntry node to Psyche.

        Creates a ReflexionEntry node with all fields serialized appropriately
        for graph storage.

        Args:
            entry: The entry to persist
        """
        cypher = """
        CREATE (e:ReflexionEntry {
            uid: $uid,
            cycle_number: $cycle_number,
            timestamp: $timestamp,
            baseline_comparison: $baseline_comparison,
            phenomenological: $phenomenological,
            modifications: $modifications,
            modifications_skipped: $modifications_skipped,
            overall_coherence: $overall_coherence,
            narrative_summary: $narrative_summary
        })
        """

        props = entry.to_cypher_props()
        await self._psyche.execute(cypher, props)

    async def _link_to_previous(self, entry: ReflexionEntry) -> None:
        """Create FOLLOWS relationship to previous entry.

        Finds the entry with cycle_number - 1 and creates a FOLLOWS
        relationship from previous to current.

        Args:
            entry: The current entry
        """
        # No previous for cycle 0
        if entry.cycle_number <= 0:
            return

        previous_cycle = entry.cycle_number - 1

        # Find previous entry
        query_cypher = """
        MATCH (prev:ReflexionEntry {cycle_number: $prev_cycle})
        RETURN prev.uid AS uid
        LIMIT 1
        """

        results = await self._psyche.query(
            query_cypher,
            {"prev_cycle": previous_cycle},
        )

        if not results:
            return

        prev_uid = results[0].get("uid")
        if not prev_uid:
            return

        # Create FOLLOWS relationship
        link_cypher = """
        MATCH (prev:ReflexionEntry {uid: $prev_uid})
        MATCH (curr:ReflexionEntry {uid: $curr_uid})
        CREATE (prev)-[:FOLLOWS]->(curr)
        """

        await self._psyche.execute(
            link_cypher,
            {"prev_uid": prev_uid, "curr_uid": entry.uid},
        )

    async def _link_modifications(self, entry: ReflexionEntry) -> None:
        """Create MODIFIED relationships for each modification.

        For CONFIG tier: creates relationship to CognitiveParameter
        For PROMPT tier: creates relationship to PromptComponent

        Args:
            entry: The entry containing modifications
        """
        for mod in entry.modifications:
            if mod.tier == ModificationTier.CONFIG:
                await self._link_config_modification(entry, mod)
            elif mod.tier == ModificationTier.PROMPT:
                await self._link_prompt_modification(entry, mod)
            # RUNTIME modifications are transient and not linked

    async def _link_config_modification(
        self,
        entry: ReflexionEntry,
        mod: Modification,
    ) -> None:
        """Create MODIFIED relationship to CognitiveParameter.

        Args:
            entry: The reflexion entry
            mod: The CONFIG tier modification
        """
        cypher = """
        MATCH (e:ReflexionEntry {uid: $entry_uid})
        MERGE (p:CognitiveParameter {path: $parameter_path})
        CREATE (e)-[:MODIFIED {
            old_value: $old_value,
            new_value: $new_value,
            rationale: $rationale,
            confidence: $confidence,
            timestamp: $timestamp
        }]->(p)
        """

        import json

        await self._psyche.execute(
            cypher,
            {
                "entry_uid": entry.uid,
                "parameter_path": mod.parameter_path,
                "old_value": json.dumps(mod.old_value),
                "new_value": json.dumps(mod.new_value),
                "rationale": mod.rationale,
                "confidence": mod.confidence,
                "timestamp": entry.timestamp.isoformat(),
            },
        )

    async def _link_prompt_modification(
        self,
        entry: ReflexionEntry,
        mod: Modification,
    ) -> None:
        """Create MODIFIED relationship to PromptComponent.

        Args:
            entry: The reflexion entry
            mod: The PROMPT tier modification
        """
        cypher = """
        MATCH (e:ReflexionEntry {uid: $entry_uid})
        MERGE (p:PromptComponent {path: $parameter_path})
        CREATE (e)-[:MODIFIED {
            old_value: $old_value,
            new_value: $new_value,
            rationale: $rationale,
            confidence: $confidence,
            timestamp: $timestamp
        }]->(p)
        """

        import json

        await self._psyche.execute(
            cypher,
            {
                "entry_uid": entry.uid,
                "parameter_path": mod.parameter_path,
                "old_value": json.dumps(mod.old_value),
                "new_value": json.dumps(mod.new_value),
                "rationale": mod.rationale,
                "confidence": mod.confidence,
                "timestamp": entry.timestamp.isoformat(),
            },
        )

    async def get_recent_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve recent ReflexionEntry records.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entry dictionaries ordered by cycle_number DESC
        """
        cypher = """
        MATCH (e:ReflexionEntry)
        RETURN e.uid AS uid,
               e.cycle_number AS cycle_number,
               e.timestamp AS timestamp,
               e.overall_coherence AS overall_coherence,
               e.narrative_summary AS narrative_summary,
               e.baseline_comparison AS baseline_comparison,
               e.phenomenological AS phenomenological,
               e.modifications AS modifications
        ORDER BY e.cycle_number DESC
        LIMIT $limit
        """

        return await self._psyche.query(cypher, {"limit": limit})

    async def get_modification_history(
        self,
        parameter_path: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve modification history for a given parameter path.

        Args:
            parameter_path: The parameter path to get history for
            limit: Maximum number of modifications to return

        Returns:
            List of modification records ordered by timestamp DESC
        """
        cypher = """
        MATCH (e:ReflexionEntry)-[m:MODIFIED]->(p)
        WHERE p.path = $parameter_path
        RETURN e.uid AS entry_uid,
               e.cycle_number AS cycle_number,
               p.path AS parameter_path,
               m.old_value AS old_value,
               m.new_value AS new_value,
               m.rationale AS rationale,
               m.confidence AS confidence,
               m.timestamp AS timestamp
        ORDER BY e.cycle_number DESC
        LIMIT $limit
        """

        return await self._psyche.query(
            cypher,
            {"parameter_path": parameter_path, "limit": limit},
        )

    async def get_entry_by_cycle(
        self,
        cycle_number: int,
    ) -> Optional[dict[str, Any]]:
        """Retrieve a single ReflexionEntry by cycle number.

        Args:
            cycle_number: The cycle number to retrieve

        Returns:
            Entry dictionary if found, None otherwise
        """
        cypher = """
        MATCH (e:ReflexionEntry {cycle_number: $cycle_number})
        RETURN e.uid AS uid,
               e.cycle_number AS cycle_number,
               e.timestamp AS timestamp,
               e.overall_coherence AS overall_coherence,
               e.narrative_summary AS narrative_summary,
               e.baseline_comparison AS baseline_comparison,
               e.phenomenological AS phenomenological,
               e.modifications AS modifications
        LIMIT 1
        """

        results = await self._psyche.query(cypher, {"cycle_number": cycle_number})

        if results:
            return results[0]
        return None
