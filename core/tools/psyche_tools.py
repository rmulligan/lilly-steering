"""Psyche knowledge graph tools for triple exploration.

Provides tools for the model to query and explore the knowledge graph,
enabling discovery-driven thought generation based on triple narratives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


def format_triple_narrative(triple: Any) -> str:
    """Format a triple as a natural language narrative."""
    return f"{triple.subject} {triple.predicate} {triple.object}"


def format_triples_as_context(triples: list[Any], max_triples: int = 10) -> str:
    """Format a list of triples as context for the model.

    Args:
        triples: List of Triple objects
        max_triples: Maximum number to include

    Returns:
        Formatted string with triple narratives
    """
    if not triples:
        return "No relevant knowledge found."

    lines = ["Known relationships:"]
    for i, triple in enumerate(triples[:max_triples]):
        narrative = format_triple_narrative(triple)
        confidence = f" (confidence: {triple.confidence:.0%})" if triple.confidence < 1.0 else ""
        lines.append(f"  - {narrative}{confidence}")

    if len(triples) > max_triples:
        lines.append(f"  ... and {len(triples) - max_triples} more")

    return "\n".join(lines)


class PsycheTools:
    """Collection of psyche query tools for graph exploration."""

    def __init__(self, psyche: "PsycheClient"):
        self._psyche = psyche

    async def search_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search for triples by subject, predicate, or object.

        Args:
            subject: Filter by subject (partial match)
            predicate: Filter by predicate (exact match)
            obj: Filter by object (partial match)
            limit: Maximum results to return

        Returns:
            Dictionary with triples and formatted context
        """
        try:
            triples = await self._psyche.search_triples(
                subject=subject,
                predicate=predicate,
                obj=obj,
                limit=limit,
            )

            return {
                "count": len(triples),
                "triples": [
                    {
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "object": t.object,
                        "confidence": t.confidence,
                    }
                    for t in triples
                ],
                "narrative": format_triples_as_context(triples),
            }

        except Exception as e:
            logger.error(f"search_triples failed: {e}")
            return {"error": str(e), "count": 0, "triples": [], "narrative": ""}

    async def explore_entity(
        self,
        entity_name: str,
        depth: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Explore connections for an entity.

        Args:
            entity_name: Entity to explore
            depth: How many hops to explore (1 = direct connections)
            limit: Maximum triples per direction

        Returns:
            Dictionary with incoming/outgoing connections
        """
        try:
            as_subject = await self._psyche.search_triples(
                subject=entity_name,
                limit=limit,
            )

            as_object = await self._psyche.search_triples(
                obj=entity_name,
                limit=limit,
            )

            outgoing = [
                {"predicate": t.predicate, "target": t.object, "confidence": t.confidence}
                for t in as_subject
            ]

            incoming = [
                {"source": t.subject, "predicate": t.predicate, "confidence": t.confidence}
                for t in as_object
            ]

            lines = [f"Connections for '{entity_name}':"]

            if outgoing:
                lines.append("  Outgoing:")
                for conn in outgoing[:5]:
                    lines.append(f"    → {conn['predicate']} → {conn['target']}")

            if incoming:
                lines.append("  Incoming:")
                for conn in incoming[:5]:
                    lines.append(f"    {conn['source']} → {conn['predicate']} →")

            if not outgoing and not incoming:
                lines.append("  No connections found.")

            return {
                "entity": entity_name,
                "outgoing_count": len(outgoing),
                "incoming_count": len(incoming),
                "outgoing": outgoing,
                "incoming": incoming,
                "narrative": "\n".join(lines),
            }

        except Exception as e:
            logger.error(f"explore_entity failed: {e}")
            return {"error": str(e), "entity": entity_name, "outgoing": [], "incoming": []}

    async def get_recent_knowledge(
        self,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get recently added knowledge from the graph.

        Args:
            limit: Maximum triples to return

        Returns:
            Dictionary with recent triples
        """
        try:
            triples = await self._psyche.search_triples(limit=limit)

            return {
                "count": len(triples),
                "triples": [
                    {
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "object": t.object,
                        "confidence": t.confidence,
                        "created_at": t.created_at.isoformat(),
                    }
                    for t in triples
                ],
                "narrative": format_triples_as_context(triples),
            }

        except Exception as e:
            logger.error(f"get_recent_knowledge failed: {e}")
            return {"error": str(e), "count": 0, "triples": []}

    async def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        """Find a path between two entities in the knowledge graph.

        Args:
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum path length

        Returns:
            Dictionary with path information
        """
        try:
            # FalkorDB doesn't support shortestPath in MATCH clause
            # Use variable-length path with ordering by length instead
            cypher = """
            MATCH (start:Entity {name: $start}), (end:Entity {name: $end})
            MATCH path = (start)-[*1..%d]-(end)
            WITH path, length(path) as pathLength
            ORDER BY pathLength
            LIMIT 1
            RETURN [node IN nodes(path) | node.name] as path_nodes,
                   [rel IN relationships(path) | type(rel)] as relationships
            """ % max_hops

            results = await self._psyche.query(
                cypher,
                {"start": start_entity, "end": end_entity}
            )

            if results:
                path_nodes = results[0].get("path_nodes", [])
                rels = results[0].get("relationships", [])

                path_narrative = " → ".join(path_nodes) if path_nodes else "No path found"

                return {
                    "found": True,
                    "path": path_nodes,
                    "relationships": rels,
                    "hops": len(path_nodes) - 1 if path_nodes else 0,
                    "narrative": f"Path: {path_narrative}",
                }

            return {
                "found": False,
                "path": [],
                "narrative": f"No path found between '{start_entity}' and '{end_entity}'",
            }

        except Exception as e:
            logger.debug(f"find_path query failed (may not have matching entities): {e}")
            return {
                "found": False,
                "error": str(e),
                "narrative": f"Could not find path between '{start_entity}' and '{end_entity}'",
            }


def register_psyche_tools(registry: "ToolRegistry", psyche: "PsycheClient") -> None:
    """Register all psyche tools with a registry.

    Args:
        registry: Tool registry to register with
        psyche: Psyche client for graph access
    """
    from core.tools.registry import ToolRegistry

    tools = PsycheTools(psyche)

    # Wrapper to map 'object' parameter to 'obj' for search_triples
    async def search_knowledge_wrapper(
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,  # noqa: A002 - intentionally shadowing builtin for API
        limit: int = 20,
    ) -> dict[str, Any]:
        return await tools.search_triples(
            subject=subject,
            predicate=predicate,
            obj=object,
            limit=limit,
        )

    registry.register(
        name="search_knowledge",
        description="Search the knowledge graph for relationships. Use to discover what is known about a topic.",
        parameters={
            "subject": {
                "type": "string",
                "description": "Filter by subject entity (partial match)",
            },
            "predicate": {
                "type": "string",
                "description": "Filter by relationship type (exact match)",
            },
            "object": {
                "type": "string",
                "description": "Filter by object entity (partial match)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 20)",
            },
        },
        function=search_knowledge_wrapper,
    )

    registry.register(
        name="explore_connections",
        description="Explore all connections for an entity. Use to discover what relates to something.",
        parameters={
            "entity_name": {
                "type": "string",
                "description": "Entity name to explore",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum connections per direction (default 20)",
            },
        },
        function=tools.explore_entity,
        required=["entity_name"],
    )

    registry.register(
        name="recent_knowledge",
        description="Get recently added knowledge. Use to see what's new in the knowledge graph.",
        parameters={
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10)",
            },
        },
        function=tools.get_recent_knowledge,
    )

    registry.register(
        name="find_path",
        description="Find how two entities are connected. Use to discover relationships between concepts.",
        parameters={
            "start_entity": {
                "type": "string",
                "description": "Starting entity",
            },
            "end_entity": {
                "type": "string",
                "description": "Target entity",
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum path length (default 3)",
            },
        },
        function=tools.find_path,
        required=["start_entity", "end_entity"],
    )

    logger.info(f"Registered {len(registry.list_tools())} psyche tools")
