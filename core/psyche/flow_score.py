"""Information flow scoring for graph entities.

Inspired by brain research on white-matter pathway importance:
entities that bridge fast/slow processing have higher flow scores.

The flow score captures how well-connected an entity is, weighted by
the importance of its connections and the salience of its neighbors.
High-flow entities act as information hubs in the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient


@dataclass
class FlowScoreResult:
    """Information flow score for an entity.

    Attributes:
        entity_name: Name of the scored entity.
        flow_score: Computed flow score (higher = more information throughput).
        connection_count: Number of connections to neighbors.
        avg_importance: Average importance across all edges.
        weighted_salience: Sum of (importance * salience) for all connections.
    """
    entity_name: str
    flow_score: float
    connection_count: int
    avg_importance: float
    weighted_salience: float


class InformationFlowScorer:
    """Compute information flow scores for graph entities.

    Flow scores measure how much information passes through an entity
    based on its connections. Entities with many important connections
    to salient neighbors score higher.

    The formula balances total throughput against redundancy:
        Score = Sum(edge_importance * neighbor_salience) / sqrt(connection_count)

    The sqrt normalization prevents entities with many weak connections
    from scoring higher than entities with fewer strong connections.
    """

    def __init__(self, client: "PsycheClient"):
        """Initialize scorer with PsycheClient.

        Args:
            client: PsycheClient instance for graph queries.
        """
        self.client = client

    async def compute_flow_score(self, entity_name: str) -> float:
        """Compute flow score as weighted sum of connection importance.

        Score = Sum(edge_importance * neighbor_salience) / sqrt(connection_count)

        Args:
            entity_name: Name of the entity to score.

        Returns:
            Flow score (0.0 for isolated entities).
        """
        neighborhood = await self.client.get_entity_neighborhood(
            entity_name,
            limit=50,
            order_by="importance",
        )

        edges = neighborhood.get("edges", [])
        if not edges:
            return 0.0

        # Filter to non-center nodes (neighbors only)
        nodes = [n for n in neighborhood.get("nodes", []) if not n.get("is_center")]

        total_score = 0.0
        for node, edge in zip(nodes, edges):
            importance = edge.get("importance", 1.0)
            salience = node.get("salience", 0.5)
            total_score += importance * salience

        connection_count = len(edges)
        return total_score / (connection_count ** 0.5) if connection_count > 0 else 0.0

    async def compute_detailed(self, entity_name: str) -> FlowScoreResult:
        """Compute detailed flow score with all components.

        Args:
            entity_name: Name of the entity to score.

        Returns:
            FlowScoreResult with score breakdown.
        """
        neighborhood = await self.client.get_entity_neighborhood(
            entity_name,
            limit=50,
            order_by="importance",
        )

        edges = neighborhood.get("edges", [])
        nodes = [n for n in neighborhood.get("nodes", []) if not n.get("is_center")]

        if not edges:
            return FlowScoreResult(
                entity_name=entity_name,
                flow_score=0.0,
                connection_count=0,
                avg_importance=0.0,
                weighted_salience=0.0,
            )

        importances = [e.get("importance", 1.0) for e in edges]
        saliences = [n.get("salience", 0.5) for n in nodes]

        weighted_salience = sum(i * s for i, s in zip(importances, saliences))
        connection_count = len(edges)

        return FlowScoreResult(
            entity_name=entity_name,
            flow_score=weighted_salience / (connection_count ** 0.5),
            connection_count=connection_count,
            avg_importance=sum(importances) / len(importances),
            weighted_salience=weighted_salience,
        )
