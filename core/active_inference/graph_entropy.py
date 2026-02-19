"""
Graph Entropy: Global Uncertainty Measurement with Boltzmann Process Framework.

This module computes entropy-based metrics over the knowledge graph and implements
Wolpert's entropy conjecture framework for conditioning analysis.

Boltzmann Process Framework (Wolpert 2025):
    The entropy conjecture formalizes entropy dynamics as a time-symmetric,
    time-translation invariant Markov process. Crucially, this does NOT specify
    which time(s) to condition on—that choice is an independent assumption.

    Key Results:
    - Single-time conditioning (present): Boltzmann brain hypothesis
    - Single-time conditioning (past): Past Hypothesis / Second Law
    - Two-time conditioning (past + present): Changes expected dynamics

    The with_primordial_conditioning() method enables analyzing how entropy
    expectations change under different conditioning assumptions.

FEP Alignment:
    Graph entropy approximates the entropy term in variational free energy.
    High entropy = high uncertainty = Lilly should take epistemic actions.
    The BoltzmannProcess extension adds meta-epistemic awareness: Lilly can
    reason about how her conclusions depend on conditioning assumptions.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


@dataclass
class BoltzmannProcess:
    """
    Boltzmann process model for entropy dynamics.

    Implements Wolpert's framework: entropy evolves as a time-symmetric,
    time-translation invariant Markov process. The key insight is that
    different conditioning times yield different expected dynamics.

    Key concepts:
    - Single-time conditioning (past only): Entropy increases forward = Past Hypothesis / Second Law
    - Single-time conditioning (present only): Entropy increases backward = Boltzmann brain hypothesis
    - Two-time conditioning (both): Entropy peaks in middle = Wolpert's key insight

    Attributes:
        entropy_values: Known entropy values at specific times {time: entropy}
        conditioning_times: Which times to condition on
        diffusion_rate: Rate of entropy diffusion (default 0.1)
    """
    entropy_values: dict[float, float]
    conditioning_times: list[float]
    diffusion_rate: float = 0.1

    def expected_entropy_at(self, t: float) -> float:
        """
        Compute expected entropy at time t given conditioning.

        Uses Boltzmann generation lemma for multi-time conditioning:
        P(x_t|x_{t0}, x_{tf}) = P(x_{t0}|x_t) * P(x_t|x_{tf}) / P(x_{t0}|x_{tf})

        Args:
            t: Time at which to compute expected entropy

        Returns:
            Expected entropy value (0.0 to 1.0)
        """
        if not self.conditioning_times:
            # No conditioning: return equilibrium entropy (maximum)
            return 1.0

        if len(self.conditioning_times) == 1:
            # Single-time conditioning: monotonic from conditioning point
            t0 = self.conditioning_times[0]
            s0 = self.entropy_values.get(t0, 0.5)

            # Entropy increases away from conditioning point
            dt = abs(t - t0)
            # Asymptotic approach to equilibrium (1.0)
            return s0 + (1.0 - s0) * (1 - math.exp(-self.diffusion_rate * dt))

        elif len(self.conditioning_times) == 2:
            # Two-time conditioning: Wolpert's key insight
            t0, tf = sorted(self.conditioning_times)
            s0 = self.entropy_values.get(t0, 0.5)
            sf = self.entropy_values.get(tf, 0.5)

            if t <= t0:
                # Before first conditioning point
                dt = t0 - t
                return s0 + (1.0 - s0) * (1 - math.exp(-self.diffusion_rate * dt))
            elif t >= tf:
                # After second conditioning point
                dt = t - tf
                return sf + (1.0 - sf) * (1 - math.exp(-self.diffusion_rate * dt))
            else:
                # Between conditioning points: weighted interpolation with peak
                # This is the key insight: two low-entropy endpoints = higher middle
                alpha = (t - t0) / (tf - t0)
                base = s0 * (1 - alpha) + sf * alpha

                # Add hump for intermediate times (Wolpert's result)
                hump = 4 * alpha * (1 - alpha) * (1.0 - min(s0, sf))
                return min(1.0, base + 0.5 * hump)

        else:
            # Multi-time conditioning: recursive application
            # For simplicity, use nearest two conditioning times
            sorted_times = sorted(self.conditioning_times)
            for i in range(len(sorted_times) - 1):
                if sorted_times[i] <= t <= sorted_times[i + 1]:
                    sub_process = BoltzmannProcess(
                        entropy_values=self.entropy_values,
                        conditioning_times=[sorted_times[i], sorted_times[i + 1]],
                        diffusion_rate=self.diffusion_rate,
                    )
                    return sub_process.expected_entropy_at(t)

            # Outside all conditioning times, behave like single-time conditioning from the nearest endpoint.
            if t < sorted_times[0]:
                t0 = sorted_times[0]
                s0 = self.entropy_values.get(t0, 0.5)
                dt = t0 - t
                return s0 + (1.0 - s0) * (1 - math.exp(-self.diffusion_rate * dt))
            else:  # t > sorted_times[-1]
                tf = sorted_times[-1]
                sf = self.entropy_values.get(tf, 0.5)
                dt = t - tf
                return sf + (1.0 - sf) * (1 - math.exp(-self.diffusion_rate * dt))


def compute_two_time_conditional(
    s0: float,
    sf: float,
    t0: float,
    tf: float,
    t: float,
    diffusion_rate: float = 0.1,
) -> float:
    """
    Convenience function for two-time conditional entropy.

    Implements Wolpert's Boltzmann generation lemma result.

    Args:
        s0: Entropy value at time t0
        sf: Entropy value at time tf
        t0: First conditioning time
        tf: Second conditioning time
        t: Time at which to compute expected entropy
        diffusion_rate: Rate of entropy diffusion

    Returns:
        Expected entropy at time t given conditioning on t0 and tf
    """
    process = BoltzmannProcess(
        entropy_values={t0: s0, tf: sf},
        conditioning_times=[t0, tf],
        diffusion_rate=diffusion_rate,
    )
    return process.expected_entropy_at(t)


@dataclass
class EntropyResult:
    """
    Result of graph entropy computation.

    Includes BoltzmannProcess for Wolpert-style conditioning analysis.

    Attributes:
        structural_entropy: Entropy of degree distribution (0-1)
        cluster_entropy: Entropy of cluster size distribution (0-1)
        orphan_rate: Proportion of orphan nodes (0-1)
        hub_concentration: Gini coefficient of degree distribution (0-1)
        total_entropy: Combined entropy score (0-1)
        node_count: Total number of nodes
        edge_count: Total number of edges
        cluster_count: Number of detected clusters
        orphan_count: Number of orphan nodes
        should_cultivate: Whether the Weaver should activate
        cultivation_reason: Why cultivation is or isn't needed
        boltzmann_process: Boltzmann process for conditional entropy analysis
    """
    structural_entropy: float
    cluster_entropy: float
    orphan_rate: float
    hub_concentration: float
    total_entropy: float
    node_count: int
    edge_count: int
    cluster_count: int
    orphan_count: int
    should_cultivate: bool
    cultivation_reason: str
    boltzmann_process: Optional["BoltzmannProcess"] = None

    @property
    def is_fragmented(self) -> bool:
        """Graph is highly fragmented."""
        return self.total_entropy > ENTROPY_THRESHOLD_FRAGMENTED

    @property
    def is_healthy(self) -> bool:
        """Graph is well-connected."""
        return self.total_entropy < ENTROPY_THRESHOLD_HEALTHY

    @property
    def avg_degree(self) -> float:
        """Average degree (edges per node)."""
        if self.node_count == 0:
            return 0.0
        return (2 * self.edge_count) / self.node_count

    def to_dict(self) -> dict:
        """Serialize for logging and persistence."""
        result = {
            "structural_entropy": self.structural_entropy,
            "cluster_entropy": self.cluster_entropy,
            "orphan_rate": self.orphan_rate,
            "hub_concentration": self.hub_concentration,
            "total_entropy": self.total_entropy,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cluster_count": self.cluster_count,
            "orphan_count": self.orphan_count,
            "should_cultivate": self.should_cultivate,
            "cultivation_reason": self.cultivation_reason,
            "avg_degree": self.avg_degree,
        }
        if self.boltzmann_process:
            result["conditioning_times"] = self.boltzmann_process.conditioning_times
        return result

    def with_primordial_conditioning(
        self,
        primordial_entropy: float,
        primordial_time: float = -13.8e9,
    ) -> "EntropyResult":
        """
        Create new EntropyResult with two-time conditioning.

        Implements Wolpert's insight: conditioning on both present
        and primordial entropy changes expected dynamics.

        Args:
            primordial_entropy: Entropy at primordial time (typically very low)
            primordial_time: Time of primordial conditioning (negative = past)

        Returns:
            New EntropyResult with two-time conditioned BoltzmannProcess
        """
        if not self.boltzmann_process:
            return self

        # Get current conditioning
        current_entropy = self.total_entropy
        current_time = 0  # Now

        # Create two-time conditioned process
        two_time_process = BoltzmannProcess(
            entropy_values={
                primordial_time: primordial_entropy,
                current_time: current_entropy,
            },
            conditioning_times=[primordial_time, current_time],
            diffusion_rate=self.boltzmann_process.diffusion_rate,
        )

        # Return new result with updated process
        return EntropyResult(
            structural_entropy=self.structural_entropy,
            cluster_entropy=self.cluster_entropy,
            orphan_rate=self.orphan_rate,
            hub_concentration=self.hub_concentration,
            total_entropy=self.total_entropy,
            node_count=self.node_count,
            edge_count=self.edge_count,
            cluster_count=self.cluster_count,
            orphan_count=self.orphan_count,
            should_cultivate=self.should_cultivate,
            cultivation_reason=self.cultivation_reason + " [two-time conditioned]",
            boltzmann_process=two_time_process,
        )


# Thresholds for entropy-based decisions
ENTROPY_THRESHOLD_CULTIVATE = 0.6  # Above this, Weaver should activate
ENTROPY_THRESHOLD_FRAGMENTED = 0.7  # Graph is highly fragmented above this
ENTROPY_THRESHOLD_HEALTHY = 0.3  # Graph is well-connected below this
ORPHAN_RATE_CONCERN = 0.3  # More than 30% orphans is concerning
HUB_CONCENTRATION_CONCERN = 0.8  # Very high concentration = brittle

# Entropy weights for total entropy calculation
# Weights reflect importance: orphan_rate is most critical, then cluster fragmentation
WEIGHT_ORPHAN_RATE = 0.35  # Disconnected nodes are most problematic
WEIGHT_CLUSTER_ENTROPY = 0.30  # Fragmentation is second priority
WEIGHT_STRUCTURAL_ENTROPY = 0.20  # Degree distribution matters less
WEIGHT_HUB_CONCENTRATION = 0.15  # Over-concentration is least critical

# Cluster count threshold for fragmentation concern
CLUSTER_COUNT_FRAGMENTATION_THRESHOLD = 5


class GraphEntropy:
    """
    Computes entropy metrics over the knowledge graph.

    This class provides a global view of graph health for the Weaver.
    It's called periodically (e.g., during Dream Cycle) to assess whether
    cultivation actions are needed.

    The key insight: graph structure is itself a generative model.
    Entropy over the structure tells us how "uncertain" the graph is
    about the relationships it encodes.

    Attributes:
        graph: PsycheClient for graph queries
    """

    def __init__(self, graph: "PsycheClient"):
        """
        Initialize the entropy calculator.

        Args:
            graph: PsycheClient instance
        """
        self.graph = graph

    async def compute(self, tenant_id: str) -> EntropyResult:
        """
        Compute comprehensive entropy metrics for the graph.

        Args:
            tenant_id: Tenant identifier

        Returns:
            EntropyResult with all entropy metrics
        """
        # Get graph statistics
        stats = await self._get_graph_stats(tenant_id)

        node_count = stats.get("node_count", 0)
        edge_count = stats.get("edge_count", 0)
        degrees = stats.get("degrees", [])

        # Handle empty graph
        if node_count == 0:
            # Create trivial Boltzmann process with maximum entropy
            trivial_process = BoltzmannProcess(
                entropy_values={0: 1.0},  # t=0 is "now", max entropy for empty graph
                conditioning_times=[0],
                diffusion_rate=0.1,
            )
            return EntropyResult(
                structural_entropy=1.0,
                cluster_entropy=1.0,
                orphan_rate=1.0,
                hub_concentration=0.0,
                total_entropy=1.0,
                node_count=0,
                edge_count=0,
                cluster_count=0,
                orphan_count=0,
                should_cultivate=False,  # Can't cultivate empty graph
                cultivation_reason="Empty graph - nothing to cultivate",
                boltzmann_process=trivial_process,
            )

        # Compute individual entropy components
        structural_entropy = self._compute_structural_entropy(degrees)
        orphan_count = sum(1 for d in degrees if d == 0)
        orphan_rate = orphan_count / node_count if node_count > 0 else 1.0

        hub_concentration = self._compute_hub_concentration(degrees)

        # Get cluster information
        # Note: ClusterDetector not yet migrated, return empty list for now
        cluster_sizes = await self._get_cluster_sizes(tenant_id)
        cluster_count = len(cluster_sizes)
        cluster_entropy = self._compute_cluster_entropy(cluster_sizes, node_count)

        # Combine into total entropy
        total_entropy = self._combine_entropy(
            structural_entropy=structural_entropy,
            cluster_entropy=cluster_entropy,
            orphan_rate=orphan_rate,
            hub_concentration=hub_concentration,
        )

        # Determine if cultivation is needed
        should_cultivate, reason = self._should_cultivate(
            total_entropy=total_entropy,
            orphan_rate=orphan_rate,
            hub_concentration=hub_concentration,
            cluster_count=cluster_count,
        )

        # Create Boltzmann process for conditioning analysis
        # Condition on current entropy measurement (single-time, present)
        boltzmann_process = BoltzmannProcess(
            entropy_values={0: total_entropy},  # t=0 is "now"
            conditioning_times=[0],
            diffusion_rate=0.1,
        )

        return EntropyResult(
            structural_entropy=structural_entropy,
            cluster_entropy=cluster_entropy,
            orphan_rate=orphan_rate,
            hub_concentration=hub_concentration,
            total_entropy=total_entropy,
            node_count=node_count,
            edge_count=edge_count,
            cluster_count=cluster_count,
            orphan_count=orphan_count,
            should_cultivate=should_cultivate,
            cultivation_reason=reason,
            boltzmann_process=boltzmann_process,
        )

    async def _get_graph_stats(self, tenant_id: str) -> dict:
        """
        Get basic graph statistics concurrently.

        Returns dict with node_count, edge_count, degrees.

        Note: tenant_id parameter kept for API compatibility but not used.
        Lilly is single-tenant so we query all nodes.
        """
        import asyncio

        try:
            # Run queries concurrently for efficiency
            # Count Entity nodes (concepts in the knowledge graph)
            node_query = """
                MATCH (n:Entity)
                RETURN count(n) as count
            """
            # Count Triple nodes as "edges" - they represent semantic relationships
            # between entities via subject/predicate/object properties
            edge_query = """
                MATCH (t:Triple)
                RETURN count(t) as count
            """
            # Compute degree by counting Triples mentioning each Entity
            # An Entity's degree = number of Triples where it appears as subject or object
            # Use OPTIONAL MATCH with WHERE to avoid unsupported list comprehension syntax
            degree_query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (t:Triple)
                WHERE t.subject = e.name OR t.object_ = e.name
                RETURN e.uid AS uid, count(t) AS degree
            """

            # PsycheClient uses query() method for both reads and writes
            node_task = self.graph.query(node_query)
            edge_task = self.graph.query(edge_query)
            degree_task = self.graph.query(degree_query)

            node_result, edge_result, degree_result = await asyncio.gather(
                node_task, edge_task, degree_task
            )

            node_count = node_result[0]["count"] if node_result else 0
            edge_count = edge_result[0]["count"] if edge_result else 0
            degrees = [row["degree"] for row in degree_result]

            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "degrees": degrees
            }

        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")
            return {"node_count": 0, "edge_count": 0, "degrees": []}

    async def _get_cluster_sizes(self, tenant_id: str) -> list[int]:
        """
        Get sizes of connected components (clusters).

        Note: ClusterDetector is not yet migrated to this codebase.
        This method returns an empty list for now, which causes cluster_entropy
        to default to 1.0 (maximum entropy). When ClusterDetector is migrated,
        this method should be updated to use it.
        """
        # TODO: Integrate ClusterDetector when migrated
        # ClusterDetector not yet migrated - return empty for now
        logger.debug(f"ClusterDetector not available, skipping cluster analysis for {tenant_id}")
        return []

    def _compute_structural_entropy(self, degrees: list[int]) -> float:
        """
        Compute Shannon entropy of the degree distribution.

        A random graph has high entropy (uniform degree distribution).
        A scale-free graph has lower entropy (power-law distribution).
        The brain uses scale-free topology, so we prefer lower entropy.
        """
        if not degrees or len(degrees) < 2:
            return 1.0

        # Count degree frequencies
        degree_counts: dict[int, int] = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        total = len(degrees)

        # Compute Shannon entropy
        entropy = 0.0
        for count in degree_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = math.log2(len(degree_counts)) if degree_counts else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, max(0.0, normalized))

    def _compute_cluster_entropy(
        self,
        cluster_sizes: list[int],
        total_nodes: int,
    ) -> float:
        """
        Compute entropy of cluster size distribution.

        Many small clusters = high entropy (fragmented)
        Few large clusters = low entropy (connected)
        """
        if not cluster_sizes or total_nodes == 0:
            return 1.0

        # Normalize sizes to proportions
        proportions = [s / total_nodes for s in cluster_sizes]

        # Compute Shannon entropy
        entropy = 0.0
        for p in proportions:
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum entropy
        max_entropy = math.log2(len(cluster_sizes)) if cluster_sizes else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, max(0.0, normalized))

    def _compute_hub_concentration(self, degrees: list[int]) -> float:
        """
        Compute Gini coefficient of degree distribution.

        High Gini = few nodes have most connections (brittle but structured)
        Low Gini = connections are evenly distributed (resilient but flat)

        Ideal is moderate Gini (scale-free but not too concentrated).
        """
        if not degrees or len(degrees) < 2:
            return 0.0

        # Sort degrees
        sorted_degrees = sorted(degrees)
        n = len(sorted_degrees)

        # Compute Gini coefficient
        # G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
        total = sum(sorted_degrees)
        if total == 0:
            return 0.0

        cumsum = sum((i + 1) * d for i, d in enumerate(sorted_degrees))
        gini = (2 * cumsum) / (n * total) - (n + 1) / n

        return min(1.0, max(0.0, gini))

    def _combine_entropy(
        self,
        structural_entropy: float,
        cluster_entropy: float,
        orphan_rate: float,
        hub_concentration: float,
    ) -> float:
        """
        Combine entropy components into total entropy.

        Weights reflect importance:
        - Orphan rate is most important (disconnected = bad)
        - Cluster entropy next (fragmentation = bad)
        - Structural entropy (degree distribution)
        - Hub concentration (too high = brittle)
        """
        total = (
            WEIGHT_ORPHAN_RATE * orphan_rate +
            WEIGHT_CLUSTER_ENTROPY * cluster_entropy +
            WEIGHT_STRUCTURAL_ENTROPY * structural_entropy +
            WEIGHT_HUB_CONCENTRATION * hub_concentration
        )

        return min(1.0, max(0.0, total))

    def _should_cultivate(
        self,
        total_entropy: float,
        orphan_rate: float,
        hub_concentration: float,
        cluster_count: int,
    ) -> tuple[bool, str]:
        """
        Determine if the Weaver should activate cultivation.

        Returns:
            Tuple of (should_cultivate, reason)
        """
        # High overall entropy
        if total_entropy > ENTROPY_THRESHOLD_CULTIVATE:
            if orphan_rate > ORPHAN_RATE_CONCERN:
                return True, f"High orphan rate ({orphan_rate:.1%}) - many disconnected nodes"
            if cluster_count > CLUSTER_COUNT_FRAGMENTATION_THRESHOLD:
                return True, f"Graph is fragmented into {cluster_count} clusters"
            return True, f"High entropy ({total_entropy:.2f}) - graph needs cultivation"

        # Hub concentration warning (not enough for cultivation, but concerning)
        if hub_concentration > HUB_CONCENTRATION_CONCERN:
            return True, f"High hub concentration ({hub_concentration:.2f}) - graph is brittle"

        # Healthy graph
        if total_entropy < ENTROPY_THRESHOLD_HEALTHY:
            return False, "Graph is healthy - no cultivation needed"

        # Moderate entropy
        return False, f"Moderate entropy ({total_entropy:.2f}) - monitoring"


async def compute_graph_entropy(
    graph: "PsycheClient",
    tenant_id: str,
) -> EntropyResult:
    """
    Convenience function to compute graph entropy.

    Usage:
        from core.active_inference import compute_graph_entropy

        result = await compute_graph_entropy(psyche_client, tenant_id)
        if result.should_cultivate:
            await weaver.cultivate()
    """
    calculator = GraphEntropy(graph)
    return await calculator.compute(tenant_id)
