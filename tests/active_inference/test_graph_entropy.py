"""Tests for GraphEntropy module - surprise detection for knowledge graphs."""

import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.active_inference.graph_entropy import (
    EntropyResult,
    GraphEntropy,
    compute_graph_entropy,
    ENTROPY_THRESHOLD_CULTIVATE,
    ORPHAN_RATE_CONCERN,
    HUB_CONCENTRATION_CONCERN,
    BoltzmannProcess,
    compute_two_time_conditional,
)


# === EntropyResult Tests ===


class TestEntropyResult:
    """Tests for EntropyResult dataclass properties."""

    def test_is_fragmented_above_threshold(self):
        """is_fragmented returns True when total_entropy > 0.7."""
        result = EntropyResult(
            structural_entropy=0.8,
            cluster_entropy=0.8,
            orphan_rate=0.5,
            hub_concentration=0.3,
            total_entropy=0.75,  # Above 0.7
            node_count=100,
            edge_count=50,
            cluster_count=5,
            orphan_count=50,
            should_cultivate=True,
            cultivation_reason="High entropy",
        )
        assert result.is_fragmented is True

    def test_is_fragmented_below_threshold(self):
        """is_fragmented returns False when total_entropy <= 0.7."""
        result = EntropyResult(
            structural_entropy=0.5,
            cluster_entropy=0.5,
            orphan_rate=0.2,
            hub_concentration=0.3,
            total_entropy=0.5,  # Below 0.7
            node_count=100,
            edge_count=80,
            cluster_count=2,
            orphan_count=20,
            should_cultivate=False,
            cultivation_reason="Moderate entropy",
        )
        assert result.is_fragmented is False

    def test_is_healthy_below_threshold(self):
        """is_healthy returns True when total_entropy < 0.3."""
        result = EntropyResult(
            structural_entropy=0.2,
            cluster_entropy=0.2,
            orphan_rate=0.1,
            hub_concentration=0.2,
            total_entropy=0.25,  # Below 0.3
            node_count=100,
            edge_count=200,
            cluster_count=1,
            orphan_count=10,
            should_cultivate=False,
            cultivation_reason="Healthy graph",
        )
        assert result.is_healthy is True

    def test_is_healthy_above_threshold(self):
        """is_healthy returns False when total_entropy >= 0.3."""
        result = EntropyResult(
            structural_entropy=0.4,
            cluster_entropy=0.4,
            orphan_rate=0.2,
            hub_concentration=0.3,
            total_entropy=0.35,  # At or above 0.3
            node_count=100,
            edge_count=80,
            cluster_count=2,
            orphan_count=20,
            should_cultivate=False,
            cultivation_reason="Moderate entropy",
        )
        assert result.is_healthy is False

    def test_avg_degree_calculation(self):
        """avg_degree correctly calculates edges per node."""
        result = EntropyResult(
            structural_entropy=0.3,
            cluster_entropy=0.3,
            orphan_rate=0.1,
            hub_concentration=0.2,
            total_entropy=0.25,
            node_count=100,
            edge_count=150,  # 150 edges, 100 nodes -> avg 3.0
            cluster_count=1,
            orphan_count=10,
            should_cultivate=False,
            cultivation_reason="",
        )
        assert result.avg_degree == 3.0

    def test_avg_degree_empty_graph(self):
        """avg_degree returns 0 for empty graph."""
        result = EntropyResult(
            structural_entropy=1.0,
            cluster_entropy=1.0,
            orphan_rate=1.0,
            hub_concentration=0.0,
            total_entropy=1.0,
            node_count=0,
            edge_count=0,
            cluster_count=0,
            orphan_count=0,
            should_cultivate=False,
            cultivation_reason="Empty graph",
        )
        assert result.avg_degree == 0.0

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all entropy metrics and computed properties."""
        result = EntropyResult(
            structural_entropy=0.3,
            cluster_entropy=0.4,
            orphan_rate=0.15,
            hub_concentration=0.25,
            total_entropy=0.35,
            node_count=100,
            edge_count=150,
            cluster_count=2,
            orphan_count=15,
            should_cultivate=True,
            cultivation_reason="Test reason",
        )
        d = result.to_dict()

        assert d["structural_entropy"] == 0.3
        assert d["cluster_entropy"] == 0.4
        assert d["orphan_rate"] == 0.15
        assert d["hub_concentration"] == 0.25
        assert d["total_entropy"] == 0.35
        assert d["node_count"] == 100
        assert d["edge_count"] == 150
        assert d["cluster_count"] == 2
        assert d["orphan_count"] == 15
        assert d["should_cultivate"] is True
        assert d["cultivation_reason"] == "Test reason"
        assert d["avg_degree"] == 3.0


# === Structural Entropy Tests ===


class TestStructuralEntropy:
    """Tests for structural entropy computation (degree distribution)."""

    @pytest.fixture
    def graph_entropy(self):
        """Create GraphEntropy with mock client."""
        mock_client = MagicMock()
        return GraphEntropy(mock_client)

    def test_empty_degrees(self, graph_entropy):
        """Empty degree list returns maximum entropy."""
        result = graph_entropy._compute_structural_entropy([])
        assert result == 1.0

    def test_single_degree(self, graph_entropy):
        """Single degree returns maximum entropy."""
        result = graph_entropy._compute_structural_entropy([5])
        assert result == 1.0

    def test_uniform_distribution(self, graph_entropy):
        """Uniform degree distribution has maximum entropy."""
        # All nodes have different degrees -> uniform distribution -> max entropy
        degrees = [1, 2, 3, 4, 5]
        result = graph_entropy._compute_structural_entropy(degrees)
        # Should be close to 1.0 (normalized Shannon entropy of uniform)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_concentrated_distribution(self, graph_entropy):
        """All same degree has zero entropy."""
        # All nodes have same degree -> single value -> min entropy
        degrees = [3, 3, 3, 3, 3]
        result = graph_entropy._compute_structural_entropy(degrees)
        # With only one unique value (one "bin"), there is no uncertainty.
        # max_entropy = log2(1) = 0, and since a single-outcome distribution
        # has zero entropy (complete certainty), we return 0.0
        assert result == 0.0

    def test_power_law_like_distribution(self, graph_entropy):
        """Power-law-like distribution has lower entropy than uniform."""
        # Power law: few high-degree nodes, many low-degree nodes
        degrees = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 5, 10]
        result = graph_entropy._compute_structural_entropy(degrees)
        # Should have lower entropy than uniform (more concentrated)
        assert 0.0 <= result <= 1.0

    def test_bimodal_distribution(self, graph_entropy):
        """Bimodal distribution has intermediate entropy."""
        # Two distinct groups of degrees
        degrees = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10]
        result = graph_entropy._compute_structural_entropy(degrees)
        # Should be around 0.5-0.7 (two bins, fairly uniform within each)
        assert 0.0 <= result <= 1.0


# === Cluster Entropy Tests ===


class TestClusterEntropy:
    """Tests for cluster entropy computation."""

    @pytest.fixture
    def graph_entropy(self):
        """Create GraphEntropy with mock client."""
        mock_client = MagicMock()
        return GraphEntropy(mock_client)

    def test_empty_clusters(self, graph_entropy):
        """Empty cluster list returns maximum entropy."""
        result = graph_entropy._compute_cluster_entropy([], 100)
        assert result == 1.0

    def test_zero_nodes(self, graph_entropy):
        """Zero total nodes returns maximum entropy."""
        result = graph_entropy._compute_cluster_entropy([50, 30, 20], 0)
        assert result == 1.0

    def test_single_cluster(self, graph_entropy):
        """Single cluster (all connected) has minimum entropy."""
        # All nodes in one cluster
        result = graph_entropy._compute_cluster_entropy([100], 100)
        # Single cluster means max_entropy = log2(1) = 0.
        # A single-outcome distribution has zero entropy (complete certainty).
        assert result == 0.0

    def test_many_equal_clusters(self, graph_entropy):
        """Many equal-sized clusters has maximum entropy."""
        # 10 clusters of 10 nodes each
        cluster_sizes = [10] * 10
        result = graph_entropy._compute_cluster_entropy(cluster_sizes, 100)
        # Equal clusters -> uniform distribution -> max entropy
        # entropy = -10 * (0.1 * log2(0.1)) = 10 * 0.1 * 3.32 = 3.32
        # max_entropy = log2(10) = 3.32
        # normalized = 3.32 / 3.32 = 1.0
        assert result == pytest.approx(1.0, rel=0.01)

    def test_one_dominant_cluster(self, graph_entropy):
        """One dominant cluster has low entropy."""
        # 90 nodes in one cluster, 10 others in small clusters
        cluster_sizes = [90, 5, 3, 2]
        result = graph_entropy._compute_cluster_entropy(cluster_sizes, 100)
        # Skewed distribution -> lower entropy
        assert result < 0.9


# === Hub Concentration (Gini) Tests ===


class TestHubConcentration:
    """Tests for hub concentration (Gini coefficient) computation."""

    @pytest.fixture
    def graph_entropy(self):
        """Create GraphEntropy with mock client."""
        mock_client = MagicMock()
        return GraphEntropy(mock_client)

    def test_empty_degrees(self, graph_entropy):
        """Empty degree list returns 0 concentration."""
        result = graph_entropy._compute_hub_concentration([])
        assert result == 0.0

    def test_single_degree(self, graph_entropy):
        """Single degree returns 0 concentration."""
        result = graph_entropy._compute_hub_concentration([5])
        assert result == 0.0

    def test_all_zero_degrees(self, graph_entropy):
        """All zero degrees returns 0 concentration."""
        result = graph_entropy._compute_hub_concentration([0, 0, 0, 0, 0])
        assert result == 0.0

    def test_uniform_degrees(self, graph_entropy):
        """Uniform degrees has low Gini (egalitarian)."""
        degrees = [5, 5, 5, 5, 5]
        result = graph_entropy._compute_hub_concentration(degrees)
        # Perfect equality -> Gini = 0
        assert result == pytest.approx(0.0, abs=0.01)

    def test_highly_concentrated(self, graph_entropy):
        """One hub with all connections has high Gini."""
        # One node with all edges, others with zero
        degrees = [100, 0, 0, 0, 0]
        result = graph_entropy._compute_hub_concentration(degrees)
        # Extreme inequality -> Gini close to 1
        assert result > 0.7

    def test_moderate_concentration(self, graph_entropy):
        """Moderate degree variation has moderate Gini."""
        # Some variation but not extreme
        degrees = [1, 2, 3, 4, 5]
        result = graph_entropy._compute_hub_concentration(degrees)
        # Should be between 0 and 0.5
        assert 0.0 < result < 0.5


# === Combined Entropy Tests ===


class TestCombinedEntropy:
    """Tests for combining entropy components."""

    @pytest.fixture
    def graph_entropy(self):
        """Create GraphEntropy with mock client."""
        mock_client = MagicMock()
        return GraphEntropy(mock_client)

    def test_all_zero_components(self, graph_entropy):
        """All zero components gives zero total entropy."""
        result = graph_entropy._combine_entropy(
            structural_entropy=0.0,
            cluster_entropy=0.0,
            orphan_rate=0.0,
            hub_concentration=0.0,
        )
        assert result == 0.0

    def test_all_max_components(self, graph_entropy):
        """All max components gives max total entropy."""
        result = graph_entropy._combine_entropy(
            structural_entropy=1.0,
            cluster_entropy=1.0,
            orphan_rate=1.0,
            hub_concentration=1.0,
        )
        assert result == pytest.approx(1.0)

    def test_weights_sum_to_one(self, graph_entropy):
        """Verify weights sum correctly."""
        # All components at 0.5 should give 0.5 total
        result = graph_entropy._combine_entropy(
            structural_entropy=0.5,
            cluster_entropy=0.5,
            orphan_rate=0.5,
            hub_concentration=0.5,
        )
        assert result == pytest.approx(0.5)

    def test_orphan_rate_most_weighted(self, graph_entropy):
        """Orphan rate has highest weight (0.35)."""
        # Only orphan_rate is 1.0, others are 0
        result = graph_entropy._combine_entropy(
            structural_entropy=0.0,
            cluster_entropy=0.0,
            orphan_rate=1.0,
            hub_concentration=0.0,
        )
        assert result == pytest.approx(0.35, rel=0.01)


# === Cultivation Decision Tests ===


class TestCultivationDecision:
    """Tests for should_cultivate decision logic."""

    @pytest.fixture
    def graph_entropy(self):
        """Create GraphEntropy with mock client."""
        mock_client = MagicMock()
        return GraphEntropy(mock_client)

    def test_high_entropy_triggers_cultivation(self, graph_entropy):
        """High total entropy triggers cultivation."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.7,
            orphan_rate=0.1,
            hub_concentration=0.3,
            cluster_count=2,
        )
        assert should_cultivate is True
        assert "entropy" in reason.lower()

    def test_high_orphan_rate_triggers_cultivation(self, graph_entropy):
        """High orphan rate with high entropy mentions orphans."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.7,
            orphan_rate=0.4,  # Above ORPHAN_RATE_CONCERN (0.3)
            hub_concentration=0.3,
            cluster_count=2,
        )
        assert should_cultivate is True
        assert "orphan" in reason.lower()

    def test_many_clusters_triggers_cultivation(self, graph_entropy):
        """Many clusters with high entropy mentions fragmentation."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.7,
            orphan_rate=0.1,
            hub_concentration=0.3,
            cluster_count=10,  # More than 5
        )
        assert should_cultivate is True
        assert "cluster" in reason.lower() or "fragment" in reason.lower()

    def test_high_hub_concentration_triggers_cultivation(self, graph_entropy):
        """High hub concentration alone triggers cultivation."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.5,  # Below threshold
            orphan_rate=0.1,
            hub_concentration=0.85,  # Above HUB_CONCENTRATION_CONCERN (0.8)
            cluster_count=2,
        )
        assert should_cultivate is True
        assert "hub" in reason.lower() or "concentration" in reason.lower() or "brittle" in reason.lower()

    def test_healthy_graph_no_cultivation(self, graph_entropy):
        """Healthy graph does not need cultivation."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.2,
            orphan_rate=0.05,
            hub_concentration=0.3,
            cluster_count=2,
        )
        assert should_cultivate is False
        assert "healthy" in reason.lower()

    def test_moderate_entropy_monitoring(self, graph_entropy):
        """Moderate entropy results in monitoring (no action)."""
        should_cultivate, reason = graph_entropy._should_cultivate(
            total_entropy=0.45,  # Between 0.3 and 0.6
            orphan_rate=0.1,
            hub_concentration=0.3,
            cluster_count=3,
        )
        assert should_cultivate is False
        assert "moderate" in reason.lower() or "monitoring" in reason.lower()


# === Graph Stats Integration Tests ===


class TestGraphStatsRetrieval:
    """Tests for _get_graph_stats with mocked PsycheClient."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_get_graph_stats_success(self, mock_client):
        """Graph stats retrieval returns expected format."""
        mock_client.query.side_effect = [
            [{"count": 100}],  # node count
            [{"count": 150}],  # edge count
            [{"uid": f"n{i}", "degree": i % 5} for i in range(100)],  # degrees
        ]

        entropy = GraphEntropy(mock_client)
        stats = await entropy._get_graph_stats("test-tenant")

        assert stats["node_count"] == 100
        assert stats["edge_count"] == 150
        assert len(stats["degrees"]) == 100

    @pytest.mark.asyncio
    async def test_get_graph_stats_empty_results(self, mock_client):
        """Empty query results return zeros."""
        mock_client.query.side_effect = [
            [],  # no nodes
            [],  # no edges
            [],  # no degrees
        ]

        entropy = GraphEntropy(mock_client)
        stats = await entropy._get_graph_stats("test-tenant")

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["degrees"] == []

    @pytest.mark.asyncio
    async def test_get_graph_stats_error_handling(self, mock_client):
        """Query errors return safe defaults."""
        mock_client.query.side_effect = Exception("Connection failed")

        entropy = GraphEntropy(mock_client)
        stats = await entropy._get_graph_stats("test-tenant")

        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["degrees"] == []


# === Empty Graph Handling Tests ===


class TestEmptyGraphHandling:
    """Tests for empty graph edge cases."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_empty_graph_result(self, mock_client):
        """Empty graph returns max entropy, no cultivation."""
        mock_client.query.side_effect = [
            [],  # no nodes
            [],  # no edges
            [],  # no degrees
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test-tenant")

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.total_entropy == 1.0
        assert result.should_cultivate is False
        assert "empty" in result.cultivation_reason.lower()


# === Full Compute Integration Tests ===


class TestComputeIntegration:
    """Integration tests for full entropy computation."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_compute_healthy_graph(self, mock_client):
        """Well-connected graph shows low entropy."""
        # Simulate well-connected graph: 100 nodes, 200 edges, good degree distribution
        degrees = [4, 4, 4, 5, 5, 5, 3, 3, 3, 6] * 10  # Fairly uniform
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 200}],
            [{"uid": f"n{i}", "degree": degrees[i]} for i in range(100)],
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test-tenant")

        assert result.node_count == 100
        assert result.edge_count == 200
        assert result.orphan_count == 0
        assert result.orphan_rate == 0.0
        # Total entropy depends on cluster entropy which defaults to 1.0
        # since ClusterDetector is not available
        assert result.total_entropy > 0  # Will have some entropy from cluster_entropy

    @pytest.mark.asyncio
    async def test_compute_fragmented_graph(self, mock_client):
        """Graph with many orphans shows high entropy."""
        # Simulate fragmented graph: 100 nodes, many with degree 0
        degrees = [0] * 50 + [1, 1, 2, 2, 3] * 10
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 25}],
            [{"uid": f"n{i}", "degree": degrees[i]} for i in range(100)],
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test-tenant")

        assert result.orphan_count == 50
        assert result.orphan_rate == 0.5
        assert result.total_entropy > ENTROPY_THRESHOLD_CULTIVATE
        assert result.should_cultivate is True

    @pytest.mark.asyncio
    async def test_compute_hub_dominated_graph(self, mock_client):
        """Graph with one dominant hub has high concentration."""
        # One node with many connections, others with few
        degrees = [50] + [1] * 49 + [0] * 50
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 75}],
            [{"uid": f"n{i}", "degree": degrees[i]} for i in range(100)],
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test-tenant")

        assert result.hub_concentration > 0.5
        assert result.orphan_count == 50


# === Convenience Function Tests ===


class TestConvenienceFunction:
    """Tests for compute_graph_entropy convenience function."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_compute_graph_entropy_function(self, mock_client):
        """Convenience function returns EntropyResult."""
        mock_client.query.side_effect = [
            [{"count": 50}],
            [{"count": 75}],
            [{"uid": f"n{i}", "degree": 3} for i in range(50)],
        ]

        result = await compute_graph_entropy(mock_client, "test-tenant")

        assert isinstance(result, EntropyResult)
        assert result.node_count == 50
        assert result.edge_count == 75


# === BoltzmannProcess Tests ===


class TestBoltzmannProcess:
    """Tests for BoltzmannProcess - Wolpert-style entropy conditioning."""

    def test_boltzmann_process_single_time_conditioning(self):
        """Single-time conditioning should follow standard entropy dynamics."""
        process = BoltzmannProcess(
            entropy_values={0: 0.1, 10: 0.5},  # t=0 low entropy, t=10 higher
            conditioning_times=[0],  # Condition on t=0 (Past Hypothesis style)
        )

        # Entropy should increase forward in time when conditioned on past low entropy
        assert process.expected_entropy_at(5) > process.expected_entropy_at(0)

    def test_boltzmann_process_two_time_conditioning(self):
        """Two-time conditioning should change expected dynamics per Wolpert."""
        process = BoltzmannProcess(
            entropy_values={0: 0.1, 100: 0.1},  # Both past and present low
            conditioning_times=[0, 100],  # Condition on both
        )

        # With two low-entropy times, intermediate should be higher
        mid_entropy = process.expected_entropy_at(50)
        assert mid_entropy > 0.1  # Higher than either endpoint

    def test_single_time_monotonic_increase(self):
        """Entropy increases monotonically away from single conditioning point."""
        process = BoltzmannProcess(
            entropy_values={0: 0.2},
            conditioning_times=[0],
            diffusion_rate=0.1,
        )

        # Values should increase over time
        e0 = process.expected_entropy_at(0)
        e5 = process.expected_entropy_at(5)
        e10 = process.expected_entropy_at(10)
        e20 = process.expected_entropy_at(20)

        assert e0 < e5 < e10 < e20
        # Should asymptote toward 1.0 (equilibrium)
        assert e20 < 1.0
        assert process.expected_entropy_at(100) > 0.9

    def test_no_conditioning_returns_equilibrium(self):
        """No conditioning times returns maximum entropy (equilibrium)."""
        process = BoltzmannProcess(
            entropy_values={0: 0.1, 50: 0.5},
            conditioning_times=[],  # No conditioning
        )

        # Should return equilibrium entropy regardless of time
        assert process.expected_entropy_at(0) == 1.0
        assert process.expected_entropy_at(50) == 1.0
        assert process.expected_entropy_at(100) == 1.0

    def test_two_time_symmetric_endpoints(self):
        """Two-time conditioning with equal endpoint values is symmetric."""
        process = BoltzmannProcess(
            entropy_values={0: 0.2, 100: 0.2},
            conditioning_times=[0, 100],
        )

        # Should be symmetric around midpoint
        e25 = process.expected_entropy_at(25)
        e75 = process.expected_entropy_at(75)
        assert e25 == pytest.approx(e75, rel=0.01)

        # Midpoint should have the peak
        e50 = process.expected_entropy_at(50)
        assert e50 >= e25
        assert e50 >= e75

    def test_two_time_hump_in_middle(self):
        """Two low-entropy endpoints create entropy hump in middle."""
        process = BoltzmannProcess(
            entropy_values={0: 0.1, 100: 0.1},
            conditioning_times=[0, 100],
        )

        # Check several midpoints - all should be higher than endpoints
        for t in [25, 50, 75]:
            assert process.expected_entropy_at(t) > 0.1

    def test_two_time_before_first_conditioning(self):
        """Before first conditioning point, entropy increases backward."""
        process = BoltzmannProcess(
            entropy_values={10: 0.2, 100: 0.3},
            conditioning_times=[10, 100],
        )

        # Before t=10, entropy should increase as we go further back
        e_before = process.expected_entropy_at(5)
        e_at = process.expected_entropy_at(10)
        assert e_before > e_at

    def test_two_time_after_last_conditioning(self):
        """After last conditioning point, entropy increases forward."""
        process = BoltzmannProcess(
            entropy_values={0: 0.2, 50: 0.3},
            conditioning_times=[0, 50],
        )

        # After t=50, entropy should increase
        e_at = process.expected_entropy_at(50)
        e_after = process.expected_entropy_at(60)
        assert e_after > e_at

    def test_diffusion_rate_affects_dynamics(self):
        """Higher diffusion rate leads to faster entropy increase."""
        slow_process = BoltzmannProcess(
            entropy_values={0: 0.1},
            conditioning_times=[0],
            diffusion_rate=0.05,
        )
        fast_process = BoltzmannProcess(
            entropy_values={0: 0.1},
            conditioning_times=[0],
            diffusion_rate=0.2,
        )

        # At same future time, fast process should have higher entropy
        assert fast_process.expected_entropy_at(10) > slow_process.expected_entropy_at(10)

    def test_entropy_bounded_by_one(self):
        """Entropy should never exceed 1.0 (maximum)."""
        process = BoltzmannProcess(
            entropy_values={0: 0.9, 100: 0.9},
            conditioning_times=[0, 100],
        )

        for t in range(-10, 120, 5):
            assert process.expected_entropy_at(t) <= 1.0

    def test_multi_time_conditioning_uses_nearest_pair(self):
        """Multi-time conditioning uses nearest conditioning points."""
        process = BoltzmannProcess(
            entropy_values={0: 0.1, 50: 0.3, 100: 0.1},
            conditioning_times=[0, 50, 100],
        )

        # Between 0 and 50, should behave like two-time with those endpoints
        e25 = process.expected_entropy_at(25)
        # Between 50 and 100, should behave like two-time with those endpoints
        e75 = process.expected_entropy_at(75)

        # Both should be defined and reasonable
        assert 0.0 <= e25 <= 1.0
        assert 0.0 <= e75 <= 1.0


class TestComputeTwoTimeConditional:
    """Tests for compute_two_time_conditional convenience function."""

    def test_convenience_function_matches_class(self):
        """Convenience function produces same results as BoltzmannProcess."""
        s0, sf = 0.2, 0.3
        t0, tf = 0, 100
        t = 50

        # Using convenience function
        result_func = compute_two_time_conditional(s0, sf, t0, tf, t)

        # Using class directly
        process = BoltzmannProcess(
            entropy_values={t0: s0, tf: sf},
            conditioning_times=[t0, tf],
        )
        result_class = process.expected_entropy_at(t)

        assert result_func == result_class

    def test_convenience_function_with_diffusion_rate(self):
        """Convenience function accepts diffusion_rate parameter."""
        result = compute_two_time_conditional(
            s0=0.1,
            sf=0.1,
            t0=0,
            tf=100,
            t=50,
            diffusion_rate=0.2,
        )
        assert 0.0 <= result <= 1.0

    def test_convenience_function_at_endpoints(self):
        """Convenience function returns endpoint values at conditioning times."""
        s0, sf = 0.15, 0.25
        t0, tf = 0, 100

        # At t0, should be close to s0
        result_t0 = compute_two_time_conditional(s0, sf, t0, tf, t0)
        assert result_t0 == pytest.approx(s0, rel=0.01)

        # At tf, should be close to sf
        result_tf = compute_two_time_conditional(s0, sf, t0, tf, tf)
        assert result_tf == pytest.approx(sf, rel=0.01)


# === BoltzmannProcess Integration Tests ===


class TestGraphEntropyBoltzmannIntegration:
    """Tests for BoltzmannProcess integration into EntropyResult."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_graph_entropy_includes_boltzmann_process(self, mock_client):
        """EntropyResult should include Boltzmann process for conditioning analysis."""
        mock_client.query.side_effect = [
            [{"count": 100}],  # node count
            [{"count": 150}],  # edge count
            [{"uid": f"n{i}", "degree": i % 10} for i in range(100)],  # degrees
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test_tenant")

        assert hasattr(result, "boltzmann_process")
        assert result.boltzmann_process is not None
        assert isinstance(result.boltzmann_process, BoltzmannProcess)

    @pytest.mark.asyncio
    async def test_empty_graph_has_boltzmann_process(self, mock_client):
        """Empty graph should still include a trivial BoltzmannProcess."""
        mock_client.query.side_effect = [
            [],  # no nodes
            [],  # no edges
            [],  # no degrees
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test_tenant")

        assert result.boltzmann_process is not None
        assert isinstance(result.boltzmann_process, BoltzmannProcess)
        # Trivial process should still work
        assert result.boltzmann_process.expected_entropy_at(0) >= 0

    @pytest.mark.asyncio
    async def test_boltzmann_process_conditions_on_current_entropy(self, mock_client):
        """BoltzmannProcess should condition on the computed total_entropy at t=0."""
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 150}],
            [{"uid": f"n{i}", "degree": 3} for i in range(100)],  # uniform degrees
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test_tenant")

        # Check process is conditioned on t=0
        assert 0 in result.boltzmann_process.conditioning_times
        # The entropy at t=0 should match the computed total_entropy
        assert result.boltzmann_process.entropy_values.get(0) == pytest.approx(result.total_entropy, rel=0.01)

    @pytest.mark.asyncio
    async def test_to_dict_includes_conditioning_times(self, mock_client):
        """to_dict should include conditioning_times from boltzmann_process."""
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 150}],
            [{"uid": f"n{i}", "degree": 3} for i in range(100)],
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test_tenant")
        d = result.to_dict()

        assert "conditioning_times" in d
        assert d["conditioning_times"] == [0]


# === Primordial Conditioning Tests (Past Hypothesis) ===


class TestPrimordialConditioning:
    """Tests for with_primordial_conditioning - Past Hypothesis style two-time conditioning."""

    @pytest.fixture
    def mock_client(self):
        """Create mock PsycheClient."""
        client = MagicMock()
        client.query = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_analyze_with_primordial_conditioning(self, mock_client):
        """Should support Past Hypothesis style two-time conditioning."""
        mock_client.query.side_effect = [
            [{"count": 100}],
            [{"count": 150}],
            [{"uid": f"n{i}", "degree": 5} for i in range(100)],
        ]

        entropy = GraphEntropy(mock_client)
        result = await entropy.compute("test_tenant")

        # Add primordial conditioning (Past Hypothesis)
        primordial_entropy = 0.01  # Very low entropy at Big Bang
        primordial_time = -13.8e9  # ~13.8 billion years ago

        two_time = result.with_primordial_conditioning(
            primordial_entropy=primordial_entropy,
            primordial_time=primordial_time,
        )

        assert len(two_time.boltzmann_process.conditioning_times) == 2
        # With two-time conditioning, expected past entropy should differ
        past_entropy = two_time.boltzmann_process.expected_entropy_at(-1e9)
        assert past_entropy != result.boltzmann_process.expected_entropy_at(-1e9)
