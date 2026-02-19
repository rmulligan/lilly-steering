"""Tests for cross-zone coherence tracking between fast and slow steering zones.

Tests the CrossZoneCoherence class which tracks alignment between fast-adapting
exploration zones and slow-adapting identity zones. Brain research shows that
cross-timescale coherence correlates with cognitive performance.

Uses numpy for vector operations.
"""

import pytest

# Handle optional numpy dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Skip all tests if numpy not available
pytestmark = pytest.mark.skipif(
    not NUMPY_AVAILABLE,
    reason="numpy not installed"
)


# =============================================================================
# Test Constants
# =============================================================================

VECTOR_DIM = 4
HISTORY_SIZE = 5


# =============================================================================
# CrossZoneCoherence Initialization Tests
# =============================================================================

class TestCrossZoneCoherenceInitialization:
    """Tests for CrossZoneCoherence initialization."""

    def test_coherence_initialization_defaults(self):
        """CrossZoneCoherence should initialize with default history size."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        assert coherence.history_size == 50
        assert len(coherence.history) == 0

    def test_coherence_initialization_custom_history_size(self):
        """CrossZoneCoherence should accept custom history size."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence(history_size=10)

        assert coherence.history_size == 10


# =============================================================================
# Coherence Computation Tests
# =============================================================================

class TestCoherenceComputation:
    """Tests for coherence computation."""

    def test_coherence_computation_aligned_vectors(self):
        """Cross-zone coherence should measure alignment between zones."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Aligned vectors should have high coherence
        fast_vector = np.array([1.0, 0.0, 0.0, 0.0])
        slow_vector = np.array([0.9, 0.1, 0.0, 0.0])

        score = coherence.compute(fast_vector, slow_vector)
        assert 0.9 < score <= 1.0, "Aligned vectors should have high coherence"

    def test_coherence_computation_orthogonal_vectors(self):
        """Orthogonal vectors should have medium coherence (0.5 after mapping)."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        fast_vector = np.array([1.0, 0.0, 0.0, 0.0])
        orthogonal = np.array([0.0, 1.0, 0.0, 0.0])

        score = coherence.compute(fast_vector, orthogonal)
        # Orthogonal vectors have cosine 0, mapped to 0.5
        assert abs(score - 0.5) < 0.01, "Orthogonal vectors should have ~0.5 coherence"

    def test_coherence_computation_opposite_vectors(self):
        """Opposite vectors should have low coherence (near 0)."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        fast_vector = np.array([1.0, 0.0, 0.0, 0.0])
        opposite = np.array([-1.0, 0.0, 0.0, 0.0])

        score = coherence.compute(fast_vector, opposite)
        # Opposite vectors have cosine -1, mapped to 0
        assert score < 0.1, "Opposite vectors should have low coherence"

    def test_coherence_computation_identical_vectors(self):
        """Identical vectors should have perfect coherence (1.0)."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        vector = np.array([1.0, 2.0, 3.0, 4.0])

        score = coherence.compute(vector, vector)
        assert abs(score - 1.0) < 1e-6, "Identical vectors should have coherence 1.0"

    def test_coherence_computation_zero_vector(self):
        """Zero vector should return coherence 0."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        fast_vector = np.array([1.0, 0.0, 0.0, 0.0])
        zero_vector = np.array([0.0, 0.0, 0.0, 0.0])

        score = coherence.compute(fast_vector, zero_vector)
        assert score == 0.0, "Zero vector should produce coherence 0"

    def test_coherence_computation_returns_float(self):
        """Coherence should return a Python float."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        vec1 = np.random.randn(VECTOR_DIM)
        vec2 = np.random.randn(VECTOR_DIM)

        score = coherence.compute(vec1, vec2)
        assert isinstance(score, float)

    def test_coherence_computation_bounded(self):
        """Coherence should always be in [0, 1]."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        for _ in range(20):
            vec1 = np.random.randn(VECTOR_DIM)
            vec2 = np.random.randn(VECTOR_DIM)
            score = coherence.compute(vec1, vec2)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"


# =============================================================================
# History Tracking Tests
# =============================================================================

class TestCoherenceHistoryTracking:
    """Tests for coherence history tracking."""

    def test_coherence_history_tracking_bounded(self):
        """Coherence tracker should maintain history for trend analysis."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence(history_size=5)

        for i in range(10):
            vec1 = np.random.randn(VECTOR_DIM)
            vec2 = np.random.randn(VECTOR_DIM)
            coherence.record(vec1, vec2, cycle=i)

        assert len(coherence.history) == 5, "Should maintain bounded history"
        assert coherence.trend() is not None, "Should compute trend"

    def test_record_returns_score(self):
        """Record should return the computed coherence score."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0, 0.0])

        score = coherence.record(vec1, vec2, cycle=0)
        assert abs(score - 1.0) < 1e-6

    def test_record_stores_zone_names(self):
        """Record should store zone names in history."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        vec1 = np.random.randn(VECTOR_DIM)
        vec2 = np.random.randn(VECTOR_DIM)

        coherence.record(
            vec1, vec2, cycle=0,
            fast_zone="exploration",
            slow_zone="identity"
        )

        assert coherence.history[0].fast_zone == "exploration"
        assert coherence.history[0].slow_zone == "identity"

    def test_history_preserves_recent_records(self):
        """History should preserve the most recent records."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence(history_size=3)

        for i in range(5):
            vec1 = np.random.randn(VECTOR_DIM)
            vec2 = np.random.randn(VECTOR_DIM)
            coherence.record(vec1, vec2, cycle=i)

        # Should have cycles 2, 3, 4 (most recent 3)
        cycles = [r.cycle for r in coherence.history]
        assert cycles == [2, 3, 4]


# =============================================================================
# Trend Computation Tests
# =============================================================================

class TestTrendComputation:
    """Tests for trend computation."""

    def test_trend_returns_none_insufficient_history(self):
        """Trend should return None with insufficient history."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # No history
        assert coherence.trend() is None

        # One record
        coherence.record(np.random.randn(VECTOR_DIM), np.random.randn(VECTOR_DIM), cycle=0)
        assert coherence.trend() is None

        # Two records
        coherence.record(np.random.randn(VECTOR_DIM), np.random.randn(VECTOR_DIM), cycle=1)
        assert coherence.trend() is None

    def test_trend_positive_for_improving_coherence(self):
        """Trend should be positive when coherence is improving."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Create a series of increasingly aligned vectors
        base = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(5):
            # Vectors become more aligned over time
            alignment = 0.2 + i * 0.15  # 0.2, 0.35, 0.5, 0.65, 0.8
            slow = np.array([alignment, 1 - alignment, 0.0, 0.0])
            slow = slow / np.linalg.norm(slow)
            coherence.record(base, slow, cycle=i)

        trend = coherence.trend()
        assert trend is not None
        assert trend > 0, "Trend should be positive for improving coherence"

    def test_trend_negative_for_declining_coherence(self):
        """Trend should be negative when coherence is declining."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Create a series of increasingly misaligned vectors
        base = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(5):
            # Vectors become more orthogonal over time
            alignment = 0.8 - i * 0.15  # 0.8, 0.65, 0.5, 0.35, 0.2
            slow = np.array([alignment, 1 - alignment, 0.0, 0.0])
            slow = slow / np.linalg.norm(slow)
            coherence.record(base, slow, cycle=i)

        trend = coherence.trend()
        assert trend is not None
        assert trend < 0, "Trend should be negative for declining coherence"

    def test_trend_near_zero_for_stable_coherence(self):
        """Trend should be near zero for stable coherence."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Use identical vectors for stable coherence
        vec1 = np.array([1.0, 0.0, 0.0, 0.0])
        vec2 = np.array([0.9, 0.1, 0.0, 0.0])

        for i in range(5):
            coherence.record(vec1, vec2, cycle=i)

        trend = coherence.trend()
        assert trend is not None
        assert abs(trend) < 0.01, "Trend should be near zero for stable coherence"


# =============================================================================
# Recent Average Tests
# =============================================================================

class TestRecentAverage:
    """Tests for recent average computation."""

    def test_recent_average_empty_history(self):
        """Recent average should return 0.5 for empty history."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        avg = coherence.recent_average()
        assert avg == 0.5

    def test_recent_average_full_history(self):
        """Recent average should average last n records."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Record identical vectors (coherence = 1.0)
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(10):
            coherence.record(vec, vec, cycle=i)

        avg = coherence.recent_average(n=5)
        assert abs(avg - 1.0) < 1e-6

    def test_recent_average_partial_history(self):
        """Recent average should work with less than n records."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence()

        # Record only 3 records
        vec = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(3):
            coherence.record(vec, vec, cycle=i)

        # Request average of last 10, should use all 3
        avg = coherence.recent_average(n=10)
        assert abs(avg - 1.0) < 1e-6

    def test_recent_average_respects_n(self):
        """Recent average should use exactly n most recent records."""
        from core.steering.coherence import CrossZoneCoherence

        coherence = CrossZoneCoherence(history_size=20)

        # First 5: aligned (coherence ~1.0)
        aligned = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(5):
            coherence.record(aligned, aligned, cycle=i)

        # Next 5: orthogonal (coherence ~0.5)
        orthogonal = np.array([0.0, 1.0, 0.0, 0.0])
        for i in range(5, 10):
            coherence.record(aligned, orthogonal, cycle=i)

        # Average of last 5 should be ~0.5 (orthogonal)
        avg = coherence.recent_average(n=5)
        assert abs(avg - 0.5) < 0.01


# =============================================================================
# CoherenceRecord Dataclass Tests
# =============================================================================

class TestCoherenceRecord:
    """Tests for CoherenceRecord dataclass."""

    def test_coherence_record_creation(self):
        """CoherenceRecord should store all fields."""
        from core.steering.coherence import CoherenceRecord

        record = CoherenceRecord(
            cycle=42,
            score=0.85,
            fast_zone="exploration",
            slow_zone="identity"
        )

        assert record.cycle == 42
        assert record.score == 0.85
        assert record.fast_zone == "exploration"
        assert record.slow_zone == "identity"

    def test_coherence_record_defaults(self):
        """CoherenceRecord should have default zone names."""
        from core.steering.coherence import CoherenceRecord

        record = CoherenceRecord(cycle=0, score=0.5)

        assert record.fast_zone == ""
        assert record.slow_zone == ""
