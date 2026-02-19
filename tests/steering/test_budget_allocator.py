"""Tests for BudgetAllocator steering budget distribution.

Tests the BudgetAllocator which distributes steering magnitude budget between
hypothesis-driven process vectors and Evalatis emergent vectors. This replaces
magnitude capping with proportional allocation to avoid homogenized steering.

Key test scenarios:
- Basic budget split between process and emergent shares
- Proportional allocation by effectiveness score
- Vector normalization to allocated magnitudes
- Edge cases: empty vectors, single vector, zero effectiveness
"""

import pytest
import numpy as np


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_hypothesis_vectors():
    """Create sample HypothesisSteeringVectors for testing."""
    from core.steering.hypothesis_vectors import HypothesisSteeringVector

    # Three vectors with different effectiveness scores
    vec1 = HypothesisSteeringVector(
        uid="hsv_001",
        hypothesis_uid="hyp_001",
        cognitive_operation="explore_emergence",
        vector_data=[1.0, 0.0, 0.0, 0.0],  # Unit vector along axis 0
        layer=15,
        effectiveness_score=0.8,
    )
    vec2 = HypothesisSteeringVector(
        uid="hsv_002",
        hypothesis_uid="hyp_002",
        cognitive_operation="deepen_understanding",
        vector_data=[0.0, 1.0, 0.0, 0.0],  # Unit vector along axis 1
        layer=15,
        effectiveness_score=0.6,
    )
    vec3 = HypothesisSteeringVector(
        uid="hsv_003",
        hypothesis_uid="hyp_003",
        cognitive_operation="seek_contradiction",
        vector_data=[0.0, 0.0, 1.0, 0.0],  # Unit vector along axis 2
        layer=15,
        effectiveness_score=0.4,
    )
    return [vec1, vec2, vec3]


@pytest.fixture
def sample_evalatis_vector():
    """Create a sample Evalatis emergent vector."""
    # 4D vector with magnitude 2.0
    return np.array([1.0, 1.0, 1.0, 1.0])  # magnitude = 2.0


@pytest.fixture
def single_hypothesis_vector():
    """Create a single HypothesisSteeringVector for edge case testing."""
    from core.steering.hypothesis_vectors import HypothesisSteeringVector

    return HypothesisSteeringVector(
        uid="hsv_single",
        hypothesis_uid="hyp_single",
        cognitive_operation="explore",
        vector_data=[3.0, 4.0, 0.0, 0.0],  # magnitude = 5.0
        layer=15,
        effectiveness_score=0.7,
    )


@pytest.fixture
def zero_effectiveness_vectors():
    """Create vectors with zero effectiveness scores."""
    from core.steering.hypothesis_vectors import HypothesisSteeringVector

    vec1 = HypothesisSteeringVector(
        uid="hsv_zero1",
        hypothesis_uid="hyp_zero1",
        cognitive_operation="op1",
        vector_data=[1.0, 0.0, 0.0, 0.0],
        layer=15,
        effectiveness_score=0.0,
    )
    vec2 = HypothesisSteeringVector(
        uid="hsv_zero2",
        hypothesis_uid="hyp_zero2",
        cognitive_operation="op2",
        vector_data=[0.0, 1.0, 0.0, 0.0],
        layer=15,
        effectiveness_score=0.0,
    )
    return [vec1, vec2]


# =============================================================================
# BudgetAllocator Initialization Tests
# =============================================================================


class TestBudgetAllocatorInitialization:
    """Tests for BudgetAllocator initialization."""

    def test_initialization_with_defaults(self):
        """BudgetAllocator should initialize with default process share."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()

        assert allocator.default_process_share == 0.6

    def test_initialization_with_custom_process_share(self):
        """BudgetAllocator should accept custom process share."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator(default_process_share=0.7)

        assert allocator.default_process_share == 0.7


# =============================================================================
# normalize_to_magnitude Tests
# =============================================================================


class TestNormalizeToMagnitude:
    """Tests for the normalize_to_magnitude helper method."""

    def test_normalize_unit_vector(self):
        """Normalizing a unit vector should scale to target magnitude."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        target = 2.5

        result = allocator.normalize_to_magnitude(vector, target)

        assert np.isclose(np.linalg.norm(result), target)
        # Direction should be preserved
        assert np.allclose(result / np.linalg.norm(result), vector)

    def test_normalize_arbitrary_vector(self):
        """Normalizing an arbitrary vector should scale correctly."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        vector = np.array([3.0, 4.0, 0.0])  # magnitude = 5.0
        target = 1.0

        result = allocator.normalize_to_magnitude(vector, target)

        assert np.isclose(np.linalg.norm(result), target)

    def test_normalize_zero_vector(self):
        """Normalizing a zero vector should return zero vector."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        vector = np.array([0.0, 0.0, 0.0])
        target = 2.0

        result = allocator.normalize_to_magnitude(vector, target)

        assert np.allclose(result, np.zeros(3))

    def test_normalize_zero_target(self):
        """Normalizing to zero magnitude should return zero vector."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        vector = np.array([1.0, 2.0, 3.0])
        target = 0.0

        result = allocator.normalize_to_magnitude(vector, target)

        assert np.allclose(result, np.zeros(3))


# =============================================================================
# Budget Split Tests
# =============================================================================


class TestBudgetSplit:
    """Tests for the budget split logic between process and emergent."""

    def test_default_60_40_split(
        self, sample_hypothesis_vectors, sample_evalatis_vector
    ):
        """Default split should be 60% process, 40% emergent."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        process_magnitude = np.linalg.norm(process_vec)
        emergent_magnitude = np.linalg.norm(emergent_vec)

        # Process should get 60% = 1.2
        assert np.isclose(process_magnitude, 1.2, atol=0.01)
        # Emergent should get 40% = 0.8
        assert np.isclose(emergent_magnitude, 0.8, atol=0.01)

    def test_custom_split(self, sample_hypothesis_vectors, sample_evalatis_vector):
        """Custom process_share should be respected."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator(default_process_share=0.5)
        total_budget = 4.0

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        process_magnitude = np.linalg.norm(process_vec)
        emergent_magnitude = np.linalg.norm(emergent_vec)

        # Each should get 50% = 2.0
        assert np.isclose(process_magnitude, 2.0, atol=0.01)
        assert np.isclose(emergent_magnitude, 2.0, atol=0.01)

    def test_override_process_share(
        self, sample_hypothesis_vectors, sample_evalatis_vector
    ):
        """process_share parameter should override default."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator(default_process_share=0.6)
        total_budget = 2.0

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
            process_share=0.8,  # Override to 80%
        )

        process_magnitude = np.linalg.norm(process_vec)
        emergent_magnitude = np.linalg.norm(emergent_vec)

        # Process should get 80% = 1.6
        assert np.isclose(process_magnitude, 1.6, atol=0.01)
        # Emergent should get 20% = 0.4
        assert np.isclose(emergent_magnitude, 0.4, atol=0.01)


# =============================================================================
# Proportional Allocation by Effectiveness Tests
# =============================================================================


class TestProportionalAllocation:
    """Tests for proportional allocation within process budget."""

    def test_allocation_proportional_to_effectiveness(self, sample_hypothesis_vectors):
        """Each vector's budget should be proportional to effectiveness."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0
        process_share = 0.6
        process_budget = total_budget * process_share  # = 1.2

        # effectiveness_scores: [0.8, 0.6, 0.4]
        # total_effectiveness = 1.8
        # Expected allocations:
        # vec1: 1.2 * (0.8 / 1.8) = 0.533...
        # vec2: 1.2 * (0.6 / 1.8) = 0.400
        # vec3: 1.2 * (0.4 / 1.8) = 0.267...

        expected_magnitudes = [
            process_budget * (0.8 / 1.8),
            process_budget * (0.6 / 1.8),
            process_budget * (0.4 / 1.8),
        ]

        # Compute individual allocations
        budgets = allocator._compute_effectiveness_budgets(
            sample_hypothesis_vectors, process_budget
        )

        for budget, expected in zip(budgets, expected_magnitudes):
            assert np.isclose(budget, expected, atol=0.001)

    def test_single_vector_gets_full_budget(self, single_hypothesis_vector):
        """A single hypothesis vector should get the full process budget."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0
        process_share = 0.6
        process_budget = total_budget * process_share  # = 1.2

        budgets = allocator._compute_effectiveness_budgets(
            [single_hypothesis_vector], process_budget
        )

        assert len(budgets) == 1
        assert np.isclose(budgets[0], process_budget)


# =============================================================================
# Vector Combination Tests
# =============================================================================


class TestVectorCombination:
    """Tests for combining hypothesis vectors into process vector."""

    def test_combined_vector_has_process_magnitude(
        self, sample_hypothesis_vectors, sample_evalatis_vector
    ):
        """Combined process vector should have exact process budget magnitude."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0
        process_budget = total_budget * 0.6  # = 1.2

        process_vec, _ = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        assert np.isclose(np.linalg.norm(process_vec), process_budget, atol=0.01)

    def test_orthogonal_vectors_combine_correctly(
        self, sample_hypothesis_vectors, sample_evalatis_vector
    ):
        """Orthogonal vectors should combine with correct Pythagorean magnitude."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0
        process_budget = total_budget * 0.6  # = 1.2

        process_vec, _ = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        # For orthogonal unit vectors scaled to budgets [a, b, c]:
        # combined magnitude = sqrt(a^2 + b^2 + c^2)
        # But we normalize to process_budget, so final magnitude = process_budget
        assert np.isclose(np.linalg.norm(process_vec), process_budget, atol=0.01)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_hypothesis_vectors(self, sample_evalatis_vector):
        """Empty hypothesis list should return zero process vector."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=[],
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        # Process vector should be zeros
        assert np.allclose(process_vec, np.zeros_like(sample_evalatis_vector))
        # Emergent should still get full emergent budget (or full budget?)
        # Design decision: emergent gets its allocated share regardless
        assert np.isclose(np.linalg.norm(emergent_vec), total_budget * 0.4, atol=0.01)

    def test_zero_evalatis_vector(self, sample_hypothesis_vectors):
        """Zero evalatis vector should remain zero."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0
        zero_evalatis = np.zeros(4)

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=zero_evalatis,
            total_budget=total_budget,
        )

        # Emergent should be zero (can't normalize zero vector)
        assert np.allclose(emergent_vec, np.zeros(4))
        # Process should still work
        assert np.isclose(np.linalg.norm(process_vec), total_budget * 0.6, atol=0.01)

    def test_zero_effectiveness_scores_equal_split(self, zero_effectiveness_vectors):
        """Vectors with zero effectiveness should get equal shares."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        process_budget = 1.2

        budgets = allocator._compute_effectiveness_budgets(
            zero_effectiveness_vectors, process_budget
        )

        # With all zero effectiveness, should split equally
        expected = process_budget / len(zero_effectiveness_vectors)
        for budget in budgets:
            assert np.isclose(budget, expected)

    def test_zero_total_budget(self, sample_hypothesis_vectors, sample_evalatis_vector):
        """Zero total budget should return zero vectors."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=0.0,
        )

        assert np.allclose(process_vec, np.zeros_like(sample_evalatis_vector))
        assert np.allclose(emergent_vec, np.zeros_like(sample_evalatis_vector))

    def test_single_vector_allocation(
        self, single_hypothesis_vector, sample_evalatis_vector
    ):
        """Single hypothesis vector should get full process budget."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()
        total_budget = 2.0

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=[single_hypothesis_vector],
            evalatis_vector=sample_evalatis_vector,
            total_budget=total_budget,
        )

        # Process should have 60% of budget
        assert np.isclose(np.linalg.norm(process_vec), 1.2, atol=0.01)
        # Emergent should have 40%
        assert np.isclose(np.linalg.norm(emergent_vec), 0.8, atol=0.01)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full allocation workflow."""

    def test_full_allocation_example_from_spec(self):
        """Test the example from the spec document."""
        from core.steering.budget_allocator import BudgetAllocator
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        # Create vectors as in spec example
        vec1 = HypothesisSteeringVector(
            uid="hsv_1",
            hypothesis_uid="hyp_1",
            cognitive_operation="op1",
            vector_data=[1.0, 0.0, 0.0, 0.0],
            layer=15,
            effectiveness_score=0.8,
        )
        vec2 = HypothesisSteeringVector(
            uid="hsv_2",
            hypothesis_uid="hyp_2",
            cognitive_operation="op2",
            vector_data=[0.0, 1.0, 0.0, 0.0],
            layer=15,
            effectiveness_score=0.6,
        )
        vec3 = HypothesisSteeringVector(
            uid="hsv_3",
            hypothesis_uid="hyp_3",
            cognitive_operation="op3",
            vector_data=[0.0, 0.0, 1.0, 0.0],
            layer=15,
            effectiveness_score=0.4,
        )

        evalatis = np.array([1.0, 0.0, 0.0, 0.0])
        allocator = BudgetAllocator()

        # From spec:
        # total_budget = 2.0, process_share = 0.6
        # process_budget = 1.2, emergent_budget = 0.8
        # effectiveness_sum = 1.8
        # vec1 gets 1.2 * (0.8/1.8) = 0.533 magnitude
        # vec2 gets 1.2 * (0.6/1.8) = 0.400 magnitude
        # vec3 gets 1.2 * (0.4/1.8) = 0.267 magnitude

        process_vec, emergent_vec = allocator.allocate(
            hypothesis_vectors=[vec1, vec2, vec3],
            evalatis_vector=evalatis,
            total_budget=2.0,
            process_share=0.6,
        )

        # Verify magnitudes
        assert np.isclose(np.linalg.norm(process_vec), 1.2, atol=0.01)
        assert np.isclose(np.linalg.norm(emergent_vec), 0.8, atol=0.01)

    def test_allocation_preserves_vector_directions(self):
        """Allocation should preserve relative vector directions."""
        from core.steering.budget_allocator import BudgetAllocator
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        # Create a vector with a specific direction
        vec = HypothesisSteeringVector(
            uid="hsv_dir",
            hypothesis_uid="hyp_dir",
            cognitive_operation="op",
            vector_data=[3.0, 4.0, 0.0, 0.0],  # 3-4-5 triangle, magnitude 5
            layer=15,
            effectiveness_score=0.5,
        )

        evalatis = np.array([1.0, 1.0, 1.0, 1.0])
        allocator = BudgetAllocator()

        process_vec, _ = allocator.allocate(
            hypothesis_vectors=[vec],
            evalatis_vector=evalatis,
            total_budget=2.0,
        )

        # Direction should be preserved (normalized to same direction)
        original_dir = np.array(vec.vector_data) / np.linalg.norm(vec.vector_data)
        result_dir = process_vec / np.linalg.norm(process_vec)

        assert np.allclose(result_dir, original_dir)

    def test_repeated_allocations_are_consistent(
        self, sample_hypothesis_vectors, sample_evalatis_vector
    ):
        """Multiple calls with same inputs should produce same outputs."""
        from core.steering.budget_allocator import BudgetAllocator

        allocator = BudgetAllocator()

        result1 = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=2.0,
        )

        result2 = allocator.allocate(
            hypothesis_vectors=sample_hypothesis_vectors,
            evalatis_vector=sample_evalatis_vector,
            total_budget=2.0,
        )

        assert np.allclose(result1[0], result2[0])
        assert np.allclose(result1[1], result2[1])
