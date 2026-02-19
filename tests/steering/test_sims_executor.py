"""Tests for SIMS Executor - applies steering vector adjustments."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.executor import (
    SIMSExecutor,
    ExecutionResult,
    AppliedAdjustment,
)
from core.steering.sims.reflector import (
    VectorAdjustment,
    AdjustmentType,
    ReflectionResult,
)
from core.steering.sims.state_machine import SIMSContext


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestAppliedAdjustment:
    """Tests for applied adjustment records."""

    def test_stores_adjustment_details(self):
        """Should store details of applied adjustment."""
        original = torch.randn(768)
        modified = torch.randn(768)
        applied = AppliedAdjustment(
            vector_name="identity",
            adjustment_type=AdjustmentType.STRENGTHEN,
            original_magnitude=1.0,
            new_magnitude=1.2,
            success=True,
        )

        assert applied.vector_name == "identity"
        assert applied.adjustment_type == AdjustmentType.STRENGTHEN
        assert applied.success is True

    def test_to_dict_serialization(self):
        """Should serialize for logging."""
        applied = AppliedAdjustment(
            vector_name="autonomy",
            adjustment_type=AdjustmentType.WEAKEN,
            original_magnitude=1.0,
            new_magnitude=0.8,
            success=True,
        )

        result = applied.to_dict()
        assert "vector_name" in result
        assert "adjustment_type" in result
        assert "success" in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestExecutionResult:
    """Tests for execution results."""

    def test_result_stores_applied_adjustments(self):
        """Should store list of applied adjustments."""
        applied = [
            AppliedAdjustment(
                vector_name="identity",
                adjustment_type=AdjustmentType.STRENGTHEN,
                original_magnitude=1.0,
                new_magnitude=1.2,
                success=True,
            )
        ]
        result = ExecutionResult(
            applied_adjustments=applied,
            total_requested=1,
            total_applied=1,
        )

        assert len(result.applied_adjustments) == 1
        assert result.total_applied == 1

    def test_result_has_timestamp(self):
        """Should record when execution occurred."""
        result = ExecutionResult(
            applied_adjustments=[],
            total_requested=0,
            total_applied=0,
        )

        assert result.timestamp is not None

    def test_success_rate(self):
        """Should calculate success rate."""
        result = ExecutionResult(
            applied_adjustments=[],
            total_requested=10,
            total_applied=8,
        )

        assert result.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_requested(self):
        """Should handle zero requested gracefully."""
        result = ExecutionResult(
            applied_adjustments=[],
            total_requested=0,
            total_applied=0,
        )

        assert result.success_rate == pytest.approx(1.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSExecutor:
    """Tests for SIMS Executor class."""

    @pytest.fixture
    def mock_identity_hooks(self):
        """Create mock IdentityHooks."""
        hooks = MagicMock()
        hooks.identity_vector = torch.randn(768)
        hooks.autonomy_vector = torch.randn(768)
        hooks.constitutional_vector = torch.randn(768)

        # Track applied adjustments
        hooks.applied_adjustments = []

        def strengthen_vector(name, factor):
            hooks.applied_adjustments.append(("strengthen", name, factor))

        def weaken_vector(name, factor):
            hooks.applied_adjustments.append(("weaken", name, factor))

        def add_vector(name, vector):
            hooks.applied_adjustments.append(("add", name, vector.shape))

        def remove_vector(name):
            hooks.applied_adjustments.append(("remove", name))

        hooks.strengthen_vector = strengthen_vector
        hooks.weaken_vector = weaken_vector
        hooks.add_vector = add_vector
        hooks.remove_vector = remove_vector

        return hooks

    @pytest.fixture
    def mock_psyche(self):
        """Create mock PsycheClient."""
        psyche = MagicMock()
        psyche.upsert_steering_vector = AsyncMock()
        return psyche

    def test_executor_init(self, mock_identity_hooks, mock_psyche):
        """Should initialize with dependencies."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        assert executor.identity_hooks == mock_identity_hooks
        assert executor.psyche == mock_psyche

    @pytest.mark.asyncio
    async def test_execute_returns_result(self, mock_identity_hooks, mock_psyche):
        """Should return ExecutionResult."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        reflection = ReflectionResult(
            adjustments=[],
            analysis_summary="No adjustments needed",
            confidence=0.5,
        )
        context = SIMSContext(surprise_level=0.5)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_strengthen_adjustment(
        self, mock_identity_hooks, mock_psyche
    ):
        """Should apply STRENGTHEN adjustments."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    magnitude=0.2,
                    reason="Low coherence",
                )
            ],
            analysis_summary="Strengthening identity",
            confidence=0.9,
        )
        context = SIMSContext(surprise_level=0.8)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        assert result.total_applied == 1
        assert ("strengthen", "identity", 0.2) in mock_identity_hooks.applied_adjustments

    @pytest.mark.asyncio
    async def test_execute_weaken_adjustment(
        self, mock_identity_hooks, mock_psyche
    ):
        """Should apply WEAKEN adjustments."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="autonomy",
                    adjustment_type=AdjustmentType.WEAKEN,
                    magnitude=0.1,
                    reason="Too strong",
                )
            ],
            analysis_summary="Weakening autonomy",
            confidence=0.8,
        )
        context = SIMSContext(surprise_level=0.7)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        assert result.total_applied == 1
        assert ("weaken", "autonomy", 0.1) in mock_identity_hooks.applied_adjustments

    @pytest.mark.asyncio
    async def test_execute_add_vector(self, mock_identity_hooks, mock_psyche):
        """Should add new vectors."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        new_vector = torch.randn(768)
        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="curiosity",
                    adjustment_type=AdjustmentType.ADD,
                    magnitude=1.0,
                    reason="New direction",
                    new_vector=new_vector,
                )
            ],
            analysis_summary="Adding curiosity vector",
            confidence=0.85,
        )
        context = SIMSContext(surprise_level=0.6)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        assert result.total_applied == 1
        # Check that add was called
        add_calls = [a for a in mock_identity_hooks.applied_adjustments if a[0] == "add"]
        assert len(add_calls) == 1

    @pytest.mark.asyncio
    async def test_execute_protects_constitutional(
        self, mock_identity_hooks, mock_psyche
    ):
        """Should never remove constitutional vector."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="constitutional",
                    adjustment_type=AdjustmentType.REMOVE,
                    magnitude=1.0,
                    reason="Should be blocked",
                )
            ],
            analysis_summary="Attempting to remove constitutional",
            confidence=0.9,
        )
        context = SIMSContext(surprise_level=0.9)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        # Should not have applied the removal
        assert result.total_applied == 0
        remove_calls = [a for a in mock_identity_hooks.applied_adjustments if a[0] == "remove"]
        assert len(remove_calls) == 0

    @pytest.mark.asyncio
    async def test_execute_handles_missing_reflect_result(
        self, mock_identity_hooks, mock_psyche
    ):
        """Should handle missing reflection result gracefully."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        context = SIMSContext(surprise_level=0.5)
        # No reflect_result in metadata

        result = await executor.execute(context)

        assert isinstance(result, ExecutionResult)
        assert result.total_applied == 0

    @pytest.mark.asyncio
    async def test_execute_multiple_adjustments(
        self, mock_identity_hooks, mock_psyche
    ):
        """Should apply multiple adjustments."""
        executor = SIMSExecutor(
            identity_hooks=mock_identity_hooks,
            psyche=mock_psyche,
        )

        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    magnitude=0.1,
                    reason="Boost identity",
                ),
                VectorAdjustment(
                    vector_name="autonomy",
                    adjustment_type=AdjustmentType.WEAKEN,
                    magnitude=0.05,
                    reason="Reduce autonomy",
                ),
            ],
            analysis_summary="Multiple adjustments",
            confidence=0.85,
        )
        context = SIMSContext(surprise_level=0.7)
        context.metadata["reflect_result"] = reflection

        result = await executor.execute(context)

        assert result.total_requested == 2
        assert result.total_applied == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSExecutorPersistence:
    """Tests for executor persistence to Psyche."""

    @pytest.fixture
    def executor_with_mocks(self):
        """Create executor with mocked dependencies."""
        hooks = MagicMock()
        hooks.applied_adjustments = []
        hooks.strengthen_vector = lambda n, f: hooks.applied_adjustments.append(("strengthen", n, f))
        hooks.weaken_vector = lambda n, f: hooks.applied_adjustments.append(("weaken", n, f))

        psyche = MagicMock()
        psyche.upsert_steering_vector = AsyncMock()

        return SIMSExecutor(identity_hooks=hooks, psyche=psyche)

    @pytest.mark.asyncio
    async def test_persists_to_psyche(self, executor_with_mocks):
        """Should persist adjustments to Psyche."""
        reflection = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    magnitude=0.1,
                    reason="Test persistence",
                )
            ],
            analysis_summary="Test",
            confidence=0.9,
        )
        context = SIMSContext(surprise_level=0.8)
        context.metadata["reflect_result"] = reflection

        await executor_with_mocks.execute(context, persist=True)

        executor_with_mocks.psyche.upsert_steering_vector.assert_called()
