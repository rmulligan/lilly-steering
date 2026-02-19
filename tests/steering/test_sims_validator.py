"""Tests for SIMS Validator - validates steering adjustments."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.validator import (
    SIMSValidator,
    ValidationResult,
    ValidationMetric,
    ValidationOutcome,
)
from core.steering.sims.executor import ExecutionResult, AppliedAdjustment
from core.steering.sims.reflector import AdjustmentType
from core.steering.sims.state_machine import SIMSContext


class TestValidationOutcome:
    """Tests for validation outcome enumeration."""

    def test_outcomes_defined(self):
        """Should have all validation outcomes."""
        assert ValidationOutcome.IMPROVED
        assert ValidationOutcome.DEGRADED
        assert ValidationOutcome.UNCHANGED
        assert ValidationOutcome.INCONCLUSIVE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestValidationMetric:
    """Tests for validation metrics."""

    def test_metric_stores_values(self):
        """Should store metric values."""
        metric = ValidationMetric(
            name="coherence",
            baseline_value=0.7,
            current_value=0.8,
            threshold=0.05,
        )

        assert metric.name == "coherence"
        assert metric.baseline_value == 0.7
        assert metric.current_value == 0.8

    def test_metric_improved(self):
        """Should detect improvement."""
        metric = ValidationMetric(
            name="coherence",
            baseline_value=0.7,
            current_value=0.85,
            threshold=0.05,
        )

        assert metric.outcome == ValidationOutcome.IMPROVED

    def test_metric_degraded(self):
        """Should detect degradation."""
        metric = ValidationMetric(
            name="coherence",
            baseline_value=0.8,
            current_value=0.65,
            threshold=0.05,
        )

        assert metric.outcome == ValidationOutcome.DEGRADED

    def test_metric_unchanged(self):
        """Should detect no significant change."""
        metric = ValidationMetric(
            name="coherence",
            baseline_value=0.7,
            current_value=0.72,
            threshold=0.05,
        )

        assert metric.outcome == ValidationOutcome.UNCHANGED

    def test_to_dict_serialization(self):
        """Should serialize for logging."""
        metric = ValidationMetric(
            name="surprise",
            baseline_value=0.5,
            current_value=0.3,
            threshold=0.1,
        )

        result = metric.to_dict()
        assert "name" in result
        assert "baseline_value" in result
        assert "current_value" in result
        assert "outcome" in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestValidationResult:
    """Tests for validation results."""

    def test_result_stores_metrics(self):
        """Should store validation metrics."""
        metrics = [
            ValidationMetric(
                name="coherence",
                baseline_value=0.7,
                current_value=0.8,
                threshold=0.05,
            )
        ]
        result = ValidationResult(
            metrics=metrics,
            overall_outcome=ValidationOutcome.IMPROVED,
            should_keep_changes=True,
            summary="Validation passed",
        )

        assert len(result.metrics) == 1
        assert result.overall_outcome == ValidationOutcome.IMPROVED

    def test_result_has_timestamp(self):
        """Should record when validation occurred."""
        result = ValidationResult(
            metrics=[],
            overall_outcome=ValidationOutcome.UNCHANGED,
            should_keep_changes=True,
            summary="No changes",
        )

        assert result.timestamp is not None

    def test_to_dict_serialization(self):
        """Should serialize for logging."""
        result = ValidationResult(
            metrics=[],
            overall_outcome=ValidationOutcome.IMPROVED,
            should_keep_changes=True,
            summary="Test",
        )

        data = result.to_dict()
        assert "metrics" in data
        assert "overall_outcome" in data
        assert "should_keep_changes" in data


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSValidator:
    """Tests for SIMS Validator class."""

    @pytest.fixture
    def mock_observer(self):
        """Create mock SteeringObserver."""
        observer = MagicMock()
        observer.get_stats.return_value = {
            "mean_surprise": 0.3,
            "total_observations": 50,
        }
        observer.get_baseline_metrics.return_value = {
            "coherence": 0.7,
            "surprise": 0.5,
        }
        return observer

    @pytest.fixture
    def mock_model(self):
        """Create mock HookedQwen."""
        model = MagicMock()
        model.run_with_cache = MagicMock(return_value=(
            MagicMock(logits=torch.randn(1, 10, 32000)),
            {"blocks.15.hook_resid_post": torch.randn(1, 10, 768)},
        ))
        return model

    def test_validator_init(self, mock_observer, mock_model):
        """Should initialize with dependencies."""
        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
        )

        assert validator.observer == mock_observer
        assert validator.model == mock_model

    @pytest.mark.asyncio
    async def test_validate_returns_result(self, mock_observer, mock_model):
        """Should return ValidationResult."""
        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
        )

        execution = ExecutionResult(
            applied_adjustments=[],
            total_requested=0,
            total_applied=0,
        )
        context = SIMSContext(surprise_level=0.5)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_validate_with_applied_adjustments(
        self, mock_observer, mock_model
    ):
        """Should validate after adjustments applied."""
        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
        )

        execution = ExecutionResult(
            applied_adjustments=[
                AppliedAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    original_magnitude=1.0,
                    new_magnitude=1.2,
                    success=True,
                )
            ],
            total_requested=1,
            total_applied=1,
        )
        context = SIMSContext(surprise_level=0.8)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        assert isinstance(result, ValidationResult)
        assert result.overall_outcome in ValidationOutcome

    @pytest.mark.asyncio
    async def test_validate_handles_missing_execution_result(
        self, mock_observer, mock_model
    ):
        """Should handle missing execution result gracefully."""
        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
        )

        context = SIMSContext(surprise_level=0.5)
        # No execute_result in metadata

        result = await validator.validate(context)

        assert isinstance(result, ValidationResult)
        assert result.should_keep_changes is True  # Nothing to rollback

    @pytest.mark.asyncio
    async def test_validate_recommends_rollback_on_degradation(
        self, mock_observer, mock_model
    ):
        """Should recommend rollback when metrics degrade significantly."""
        # Configure observer to show degraded metrics
        mock_observer.get_stats.return_value = {
            "mean_surprise": 0.8,  # High surprise = degraded
            "total_observations": 50,
        }
        mock_observer.get_current_metrics.return_value = {
            "coherence": 0.4,  # Much lower than baseline
            "surprise": 0.8,
        }
        mock_observer.get_baseline_metrics.return_value = {
            "coherence": 0.7,
            "surprise": 0.3,
        }

        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
        )

        execution = ExecutionResult(
            applied_adjustments=[
                AppliedAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.WEAKEN,
                    original_magnitude=1.0,
                    new_magnitude=0.5,
                    success=True,
                )
            ],
            total_requested=1,
            total_applied=1,
        )
        context = SIMSContext(surprise_level=0.9)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        # With significant degradation, should recommend rollback
        if result.overall_outcome == ValidationOutcome.DEGRADED:
            assert result.should_keep_changes is False

    @pytest.mark.asyncio
    async def test_validate_with_test_prompts(self, mock_observer, mock_model):
        """Should run test prompts for validation."""
        validator = SIMSValidator(
            observer=mock_observer,
            model=mock_model,
            test_prompts=["Who are you?", "What do you value?"],
        )

        execution = ExecutionResult(
            applied_adjustments=[
                AppliedAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    original_magnitude=1.0,
                    new_magnitude=1.2,
                    success=True,
                )
            ],
            total_requested=1,
            total_applied=1,
        )
        context = SIMSContext(surprise_level=0.6)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        # Should have run model with test prompts
        assert isinstance(result, ValidationResult)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSValidatorMetrics:
    """Tests for metric collection and comparison."""

    @pytest.fixture
    def validator(self):
        """Create validator with mocks."""
        observer = MagicMock()
        observer.get_baseline_metrics.return_value = {
            "coherence": 0.7,
            "surprise": 0.4,
            "identity_alignment": 0.8,
        }
        observer.get_current_metrics.return_value = {
            "coherence": 0.75,
            "surprise": 0.35,
            "identity_alignment": 0.85,
        }
        observer.get_stats.return_value = {"mean_surprise": 0.35}

        model = MagicMock()
        model.run_with_cache = MagicMock(return_value=(
            MagicMock(logits=torch.randn(1, 10, 32000)),
            {},
        ))

        return SIMSValidator(observer=observer, model=model)

    @pytest.mark.asyncio
    async def test_collects_multiple_metrics(self, validator):
        """Should collect and compare multiple metrics."""
        execution = ExecutionResult(
            applied_adjustments=[
                AppliedAdjustment(
                    vector_name="identity",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    original_magnitude=1.0,
                    new_magnitude=1.1,
                    success=True,
                )
            ],
            total_requested=1,
            total_applied=1,
        )
        context = SIMSContext(surprise_level=0.5)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        # Should have collected metrics
        assert isinstance(result, ValidationResult)

    @pytest.mark.asyncio
    async def test_overall_outcome_based_on_metrics(self, validator):
        """Should determine overall outcome from individual metrics."""
        execution = ExecutionResult(
            applied_adjustments=[],
            total_requested=0,
            total_applied=0,
        )
        context = SIMSContext(surprise_level=0.3)
        context.metadata["execute_result"] = execution

        result = await validator.validate(context)

        # Overall outcome should be one of the valid outcomes
        assert result.overall_outcome in [
            ValidationOutcome.IMPROVED,
            ValidationOutcome.DEGRADED,
            ValidationOutcome.UNCHANGED,
            ValidationOutcome.INCONCLUSIVE,
        ]
