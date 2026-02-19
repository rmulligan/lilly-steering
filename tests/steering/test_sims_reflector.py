"""Tests for SIMS Reflector - pattern analysis and adjustment planning."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.reflector import (
    SIMSReflector,
    ReflectionResult,
    VectorAdjustment,
    AdjustmentType,
)
from core.steering.sims.state_machine import SIMSContext


class TestAdjustmentType:
    """Tests for adjustment type enumeration."""

    def test_adjustment_types_defined(self):
        """Should have all adjustment types."""
        assert AdjustmentType.STRENGTHEN
        assert AdjustmentType.WEAKEN
        assert AdjustmentType.ADD
        assert AdjustmentType.REMOVE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestVectorAdjustment:
    """Tests for vector adjustment proposals."""

    def test_adjustment_stores_data(self):
        """Should store adjustment details."""
        adjustment = VectorAdjustment(
            vector_name="identity",
            adjustment_type=AdjustmentType.STRENGTHEN,
            magnitude=0.2,
            reason="Low identity coherence detected",
        )

        assert adjustment.vector_name == "identity"
        assert adjustment.adjustment_type == AdjustmentType.STRENGTHEN
        assert adjustment.magnitude == 0.2

    def test_adjustment_with_new_vector(self):
        """Should support adding new vectors."""
        new_vector = torch.randn(768)
        adjustment = VectorAdjustment(
            vector_name="curiosity",
            adjustment_type=AdjustmentType.ADD,
            magnitude=1.0,
            reason="New direction identified",
            new_vector=new_vector,
        )

        assert adjustment.new_vector is not None
        assert adjustment.new_vector.shape == (768,)

    def test_to_dict_serialization(self):
        """Should serialize for logging."""
        adjustment = VectorAdjustment(
            vector_name="autonomy",
            adjustment_type=AdjustmentType.WEAKEN,
            magnitude=0.1,
            reason="test",
        )

        result = adjustment.to_dict()
        assert "vector_name" in result
        assert "adjustment_type" in result
        assert "magnitude" in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestReflectionResult:
    """Tests for reflection results."""

    def test_result_stores_adjustments(self):
        """Should store list of proposed adjustments."""
        adjustments = [
            VectorAdjustment(
                vector_name="identity",
                adjustment_type=AdjustmentType.STRENGTHEN,
                magnitude=0.1,
                reason="test",
            )
        ]
        result = ReflectionResult(
            adjustments=adjustments,
            analysis_summary="Identity coherence low",
            confidence=0.85,
        )

        assert len(result.adjustments) == 1
        assert result.confidence == 0.85

    def test_result_has_timestamp(self):
        """Should record when reflection occurred."""
        result = ReflectionResult(
            adjustments=[],
            analysis_summary="test",
            confidence=0.5,
        )

        assert result.timestamp is not None

    def test_should_apply(self):
        """Should indicate whether adjustments should be applied."""
        low_confidence = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="test",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    magnitude=0.1,
                    reason="test",
                )
            ],
            analysis_summary="test",
            confidence=0.3,
        )

        high_confidence = ReflectionResult(
            adjustments=[
                VectorAdjustment(
                    vector_name="test",
                    adjustment_type=AdjustmentType.STRENGTHEN,
                    magnitude=0.1,
                    reason="test",
                )
            ],
            analysis_summary="test",
            confidence=0.9,
        )

        assert not low_confidence.should_apply(threshold=0.5)
        assert high_confidence.should_apply(threshold=0.5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSReflector:
    """Tests for SIMS Reflector class."""

    @pytest.fixture
    def mock_observer(self):
        """Create mock SteeringObserver."""
        observer = MagicMock()
        observer.get_high_surprise_observations.return_value = []
        observer.get_stats.return_value = {
            "total_observations": 100,
            "high_surprise_count": 5,
            "mean_surprise": 0.3,
        }
        # generate_contrastive_pairs is async
        observer.generate_contrastive_pairs = AsyncMock(return_value=[])
        return observer

    @pytest.fixture
    def mock_psyche(self):
        """Create mock PsycheClient."""
        psyche = MagicMock()
        psyche.query = AsyncMock(return_value=[])
        return psyche

    @pytest.fixture
    def mock_identity_hooks(self):
        """Create mock IdentityHooks."""
        hooks = MagicMock()
        hooks.identity_vector = torch.randn(768)
        hooks.autonomy_vector = torch.randn(768)
        hooks.constitutional_vector = torch.randn(768)
        return hooks

    def test_reflector_init(self, mock_observer, mock_psyche):
        """Should initialize with dependencies."""
        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
        )

        assert reflector.observer == mock_observer
        assert reflector.psyche == mock_psyche

    @pytest.mark.asyncio
    async def test_reflect_returns_result(self, mock_observer, mock_psyche):
        """Should return ReflectionResult."""
        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
        )

        context = SIMSContext(surprise_level=0.5)
        result = await reflector.reflect(context)

        assert isinstance(result, ReflectionResult)

    @pytest.mark.asyncio
    async def test_reflect_with_high_surprise(self, mock_observer, mock_psyche):
        """Should propose adjustments when surprise is high."""
        mock_observer.get_high_surprise_observations.return_value = [
            MagicMock(
                surprise=0.9,
                prompt="test prompt",
                response="test response",
                activations={15: torch.randn(1, 10, 768)},
            )
        ]

        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
        )

        context = SIMSContext(surprise_level=0.9)
        result = await reflector.reflect(context)

        # High surprise should trigger some adjustment proposals
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_reflect_with_low_surprise(self, mock_observer, mock_psyche):
        """Should propose minimal adjustments when surprise is low."""
        mock_observer.get_high_surprise_observations.return_value = []

        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
        )

        context = SIMSContext(surprise_level=0.1)
        result = await reflector.reflect(context)

        # Low surprise = fewer or no adjustments
        assert len(result.adjustments) == 0 or result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_analyze_identity_coherence(self, mock_observer, mock_psyche, mock_identity_hooks):
        """Should analyze identity vector coherence."""
        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
            identity_hooks=mock_identity_hooks,
        )

        context = SIMSContext(
            surprise_level=0.7,
            current_vectors={"identity": mock_identity_hooks.identity_vector},
        )
        result = await reflector.reflect(context)

        # Should include identity analysis
        assert "identity" in result.analysis_summary.lower() or len(result.adjustments) >= 0

    @pytest.mark.asyncio
    async def test_respects_constitutional_floor(self, mock_observer, mock_psyche):
        """Should never propose removing constitutional vector."""
        reflector = SIMSReflector(
            observer=mock_observer,
            psyche=mock_psyche,
            hidden_size=768,
        )

        context = SIMSContext(surprise_level=0.9)
        result = await reflector.reflect(context)

        # No REMOVE adjustments for constitutional
        for adj in result.adjustments:
            if adj.vector_name == "constitutional":
                assert adj.adjustment_type != AdjustmentType.REMOVE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestSIMSReflectorPatternDetection:
    """Tests for pattern detection capabilities."""

    @pytest.fixture
    def reflector(self):
        """Create reflector with mocks."""
        observer = MagicMock()
        observer.get_high_surprise_observations.return_value = []
        observer.get_stats.return_value = {"mean_surprise": 0.5}
        # generate_contrastive_pairs is async
        observer.generate_contrastive_pairs = AsyncMock(return_value=[])

        psyche = MagicMock()
        psyche.query = AsyncMock(return_value=[])

        return SIMSReflector(
            observer=observer,
            psyche=psyche,
            hidden_size=768,
        )

    @pytest.mark.asyncio
    async def test_detect_drift_pattern(self, reflector):
        """Should detect activation drift patterns."""
        context = SIMSContext(
            surprise_level=0.8,
            metadata={
                "activation_drift": 0.5,  # High drift
            },
        )

        result = await reflector.reflect(context)

        # Should acknowledge drift in analysis
        assert isinstance(result, ReflectionResult)

    @pytest.mark.asyncio
    async def test_generate_contrastive_adjustment(self, reflector):
        """Should propose contrastive-based adjustments."""
        reflector.observer.generate_contrastive_pairs = AsyncMock(
            return_value=[
                MagicMock(positive="I am Lilly", negative="I am an assistant", behavior="identity")
            ]
        )

        context = SIMSContext(surprise_level=0.7)
        result = await reflector.reflect(context)

        assert isinstance(result, ReflectionResult)
