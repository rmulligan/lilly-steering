"""Tests for CoefficientOptimizer."""

import pytest
import torch

from core.steering.vector_library import VectorLibrary
from core.steering.coefficient_optimizer import CoefficientOptimizer, ValenceFeedback


class TestValenceFeedback:
    """Tests for ValenceFeedback dataclass."""

    def test_creation(self):
        from datetime import datetime, timezone

        feedback = ValenceFeedback(
            valence=0.5,
            arousal=0.7,
            active_vectors=["v1", "v2"],
            timestamp=datetime.now(timezone.utc),
        )

        assert feedback.valence == 0.5
        assert feedback.arousal == 0.7
        assert len(feedback.active_vectors) == 2


class TestCoefficientOptimizer:
    """Tests for CoefficientOptimizer."""

    @pytest.fixture
    def library(self, tmp_path):
        lib = VectorLibrary(storage_path=tmp_path / "vectors")
        lib.add_vector("v1", torch.randn(3584), "test", "V1", "test", (18, 27), 1.0)
        lib.add_vector("v2", torch.randn(3584), "test", "V2", "test", (18, 27), 1.0)
        return lib

    @pytest.fixture
    def optimizer(self, library):
        return CoefficientOptimizer(library, learning_rate=0.1)

    def test_positive_feedback_reinforces(self, optimizer, library):
        optimizer.record_feedback(
            valence=0.8,
            arousal=0.5,
            active_vectors=["v1"],
        )

        _, meta = library.get_vector("v1")
        assert meta.coefficient > 1.0

    def test_negative_feedback_weakens(self, optimizer, library):
        optimizer.record_feedback(
            valence=-0.5,
            arousal=0.5,
            active_vectors=["v1"],
        )

        _, meta = library.get_vector("v1")
        assert meta.coefficient < 1.0

    def test_neutral_no_change(self, optimizer, library):
        optimizer.record_feedback(
            valence=0.1,  # Between thresholds
            arousal=0.5,
            active_vectors=["v1"],
        )

        _, meta = library.get_vector("v1")
        assert meta.coefficient == 1.0

    def test_arousal_scales_adjustment(self, library, tmp_path):
        # Create two optimizers with same settings
        lib1 = VectorLibrary(storage_path=tmp_path / "lib1")
        lib1.add_vector("v1", torch.randn(3584), "test", "V1", "test", (18, 27), 1.0)
        opt_low_arousal = CoefficientOptimizer(lib1, learning_rate=0.1)

        lib2 = VectorLibrary(storage_path=tmp_path / "lib2")
        lib2.add_vector("v1", torch.randn(3584), "test", "V1", "test", (18, 27), 1.0)
        opt_high_arousal = CoefficientOptimizer(lib2, learning_rate=0.1)

        # Same valence, different arousal
        opt_low_arousal.record_feedback(valence=0.8, arousal=0.1, active_vectors=["v1"])
        opt_high_arousal.record_feedback(valence=0.8, arousal=0.9, active_vectors=["v1"])

        _, meta1 = lib1.get_vector("v1")
        _, meta2 = lib2.get_vector("v1")

        # Higher arousal should result in larger adjustment
        assert meta2.coefficient > meta1.coefficient

    def test_multiple_vectors_affected(self, optimizer, library):
        optimizer.record_feedback(
            valence=0.8,
            arousal=0.5,
            active_vectors=["v1", "v2"],
        )

        _, meta1 = library.get_vector("v1")
        _, meta2 = library.get_vector("v2")

        assert meta1.coefficient > 1.0
        assert meta2.coefficient > 1.0

    def test_actions_returned(self, optimizer):
        actions = optimizer.record_feedback(
            valence=0.8,
            arousal=0.5,
            active_vectors=["v1", "v2"],
        )

        assert actions["v1"] == "reinforced"
        assert actions["v2"] == "reinforced"

    def test_neutral_actions(self, optimizer):
        actions = optimizer.record_feedback(
            valence=0.1,
            arousal=0.5,
            active_vectors=["v1"],
        )

        assert actions["v1"] == "unchanged"

    def test_get_problematic_vectors(self, optimizer, library):
        # Record many negative experiences with v1
        for _ in range(15):
            optimizer.record_feedback(valence=-0.8, arousal=0.5, active_vectors=["v1"])
            optimizer.record_feedback(valence=0.8, arousal=0.5, active_vectors=["v2"])

        problematic = optimizer.get_problematic_vectors(min_negative_ratio=0.6)

        assert "v1" in problematic
        assert "v2" not in problematic

    def test_not_enough_data_for_problematic(self, optimizer):
        # Only a few feedbacks
        for _ in range(3):
            optimizer.record_feedback(valence=-0.8, arousal=0.5, active_vectors=["v1"])

        problematic = optimizer.get_problematic_vectors()

        # Need at least 10 samples
        assert "v1" not in problematic

    def test_get_statistics_empty(self, optimizer):
        stats = optimizer.get_statistics()

        assert stats["feedback_count"] == 0

    def test_get_statistics_with_data(self, optimizer):
        optimizer.record_feedback(valence=0.8, arousal=0.5, active_vectors=["v1"])
        optimizer.record_feedback(valence=-0.5, arousal=0.5, active_vectors=["v1"])
        optimizer.record_feedback(valence=0.1, arousal=0.5, active_vectors=["v1"])

        stats = optimizer.get_statistics()

        assert stats["feedback_count"] == 3
        assert stats["positive_count"] == 1
        assert stats["negative_count"] == 1
        assert stats["neutral_count"] == 1

    def test_history_limit(self, optimizer):
        # Record more than max_history
        for i in range(1200):
            optimizer.record_feedback(valence=0.5, arousal=0.5, active_vectors=["v1"])

        assert len(optimizer._feedback_history) == optimizer._max_history

    def test_custom_thresholds(self, library):
        optimizer = CoefficientOptimizer(
            library,
            learning_rate=0.1,
            positive_threshold=0.5,  # Higher threshold
            negative_threshold=-0.5,  # Lower threshold
        )

        # 0.4 is below positive_threshold now
        optimizer.record_feedback(valence=0.4, arousal=0.5, active_vectors=["v1"])
        _, meta = library.get_vector("v1")
        assert meta.coefficient == 1.0  # Unchanged

        # 0.6 is above positive_threshold
        optimizer.record_feedback(valence=0.6, arousal=0.5, active_vectors=["v1"])
        _, meta = library.get_vector("v1")
        assert meta.coefficient > 1.0  # Reinforced
