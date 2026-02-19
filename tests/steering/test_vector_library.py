"""Tests for VectorLibrary."""

import pytest
import torch
from pathlib import Path

from core.steering.vector_library import VectorLibrary, VectorMetadata


class TestVectorMetadata:
    """Tests for VectorMetadata dataclass."""

    def test_to_dict(self):
        meta = VectorMetadata(
            name="test",
            category="identity",
            description="Test vector",
            source="manual",
            layer_range=(18, 27),
            coefficient=1.5,
        )

        data = meta.to_dict()

        assert data["name"] == "test"
        assert data["category"] == "identity"
        assert data["coefficient"] == 1.5
        assert data["layer_range"] == [18, 27]

    def test_from_dict_roundtrip(self):
        meta = VectorMetadata(
            name="test",
            category="identity",
            description="Test vector",
            source="manual",
            layer_range=(18, 27),
            coefficient=2.0,
        )

        data = meta.to_dict()
        restored = VectorMetadata.from_dict(data)

        assert restored.name == meta.name
        assert restored.category == meta.category
        assert restored.coefficient == meta.coefficient
        assert restored.layer_range == meta.layer_range


class TestVectorLibrary:
    """Tests for VectorLibrary."""

    @pytest.fixture
    def library(self, tmp_path):
        return VectorLibrary(storage_path=tmp_path / "vectors")

    def test_add_vector(self, library):
        vec = torch.randn(3584)
        library.add_vector(
            name="test_vec",
            vector=vec,
            category="test",
            description="Test vector",
            source="test",
            layer_range=(18, 27),
        )

        assert "test_vec" in library.vector_names

    def test_get_vector(self, library):
        vec = torch.randn(3584)
        library.add_vector(
            name="test_vec",
            vector=vec,
            category="test",
            description="Test",
            source="test",
            layer_range=(18, 27),
        )

        result = library.get_vector("test_vec")
        assert result is not None
        tensor, meta = result
        assert meta.name == "test_vec"

    def test_get_nonexistent_vector(self, library):
        result = library.get_vector("nonexistent")
        assert result is None

    def test_vector_normalized(self, library):
        vec = torch.randn(3584) * 100  # Large vector
        library.add_vector(
            name="test_vec",
            vector=vec,
            category="test",
            description="Test",
            source="test",
            layer_range=(18, 27),
        )

        tensor, _ = library.get_vector("test_vec")
        assert abs(tensor.norm().item() - 1.0) < 0.01

    def test_persistence(self, tmp_path):
        path = tmp_path / "vectors"

        # Create and save
        lib1 = VectorLibrary(storage_path=path)
        lib1.add_vector(
            name="persistent",
            vector=torch.randn(3584),
            category="test",
            description="Test",
            source="test",
            layer_range=(18, 27),
        )

        # Load in new instance
        lib2 = VectorLibrary(storage_path=path)
        assert "persistent" in lib2.vector_names

    def test_get_by_category(self, library):
        library.add_vector("v1", torch.randn(3584), "identity", "V1", "test", (18, 27))
        library.add_vector("v2", torch.randn(3584), "drives", "V2", "test", (18, 27))
        library.add_vector("v3", torch.randn(3584), "identity", "V3", "test", (18, 27))

        identity_vectors = library.get_by_category("identity")

        assert len(identity_vectors) == 2
        assert "v1" in identity_vectors
        assert "v3" in identity_vectors

    def test_get_all_active(self, library):
        library.add_vector("v1", torch.randn(3584), "test", "V1", "test", (18, 27), 1.0)
        library.add_vector("v2", torch.randn(3584), "test", "V2", "test", (18, 27), 0.0)
        library.add_vector("v3", torch.randn(3584), "test", "V3", "test", (20, 25), 2.0)

        active = library.get_all_active()

        assert len(active) == 2
        assert "v1" in active
        assert "v3" in active
        assert "v2" not in active  # coefficient is 0

    def test_update_coefficient(self, library):
        library.add_vector("test", torch.randn(3584), "test", "Test", "test", (18, 27), 1.0)

        library.update_coefficient("test", 2.5)

        _, meta = library.get_vector("test")
        assert meta.coefficient == 2.5

    def test_reinforce(self, library):
        library.add_vector(
            name="test_vec",
            vector=torch.randn(3584),
            category="test",
            description="Test",
            source="test",
            layer_range=(18, 27),
            coefficient=1.0,
        )

        library.reinforce("test_vec", 0.5)
        _, meta = library.get_vector("test_vec")
        assert meta.coefficient == 1.5
        assert meta.positive_reinforcements == 1

    def test_weaken(self, library):
        library.add_vector(
            name="test_vec",
            vector=torch.randn(3584),
            category="test",
            description="Test",
            source="test",
            layer_range=(18, 27),
            coefficient=1.0,
        )

        library.weaken("test_vec", 0.3)
        _, meta = library.get_vector("test_vec")
        assert meta.coefficient == 0.7
        assert meta.negative_adjustments == 1

    def test_coefficient_bounds(self, library):
        library.add_vector("test", torch.randn(3584), "test", "Test", "test", (18, 27), 1.0)

        # Can't go below 0
        library.weaken("test", 10.0)
        _, meta = library.get_vector("test")
        assert meta.coefficient == 0.0

        # Can't go above 5
        for _ in range(100):
            library.reinforce("test", 1.0)
        _, meta = library.get_vector("test")
        assert meta.coefficient == 5.0

    def test_orthogonalization(self, library):
        # Add two similar vectors
        v1 = torch.randn(3584)
        v2 = v1 + torch.randn(3584) * 0.1  # Slightly different

        library.add_vector("v1", v1, "test", "V1", "test", (18, 27))
        library.add_vector("v2", v2, "test", "V2", "test", (18, 27))

        library.compute_orthogonal_basis()

        o1 = library.get_orthogonal_vector("v1")
        o2 = library.get_orthogonal_vector("v2")

        # Should be orthogonal
        dot = torch.dot(o1, o2).abs().item()
        assert dot < 0.01

    def test_get_orthogonal_auto_computes(self, library):
        library.add_vector("v1", torch.randn(3584), "test", "V1", "test", (18, 27))
        library.add_vector("v2", torch.randn(3584), "test", "V2", "test", (18, 27))

        # Should auto-compute basis
        o1 = library.get_orthogonal_vector("v1")
        assert o1 is not None

    def test_analyze_overlap(self, library):
        v1 = torch.randn(3584)
        library.add_vector("v1", v1, "test", "V1", "test", (18, 27))
        library.add_vector("v2", v1 * 0.9 + torch.randn(3584) * 0.1, "test", "V2", "test", (18, 27))
        library.add_vector("v3", torch.randn(3584), "test", "V3", "test", (18, 27))

        overlap = library.analyze_overlap()

        # v1 and v2 should have high similarity
        assert overlap["v1"]["v2"] > 0.7
        # v1 and v3 should have lower similarity
        assert abs(overlap["v1"]["v3"]) < 0.5

    def test_summary(self, library):
        library.add_vector("v1", torch.randn(3584), "identity", "V1", "test", (18, 27))
        library.add_vector("v2", torch.randn(3584), "drives", "V2", "test", (18, 27))

        summary = library.summary()

        assert summary["total_vectors"] == 2
        assert "identity" in summary["categories"]
        assert "drives" in summary["categories"]

    def test_categories_property(self, library):
        library.add_vector("v1", torch.randn(3584), "identity", "V1", "test", (18, 27))
        library.add_vector("v2", torch.randn(3584), "drives", "V2", "test", (18, 27))
        library.add_vector("v3", torch.randn(3584), "identity", "V3", "test", (18, 27))

        categories = library.categories

        assert len(categories) == 2
        assert "identity" in categories
        assert "drives" in categories
