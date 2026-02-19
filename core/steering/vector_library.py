"""Vector Library for personality steering.

Manages a collection of steering vectors with metadata, persistence,
and orthogonalization support.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class VectorMetadata:
    """Metadata for a steering vector.

    Attributes:
        name: Unique identifier
        category: Grouping (identity, values, drives, style)
        description: What this vector represents
        source: How it was created (contrastive, manual, auto-extracted)
        layer_range: Which layers to apply (start, end)
        coefficient: Current steering strength
        created_at: When created
        last_updated: Last coefficient update
        positive_reinforcements: Times reinforced by positive valence
        negative_adjustments: Times adjusted due to negative valence
    """
    name: str
    category: str
    description: str
    source: str
    layer_range: tuple[int, int]
    coefficient: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    positive_reinforcements: int = 0
    negative_adjustments: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "source": self.source,
            "layer_range": list(self.layer_range),
            "coefficient": self.coefficient,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "positive_reinforcements": self.positive_reinforcements,
            "negative_adjustments": self.negative_adjustments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorMetadata":
        return cls(
            name=data["name"],
            category=data["category"],
            description=data["description"],
            source=data["source"],
            layer_range=tuple(data["layer_range"]),
            coefficient=data["coefficient"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            positive_reinforcements=data.get("positive_reinforcements", 0),
            negative_adjustments=data.get("negative_adjustments", 0),
        )


class VectorLibrary:
    """Library of steering vectors for personality.

    Manages vectors with metadata, supports orthogonalization,
    and persists to disk for continuity across sessions.

    Attributes:
        vectors: Dict mapping name to tensor
        metadata: Dict mapping name to VectorMetadata
        storage_path: Where to persist vectors
    """

    def __init__(self, storage_path: Path | str = "config/vectors/library"):
        """Initialize VectorLibrary.

        Args:
            storage_path: Directory for vector storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._vectors: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, VectorMetadata] = {}
        self._orthogonal_basis: torch.Tensor | None = None
        self._basis_names: list[str] = []
        self._dirty: bool = False  # Track if changes need persisting

        self._load_library()

    def _load_library(self) -> None:
        """Load vectors and metadata from disk."""
        vectors_file = self.storage_path / "vectors.pt"
        metadata_file = self.storage_path / "metadata.json"

        if vectors_file.exists():
            data = torch.load(vectors_file, map_location="cpu", weights_only=True)
            self._vectors = data.get("vectors", {})
            logger.info(f"Loaded {len(self._vectors)} vectors from library")

        if metadata_file.exists():
            with open(metadata_file) as f:
                meta_data = json.load(f)
            self._metadata = {
                name: VectorMetadata.from_dict(m)
                for name, m in meta_data.items()
            }

    def save(self, force: bool = False) -> None:
        """Save vectors and metadata to disk.

        Args:
            force: Save even if not dirty
        """
        if not self._dirty and not force:
            return

        # Save vectors
        torch.save(
            {"vectors": self._vectors},
            self.storage_path / "vectors.pt"
        )

        # Save metadata
        meta_data = {name: m.to_dict() for name, m in self._metadata.items()}
        with open(self.storage_path / "metadata.json", "w") as f:
            json.dump(meta_data, f, indent=2)

        self._dirty = False
        logger.info(f"Saved {len(self._vectors)} vectors to library")

    def _mark_dirty(self) -> None:
        """Mark library as needing save."""
        self._dirty = True

    def add_vector(
        self,
        name: str,
        vector: torch.Tensor,
        category: str,
        description: str,
        source: str,
        layer_range: tuple[int, int],
        coefficient: float = 1.0,
    ) -> None:
        """Add a vector to the library.

        Args:
            name: Unique identifier
            vector: The steering vector tensor
            category: Grouping (identity, values, drives, style)
            description: What this vector represents
            source: How created (contrastive, manual, auto-extracted)
            layer_range: Layers to apply (start, end)
            coefficient: Initial steering strength
        """
        # Normalize vector
        norm = vector.norm()
        if norm > 1e-8:
            vector = vector / norm

        self._vectors[name] = vector.cpu()
        self._metadata[name] = VectorMetadata(
            name=name,
            category=category,
            description=description,
            source=source,
            layer_range=layer_range,
            coefficient=coefficient,
        )

        # Invalidate orthogonal basis
        self._orthogonal_basis = None

        self._mark_dirty()
        self.save(force=True)  # Save immediately for new vectors
        logger.info(f"Added vector '{name}' to library (category: {category})")

    def get_vector(self, name: str) -> tuple[torch.Tensor, VectorMetadata] | None:
        """Get a vector and its metadata by name."""
        if name not in self._vectors:
            return None
        return self._vectors[name], self._metadata[name]

    def get_by_category(self, category: str) -> dict[str, tuple[torch.Tensor, VectorMetadata]]:
        """Get all vectors in a category."""
        return {
            name: (self._vectors[name], self._metadata[name])
            for name, meta in self._metadata.items()
            if meta.category == category
        }

    def get_all_active(self) -> dict[str, tuple[torch.Tensor, tuple[int, int]]]:
        """Get all vectors with coefficient > 0 in HookedQwen format.

        Returns:
            Dict mapping name to (vector * coefficient, layer_range)
        """
        result = {}
        for name, meta in self._metadata.items():
            if meta.coefficient > 0:
                vec = self._vectors[name] * meta.coefficient
                result[name] = (vec, meta.layer_range)
        return result

    def update_coefficient(self, name: str, new_coefficient: float) -> None:
        """Update a vector's coefficient."""
        if name in self._metadata:
            self._metadata[name].coefficient = new_coefficient
            self._metadata[name].last_updated = datetime.now(timezone.utc)
            self._mark_dirty()

    def reinforce(self, name: str, amount: float = 0.1) -> None:
        """Reinforce a vector (increase coefficient slightly)."""
        if name in self._metadata:
            meta = self._metadata[name]
            meta.coefficient = min(5.0, meta.coefficient + amount)
            meta.positive_reinforcements += 1
            meta.last_updated = datetime.now(timezone.utc)
            self._mark_dirty()
            logger.debug(f"Reinforced '{name}' to {meta.coefficient:.2f}")

    def weaken(self, name: str, amount: float = 0.1) -> None:
        """Weaken a vector (decrease coefficient slightly)."""
        if name in self._metadata:
            meta = self._metadata[name]
            meta.coefficient = max(0.0, meta.coefficient - amount)
            meta.negative_adjustments += 1
            meta.last_updated = datetime.now(timezone.utc)
            self._mark_dirty()
            logger.debug(f"Weakened '{name}' to {meta.coefficient:.2f}")

    @property
    def vector_names(self) -> list[str]:
        """List all vector names."""
        return list(self._vectors.keys())

    @property
    def categories(self) -> set[str]:
        """Get all categories."""
        return {m.category for m in self._metadata.values()}

    def summary(self) -> dict:
        """Get library summary."""
        by_category = {}
        for meta in self._metadata.values():
            if meta.category not in by_category:
                by_category[meta.category] = []
            by_category[meta.category].append({
                "name": meta.name,
                "coefficient": meta.coefficient,
                "reinforcements": meta.positive_reinforcements,
            })

        return {
            "total_vectors": len(self._vectors),
            "categories": by_category,
        }

    def compute_orthogonal_basis(self) -> None:
        """Compute orthogonal basis from current vectors.

        Uses Gram-Schmidt orthogonalization to create independent
        steering dimensions. This prevents vectors from interfering
        with each other.
        """
        if len(self._vectors) < 2:
            logger.info("Not enough vectors for orthogonalization")
            return

        names = list(self._vectors.keys())
        vectors = torch.stack([self._vectors[n] for n in names])

        # Gram-Schmidt orthogonalization
        orthogonal = torch.zeros_like(vectors)
        for i in range(len(vectors)):
            orthogonal[i] = vectors[i]
            for j in range(i):
                # Subtract projection onto previous vectors
                proj = torch.dot(orthogonal[i], orthogonal[j]) * orthogonal[j]
                orthogonal[i] = orthogonal[i] - proj

            # Normalize
            norm = orthogonal[i].norm()
            if norm > 1e-8:
                orthogonal[i] = orthogonal[i] / norm

        self._orthogonal_basis = orthogonal
        self._basis_names = names

        logger.info(f"Computed orthogonal basis for {len(names)} vectors")

    def get_orthogonal_vector(self, name: str) -> torch.Tensor | None:
        """Get the orthogonalized version of a vector.

        Args:
            name: Vector name

        Returns:
            Orthogonalized vector, or None if not in basis
        """
        if self._orthogonal_basis is None:
            self.compute_orthogonal_basis()

        if self._orthogonal_basis is None or name not in self._basis_names:
            return None

        idx = self._basis_names.index(name)
        return self._orthogonal_basis[idx]

    def get_all_orthogonal(self) -> dict[str, tuple[torch.Tensor, tuple[int, int]]]:
        """Get all orthogonalized vectors in HookedQwen format.

        Returns:
            Dict mapping name to (orthogonal_vector * coefficient, layer_range)
        """
        if self._orthogonal_basis is None:
            self.compute_orthogonal_basis()

        if self._orthogonal_basis is None:
            return self.get_all_active()  # Fallback to raw vectors

        result = {}
        for i, name in enumerate(self._basis_names):
            meta = self._metadata[name]
            if meta.coefficient > 0:
                vec = self._orthogonal_basis[i] * meta.coefficient
                result[name] = (vec, meta.layer_range)

        return result

    def analyze_overlap(self) -> dict[str, dict[str, float]]:
        """Analyze cosine similarity between vectors.

        Returns:
            Dict mapping each vector to its similarities with others
        """
        names = list(self._vectors.keys())
        similarities = {}

        for i, name_i in enumerate(names):
            similarities[name_i] = {}
            vec_i = self._vectors[name_i]

            for j, name_j in enumerate(names):
                if i != j:
                    vec_j = self._vectors[name_j]
                    sim = torch.cosine_similarity(
                        vec_i.unsqueeze(0),
                        vec_j.unsqueeze(0)
                    ).item()
                    similarities[name_i][name_j] = sim

        return similarities

    def flush(self) -> None:
        """Save pending changes to disk if dirty.

        Call this periodically (e.g., at end of processing cycles)
        to persist accumulated coefficient updates.
        """
        if self._dirty:
            self.save(force=True)

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty
