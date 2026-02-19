"""Learned feature embeddings with attractor dynamics.

The EmbeddingSpace maps SAE features to a dense embedding space where
features can learn to cluster around semantic attractors representing
entities, zettels, moods, and emergent concepts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.substrate.schemas import Attractor, FeatureActivation

if TYPE_CHECKING:
    pass


class EmbeddingSpace:
    """Dense embedding space for SAE features with attractor dynamics.

    Features are embedded in a d-dimensional space where:
    - Co-activated features move closer together (Hebbian learning)
    - Features are pulled toward nearby attractors (gravitational dynamics)
    - Basin membership determines which attractor "owns" each feature

    Attributes:
        n_features: Number of SAE features (163840 for Qwen3-8B)
        embed_dim: Dimensionality of embedding space
        learning_rate: Rate for embedding updates
        embeddings: Feature embedding matrix [n_features, embed_dim]
        attractors: Registry of attractors by uid
    """

    def __init__(
        self,
        n_features: int = 163840,
        embed_dim: int = 64,
        learning_rate: float = 0.01,
        seed: int | None = None,
    ):
        """Initialize the embedding space.

        Args:
            n_features: Number of SAE features
            embed_dim: Embedding dimensionality (default 64)
            learning_rate: Learning rate for updates
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

        # Initialize random embeddings normalized to unit sphere
        rng = np.random.default_rng(seed)
        self.embeddings = rng.normal(0, 1, (n_features, embed_dim)).astype(
            np.float32
        )
        # Normalize to unit sphere
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Prevent division by zero
        self.embeddings /= norms

        # Attractor registry
        self.attractors: dict[str, Attractor] = {}

    def get_embedding(self, feature_idx: int) -> np.ndarray:
        """Get embedding vector for a specific feature.

        Args:
            feature_idx: Feature index

        Returns:
            Embedding vector of shape [embed_dim]

        Raises:
            IndexError: If feature_idx is out of bounds [0, n_features)
        """
        if not 0 <= feature_idx < self.n_features:
            raise IndexError(
                f"feature_idx {feature_idx} out of bounds [0, {self.n_features})"
            )
        return self.embeddings[feature_idx].copy()

    def set_embedding(self, feature_idx: int, embedding: np.ndarray) -> None:
        """Set embedding vector for a specific feature.

        Args:
            feature_idx: Feature index
            embedding: New embedding vector of shape [embed_dim]

        Raises:
            IndexError: If feature_idx is out of bounds [0, n_features)
        """
        if not 0 <= feature_idx < self.n_features:
            raise IndexError(
                f"feature_idx {feature_idx} out of bounds [0, {self.n_features})"
            )
        self.embeddings[feature_idx] = embedding.astype(np.float32)

    def update_from_coactivation(
        self, i: int, j: int, strength: float = 1.0
    ) -> None:
        """Update embeddings for co-activated features.

        Moves features i and j closer together proportional to strength.

        Args:
            i: First feature index
            j: Second feature index
            strength: Coactivation strength (product of activations)
        """
        if strength <= 0:
            return

        emb_i = self.embeddings[i]
        emb_j = self.embeddings[j]

        # Compute direction from i to j
        diff = emb_j - emb_i
        delta = self.learning_rate * strength * diff

        # Move both embeddings toward each other
        self.embeddings[i] = emb_i + delta
        self.embeddings[j] = emb_j - delta

    def batch_update_coactivation(
        self, pairs: list[tuple[int, int]], strength: float = 1.0
    ) -> None:
        """Update embeddings for multiple co-activated feature pairs.

        Args:
            pairs: List of (feature_i, feature_j) pairs
            strength: Common coactivation strength
        """
        for i, j in pairs:
            self.update_from_coactivation(i, j, strength)

    def add_attractor(self, attractor: Attractor) -> None:
        """Add an attractor to the embedding space.

        Args:
            attractor: Attractor to add
        """
        self.attractors[attractor.uid] = attractor

    def remove_attractor(self, uid: str) -> None:
        """Remove an attractor from the embedding space.

        Args:
            uid: Attractor uid to remove
        """
        self.attractors.pop(uid, None)

    def get_attractor(self, uid: str) -> Attractor | None:
        """Get an attractor by uid.

        Args:
            uid: Attractor uid

        Returns:
            Attractor if found, None otherwise
        """
        return self.attractors.get(uid)

    def apply_attractor_pull(self, active_features: list[FeatureActivation]) -> None:
        """Apply gravitational pull from nearby attractors to active features.

        Features within an attractor's pull_radius are pulled toward it
        proportional to pull_strength, activation, and inverse distance.

        Args:
            active_features: Currently active features with activations
        """
        if not self.attractors:
            return

        for feat in active_features:
            idx = feat.feature_idx
            activation = feat.activation
            emb = self.embeddings[idx]

            for attractor in self.attractors.values():
                attr_pos = np.array(attractor.position, dtype=np.float32)

                # Compute distance to attractor
                dist = np.linalg.norm(emb - attr_pos)

                # Check if within pull radius
                if dist > attractor.pull_radius or dist < 1e-6:
                    continue

                # Compute pull direction and magnitude
                direction = attr_pos - emb
                direction_normalized = direction / dist

                # Pull strength: stronger when closer, modulated by activation
                pull_magnitude = (
                    self.learning_rate
                    * attractor.pull_strength
                    * activation
                    * (1.0 - dist / attractor.pull_radius)  # Falloff with distance
                )

                # Apply pull
                self.embeddings[idx] = emb + pull_magnitude * direction_normalized

    def find_nearest_attractors(
        self, feature_idx: int, top_k: int = 5
    ) -> list[tuple[Attractor, float]]:
        """Find the nearest attractors to a feature.

        Args:
            feature_idx: Feature index
            top_k: Maximum number of attractors to return

        Returns:
            List of (attractor, distance) tuples sorted by distance
        """
        if not self.attractors:
            return []

        emb = self.embeddings[feature_idx]
        distances: list[tuple[Attractor, float]] = []

        for attractor in self.attractors.values():
            attr_pos = np.array(attractor.position, dtype=np.float32)
            dist = float(np.linalg.norm(emb - attr_pos))
            distances.append((attractor, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        return distances[:top_k]

    def compute_basin_membership(
        self, feature_indices: list[int]
    ) -> dict[int, str]:
        """Assign features to their nearest attractor basin.

        Args:
            feature_indices: Feature indices to classify

        Returns:
            Mapping of feature_idx -> attractor_uid for nearest attractor
        """
        if not self.attractors:
            return {}

        membership: dict[int, str] = {}

        # Precompute attractor positions
        attr_list = list(self.attractors.values())
        attr_positions = np.array(
            [a.position for a in attr_list], dtype=np.float32
        )

        for idx in feature_indices:
            emb = self.embeddings[idx]

            # Compute distances to all attractors
            dists = np.linalg.norm(attr_positions - emb, axis=1)

            # Find nearest
            nearest_idx = int(np.argmin(dists))
            membership[idx] = attr_list[nearest_idx].uid

        return membership

    def get_attractor_activations(
        self, active_features: list[FeatureActivation]
    ) -> dict[str, float]:
        """Aggregate feature activations per attractor basin.

        Args:
            active_features: Currently active features

        Returns:
            Mapping of attractor_uid -> total activation from features in basin
        """
        if not self.attractors or not active_features:
            return {}

        # Get basin membership for active features
        indices = [f.feature_idx for f in active_features]
        activations = {f.feature_idx: f.activation for f in active_features}
        basins = self.compute_basin_membership(indices)

        # Aggregate activations per attractor
        attractor_acts: dict[str, float] = {uid: 0.0 for uid in self.attractors}
        for idx, attr_uid in basins.items():
            attractor_acts[attr_uid] += activations[idx]

        return attractor_acts

    def variance(self) -> float:
        """Compute variance of embeddings as health metric.

        Returns:
            Mean variance across embedding dimensions
        """
        return float(np.var(self.embeddings))

    def save_state(self) -> dict:
        """Serialize embedding space state for persistence.

        Returns:
            Serializable state dictionary
        """
        return {
            "n_features": self.n_features,
            "embed_dim": self.embed_dim,
            "learning_rate": self.learning_rate,
            "embeddings": self.embeddings.tolist(),
            "attractors": {
                uid: attractor.model_dump()
                for uid, attractor in self.attractors.items()
            },
        }

    @classmethod
    def load_state(cls, state: dict) -> "EmbeddingSpace":
        """Deserialize embedding space from saved state.

        Args:
            state: Previously saved state dictionary

        Returns:
            Reconstructed EmbeddingSpace instance
        """
        space = cls(
            n_features=state["n_features"],
            embed_dim=state["embed_dim"],
            learning_rate=state["learning_rate"],
        )
        space.embeddings = np.array(state["embeddings"], dtype=np.float32)

        # Restore attractors
        for uid, attr_data in state.get("attractors", {}).items():
            space.attractors[uid] = Attractor(**attr_data)

        return space
