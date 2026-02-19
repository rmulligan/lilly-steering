"""Sparse co-activation trace matrix with Hebbian learning.

The TraceMatrix tracks feature co-activation patterns over time using
Hebbian learning: connections strengthen when features activate together,
modulated by a value signal. This captures temporal dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from core.substrate.schemas import FeatureActivation


class TraceMatrix:
    """Sparse symmetric matrix tracking feature co-activation strength.

    Implements Hebbian learning with value modulation:
        Δw_ij = η * a_i * a_j * value

    The matrix is symmetric (w_ij = w_ji) and sparse for memory efficiency.

    Attributes:
        n_features: Number of SAE features
        learning_rate: η in the Hebbian update rule
        decay_rate: Multiplicative decay applied during consolidation
    """

    def __init__(
        self,
        n_features: int = 163840,
        learning_rate: float = 0.01,
        decay_rate: float = 0.995,
    ):
        """Initialize the trace matrix.

        Args:
            n_features: Number of SAE features
            learning_rate: Hebbian learning rate (η)
            decay_rate: Per-step decay rate for forgetting
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        # Use dok_matrix for efficient incremental updates
        self._matrix = sparse.dok_matrix(
            (n_features, n_features), dtype=np.float32
        )

    @property
    def matrix(self) -> sparse.spmatrix:
        """Return the trace matrix in CSR format for efficient operations."""
        return self._matrix.tocsr()

    @property
    def sparsity(self) -> float:
        """Return sparsity (fraction of zero entries)."""
        nnz = self._matrix.nnz
        total = self.n_features * self.n_features
        return 1.0 - (nnz / total)

    def hebbian_update(
        self, features: list[FeatureActivation], value: float
    ) -> None:
        """Apply Hebbian update for co-activated features.

        Updates connection strength between all pairs of active features:
            Δw_ij = η * a_i * a_j * value

        Args:
            features: List of active features with activation strengths
            value: Value signal modulating the update magnitude
        """
        if len(features) < 2 or value <= 0:
            return

        # Extract indices and activations
        indices = [f.feature_idx for f in features]
        activations = {f.feature_idx: f.activation for f in features}

        # Update all pairs (symmetric)
        for i, idx_i in enumerate(indices):
            for idx_j in indices[i + 1:]:
                a_i = activations[idx_i]
                a_j = activations[idx_j]
                delta = self.learning_rate * a_i * a_j * value

                # Symmetric update
                self._matrix[idx_i, idx_j] += delta
                self._matrix[idx_j, idx_i] += delta

    def decay(self) -> None:
        """Apply decay to all connections."""
        self._matrix *= self.decay_rate

    def get_strength(self, i: int, j: int) -> float:
        """Get connection strength between two features.

        Args:
            i: First feature index
            j: Second feature index

        Returns:
            Connection strength (0 if no connection)
        """
        return float(self._matrix.get((i, j), 0.0))

    def get_associated_features(
        self, feature_idx: int, top_k: int = 10, min_strength: float = 0.0
    ) -> list[tuple[int, float]]:
        """Get features most strongly associated with the given feature.

        Args:
            feature_idx: Query feature index
            top_k: Maximum number of associations to return
            min_strength: Minimum strength threshold

        Returns:
            List of (feature_idx, strength) tuples, sorted by strength
        """
        # Get row for this feature
        row = self._matrix.tocsr().getrow(feature_idx).toarray().flatten()

        # Find non-zero entries above threshold
        mask = row > min_strength
        indices = np.where(mask)[0]
        strengths = row[mask]

        # Sort by strength descending
        sorted_indices = np.argsort(-strengths)[:top_k]

        return [
            (int(indices[i]), float(strengths[i]))
            for i in sorted_indices
        ]

    def prune(self, threshold: float = 0.001) -> int:
        """Remove connections below threshold.

        Args:
            threshold: Minimum strength to retain

        Returns:
            Number of connections pruned
        """
        # Convert to COO for efficient filtering
        coo = self._matrix.tocoo()
        mask = np.abs(coo.data) >= threshold
        pruned_count = len(coo.data) - np.sum(mask)

        # Rebuild matrix with only strong connections
        self._matrix = sparse.dok_matrix(
            sparse.csr_matrix(
                (coo.data[mask], (coo.row[mask], coo.col[mask])),
                shape=(self.n_features, self.n_features),
            )
        )

        return int(pruned_count)

    def get_primed_features(
        self, active_features: list[int], top_k: int = 20
    ) -> list[tuple[int, float]]:
        """Get features primed (predicted) by active features.

        Aggregates association strengths across all active features
        to predict which features might activate next.

        Args:
            active_features: Currently active feature indices
            top_k: Number of primed features to return

        Returns:
            List of (feature_idx, priming_strength) tuples
        """
        if not active_features:
            return []

        # Sum association strengths across active features using sparse matrix slicing
        csr = self.matrix
        priming_vector = csr[active_features, :].sum(axis=0)
        priming = np.asarray(priming_vector).flatten()

        # Exclude already-active features
        for idx in active_features:
            priming[idx] = 0.0

        # Get top-k
        top_indices = np.argsort(-priming)[:top_k]
        return [
            (int(idx), float(priming[idx]))
            for idx in top_indices
            if priming[idx] > 0
        ]

    def save_state(self) -> dict:
        """Serialize matrix state for persistence."""
        csr = self.matrix
        return {
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "data": csr.data.tolist(),
            "indices": csr.indices.tolist(),
            "indptr": csr.indptr.tolist(),
        }

    @classmethod
    def load_state(cls, state: dict) -> "TraceMatrix":
        """Deserialize matrix from saved state."""
        trace = cls(
            n_features=state["n_features"],
            learning_rate=state["learning_rate"],
            decay_rate=state["decay_rate"],
        )
        csr = sparse.csr_matrix(
            (state["data"], state["indices"], state["indptr"]),
            shape=(state["n_features"], state["n_features"]),
        )
        trace._matrix = csr.todok()
        return trace
