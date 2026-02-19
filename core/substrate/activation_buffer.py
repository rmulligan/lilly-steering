"""Rolling window buffer for recent feature activations.

The ActivationBuffer maintains a fixed-capacity circular buffer of sparse
feature activation vectors. It supports efficient co-activation detection
and feeds into the trace matrix for Hebbian learning.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np
from scipy import sparse

from core.substrate.schemas import FeatureActivation


class ActivationBuffer:
    """Rolling window of recent SAE feature activations.

    Maintains a circular buffer of sparse activation vectors for:
    - Co-activation detection within temporal windows
    - Feeding the trace matrix with recent patterns
    - Computing running statistics on activation patterns

    Attributes:
        capacity: Maximum number of activation snapshots to retain
        n_features: Total number of SAE features (163840 for Qwen3-8B)
    """

    def __init__(self, capacity: int = 10, n_features: int = 163840):
        """Initialize the activation buffer.

        Args:
            capacity: Maximum snapshots to retain (default 10 cycles)
            n_features: Number of SAE features
        """
        self.capacity = capacity
        self.n_features = n_features
        self._buffer: deque[sparse.csr_matrix] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """Return number of snapshots currently in buffer."""
        return len(self._buffer)

    def add(self, features: list[FeatureActivation]) -> None:
        """Add a new activation snapshot to the buffer.

        Args:
            features: List of active features with their strengths
        """
        if not features:
            # Add empty row
            row = sparse.csr_matrix((1, self.n_features), dtype=np.float32)
        else:
            # Build sparse row from features
            indices = [f.feature_idx for f in features]
            values = [f.activation for f in features]
            row = sparse.csr_matrix(
                (values, ([0] * len(indices), indices)),
                shape=(1, self.n_features),
                dtype=np.float32,
            )
        self._buffer.append(row)

    def get_recent(self, n: int = 1) -> sparse.csr_matrix:
        """Get the n most recent activation snapshots.

        Args:
            n: Number of recent snapshots to return

        Returns:
            Sparse matrix of shape [n, n_features]
        """
        if not self._buffer:
            return sparse.csr_matrix((0, self.n_features), dtype=np.float32)

        if n == 0:
            return sparse.csr_matrix((0, self.n_features), dtype=np.float32)

        n = min(n, len(self._buffer))
        recent = list(self._buffer)[-n:]
        return sparse.vstack(recent)

    def get_all_sparse(self) -> sparse.csr_matrix:
        """Get all buffered activations as sparse matrix.

        Returns:
            Sparse matrix of shape [len(buffer), n_features]
        """
        if not self._buffer:
            return sparse.csr_matrix((0, self.n_features), dtype=np.float32)
        return sparse.vstack(list(self._buffer))

    def get_coactivation_pairs(
        self, window: int = 1, min_activation: float = 0.1
    ) -> set[tuple[int, int]]:
        """Find feature pairs that co-activated within a window.

        Args:
            window: Number of recent snapshots to consider
            min_activation: Minimum activation strength to count

        Returns:
            Set of (feature_i, feature_j) pairs where i < j
        """
        if len(self._buffer) == 0:
            return set()

        recent = self.get_recent(n=window)
        pairs: set[tuple[int, int]] = set()

        # Find non-zero entries per row, then find pairs
        for row_idx in range(recent.shape[0]):
            row = recent.getrow(row_idx)
            active_indices = row.indices[row.data >= min_activation]

            # Generate all pairs from this snapshot
            for i, idx_i in enumerate(active_indices):
                for idx_j in active_indices[i + 1:]:
                    pairs.add((int(idx_i), int(idx_j)))

        return pairs

    def get_active_features(self, n: int = 1) -> list[int]:
        """Get unique feature indices active in last n snapshots.

        Args:
            n: Number of recent snapshots to consider

        Returns:
            Sorted list of unique active feature indices
        """
        if not self._buffer:
            return []

        recent = self.get_recent(n=n)
        return sorted(set(recent.indices.tolist()))

    def clear(self) -> None:
        """Clear all buffered activations."""
        self._buffer.clear()

    def __iter__(self) -> Iterator[sparse.csr_matrix]:
        """Iterate over buffered activation snapshots."""
        return iter(self._buffer)
