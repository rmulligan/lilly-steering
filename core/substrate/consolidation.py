"""Surprise-modulated consolidation for dream cycles.

Consolidation transfers patterns between substrate layers at different
timescales, modulated by surprise level (flashbulb memory effect).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.substrate.schemas import DreamCycleType, FeatureActivation

if TYPE_CHECKING:
    from core.substrate.activation_buffer import ActivationBuffer
    from core.substrate.trace_matrix import TraceMatrix
    from core.substrate.embedding_space import EmbeddingSpace


class ConsolidationManager:
    """Manages consolidation across dream cycle types.

    Consolidation transfers information between substrate layers:
    - Micro: Buffer -> Trace (per-cycle)
    - Nap: Trace pruning, strong patterns -> Embedding updates
    - Full: Embedding clusters -> Graph, weight decay
    - Deep: Aggressive pruning, topology restructuring

    All consolidation is modulated by surprise level.

    Attributes:
        buffer: ActivationBuffer to read patterns from
        trace: TraceMatrix for Hebbian updates
        embeddings: EmbeddingSpace for feature vectors
    """

    def __init__(
        self,
        buffer: "ActivationBuffer",
        trace: "TraceMatrix",
        embeddings: "EmbeddingSpace",
        surprise_coefficient: float = 1.0,
        nap_prune_threshold: float = 0.01,
        deep_prune_threshold: float = 0.1,
        embedding_transfer_threshold: float = 0.1,
    ):
        """Initialize the consolidation manager.

        Args:
            buffer: Activation buffer component
            trace: Trace matrix component
            embeddings: Embedding space component
            surprise_coefficient: beta in consolidation_rate = base * (1 + beta * surprise)
            nap_prune_threshold: Minimum trace strength to retain during nap
            deep_prune_threshold: Minimum trace strength for deep reflection
            embedding_transfer_threshold: Min trace strength to update embeddings
        """
        self.buffer = buffer
        self.trace = trace
        self.embeddings = embeddings

        self.surprise_coefficient = surprise_coefficient
        self.nap_prune_threshold = nap_prune_threshold
        self.deep_prune_threshold = deep_prune_threshold
        self.embedding_transfer_threshold = embedding_transfer_threshold

    def consolidate(
        self,
        cycle_type: DreamCycleType,
        surprise: float = 0.0,
        value: float = 0.0,
    ) -> dict:
        """Run consolidation appropriate to cycle type.

        Args:
            cycle_type: Type of dream cycle
            surprise: Current surprise level for modulation
            value: Current value signal

        Returns:
            Dict with consolidation statistics
        """
        # Calculate consolidation rate
        rate_multiplier = 1.0 + self.surprise_coefficient * surprise

        if cycle_type == DreamCycleType.MICRO:
            return self._consolidate_micro(value, rate_multiplier)
        elif cycle_type == DreamCycleType.NAP:
            return self._consolidate_nap(rate_multiplier)
        elif cycle_type == DreamCycleType.FULL:
            return self._consolidate_full(rate_multiplier)
        elif cycle_type == DreamCycleType.DEEP:
            return self._consolidate_deep(rate_multiplier)
        else:
            return {}

    def _consolidate_micro(
        self, value: float, rate_multiplier: float
    ) -> dict:
        """Micro-dream: Flush buffer to trace matrix.

        Transfers recent co-activation patterns to trace with Hebbian
        learning, modulated by value signal and surprise.
        """
        stats = {"patterns_transferred": 0, "pairs_updated": 0}

        # Get all buffered activations
        for row in self.buffer:
            # Extract active features from sparse row
            active_indices = row.indices
            active_values = row.data

            if len(active_indices) < 2:
                continue

            features = [
                FeatureActivation(feature_idx=int(idx), activation=float(val))
                for idx, val in zip(active_indices, active_values)
            ]

            # Apply Hebbian update with modulated value
            effective_value = value * rate_multiplier
            self.trace.hebbian_update(features, effective_value)

            stats["patterns_transferred"] += 1
            stats["pairs_updated"] += len(features) * (len(features) - 1) // 2

        return stats

    def _consolidate_nap(self, rate_multiplier: float) -> dict:
        """Nap: Prune weak traces, transfer strong patterns to embeddings.

        1. Prune trace connections below threshold
        2. For strong connections, move features closer in embedding space
        """
        stats = {"pruned": 0, "embeddings_updated": 0}

        # Prune weak connections (threshold adjusted by surprise)
        threshold = self.nap_prune_threshold / rate_multiplier
        stats["pruned"] = self.trace.prune(threshold=threshold)

        # Transfer strong patterns to embeddings
        csr = self.trace.matrix.tocoo()
        for i, j, strength in zip(csr.row, csr.col, csr.data):
            if i < j and strength > self.embedding_transfer_threshold:
                self.embeddings.update_from_coactivation(
                    int(i), int(j), float(strength) * rate_multiplier
                )
                stats["embeddings_updated"] += 1

        return stats

    def _consolidate_full(self, rate_multiplier: float) -> dict:
        """Full dream: Apply decay, consolidate to graph (not implemented here).

        Graph consolidation requires PsycheClient, handled by FeatureSubstrate.
        """
        stats = {"decay_applied": True}

        # Apply decay to trace matrix
        self.trace.decay()

        return stats

    def _consolidate_deep(self, rate_multiplier: float) -> dict:
        """Deep reflection: Aggressive pruning, topology restructuring.

        Uses higher threshold to remove moderate-strength connections,
        keeping only the strongest patterns.
        """
        stats = {"pruned": 0, "attractors_dissolved": 0}

        # Aggressive pruning
        threshold = self.deep_prune_threshold / rate_multiplier
        stats["pruned"] = self.trace.prune(threshold=threshold)

        # TODO: Attractor dissolution when graph integration is added

        return stats

    def get_consolidation_stats(self) -> dict:
        """Get current consolidation statistics."""
        return {
            "buffer_size": len(self.buffer),
            "trace_sparsity": self.trace.sparsity,
            "embedding_variance": self.embeddings.variance(),
            "attractor_count": len(self.embeddings.attractors),
        }
