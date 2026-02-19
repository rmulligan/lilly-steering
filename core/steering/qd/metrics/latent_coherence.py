"""Latent Coherence Metric: Rewards consistent-but-diverse latent paths.

Inspired by ATP-Latent (Zheng & Lee 2025): Penalizes both mode collapse
(too similar to recent) AND incoherence (too dissimilar to context).

The sweet spot: Semantically coherent + sufficiently novel.

This is the PROACTIVE layer for semantic diversity - it guides steering
to avoid stagnation rather than detecting it after the fact (which is
what WeaverControlLoop does as a fallback).
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LatentCoherenceConfig:
    """Configuration for latent coherence scoring.

    Attributes:
        window_size: Number of recent embeddings to compare against.
            Shorter = more diversity pressure, longer = more stability.
        collapse_threshold: Similarity above this = mode collapse (penalize).
        coherence_floor: Similarity below this = incoherent (penalize).
        optimal_similarity: Target similarity for ideal balance.
            The scoring function peaks here.
    """

    window_size: int = 10
    collapse_threshold: float = 0.9
    coherence_floor: float = 0.3
    optimal_similarity: float = 0.6


class LatentCoherenceMetric:
    """Computes coherence reward from embedding similarity dynamics.

    Uses the ATP-Latent insight: Good latent planning produces outputs that are
    CONSISTENT (semantically related to context) but DIVERSE (not repeating).

    Score function (inverted U-shape):
        - sim < coherence_floor: Low score (incoherent, penalize)
        - sim > collapse_threshold: Low score (mode collapse, penalize)
        - sim ≈ optimal_similarity: High score (ideal balance)

    Attributes:
        config: Latent coherence configuration.
    """

    def __init__(self, config: Optional[LatentCoherenceConfig] = None):
        """Initialize the metric.

        Args:
            config: Configuration for thresholds and window size.
        """
        self.config = config or LatentCoherenceConfig()
        self._recent_embeddings: deque[np.ndarray] = deque(
            maxlen=self.config.window_size
        )
        self._last_score: float = 0.5
        self._last_avg_similarity: float = 0.0

    def compute(
        self,
        current_embedding: Optional[np.ndarray],
        context_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute latent coherence score.

        Args:
            current_embedding: Embedding of current thought (SAE or golden).
                If None, returns neutral score.
            context_embedding: Optional context for coherence check.

        Returns:
            Score in [0, 1] where 1 = optimal coherence-diversity balance.
        """
        if current_embedding is None:
            return 0.5  # Neutral when no embedding available

        # Normalize embedding
        norm = np.linalg.norm(current_embedding)
        if norm < 1e-8:
            return 0.5
        current_normalized = current_embedding / norm

        # First observation - neutral score, just record
        if len(self._recent_embeddings) == 0:
            self._recent_embeddings.append(current_normalized.copy())
            self._last_score = 0.5
            self._last_avg_similarity = 0.0
            return 0.5

        # Compute similarity to recent embeddings (collapse detection)
        recent_sims = [
            self._cosine_similarity(current_normalized, past)
            for past in self._recent_embeddings
        ]
        avg_recent_sim = float(np.mean(recent_sims))
        max_recent_sim = float(np.max(recent_sims))
        self._last_avg_similarity = avg_recent_sim

        # Compute context coherence if provided
        if context_embedding is not None:
            context_norm = np.linalg.norm(context_embedding)
            if context_norm > 1e-8:
                context_sim = self._cosine_similarity(
                    current_normalized, context_embedding / context_norm
                )
            else:
                context_sim = 0.5
        else:
            context_sim = 0.5  # Neutral when no context

        cfg = self.config

        # === Diversity score (penalize collapse) ===
        # Using max similarity to detect exact repeats
        if max_recent_sim > cfg.collapse_threshold:
            # Severe penalty for near-exact repeats
            collapse_severity = (max_recent_sim - cfg.collapse_threshold) / (
                1.0 - cfg.collapse_threshold
            )
            diversity_score = max(0.0, 1.0 - collapse_severity * 2.0)
        elif avg_recent_sim > cfg.optimal_similarity:
            # Mild penalty for drifting toward collapse
            drift = (avg_recent_sim - cfg.optimal_similarity) / (
                cfg.collapse_threshold - cfg.optimal_similarity
            )
            diversity_score = 1.0 - drift * 0.3
        else:
            diversity_score = 1.0

        # === Coherence score (penalize incoherence) ===
        if context_sim < cfg.coherence_floor:
            # Penalty for being too dissimilar to context
            coherence_score = context_sim / cfg.coherence_floor
        else:
            coherence_score = 1.0

        # === Optimality bonus (near ideal similarity) ===
        distance_from_optimal = abs(avg_recent_sim - cfg.optimal_similarity)
        # Peak at optimal, decay to 0.7 at extremes
        optimality_score = 1.0 - min(distance_from_optimal / 0.4, 0.3)

        # === Combined score ===
        # Must pass both diversity and coherence checks
        # Optimality provides bonus within the valid range
        final_score = diversity_score * coherence_score * optimality_score

        # Record for future comparisons
        self._recent_embeddings.append(current_normalized.copy())

        self._last_score = float(np.clip(final_score, 0.0, 1.0))

        logger.debug(
            f"LatentCoherence: score={self._last_score:.3f}, "
            f"avg_sim={avg_recent_sim:.3f}, max_sim={max_recent_sim:.3f}, "
            f"context_sim={context_sim:.3f}, diversity={diversity_score:.3f}, "
            f"coherence={coherence_score:.3f}"
        )

        return self._last_score

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors.

        The caller ensures that inputs are normalized unit vectors.
        """
        return float(np.dot(a, b))

    def clear_history(self) -> None:
        """Clear the recent embeddings history."""
        self._recent_embeddings.clear()
        self._last_score = 0.5
        self._last_avg_similarity = 0.0

    def get_stats(self) -> dict:
        """Get metric statistics for monitoring.

        Returns:
            Dictionary with current state and history info.
        """
        return {
            "history_size": len(self._recent_embeddings),
            "window_size": self.config.window_size,
            "last_score": self._last_score,
            "last_avg_similarity": self._last_avg_similarity,
            "collapse_threshold": self.config.collapse_threshold,
            "coherence_floor": self.config.coherence_floor,
            "optimal_similarity": self.config.optimal_similarity,
        }
