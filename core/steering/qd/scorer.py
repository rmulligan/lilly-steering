"""QDScorer: Quality Diversity scoring orchestrator.

Combines the five QD metrics (Coherence, Novelty, Surprise, Presence, LatentCoherence)
into a weighted score for crystal selection in EvalatisSteerer.

The LatentCoherence metric is inspired by ATP-Latent (Zheng & Lee 2025) and provides
proactive diversity maintenance by rewarding consistent-but-diverse latent paths.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.steering.qd.config import QDConfig
from core.steering.qd.metrics.coherence import CoherenceMetric
from core.steering.qd.metrics.latent_coherence import (
    LatentCoherenceConfig,
    LatentCoherenceMetric,
)
from core.steering.qd.metrics.novelty import NoveltyMetric
from core.steering.qd.metrics.presence import PresenceMetric
from core.steering.qd.metrics.surprise import SurpriseMetric

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService
    from core.recognition.feature_tracker import ApprovedFeatureTracker
    from core.steering.crystal import CrystalEntry
    from core.steering.emergent import EmergentSlot

logger = logging.getLogger(__name__)


@dataclass
class QDContext:
    """Context for QD scoring.

    Provides the current state information needed for metric computation.

    Attributes:
        zone_name: Name of the steering zone being scored
        current_cycle: Current cognitive cycle number
        context_embedding: Optional embedding for coherence calculation
        current_embedding: Optional embedding of current thought for latent coherence
        sae_features: Optional SAE features for presence calculation
        approval_bonus: Optional pre-computed approval bonus
    """

    zone_name: str
    current_cycle: int = 0
    context_embedding: Optional[np.ndarray] = None
    current_embedding: Optional[np.ndarray] = None  # NEW: for latent coherence
    sae_features: Optional[list[tuple[int, float]]] = None
    approval_bonus: float = 0.0


@dataclass
class QDScore:
    """Result of QD scoring.

    Contains the total weighted score and individual metric scores.

    Attributes:
        total: Weighted total score (0 if below floor)
        coherence: Coherence metric score [0, 1]
        novelty: Novelty metric score [0, 1]
        surprise: Surprise metric score [0, 1]
        presence: Presence metric score [0, 1]
        latent_coherence: Latent coherence metric score [0, 1]
        passed_floor: Whether crystal passed quality floor
    """

    total: float
    coherence: float
    novelty: float
    surprise: float
    presence: float
    latent_coherence: float = 0.5  # NEW: default to neutral
    passed_floor: bool = True


class QDScorer:
    """Quality Diversity scoring orchestrator.

    Combines five metrics with configurable (and optionally adaptive) weights:
    - COHERENCE (15%): Quality floor via embedding alignment
    - NOVELTY (30%): Diversity via inverse similarity
    - SURPRISE (20%): Emergence signal from activations
    - PRESENCE (15%): Authenticity from approval patterns
    - LATENT_COHERENCE (20%): Proactive diversity via embedding dynamics

    The coherence threshold acts as a quality floor - crystals scoring
    below it receive a total score of 0 regardless of other metrics.

    When config.adaptive=True, weights adjust based on cycle outcomes
    to learn configurations that improve H_sem, D, and verification_rate.

    Attributes:
        config: QD configuration with weights and thresholds
        coherence: CoherenceMetric instance
        novelty: NoveltyMetric instance
        surprise: SurpriseMetric instance
        presence: PresenceMetric instance
        latent_coherence: LatentCoherenceMetric instance
    """

    def __init__(
        self,
        config: Optional[QDConfig] = None,
        embedding_service: Optional["TieredEmbeddingService"] = None,
        feature_tracker: Optional["ApprovedFeatureTracker"] = None,
    ):
        """Initialize QD scorer.

        Args:
            config: QD configuration. Uses defaults if None.
            embedding_service: Service for coherence embeddings.
            feature_tracker: Tracker for presence scoring.
        """
        self.config = config or QDConfig()

        # Initialize existing metrics
        self.coherence = CoherenceMetric(embedding_service)
        self.novelty = NoveltyMetric(
            window_size=self.config.novelty_window,
            decay=self.config.novelty_similarity_decay,
        )
        self.surprise = SurpriseMetric(
            normalize_max=self.config.surprise_normalize_max,
        )
        self.presence = PresenceMetric(
            feature_tracker=feature_tracker,
            ema_alpha=self.config.presence_ema_alpha,
        )

        # Initialize latent coherence metric (ATP-Latent inspired)
        latent_config = LatentCoherenceConfig(
            window_size=self.config.latent_window_size,
            collapse_threshold=self.config.latent_collapse_threshold,
            coherence_floor=self.config.latent_coherence_floor,
            optimal_similarity=self.config.latent_optimal_similarity,
        )
        self.latent_coherence = LatentCoherenceMetric(latent_config)

        # Initialize curvature tracker for natural gradient-inspired adaptation
        from core.steering.qd.curvature import CurvatureTracker
        self.curvature_tracker = CurvatureTracker(
            ema_alpha=self.config.curvature_ema_alpha,
        )

        # Recent selections for novelty (owned by scorer, not metric)
        self._recent_selections: deque[np.ndarray] = deque(
            maxlen=self.config.novelty_window
        )

        # Track last latent coherence score for monitoring
        self._last_latent_coherence_score: float = 0.5

    def score(self, crystal: "CrystalEntry", context: QDContext) -> QDScore:
        """Compute weighted QD score for a crystal.

        Args:
            crystal: The crystal entry to score
            context: Current context for metric computation

        Returns:
            QDScore with total and individual metric scores
        """
        # Compute individual metrics
        coherence_score = self.coherence.compute(crystal, context.context_embedding)
        novelty_score = self.novelty.compute(crystal, self._recent_selections)
        surprise_score = self.surprise.compute(crystal)
        presence_score = self.presence.compute(crystal, context.sae_features)

        # Compute latent coherence score
        latent_coherence_score = self.latent_coherence.compute(
            current_embedding=context.current_embedding,
            context_embedding=context.context_embedding,
        )
        self._last_latent_coherence_score = latent_coherence_score

        # Quality floor check
        if coherence_score < self.config.coherence_threshold:
            return QDScore(
                total=0.0,
                coherence=coherence_score,
                novelty=novelty_score,
                surprise=surprise_score,
                presence=presence_score,
                latent_coherence=latent_coherence_score,
                passed_floor=False,
            )

        # Weighted combination
        total = (
            self.config.coherence_weight * coherence_score
            + self.config.novelty_weight * novelty_score
            + self.config.surprise_weight * surprise_score
            + self.config.presence_weight * presence_score
            + self.config.latent_coherence_weight * latent_coherence_score
        )

        return QDScore(
            total=total,
            coherence=coherence_score,
            novelty=novelty_score,
            surprise=surprise_score,
            presence=presence_score,
            latent_coherence=latent_coherence_score,
            passed_floor=True,
        )

    def score_emergent(
        self,
        emergent: "EmergentSlot",
        context: QDContext,
    ) -> QDScore:
        """Compute QD score for an emergent slot.

        Creates a temporary "virtual crystal" from the emergent slot's
        current state to enable consistent scoring.

        Args:
            emergent: The emergent slot to score
            context: Current context for metric computation

        Returns:
            QDScore for the emergent slot
        """
        # Create virtual crystal-like object for scoring
        # We can't import CrystalEntry here to avoid circular imports,
        # so we create an ad-hoc object with the needed properties

        class EmergentProxy:
            """Proxy object for scoring emergent slots like crystals."""

            def __init__(self, slot: "EmergentSlot"):
                self.vector = slot.vector
                self.name = "emergent"
                # Use surprise_ema as avg_surprise for emergent
                self.avg_surprise = slot.surprise_ema

        proxy = EmergentProxy(emergent)

        # Score like a crystal
        coherence_score = self.coherence.compute(proxy, context.context_embedding)  # type: ignore
        novelty_score = self.novelty.compute(proxy, self._recent_selections)  # type: ignore
        surprise_score = self.surprise.compute(proxy)  # type: ignore
        presence_score = self.presence.compute(proxy, context.sae_features)  # type: ignore

        # Compute latent coherence score
        latent_coherence_score = self.latent_coherence.compute(
            current_embedding=context.current_embedding,
            context_embedding=context.context_embedding,
        )
        self._last_latent_coherence_score = latent_coherence_score

        # Quality floor check
        if coherence_score < self.config.coherence_threshold:
            return QDScore(
                total=0.0,
                coherence=coherence_score,
                novelty=novelty_score,
                surprise=surprise_score,
                presence=presence_score,
                latent_coherence=latent_coherence_score,
                passed_floor=False,
            )

        # Weighted combination
        total = (
            self.config.coherence_weight * coherence_score
            + self.config.novelty_weight * novelty_score
            + self.config.surprise_weight * surprise_score
            + self.config.presence_weight * presence_score
            + self.config.latent_coherence_weight * latent_coherence_score
        )

        return QDScore(
            total=total,
            coherence=coherence_score,
            novelty=novelty_score,
            surprise=surprise_score,
            presence=presence_score,
            latent_coherence=latent_coherence_score,
            passed_floor=True,
        )

    def adapt_weights(
        self, 
        outcomes: dict[str, float],
        health_status: str | None = None,
    ) -> dict[str, float]:
        """Adapt QD weights based on cycle outcomes with optional preconditioning.

        Delegates to QDConfig.adapt_weights() which implements the
        gradient-free optimization logic. Passes curvature tracker for
        natural gradient-inspired preconditioning and health status for
        adaptive throttling.

        Args:
            outcomes: Dictionary with keys 'H_sem', 'D', 'verification_rate'.
            health_status: Optional health status for adaptation throttling
                (CRITICAL/STRESSED/STABLE/THRIVING)

        Returns:
            Dictionary of weight deltas applied (empty if adaptation disabled).
        """
        return self.config.adapt_weights(
            outcomes,
            curvature_tracker=self.curvature_tracker,
            health_status=health_status,
        )

    def record_selection(self, vector: np.ndarray) -> None:
        """Record a selected vector for novelty computation.

        Call this after a vector is selected to update the novelty
        tracking history.

        Args:
            vector: The selected steering vector
        """
        self._recent_selections.append(vector.copy())

    def clear_history(self) -> None:
        """Clear selection history and metric state."""
        self._recent_selections.clear()
        self.novelty.clear_history()
        self.latent_coherence.clear_history()

    def get_stats(self) -> dict:
        """Get scorer statistics.

        Returns:
            Dictionary with scoring statistics
        """
        return {
            "recent_selections": len(self._recent_selections),
            "window_size": self.config.novelty_window,
            "coherence_threshold": self.config.coherence_threshold,
            "presence_stats": self.presence.get_stats(),
            "latent_coherence_stats": self.latent_coherence.get_stats(),
            "weights": self.config.get_weights(),
            "adaptive": self.config.adaptive,
            "frozen_weights": self.config.frozen_weights,
            "last_latent_coherence_score": self._last_latent_coherence_score,
        }
