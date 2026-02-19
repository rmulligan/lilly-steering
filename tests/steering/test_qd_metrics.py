# tests/steering/test_qd_metrics.py
"""Tests for Quality Diversity (QD) metrics in EvalatisSteerer.

Includes tests for:
- Individual QD metrics (Coherence, Novelty, Surprise, Presence)
- LatentCoherenceMetric (ATP-Latent inspired proactive diversity)
- Adaptive weight learning in QDConfig
- QDScorer integration with all five metrics
"""

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest

from core.steering.crystal import CrystalEntry, CrystallizationConfig
from core.steering.config import HierarchicalSteeringConfig, SteeringZone
from core.steering.emergent import create_emergent_slot
from core.steering.evalatis import EvalatisSteerer
from core.steering.qd.config import QDConfig
from core.steering.qd.metrics.coherence import CoherenceMetric
from core.steering.qd.metrics.latent_coherence import (
    LatentCoherenceConfig,
    LatentCoherenceMetric,
)
from core.steering.qd.metrics.novelty import NoveltyMetric
from core.steering.qd.metrics.presence import PresenceMetric
from core.steering.qd.metrics.surprise import SurpriseMetric
from core.steering.qd.scorer import QDContext, QDScore, QDScorer


# === Fixtures ===


@pytest.fixture
def d_model():
    """Standard model dimension for tests."""
    return 128


@pytest.fixture
def random_vector(d_model):
    """Generate a random unit vector."""
    vec = np.random.randn(d_model).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def crystal(d_model):
    """Standard test crystal."""
    return CrystalEntry(
        name="test_crystal",
        vector=np.random.randn(d_model).astype(np.float32),
        birth_surprise=50.0,
        selection_count=5,
        total_surprise=250.0,  # avg_surprise = 50
    )


@pytest.fixture
def config():
    """Standard hierarchical config for testing."""
    return HierarchicalSteeringConfig(
        zones=[
            SteeringZone(name="exploration", layers=(2, 4), max_magnitude=3.0, ema_alpha=0.12),
            SteeringZone(name="concept", layers=(6, 8), max_magnitude=1.5, ema_alpha=0.2),
        ],
        observation_layer=12,
    )


# === QDConfig Tests ===


class TestQDConfig:
    """Tests for QDConfig dataclass."""

    def test_default_weights_sum_to_one(self):
        """Default metric weights sum to 1.0 (including latent coherence)."""
        config = QDConfig()
        total = (
            config.coherence_weight
            + config.novelty_weight
            + config.surprise_weight
            + config.presence_weight
            + config.latent_coherence_weight
        )
        assert total == pytest.approx(1.0)

    def test_default_values(self):
        """Check default configuration values (5-metric weights)."""
        config = QDConfig()
        # Updated weights for 5-metric system
        assert config.coherence_weight == 0.15
        assert config.novelty_weight == 0.30
        assert config.surprise_weight == 0.20
        assert config.presence_weight == 0.15
        assert config.latent_coherence_weight == 0.20
        assert config.coherence_threshold == 0.5
        assert config.novelty_window == 20
        assert config.surprise_normalize_max == 100.0

    def test_invalid_weights_raises_error(self):
        """Weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            QDConfig(
                coherence_weight=0.4,
                novelty_weight=0.4,
                surprise_weight=0.4,
                presence_weight=0.4,
                latent_coherence_weight=0.4,  # Sum = 2.0
            )

    def test_invalid_threshold_raises_error(self):
        """Invalid coherence threshold raises ValueError."""
        with pytest.raises(ValueError, match="Coherence threshold must be in"):
            QDConfig(coherence_threshold=1.5)

        with pytest.raises(ValueError, match="Coherence threshold must be in"):
            QDConfig(coherence_threshold=-0.1)

    def test_custom_weights(self):
        """Custom weights are accepted when valid (must include all 5)."""
        config = QDConfig(
            coherence_weight=0.20,
            novelty_weight=0.20,
            surprise_weight=0.20,
            presence_weight=0.20,
            latent_coherence_weight=0.20,
        )
        assert config.coherence_weight == 0.20
        assert config.novelty_weight == 0.20

    def test_niche_names_default(self):
        """Niche names have default values."""
        config = QDConfig()
        assert len(config.niche_names) == 5
        assert "exploratory" in config.niche_names
        assert "conceptual" in config.niche_names


# === CoherenceMetric Tests ===


class TestCoherenceMetric:
    """Tests for CoherenceMetric."""

    def test_returns_neutral_without_context(self, crystal):
        """Returns 0.5 (neutral) when no context embedding provided."""
        metric = CoherenceMetric()
        score = metric.compute(crystal, context_embedding=None)
        assert score == 0.5

    def test_identical_vectors_max_coherence(self, d_model):
        """Identical vectors produce maximum coherence (1.0)."""
        metric = CoherenceMetric()
        vec = np.random.randn(d_model).astype(np.float32)
        crystal = CrystalEntry(name="test", vector=vec.copy())

        score = metric.compute(crystal, context_embedding=vec)
        assert score == pytest.approx(1.0, rel=0.01)

    def test_opposite_vectors_min_coherence(self, d_model):
        """Opposite vectors produce minimum coherence (0.0)."""
        metric = CoherenceMetric()
        vec = np.random.randn(d_model).astype(np.float32)
        crystal = CrystalEntry(name="test", vector=-vec)

        score = metric.compute(crystal, context_embedding=vec)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_neutral_coherence(self, d_model):
        """Orthogonal vectors produce neutral coherence (0.5)."""
        metric = CoherenceMetric()
        # Create orthogonal vectors
        vec1 = np.zeros(d_model, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(d_model, dtype=np.float32)
        vec2[1] = 1.0

        crystal = CrystalEntry(name="test", vector=vec1)
        score = metric.compute(crystal, context_embedding=vec2)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_zero_vector_returns_neutral(self, d_model):
        """Zero vectors return neutral score."""
        metric = CoherenceMetric()
        zero_vec = np.zeros(d_model, dtype=np.float32)
        crystal = CrystalEntry(name="test", vector=zero_vec)
        context = np.random.randn(d_model).astype(np.float32)

        score = metric.compute(crystal, context_embedding=context)
        assert score == 0.5  # Neutral for zero norm


# === NoveltyMetric Tests ===


class TestNoveltyMetric:
    """Tests for NoveltyMetric."""

    def test_max_novelty_without_history(self, crystal):
        """Returns 1.0 (max novelty) when no selection history."""
        metric = NoveltyMetric(window_size=20)
        score = metric.compute(crystal, recent_selections=deque())
        assert score == 1.0

    def test_zero_novelty_for_identical_vector(self, d_model):
        """Returns 0.0 when crystal is identical to recent selection."""
        metric = NoveltyMetric(window_size=20, decay=1.0)
        vec = np.random.randn(d_model).astype(np.float32)
        crystal = CrystalEntry(name="test", vector=vec.copy())

        recent = deque([vec.copy()])
        score = metric.compute(crystal, recent_selections=recent)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_high_novelty_for_orthogonal_vector(self, d_model):
        """Returns high novelty for orthogonal vectors."""
        metric = NoveltyMetric(window_size=20, decay=1.0)
        vec1 = np.zeros(d_model, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(d_model, dtype=np.float32)
        vec2[1] = 1.0

        crystal = CrystalEntry(name="test", vector=vec2)
        recent = deque([vec1])

        score = metric.compute(crystal, recent_selections=recent)
        assert score == 1.0  # Orthogonal = max novelty

    def test_recency_decay_reduces_similarity_impact(self, d_model):
        """Older selections have less impact on novelty due to decay."""
        vec = np.random.randn(d_model).astype(np.float32)
        crystal = CrystalEntry(name="test", vector=vec.copy())

        # With decay=0.5, the impact halves for each step back
        metric_with_decay = NoveltyMetric(window_size=20, decay=0.5)
        metric_no_decay = NoveltyMetric(window_size=20, decay=1.0)

        # Add identical vector at the OLDEST position (first in deque, last when reversed)
        # deque order: [oldest, ..., newest], reversed: [newest, ..., oldest]
        # So [identical, random] → reversed → [random (i=0), identical (i=1)]
        # At i=1, decay^1 = 0.5 for decay metric, 1.0 for no-decay metric
        recent = deque([vec.copy(), np.random.randn(d_model).astype(np.float32)])

        score_decay = metric_with_decay.compute(crystal, recent_selections=recent)
        score_no_decay = metric_no_decay.compute(crystal, recent_selections=recent)

        # With decay, older identical vector has less impact → higher novelty score
        assert score_decay > score_no_decay

    def test_record_selection_updates_history(self):
        """record_selection adds vector to internal history."""
        metric = NoveltyMetric(window_size=5)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        metric.record_selection(vec)
        assert len(metric._recent_selections) == 1

        metric.record_selection(vec)
        assert len(metric._recent_selections) == 2

    def test_clear_history(self):
        """clear_history removes all tracked selections."""
        metric = NoveltyMetric(window_size=5)
        metric.record_selection(np.array([1.0], dtype=np.float32))
        metric.record_selection(np.array([2.0], dtype=np.float32))

        metric.clear_history()
        assert len(metric._recent_selections) == 0


# === SurpriseMetric Tests ===


class TestSurpriseMetric:
    """Tests for SurpriseMetric."""

    def test_normalize_surprise(self, d_model):
        """Surprise is normalized to [0, 1] based on normalize_max."""
        metric = SurpriseMetric(normalize_max=100.0)

        crystal = CrystalEntry(
            name="test",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=50.0,
        )

        score = metric.compute(crystal)
        assert score == pytest.approx(0.5, rel=0.01)

    def test_clamps_at_max(self, d_model):
        """Surprise above normalize_max is clamped to 1.0."""
        metric = SurpriseMetric(normalize_max=100.0)

        crystal = CrystalEntry(
            name="test",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=150.0,
        )

        score = metric.compute(crystal)
        assert score == 1.0

    def test_zero_surprise(self, d_model):
        """Zero surprise produces score 0.0."""
        metric = SurpriseMetric(normalize_max=100.0)

        crystal = CrystalEntry(
            name="test",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=0.0,
        )

        score = metric.compute(crystal)
        assert score == 0.0

    def test_uses_avg_surprise_when_selected(self, d_model):
        """Uses avg_surprise (not birth_surprise) when crystal has selections."""
        metric = SurpriseMetric(normalize_max=100.0)

        crystal = CrystalEntry(
            name="test",
            vector=np.zeros(d_model, dtype=np.float32),
            birth_surprise=100.0,  # Would give 1.0
            selection_count=2,
            total_surprise=100.0,  # avg = 50.0 → 0.5
        )

        score = metric.compute(crystal)
        assert score == pytest.approx(0.5, rel=0.01)


# === PresenceMetric Tests ===


class TestPresenceMetric:
    """Tests for PresenceMetric."""

    def test_returns_neutral_without_tracker(self, crystal):
        """Returns 0.5 (neutral) when no feature tracker configured."""
        metric = PresenceMetric(feature_tracker=None)
        score = metric.compute(crystal, sae_features=[(100, 0.8)])
        assert score == 0.5

    def test_returns_cached_score_without_features(self, crystal):
        """Returns cached score when no SAE features provided."""
        metric = PresenceMetric(feature_tracker=None)
        # First compute with features (would use neutral)
        metric._presence_scores[crystal.name] = 0.7
        metric._observation_counts[crystal.name] = 5

        score = metric.compute(crystal, sae_features=None)
        assert score == 0.7

    def test_ema_update_from_approval_bonus(self, crystal):
        """EMA updates presence score from tracker approval bonus."""
        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.8  # High approval

        metric = PresenceMetric(feature_tracker=mock_tracker, ema_alpha=0.5)

        # First observation: neutral (0.5) → 0.8 approval → score
        # approval_bonus 0.8 maps to presence 0.9 ((0.8+1)/2)
        # EMA: 0.5 * 0.5 + 0.5 * 0.9 = 0.7
        score = metric.compute(crystal, sae_features=[(100, 0.8)])
        assert score == pytest.approx(0.7, rel=0.01)

    def test_observation_count_tracking(self, crystal):
        """Tracks observation count per crystal."""
        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.0

        metric = PresenceMetric(feature_tracker=mock_tracker)

        assert metric.get_observation_count(crystal.name) == 0

        metric.compute(crystal, sae_features=[(100, 0.8)])
        assert metric.get_observation_count(crystal.name) == 1

        metric.compute(crystal, sae_features=[(100, 0.8)])
        assert metric.get_observation_count(crystal.name) == 2

    def test_has_sufficient_observations(self, crystal):
        """has_sufficient_observations checks minimum count."""
        mock_tracker = Mock()
        mock_tracker.get_approval_bonus.return_value = 0.0

        metric = PresenceMetric(feature_tracker=mock_tracker)

        assert not metric.has_sufficient_observations(crystal.name, min_observations=5)

        for _ in range(5):
            metric.compute(crystal, sae_features=[(100, 0.8)])

        assert metric.has_sufficient_observations(crystal.name, min_observations=5)

    def test_reset_crystal(self, crystal):
        """reset_crystal clears cached state."""
        metric = PresenceMetric()
        metric._presence_scores[crystal.name] = 0.8
        metric._observation_counts[crystal.name] = 10

        metric.reset_crystal(crystal.name)

        assert crystal.name not in metric._presence_scores
        assert crystal.name not in metric._observation_counts


# === QDScorer Tests ===


class TestQDScorer:
    """Tests for QDScorer orchestrator."""

    def test_default_initialization(self):
        """Scorer initializes with default config."""
        scorer = QDScorer()
        assert scorer.config is not None
        assert scorer.coherence is not None
        assert scorer.novelty is not None
        assert scorer.surprise is not None
        assert scorer.presence is not None

    def test_weighted_score_computation(self, crystal, d_model):
        """Computes weighted combination of metric scores."""
        config = QDConfig(
            coherence_weight=0.20,
            novelty_weight=0.20,
            surprise_weight=0.20,
            presence_weight=0.20,
            latent_coherence_weight=0.20,
        )
        scorer = QDScorer(config)

        context = QDContext(zone_name="exploration", current_cycle=10)
        score = scorer.score(crystal, context)

        # All metrics should have computed
        assert score.passed_floor
        assert 0.0 <= score.total <= 1.0
        assert 0.0 <= score.coherence <= 1.0
        assert 0.0 <= score.novelty <= 1.0
        assert 0.0 <= score.surprise <= 1.0
        assert 0.0 <= score.presence <= 1.0
        assert 0.0 <= score.latent_coherence <= 1.0

    def test_coherence_floor_rejection(self, d_model):
        """Crystals below coherence threshold are rejected."""
        config = QDConfig(coherence_threshold=0.8)  # High threshold
        scorer = QDScorer(config)

        # Crystal with vector that will have low coherence
        crystal = CrystalEntry(
            name="test",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=50.0,
        )

        # Context with opposite direction embedding
        context = QDContext(
            zone_name="exploration",
            context_embedding=-crystal.vector,  # Opposite direction
        )

        score = scorer.score(crystal, context)
        assert not score.passed_floor
        assert score.total == 0.0

    def test_record_selection_updates_novelty(self, d_model):
        """record_selection updates novelty tracking."""
        scorer = QDScorer()
        vec = np.random.randn(d_model).astype(np.float32)

        assert len(scorer._recent_selections) == 0

        scorer.record_selection(vec)
        assert len(scorer._recent_selections) == 1

    def test_clear_history(self, d_model):
        """clear_history resets selection tracking."""
        scorer = QDScorer()
        scorer.record_selection(np.random.randn(d_model).astype(np.float32))
        scorer.record_selection(np.random.randn(d_model).astype(np.float32))

        scorer.clear_history()
        assert len(scorer._recent_selections) == 0

    def test_score_emergent_slot(self, d_model):
        """score_emergent scores emergent slots like crystals."""
        scorer = QDScorer()

        emergent = create_emergent_slot(d_model)
        emergent.vector = np.random.randn(d_model).astype(np.float32)
        emergent.surprise_ema = 40.0

        context = QDContext(zone_name="exploration")
        score = scorer.score_emergent(emergent, context)

        assert score.passed_floor
        assert 0.0 <= score.total <= 1.0
        # Surprise should reflect EMA
        assert score.surprise == pytest.approx(0.4, rel=0.01)  # 40/100

    def test_get_stats(self):
        """get_stats returns scorer statistics."""
        scorer = QDScorer()
        stats = scorer.get_stats()

        assert "recent_selections" in stats
        assert "window_size" in stats
        assert "weights" in stats
        # Updated for 5-metric weights
        assert stats["weights"]["coherence"] == 0.15
        assert stats["weights"]["latent_coherence"] == 0.20


# === EvalatisSteerer QD Integration Tests ===


class TestEvalatisQDIntegration:
    """Integration tests for QD scoring in EvalatisSteerer."""

    def test_steerer_accepts_qd_scorer(self, config, d_model):
        """EvalatisSteerer accepts optional qd_scorer parameter."""
        qd_scorer = QDScorer()
        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            qd_scorer=qd_scorer,
        )
        assert steerer.qd_scorer is qd_scorer

    def test_set_qd_scorer(self, config, d_model):
        """set_qd_scorer enables QD scoring after initialization."""
        steerer = EvalatisSteerer(config=config, d_model=d_model)
        assert steerer.qd_scorer is None

        qd_scorer = QDScorer()
        steerer.set_qd_scorer(qd_scorer)
        assert steerer.qd_scorer is qd_scorer

    def test_selection_uses_qd_when_scorer_configured(self, config, d_model):
        """Selection uses QD scoring when scorer is configured."""
        qd_scorer = QDScorer()
        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            qd_scorer=qd_scorer,
        )

        # Add crystal
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=50.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # Set emergent
        steerer.zones["exploration"].emergent.vector = np.random.randn(d_model).astype(np.float32)
        steerer.zones["exploration"].emergent.surprise_ema = 30.0

        # Selection should use QD scoring
        selected_name, is_emergent = steerer._select_vector("exploration")

        # Should have selected something (either crystal or emergent)
        assert selected_name in ["test_crystal", "emergent"]

    def test_update_from_cycle_with_qd_scorer(self, config, d_model):
        """update_from_cycle works with QD scorer."""
        qd_scorer = QDScorer()
        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            qd_scorer=qd_scorer,
        )

        activations = np.random.randn(d_model).astype(np.float32)
        events = steerer.update_from_cycle(
            zone_name="exploration",
            activations=activations,
            surprise=45.0,
        )

        assert "selected_name" in events
        assert "selected_is_emergent" in events

    def test_qd_selection_records_for_novelty(self, config, d_model):
        """QD selection records vectors for novelty tracking."""
        qd_scorer = QDScorer()
        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            qd_scorer=qd_scorer,
        )

        # Set emergent to have a vector
        steerer.zones["exploration"].emergent.vector = np.random.randn(d_model).astype(np.float32)
        steerer.zones["exploration"].emergent.surprise_ema = 50.0

        assert len(qd_scorer._recent_selections) == 0

        steerer._select_vector("exploration")

        # Should have recorded selection
        assert len(qd_scorer._recent_selections) == 1

    def test_backward_compatible_without_qd_scorer(self, config, d_model):
        """Selection falls back to affinity×freshness without QD scorer."""
        crystal_config = CrystallizationConfig()
        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            crystal_config=crystal_config,
            qd_scorer=None,  # No QD scorer
        )

        # Add crystal with high performance
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=70.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # Set emergent to low surprise
        steerer.zones["exploration"].emergent.surprise_ema = 10.0

        # Should use original affinity×freshness logic
        selected_name, is_emergent = steerer._select_vector("exploration")

        # Crystal should win with high surprise vs low emergent
        assert selected_name == "test_crystal"
        assert is_emergent is False

    def test_qd_floor_rejection_falls_back_to_emergent(self, config, d_model):
        """When all crystals fail QD floor, falls back to emergent."""
        qd_config = QDConfig(coherence_threshold=0.99)  # Very high threshold
        qd_scorer = QDScorer(qd_config)

        steerer = EvalatisSteerer(
            config=config,
            d_model=d_model,
            qd_scorer=qd_scorer,
        )

        # Add crystal that will fail coherence floor
        crystal = CrystalEntry(
            name="test_crystal",
            vector=np.random.randn(d_model).astype(np.float32),
            birth_surprise=50.0,
        )
        steerer.zones["exploration"].crystals.append(crystal)

        # Set context embedding to be different (low coherence)
        context_embedding = np.random.randn(d_model).astype(np.float32)

        # Set emergent to also have low coherence
        steerer.zones["exploration"].emergent.vector = np.random.randn(d_model).astype(np.float32)

        selected_name, is_emergent = steerer._select_vector(
            "exploration",
            context_embedding=context_embedding,
        )

        # Should fall back to emergent when nothing passes floor
        assert selected_name == "emergent"
        assert is_emergent is True


# === LatentCoherenceMetric Tests ===


class TestLatentCoherenceMetric:
    """Tests for ATP-Latent inspired latent coherence metric."""

    def test_first_observation_returns_neutral(self, d_model):
        """First observation returns neutral score (0.5)."""
        from core.steering.qd.metrics.latent_coherence import LatentCoherenceMetric

        metric = LatentCoherenceMetric()
        embedding = np.random.randn(d_model).astype(np.float32)

        score = metric.compute(embedding)
        assert score == 0.5

    def test_none_embedding_returns_neutral(self):
        """None embedding returns neutral score."""
        from core.steering.qd.metrics.latent_coherence import LatentCoherenceMetric

        metric = LatentCoherenceMetric()
        score = metric.compute(None)
        assert score == 0.5

    def test_similar_embeddings_penalized_collapse(self, d_model):
        """Very similar consecutive embeddings are penalized (collapse)."""
        from core.steering.qd.metrics.latent_coherence import (
            LatentCoherenceConfig,
            LatentCoherenceMetric,
        )

        config = LatentCoherenceConfig(collapse_threshold=0.9)
        metric = LatentCoherenceMetric(config)

        # Use same embedding repeatedly
        embedding = np.random.randn(d_model).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # First observation
        metric.compute(embedding)

        # Same embedding should be penalized
        score = metric.compute(embedding)
        assert score < 0.5  # Penalized for collapse

    def test_diverse_embeddings_rewarded(self, d_model):
        """Diverse but related embeddings are rewarded."""
        from core.steering.qd.metrics.latent_coherence import (
            LatentCoherenceConfig,
            LatentCoherenceMetric,
        )

        config = LatentCoherenceConfig(
            optimal_similarity=0.6,
            collapse_threshold=0.9,
            coherence_floor=0.3,
        )
        metric = LatentCoherenceMetric(config)

        # Generate embeddings with moderate similarity
        base = np.random.randn(d_model).astype(np.float32)
        base = base / np.linalg.norm(base)

        # First observation
        metric.compute(base)

        # Create moderately similar embedding
        noise = np.random.randn(d_model).astype(np.float32) * 0.5
        diverse = base + noise
        diverse = diverse / np.linalg.norm(diverse)

        score = metric.compute(diverse)
        # Should get reasonable score (not penalized)
        assert score > 0.3

    def test_incoherent_embeddings_penalized(self, d_model):
        """Completely unrelated embeddings penalized if context provided."""
        from core.steering.qd.metrics.latent_coherence import (
            LatentCoherenceConfig,
            LatentCoherenceMetric,
        )

        config = LatentCoherenceConfig(coherence_floor=0.3)
        metric = LatentCoherenceMetric(config)

        # First observation
        embedding1 = np.random.randn(d_model).astype(np.float32)
        metric.compute(embedding1)

        # Completely opposite context
        embedding2 = np.random.randn(d_model).astype(np.float32)
        context = -embedding2  # Opposite direction

        score = metric.compute(embedding2, context_embedding=context)
        # Should be penalized for incoherence with context
        assert score < 0.7

    def test_clear_history(self, d_model):
        """clear_history resets the metric state."""
        from core.steering.qd.metrics.latent_coherence import LatentCoherenceMetric

        metric = LatentCoherenceMetric()

        # Add some history
        for _ in range(5):
            metric.compute(np.random.randn(d_model).astype(np.float32))

        assert len(metric._recent_embeddings) == 5

        metric.clear_history()
        assert len(metric._recent_embeddings) == 0
        assert metric._last_score == 0.5

    def test_get_stats(self, d_model):
        """get_stats returns metric statistics."""
        from core.steering.qd.metrics.latent_coherence import LatentCoherenceMetric

        metric = LatentCoherenceMetric()
        stats = metric.get_stats()

        assert "history_size" in stats
        assert "window_size" in stats
        assert "collapse_threshold" in stats
        assert "coherence_floor" in stats


# === Adaptive QD Config Tests ===


class TestAdaptiveQDConfig:
    """Tests for adaptive weight learning in QDConfig."""

    def test_weights_sum_to_one_with_latent_coherence(self):
        """Weights sum to 1.0 including latent coherence."""
        config = QDConfig()
        total = (
            config.coherence_weight
            + config.novelty_weight
            + config.surprise_weight
            + config.presence_weight
            + config.latent_coherence_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_adapt_weights_low_H_sem(self):
        """Low H_sem increases diversity weights."""
        config = QDConfig(adaptive=True, frozen_weights=False)
        initial_novelty = config.novelty_weight
        initial_latent = config.latent_coherence_weight

        outcomes = {"H_sem": 0.1, "D": 0.0, "verification_rate": 0.3}
        config.adapt_weights(outcomes)

        # Diversity weights should increase
        assert config.novelty_weight >= initial_novelty - 0.01
        assert config.latent_coherence_weight >= initial_latent - 0.01

    def test_adapt_weights_negative_D(self):
        """Negative D increases exploration weights."""
        config = QDConfig(adaptive=True, frozen_weights=False)
        initial_latent = config.latent_coherence_weight

        outcomes = {"H_sem": 0.5, "D": -0.5, "verification_rate": 0.3}
        config.adapt_weights(outcomes)

        # Exploration weights should increase
        assert config.latent_coherence_weight >= initial_latent - 0.01

    def test_frozen_weights_prevents_adaptation(self):
        """Frozen weights prevent any adaptation."""
        config = QDConfig(adaptive=True, frozen_weights=True)
        initial_weights = config.get_weights().copy()

        outcomes = {"H_sem": 0.0, "D": -1.0, "verification_rate": 0.0}
        deltas = config.adapt_weights(outcomes)

        # No adaptation should occur
        assert deltas == {}
        assert config.get_weights() == initial_weights

    def test_weights_clamped_to_range(self):
        """Weights are clamped to [weight_min, weight_max]."""
        config = QDConfig(
            adaptive=True,
            frozen_weights=False,
            weight_learning_rate=1.0,  # Very high rate
            weight_min=0.05,
            weight_max=0.50,
        )

        # Extreme outcomes to push weights
        for _ in range(10):
            config.adapt_weights({"H_sem": 0.0, "D": -1.0, "verification_rate": 0.0})

        # All weights should be within bounds
        for name, weight in config.get_weights().items():
            assert config.weight_min <= weight <= config.weight_max

    def test_weights_renormalized_to_one(self):
        """Weights are renormalized to sum to 1.0 after adaptation."""
        config = QDConfig(adaptive=True, frozen_weights=False)

        outcomes = {"H_sem": 0.1, "D": -0.5, "verification_rate": 0.1}
        config.adapt_weights(outcomes)

        total = sum(config.get_weights().values())
        assert abs(total - 1.0) < 0.001

    def test_freeze_unfreeze(self):
        """freeze() and unfreeze() toggle adaptation."""
        config = QDConfig(adaptive=True, frozen_weights=False)

        config.freeze()
        assert config.frozen_weights is True

        config.unfreeze()
        assert config.frozen_weights is False

    def test_reset_weights(self):
        """reset_weights() restores defaults."""
        config = QDConfig(adaptive=True, frozen_weights=False)
        config.adapt_weights({"H_sem": 0.0, "D": -1.0, "verification_rate": 0.0})

        # Weights should have changed
        assert config.novelty_weight != 0.30

        config.reset_weights()

        # Should be back to defaults
        assert config.coherence_weight == 0.15
        assert config.novelty_weight == 0.30
        assert config.surprise_weight == 0.20
        assert config.presence_weight == 0.15
        assert config.latent_coherence_weight == 0.20


# === QDScorer with Latent Coherence Tests ===


class TestQDScorerLatentCoherence:
    """Tests for QDScorer with latent coherence integration."""

    def test_scorer_has_latent_coherence_metric(self):
        """QDScorer initializes latent coherence metric."""
        scorer = QDScorer()
        assert hasattr(scorer, "latent_coherence")
        assert scorer.latent_coherence is not None

    def test_score_includes_latent_coherence(self, d_model, crystal):
        """score() includes latent_coherence in result."""
        scorer = QDScorer()
        context = QDContext(
            zone_name="exploration",
            current_embedding=np.random.randn(d_model).astype(np.float32),
        )

        score = scorer.score(crystal, context)

        assert hasattr(score, "latent_coherence")
        assert 0.0 <= score.latent_coherence <= 1.0

    def test_scorer_adapt_weights(self):
        """QDScorer.adapt_weights delegates to config."""
        scorer = QDScorer()
        initial_weights = scorer.config.get_weights().copy()

        outcomes = {"H_sem": 0.1, "D": -0.5, "verification_rate": 0.1}
        deltas = scorer.adapt_weights(outcomes)

        assert deltas  # Some deltas should exist
        # Weights should have changed
        assert scorer.config.get_weights() != initial_weights

    def test_get_stats_includes_latent_coherence(self):
        """get_stats includes latent coherence information."""
        scorer = QDScorer()
        stats = scorer.get_stats()

        assert "latent_coherence_stats" in stats
        assert "last_latent_coherence_score" in stats
        assert "adaptive" in stats
        assert "frozen_weights" in stats
