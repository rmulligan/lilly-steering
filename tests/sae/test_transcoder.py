"""Tests for TranscoderManager logit contribution computation.

Tests the SAE-Steering inspired feature recall pipeline (arXiv:2601.03595):
- Stage 1: Filter features by keyword logit contribution
- Stage 2: Create steering vectors from top candidates

Uses mocks to avoid requiring actual GPU/SAE loading.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import fields

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed"
)


# =============================================================================
# Test Constants
# =============================================================================

D_MODEL = 768  # Model dimension
D_SAE = 1000  # SAE feature count (small for testing)
VOCAB_SIZE = 5000  # Vocabulary size
TEST_LAYER = 15


# =============================================================================
# StrategyFeatureCandidate Tests
# =============================================================================

class TestStrategyFeatureCandidate:
    """Tests for StrategyFeatureCandidate dataclass."""

    def test_dataclass_creation(self):
        """StrategyFeatureCandidate should store all required fields."""
        from core.sae.transcoder import StrategyFeatureCandidate

        candidate = StrategyFeatureCandidate(
            index=42,
            logit_contribution=0.85,
            top_tokens=[100, 200, 300],
            keyword_hits=2,
            neuronpedia_id="qwen3-8b-test/layer_15/42",
        )

        assert candidate.index == 42
        assert candidate.logit_contribution == 0.85
        assert candidate.top_tokens == [100, 200, 300]
        assert candidate.keyword_hits == 2
        assert candidate.neuronpedia_id == "qwen3-8b-test/layer_15/42"

    def test_dataclass_has_expected_fields(self):
        """StrategyFeatureCandidate should have exactly the expected fields."""
        from core.sae.transcoder import StrategyFeatureCandidate

        field_names = {f.name for f in fields(StrategyFeatureCandidate)}
        expected = {"index", "logit_contribution", "top_tokens", "keyword_hits", "neuronpedia_id"}

        assert field_names == expected

    def test_candidates_are_sortable_by_contribution(self):
        """StrategyFeatureCandidates should be sortable by logit_contribution."""
        from core.sae.transcoder import StrategyFeatureCandidate

        candidates = [
            StrategyFeatureCandidate(0, 0.5, [], 1, "a"),
            StrategyFeatureCandidate(1, 0.9, [], 2, "b"),
            StrategyFeatureCandidate(2, 0.3, [], 1, "c"),
        ]

        sorted_candidates = sorted(candidates, key=lambda c: c.logit_contribution, reverse=True)

        assert sorted_candidates[0].index == 1
        assert sorted_candidates[1].index == 0
        assert sorted_candidates[2].index == 2


# =============================================================================
# Mock SAE Fixture
# =============================================================================

@pytest.fixture
def mock_sae():
    """Create a mock SAE with controlled W_dec."""
    sae = MagicMock()
    # W_dec in SAELens is [d_sae, d_in] (feature_dim x model_dim)
    # Our get_decoder_weights transposes to [d_in, d_sae]
    sae.W_dec = torch.randn(D_SAE, D_MODEL)
    return sae


@pytest.fixture
def transcoder_manager(mock_sae):
    """Create a TranscoderManager with mocked SAE."""
    from core.sae.transcoder import TranscoderManager

    with patch.object(TranscoderManager, 'load'):
        manager = TranscoderManager(layer=TEST_LAYER)
        manager._sae = mock_sae
        manager._d_in = D_MODEL
        manager._d_sae = D_SAE
        return manager


# =============================================================================
# get_decoder_weights Tests
# =============================================================================

class TestGetDecoderWeights:
    """Tests for TranscoderManager.get_decoder_weights."""

    def test_returns_transposed_W_dec(self, transcoder_manager, mock_sae):
        """get_decoder_weights should return transposed decoder weights."""
        W_dec = transcoder_manager.get_decoder_weights()

        # Should be transposed from [d_sae, d_in] to [d_in, d_sae]
        assert W_dec.shape == (D_MODEL, D_SAE)

    def test_values_match_sae_transpose(self, transcoder_manager, mock_sae):
        """get_decoder_weights should match SAE W_dec transposed."""
        W_dec = transcoder_manager.get_decoder_weights()
        expected = mock_sae.W_dec.T

        assert torch.allclose(W_dec, expected)

    def test_raises_if_not_loaded(self):
        """get_decoder_weights should raise if SAE not loaded."""
        from core.sae.transcoder import TranscoderManager

        with patch.object(TranscoderManager, 'load'):
            manager = TranscoderManager(layer=TEST_LAYER)
            # _sae is None

            with pytest.raises(RuntimeError, match="not loaded"):
                manager.get_decoder_weights()


# =============================================================================
# compute_logit_contribution_matrix Tests
# =============================================================================

class TestComputeLogitContributionMatrix:
    """Tests for TranscoderManager.compute_logit_contribution_matrix."""

    def test_returns_correct_shape(self, transcoder_manager):
        """compute_logit_contribution_matrix should return [d_sae, vocab_size]."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)

        L = transcoder_manager.compute_logit_contribution_matrix(unembed)

        assert L.shape == (D_SAE, VOCAB_SIZE)

    def test_computes_correct_matmul(self, transcoder_manager, mock_sae):
        """compute_logit_contribution_matrix should compute W_dec.T @ W_U."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)

        L = transcoder_manager.compute_logit_contribution_matrix(unembed)

        # Manual computation: [d_sae, d_in] @ [d_in, vocab] = [d_sae, vocab]
        expected = mock_sae.W_dec @ unembed

        assert torch.allclose(L, expected, atol=1e-5)

    def test_raises_if_not_loaded(self):
        """compute_logit_contribution_matrix should raise if SAE not loaded."""
        from core.sae.transcoder import TranscoderManager

        with patch.object(TranscoderManager, 'load'):
            manager = TranscoderManager(layer=TEST_LAYER)
            unembed = torch.randn(D_MODEL, VOCAB_SIZE)

            with pytest.raises(RuntimeError, match="not loaded"):
                manager.compute_logit_contribution_matrix(unembed)

    def test_no_gradient_tracking(self, transcoder_manager):
        """compute_logit_contribution_matrix should not track gradients."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE, requires_grad=True)

        L = transcoder_manager.compute_logit_contribution_matrix(unembed)

        assert not L.requires_grad


# =============================================================================
# recall_strategy_features Tests
# =============================================================================

class TestRecallStrategyFeatures:
    """Tests for TranscoderManager.recall_strategy_features (Stage 1)."""

    def test_returns_list_of_candidates(self, transcoder_manager):
        """recall_strategy_features should return list of StrategyFeatureCandidate."""
        from core.sae.transcoder import StrategyFeatureCandidate

        unembed = torch.randn(D_MODEL, VOCAB_SIZE)
        keyword_ids = [10, 20, 30]

        candidates = transcoder_manager.recall_strategy_features(
            unembed_matrix=unembed,
            keyword_token_ids=keyword_ids,
            n_keywords_required=1,  # Relaxed for testing
            logit_threshold=0.0,  # No threshold
        )

        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, StrategyFeatureCandidate)

    def test_filters_by_n_keywords_required(self):
        """recall_strategy_features should filter by minimum keyword hits."""
        from core.sae.transcoder import TranscoderManager

        # Ensure deterministic behavior for reproducible tests
        torch.manual_seed(0)

        with patch.object(TranscoderManager, 'load'):
            manager = TranscoderManager(layer=TEST_LAYER)

            # Create orthogonal feature directions for clean separation
            # Use one-hot-like vectors in a small subspace to avoid cross-talk
            W_dec = torch.zeros(3, D_MODEL)
            W_dec[0, 0:100] = 1.0  # Feature 0: strong in dimensions 0-99
            W_dec[1, 100:200] = 1.0  # Feature 1: strong in dimensions 100-199
            W_dec[2, 200:300] = 0.001  # Feature 2: very weak everywhere

            mock_sae = MagicMock()
            mock_sae.W_dec = W_dec
            manager._sae = mock_sae
            manager._d_in = D_MODEL
            manager._d_sae = 3

            # Create unembed that makes specific tokens get high logits
            # L[f, t] = W_dec[f] @ unembed[:, t]
            unembed = torch.zeros(D_MODEL, 100)

            # Token 10: aligned with feature 0 (L[0,10] = 100, L[1,10] = 0)
            unembed[0:100, 10] = 1.0

            # Token 20: also aligned with feature 0 (L[0,20] = 100, L[1,20] = 0)
            unembed[0:100, 20] = 1.0

            # Token 30: aligned with feature 1 only (L[0,30] = 0, L[1,30] = 100)
            unembed[100:200, 30] = 1.0

            keyword_ids = [10, 20]

            # Require 2 keywords - only feature 0 should qualify
            candidates = manager.recall_strategy_features(
                unembed_matrix=unembed,
                keyword_token_ids=keyword_ids,
                n_keywords_required=2,
                logit_threshold=0.1,
                top_k_tokens=10,
            )

            # Only feature 0 should qualify (has both keywords in top tokens)
            feature_indices = [c.index for c in candidates]
            assert feature_indices == [0], (
                f"Expected only feature 0, got {feature_indices}. "
                "Feature 0 should hit both keywords 10 and 20; "
                "Feature 1 should hit neither (orthogonal); "
                "Feature 2 should be too weak."
            )

    def test_sorted_by_logit_contribution_descending(self, transcoder_manager):
        """recall_strategy_features should sort candidates by contribution."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)
        keyword_ids = list(range(50))  # Many keywords to increase hits

        candidates = transcoder_manager.recall_strategy_features(
            unembed_matrix=unembed,
            keyword_token_ids=keyword_ids,
            n_keywords_required=1,
            logit_threshold=-1.0,  # Allow any
            top_k_tokens=50,
        )

        if len(candidates) >= 2:
            # Check descending order
            for i in range(len(candidates) - 1):
                assert candidates[i].logit_contribution >= candidates[i + 1].logit_contribution

    def test_candidate_has_correct_neuronpedia_id(self, transcoder_manager):
        """recall_strategy_features should set correct neuronpedia_id."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)
        keyword_ids = list(range(100))

        candidates = transcoder_manager.recall_strategy_features(
            unembed_matrix=unembed,
            keyword_token_ids=keyword_ids,
            n_keywords_required=1,
            logit_threshold=-1.0,
            top_k_tokens=100,
        )

        if candidates:
            # Should match format "neuronpedia_base/feature_idx"
            first = candidates[0]
            expected_id = f"{transcoder_manager.neuronpedia_id}/{first.index}"
            assert first.neuronpedia_id == expected_id

    def test_raises_if_not_loaded(self):
        """recall_strategy_features should raise if SAE not loaded."""
        from core.sae.transcoder import TranscoderManager

        with patch.object(TranscoderManager, 'load'):
            manager = TranscoderManager(layer=TEST_LAYER)
            unembed = torch.randn(D_MODEL, VOCAB_SIZE)

            with pytest.raises(RuntimeError, match="not loaded"):
                manager.recall_strategy_features(unembed, keyword_token_ids=[1, 2, 3])


# =============================================================================
# get_strategy_steering_vectors Tests
# =============================================================================

class TestGetStrategySteeringVectors:
    """Tests for TranscoderManager.get_strategy_steering_vectors."""

    @pytest.fixture
    def candidates(self):
        """Create test candidates."""
        from core.sae.transcoder import StrategyFeatureCandidate

        return [
            StrategyFeatureCandidate(0, 0.9, [1, 2, 3], 3, "test/0"),
            StrategyFeatureCandidate(1, 0.7, [4, 5, 6], 2, "test/1"),
            StrategyFeatureCandidate(2, 0.5, [7, 8, 9], 2, "test/2"),
        ]

    def test_returns_tensor(self, transcoder_manager, candidates):
        """get_strategy_steering_vectors should return a tensor."""
        with patch.object(
            transcoder_manager, 'get_feature_steering_vector',
            return_value=torch.randn(D_MODEL)
        ):
            vector = transcoder_manager.get_strategy_steering_vectors(
                candidates, top_k=2
            )

            assert isinstance(vector, torch.Tensor)

    def test_top1_mode_uses_single_feature(self, transcoder_manager, candidates):
        """top1 mode should only use the best feature."""
        with patch.object(
            transcoder_manager, 'get_feature_steering_vector',
            return_value=torch.randn(D_MODEL)
        ) as mock_get:
            transcoder_manager.get_strategy_steering_vectors(
                candidates, top_k=5, combine_mode="top1"
            )

            # Should call with just the top candidate's index
            mock_get.assert_called_once_with([0])

    def test_mean_mode_equal_weights(self, transcoder_manager, candidates):
        """mean mode should use equal weights."""
        with patch.object(
            transcoder_manager, 'get_feature_steering_vector',
            return_value=torch.randn(D_MODEL)
        ) as mock_get:
            transcoder_manager.get_strategy_steering_vectors(
                candidates, top_k=3, combine_mode="mean"
            )

            call_args = mock_get.call_args
            indices, weights = call_args[0]

            assert indices == [0, 1, 2]
            # Equal weights for 3 candidates
            expected_weight = 1.0 / 3
            for w in weights:
                assert abs(w - expected_weight) < 1e-6

    def test_weighted_mode_proportional_weights(self, transcoder_manager, candidates):
        """weighted mode should weight by logit_contribution."""
        with patch.object(
            transcoder_manager, 'get_feature_steering_vector',
            return_value=torch.randn(D_MODEL)
        ) as mock_get:
            transcoder_manager.get_strategy_steering_vectors(
                candidates, top_k=3, combine_mode="weighted"
            )

            call_args = mock_get.call_args
            indices, weights = call_args[0]

            assert indices == [0, 1, 2]

            # Weights should be proportional to contributions: 0.9, 0.7, 0.5
            total = 0.9 + 0.7 + 0.5
            expected_weights = [0.9 / total, 0.7 / total, 0.5 / total]
            for w, exp in zip(weights, expected_weights):
                assert abs(w - exp) < 1e-6

    def test_top_k_limits_candidates(self, transcoder_manager, candidates):
        """top_k should limit the number of candidates used."""
        with patch.object(
            transcoder_manager, 'get_feature_steering_vector',
            return_value=torch.randn(D_MODEL)
        ) as mock_get:
            transcoder_manager.get_strategy_steering_vectors(
                candidates, top_k=2, combine_mode="mean"
            )

            call_args = mock_get.call_args
            indices, weights = call_args[0]

            # Should only use top 2
            assert len(indices) == 2
            assert len(weights) == 2

    def test_raises_on_empty_candidates(self, transcoder_manager):
        """get_strategy_steering_vectors should raise on empty candidates."""
        with pytest.raises(ValueError, match="cannot be empty"):
            transcoder_manager.get_strategy_steering_vectors([])

    def test_raises_if_not_loaded(self):
        """get_strategy_steering_vectors should raise if SAE not loaded."""
        from core.sae.transcoder import TranscoderManager, StrategyFeatureCandidate

        with patch.object(TranscoderManager, 'load'):
            manager = TranscoderManager(layer=TEST_LAYER)

            candidates = [StrategyFeatureCandidate(0, 0.5, [], 1, "test")]

            with pytest.raises(RuntimeError, match="not loaded"):
                manager.get_strategy_steering_vectors(candidates)


# =============================================================================
# Integration-style Tests
# =============================================================================

class TestLogitContributionPipeline:
    """Integration tests for the full SAE-Steering pipeline."""

    def test_full_recall_pipeline(self, transcoder_manager):
        """Test full pipeline from unembed to steering vector."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)
        keyword_ids = list(range(200))  # Lots of keywords

        # Stage 1: Recall features
        candidates = transcoder_manager.recall_strategy_features(
            unembed_matrix=unembed,
            keyword_token_ids=keyword_ids,
            n_keywords_required=1,
            logit_threshold=-2.0,  # Permissive for random data
            top_k_tokens=200,
        )

        if candidates:
            # Stage 2: Create steering vector
            with patch.object(
                transcoder_manager, 'get_feature_steering_vector',
                return_value=torch.randn(D_MODEL)
            ):
                vector = transcoder_manager.get_strategy_steering_vectors(
                    candidates, top_k=5, combine_mode="weighted"
                )

                assert vector.shape == (D_MODEL,)

    def test_logit_matrix_is_deterministic(self, transcoder_manager):
        """Same inputs should produce same logit contribution matrix."""
        unembed = torch.randn(D_MODEL, VOCAB_SIZE)

        L1 = transcoder_manager.compute_logit_contribution_matrix(unembed)
        L2 = transcoder_manager.compute_logit_contribution_matrix(unembed)

        assert torch.allclose(L1, L2)
