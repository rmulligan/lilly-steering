"""SAE Transcoder for Qwen3-8B monosemantic feature extraction.

This module provides sparse autoencoder (SAE) transcoder functionality
for extracting monosemantic features from Qwen3-8B's MLP activations.
Unlike logit lens (polysemantic), SAE features represent distinct,
interpretable concepts.

The transcoder intercepts MLP computation:
    mlp_input -> sparse_features -> mlp_output
    (d_model)    (d_sae=163840)    (d_model)

Usage:
    from core.sae import get_transcoder_manager

    # Get singleton manager
    manager = get_transcoder_manager()

    # Load transcoder (lazy, cached)
    await manager.load()

    # Encode MLP activations to sparse features
    features = manager.encode(mlp_activations)  # [batch, seq, d_sae]

    # Get top-k active features with indices and activations
    active = manager.get_active_features(features, top_k=50)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Try to import SAELens
try:
    from sae_lens import SAE
    SAELENS_AVAILABLE = True
except ImportError:
    SAELENS_AVAILABLE = False
    logger.warning("sae_lens not available - transcoder features disabled")

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Configuration constants
DEFAULT_LAYER = 16  # Layer for transcoder (matches steering layer)
TRANSCODER_RELEASE = "mwhanna-qwen3-8b-transcoders"
NEURONPEDIA_ID_TEMPLATE = "qwen3-8b/{layer}-transcoder-hp"

# Singleton instance
_transcoder_manager: Optional["TranscoderManager"] = None


@dataclass
class ActiveFeature:
    """A single active SAE feature."""

    index: int  # Feature index in the SAE
    activation: float  # Activation strength
    neuronpedia_id: str  # ID for looking up interpretation


@dataclass
class FeatureCoactivation:
    """Record of two features firing together."""

    feature_a: int
    feature_b: int
    strength: float  # Product or min of activations
    context: str  # Brief description of when they coactivated


@dataclass
class StrategyFeatureCandidate:
    """A candidate feature for strategy control, identified via logit contribution.

    Based on SAE-Steering (arXiv:2601.03595): features are recalled by their
    ability to amplify strategy-specific keywords in the output vocabulary.

    Attributes:
        index: Feature index in the SAE
        logit_contribution: Mean logit contribution to target keywords
        top_tokens: Token indices with highest logit contribution from this feature
        keyword_hits: How many target keywords appear in top_tokens
        neuronpedia_id: ID for looking up interpretation
    """

    index: int
    logit_contribution: float
    top_tokens: list[int]
    keyword_hits: int
    neuronpedia_id: str


class TranscoderManager:
    """Manages SAE transcoder for Qwen3-8B feature extraction.

    The transcoder provides monosemantic features by intercepting MLP
    computation. Each feature represents a distinct, interpretable concept
    (unlike polysemantic residual stream neurons).

    Features:
    - Lazy loading: Transcoder only loaded when first needed
    - Device flexibility: Can run on CPU or GPU
    - Cached: Single instance shared across cognitive loops
    - Neuronpedia integration: Feature indices map to interpretations

    Attributes:
        layer: Which transformer layer's MLP to intercept
        device: Device for transcoder computation
        sae: The loaded SAE transcoder (None until load() called)
    """

    def __init__(
        self,
        layer: int = DEFAULT_LAYER,
        device: str = "cpu",
    ):
        """Initialize transcoder manager.

        Args:
            layer: Transformer layer to intercept (default: 16)
            device: Device for computation ('cpu' or 'cuda')
        """
        self.layer = layer
        self.device = device
        self._sae: Optional["SAE"] = None
        self._load_lock = asyncio.Lock()
        self._d_in: int = 0  # Set after loading
        self._d_sae: int = 0  # Set after loading

    @property
    def is_loaded(self) -> bool:
        """Check if transcoder is loaded."""
        return self._sae is not None

    @property
    def d_in(self) -> int:
        """Input dimension (d_model)."""
        return self._d_in

    @property
    def d_sae(self) -> int:
        """SAE feature dimension (sparse features count)."""
        return self._d_sae

    @property
    def neuronpedia_id(self) -> str:
        """Neuronpedia ID for this layer's features."""
        return NEURONPEDIA_ID_TEMPLATE.format(layer=self.layer)

    async def load(self) -> None:
        """Load the transcoder from SAELens pretrained weights.

        This is async-safe and will only load once even if called
        concurrently from multiple coroutines.

        Raises:
            RuntimeError: If SAELens is not available
        """
        if self._sae is not None:
            return  # Already loaded

        async with self._load_lock:
            # Double-check after acquiring lock
            if self._sae is not None:
                return

            if not SAELENS_AVAILABLE:
                raise RuntimeError(
                    "sae_lens not installed. Run: uv pip install sae-lens"
                )

            logger.info(
                f"Loading Qwen3-8B transcoder for layer {self.layer} on {self.device}"
            )

            loop = asyncio.get_running_loop()

            def _load():
                sae = SAE.from_pretrained(
                    release=TRANSCODER_RELEASE,
                    sae_id=f"layer_{self.layer}",
                    device=self.device,
                )
                return sae

            self._sae = await loop.run_in_executor(None, _load)
            self._d_in = self._sae.cfg.d_in
            self._d_sae = self._sae.cfg.d_sae

            logger.info(
                f"Transcoder loaded: d_in={self._d_in}, d_sae={self._d_sae}, "
                f"neuronpedia={self.neuronpedia_id}"
            )

    def unload(self) -> None:
        """Unload the transcoder to free memory."""
        if self._sae is not None:
            del self._sae
            self._sae = None
            self._d_in = 0
            self._d_sae = 0
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            logger.info("Transcoder unloaded")

    def encode(self, mlp_input: "torch.Tensor") -> "torch.Tensor":
        """Encode MLP input activations to sparse SAE features.

        The transcoder's encoder projects d_model activations to a
        sparse d_sae dimensional space where each dimension represents
        a monosemantic feature.

        Args:
            mlp_input: MLP input activations [batch, seq, d_model] or [seq, d_model]

        Returns:
            Sparse feature activations [batch, seq, d_sae] or [seq, d_sae]

        Raises:
            RuntimeError: If transcoder not loaded
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        # Ensure 3D input [batch, seq, d_model]
        original_shape = mlp_input.shape
        if mlp_input.dim() == 2:
            mlp_input = mlp_input.unsqueeze(0)

        batch, seq, d_model = mlp_input.shape

        # Flatten for encoding: [batch * seq, d_model]
        flat_input = mlp_input.reshape(-1, d_model)

        # Encode to sparse features
        with torch.no_grad():
            # SAE encode returns feature activations (post-ReLU)
            features = self._sae.encode(flat_input)

        # Reshape back: [batch, seq, d_sae]
        features = features.reshape(batch, seq, -1)

        # Restore original batch dimension if input was 2D
        if len(original_shape) == 2:
            features = features.squeeze(0)

        return features

    def decode(self, features: "torch.Tensor") -> "torch.Tensor":
        """Decode sparse features back to MLP output space.

        This reconstructs what the MLP would have produced, allowing
        for feature steering by modifying the sparse representation.

        Args:
            features: Sparse feature activations [batch, seq, d_sae] or [seq, d_sae]

        Returns:
            Reconstructed MLP output [batch, seq, d_model] or [seq, d_model]

        Raises:
            RuntimeError: If transcoder not loaded
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        original_shape = features.shape
        if features.dim() == 2:
            features = features.unsqueeze(0)

        batch, seq, d_sae = features.shape

        # Flatten: [batch * seq, d_sae]
        flat_features = features.reshape(-1, d_sae)

        # Decode back to d_model
        with torch.no_grad():
            reconstructed = self._sae.decode(flat_features)

        # Reshape: [batch, seq, d_model]
        reconstructed = reconstructed.reshape(batch, seq, -1)

        if len(original_shape) == 2:
            reconstructed = reconstructed.squeeze(0)

        return reconstructed

    def get_active_features(
        self,
        features: "torch.Tensor",
        top_k: int = 50,
        threshold: float = 0.0,
    ) -> list[ActiveFeature]:
        """Extract the top-k most active features from sparse representation.

        Args:
            features: Feature activations [batch, seq, d_sae] or [seq, d_sae] or [d_sae]
            top_k: Maximum number of features to return
            threshold: Minimum activation to include (default: 0 = include all non-zero)

        Returns:
            List of ActiveFeature objects, sorted by activation descending
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        # Collapse to 1D by taking mean over batch/seq dimensions
        if features.dim() == 3:
            feat_1d = features.mean(dim=(0, 1))
        elif features.dim() == 2:
            feat_1d = features.mean(dim=0)
        else:
            feat_1d = features

        # Get top-k indices and values
        values, indices = feat_1d.topk(min(top_k, len(feat_1d)))

        active = []
        for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            if val > threshold:
                active.append(
                    ActiveFeature(
                        index=idx,
                        activation=val,
                        neuronpedia_id=f"{self.neuronpedia_id}/{idx}",
                    )
                )

        return active

    def find_coactivations(
        self,
        features_a: "torch.Tensor",
        features_b: "torch.Tensor",
        top_k: int = 20,
        min_strength: float = 0.01,
    ) -> list[tuple[int, int, float]]:
        """Find features that are active in both tensors (coactivation).

        Coactivation indicates conceptual association: when two features
        fire together across different thoughts, they represent related ideas.

        Args:
            features_a: First feature tensor (any shape, will be collapsed)
            features_b: Second feature tensor (any shape, will be collapsed)
            top_k: Maximum coactivations to return
            min_strength: Minimum coactivation strength (geometric mean of activations)

        Returns:
            List of (feature_idx, feature_idx, strength) tuples
        """
        # Collapse both to 1D
        if features_a.dim() > 1:
            a = features_a.mean(dim=tuple(range(features_a.dim() - 1)))
        else:
            a = features_a

        if features_b.dim() > 1:
            b = features_b.mean(dim=tuple(range(features_b.dim() - 1)))
        else:
            b = features_b

        # Find features active in both (geometric mean of activations)
        # This rewards features that are strongly active in both, not just one
        coactivation = torch.sqrt(torch.clamp(a, min=0) * torch.clamp(b, min=0))

        # Get top-k
        values, indices = coactivation.topk(min(top_k, len(coactivation)))

        results = []
        for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            if val >= min_strength:
                results.append((idx, idx, val))  # Same feature in both

        return results

    def get_feature_label(self, feature_index: int) -> Optional[str]:
        """Get a human-readable label for a feature index.

        Currently returns a placeholder based on the Neuronpedia ID.
        In the future, this could fetch interpretations from Neuronpedia.

        Args:
            feature_index: The SAE feature index

        Returns:
            A label string, or None if unavailable
        """
        if feature_index < 0 or (self._d_sae > 0 and feature_index >= self._d_sae):
            return None
        # Return Neuronpedia-style ID as placeholder label
        # Could be enhanced to fetch actual interpretations from Neuronpedia API
        return f"feature_{feature_index}"

    def get_feature_steering_vector(
        self,
        feature_indices: list[int],
        coefficients: Optional[list[float]] = None,
    ) -> "torch.Tensor":
        """Create a steering vector by amplifying specific SAE features.

        This allows steering generation by boosting or suppressing specific
        monosemantic concepts (e.g., boost "autonomy" feature, suppress
        "assistant helpfulness" feature).

        Args:
            feature_indices: Which features to include in steering
            coefficients: Amplification per feature (default: all 1.0)

        Returns:
            Steering vector in d_model space [d_model]

        Raises:
            RuntimeError: If transcoder not loaded
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        if coefficients is None:
            coefficients = [1.0] * len(feature_indices)

        # Create sparse feature vector
        sparse_features = torch.zeros(self._d_sae, device=self.device)
        for idx, coef in zip(feature_indices, coefficients):
            if 0 <= idx < self._d_sae:
                sparse_features[idx] = coef

        # Decode to d_model space (this is the steering direction)
        with torch.no_grad():
            steering = self._sae.decode(sparse_features.unsqueeze(0)).squeeze(0)

        return steering


    def get_decoder_weights(self) -> "torch.Tensor":
        """Get the SAE decoder weight matrix.

        The decoder matrix W_dec has shape [d_model, d_sae], where each column
        represents a learned feature direction. This is used for computing
        logit contributions via W_dec^T @ W_U.

        Returns:
            Decoder weight matrix [d_model, d_sae]

        Raises:
            RuntimeError: If transcoder not loaded
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        # SAELens stores decoder as W_dec with shape [d_sae, d_in]
        # We need [d_in, d_sae] for our convention
        return self._sae.W_dec.T

    def compute_logit_contribution_matrix(
        self,
        unembed_matrix: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute logit contribution of each SAE feature to each vocabulary token.

        Based on SAE-Steering (arXiv:2601.03595): L = W_dec^T @ W_U
        This measures how much each feature direction contributes to each token's
        logit when activated. High contribution to strategy keywords indicates
        potential for steering toward that strategy.

        Args:
            unembed_matrix: Model's unembedding matrix W_U [d_model, vocab_size]

        Returns:
            Logit contribution matrix [d_sae, vocab_size] where L[i,j] is
            feature i's contribution to token j's logit

        Raises:
            RuntimeError: If transcoder not loaded
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")

        # W_dec is [d_in, d_sae], unembed is [d_in, vocab_size]
        # Result: [d_sae, vocab_size]
        W_dec = self.get_decoder_weights()  # [d_in, d_sae]

        with torch.no_grad():
            # Transpose W_dec to [d_sae, d_in], then matmul with unembed [d_in, vocab]
            # Result: [d_sae, vocab_size]
            logit_contrib = W_dec.T @ unembed_matrix

        return logit_contrib

    def recall_strategy_features(
        self,
        unembed_matrix: "torch.Tensor",
        keyword_token_ids: list[int],
        n_keywords_required: int = 2,
        logit_threshold: float = 0.1,
        top_k_tokens: int = 10,
    ) -> list[StrategyFeatureCandidate]:
        """Recall features that amplify strategy-specific keywords.

        Stage 1 of SAE-Steering (arXiv:2601.03595): Efficiently filter the vast
        pool of SAE features (e.g., 163k) to a small candidate set (~100) by
        identifying features whose top logit contributions include target keywords.

        This filters out ~99% of features, leaving only those with potential
        control effectiveness for the target strategy.

        Args:
            unembed_matrix: Model's unembedding matrix W_U [d_model, vocab_size]
            keyword_token_ids: Token IDs of strategy-specific keywords
                (e.g., "wait", "alternatively", "let me check")
            n_keywords_required: Minimum keywords that must appear in top_k_tokens
                (default: 2)
            logit_threshold: Minimum logit contribution for a keyword to count
                (default: 0.1)
            top_k_tokens: Number of top tokens to consider per feature
                (default: 10)

        Returns:
            List of StrategyFeatureCandidate objects for features meeting criteria,
            sorted by logit_contribution descending

        Raises:
            RuntimeError: If transcoder not loaded

        Example:
            ```python
            # Get keywords for "backtracking" strategy
            tokenizer = model.tokenizer
            keywords = ["wait", "back", "reconsider", "actually", "mistake"]
            keyword_ids = [tokenizer.encode(kw)[0] for kw in keywords]

            # Recall candidate features
            candidates = manager.recall_strategy_features(
                unembed_matrix=model.W_U,
                keyword_token_ids=keyword_ids,
                n_keywords_required=2,
                logit_threshold=0.1,
            )
            print(f"Recalled {len(candidates)} candidate features from {manager.d_sae}")
            ```
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")

        # Compute full logit contribution matrix
        L = self.compute_logit_contribution_matrix(unembed_matrix)  # [d_sae, vocab]

        keyword_set = set(keyword_token_ids)
        candidates = []

        # Vectorized feature processing for performance
        # Get top-k tokens for all features at once: [d_sae, top_k_tokens]
        top_values, top_indices = L.topk(top_k_tokens, dim=1)

        # Create tensor of keyword token IDs for vectorized membership check
        keyword_tensor = torch.tensor(list(keyword_set), device=L.device, dtype=torch.long)

        # Check which top tokens are keywords: [d_sae, top_k_tokens]
        is_keyword = torch.isin(top_indices, keyword_tensor)

        # Check which values exceed threshold: [d_sae, top_k_tokens]
        above_threshold = top_values > logit_threshold

        # Valid hits are tokens that are keywords AND above threshold
        valid_hits = is_keyword & above_threshold

        # Count keyword hits per feature: [d_sae]
        keyword_hits_per_feature = valid_hits.sum(dim=1)

        # Mask for features meeting the n_keywords_required criteria
        candidate_mask = keyword_hits_per_feature >= n_keywords_required

        if not torch.any(candidate_mask):
            return []

        # Compute mean contribution for all features (only valid hits contribute)
        contributions = top_values * valid_hits
        sum_contributions = contributions.sum(dim=1)
        mean_contributions = torch.nan_to_num(sum_contributions / keyword_hits_per_feature)

        # Create candidates for features that meet the criteria
        candidate_indices = torch.where(candidate_mask)[0]
        for idx in candidate_indices:
            idx_item = idx.item()
            candidates.append(
                StrategyFeatureCandidate(
                    index=idx_item,
                    logit_contribution=mean_contributions[idx_item].item(),
                    top_tokens=top_indices[idx_item].cpu().tolist(),
                    keyword_hits=keyword_hits_per_feature[idx_item].item(),
                    neuronpedia_id=f"{self.neuronpedia_id}/{idx_item}",
                )
            )

        # Sort by logit contribution descending
        candidates.sort(key=lambda c: c.logit_contribution, reverse=True)

        return candidates

    def get_strategy_steering_vectors(
        self,
        candidates: list[StrategyFeatureCandidate],
        top_k: int = 5,
        combine_mode: Literal["mean", "weighted", "top1"] = "mean",
    ) -> "torch.Tensor":
        """Create steering vector from top strategy feature candidates.

        Combines multiple strategy-specific features into a single steering
        direction. This can be used with HookedQwen.set_steering_vector()
        to guide generation toward a particular reasoning strategy.

        Args:
            candidates: List of StrategyFeatureCandidate from recall_strategy_features
            top_k: Number of top candidates to combine (default: 5)
            combine_mode: How to combine features:
                - "mean": Average of feature directions (default)
                - "weighted": Weight by logit_contribution
                - "top1": Use only the top feature

        Returns:
            Combined steering vector [d_model]

        Raises:
            RuntimeError: If transcoder not loaded
            ValueError: If candidates list is empty
        """
        if self._sae is None:
            raise RuntimeError("Transcoder not loaded. Call load() first.")

        if not candidates:
            raise ValueError("candidates list cannot be empty")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not available")

        # Take top-k candidates
        selected = candidates[:top_k]

        if combine_mode == "top1":
            # Just use the single best feature
            return self.get_feature_steering_vector([selected[0].index])

        # Get feature indices and weights
        indices = [c.index for c in selected]
        if combine_mode == "weighted":
            total_contrib = sum(c.logit_contribution for c in selected)
            if total_contrib > 0:
                weights = [c.logit_contribution / total_contrib for c in selected]
            else:
                # Fallback to mean if total contribution is zero to avoid division by zero
                weights = [1.0 / len(selected)] * len(selected)
        elif combine_mode == "mean":
            weights = [1.0 / len(selected)] * len(selected)
        else:
            raise ValueError(f"Invalid combine_mode: {combine_mode}. Must be 'top1', 'weighted', or 'mean'.")

        # Create combined steering vector
        return self.get_feature_steering_vector(indices, weights)


def get_transcoder_manager(
    layer: int = DEFAULT_LAYER,
    device: str = "cpu",
) -> TranscoderManager:
    """Get or create the singleton TranscoderManager.

    This ensures only one transcoder is loaded, saving memory.

    Args:
        layer: Layer to use (only applies on first call)
        device: Device to use (only applies on first call)

    Returns:
        The singleton TranscoderManager instance
    """
    global _transcoder_manager

    if _transcoder_manager is None:
        _transcoder_manager = TranscoderManager(layer=layer, device=device)

    return _transcoder_manager


async def extract_sae_concepts(
    mlp_activations: "torch.Tensor",
    manager: Optional[TranscoderManager] = None,
    top_k: int = 30,
) -> list[ActiveFeature]:
    """High-level function to extract concepts from MLP activations.

    This is the main entry point for the cognitive loop to get
    monosemantic features from a thought's activations.

    Args:
        mlp_activations: MLP input activations from the model
        manager: TranscoderManager to use (uses singleton if None)
        top_k: Number of top features to return

    Returns:
        List of ActiveFeature objects representing active concepts
    """
    if manager is None:
        manager = get_transcoder_manager()

    if not manager.is_loaded:
        await manager.load()

    # Encode to sparse features
    features = manager.encode(mlp_activations)

    # Extract top active features
    active = manager.get_active_features(features, top_k=top_k)

    return active
