"""SAE Feature Interpreter using logit lens projection.

This module provides interpretation of SAE features by projecting their
decoder directions through the model's unembedding matrix to find which
vocabulary tokens each feature most strongly predicts.

This is an alternative to Neuronpedia when external interpretations aren't available.

Usage:
    from core.sae.feature_interpreter import FeatureInterpreter

    interpreter = FeatureInterpreter(transcoder, hooked_model)
    await interpreter.initialize()

    # Get interpretation for a feature
    interpretation = interpreter.get_feature_interpretation(49431)
    # Returns: "mathematical, equation, formula" or similar token-based label
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Try imports
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None

if TYPE_CHECKING:
    from core.sae.transcoder import TranscoderManager
    from core.model.hooked_qwen import HookedQwen

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "lilly" / "sae_interpretations"
CACHE_FILE = CACHE_DIR / "qwen3_8b_layer16_logit_lens.json"


@dataclass
class FeatureInterpretation:
    """Interpretation of an SAE feature based on logit lens."""

    index: int
    top_tokens: list[str]  # Top vocabulary tokens this feature predicts
    interpretation: str  # Human-readable summary
    confidence: float  # How concentrated the feature is (higher = more interpretable)


class FeatureInterpreter:
    """Interprets SAE features using logit lens projection.

    Projects SAE decoder weights through the model's unembedding matrix
    to find which vocabulary tokens each feature most strongly predicts.

    Computes projections on-demand for specific features rather than
    precomputing all (which would require ~93GB for Qwen3-8B).

    Attributes:
        transcoder: The SAE transcoder manager
        model: The hooked model (for accessing unembedding matrix)
        _W_dec: Cached decoder weights
        _W_U: Cached unembedding matrix
    """

    def __init__(
        self,
        transcoder: "TranscoderManager",
        model: "HookedQwen",
    ):
        """Initialize interpreter.

        Args:
            transcoder: SAE transcoder manager (must be loaded)
            model: Hooked model (must be loaded)
        """
        self.transcoder = transcoder
        self.model = model
        self._W_dec: Optional["torch.Tensor"] = None  # [d_sae, d_model]
        self._W_U: Optional["torch.Tensor"] = None     # [d_model, vocab_size]
        self._interpretation_cache: dict[int, FeatureInterpretation] = {}
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if ready for interpretation."""
        return self._initialized

    async def initialize(self) -> None:
        """Cache weight matrices for on-demand projection.

        Results are cached to disk.
        """
        if self._initialized:
            return

        # Try to load from cache first
        if self._load_cache():
            self._initialized = True
            logger.info(f"Loaded {len(self._interpretation_cache)} cached feature interpretations")
            return

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for feature interpretation")

        if not self.transcoder.is_loaded:
            raise RuntimeError("Transcoder must be loaded before initialization")

        if not self.model.is_loaded:
            raise RuntimeError("Model must be loaded before initialization")

        logger.info("Initializing on-demand logit lens interpreter...")

        # Cache weight matrices (don't compute full projection - too large!)
        # W_dec: [163840, 4096], W_U: [4096, 151936]
        # Full projection would be [163840, 151936] = 93GB!
        # Instead, compute per-feature: [4096] @ [4096, 151936] = [151936] = 600KB
        self._W_dec = self.transcoder._sae.W_dec.detach().cpu().float()
        self._W_U = self.model._model.W_U.detach().cpu().float()

        self._initialized = True
        logger.info("Feature interpreter ready for on-demand projections")

    def _compute_feature_projection(self, feature_idx: int) -> Optional["torch.Tensor"]:
        """Compute logit lens projection for a single feature.

        Args:
            feature_idx: SAE feature index

        Returns:
            Projection over vocabulary [vocab_size] or None if not initialized
        """
        if self._W_dec is None or self._W_U is None:
            return None

        with torch.no_grad():
            # Get this feature's decoder direction: [d_model]
            feature_direction = self._W_dec[feature_idx]
            # Project through unembedding: [d_model] @ [d_model, vocab_size] = [vocab_size]
            projection = feature_direction @ self._W_U
            return projection

    def _compute_interpretation(
        self,
        feature_idx: int,
        tokenizer,
        top_k_tokens: int = 20,
    ) -> Optional[FeatureInterpretation]:
        """Compute interpretation for a single feature using on-demand projection.

        Args:
            feature_idx: SAE feature index
            tokenizer: Model tokenizer for decoding
            top_k_tokens: Number of top tokens to examine (increased to find ASCII tokens)
        """
        # Compute projection for this specific feature
        feature_proj = self._compute_feature_projection(feature_idx)
        if feature_proj is None:
            return None

        # Get top-k tokens (look at more to find ASCII tokens in multilingual models)
        top_values, top_indices = feature_proj.topk(top_k_tokens)

        # Convert to token strings
        top_tokens = []
        for idx in top_indices.tolist():
            try:
                token = tokenizer.decode([idx]).strip()
                if token:  # Skip empty tokens
                    top_tokens.append(token)
            except Exception:
                continue

        if not top_tokens:
            return None

        # Debug: log the raw tokens found
        logger.debug(f"Feature {feature_idx} top tokens: {top_tokens[:10]}")

        # Compute confidence (how peaked the distribution is)
        # Higher std in top-k projections = more selective feature
        confidence = float(top_values.std() / (top_values.mean() + 1e-6))
        confidence = min(1.0, confidence)  # Cap at 1.0

        # Create human-readable interpretation
        # Prefer longer ASCII alphabetic tokens (more likely to be complete words)
        # Chinese characters pass isalpha() but aren't ideal for English TTS
        ascii_tokens = [t for t in top_tokens if len(t) > 1 and t.isascii() and t.isalpha()]
        # Sort by length (prefer longer, more word-like tokens)
        ascii_tokens.sort(key=len, reverse=True)
        # Filter for word-like tokens (4+ chars) if available, else use any ASCII
        word_like = [t for t in ascii_tokens if len(t) >= 4]
        non_ascii_alpha = [t for t in top_tokens if len(t) > 1 and t.isalpha() and not t.isascii()]

        if word_like:
            # Prefer longer ASCII tokens that look like words
            interpretation = ", ".join(word_like[:3])
        elif ascii_tokens:
            # Fall back to shorter ASCII tokens
            interpretation = ", ".join(ascii_tokens[:3])
        elif non_ascii_alpha:
            # Fall back to non-ASCII alphabetic (Chinese, etc.) if no ASCII available
            interpretation = ", ".join(non_ascii_alpha[:3])
        elif top_tokens:
            # Use any available tokens (numbers, symbols, etc.) rather than generic fallback
            # Filter out pure whitespace and very short tokens
            usable_tokens = [t for t in top_tokens if t.strip() and len(t.strip()) > 0]
            if usable_tokens:
                interpretation = ", ".join(usable_tokens[:3])
            else:
                interpretation = f"tokens: {', '.join(top_tokens[:3])}"
        else:
            interpretation = f"feature {feature_idx}"

        return FeatureInterpretation(
            index=feature_idx,
            top_tokens=top_tokens,
            interpretation=interpretation,
            confidence=confidence,
        )

    def get_feature_interpretation(self, feature_idx: int) -> str:
        """Get interpretation for a feature index.

        Args:
            feature_idx: SAE feature index

        Returns:
            Human-readable interpretation string
        """
        # Check cache first
        if feature_idx in self._interpretation_cache:
            return self._interpretation_cache[feature_idx].interpretation

        # Compute on-demand if initialized with weight matrices
        if self._W_dec is not None and self._W_U is not None and self.model.is_loaded:
            tokenizer = self.model._model.tokenizer
            interp = self._compute_interpretation(feature_idx, tokenizer)
            if interp:
                self._interpretation_cache[feature_idx] = interp
                logger.debug(f"Feature {feature_idx}: computed '{interp.interpretation}' from tokens {interp.top_tokens}")
                return interp.interpretation
            else:
                logger.debug(f"Feature {feature_idx}: _compute_interpretation returned None")
        elif self._W_dec is None or self._W_U is None:
            logger.debug(f"Feature {feature_idx}: weight matrices not initialized")
        elif not self.model.is_loaded:
            logger.debug(f"Feature {feature_idx}: model not loaded (tokenizer unavailable)")

        # Fallback
        return f"feature {feature_idx}"

    def get_feature_tokens(self, feature_idx: int, top_k: int = 5) -> list[str]:
        """Get top tokens for a feature.

        Args:
            feature_idx: SAE feature index
            top_k: Number of tokens to return

        Returns:
            List of top predicted tokens
        """
        if feature_idx in self._interpretation_cache:
            return self._interpretation_cache[feature_idx].top_tokens[:top_k]

        if self._W_dec is not None and self._W_U is not None and self.model.is_loaded:
            tokenizer = self.model._model.tokenizer
            interp = self._compute_interpretation(feature_idx, tokenizer, top_k)
            if interp:
                return interp.top_tokens

        return []

    def batch_get_interpretations(self, feature_indices: list[int]) -> dict[int, str]:
        """Get interpretations for multiple features.

        Args:
            feature_indices: List of feature indices

        Returns:
            Dict mapping feature index to interpretation
        """
        results = {}
        for idx in feature_indices:
            results[idx] = self.get_feature_interpretation(idx)
        return results

    def _load_cache(self) -> bool:
        """Load interpretations from disk cache."""
        if not CACHE_FILE.exists():
            return False

        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)

            for idx_str, interp_data in data.items():
                idx = int(idx_str)
                self._interpretation_cache[idx] = FeatureInterpretation(
                    index=idx,
                    top_tokens=interp_data["top_tokens"],
                    interpretation=interp_data["interpretation"],
                    confidence=interp_data.get("confidence", 0.5),
                )

            return True
        except Exception as e:
            logger.warning(f"Failed to load interpretation cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save interpretations to disk cache."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            data = {}
            for idx, interp in self._interpretation_cache.items():
                data[str(idx)] = {
                    "top_tokens": interp.top_tokens,
                    "interpretation": interp.interpretation,
                    "confidence": interp.confidence,
                }

            with open(CACHE_FILE, "w") as f:
                json.dump(data, f)

            logger.info(f"Saved {len(data)} feature interpretations to cache")
        except Exception as e:
            logger.warning(f"Failed to save interpretation cache: {e}")

    def clear_cache(self) -> None:
        """Clear the interpretation cache."""
        self._interpretation_cache = {}
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()


# Singleton instance
_interpreter: Optional[FeatureInterpreter] = None


def get_feature_interpreter(
    transcoder: "TranscoderManager",
    model: "HookedQwen",
) -> FeatureInterpreter:
    """Get or create the singleton FeatureInterpreter."""
    global _interpreter

    if _interpreter is None:
        _interpreter = FeatureInterpreter(transcoder, model)

    return _interpreter
