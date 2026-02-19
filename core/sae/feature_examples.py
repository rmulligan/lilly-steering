"""Pre-computed SAE feature examples from mwhanna transcoders.

This module loads feature interpretability data from the mwhanna/qwen3-8b-transcoders
repository, which contains pre-computed activation examples for each SAE feature.

Each feature has examples showing:
- Which text contexts activate the feature most strongly
- Token-level activation values
- Examples from different quantiles (top activations, samples, bottom)

This provides ground-truth interpretability data without requiring on-the-fly
logit lens computation.

Usage:
    from core.sae.feature_examples import FeatureExamplesLoader

    loader = FeatureExamplesLoader("/path/to/qwen3-8b-transcoders")
    await loader.load_index()

    examples = loader.get_feature_examples(10000, layer=16)
    print(examples.top_contexts)  # Text contexts that activate this feature
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Stopwords to filter out from interpretations (grammatical tokens, not semantic)
# Includes common function words and contracted forms
STOPWORDS = frozenset({
    # Articles and determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Pronouns
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "i", "my", "me", "he", "she", "him", "her", "his",
    # Copula and auxiliaries
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    # Modals
    "will", "would", "could", "should", "may", "might", "must", "can",
    # Prepositions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    # Conjunctions
    "and", "or", "but", "if", "then", "than", "so",
    # Question words
    "what", "when", "where", "who", "which", "how", "why",
    # Quantifiers
    "all", "each", "more", "most", "other", "some", "such", "no", "not",
    "only", "also", "just", "even", "still", "already", "much",
    # Contractions (various forms that might appear)
    "'s", "s", "'re", "re", "'ve", "ve", "'ll", "ll", "'d", "d",
})


@dataclass
class ActivationExample:
    """A single example showing feature activation in context.

    Attributes:
        tokens: List of tokens in the example context
        activations: Activation value at each token position
        focus_token_idx: The token position to focus on (train_token_ind)
        max_activation: The highest activation value in this example
        max_token: The token with highest activation
    """

    tokens: list[str]
    activations: list[float]
    focus_token_idx: int

    @property
    def max_activation(self) -> float:
        """Get the maximum activation value."""
        return max(self.activations) if self.activations else 0.0

    @property
    def max_token(self) -> str:
        """Get the token with maximum activation."""
        if not self.activations or not self.tokens:
            return ""
        max_idx = self.activations.index(max(self.activations))
        return self.tokens[max_idx] if max_idx < len(self.tokens) else ""

    @property
    def context_text(self) -> str:
        """Get the full context as a string."""
        return "".join(self.tokens)

    @property
    def focus_token(self) -> str:
        """Get the focus token."""
        if 0 <= self.focus_token_idx < len(self.tokens):
            return self.tokens[self.focus_token_idx]
        return ""


@dataclass
class FeatureExamples:
    """Pre-computed examples for an SAE feature.

    Attributes:
        transcoder_id: Identifier like "Qwen3-8b-relu-16"
        index: Feature index (0-163839)
        top_examples: Examples with highest activations
        bottom_examples: Examples with lowest activations
        sample_examples: Representative samples from the distribution
    """

    transcoder_id: str
    index: int
    top_examples: list[ActivationExample] = field(default_factory=list)
    bottom_examples: list[ActivationExample] = field(default_factory=list)
    sample_examples: list[ActivationExample] = field(default_factory=list)

    @property
    def top_contexts(self) -> list[str]:
        """Get context strings from top examples."""
        return [ex.context_text for ex in self.top_examples]

    @property
    def top_tokens(self) -> list[str]:
        """Get the focus tokens from top examples (most activated tokens)."""
        return [ex.max_token for ex in self.top_examples if ex.max_token]

    @property
    def interpretation_hint(self) -> str:
        """Generate an interpretation hint from top examples.

        Extracts common tokens that appear with high activation across examples.
        Filters out stopwords and grammatical tokens to focus on semantic content.
        """
        # Collect tokens with high activations
        high_act_tokens: list[str] = []
        for ex in self.top_examples[:5]:
            # Get tokens where activation is > 50% of max
            threshold = ex.max_activation * 0.5
            for i, (tok, act) in enumerate(zip(ex.tokens, ex.activations)):
                if act > threshold and tok.strip():
                    # Clean token (remove leading space, newlines, normalize quotes)
                    clean = tok.strip().replace("\u23ce", "")
                    if clean and len(clean) > 1:
                        high_act_tokens.append(clean)

        if not high_act_tokens:
            return f"feature {self.index}"

        # Find most common tokens
        counts = Counter(high_act_tokens)
        common = [tok for tok, _ in counts.most_common(10)]  # Get more candidates

        def is_stopword(token: str) -> bool:
            """Check if token is a stopword, normalizing apostrophes."""
            # Normalize curly quotes to straight for comparison
            normalized = token.lower().replace("'", "'").replace("'", "'")
            return normalized in STOPWORDS

        # Filter: word-like (3+ chars, alphabetic) AND not a stopword
        semantic = [
            t for t in common
            if len(t) >= 3 and t.isalpha() and not is_stopword(t)
        ]
        if semantic:
            return ", ".join(semantic[:3])

        # Fall back to any non-stopword tokens
        non_stop = [t for t in common if not is_stopword(t)]
        if non_stop:
            return ", ".join(non_stop[:3])

        # All tokens are stopwords - this is a grammatical feature
        return f"feature {self.index}"


class FeatureExamplesLoader:
    """Loads pre-computed feature examples from mwhanna transcoder repo.

    The repo structure is:
        features/
            index.json.gz    # Offsets for each feature in layer files
            layer_0.bin      # Compressed feature data for layer 0
            layer_16.bin     # etc.
            ...

    Each feature's data is individually gzipped within the .bin files,
    with a 4-byte length prefix.

    Attributes:
        repo_path: Path to the cloned mwhanna/qwen3-8b-transcoders repo
        _index: Loaded index data (layer -> {filename, offsets})
        _layer_files: Paths to layer binary files
    """

    def __init__(self, repo_path: str | Path):
        """Initialize loader.

        Args:
            repo_path: Path to cloned mwhanna/qwen3-8b-transcoders repo
        """
        self.repo_path = Path(repo_path)
        self._index: dict = {}
        self._layer_files: dict[int, Path] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._loaded

    @property
    def available_layers(self) -> list[int]:
        """Get list of available layer numbers."""
        if not self._loaded:
            return []
        return sorted(int(k) for k in self._index.keys() if k not in ("version", "format"))

    def load_index(self) -> None:
        """Load the feature index from disk.

        This loads the index.json.gz which contains offsets for
        looking up individual features in the binary files.
        """
        if self._loaded:
            return

        index_path = self.repo_path / "features" / "index.json.gz"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Feature index not found at {index_path}. "
                f"Download from: https://huggingface.co/mwhanna/qwen3-8b-transcoders"
            )

        logger.info(f"Loading feature index from {index_path}")
        with gzip.open(index_path, "rt") as f:
            self._index = json.load(f)

        # Map layer numbers to binary file paths
        for key, value in self._index.items():
            if key in ("version", "format"):
                continue
            layer = int(key)
            bin_path = self.repo_path / "features" / value["filename"]
            if bin_path.exists():
                self._layer_files[layer] = bin_path
            else:
                logger.debug(f"Layer {layer} binary not found at {bin_path}")

        self._loaded = True
        logger.info(
            f"Feature index loaded: {len(self._layer_files)} layers available, "
            f"version={self._index.get('version')}"
        )

    def get_feature_examples(
        self,
        feature_idx: int,
        layer: int = 16,
        max_examples: int = 10,
    ) -> Optional[FeatureExamples]:
        """Load examples for a specific feature.

        Args:
            feature_idx: Feature index (0-163839)
            layer: Transformer layer (default: 16)
            max_examples: Maximum examples to load per quantile

        Returns:
            FeatureExamples or None if not available
        """
        if not self._loaded:
            logger.warning("Index not loaded. Call load_index() first.")
            return None

        layer_key = str(layer)
        if layer_key not in self._index:
            logger.warning(f"Layer {layer} not in index")
            return None

        if layer not in self._layer_files:
            logger.warning(f"Layer {layer} binary file not downloaded")
            return None

        layer_data = self._index[layer_key]
        offsets = layer_data["offsets"]

        # Validate feature index
        if feature_idx < 0 or feature_idx >= len(offsets) - 1:
            logger.warning(f"Feature index {feature_idx} out of range (0-{len(offsets)-2})")
            return None

        # Read feature data from binary file
        bin_path = self._layer_files[layer]
        start = offsets[feature_idx]
        end = offsets[feature_idx + 1]

        try:
            with open(bin_path, "rb") as f:
                f.seek(start)
                raw_data = f.read(end - start)

            # Skip 4-byte length prefix and decompress
            compressed = raw_data[4:]
            decompressed = gzip.decompress(compressed)
            feature_data = json.loads(decompressed.decode("utf-8"))

            return self._parse_feature_data(feature_data, max_examples)

        except Exception as e:
            logger.error(f"Failed to load feature {feature_idx} from layer {layer}: {e}")
            return None

    def _parse_feature_data(
        self,
        data: dict,
        max_examples: int,
    ) -> FeatureExamples:
        """Parse raw feature JSON into FeatureExamples.

        Args:
            data: Raw JSON data for a feature
            max_examples: Maximum examples per quantile

        Returns:
            Parsed FeatureExamples
        """
        result = FeatureExamples(
            transcoder_id=data.get("transcoder_id", "unknown"),
            index=data.get("index", -1),
        )

        # Parse examples from each quantile
        for quantile in data.get("examples_quantiles", []):
            quantile_name = quantile.get("quantile_name", "")
            examples = quantile.get("examples", [])[:max_examples]

            parsed_examples = []
            for ex in examples:
                tokens = ex.get("tokens", [])
                activations = ex.get("tokens_acts_list", [])
                focus_idx = ex.get("train_token_ind", 0)

                parsed_examples.append(
                    ActivationExample(
                        tokens=tokens,
                        activations=activations,
                        focus_token_idx=focus_idx,
                    )
                )

            if quantile_name == "Top":
                result.top_examples = parsed_examples
            elif quantile_name == "Bottom":
                result.bottom_examples = parsed_examples
            elif "Subsample" in quantile_name or "interval" in quantile_name.lower():
                result.sample_examples.extend(parsed_examples)

        return result

    def batch_get_features(
        self,
        feature_indices: list[int],
        layer: int = 16,
    ) -> dict[int, FeatureExamples]:
        """Load examples for multiple features.

        Args:
            feature_indices: List of feature indices
            layer: Transformer layer

        Returns:
            Dict mapping feature index to examples (missing features omitted)
        """
        results = {}
        for idx in feature_indices:
            examples = self.get_feature_examples(idx, layer)
            if examples:
                results[idx] = examples
        return results


# Default path for the transcoders repo (can be overridden)
DEFAULT_TRANSCODERS_PATH = Path.home() / "qwen3-8b-transcoders"

# Singleton instance
_examples_loader: Optional[FeatureExamplesLoader] = None


def get_examples_loader(
    repo_path: str | Path | None = None,
) -> FeatureExamplesLoader:
    """Get or create the singleton FeatureExamplesLoader.

    Args:
        repo_path: Optional path to transcoders repo. If None, uses default.

    Returns:
        FeatureExamplesLoader instance
    """
    global _examples_loader

    if _examples_loader is None:
        path = Path(repo_path) if repo_path else DEFAULT_TRANSCODERS_PATH
        _examples_loader = FeatureExamplesLoader(path)

    return _examples_loader
