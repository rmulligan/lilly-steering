"""Polarity detection using logit lens for dialectical exploration.

This module detects semantic opposition using the model's unembedding matrix.
When the cognitive loop stagnates, polarity detection identifies tokens and
directions that oppose the current thought, enabling dialectical progression.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PolarityDetector:
    """Detects semantic opposition using logit lens projection.

    Uses the model's unembedding matrix (W_U) to find tokens that are
    semantically opposed to the current activation direction. This enables
    dialectical exploration by identifying concepts that contradict or
    challenge the current thought.

    Attributes:
        model: The HookedQwen model with access to W_U
        _cache_opposing: Cache of recently computed opposing tokens
    """

    def __init__(self, model: "HookedQwen"):
        """Initialize polarity detector.

        Args:
            model: HookedQwen model with W_U exposed
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch required for PolarityDetector")

        self.model = model
        self._cache_opposing: dict[int, list[tuple[str, float]]] = {}

    @property
    def W_U(self) -> Optional["torch.Tensor"]:
        """Get the unembedding matrix."""
        return self.model.W_U

    @property
    def tokenizer(self):
        """Get the tokenizer."""
        return self.model.tokenizer

    def find_opposing_tokens(
        self,
        activation: "torch.Tensor",
        top_k: int = 20,
        filter_special: bool = True,
    ) -> list[tuple[str, float]]:
        """Find tokens most opposed to current activation direction.

        Projects the activation through W_U to get logits, then returns
        the tokens with the most negative logits (semantically opposed).

        Args:
            activation: Activation vector [d_model] (numpy or torch)
            top_k: Number of opposing tokens to return
            filter_special: Whether to filter out special/punctuation tokens

        Returns:
            List of (token, opposition_strength) tuples, sorted by opposition
        """
        W_U = self.W_U
        if W_U is None:
            logger.warning("W_U not available - model may not be loaded")
            return []

        # Convert numpy to torch if needed
        if hasattr(activation, 'numpy'):
            # Already torch tensor
            act_tensor = activation
        else:
            # Numpy array
            act_tensor = torch.from_numpy(activation)

        # Ensure correct device and dtype
        act_tensor = act_tensor.to(device=W_U.device, dtype=W_U.dtype)

        # Ensure 1D
        if act_tensor.dim() > 1:
            act_tensor = act_tensor.mean(dim=tuple(range(act_tensor.dim() - 1)))

        # Project to logit space: [d_model] @ [d_model, vocab_size] -> [vocab_size]
        with torch.no_grad():
            logits = act_tensor @ W_U

        # Get most negative logits (opposed to current direction)
        values, indices = logits.topk(top_k * 3, largest=False)  # Get extra for filtering

        opposing = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            try:
                token = self.tokenizer.decode([idx])
            except Exception:
                continue

            # Filter criteria
            token_stripped = token.strip()
            if not token_stripped:
                continue
            if len(token_stripped) < 2:
                continue
            if filter_special:
                # Skip punctuation-only tokens
                if all(c in '.,!?;:\'"()[]{}/<>@#$%^&*-_=+\\|`~' for c in token_stripped):
                    continue
                # Skip whitespace-heavy tokens
                if len(token_stripped) < len(token) / 2:
                    continue

            opposing.append((token_stripped, -val))  # Negate so higher = more opposed

            if len(opposing) >= top_k:
                break

        return opposing

    def find_opposing_direction(
        self,
        activation: "torch.Tensor",
        strength: float = 1.0,
    ) -> "torch.Tensor":
        """Get a steering vector pointing away from current activation.

        This creates a direction that, when added to the steering vector,
        will push the model away from its current semantic territory.

        Args:
            activation: Current activation vector [d_model]
            strength: Multiplier for opposition strength

        Returns:
            Opposition steering vector [d_model]
        """
        # Convert to torch if needed
        if hasattr(activation, 'numpy'):
            act_tensor = activation
        else:
            act_tensor = torch.from_numpy(activation)

        # Simple opposition: negate the direction
        # More sophisticated: project out the mean and negate
        if act_tensor.dim() > 1:
            act_tensor = act_tensor.mean(dim=tuple(range(act_tensor.dim() - 1)))

        # Normalize then negate
        norm = torch.norm(act_tensor)
        if norm > 0:
            opposing = -act_tensor / norm * strength
        else:
            opposing = torch.zeros_like(act_tensor)

        return opposing

    def get_dialectical_concepts(
        self,
        activation: "torch.Tensor",
        top_k: int = 5,
    ) -> list[str]:
        """Extract concept words (nouns, verbs) that oppose current direction.

        Filters opposing tokens to find meaningful concept words rather
        than function words or fragments.

        Args:
            activation: Current activation vector
            top_k: Number of concepts to return

        Returns:
            List of opposing concept words
        """
        opposing = self.find_opposing_tokens(activation, top_k=top_k * 4)

        concepts = []
        for token, _ in opposing:
            # Heuristic: longer tokens more likely to be concepts
            if len(token) >= 4:
                # Clean up token
                clean = token.strip().lower()
                if clean and clean not in concepts:
                    concepts.append(clean)

            if len(concepts) >= top_k:
                break

        return concepts

    def build_dialectical_prompt(
        self,
        current_thought: str,
        current_insight: str,
        opposing_concepts: list[str],
    ) -> str:
        """Build a prompt that forces engagement with opposing concepts.

        Creates a prompt that acknowledges the previous thought but
        introduces tension through opposing concepts.

        Args:
            current_thought: The previous thought content
            current_insight: The extracted insight from previous thought
            opposing_concepts: Concepts that oppose the current direction

        Returns:
            Dialectical prompt string
        """
        # Truncate thought for prompt
        thought_preview = current_thought[:300]
        if len(current_thought) > 300:
            thought_preview += "..."

        opposition_str = ", ".join(opposing_concepts[:5])

        return f"""I've been exploring: "{thought_preview}"

My insight was: {current_insight or "still forming"}

But I notice tension with: {opposition_str}

Rather than continuing in the same direction, I want to challenge my previous thought.
What would oppose, contradict, or complicate that perspective?
What am I not seeing? What assumption might be wrong?

Let me think differently..."""


def detect_repetition(
    current_thought: str,
    previous_thoughts: list[str],
    threshold: float = 0.5,
) -> bool:
    """Detect if current thought is too similar to recent thoughts.

    Uses simple substring matching to detect verbatim or near-verbatim
    repetition that indicates stagnation.

    Args:
        current_thought: The thought to check
        previous_thoughts: Recent thoughts to compare against
        threshold: Similarity threshold (0-1) above which counts as repetition

    Returns:
        True if repetition detected
    """
    if not previous_thoughts:
        return False

    current_lower = current_thought.lower()

    for prev in previous_thoughts[-5:]:  # Check last 5 thoughts
        prev_lower = prev.lower()

        # Check for long shared substrings
        # Find longest common substring
        min_len = min(len(current_lower), len(prev_lower))
        if min_len < 50:
            continue

        # Simple check: look for 50+ char shared sequences
        for i in range(0, len(current_lower) - 50, 20):
            chunk = current_lower[i:i+50]
            if chunk in prev_lower:
                logger.debug(f"Repetition detected: '{chunk[:30]}...'")
                return True

    return False
