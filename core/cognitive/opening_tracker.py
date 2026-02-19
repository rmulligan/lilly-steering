"""Opening diversity tracker for cognitive loop.

Tracks recent thought openings to detect and prevent repetitive starts.
When repetition is detected, provides diversity directives to inject
into prompts, forcing varied generation.

Enhanced with semantic similarity detection to catch paraphrases like
"Truth whispers through the cosmos" vs "The cosmos murmurs truth"
which would evade word-based Jaccard overlap.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService

logger = logging.getLogger(__name__)


# Diversity directives injected when repetition detected
DIVERSITY_DIRECTIVES = [
    "Begin with a question that challenges me.",
    "Start from tension or contradiction.",
    "Open with what surprises or unsettles me.",
    "Lead with uncertainty or doubt.",
    "Begin by naming what I don't understand.",
    "Start with an image or sensation.",
    "Open with a memory or association.",
    "Lead with what feels urgent right now.",
]


def extract_opening(thought: str, max_chars: int = 60) -> str:
    """Extract the opening phrase from a thought.

    Takes the first sentence or first chunk of text to use as
    a fingerprint for repetition detection.

    Args:
        thought: Full thought text
        max_chars: Maximum characters for opening if no sentence end found

    Returns:
        Normalized opening phrase
    """
    if not thought:
        return ""

    # Normalize whitespace
    text = " ".join(thought.split())

    # Try to get first sentence
    for end_char in [". ", ".\n", "! ", "? "]:
        if end_char in text[:max_chars + 20]:
            idx = text.find(end_char)
            if idx > 0 and idx < max_chars + 20:
                return text[:idx + 1].strip()

    # Fallback: truncate at max_chars, try to break at word boundary
    if len(text) > max_chars:
        truncated = text[:max_chars]
        # Find last space
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            return truncated[:last_space].strip()
        return truncated.strip()

    return text.strip()


def _fingerprint(text: str) -> str:
    """Create a fingerprint for similarity comparison.

    Normalizes text to lowercase, removes punctuation, and hashes
    for fast comparison.

    Args:
        text: Text to fingerprint

    Returns:
        MD5 hash of normalized text
    """
    # Normalize: lowercase, remove punctuation, collapse whitespace
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _word_overlap(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity coefficient (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Cosine similarity (-1 to 1, typically 0 to 1 for embeddings)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class OpeningTracker:
    """Tracks thought openings to detect and prevent repetition.

    Maintains a sliding window of recent openings and provides
    diversity directives when repetition is detected.

    Enhanced with semantic similarity detection (via embeddings) to catch
    paraphrases that would evade word-based Jaccard overlap.

    Attributes:
        window_size: Number of recent openings to track
        similarity_threshold: Jaccard similarity above which is "repetitive"
        semantic_threshold: Cosine similarity above which is "semantically repetitive"
        _openings: Deque of recent opening texts
        _fingerprints: Set of fingerprints for exact match detection
        _embeddings: Deque of cached embeddings for semantic comparison
    """

    def __init__(
        self,
        window_size: int = 5,
        similarity_threshold: float = 0.7,
        semantic_threshold: float = 0.85,
    ):
        """Initialize tracker.

        Args:
            window_size: Number of recent openings to remember
            similarity_threshold: Similarity threshold for fuzzy word matching
            semantic_threshold: Cosine similarity threshold for semantic matching
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.semantic_threshold = semantic_threshold
        self._openings: deque[str] = deque(maxlen=window_size)
        self._fingerprints: set[str] = set()
        self._embeddings: deque[np.ndarray] = deque(maxlen=window_size)

    def is_repetitive(self, opening: str) -> bool:
        """Check if opening is repetitive compared to recent history (syntactic).

        Uses fingerprint (exact match) and word overlap (fuzzy match).
        For semantic similarity, use is_repetitive_semantic() instead.

        Args:
            opening: Opening phrase to check

        Returns:
            True if opening is too similar to recent openings
        """
        if not opening:
            return False

        # Check exact match via fingerprint
        fp = _fingerprint(opening)
        if fp in self._fingerprints:
            return True

        # Check fuzzy match via word overlap
        for prev in self._openings:
            similarity = _word_overlap(opening, prev)
            if similarity >= self.similarity_threshold:
                return True

        return False

    async def is_repetitive_semantic(
        self,
        opening: str,
        embedder: "TieredEmbeddingService",
    ) -> bool:
        """Check if opening is semantically repetitive using embeddings.

        This catches paraphrases like "Truth whispers through the cosmos"
        vs "The cosmos murmurs truth" that would evade word-based detection.

        Args:
            opening: Opening phrase to check
            embedder: Embedding service for semantic comparison

        Returns:
            True if opening is semantically too similar to recent openings
        """
        if not opening or not self._embeddings:
            return False

        try:
            from core.embedding.service import EmbeddingTier

            # Embed the new opening
            result = await embedder.encode(opening, tier=EmbeddingTier.RETRIEVAL)
            new_embedding = np.array(result.to_list())

            # Check semantic similarity against cached embeddings
            for prev_emb in self._embeddings:
                similarity = _cosine_similarity(new_embedding, prev_emb)
                if similarity >= self.semantic_threshold:
                    logger.debug(
                        f"Semantic repetition detected: similarity={similarity:.3f} >= {self.semantic_threshold}"
                    )
                    return True

            return False

        except Exception as e:
            logger.warning(f"Semantic similarity check failed: {e}")
            return False  # Fallback to syntactic check only

    async def record_with_embedding(
        self,
        opening: str,
        embedder: "TieredEmbeddingService",
    ) -> None:
        """Record an opening with its embedding for semantic comparison.

        Args:
            opening: Opening phrase to record
            embedder: Embedding service for computing embedding
        """
        if not opening:
            return

        # If at capacity, remove oldest fingerprint
        if len(self._openings) >= self.window_size:
            oldest = self._openings[0]
            oldest_fp = _fingerprint(oldest)
            self._fingerprints.discard(oldest_fp)

        # Add new opening
        self._openings.append(opening)
        self._fingerprints.add(_fingerprint(opening))

        # Compute and cache embedding
        try:
            from core.embedding.service import EmbeddingTier

            result = await embedder.encode(opening, tier=EmbeddingTier.RETRIEVAL)
            self._embeddings.append(np.array(result.to_list()))
        except Exception as e:
            logger.debug(f"Failed to cache embedding for opening: {e}")

    def record(self, opening: str) -> None:
        """Record an opening in history.

        Args:
            opening: Opening phrase to record
        """
        if not opening:
            return

        # If at capacity, remove oldest fingerprint
        if len(self._openings) >= self.window_size:
            oldest = self._openings[0]
            oldest_fp = _fingerprint(oldest)
            self._fingerprints.discard(oldest_fp)

        # Add new opening
        self._openings.append(opening)
        self._fingerprints.add(_fingerprint(opening))

    def get_diversity_directive(self, cycle_count: int) -> str:
        """Get a diversity directive to inject into prompt.

        Rotates through directives based on cycle count.

        Args:
            cycle_count: Current cycle number

        Returns:
            Directive string to add to prompt
        """
        idx = cycle_count % len(DIVERSITY_DIRECTIVES)
        return DIVERSITY_DIRECTIVES[idx]

    def check_and_record(self, thought: str) -> tuple[bool, str]:
        """Check for repetition and record opening in one call.

        Convenience method for the cognitive loop.

        Args:
            thought: Full thought text

        Returns:
            Tuple of (was_repetitive, opening_used)
        """
        opening = extract_opening(thought)
        was_repetitive = self.is_repetitive(opening)
        self.record(opening)
        return was_repetitive, opening
