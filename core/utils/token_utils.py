"""
Token counting and manipulation utilities for Lilly's cognitive system.

Provides consistent token operations across the codebase using tiktoken
with the cl100k_base encoding (compatible with GPT-4 and Claude).
"""

from __future__ import annotations

import tiktoken

# Cache the encoder instance for performance
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get tiktoken encoder (cached for performance)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.

    Uses cl100k_base encoding (GPT-4, Claude compatible).

    Args:
        text: Text to count tokens in

    Returns:
        Token count
    """
    try:
        enc = _get_encoder()
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


# Alias for backwards compatibility
estimate_tokens = count_tokens


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.

    Uses binary search on word boundaries to find the longest
    prefix that fits within the token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated text (unchanged if already within limit)
    """
    if count_tokens(text) <= max_tokens:
        return text

    # Binary search for cutoff point on word boundaries
    words = text.split()
    low, high = 0, len(words)

    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(" ".join(words[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1

    return " ".join(words[:low])
