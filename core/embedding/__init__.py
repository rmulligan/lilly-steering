"""Tiered embedding service for retrieval and golden embeddings.

This module provides a tiered embedding strategy:
- Retrieval embeddings: Fast, CPU-based, lower dimension (1024-dim)
- Golden embeddings: High-quality, GPU-batched, higher dimension (4096-dim)

The separation prevents GPU contention between embedding generation and
inference operations.
"""

from core.embedding.service import (
    EmbeddingTier,
    TieredEmbeddingService,
)

__all__ = [
    "EmbeddingTier",
    "TieredEmbeddingService",
]
