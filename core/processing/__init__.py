"""
Document processing pipeline for GPU-based knowledge extraction.

This module provides:
- GPUEmbedder: Generate embeddings using Qwen3-Embedding-8B
- HippoRAGProcessor: Extract entities and triples using Qwen3-8B
- DocumentProcessingCoordinator: Orchestrate GPU time-sharing
"""

from core.processing.gpu_embedder import GPUEmbedder
from core.processing.hipporag import HippoRAGProcessor
from core.processing.coordinator import DocumentProcessingCoordinator

__all__ = ["GPUEmbedder", "HippoRAGProcessor", "DocumentProcessingCoordinator"]
