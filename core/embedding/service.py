"""Tiered Embedding Service: CPU retrieval + GPU golden embeddings.

This module implements the tiered embedding strategy recommended for
non-blocking embedding generation. It separates:

1. Retrieval embeddings (CPU, 1024-dim):
   - Fast lookups that don't contend with GPU inference
   - Non-blocking, runs on CPU
   - Used for real-time retrieval queries

2. Golden embeddings (GPU, 4096-dim):
   - High-resolution embeddings for storage/persistence
   - Batched operations to avoid GPU contention with inference
   - Can be blocking since batched off critical path

Architecture:
    The service uses asyncio.Semaphore for GPU access control and
    a batching queue for efficient GPU utilization. Retrieval embeddings
    run concurrently without blocking.

Usage:
    from core.embedding import TieredEmbeddingService, EmbeddingTier

    service = await TieredEmbeddingService.create()

    # Fast retrieval (CPU)
    retrieval_emb = await service.encode("query text", tier=EmbeddingTier.RETRIEVAL)

    # High-quality storage (GPU, batched)
    golden_emb = await service.encode("document text", tier=EmbeddingTier.GOLDEN)

    # Batch encode for efficiency
    embeddings = await service.encode_batch(texts, tier=EmbeddingTier.GOLDEN)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Configuration constants
RETRIEVAL_DIM = 1024  # Lower dimension for fast retrieval (Qwen3-0.6B)
GOLDEN_DIM = 4096  # Higher dimension for semantic precision (Qwen3-8B)
GPU_SEMAPHORE_LIMIT = 1  # Only one GPU operation at a time

# Default models
RETRIEVAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # CPU-friendly, 1024-dim
GOLDEN_MODEL = "Qwen/Qwen3-Embedding-8B"  # GPU, 4096-dim


class EmbeddingTier(Enum):
    """Embedding quality/speed tier."""
    RETRIEVAL = "retrieval"  # Fast, CPU-based, lower dimension
    GOLDEN = "golden"  # High-quality, GPU-batched, higher dimension


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embedding: The embedding vector
        tier: Which tier was used
        dimension: Embedding dimension
        latency_ms: Time taken in milliseconds
        was_batched: Whether this was part of a batch operation
    """
    embedding: np.ndarray
    tier: EmbeddingTier
    dimension: int
    latency_ms: float
    was_batched: bool = False

    def to_list(self) -> list[float]:
        """Convert to list for JSON serialization."""
        return self.embedding.tolist()


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for model backends that provide embeddings.

    Implementations should handle the actual model loading and inference.
    This allows swapping between different embedding models.
    """

    async def encode_async(self, text: str) -> np.ndarray:
        """Encode a single text asynchronously."""
        ...

    async def encode_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts asynchronously."""
        ...

    def unload(self) -> None:
        """Unload model and free memory (optional).

        Implementations may choose to implement this to release resources.
        If not implemented, the backend will be garbage collected normally.
        """
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension produced by this backend."""
        ...


class GPUEmbedderBackend:
    """ModelBackend adapter for GPUEmbedder.

    Wraps GPUEmbedder to implement the ModelBackend protocol, allowing it
    to be used with TieredEmbeddingService.

    Usage:
        from core.processing.gpu_embedder import GPUEmbedder

        embedder = GPUEmbedder(model_id="Qwen/Qwen3-Embedding-0.6B", device="cpu")
        embedder.load()
        backend = GPUEmbedderBackend(embedder)

        # Now use as ModelBackend
        embedding = await backend.encode_async("text")
    """

    def __init__(self, embedder: "GPUEmbedder"):
        """Initialize the backend adapter.

        Args:
            embedder: An initialized GPUEmbedder instance (must be loaded)
        """
        self._embedder = embedder

    async def encode_async(self, text: str) -> np.ndarray:
        """Encode a single text asynchronously.

        Args:
            text: Text to encode

        Returns:
            numpy array of embeddings
        """
        result = await self._embedder.embed_single(text)
        return np.array(result, dtype=np.float32)

    async def encode_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts asynchronously.

        Args:
            texts: Texts to encode

        Returns:
            List of numpy arrays
        """
        results = await self._embedder.embed(texts)
        return [np.array(r, dtype=np.float32) for r in results]

    @property
    def dimension(self) -> int:
        """Embedding dimension produced by this backend."""
        return self._embedder.embedding_dim

    def unload(self) -> None:
        """Unload the underlying embedder and free GPU memory."""
        self._embedder.unload()


# Forward reference for type hint
GPUEmbedder = "GPUEmbedder"


class TieredEmbeddingService:
    """Tiered embedding service with CPU retrieval and GPU golden embeddings.

    This service provides non-blocking embedding generation by:
    1. Using CPU for fast retrieval embeddings
    2. Batching GPU operations for golden embeddings
    3. Using semaphores to prevent GPU contention

    Attributes:
        retrieval_backend: Backend for fast CPU-based embeddings
        golden_backend: Backend for high-quality GPU embeddings
        gpu_semaphore: Limits concurrent GPU operations
    """

    def __init__(
        self,
        retrieval_backend: Optional[ModelBackend] = None,
        golden_backend: Optional[ModelBackend] = None,
        gpu_semaphore_limit: int = GPU_SEMAPHORE_LIMIT,
    ):
        """Initialize the tiered embedding service.

        Args:
            retrieval_backend: Backend for retrieval embeddings (CPU)
            golden_backend: Backend for golden embeddings (GPU)
            gpu_semaphore_limit: Max concurrent GPU operations
        """
        self.retrieval_backend = retrieval_backend
        self.golden_backend = golden_backend
        self._gpu_semaphore = asyncio.Semaphore(gpu_semaphore_limit)

        # Statistics
        self._retrieval_count = 0
        self._golden_count = 0
        self._batch_count = 0

    @classmethod
    async def create(
        cls,
        retrieval_backend: Optional[ModelBackend] = None,
        golden_backend: Optional[ModelBackend] = None,
    ) -> "TieredEmbeddingService":
        """Create a tiered embedding service.

        This is the recommended way to create the service as it allows
        for async initialization of backends if needed.

        Args:
            retrieval_backend: Backend for retrieval embeddings
            golden_backend: Backend for golden embeddings

        Returns:
            Initialized TieredEmbeddingService
        """
        service = cls(
            retrieval_backend=retrieval_backend,
            golden_backend=golden_backend,
        )
        return service

    async def encode(
        self,
        text: str,
        tier: EmbeddingTier = EmbeddingTier.RETRIEVAL,
    ) -> EmbeddingResult:
        """Encode a single text.

        For retrieval tier: Returns immediately using CPU backend.
        For golden tier: May batch with other requests for GPU efficiency.

        Args:
            text: Text to encode
            tier: Which embedding tier to use

        Returns:
            EmbeddingResult with the embedding and metadata

        Raises:
            ValueError: If the requested backend is not configured
        """
        start = datetime.now(timezone.utc)

        if tier == EmbeddingTier.RETRIEVAL:
            embedding = await self._encode_retrieval(text)
            was_batched = False
            dimension = RETRIEVAL_DIM
            self._retrieval_count += 1
        else:
            embedding = await self._encode_golden_single(text)
            was_batched = False  # Single request, not batched
            dimension = GOLDEN_DIM
            self._golden_count += 1

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return EmbeddingResult(
            embedding=embedding,
            tier=tier,
            dimension=dimension,
            latency_ms=elapsed,
            was_batched=was_batched,
        )

    async def encode_batch(
        self,
        texts: list[str],
        tier: EmbeddingTier = EmbeddingTier.GOLDEN,
    ) -> list[EmbeddingResult]:
        """Encode multiple texts.

        For golden tier: Batches efficiently on GPU.
        For retrieval tier: Processes concurrently on CPU.

        Args:
            texts: Texts to encode
            tier: Which embedding tier to use

        Returns:
            List of EmbeddingResult objects

        Raises:
            ValueError: If the requested backend is not configured
        """
        if not texts:
            return []

        start = datetime.now(timezone.utc)

        if tier == EmbeddingTier.RETRIEVAL:
            # Process concurrently on CPU
            tasks = [self._encode_retrieval(t) for t in texts]
            embeddings = await asyncio.gather(*tasks)
            was_batched = False
            dimension = RETRIEVAL_DIM
            self._retrieval_count += len(texts)
        else:
            # Batch on GPU with semaphore
            embeddings = await self._encode_golden_batch(texts)
            was_batched = True
            dimension = GOLDEN_DIM
            self._golden_count += len(texts)
            self._batch_count += 1

        elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        per_item_latency = elapsed / len(texts) if texts else 0

        return [
            EmbeddingResult(
                embedding=emb,
                tier=tier,
                dimension=dimension,
                latency_ms=per_item_latency,
                was_batched=was_batched,
            )
            for emb in embeddings
        ]

    async def _encode_retrieval(self, text: str) -> np.ndarray:
        """Encode using retrieval backend (CPU, non-blocking)."""
        if self.retrieval_backend is None:
            raise ValueError("Retrieval embedding backend is not configured.")

        return await self.retrieval_backend.encode_async(text)

    # TODO: Implement request queuing for efficient batching of single golden
    # embedding requests. Currently processes each request individually.
    async def _encode_golden_single(self, text: str) -> np.ndarray:
        """Encode a single text using golden backend (GPU, batched)."""
        results = await self._encode_golden_batch([text])
        return results[0]

    async def _encode_golden_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode batch using golden backend (GPU, with semaphore)."""
        if self.golden_backend is None:
            raise ValueError("Golden embedding backend is not configured.")

        # Acquire GPU semaphore to prevent contention
        async with self._gpu_semaphore:
            return await self.golden_backend.encode_batch_async(texts)

    def get_stats(self) -> dict:
        """Get embedding service statistics."""
        return {
            "retrieval_count": self._retrieval_count,
            "golden_count": self._golden_count,
            "batch_count": self._batch_count,
            "retrieval_dimension": RETRIEVAL_DIM,
            "golden_dimension": GOLDEN_DIM,
        }

    async def ensure_golden_loaded(self) -> None:
        """Ensure golden embedding backend is loaded and ready.

        For the three-phase cognitive cycle, this is called before the
        integration phase to prepare for zettel embedding.

        Note: Currently golden embeddings are optional, so this method
        gracefully handles the case where no golden backend is configured.
        """
        if self.golden_backend is None:
            logger.debug("No golden backend configured, using retrieval backend for embeddings")
            return

        # If the backend has a load method, call it
        if hasattr(self.golden_backend, 'load'):
            await self.golden_backend.load()
            logger.info("Golden embedding backend loaded")

    def unload_golden(self) -> None:
        """Unload the golden backend and free GPU memory.

        Call this after completing startup semantic operations
        before loading the main LLM.
        """
        if self.golden_backend is not None:
            # unload() is part of ModelBackend protocol but optional to implement
            if hasattr(self.golden_backend, 'unload'):
                self.golden_backend.unload()
                logger.info("Golden embedding backend unloaded")
            self.golden_backend = None


async def create_tiered_embedding_service(
    retrieval_backend: Optional[ModelBackend] = None,
    golden_backend: Optional[ModelBackend] = None,
    auto_create_retrieval: bool = False,
    auto_create_golden: bool = False,
    retrieval_device: str = "cpu",
    golden_device: str = "cuda",
) -> TieredEmbeddingService:
    """Create a tiered embedding service.

    This is a convenience function that wraps TieredEmbeddingService.create().

    Args:
        retrieval_backend: Pre-configured retrieval backend
        golden_backend: Pre-configured golden backend
        auto_create_retrieval: If True and no retrieval_backend provided,
            automatically create one using Qwen3-Embedding-0.6B
        auto_create_golden: If True and no golden_backend provided,
            automatically create one using Qwen3-Embedding-8B
        retrieval_device: Device for auto-created retrieval backend
        golden_device: Device for auto-created golden backend

    Usage:
        # With both backends auto-created
        service = await create_tiered_embedding_service(
            auto_create_retrieval=True,
            auto_create_golden=True,
        )

        # Fast retrieval (CPU, 1024-dim)
        result = await service.encode("query", tier=EmbeddingTier.RETRIEVAL)

        # High-quality golden embedding (GPU, 4096-dim)
        result = await service.encode("document", tier=EmbeddingTier.GOLDEN)
    """
    if retrieval_backend is None and auto_create_retrieval:
        retrieval_backend = await create_retrieval_backend(device=retrieval_device)

    if golden_backend is None and auto_create_golden:
        try:
            golden_backend = await create_golden_backend(device=golden_device)
        except Exception as e:
            logger.warning(
                f"Failed to create golden backend on {golden_device}: {e}. "
                "Fragment embeddings will use retrieval backend."
            )
            golden_backend = None
            # Clean up any leaked CUDA memory from failed load
            if golden_device == "cuda":
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

    return await TieredEmbeddingService.create(
        retrieval_backend=retrieval_backend,
        golden_backend=golden_backend,
    )


async def create_retrieval_backend(
    model_id: str = RETRIEVAL_MODEL,
    device: str = "cpu",
) -> GPUEmbedderBackend:
    """Create a retrieval embedding backend using Qwen3-Embedding-0.6B.

    This creates a CPU-based embedding backend suitable for fast retrieval
    operations that don't block GPU inference.

    Args:
        model_id: Model to use (default: Qwen3-Embedding-0.6B)
        device: Device to load model on (default: cpu)

    Returns:
        GPUEmbedderBackend wrapping the loaded model
    """
    from core.processing.gpu_embedder import GPUEmbedder

    logger.info(f"Creating retrieval backend with {model_id} on {device}")

    embedder = GPUEmbedder(
        model_id=model_id,
        device=device,
        default_instruction="search",
    )

    # Load synchronously since this is an async function
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, embedder.load)

    logger.info(f"Retrieval backend ready (dim={embedder.embedding_dim})")

    return GPUEmbedderBackend(embedder)


async def create_golden_backend(
    model_id: str = GOLDEN_MODEL,
    device: str = "cuda",
) -> GPUEmbedderBackend:
    """Create a golden embedding backend using Qwen3-Embedding-8B.

    This creates a GPU-based embedding backend for high-quality embeddings
    used in fragment storage. These embeddings have higher dimensionality
    (4096-dim) for better semantic precision.

    Args:
        model_id: Model to use (default: Qwen3-Embedding-8B)
        device: Device to load model on (default: cuda)

    Returns:
        GPUEmbedderBackend wrapping the loaded model
    """
    from core.processing.gpu_embedder import GPUEmbedder

    logger.info(f"Creating golden backend with {model_id} on {device}")

    embedder = GPUEmbedder(
        model_id=model_id,
        device=device,
        default_instruction="passage",  # Optimized for document embedding
        fallback_to_cpu=False,  # 8B model too large for CPU fallback
    )

    # Load synchronously since this is an async function
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, embedder.load)

    logger.info(f"Golden backend ready (dim={embedder.embedding_dim})")

    return GPUEmbedderBackend(embedder)
