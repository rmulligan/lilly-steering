"""
GPU Embedder - Generate embeddings using Qwen3-Embedding-8B.

Uses SentenceTransformer with Qwen3-Embedding-8B to generate
4096-dimensional embeddings for semantic search and HippoRAG.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default embedding model - Qwen3-Embedding-8B gives 4096-dim embeddings
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# Instruction library for Qwen3 instruction-aware embeddings
QWEN3_INSTRUCTIONS = {
    "search": "Represent this text for semantic search retrieval",
    "similarity": "Represent this text for finding similar content",
    "clustering": "Represent this text for clustering with related concepts",
    "graph_path": "Represent this sequence of connected knowledge fragments as a coherent reasoning path",
    "causal_chain": "Represent this text emphasizing cause-and-effect relationships",
    "find_prerequisites": "Represent this concept to find knowledge that must be understood first",
    "find_applications": "Represent this concept to find practical applications and examples",
    "find_contradictions": "Represent this claim to find potentially conflicting beliefs",
    "merge_candidate": "Represent this fragment to find similar fragments for potential merging",
}


class GPUEmbedder:
    """Generate embeddings using Qwen3-Embedding-8B.

    Uses SentenceTransformer with the Qwen3-Embedding-8B model to generate
    4096-dimensional embeddings with instruction-aware formatting.

    The model supports explicit load/unload for GPU memory management during
    document processing.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda",
        default_instruction: str = "search",
        fallback_to_cpu: bool = True,
    ):
        """Initialize the GPU embedder.

        Args:
            model_id: HuggingFace model ID for embeddings
            device: Device to load model on ("cuda" or "cpu")
            default_instruction: Default instruction key from QWEN3_INSTRUCTIONS
            fallback_to_cpu: If True and CUDA fails, fall back to CPU (default True).
                Set to False for large models to avoid OOM on CPU fallback.
        """
        self.model_id = model_id
        self.device = device
        self.fallback_to_cpu = fallback_to_cpu
        self.default_instruction = QWEN3_INSTRUCTIONS.get(
            default_instruction, QWEN3_INSTRUCTIONS["search"]
        )

        self._model = None
        self._dimension_cache: Optional[int] = None

        # Determine dimension based on model variant
        if "0.6B" in model_id or "0.6b" in model_id:
            self._dimension_cache = 1024
        elif "4B" in model_id or "4b" in model_id:
            self._dimension_cache = 2560
        else:  # 8B default
            self._dimension_cache = 4096

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._dimension_cache or 4096

    def load(self) -> None:
        """Load the embedding model to GPU."""
        if self._model is not None:
            logger.warning("Embedding model already loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Qwen3 embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading embedding model: {self.model_id}")

        try:
            # Load in bfloat16 to reduce memory usage (8B model needs ~16GB in fp16)
            model_kwargs = {}
            if self.device == "cuda" and TORCH_AVAILABLE:
                model_kwargs["torch_dtype"] = torch.bfloat16

            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            logger.info(f"Embedding model loaded on {self.device}")
        except (RuntimeError, ImportError) as e:
            if self.device == "cuda" and self.fallback_to_cpu:
                logger.warning(f"Failed to load on CUDA, falling back to CPU: {e}")
                self._model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                    trust_remote_code=True,
                )
                logger.info("Embedding model loaded on CPU (fallback)")
            else:
                raise

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is None:
            return

        logger.info("Unloading embedding model...")

        del self._model
        self._model = None

        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Embedding model unloaded")

    def _format_prompt(self, text: str, instruction: Optional[str] = None) -> str:
        """Format text with instruction for Qwen3-Embedding.

        Qwen3-Embedding uses instruction-aware format:
        - For queries: "Instruct: {instruction}\nQuery: {text}"
        - For documents (no instruction): just the text

        Args:
            text: Text to embed
            instruction: Optional instruction string

        Returns:
            Formatted prompt
        """
        effective_instruction = instruction or self.default_instruction
        if effective_instruction:
            return f"Instruct: {effective_instruction}\nQuery: {text}"
        return text

    async def embed(
        self,
        texts: list[str],
        instruction: Optional[str] = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed
            instruction: Optional instruction for instruction-aware embedding
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors (each 4096-dim for Qwen3-8B)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not texts:
            return []

        # Format all texts with instruction
        formatted = [
            self._format_prompt(t.strip() if t else "", instruction)
            for t in texts
        ]

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                formatted,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(formatted) > 100,
            )
        )

        result = [emb.tolist() for emb in embeddings]
        logger.info(f"Generated {len(result)} embeddings of dimension {self.embedding_dim}")
        return result

    async def embed_single(
        self,
        text: str,
        instruction: Optional[str] = None,
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            instruction: Optional instruction for instruction-aware embedding

        Returns:
            Embedding vector as list of floats
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        formatted = self._format_prompt(text.strip() if text else "", instruction)

        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(formatted, normalize_embeddings=True)
        )

        return embedding.tolist()
