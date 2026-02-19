"""Chunked Text Generation for PLaT-Lite Latent Observation.

Generates text in small increments (16 tokens by default) with activation
capture at each chunk. This enables real-time latent state observation
and intentional steering during generation.

Architecture:
    ChunkedGenerator wraps HookedQwen and produces ChunkResult objects
    that contain:
    - Generated tokens and text for this chunk
    - Layer 16 activations for latent observation
    - MLP input for SAE feature extraction
    - Termination detection (EOS, complete thought)

Key Insight from PLaT:
    By observing activations after each chunk, we can track the "planning
    trajectory" in latent space and steer before it collapses into text.
    This gives PLaT-like observability without retraining the model.

Usage:
    from core.cognitive.stream.chunked_generator import ChunkedGenerator

    generator = ChunkedGenerator(hooked_model, tokenizer)

    full_text = ""
    for _ in range(generator.MAX_CHUNKS):
        chunk = await generator.generate_chunk(
            context=prompt + full_text,
            chunk_size=16,
            steering=steering_state,
        )
        full_text += chunk.text

        # Observe latent state, adjust steering...

        if chunk.is_complete:
            break

Reference:
    Inspired by PLaT (Planning with Latent Thoughts, arXiv:2601.21358) -
    reasoning in continuous latent space, verbalization when needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from core.model.hooked_qwen import HookedQwen, ActivationSnapshot
    from core.steering.hierarchical import HierarchicalSteerer

logger = logging.getLogger(__name__)

# Lazy torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class ChunkResult:
    """Result of generating one chunk.

    Attributes:
        chunk_idx: Index of this chunk in the generation sequence
        tokens: Generated token IDs for this chunk
        text: Decoded text for this chunk
        activations: Layer 16 activations [1, seq, d_model] (optional)
        mlp_input: MLP input activations for SAE encoding (optional)
        is_complete: True if natural termination detected (EOS, period+EOS)
        cumulative_tokens: Total tokens generated so far including this chunk
    """

    chunk_idx: int
    tokens: list[int]
    text: str
    activations: Optional["torch.Tensor"] = None
    mlp_input: Optional["torch.Tensor"] = None
    is_complete: bool = False
    cumulative_tokens: int = 0


class ChunkedGenerator:
    """Generates text in small chunks with activation capture.

    Enables PLaT-style latent observation by producing partial generation
    results that include activations. External components can observe and
    steer between chunks.

    Constants:
        DEFAULT_CHUNK_SIZE: 16 tokens per chunk (balances granularity and overhead)
        MAX_CHUNKS: 32 chunks maximum (~512 tokens total)
        CAPTURE_LAYER: 16 (mid-model "planning" layer per PLaT insights)
    """

    DEFAULT_CHUNK_SIZE = 16
    MAX_CHUNKS = 32
    CAPTURE_LAYER = 16

    # Termination patterns (common thought-ending indicators)
    # Note: Double newline ("\n\n") is handled separately in _detect_termination
    # because it requires checking the original text ending, not stripped text
    TERMINATION_TOKENS = {"</s>", "<|endoftext|>", "<|im_end|>"}

    def __init__(
        self,
        model: "HookedQwen",
        temperature: float = 0.7,
    ):
        """Initialize the chunked generator.

        Args:
            model: HookedQwen instance for generation
            temperature: Sampling temperature (default: 0.7)
        """
        self._model = model
        self._temperature = temperature
        self._chunk_idx = 0
        self._total_tokens = 0

    def reset(self) -> None:
        """Reset chunk counter for new generation."""
        self._chunk_idx = 0
        self._total_tokens = 0

    @property
    def tokenizer(self):
        """Access the model's tokenizer."""
        try:
            return self._model.tokenizer
        except AttributeError:
            return None

    async def generate_chunk(
        self,
        context: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        steering: Optional["HierarchicalSteerer"] = None,
        capture_mlp_input: bool = True,
    ) -> ChunkResult:
        """Generate one chunk and capture activations.

        Args:
            context: Full context (prompt + previously generated text)
            chunk_size: Number of tokens to generate this chunk
            steering: Optional HierarchicalSteerer for steering during generation
            capture_mlp_input: Whether to capture MLP input for SAE encoding

        Returns:
            ChunkResult with tokens, text, activations, and completion status
        """
        if not self._model._model:
            raise RuntimeError("Model not loaded")

        # Generate chunk with activation capture
        result = await self._model.generate(
            prompt=context,
            max_tokens=chunk_size,
            temperature=self._temperature,
            capture_activations=True,
            capture_layers=[self.CAPTURE_LAYER],
            capture_mlp_input=capture_mlp_input,
            hierarchical_steerer=steering,
        )

        # Extract activations from snapshots
        activations = None
        mlp_input = None
        for snapshot in result.snapshots:
            if snapshot.layer == self.CAPTURE_LAYER:
                if snapshot.hook_point == "resid_post":
                    activations = snapshot.activations
                elif snapshot.hook_point == "mlp_in":
                    mlp_input = snapshot.activations

        # Detect natural termination
        is_complete = self._detect_termination(result.tokens, result.text)

        # Update counters
        self._total_tokens += len(result.tokens)
        chunk_result = ChunkResult(
            chunk_idx=self._chunk_idx,
            tokens=result.tokens,
            text=result.text,
            activations=activations,
            mlp_input=mlp_input,
            is_complete=is_complete,
            cumulative_tokens=self._total_tokens,
        )

        self._chunk_idx += 1

        logger.debug(
            f"[CHUNKED_GEN] Chunk {chunk_result.chunk_idx}: "
            f"{len(result.tokens)} tokens, complete={is_complete}"
        )

        return chunk_result

    def _detect_termination(self, tokens: list[int], text: str) -> bool:
        """Detect if generation has naturally completed.

        Checks for:
        1. EOS token in generated tokens
        2. Known termination patterns in text
        3. Complete sentence followed by newline

        Args:
            tokens: Generated token IDs
            text: Decoded text

        Returns:
            True if generation appears complete
        """
        if not self.tokenizer:
            return False

        # Check for EOS token
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id in tokens:
            return True

        # Check for termination patterns (on stripped text for endoftext/im_end)
        text_stripped = text.strip()
        for pattern in self.TERMINATION_TOKENS:
            if text_stripped.endswith(pattern):
                return True

        # Check for sentence end followed by newline (common thought completion)
        # Use original text to preserve trailing newlines
        if text.rstrip(" ").endswith((".\n", "?\n", "!\n")):
            return True

        # Check for double newline at end of chunk (paragraph break)
        if text.endswith("\n\n"):
            return True

        return False

    async def generate_full(
        self,
        prompt: str,
        steering: Optional["HierarchicalSteerer"] = None,
        on_chunk: Optional[callable] = None,
    ) -> tuple[str, list[ChunkResult]]:
        """Generate complete text with chunk-by-chunk observation.

        Convenience method that handles the full generation loop,
        calling an optional callback after each chunk for observation
        and steering adjustments.

        Args:
            prompt: Initial prompt text
            steering: Optional HierarchicalSteerer (can be modified in callback)
            on_chunk: Optional async callback(ChunkResult) -> Optional[HierarchicalSteerer]
                     Return new steerer to adjust, or None to continue unchanged

        Returns:
            Tuple of (full_text, list[ChunkResult])
        """
        self.reset()
        full_text = ""
        chunks: list[ChunkResult] = []

        for _ in range(self.MAX_CHUNKS):
            chunk = await self.generate_chunk(
                context=prompt + full_text,
                steering=steering,
            )
            chunks.append(chunk)
            full_text += chunk.text

            # Call observation callback
            if on_chunk:
                new_steering = await on_chunk(chunk)
                if new_steering is not None:
                    steering = new_steering

            # Check termination
            if chunk.is_complete:
                logger.info(
                    f"[CHUNKED_GEN] Generation complete after {len(chunks)} chunks, "
                    f"{self._total_tokens} total tokens"
                )
                break

        return full_text, chunks
