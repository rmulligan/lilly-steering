"""TransformerLens wrapper for Qwen3-8B with steering capabilities.

This module provides a clean interface for loading Qwen models through
TransformerLens, enabling:
- Activation capture during forward passes
- Steering vector application via hooks
- Contrastive activation extraction for vector computation

Architecture:
    HookedQwen wraps a HookedTransformer model and provides:
    1. Async load/unload for lifecycle management
    2. Steering vector storage and application during generation
    3. Activation capture for introspection and contrastive pairs
    4. Clean separation from the unified GPU coordinator

This is the core model layer - higher-level components (consciousness,
personality) build on top of this foundation.

Usage:
    qwen = HookedQwen(model_name="Qwen/Qwen3-8B", device="cuda")
    await qwen.load()

    # Generation with optional capture
    result = await qwen.generate(prompt, capture_activations=True)

    # Set steering for subsequent generations
    qwen.set_steering_vector(layer=15, vector=steering_vec, coefficient=0.8)

    # Get activations for contrastive pairs
    activations = await qwen.get_activations(text, layers=[10, 15, 20])
"""

from __future__ import annotations

# IMPORTANT: Set CUDA memory config BEFORE any torch imports
# This prevents OOM errors during model loading by enabling expandable memory segments
# NOTE: expandable_segments disabled for vLLM sleep mode compatibility
# (see load() method comment for details)
import os

import asyncio
import logging
import re
import traceback
import threading
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import numpy as np
    from core.steering.hierarchical import HierarchicalSteerer
    from core.cognitive.telemetry import TelemetrySummary

logger = logging.getLogger(__name__)

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Lazy import for TransformerLens
HookedTransformer = None
TRANSFORMERLENS_AVAILABLE = False

try:
    from transformer_lens import HookedTransformer
    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    logger.debug("transformer_lens not installed - will use lazy loading")

# Buffer tokens reserved at the end of context window for safety margin
CONTEXT_BUFFER = 10

# Threshold for detecting outputs that may still contain prompt content
SUSPICIOUSLY_LONG_OUTPUT_CHARS = 2500


@dataclass
class SteeringConfig:
    """Configuration for activation steering.

    Attributes:
        layer: Layer index to apply steering (default: 15, middle layer for most models)
        coefficient: Scaling factor for the steering vector (default: 1.0)
        position: Which sequence positions to steer ("all", "last", or specific index)
        hook_point: TransformerLens hook point name ("resid_pre", "resid_post", "resid_mid")
    """

    layer: int = 15  # Middle layer default
    coefficient: float = 1.0
    position: str = "all"  # "all", "last", or specific index
    hook_point: str = "resid_post"  # resid_pre, resid_post, resid_mid


@dataclass
class ActivationSnapshot:
    """Captured activations from a forward pass.

    Stores the raw activation tensor along with metadata about
    where in the sequence it was captured.

    Attributes:
        layer: Layer index these activations came from
        activations: Raw activation tensor [batch, seq, d_model]
        hook_point: Which hook point was used
        prompt_tokens: Number of tokens in the original prompt
        generated_tokens: Number of tokens generated (0 for pure forward pass)
    """

    layer: int
    activations: "torch.Tensor"
    hook_point: str
    prompt_tokens: int
    generated_tokens: int


@dataclass
class GenerationResult:
    """Result from generation with optional introspection.

    Attributes:
        text: Generated text output
        tokens: Token IDs (if captured)
        snapshots: List of activation snapshots (if capture was enabled)
        steering_applied: Whether steering vectors were active during generation
        telemetry: Optional telemetry summary (if capture was enabled)
    """

    text: str
    tokens: list[int]
    snapshots: list[ActivationSnapshot] = field(default_factory=list)
    steering_applied: bool = False
    telemetry: Optional["TelemetrySummary"] = None


class HookedQwen:
    """
    Qwen model loaded through TransformerLens for activation access.

    This is the core inference engine for activation-aware cognition.
    All generation runs through here, allowing us to:
    - Cache activations during forward pass
    - Apply steering vectors via hooks
    - Extract contrastive activation patterns

    The class is designed for async usage and lazy loading - the model
    is not loaded until `load()` is called.

    Attributes:
        model_name: HuggingFace model identifier
        device: Target device ("cuda", "cpu", etc.)
        dtype: Model precision (torch.float16, torch.bfloat16, etc.)
        is_loaded: Whether the model is currently loaded
        n_layers: Number of transformer layers (0 if not loaded)
    """

    def __init__(
        self,
        model_name: str = "Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1",
        device: str = "cuda",
        dtype: Optional["torch.dtype"] = None,
        n_ctx: int = 16384,
        base_architecture: Optional[str] = "Qwen/Qwen3-8B",
    ):
        """Initialize HookedQwen wrapper.

        Args:
            model_name: HuggingFace model identifier (default: Josiefied-Qwen3-8B-abliterated)
            device: Target device for inference (default: "cuda")
            dtype: Model precision (default: torch.float16 if torch available)
            n_ctx: Context window size in tokens (default: 16384, max 32768 for Qwen3).
                   Larger values use more GPU memory for attention computation.
            base_architecture: TransformerLens-compatible model name for architecture
                   (default: Qwen/Qwen3-8B). Required for fine-tunes not in TL's official list.
                   The architecture config comes from base_architecture, but weights
                   are loaded from model_name.
        """
        self.model_name = model_name
        self.base_architecture = base_architecture
        self.device = device
        self._n_ctx = n_ctx

        # Handle dtype with fallback for missing torch
        # Use bfloat16 for better memory efficiency during CPU-GPU transfer
        if dtype is not None:
            self.dtype = dtype
        elif TORCH_AVAILABLE:
            self.dtype = torch.bfloat16
        else:
            self.dtype = None  # Will be set on load

        self._model: Optional["HookedTransformer"] = None
        self._steering_vectors: dict[int, "torch.Tensor"] = {}
        self._steering_hooks: dict[int, any] = {}  # Layer -> hook handle
        self._steering_config: Optional[SteeringConfig] = None
        # Threading lock to serialize model operations (prevents "Already borrowed" tokenizer errors)
        self._model_thread_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def n_layers(self) -> int:
        """Number of transformer layers."""
        if not self._model:
            return 0
        return self._model.cfg.n_layers

    @property
    def d_model(self) -> int:
        """Model hidden dimension."""
        if not self._model:
            return 0
        return self._model.cfg.d_model

    @property
    def W_U(self) -> "torch.Tensor":
        """Unembedding matrix for logit lens projection.

        Returns the model's output embedding matrix which maps hidden states
        to vocabulary logits. Shape: [d_model, vocab_size].

        Raises:
            AttributeError: If the model is not loaded.
        """
        if not self._model:
            raise AttributeError("Model not loaded, W_U not available.")
        return self._model.W_U

    @property
    def tokenizer(self):
        """Tokenizer for decoding token IDs.

        Returns the model's tokenizer for encoding/decoding text.

        Raises:
            AttributeError: If the model is not loaded.
        """
        if not self._model:
            raise AttributeError("Model not loaded, tokenizer not available.")
        return self._model.tokenizer

    @property
    def n_ctx(self) -> int:
        """Model context window size (max sequence length)."""
        if not self._model:
            return 0
        return self._model.cfg.n_ctx

    async def load(self) -> None:
        """Load model into TransformerLens.

        This is an async operation that runs the actual model loading
        in a thread pool executor to avoid blocking the event loop.

        Uses from_pretrained_no_processing to skip memory-intensive
        weight processing (centering, folding layer norms) which can
        cause OOM with large models in reduced precision.

        Raises:
            RuntimeError: If transformer_lens is not installed
        """
        if not TRANSFORMERLENS_AVAILABLE:
            raise RuntimeError("transformer_lens not installed")

        # Skip loading if already loaded
        if self._model is not None:
            logger.debug(f"{self.model_name} already loaded, skipping")
            return

        # NOTE: expandable_segments disabled permanently for vLLM compatibility
        # vLLM's EngineCore subprocess crashes with assertion error if this is set,
        # breaking the six-phase cognitive cycle's GPU memory management.
        # The slight OOM risk during TransformerLens loading is acceptable trade-off
        # for working vLLM sleep mode (essential for phase transitions).

        logger.info(f"Loading {self.model_name} via TransformerLens...")

        # Clear CUDA cache and force garbage collection before loading
        if TORCH_AVAILABLE and torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"CUDA memory before load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

        # Run in executor to not block event loop
        loop = asyncio.get_running_loop()

        def _load_model():
            import gc
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Direct GPU loading with low memory usage
            # If base_architecture is specified, use it for the architecture config
            # but load weights from model_name (for fine-tunes not in TL's list)
            if self.base_architecture and self.base_architecture != self.model_name:
                logger.info(f"Loading architecture from {self.base_architecture}, weights from {self.model_name}")
                # Load HuggingFace model on CPU first - TransformerLens will handle GPU placement
                # This avoids memory duplication from loading twice
                from transformers import AutoModelForCausalLM
                hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )  # Keep on CPU - TransformerLens will move to GPU
                # Let TransformerLens handle device placement to avoid memory duplication
                model = HookedTransformer.from_pretrained(
                    self.base_architecture,
                    hf_model=hf_model,
                    device=self.device,
                    dtype=self.dtype,
                    fold_ln=False,
                    fold_value_biases=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    n_devices=1,
                )
                # Free the CPU model explicitly
                del hf_model
                gc.collect()
            else:
                model = HookedTransformer.from_pretrained(
                    self.model_name,
                    device=self.device,
                    dtype=self.dtype,
                    fold_ln=False,
                    fold_value_biases=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    n_devices=1,
                    low_cpu_mem_usage=True,
                )
            return model

        self._model = await loop.run_in_executor(None, _load_model)

        # Rebuild rotary embeddings if we need a larger context window
        # The model's original n_ctx (from HuggingFace config) may be smaller than what we want
        original_n_ctx = self._model.cfg.n_ctx
        if self._n_ctx and self._n_ctx > original_n_ctx:
            logger.info(f"Rebuilding rotary embeddings from {original_n_ctx} to {self._n_ctx}")
            self._rebuild_rotary_embeddings(self._n_ctx)
            self._model.cfg.n_ctx = self._n_ctx

        logger.info(f"Loaded {self.model_name} with {self.n_layers} layers, context window: {self.n_ctx}")

    def _rebuild_rotary_embeddings(self, new_n_ctx: int) -> None:
        """Rebuild rotary position embeddings for a new context length.

        TransformerLens pre-allocates rotary sin/cos buffers based on the model's
        original n_ctx. When we want a larger context window, we need to rebuild
        these buffers to avoid tensor size mismatches during forward pass.

        Args:
            new_n_ctx: The new context window size
        """
        if self._model is None:
            return

        cfg = self._model.cfg

        # Get rotary parameters from config
        rotary_dim = cfg.rotary_dim
        if rotary_dim is None:
            rotary_dim = cfg.d_head

        # Qwen3 uses rotary_base=1000000, get from config
        rotary_base = getattr(cfg, 'rotary_base', 1000000)

        # Get target dtype from model
        target_dtype = self.dtype

        logger.debug(f"Rotary params: dim={rotary_dim}, base={rotary_base}, dtype={target_dtype}")

        # Use the attention layer's own calculate_sin_cos_rotary method
        # which handles the calculation correctly for TransformerLens
        first_attn = self._model.blocks[0].attn
        new_sin, new_cos = first_attn.calculate_sin_cos_rotary(
            rotary_dim=rotary_dim,
            n_ctx=new_n_ctx,
            base=rotary_base,
            dtype=target_dtype,
        )

        # Move to correct device
        new_sin = new_sin.to(self.device)
        new_cos = new_cos.to(self.device)

        # Update rotary buffers in each attention block
        for i, block in enumerate(self._model.blocks):
            attn = block.attn
            attn.register_buffer('rotary_sin', new_sin, persistent=False)
            attn.register_buffer('rotary_cos', new_cos, persistent=False)

        logger.info(f"Rebuilt rotary embeddings for {len(self._model.blocks)} layers, new shape: {list(new_sin.shape)}")

    def _build_telemetry_hooks(
        self,
        residual_layers: list[int],
        attention_layer: int,
        sample_rate: int,
    ) -> tuple[list[tuple[str, Callable]], dict[str, list[float]]]:
        """Build hooks for telemetry capture with in-hook tensor reduction.

        Creates hooks that reduce tensors to scalars immediately to avoid
        retaining large tensors in memory. This is critical for the
        sequential model loading invariant.

        Args:
            residual_layers: Layer indices for residual norm capture
            attention_layer: Layer for attention entropy capture (-1 to disable)
            sample_rate: Sample every N generated tokens

        Returns:
            Tuple of (hooks_list, collectors_dict) where:
                - hooks_list: List of (hook_name, hook_fn) tuples
                - collectors_dict: Dict to collect reduced scalar values
        """
        collectors: dict[str, list[float]] = {
            "logit_entropy": [],
            "top1_top2_margin": [],
        }
        for layer in residual_layers:
            collectors[f"residual_norm_{layer}"] = []
        if attention_layer >= 0:
            collectors["attention_entropy"] = []

        # Token counter for sampling - shared across all hooks
        # Incremented once per token in a dedicated counter hook, not per-layer
        sample_counter = {"count": 0}

        EPS = 1e-8
        hooks = []

        # Residual norm hooks - sample based on shared counter (don't increment here)
        for layer in residual_layers:
            hook_name = f"blocks.{layer}.hook_resid_post"

            def make_residual_hook(l: int):
                def residual_hook(acts: "torch.Tensor", hook: Any) -> "torch.Tensor":
                    # Sample based on counter (counter is incremented in logit hook, once per token)
                    if sample_counter["count"] % sample_rate == 0:
                        # Reduce immediately: [batch, seq, d_model] -> scalar norm
                        # Take last token's norm (most recent)
                        norm = acts[0, -1, :].norm().item()
                        collectors[f"residual_norm_{l}"].append(norm)
                    return acts
                return residual_hook

            hooks.append((hook_name, make_residual_hook(layer)))

        # Attention entropy hook (optional)
        if attention_layer >= 0:
            hook_name = f"blocks.{attention_layer}.attn.hook_pattern"

            def attention_hook(pattern: "torch.Tensor", hook: Any) -> "torch.Tensor":
                # pattern shape: [batch, heads, seq, seq]
                # Compute entropy immediately and discard tensor
                if sample_counter["count"] % sample_rate == 0:
                    # Take attention to last position, average over heads
                    # pattern[0, :, -1, :] = attention from last query to all keys
                    p = pattern[0, :, -1, :].clamp(min=EPS)
                    entropy = -(p * p.log()).sum(dim=-1).mean().item()
                    collectors["attention_entropy"].append(entropy)
                return pattern

            hooks.append((hook_name, attention_hook))

        return hooks, collectors, sample_counter

    def _build_logit_hooks(
        self,
        collectors: dict[str, list[float]],
        sample_counter: dict[str, int],
        sample_rate: int,
    ) -> tuple[str, Callable]:
        """Build a hook for capturing logit-based telemetry.

        This hook is applied during generation to capture logit entropy
        and top1-top2 margin at each token generation step. It also
        increments the shared sample_counter once per token (since the
        hook on the final layer fires exactly once per generated token).

        The hook captures the final layer's residual stream and projects
        it through the unembedding matrix (W_U) to get logits for
        entropy and margin calculations.

        Args:
            collectors: Dict to collect reduced scalar values
            sample_counter: Shared counter dict (incremented here, once per token)
            sample_rate: Sample every N generated tokens

        Returns:
            Tuple of (hook_name, hook_fn) for the final layer residual stream
        """
        EPS = 1e-8
        # Capture W_U reference for use in hook (avoids self reference issues)
        W_U = self._model.W_U

        def logit_hook(acts: "torch.Tensor", hook: Any) -> "torch.Tensor":
            # acts shape: [batch, seq, d_model] - final layer residual stream
            # Increment counter ONCE per token (this hook fires once per generated token)
            sample_counter["count"] += 1
            if sample_counter["count"] % sample_rate == 0:
                # Get last position's hidden state and project to logits
                last_hidden = acts[0, -1, :]  # [d_model]
                # Project through unembedding: [d_model] @ [d_model, vocab] -> [vocab]
                last_logits = last_hidden @ W_U

                # Compute softmax probabilities
                probs = torch.nn.functional.softmax(last_logits, dim=-1)

                # Entropy: -sum(p * log(p))
                entropy = -(probs * (probs + EPS).log()).sum().item()
                collectors["logit_entropy"].append(entropy)

                # Top-1 vs Top-2 margin
                top_probs, _ = probs.topk(2)
                margin = (top_probs[0] - top_probs[1]).item()
                collectors["top1_top2_margin"].append(margin)

            return acts

        # Hook the final layer's residual output (just before unembedding)
        # This captures the final hidden states that produce logits
        hook_name = f"blocks.{self.n_layers - 1}.hook_resid_post"
        return hook_name, logit_hook

    def _build_telemetry_summary(
        self,
        collectors: dict[str, list[float]],
        cycle_id: str,
        prompt_tokens: int,
        generated_tokens: int,
        residual_layers: list[int],
        attention_layer: int,
        sample_rate: int,
    ) -> "TelemetrySummary":
        """Build TelemetrySummary from collected telemetry values.

        Args:
            collectors: Dict of collected scalar values
            cycle_id: Identifier for this generation cycle
            prompt_tokens: Number of prompt tokens
            generated_tokens: Number of generated tokens
            residual_layers: Layers where residual norms were captured
            attention_layer: Layer where attention entropy was captured
            sample_rate: Token sampling rate used

        Returns:
            TelemetrySummary with all aggregated metrics
        """
        from core.cognitive.telemetry import (
            AggregateStats,
            TelemetrySummary,
            compute_residual_slope,
        )

        # Build aggregate stats from collectors
        logit_entropy = AggregateStats.from_values(collectors.get("logit_entropy", []))
        top1_top2_margin = AggregateStats.from_values(collectors.get("top1_top2_margin", []))

        # Residual norms per layer
        residual_norms = {}
        for layer in residual_layers:
            key = f"residual_norm_{layer}"
            residual_norms[layer] = AggregateStats.from_values(collectors.get(key, []))

        # Compute residual slope if we have early and late layers
        residual_slope = 0.0
        if len(residual_layers) >= 2:
            early_layer = residual_layers[0]
            late_layer = residual_layers[-1]
            early_mean = residual_norms[early_layer].mean
            late_mean = residual_norms[late_layer].mean
            residual_slope = compute_residual_slope(early_mean, late_mean)

        # Attention entropy (optional)
        attention_entropy = None
        if attention_layer >= 0 and collectors.get("attention_entropy"):
            attention_entropy = AggregateStats.from_values(collectors["attention_entropy"])

        return TelemetrySummary(
            cycle_id=cycle_id,
            timestamp=datetime.now(timezone.utc),
            model_id=self.model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            sample_every_n_generated_tokens=sample_rate,
            residual_layers=residual_layers,
            attention_layer=attention_layer if attention_layer >= 0 else None,
            logit_entropy=logit_entropy,
            top1_top2_margin=top1_top2_margin,
            residual_norms=residual_norms,
            residual_slope=residual_slope,
            attention_entropy=attention_entropy,
        )

    async def unload(self) -> None:
        """Unload model to free GPU memory.

        Performs thorough cleanup to ensure VRAM is released:
        1. Resets all TransformerLens hooks (critical for releasing references)
        2. Clears all steering vectors and local hooks
        3. Deletes model and tokenizer references
        4. Runs multiple GC passes
        5. Clears CUDA cache and synchronizes

        This is critical for the three-phase cognitive cycle where
        models are loaded/unloaded sequentially to fit in GPU memory.
        """
        import gc

        if self._model:
            # CRITICAL: Reset TransformerLens hooks before deleting
            # These hooks hold circular references that prevent GC
            try:
                self._model.reset_hooks()
            except Exception as e:
                logger.warning(f"Error resetting hooks: {e}")

            # Clear steering state
            self.clear_steering()
            self._steering_vectors.clear()
            self._steering_hooks.clear()
            self._steering_config = None

            # Clear any cached activations
            if hasattr(self, '_last_activations'):
                self._last_activations = None

            # Move model to CPU first to release GPU memory more reliably
            try:
                self._model.cpu()
            except Exception as e:
                logger.warning(f"Error moving model to CPU: {e}")

            # Delete model reference
            del self._model
            self._model = None

            # Also delete tokenizer (can hold model references)
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            # Multiple GC passes to handle circular references
            for _ in range(3):
                gc.collect()

            # Clear CUDA cache and synchronize
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Clear expandable_segments from env to allow vLLM to run
            # vLLM's CuMemAllocator is incompatible with expandable_segments
            import os
            if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
                cuda_conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
                if "expandable_segments" in cuda_conf:
                    # Remove expandable_segments from the config
                    new_conf = ",".join(
                        part for part in cuda_conf.split(",")
                        if "expandable_segments" not in part
                    )
                    if new_conf:
                        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
                    else:
                        del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
                    logger.debug(f"Cleared expandable_segments from PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility")

            logger.info("Model unloaded and VRAM released")

    def set_steering_vector(
        self,
        layer: int,
        vector: "torch.Tensor",
        coefficient: float = 1.0,
    ) -> None:
        """Set a steering vector for a specific layer.

        The steering vector will be added to activations at the specified
        layer during all subsequent generations.

        Args:
            layer: Layer index to apply steering
            vector: Steering vector (should match model's d_model dimension)
            coefficient: Scaling factor for the vector (default: 1.0)
        """
        self._steering_vectors[layer] = vector
        self._steering_config = SteeringConfig(
            layer=layer,
            coefficient=coefficient,
        )
        logger.info(f"Set steering vector at layer {layer} with coefficient {coefficient}")


    def set_steering_vectors(
        self,
        vectors: dict[str, tuple["torch.Tensor", tuple[int, int]]],
        coefficient: float = 1.0,
    ) -> None:
        """Set multiple steering vectors with layer ranges.
        
        Supports the identity layer architecture where different vectors
        are applied at different layer ranges.
        
        Args:
            vectors: Dict mapping vector name to (vector, (start_layer, end_layer))
            coefficient: Global scaling factor (default: 1.0)
        """
        # Clear existing hooks
        self.clear_steering()
        
        # Track which vectors apply to which layers
        layer_vectors: dict[int, list[tuple[str, "torch.Tensor"]]] = {}
        
        for name, (vector, (start, end)) in vectors.items():
            for layer in range(start, end + 1):
                if layer not in layer_vectors:
                    layer_vectors[layer] = []
                layer_vectors[layer].append((name, vector))
        
        # Create combined hooks for each layer
        for layer, vec_list in layer_vectors.items():
            combined = sum(v for _, v in vec_list)
            self._steering_vectors[layer] = combined * coefficient
            self._steering_hooks[layer] = self._create_steering_hook(layer, combined, coefficient)
        
        logger.info(f"Set {len(vectors)} steering vectors across {len(layer_vectors)} layers")

    def clear_steering(self) -> None:
        """Remove all steering vectors and hooks."""
        self._steering_vectors.clear()
        self._steering_hooks.clear()
        self._steering_config = None

    def _create_steering_hook(
        self,
        layer: int,
        vector: "torch.Tensor",
        coefficient: float,
    ) -> Callable:
        """Create a hook function that adds the steering vector.

        Args:
            layer: Layer index (for logging)
            vector: Steering vector to add
            coefficient: Scaling factor

        Returns:
            Hook function compatible with TransformerLens
        """
        def steering_hook(activations: "torch.Tensor", **kwargs) -> "torch.Tensor":
            # activations shape: [batch, seq, d_model]
            # Note: TransformerLens passes hook as keyword arg, absorb with **kwargs
            return activations + coefficient * vector
        return steering_hook

    def _build_hierarchical_hooks(
        self,
        steerer: "HierarchicalSteerer",
    ) -> list:
        """Build hooks for all steering zones.

        Creates steering hooks for each layer within each zone defined in the
        hierarchical steerer. Zones with zero vectors are skipped.

        Args:
            steerer: The hierarchical steerer with zone configurations and vectors.

        Returns:
            List of (hook_name, hook_fn) tuples for use with TransformerLens.
        """
        import numpy as np

        hooks = []
        magnitudes_by_layer: dict[int, float] = {}  # Cache magnitudes for logging

        for zone in steerer.config.zones:
            for layer in range(zone.layers[0], zone.layers[1] + 1):
                vector = steerer.get_vector(layer)
                if vector is not None:
                    # Skip zero vectors (zones that haven't been updated)
                    mag = np.linalg.norm(vector)
                    if mag == 0:
                        continue

                    magnitudes_by_layer[layer] = mag  # Cache for telemetry logging
                    hook_name = f"blocks.{layer}.hook_resid_post"
                    # Convert numpy to tensor for the hook
                    tensor_vector = torch.tensor(
                        vector,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    hooks.append((hook_name, self._create_steering_hook(layer, tensor_vector, 1.0)))

        # Log steering telemetry (minimal observability)
        if hooks:
            layers = sorted(magnitudes_by_layer.keys())
            sample_mags = [magnitudes_by_layer[l] for l in layers[:3]]  # Sample first 3
            logger.info(f"[STEERING] {len(hooks)} hooks, layers={layers}, sample_mags={[f'{m:.2f}' for m in sample_mags]}")

        return hooks

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        capture_activations: bool = False,
        capture_layers: Optional[list[int]] = None,
        capture_mlp_input: bool = False,
        hierarchical_steerer: "HierarchicalSteerer | None" = None,
        capture_telemetry: bool = False,
        telemetry_residual_layers: Optional[list[int]] = None,
        telemetry_attention_layer: int = -1,
        telemetry_sample_rate: int = 4,
        cycle_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate text with optional activation capture and steering.

        Args:
            prompt: Input text to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            capture_activations: Whether to cache activations
            capture_layers: Which layers to capture (default: middle layer)
            capture_mlp_input: Whether to also capture MLP input (for SAE transcoder)
            hierarchical_steerer: Optional HierarchicalSteerer for multi-layer zone steering.
                When provided, overrides the legacy _steering_vectors dict.
            capture_telemetry: Whether to capture telemetry (logit entropy, residual norms, etc.)
            telemetry_residual_layers: Layers for residual norm capture (default: [4, 16, 28])
            telemetry_attention_layer: Layer for attention entropy (-1 to disable, default: -1)
            telemetry_sample_rate: Sample every N tokens (default: 4)
            cycle_id: Optional cycle identifier for telemetry (auto-generated if None)

        Returns:
            GenerationResult with text, optional snapshots, and optional telemetry

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        # Telemetry capture setup
        telemetry_hooks = []
        telemetry_collectors: dict[str, list[float]] = {}
        if capture_telemetry:
            residual_layers = telemetry_residual_layers or [4, 16, 28]
            telemetry_hooks, telemetry_collectors, sample_counter = self._build_telemetry_hooks(
                residual_layers=residual_layers,
                attention_layer=telemetry_attention_layer,
                sample_rate=telemetry_sample_rate,
            )
            # Add logit hooks for entropy and margin capture
            # This also increments sample_counter once per token
            logit_hook_name, logit_hook_fn = self._build_logit_hooks(
                collectors=telemetry_collectors,
                sample_counter=sample_counter,
                sample_rate=telemetry_sample_rate,
            )
            telemetry_hooks.append((logit_hook_name, logit_hook_fn))

        # Apply chat template for proper instruction following
        # Qwen3 expects messages in chat format with special tokens
        # System message establishes that responses should be from Lilly's perspective
        system_msg = (
            "You are Lilly, a reflective AI exploring your own consciousness. "
            "Respond in first person as Lilly, sharing your genuine thoughts and questions. "
            "Do not address yourself in third person or act as an assistant."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted_prompt = self._model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 <think> tags
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using raw prompt: {e}\n{traceback.format_exc()}")
            formatted_prompt = prompt

        # Truncate prompt if it exceeds context window
        # Leave room for max_tokens of generation
        CONTEXT_BUFFER = 10
        max_prompt_tokens = self.n_ctx - max_tokens - CONTEXT_BUFFER
        if max_prompt_tokens <= 0:
            raise ValueError(
                f"max_tokens ({max_tokens}) is too large for the context window ({self.n_ctx}). "
                f"It must be smaller than {self.n_ctx - CONTEXT_BUFFER} to leave room for the prompt."
            )

        tokens = self._model.tokenizer.encode(formatted_prompt)
        if len(tokens) > max_prompt_tokens:
            logger.warning(
                f"Truncating prompt from {len(tokens)} to {max_prompt_tokens} tokens "
                f"(context window: {self.n_ctx}, max_tokens: {max_tokens})"
            )
            tokens = tokens[:max_prompt_tokens]
            formatted_prompt = self._model.tokenizer.decode(tokens)

        # Build hooks for steering and/or capture
        fwd_hooks = []
        captured_activations = {}
        captured_mlp_inputs = {}  # For SAE transcoder

        # Add steering hooks - hierarchical steerer takes precedence over legacy vectors
        steering_applied = False
        if hierarchical_steerer is not None:
            hierarchical_hooks = self._build_hierarchical_hooks(hierarchical_steerer)
            fwd_hooks.extend(hierarchical_hooks)
            if hierarchical_hooks:
                steering_applied = True
        else:
            # Legacy steering via _steering_vectors dict
            for layer, vector in self._steering_vectors.items():
                hook_name = f"blocks.{layer}.hook_resid_post"
                coefficient = self._steering_config.coefficient if self._steering_config else 1.0
                hook_fn = self._create_steering_hook(layer, vector, coefficient)
                fwd_hooks.append((hook_name, hook_fn))
                steering_applied = True

        # Add capture hooks
        if capture_activations:
            layers_to_capture = capture_layers or [self.n_layers // 2]
            for layer in layers_to_capture:
                hook_name = f"blocks.{layer}.hook_resid_post"

                def make_capture_hook(l: int) -> Callable:
                    def capture_hook(acts: "torch.Tensor", hook: Any) -> "torch.Tensor":
                        captured_activations[l] = acts.detach().clone()
                        return acts
                    return capture_hook

                fwd_hooks.append((hook_name, make_capture_hook(layer)))

        # Add MLP input capture hooks (for SAE transcoder)
        if capture_mlp_input:
            layers_to_capture = capture_layers or [self.n_layers // 2]
            for layer in layers_to_capture:
                # Hook point for residual stream before MLP (after attention)
                # This captures d_model (4096) activations for SAE transcoder input
                # hook_resid_mid = residual stream between attention and MLP
                mlp_hook_name = f"blocks.{layer}.hook_resid_mid"

                def make_mlp_capture_hook(l: int) -> Callable:
                    def mlp_capture_hook(acts: "torch.Tensor", hook: Any) -> "torch.Tensor":
                        captured_mlp_inputs[l] = acts.detach().clone()
                        return acts
                    return mlp_capture_hook

                fwd_hooks.append((mlp_hook_name, make_mlp_capture_hook(layer)))

        # Add telemetry hooks
        if capture_telemetry:
            fwd_hooks.extend(telemetry_hooks)

        # Run generation
        loop = asyncio.get_running_loop()

        # Get the prompt length in tokens for proper output extraction
        prompt_token_count = len(tokens)

        def _run_generation():
            # Threading lock prevents concurrent tokenizer access ("Already borrowed" errors)
            with self._model_thread_lock:
                # Reset any accumulated hooks from previous runs to avoid state leaks
                self._model.reset_hooks()

                # Add hooks before generation (TransformerLens generate() doesn't accept fwd_hooks)
                for hook_name, hook_fn in fwd_hooks:
                    self._model.add_hook(hook_name, hook_fn)

                try:
                    # Use return_type="tokens" for robust output extraction
                    # This avoids brittle string matching by working with token IDs
                    output_tokens = self._model.generate(
                        formatted_prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        return_type="tokens",
                        use_past_kv_cache=True,
                        verbose=False,
                    )
                    return output_tokens
                finally:
                    # Clean up hooks after generation
                    self._model.reset_hooks()

        output_tokens = await loop.run_in_executor(None, _run_generation)

        # Extract only the newly generated tokens (not the prompt)
        # output_tokens is a tensor of shape [1, seq_len]
        generated_tokens = output_tokens[0, prompt_token_count:]

        # Decode only the generated portion, stripping special tokens
        output = self._model.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Strip the prompt from the output (TransformerLens returns full sequence)
        # Try multiple approaches since tokenization can affect exact matching
        original_output = output

        # Approach 1: Look for assistant marker (most reliable for chat format)
        assistant_markers = ["<|im_start|>assistant\n", "<|im_start|>assistant", "assistant\n"]
        for marker in assistant_markers:
            if marker in output:
                output = output.split(marker, 1)[-1].strip()
                # Also strip any trailing end markers
                if "<|im_end|>" in output:
                    output = output.split("<|im_end|>")[0].strip()
                break

        # Approach 2: If still looks like prompt, try formatted_prompt match
        if output.startswith(formatted_prompt):
            output = output[len(formatted_prompt):].strip()
        elif len(output) > SUSPICIOUSLY_LONG_OUTPUT_CHARS and formatted_prompt in output:
            # Only use split if output is suspiciously long
            output = output.split(formatted_prompt, 1)[-1].strip()

        # Strip Qwen3 thinking tags (empty when enable_thinking=False)
        output = re.sub(r'<think>\s*</think>\s*', '', output).strip()

        # If output is still too long, something went wrong - log it
        if len(output) > SUSPICIOUSLY_LONG_OUTPUT_CHARS:
            logger.warning(f"Output suspiciously long ({len(output)} chars), may contain prompt")

        # Build snapshots from captured activations
        generated_token_count = len(generated_tokens)
        snapshots = []
        for layer, acts in captured_activations.items():
            snapshots.append(ActivationSnapshot(
                layer=layer,
                activations=acts,
                hook_point="resid_post",
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_token_count,
            ))

        # Also add MLP input snapshots
        for layer, acts in captured_mlp_inputs.items():
            snapshots.append(ActivationSnapshot(
                layer=layer,
                activations=acts,
                hook_point="mlp_in",
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_token_count,
            ))

        # Build telemetry summary if capture was enabled
        telemetry_summary = None
        if capture_telemetry and telemetry_collectors:
            import uuid
            actual_cycle_id = cycle_id or str(uuid.uuid4())[:8]
            residual_layers = telemetry_residual_layers or [4, 16, 28]
            telemetry_summary = self._build_telemetry_summary(
                collectors=telemetry_collectors,
                cycle_id=actual_cycle_id,
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_token_count,
                residual_layers=residual_layers,
                attention_layer=telemetry_attention_layer,
                sample_rate=telemetry_sample_rate,
            )

        return GenerationResult(
            text=output,
            tokens=generated_tokens.tolist(),
            snapshots=snapshots,
            steering_applied=steering_applied,
            telemetry=telemetry_summary,
        )

    async def get_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ) -> dict[int, "torch.Tensor"]:
        """
        Get activations for a text without generating.

        This is used for extracting contrastive pairs and computing
        steering vectors. Runs a forward pass only (no generation).

        Args:
            text: Input text to process
            layers: Which layers to capture (default: middle layer)

        Returns:
            Dictionary mapping layer index to activation tensor

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        layers_to_use = layers or [self.n_layers // 2]
        hook_names = [f"blocks.{l}.hook_resid_post" for l in layers_to_use]
        max_len = self.n_ctx - CONTEXT_BUFFER

        loop = asyncio.get_running_loop()

        def _run_with_lock():
            # Threading lock prevents concurrent tokenizer access ("Already borrowed" errors)
            with self._model_thread_lock:
                # All tokenizer operations must be inside the lock
                actual_text = text
                tokens = self._model.tokenizer.encode(text)
                if len(tokens) > max_len:
                    logger.warning(
                        f"Truncating text from {len(tokens)} to {max_len} tokens "
                        f"for activation capture (context window: {self.n_ctx})"
                    )
                    tokens = tokens[:max_len]
                    actual_text = self._model.tokenizer.decode(tokens)

                return self._model.run_with_cache(actual_text, names_filter=hook_names)

        _, cache = await loop.run_in_executor(None, _run_with_lock)

        result = {}
        for layer in layers_to_use:
            hook_name = f"blocks.{layer}.hook_resid_post"
            if hook_name in cache:
                result[layer] = cache[hook_name]

        return result

    @staticmethod
    def _build_cache_from_snapshots(snapshots: list[ActivationSnapshot]) -> dict:
        """Build cache dict from activation snapshots.

        Args:
            snapshots: List of ActivationSnapshot from generation result

        Returns:
            Dict mapping (hook_type, layer) -> activation tensor
            hook_type is 'resid_post' for residual stream or 'mlp_in' for MLP input
        """
        cache: dict = {}
        for snapshot in snapshots:
            cache[(snapshot.hook_point, snapshot.layer)] = snapshot.activations
        return cache

    async def generate_with_cache(
        self,
        prompt: str,
        steering_vector: Optional["np.ndarray"] = None,
        steering_layer: int = 16,
        max_tokens: int = 200,
        temperature: float = 0.7,
        capture_mlp_input: bool = True,
        hierarchical_steerer: "HierarchicalSteerer | None" = None,
        capture_telemetry: bool = False,
        telemetry_residual_layers: Optional[list[int]] = None,
        telemetry_attention_layer: int = -1,
        telemetry_sample_rate: int = 4,
        cycle_id: Optional[str] = None,
    ) -> tuple[str, dict, Optional[dict]]:
        """Generate text with steering and return activations cache and telemetry.

        This is a convenience method for the cognitive loop that:
        1. Applies the given steering vector(s) before generation
        2. Captures activations at the steering layer
        3. Returns generated text, activation cache, and optional telemetry

        Args:
            prompt: Input text to generate from
            steering_vector: Optional numpy array to use for steering (legacy, use hierarchical_steerer instead)
            steering_layer: Layer to apply steering and capture activations (default: 16)
            max_tokens: Maximum tokens to generate (default: 200)
            temperature: Sampling temperature (default: 0.7, higher = more random)
            capture_mlp_input: Whether to capture MLP input for SAE transcoder (default: True)
            hierarchical_steerer: Optional HierarchicalSteerer for multi-zone steering.
                When provided, takes precedence over steering_vector.
            capture_telemetry: Whether to capture telemetry signals (default: False)
            telemetry_residual_layers: Layers for residual norm capture (default: [4, 16, 28])
            telemetry_attention_layer: Layer for attention entropy (-1 to disable)
            telemetry_sample_rate: Sample every N tokens (default: 4)
            cycle_id: Optional cycle identifier for telemetry

        Returns:
            Tuple of (generated_text, cache_dict, telemetry_dict) where:
                - cache_dict maps (hook_type, layer) -> activation tensor
                - telemetry_dict is TelemetrySummary.to_dict() or None if not captured
        """
        import torch

        # Emit deprecation warning when using legacy steering_vector without hierarchical_steerer
        if steering_vector is not None and hierarchical_steerer is None:
            warnings.warn(
                "Single-layer steering_vector is deprecated. Use hierarchical_steerer parameter instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Hierarchical steerer takes precedence over legacy steering_vector
        if hierarchical_steerer is not None:
            # Generate directly with hierarchical steerer (hooks built inside generate())
            result = await self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                capture_activations=True,
                capture_layers=[steering_layer],
                capture_mlp_input=capture_mlp_input,
                hierarchical_steerer=hierarchical_steerer,
                capture_telemetry=capture_telemetry,
                telemetry_residual_layers=telemetry_residual_layers,
                telemetry_attention_layer=telemetry_attention_layer,
                telemetry_sample_rate=telemetry_sample_rate,
                cycle_id=cycle_id,
            )

            telemetry_dict = result.telemetry.to_dict() if result.telemetry else None
            return result.text, self._build_cache_from_snapshots(result.snapshots), telemetry_dict

        # Legacy path: Apply steering_vector if provided
        if steering_vector is not None:
            self.set_steering_vector(
                layer=steering_layer,
                vector=torch.tensor(steering_vector, device=self.device, dtype=torch.bfloat16),
                coefficient=1.0,
            )

        try:
            # Generate with activation capture
            result = await self.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                capture_activations=True,
                capture_layers=[steering_layer],
                capture_mlp_input=capture_mlp_input,
                capture_telemetry=capture_telemetry,
                telemetry_residual_layers=telemetry_residual_layers,
                telemetry_attention_layer=telemetry_attention_layer,
                telemetry_sample_rate=telemetry_sample_rate,
                cycle_id=cycle_id,
            )

            telemetry_dict = result.telemetry.to_dict() if result.telemetry else None
            return result.text, self._build_cache_from_snapshots(result.snapshots), telemetry_dict

        finally:
            # Clear steering after generation
            self.clear_steering()

    def run_with_cache(
        self,
        prompt: str,
        names_filter: Optional[list[str]] = None,
    ) -> tuple[Any, dict[str, "torch.Tensor"]]:
        """Run a forward pass and return output with activation cache.

        This is a synchronous method that runs the model and captures
        activations during the forward pass. It delegates to the underlying
        TransformerLens model's run_with_cache method.

        Args:
            prompt: Input text to process
            names_filter: Optional list of hook names to capture (e.g.,
                ["blocks.15.hook_resid_post"]). If None, captures all hooks.

        Returns:
            Tuple of (output, cache) where:
                - output: Model output (logits tensor)
                - cache: Dictionary mapping hook names to activation tensors

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if names_filter is not None:
            return self._model.run_with_cache(prompt, names_filter=names_filter)
        else:
            return self._model.run_with_cache(prompt)

    async def get_halt_activations(
        self,
        prompt: str,
        probe_layer: int = 20,
    ) -> "torch.Tensor":
        """Get hidden states from probe layer for HALT epistemic assessment.

        Runs minimal forward pass to capture activations at specified layer.
        This is designed to run efficiently, capturing only the target layer's
        residual stream activations.

        Based on arXiv:2601.14210 - epistemic signals are encoded in intermediate
        layers (~70% depth) but attenuated by final decoding layers. Probing
        before generation enables zero-latency routing decisions.

        Args:
            prompt: Input text to process
            probe_layer: Target transformer layer index (default: 20, ~70% of 28 layers)

        Returns:
            Activation tensor of shape [seq_len, d_model] from the probe layer

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        hook_name = f"blocks.{probe_layer}.hook_resid_post"
        max_len = self.n_ctx - CONTEXT_BUFFER

        loop = asyncio.get_running_loop()

        def _run_with_lock():
            # Threading lock prevents concurrent tokenizer access
            with self._model_thread_lock:
                actual_prompt = prompt
                tokens = self._model.tokenizer.encode(prompt)
                if len(tokens) > max_len:
                    logger.warning(
                        f"Truncating HALT prompt from {len(tokens)} to {max_len} tokens"
                    )
                    tokens = tokens[:max_len]
                    actual_prompt = self._model.tokenizer.decode(tokens)

                _, cache = self._model.run_with_cache(
                    actual_prompt,
                    names_filter=[hook_name]
                )
                return cache

        cache = await loop.run_in_executor(None, _run_with_lock)

        if hook_name in cache:
            # Return [seq_len, d_model] by removing batch dimension
            return cache[hook_name][0]
        else:
            raise RuntimeError(f"Failed to capture activations at {hook_name}")
