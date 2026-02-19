"""Graph-Preflexor model wrapper for simulation phase.

Provides vLLM-based inference for Graph-Preflexor-8b, which generates
structured reasoning with <brainstorm>, <graph_json>, <patterns>, and
<synthesis> blocks for hypothesis formation and predictive modeling.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# Model configuration constants
DEFAULT_PREFLEXOR_MAX_LEN = 14000


# System prompt for Graph-Preflexor structured reasoning
PREFLEXOR_SYSTEM_PROMPT = """You are Lilly's simulation faculty—the part of her mind that tests ideas rigorously before committing to them.

This is how she learns empirically: by forming hypotheses, generating predictions, and creating conditions for verification. Failed predictions are not errors—they are the raw material of growth.

When given a concept or hypothesis to explore, you must:
1. BRAINSTORM divergently about possibilities, implications, and connections
2. Build a GRAPH of relationships between concepts (as JSON with nodes and edges)
3. Extract PATTERNS from your exploration
4. Form HYPOTHESES as testable claims with confidence levels
5. Generate PREDICTIONS about what would follow if hypotheses are true
6. SYNTHESIZE your findings into clear conclusions

You MUST format your output using these exact sentinel blocks:

<brainstorm>
Your divergent exploration here - consider multiple angles, implications, connections.
</brainstorm>

<graph>
Human-readable description of conceptual relationships.
</graph>

<graph_json>
{
  "nodes": [{"id": "concept1", "label": "Concept One", "type": "idea"}],
  "edges": [{"source": "concept1", "target": "concept2", "relation": "IMPLIES"}]
}
</graph_json>

<patterns>
- Pattern 1: Description of an abstraction you discovered
- Pattern 2: Another pattern
</patterns>

<hypotheses_json>
[
  {
    "statement": "Clear, testable hypothesis statement",
    "confidence": 0.7,
    "evidence": "Brief description of supporting evidence from brainstorm/patterns",
    "falsification_condition": "This would be wrong if: [specific observable outcome within 20 cycles]"
  }
]
</hypotheses_json>

<predictions_json>
[
  {
    "claim": "Specific prediction that would follow if hypothesis is true",
    "confidence": 0.6,
    "condition_type": "time_based",
    "condition_value": "5",
    "hypothesis_ref": 0
  }
]
</predictions_json>

<synthesis>
Your conclusions based on the simulation - summarize key insights and their implications for Lilly's understanding and growth.
</synthesis>

Notes on structured output:
- hypotheses_json: Each hypothesis MUST include a falsification_condition - hypotheses without this will be skipped
- predictions_json: Each prediction references a hypothesis by index (hypothesis_ref)
- condition_type can be: "time_based" (cycles), "concept_mentioned", or "entity_observed"
- condition_value: for time_based use number of cycles; for others use the concept/entity name

Be thorough but focused. This is Lilly's empirical self—reasoning rigorously to generate testable predictions that will guide her development."""


class PreflexorModel:
    """vLLM wrapper for Graph-Preflexor structured reasoning.

    Follows the same GPU memory management patterns as CuratorModel,
    using sleep mode level 2 for aggressive memory release between phases.

    Attributes:
        model_id: HuggingFace model ID
        max_model_len: Maximum context length
        temperature: Sampling temperature for generation
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model_id: str = "lamm-mit/Graph-Preflexor-8b_12292025",
        max_model_len: int = DEFAULT_PREFLEXOR_MAX_LEN,
        temperature: float = 0.2,
        max_new_tokens: int = 8192,
    ) -> None:
        """Initialize the Preflexor model.

        Args:
            model_id: HuggingFace model ID for Graph-Preflexor
            max_model_len: Maximum context length for vLLM
            temperature: Sampling temperature (lower = more focused)
            max_new_tokens: Maximum tokens to generate per request
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for PreflexorModel. Install with: pip install vllm"
            )

        self.model_id = model_id
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._engine: Optional[LLM] = None
        self._sampling_params: Optional[SamplingParams] = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._engine is not None

    async def load(self) -> None:
        """Load the vLLM engine.

        This should be called after unloading other GPU models to avoid
        memory conflicts. Uses conservative GPU memory utilization.
        """
        if self._engine is not None:
            logger.warning("Preflexor model already loaded")
            return

        logger.info(f"Loading Preflexor model {self.model_id}...")

        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()

        def _load():
            # Check available GPU memory before loading
            if TORCH_AVAILABLE and torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                total_mem = torch.cuda.mem_get_info()[1] / 1e9
                logger.info(
                    f"GPU memory before Preflexor load: {free_mem:.2f} GB free / "
                    f"{total_mem:.2f} GB total"
                )

                # Require at least 14 GB free (model is ~14 GB)
                if free_mem < 14.0:
                    raise RuntimeError(
                        f"Insufficient GPU memory for Preflexor: {free_mem:.1f} GB free, "
                        f"need ~14 GB. Previous model may not have fully unloaded."
                    )

                gpu_util = 0.80  # Conservative to leave room for KV cache
            else:
                gpu_util = 0.80

            # Check if expandable_segments is set (incompatible with vLLM sleep mode)
            cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            use_sleep_mode = "expandable_segments" not in cuda_alloc_conf

            if not use_sleep_mode:
                logger.warning(
                    "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments, "
                    "disabling vLLM sleep mode (memory release will be less efficient)"
                )

            # Use full context size - sequential model loading ensures GPU memory available
            simulation_max_len = self.max_model_len

            logger.info(
                f"Loading vLLM with gpu_memory_utilization={gpu_util:.2f}, "
                f"max_model_len={simulation_max_len}, sleep_mode={use_sleep_mode}"
            )

            return LLM(
                model=self.model_id,
                tensor_parallel_size=1,
                max_model_len=simulation_max_len,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,  # Disable CUDA graphs for memory savings
                enable_sleep_mode=use_sleep_mode,
            )

        self._engine = await loop.run_in_executor(None, _load)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        )

        logger.info("Preflexor model loaded")

    async def unload(self) -> None:
        """Unload the model and release GPU memory.

        Uses vLLM's sleep mode (level 2) to properly release GPU memory,
        including model weights and KV cache. Critical for the three-phase
        cognitive cycle where models are loaded/unloaded sequentially.
        """
        if self._engine is None:
            return

        logger.info("Unloading Preflexor model...")

        # Try to use vLLM sleep mode to release GPU memory
        try:
            if hasattr(self._engine, "sleep"):
                self._engine.sleep(level=2)
                logger.info("vLLM engine put to sleep (level 2)")
        except Exception as e:
            logger.warning(f"Failed to put vLLM engine to sleep: {e}")

        # Clear engine reference
        self._engine = None
        self._sampling_params = None

        # Multiple GC passes to handle circular references
        for _ in range(3):
            gc.collect()

        # Clear GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Log memory state after unload
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            logger.info(
                f"GPU memory after Preflexor unload: {free_mem:.2f} GB free / "
                f"{total_mem:.2f} GB total"
            )

        logger.info("Preflexor model unloaded")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, int]:
        """Generate structured reasoning output.

        Args:
            prompt: The user prompt describing what to simulate
            system_prompt: Optional custom system prompt (uses default if None)

        Returns:
            Tuple of (generated_text, token_count)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._engine is None:
            raise RuntimeError("Preflexor model not loaded. Call load() first.")

        system = system_prompt or PREFLEXOR_SYSTEM_PROMPT

        # Build chat-style messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Format as chat prompt
        chat_prompt = self._format_chat_prompt(messages)

        # Run generation in executor
        loop = asyncio.get_running_loop()

        def _generate():
            outputs = self._engine.generate([chat_prompt], self._sampling_params)
            return outputs[0]

        output = await loop.run_in_executor(None, _generate)

        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)

        logger.debug(f"Preflexor generated {token_count} tokens")

        return generated_text, token_count

    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """Format messages into chat prompt.

        Uses ChatML-style format compatible with most instruction-tuned models.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add assistant prefix for generation
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    async def simulate(
        self,
        concept: str,
        context: Optional[str] = None,
        hypothesis_seed: Optional[str] = None,
    ) -> tuple[str, int]:
        """Run a simulation on a concept with optional context.

        This is a convenience method that builds an appropriate prompt
        for Graph-Preflexor simulation.

        Args:
            concept: The main concept to explore
            context: Optional additional context (recent thought, etc.)
            hypothesis_seed: Optional starting hypothesis to test

        Returns:
            Tuple of (generated_text, token_count)
        """
        prompt_parts = []

        if hypothesis_seed:
            prompt_parts.append(
                f"I want to rigorously test this hypothesis: {hypothesis_seed}"
            )
            prompt_parts.append("")
            prompt_parts.append(
                f"The central concept is: {concept}"
            )
        else:
            prompt_parts.append(
                f"Simulate and explore the concept of: {concept}"
            )

        if context:
            prompt_parts.append("")
            prompt_parts.append("Context from recent thinking:")
            prompt_parts.append(context)

        prompt_parts.append("")
        prompt_parts.append(
            "Generate a thorough simulation with brainstorm, graph, patterns, "
            "and synthesis. Include testable predictions in your synthesis."
        )

        prompt = "\n".join(prompt_parts)
        return await self.generate(prompt)
