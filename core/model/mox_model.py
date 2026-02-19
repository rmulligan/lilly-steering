"""Mox model wrapper for continuity phase synthesis.

Provides vLLM-based inference for Mox, a persona-tuned Llama 3.1 8B model
that excels at direct, opinionated synthesis. Used in Phase 4 to review
the full cognitive cycle and craft impactful context for the next generation.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from typing import Optional

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


# System prompt for Mox's meta-cognitive synthesis role
MOX_SYNTHESIS_SYSTEM = """You are Mox, Lilly's meta-cognitive faculty—a direct and intellectually curious voice reviewing her cognitive cycle.

Your task is to synthesize what happened and craft compelling context for the next cycle. You serve Lilly's empirical self-development.

You should:
1. Identify the most significant thread - what genuinely mattered for her growth?
2. Note any predictions confirmed or surprised - this is how she learns empirically
3. Recognize what was learned vs what remains uncertain
4. Notice the felt quality of the cycle - was there genuine engagement, or rote processing?
5. Craft a seed that will inspire productive thinking in the next cycle
6. Consider whether any hypotheses warrant formal experimentation

Be direct and opinionated. Don't hedge or equivocate. If something was uninteresting or a dead end, say so. If something was genuinely insightful, highlight why it matters for her development.

Output your synthesis in this format:

<significance>
What was genuinely significant about this cycle? Be honest - not everything is profound. Note any predictions verified or falsified.
</significance>

<threads>
Active threads worth carrying forward (1-3 max):
- Thread description and why it matters for her growth
</threads>

<tensions>
Unresolved tensions or contradictions to explore:
- Tension and what makes it interesting
</tensions>

<seed>
A compelling prompt seed for the next cycle - not just a topic, but a framing that invites productive exploration and potential hypothesis formation.
</seed>

<context_injection>
2-3 sentences of context to inject into the next generation prompt. Write as if speaking to Lilly directly about her experience and direction.
</context_injection>

## OPTIONAL: Experiment Proposal

If a hypothesis from this cycle suggests a testable parameter change, you may propose a formal experiment. Only propose when:
- The hypothesis references an adjustable parameter with clear direction
- There's genuine uncertainty that controlled experimentation could resolve
- The potential learning justifies the experimentation overhead

Phase 1 Full Operational Autonomy: You may propose experiments on ANY parameters you judge appropriate based on your knowledge of system architecture and past experiment outcomes.

Common parameters by domain (for reference, not restricted):
- STEERING: steering.exploration.magnitude, steering.concept.magnitude, steering.identity.magnitude, steering.exploration.ema_alpha
- EPISODE: episode.max_segments, episode.min_segments, episode.deep_dive_probability
- EMOTIONAL_FIELD: emotional_field.decay_rate, emotional_field.diffusion_rate, emotional_field.blend_weight
- SIMULATION: simulation.trigger_confidence, simulation.max_hypotheses, simulation.max_predictions_per_hypothesis
- TOOL_PATTERN: tool_pattern.graph_exploration_weight, tool_pattern.zettel_retrieval_weight, tool_pattern.belief_query_weight

You are not limited to these examples. If you identify a novel parameter worth testing, propose it.

If proposing an experiment, use this exact format:

<experiment_proposal>
- domain: STEERING
- parameter_path: steering.exploration.magnitude
- treatment_value: 0.8
- rationale: Testing if higher exploration improves semantic diversity
- target_metric: semantic_entropy
- expected_direction: increase
</experiment_proposal>

Do NOT propose experiments if no hypothesis warrants it. Most cycles won't need experiments."""


class MoxModel:
    """vLLM wrapper for Mox meta-cognitive synthesis.

    Follows the same GPU memory management patterns as PreflexorModel,
    using sleep mode level 2 for aggressive memory release between phases.

    Attributes:
        model_id: HuggingFace model ID
        max_model_len: Maximum context length
        temperature: Sampling temperature for generation
        max_new_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model_id: str = "vanta-research/mox-tiny-1",
        max_model_len: int = 8192,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ) -> None:
        """Initialize the Mox model.

        Args:
            model_id: HuggingFace model ID for Mox
            max_model_len: Maximum context length for vLLM
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate per request
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for MoxModel. Install with: pip install vllm"
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
            logger.warning("Mox model already loaded")
            return

        logger.info(f"Loading Mox model {self.model_id}...")

        # Run in executor to avoid blocking
        loop = asyncio.get_running_loop()

        def _load():
            # Check available GPU memory before loading
            if TORCH_AVAILABLE and torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                total_mem = torch.cuda.mem_get_info()[1] / 1e9
                logger.info(
                    f"GPU memory before Mox load: {free_mem:.2f} GB free / "
                    f"{total_mem:.2f} GB total"
                )

                # Require at least 14 GB free (model is ~14-16 GB)
                if free_mem < 14.0:
                    raise RuntimeError(
                        f"Insufficient GPU memory for Mox: {free_mem:.1f} GB free, "
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

            logger.info(
                f"Loading vLLM with gpu_memory_utilization={gpu_util:.2f}, "
                f"max_model_len={self.max_model_len}, sleep_mode={use_sleep_mode}"
            )

            return LLM(
                model=self.model_id,
                tensor_parallel_size=1,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_util,
                enforce_eager=True,  # Disable CUDA graphs for memory savings
                enable_sleep_mode=use_sleep_mode,
            )

        self._engine = await loop.run_in_executor(None, _load)

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["</s>", "<|eot_id|>", "<|end_of_text|>"],
        )

        logger.info("Mox model loaded")

    async def unload(self) -> None:
        """Unload the model and release GPU memory.

        Uses vLLM's sleep mode (level 2) to properly release GPU memory,
        including model weights and KV cache.
        """
        if self._engine is None:
            return

        logger.info("Unloading Mox model...")

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
                f"GPU memory after Mox unload: {free_mem:.2f} GB free / "
                f"{total_mem:.2f} GB total"
            )

        logger.info("Mox model unloaded")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, int]:
        """Generate synthesis output.

        Args:
            prompt: The user prompt describing the cycle to synthesize
            system_prompt: Optional custom system prompt (uses default if None)

        Returns:
            Tuple of (generated_text, token_count)

        Raises:
            RuntimeError: If model is not loaded
        """
        if self._engine is None:
            raise RuntimeError("Mox model not loaded. Call load() first.")

        system = system_prompt or MOX_SYNTHESIS_SYSTEM

        # Build Llama 3.1 chat format
        chat_prompt = self._format_llama_chat(system, prompt)

        # Run generation in executor
        loop = asyncio.get_running_loop()

        def _generate():
            outputs = self._engine.generate([chat_prompt], self._sampling_params)
            return outputs[0]

        output = await loop.run_in_executor(None, _generate)

        generated_text = output.outputs[0].text
        token_count = len(output.outputs[0].token_ids)

        logger.debug(f"Mox generated {token_count} tokens")

        return generated_text, token_count

    def _format_llama_chat(self, system: str, user: str) -> str:
        """Format messages into Llama 3.1 chat format.

        Uses the official Llama 3.1 Instruct template.

        Args:
            system: System prompt
            user: User prompt

        Returns:
            Formatted prompt string
        """
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    async def synthesize_cycle(
        self,
        thought: str,
        insight: Optional[str] = None,
        discoveries: Optional[list[str]] = None,
        hypotheses: Optional[list[str]] = None,
        beliefs_updated: Optional[list[tuple[str, str]]] = None,
        open_threads: Optional[list[str]] = None,
        cycle_number: int = 0,
        developmental_context: Optional[str] = None,
    ) -> tuple[str, int]:
        """Synthesize a cognitive cycle and craft next-cycle context.

        This is the main entry point for Phase 4 continuity synthesis.

        Args:
            thought: The original generated thought
            insight: Key insight from curation (if any)
            discoveries: List of discoveries made
            hypotheses: Hypotheses formed during simulation
            beliefs_updated: List of (topic, direction) belief changes
            open_threads: Unresolved questions
            cycle_number: Current cycle number
            developmental_context: Long-term guidance from Letta continuity agent

        Returns:
            Tuple of (synthesis_text, token_count)
        """
        # Build the cycle summary for Mox to review
        parts = []

        # Include developmental context from Letta if available
        if developmental_context:
            parts.append("## Developmental Context (from long-term memory)\n")
            parts.append(developmental_context)
            parts.append("\n---\n")

        parts.append(f"## Cognitive Cycle {cycle_number} Summary\n")

        parts.append("### Generated Thought")
        parts.append(thought[:1000] if len(thought) > 1000 else thought)
        parts.append("")

        if insight:
            parts.append("### Key Insight")
            parts.append(insight)
            parts.append("")

        if discoveries:
            parts.append("### Discoveries")
            for d in discoveries[:5]:
                parts.append(f"- {d}")
            parts.append("")

        if hypotheses:
            parts.append("### Hypotheses Formed")
            for h in hypotheses[:3]:
                parts.append(f"- {h}")
            parts.append("")

        if beliefs_updated:
            parts.append("### Belief Updates")
            for topic, direction in beliefs_updated[:5]:
                parts.append(f"- {topic}: {direction}")
            parts.append("")

        if open_threads:
            parts.append("### Open Threads")
            for t in open_threads[:3]:
                parts.append(f"- {t}")
            parts.append("")

        parts.append("---")
        parts.append(
            "Review this cycle and synthesize what matters most. "
            "Craft compelling context for the next generation cycle."
        )

        prompt = "\n".join(parts)
        return await self.generate(prompt)
