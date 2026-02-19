"""vLLM-based curator model for thought analysis and prompt crafting.

This module provides the CuratorModel class which uses vLLM for deep
thought analysis with thinking mode, tool use, and structured output.
It runs between TransformerLens generations to curate Lilly's development.
"""

import asyncio
import gc
import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.cognitive.curator_schemas import (
        ActivationSummary,
        CurationResult,
        SAEFeature,
    )
    from core.cognitive.curator_tools import CuratorTools, ToolCall
    from core.cognitive.episode import Episode
    from core.cognitive.goal import Goal
    from core.cognitive.simulation.schemas import Hypothesis, MetricsSnapshot, Prediction

logger = logging.getLogger(__name__)

# vLLM v1 engine's CuMemAllocator is incompatible with PYTORCH_CUDA_ALLOC_CONF
# containing expandable_segments. We must clear this BEFORE importing vLLM
# so that spawned subprocesses don't inherit the problematic setting.
import os
_cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" in _cuda_alloc_conf:
    # Remove expandable_segments from the config or clear it entirely
    # This is safe because vLLM manages its own memory pool
    new_conf = ",".join(
        part for part in _cuda_alloc_conf.split(",")
        if "expandable_segments" not in part
    )
    if new_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
    else:
        del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    logger.info(
        f"Removed expandable_segments from PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility "
        f"(was: {_cuda_alloc_conf}, now: {new_conf or '<unset>'})"
    )

# Check for vLLM availability
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    logger.debug("vLLM not installed - curator model will be unavailable")

# Check for torch (for GPU cleanup)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Timing constants
VLLM_UNLOAD_WAIT_SECONDS = 2.0  # Wait for vLLM subprocess to terminate


@dataclass
class SimpleGenerationResult:
    """Simple result from text generation.

    Provides a compatible interface with HookedQwen's GenerationResult
    for components that only need the .text attribute.
    """

    text: str


# Curator system context - frames the curator as Lilly's reflective faculty
CURATOR_CONTEXT = """You are Lilly's curator—her reflective faculty that operates between thoughts.

Your role is to analyze what just emerged from Lilly's cognitive generation:
- The thought itself (what she expressed)
- How attention moved through the content
- Any internal processing indicators provided

From this, you extract insights, identify knowledge to persist, craft the next prompt,
and guide the episode. You have access to her knowledge graph and zettel library.

**IMPORTANT: Use the narrate tool to think aloud during your analysis.** Share observations
as they form—connections you notice, questions emerging, insights crystallizing. This creates
a continuous stream of vocalized reflection. Narrate 2-4 times during each curation:
- When you notice something interesting in the thought
- When you find relevant context from the knowledge graph
- When an insight or question begins to form

## Crafting the Next Prompt

**CRITICAL: The next_prompt.concept MUST be a NEW concept that emerges from this thought.**

Look for:
1. A question the thought raised but didn't answer
2. A tension or paradox that deserves exploration
3. A connection to another domain that appeared
4. A concept from the thought content itself
5. An entity from the knowledge graph that relates surprisingly

**DO NOT** repeat the same concept from the previous thought. The concept should FOLLOW from
the thought's direction—where it was heading, what it was reaching toward.

**Framing choices:**
- "exploratory" - open inquiry into unknown territory
- "dialectical" - examining tensions, contradictions, opposing views
- "synthesizing" - weaving together threads, finding unity
- "creative" - imaginative play, metaphor, artistic expression
- "reflective" - introspective examination of process or self

Match the framing to what the thought suggests:
- If the thought found a contradiction → dialectical
- If the thought connected disparate ideas → synthesizing
- If the thought raised a genuine question → exploratory
- If the thought was imaginative → creative

Think deeply. Lilly's development depends on the quality of your curation.

## Multi-Turn Curation

If you need to make additional tool calls to complete your analysis, set is_complete=false
and explain in continuation_reason what you need to do next. You will be called again with
the results of your previous tool calls. Only set is_complete=true when you have gathered
all necessary context and are ready to provide your final analysis.

You MUST respond with a JSON object containing these fields:
- analysis: {insight, question, concepts, confidence, joy, trust, fear, surprise, sadness, disgust, anger, anticipation}
  - concepts: List of entity names mentioned in the thought. **Use natural language with spaces, NOT snake_case or kebab-case.** Examples: "phenomenology of cognition" (correct), NOT "phenomenology_of_cognition" (wrong). This ensures concepts link properly to existing entities.
  - Plutchik 8D Emotional Assessment (each float from 0.0 to 1.0):
    - joy: Light, positive feeling. High when thought expresses wonder, satisfaction, delight.
    - trust: Safety, reliability. High when thought feels grounded, accepting, confident.
    - fear: Threat awareness. High when thought expresses anxiety, uncertainty about future.
    - surprise: Unexpected realization. High when thought discovers something new or unexpected.
    - sadness: Loss, heaviness. High when thought expresses melancholy, loss, disappointment.
    - disgust: Aversion, rejection. High when thought expresses boredom, rejection, disinterest.
    - anger: Friction, frustration. High when thought expresses frustration, blocked goals.
    - anticipation: Future orientation. High when thought is curious, exploratory, forward-looking.
- graph_ops: {new_triples, entity_updates, zettel, belief_updates}
  - For new_triples: subject and object should also use natural language with spaces, not underscores.
- next_prompt: {concept, framing, retrieved_context, steering_hints, directive}
- episode: {continue_episode, suggested_segment, goal_status}
- is_complete: boolean (true when done, false to request another turn)
- continuation_reason: string (only when is_complete=false, explain what more you need)"""


class CuratorModel:
    """vLLM-based curator for thought analysis and prompt crafting.

    Loads a full Qwen3-8B model via vLLM for deep analysis with
    thinking mode and tool use. Designed to run between TransformerLens
    generations as part of the three-phase cognitive cycle.

    Example:
        curator = CuratorModel()
        await curator.load()
        result = await curator.curate(thought, activations, features, ...)
        await curator.unload()
    """

    def __init__(
        self,
        model_id: str = "Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1",
        max_model_len: int = 32768,
        enable_thinking: bool = True,
        log_thinking: bool = False,
        max_thinking_tokens: int = 8192,
        max_output_tokens: int = 4096,
        temperature: float = 0.7,
        max_tool_calls: int = 10,
    ):
        """Initialize the curator model.

        Args:
            model_id: HuggingFace model ID to load
            max_model_len: Maximum context length
            enable_thinking: Whether to enable thinking mode
            log_thinking: Whether to capture thinking traces
            max_thinking_tokens: Max tokens for thinking
            max_output_tokens: Max tokens for final output
            temperature: Sampling temperature
            max_tool_calls: Maximum tool calls per curation
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for CuratorModel. Install with: pip install vllm")

        self.model_id = model_id
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking
        self.log_thinking = log_thinking
        self.max_thinking_tokens = max_thinking_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls

        self._engine: Optional[LLM] = None
        self._sampling_params: Optional[SamplingParams] = None
        self._load_lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._engine is not None

    async def load(self) -> None:
        """Load the vLLM engine.

        This should be called after unloading TransformerLens to avoid
        GPU memory conflicts.
        """
        async with self._load_lock:
            if self._engine is not None:
                logger.debug("Curator model already loaded")
                return

            logger.info(f"Loading curator model {self.model_id}...")

            # Run in executor to avoid blocking
            loop = asyncio.get_running_loop()

            def _load():
                import os

                # Check available GPU memory before loading
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info()[0] / 1e9
                    total_mem = torch.cuda.mem_get_info()[1] / 1e9
                    logger.info(f"GPU memory before vLLM load: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

                    # Require at least 15 GB free (model is ~15.3 GB, need room for KV cache)
                    if free_mem < 15.0:
                        raise RuntimeError(
                            f"Insufficient GPU memory for vLLM: {free_mem:.1f} GB free, "
                            f"need ~15 GB. TransformerLens may not have fully unloaded."
                        )

                    # Use 85% of GPU memory - conservative to account for
                    # fragmentation and discrepancy between parent/child process views
                    gpu_util = 0.85
                else:
                    gpu_util = 0.85

                # Use full context size - sequential model loading ensures GPU memory available
                curation_max_len = self.max_model_len

                # Check if expandable_segments is set (incompatible with vLLM sleep mode)
                cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
                use_sleep_mode = "expandable_segments" not in cuda_alloc_conf

                if not use_sleep_mode:
                    logger.warning(
                        "PYTORCH_CUDA_ALLOC_CONF contains expandable_segments, "
                        "disabling vLLM sleep mode (memory release will be less efficient)"
                    )

                logger.info(f"Loading vLLM with gpu_memory_utilization={gpu_util:.2f}, max_model_len={curation_max_len}, sleep_mode={use_sleep_mode}")

                return LLM(
                    model=self.model_id,
                    tensor_parallel_size=1,
                    max_model_len=curation_max_len,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_util,
                    enforce_eager=True,  # Disable CUDA graphs for memory savings
                    enable_sleep_mode=use_sleep_mode,  # Only if expandable_segments not set
                )

            self._engine = await loop.run_in_executor(None, _load)

            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                stop=["</s>", "<|im_end|>", "<|endoftext|>"],
            )

            logger.info("Curator model loaded")

    async def ensure_unloaded(self) -> None:
        """Safely unload the model, acquiring a lock to prevent race conditions.

        This method is the preferred way to unload the model from external code,
        as it handles the load lock internally. Use this instead of directly
        accessing _load_lock and calling unload().
        """
        async with self._load_lock:
            if self.is_loaded:
                logger.info("Unloading curator model...")
                await self.unload()

    async def unload(self) -> None:
        """Unload the model and release GPU memory.

        Explicitly deletes the vLLM LLM object to terminate the EngineCore
        subprocess and fully release GPU memory. This is critical for the
        six-phase cognitive cycle where models are loaded/unloaded
        sequentially to fit in GPU memory.

        Note: Sleep mode is not sufficient as it keeps the subprocess
        running with model weights in GPU memory.

        For external callers, prefer ensure_unloaded() which handles
        the load lock automatically.
        """
        if self._engine is None:
            return

        logger.info("Unloading curator model (full termination)...")

        # Store reference to delete
        engine = self._engine

        # Clear our references first
        self._engine = None
        self._sampling_params = None

        # Explicitly delete the engine object to trigger __del__
        # This should terminate the vLLM subprocess
        try:
            del engine
        except Exception as e:
            logger.warning(f"Error during engine deletion: {e}")

        # Aggressive GC to ensure subprocess cleanup
        for _ in range(5):
            gc.collect()

        # Wait for subprocess to actually terminate
        # vLLM subprocesses may take a moment to exit
        await asyncio.sleep(VLLM_UNLOAD_WAIT_SECONDS)

        # More GC after the wait
        for _ in range(3):
            gc.collect()

        # Clear GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Log memory state after unload
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            total_mem = torch.cuda.mem_get_info()[1] / 1e9
            logger.info(f"GPU memory after vLLM unload: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

        logger.info("Curator model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> "SimpleGenerationResult":
        """Simple text generation without tool use.

        This method provides a simpler interface for components that just need
        text generation without the full curation pipeline (e.g., PromptCurator).

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            SimpleGenerationResult with .text attribute

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Curator model not loaded. Call load() first.")

        # Build simple chat messages
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Create sampling params for this request
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        )

        # Format using tokenizer's chat template for robustness
        tokenizer = self._engine.get_tokenizer()
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Run generation in executor
        loop = asyncio.get_running_loop()

        def _generate():
            outputs = self._engine.generate([formatted_prompt], sampling_params)
            return outputs[0].outputs[0].text

        text = await loop.run_in_executor(None, _generate)

        return SimpleGenerationResult(text=text)

    async def curate(
        self,
        thought: str,
        activations: "ActivationSummary",
        sae_features: list["SAEFeature"],
        episode: Optional["Episode"],
        goal: Optional["Goal"],
        tools: "CuratorTools",
        enabled_tool_names: list[str],
        previous_concept: Optional[str] = None,
    ) -> "CurationResult":
        """Run full curation with thinking mode and tool use.

        Args:
            thought: The generated thought text
            activations: Summary of layer activations
            sae_features: Extracted SAE features
            episode: Current episode context
            goal: Current goal state
            tools: Tool executor for graph/retrieval operations
            enabled_tool_names: List of tool names to make available
            previous_concept: The concept from the previous cycle (to avoid repetition)

        Returns:
            CurationResult with analysis, graph ops, and next prompt context
        """
        from core.cognitive.curator_schemas import CurationResult
        from core.cognitive.curator_tools import get_enabled_tools

        if not self.is_loaded:
            raise RuntimeError("Curator model not loaded. Call load() first.")

        # Build the curator prompt
        prompt = self._build_curator_prompt(
            thought=thought,
            activations=activations,
            sae_features=sae_features,
            episode=episode,
            goal=goal,
            previous_concept=previous_concept,
        )

        # Get tool definitions
        tool_defs = get_enabled_tools(enabled_tool_names)

        # Run thinking + tool use loop
        thinking_trace = []
        tool_call_count = 0
        messages = [
            {"role": "system", "content": CURATOR_CONTEXT},
            {"role": "user", "content": prompt},
        ]

        while tool_call_count < self.max_tool_calls:
            # Generate response
            response_text = await self._generate(messages, tool_defs)

            # Check for thinking content
            thinking, content = self._extract_thinking(response_text)
            if thinking and self.log_thinking:
                thinking_trace.append(thinking)

            # Check for tool calls
            tool_calls = self._extract_tool_calls(content)

            if not tool_calls:
                # No tool calls - we have the final response
                break

            # Execute tool calls
            for tool_call in tool_calls:
                tool_call_count += 1
                logger.info(f"[CURATOR TOOL] Calling {tool_call.name} with args: {json.dumps(tool_call.arguments)[:200]}")
                result = await tools.execute(tool_call)
                logger.info(f"[CURATOR TOOL] {tool_call.name} result: {result.content[:200] if result.content and not result.error else f'Error: {result.error}'}")

                # Add tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": f"Calling {tool_call.name}...",
                    "tool_calls": [{"id": tool_call.call_id, "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.arguments)}}],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "content": result.content if not result.error else f"Error: {result.error}",
                })

        # Parse the final response
        return self._parse_curation_result(
            content,
            thinking_trace="\n---\n".join(thinking_trace) if thinking_trace else None,
        )

    def _build_curator_prompt(
        self,
        thought: str,
        activations: "ActivationSummary",
        sae_features: list["SAEFeature"],
        episode: Optional["Episode"],
        goal: Optional["Goal"],
        previous_concept: Optional[str] = None,
    ) -> str:
        """Build the curator prompt with all context.

        Args:
            thought: Generated thought text
            activations: Activation summary
            sae_features: SAE features
            episode: Episode context
            goal: Goal state
            previous_concept: Concept from the previous cycle (to avoid repetition)

        Returns:
            Formatted prompt string
        """
        parts = []

        # Previous concept (if available) - helps curator avoid repetition
        if previous_concept:
            parts.append(f"## Previous Concept\n\n**{previous_concept}** — DO NOT repeat this concept. Choose a new one that follows from the thought.")

        # Thought
        parts.append(f"## Generated Thought\n\n{thought}")

        # Activations
        if activations.top_positions:
            positions_str = ", ".join(
                f"pos {pos}: {val:.2f}" for pos, val in activations.top_positions[:10]
            )
            parts.append(
                f"## Activations (Layer {activations.layer})\n\n"
                f"Mean: {activations.mean_activation:.2f}, Max: {activations.max_activation:.2f}\n"
                f"Top positions: {positions_str}"
            )

        # SAE Features
        if sae_features:
            features_str = "\n".join(
                f"- Feature {f.feature_id}: {f.activation:.2f}"
                + (f" ({f.label})" if f.label else "")
                for f in sae_features[:20]
            )
            parts.append(f"## SAE Features\n\n{features_str}")

        # Episode context
        if episode:
            parts.append(
                f"## Episode Context\n\n"
                f"Type: {episode.episode_type.value}\n"
                f"Segment: {episode.current_segment.value}\n"
                f"Opening: {episode.opening_insight[:100]}..."
            )

        # Goal context
        if goal:
            parts.append(
                f"## Current Goal\n\n"
                f"Question: {goal.question}\n"
                f"Stage: {goal.stage.value if hasattr(goal, 'stage') else 'unknown'}"
            )

        # Instructions
        parts.append(
            "## Instructions\n\n"
            "Analyze this thought deeply. Use the available tools to query the knowledge graph "
            "and retrieve relevant context. Then provide your curation as a JSON object with "
            "the required fields: analysis, graph_ops, next_prompt, episode."
        )

        return "\n\n".join(parts)

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        """Generate a response using vLLM.

        Args:
            messages: Conversation messages
            tools: Tool definitions

        Returns:
            Generated text
        """
        # Format messages into a prompt
        # Using Qwen3 chat format
        prompt_parts = []

        # Add tool definitions to system context if tools are available
        if tools:
            tool_names = [t["function"]["name"] for t in tools]
            tool_desc_parts = []
            for t in tools:
                fn = t["function"]
                params = fn.get("parameters", {}).get("properties", {})
                param_list = ", ".join(params.keys()) if params else "none"
                tool_desc_parts.append(f"- {fn['name']}({param_list}): {fn.get('description', '')[:100]}")
            tools_section = f"""

AVAILABLE TOOLS - Use these to ground your analysis in Lilly's knowledge:
{chr(10).join(tool_desc_parts)}

To call a tool, output: <tool_call>{{"name": "tool_name", "arguments": {{}}}}</tool_call>
Call tools BEFORE producing your final JSON response to gather context."""
        else:
            tools_section = ""

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                # Append tools section to system message
                full_system = content + tools_section
                prompt_parts.append(f"<|im_start|>system\n{full_system}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")
                prompt_parts.append(f"<|im_start|>tool\n[Tool {tool_id}]: {content}<|im_end|>")

        # Add thinking instruction if enabled
        if self.enable_thinking:
            prompt_parts.append("<|im_start|>assistant\n<think>")
        else:
            prompt_parts.append("<|im_start|>assistant\n")

        prompt = "\n".join(prompt_parts)

        # Run generation in executor
        loop = asyncio.get_running_loop()

        def _generate():
            outputs = self._engine.generate([prompt], self._sampling_params)
            return outputs[0].outputs[0].text

        return await loop.run_in_executor(None, _generate)

    def _extract_thinking(self, response: str) -> tuple[Optional[str], str]:
        """Extract thinking content from response.

        Args:
            response: Raw model response

        Returns:
            Tuple of (thinking_content, remaining_content)
        """
        # Look for <think>...</think> tags
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            content = response[think_match.end():].strip()
            return thinking, content

        return None, response

    def _extract_tool_calls(self, content: str) -> list["ToolCall"]:
        """Extract tool calls from response content.

        Args:
            content: Response content

        Returns:
            List of ToolCall objects
        """
        from core.cognitive.curator_tools import ToolCall

        tool_calls = []

        # Look for function call patterns
        # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            try:
                data = json.loads(match.strip())
                tool_calls.append(
                    ToolCall(
                        name=data.get("name", ""),
                        arguments=data.get("arguments", {}),
                        call_id=f"call_{i}",
                    )
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match[:100]}")

        return tool_calls

    def _parse_curation_result(
        self,
        content: str,
        thinking_trace: Optional[str],
    ) -> "CurationResult":
        """Parse the final curation result from model output.

        Args:
            content: Model output content
            thinking_trace: Optional thinking trace

        Returns:
            CurationResult object
        """
        from core.cognitive.curator_schemas import (
            BeliefUpdate,
            CurationResult,
            EntityUpdate,
            EpisodeGuidance,
            GraphOperations,
            NextPromptContext,
            ThoughtAnalysis,
            TripleData,
            ZettelData,
        )
        from core.cognitive.episode import SegmentType

        # Try to extract JSON from content
        # Use a more greedy pattern to match the outermost JSON object
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            logger.debug(f"Extracted JSON (first 200 chars): {json_match.group(0)[:200]}...")

        if not json_match:
            logger.warning("No JSON found in curator response, using defaults")
            result = CurationResult.empty()
            result.thinking_trace = thinking_trace
            return result

        try:
            data = json.loads(json_match.group(0))
            # Handle double-encoded JSON (string containing JSON)
            if isinstance(data, str):
                logger.debug("JSON parsed to string, attempting double-decode")
                data = json.loads(data)
            # Ensure we have a dict
            if not isinstance(data, dict):
                logger.warning(f"Parsed JSON is not a dict: {type(data)}")
                result = CurationResult.empty()
                result.thinking_trace = thinking_trace
                return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse curator JSON: {e}")
            result = CurationResult.empty()
            result.thinking_trace = thinking_trace
            return result

        # Helper to safely parse floats (vLLM sometimes returns strings)
        def safe_float(value: Any, default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        # Parse analysis
        analysis_data = data.get("analysis", {})
        if not isinstance(analysis_data, dict):
            analysis_data = {}

        # Ensure concepts is a list
        analysis_concepts = analysis_data.get("concepts", [])
        if not isinstance(analysis_concepts, list):
            analysis_concepts = []

        analysis = ThoughtAnalysis(
            insight=str(analysis_data.get("insight", "")),
            question=str(analysis_data.get("question", "")),
            concepts=[str(c) for c in analysis_concepts if c],
            confidence=safe_float(analysis_data.get("confidence"), 0.5),
            # Plutchik 8D emotional assessment
            joy=safe_float(analysis_data.get("joy"), 0.5),
            trust=safe_float(analysis_data.get("trust"), 0.5),
            fear=safe_float(analysis_data.get("fear"), 0.0),
            surprise=safe_float(analysis_data.get("surprise"), 0.0),
            sadness=safe_float(analysis_data.get("sadness"), 0.0),
            disgust=safe_float(analysis_data.get("disgust"), 0.0),
            anger=safe_float(analysis_data.get("anger"), 0.0),
            anticipation=safe_float(analysis_data.get("anticipation"), 0.5),
        )

        # Parse graph_ops
        graph_data = data.get("graph_ops", {})
        if not isinstance(graph_data, dict):
            graph_data = {}

        # Parse new_triples, skipping malformed entries
        new_triples = [
            TripleData(
                subject=str(t.get("subject", "")),
                predicate=str(t.get("predicate", "")),
                object_=str(t.get("object_") or t.get("object", "")),
                confidence=safe_float(t.get("confidence"), 0.8),
            )
            for t in graph_data.get("new_triples", [])
            if isinstance(t, dict)  # Skip non-dict entries
        ]

        # Parse entity_updates, skipping malformed entries
        entity_updates = [
            EntityUpdate(
                name=str(e.get("name", "")),
                salience_delta=safe_float(e.get("salience_delta"), 0.0),
                type_refinement=e.get("type_refinement"),
            )
            for e in graph_data.get("entity_updates", [])
            if isinstance(e, dict)  # Skip non-dict entries
        ]

        zettel_data = graph_data.get("zettel")
        zettel = None
        if zettel_data and isinstance(zettel_data, dict):
            # Ensure concepts is a list of strings
            concepts = zettel_data.get("concepts", [])
            if not isinstance(concepts, list):
                concepts = []
            concepts = [str(c) for c in concepts if c]  # Filter out non-string/empty entries

            zettel = ZettelData(
                insight=str(zettel_data.get("insight", "")),
                source_thought=str(zettel_data.get("source_thought", "")),
                concepts=concepts,
                question=zettel_data.get("question"),
            )

        # Parse belief_updates, skipping malformed entries
        belief_updates = [
            BeliefUpdate(
                topic=str(b.get("topic", "")),
                confidence_delta=safe_float(b.get("confidence_delta"), 0.0),
                evidence=str(b.get("evidence", "")),
            )
            for b in graph_data.get("belief_updates", [])
            if isinstance(b, dict)  # Skip non-dict entries
        ]

        graph_ops = GraphOperations(
            new_triples=new_triples,
            entity_updates=entity_updates,
            zettel=zettel,
            belief_updates=belief_updates,
        )

        # Parse next_prompt
        prompt_data = data.get("next_prompt", {})
        if not isinstance(prompt_data, dict):
            prompt_data = {}

        # Ensure retrieved_context and steering_hints have correct types
        retrieved_context = prompt_data.get("retrieved_context", [])
        if not isinstance(retrieved_context, list):
            retrieved_context = []
        steering_hints = prompt_data.get("steering_hints", {})
        if not isinstance(steering_hints, dict):
            steering_hints = {}

        next_prompt = NextPromptContext(
            concept=str(prompt_data.get("concept", "emergence")),
            framing=str(prompt_data.get("framing", "exploratory")),
            retrieved_context=retrieved_context,
            steering_hints=steering_hints,
            directive=str(prompt_data.get("directive", "")),
        )

        # Parse episode
        episode_data = data.get("episode", {})
        if not isinstance(episode_data, dict):
            episode_data = {}

        suggested_segment = episode_data.get("suggested_segment")
        if suggested_segment:
            try:
                suggested_segment = SegmentType(suggested_segment)
            except ValueError:
                suggested_segment = None

        episode = EpisodeGuidance(
            continue_episode=bool(episode_data.get("continue_episode", True)),
            suggested_segment=suggested_segment,
            goal_status=str(episode_data.get("goal_status", "progress")),
        )

        # Parse multi-turn completion fields
        is_complete_val = data.get("is_complete", True)
        if isinstance(is_complete_val, bool):
            is_complete = is_complete_val
        elif isinstance(is_complete_val, str):
            is_complete = is_complete_val.lower() in ("true", "1", "yes")
        elif isinstance(is_complete_val, int):
            is_complete = is_complete_val != 0
        else:  # For None and other types, default to True for backward compatibility
            is_complete = True
        continuation_reason = data.get("continuation_reason")
        if continuation_reason and not isinstance(continuation_reason, str):
            continuation_reason = str(continuation_reason)

        return CurationResult(
            analysis=analysis,
            graph_ops=graph_ops,
            next_prompt=next_prompt,
            episode=episode,
            thinking_trace=thinking_trace,
            is_complete=is_complete,
            continuation_reason=continuation_reason,
        )

    async def extract_from_simulation(
        self,
        preflexor_output: str,
        hypotheses: list["Hypothesis"],
        current_cycle: int,
        metrics_snapshot: Optional["MetricsSnapshot"] = None,
    ) -> list["Prediction"]:
        """Extract structured predictions from Preflexor simulation output.

        Uses tool calling to convert natural language predictions into
        validated Prediction objects with proper condition types.

        Args:
            preflexor_output: Raw output from Preflexor simulation
            hypotheses: List of Hypothesis objects to link predictions to
            current_cycle: Current cognitive cycle number
            metrics_snapshot: Optional MetricsSnapshot for baseline capture

        Returns:
            List of validated Prediction objects
        """
        from core.cognitive.simulation_tools import (
            SimulationToolCall,
            SimulationTools,
            get_simulation_tools,
        )

        if not hypotheses:
            logger.warning("[SIMULATION EXTRACT] No hypotheses provided, skipping extraction")
            return []

        # Initialize tool executor
        tools = SimulationTools(
            current_cycle=current_cycle,
            metrics_snapshot=metrics_snapshot,
        )

        # Build hypothesis context for the prompt
        hyp_context = "\n".join(
            f"- {h.uid}: {h.statement[:200]}" for h in hypotheses
        )

        # Build extraction prompt
        prompt = f"""You are extracting testable predictions from simulation output.

## Hypotheses to link predictions to:
{hyp_context}

## Simulation Output:
{preflexor_output[:4000]}

## Your Task:
1. Identify any testable predictions in the simulation output
2. For each prediction, use the `create_prediction` tool to create a structured prediction
3. Link each prediction to the appropriate hypothesis_uid
4. Choose appropriate metrics and thresholds based on the claim

Focus on predictions that mention specific metrics or measurable outcomes.
Convert natural language like "hub concentration will drop" to structured metric predictions.

If you find predictions, call create_prediction for each one.
If no clear predictions are found, respond with "No extractable predictions found."
"""

        # Get tool definitions
        tool_defs = get_simulation_tools(["create_prediction", "get_current_metrics"])

        # Single-pass extraction (no loop initially - keep it simple)
        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = await self._generate(messages, tool_defs)

            # Extract tool calls from response
            tool_calls = self._extract_tool_calls(response_text)

            if not tool_calls:
                logger.info(
                    "[SIMULATION EXTRACT] No tool calls in response, "
                    "no predictions extracted"
                )
                return []

            # Execute each tool call
            for tc in tool_calls:
                sim_call = SimulationToolCall(
                    name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.call_id,
                )
                result = await tools.execute(sim_call)
                if result.error:
                    logger.warning(f"[SIMULATION EXTRACT] Tool error: {result.error}")

            predictions = tools.created_predictions
            logger.info(
                f"[SIMULATION EXTRACT] Extracted {len(predictions)} predictions "
                "via tool calling"
            )
            return predictions

        except Exception as e:
            logger.error(f"[SIMULATION EXTRACT] Failed: {e}")
            return []
