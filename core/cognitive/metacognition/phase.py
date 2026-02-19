"""Metacognition phase implementation.

Provides bird's-eye pattern detection across cognitive cycles using a local
Gemma-2-9B-IT model via vLLM.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import TYPE_CHECKING, Optional

from core.cognitive.metacognition.buffer import CycleSummary, MetacognitionBuffer
from core.cognitive.metacognition.memory import MetacognitionMemory
from core.cognitive.metacognition.prompts import (
    METACOGNITION_SYSTEM_PROMPT,
    build_metacognition_prompt,
    parse_memory_updates,
)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.cognitive.state import CognitiveState
    from integrations.liquidsoap.client import LiquidsoapClient

logger = logging.getLogger(__name__)

# Minimum GPU memory required (GB)
MIN_GPU_MEMORY_GB = 18.0

# vLLM availability flag
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# Torch for GPU memory checks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


class MetacognitionPhase:
    """Local metacognition phase using Gemma-2-9B-IT.

    Runs after the Continuity phase to analyze patterns across the last N cycles
    and provide guidance for the next cycle.

    Attributes:
        _settings: Application settings
        _liquidsoap: TTS client for narration
        _memory: Persistent memory blocks
        _buffer: Rolling buffer of cycle summaries
        _engine: vLLM engine (loaded on demand)
    """

    def __init__(
        self,
        settings: "Settings",
        liquidsoap: Optional["LiquidsoapClient"] = None,
    ):
        """Initialize the metacognition phase.

        Args:
            settings: Application settings
            liquidsoap: Optional TTS client for narrating guidance
        """
        self._settings = settings
        self._liquidsoap = liquidsoap
        self._memory = MetacognitionMemory.load()

        # Load buffer from disk (persists across restarts)
        buffer_size = getattr(settings, "metacognition_buffer_size", 5)
        self._buffer = MetacognitionBuffer.load(maxlen=buffer_size)

        self._engine: Optional["LLM"] = None
        self._sampling_params: Optional["SamplingParams"] = None

    @property
    def is_enabled(self) -> bool:
        """Check if metacognition is enabled in settings."""
        return getattr(self._settings, "metacognition_enabled", False)

    @property
    def buffer(self) -> MetacognitionBuffer:
        """Access the cycle buffer."""
        return self._buffer

    def add_cycle_summary(self, summary: CycleSummary) -> None:
        """Add a cycle summary to the buffer.

        Args:
            summary: Summary of the completed cycle
        """
        self._buffer.append(summary)
        self._buffer.save()  # Persist across restarts
        logger.info(
            "Metacognition buffer: %d/%d cycles (persisted)",
            len(self._buffer),
            self._buffer.maxlen,
        )

    async def run(self, state: "CognitiveState") -> Optional[str]:
        """Run the metacognition phase.

        Args:
            state: Current cognitive state (for context)

        Returns:
            Guidance string if generated, None if skipped or failed
        """
        if not self.is_enabled:
            logger.debug("Metacognition disabled, skipping")
            return None

        if not VLLM_AVAILABLE:
            logger.warning("Metacognition skipped: vLLM not available")
            return None

        # Check warmup period
        min_cycles = getattr(self._settings, "metacognition_min_cycles", 3)
        if len(self._buffer) < min_cycles:
            logger.debug(
                "Metacognition skipped: warming up (%d/%d cycles)",
                len(self._buffer),
                min_cycles,
            )
            return None

        # Check GPU memory
        if not self._check_gpu_available():
            logger.warning("Metacognition skipped: insufficient GPU memory")
            return None

        try:
            # Load model
            await self._load_model()

            # Run inference
            response = await self._run_inference()

            # Parse and apply updates
            updates = parse_memory_updates(response)
            if updates:
                self._memory.update_from_response(updates)
                self._memory.save()
                logger.info("Metacognition updated %d memory blocks", len(updates))

            # Get and narrate guidance
            guidance = self._memory.get_guidance()
            if guidance:
                await self._narrate_guidance(guidance)
                logger.info("Metacognition guidance: %s", guidance[:100])

            return guidance

        except Exception as e:
            logger.error("Metacognition failed: %s", e, exc_info=True)
            return None

        finally:
            await self._unload_model()

    def _check_gpu_available(self) -> bool:
        """Check if sufficient GPU memory is available.

        Returns:
            True if GPU has enough free memory, False otherwise
        """
        if not TORCH_AVAILABLE:
            return False

        try:
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                if free_mem < MIN_GPU_MEMORY_GB:
                    logger.debug(
                        "GPU memory insufficient: %.1f GB free, need %.1f GB",
                        free_mem,
                        MIN_GPU_MEMORY_GB,
                    )
                    return False
                return True
        except Exception as e:
            logger.debug("GPU check failed: %s", e)

        return False

    async def _load_model(self) -> None:
        """Load the vLLM engine for inference."""
        if self._engine is not None:
            return

        model_id = getattr(
            self._settings, "metacognition_model", "google/gemma-2-9b-it"
        )
        logger.info("Loading metacognition model: %s", model_id)

        loop = asyncio.get_running_loop()

        def _load():
            return LLM(
                model=model_id,
                tensor_parallel_size=1,
                max_model_len=8192,
                trust_remote_code=True,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
            )

        self._engine = await loop.run_in_executor(None, _load)
        self._sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            stop=["```\n\n", "</json>"],
        )
        logger.info("Metacognition model loaded")

    async def _unload_model(self) -> None:
        """Unload the model and release GPU memory."""
        if self._engine is None:
            return

        logger.info("Unloading metacognition model...")

        # Try sleep mode if available
        try:
            if hasattr(self._engine, "sleep"):
                self._engine.sleep(level=2)
        except Exception as e:
            logger.debug("vLLM sleep failed: %s", e)

        self._engine = None
        self._sampling_params = None

        # Aggressive memory cleanup
        for _ in range(3):
            gc.collect()

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info("GPU memory after metacognition unload: %.1f GB free", free_mem)

    async def _run_inference(self) -> str:
        """Run inference on the cycle buffer.

        Returns:
            Raw LLM response text
        """
        if self._engine is None:
            raise RuntimeError("Model not loaded")

        # Build the prompt
        prompt = build_metacognition_prompt(self._buffer, self._memory)

        # Format as chat
        messages = [
            {"role": "system", "content": METACOGNITION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self._format_chat_prompt(messages)

        # Run generation
        loop = asyncio.get_running_loop()

        def _generate():
            outputs = self._engine.generate([chat_prompt], self._sampling_params)
            return outputs[0].outputs[0].text

        response = await loop.run_in_executor(None, _generate)
        logger.debug("Metacognition response length: %d chars", len(response))

        return response

    def _format_chat_prompt(self, messages: list[dict]) -> str:
        """Format messages as chat prompt.

        Uses Gemma's chat template format.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                # Gemma treats system as user context
                parts.append(f"<start_of_turn>user\n[System]\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

        # Add model turn for generation
        parts.append("<start_of_turn>model\n")

        return "\n".join(parts)

    async def _narrate_guidance(self, guidance: str) -> None:
        """Narrate the guidance with the metacognitive voice.

        Args:
            guidance: Guidance text to narrate
        """
        if not self._liquidsoap or not guidance.strip():
            return

        # Get voice setting
        voice = getattr(
            self._settings,
            "voice_metacognitive",
            "hf://kyutai/tts-voices/expresso/ex03-ex01_calm_001_channel1_1143s.wav",
        )

        # Brief intro before guidance
        intro = "A moment of reflection."
        full_text = f"{intro} {guidance}"

        try:
            await self._liquidsoap.narrate(full_text, voice=voice)
        except Exception as e:
            logger.warning("Failed to narrate metacognition guidance: %s", e)

    def get_guidance(self) -> str:
        """Get current guidance for injection into Generation.

        Returns:
            Current guidance string, or empty if none
        """
        return self._memory.get_guidance()

    def get_memory(self) -> MetacognitionMemory:
        """Access the memory blocks.

        Returns:
            Current MetacognitionMemory instance
        """
        return self._memory
