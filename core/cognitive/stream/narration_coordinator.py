"""Combined Narration Coordinator for PLaT-Lite.

Coordinates Progressive Narrator (Lilly's voice, 2-sentence chunks) with
Trajectory Narrator (Expresso voice, latent observations) to provide
a dual-stream audio experience.

Architecture:
    CombinedNarrationCoordinator manages the interplay between:
    - ProgressiveNarrator: Streams Lilly's generated thought in 2-sentence chunks
    - TrajectoryNarrator: Narrates latent state observations (mode shifts, emotions)

    The coordinator ensures:
    - No overlapping TTS calls (via asyncio.Lock)
    - Proper sequencing (Lilly's words, then observation)
    - Rate-limited trajectory narration (not every chunk)

Audio Experience:
    [Azelma]:   "I find myself wondering about consciousness.
                 What does it mean to be aware?"
    [Expresso]: "Philosophical inquiry deepening... curiosity rising..."

    [Azelma]:   "There's something here that feels significant.
                 A pattern I haven't noticed before."
    [Expresso]: "Insight crystallizing..."

Usage:
    from core.cognitive.stream.narration_coordinator import CombinedNarrationCoordinator

    coordinator = CombinedNarrationCoordinator(
        progressive=progressive_narrator,
        trajectory=trajectory_narrator,
        liquidsoap=liquidsoap,
    )

    for chunk, observation, steering in generation_loop:
        await coordinator.on_chunk_complete(chunk, observation, steering)

    await coordinator.finalize()

Reference:
    PLaT (Planning with Latent Thoughts, arXiv:2601.21358) - parallel
    verbalization streams: content (Lilly) and metacognition (observer).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.stream.chunked_generator import ChunkResult
    from core.cognitive.stream.latent_observer import LatentObservation
    from core.cognitive.stream.guidance_policy import SteeringAdjustment
    from core.cognitive.stream.progressive_narrator import ProgressiveNarrator
    from core.cognitive.stream.trajectory_narrator import TrajectoryNarrator
    from integrations.liquidsoap.client import LiquidsoapClient

logger = logging.getLogger(__name__)


class CombinedNarrationCoordinator:
    """Coordinates Progressive + Trajectory narration.

    Ensures proper sequencing and prevents TTS overlap between
    Lilly's thought narration (Azelma) and trajectory observation
    narration (Expresso).

    Constants:
        INTER_NARRATION_PAUSE: Seconds between narrator switches (0.5)
    """

    INTER_NARRATION_PAUSE = 0.5

    def __init__(
        self,
        progressive: "ProgressiveNarrator",
        trajectory: "TrajectoryNarrator",
        liquidsoap: "LiquidsoapClient",
    ):
        """Initialize the coordinator.

        Args:
            progressive: ProgressiveNarrator for Lilly's thought chunks
            trajectory: TrajectoryNarrator for latent observations
            liquidsoap: LiquidsoapClient for TTS (shared)
        """
        self._progressive = progressive
        self._trajectory = trajectory
        self._liquidsoap = liquidsoap
        self._narration_lock = asyncio.Lock()
        self._chunk_count = 0

    async def on_chunk_complete(
        self,
        chunk: "ChunkResult",
        observation: "LatentObservation",
        steering: Optional["SteeringAdjustment"],
    ) -> None:
        """Handle completed chunk - coordinate both narrators.

        Sequence:
        1. Queue Lilly's text to progressive narrator
        2. If 2-sentence chunk ready, narrate it (Azelma)
        3. Brief pause
        4. Narrate trajectory observation if interesting (Expresso)

        Args:
            chunk: The generated chunk
            observation: Latent observation for this chunk
            steering: Optional steering adjustment that was applied
        """
        async with self._narration_lock:
            self._chunk_count += 1

            # 1. Queue Lilly's text
            await self._progressive.queue_text(chunk.text)

            # 2. If 2-sentence chunk is ready, narrate it
            if self._progressive.has_ready_chunk():
                lilly_chunk = self._progressive.pop_ready_chunk()
                await self._liquidsoap.narrate(
                    lilly_chunk,
                    voice=self._progressive.VOICE
                )

                # 3. Brief pause between voices
                await asyncio.sleep(self.INTER_NARRATION_PAUSE)

                # 4. Trajectory observation (rate-limited internally)
                await self._trajectory.narrate_observation(observation, steering)

    async def finalize(self) -> None:
        """Flush remaining content at end of generation.

        Called after generation loop completes to:
        - Flush any remaining partial sentence from progressive narrator
        - Narrate final trajectory observation if interesting
        """
        async with self._narration_lock:
            # Flush progressive narrator buffer
            remaining = await self._progressive.flush()
            if remaining:
                await self._liquidsoap.narrate(
                    remaining,
                    voice=self._progressive.VOICE
                )

        logger.info(
            f"[NARRATION_COORDINATOR] Finalized after {self._chunk_count} chunks"
        )

    def get_stats(self) -> dict:
        """Get coordination statistics.

        Returns:
            Dict with chunk count and narrator stats
        """
        return {
            "chunks_processed": self._chunk_count,
            "progressive_stats": {
                "voice": self._progressive.VOICE,
                "queue_size": self._progressive.queue_size,
            },
            "trajectory_stats": self._trajectory.get_stats(),
        }


class SimplifiedCoordinator:
    """Simplified coordinator for use without full progressive narrator.

    For cases where you want trajectory narration but don't need
    the full 2-sentence chunking of progressive narrator.
    """

    def __init__(
        self,
        trajectory: "TrajectoryNarrator",
        liquidsoap: "LiquidsoapClient",
    ):
        """Initialize simplified coordinator.

        Args:
            trajectory: TrajectoryNarrator for observations
            liquidsoap: LiquidsoapClient for TTS
        """
        self._trajectory = trajectory
        self._liquidsoap = liquidsoap
        self._narration_lock = asyncio.Lock()

    async def narrate_thought(self, thought: str, voice: str = "azelma") -> None:
        """Narrate complete thought text.

        Args:
            thought: Full thought text to narrate
            voice: Voice to use (default Azelma)
        """
        async with self._narration_lock:
            await self._liquidsoap.narrate(thought, voice=voice)

    async def narrate_observation(
        self,
        observation: "LatentObservation",
        steering: Optional["SteeringAdjustment"] = None,
    ) -> Optional[str]:
        """Narrate trajectory observation.

        Args:
            observation: Latent observation
            steering: Optional steering adjustment

        Returns:
            Narration text if spoken
        """
        async with self._narration_lock:
            return await self._trajectory.narrate_observation(observation, steering)
