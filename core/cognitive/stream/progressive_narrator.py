"""Progressive Narration for Eliminating Dead Air.

Streams narration progressively to eliminate silence gaps between
cognitive phases. While curation/integration processes in the background,
the narrator queues and plays thought chunks sequentially.

Architecture:
    ProgressiveNarrator maintains a priority queue of narration chunks.
    Generation output has highest priority, curation insights have
    lower priority. A background loop continuously narrates queued
    chunks, preventing dead air without modifying the generation loop.

Key Insight:
    TransformerLens generates all tokens at once (no mid-generation streaming).
    However, narration is slow (~2-3s per sentence) and curation is slow
    (~5-10s for analysis). We can narrate generation output while curation
    runs, eliminating the gap without streaming generation.

Usage:
    from core.cognitive.stream.progressive_narrator import ProgressiveNarrator

    narrator = ProgressiveNarrator(liquidsoap, settings)
    await narrator.start()

    # After generation phase
    await narrator.queue_thought(thought, source="generation")

    # During/after curation
    await narrator.queue_insight(insight, source="curation")

    # When done
    await narrator.stop()

Reference:
    Inspired by PLaT (Latent Chain-of-Thought as Planning) - verbalization
    streams independently of reasoning. The "planner" (generation) and
    "decoder" (narration) operate asynchronously.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config.settings import Settings
    from integrations.liquidsoap.client import LiquidsoapClient

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedChunk:
    """A narration chunk with priority for queue ordering.

    Lower priority value = higher priority (processed first).
    Uses @dataclass(order=True) for automatic comparison based
    on field order.

    Attributes:
        priority: Priority level (0=generation, 1=curation, 2=simulation)
        queued_at: Timestamp for FIFO within same priority
        text: The text to narrate (excluded from comparison)
        source: Origin of this chunk (excluded from comparison)
        voice: Optional voice override (excluded from comparison)
    """

    priority: int
    queued_at: datetime = field(compare=True)
    text: str = field(compare=False)
    source: str = field(compare=False)
    voice: Optional[str] = field(default=None, compare=False)


class ProgressiveNarrator:
    """Streams narration progressively to eliminate dead air.

    Maintains a priority queue of narration chunks and a background
    loop that continuously narrates queued content. Coordinates with
    SilenceMonitor to prevent overlapping narrations.

    Priority Levels (lower = higher priority):
        PRIORITY_GENERATION (0): Lilly's thoughts - highest priority
        PRIORITY_CURATION (1): Curator insights
        PRIORITY_SIMULATION (2): Simulation results

    Two Operating Modes:
        1. Autonomous: Uses queue_thought() with background streaming loop
        2. Coordinated: Uses queue_text() with CombinedNarrationCoordinator
           - Text buffered until 2 sentences ready
           - Coordinator calls has_ready_chunk()/pop_ready_chunk() to narrate

    Thread Safety:
        Uses asyncio.Lock to prevent overlapping TTS calls.
        The is_active property indicates when narration is in progress.
    """

    # Priority levels (lower = higher priority)
    PRIORITY_GENERATION = 0
    PRIORITY_CURATION = 1
    PRIORITY_SIMULATION = 2

    # Playback estimation constants
    WORDS_PER_SECOND = 2.5  # ~0.4s per word at normal TTS speed
    OVERLAP_FACTOR = 0.7  # Wait 70% of estimated duration before next chunk

    # Coordinated mode constants
    VOICE = "azelma"  # Default voice for Lilly's thoughts
    SENTENCES_PER_CHUNK = 2  # Sentences per narration chunk in coordinated mode

    def __init__(
        self,
        liquidsoap: "LiquidsoapClient",
        settings: Optional["Settings"] = None,
    ):
        """Initialize the progressive narrator.

        Args:
            liquidsoap: LiquidsoapClient for TTS narration
            settings: Application settings for voice configuration
        """
        self._liquidsoap = liquidsoap
        self._settings = settings
        self._queue: asyncio.PriorityQueue[PrioritizedChunk] = asyncio.PriorityQueue()
        self._streaming_task: Optional[asyncio.Task] = None
        self._active = False
        self._narrating = False  # True while actually narrating a chunk
        self._narration_lock = asyncio.Lock()

        # Coordinated mode state (for CombinedNarrationCoordinator)
        self._text_buffer: str = ""
        self._ready_chunks: list[str] = []

    @property
    def _voice_subject(self) -> str:
        """Get subject (Lilly) voice from settings or default."""
        if self._settings:
            return self._settings.voice_subject
        return "azelma"

    @property
    def _voice_curator(self) -> str:
        """Get curator voice from settings or default."""
        if self._settings:
            return self._settings.voice_curator
        return "eponine"

    @property
    def is_active(self) -> bool:
        """Whether the narrator is active (started and has content or is narrating)."""
        return self._active or self._narrating or not self._queue.empty()

    @property
    def queue_size(self) -> int:
        """Number of chunks waiting in the queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the background streaming loop."""
        if self._active:
            return

        self._active = True
        self._streaming_task = asyncio.create_task(self._stream_loop())
        logger.info("[PROGRESSIVE_NARRATOR] Started")

    async def stop(self) -> None:
        """Stop the background streaming loop gracefully.

        Allows current narration to complete and drains remaining queue.
        """
        if not self._active and (not self._streaming_task or self._streaming_task.done()):
            return

        self._active = False

        if self._streaming_task and not self._streaming_task.done():
            try:
                # The stream loop will exit once the queue is empty.
                await asyncio.wait_for(self._streaming_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("[PROGRESSIVE_NARRATOR] Stop timed out, cancelling task.")
                self._streaming_task.cancel()
                try:
                    await self._streaming_task
                except asyncio.CancelledError:
                    pass  # Expected after cancellation
            except asyncio.CancelledError:
                pass  # Task was already cancelled.

        logger.info("[PROGRESSIVE_NARRATOR] Stopped")

    async def queue_thought(
        self,
        thought: str,
        source: str = "generation",
    ) -> None:
        """Queue thought text for progressive narration.

        Splits the thought into sentences and queues each as a separate
        chunk with generation priority.

        Args:
            thought: The thought text to narrate
            source: Origin identifier (default "generation")
        """
        if not thought or not thought.strip():
            return

        chunks = self._split_by_sentences(thought)
        now = datetime.now(timezone.utc)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            # Slightly increment time to preserve order within same source
            queued_at = datetime.fromtimestamp(
                now.timestamp() + i * 0.001, tz=timezone.utc
            )

            await self._queue.put(
                PrioritizedChunk(
                    priority=self.PRIORITY_GENERATION,
                    queued_at=queued_at,
                    text=chunk.strip(),
                    source=source,
                    voice=self._voice_subject,
                )
            )

        logger.debug(
            f"[PROGRESSIVE_NARRATOR] Queued {len(chunks)} generation chunks"
        )

    async def queue_insight(
        self,
        insight: str,
        source: str = "curation",
    ) -> None:
        """Queue curator insight for narration.

        Curation insights have lower priority than generation chunks.

        Args:
            insight: The insight text to narrate
            source: Origin identifier (default "curation")
        """
        if not insight or not insight.strip():
            return

        chunks = self._split_by_sentences(insight)
        now = datetime.now(timezone.utc)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            queued_at = datetime.fromtimestamp(
                now.timestamp() + i * 0.001, tz=timezone.utc
            )

            await self._queue.put(
                PrioritizedChunk(
                    priority=self.PRIORITY_CURATION,
                    queued_at=queued_at,
                    text=f"She reflects: {chunk.strip()}",
                    source=source,
                    voice=self._voice_curator,
                )
            )

        logger.debug(
            f"[PROGRESSIVE_NARRATOR] Queued {len(chunks)} curation chunks"
        )

    async def queue_simulation_result(
        self,
        result: str,
        source: str = "simulation",
    ) -> None:
        """Queue simulation result for narration.

        Simulation results have the lowest priority.

        Args:
            result: The simulation result text
            source: Origin identifier (default "simulation")
        """
        if not result or not result.strip():
            return

        await self._queue.put(
            PrioritizedChunk(
                priority=self.PRIORITY_SIMULATION,
                queued_at=datetime.now(timezone.utc),
                text=result.strip(),
                source=source,
                voice=self._voice_curator,
            )
        )

        logger.debug("[PROGRESSIVE_NARRATOR] Queued simulation result")

    async def clear_queue(self) -> None:
        """Clear all pending chunks from the queue.

        Does not interrupt currently narrating chunk.
        """
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("[PROGRESSIVE_NARRATOR] Queue cleared")

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text into sentences for progressive narration.

        Handles common sentence terminators (.!?) and preserves
        sentences that are too short by merging them.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        # Split on sentence terminators followed by whitespace or end
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Filter empty and merge very short sentences
        result = []
        buffer = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) < 20 and buffer:
                # Merge short sentence with previous
                buffer = f"{buffer} {sentence}"
            elif len(sentence) < 20 and not buffer:
                # Start buffer with short sentence
                buffer = sentence
            else:
                # Flush buffer and add sentence
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(sentence)

        # Flush remaining buffer
        if buffer:
            result.append(buffer)

        return result

    async def _stream_loop(self) -> None:
        """Background loop that narrates queued chunks.

        Continuously pulls from the priority queue and narrates each
        chunk, estimating playback duration to prevent overlap.
        """
        while self._active or not self._queue.empty():
            try:
                # Wait for next chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        self._queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    if not self._active:
                        break
                    continue

                # Narrate with lock to prevent overlap
                async with self._narration_lock:
                    self._narrating = True
                    try:
                        voice = chunk.voice or self._voice_subject
                        await self._liquidsoap.narrate(chunk.text, voice=voice)

                        # Estimate playback duration and wait to prevent overlap
                        word_count = len(chunk.text.split())
                        estimated_duration = word_count / self.WORDS_PER_SECOND
                        wait_time = estimated_duration * self.OVERLAP_FACTOR

                        # Cap wait time to avoid blocking too long
                        wait_time = min(wait_time, 10.0)
                        await asyncio.sleep(wait_time)

                    except Exception as e:
                        logger.warning(
                            f"[PROGRESSIVE_NARRATOR] Narration failed: {e}"
                        )
                    finally:
                        self._narrating = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[PROGRESSIVE_NARRATOR] Stream loop error: {e}")

        logger.debug("[PROGRESSIVE_NARRATOR] Stream loop exited")

    async def acquire_lock(self) -> None:
        """Acquire the narration lock (for external coordination).

        Use this when SilenceMonitor needs to narrate without
        overlapping with ProgressiveNarrator.
        """
        await self._narration_lock.acquire()

    def release_lock(self) -> None:
        """Release the narration lock."""
        try:
            self._narration_lock.release()
        except RuntimeError:
            pass  # Lock wasn't held

    # -------------------------------------------------------------------------
    # Coordinated Mode Methods (for CombinedNarrationCoordinator)
    # -------------------------------------------------------------------------

    async def queue_text(self, text: str) -> None:
        """Queue text for coordinated narration (2-sentence chunks).

        Buffers incoming text and splits into ready chunks when
        SENTENCES_PER_CHUNK complete sentences are accumulated.
        Used by CombinedNarrationCoordinator instead of queue_thought().

        Args:
            text: Text to add to the buffer
        """
        if not text:
            return

        # Append to buffer
        if self._text_buffer:
            self._text_buffer += " " + text
        else:
            self._text_buffer = text

        # Check for complete sentences and move to ready chunks
        self._process_buffer()

    def _process_buffer(self) -> None:
        """Process buffer and move complete sentence groups to ready chunks."""
        # Split buffer into sentences
        sentences = self._extract_sentences(self._text_buffer)

        # Group into SENTENCES_PER_CHUNK chunks
        while len(sentences) >= self.SENTENCES_PER_CHUNK:
            chunk_sentences = sentences[: self.SENTENCES_PER_CHUNK]
            chunk_text = " ".join(chunk_sentences)
            self._ready_chunks.append(chunk_text)
            sentences = sentences[self.SENTENCES_PER_CHUNK :]

        # Reconstruct buffer from remaining sentences
        self._text_buffer = " ".join(sentences)

    def _extract_sentences(self, text: str) -> list[str]:
        """Extract complete sentences from text.

        Only returns sentences that end with sentence terminators.
        Incomplete sentences remain in the buffer.

        Uses re.finditer to iterate through potential sentence endings
        and decides whether each one is a real sentence boundary or
        part of an abbreviation/decimal number.

        Args:
            text: Text to extract sentences from

        Returns:
            List of complete sentences
        """
        if not text:
            return []

        # Common abbreviations that shouldn't end sentences
        # Format: abbreviations without their trailing period
        abbreviations = {"Mr", "Mrs", "Ms", "Dr", "Prof", "vs", "etc", "eg", "ie"}
        # Multi-part abbreviations like "e.g." and "i.e." - match the full pattern
        multi_part_abbrevs = {"e.g", "i.e"}

        sentences = []
        sentence_start = 0

        # Find all potential sentence terminators
        for match in re.finditer(r"[.!?]", text):
            terminator_pos = match.start()
            after_terminator = match.end()

            # Get text before and after the terminator for context
            before = text[sentence_start:terminator_pos]
            after = text[after_terminator:] if after_terminator < len(text) else ""

            # Check if terminator is part of an abbreviation
            # Extract the word immediately before the terminator
            before_stripped = before.rstrip()

            # Find the last word before the period
            word_match = re.search(r"(\S+)$", before_stripped)
            last_word = word_match.group(1) if word_match else ""

            # Check simple abbreviations (word without period)
            is_simple_abbrev = last_word in abbreviations

            # Check multi-part abbreviations like "e.g" or "i.e"
            # These contain embedded periods, so check if text ends with them
            is_multi_part_abbrev = any(
                before_stripped.endswith(abbr) for abbr in multi_part_abbrevs
            )

            # Check if terminator is a decimal point (digit.digit)
            is_decimal = (
                before_stripped
                and before_stripped[-1].isdigit()
                and after
                and after[0].isdigit()
            )

            # Check if this looks like a sentence end (followed by space + capital
            # or end of text). Abbreviations mid-sentence usually have lowercase after.
            looks_like_sentence_end = (
                not after  # End of text
                or (after and after[0].isspace()
                    and len(after) > 1
                    and after.lstrip()
                    and after.lstrip()[0].isupper())
            )

            if is_simple_abbrev or is_multi_part_abbrev or is_decimal:
                # Not a real sentence end - continue to next terminator
                continue

            # Also skip if followed by lowercase (likely abbreviation)
            if after and after[0].islower():
                continue

            # This is a real sentence end
            sentence = text[sentence_start:after_terminator].strip()
            if sentence:
                sentences.append(sentence)
            # Move start to after the terminator, skipping leading whitespace
            sentence_start = after_terminator
            while sentence_start < len(text) and text[sentence_start].isspace():
                sentence_start += 1

        return sentences

    def has_ready_chunk(self) -> bool:
        """Check if a 2-sentence chunk is ready for narration.

        Returns:
            True if at least one chunk is ready
        """
        return len(self._ready_chunks) > 0

    def pop_ready_chunk(self) -> str:
        """Get and remove the next ready chunk.

        Returns:
            The next 2-sentence chunk, or empty string if none ready
        """
        if self._ready_chunks:
            return self._ready_chunks.pop(0)
        return ""

    async def flush(self) -> Optional[str]:
        """Flush any remaining buffered text.

        Called at end of generation to retrieve partial content
        that didn't form a complete 2-sentence chunk.

        Returns:
            Remaining text, or None if buffer is empty
        """
        # Process any remaining buffer content
        if self._text_buffer:
            remaining = self._text_buffer.strip()
            self._text_buffer = ""

            # Also include any ready chunks that weren't popped
            if self._ready_chunks:
                all_remaining = self._ready_chunks + [remaining] if remaining else self._ready_chunks
                self._ready_chunks = []
                return " ".join(all_remaining)

            return remaining if remaining else None

        # Just ready chunks remaining
        if self._ready_chunks:
            result = " ".join(self._ready_chunks)
            self._ready_chunks = []
            return result

        return None

    def reset_buffer(self) -> None:
        """Reset the coordinated mode buffer state.

        Call at the start of a new generation to clear any
        leftover state from previous generations.
        """
        self._text_buffer = ""
        self._ready_chunks = []
