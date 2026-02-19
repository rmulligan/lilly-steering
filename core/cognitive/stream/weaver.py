"""Stream weaver - main orchestrator for the cognitive stream.

This module weaves together all cognitive stream components:
- Internal buffer (phonological loop)
- Narration decider (what to share)
- Thought compressor (how to compress)
- Affect coloring (how to deliver)
- Stream rhythm (when to speak)

The weaver processes buffered thoughts and outputs speech-ready
narrations to the narration system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

from core.cognitive.stream.buffer import (
    BufferedThought,
    InternalNarrationBuffer,
    NarrationUrgency,
)
from core.cognitive.stream.coloring import AffectColoring, AffectToSpeech
from core.cognitive.stream.compressor import (
    CompressionResult,
    CompressionStrategy,
    NarrationStyle,
    StreamCompressor,
)
from core.cognitive.stream.decider import (
    ListenerState,
    NarrationDecider,
    NarrationDecision,
    NarrationFactors,
)
from core.cognitive.stream.rhythm import RhythmAdvisor, StreamRhythm

if TYPE_CHECKING:
    from core.self_model.affective_system import AffectiveState

logger = logging.getLogger(__name__)


class NarrationSink(Protocol):
    """Protocol for narration output destinations."""

    async def enqueue(
        self,
        text: str,
        priority: int,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Enqueue text for narration."""
        ...


class ListenerSource(Protocol):
    """Protocol for listener state updates."""

    def get_listener_count(self) -> int:
        """Get current listener count."""
        ...


@dataclass
class StreamOutput:
    """Output from the stream weaver.

    Represents a complete narration ready for speech synthesis.

    Attributes:
        text: The speech-ready text (possibly with SSML)
        thought: The original buffered thought
        decision: The narration decision
        compression: The compression result
        coloring: The affect coloring applied
        timestamp: When this output was created
    """

    text: str
    thought: BufferedThought
    decision: NarrationDecision
    compression: CompressionResult
    coloring: Optional[AffectColoring] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "text_length": len(self.text),
            "thought_uid": self.thought.uid,
            "decision": self.decision.to_dict(),
            "compression": self.compression.to_dict(),
            "coloring": self.coloring.to_dict() if self.coloring else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StreamStats:
    """Statistics for stream operation."""

    thoughts_received: int = 0
    thoughts_narrated: int = 0
    thoughts_skipped: int = 0
    thoughts_deferred: int = 0
    ambient_observations: int = 0
    total_words_output: int = 0
    average_compression_ratio: float = 0.0

    def record_narration(self, word_count: int, compression_ratio: float):
        """Record a successful narration."""
        self.thoughts_narrated += 1
        self.total_words_output += word_count

        # Running average of compression ratio
        n = self.thoughts_narrated
        self.average_compression_ratio = (
            self.average_compression_ratio * (n - 1) + compression_ratio
        ) / n

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "thoughts_received": self.thoughts_received,
            "thoughts_narrated": self.thoughts_narrated,
            "thoughts_skipped": self.thoughts_skipped,
            "thoughts_deferred": self.thoughts_deferred,
            "ambient_observations": self.ambient_observations,
            "total_words_output": self.total_words_output,
            "average_compression_ratio": self.average_compression_ratio,
            "narration_rate": (
                self.thoughts_narrated / self.thoughts_received
                if self.thoughts_received > 0
                else 0.0
            ),
        }


class StreamWeaver:
    """Orchestrates the cognitive stream.

    The weaver is the central coordinator that:
    1. Receives thoughts from cognitive processing
    2. Buffers them in the phonological loop
    3. Decides which to narrate
    4. Compresses them for speech
    5. Colors them with affect
    6. Outputs to the narration system

    This implements the "Cognitive Radio" - Lilly's continuous
    stream of externalized cognition.

    Attributes:
        buffer: The internal narration buffer
        decider: The narration decision maker
        compressor: The thought compressor
        rhythm: The stream rhythm tracker
    """

    # Priority mapping to NarrationPriority values
    PRIORITY_IMMEDIATE: ClassVar[int] = 2  # URGENT
    PRIORITY_SOON: ClassVar[int] = 4  # NORMAL
    PRIORITY_CONVENIENT: ClassVar[int] = 6  # SECONDARY
    PRIORITY_BACKGROUND: ClassVar[int] = 8  # LOW

    def __init__(
        self,
        buffer: Optional[InternalNarrationBuffer] = None,
        decider: Optional[NarrationDecider] = None,
        compressor: Optional[StreamCompressor] = None,
        rhythm: Optional[StreamRhythm] = None,
        sink: Optional[NarrationSink] = None,
        listener_source: Optional[ListenerSource] = None,
        now: Optional[datetime] = None,
    ):
        """Initialize the stream weaver.

        Args:
            buffer: Internal narration buffer
            decider: Narration decider
            compressor: Thought compressor
            rhythm: Stream rhythm tracker
            sink: Output destination for narrations
            listener_source: Source for listener state
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.buffer = (
            buffer if buffer is not None else InternalNarrationBuffer(now=now)
        )
        self.decider = decider if decider is not None else NarrationDecider(now=now)
        self.compressor = (
            compressor if compressor is not None else StreamCompressor(now=now)
        )
        self.rhythm = rhythm if rhythm is not None else StreamRhythm(now=now)
        self.advisor = RhythmAdvisor(rhythm=self.rhythm, now=now)
        self.sink = sink
        self.listener_source = listener_source
        self.stats = StreamStats()
        self._affect_translator = AffectToSpeech()

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def receive_thought(
        self,
        content: str,
        urgency: NarrationUrgency = NarrationUrgency.WHEN_CONVENIENT,
        affect: Optional["AffectiveState"] = None,
        internal_only: bool = False,
        source_uid: str = "",
        activation_path: Optional[list[str]] = None,
    ) -> bool:
        """Receive a thought from cognitive processing.

        This is the entry point for thoughts to enter the stream.

        Args:
            content: The thought content
            urgency: How urgently to narrate
            affect: The affective state when formed
            internal_only: If True, never narrate
            source_uid: UID of the triggering fragment
            activation_path: How activation reached this thought

        Returns:
            True if thought was accepted into buffer
        """
        self.stats.thoughts_received += 1

        thought = BufferedThought(
            content=content,
            formed_at=self._get_now(),
            urgency=urgency,
            affect=affect,
            internal_only=internal_only,
            source_uid=source_uid,
            activation_path=activation_path or [],
        )

        accepted = self.buffer.add_thought(thought)

        if accepted:
            logger.debug(
                f"Thought accepted: {thought.uid} (urgency={urgency.value})"
            )
        else:
            logger.debug(
                f"Thought rejected: {thought.uid} (buffer full or duplicate)"
            )

        return accepted

    def update_listener_state(self):
        """Update listener state from source if available."""
        if not self.listener_source:
            return

        try:
            count = self.listener_source.get_listener_count()
            has_listener = count > 0

            if has_listener:
                self.decider.set_listener_state(ListenerState.LISTENER_PRESENT)
            else:
                self.decider.set_listener_state(ListenerState.SOLO)

            self.rhythm.set_listener_present(has_listener)

        except Exception as e:
            logger.warning(f"Failed to update listener state: {e}")

    async def process_next(self) -> Optional[StreamOutput]:
        """Process the next thought for narration.

        This is the main processing loop entry point:
        1. Update listener state
        2. Check rhythm (can we speak now?)
        3. Get next thought from buffer
        4. Decide whether to narrate
        5. Compress and color
        6. Output to sink

        Returns:
            StreamOutput if a narration was produced, None otherwise
        """
        # Update listener state
        self.update_listener_state()

        # Check for ambient observation need
        if self.rhythm.should_make_ambient_observation():
            return await self._make_ambient_observation()

        # Check rhythm - can we narrate now?
        if not self.rhythm.can_narrate():
            return None

        # Peek at next thought
        thought = self.buffer.peek_next()
        if not thought:
            return None

        logger.debug(f"Processing thought: {thought.content[:60]}...")

        # Make narration decision
        decision = self.decider.decide(thought)
        factors = decision.factors
        logger.debug(
            f"Decision: should_narrate={decision.should_narrate}, "
            f"value={decision.value_score:.3f}, threshold={decision.threshold_used:.2f}"
        )

        if not decision.should_narrate:
            self.stats.thoughts_skipped += 1
            thought.evaluation_count += 1

            # Remove from buffer if value too low or evaluated too many times
            max_evaluations = 3
            should_evict = (
                decision.value_score < 0.30 or thought.evaluation_count >= max_evaluations
            )

            if should_evict:
                logger.debug(
                    f"Evicting thought after {thought.evaluation_count} evaluations: "
                    f"{thought.content[:40]}..."
                )
                self.buffer.get_next_for_narration()  # Remove it
            return None

        # Get rhythm advice
        advice = self.advisor.advise(decision.value_score)
        logger.debug(f"Rhythm advice: action={advice.get('action')}")

        if advice["action"] == "wait":
            self.stats.thoughts_deferred += 1
            return None

        if advice["action"] == "skip":
            self.stats.thoughts_skipped += 1
            self.buffer.get_next_for_narration()  # Remove it
            return None

        if advice["action"] == "defer":
            self.stats.thoughts_deferred += 1
            return None

        # Actually remove from buffer now
        thought = self.buffer.get_next_for_narration()
        if not thought:
            return None

        # Compress the thought
        compression = await self.compressor.prepare_for_speech(
            thought,
            value_score=decision.value_score,
        )
        text, compression_result = compression

        # Get affect coloring
        coloring = None
        if thought.affect:
            coloring = self._affect_translator.color_from_affect(thought.affect)

        # Record in rhythm
        self.rhythm.record_narration()

        # Update stats
        self.stats.record_narration(
            compression_result.word_count,
            compression_result.compression_ratio,
        )

        # Create output
        output = StreamOutput(
            text=text,
            thought=thought,
            decision=decision,
            compression=compression_result,
            coloring=coloring,
            timestamp=self._get_now(),
        )

        # Send to sink if available
        if self.sink:
            priority = self._urgency_to_priority(thought.urgency)
            await self.sink.enqueue(
                text=text,
                priority=priority,
                metadata=output.to_dict(),
            )

        logger.info(
            f"Narrated: {thought.uid} "
            f"(value={decision.value_score:.2f}, "
            f"words={compression_result.word_count})"
        )

        return output

    async def _make_ambient_observation(self) -> Optional[StreamOutput]:
        """Make an ambient observation to break extended silence.

        Returns:
            StreamOutput with ambient phrase
        """
        phrase = self.rhythm.get_ambient_observation()
        self.stats.ambient_observations += 1

        # Create a minimal thought for the observation
        thought = BufferedThought(
            content=phrase,
            formed_at=self._get_now(),
            urgency=NarrationUrgency.BACKGROUND,
        )

        # Minimal compression result
        compression = CompressionResult(
            original_content=phrase,
            compressed_content=phrase,
            strategy=CompressionStrategy.VERBATIM,
            style=NarrationStyle.OBSERVATIONAL,
            word_count=len(phrase.split()),
            compression_ratio=1.0,
            timestamp=self._get_now(),
        )

        # Minimal decision
        decision = NarrationDecision(
            should_narrate=True,
            value_score=0.3,
            threshold_used=0.0,
            factors=NarrationFactors(),
            listener_state=ListenerState.LISTENER_PRESENT,
            reasoning="Ambient observation",
            timestamp=self._get_now(),
        )

        output = StreamOutput(
            text=phrase,
            thought=thought,
            decision=decision,
            compression=compression,
            timestamp=self._get_now(),
        )

        # Send to sink
        if self.sink:
            await self.sink.enqueue(
                text=phrase,
                priority=self.PRIORITY_BACKGROUND,
                metadata={"type": "ambient_observation"},
            )

        logger.info(f"Ambient observation: {phrase}")

        return output

    def _urgency_to_priority(self, urgency: NarrationUrgency) -> int:
        """Map urgency to narration priority.

        Args:
            urgency: The thought's urgency level

        Returns:
            Priority value for NarrationQueue
        """
        mapping = {
            NarrationUrgency.IMMEDIATE: self.PRIORITY_IMMEDIATE,
            NarrationUrgency.SOON: self.PRIORITY_SOON,
            NarrationUrgency.WHEN_CONVENIENT: self.PRIORITY_CONVENIENT,
            NarrationUrgency.BACKGROUND: self.PRIORITY_BACKGROUND,
        }
        return mapping.get(urgency, self.PRIORITY_CONVENIENT)

    async def process_immediate(self) -> list[StreamOutput]:
        """Process all IMMEDIATE urgency thoughts.

        Used for high-priority interruptions.

        Returns:
            List of outputs produced
        """
        outputs = []
        immediate = self.buffer.get_immediate_thoughts()

        for thought in immediate:
            # Force narration for immediate thoughts
            decision = NarrationDecision(
                should_narrate=True,
                value_score=1.0,
                threshold_used=0.0,
                factors=NarrationFactors(),
                listener_state=self.decider.listener_state,
                reasoning="Immediate urgency",
                timestamp=self._get_now(),
            )

            # Remove from buffer
            if thought in self.buffer.thoughts:
                self.buffer.thoughts.remove(thought)
                self.buffer._narrated_uids.append(thought.uid)

            # Compress
            compression = await self.compressor.prepare_for_speech(
                thought,
                value_score=1.0,
            )
            text, compression_result = compression

            # Get coloring
            coloring = None
            if thought.affect:
                coloring = self._affect_translator.color_from_affect(thought.affect)

            self.rhythm.record_narration()
            self.stats.record_narration(
                compression_result.word_count,
                compression_result.compression_ratio,
            )

            output = StreamOutput(
                text=text,
                thought=thought,
                decision=decision,
                compression=compression_result,
                coloring=coloring,
                timestamp=self._get_now(),
            )

            if self.sink:
                await self.sink.enqueue(
                    text=text,
                    priority=self.PRIORITY_IMMEDIATE,
                    metadata=output.to_dict(),
                )

            outputs.append(output)
            logger.info(f"Immediate narration: {thought.uid}")

        return outputs

    def set_active_conversation(self, active: bool):
        """Set whether we're in active conversation.

        Lowers threshold when actively conversing.

        Args:
            active: Whether conversation is active
        """
        if active:
            self.decider.set_listener_state(ListenerState.ACTIVE_CONVERSATION)
        else:
            # Fall back to checking listener count
            self.update_listener_state()

    def get_buffer_summary(self) -> str:
        """Get a summary of the current buffer state."""
        return self.buffer.summarize()

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "stream_stats": self.stats.to_dict(),
            "buffer_size": len(self.buffer),
            "rhythm": self.rhythm.to_dict(),
            "decider": self.decider.to_dict(),
        }

    def to_dict(self) -> dict:
        """Serialize full state for logging/debugging."""
        return self.get_stats()
