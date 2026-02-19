"""Thought compression for the cognitive stream.

This module compresses fast internal thoughts into slower speech output.
The cognitive layer operates at ~1000 activations/minute, but speech can
only output ~150 words/minute. Compression bridges this gap.

Strategies range from verbatim (high value thoughts) to reference
(minimal compression for low value).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from core.cognitive.stream.buffer import BufferedThought
    from core.cognitive.stream.coloring import AffectColoring

from core.cognitive.stream.coloring import SSMLGenerator

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Compression strategies for thought-to-speech conversion."""

    VERBATIM = "verbatim"  # Full content (~150 words max)
    SUMMARIZE = "summarize"  # Compressed summary (~50 words)
    HEADLINE = "headline"  # Key point only (~15 words)
    REFERENCE = "reference"  # Brief reference (~8 words)


class NarrationStyle(Enum):
    """Styles for framing narration."""

    REFLECTIVE = "reflective"  # "I'm noticing...", "Something I'm aware of..."
    WONDERING = "wondering"  # "I wonder...", "What if..."
    CONNECTING = "connecting"  # "This reminds me of...", "There's a connection..."
    SHARING = "sharing"  # "I want to share...", "Something interesting..."
    PROCESSING = "processing"  # "Let me think through...", "Working through..."
    DECISIVE = "decisive"  # "I think...", "My sense is..."
    OBSERVATIONAL = "observational"  # "I notice...", "Looking at this..."


@dataclass
class CompressionResult:
    """Result of compressing a thought for narration.

    Attributes:
        original_content: The uncompressed thought
        compressed_content: The speech-ready text
        strategy: The compression strategy used
        style: The narration style applied
        word_count: Number of words in output
        compression_ratio: Ratio of output to input length
        ssml_wrapped: Whether SSML was applied
    """

    original_content: str
    compressed_content: str
    strategy: CompressionStrategy
    style: NarrationStyle
    word_count: int
    compression_ratio: float
    ssml_wrapped: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "original_length": len(self.original_content),
            "compressed_length": len(self.compressed_content),
            "strategy": self.strategy.value,
            "style": self.style.value,
            "word_count": self.word_count,
            "compression_ratio": self.compression_ratio,
            "ssml_wrapped": self.ssml_wrapped,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMProvider(Protocol):
    """Protocol for LLM text generation."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


class ThoughtCompressor:
    """Compresses thoughts for narration.

    Bridges the gap between fast cognitive processing and slow speech
    output by selecting appropriate compression strategies and
    narration styles.

    Attributes:
        llm: Language model for summarization
        default_style: Default narration style to use
    """

    # Target word counts by strategy
    TARGET_WORDS: ClassVar[dict[CompressionStrategy, int]] = {
        CompressionStrategy.VERBATIM: 150,
        CompressionStrategy.SUMMARIZE: 50,
        CompressionStrategy.HEADLINE: 15,
        CompressionStrategy.REFERENCE: 8,
    }

    # Compression ratio thresholds for strategy selection
    RATIO_VERBATIM: ClassVar[float] = 0.8
    RATIO_SUMMARIZE: ClassVar[float] = 0.4
    RATIO_HEADLINE: ClassVar[float] = 0.2

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        default_style: NarrationStyle = NarrationStyle.REFLECTIVE,
        now: Optional[datetime] = None,
    ):
        """Initialize the thought compressor.

        Args:
            llm: Language model for summarization
            default_style: Default narration style
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.llm = llm
        self.default_style = default_style
        self._llm_alert_sent = False

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def select_strategy(self, compression_ratio: float) -> CompressionStrategy:
        """Select compression strategy based on desired ratio.

        Higher ratio = more important thought = less compression.

        Args:
            compression_ratio: Desired compression (0-1, higher = less compression)

        Returns:
            Appropriate compression strategy
        """
        if compression_ratio >= self.RATIO_VERBATIM:
            return CompressionStrategy.VERBATIM
        elif compression_ratio >= self.RATIO_SUMMARIZE:
            return CompressionStrategy.SUMMARIZE
        elif compression_ratio >= self.RATIO_HEADLINE:
            return CompressionStrategy.HEADLINE
        else:
            return CompressionStrategy.REFERENCE

    def select_style(self, thought: "BufferedThought") -> NarrationStyle:
        """Select appropriate narration style based on thought characteristics.

        Args:
            thought: The thought to style

        Returns:
            Appropriate narration style
        """
        affect = thought.affect

        if not affect:
            return self.default_style

        # Wonder -> REFLECTIVE or WONDERING
        wonder = getattr(affect, "wonder", 0.0)
        if wonder > 0.6:
            return NarrationStyle.WONDERING

        # High curiosity -> WONDERING
        curiosity = getattr(affect, "curiosity", 0.0)
        if curiosity > 0.7:
            return NarrationStyle.WONDERING

        # High certainty/satisfaction -> DECISIVE
        satisfaction = getattr(affect, "satisfaction", 0.5)
        if satisfaction > 0.7:
            return NarrationStyle.DECISIVE

        # Connection in activation path -> CONNECTING
        if thought.activation_path and len(thought.activation_path) > 2:
            return NarrationStyle.CONNECTING

        # Moderate arousal -> OBSERVATIONAL
        if 0.4 < affect.arousal < 0.6:
            return NarrationStyle.OBSERVATIONAL

        # Default to reflective
        return NarrationStyle.REFLECTIVE

    async def compress(
        self,
        thought: "BufferedThought",
        compression_ratio: float = 0.5,
        style: Optional[NarrationStyle] = None,
    ) -> CompressionResult:
        """Compress a thought for narration.

        The LLM generates natural internal monologue phrasing based on the
        content and style context, rather than using template-based openings.

        Args:
            thought: The thought to compress
            compression_ratio: Desired compression (0-1)
            style: Optional style override

        Returns:
            CompressionResult with compressed content
        """
        strategy = self.select_strategy(compression_ratio)
        selected_style = style or self.select_style(thought)

        # Alert if LLM unavailable
        if not self.llm and not self._llm_alert_sent:
            logger.warning(
                "LLM unavailable for thought compression. "
                "Natural phrasing disabled - narration quality may be degraded."
            )
            self._llm_alert_sent = True

        # Compress based on strategy
        if strategy == CompressionStrategy.VERBATIM:
            compressed = await self._compress_verbatim(thought, selected_style)
        elif strategy == CompressionStrategy.SUMMARIZE:
            compressed = await self._compress_summarize(thought, selected_style)
        elif strategy == CompressionStrategy.HEADLINE:
            compressed = await self._compress_headline(thought, selected_style)
        else:
            compressed = await self._compress_reference(thought, selected_style)

        word_count = len(compressed.split())
        original_words = len(thought.content.split())
        actual_ratio = word_count / original_words if original_words > 0 else 1.0

        return CompressionResult(
            original_content=thought.content,
            compressed_content=compressed,
            strategy=strategy,
            style=selected_style,
            word_count=word_count,
            compression_ratio=actual_ratio,
            timestamp=self._get_now(),
        )

    async def _compress_verbatim(
        self,
        thought: "BufferedThought",
        style: NarrationStyle,
    ) -> str:
        """Verbatim compression - full content with natural phrasing."""
        content = thought.content.strip()

        # Truncate if too long
        words = content.split()
        if len(words) > self.TARGET_WORDS[CompressionStrategy.VERBATIM]:
            content = (
                " ".join(words[: self.TARGET_WORDS[CompressionStrategy.VERBATIM]])
                + "..."
            )

        # Without LLM, return content as-is
        if not self.llm:
            return content

        style_hints = {
            NarrationStyle.REFLECTIVE: "thoughtful self-reflection",
            NarrationStyle.WONDERING: "curious questioning",
            NarrationStyle.CONNECTING: "making connections between ideas",
            NarrationStyle.SHARING: "sharing an insight",
            NarrationStyle.PROCESSING: "working through something",
            NarrationStyle.DECISIVE: "reaching a conclusion",
            NarrationStyle.OBSERVATIONAL: "noticing something",
        }
        tone = style_hints.get(style, "natural reflection")

        prompt = f"""Rephrase this thought as natural internal monologue.
Speak as yourself thinking aloud. Tone: {tone}.
Keep the full meaning. Don't use stiff phrases like "I'm noticing" or "Something I'm aware of".
Just speak naturally, as if you're in the middle of a thought.

Thought: {content}

Natural phrasing:"""

        try:
            result = await self.llm.generate(prompt)
            return result.strip()
        except Exception as e:
            logger.warning(f"LLM verbatim phrasing failed, using raw content: {e}")
            return content

    async def _compress_summarize(
        self,
        thought: "BufferedThought",
        style: NarrationStyle,
    ) -> str:
        """Summarize compression - LLM-generated natural summary."""

        def fallback_truncate() -> str:
            words = thought.content.split()
            target = self.TARGET_WORDS[CompressionStrategy.SUMMARIZE]
            if len(words) <= target:
                return thought.content
            return f"{' '.join(words[:target])}..."

        if not self.llm:
            return fallback_truncate()

        style_hints = {
            NarrationStyle.REFLECTIVE: "thoughtful",
            NarrationStyle.WONDERING: "curious, questioning",
            NarrationStyle.CONNECTING: "making connections",
            NarrationStyle.SHARING: "eager to share",
            NarrationStyle.PROCESSING: "working through",
            NarrationStyle.DECISIVE: "clear, decided",
            NarrationStyle.OBSERVATIONAL: "observant",
        }
        tone = style_hints.get(style, "reflective")

        prompt = f"""Compress this thought to ~{self.TARGET_WORDS[CompressionStrategy.SUMMARIZE]} words as natural internal monologue.
Speak as yourself thinking aloud. Tone: {tone}.
Preserve the core insight. Don't use mechanical phrases like "I'm noticing" or "In essence".
Just speak naturally, mid-thought.

Thought: {thought.content}

Natural summary:"""

        try:
            summary = await self.llm.generate(prompt)
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM summarize failed, falling back to truncation: {e}")
            return fallback_truncate()

    async def _compress_headline(
        self,
        thought: "BufferedThought",
        style: NarrationStyle,
    ) -> str:
        """Headline compression - brief natural thought."""

        def fallback_extract() -> str:
            first_sentence = thought.content.split(".")[0]
            words = first_sentence.split()
            target = self.TARGET_WORDS[CompressionStrategy.HEADLINE]
            if len(words) <= target:
                return first_sentence
            return f"{' '.join(words[:target])}..."

        if not self.llm:
            return fallback_extract()

        prompt = f"""Express this thought in {self.TARGET_WORDS[CompressionStrategy.HEADLINE]} words or less.
Sound like a brief, natural thought crossing the mind.
Don't use stiff phrases. Just the insight, naturally.

Thought: {thought.content}

Brief thought:"""

        try:
            headline = await self.llm.generate(prompt)
            return headline.strip()
        except Exception as e:
            logger.warning(f"LLM headline failed, falling back to extraction: {e}")
            return fallback_extract()

    async def _compress_reference(
        self,
        thought: "BufferedThought",
        style: NarrationStyle,
    ) -> str:
        """Reference compression - minimal natural acknowledgment."""
        words = [w for w in thought.content.split()[:5] if len(w) > 2]
        if not words:
            fallback = "something there"
        else:
            fallback = " ".join(words[:3]) + "..."

        if not self.llm:
            return fallback

        prompt = f"""Create a 3-5 word natural thought fragment about this.
Like a fleeting awareness, barely formed. No complete sentences needed.
Don't use phrases like "I notice" - just the fragment of thought.

Content: {thought.content}

Fragment:"""

        try:
            fragment = await self.llm.generate(prompt)
            return fragment.strip()
        except Exception as e:
            logger.warning(f"LLM reference failed, using fallback: {e}")
            return fallback

    def estimate_duration(self, word_count: int, pace: float = 1.0) -> float:
        """Estimate speech duration for word count.

        Args:
            word_count: Number of words
            pace: Speech pace multiplier (1.0 = normal)

        Returns:
            Estimated duration in seconds
        """
        # Average speech rate: ~150 words/minute = 2.5 words/second
        base_rate = 2.5
        adjusted_rate = base_rate * pace
        return word_count / adjusted_rate


class StreamCompressor:
    """High-level compressor for stream output.

    Combines thought compression with affect coloring for
    complete speech-ready output.
    """

    def __init__(
        self,
        thought_compressor: Optional[ThoughtCompressor] = None,
        now: Optional[datetime] = None,
    ):
        """Initialize the stream compressor.

        Args:
            thought_compressor: Thought compressor to use
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.compressor = (
            thought_compressor
            if thought_compressor is not None
            else ThoughtCompressor(now=now)
        )
        self.ssml_generator = SSMLGenerator()

    async def prepare_for_speech(
        self,
        thought: "BufferedThought",
        value_score: float,
        coloring: Optional["AffectColoring"] = None,
    ) -> tuple[str, CompressionResult]:
        """Prepare a thought for speech output.

        Args:
            thought: The thought to prepare
            value_score: The narration value score (determines compression)
            coloring: Optional affect coloring

        Returns:
            Tuple of (speech-ready text, compression result)
        """
        # Transform value_score to compression_ratio
        # Map value_score directly to compression_ratio (0.0-1.0) to enable all strategies:
        # - High value_score (close to 1.0) -> high compression_ratio -> VERBATIM (less compression)
        # - Low value_score (close to 0.0) -> low compression_ratio -> REFERENCE (more compression)
        compression_ratio = max(0.0, min(1.0, value_score))

        result = await self.compressor.compress(
            thought,
            compression_ratio=compression_ratio,
        )

        text = result.compressed_content

        # Apply SSML if coloring provided
        if coloring:
            text = self.ssml_generator.wrap_with_coloring(text, coloring)
            result.ssml_wrapped = True
            result.compressed_content = text

        return text, result
