"""Cognitive stream: Externalized cognition and narration pipeline.

This package implements Lilly's "Cognitive Radio" - the continuous stream
of externalized thought that bridges fast internal processing with slow
speech output.

Architecture (5 layers):
- Buffer: Phonological loop staging area
- Decider: What to share (5-factor weighted decision)
- Compressor: How to compress (verbatim -> reference)
- Coloring: How to deliver (affect -> speech parameters)
- Rhythm: When to speak (silence management)

The StreamWeaver orchestrates all components to produce speech-ready
narrations from internal thoughts.
"""

from core.cognitive.stream.buffer import (
    BufferedThought,
    InternalNarrationBuffer,
    NarrationUrgency,
)
from core.cognitive.stream.coloring import (
    AffectColoring,
    AffectToSpeech,
    SpeechEmotion,
    SSMLGenerator,
    color_text_for_affect,
)
from core.cognitive.stream.compressor import (
    CompressionResult,
    CompressionStrategy,
    LLMProvider,
    NarrationStyle,
    StreamCompressor,
    ThoughtCompressor,
)
from core.cognitive.stream.decider import (
    ListenerState,
    NarrationDecider,
    NarrationDecision,
    NarrationFactors,
)
from core.cognitive.stream.rhythm import (
    RhythmAdvisor,
    SilenceState,
    SilenceType,
    StreamRhythm,
)
from core.cognitive.stream.silence_monitor import (
    ContentType,
    SilenceMonitor,
)
from core.cognitive.stream.progressive_narrator import (
    PrioritizedChunk,
    ProgressiveNarrator,
)
from core.cognitive.stream.chunker import (
    NarrationChunk,
    NarrationChunker,
)
from core.cognitive.stream.weaver import (
    ListenerSource,
    NarrationSink,
    StreamOutput,
    StreamStats,
    StreamWeaver,
)

__all__ = [
    # Buffer
    "NarrationUrgency",
    "BufferedThought",
    "InternalNarrationBuffer",
    # Decider
    "ListenerState",
    "NarrationFactors",
    "NarrationDecision",
    "NarrationDecider",
    # Compressor
    "CompressionStrategy",
    "NarrationStyle",
    "CompressionResult",
    "LLMProvider",
    "ThoughtCompressor",
    "StreamCompressor",
    # Coloring
    "SpeechEmotion",
    "AffectColoring",
    "AffectToSpeech",
    "SSMLGenerator",
    "color_text_for_affect",
    # Rhythm
    "SilenceType",
    "SilenceState",
    "StreamRhythm",
    "RhythmAdvisor",
    # Silence Monitor
    "ContentType",
    "SilenceMonitor",
    # Progressive Narrator
    "PrioritizedChunk",
    "ProgressiveNarrator",
    # Chunker
    "NarrationChunk",
    "NarrationChunker",
    # Weaver
    "NarrationSink",
    "ListenerSource",
    "StreamOutput",
    "StreamStats",
    "StreamWeaver",
]
