"""Chunks long narrations for better listener experience.

Long monologues (300+ words) are segmented into digestible chunks with
ambient music fades between them, improving listener engagement and
reducing fatigue.
"""

from dataclasses import dataclass


@dataclass
class NarrationChunk:
    """A chunk of narration with fade metadata.

    Attributes:
        text: The narration text for this chunk.
        fade_after: If True, play an ambient fade after this chunk.
        fade_duration_ms: Duration of the fade in milliseconds.
    """

    text: str
    fade_after: bool = False
    fade_duration_ms: int = 2000


class NarrationChunker:
    """Splits long narrations into digestible chunks.

    Chunks are split at word boundaries to avoid mid-word breaks.
    All chunks except the last have fade_after=True to signal
    that an ambient music fade should play between chunks.

    Args:
        max_words: Maximum words per chunk (default 200).
        fade_duration_ms: Duration of fades between chunks (default 2000ms).
    """

    def __init__(self, max_words: int = 200, fade_duration_ms: int = 2000):
        self._max_words = max_words
        self._fade_duration_ms = fade_duration_ms

    def chunk(self, text: str) -> list[NarrationChunk]:
        """Split long text into chunks with fade markers.

        Args:
            text: The full narration text to chunk.

        Returns:
            List of NarrationChunk objects. Short text (<= max_words)
            returns a single chunk with fade_after=False. Longer text
            is split into multiple chunks, all with fade_after=True
            except the final chunk.
        """
        words = text.split()

        # Short text - no chunking needed
        if len(words) <= self._max_words:
            return [NarrationChunk(text=text, fade_after=False)]

        # Split into chunks at word boundaries
        chunks = []
        for i in range(0, len(words), self._max_words):
            chunk_words = words[i : i + self._max_words]
            chunk_text = " ".join(chunk_words)
            is_last = i + self._max_words >= len(words)

            chunks.append(
                NarrationChunk(
                    text=chunk_text,
                    fade_after=not is_last,
                    fade_duration_ms=self._fade_duration_ms,
                )
            )

        return chunks
