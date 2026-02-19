"""Buffer for batching curator narrations to reduce micro-narration fatigue.

The listener UX audit found 15-25 micro-narrations per cycle during curation
phase. This buffer consolidates discovery narrations into batched summaries,
targeting 8-10 total narrations per cycle.
"""

import time
from typing import Optional


class CuratorNarrationBuffer:
    """Batches curator discoveries for consolidated narration.

    Instead of narrating each discovery immediately (e.g., "Found entity: X",
    "Found relationship: Y", etc.), this buffer collects discoveries and
    flushes them as a single narration when either:
    - The buffer reaches max_buffer items
    - min_interval_seconds has elapsed since last narration

    Usage:
        buffer = CuratorNarrationBuffer(max_buffer=5, min_interval_seconds=30.0)

        # In tool execution:
        text = buffer.add("Found entity: concept_drift")
        if text:
            await narrate(text)

        # At end of curation phase:
        text = buffer.flush_remaining()
        if text:
            await narrate(text)
    """

    def __init__(self, max_buffer: int = 5, min_interval_seconds: float = 30.0):
        """Initialize the narration buffer.

        Args:
            max_buffer: Maximum discoveries before automatic flush.
            min_interval_seconds: Minimum time between narrations.
        """
        self._buffer: list[str] = []
        self._last_narration: float = time.time()
        self._max_buffer = max_buffer
        self._min_interval = min_interval_seconds

    def add(self, discovery: str) -> Optional[str]:
        """Add discovery to buffer. Returns narration text if ready to flush.

        Args:
            discovery: The discovery text to buffer (e.g., "Found entity: X").

        Returns:
            Consolidated narration text if buffer should flush, None otherwise.
        """
        self._buffer.append(discovery)

        now = time.time()
        elapsed = now - self._last_narration

        if len(self._buffer) >= self._max_buffer or elapsed >= self._min_interval:
            return self._flush()
        return None

    def _flush(self) -> str:
        """Combine buffered discoveries into single narration.

        Returns:
            Consolidated narration text.
        """
        if len(self._buffer) == 1:
            text = self._buffer[0]
        elif len(self._buffer) <= 3:
            text = f"Several discoveries: {', '.join(self._buffer)}"
        else:
            text = f"Several discoveries: {', '.join(self._buffer[:3])}"
            text += f" and {len(self._buffer) - 3} more"

        self._buffer.clear()
        self._last_narration = time.time()
        return text

    def flush_remaining(self) -> Optional[str]:
        """Flush any remaining buffered discoveries.

        Call this at the end of curation phase to ensure all discoveries
        are narrated before transitioning to the next phase.

        Returns:
            Consolidated narration text if buffer has content, None otherwise.
        """
        if self._buffer:
            return self._flush()
        return None
