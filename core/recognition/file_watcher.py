"""File-based recognition signal acquisition.

Watches a JSONL file for new recognition signals from Ryan.
This is the simplest approach - Ryan can edit the file directly.

File format (config/recognition_signals.jsonl):
    {"uid": "rec_001", "signal": "approve", "thought_uid": "th_abc123", "thought_text": "...", "context": "..."}
    {"uid": "rec_002", "signal": "disapprove", "thought_uid": "th_def456", "note": "too abstract"}
    {"uid": "rec_003", "signal": "curious", "thought_uid": "th_ghi789", "context": "unexpected connection"}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.recognition.schema import RecognitionSignal, SignalType, RecognitionStats

logger = logging.getLogger(__name__)

# Default location for recognition signals
DEFAULT_RECOGNITION_FILE = Path("config/recognition_signals.jsonl")


class RecognitionFileWatcher:
    """Watch for new recognition signals in JSONL file.

    Maintains file position to only read new signals since last check.
    Thread-safe for async usage.

    Attributes:
        file_path: Path to the JSONL signal file
        stats: Aggregated statistics about signals received
    """

    def __init__(self, file_path: Optional[Path] = None):
        """Initialize the watcher.

        Args:
            file_path: Path to signal file, defaults to config/recognition_signals.jsonl
        """
        self.file_path = file_path or DEFAULT_RECOGNITION_FILE
        self._last_position: int = 0
        self._signal_count: int = 0
        self.stats = RecognitionStats()
        self._initialized = False

    def _init_position(self) -> None:
        """Initialize file position to end of file (skip existing signals)."""
        if self._initialized:
            return

        if self.file_path.exists():
            with open(self.file_path) as f:
                f.seek(0, 2)  # Seek to end
                self._last_position = f.tell()
            logger.info(f"Recognition watcher initialized at position {self._last_position}")
        else:
            logger.info(f"Recognition file does not exist yet: {self.file_path}")

        self._initialized = True

    async def check_for_signals(self) -> list[RecognitionSignal]:
        """Read new signals since last check.

        Returns:
            List of new RecognitionSignal objects
        """
        self._init_position()

        if not self.file_path.exists():
            return []

        new_signals: list[RecognitionSignal] = []

        try:
            with open(self.file_path, encoding="utf-8") as f:
                f.seek(self._last_position)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        signal = self._parse_signal_line(line)
                        if signal:
                            new_signals.append(signal)
                            self.stats.update(signal)
                            self._signal_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse signal line: {e}")
                        continue

                self._last_position = f.tell()

        except Exception as e:
            logger.error(f"Failed to read recognition file: {e}")

        if new_signals:
            logger.info(f"Read {len(new_signals)} new recognition signal(s)")

        return new_signals

    def _parse_signal_line(self, line: str) -> Optional[RecognitionSignal]:
        """Parse a single JSONL line into a RecognitionSignal.

        Args:
            line: JSON string from the file

        Returns:
            RecognitionSignal if valid, None otherwise
        """
        data = json.loads(line)

        # Validate required fields
        if "signal" not in data:
            logger.warning(f"Signal line missing 'signal' field: {line[:50]}")
            return None

        # Parse signal type
        try:
            signal_type = SignalType(data["signal"])
        except ValueError:
            logger.warning(f"Invalid signal type: {data['signal']}")
            return None

        # Parse timestamp if present
        timestamp = datetime.now(timezone.utc)
        if "timestamp" in data:
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass  # Use current time

        # Generate UID if not provided
        uid = data.get("uid", f"rec_{self._signal_count:06d}")

        return RecognitionSignal(
            uid=uid,
            signal_type=signal_type,
            thought_uid=data.get("thought_uid", ""),
            thought_text=data.get("thought_text", ""),
            context=data.get("context", ""),
            timestamp=timestamp,
            confidence=data.get("confidence", 1.0),
            note=data.get("note"),
        )

    def reset_to_beginning(self) -> None:
        """Reset file position to beginning (reprocess all signals)."""
        self._last_position = 0
        self._initialized = True  # Mark as initialized to prevent _init_position from seeking to end
        logger.info("Recognition watcher reset to beginning")

    def get_signal_count(self) -> int:
        """Get total number of signals processed."""
        return self._signal_count

    def get_stats(self) -> RecognitionStats:
        """Get aggregated statistics."""
        return self.stats
