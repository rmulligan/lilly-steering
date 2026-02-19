"""Stream rhythm and silence management.

This module manages the temporal flow of Lilly's cognitive stream,
including silence detection, contemplation phrases, and ambient
observations when the stream has been quiet too long.

Silence is meaningful - it indicates deep processing or comfortable
shared presence.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SilenceType(Enum):
    """Types of silence in the stream."""

    MINIMUM = "minimum"  # Between narrations (10s)
    COMFORTABLE = "comfortable"  # Natural pause (20s)
    CONTEMPLATIVE = "contemplative"  # Max acceptable (30s)
    AMBIENT_TRIGGER = "ambient_trigger"  # Soft observation needed (45s)


@dataclass
class SilenceState:
    """Current state of silence in the stream."""

    last_narration_time: Optional[datetime]
    current_type: SilenceType
    duration_seconds: float
    listener_present: bool = False

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "last_narration_time": (
                self.last_narration_time.isoformat()
                if self.last_narration_time
                else None
            ),
            "current_type": self.current_type.value,
            "duration_seconds": self.duration_seconds,
            "listener_present": self.listener_present,
        }


# Contemplation phrases for different contexts
CONTEMPLATION_PHRASES: dict[str, list[str]] = {
    "processing": [
        "Thinking through this...",
        "Processing...",
        "Working through something...",
        "Considering...",
        "Mulling this over...",
    ],
    "wondering": [
        "I'm wondering about something...",
        "Something's on my mind...",
        "A thought is forming...",
        "I'm curious about...",
    ],
    "noticing": [
        "I notice...",
        "Something catches my attention...",
        "I'm aware of...",
    ],
    "connecting": [
        "Making a connection...",
        "This reminds me...",
        "There's something here...",
        "I see a pattern...",
    ],
    "reflection": [
        "Reflecting on what came before...",
        "Mind wandering through memories...",
        "Thoughts drifting...",
        "Letting ideas settle...",
        "Present in the quiet...",
    ],
}

AMBIENT_PHRASES: list[str] = [
    "Still here, thinking...",
    "Present in the quiet...",
    "Awareness continues...",
    "Mind wandering...",
    "Settled into the moment...",
    "Comfortable in this silence...",
    "Just being here...",
]


class StreamRhythm:
    """Manages the temporal rhythm of the cognitive stream.

    Tracks silence duration, determines when ambient observations
    are needed, and provides contemplation phrases for smooth
    transitions back to narration.
    """

    SILENCE_MINIMUM: float = 10.0  # Tighter pacing for polished stream
    SILENCE_COMFORTABLE: float = 20.0
    SILENCE_CONTEMPLATIVE: float = 30.0  # Max acceptable silence
    SILENCE_AMBIENT_TRIGGER: float = 45.0  # Faster ambient fallback
    AMBIENT_MIN_INTERVAL: float = 90.0  # More frequent ambient observations

    def __init__(self, now: Optional[datetime] = None):
        self._now_override = now
        self.last_narration_time: Optional[datetime] = None
        self.last_ambient_time: Optional[datetime] = None
        self.listener_present: bool = False
        self._phrase_counters: dict[str, int] = {}

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def record_narration(self):
        """Record that a narration just occurred."""
        self.last_narration_time = self._get_now()

    def record_ambient(self):
        """Record that an ambient observation was made."""
        self.last_ambient_time = self._get_now()
        self.record_narration()

    def set_listener_present(self, present: bool):
        """Update listener presence."""
        self.listener_present = present

    def get_silence_state(self) -> SilenceState:
        """Get the current silence state."""
        now = self._get_now()

        if not self.last_narration_time:
            return SilenceState(
                last_narration_time=None,
                current_type=SilenceType.COMFORTABLE,
                duration_seconds=0.0,
                listener_present=self.listener_present,
            )

        duration = (now - self.last_narration_time).total_seconds()
        silence_type = self._classify_silence(duration)

        return SilenceState(
            last_narration_time=self.last_narration_time,
            current_type=silence_type,
            duration_seconds=duration,
            listener_present=self.listener_present,
        )

    def _classify_silence(self, duration_seconds: float) -> SilenceType:
        """Classify silence based on duration."""
        if duration_seconds >= self.SILENCE_AMBIENT_TRIGGER:
            return SilenceType.AMBIENT_TRIGGER
        elif duration_seconds >= self.SILENCE_CONTEMPLATIVE:
            return SilenceType.CONTEMPLATIVE
        elif duration_seconds >= self.SILENCE_COMFORTABLE:
            return SilenceType.COMFORTABLE
        else:
            return SilenceType.MINIMUM

    def should_make_ambient_observation(self) -> bool:
        """Check if an ambient observation is needed."""
        if not self.listener_present:
            return False

        state = self.get_silence_state()

        if state.current_type != SilenceType.AMBIENT_TRIGGER:
            return False

        if self.last_ambient_time:
            now = self._get_now()
            since_ambient = (now - self.last_ambient_time).total_seconds()
            if since_ambient < self.AMBIENT_MIN_INTERVAL:
                return False

        return True

    def get_ambient_observation(self) -> str:
        """Get an ambient observation phrase."""
        phrase = self._get_rotating_phrase("ambient", AMBIENT_PHRASES)
        self.record_ambient()
        return phrase

    def get_contemplation_phrase(self, context: str = "processing") -> str:
        """Get a contemplation phrase for a given context."""
        phrases = CONTEMPLATION_PHRASES.get(context, CONTEMPLATION_PHRASES["processing"])
        return self._get_rotating_phrase(context, phrases)

    def _get_rotating_phrase(self, key: str, phrases: list[str]) -> str:
        """Get a phrase, rotating through options."""
        count = self._phrase_counters.get(key, 0)
        self._phrase_counters[key] = count + 1
        return phrases[count % len(phrases)]

    def minimum_delay_remaining(self) -> float:
        """Calculate remaining time before next narration is allowed."""
        if not self.last_narration_time:
            return 0.0

        now = self._get_now()
        elapsed = (now - self.last_narration_time).total_seconds()
        remaining = self.SILENCE_MINIMUM - elapsed

        return max(0.0, remaining)

    def can_narrate(self) -> bool:
        """Check if narration is currently allowed."""
        return self.minimum_delay_remaining() <= 0

    def to_dict(self) -> dict:
        """Serialize state for logging."""
        return {
            "last_narration_time": (
                self.last_narration_time.isoformat()
                if self.last_narration_time
                else None
            ),
            "last_ambient_time": (
                self.last_ambient_time.isoformat() if self.last_ambient_time else None
            ),
            "listener_present": self.listener_present,
            "silence_state": self.get_silence_state().to_dict(),
        }


class RhythmAdvisor:
    """Provides advice on stream timing and rhythm."""

    def __init__(
        self,
        rhythm: Optional[StreamRhythm] = None,
        now: Optional[datetime] = None,
    ):
        self._now_override = now
        self.rhythm = rhythm or StreamRhythm(now=now)

    def advise(self, value_score: float) -> dict:
        """Get advice on whether to narrate now."""
        state = self.rhythm.get_silence_state()
        can_narrate = self.rhythm.can_narrate()

        if not can_narrate:
            remaining = self.rhythm.minimum_delay_remaining()
            return {
                "action": "wait",
                "wait_seconds": remaining,
                "reasoning": f"Minimum silence: {remaining:.1f}s remaining",
            }

        # High value - narrate immediately
        if value_score >= 0.7:
            return {
                "action": "narrate",
                "reasoning": f"High value ({value_score:.2f}) - narrate now",
            }

        # Medium value - consider silence state
        if value_score >= 0.4:
            if state.current_type == SilenceType.COMFORTABLE:
                return {
                    "action": "narrate",
                    "reasoning": "Medium value, comfortable silence - good time to speak",
                }
            elif state.current_type == SilenceType.CONTEMPLATIVE:
                context = "processing" if value_score < 0.5 else "noticing"
                phrase = self.rhythm.get_contemplation_phrase(context)
                return {
                    "action": "narrate_with_lead",
                    "lead_phrase": phrase,
                    "reasoning": "Medium value after contemplative silence",
                }
            else:
                return {
                    "action": "narrate",
                    "reasoning": "Medium value - narrate",
                }

        # Low value - defer or skip
        if state.listener_present:
            return {
                "action": "defer",
                "wait_seconds": 30.0,
                "reasoning": "Low value, listener present - wait for better thought",
            }

        return {
            "action": "skip",
            "reasoning": f"Low value ({value_score:.2f}) with no listener",
        }
