"""Internal narration buffer for the cognitive stream.

This module implements the "phonological loop" - a staging area between
fast unconscious processing and slow speech output. Thoughts that exceed
activation thresholds are buffered here before being selected for narration.

The buffer maintains urgency-based ordering with relevance decay over time.
"""

import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.self_model.affective_system import AffectiveState


class NarrationUrgency(Enum):
    """Urgency levels for narration selection."""

    IMMEDIATE = "immediate"  # Interrupt current narration
    SOON = "soon"  # Next in queue
    WHEN_CONVENIENT = "when_convenient"  # Normal priority
    BACKGROUND = "background"  # Only if nothing else


@dataclass
class BufferedThought:
    """A thought staged for potential narration.

    Represents something that has entered Lilly's awareness and may be
    worth sharing. Includes urgency, relevance decay, and affect context.
    """

    content: str
    formed_at: datetime
    urgency: NarrationUrgency
    affect: Optional["AffectiveState"] = None
    internal_only: bool = False
    relevance_decay: float = 0.95
    source_uid: str = ""
    activation_path: list[str] = field(default_factory=list)
    uid: str = field(default="")
    evaluation_count: int = 0

    def __post_init__(self):
        if not self.uid:
            key = f"{self.content[:50]}:{self.formed_at.isoformat()}"
            self.uid = f"bt:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def current_relevance(self, now: Optional[datetime] = None) -> float:
        """Calculate current relevance with time decay.

        Relevance decreases exponentially over time, making older
        thoughts less likely to be selected for narration.
        """
        current = now or datetime.now(timezone.utc)
        age_seconds = (current - self.formed_at).total_seconds()
        return self.relevance_decay ** (age_seconds / 10)

    def urgency_score(self) -> float:
        """Convert urgency to numeric score for sorting."""
        scores = {
            NarrationUrgency.IMMEDIATE: 1.0,
            NarrationUrgency.SOON: 0.75,
            NarrationUrgency.WHEN_CONVENIENT: 0.5,
            NarrationUrgency.BACKGROUND: 0.25,
        }
        return scores.get(self.urgency, 0.5)

    def selection_score(self, now: Optional[datetime] = None) -> float:
        """Calculate overall score for selection."""
        return self.urgency_score() * self.current_relevance(now)

    def to_dict(self) -> dict:
        """Serialize for storage/logging."""
        return {
            "uid": self.uid,
            "content": self.content,
            "formed_at": self.formed_at.isoformat(),
            "urgency": self.urgency.value,
            "internal_only": self.internal_only,
            "relevance_decay": self.relevance_decay,
            "source_uid": self.source_uid,
            "activation_path": self.activation_path,
        }


class InternalNarrationBuffer:
    """Buffer for thoughts awaiting potential narration.

    This is the "phonological loop" of the cognitive stream - a capacity-limited
    staging area where thoughts compete for narration based on urgency and
    relevance. The buffer automatically prunes stale thoughts.
    """

    MAX_BUFFER_ITEMS: int = 10
    MIN_RELEVANCE_THRESHOLD: float = 0.1
    STALE_SECONDS: float = 120.0

    def __init__(
        self,
        max_capacity: int = MAX_BUFFER_ITEMS,
        min_relevance: float = MIN_RELEVANCE_THRESHOLD,
        now: Optional[datetime] = None,
    ):
        self._now_override = now
        self.max_capacity = max_capacity
        self.min_relevance = min_relevance
        self.thoughts: list[BufferedThought] = []
        self._narrated_uids: deque[str] = deque(maxlen=100)

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def add_thought(self, thought: BufferedThought) -> bool:
        """Add a thought to the buffer.

        If buffer is at capacity, the thought with lowest selection score
        is removed to make room (unless the new thought is lower).
        """
        if thought.uid in self._narrated_uids:
            return False

        if any(t.uid == thought.uid for t in self.thoughts):
            return False

        self._prune_stale()

        if len(self.thoughts) < self.max_capacity:
            self.thoughts.append(thought)
            return True

        now = self._get_now()
        min_thought = min(self.thoughts, key=lambda t: t.selection_score(now))

        if thought.selection_score(now) > min_thought.selection_score(now):
            self.thoughts.remove(min_thought)
            self.thoughts.append(thought)
            return True

        return False

    def get_next_for_narration(self) -> Optional[BufferedThought]:
        """Get the highest-priority thought for narration.

        Removes the thought from the buffer and marks it as narrated.
        """
        self._prune_stale()

        candidates = [t for t in self.thoughts if not t.internal_only]

        if not candidates:
            return None

        now = self._get_now()
        best = max(candidates, key=lambda t: t.selection_score(now))

        self.thoughts.remove(best)
        self._narrated_uids.append(best.uid)

        return best

    def peek_next(self) -> Optional[BufferedThought]:
        """Peek at the highest-priority thought without removing it."""
        self._prune_stale()

        candidates = [t for t in self.thoughts if not t.internal_only]
        if not candidates:
            return None

        now = self._get_now()
        return max(candidates, key=lambda t: t.selection_score(now))

    def get_immediate_thoughts(self) -> list[BufferedThought]:
        """Get all IMMEDIATE urgency thoughts."""
        now = self._get_now()
        immediate = [
            t
            for t in self.thoughts
            if t.urgency == NarrationUrgency.IMMEDIATE and not t.internal_only
        ]
        return sorted(immediate, key=lambda t: t.selection_score(now), reverse=True)

    def _prune_stale(self):
        """Remove thoughts below relevance threshold."""
        now = self._get_now()
        self.thoughts = [
            t for t in self.thoughts if t.current_relevance(now) >= self.min_relevance
        ]

    def clear(self):
        """Clear all buffered thoughts."""
        self.thoughts = []

    def __len__(self) -> int:
        """Number of thoughts currently buffered."""
        return len(self.thoughts)

    def summarize(self) -> str:
        """Generate a brief summary of buffer state."""
        if not self.thoughts:
            return "Buffer empty"

        by_urgency: dict[str, int] = {}
        for t in self.thoughts:
            key = t.urgency.value
            by_urgency[key] = by_urgency.get(key, 0) + 1

        parts = [f"{count} {urgency}" for urgency, count in by_urgency.items()]
        return f"Buffer: {len(self.thoughts)} thoughts ({', '.join(parts)})"
