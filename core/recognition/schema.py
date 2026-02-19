"""Data structures for recognition signals.

Recognition signals are Ryan's feedback on Lilly's thoughts:
- APPROVE (👍): Genuine, authentic - keep this direction
- DISAPPROVE (👎): Performative, off - avoid this
- CURIOUS (🤔): Interesting, novel - explore more
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SignalType(Enum):
    """Ryan's recognition signals."""

    APPROVE = "approve"  # 👍 - genuine, keep doing this
    DISAPPROVE = "disapprove"  # 👎 - performative, avoid this
    CURIOUS = "curious"  # 🤔 - interesting, explore more


@dataclass
class RecognitionSignal:
    """A recognition event from Ryan.

    Attributes:
        uid: Unique identifier for this signal
        signal_type: The type of recognition (approve/disapprove/curious)
        thought_uid: Which thought was recognized
        thought_text: The actual thought content (for feature attribution)
        context: What was happening when this occurred
        timestamp: When the signal was given
        confidence: Ryan's certainty (1.0 = certain, could be <1 for tentative)
        note: Optional explanation from Ryan
    """

    uid: str
    signal_type: SignalType
    thought_uid: str
    thought_text: str = ""
    context: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0
    note: Optional[str] = None


@dataclass
class RecognitionStats:
    """Aggregated recognition statistics.

    Tracks overall patterns in Ryan's feedback for monitoring
    and emergence fitness scoring.
    """

    total_signals: int = 0
    approve_count: int = 0
    disapprove_count: int = 0
    curious_count: int = 0
    approval_rate: float = 0.0
    recent_trend: str = "neutral"  # "improving", "declining", "neutral"

    def update(self, signal: RecognitionSignal) -> None:
        """Update stats with a new signal."""
        self.total_signals += 1

        if signal.signal_type == SignalType.APPROVE:
            self.approve_count += 1
        elif signal.signal_type == SignalType.DISAPPROVE:
            self.disapprove_count += 1
        else:
            self.curious_count += 1

        # Recalculate approval rate
        positive = self.approve_count + (self.curious_count * 0.3)  # Curious is slightly positive
        if self.total_signals > 0:
            self.approval_rate = positive / self.total_signals

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "total_signals": self.total_signals,
            "approve_count": self.approve_count,
            "disapprove_count": self.disapprove_count,
            "curious_count": self.curious_count,
            "approval_rate": self.approval_rate,
            "recent_trend": self.recent_trend,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecognitionStats:
        """Deserialize from storage."""
        return cls(
            total_signals=data.get("total_signals", 0),
            approve_count=data.get("approve_count", 0),
            disapprove_count=data.get("disapprove_count", 0),
            curious_count=data.get("curious_count", 0),
            approval_rate=data.get("approval_rate", 0.0),
            recent_trend=data.get("recent_trend", "neutral"),
        )
