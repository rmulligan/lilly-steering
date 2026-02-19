"""
Base infrastructure for Lilly's dream cycles.

Dream cycles are tiered consolidation processes inspired by how biological
sleep enables memory consolidation and learning. Each cycle type serves
a different purpose in Lilly's cognitive development.

Dream Hierarchy:
    - Micro-dream: Per-interaction, triggered by surprise (high free energy)
    - Nap: Every few hours, pattern detection and small adjustments
    - Full dream: Daily deep consolidation, vector refinement, journaling
    - Deep reflection: Weekly existential inquiry, goal revision

Integration Points:
    - Active Inference: Uses graph entropy for surprise detection
    - Self-Model: Updates affective state, goals, preferences
    - Psyche: Reads/writes episodic memories and steering vectors
    - Event Bus: Receives triggers, emits completion events
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from services.event_bus import EventBus

logger = logging.getLogger(__name__)


class DreamDepth(Enum):
    """Processing depth for dream cycles."""
    LIGHT = "light"      # Quick scan, flag for later
    MEDIUM = "medium"    # Pattern detection, small adjustments
    DEEP = "deep"        # Full consolidation, vector refinement
    INTENSIVE = "intensive"  # Existential inquiry, goal revision


@dataclass
class DreamContext:
    """
    Context passed to dream cycles.

    Contains references to system components and trigger information.
    """
    psyche: Optional["PsycheClient"] = None
    event_bus: Optional["EventBus"] = None
    trigger_event: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def trigger_type(self) -> str:
        """Get the type of event that triggered this dream."""
        if self.trigger_event:
            return self.trigger_event.get("trigger", "scheduled")
        return "scheduled"

    @property
    def surprise_score(self) -> Optional[float]:
        """Get surprise score if this was a surprise-triggered dream."""
        if self.trigger_event:
            return self.trigger_event.get("score")
        return None


@dataclass
class DreamInsight:
    """
    An insight discovered during a dream cycle.

    Insights are observations, patterns, or discoveries that may
    influence Lilly's development.
    """
    content: str
    category: str  # "pattern", "surprise", "preference", "question", "goal"
    confidence: float = 0.5  # 0-1
    actionable: bool = False
    suggested_action: Optional[str] = None
    source_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "actionable": self.actionable,
            "suggested_action": self.suggested_action,
            "source_memories": self.source_memories,
        }


@dataclass
class DreamResult:
    """
    Result of a dream cycle execution.

    Contains insights, metrics, and actions taken during the dream.
    """
    cycle_type: str
    depth: DreamDepth
    duration_ms: float
    insights: list[DreamInsight] = field(default_factory=list)
    memories_processed: int = 0
    patterns_detected: int = 0
    vectors_updated: int = 0
    journal_entry: Optional[str] = None
    narration: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success(self) -> bool:
        """Dream completed without critical errors."""
        return len(self.errors) == 0

    @property
    def has_insights(self) -> bool:
        """Dream produced insights worth noting."""
        return len(self.insights) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging and storage."""
        return {
            "cycle_type": self.cycle_type,
            "depth": self.depth.value,
            "duration_ms": self.duration_ms,
            "insights": [i.to_dict() for i in self.insights],
            "memories_processed": self.memories_processed,
            "patterns_detected": self.patterns_detected,
            "vectors_updated": self.vectors_updated,
            "journal_entry": self.journal_entry,
            "narration": self.narration,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }


class BaseDream(ABC):
    """
    Abstract base class for dream cycle implementations.

    Each dream type inherits from this and implements its specific
    processing logic.
    """

    def __init__(
        self,
        cycle_type: str,
        depth: DreamDepth,
    ):
        self.cycle_type = cycle_type
        self.depth = depth
        self._start_time: Optional[datetime] = None

    @abstractmethod
    async def execute(self, context: DreamContext) -> DreamResult:
        """
        Execute the dream cycle.

        Args:
            context: Dream context with system references

        Returns:
            DreamResult with insights and metrics
        """
        pass

    def _start_timer(self) -> None:
        """Start timing the dream execution."""
        self._start_time = datetime.now(timezone.utc)

    def _get_duration_ms(self) -> float:
        """Get duration since start in milliseconds."""
        if self._start_time is None:
            return 0.0
        elapsed = datetime.now(timezone.utc) - self._start_time
        return elapsed.total_seconds() * 1000

    def _create_result(
        self,
        insights: Optional[list[DreamInsight]] = None,
        **kwargs: Any,
    ) -> DreamResult:
        """Create a result object with common fields filled in."""
        return DreamResult(
            cycle_type=self.cycle_type,
            depth=self.depth,
            duration_ms=self._get_duration_ms(),
            insights=insights or [],
            **kwargs,
        )


class DreamCycleError(Exception):
    """Error during dream cycle execution."""
    pass


# Thresholds for dream triggers
SURPRISE_THRESHOLD_MICRO = 0.7  # Trigger micro-dream
SURPRISE_THRESHOLD_HIGH = 0.85  # Worth immediate attention
PATTERN_DETECTION_MIN_SAMPLES = 5  # Min episodes for pattern detection
