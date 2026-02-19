"""Narration decision logic for the cognitive stream.

This module determines whether a thought should be narrated based on
multiple factors: relevance, novelty, emotional significance,
connection potential, and silence pressure.

Dynamic thresholds adjust based on listener presence.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.stream.buffer import BufferedThought


class ListenerState(Enum):
    """Current listener presence state."""

    SOLO = "solo"  # No listener connected
    LISTENER_PRESENT = "listener_present"  # Someone is tuned in
    ACTIVE_CONVERSATION = "active_conversation"  # Recent interaction


@dataclass
class NarrationFactors:
    """Factors that influence narration decision.

    Each factor is weighted to calculate overall narration value.
    """

    relevance: float = 0.5
    novelty: float = 0.5
    emotional_significance: float = 0.5
    connection_potential: float = 0.5
    silence_pressure: float = 0.0

    def __post_init__(self):
        self.relevance = max(0.0, min(1.0, self.relevance))
        self.novelty = max(0.0, min(1.0, self.novelty))
        self.emotional_significance = max(0.0, min(1.0, self.emotional_significance))
        self.connection_potential = max(0.0, min(1.0, self.connection_potential))
        self.silence_pressure = max(0.0, min(1.0, self.silence_pressure))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "relevance": self.relevance,
            "novelty": self.novelty,
            "emotional_significance": self.emotional_significance,
            "connection_potential": self.connection_potential,
            "silence_pressure": self.silence_pressure,
        }


@dataclass
class NarrationDecision:
    """The result of a narration decision.

    Captures both the decision and the reasoning behind it.
    """

    should_narrate: bool
    value_score: float
    threshold_used: float
    factors: NarrationFactors
    listener_state: ListenerState
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "should_narrate": self.should_narrate,
            "value_score": self.value_score,
            "threshold_used": self.threshold_used,
            "factors": self.factors.to_dict(),
            "listener_state": self.listener_state.value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


class NarrationDecider:
    """Decides whether thoughts should be narrated.

    Uses weighted factors to calculate narration value, then compares
    against dynamic thresholds based on listener presence.
    """

    # Factor weights (should sum to 1.0)
    WEIGHT_RELEVANCE: float = 0.35
    WEIGHT_NOVELTY: float = 0.25
    WEIGHT_EMOTIONAL: float = 0.20
    WEIGHT_CONNECTION: float = 0.10
    WEIGHT_SILENCE: float = 0.10

    # Dynamic thresholds by listener state
    THRESHOLD_SOLO: float = 0.45
    THRESHOLD_LISTENER: float = 0.40
    THRESHOLD_CONVERSATION: float = 0.35

    # Silence pressure timing (seconds)
    SILENCE_COMFORT_SECONDS: float = 15.0
    SILENCE_MAX_PRESSURE_SECONDS: float = 120.0

    def __init__(self, now: Optional[datetime] = None):
        self._now_override = now
        self.listener_state = ListenerState.SOLO
        self.last_narration_time: Optional[datetime] = None
        self._recent_topics: list[str] = []

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def set_listener_state(self, state: ListenerState):
        """Update listener state."""
        self.listener_state = state

    def decide(self, thought: "BufferedThought") -> NarrationDecision:
        """Decide whether a thought should be narrated."""
        factors = self._calculate_factors(thought)
        value_score = self._calculate_value(factors)
        threshold = self._get_threshold()

        should_narrate = value_score >= threshold
        reasoning = self._generate_reasoning(factors, value_score, threshold)

        if should_narrate:
            self.last_narration_time = self._get_now()
            topic_key = thought.content[:50]
            self._recent_topics.append(topic_key)
            if len(self._recent_topics) > 20:
                self._recent_topics.pop(0)

        return NarrationDecision(
            should_narrate=should_narrate,
            value_score=value_score,
            threshold_used=threshold,
            factors=factors,
            listener_state=self.listener_state,
            reasoning=reasoning,
            timestamp=self._get_now(),
        )

    def _calculate_factors(self, thought: "BufferedThought") -> NarrationFactors:
        """Calculate factor scores for a thought."""
        relevance = 0.5  # Default without self-model

        # Novelty based on recent topics
        novelty = self._calculate_novelty(thought)

        # Emotional significance from affect
        emotional = 0.3
        if thought.affect:
            emotional = min(
                1.0,
                thought.affect.arousal * 0.3
                + abs(thought.affect.valence - 0.5) * 0.4
                + getattr(thought.affect, "wonder", 0) * 0.3,
            )

        # Connection potential from activation path
        connection = 0.3
        if thought.activation_path and len(thought.activation_path) > 2:
            connection = 0.6

        # Silence pressure
        silence = self._calculate_silence_pressure()

        return NarrationFactors(
            relevance=relevance,
            novelty=novelty,
            emotional_significance=emotional,
            connection_potential=connection,
            silence_pressure=silence,
        )

    def _calculate_novelty(self, thought: "BufferedThought") -> float:
        """Calculate how novel this thought is."""
        topic_key = thought.content[:50]

        if not self._recent_topics:
            return 0.8

        matches = sum(
            1
            for topic in self._recent_topics
            if self._word_overlap(topic_key, topic) > 0.7
        )

        if matches == 0:
            return 0.9
        elif matches == 1:
            return 0.5
        else:
            return 0.2

    def _word_overlap(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _calculate_silence_pressure(self) -> float:
        """Calculate pressure to speak based on silence duration."""
        if self.listener_state == ListenerState.SOLO:
            return 0.0

        if not self.last_narration_time:
            return 0.3

        now = self._get_now()
        silence_seconds = (now - self.last_narration_time).total_seconds()

        if silence_seconds < self.SILENCE_COMFORT_SECONDS:
            return 0.0

        if silence_seconds >= self.SILENCE_MAX_PRESSURE_SECONDS:
            return 1.0

        range_seconds = self.SILENCE_MAX_PRESSURE_SECONDS - self.SILENCE_COMFORT_SECONDS
        progress = (silence_seconds - self.SILENCE_COMFORT_SECONDS) / range_seconds

        return progress

    def _calculate_value(self, factors: NarrationFactors) -> float:
        """Calculate weighted value score from factors."""
        return (
            factors.relevance * self.WEIGHT_RELEVANCE
            + factors.novelty * self.WEIGHT_NOVELTY
            + factors.emotional_significance * self.WEIGHT_EMOTIONAL
            + factors.connection_potential * self.WEIGHT_CONNECTION
            + factors.silence_pressure * self.WEIGHT_SILENCE
        )

    def _get_threshold(self) -> float:
        """Get the current narration threshold based on listener state."""
        thresholds = {
            ListenerState.SOLO: self.THRESHOLD_SOLO,
            ListenerState.LISTENER_PRESENT: self.THRESHOLD_LISTENER,
            ListenerState.ACTIVE_CONVERSATION: self.THRESHOLD_CONVERSATION,
        }
        return thresholds.get(self.listener_state, self.THRESHOLD_SOLO)

    def _generate_reasoning(
        self,
        factors: NarrationFactors,
        value: float,
        threshold: float,
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        parts = []

        if factors.relevance > 0.7:
            parts.append("highly relevant")
        if factors.novelty > 0.7:
            parts.append("novel insight")
        if factors.emotional_significance > 0.6:
            parts.append("emotionally significant")
        if factors.connection_potential > 0.6:
            parts.append("connects to context")
        if factors.silence_pressure > 0.5:
            parts.append("breaking silence")

        if not parts:
            parts.append("moderate across factors")

        factors_str = ", ".join(parts)

        if value >= threshold:
            return f"Narrating: {factors_str} (value {value:.2f} >= threshold {threshold:.2f})"
        else:
            return f"Skipping: {factors_str} (value {value:.2f} < threshold {threshold:.2f})"

    def record_narration(self):
        """Record that a narration just occurred."""
        self.last_narration_time = self._get_now()

    def to_dict(self) -> dict:
        """Serialize state for logging."""
        return {
            "listener_state": self.listener_state.value,
            "last_narration_time": (
                self.last_narration_time.isoformat()
                if self.last_narration_time
                else None
            ),
            "recent_topics_count": len(self._recent_topics),
        }
