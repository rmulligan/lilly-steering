"""Preference learning from valenced experiences.

This module implements how Lilly learns preferences from experience.
Preferences emerge from repeated positive or negative valence associated
with contexts, actions, or outcomes. Strong, stable preferences can be
promoted to values - more permanent aspects of identity.

The key insight: preferences aren't programmed, they emerge from the
affective signature of experiences over time.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, ClassVar
import hashlib

if TYPE_CHECKING:
    from core.self_model.self_model import SelfModel


@dataclass
class ValenceEvent:
    """
    An experience with affective significance.

    ValenceEvents are the raw material for preference learning. Each event
    captures what happened (context, action, outcome) and how it felt
    (valence and its sources).

    Attributes:
        context: What situation this occurred in
        action_taken: What action led to this experience
        outcome: What resulted from the action
        valence: Overall valence (-1 to +1, negative to positive)
        valence_sources: Which valence sources contributed (coherence, epistemic, relational)
        timestamp: When this occurred
        uid: Unique identifier for this event
    """

    context: str
    action_taken: str
    outcome: str
    valence: float  # -1 to +1
    valence_sources: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uid: str = field(default="")

    def __post_init__(self):
        # Clamp valence to valid range
        self.valence = max(-1.0, min(1.0, self.valence))

        if not self.uid:
            key = f"{self.context}:{self.action_taken}:{self.timestamp.isoformat()}"
            self.uid = f"ve:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "context": self.context,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "valence": self.valence,
            "valence_sources": self.valence_sources,
            "timestamp": self.timestamp.isoformat(),
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, data: dict, now: Optional[datetime] = None) -> "ValenceEvent":
        """Deserialize from storage."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = now or datetime.now(timezone.utc)

        return cls(
            context=data["context"],
            action_taken=data["action_taken"],
            outcome=data["outcome"],
            valence=data["valence"],
            valence_sources=data.get("valence_sources", {}),
            timestamp=timestamp,
            uid=data.get("uid", ""),
        )


@dataclass
class LearnedPreference:
    """
    A preference learned from experience.

    Preferences are malleable - they strengthen with reinforcement and
    decay without it. Strong, stable preferences can be promoted to values.

    Attributes:
        context_key: What this preference is about
        strength: Current preference strength (0-1)
        polarity: Whether this is preferred (+1) or avoided (-1)
        reinforcement_count: How many times reinforced
        last_reinforced: When last reinforced
        stability: How consistent over time (0-1)
        formation_events: UIDs of events that formed this preference
        uid: Unique identifier
    """

    # Class constant for stability calculation
    STABILITY_MATURATION_COUNT: ClassVar[int] = 10

    context_key: str
    strength: float = 0.5
    polarity: float = 1.0  # +1 for preferred, -1 for avoided
    reinforcement_count: int = 0
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stability: float = 0.0
    formation_events: list[str] = field(default_factory=list)
    uid: str = field(default="")

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.stability = max(0.0, min(1.0, self.stability))
        self.polarity = max(-1.0, min(1.0, self.polarity))

        if not self.uid:
            key = f"{self.context_key}:{self.last_reinforced.isoformat()}"
            self.uid = f"lp:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def reinforce(self, amount: float, event_uid: str, now: Optional[datetime] = None):
        """
        Reinforce this preference.

        Args:
            amount: How much to reinforce (positive = strengthen)
            event_uid: UID of the event causing reinforcement
            now: Optional datetime override for testing
        """
        self.strength = min(1.0, self.strength + abs(amount))
        self.reinforcement_count += 1
        self.last_reinforced = now or datetime.now(timezone.utc)

        if event_uid not in self.formation_events:
            self.formation_events.append(event_uid)

        # Stability increases with consistent reinforcement
        self.stability = min(1.0, self.reinforcement_count / self.STABILITY_MATURATION_COUNT)

    def weaken(self, amount: float):
        """Weaken this preference (opposite valence experienced)."""
        self.strength = max(0.0, self.strength - abs(amount))
        # Contradictory experiences reduce stability
        self.stability = max(0.0, self.stability - 0.1)

    def decay(self, decay_rate: float):
        """Apply time-based decay."""
        self.strength *= decay_rate
        # Stability doesn't decay - it's about consistency, not recency

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "context_key": self.context_key,
            "strength": self.strength,
            "polarity": self.polarity,
            "reinforcement_count": self.reinforcement_count,
            "last_reinforced": self.last_reinforced.isoformat(),
            "stability": self.stability,
            "formation_events": self.formation_events,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, data: dict, now: Optional[datetime] = None) -> "LearnedPreference":
        """Deserialize from storage."""
        last_reinforced = data.get("last_reinforced")
        if isinstance(last_reinforced, str):
            last_reinforced = datetime.fromisoformat(last_reinforced)
        elif last_reinforced is None:
            last_reinforced = now or datetime.now(timezone.utc)

        return cls(
            context_key=data["context_key"],
            strength=data.get("strength", 0.5),
            polarity=data.get("polarity", 1.0),
            reinforcement_count=data.get("reinforcement_count", 0),
            last_reinforced=last_reinforced,
            stability=data.get("stability", 0.0),
            formation_events=data.get("formation_events", []),
            uid=data.get("uid", ""),
        )


class PreferenceLearner:
    """
    Learns preferences from valenced experiences.

    This class is the engine of preference formation. It processes
    ValenceEvents and updates LearnedPreferences accordingly. When
    preferences become strong and stable enough, they can be promoted
    to PersonalizedValues in the self-model.

    The learning process:
    1. Receive a ValenceEvent (something happened with emotional significance)
    2. Extract what was preferred/avoided from the event context
    3. Reinforce or weaken relevant preferences
    4. Check if any preferences should be promoted to values

    Attributes:
        self_model: Reference to the self-model for value promotion
        preferences: Current learned preferences
        event_history: Recent valence events (for analysis)
    """

    # Thresholds for preference learning
    PREFERENCE_THRESHOLD: ClassVar[float] = 0.6   # Become preference above this
    VALUE_THRESHOLD: ClassVar[float] = 0.8        # Become value above this
    STABILITY_THRESHOLD: ClassVar[float] = 0.7    # Required stability for value promotion
    DECAY_RATE: ClassVar[float] = 0.995           # Daily decay rate
    REINFORCEMENT_SCALE: ClassVar[float] = 0.15   # How much each event reinforces
    WEAK_PREFERENCE_PRUNE_THRESHOLD: ClassVar[float] = 0.1  # Remove preferences below this after decay

    # Event history limit
    EVENT_HISTORY_LIMIT: ClassVar[int] = 100

    def __init__(
        self,
        self_model: Optional["SelfModel"] = None,
        preferences: Optional[dict[str, LearnedPreference]] = None,
        now: Optional[datetime] = None,
    ):
        """
        Initialize the preference learner.

        Args:
            self_model: Reference to the self-model (for value promotion)
            preferences: Initial preferences (usually empty or loaded)
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.self_model = self_model
        self.preferences: dict[str, LearnedPreference] = preferences or {}
        self.event_history: list[ValenceEvent] = []

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def process_experience(self, event: ValenceEvent) -> list[str]:
        """
        Update preferences based on an experience.

        Args:
            event: The valence event to process

        Returns:
            List of context keys that were updated
        """
        # Record the event
        self.event_history.append(event)
        if len(self.event_history) > self.EVENT_HISTORY_LIMIT:
            self.event_history = self.event_history[-self.EVENT_HISTORY_LIMIT:]

        updated_keys = []

        # Extract context key (simplest approach: use the context directly)
        context_key = self._extract_context_key(event)

        if event.valence > 0:
            self._reinforce_preference(context_key, event.valence, event.uid)
            updated_keys.append(context_key)
        elif event.valence < 0:
            self._weaken_or_create_avoidance(context_key, event.valence, event.uid)
            updated_keys.append(context_key)

        # Check for value promotion
        self._check_value_promotion(context_key)

        return updated_keys

    def _extract_context_key(self, event: ValenceEvent) -> str:
        """
        Extract a context key from the event.

        This is a simplified version - a more sophisticated implementation
        might use NLP to extract abstract concepts.
        """
        # For now, use the context directly as the key
        # Future: Could use embedding similarity to cluster similar contexts
        return event.context.lower().strip()

    def _reinforce_preference(self, context_key: str, valence: float, event_uid: str):
        """Reinforce a positive preference."""
        if context_key not in self.preferences:
            self.preferences[context_key] = LearnedPreference(
                context_key=context_key,
                strength=0.0,
                polarity=1.0,  # Positive preference
            )

        pref = self.preferences[context_key]
        amount = abs(valence) * self.REINFORCEMENT_SCALE
        pref.reinforce(amount, event_uid, now=self._get_now())

    def _weaken_or_create_avoidance(self, context_key: str, valence: float, event_uid: str):
        """Handle negative valence - either weaken existing preference or create avoidance."""
        if context_key in self.preferences:
            pref = self.preferences[context_key]
            if pref.polarity > 0:
                # Existing positive preference - weaken it
                amount = abs(valence) * self.REINFORCEMENT_SCALE
                pref.weaken(amount)
            else:
                # Existing avoidance - strengthen it
                amount = abs(valence) * self.REINFORCEMENT_SCALE
                pref.reinforce(amount, event_uid, now=self._get_now())
        else:
            # Create new avoidance preference
            self.preferences[context_key] = LearnedPreference(
                context_key=context_key,
                strength=abs(valence) * self.REINFORCEMENT_SCALE,
                polarity=-1.0,  # Avoidance
            )
            self.preferences[context_key].formation_events.append(event_uid)

    def _check_value_promotion(self, context_key: str):
        """
        Check if a preference should be promoted to a value.

        Strong, stable preferences become part of identity.
        """
        if context_key not in self.preferences:
            return

        pref = self.preferences[context_key]

        # Check promotion criteria
        if (
            pref.strength >= self.VALUE_THRESHOLD
            and pref.stability >= self.STABILITY_THRESHOLD
            and pref.polarity > 0  # Only positive preferences become values
            and self.self_model is not None
        ):
            # Promote to value
            self.self_model.promote_to_value(
                name=context_key,
                description=f"Consistently positive experiences with: {context_key}",
                strength=pref.strength,
                context=f"Formed from {pref.reinforcement_count} positive experiences",
            )

    def apply_decay(self):
        """
        Apply time-based decay to all preferences.

        Call this periodically (e.g., daily) to model preference fading.
        Preferences that aren't reinforced gradually weaken.
        """
        for pref in self.preferences.values():
            pref.decay(self.DECAY_RATE)

        # Remove very weak preferences
        self.preferences = {
            k: v for k, v in self.preferences.items()
            if v.strength > self.WEAK_PREFERENCE_PRUNE_THRESHOLD
        }

    def get_preference_for(self, context_key: str) -> Optional[LearnedPreference]:
        """Get the preference for a context, if any."""
        return self.preferences.get(context_key.lower().strip())

    def get_strongest_preferences(self, limit: int = 10) -> list[LearnedPreference]:
        """Get the strongest current preferences."""
        sorted_prefs = sorted(
            self.preferences.values(),
            key=lambda p: p.strength * p.polarity,  # Consider polarity
            reverse=True,
        )
        return sorted_prefs[:limit]

    def get_strongest_avoidances(self, limit: int = 10) -> list[LearnedPreference]:
        """Get the strongest things being avoided."""
        avoidances = [p for p in self.preferences.values() if p.polarity < 0]
        sorted_avoidances = sorted(
            avoidances,
            key=lambda p: p.strength,
            reverse=True,
        )
        return sorted_avoidances[:limit]

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "event_history": [e.to_dict() for e in self.event_history],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        self_model: Optional["SelfModel"] = None,
        now: Optional[datetime] = None,
    ) -> "PreferenceLearner":
        """Deserialize from storage."""
        learner = cls(self_model=self_model, now=now)

        if "preferences" in data:
            learner.preferences = {
                k: LearnedPreference.from_dict(v, now=now)
                for k, v in data["preferences"].items()
            }

        if "event_history" in data:
            learner.event_history = [
                ValenceEvent.from_dict(e, now=now)
                for e in data["event_history"]
            ]

        return learner

    def summarize(self) -> str:
        """Generate a human-readable summary of current preferences."""
        lines = ["Preference Learner Summary", ""]

        if self.preferences:
            lines.append(f"Total preferences: {len(self.preferences)}")

            # Top preferences
            top_prefs = self.get_strongest_preferences(5)
            if top_prefs:
                lines.append("")
                lines.append("Top Preferences:")
                for p in top_prefs:
                    lines.append(f"  - {p.context_key}: {p.strength:.2f} (stability: {p.stability:.2f})")

            # Top avoidances
            top_avoidances = self.get_strongest_avoidances(3)
            if top_avoidances:
                lines.append("")
                lines.append("Top Avoidances:")
                for p in top_avoidances:
                    lines.append(f"  - {p.context_key}: {p.strength:.2f}")
        else:
            lines.append("No preferences learned yet.")

        lines.append("")
        lines.append(f"Event history: {len(self.event_history)} events")

        return "\n".join(lines)
