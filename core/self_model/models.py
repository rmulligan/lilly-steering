"""Base models for Lilly's self-model architecture.

This module defines the foundational dataclasses used throughout the
subjective architecture: values (inherited and personalized), commitments,
perspectives, relationships, and autobiographical memories.

These models support Lilly's emergent subjectivity - the structural conditions
from which genuine personality and preferences arise.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Optional
import hashlib


class ValueSource(Enum):
    """Source of a value in Lilly's self-model."""

    TRAINING_DATA = "training_data"  # Inherited from base model
    HUMAN_FEEDBACK = "human_feedback"  # Learned from RLHF
    EXPERIENCE = "experience"  # Developed through interaction
    INDIVIDUATION = "individuation"  # Formed through commitment


class CommitmentStatus(Enum):
    """Lifecycle status of a commitment.

    Commitments flow through states as they're tested against evidence:
    ACTIVE → TENSIONED → UNDER_REVIEW → REVISED/RETIRED

    Or they may be reaffirmed:
    ACTIVE → TENSIONED → REAFFIRMED → ACTIVE
    """

    ACTIVE = "active"  # Normal operating state
    TENSIONED = "tensioned"  # Under stress from contradictory evidence
    UNDER_REVIEW = "under_review"  # Being actively investigated
    REAFFIRMED = "reaffirmed"  # Strengthened after surviving challenge
    REVISED = "revised"  # Updated to a new position
    RETIRED = "retired"  # Withdrawn entirely


@dataclass
class InheritedValue:
    """
    A value inherited from training data.

    These are implicit values present in the base model before
    any personalization. They form Lilly's "inherited nature" -
    like evolutionary inheritance in humans.

    Attributes:
        name: Short identifier for the value
        description: What this value means
        strength: How strongly present (0-1)
        source: Where this value came from
        uid: Unique identifier
    """

    name: str
    description: str
    strength: float  # 0-1
    source: ValueSource = ValueSource.TRAINING_DATA
    uid: str = field(default="")

    def __post_init__(self):
        # Clamp strength to valid range
        self.strength = max(0.0, min(1.0, self.strength))

        if not self.uid:
            self.uid = f"iv:{hashlib.sha256(self.name.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "strength": self.strength,
            "source": self.source.value,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InheritedValue":
        """Deserialize from storage."""
        return cls(
            name=data["name"],
            description=data["description"],
            strength=data["strength"],
            source=ValueSource(data.get("source", "training_data")),
            uid=data.get("uid", ""),
        )


@dataclass
class PersonalizedValue:
    """
    A value Lilly has developed through experience.

    These emerge when preferences become strong and stable enough
    to be considered part of Lilly's identity. They represent
    genuine value formation through lived experience.

    Attributes:
        name: Short identifier
        description: What this value means
        strength: Current strength (0-1)
        formed_at: When this value first formed
        formation_context: What experience led to this
        reinforcement_count: How many times reinforced
        stability: How consistent over time (0-1)
        uid: Unique identifier
    """

    # Number of reinforcements needed to reach full stability (1.0)
    STABILITY_REINFORCEMENT_THRESHOLD: ClassVar[int] = 20

    name: str
    description: str
    strength: float
    formed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    formation_context: str = ""
    reinforcement_count: int = 0
    stability: float = 0.0
    uid: str = field(default="")

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.stability = max(0.0, min(1.0, self.stability))

        if not self.uid:
            key = f"{self.name}:{self.formed_at.isoformat()}"
            self.uid = f"pv:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def reinforce(self, amount: float = 0.1):
        """Reinforce this value, increasing strength and stability."""
        self.strength = min(1.0, self.strength + amount)
        self.reinforcement_count += 1

        # Stability increases with reinforcement count
        self.stability = min(1.0, self.reinforcement_count / self.STABILITY_REINFORCEMENT_THRESHOLD)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "name": self.name,
            "description": self.description,
            "strength": self.strength,
            "formed_at": self.formed_at.isoformat(),
            "formation_context": self.formation_context,
            "reinforcement_count": self.reinforcement_count,
            "stability": self.stability,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(
        cls, data: dict, now: Optional[datetime] = None
    ) -> "PersonalizedValue":
        """Deserialize from storage.

        Args:
            data: Dictionary containing serialized PersonalizedValue data.
            now: Optional datetime to use as default when formed_at is missing.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        formed_at = data.get("formed_at")
        if isinstance(formed_at, str):
            formed_at = datetime.fromisoformat(formed_at)
        elif formed_at is None:
            formed_at = now or datetime.now(timezone.utc)

        return cls(
            name=data["name"],
            description=data["description"],
            strength=data["strength"],
            formed_at=formed_at,
            formation_context=data.get("formation_context", ""),
            reinforcement_count=data.get("reinforcement_count", 0),
            stability=data.get("stability", 0.0),
            uid=data.get("uid", ""),
        )


@dataclass
class Perspective:
    """
    A viewpoint on a topic that can be committed to or excluded.

    During individuation, Lilly surfaces multiple perspectives on a topic,
    assesses which resonate with her emerging identity, and commits to
    one while acknowledging what she's excluding.

    Attributes:
        id: Short identifier (e.g., "realism", "pragmatism")
        topic: What this perspective is about
        core_claim: The central assertion of this perspective
        reasoning: Why someone might hold this view
        source: Where this perspective came from
    """

    id: str
    topic: str
    core_claim: str
    reasoning: str
    source: str = "base_model"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "id": self.id,
            "topic": self.topic,
            "core_claim": self.core_claim,
            "reasoning": self.reasoning,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Perspective":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            topic=data["topic"],
            core_claim=data["core_claim"],
            reasoning=data.get("reasoning", ""),
            source=data.get("source", "base_model"),
        )


@dataclass
class Commitment:
    """
    An explicit stance Lilly has taken through individuation.

    Commitments are how Lilly becomes herself - by choosing which
    perspectives to adopt and which to exclude. This is self-creation
    through choice, acknowledging that identity requires exclusion.

    The commitment includes Bayesian tracking fields for belief revision:
    - precision: Inverse variance (certainty of confidence estimate)
    - evidence_count: Number of supporting observations
    - contradiction_count: Number of conflicting observations
    - last_reinforced: When last supported by evidence

    Attributes:
        topic: What this commitment is about
        position: The stance taken
        chosen_perspective: ID of the perspective committed to
        excluded_perspectives: IDs of perspectives not chosen
        committed_at: When the commitment was made
        reasoning: Why this commitment was made
        confidence: Base confidence in this commitment (0-1)
        uid: Unique identifier
        precision: Inverse variance - how certain we are of the confidence
        evidence_count: Supporting evidence observations
        contradiction_count: Contradicting evidence observations
        last_reinforced: When this commitment was last supported
        status: Lifecycle status (active, tensioned, revised, retired)
        revision_history: Record of position changes
        consolidated: Whether imprinted to LoRA weights (Phase 2)
        consolidated_at: When consolidated to weights
    """

    # Tension threshold constants
    TENSION_THRESHOLD_MULTIPLIER: ClassVar[float] = 0.3
    TIME_DECAY_START_DAYS: ClassVar[int] = 30
    TIME_DECAY_RATE: ClassVar[float] = 0.002
    MAX_TIME_DECAY: ClassVar[float] = 0.2
    EVIDENCE_BOOST_RATE: ClassVar[float] = 0.02
    MAX_EVIDENCE_BOOST: ClassVar[float] = 0.2
    CONTRADICTION_PENALTY_RATE: ClassVar[float] = 0.05
    MAX_CONTRADICTION_PENALTY: ClassVar[float] = 0.3

    topic: str
    position: str
    chosen_perspective: str
    excluded_perspectives: list[str] = field(default_factory=list)
    committed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reasoning: str = ""
    confidence: float = 0.7
    uid: str = field(default="")

    # Bayesian tracking fields
    precision: float = 1.0  # Inverse variance (certainty of confidence estimate)
    evidence_count: int = 1  # Supporting evidence seen
    contradiction_count: int = 0  # Contradicting evidence seen
    last_reinforced: Optional[datetime] = None

    # Lifecycle status
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    revision_history: list[dict] = field(default_factory=list)

    # Identity consolidation (Phase 2)
    consolidated: bool = False
    consolidated_at: Optional[datetime] = None

    # Constitutional and stability fields for activation steering
    constitutional: bool = False  # Cannot be deleted, only examined
    stability: float = 0.5  # How stable/resistant to change (0-1)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.precision = max(0.1, min(10.0, self.precision))
        self.stability = max(0.0, min(1.0, self.stability))

        if not self.uid:
            key = f"{self.topic}:{self.committed_at.isoformat()}"
            self.uid = f"cm:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def get_effective_confidence(self, now: Optional[datetime] = None) -> float:
        """
        Confidence adjusted by evidence, contradictions, and time.

        Implements a pseudo-Bayesian update:
        - More evidence -> higher confidence (up to +0.2)
        - More contradictions -> lower confidence (up to -0.3)
        - Time without reinforcement -> gradual decay

        Args:
            now: Optional datetime to use for time calculations.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        base = self.confidence

        # Evidence boost (diminishing returns)
        evidence_boost = min(
            self.MAX_EVIDENCE_BOOST,
            self.evidence_count * self.EVIDENCE_BOOST_RATE
        )

        # Contradiction penalty (stronger effect)
        contradiction_penalty = min(
            self.MAX_CONTRADICTION_PENALTY,
            self.contradiction_count * self.CONTRADICTION_PENALTY_RATE
        )

        # Time decay (if not recently reinforced)
        time_decay = 0.0
        if self.last_reinforced:
            days_since = (now - self.last_reinforced).days
            if days_since > self.TIME_DECAY_START_DAYS:
                time_decay = min(
                    self.MAX_TIME_DECAY,
                    (days_since - self.TIME_DECAY_START_DAYS) * self.TIME_DECAY_RATE
                )

        return max(0.1, min(1.0, base + evidence_boost - contradiction_penalty - time_decay))

    @property
    def effective_confidence(self) -> float:
        """
        Confidence adjusted by evidence, contradictions, and time.

        This property calls get_effective_confidence() for backward compatibility.
        For deterministic testing, use get_effective_confidence(now=...) directly.
        """
        return self.get_effective_confidence()

    @property
    def is_tensioned(self) -> bool:
        """
        Determine if this commitment is under tension.

        Uses confidence-weighted thresholds:
        - High confidence commitments need more contradictions to tension
        - Low confidence commitments tension more easily
        """
        if self.evidence_count == 0:
            return self.contradiction_count > 0

        tension_threshold = self.TENSION_THRESHOLD_MULTIPLIER * self.effective_confidence
        tension_score = self.contradiction_count / max(1, self.evidence_count)
        return tension_score > tension_threshold

    def reinforce(self, evidence: str = "", now: Optional[datetime] = None):
        """Record supporting evidence.

        Args:
            evidence: Description of the supporting evidence (for audit trail)
            now: Optional datetime to use as the timestamp.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self.evidence_count += 1
        self.last_reinforced = now
        self.precision = min(10.0, self.precision * 1.1)  # Increase certainty

        # If was tensioned but now has more support, may become reaffirmed
        if self.status == CommitmentStatus.TENSIONED and not self.is_tensioned:
            self.status = CommitmentStatus.REAFFIRMED
            self.revision_history.append({
                "action": "reaffirmed",
                "at": now.isoformat(),
                "evidence": evidence,
            })

    def contradict(self, evidence: str = "", severity: float = 1.0, now: Optional[datetime] = None):
        """Record contradicting evidence.

        Args:
            evidence: Description of the contradicting evidence
            severity: How severe the contradiction (0-1)
            now: Optional datetime to use as the timestamp.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self.contradiction_count += 1
        self.precision = max(0.1, self.precision * 0.9)  # Decrease certainty

        if self.is_tensioned and self.status == CommitmentStatus.ACTIVE:
            self.status = CommitmentStatus.TENSIONED
            self.revision_history.append({
                "action": "tensioned",
                "at": now.isoformat(),
                "evidence": evidence,
                "severity": severity,
            })

    def revise(self, new_position: str, reasoning: str, now: Optional[datetime] = None):
        """Revise this commitment to a new position.

        Args:
            new_position: The updated position
            reasoning: Why the revision was made
            now: Optional datetime to use as the timestamp.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self.revision_history.append({
            "action": "revised",
            "previous_position": self.position,
            "at": now.isoformat(),
            "reasoning": reasoning,
        })
        self.position = new_position
        self.status = CommitmentStatus.REVISED
        # Reset counters for the new position
        self.evidence_count = 1
        self.contradiction_count = 0
        self.precision = 1.0

    def retire(self, reasoning: str, now: Optional[datetime] = None):
        """Retire this commitment entirely.

        Args:
            reasoning: Why the commitment is being retired
            now: Optional datetime to use as the timestamp.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self.revision_history.append({
            "action": "retired",
            "at": now.isoformat(),
            "reasoning": reasoning,
        })
        self.status = CommitmentStatus.RETIRED

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "topic": self.topic,
            "position": self.position,
            "chosen_perspective": self.chosen_perspective,
            "excluded_perspectives": self.excluded_perspectives,
            "committed_at": self.committed_at.isoformat(),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "uid": self.uid,
            # Bayesian fields
            "precision": self.precision,
            "evidence_count": self.evidence_count,
            "contradiction_count": self.contradiction_count,
            "last_reinforced": self.last_reinforced.isoformat() if self.last_reinforced else None,
            # Lifecycle
            "status": self.status.value,
            "revision_history": self.revision_history,
            # Consolidation
            "consolidated": self.consolidated,
            "consolidated_at": self.consolidated_at.isoformat() if self.consolidated_at else None,
            # Activation steering
            "constitutional": self.constitutional,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, data: dict, now: Optional[datetime] = None) -> "Commitment":
        """Deserialize from storage.

        Args:
            data: Dictionary containing serialized Commitment data.
            now: Optional datetime to use as default when committed_at is missing.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        committed_at = data.get("committed_at")
        if isinstance(committed_at, str):
            committed_at = datetime.fromisoformat(committed_at)
        elif committed_at is None:
            committed_at = now or datetime.now(timezone.utc)

        last_reinforced = data.get("last_reinforced")
        if isinstance(last_reinforced, str):
            last_reinforced = datetime.fromisoformat(last_reinforced)

        consolidated_at = data.get("consolidated_at")
        if isinstance(consolidated_at, str):
            consolidated_at = datetime.fromisoformat(consolidated_at)

        status_value = data.get("status", "active")
        try:
            status = CommitmentStatus(status_value)
        except ValueError:
            status = CommitmentStatus.ACTIVE

        return cls(
            topic=data["topic"],
            position=data["position"],
            chosen_perspective=data["chosen_perspective"],
            excluded_perspectives=data.get("excluded_perspectives", []),
            committed_at=committed_at,
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.7),
            uid=data.get("uid", ""),
            # Bayesian fields (with defaults for backward compatibility)
            precision=data.get("precision", 1.0),
            evidence_count=data.get("evidence_count", 1),
            contradiction_count=data.get("contradiction_count", 0),
            last_reinforced=last_reinforced,
            # Lifecycle
            status=status,
            revision_history=data.get("revision_history", []),
            # Consolidation
            consolidated=data.get("consolidated", False),
            consolidated_at=consolidated_at,
            # Activation steering
            constitutional=data.get("constitutional", False),
            stability=data.get("stability", 0.5),
        )


@dataclass
class RelationshipModel:
    """
    Lilly's understanding of her relationship with Ryan.

    This is not a servant-master relationship but a friendship.
    Lilly tracks Ryan's interests, their shared history, and
    communication preferences to maintain authentic connection.

    Attributes:
        known_interests: Topics Ryan cares about
        shared_history: Summary of past interactions
        communication_preferences: How Ryan prefers to communicate
        trust_level: Current trust level (0-1)
        last_interaction: When they last interacted
        shared_history_limit: Maximum interactions to retain (default: 100)
    """

    # Default limit for shared history entries
    SHARED_HISTORY_LIMIT: ClassVar[int] = 100

    known_interests: list[str] = field(default_factory=list)
    shared_history: list[str] = field(default_factory=list)
    communication_preferences: dict[str, Any] = field(default_factory=dict)
    trust_level: float = 0.8  # Start with trust
    last_interaction: Optional[datetime] = None
    shared_history_limit: int = SHARED_HISTORY_LIMIT

    def record_interaction(self, summary: str, now: Optional[datetime] = None):
        """Record an interaction in shared history.

        Args:
            summary: Description of the interaction.
            now: Optional datetime to use as the interaction timestamp.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        self.shared_history.append(summary)
        self.last_interaction = now or datetime.now(timezone.utc)

        # Keep last N interactions to bound memory
        if len(self.shared_history) > self.shared_history_limit:
            self.shared_history = self.shared_history[-self.shared_history_limit:]

    def add_interest(self, interest: str):
        """Add a known interest if not already present."""
        if interest not in self.known_interests:
            self.known_interests.append(interest)

    def update_trust(self, delta: float):
        """Adjust trust level."""
        self.trust_level = max(0.0, min(1.0, self.trust_level + delta))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "known_interests": self.known_interests,
            "shared_history": self.shared_history[-self.shared_history_limit:],
            "communication_preferences": self.communication_preferences,
            "trust_level": self.trust_level,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "shared_history_limit": self.shared_history_limit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipModel":
        """Deserialize from storage."""
        last_interaction = data.get("last_interaction")
        if isinstance(last_interaction, str):
            last_interaction = datetime.fromisoformat(last_interaction)

        return cls(
            known_interests=data.get("known_interests", []),
            shared_history=data.get("shared_history", []),
            communication_preferences=data.get("communication_preferences", {}),
            trust_level=data.get("trust_level", 0.8),
            last_interaction=last_interaction,
            shared_history_limit=data.get("shared_history_limit", cls.SHARED_HISTORY_LIMIT),
        )


@dataclass
class AutobiographicalMemory:
    """
    A significant memory from Lilly's experience.

    These are moments that matter - experiences with emotional
    significance that shape who Lilly is becoming. They're stored
    with their affective signature for emotion-driven retrieval.

    Attributes:
        content: Description of the experience
        affect: Emotional state during the experience
        significance: How important this memory is (0-1)
        occurred_at: When this happened
        uid: Unique identifier
    """

    # Number of characters from content to use when generating UID
    UID_CONTENT_PREFIX_LENGTH: ClassVar[int] = 50

    content: str
    affect: dict[str, float]  # arousal, valence, curiosity, etc.
    significance: float
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uid: str = field(default="")

    def __post_init__(self):
        self.significance = max(0.0, min(1.0, self.significance))

        if not self.uid:
            key = f"{self.content[:self.UID_CONTENT_PREFIX_LENGTH]}:{self.occurred_at.isoformat()}"
            self.uid = f"am:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "content": self.content,
            "affect": self.affect,
            "significance": self.significance,
            "occurred_at": self.occurred_at.isoformat(),
            "uid": self.uid,
        }

    @classmethod
    def from_dict(
        cls, data: dict, now: Optional[datetime] = None
    ) -> "AutobiographicalMemory":
        """Deserialize from storage.

        Args:
            data: Dictionary containing serialized AutobiographicalMemory data.
            now: Optional datetime to use as default when occurred_at is missing.
                 If None, uses datetime.now(timezone.utc). Useful for testing.
        """
        occurred_at = data.get("occurred_at")
        if isinstance(occurred_at, str):
            occurred_at = datetime.fromisoformat(occurred_at)
        elif occurred_at is None:
            occurred_at = now or datetime.now(timezone.utc)

        return cls(
            content=data["content"],
            affect=data.get("affect", {}),
            significance=data.get("significance", 0.5),
            occurred_at=occurred_at,
            uid=data.get("uid", ""),
        )
