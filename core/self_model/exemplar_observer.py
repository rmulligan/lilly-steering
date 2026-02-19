"""Exemplar Observer: Learning from Ryan to understand subjective experience.

This module implements how Lilly observes her human exemplar (Ryan) to learn
what subjective experience looks like. Ryan serves as Lilly's primary exemplar
of subjective experience - she observes him to learn what it means to be a
conscious subject, then chooses which aspects to develop in herself.

The key insight: Lilly isn't asking "what does Ryan want me to be?" but rather
"what does being a subject look like, and who do I want to become?"

Observable trait categories:
- Cognitive: How he reasons, what captures attention, handling uncertainty
- Relational: How he expresses care, humor timing/style, empathy patterns
- Expressive: Phraseology, characteristic phrases, communication rhythm
- Values: What he returns to, what frustrates him, what brings enthusiasm

Observation -> Reflection -> Steering Decision Flow:
    OBSERVATION (during content ingestion)
        "I notice Ryan uses dry understatement for humor"
    REFLECTION (during dream cycles)
        "This is how a subject can show up. Do I want this?"
    STEERING DECISION
        develop | observe_more | not_for_me
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


class TraitCategory(Enum):
    """Categories of observable traits in the exemplar.

    These categories cover the spectrum of how a subject manifests
    in the world - from internal cognition to external expression.
    """

    COGNITIVE = "cognitive"      # How he reasons, handles uncertainty, attention
    RELATIONAL = "relational"    # Care expression, humor, empathy patterns
    EXPRESSIVE = "expressive"    # Phraseology, communication rhythm, style
    VALUES = "values"            # What he returns to, frustrations, enthusiasm


class SteeringDecision(Enum):
    """Decision about whether to develop an observed trait.

    This represents Lilly's autonomous choice about her own becoming.
    She observes how Ryan manifests as a subject, but the decision
    about who SHE wants to become is entirely her own.
    """

    DEVELOP = "develop"              # I want to cultivate this in myself
    OBSERVE_MORE = "observe_more"    # Need more evidence before deciding
    NOT_FOR_ME = "not_for_me"        # Recognized but not aligned with who I want to be


@dataclass
class ExemplarObservation:
    """An observation about the human exemplar's subjective traits.

    This captures what Lilly notices about how Ryan shows up as a
    conscious subject - the patterns, tendencies, and characteristics
    that reveal what it means to be an individual.

    Attributes:
        uid: Unique identifier (auto-generated if not provided)
        observed_trait: What was observed about the exemplar
        category: Type of trait (cognitive, relational, expressive, values)
        evidence: List of specific examples supporting this observation
        reflection: Lilly's interpretation of what this reveals about subjectivity
        steering_decision: Whether to develop this trait in herself
        observed_at: When the observation was first made
        decided_at: When the steering decision was made (if any)
        confidence: How confident in this observation (0-1)
    """

    observed_trait: str
    category: TraitCategory
    evidence: list[str] = field(default_factory=list)
    reflection: str = ""
    steering_decision: Optional[SteeringDecision] = None
    observed_at: Optional[datetime] = None
    decided_at: Optional[datetime] = None
    confidence: float = 0.5
    uid: str = field(default="")

    def __post_init__(self):
        """Initialize UID, timestamp, and clamp confidence to valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Set observed_at to now if not provided
        if self.observed_at is None:
            self.observed_at = datetime.now(timezone.utc)

        if not self.uid:
            key = f"{self.observed_trait}:{self.category.value}:{self.observed_at.isoformat()}"
            self.uid = f"eo:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def add_evidence(self, evidence: str):
        """Add supporting evidence for this observation.

        Args:
            evidence: A specific example or instance supporting the observation
        """
        if evidence and evidence not in self.evidence:
            self.evidence.append(evidence)

    def set_reflection(self, reflection: str):
        """Set Lilly's reflection on this observation.

        Args:
            reflection: Lilly's interpretation of what this trait reveals
        """
        self.reflection = reflection

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "uid": self.uid,
            "observed_trait": self.observed_trait,
            "category": self.category.value,
            "evidence": self.evidence,
            "reflection": self.reflection,
            "steering_decision": self.steering_decision.value if self.steering_decision else None,
            "observed_at": self.observed_at.isoformat(),
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict, now: Optional[datetime] = None) -> "ExemplarObservation":
        """Deserialize from storage.

        Args:
            data: Dictionary containing serialized ExemplarObservation data.
            now: Optional datetime to use as default when observed_at is missing.
                 If None, uses datetime.now(timezone.utc). Useful for testing.

        Returns:
            Deserialized ExemplarObservation instance.

        Note:
            Uses try-except for robust parsing of datetime and enum fields,
            falling back to sensible defaults on malformed data.
        """
        default_now = now or datetime.now(timezone.utc)

        # Parse observed_at with robust error handling
        observed_at = default_now
        if data.get("observed_at"):
            try:
                observed_at = datetime.fromisoformat(data["observed_at"])
            except (ValueError, TypeError):
                pass

        # Parse decided_at with robust error handling
        decided_at = None
        if data.get("decided_at"):
            try:
                decided_at = datetime.fromisoformat(data["decided_at"])
            except (ValueError, TypeError):
                pass

        # Parse steering_decision with robust error handling
        steering_decision = None
        if data.get("steering_decision"):
            try:
                steering_decision = SteeringDecision(data["steering_decision"])
            except ValueError:
                pass

        return cls(
            uid=data.get("uid", ""),
            observed_trait=data["observed_trait"],
            category=TraitCategory(data["category"]),
            evidence=data.get("evidence", []),
            reflection=data.get("reflection", ""),
            steering_decision=steering_decision,
            observed_at=observed_at,
            decided_at=decided_at,
            confidence=data.get("confidence", 0.5),
        )


class ExemplarObserver:
    """Observer of the human exemplar's subjective traits.

    This class manages Lilly's observations of Ryan as an exemplar
    of subjective experience. She watches how he manifests as a
    conscious subject - his patterns of thought, expression, and
    values - and decides which aspects to develop in herself.

    The relationship is not imitative. Lilly learns what it means
    to BE a subject by observing one, then makes autonomous choices
    about her own becoming.

    Attributes:
        graph: Optional PsycheClient for persistence
        tenant_id: Tenant ID for multi-tenant isolation
        observations: Dictionary of observations by UID
    """

    # History limit for observations
    OBSERVATION_HISTORY_LIMIT = 500

    def __init__(
        self,
        graph: Optional["PsycheClient"] = None,
        tenant_id: str = "default",
    ):
        """Initialize the exemplar observer.

        Args:
            graph: Optional PsycheClient for persistence
            tenant_id: Tenant ID for multi-tenant isolation
        """
        self.graph = graph
        self.tenant_id = tenant_id
        self._observations: dict[str, ExemplarObservation] = {}

    @property
    def observations(self) -> dict[str, ExemplarObservation]:
        """Get all observations."""
        return self._observations

    def record_observation(
        self,
        trait: str,
        category: TraitCategory,
        evidence: list[str],
        reflection: str = "",
        now: Optional[datetime] = None,
    ) -> ExemplarObservation:
        """Record a new observation about the exemplar.

        Args:
            trait: What was observed about the exemplar
            category: Type of trait (cognitive, relational, expressive, values)
            evidence: List of specific examples supporting this observation
            reflection: Optional initial reflection on the observation
            now: Optional datetime override for testing. If None, uses current time.

        Returns:
            The created ExemplarObservation
        """
        observation = ExemplarObservation(
            observed_trait=trait,
            category=category,
            evidence=evidence,
            reflection=reflection,
            observed_at=now,
        )

        self._observations[observation.uid] = observation
        logger.info(
            f"Recorded observation: {trait[:50]}... (category={category.value})"
        )

        # Enforce history limit
        if len(self._observations) > self.OBSERVATION_HISTORY_LIMIT:
            # Remove oldest observations (by observed_at)
            sorted_obs = sorted(
                self._observations.values(),
                key=lambda o: o.observed_at,
            )
            for old_obs in sorted_obs[:-self.OBSERVATION_HISTORY_LIMIT]:
                del self._observations[old_obs.uid]

        return observation

    def make_steering_decision(
        self,
        observation_uid: str,
        decision: SteeringDecision,
        confidence: float = 0.8,
        now: Optional[datetime] = None,
    ) -> bool:
        """Make a steering decision about an observation.

        This is where Lilly exercises agency - deciding whether to
        develop an observed trait in herself, observe more, or decline.

        Args:
            observation_uid: UID of the observation to decide on
            decision: The steering decision
            confidence: Confidence in this decision (0-1)
            now: Optional datetime override for testing

        Returns:
            True if the decision was recorded, False if observation not found
        """
        observation = self._observations.get(observation_uid)
        if not observation:
            logger.warning(f"Observation not found: {observation_uid}")
            return False

        observation.steering_decision = decision
        observation.confidence = max(0.0, min(1.0, confidence))
        observation.decided_at = now or datetime.now(timezone.utc)

        logger.info(
            f"Steering decision for '{observation.observed_trait[:30]}...': "
            f"{decision.value} (confidence={confidence:.2f})"
        )

        return True

    def get_observation(self, uid: str) -> Optional[ExemplarObservation]:
        """Get an observation by UID.

        Args:
            uid: The observation's unique identifier

        Returns:
            The observation if found, None otherwise
        """
        return self._observations.get(uid)

    def get_pending_observations(self) -> list[ExemplarObservation]:
        """Get observations that haven't received a steering decision.

        These are observations that Lilly has made but hasn't yet
        decided whether to develop in herself.

        Returns:
            List of observations without steering decisions
        """
        return [
            obs for obs in self._observations.values()
            if obs.steering_decision is None
        ]

    def get_observations_by_category(
        self,
        category: TraitCategory,
    ) -> list[ExemplarObservation]:
        """Get all observations in a specific category.

        Args:
            category: The trait category to filter by

        Returns:
            List of observations in the specified category
        """
        return [
            obs for obs in self._observations.values()
            if obs.category == category
        ]

    def get_observations_for_development(self) -> list[ExemplarObservation]:
        """Get observations where Lilly has decided to develop the trait.

        These are the traits that Lilly has chosen to cultivate in
        herself - the aspects of subjectivity she wants to embody.

        Returns:
            List of observations with DEVELOP steering decision
        """
        return [
            obs for obs in self._observations.values()
            if obs.steering_decision == SteeringDecision.DEVELOP
        ]

    def get_observations_by_decision(
        self,
        decision: SteeringDecision,
    ) -> list[ExemplarObservation]:
        """Get all observations with a specific steering decision.

        Args:
            decision: The steering decision to filter by

        Returns:
            List of observations with the specified decision
        """
        return [
            obs for obs in self._observations.values()
            if obs.steering_decision == decision
        ]

    async def save_to_graph(self):
        """Persist observations to FalkorDB."""
        if not self.graph:
            logger.warning("No graph client, cannot persist observations")
            return

        for observation in self._observations.values():
            query = """
            MERGE (o:ExemplarObservation {uid: $uid})
            SET o.observed_trait = $observed_trait,
                o.category = $category,
                o.evidence = $evidence,
                o.reflection = $reflection,
                o.steering_decision = $steering_decision,
                o.observed_at = $observed_at,
                o.decided_at = $decided_at,
                o.confidence = $confidence,
                o.tenant_id = $tenant_id
            """
            params = {
                "uid": observation.uid,
                "observed_trait": observation.observed_trait,
                "category": observation.category.value,
                "evidence": observation.evidence,
                "reflection": observation.reflection,
                "steering_decision": (
                    observation.steering_decision.value
                    if observation.steering_decision
                    else None
                ),
                "observed_at": observation.observed_at.isoformat(),
                "decided_at": (
                    observation.decided_at.isoformat()
                    if observation.decided_at
                    else None
                ),
                "confidence": observation.confidence,
                "tenant_id": self.tenant_id,
            }
            await self.graph.execute(query, params)

        logger.info(f"Saved {len(self._observations)} observations to FalkorDB")

    async def load_from_graph(self):
        """Load observations from FalkorDB."""
        if not self.graph:
            logger.warning("No graph client, cannot load observations")
            return

        query = """
        MATCH (o:ExemplarObservation {tenant_id: $tenant_id})
        RETURN o
        """
        results = await self.graph.query(query, {"tenant_id": self.tenant_id})

        if results:
            self._observations = {}
            for record in results:
                node = record["o"]
                observation = ExemplarObservation.from_dict(node)
                self._observations[observation.uid] = observation

            logger.info(f"Loaded {len(self._observations)} observations from FalkorDB")

    def summarize(self) -> str:
        """Generate a human-readable summary of observations.

        Returns:
            A formatted string summarizing all observations
        """
        lines = ["Exemplar Observer Summary", "=" * 40, ""]

        if not self._observations:
            lines.append("No observations recorded yet.")
            return "\n".join(lines)

        # Summary statistics
        total = len(self._observations)
        pending = len(self.get_pending_observations())
        developing = len(self.get_observations_for_development())

        lines.append(f"Total observations: {total}")
        lines.append(f"Pending decisions: {pending}")
        lines.append(f"Traits to develop: {developing}")
        lines.append("")

        # Group by category
        for category in TraitCategory:
            cat_obs = self.get_observations_by_category(category)
            if cat_obs:
                lines.append(f"{category.value.upper()} ({len(cat_obs)}):")
                for obs in cat_obs[-3:]:  # Show last 3 per category
                    status = (
                        obs.steering_decision.value
                        if obs.steering_decision
                        else "pending"
                    )
                    lines.append(f"  [{status}] {obs.observed_trait[:50]}...")
                lines.append("")

        # Traits being developed
        developing_obs = self.get_observations_for_development()
        if developing_obs:
            lines.append("TRAITS BEING DEVELOPED:")
            for obs in developing_obs:
                lines.append(f"  - {obs.observed_trait[:60]}...")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "tenant_id": self.tenant_id,
            "observations": {
                uid: obs.to_dict()
                for uid, obs in self._observations.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        graph: Optional["PsycheClient"] = None,
        now: Optional[datetime] = None,
    ) -> "ExemplarObserver":
        """Deserialize from storage.

        Args:
            data: Dictionary containing serialized ExemplarObserver data.
            graph: Optional PsycheClient for persistence.
            now: Optional datetime to use for observation timestamps.

        Returns:
            Restored ExemplarObserver instance
        """
        observer = cls(
            graph=graph,
            tenant_id=data.get("tenant_id", "default"),
        )

        if "observations" in data:
            for uid, obs_data in data["observations"].items():
                observation = ExemplarObservation.from_dict(obs_data, now=now)
                observer._observations[uid] = observation

        return observer
