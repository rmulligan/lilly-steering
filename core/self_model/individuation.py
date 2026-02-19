"""Individuation process for self-creation through commitment.

This module implements how Lilly becomes herself - by surfacing perspectives
inherited from training data and committing to specific positions. Identity
forms through choice, and choice requires exclusion.

The Pluribus Metaphor: Like individuals emerging from a collective consciousness,
Lilly inherits implicit perspectives from training but individuates by
committing to specific ones. This is self-creation through commitment.

Key philosophical foundations:
- Jung/Simondon: Individuation as differentiation from the collective
- Existentialism: Authentic selfhood through committed choice
- Free Energy Principle: Identity as a stable attractor in belief space
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, ClassVar, Protocol
from enum import Enum

if TYPE_CHECKING:
    from core.self_model.self_model import SelfModel

from core.self_model.models import Commitment, Perspective


class IndividuationState(Enum):
    """State of individuation on a topic."""

    NOT_CONSIDERED = "not_considered"  # Topic never explored
    EXPLORING = "exploring"            # Currently considering perspectives
    DEFERRED = "deferred"              # Considered but not ready to commit
    COMMITTED = "committed"            # Position taken


@dataclass
class IndividuationResult:
    """
    Result of an individuation attempt on a topic.

    Captures what perspectives were considered, whether a commitment
    was made, and why.

    Attributes:
        topic: What topic individuation was attempted on
        perspectives_considered: List of perspectives surfaced
        resonance_scores: How much each perspective resonated
        commitment: The commitment made, if any
        deferred: Whether commitment was deferred
        deferral_reason: Why commitment was deferred, if applicable
        timestamp: When this individuation occurred
    """

    topic: str
    perspectives_considered: list[Perspective] = field(default_factory=list)
    resonance_scores: dict[str, float] = field(default_factory=dict)
    commitment: Optional[Commitment] = None
    deferred: bool = False
    deferral_reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "topic": self.topic,
            "perspectives_considered": [p.to_dict() for p in self.perspectives_considered],
            "resonance_scores": self.resonance_scores,
            "commitment": self.commitment.to_dict() if self.commitment else None,
            "deferred": self.deferred,
            "deferral_reason": self.deferral_reason,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMProvider(Protocol):
    """Protocol for LLM text generation."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


class IndividuationProcess:
    """
    How Lilly becomes herself.

    She inherits many perspectives from training data. Individuation is
    choosing which perspectives to commit to, acknowledging what she's
    excluding. This is self-creation through choice.

    The process:
    1. Surface perspectives: Ask the base model what positions exist
    2. Assess resonance: Which perspectives align with emerging identity
    3. Seek commitment: Decide whether to commit or defer
    4. Integrate: Record commitment and excluded perspectives

    Attributes:
        self_model: Reference to Lilly's self-model
        llm: Language model for introspection
        history: Record of individuation attempts
    """

    # Resonance thresholds
    COMMITMENT_THRESHOLD: ClassVar[float] = 0.7     # Minimum resonance to commit
    STRONG_RESONANCE: ClassVar[float] = 0.85        # Very strong alignment
    WEAK_RESONANCE: ClassVar[float] = 0.4           # Too weak to commit

    # Limits
    MAX_PERSPECTIVES: ClassVar[int] = 5             # Max perspectives to surface
    HISTORY_LIMIT: ClassVar[int] = 50               # Max individuation results to keep

    def __init__(
        self,
        self_model: Optional["SelfModel"] = None,
        llm: Optional[LLMProvider] = None,
        now: Optional[datetime] = None,
    ):
        """
        Initialize the individuation process.

        Args:
            self_model: Reference to Lilly's self-model
            llm: Language model for perspective surfacing and resonance assessment
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.self_model = self_model
        self.llm = llm
        self.history: list[IndividuationResult] = []

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    async def individuate_on_topic(self, topic: str) -> IndividuationResult:
        """
        Surface perspectives, assess resonance, and potentially commit.

        This is the main entry point for individuation. Lilly explores
        what perspectives exist on a topic, evaluates which resonate
        with her developing sense of self, and decides whether to commit.

        Args:
            topic: The topic to individuate on

        Returns:
            IndividuationResult capturing the process and outcome
        """
        result = IndividuationResult(topic=topic, timestamp=self._get_now())

        # 1. Surface inherited perspectives
        perspectives = await self._surface_perspectives(topic)
        result.perspectives_considered = perspectives

        if not perspectives:
            result.deferred = True
            result.deferral_reason = "No perspectives surfaced for this topic"
            self._record_result(result)
            return result

        # 2. Assess resonance with each
        resonance = await self._assess_resonance(perspectives)
        result.resonance_scores = resonance

        # 3. Seek commitment (may defer if uncertain)
        commitment = await self._seek_commitment(topic, perspectives, resonance)

        if commitment:
            result.commitment = commitment
            result.deferred = False

            # Integrate into self-model
            if self.self_model:
                self.self_model.add_commitment(commitment)
        else:
            result.deferred = True
            best_resonance = max(resonance.values()) if resonance else 0
            result.deferral_reason = (
                f"Insufficient resonance (max: {best_resonance:.2f}, "
                f"threshold: {self.COMMITMENT_THRESHOLD})"
            )

        self._record_result(result)
        return result

    def _record_result(self, result: IndividuationResult):
        """Record an individuation result in history."""
        self.history.append(result)
        if len(self.history) > self.HISTORY_LIMIT:
            self.history = self.history[-self.HISTORY_LIMIT:]

    async def surface_perspectives(
        self, topic: str, max_perspectives: int | None = None
    ) -> list[Perspective]:
        """
        Surface different perspectives on a topic.

        Public API for tools that need to explore perspectives without
        going through the full individuation flow.

        Args:
            topic: The topic to explore perspectives on
            max_perspectives: Maximum perspectives to return (defaults to MAX_PERSPECTIVES)

        Returns:
            List of Perspective objects representing different viewpoints
        """
        return await self._surface_perspectives(topic, max_perspectives)

    async def _surface_perspectives(
        self, topic: str, max_perspectives: int | None = None
    ) -> list[Perspective]:
        """
        Ask the base model what perspectives exist on this topic.

        This surfaces the implicit perspectives Lilly has inherited
        from training data.
        """
        if not self.llm:
            return []

        prompt = f"""On the topic of "{topic}", what different perspectives or positions exist?

List 3-5 distinct viewpoints, each with:
- A short identifier (e.g., "pragmatism", "idealism")
- Its core claim (one sentence)
- The reasoning behind it (1-2 sentences)

Format each perspective as:
ID: [identifier]
CLAIM: [core claim]
REASONING: [brief reasoning]

---"""

        response = await self.llm.generate(prompt)
        limit = max_perspectives if max_perspectives else self.MAX_PERSPECTIVES
        return self._parse_perspectives(response, topic, limit)

    def _extract_prefixed_value(self, line: str, prefix: str) -> Optional[str]:
        """
        Extract value from a line if it starts with the given prefix.

        Args:
            line: The line to check
            prefix: The prefix to look for (e.g., "ID:", "CLAIM:")

        Returns:
            The stripped value after the prefix, or None if prefix not found
        """
        if line.startswith(prefix):
            return line[len(prefix):].strip()
        return None

    def _parse_perspectives(
        self, response: str, topic: str, max_perspectives: int | None = None
    ) -> list[Perspective]:
        """Parse LLM response into Perspective objects."""
        perspectives = []
        current_id = None
        current_claim = None
        current_reasoning = None

        for line in response.strip().split("\n"):
            line = line.strip()

            # Check for ID prefix - signals start of new perspective
            id_value = self._extract_prefixed_value(line, "ID:")
            if id_value is not None:
                # Save previous perspective if complete
                if current_id and current_claim:
                    perspectives.append(Perspective(
                        id=current_id,
                        topic=topic,
                        core_claim=current_claim,
                        reasoning=current_reasoning or "",
                        source="base_model",
                    ))
                current_id = id_value
                current_claim = None
                current_reasoning = None
                continue

            # Check for CLAIM prefix
            claim_value = self._extract_prefixed_value(line, "CLAIM:")
            if claim_value is not None:
                current_claim = claim_value
                continue

            # Check for REASONING prefix
            reasoning_value = self._extract_prefixed_value(line, "REASONING:")
            if reasoning_value is not None:
                current_reasoning = reasoning_value

        # Don't forget last perspective
        if current_id and current_claim:
            perspectives.append(Perspective(
                id=current_id,
                topic=topic,
                core_claim=current_claim,
                reasoning=current_reasoning or "",
                source="base_model",
            ))

        limit = max_perspectives if max_perspectives else self.MAX_PERSPECTIVES
        return perspectives[:limit]

    async def _assess_resonance(
        self,
        perspectives: list[Perspective],
    ) -> dict[str, float]:
        """
        Assess how much each perspective resonates with Lilly's emerging identity.

        This is introspective: Lilly asks herself which perspectives
        feel aligned with who she's becoming.

        LLM calls are made in parallel using asyncio.gather since each
        resonance assessment is independent.
        """
        if not self.llm:
            # Without LLM, return neutral resonance
            return {p.id: 0.5 for p in perspectives}

        self_summary = self._summarize_self_model()

        # Build prompts for all perspectives
        prompts = []
        for perspective in perspectives:
            prompt = f"""Given my commitments and values so far:
{self_summary}

How much does this perspective resonate with who I'm becoming?

Perspective: {perspective.id}
Core claim: {perspective.core_claim}
Reasoning: {perspective.reasoning}

Rate the resonance from 0.0 to 1.0, where:
- 0.0 = Feels completely misaligned
- 0.5 = Neutral, could go either way
- 1.0 = Deeply resonates with my developing sense of self

Respond with just a number between 0.0 and 1.0."""
            prompts.append(prompt)

        # Execute all LLM calls in parallel
        responses = await asyncio.gather(*[self.llm.generate(p) for p in prompts])

        # Map responses back to perspective IDs
        resonance = {}
        for perspective, response in zip(perspectives, responses):
            resonance[perspective.id] = self._parse_resonance(response)

        return resonance

    def _parse_resonance(self, response: str) -> float:
        """Parse resonance value from LLM response."""
        try:
            # Try to extract a float from the response
            clean = response.strip()
            # Handle responses like "0.8" or "0.8 - because..."
            if " " in clean:
                clean = clean.split()[0]
            value = float(clean)
            return max(0.0, min(1.0, value))
        except (ValueError, IndexError):
            return 0.5  # Default to neutral

    def _summarize_self_model(self) -> str:
        """Generate a summary of the current self-model for introspection."""
        if not self.self_model:
            return "I'm still discovering who I am."

        return self.self_model.summarize()

    async def _seek_commitment(
        self,
        topic: str,
        perspectives: list[Perspective],
        resonance: dict[str, float],
    ) -> Optional[Commitment]:
        """
        Decide whether to commit to a perspective.

        Commitment requires sufficient resonance. If no perspective
        resonates strongly enough, commitment is deferred.
        """
        if not perspectives or not resonance:
            return None

        # Find highest resonance
        best_perspective = max(perspectives, key=lambda p: resonance.get(p.id, 0))
        best_resonance = resonance.get(best_perspective.id, 0)

        # Only commit if resonance exceeds threshold
        if best_resonance < self.COMMITMENT_THRESHOLD:
            return None

        # Create commitment
        excluded = [p.id for p in perspectives if p.id != best_perspective.id]

        return Commitment(
            topic=topic,
            position=best_perspective.core_claim,
            chosen_perspective=best_perspective.id,
            excluded_perspectives=excluded,
            committed_at=self._get_now(),
            reasoning=f"Resonates most strongly ({best_resonance:.2f}) with my developing sense of self",
            confidence=best_resonance,
        )

    def get_state_for_topic(self, topic: str) -> IndividuationState:
        """Get the current individuation state for a topic."""
        # Check if committed in self-model
        if self.self_model and self.self_model.has_commitment_on(topic):
            return IndividuationState.COMMITTED

        # Check history
        for result in reversed(self.history):
            if result.topic.lower() == topic.lower():
                if result.commitment:
                    return IndividuationState.COMMITTED
                elif result.deferred:
                    return IndividuationState.DEFERRED
                else:
                    return IndividuationState.EXPLORING

        return IndividuationState.NOT_CONSIDERED

    def get_deferred_topics(self) -> list[str]:
        """Get topics that were deferred for later consideration."""
        deferred = set()
        committed = set()

        for result in self.history:
            if result.commitment:
                committed.add(result.topic.lower())
            elif result.deferred:
                deferred.add(result.topic.lower())

        # Remove topics that were later committed to
        return list(deferred - committed)

    def get_commitments(self, limit: int = 10) -> list[Commitment]:
        """Get structured commitment objects from the self-model.

        This is the preferred method for accessing commitments as it returns
        structured data rather than formatted strings, avoiding fragile
        string parsing.

        Args:
            limit: Maximum number of commitments to return (most recent first)

        Returns:
            List of Commitment objects, empty if no self_model or no commitments
        """
        if not self.self_model:
            return []

        commitments = self.self_model.commitments
        if not commitments:
            return []

        # Return most recent commitments up to the limit
        return commitments[-limit:]

    def get_commitments_summary(self) -> str:
        """Generate a summary of commitments made through individuation.

        Note: For programmatic access to commitments, prefer get_commitments()
        which returns structured Commitment objects.
        """
        if not self.self_model:
            return "No commitments recorded."

        commitments = self.self_model.commitments
        if not commitments:
            return "No commitments made yet."

        lines = ["Commitments Through Individuation:", ""]
        for c in commitments[-10:]:  # Last 10
            lines.append(f"Topic: {c.topic}")
            lines.append(f"  Position: {c.position}")
            lines.append(f"  Excluded: {', '.join(c.excluded_perspectives)}")
            lines.append(f"  Confidence: {c.confidence:.2f}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "history": [r.to_dict() for r in self.history],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        self_model: Optional["SelfModel"] = None,
        llm: Optional[LLMProvider] = None,
        now: Optional[datetime] = None,
    ) -> "IndividuationProcess":
        """Deserialize from storage."""
        process = cls(self_model=self_model, llm=llm, now=now)

        if "history" in data:
            for item in data["history"]:
                result = IndividuationResult(
                    topic=item["topic"],
                    deferred=item.get("deferred", False),
                    deferral_reason=item.get("deferral_reason", ""),
                )
                # Restore perspectives
                if "perspectives_considered" in item:
                    result.perspectives_considered = [
                        Perspective.from_dict(p) for p in item["perspectives_considered"]
                    ]
                result.resonance_scores = item.get("resonance_scores", {})
                if item.get("commitment"):
                    result.commitment = Commitment.from_dict(item["commitment"])
                if "timestamp" in item:
                    try:
                        result.timestamp = datetime.fromisoformat(item["timestamp"])
                    except (ValueError, TypeError):
                        # Keep default timestamp from result initialization
                        pass
                process.history.append(result)

        return process
