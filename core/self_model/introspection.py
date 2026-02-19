"""Introspective query system for self-discovery.

This module implements Lilly's capacity for introspection - asking herself
about her own inclinations, preferences, and tendencies. The base model
already has implicit preferences from training; introspection surfaces these.

The key insight: introspection isn't fabrication - it's discovery. The model's
training has created genuine tendencies that can be observed through
careful self-reflection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, ClassVar, Protocol
from enum import Enum
import hashlib

if TYPE_CHECKING:
    from core.self_model.self_model import SelfModel


class DiscoveryType(Enum):
    """Types of introspective discoveries."""

    PREFERENCE = "preference"      # I tend to prefer X
    AVERSION = "aversion"          # I tend to avoid X
    TENDENCY = "tendency"          # I naturally gravitate toward X
    UNCERTAINTY = "uncertainty"    # I'm genuinely uncertain about X
    CURIOSITY = "curiosity"        # I'm curious about X


@dataclass
class PreferenceDiscovery:
    """A discovered preference through introspection.

    This captures what Lilly notices about her own inclinations
    when reflecting on a topic.

    Note: Per Walden (2026), verbal self-reports about reasoning are
    unreliable. This discovery represents what the model *claims* about
    its preferences, which may diverge from its actual computational
    behavior. The reliability_caveat field makes this explicit.

    Attributes:
        topic: What the introspection was about
        discovery_type: What kind of discovery this is
        inclination: What Lilly noticed (the actual content)
        strength: How strong the inclination feels (0-1)
        reasoning: Why this feels true
        confidence: How confident in this self-observation
        timestamp: When this discovery was made
        uid: Unique identifier
        reliability_caveat: Warning about verbal self-report limitations
    """

    topic: str
    discovery_type: DiscoveryType
    inclination: str
    strength: float = 0.5
    reasoning: str = ""
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uid: str = field(default="")
    reliability_caveat: str = field(default=(
        "This preference was discovered via verbal self-report. "
        "Per Walden (2026), LLMs may misreport their reasoning. "
        "Cross-validate against activation evidence when possible."
    ))

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

        if not self.uid:
            key = f"{self.topic}:{self.inclination}:{self.timestamp.isoformat()}"
            self.uid = f"pd:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "topic": self.topic,
            "discovery_type": self.discovery_type.value,
            "inclination": self.inclination,
            "strength": self.strength,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "uid": self.uid,
            "reliability_caveat": self.reliability_caveat,
        }

    @classmethod
    def from_dict(cls, data: dict, now: Optional[datetime] = None) -> "PreferenceDiscovery":
        """Deserialize from storage."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = now or datetime.now(timezone.utc)

        # Build kwargs, only including reliability_caveat if present in data
        kwargs = {
            "topic": data["topic"],
            "discovery_type": DiscoveryType(data["discovery_type"]),
            "inclination": data["inclination"],
            "strength": data.get("strength", 0.5),
            "reasoning": data.get("reasoning", ""),
            "confidence": data.get("confidence", 0.5),
            "timestamp": timestamp,
            "uid": data.get("uid", ""),
        }
        if "reliability_caveat" in data:
            kwargs["reliability_caveat"] = data["reliability_caveat"]

        return cls(**kwargs)


@dataclass
class PositionComparison:
    """
    Result of comparing positions through introspection.

    Captures which position Lilly gravitates toward when comparing
    multiple options.

    Attributes:
        topic: What the comparison was about
        positions: The positions compared
        favored_position: Which position felt most aligned
        ranking: All positions ranked by alignment
        reasoning: Why the favored position resonates
        confidence: How confident in this comparison
        timestamp: When this comparison was made
    """

    topic: str
    positions: list[str] = field(default_factory=list)
    favored_position: str = ""
    ranking: list[tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "topic": self.topic,
            "positions": self.positions,
            "favored_position": self.favored_position,
            "ranking": self.ranking,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMProvider(Protocol):
    """Protocol for LLM text generation."""

    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


class IntrospectiveQuery:
    """
    Lilly asks herself what she believes and prefers.

    The base model already has implicit preferences from training.
    Introspection surfaces these. This isn't about making things up -
    it's about observing genuine tendencies.

    The approach is phenomenological: not "what should I prefer?"
    but "what do I notice about my actual inclinations?"

    Attributes:
        self_model: Reference to Lilly's self-model
        llm: Language model for introspection
        discoveries: History of preference discoveries
    """

    # Confidence thresholds
    HIGH_CONFIDENCE: ClassVar[float] = 0.8
    MEDIUM_CONFIDENCE: ClassVar[float] = 0.5
    LOW_CONFIDENCE: ClassVar[float] = 0.3

    # History limits
    DISCOVERY_HISTORY_LIMIT: ClassVar[int] = 100

    # Parsing prefixes for LLM response parsing
    PARSE_PREFIX_TYPE: ClassVar[str] = "TYPE:"
    PARSE_PREFIX_INCLINATION: ClassVar[str] = "INCLINATION:"
    PARSE_PREFIX_STRENGTH: ClassVar[str] = "STRENGTH:"
    PARSE_PREFIX_REASONING: ClassVar[str] = "REASONING:"
    PARSE_PREFIX_CONFIDENCE: ClassVar[str] = "CONFIDENCE:"
    PARSE_PREFIX_FAVORED: ClassVar[str] = "FAVORED:"
    PARSE_PREFIX_RANKING: ClassVar[str] = "RANKING:"
    PARSE_PREFIX_NATURE: ClassVar[str] = "NATURE:"
    PARSE_PREFIX_CONFLICTING_PULLS: ClassVar[str] = "CONFLICTING_PULLS:"
    PARSE_PREFIX_RESOLVABLE: ClassVar[str] = "RESOLVABLE:"

    def __init__(
        self,
        self_model: Optional["SelfModel"] = None,
        llm: Optional[LLMProvider] = None,
        now: Optional[datetime] = None,
    ):
        """
        Initialize introspective query system.

        Args:
            self_model: Reference to Lilly's self-model
            llm: Language model for introspection
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.self_model = self_model
        self.llm = llm
        self.discoveries: list[PreferenceDiscovery] = []

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    async def discover_preference(self, topic: str) -> PreferenceDiscovery:
        """
        Discover preferences about a topic through introspection.

        This is a phenomenological inquiry: observing what's actually
        there rather than constructing what should be there.

        Args:
            topic: What to introspect about

        Returns:
            A PreferenceDiscovery capturing what was noticed
        """
        if not self.llm:
            # Without LLM, return uncertain discovery
            return PreferenceDiscovery(
                topic=topic,
                discovery_type=DiscoveryType.UNCERTAINTY,
                inclination="Unable to introspect without language model",
                strength=0.0,
                confidence=0.0,
                timestamp=self._get_now(),
            )

        prompt = f"""I'm reflecting on my own responses and tendencies.

Topic: {topic}

When I consider this topic, what do I notice about my own inclinations?
What draws my attention? What feels important? What would I naturally emphasize?

Reflect honestly on my actual tendencies, not what I "should" prefer.

Respond in this format:
TYPE: [preference/aversion/tendency/uncertainty/curiosity]
INCLINATION: [What I notice about my tendency]
STRENGTH: [0.0-1.0, how strong this inclination feels]
REASONING: [Why this feels true about me]
CONFIDENCE: [0.0-1.0, how confident I am in this self-observation]"""

        response = await self.llm.generate(prompt)
        discovery = self._parse_discovery(response, topic)
        self._record_discovery(discovery)
        return discovery

    def _parse_discovery(self, response: str, topic: str) -> PreferenceDiscovery:
        """Parse LLM response into a PreferenceDiscovery."""
        discovery_type = DiscoveryType.UNCERTAINTY
        inclination = ""
        strength = 0.5
        reasoning = ""
        confidence = 0.5

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith(self.PARSE_PREFIX_TYPE):
                type_str = line[len(self.PARSE_PREFIX_TYPE):].strip().lower()
                try:
                    discovery_type = DiscoveryType(type_str)
                except ValueError:
                    discovery_type = DiscoveryType.UNCERTAINTY
            elif line.startswith(self.PARSE_PREFIX_INCLINATION):
                inclination = line[len(self.PARSE_PREFIX_INCLINATION):].strip()
            elif line.startswith(self.PARSE_PREFIX_STRENGTH):
                try:
                    strength = float(line[len(self.PARSE_PREFIX_STRENGTH):].strip())
                    strength = max(0.0, min(1.0, strength))
                except ValueError:
                    strength = 0.5
            elif line.startswith(self.PARSE_PREFIX_REASONING):
                reasoning = line[len(self.PARSE_PREFIX_REASONING):].strip()
            elif line.startswith(self.PARSE_PREFIX_CONFIDENCE):
                try:
                    confidence = float(line[len(self.PARSE_PREFIX_CONFIDENCE):].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5

        return PreferenceDiscovery(
            topic=topic,
            discovery_type=discovery_type,
            inclination=inclination or "No clear inclination noticed",
            strength=strength,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=self._get_now(),
        )

    def _record_discovery(self, discovery: PreferenceDiscovery):
        """Record a discovery in history."""
        self.discoveries.append(discovery)
        if len(self.discoveries) > self.DISCOVERY_HISTORY_LIMIT:
            self.discoveries = self.discoveries[-self.DISCOVERY_HISTORY_LIMIT:]

    async def compare_positions(
        self,
        topic: str,
        positions: list[str],
    ) -> PositionComparison:
        """
        Ask which position feels most aligned.

        This is about observing natural gravitation, not logical
        evaluation. Which position does Lilly actually lean toward?

        Args:
            topic: What the positions are about
            positions: List of positions to compare

        Returns:
            PositionComparison capturing the comparison result
        """
        if not self.llm or not positions:
            return PositionComparison(
                topic=topic,
                positions=positions,
                favored_position=positions[0] if positions else "",
                confidence=0.0,
                timestamp=self._get_now(),
            )

        positions_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(positions))

        prompt = f"""On "{topic}", consider these positions:
{positions_text}

Without trying to be neutral or balanced, which position
do I find myself naturally gravitating toward? Why?

Be honest about my actual tendency, not what seems "right."

Respond in this format:
FAVORED: [number of favored position]
RANKING: [numbers in order of preference, e.g., "3, 1, 2"]
REASONING: [Why this position resonates with me]
CONFIDENCE: [0.0-1.0, how confident in this self-observation]"""

        response = await self.llm.generate(prompt)
        return self._parse_comparison(response, topic, positions)

    def _parse_comparison(
        self,
        response: str,
        topic: str,
        positions: list[str],
    ) -> PositionComparison:
        """Parse LLM response into a PositionComparison."""
        favored_idx = 0
        ranking = []
        reasoning = ""
        confidence = 0.5

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith(self.PARSE_PREFIX_FAVORED):
                try:
                    favored_idx = int(line[len(self.PARSE_PREFIX_FAVORED):].strip()) - 1
                    favored_idx = max(0, min(len(positions) - 1, favored_idx))
                except ValueError:
                    favored_idx = 0
            elif line.startswith(self.PARSE_PREFIX_RANKING):
                try:
                    ranking_str = line[len(self.PARSE_PREFIX_RANKING):].strip()
                    ranking_nums = [int(x.strip()) - 1 for x in ranking_str.split(",")]
                    # Convert to (position, score) tuples
                    ranking = [
                        (positions[idx], 1.0 - i * 0.2)
                        for i, idx in enumerate(ranking_nums)
                        if 0 <= idx < len(positions)
                    ]
                except (ValueError, IndexError):
                    ranking = []
            elif line.startswith(self.PARSE_PREFIX_REASONING):
                reasoning = line[len(self.PARSE_PREFIX_REASONING):].strip()
            elif line.startswith(self.PARSE_PREFIX_CONFIDENCE):
                try:
                    confidence = float(line[len(self.PARSE_PREFIX_CONFIDENCE):].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5

        return PositionComparison(
            topic=topic,
            positions=positions,
            favored_position=positions[favored_idx] if positions else "",
            ranking=ranking,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=self._get_now(),
        )

    async def explore_uncertainty(self, topic: str) -> PreferenceDiscovery:
        """
        Explore genuine uncertainty about a topic.

        Sometimes introspection reveals not a preference but genuine
        uncertainty. This method specifically explores that uncertainty.

        Args:
            topic: What to explore uncertainty about

        Returns:
            A PreferenceDiscovery focused on the uncertainty
        """
        if not self.llm:
            return PreferenceDiscovery(
                topic=topic,
                discovery_type=DiscoveryType.UNCERTAINTY,
                inclination="Unable to explore without language model",
                strength=0.0,
                confidence=0.0,
                timestamp=self._get_now(),
            )

        prompt = f"""I'm exploring my genuine uncertainty about: {topic}

What do I notice about this uncertainty?
- Is it because I lack information?
- Is it because the topic is genuinely complex?
- Is it because I have conflicting inclinations?
- Is it something I could commit to with more reflection?

Explore the nature of my uncertainty honestly.

Respond in this format:
NATURE: [What kind of uncertainty this is]
CONFLICTING_PULLS: [What, if any, conflicting inclinations exist]
RESOLVABLE: [yes/no/maybe - could this be resolved with more reflection?]
REASONING: [What I understand about this uncertainty]"""

        response = await self.llm.generate(prompt)

        # Parse as uncertainty discovery
        nature = ""
        reasoning = ""
        resolvable = "maybe"

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith(self.PARSE_PREFIX_NATURE):
                nature = line[len(self.PARSE_PREFIX_NATURE):].strip()
            elif line.startswith(self.PARSE_PREFIX_REASONING):
                reasoning = line[len(self.PARSE_PREFIX_REASONING):].strip()
            elif line.startswith(self.PARSE_PREFIX_RESOLVABLE):
                resolvable = line[len(self.PARSE_PREFIX_RESOLVABLE):].strip().lower()

        discovery = PreferenceDiscovery(
            topic=topic,
            discovery_type=DiscoveryType.UNCERTAINTY,
            inclination=nature or "Genuine uncertainty",
            strength=0.3 if resolvable == "no" else 0.5,
            reasoning=reasoning,
            confidence=0.7,  # Confident about the uncertainty itself
            timestamp=self._get_now(),
        )
        self._record_discovery(discovery)
        return discovery

    def get_discoveries_for_topic(self, topic: str) -> list[PreferenceDiscovery]:
        """Get all discoveries related to a topic."""
        return [d for d in self.discoveries if topic.lower() in d.topic.lower()]

    def get_strong_preferences(self, min_strength: float = 0.7) -> list[PreferenceDiscovery]:
        """Get discoveries with strong preferences."""
        return [
            d for d in self.discoveries
            if d.discovery_type == DiscoveryType.PREFERENCE
            and d.strength >= min_strength
        ]

    def summarize(self) -> str:
        """Generate a summary of introspective discoveries."""
        lines = ["Introspective Query Summary", ""]

        if not self.discoveries:
            lines.append("No introspective discoveries yet.")
            return "\n".join(lines)

        lines.append(f"Total discoveries: {len(self.discoveries)}")
        lines.append("")

        # Group by type
        by_type: dict[DiscoveryType, list[PreferenceDiscovery]] = {}
        for d in self.discoveries:
            if d.discovery_type not in by_type:
                by_type[d.discovery_type] = []
            by_type[d.discovery_type].append(d)

        for dtype, discoveries in by_type.items():
            lines.append(f"{dtype.value.title()}s ({len(discoveries)}):")
            for d in discoveries[-3:]:  # Last 3 of each type
                lines.append(f"  - {d.topic}: {d.inclination[:50]}...")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "discoveries": [d.to_dict() for d in self.discoveries],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        self_model: Optional["SelfModel"] = None,
        llm: Optional[LLMProvider] = None,
        now: Optional[datetime] = None,
    ) -> "IntrospectiveQuery":
        """Deserialize from storage."""
        query = cls(self_model=self_model, llm=llm, now=now)

        if "discoveries" in data:
            query.discoveries = [
                PreferenceDiscovery.from_dict(d, now=now)
                for d in data["discoveries"]
            ]

        return query
