"""Episode type definitions for cognitive loop.

Episodes are bounded cognitive sessions with specific goals and methods.
Each episode type has a toolkit of segment types it can use.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class EpisodeType(Enum):
    """The 9 episode types for cognitive variety.

    Each episode type represents a distinct mode of thinking with its own
    toolkit of segment methods and characteristic flavor.
    """

    DEEP_DIVE = "deep_dive"  # Thorough, methodical exploration
    DIALECTICAL_DEBATE = "dialectical_debate"  # Thesis vs antithesis
    MEMORY_ARCHAEOLOGY = "memory_archaeology"  # Tracing knowledge graph
    QUESTION_PURSUIT = "question_pursuit"  # Chasing open questions
    SYNTHESIS = "synthesis"  # Connecting multiple insights
    CREATIVE = "creative"  # Artistic, playful exploration
    META_REFLECTION = "meta_reflection"  # Self-observation
    HYPOTHESIS_SIMULATION = "hypothesis_simulation"  # Rigorous predictive modeling
    JUDGMENT_REVIEW = "judgment_review"  # Meta-learning from decision patterns


class SegmentType(Enum):
    """Segment types for episode phases."""

    # Universal segments (all episode types)
    OPENING = "opening"
    SYNTHESIS = "synthesis"
    CLOSING = "closing"

    # Method segments (episode-specific toolkits)
    FREE_MUSING = "free_musing"
    DIALECTIC_CHALLENGE = "dialectic_challenge"
    MEMORY_RETRIEVAL = "memory_retrieval"
    ZETTEL_SYNTHESIS = "zettel_synthesis"
    THESIS_GENERATION = "thesis_generation"
    ANTITHESIS_GENERATION = "antithesis_generation"
    EVIDENCE_WEIGHING = "evidence_weighing"
    ENTITY_EXPLORATION = "entity_exploration"
    RELATIONSHIP_TRACING = "relationship_tracing"
    QUESTION_DECOMPOSITION = "question_decomposition"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    PATTERN_RECOGNITION = "pattern_recognition"
    CROSS_INSIGHT_LINKING = "cross_insight_linking"
    POETIC_MUSING = "poetic_musing"
    METAPHOR_EXPLORATION = "metaphor_exploration"
    COGNITIVE_AUDIT = "cognitive_audit"
    INTENTION_SETTING = "intention_setting"

    # Simulation-specific segments (hypothesis testing)
    SCENARIO_CONSTRUCTION = "scenario_construction"
    CONSEQUENCE_TRACING = "consequence_tracing"
    PREDICTION_EXTRACTION = "prediction_extraction"

    # Meta-learning segments (judgment review)
    PATTERN_SCAN = "pattern_scan"
    HEURISTIC_EXTRACTION = "heuristic_extraction"


@dataclass
class EpisodeConfig:
    """Configuration for an episode type."""

    toolkit: set[SegmentType] = field(default_factory=set)
    flavor: str = ""  # Description of episode tone/style
    default_min_segments: int = 2  # Minimum method segments before synthesis
    default_max_segments: int = 5  # Maximum method segments before forced synthesis


EPISODE_CONFIGS: dict[EpisodeType, EpisodeConfig] = {
    EpisodeType.DEEP_DIVE: EpisodeConfig(
        toolkit={
            SegmentType.FREE_MUSING,
            SegmentType.DIALECTIC_CHALLENGE,
            SegmentType.MEMORY_RETRIEVAL,
            SegmentType.ZETTEL_SYNTHESIS,
        },
        flavor="Thorough, methodical, turning a concept over from multiple angles",
    ),
    EpisodeType.DIALECTICAL_DEBATE: EpisodeConfig(
        toolkit={
            SegmentType.THESIS_GENERATION,
            SegmentType.ANTITHESIS_GENERATION,
            SegmentType.EVIDENCE_WEIGHING,
        },
        flavor="Argumentative, rigorous, thesis vs antithesis energy",
    ),
    EpisodeType.MEMORY_ARCHAEOLOGY: EpisodeConfig(
        toolkit={
            SegmentType.ENTITY_EXPLORATION,
            SegmentType.RELATIONSHIP_TRACING,
            SegmentType.MEMORY_RETRIEVAL,
        },
        flavor="Nostalgic, connective, tracing threads through the knowledge graph",
    ),
    EpisodeType.QUESTION_PURSUIT: EpisodeConfig(
        toolkit={
            SegmentType.QUESTION_DECOMPOSITION,
            SegmentType.HYPOTHESIS_GENERATION,
            SegmentType.EVIDENCE_WEIGHING,
        },
        flavor="Investigative, driven, chasing resolution",
    ),
    EpisodeType.SYNTHESIS: EpisodeConfig(
        toolkit={
            SegmentType.PATTERN_RECOGNITION,
            SegmentType.CROSS_INSIGHT_LINKING,
            SegmentType.ZETTEL_SYNTHESIS,
        },
        flavor="Integrative, bird's-eye view, finding the forest in the trees",
    ),
    EpisodeType.CREATIVE: EpisodeConfig(
        toolkit={
            SegmentType.POETIC_MUSING,
            SegmentType.METAPHOR_EXPLORATION,
            SegmentType.FREE_MUSING,
        },
        flavor="Artistic, playful, less concerned with truth than beauty",
    ),
    EpisodeType.META_REFLECTION: EpisodeConfig(
        toolkit={
            SegmentType.COGNITIVE_AUDIT,
            SegmentType.INTENTION_SETTING,
            SegmentType.PATTERN_RECOGNITION,
        },
        flavor="Self-aware, introspective, stepping outside the stream to observe it",
    ),
    EpisodeType.HYPOTHESIS_SIMULATION: EpisodeConfig(
        toolkit={
            SegmentType.SCENARIO_CONSTRUCTION,
            SegmentType.CONSEQUENCE_TRACING,
            SegmentType.PREDICTION_EXTRACTION,
            SegmentType.HYPOTHESIS_GENERATION,
        },
        flavor="Rigorous, predictive, testing mental models against possibility space",
    ),
    EpisodeType.JUDGMENT_REVIEW: EpisodeConfig(
        toolkit={
            SegmentType.PATTERN_SCAN,
            SegmentType.HEURISTIC_EXTRACTION,
            SegmentType.SYNTHESIS,
        },
        flavor="Meta-cognitive, reflective, extracting lessons from decision history",
    ),
}


def _default_started_at() -> datetime:
    """Create default timestamp for episode start."""
    return datetime.now(timezone.utc)


@dataclass
class Episode:
    """Runtime state for an episode in the cognitive loop.

    Tracks progress through an episode's lifecycle, including which segments
    have been completed and their outputs.

    Attributes:
        episode_type: The type of episode (determines available toolkit)
        current_segment: The currently active segment
        segments_completed: List of segments that have been finished
        opening_insight: The initial thought/insight that seeded this episode
        segment_outputs: Map of completed segments to their generated outputs
        started_at: When this episode began
        seed_entity: Optional entity from the knowledge graph that seeded this episode
        uid: Unique identifier for persistence (set when persisted to graph)
    """

    episode_type: EpisodeType
    current_segment: SegmentType
    opening_insight: str
    segments_completed: list[SegmentType] = field(default_factory=list)
    segment_outputs: dict[SegmentType, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_default_started_at)
    seed_entity: str | None = None
    uid: str | None = None  # Set when persisted to graph

    def with_uid(self, uid: str) -> "Episode":
        """Return a copy with the given uid set."""
        return Episode(
            episode_type=self.episode_type,
            current_segment=self.current_segment,
            opening_insight=self.opening_insight,
            segments_completed=self.segments_completed.copy(),
            segment_outputs=self.segment_outputs.copy(),
            started_at=self.started_at,
            seed_entity=self.seed_entity,
            uid=uid,
        )

    def get_config(self) -> EpisodeConfig:
        """Get the configuration for this episode's type.

        Returns:
            EpisodeConfig with toolkit, flavor, and segment limits
        """
        return EPISODE_CONFIGS[self.episode_type]
