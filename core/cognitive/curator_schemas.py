"""Structured output schemas for the curator phase.

These dataclasses define the structured outputs that the curator produces,
enabling consistent graph updates and prompt crafting between generations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from core.cognitive.episode import SegmentType

if TYPE_CHECKING:
    from core.cognitive.experimentation.schemas import ExperimentProposal


@dataclass
class ThoughtAnalysis:
    """Core extraction from the generated thought.

    Attributes:
        insight: Key realization or understanding from the thought
        question: Driving question that emerges for further exploration
        concepts: Entities and concepts mentioned in the thought
        confidence: How certain the thought feels, 0 to 1
        joy: Plutchik joy intensity (0-1)
        trust: Plutchik trust intensity (0-1)
        fear: Plutchik fear intensity (0-1)
        surprise: Plutchik surprise intensity (0-1)
        sadness: Plutchik sadness intensity (0-1)
        disgust: Plutchik disgust intensity (0-1)
        anger: Plutchik anger intensity (0-1)
        anticipation: Plutchik anticipation intensity (0-1)
    """

    insight: str
    question: str
    concepts: list[str] = field(default_factory=list)
    confidence: float = 0.5
    # Plutchik 8D emotional field
    joy: float = 0.5
    trust: float = 0.5
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.5

    def __post_init__(self):
        """Validate ranges - clamp all emotions to [0, 1]."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.joy = max(0.0, min(1.0, self.joy))
        self.trust = max(0.0, min(1.0, self.trust))
        self.fear = max(0.0, min(1.0, self.fear))
        self.surprise = max(0.0, min(1.0, self.surprise))
        self.sadness = max(0.0, min(1.0, self.sadness))
        self.disgust = max(0.0, min(1.0, self.disgust))
        self.anger = max(0.0, min(1.0, self.anger))
        self.anticipation = max(0.0, min(1.0, self.anticipation))

    @property
    def emotional_valence(self) -> float:
        """Compute emotional valence from Plutchik emotions.

        Positive emotions (joy, trust, anticipation, surprise) contribute positively.
        Negative emotions (sadness, fear, disgust, anger) contribute negatively.
        Returns value in [-1, 1].
        """
        positive = (self.joy + self.trust + self.anticipation + self.surprise * 0.5) / 3.0
        negative = (self.sadness + self.fear + self.disgust + self.anger) / 4.0
        return max(-1.0, min(1.0, positive - negative))


@dataclass
class EntityUpdate:
    """Update to an existing entity in the knowledge graph.

    Attributes:
        name: Entity name to update
        salience_delta: Change in salience score
        type_refinement: Optional new or refined entity type
    """

    name: str
    salience_delta: float = 0.0
    type_refinement: Optional[str] = None


@dataclass
class BeliefUpdate:
    """Update to a committed belief.

    Attributes:
        topic: Topic the belief is about
        confidence_delta: Change in confidence level
        evidence: New evidence supporting the update
    """

    topic: str
    confidence_delta: float
    evidence: str


@dataclass
class TripleData:
    """Data for a new triple to add to the graph.

    Attributes:
        subject: Subject entity
        predicate: Relationship type
        object_: Object entity (named object_ to avoid Python keyword)
        confidence: Confidence in this triple
    """

    subject: str
    predicate: str
    object_: str
    confidence: float = 0.8


@dataclass
class ZettelData:
    """Data for creating an InsightZettel.

    Attributes:
        insight: The core insight text
        source_thought: The thought this emerged from
        concepts: Related concepts for linking
        question: Open question if any
    """

    insight: str
    source_thought: str
    concepts: list[str] = field(default_factory=list)
    question: Optional[str] = None


@dataclass
class GraphOperations:
    """What to persist to the knowledge graph.

    Attributes:
        new_triples: Triples to add
        entity_updates: Updates to existing entities
        zettel: Zettel to create if insight is significant
        belief_updates: Updates to committed beliefs
    """

    new_triples: list[TripleData] = field(default_factory=list)
    entity_updates: list[EntityUpdate] = field(default_factory=list)
    zettel: Optional[ZettelData] = None
    belief_updates: list[BeliefUpdate] = field(default_factory=list)


@dataclass
class NextPromptContext:
    """Curated setup for next generation.

    Attributes:
        concept: Primary concept to explore
        framing: How to approach it (dialectical, exploratory, creative, synthesizing)
        retrieved_context: Relevant zettels/memories to include in prompt
        steering_hints: Zone adjustments (zone_name -> delta)
        directive: Specific instruction or constraint for generation
    """

    concept: str
    framing: str = "exploratory"
    retrieved_context: list[str] = field(default_factory=list)
    steering_hints: dict[str, float] = field(default_factory=dict)
    directive: str = ""

    VALID_FRAMINGS = {"dialectical", "exploratory", "creative", "synthesizing"}

    def __post_init__(self):
        """Validate framing."""
        if self.framing not in self.VALID_FRAMINGS:
            self.framing = "exploratory"


@dataclass
class EpisodeGuidance:
    """Meta-level episode decisions.

    Attributes:
        continue_episode: Whether to continue current episode
        suggested_segment: Next segment type if continuing
        goal_status: Progress status (progress, pivot, complete)
    """

    continue_episode: bool = True
    suggested_segment: Optional[SegmentType] = None
    goal_status: str = "progress"

    VALID_STATUSES = {"progress", "pivot", "complete"}

    def __post_init__(self):
        """Validate goal_status."""
        if self.goal_status not in self.VALID_STATUSES:
            self.goal_status = "progress"


@dataclass
class SimulationHint:
    """Hint from curator about whether to run simulation phase.

    The simulation phase (Phase 2.5) uses Graph-Preflexor for rigorous
    hypothesis testing and predictive modeling. This hint allows the
    curator to explicitly trigger simulation when insights are worth
    stress-testing.

    Attributes:
        should_simulate: Whether to trigger simulation phase
        trigger_reason: Why simulation was triggered (for logging/narration)
        focus_concept: Primary concept to simulate
        hypothesis_seed: Optional starting hypothesis to test
    """

    should_simulate: bool = False
    trigger_reason: str = ""
    focus_concept: str = ""
    hypothesis_seed: str = ""


@dataclass
class CurationResult:
    """Complete output from curator phase.

    Attributes:
        analysis: Thought analysis with insight/question extraction
        graph_ops: Operations to perform on knowledge graph
        next_prompt: Context for crafting next generation prompt
        episode: Episode-level guidance
        simulation_hint: Hint about whether to run simulation phase
        reliability: Assessment of data reliability (Wolpert framework)
        faithfulness: Activation-verbal consistency score (Walden framework)
        thinking_trace: Curator's thinking trace (when debug enabled)
        reflection: Verbal reflection on thought quality (Reflexion framework)
        reflection_trigger: What triggered the reflection (e.g., "low_faithfulness")
        experiment_proposal: Optional self-experiment proposal for parameter tuning
        is_complete: Whether curation is complete (False to request another turn)
        continuation_reason: Why more work is needed (when is_complete=False)
    """

    analysis: ThoughtAnalysis
    graph_ops: GraphOperations
    next_prompt: NextPromptContext
    episode: EpisodeGuidance
    simulation_hint: Optional["SimulationHint"] = None
    reliability: Optional["ReliabilityAssessment"] = None
    faithfulness: Optional["FaithfulnessScore"] = None
    thinking_trace: Optional[str] = None
    reflection: Optional[str] = None  # Verbal reflection on thought quality
    reflection_trigger: Optional[str] = None  # What triggered reflection
    experiment_proposal: Optional["ExperimentProposal"] = None  # Self-experiment proposal
    is_complete: bool = True  # Default true for backwards compatibility
    continuation_reason: Optional[str] = None  # Why more work is needed

    @classmethod
    def empty(cls, concept: str = "emergence") -> "CurationResult":
        """Create an empty/default curation result for fallback."""
        return cls(
            analysis=ThoughtAnalysis(insight="", question=""),
            graph_ops=GraphOperations(),
            next_prompt=NextPromptContext(concept=concept),
            episode=EpisodeGuidance(),
            simulation_hint=None,
            reliability=None,
            faithfulness=None,
            thinking_trace=None,
        )


@dataclass
class ActivationSummary:
    """Summary of activations for curator consumption.

    Attributes:
        layer: Which layer the activations are from
        top_positions: Top activating token positions with values
        mean_activation: Mean activation magnitude
        max_activation: Maximum activation value
    """

    layer: int
    top_positions: list[tuple[int, float]] = field(default_factory=list)
    mean_activation: float = 0.0
    max_activation: float = 0.0


@dataclass
class SAEFeature:
    """A single SAE feature with metadata.

    Attributes:
        feature_id: Feature index in the SAE
        activation: Activation strength
        label: Human-readable label (if available)
        interpretation_source: Where the label came from:
            - "neuronpedia": Verified interpretation from Neuronpedia API
            - "logit_lens": Inferred from vocabulary projection (less reliable)
            - None: No interpretation available
    """

    feature_id: int
    activation: float
    label: Optional[str] = None
    interpretation_source: Optional[str] = None


@dataclass
class ReliabilityAssessment:
    """Assessment of data reliability for epistemic transparency.

    Following Wolpert's framework: conclusions depend on which data
    we assume is reliable. This makes that assumption explicit.

    Attributes:
        data_sources: All data sources consulted (memory, graph, etc.)
        assumed_reliable: Subset of sources treated as reliable
        reliability_justification: Why these sources are trusted
        conditioning_implications: What this conditioning choice implies
    """

    data_sources: list[str] = field(default_factory=list)
    assumed_reliable: list[str] = field(default_factory=list)
    reliability_justification: str = ""
    conditioning_implications: str = ""


@dataclass
class FaithfulnessScore:
    """Cross-validation of verbal claims against activation evidence.

    Following Walden (2026) "Reasoning Models Will Blatantly Lie About Their
    Reasoning": models deny using information even when behavioral evidence
    shows they are using it. This score detects such divergence.

    Attributes:
        claimed_influences: What the curator says influenced the thought
        sae_evidence: What SAE features indicate actually activated
        overlap_ratio: Proportion of claims supported by activation evidence
        missing_from_verbal: Active influences not mentioned verbally
        unsupported_claims: Verbal claims without activation support
    """

    claimed_influences: list[str] = field(default_factory=list)
    sae_evidence: list[str] = field(default_factory=list)
    overlap_ratio: float = 0.0
    missing_from_verbal: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)

    # HALT probe epistemic confidence (arXiv:2601.14210)
    # Probability of reliable answer from intermediate layer probe
    halt_epistemic_confidence: Optional[float] = None

    @property
    def combined_reliability(self) -> float:
        """Combined reliability score from faithfulness overlap and HALT probe.

        Weights faithfulness (60%) and HALT epistemic confidence (40%) when
        HALT is available. Falls back to overlap_ratio alone otherwise.

        Returns:
            Combined reliability score in [0, 1]
        """
        if self.halt_epistemic_confidence is not None:
            # Weighted average: 60% faithfulness, 40% HALT
            return 0.6 * self.overlap_ratio + 0.4 * self.halt_epistemic_confidence
        return self.overlap_ratio

    @property
    def is_faithful(self) -> bool:
        """Check if verbal claims are reasonably faithful to activations.

        Returns True if:
        - Overlap ratio >= 0.5 (at least half of claims supported)
        - No more than 1 unsupported claim
        """
        return self.overlap_ratio >= 0.5 and len(self.unsupported_claims) <= 1

    @property
    def divergence_severity(self) -> str:
        """Classify the severity of activation-verbal divergence."""
        if self.overlap_ratio >= 0.8 and not self.unsupported_claims:
            return "none"
        elif self.overlap_ratio >= 0.5:
            return "low"
        elif self.overlap_ratio >= 0.25:
            return "moderate"
        else:
            return "high"
