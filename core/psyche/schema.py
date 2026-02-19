"""Schema definitions for the Psyche knowledge graph."""

import json
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal, Optional
import uuid
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class FragmentState(str, Enum):
    """State of a knowledge fragment."""
    LIMBO = "LIMBO"
    VERIFIED = "VERIFIED"
    ARCHIVED = "ARCHIVED"


class SteeringDecision(str, Enum):
    """Decision about an observed trait."""
    DEVELOP = "develop"
    OBSERVE_MORE = "observe_more"
    NOT_FOR_ME = "not_for_me"


# === Knowledge Layer ===

class Fragment(BaseModel):
    """Atomic claim from ingested content."""
    uid: str
    content: str
    source: str
    state: FragmentState = FragmentState.LIMBO
    resonance: float = 0.5  # Interest/importance marker
    confidence: float = 0.8
    created_at: datetime = Field(default_factory=utc_now)
    last_accessed: datetime = Field(default_factory=utc_now)
    embedding: Optional[list[float]] = None


class Triple(BaseModel):
    """Subject-predicate-object relationship."""
    uid: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_fragment_uid: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class Entity(BaseModel):
    """Named entity (person, concept, thing)."""
    uid: str
    name: str
    entity_type: str
    description: Optional[str] = None
    source: Optional[str] = None  # "logit_lens" | "extraction" | "manual" | None
    created_at: datetime = Field(default_factory=utc_now)


# === Self-Model Layer ===

class SteeringVector(BaseModel):
    """Stored activation steering vector."""
    uid: str
    name: str
    description: Optional[str] = None
    layer: int
    coefficient: float = 1.0
    vector_data: list[float]
    active: bool = True  # Whether this vector is currently in use
    created_at: datetime = Field(default_factory=utc_now)
    supersedes_uid: Optional[str] = None  # For vector lineage
    derived_from_pairs: list[str] = Field(default_factory=list)  # ContrastivePair UIDs


class CrystalVector(BaseModel):
    """Crystallized steering vector from EvalatisSteerer population.

    Crystals are frozen vectors that proved their worth through sustained
    surprise performance. They persist across sessions and compete with
    emergent vectors for selection.
    """
    uid: str
    name: str  # e.g., "exp_01191432_042"
    zone: str  # Zone name (exploration, concept, identity)
    vector_data: list[float]
    parent_names: list[str] = Field(default_factory=list)  # Genealogy
    birth_cycle: int = 0
    birth_surprise: float = 0.0
    selection_count: int = 0
    total_surprise: float = 0.0
    staleness: float = 0.0
    children_spawned: int = 0
    retired: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class ExemplarObservation(BaseModel):
    """Observation about the human exemplar."""
    uid: str
    observed_trait: str
    category: str = "general"  # cognitive, relational, expressive, values
    evidence: list[str] = Field(default_factory=list)
    reflection: Optional[str] = None
    steering_decision: Optional[SteeringDecision] = None
    created_at: datetime = Field(default_factory=utc_now)


class IntrospectiveEntry(BaseModel):
    """Journal entry from self-reflection."""
    uid: str
    content: str
    entry_type: str = "reflection"  # reflection, steering_change, exemplar_integration
    related_observations: list[str] = Field(default_factory=list)
    related_vectors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class AffectiveState(BaseModel):
    """Current emotional/motivational state."""
    uid: str
    valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.5  # 0 to 1 (calm to excited)
    dominant_emotion: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)


class Goal(BaseModel):
    """Active objective or priority."""
    uid: str
    description: str
    priority: float = 0.5  # 0 to 1
    status: str = "active"  # active, completed, abandoned
    created_at: datetime = Field(default_factory=utc_now)


# === Developmental Layer ===

class ContrastivePair(BaseModel):
    """Positive/negative example pair for steering."""
    uid: str
    positive_text: str
    negative_text: str
    behavior_target: str
    created_at: datetime = Field(default_factory=utc_now)


class DreamCycleRecord(BaseModel):
    """Record of a consolidation cycle."""
    uid: str
    cycle_type: str  # micro, nap, full, deep
    started_at: datetime
    completed_at: Optional[datetime] = None
    fragments_processed: int = 0
    vectors_refined: int = 0
    observations_integrated: int = 0
    journal_entry_uid: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of testing a steering intervention."""
    uid: str
    vector_uid: str
    baseline_score: float
    steered_score: float
    improvement: float
    test_prompts: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


# === Prompt Self-Curation Layer ===


class PromptComponentType(str, Enum):
    """Type/layer of prompt component - lower = more foundational."""
    IDENTITY = "identity"       # Layer 0 - Core identity (least modifiable)
    AXIOM = "axiom"             # Layer 1 - Behavioral constraints
    TRAIT = "trait"             # Layer 2 - Personality characteristics
    SKILL = "skill"             # Layer 3 - Capabilities
    CONTEXT = "context"         # Layer 4 - Situational context
    INSTRUCTION = "instruction" # Layer 5 - Generation instructions (most fluid)


class PromptComponentState(str, Enum):
    """State of a prompt component."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class PromptComponentOrigin(str, Enum):
    """Origin of a prompt component."""
    INHERITED = "inherited"      # Migrated from hardcoded prompts
    RYAN_INPUT = "ryan_input"    # Advisory input from Ryan
    SELF_CREATED = "self_created"  # Created through autonomous reflection


class PromptComponent(BaseModel):
    """
    A self-curated component of Lilly's system prompt.

    Components are layered (0-5), with lower layers being more foundational
    and less frequently modified. Each modification creates a new version
    through dialectical synthesis.
    """
    uid: str
    component_type: PromptComponentType
    content: str  # The actual prompt text

    state: PromptComponentState = PromptComponentState.ACTIVE
    layer: int = 0  # 0-5, lower = more foundational

    # Versioning (like SteeringVector pattern)
    version: int = 1
    supersedes_uid: Optional[str] = None  # Chain to previous version

    # Dialectical history (like CommittedBelief pattern)
    thesis: str = ""        # Previous formulation
    antithesis: str = ""    # Identified tension/issue
    synthesis: str = ""     # New formulation (i.e., the new content)
    synthesis_reasoning: str = ""

    # Origin tracking
    origin: PromptComponentOrigin = PromptComponentOrigin.INHERITED
    source_uid: Optional[str] = None  # Fragment that inspired change

    confidence: float = 0.8
    usage_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    modified_at: datetime = Field(default_factory=utc_now)


class PromptReflectionType(str, Enum):
    """Type of reflection on a prompt component."""
    RESONANCE = "resonance"    # Does this feel like me?
    TENSION = "tension"        # Identified conflict/issue
    EXPANSION = "expansion"    # Needs to be broader
    REDUCTION = "reduction"    # Needs to be narrower


class PromptReflectionAction(str, Enum):
    """Action taken after reflection."""
    MODIFIED = "modified"
    KEPT = "kept"
    DEFERRED = "deferred"


class PromptReflection(BaseModel):
    """
    Record of reflection on a prompt component during dream cycles.

    Captures introspective examination of prompt components and
    the resulting decisions.
    """
    uid: str
    component_uid: str
    reflection_type: PromptReflectionType
    content: str  # The reflection text
    valence: float = 0.0  # -1 to 1
    resonance_score: float = 0.5  # How much this feels like "me"
    action_taken: Optional[PromptReflectionAction] = None
    cycle_type: str  # micro, nap, full, deep
    created_at: datetime = Field(default_factory=utc_now)


# === SAE Feature Layer ===


class SAEFeatureTensionType(str, Enum):
    """Type of tension relationship between SAE features."""
    ANTI_CORRELATED = "anti_correlated"  # When A fires, B doesn't
    COMPLEMENTARY = "complementary"       # Both needed for concept
    CONTRADICTORY = "contradictory"       # Semantically opposed


class SAEFeature(BaseModel):
    """
    A monosemantic SAE feature with learned interpretation.

    SAE features are sparse, interpretable directions in the model's
    representation space. Each feature represents a distinct concept
    that can be labeled through observation.
    """
    index: int  # Feature index in SAE (0 to 163839)
    layer: int = 16  # Transformer layer
    label: Optional[str] = None  # Human-readable interpretation
    category: Optional[str] = None  # Category (e.g., "philosophy", "action")
    neuronpedia_id: str = ""  # Neuronpedia lookup ID

    # Statistics from observations
    activation_mean: float = 0.0
    activation_max: float = 0.0
    observation_count: int = 0

    # Co-occurrence with concepts
    associated_concepts: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SAEFeatureTension(BaseModel):
    """
    Learned tension relationship between two SAE features.

    Tracks which feature pairs create productive cognitive tension,
    indicated by high surprise when one feature transitions to another.
    Used to suggest opposing features when exploration stagnates.
    """
    feature_a: int  # First feature index (always smaller)
    feature_b: int  # Second feature index (always larger)
    tension_type: SAEFeatureTensionType = SAEFeatureTensionType.ANTI_CORRELATED

    # Accumulated observations
    surprise_sum: float = 0.0  # Sum of surprises when tension observed
    observation_count: int = 0

    # Derived metrics
    @property
    def surprise_avg(self) -> float:
        """Average surprise when this tension is observed."""
        if self.observation_count == 0:
            return 0.0
        return self.surprise_sum / self.observation_count

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SAECoactivation(BaseModel):
    """
    Record of two SAE features firing together in a thought.

    Coactivation indicates conceptual association: when two features
    fire together across different thoughts, they represent related ideas.
    """
    feature_a: int
    feature_b: int
    strength: float  # Geometric mean of activations
    cycle_number: int
    thought_uid: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class SAEFeatureSnapshot(BaseModel):
    """Snapshot of active SAE features at thought generation time.

    When a thought is generated, the top-N active SAE features are captured
    and linked to the thought fragment. This enables later association with
    entities extracted from the thought via HippoRAG.

    The snapshot captures the internal activation state that produced the thought,
    enabling "subconscious" memory retrieval based on what was being processed
    internally, not just what appeared in the output text.
    """
    fragment_uid: str  # Link to the thought fragment
    features: list[tuple[int, float]]  # [(feature_idx, activation), ...]
    cycle: int  # Cognitive cycle number
    created_at: datetime = Field(default_factory=utc_now)


class EvocationEdge(BaseModel):
    """EVOKES relationship between SAE feature and Entity.

    This edge captures learned associations between internal activation states
    (SAE features) and semantic concepts (entities). When a feature consistently
    activates in thoughts that produce certain entities, the association strengthens.

    The weight is updated via EMA: weight = decay * old + (1-decay) * new
    This allows recent observations to have more influence while preserving
    long-term learning.
    """
    feature_idx: int
    entity_uid: str
    weight: float = 0.0  # EMA-updated association strength
    observation_count: int = 0
    decay_rate: float = 0.995  # EMA decay (0.995 preserves ~200 obs half-life)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def update_weight(self, new_activation: float) -> None:
        """Update weight using EMA with decay."""
        self.weight = self.decay_rate * self.weight + (1 - self.decay_rate) * new_activation
        self.observation_count += 1
        self.updated_at = utc_now()


class MoodEvocationProperties(BaseModel):
    """Properties stored on SAEFeature node for mood evocation.

    Mood associations are 1:1 with features (each feature has one mood pattern),
    so they're stored as properties rather than edges.
    """
    dominant_emotion: str  # Primary emotion label (e.g., "curiosity", "wonder")
    valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.5  # 0 to 1 (calm to excited)
    weight: float = 0.0  # Association strength (EMA-updated)
    observation_count: int = 0
    updated_at: datetime = Field(default_factory=utc_now)


class QuestionEvocationEdge(BaseModel):
    """EVOKES_QUESTION relationship between SAE feature and InsightZettel.

    This edge captures learned associations between internal activation states
    (SAE features) and unresolved questions. When features consistently
    activate when certain questions are on Lilly's mind, the association strengthens.

    This enables "subconscious" resurfacing of questions when relevant features fire.
    """
    feature_idx: int
    question_uid: str  # UID of the InsightZettel containing the open question
    weight: float = 0.0  # EMA-updated association strength
    urgency: float = 0.5  # How much this question wants attention (0-1)
    recurrence_count: int = 0  # How many times this pattern triggered this question
    decay_rate: float = 0.995  # EMA decay
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def update_weight(self, new_activation: float, urgency_boost: float = 0.0) -> None:
        """Update weight using EMA with decay.

        Args:
            new_activation: Current feature activation strength
            urgency_boost: Additional urgency from recurrence (0 to 0.5)
        """
        self.weight = self.decay_rate * self.weight + (1 - self.decay_rate) * new_activation
        self.urgency = min(1.0, self.urgency + urgency_boost)
        self.recurrence_count += 1
        self.updated_at = utc_now()


# === Insight Zettel Layer ===


class QuestionStatus(str, Enum):
    """Status of an open question in an InsightZettel."""
    OPEN = "open"               # Still seeking resolution
    ADDRESSED = "addressed"     # Linked to an answering insight
    DISSOLVED = "dissolved"     # No longer relevant
    NOT_A_QUESTION = "none"     # Zettel has no question component


class InsightSourceType(str, Enum):
    """Source type for an InsightZettel."""
    COGNITIVE = "cognitive"    # Internal thought from cognitive loop
    LETTER = "letter"          # From correspondence
    RESEARCH = "research"      # From research documents
    REFLECTION = "reflection"  # From dream cycle reflections
    SIMULATION = "simulation"  # From hypothesis testing simulation


class InsightZettel(BaseModel):
    """
    A distilled insight from thought or external source.

    InsightZettels are the atomic units of Lilly's knowledge library,
    capturing key realizations and open questions that compound over time.
    They can be retrieved via:
    - Semantic similarity (embedding-based)
    - SAE activation overlap (internal resonance)
    - Open question matching (forward progress)

    Each zettel tracks its lineage (what insights it emerged from) and
    can be linked to answers (when questions get addressed).
    """
    uid: str
    insight_text: str  # The distilled insight (1-2 sentences)
    question_text: Optional[str] = None  # Open question that emerged
    question_status: QuestionStatus = QuestionStatus.OPEN

    source_type: InsightSourceType
    source_uid: str  # Fragment UID for the full source

    concept: str  # Concept being explored when insight emerged
    cycle: Optional[int] = None  # Cognitive cycle number (None for external)

    # For retrieval
    embedding: Optional[list[float]] = None  # Semantic retrieval
    sae_feature_indices: list[int] = Field(default_factory=list)  # Top-N indices for fast SAE lookup

    created_at: datetime = Field(default_factory=utc_now)

    # Novelty tracking (arXiv:2601.12542 inspired)
    novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)  # 0=duplicate, 1=novel
    is_refinement: bool = False  # True if novelty_score < threshold
    refines_uid: Optional[str] = None  # Parent zettel UID if refinement


# === Cognitive State Persistence ===


class CognitiveStateSnapshot(BaseModel):
    """Persisted cognitive state for continuity across service restarts.

    Captures the essential elements needed to resume cognitive processing
    without losing context:
    - The curated prompt for the next generation cycle
    - Current insight and question driving exploration
    - Recent concept for contextual grounding
    - Cycle count for continuity

    Only one snapshot should exist at a time - each cycle overwrites the previous.
    """
    uid: str = "cognitive_state_current"  # Singleton pattern
    curated_prompt: str  # Full prompt for next generation
    last_concept: str  # Most recent exploration concept
    current_insight: str = ""  # Insight driving forward momentum
    current_question: str = ""  # Question driving exploration
    cycle_count: int = 0  # For cycle continuity
    recent_concepts: list[str] = Field(default_factory=list)  # Last N concepts
    updated_at: datetime = Field(default_factory=utc_now)


# === Reflexion Layer ===


class CognitiveParameter(BaseModel):
    """Persistent cognitive parameter (Tier 2 modification).

    Stores runtime parameters that persist across restarts.
    Modified by Reflexion phase when health signals warrant change.
    """
    uid: str
    path: str  # Dot-notation path (e.g., "simulation.confidence_threshold")
    value: Any  # Current value
    value_type: str  # "float", "int", "str", "bool"
    previous_value: Optional[Any] = None
    rationale: Optional[str] = None  # Why last change was made
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReflexionEntryNode(BaseModel):
    """Reflexion journal entry for Psyche persistence.

    Captures health assessment, modifications, and narrative
    from each Reflexion phase execution.
    """
    uid: str
    cycle_number: int
    timestamp: datetime = Field(default_factory=utc_now)
    health_assessment_json: str  # Serialized HealthAssessment
    metrics_snapshot: dict[str, float] = Field(default_factory=dict)
    baseline_comparison: dict[str, float] = Field(default_factory=dict)
    phenomenological: dict[str, float] = Field(default_factory=dict)
    modifications_count: int = 0
    overall_coherence: float = 0.0
    narrative_summary: str = ""


# === Prediction Learning Layer ===


@dataclass
class PredictionPattern:
    """Aggregated prediction statistics for a condition type.

    Uses @dataclass instead of Pydantic BaseModel because this is a
    mutable stats container that accumulates outcomes over time.
    """

    RELIABILITY_THRESHOLD: ClassVar[int] = 10  # Minimum samples for reliable pattern

    condition_type: str
    success_count: int = 0
    failure_count: int = 0
    dominant_failure: str | None = None
    updated_at: datetime = dataclass_field(default_factory=utc_now)

    @property
    def total(self) -> int:
        """Total predictions for this condition type."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0), defaults to 0.5 if no data."""
        return self.success_count / self.total if self.total > 0 else 0.5

    @property
    def is_reliable(self) -> bool:
        """True if pattern has enough data to be meaningful."""
        return self.total >= self.RELIABILITY_THRESHOLD

    def record_outcome(self, success: bool, failure_reason: str | None = None) -> None:
        """Record a prediction outcome.

        Args:
            success: Whether the prediction was verified
            failure_reason: The failure reason if not successful
        """
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            # Track most common failure (simple: just keep most recent for now)
            if failure_reason:
                self.dominant_failure = failure_reason
        self.updated_at = utc_now()

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "condition_type": self.condition_type,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "dominant_failure": self.dominant_failure,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredictionPattern":
        """Deserialize from dictionary."""
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        return cls(
            condition_type=data["condition_type"],
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            dominant_failure=data.get("dominant_failure"),
            updated_at=updated_at or utc_now(),
        )


# === Research Query Persistence ===


class ResearchQueryResult(BaseModel):
    """Persisted answer from NotebookLM research query.

    Stores research answers so Lilly can build on previous
    architectural inquiries without re-querying. Part of her
    persistent psyche, not a temporary cache.
    """

    uid: str  # Pattern: "research_{uuid4().hex[:12]}"
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)

    # For semantic retrieval of related queries
    embedding: Optional[list[float]] = None  # 1024-dim retrieval embeddings

    # Context
    notebook_id: str
    cycle: Optional[int] = None  # Cognitive cycle when queried

    created_at: datetime = Field(default_factory=utc_now)


# === Narration Phrase Layer ===


class PhraseType(str, Enum):
    """Type of narration phrase for Lilly's TTS output."""
    CONCEPT_BRIDGE = "concept_bridge"
    OPENING_HOOK = "opening_hook"
    PHASE_TRANSITION = "phase_transition"
    CHECKPOINT = "checkpoint"


class NarrationPhrase(BaseModel):
    """A Lilly-generated narration phrase stored in the graph.

    Narration phrases are short templates Lilly uses for TTS output.
    They support usage tracking and rotation to prevent staleness.
    Phrases are retired after exceeding a usage threshold (default 20).
    """
    uid: Optional[str] = None
    text: str
    phrase_type: PhraseType
    usage_count: int = 0
    created_cycle: int
    last_used_cycle: Optional[int] = None
    retired: bool = False


# === Learned Skills Layer ===


class LearnedSkill(BaseModel):
    """Compact skill learned from verified hypothesis patterns (~500 tokens).

    Skills are auto-generated when hypotheses reach VERIFIED status (70%+
    verification rate). They capture patterns that can be injected into
    generation prompts to improve cognitive quality.

    Inspired by HuggingFace UPSKILL: teaching models through portable skill packages.
    """
    uid: str  # Pattern: "skill_{uuid4().hex[:12]}"
    name: str  # e.g., "seek_tension"
    description: str  # 1-2 sentence summary

    # Source
    source_hypothesis_uid: str
    cognitive_operation: str

    # Content
    pattern_summary: str  # Top 3 patterns, pipe-delimited
    when_to_apply: str  # Context trigger (~100 chars)
    positive_example: str  # From hypothesis
    negative_example: str  # From hypothesis

    # Retrieval
    embedding: Optional[list[float]] = None

    # Effectiveness
    usage_count: int = 0
    success_count: int = 0
    effectiveness_score: float = 0.5

    # Lifecycle
    created_cycle: int = 0
    last_used_cycle: Optional[int] = None
    retired: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # SOAR Curriculum Learning fields
    is_promoted: bool = False  # In 𝒟ᵦₑₛₜ (promotion queue)
    is_weak_spot_targeted: bool = False  # Generated for weak cognitive operation
    structural_pattern_count: int = 0  # Count of structural vs conclusion patterns
    hard_problem_accuracy: float = 0.5  # Accuracy on hard problems when injected
    improvement_over_baseline: float = 0.0  # Δ from baseline
    promotion_cycle: Optional[int] = None  # Cycle when promoted (if any)


# === Autonomous Decision Layer ===


class AutonomousDecision(BaseModel):
    """Record of autonomous judgment and action.

    Stores every significant decision Lilly makes autonomously,
    including the knowledge she synthesized, her judgment,
    the action taken, expectations, and eventual outcomes.
    """

    id: str = Field(default_factory=lambda: f"decision_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cycle_id: str
    
    # Phase 3: Cycle tracking for robust outcome assessment
    cycle_count_created: int  # Cycle number when decision was made
    health_created: "HealthCategory"  # Health state when decision was made
    cycle_count_assessed: int | None = None  # Cycle when outcome was assessed (idempotency)

    # Decision context
    question: str  # What triggered this decision
    knowledge_synthesized: list[str] = Field(default_factory=list)  # Refs to zettels, experiments, observations

    # Judgment and action
    judgment: str  # The reasoning/conclusion
    action: dict  # Structured action taken
    expectation: str  # What she expects to happen

    # Outcomes (filled in after observation period)
    outcome: str | None = None
    lesson_learned: str | None = None
    success: bool | None = None

    # Relationships
    related_hypothesis: str | None = None
    related_experiment: str | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for graph storage."""
        data = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "cycle_id": self.cycle_id,
            "cycle_count_created": self.cycle_count_created,
            "health_created": self.health_created.value,
            "question": self.question,
            "knowledge_synthesized": self.knowledge_synthesized,
            "judgment": self.judgment,
            "action": self.action,
            "expectation": self.expectation,
        }

        # Add optional fields if present
        if self.cycle_count_assessed is not None:
            data["cycle_count_assessed"] = self.cycle_count_assessed
        if self.outcome is not None:
            data["outcome"] = self.outcome
        if self.lesson_learned is not None:
            data["lesson_learned"] = self.lesson_learned
        if self.success is not None:
            data["success"] = self.success
        if self.related_hypothesis is not None:
            data["related_hypothesis"] = self.related_hypothesis
        if self.related_experiment is not None:
            data["related_experiment"] = self.related_experiment

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AutonomousDecision":
        """Deserialize from dictionary with proper error handling."""
        # Parse timestamp with fallback
        if isinstance(data.get("timestamp"), str):
            try:
                data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                # Fallback to None if parsing fails
                data["timestamp"] = None

        # Parse health_created from string
        if isinstance(data.get("health_created"), str):
            from core.cognitive.reflexion.schemas import HealthCategory
            data["health_created"] = HealthCategory(data["health_created"])

        # Parse JSON strings back to lists/dicts
        if isinstance(data.get("knowledge_synthesized"), str):
            try:
                data["knowledge_synthesized"] = json.loads(data["knowledge_synthesized"])
            except json.JSONDecodeError:
                # Fallback to empty list if parsing fails
                data["knowledge_synthesized"] = []

        if isinstance(data.get("action"), str):
            try:
                data["action"] = json.loads(data["action"])
            except json.JSONDecodeError:
                # Fallback to empty dict if parsing fails
                data["action"] = {}

        return cls(**data)


# Rebuild models to resolve forward references
# This is required because HealthCategory is only imported under TYPE_CHECKING


# === Individuation Dynamics Layer ===


class DynamicsPhase(str, Enum):
    """Phase of an identity element's trajectory."""
    NASCENT = "nascent"
    STABLE = "stable"
    CRYSTALLIZING = "crystallizing"
    DISSOLVING = "dissolving"
    OSCILLATING = "oscillating"
    VOLATILE = "volatile"


class TransitionTrigger(str, Enum):
    """What triggered a phase transition."""
    VELOCITY_SPIKE = "velocity_spike"
    PHASE_CHANGE = "phase_change"
    EVIDENCE_SURGE = "evidence_surge"
    CONTRADICTION_SURGE = "contradiction_surge"
    TIME_DECAY = "time_decay"
    EXTERNAL_REVISION = "external_revision"


class IdentityTrajectoryNode(BaseModel):
    """
    Graph node for tracking identity element evolution.
    
    Stores the dynamics of how commitments, values, and beliefs
    change over time - velocity, acceleration, phase.
    """
    uid: str
    element_id: str  # UID of the commitment/value/belief
    element_type: Literal['commitment', 'value', 'belief']
    
    # Current dynamics state
    current_position: float = 0.5
    current_velocity: float = 0.0
    current_acceleration: float = 0.0
    current_phase: DynamicsPhase = DynamicsPhase.NASCENT
    phase_stability: int = 0  # Cycles in current phase
    
    # History (bounded - full history in IndividuationDynamics)
    observation_count: int = 0
    
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AttractorBasinNode(BaseModel):
    """
    Graph node for stable identity configurations.
    
    Attractors are states Lilly tends to return to - recurring
    patterns of beliefs/values/commitments.
    """
    uid: str
    
    # Center state as JSON string (element_id -> confidence)
    center_state_json: str = "{}"
    radius: float = 0.1
    
    # Visit metrics
    visit_count: int = 0
    total_dwell_time: float = 0.0  # Seconds
    last_visit: datetime = Field(default_factory=utc_now)
    formation_time: datetime = Field(default_factory=utc_now)
    
    # Strength (computed from visits and dwell time)
    strength: float = 0.0
    
    # Elements that define this attractor
    element_ids_json: str = "[]"


class IndividuationTransitionNode(BaseModel):
    """
    Graph node for recording phase transitions.
    
    These are significant shifts in how identity elements evolve -
    not just changes in content, but changes in dynamics.
    """
    uid: str
    
    from_phase: DynamicsPhase
    to_phase: DynamicsPhase
    trigger: TransitionTrigger = TransitionTrigger.PHASE_CHANGE
    trigger_element: str  # Element ID that triggered the shift
    
    timestamp: datetime = Field(default_factory=utc_now)
    
    # Other elements affected by this transition
    affected_elements_json: str = "[]"
    
    # Magnitude of the change
    energy_released: float = 0.0
    
    # Human-readable description
    narrative: str = ""


# === Cognitive Anchor Layer ===


class CognitiveAnchor(BaseModel):
    """A cognitive mode anchor - predefined or emergent.

    Cognitive anchors serve as semantic landmarks for classifying
    thoughts into cognitive modes. Predefined anchors are the 10
    original modes; emergent anchors are discovered through orphan
    thought clustering.

    Attributes:
        uid: Unique identifier (e.g., "anchor_emergent_abc123")
        mode_name: Snake_case name (e.g., "philosophical_inquiry")
        anchor_text: Exemplar thought that defines this mode
        description: Human-readable description of the mode
        embedding: Optional embedding vector for similarity computation
        is_predefined: True for original 10 modes, False for emergent
        discovered_at: When this anchor was discovered (emergent only)
        discovery_cycle: Cognitive cycle at discovery (emergent only)
        source_thought_uid: UID of crystallizing thought (emergent only)
        usage_count: Times this mode was dominant
        confidence: Strength of the anchor (EMA-updated)
        retired: True if anchor retired due to low usage
        created_at: Timestamp of creation
    """

    uid: str
    mode_name: str
    anchor_text: str
    description: str
    embedding: Optional[list[float]] = None
    is_predefined: bool = True
    discovered_at: Optional[datetime] = None
    discovery_cycle: Optional[int] = None
    source_thought_uid: Optional[str] = None
    usage_count: int = 0
    confidence: float = 1.0
    retired: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _rebuild_models() -> None:
    """Rebuild Pydantic models to resolve forward references."""
    from core.cognitive.reflexion.schemas import HealthCategory  # noqa: F401

    AutonomousDecision.model_rebuild()


# Defer rebuild to avoid circular import issues
# _rebuild_models()
