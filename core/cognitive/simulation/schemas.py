"""Schemas for the Graph-Preflexor simulation phase.

This module defines the data structures for hypotheses, predictions, and
simulation results used in Phase 2.5 of the cognitive cycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class HypothesisStatus(Enum):
    """Status of a hypothesis in its lifecycle."""

    PROPOSED = "proposed"  # Just created, not yet tested
    ACTIVE = "active"  # Being actively tested with predictions
    VERIFIED = "verified"  # Predictions mostly confirmed
    FALSIFIED = "falsified"  # Predictions mostly failed
    ABANDONED = "abandoned"  # Replaced or no longer relevant


class PredictionStatus(Enum):
    """Status of a prediction awaiting verification."""

    PENDING = "pending"  # Awaiting verification conditions
    VERIFIED = "verified"  # Prediction confirmed
    FALSIFIED = "falsified"  # Prediction failed
    EXPIRED = "expired"  # Condition window passed without resolution


class FailureReason(str, Enum):
    """Why a prediction was falsified. Three categories only."""

    EXPIRED = "expired"  # Condition never triggered within window
    WRONG_DIRECTION = "wrong_direction"  # Outcome opposite of prediction
    OVERCONFIDENT = "overconfident"  # High confidence prediction falsified


class PredictionConditionType(Enum):
    """Type of condition that triggers prediction verification."""

    TIME_BASED = "time_based"  # After N cycles
    CONCEPT_MENTIONED = "concept_mentioned"  # Concept appears in thought
    ENTITY_OBSERVED = "entity_observed"  # Entity in extraction
    METRIC_THRESHOLD = "metric_threshold"  # Graph/semantic metric crosses threshold
    BELIEF_CHANGE = "belief_change"  # Belief confidence changes
    SAE_FEATURE_PATTERN = "sae_feature_pattern"  # SAE feature above/below threshold
    EMOTIONAL_SHIFT = "emotional_shift"  # Emotional field changes (valence/arousal)
    GOAL_PROGRESS = "goal_progress"  # Goal alignment changes (delta or absolute)


class MetricsSnapshot(BaseModel):
    """Per-cycle graph and semantic metrics for computable verification.

    Captures the state of the knowledge graph at a specific cognitive cycle,
    enabling predictions to make quantitative claims that can be verified
    against measured changes in graph structure and semantic diversity.

    Attributes:
        cycle: Cognitive cycle number when snapshot was taken
        timestamp: When the snapshot was captured

        Structural metrics (from GraphEntropy):
        structural_entropy: Shannon entropy of degree distribution (0-1)
        cluster_entropy: Entropy of cluster size distribution (0-1)
        orphan_rate: Proportion of disconnected nodes (0-1)
        hub_concentration: Gini coefficient of node degrees (0-1, higher = more hubs)
        node_count: Total number of nodes in graph
        edge_count: Total number of edges in graph
        cluster_count: Number of connected components

        Semantic metrics (from SemanticEntropy):
        semantic_entropy: Spectral entropy of embedding similarity (0-1)
        topic_concentration: Inverse of semantic entropy (0-1, higher = fewer topics)
        effective_dimensions: Number of significant semantic dimensions

        Composite metrics (from DiscoveryParameter):
        discovery_parameter: H_sem - H_struct (positive = semantic pressure)
    """

    cycle: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Structural metrics
    structural_entropy: float = 0.0
    cluster_entropy: float = 0.0
    orphan_rate: float = 0.0
    hub_concentration: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    cluster_count: int = 0

    # Semantic metrics
    semantic_entropy: float = 0.0
    topic_concentration: float = 0.0
    effective_dimensions: float = 0.0

    # Composite metrics
    discovery_parameter: float = 0.0

    # Verification metrics (computed from predictions)
    verification_rate: float = 0.0  # verified / (verified + falsified)

    # Psychological metrics (computed every cycle)
    self_understanding: float = 0.0
    alignment_correlation: float = 0.0

    # Expensive metrics (computed only during experiments)
    exploration_depth: float | None = None
    agency_development: float | None = None

    # Experiment context
    active_experiment_uid: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp.isoformat(),
            "structural_entropy": self.structural_entropy,
            "cluster_entropy": self.cluster_entropy,
            "orphan_rate": self.orphan_rate,
            "hub_concentration": self.hub_concentration,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "cluster_count": self.cluster_count,
            "semantic_entropy": self.semantic_entropy,
            "topic_concentration": self.topic_concentration,
            "effective_dimensions": self.effective_dimensions,
            "discovery_parameter": self.discovery_parameter,
            "verification_rate": self.verification_rate,
            "self_understanding": self.self_understanding,
            "alignment_correlation": self.alignment_correlation,
            "exploration_depth": self.exploration_depth,
            "agency_development": self.agency_development,
            "active_experiment_uid": self.active_experiment_uid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsSnapshot":
        """Deserialize from dictionary."""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
        except (ValueError, KeyError):
            timestamp = datetime.now(timezone.utc)
        return cls(
            cycle=data.get("cycle", 0),
            timestamp=timestamp,
            structural_entropy=data.get("structural_entropy", 0.0),
            cluster_entropy=data.get("cluster_entropy", 0.0),
            orphan_rate=data.get("orphan_rate", 0.0),
            hub_concentration=data.get("hub_concentration", 0.0),
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
            cluster_count=data.get("cluster_count", 0),
            semantic_entropy=data.get("semantic_entropy", 0.0),
            topic_concentration=data.get("topic_concentration", 0.0),
            effective_dimensions=data.get("effective_dimensions", 0.0),
            discovery_parameter=data.get("discovery_parameter", 0.0),
            verification_rate=data.get("verification_rate", 0.0),
            self_understanding=data.get("self_understanding", 0.0),
            alignment_correlation=data.get("alignment_correlation", 0.0),
            exploration_depth=data.get("exploration_depth"),
            agency_development=data.get("agency_development"),
            active_experiment_uid=data.get("active_experiment_uid"),
        )

    def get_metric(self, name: str) -> float:
        """Get a metric value by name.

        Args:
            name: Metric name (e.g., 'semantic_entropy', 'orphan_rate')

        Returns:
            The metric value, or 0.0 if not found
        """
        return getattr(self, name, 0.0)


class Hypothesis(BaseModel):
    """A hypothesis generated through Graph-Preflexor simulation.

    Hypotheses capture structured reasoning about potential relationships,
    patterns, or outcomes. They are tracked and updated as predictions
    are verified or falsified.

    Attributes:
        uid: Unique identifier for the hypothesis
        statement: The hypothesis statement being tested
        source_zettel_uid: UID of zettel that seeded this hypothesis (if any)
        source_belief_uid: UID of belief that seeded this hypothesis (if any)
        source_thought: The thought that triggered simulation
        status: Current lifecycle status
        confidence: Confidence level 0.0-1.0
        brainstorm_trace: Full brainstorm block from simulation
        graph_structure: JSON graph from simulation (nodes/edges)
        patterns_extracted: Patterns discovered during simulation
        synthesis_narrative: Final synthesis from simulation
        cycle_generated: Cognitive cycle when this hypothesis was created
        predictions_count: Total predictions derived from this hypothesis
        verified_count: Number of verified predictions
        falsified_count: Number of falsified predictions
        created_at: Timestamp of creation
        cognitive_operation: The cognitive operation this hypothesis suggests
            (e.g., "explore_emergence", "deepen_understanding", "seek_contradiction")
        positive_example: Example text demonstrating the desired cognitive pattern
        negative_example: Example text demonstrating what to avoid
        steering_vector_uid: UID of the associated HypothesisSteeringVector (if extracted)
        falsification_condition: Specific observable outcome that would prove hypothesis wrong
    """

    uid: str = Field(default_factory=lambda: f"hyp_{uuid4().hex[:8]}")
    statement: str = ""
    source_zettel_uid: Optional[str] = None
    source_belief_uid: Optional[str] = None
    source_thought: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5
    brainstorm_trace: str = ""
    graph_structure: dict = Field(default_factory=dict)
    patterns_extracted: list[str] = Field(default_factory=list)
    synthesis_narrative: str = ""
    cycle_generated: int = 0
    predictions_count: int = 0
    verified_count: int = 0
    falsified_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Outcome-based steering fields
    cognitive_operation: str = ""
    positive_example: str = ""
    negative_example: str = ""
    steering_vector_uid: Optional[str] = None
    # Falsification condition - what would disprove this hypothesis
    falsification_condition: Optional[str] = None
    # Novelty statement - what NEW claim this makes and how it differs from previous cycles
    novelty_statement: Optional[str] = Field(
        default=None,
        description="What NEW claim this makes and how it differs from previous cycles"
    )
    # Follow-up simulation cooldown tracking
    last_follow_up_cycle: Optional[int] = None
    # Lineage tracking for refined hypotheses (accumulates verification history)
    parent_hypothesis_uid: Optional[str] = None
    # Accumulated counts across lineage (includes parent counts)
    lineage_verified_count: int = 0
    lineage_falsified_count: int = 0

    # Experiment extension fields
    is_experiment: bool = False
    experiment_domain: Optional[str] = None  # ExperimentDomain value as string

    # Intervention specification
    parameter_path: Optional[str] = None
    control_value: Optional[float] = None
    treatment_value: Optional[float] = None

    # Experiment lifecycle
    baseline_cycles: int = 5
    treatment_cycles: int = 10
    washout_cycles: int = 3
    current_phase: str = "pending"  # ExperimentPhase value
    phase_start_cycle: Optional[int] = None

    # Success criteria
    target_metric: Optional[str] = None
    expected_direction: str = "increase"
    min_effect_size: float = 0.1
    rollback_trigger: Optional[float] = None

    # Outcome (populated after COMPLETE)
    baseline_mean: Optional[float] = None
    treatment_mean: Optional[float] = None
    effect_size: Optional[float] = None
    recommendation: Optional[str] = None  # "ADOPT", "REJECT", "INCONCLUSIVE"

    # Lifecycle management fields
    calibration_error: float = 0.0  # |confidence - actual_rate| EMA
    last_evaluated_cycle: int = 0  # Cycle when lifecycle was last evaluated
    pending_count: int = 0  # Number of pending predictions

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Validate confidence range."""
        return max(0.0, min(1.0, v))

    @field_validator("experiment_domain", mode="before")
    @classmethod
    def convert_experiment_domain(cls, v):
        """Convert ExperimentDomain enum to string value."""
        if v is None:
            return None
        # If it's an enum, get its value
        if hasattr(v, "value"):
            return v.value
        return str(v)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "uid": self.uid,
            "statement": self.statement,
            "source_zettel_uid": self.source_zettel_uid,
            "source_belief_uid": self.source_belief_uid,
            "source_thought": self.source_thought,
            "status": self.status.value,
            "confidence": self.confidence,
            "brainstorm_trace": self.brainstorm_trace,
            "graph_structure": self.graph_structure,
            "patterns_extracted": self.patterns_extracted,
            "synthesis_narrative": self.synthesis_narrative,
            "cycle_generated": self.cycle_generated,
            "predictions_count": self.predictions_count,
            "verified_count": self.verified_count,
            "falsified_count": self.falsified_count,
            "created_at": self.created_at.isoformat(),
            "cognitive_operation": self.cognitive_operation,
            "positive_example": self.positive_example,
            "negative_example": self.negative_example,
            "steering_vector_uid": self.steering_vector_uid,
            "falsification_condition": self.falsification_condition,
            "novelty_statement": self.novelty_statement,
            "last_follow_up_cycle": self.last_follow_up_cycle,
            # Lineage tracking fields
            "parent_hypothesis_uid": self.parent_hypothesis_uid,
            "lineage_verified_count": self.lineage_verified_count,
            "lineage_falsified_count": self.lineage_falsified_count,
            # Experiment extension fields
            "is_experiment": self.is_experiment,
            "experiment_domain": self.experiment_domain,
            "parameter_path": self.parameter_path,
            "control_value": self.control_value,
            "treatment_value": self.treatment_value,
            "baseline_cycles": self.baseline_cycles,
            "treatment_cycles": self.treatment_cycles,
            "washout_cycles": self.washout_cycles,
            "current_phase": self.current_phase,
            "phase_start_cycle": self.phase_start_cycle,
            "target_metric": self.target_metric,
            "expected_direction": self.expected_direction,
            "min_effect_size": self.min_effect_size,
            "rollback_trigger": self.rollback_trigger,
            "baseline_mean": self.baseline_mean,
            "treatment_mean": self.treatment_mean,
            "effect_size": self.effect_size,
            "recommendation": self.recommendation,
            # Lifecycle management fields
            "calibration_error": self.calibration_error,
            "last_evaluated_cycle": self.last_evaluated_cycle,
            "pending_count": self.pending_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        """Deserialize from dictionary."""
        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except (ValueError, KeyError):
            created_at = datetime.now(timezone.utc)

        return cls(
            uid=data["uid"],
            statement=data["statement"],
            source_zettel_uid=data.get("source_zettel_uid"),
            source_belief_uid=data.get("source_belief_uid"),
            source_thought=data.get("source_thought", ""),
            status=HypothesisStatus(data["status"]),
            confidence=data.get("confidence", 0.5),
            brainstorm_trace=data.get("brainstorm_trace", ""),
            graph_structure=data.get("graph_structure", {}),
            patterns_extracted=data.get("patterns_extracted", []),
            synthesis_narrative=data.get("synthesis_narrative", ""),
            cycle_generated=data.get("cycle_generated", 0),
            predictions_count=data.get("predictions_count", 0),
            verified_count=data.get("verified_count", 0),
            falsified_count=data.get("falsified_count", 0),
            created_at=created_at,
            cognitive_operation=data.get("cognitive_operation", ""),
            positive_example=data.get("positive_example", ""),
            negative_example=data.get("negative_example", ""),
            steering_vector_uid=data.get("steering_vector_uid"),
            falsification_condition=data.get("falsification_condition"),
            novelty_statement=data.get("novelty_statement"),
            last_follow_up_cycle=data.get("last_follow_up_cycle"),
            # Lineage tracking fields
            parent_hypothesis_uid=data.get("parent_hypothesis_uid"),
            lineage_verified_count=data.get("lineage_verified_count", 0),
            lineage_falsified_count=data.get("lineage_falsified_count", 0),
            # Experiment extension fields
            is_experiment=data.get("is_experiment", False),
            experiment_domain=data.get("experiment_domain"),
            parameter_path=data.get("parameter_path"),
            control_value=data.get("control_value"),
            treatment_value=data.get("treatment_value"),
            baseline_cycles=data.get("baseline_cycles", 5),
            treatment_cycles=data.get("treatment_cycles", 10),
            washout_cycles=data.get("washout_cycles", 3),
            current_phase=data.get("current_phase", "pending"),
            phase_start_cycle=data.get("phase_start_cycle"),
            target_metric=data.get("target_metric"),
            expected_direction=data.get("expected_direction", "increase"),
            min_effect_size=data.get("min_effect_size", 0.1),
            rollback_trigger=data.get("rollback_trigger"),
            baseline_mean=data.get("baseline_mean"),
            treatment_mean=data.get("treatment_mean"),
            effect_size=data.get("effect_size"),
            recommendation=data.get("recommendation"),
            # Lifecycle management fields
            calibration_error=data.get("calibration_error", 0.0),
            last_evaluated_cycle=data.get("last_evaluated_cycle", 0),
            pending_count=data.get("pending_count", 0),
        )

    @property
    def verification_rate(self) -> float:
        """Proportion of predictions that were verified."""
        total = self.verified_count + self.falsified_count
        if total == 0:
            return 0.0
        return self.verified_count / total

    @property
    def lineage_verification_rate(self) -> float:
        """Proportion of predictions verified across entire hypothesis lineage.

        This includes predictions from parent hypotheses, providing a more
        accurate measure of a hypothesis line's overall success rate.
        """
        total = self.lineage_verified_count + self.lineage_falsified_count
        if total == 0:
            return 0.0
        return self.lineage_verified_count / total

    @property
    def total_lineage_predictions(self) -> int:
        """Total predictions across the entire hypothesis lineage."""
        return self.lineage_verified_count + self.lineage_falsified_count

    @property
    def total_experiment_cycles(self) -> int:
        """Total duration of experiment in cycles."""
        return self.baseline_cycles + self.treatment_cycles + self.washout_cycles


class Prediction(BaseModel):
    """A testable prediction derived from a hypothesis.

    Predictions have conditions that determine when they can be verified
    (e.g., after N cycles, when a concept is mentioned). They track
    outcomes for updating hypothesis confidence.

    Attributes:
        uid: Unique identifier for the prediction
        hypothesis_uid: Parent hypothesis this prediction tests
        claim: The specific claim being predicted
        condition_type: What type of condition triggers verification
        condition_value: The specific condition value (e.g., "5" cycles, "emergence")
        status: Current verification status
        confidence: Confidence in this specific prediction (0.0-1.0)
        outcome: Description of what actually happened (when verified/falsified)
        accuracy_score: How accurate the prediction was (0.0-1.0)
        earliest_verify_cycle: Earliest cycle when this can be verified
        expiry_cycle: Cycle after which prediction expires without verification
        verification_cycle: Cycle when prediction was actually verified
        created_at: Timestamp of creation
    """

    uid: str = Field(default_factory=lambda: f"pred_{uuid4().hex[:8]}")
    hypothesis_uid: str = ""
    claim: str = ""
    condition_type: PredictionConditionType = PredictionConditionType.TIME_BASED
    condition_value: str = ""
    status: PredictionStatus = PredictionStatus.PENDING
    confidence: float = 0.5
    outcome: Optional[str] = None
    accuracy_score: Optional[float] = None
    earliest_verify_cycle: Optional[int] = None
    expiry_cycle: Optional[int] = None
    verification_cycle: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Goal-based verification fields
    target_goal_uid: Optional[str] = None  # e.g., "goal:epistemic_growth"
    expected_goal_delta: float = 0.0  # Positive = expect improvement
    actual_goal_delta: Optional[float] = None  # Measured after verification
    goal_snapshot_before: Optional[float] = None  # Goal alignment at prediction creation
    goal_snapshot_after: Optional[float] = None  # Goal alignment at verification
    # Metric-based verification fields
    baseline_cycle: Optional[int] = None  # Cycle when baseline_metrics was captured
    baseline_metrics: Optional[dict] = None  # MetricsSnapshot.to_dict() at creation
    # Temporal tracking for all predictions
    cycle_created: Optional[int] = None  # Cycle when prediction was created
    # Follow-up tracking - prevents same prediction from triggering multiple follow-ups
    triggered_follow_up: bool = False

    # Failure attribution (populated when falsified)
    failure_reason: FailureReason | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Validate confidence range."""
        return max(0.0, min(1.0, v))

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "uid": self.uid,
            "hypothesis_uid": self.hypothesis_uid,
            "claim": self.claim,
            "condition_type": self.condition_type.value,
            "condition_value": self.condition_value,
            "status": self.status.value,
            "confidence": self.confidence,
            "outcome": self.outcome,
            "accuracy_score": self.accuracy_score,
            "earliest_verify_cycle": self.earliest_verify_cycle,
            "expiry_cycle": self.expiry_cycle,
            "verification_cycle": self.verification_cycle,
            "created_at": self.created_at.isoformat(),
            "target_goal_uid": self.target_goal_uid,
            "expected_goal_delta": self.expected_goal_delta,
            "actual_goal_delta": self.actual_goal_delta,
            "goal_snapshot_before": self.goal_snapshot_before,
            "goal_snapshot_after": self.goal_snapshot_after,
            "baseline_cycle": self.baseline_cycle,
            "baseline_metrics": self.baseline_metrics,
            "cycle_created": self.cycle_created,
            "triggered_follow_up": self.triggered_follow_up,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        """Deserialize from dictionary."""
        try:
            created_at = datetime.fromisoformat(data["created_at"])
        except (ValueError, KeyError):
            created_at = datetime.now(timezone.utc)
        return cls(
            uid=data["uid"],
            hypothesis_uid=data["hypothesis_uid"],
            claim=data["claim"],
            condition_type=PredictionConditionType(data["condition_type"]),
            condition_value=data.get("condition_value", ""),
            status=PredictionStatus(data["status"]),
            confidence=data.get("confidence", 0.5),
            outcome=data.get("outcome"),
            accuracy_score=data.get("accuracy_score"),
            earliest_verify_cycle=data.get("earliest_verify_cycle"),
            expiry_cycle=data.get("expiry_cycle"),
            verification_cycle=data.get("verification_cycle"),
            created_at=created_at,
            target_goal_uid=data.get("target_goal_uid"),
            expected_goal_delta=data.get("expected_goal_delta", 0.0),
            actual_goal_delta=data.get("actual_goal_delta"),
            goal_snapshot_before=data.get("goal_snapshot_before"),
            goal_snapshot_after=data.get("goal_snapshot_after"),
            baseline_cycle=data.get("baseline_cycle"),
            baseline_metrics=data.get("baseline_metrics"),
            cycle_created=data.get("cycle_created"),
            triggered_follow_up=data.get("triggered_follow_up", False),
            failure_reason=FailureReason(data["failure_reason"])
            if data.get("failure_reason")
            else None,
        )

    def is_verifiable_at(self, cycle: int) -> bool:
        """Check if prediction can be verified at given cycle.

        Args:
            cycle: Current cognitive cycle

        Returns:
            True if within verification window, False otherwise
        """
        if self.status != PredictionStatus.PENDING:
            return False
        if self.earliest_verify_cycle and cycle < self.earliest_verify_cycle:
            return False
        if self.expiry_cycle and cycle > self.expiry_cycle:
            return False
        return True


class SimulationResult(BaseModel):
    """Complete output from a Graph-Preflexor simulation phase.

    Contains all extracted hypotheses, predictions, graph structures,
    and metadata from the simulation.

    Attributes:
        hypotheses: List of hypotheses extracted from simulation
        predictions: List of testable predictions derived from hypotheses
        experiment_proposals: Auto-generated experiment proposals from hypotheses
        graph_edges: Edge data for Triple creation
        pattern_zettels: Data for InsightZettel creation from patterns
        thinking_trace: Full thinking/brainstorm trace
        synthesis: Final synthesis narrative
        trigger_reason: Why simulation was triggered
        duration_seconds: How long the simulation took
        tokens_generated: Number of tokens generated
    """

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    predictions: list[Prediction] = Field(default_factory=list)
    experiment_proposals: list[dict] = Field(default_factory=list)
    graph_edges: list[dict] = Field(default_factory=list)
    pattern_zettels: list[dict] = Field(default_factory=list)
    thinking_trace: str = ""
    synthesis: str = ""
    trigger_reason: str = ""
    duration_seconds: float = 0.0
    tokens_generated: int = 0

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "predictions": [p.to_dict() for p in self.predictions],
            "experiment_proposals": self.experiment_proposals,
            "graph_edges": self.graph_edges,
            "pattern_zettels": self.pattern_zettels,
            "thinking_trace": self.thinking_trace,
            "synthesis": self.synthesis,
            "trigger_reason": self.trigger_reason,
            "duration_seconds": self.duration_seconds,
            "tokens_generated": self.tokens_generated,
        }

    @property
    def has_hypotheses(self) -> bool:
        """Check if simulation produced any hypotheses."""
        return len(self.hypotheses) > 0

    @property
    def total_predictions(self) -> int:
        """Total number of predictions generated."""
        return len(self.predictions)
