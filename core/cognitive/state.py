"""Cognitive state for Markovian cognitive loop.

This module defines the minimal state needed for continuous cognition
where each thought depends only on the previous thought, external context,
and a self-concept steering vector with surprise amplification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.continuity import ContinuityContext
    from core.cognitive.episode import Episode
    from core.cognitive.opening_tracker import OpeningTracker
    from core.cognitive.telemetry import TelemetrySummary
    from core.steering.hierarchical import HierarchicalSteerer
    from core.steering.evalatis import EvalatisSteerer
    from core.cognitive.goal import InquiryGoal

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Qwen3-8B model dimension
D_MODEL = 4096

# Maximum number of recent concepts to retain for rotation (hard exclusion)
MAX_RECENT_CONCEPTS = 10

# Exponential decay halflife for concept cooldown (in cycles)
# After 20 cycles, a concept is at 50% availability; after 40 cycles, 75%; etc.
CONCEPT_DECAY_HALFLIFE = 20

# Sentinel for "parameter not provided" in with_update
_UNSET = object()


def _default_vector() -> "np.ndarray":
    """Create default zero steering vector."""
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy required for CognitiveState")
    return np.zeros(D_MODEL)


def _default_opening_tracker() -> "OpeningTracker":
    """Create default OpeningTracker for diversity tracking."""
    from core.cognitive.opening_tracker import OpeningTracker
    return OpeningTracker(window_size=5)


def _default_steerer() -> "EvalatisSteerer":
    """Create default EvalatisSteerer with standard zones.

    EvalatisSteerer provides hybrid emergence-selection steering:
    - Emergent vectors evolve continuously from activations
    - Crystals are frozen high-performers that compete for selection
    - Crystallization, spawning, and pruning manage the population
    """
    from core.steering.config import HierarchicalSteeringConfig
    from core.steering.evalatis import EvalatisSteerer
    return EvalatisSteerer(HierarchicalSteeringConfig(), d_model=D_MODEL)


def _hierarchical_steerer() -> "HierarchicalSteerer":
    """Create HierarchicalSteerer (legacy fallback)."""
    from core.steering.config import HierarchicalSteeringConfig
    from core.steering.hierarchical import HierarchicalSteerer
    return HierarchicalSteerer(HierarchicalSteeringConfig(), d_model=D_MODEL)



@dataclass
class ReflectionEntry:
    """A single verbal reflection on thinking quality.

    Inspired by Reflexion (Shinn et al. 2023), these entries capture
    linguistic feedback about reasoning quality that persists across
    cognitive cycles, enabling learning from mistakes without weight updates.

    Attributes:
        cycle: Cognitive cycle when reflection was generated
        reflection_text: What went wrong / what worked / what to improve
        trigger: What triggered this reflection (e.g., "low_faithfulness",
            "prediction_failure", "repetition", "insight_quality")
        severity: How significant the observation is ("minor", "notable", "critical")
        created_at: When the reflection was created
    """

    cycle: int
    reflection_text: str
    trigger: str
    severity: str  # "minor", "notable", "critical"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReflectionBuffer:
    """Accumulates verbal reflections across cognitive cycles.

    Implements Reflexion-style verbal reinforcement learning where
    linguistic feedback persists across episodes, enabling the system
    to learn from mistakes without parameter updates.

    Key triggers that generate reflections:
    - Faithfulness divergence: SAE features don't match verbal claims
    - Prediction failures: Simulation hypotheses were disconfirmed
    - Repetitive patterns: Semantic collapse or opening repetition detected
    - Low-quality insights: Curator assessment indicates shallow thinking

    The buffer maintains a rolling window of recent reflections that are
    injected into generation prompts to guide future reasoning.

    Attributes:
        entries: List of reflection entries, newest last
        max_entries: Maximum entries to retain (rolling window)
    """

    entries: list[ReflectionEntry] = field(default_factory=list)
    max_entries: int = 20

    def add(
        self,
        cycle: int,
        text: str,
        trigger: str,
        severity: str = "minor",
    ) -> None:
        """Add a reflection, maintaining rolling window.

        Args:
            cycle: Cognitive cycle number
            text: The reflection text
            trigger: What triggered this reflection
            severity: How significant ("minor", "notable", "critical")
        """
        self.entries.append(
            ReflectionEntry(
                cycle=cycle,
                reflection_text=text,
                trigger=trigger,
                severity=severity,
            )
        )
        # Keep only recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def get_recent(self, n: int = 5) -> list[ReflectionEntry]:
        """Get n most recent reflections.

        Args:
            n: Number of recent reflections to return

        Returns:
            List of most recent ReflectionEntry objects
        """
        return self.entries[-n:]

    def get_by_trigger(self, trigger: str) -> list[ReflectionEntry]:
        """Get reflections by trigger type.

        Args:
            trigger: The trigger type to filter by

        Returns:
            List of ReflectionEntry objects with matching trigger
        """
        return [e for e in self.entries if e.trigger == trigger]

    def get_by_severity(self, severity: str) -> list[ReflectionEntry]:
        """Get reflections by severity level.

        Args:
            severity: The severity level to filter by

        Returns:
            List of ReflectionEntry objects with matching severity
        """
        return [e for e in self.entries if e.severity == severity]

    def format_for_prompt(self, max_reflections: int = 3) -> str:
        """Format recent reflections for inclusion in generation prompt.

        Prioritizes notable/critical reflections over minor ones.

        Args:
            max_reflections: Maximum reflections to include

        Returns:
            Formatted string for prompt injection, or empty string if no reflections
        """
        if not self.entries:
            return ""

        # Prioritize by severity: critical > notable > minor
        severity_order = {"critical": 0, "notable": 1, "minor": 2}
        sorted_entries = sorted(
            self.entries[-10:],  # Consider last 10
            key=lambda e: (severity_order.get(e.severity, 2), -e.cycle),
        )

        selected = sorted_entries[:max_reflections]
        if not selected:
            return ""

        lines = ["<previous_reflections>"]
        lines.append("Recent observations about my thinking patterns:")
        for entry in selected:
            severity_marker = "⚠️" if entry.severity == "critical" else "•"
            lines.append(f"  {severity_marker} [{entry.trigger}] {entry.reflection_text}")
        lines.append("</previous_reflections>")

        return "\n".join(lines)

    def has_recent_critical(self, within_cycles: int = 5, current_cycle: int = 0) -> bool:
        """Check if there are critical reflections within recent cycles.

        Args:
            within_cycles: How many cycles back to check
            current_cycle: Current cycle number

        Returns:
            True if critical reflection exists within window
        """
        cutoff = current_cycle - within_cycles
        return any(
            e.severity == "critical" and e.cycle >= cutoff
            for e in self.entries
        )

    def clear(self) -> None:
        """Clear all reflections (e.g., on episode boundary)."""
        self.entries.clear()


def _default_reflection_buffer() -> ReflectionBuffer:
    """Create default empty ReflectionBuffer."""
    return ReflectionBuffer()


def _default_continuity_context() -> "ContinuityContext":
    """Create default ContinuityContext for cycle continuity tracking."""
    from core.cognitive.continuity import ContinuityContext
    return ContinuityContext()

@dataclass
class CognitiveState:
    """Minimal state for Markovian cognition with insight-driven momentum.

    Each cognitive cycle depends only on:
    - Previous thought (what Lilly was thinking)
    - Current insight (what was discovered/realized)
    - Current question (what curiosity emerged to drive next cycle)
    - Hierarchical steerer (multi-zone steering vectors)
    - Baseline vector (for surprise calculation)
    - Recent concepts (to avoid repetition)

    The insight/question pair creates directional momentum:
    - Insight grounds what was learned
    - Question drives exploration toward new territory

    The hierarchical steerer manages steering vectors across multiple layer zones:
    - Exploration zone (early layers): behavioral patterns
    - Concept zone (middle layers): topic/concept focus
    - Identity zone (later layers): self-expression

    Attributes:
        thought: The previous generated thought content
        current_insight: Key insight or realization from the thought
        current_question: Open question that emerged, driving next exploration
        vector: Legacy self-concept steering vector (for backward compatibility)
        baseline: Baseline activations for surprise calculation
        steerer: Hierarchical steerer for multi-zone steering
        cycle_count: Number of cycles completed (for diagnostics)
        recent_concepts: Recently explored concepts (for rotation)
        last_cycle_at: Timestamp of last completed cycle
        consecutive_low_surprise: Counter for detecting stale exploration (triggers perturbation)
        opening_tracker: Tracks opening phrases for diversity enforcement
        sae_features: SAE monosemantic features from transcoder
        current_goal: Active inquiry goal being pursued through dialectical stages
    """

    thought: str = ""
    current_insight: str = ""
    current_question: str = ""
    vector: "np.ndarray" = field(default_factory=_default_vector)
    baseline: "np.ndarray" = field(default_factory=_default_vector)
    steerer: "HierarchicalSteerer | EvalatisSteerer" = field(default_factory=_default_steerer)
    cycle_count: int = 0
    recent_concepts: list[str] = field(default_factory=list)
    last_concept: str = ""  # Most recent concept for next cycle's prompt
    curated_prompt: str = ""  # Curator-crafted prompt for next generation cycle
    last_cycle_at: Optional[datetime] = None
    active_concepts: list[tuple[str, float]] = field(default_factory=list)
    bridge_candidates: list[str] = field(default_factory=list)
    opening_tracker: "OpeningTracker" = field(default_factory=_default_opening_tracker)
    consecutive_low_surprise: int = 0  # Tracks staleness for exploration boost
    cumulative_low_surprise_in_stage: int = 0  # Total low-surprise cycles in current stage (not reset by high-surprise)
    repetition_count_in_stage: int = 0  # Number of repetitions in current stage
    sae_features: list[tuple[int, float]] = field(default_factory=list)  # SAE monosemantic features
    current_goal: Optional["InquiryGoal"] = None  # Active inquiry goal for progressive thinking
    concept_usage_history: dict[str, int] = field(default_factory=dict)  # concept -> cycle_count when last used
    recent_zettel_uids: list[str] = field(default_factory=list)  # For EMERGED_FROM lineage tracking
    retrieved_question_uids: list[str] = field(default_factory=list)  # For ADDRESSED_BY tracking
    current_episode: Optional["Episode"] = None  # Active episode for episode-based cognition

    # Faithfulness tracking (Walden framework)
    # Each entry is (overlap_ratio, severity) from FaithfulnessScore
    faithfulness_history: list[tuple[float, str]] = field(default_factory=list)

    # Simulation awareness (Graph-Preflexor phase)
    active_hypotheses: list[str] = field(default_factory=list)  # UIDs of hypotheses being tested
    pending_prediction_count: int = 0  # Number of predictions awaiting verification
    recent_prediction_outcomes: list[tuple[str, bool]] = field(default_factory=list)  # (claim_preview, verified)
    queued_follow_up_simulations: list[str] = field(default_factory=list)  # Hypothesis UIDs needing follow-up

    # Per-cycle graph metrics for computable verification
    # Stores MetricsSnapshot.to_dict() for hypothesis predictions
    metrics_snapshot: Optional[dict] = None

    # Reflexion buffer (verbal reinforcement learning)
    # Accumulates linguistic feedback across cycles to guide future reasoning
    reflection_buffer: ReflectionBuffer = field(default_factory=_default_reflection_buffer)

    # Continuity context (cycle recap and narrative thread)
    # Compresses recent cycles for prompt injection and narrative continuity
    continuity_context: "ContinuityContext" = field(default_factory=_default_continuity_context)

    # Reflexion tracking (ReflexionPhase integration)
    recent_reflexion_entries: list[str] = field(default_factory=list)  # UIDs of recent reflexion entries
    last_cycle_catastrophic: bool = False  # Skip reflexion if previous cycle failed badly

    # HALT probe epistemic confidence (arXiv:2601.14210)
    # Probability of reliable answer from intermediate layer probe
    epistemic_confidence: float = 0.5  # Default neutral (untrained probe)
    halt_probe_latency_ms: float = 0.0  # Probe execution time for monitoring

    # Existential drive tracking
    # Tracks when existential questions were last injected and which have been asked
    existential_prompt_last_cycle: int = 0  # Cycle when last existential prompt was injected
    existential_questions_asked: list[str] = field(default_factory=list)  # Questions asked to avoid repetition

    # Biofeedback telemetry (v0: logit dynamics, residual slopes)
    # Captured during generation phase, evaluated by Reflexion for health signals
    telemetry_summary: Optional["TelemetrySummary"] = None

    @property
    def average_faithfulness(self) -> float:
        """Average faithfulness overlap ratio across all tracked cycles."""
        if not self.faithfulness_history:
            return 1.0
        return sum(f[0] for f in self.faithfulness_history) / len(self.faithfulness_history)

    @property
    def max_divergence_severity(self) -> str:
        """Worst divergence severity observed."""
        severity_order = ["none", "low", "moderate", "high"]
        if not self.faithfulness_history:
            return "none"
        worst = max(self.faithfulness_history, key=lambda f: severity_order.index(f[1]))
        return worst[1]

    def __post_init__(self):
        """Validate state after initialization."""
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy required for CognitiveState")

        # Ensure vectors are numpy arrays
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.array(self.vector)
        if not isinstance(self.baseline, np.ndarray):
            self.baseline = np.array(self.baseline)

    def with_update(
        self,
        thought: str,
        vector: "np.ndarray",
        baseline: Optional["np.ndarray"] = None,
        concept: Optional[str] = None,
        insight: Optional[str] = None,
        question: Optional[str] = None,
        active_concepts: Optional[list[tuple[str, float]]] = None,
        bridge_candidates: Optional[list[str]] = None,
        consecutive_low_surprise: Optional[int] = None,
        cumulative_low_surprise_in_stage: Optional[int] = None,
        repetition_count_in_stage: Optional[int] = None,
        sae_features: Optional[list[tuple[int, float]]] = None,
        steerer: Optional["HierarchicalSteerer"] = None,
        current_goal: "Optional[InquiryGoal] | object" = _UNSET,
        recent_zettel_uids: Optional[list[str]] = None,
        retrieved_question_uids: Optional[list[str]] = None,
        current_episode: "Optional[Episode] | object" = _UNSET,
        curated_prompt: Optional[str] = None,
        faithfulness_history: Optional[list[tuple[float, str]]] = None,
        active_hypotheses: Optional[list[str]] = None,
        pending_prediction_count: Optional[int] = None,
        recent_prediction_outcomes: Optional[list[tuple[str, bool]]] = None,
        queued_follow_up_simulations: Optional[list[str]] = None,
        recent_reflexion_entries: Optional[list[str]] = None,
        last_cycle_catastrophic: Optional[bool] = None,
        epistemic_confidence: Optional[float] = None,
        halt_probe_latency_ms: Optional[float] = None,
        existential_prompt_last_cycle: Optional[int] = None,
        existential_questions_asked: Optional[list[str]] = None,
        metrics_snapshot: Optional[dict] = None,
        telemetry_summary: "Optional[TelemetrySummary]" = None,
    ) -> "CognitiveState":
        """Create new state with updated values.

        Returns a new CognitiveState rather than mutating in place,
        making state transitions explicit and traceable.

        Args:
            thought: New thought content
            vector: Updated steering vector (legacy, for backward compatibility)
            baseline: Updated baseline vector (optional, keeps previous if None)
            concept: Concept explored this cycle (optional)
            insight: Key insight extracted from thought (optional)
            question: Open question extracted from thought (optional)
            active_concepts: List of (concept, salience) tuples from extraction (optional)
            bridge_candidates: List of potential bridge concepts for exploration (optional)
            consecutive_low_surprise: Counter for staleness detection (optional, keeps previous if None)
            cumulative_low_surprise_in_stage: Cumulative low-surprise cycles in current stage
                (does not reset on high-surprise, only on stage transition)
            repetition_count_in_stage: Number of repetitions detected in current stage
            sae_features: List of (feature_index, activation) tuples from SAE transcoder (optional)
            steerer: Hierarchical steerer (optional, keeps previous if None)
            current_goal: Active inquiry goal (optional, uses _UNSET sentinel to distinguish
                "not provided" from "explicitly None" - pass None to clear goal)
            recent_zettel_uids: UIDs of recently created InsightZettels for EMERGED_FROM tracking
            retrieved_question_uids: UIDs of retrieved open questions for ADDRESSED_BY tracking
            current_episode: Active episode for episode-based cognition (uses _UNSET sentinel to distinguish
                "not provided" from "explicitly None" - pass None to end episode)
            faithfulness_history: List of (overlap_ratio, severity) tuples tracking faithfulness
                across cognitive cycles (Walden framework)
            active_hypotheses: UIDs of hypotheses being tested (simulation awareness)
            pending_prediction_count: Number of predictions awaiting verification
            recent_prediction_outcomes: List of (claim_preview, verified) tuples
            queued_follow_up_simulations: Hypothesis UIDs needing follow-up simulation
            recent_reflexion_entries: UIDs of recent reflexion entries (ReflexionPhase)
            last_cycle_catastrophic: Whether previous cycle failed badly (skip reflexion)
            epistemic_confidence: HALT probe confidence score in [0, 1] (arXiv:2601.14210)
            halt_probe_latency_ms: HALT probe execution time for monitoring
            existential_prompt_last_cycle: Cycle when last existential prompt was injected
            existential_questions_asked: List of existential questions asked (for diversity)
            metrics_snapshot: Cycle metrics for prediction verification (RS-404 fix)
            telemetry_summary: Biofeedback telemetry from generation (logit dynamics, residual slopes)

        Returns:
            New CognitiveState with updated values
        """
        recent = self.recent_concepts[-(MAX_RECENT_CONCEPTS - 1):] + ([concept] if concept else [])

        # Update concept usage history for exponential decay
        updated_concept_history = self.concept_usage_history.copy()
        if concept:
            updated_concept_history[concept] = self.cycle_count + 1  # Record when used

        return CognitiveState(
            thought=thought,
            current_insight=insight or "",
            current_question=question or "",
            vector=vector,
            baseline=baseline if baseline is not None else self.baseline,
            steerer=steerer if steerer is not None else self.steerer,
            cycle_count=self.cycle_count + 1,
            recent_concepts=recent,
            last_concept=concept if concept else self.last_concept,
            curated_prompt=curated_prompt if curated_prompt is not None else "",
            last_cycle_at=datetime.now(timezone.utc),
            active_concepts=active_concepts if active_concepts is not None else [],
            bridge_candidates=bridge_candidates if bridge_candidates is not None else [],
            opening_tracker=self.opening_tracker,  # Preserve tracker across updates
            consecutive_low_surprise=consecutive_low_surprise if consecutive_low_surprise is not None else self.consecutive_low_surprise,
            cumulative_low_surprise_in_stage=cumulative_low_surprise_in_stage if cumulative_low_surprise_in_stage is not None else self.cumulative_low_surprise_in_stage,
            repetition_count_in_stage=repetition_count_in_stage if repetition_count_in_stage is not None else self.repetition_count_in_stage,
            sae_features=sae_features if sae_features is not None else [],
            current_goal=current_goal if current_goal is not _UNSET else self.current_goal,
            concept_usage_history=updated_concept_history,
            recent_zettel_uids=recent_zettel_uids if recent_zettel_uids is not None else [],
            retrieved_question_uids=retrieved_question_uids if retrieved_question_uids is not None else [],
            current_episode=current_episode if current_episode is not _UNSET else self.current_episode,
            faithfulness_history=faithfulness_history if faithfulness_history is not None else self.faithfulness_history,
            active_hypotheses=active_hypotheses if active_hypotheses is not None else self.active_hypotheses,
            pending_prediction_count=pending_prediction_count if pending_prediction_count is not None else self.pending_prediction_count,
            recent_prediction_outcomes=recent_prediction_outcomes if recent_prediction_outcomes is not None else self.recent_prediction_outcomes,
            queued_follow_up_simulations=queued_follow_up_simulations if queued_follow_up_simulations is not None else self.queued_follow_up_simulations,
            reflection_buffer=self.reflection_buffer,  # Preserve buffer across updates
            continuity_context=self.continuity_context,  # Preserve continuity across updates
            metrics_snapshot=metrics_snapshot if metrics_snapshot is not None else self.metrics_snapshot,
            recent_reflexion_entries=recent_reflexion_entries if recent_reflexion_entries is not None else self.recent_reflexion_entries,
            last_cycle_catastrophic=last_cycle_catastrophic if last_cycle_catastrophic is not None else self.last_cycle_catastrophic,
            epistemic_confidence=epistemic_confidence if epistemic_confidence is not None else self.epistemic_confidence,
            halt_probe_latency_ms=halt_probe_latency_ms if halt_probe_latency_ms is not None else self.halt_probe_latency_ms,
            existential_prompt_last_cycle=existential_prompt_last_cycle if existential_prompt_last_cycle is not None else self.existential_prompt_last_cycle,
            existential_questions_asked=existential_questions_asked if existential_questions_asked is not None else self.existential_questions_asked,
            telemetry_summary=telemetry_summary,  # Pass through (not preserved across cycles - each generation captures fresh)
        )

    def vector_magnitude(self) -> float:
        """Get the L2 norm of the steering vector."""
        return float(np.linalg.norm(self.vector))

    def get_concept_availability(self, concept: str) -> float:
        """Get availability weight for a concept based on exponential decay.

        Implements the "20-cycle halflife" cooldown from the semantic collapse fix.
        Recently used concepts have low availability; concepts not used recently
        approach 1.0.

        Args:
            concept: The concept to check

        Returns:
            Float between 0.0 (just used) and 1.0 (fully available)
        """
        if concept not in self.concept_usage_history:
            return 1.0  # Never used = fully available

        cycles_since_use = self.cycle_count - self.concept_usage_history[concept]
        if cycles_since_use <= 0:
            return 0.0  # Currently being used

        # Exponential decay: availability = 1 - 0.5^(cycles/halflife)
        # At halflife cycles, availability = 0.5
        # At 2*halflife cycles, availability = 0.75
        # At 3*halflife cycles, availability = 0.875
        decay_factor = 0.5 ** (cycles_since_use / CONCEPT_DECAY_HALFLIFE)
        return 1.0 - decay_factor

    @property
    def in_episode(self) -> bool:
        """Check if an episode is currently active.

        Returns:
            True if there is an active episode, False otherwise
        """
        return self.current_episode is not None

    def episode_duration(self, now: Optional[datetime] = None) -> float:
        """Get the duration of the current episode in seconds.

        Args:
            now: Current time (defaults to now)

        Returns:
            Duration in seconds, or 0.0 if no episode is active
        """
        if self.current_episode is None:
            return 0.0
        if now is None:
            now = datetime.now(timezone.utc)
        return (now - self.current_episode.started_at).total_seconds()

    def episode_segment_count(self) -> int:
        """Get the number of segments completed in the current episode.

        Returns:
            Number of completed segments, or 0 if no episode is active
        """
        if self.current_episode is None:
            return 0
        return len(self.current_episode.segments_completed)
