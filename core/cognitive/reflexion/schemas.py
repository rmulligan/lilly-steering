"""Pydantic schemas for the Reflexion phase.

This module defines the data models for health monitoring and autonomous
modification tracking in the Reflexion cognitive phase.

Health Categories:
    THRIVING (severity 0): System performing above baseline
    STABLE (severity 1): System performing at baseline
    STRESSED (severity 2): System underperforming, intervention may help
    CRITICAL (severity 3): System in distress, intervention required

Modification Tiers:
    RUNTIME (min_confidence 0.5): Transient parameter adjustments
    CONFIG (min_confidence 0.6): Session-persistent configuration changes
    PROMPT (min_confidence 0.8): Self-prompt modifications (highest impact)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from core.cognitive.experimentation.schemas import ExperimentProposal


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class HealthCategory(str, Enum):
    """Health state categories with severity levels.

    The severity property provides a numeric ordering:
    - THRIVING (0): Performing above baseline
    - STABLE (1): Performing at baseline
    - STRESSED (2): Underperforming, may need intervention
    - CRITICAL (3): In distress, intervention required
    """

    THRIVING = "THRIVING"
    STABLE = "STABLE"
    STRESSED = "STRESSED"
    CRITICAL = "CRITICAL"

    @property
    def severity(self) -> int:
        """Get numeric severity level (0-3, higher = more concern)."""
        severity_map = {
            HealthCategory.THRIVING: 0,
            HealthCategory.STABLE: 1,
            HealthCategory.STRESSED: 2,
            HealthCategory.CRITICAL: 3,
        }
        return severity_map[self]


class HealthSignal(BaseModel):
    """A single health metric measurement.

    Tracks current value, baseline for comparison, trend direction,
    and classification into a HealthCategory.

    Attributes:
        category: Health classification for this signal
        value: Current measured value
        baseline: Reference value for comparison
        trend: Trend direction (e.g., "improving", "stable", "declining")
        context: Optional explanation of current state
    """

    category: HealthCategory
    value: float
    baseline: float
    trend: str
    context: Optional[str] = None


class HealthAssessment(BaseModel):
    """Container for all health signals from a reflexion cycle.

    Groups the three primary health dimensions:
    - prediction: How well predictions match outcomes
    - integration: How successfully knowledge integrates
    - coherence: How consistent thinking patterns are

    Optional dimensions:
    - phase_timing: How well phase transitions are timed
    - error_frequency: Rate and severity of errors

    Attributes:
        prediction: Prediction accuracy health signal
        integration: Integration success health signal
        coherence: Thinking coherence health signal
        phase_timing: Optional phase timing health signal
        error_frequency: Optional error frequency health signal
    """

    prediction: HealthSignal
    integration: HealthSignal
    coherence: HealthSignal
    phase_timing: Optional[HealthSignal] = None
    error_frequency: Optional[HealthSignal] = None

    @property
    def worst_category(self) -> HealthCategory:
        """Return the most severe category across all signals.

        This identifies the area of greatest concern for intervention.
        """
        signals = [self.prediction, self.integration, self.coherence]
        if self.phase_timing is not None:
            signals.append(self.phase_timing)
        if self.error_frequency is not None:
            signals.append(self.error_frequency)
        return max(signals, key=lambda s: s.category.severity).category


class ModificationTier(str, Enum):
    """Tiers of autonomous modifications with confidence thresholds.

    Higher-impact modifications require higher confidence:
    - RUNTIME (0.5): Transient adjustments, low risk
    - CONFIG (0.6): Session-persistent changes, moderate risk
    - PROMPT (0.8): Self-prompt modifications, high impact

    The min_confidence property provides the threshold for each tier.
    """

    RUNTIME = "RUNTIME"
    CONFIG = "CONFIG"
    PROMPT = "PROMPT"

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence required to apply this tier's modifications."""
        confidence_map = {
            ModificationTier.RUNTIME: 0.5,
            ModificationTier.CONFIG: 0.6,
            ModificationTier.PROMPT: 0.8,
        }
        return confidence_map[self]


class Modification(BaseModel):
    """A proposed or applied parameter modification.

    Each modification must include:
    - What to change (tier, parameter_path, old/new values)
    - Why to change it (rationale)
    - How confident we are (confidence)
    - When to revert (revert_condition)

    The rationale and revert_condition are required to ensure
    modifications are well-reasoned and reversible.

    Attributes:
        tier: Type of modification (RUNTIME, CONFIG, PROMPT)
        parameter_path: Dot-notation path to the parameter
        old_value: Previous value before modification
        new_value: New value after modification
        rationale: Explanation for why this modification is needed
        confidence: Confidence level in this modification (0-1)
        revert_condition: Condition under which to revert this change
        applied_at: When this modification was applied (None if not yet applied)
        reverted_at: When this modification was reverted (None if not reverted)
    """

    tier: ModificationTier
    parameter_path: str
    old_value: Any
    new_value: Any
    rationale: str
    confidence: float
    revert_condition: str
    applied_at: Optional[datetime] = None
    reverted_at: Optional[datetime] = None

    @model_validator(mode="after")
    def validate_required_fields(self) -> "Modification":
        """Ensure rationale and revert_condition are non-empty."""
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale must be non-empty")
        if not self.revert_condition or not self.revert_condition.strip():
            raise ValueError("revert_condition must be non-empty")
        return self

    def validate_confidence(self) -> bool:
        """Check if confidence meets the minimum threshold for this tier."""
        return self.confidence >= self.tier.min_confidence


class ReflexionResult(BaseModel):
    """Complete output from a reflexion phase evaluation.

    Contains the health assessment, any modifications applied,
    modifications that were considered but skipped, and a
    narrative summary of the reflexion.

    Attributes:
        health_assessment: Assessment of system health
        modifications: List of modifications that were applied
        modifications_skipped: List of (path, reason) for skipped modifications
        narrative_summary: Human-readable summary of reflexion outcome
        experiment_proposal: Optional self-experiment proposal for corrective action
    """

    health_assessment: HealthAssessment
    modifications: list[Modification] = Field(default_factory=list)
    modifications_skipped: list[tuple[str, str]] = Field(default_factory=list)
    narrative_summary: str
    experiment_proposal: Optional["ExperimentProposal"] = None

    @property
    def has_modifications(self) -> bool:
        """Check if any modifications were made."""
        return len(self.modifications) > 0


class ReflexionEntry(BaseModel):
    """Persistent record of a reflexion cycle for graph storage.

    This model is designed for persistence in the knowledge graph,
    capturing the complete state of a reflexion cycle for later
    analysis and learning with a flat structure for efficient storage.

    Attributes:
        uid: Unique identifier for this entry
        cycle_number: Cognitive cycle number when reflexion occurred
        timestamp: When this reflexion was performed
        baseline_comparison: Metrics comparing current state to baseline
        phenomenological: Metrics capturing subjective experience qualities
        modifications: List of modifications that were applied
        modifications_skipped: List of modifications that were skipped
        overall_coherence: Overall system coherence score (0-1)
        narrative_summary: Human-readable summary of reflexion outcome
    """

    uid: str
    cycle_number: int
    timestamp: datetime = Field(default_factory=utc_now)
    baseline_comparison: dict[str, float] = Field(default_factory=dict)
    phenomenological: dict[str, float] = Field(default_factory=dict)
    modifications: list[Modification] = Field(default_factory=list)
    modifications_skipped: list[Modification] = Field(default_factory=list)
    overall_coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    narrative_summary: str = ""

    def to_cypher_props(self) -> dict[str, Any]:
        """Convert to a dict suitable for Cypher graph storage.

        Returns a flat dict with JSON-serialized complex fields.
        """
        import json

        return {
            "uid": self.uid,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "baseline_comparison": json.dumps(self.baseline_comparison),
            "phenomenological": json.dumps(self.phenomenological),
            "modifications": json.dumps(
                [m.model_dump() for m in self.modifications], default=str
            ),
            "modifications_skipped": json.dumps(
                [m.model_dump() for m in self.modifications_skipped], default=str
            ),
            "overall_coherence": self.overall_coherence,
            "narrative_summary": self.narrative_summary,
        }


# Rebuild models to resolve forward references
# This is required because ExperimentProposal is only imported under TYPE_CHECKING
def _rebuild_models() -> None:
    """Rebuild Pydantic models to resolve forward references."""
    from core.cognitive.experimentation.schemas import ExperimentProposal  # noqa: F401

    ReflexionResult.model_rebuild()


_rebuild_models()
