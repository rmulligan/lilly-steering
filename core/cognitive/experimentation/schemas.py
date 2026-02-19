"""Schemas for the self-experimentation framework.

This module defines the data structures for experiment proposals,
measurements, and the allowed parameter domains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from core.cognitive.simulation.schemas import MetricsSnapshot


class ExperimentDomain(str, Enum):
    """Domains where Lilly can run experiments."""

    STEERING = "steering"
    EPISODE = "episode"
    EMOTIONAL_FIELD = "emotional_field"
    SIMULATION = "simulation"
    TOOL_PATTERN = "tool_pattern"


class ExperimentPhase(str, Enum):
    """Phases of an experiment lifecycle."""

    PENDING = "pending"
    BASELINE = "baseline"
    TREATMENT = "treatment"
    WASHOUT = "washout"
    COMPLETE = "complete"
    ABORTED = "aborted"


# REMOVED: ALLOWED_PARAMETERS whitelist
# Phase 1 Full Operational Autonomy: Lilly designs experiments with any parameters
# she judges appropriate based on her knowledge of parameter effects.
#
# Previously restricted parameters per domain (bounded autonomy):
# - STEERING: exploration.magnitude, concept.magnitude, identity.magnitude, ema_alpha
# - EPISODE: max_segments, min_segments, deep_dive_probability
# - EMOTIONAL_FIELD: decay_rate, diffusion_rate, blend_weight
# - SIMULATION: trigger_confidence, max_hypotheses, max_predictions_per_hypothesis
# - TOOL_PATTERN: graph_exploration_weight, zettel_retrieval_weight, belief_query_weight
#
# These examples remain useful for documentation but are no longer enforced.

# Experiment constraints
MAX_EXPERIMENT_DURATION = 50  # cycles
DOMAIN_COOLDOWN = 20  # cycles between experiments on same domain


class ExperimentProposal(BaseModel):
    """Proposal for a self-experiment.

    Created by curator (exploratory) or reflexion (corrective) phases.
    """

    domain: ExperimentDomain
    parameter_path: str
    treatment_value: float
    rationale: str
    target_metric: str
    expected_direction: str = "increase"  # "increase" or "decrease"
    min_effect_size: float = 0.1
    rollback_trigger: float | None = None
    baseline_cycles: int = 5
    treatment_cycles: int = 10
    washout_cycles: int = 3

    def is_valid_parameter(self) -> bool:
        """Check if parameter is valid.

        Phase 1 Full Operational Autonomy: All parameters are valid.
        Lilly makes judgments about appropriate parameters based on her
        knowledge of system architecture and past experiment outcomes.

        Returns:
            Always True (no restrictions)
        """
        # REMOVED: Whitelist validation
        # Lilly judges parameter appropriateness autonomously
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "domain": self.domain.value,
            "parameter_path": self.parameter_path,
            "treatment_value": self.treatment_value,
            "rationale": self.rationale,
            "target_metric": self.target_metric,
            "expected_direction": self.expected_direction,
            "min_effect_size": self.min_effect_size,
            "rollback_trigger": self.rollback_trigger,
            "baseline_cycles": self.baseline_cycles,
            "treatment_cycles": self.treatment_cycles,
            "washout_cycles": self.washout_cycles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentProposal":
        """Deserialize from dictionary."""
        return cls(
            domain=ExperimentDomain(data["domain"]),
            parameter_path=data["parameter_path"],
            treatment_value=data["treatment_value"],
            rationale=data["rationale"],
            target_metric=data["target_metric"],
            expected_direction=data.get("expected_direction", "increase"),
            min_effect_size=data.get("min_effect_size", 0.1),
            rollback_trigger=data.get("rollback_trigger"),
            baseline_cycles=data.get("baseline_cycles", 5),
            treatment_cycles=data.get("treatment_cycles", 10),
            washout_cycles=data.get("washout_cycles", 3),
        )


class ExperimentMeasurement(BaseModel):
    """Single measurement during an experiment."""

    experiment_uid: str
    cycle: int
    phase: ExperimentPhase
    snapshot: MetricsSnapshot
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"arbitrary_types_allowed": True}
