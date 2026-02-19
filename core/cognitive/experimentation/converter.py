"""Converter for simulation hypotheses to experiment proposals.

Analyzes hypothesis statements to detect references to adjustable parameters
and converts them into formal ExperimentProposal objects.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

from core.cognitive.experimentation.schemas import (
    ExperimentDomain,
    ExperimentProposal,
)

if TYPE_CHECKING:
    from core.cognitive.simulation.schemas import Hypothesis

logger = logging.getLogger(__name__)

# ============================================================================
# Treatment Value Constants
# ============================================================================
# These constants control how treatment values are computed when converting
# hypotheses to experiments. They determine adjustment multipliers, bounds,
# and default values for different parameter types.

# Multipliers for computing treatment values from current values
INCREASE_MULTIPLIER = 1.2  # 20% increase for general parameters
DECREASE_MULTIPLIER = 0.8  # 20% decrease for general parameters
SEGMENT_INCREASE_MULTIPLIER = 1.25  # 25% increase for segment counts
SEGMENT_DECREASE_MULTIPLIER = 0.75  # 25% decrease for segment counts

# Bounds for normalized parameters (magnitudes, weights)
NORMALIZED_PARAM_MAX = 1.0  # Maximum value for normalized parameters
NORMALIZED_PARAM_MIN = 0.1  # Minimum floor for normalized parameters
SEGMENT_MIN_VALUE = 3  # Minimum floor for segment counts

# Default values for magnitude/weight parameters when no current value exists
DEFAULT_MAGNITUDE_DECREASE = 0.7
DEFAULT_MAGNITUDE_INCREASE = 0.9

# Default values for segment parameters
DEFAULT_SEGMENTS_DECREASE = 8
DEFAULT_SEGMENTS_INCREASE = 15

# Default values for probability/rate parameters
DEFAULT_PROBABILITY_DECREASE = 0.3
DEFAULT_PROBABILITY_INCREASE = 0.7

# Default values for confidence parameters
DEFAULT_CONFIDENCE_DECREASE = 0.5
DEFAULT_CONFIDENCE_INCREASE = 0.8

# Default values for hypothesis count parameters
DEFAULT_HYPOTHESES_DECREASE = 2
DEFAULT_HYPOTHESES_INCREASE = 4

# ============================================================================
# Parameter Detection Patterns
# ============================================================================

# Patterns for detecting parameter references in hypothesis statements
PARAMETER_PATTERNS: dict[str, tuple[ExperimentDomain, str]] = {
    "exploration magnitude": (ExperimentDomain.STEERING, "steering.exploration.magnitude"),
    "exploration": (ExperimentDomain.STEERING, "steering.exploration.magnitude"),
    "concept magnitude": (ExperimentDomain.STEERING, "steering.concept.magnitude"),
    "identity magnitude": (ExperimentDomain.STEERING, "steering.identity.magnitude"),
    "identity steering": (ExperimentDomain.STEERING, "steering.identity.magnitude"),
    "ema alpha": (ExperimentDomain.STEERING, "steering.exploration.ema_alpha"),
    "episode length": (ExperimentDomain.EPISODE, "episode.max_segments"),
    "episode duration": (ExperimentDomain.EPISODE, "episode.max_segments"),
    "max segments": (ExperimentDomain.EPISODE, "episode.max_segments"),
    "min segments": (ExperimentDomain.EPISODE, "episode.min_segments"),
    "deep dive": (ExperimentDomain.EPISODE, "episode.deep_dive_probability"),
    "decay rate": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.decay_rate"),
    "emotional decay": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.decay_rate"),
    "diffusion rate": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.diffusion_rate"),
    "emotional diffusion": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.diffusion_rate"),
    "blend weight": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.blend_weight"),
    "emotional blend": (ExperimentDomain.EMOTIONAL_FIELD, "emotional_field.blend_weight"),
    "simulation trigger": (ExperimentDomain.SIMULATION, "simulation.trigger_confidence"),
    "trigger confidence": (ExperimentDomain.SIMULATION, "simulation.trigger_confidence"),
    "max hypotheses": (ExperimentDomain.SIMULATION, "simulation.max_hypotheses"),
    "hypothesis count": (ExperimentDomain.SIMULATION, "simulation.max_hypotheses"),
    "predictions per hypothesis": (
        ExperimentDomain.SIMULATION,
        "simulation.max_predictions_per_hypothesis",
    ),
    "graph exploration": (ExperimentDomain.TOOL_PATTERN, "tool_pattern.graph_exploration_weight"),
    "zettel retrieval": (ExperimentDomain.TOOL_PATTERN, "tool_pattern.zettel_retrieval_weight"),
    "belief query": (ExperimentDomain.TOOL_PATTERN, "tool_pattern.belief_query_weight"),
}

# Direction keywords - include verb forms (e.g., "increasing", "increased")
# Note: "improv" is intentionally excluded from increase as it's ambiguous
# ("reduce X to improve Y" - the direction is decrease, not increase)
INCREASE_KEYWORDS = [
    "increas",  # Matches: increase, increasing, increased
    "higher",
    "more",
    "rais",  # Matches: raise, raising, raised
    "boost",
    "amplif",  # Matches: amplify, amplifying, amplified
    "strengthen",
    "expand",
    "extend",
    "longer",
    "greater",
]
DECREASE_KEYWORDS = [
    "decreas",  # Matches: decrease, decreasing, decreased
    "lower",
    "less",
    "reduc",  # Matches: reduce, reducing, reduced
    "diminish",
    "weaken",
    "shrink",
    "shorten",
    "smaller",
    "fewer",
]

# Value extraction pattern (e.g., "0.5", "0.5 to 0.7", "from 0.5 to 0.7")
VALUE_PATTERN = re.compile(
    r"(?:from\s+)?(\d+\.?\d*)\s*(?:to|->|→)\s*(\d+\.?\d*)|(\d+\.?\d*)"
)


class HypothesisToExperimentConverter:
    """Converts testable hypotheses into experiment proposals.

    Analyzes hypothesis statements for:
    1. References to adjustable parameters
    2. Suggested direction of change (increase/decrease)
    3. Target metrics for measurement
    4. Optional specific values
    """

    def __init__(
        self,
        current_values: dict[str, float] | None = None,
        min_confidence: float = 0.6,
    ):
        """Initialize converter.

        Args:
            current_values: Current parameter values for reference
            min_confidence: Minimum hypothesis confidence to convert
        """
        self._current_values = current_values or {}
        self._min_confidence = min_confidence

    def convert(self, hypothesis: "Hypothesis") -> Optional[ExperimentProposal]:
        """Attempt to convert a hypothesis to an experiment proposal.

        Args:
            hypothesis: Hypothesis to analyze

        Returns:
            ExperimentProposal if hypothesis is testable via parameter change,
            None otherwise
        """
        # Skip if already an experiment
        if hypothesis.is_experiment:
            logger.debug(f"Skipping {hypothesis.uid}: already an experiment")
            return None

        # Skip low-confidence hypotheses
        if hypothesis.confidence < self._min_confidence:
            logger.debug(
                f"Skipping {hypothesis.uid}: confidence {hypothesis.confidence:.2f} "
                f"< {self._min_confidence}"
            )
            return None

        statement = hypothesis.statement.lower()

        # Find parameter reference
        domain, parameter_path = self._find_parameter_reference(statement)
        if not domain or not parameter_path:
            logger.debug(f"Skipping {hypothesis.uid}: no parameter reference found")
            return None

        # Determine direction
        direction = self._determine_direction(statement)
        if not direction:
            logger.debug(
                f"Skipping {hypothesis.uid}: could not determine direction"
            )
            return None

        # Extract or compute treatment value
        treatment_value = self._extract_treatment_value(
            statement, parameter_path, direction
        )
        if treatment_value is None:
            logger.debug(
                f"Skipping {hypothesis.uid}: could not determine treatment value"
            )
            return None

        # Determine target metric
        target_metric = self._determine_target_metric(statement, domain)

        # Build rationale
        rationale = (
            f"Auto-converted from hypothesis: {hypothesis.statement[:100]}... "
            f"(confidence: {hypothesis.confidence:.2f})"
        )

        proposal = ExperimentProposal(
            domain=domain,
            parameter_path=parameter_path,
            treatment_value=treatment_value,
            rationale=rationale,
            target_metric=target_metric,
            expected_direction=direction,
            min_effect_size=0.1,
        )

        # REMOVED: Whitelist validation (Phase 1 Full Operational Autonomy)
        # Lilly autonomously judges parameter appropriateness

        logger.info(
            f"Converted hypothesis {hypothesis.uid} to experiment proposal: "
            f"{parameter_path} -> {treatment_value}"
        )

        return proposal

    def _find_parameter_reference(
        self, statement: str
    ) -> tuple[Optional[ExperimentDomain], Optional[str]]:
        """Find parameter reference in statement.

        Searches for known parameter keywords in the statement.
        Returns the most specific match (longest keyword).

        Args:
            statement: Lowercase hypothesis statement

        Returns:
            Tuple of (domain, parameter_path) or (None, None)
        """
        matches = []
        for keyword, (domain, param) in PARAMETER_PATTERNS.items():
            if keyword in statement:
                matches.append((len(keyword), domain, param))

        if not matches:
            return None, None

        # Return the most specific match (longest keyword)
        matches.sort(reverse=True)
        _, domain, param = matches[0]
        return domain, param

    def _determine_direction(self, statement: str) -> Optional[str]:
        """Determine expected direction from statement.

        Args:
            statement: Lowercase hypothesis statement

        Returns:
            "increase" or "decrease" or None
        """
        for keyword in INCREASE_KEYWORDS:
            if keyword in statement:
                return "increase"
        for keyword in DECREASE_KEYWORDS:
            if keyword in statement:
                return "decrease"
        return None

    def _extract_treatment_value(
        self,
        statement: str,
        parameter_path: str,
        direction: str,
    ) -> Optional[float]:
        """Extract or compute treatment value.

        Priority:
        1. Explicit value in statement (e.g., "to 0.7")
        2. Computed from current value + direction
        3. Domain-specific defaults

        Args:
            statement: Lowercase hypothesis statement
            parameter_path: Target parameter path
            direction: "increase" or "decrease"

        Returns:
            Treatment value or None
        """
        # Try to extract explicit value
        match = VALUE_PATTERN.search(statement)
        if match:
            try:
                # Pattern: "from X to Y" or "X to Y"
                if match.group(1) and match.group(2):
                    return float(match.group(2))
                # Pattern: just "X"
                elif match.group(3):
                    return float(match.group(3))
            except ValueError:
                pass

        # Compute based on current value and direction
        current = self._current_values.get(parameter_path)
        if current is not None:
            if direction == "increase":
                # Apply increase multiplier, cap at max for normalized params
                if "magnitude" in parameter_path or "weight" in parameter_path:
                    return min(NORMALIZED_PARAM_MAX, current * INCREASE_MULTIPLIER)
                elif "segments" in parameter_path:
                    return int(current * SEGMENT_INCREASE_MULTIPLIER)
                else:
                    return current * INCREASE_MULTIPLIER
            else:
                # Apply decrease multiplier, floor at min for normalized params
                if "magnitude" in parameter_path or "weight" in parameter_path:
                    return max(NORMALIZED_PARAM_MIN, current * DECREASE_MULTIPLIER)
                elif "segments" in parameter_path:
                    return max(SEGMENT_MIN_VALUE, int(current * SEGMENT_DECREASE_MULTIPLIER))
                else:
                    return current * DECREASE_MULTIPLIER

        # Default values by parameter type
        if "magnitude" in parameter_path or "weight" in parameter_path:
            return DEFAULT_MAGNITUDE_DECREASE if direction == "decrease" else DEFAULT_MAGNITUDE_INCREASE
        elif "segments" in parameter_path:
            return DEFAULT_SEGMENTS_DECREASE if direction == "decrease" else DEFAULT_SEGMENTS_INCREASE
        elif "probability" in parameter_path:
            return DEFAULT_PROBABILITY_DECREASE if direction == "decrease" else DEFAULT_PROBABILITY_INCREASE
        elif "rate" in parameter_path:
            return DEFAULT_PROBABILITY_DECREASE if direction == "decrease" else DEFAULT_PROBABILITY_INCREASE
        elif "confidence" in parameter_path:
            return DEFAULT_CONFIDENCE_DECREASE if direction == "decrease" else DEFAULT_CONFIDENCE_INCREASE
        elif "hypotheses" in parameter_path:
            return DEFAULT_HYPOTHESES_DECREASE if direction == "decrease" else DEFAULT_HYPOTHESES_INCREASE

        return None

    def _determine_target_metric(
        self, statement: str, domain: ExperimentDomain
    ) -> str:
        """Determine target metric from statement or domain.

        Args:
            statement: Lowercase hypothesis statement
            domain: Experiment domain

        Returns:
            Target metric name
        """
        # Check for explicit metric mentions
        metric_keywords = {
            "coherence": "alignment_correlation",
            "alignment": "alignment_correlation",
            "understanding": "self_understanding",
            "prediction": "self_understanding",
            "entropy": "semantic_entropy",
            "diversity": "semantic_entropy",
            "integration": "alignment_correlation",
            "focus": "alignment_correlation",
            "exploration": "semantic_entropy",
        }

        for keyword, metric in metric_keywords.items():
            if keyword in statement:
                return metric

        # Default by domain
        domain_defaults = {
            ExperimentDomain.STEERING: "alignment_correlation",
            ExperimentDomain.EPISODE: "self_understanding",
            ExperimentDomain.EMOTIONAL_FIELD: "alignment_correlation",
            ExperimentDomain.SIMULATION: "self_understanding",
            ExperimentDomain.TOOL_PATTERN: "alignment_correlation",
        }

        return domain_defaults.get(domain, "alignment_correlation")

    def update_current_values(self, values: dict[str, float]) -> None:
        """Update current parameter values.

        Args:
            values: Dict mapping parameter paths to current values
        """
        self._current_values.update(values)
