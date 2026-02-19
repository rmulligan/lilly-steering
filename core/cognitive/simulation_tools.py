"""Simulation tools for structured prediction extraction.

Provides tools that enable the Curator model to convert natural language
simulation output from Preflexor into validated Prediction objects,
eliminating fragile regex parsing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from core.cognitive.simulation.schemas import (
    MetricsSnapshot,
    Prediction,
    PredictionConditionType,
)

logger = logging.getLogger(__name__)

# Default values for prediction creation
DEFAULT_CYCLES_TO_VERIFY = 5
DEFAULT_PREDICTION_CONFIDENCE = 0.5
PREDICTION_EXPIRY_GRACE_CYCLES = 20

# Valid metric names from MetricsSnapshot for METRIC_THRESHOLD predictions
# Dynamically generated from schema to stay in sync with MetricsSnapshot model
VALID_METRICS = frozenset(
    field for field in MetricsSnapshot.model_fields
    if field not in {"cycle", "timestamp", "active_experiment_uid"}
)

# Tool definitions in OpenAI function calling format
SIMULATION_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "create_prediction": {
        "type": "function",
        "function": {
            "name": "create_prediction",
            "description": (
                "Create a testable prediction from natural language. "
                "Use this to convert Preflexor's predictions into structured, verifiable format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "The human-readable prediction claim "
                            "(what Preflexor predicted)"
                        ),
                    },
                    "hypothesis_uid": {
                        "type": "string",
                        "description": "UID of the hypothesis this prediction tests",
                    },
                    "metric": {
                        "type": "string",
                        "enum": sorted(VALID_METRICS),
                        "description": (
                            "System metric to measure "
                            "(must be one of the valid metrics)"
                        ),
                    },
                    "operator": {
                        "type": "string",
                        "enum": [">", "<", ">=", "<=", "==", "increases", "decreases"],
                        "description": "Comparison operator for the threshold",
                    },
                    "threshold": {
                        "type": "string",
                        "description": (
                            "Threshold value as string (e.g., '0.7', 'baseline + 0.1'). "
                            "Use 'baseline + X' for relative predictions."
                        ),
                    },
                    "cycles_to_verify": {
                        "type": "integer",
                        "description": "Number of cycles before prediction can be verified",
                        "default": DEFAULT_CYCLES_TO_VERIFY,
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in this prediction (0.0 to 1.0)",
                        "default": DEFAULT_PREDICTION_CONFIDENCE,
                    },
                },
                "required": ["claim", "hypothesis_uid", "metric", "operator", "threshold"],
            },
        },
    },
    "get_current_metrics": {
        "type": "function",
        "function": {
            "name": "get_current_metrics",
            "description": (
                "Get current values of system metrics. "
                "Use this to ground predictions in current state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": sorted(VALID_METRICS)},
                        "description": "List of metric names to retrieve",
                    },
                },
                "required": ["metrics"],
            },
        },
    },
}


def get_simulation_tools(enabled_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return tool definitions for simulation extraction.

    Args:
        enabled_names: List of tool names to enable. If None, returns all tools.

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    if enabled_names is None:
        return list(SIMULATION_TOOL_REGISTRY.values())
    return [
        SIMULATION_TOOL_REGISTRY[name]
        for name in enabled_names
        if name in SIMULATION_TOOL_REGISTRY
    ]


@dataclass
class SimulationToolCall:
    """Represents a tool call from simulation extraction.

    Attributes:
        name: Tool function name
        arguments: JSON arguments for the tool
        call_id: Unique identifier for this call
    """

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class SimulationToolResult:
    """Result from executing a simulation tool.

    Attributes:
        call_id: ID of the tool call this responds to
        content: JSON string result
        error: Error message if tool failed
    """

    call_id: str
    content: str
    error: Optional[str] = None


class SimulationTools:
    """Executor for simulation extraction tools.

    Provides methods to create validated Prediction objects from
    natural language, eliminating regex-based detection.
    """

    def __init__(
        self,
        current_cycle: int,
        metrics_snapshot: Optional[MetricsSnapshot] = None,
    ):
        """Initialize simulation tools.

        Args:
            current_cycle: Current cognitive cycle number
            metrics_snapshot: Current metrics for grounding predictions (optional)
        """
        self._current_cycle = current_cycle
        self._metrics_snapshot = metrics_snapshot
        self._created_predictions: list[Prediction] = []

    @property
    def created_predictions(self) -> list[Prediction]:
        """Get all predictions created during this extraction session."""
        return self._created_predictions

    async def execute(self, tool_call: SimulationToolCall) -> SimulationToolResult:
        """Execute a tool call and return the result.

        Args:
            tool_call: The tool call to execute

        Returns:
            SimulationToolResult with JSON content or error
        """
        try:
            if tool_call.name == "create_prediction":
                result = await self.create_prediction(**tool_call.arguments)
            elif tool_call.name == "get_current_metrics":
                result = await self.get_current_metrics(**tool_call.arguments)
            else:
                return SimulationToolResult(
                    call_id=tool_call.call_id,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )

            return SimulationToolResult(
                call_id=tool_call.call_id,
                content=json.dumps(result, default=str),
            )

        except Exception as e:
            logger.warning(f"Simulation tool {tool_call.name} failed: {e}")
            return SimulationToolResult(
                call_id=tool_call.call_id,
                content="",
                error=str(e),
            )

    async def create_prediction(
        self,
        claim: str,
        hypothesis_uid: str,
        metric: str,
        operator: str,
        threshold: str,
        cycles_to_verify: int = DEFAULT_CYCLES_TO_VERIFY,
        confidence: float = DEFAULT_PREDICTION_CONFIDENCE,
    ) -> dict[str, Any]:
        """Create a validated Prediction object.

        Args:
            claim: Human-readable prediction claim
            hypothesis_uid: UID of parent hypothesis
            metric: System metric to measure
            operator: Comparison operator
            threshold: Threshold value (can be relative to baseline)
            cycles_to_verify: Cycles before verification
            confidence: Prediction confidence

        Returns:
            Dict with prediction details

        Raises:
            ValueError: If metric is not valid
        """
        # Validate metric name
        if metric not in VALID_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {sorted(VALID_METRICS)}"
            )

        # Validate threshold format
        # Reject values with invalid characters that would fail verifier parsing
        threshold = threshold.strip()
        if not threshold:
            raise ValueError("Threshold cannot be empty")

        # Check for invalid characters (%, etc.)
        invalid_chars = frozenset(("%", "$", "#", "@", "!"))
        if first_invalid := next((c for c in threshold if c in invalid_chars), None):
            raise ValueError(
                f"Invalid threshold '{threshold}': contains '{first_invalid}'. "
                f"Use decimal format (e.g., '0.05' instead of '5%')"
            )

        # Validate threshold is parseable (unless it's baseline-relative)
        if "baseline" not in threshold.lower():
            try:
                float(threshold)
            except ValueError:
                raise ValueError(
                    f"Invalid threshold '{threshold}': must be a number or "
                    f"baseline-relative expression (e.g., 'baseline + 0.1')"
                )

        # Normalize operator
        op_map = {
            "increases": ">",
            "decreases": "<",
        }
        normalized_op = op_map.get(operator, operator)

        # Build condition_value string
        condition_value = f"{metric} {normalized_op} {threshold}"

        # Capture baseline if we have metrics and threshold is relative
        baseline_metrics = None
        baseline_cycle = None
        if self._metrics_snapshot:
            baseline_metrics = self._metrics_snapshot.to_dict()
            baseline_cycle = self._metrics_snapshot.cycle

            # If threshold is relative (e.g., "baseline + 0.1"), resolve it
            if "baseline" in threshold.lower():
                current_value = self._metrics_snapshot.get_metric(metric)
                # Parse "baseline + X" or "baseline - X"
                if "+" in threshold:
                    delta = float(threshold.split("+")[1].strip())
                    resolved_threshold = current_value + delta
                elif "-" in threshold:
                    delta = float(threshold.split("-")[1].strip())
                    resolved_threshold = current_value - delta
                else:
                    resolved_threshold = current_value
                condition_value = f"{metric} {normalized_op} {resolved_threshold:.4f}"

        # Create the prediction
        prediction = Prediction(
            hypothesis_uid=hypothesis_uid,
            claim=claim,
            condition_type=PredictionConditionType.METRIC_THRESHOLD,
            condition_value=condition_value,
            confidence=max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
            earliest_verify_cycle=self._current_cycle + cycles_to_verify,
            expiry_cycle=self._current_cycle + cycles_to_verify + PREDICTION_EXPIRY_GRACE_CYCLES,
            cycle_created=self._current_cycle,
            baseline_cycle=baseline_cycle,
            baseline_metrics=baseline_metrics,
        )

        self._created_predictions.append(prediction)

        logger.info(
            f"[SIMULATION TOOL] Created prediction: {prediction.uid} "
            f"({metric} {normalized_op} {threshold}) for hypothesis {hypothesis_uid}"
        )

        return {
            "success": True,
            "prediction_uid": prediction.uid,
            "condition_type": "metric_threshold",
            "condition_value": condition_value,
            "earliest_verify_cycle": prediction.earliest_verify_cycle,
            "baseline_captured": baseline_metrics is not None,
        }

    async def get_current_metrics(
        self,
        metrics: list[str],
    ) -> dict[str, Any]:
        """Get current values of system metrics.

        Args:
            metrics: List of metric names to retrieve

        Returns:
            Dict with metric values
        """
        if not self._metrics_snapshot:
            return {
                "success": False,
                "error": "Metrics snapshot not available",
                "metrics": {},
            }

        result_metrics = {}
        for metric in metrics:
            if metric not in VALID_METRICS:
                result_metrics[metric] = {"error": f"Invalid metric: {metric}"}
            else:
                value = self._metrics_snapshot.get_metric(metric)
                result_metrics[metric] = value

        return {
            "success": True,
            "cycle": self._metrics_snapshot.cycle,
            "metrics": result_metrics,
        }
