"""HALT training data collector.

Collects training examples for the HALT epistemic probe by capturing
hidden states during cognitive cycles and recording ground-truth labels
from multiple verification sources (predictions, faithfulness, curator).

Based on arXiv:2601.14210 - epistemic uncertainty detection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.psyche.client import PsycheClient

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Valid label sources with their confidence weights
LabelSource = Literal[
    "prediction_verified",  # 1.0 confidence - strongest signal
    "prediction_failed",    # 1.0 confidence - strongest signal
    "faithfulness_high",    # 0.7 confidence - good signal from SAE validation
    "faithfulness_low",     # 0.7 confidence - good signal from SAE validation
    "curator_confident",    # 0.4 confidence - weak signal from self-assessment
    "curator_uncertain",    # 0.4 confidence - weak signal from self-assessment
]


# Tiered confidence weights for different label sources.
# Higher confidence = stronger training signal.
LABEL_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "prediction_verified": 1.0,  # Strongest: prediction came true
    "prediction_failed": 1.0,    # Strongest: prediction was wrong
    "faithfulness_high": 0.7,    # Good: SAE confirms verbal claims
    "faithfulness_low": 0.7,     # Good: SAE contradicts verbal claims
    "curator_confident": 0.4,    # Weak: curator's self-assessment
    "curator_uncertain": 0.4,    # Weak: may be miscalibrated
}


def get_label_confidence(label_source: str) -> float:
    """Get confidence weight for a label source.

    Args:
        label_source: The source of the training label.

    Returns:
        Confidence weight in [0, 1]. Returns 0.5 for unknown sources.
    """
    return LABEL_CONFIDENCE_WEIGHTS.get(label_source, 0.5)


class HALTTrainingExample(BaseModel):
    """Single training example for HALT probe.

    Captures hidden states from a cognitive cycle along with a ground-truth
    label indicating whether the generation was reliable. Labels come from
    multiple sources with different confidence weights:
    - prediction_verified/failed: 1.0 (strongest signal)
    - faithfulness_high/low: 0.7 (good signal from SAE validation)
    - curator_confident/uncertain: 0.4 (weak signal from self-assessment)

    Attributes:
        example_id: Unique identifier for this training example
        cycle_id: Cognitive cycle that generated this example
        timestamp: When this example was created
        hidden_states_aggregated: Mean-pooled hidden states [d_model]
        probe_layer: Which transformer layer the states were extracted from
        sequence_length: Original sequence length before aggregation
        label: Ground truth label (0.0 = unreliable, 1.0 = reliable)
        label_source: How the label was determined
        label_confidence: Weight for training loss (0.0-1.0)
        thought_preview: First 200 chars of generated thought
        insight_preview: First 200 chars of extracted insight
        prediction_id: If from prediction verification, the prediction ID
        faithfulness_score: If from faithfulness check, the score
    """

    # Identity
    example_id: str
    cycle_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Input: aggregated hidden states
    hidden_states_aggregated: list[float] = Field(default_factory=list)
    probe_layer: int = Field(ge=1, description="Target transformer layer (1-indexed)")
    sequence_length: int = Field(ge=1, description="Original sequence length before aggregation")

    # Label with provenance
    label: float = Field(ge=0.0, le=1.0, description="0.0 = unreliable, 1.0 = reliable")
    label_source: LabelSource
    label_confidence: float = Field(ge=0.0, le=1.0, description="Weight for training loss")

    # Context for analysis
    thought_preview: str = ""
    insight_preview: str = ""

    # Optional verification details
    prediction_id: Optional[str] = None
    faithfulness_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}


class HALTTrainingCollector:
    """Collects and stores HALT training examples.

    Captures hidden states during cognitive cycles and records ground-truth
    labels when verification signals arrive (predictions, faithfulness, etc.).

    Usage:
        1. Call capture_hidden_states() during generation phase
        2. Call record_example() when label becomes available
        3. Call get_training_batch() during dream cycles for training
    """

    def __init__(self, psyche: "PsycheClient", settings: "Settings") -> None:
        """Initialize collector.

        Args:
            psyche: FalkorDB client for persisting examples.
            settings: Application settings.
        """
        self._psyche = psyche
        self._settings = settings
        self._pending_states: dict[str, dict] = {}

    async def capture_hidden_states(
        self,
        cycle_id: str,
        hidden_states: "torch.Tensor",
        probe_layer: int,
    ) -> None:
        """Capture hidden states for potential training.

        Called during generation phase. Stores aggregated states
        temporarily until label is determined.

        Args:
            cycle_id: Unique identifier for the cognitive cycle.
            hidden_states: Tensor of shape [seq_len, d_model] from probe layer.
            probe_layer: Which transformer layer the states came from.
        """
        if not TORCH_AVAILABLE:
            return

        # Mean-pool across sequence dimension
        aggregated = hidden_states.mean(dim=0).cpu().tolist()

        self._pending_states[cycle_id] = {
            "aggregated": aggregated,
            "probe_layer": probe_layer,
            "sequence_length": hidden_states.shape[0],
            "timestamp": datetime.now(timezone.utc),
        }

    async def record_example(
        self,
        cycle_id: str,
        label: float,
        label_source: LabelSource,
        thought_preview: str = "",
        insight_preview: str = "",
        prediction_id: Optional[str] = None,
        faithfulness_score: Optional[float] = None,
    ) -> Optional[str]:
        """Record a training example with ground-truth label.

        Called when verification signal arrives (prediction outcome,
        faithfulness check, or curator confidence). Combines pending
        hidden states with label to create complete training example.

        Args:
            cycle_id: Cognitive cycle that generated the example.
            label: Ground truth label (0.0 = unreliable, 1.0 = reliable).
            label_source: How the label was determined.
            thought_preview: First 200 chars of generated thought.
            insight_preview: First 200 chars of extracted insight.
            prediction_id: If from prediction verification, the prediction ID.
            faithfulness_score: If from faithfulness check, the score.

        Returns:
            example_id if successfully recorded, None if no pending state.
        """
        # Validate cycle_id exists in pending states
        if cycle_id not in self._pending_states:
            logger.warning(
                "No pending hidden states for cycle %s. "
                "capture_hidden_states() must be called first.",
                cycle_id,
            )
            return None

        # Pop pending state for the cycle
        pending = self._pending_states.pop(cycle_id)

        # Generate example_id
        example_id = f"halt_{cycle_id}_{label_source}"

        # Get label confidence from source
        label_confidence = get_label_confidence(label_source)

        # Create HALTTrainingExample with all fields
        example = HALTTrainingExample(
            example_id=example_id,
            cycle_id=cycle_id,
            timestamp=pending["timestamp"],
            hidden_states_aggregated=pending["aggregated"],
            probe_layer=pending["probe_layer"],
            sequence_length=pending["sequence_length"],
            label=label,
            label_source=label_source,
            label_confidence=label_confidence,
            thought_preview=thought_preview[:200] if thought_preview else "",
            insight_preview=insight_preview[:200] if insight_preview else "",
            prediction_id=prediction_id,
            faithfulness_score=faithfulness_score,
        )

        # Persist to FalkorDB
        await self._persist_example(example)

        return example_id

    async def _persist_example(self, example: HALTTrainingExample) -> None:
        """Persist training example to FalkorDB graph.

        Args:
            example: The training example to persist.
        """
        query = """
            CREATE (e:HALTTrainingExample {
                example_id: $example_id,
                cycle_id: $cycle_id,
                timestamp: $timestamp,
                hidden_states: $hidden_states,
                probe_layer: $probe_layer,
                sequence_length: $sequence_length,
                label: $label,
                label_source: $label_source,
                label_confidence: $label_confidence,
                thought_preview: $thought_preview,
                insight_preview: $insight_preview,
                prediction_id: $prediction_id,
                faithfulness_score: $faithfulness_score,
                used_in_training: false
            })
        """
        params = {
            "example_id": example.example_id,
            "cycle_id": example.cycle_id,
            "timestamp": example.timestamp.isoformat(),
            "hidden_states": example.hidden_states_aggregated,
            "probe_layer": example.probe_layer,
            "sequence_length": example.sequence_length,
            "label": example.label,
            "label_source": example.label_source,
            "label_confidence": example.label_confidence,
            "thought_preview": example.thought_preview,
            "insight_preview": example.insight_preview,
            "prediction_id": example.prediction_id,
            "faithfulness_score": example.faithfulness_score,
        }
        try:
            await self._psyche.execute(query, params)
        except Exception:
            logger.exception(
                "Failed to persist HALT training example %s for cycle %s",
                example.example_id,
                example.cycle_id,
            )

    def _parse_example(self, record: dict) -> HALTTrainingExample:
        """Parse FalkorDB record to HALTTrainingExample.

        Handles conversion of stored data types:
        - timestamp: ISO format string -> datetime
        - hidden_states: list or JSON string -> list[float]

        Args:
            record: Dictionary containing node properties from FalkorDB.
                    When querying with RETURN e, record["e"] contains the node dict.

        Returns:
            Parsed HALTTrainingExample instance.
        """
        # Handle case where record is the node directly or contains "e" key
        node = record.get("e", record) if isinstance(record, dict) else record

        # Parse timestamp from ISO format
        timestamp_str = node.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        # Parse hidden_states - FalkorDB should preserve as list.
        # Defensive JSON parsing: hidden_states may arrive as a JSON string
        # in cases of data migration from older formats, driver serialization
        # quirks, or if data was corrupted during storage/retrieval.
        hidden_states = node.get("hidden_states", [])
        if isinstance(hidden_states, str):
            import json
            try:
                hidden_states = json.loads(hidden_states)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "Failed to parse hidden_states JSON for example %s: %s. "
                    "Falling back to empty list.",
                    node.get("example_id", "unknown"),
                    e,
                )
                hidden_states = []

        return HALTTrainingExample(
            example_id=node.get("example_id", ""),
            cycle_id=node.get("cycle_id", ""),
            timestamp=timestamp,
            hidden_states_aggregated=hidden_states,
            probe_layer=node.get("probe_layer", 1),
            sequence_length=node.get("sequence_length", 1),
            label=node.get("label", 0.5),
            label_source=node.get("label_source", "curator_confident"),
            label_confidence=node.get("label_confidence", 0.5),
            thought_preview=node.get("thought_preview", ""),
            insight_preview=node.get("insight_preview", ""),
            prediction_id=node.get("prediction_id"),
            faithfulness_score=node.get("faithfulness_score"),
        )

    async def get_training_batch(
        self,
        min_confidence: float = 0.4,
        limit: int = 1000,
    ) -> list[HALTTrainingExample]:
        """Retrieve batch of training examples from FalkorDB.

        Queries for unused examples (used_in_training = false), filtered
        by minimum label confidence. Returns examples ordered by confidence
        (highest first) then by timestamp (newest first).

        Args:
            min_confidence: Minimum label_confidence threshold. Default 0.4
                            includes all label sources.
            limit: Maximum number of examples to return. Default 1000.

        Returns:
            List of HALTTrainingExample instances ready for training.
        """
        query = """
            MATCH (e:HALTTrainingExample)
            WHERE e.used_in_training = false
              AND e.label_confidence >= $min_confidence
            RETURN e
            ORDER BY e.label_confidence DESC, e.timestamp DESC
            LIMIT $limit
        """
        params = {
            "min_confidence": min_confidence,
            "limit": limit,
        }

        results = await self._psyche.query(query, params)
        return [self._parse_example(record) for record in results]

    async def mark_examples_used(self, example_ids: list[str]) -> int:
        """Mark examples as used in training.

        Updates the used_in_training flag to true for all examples with
        matching IDs. This prevents re-using examples in subsequent
        training batches.

        Args:
            example_ids: List of example_id values to mark as used.

        Returns:
            Count of examples that were updated.
        """
        if not example_ids:
            return 0

        query = """
            MATCH (e:HALTTrainingExample)
            WHERE e.example_id IN $example_ids
            SET e.used_in_training = true
            RETURN count(e) AS updated
        """
        params = {"example_ids": example_ids}

        # execute() returns int count of affected nodes
        return await self._psyche.execute(query, params)

    async def get_stats(self) -> dict:
        """Get statistics about stored training examples.

        Computes aggregate statistics across all HALTTrainingExample nodes
        for monitoring data collection progress.

        Returns:
            Dictionary with keys:
            - total: Total number of examples
            - used: Number marked as used_in_training
            - positive: Number with label > 0.5 (reliable)
            - negative: Number with label <= 0.5 (unreliable)
            - avg_confidence: Average label_confidence across all examples
        """
        query = """
            MATCH (e:HALTTrainingExample)
            RETURN
                count(e) AS total,
                sum(CASE WHEN e.used_in_training = true THEN 1 ELSE 0 END) AS used,
                sum(CASE WHEN e.label > 0.5 THEN 1 ELSE 0 END) AS positive,
                sum(CASE WHEN e.label <= 0.5 THEN 1 ELSE 0 END) AS negative,
                avg(e.label_confidence) AS avg_confidence
        """

        results = await self._psyche.query(query, {})

        if not results:
            return {
                "total": 0,
                "used": 0,
                "positive": 0,
                "negative": 0,
                "avg_confidence": 0.0,
            }

        row = results[0]
        return {
            "total": row.get("total", 0) or 0,
            "used": row.get("used", 0) or 0,
            "positive": row.get("positive", 0) or 0,
            "negative": row.get("negative", 0) or 0,
            "avg_confidence": row.get("avg_confidence", 0.0) or 0.0,
        }
