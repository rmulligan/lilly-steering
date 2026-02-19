"""Rolling buffer for metacognitive analysis across cycles.

Maintains a fixed-size window of cycle summaries for pattern detection.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_PATH = Path("~/.lilly/metacognition_buffer.json").expanduser()


@dataclass
class CycleSummary:
    """Summary of a single cognitive cycle for metacognitive analysis."""

    cycle_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Thought content (truncated)
    thought_snippet: str = ""

    # Curation insights
    curation_insights: list[str] = field(default_factory=list)

    # Emotional state
    emotional_state: dict[str, float] = field(default_factory=dict)

    # Metrics snapshot
    metrics: dict[str, float] = field(default_factory=dict)

    # Hypothesis activity
    hypothesis_activity: dict[str, Any] = field(default_factory=dict)

    # Reflexion health category
    reflexion_category: str = "STABLE"

    # Telemetry signals (biofeedback)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        cycle: int,
        thought: str,
        curation_insights: list[str] | None = None,
        emotional_state: dict[str, float] | None = None,
        metrics: dict[str, float] | None = None,
        hypothesis_activity: dict[str, Any] | None = None,
        reflexion_category: str = "STABLE",
        telemetry: dict[str, Any] | None = None,
    ) -> CycleSummary:
        """Create a cycle summary from cognitive state components.

        Args:
            cycle: Current cycle number
            thought: Raw thought text (will be truncated to 500 chars)
            curation_insights: Key insights from curator phase
            emotional_state: Dict with valence, arousal, dominant_affect
            metrics: Dict with prediction_rate, integration_success, etc.
            hypothesis_activity: Dict with new_hypotheses, verifications, failures
            reflexion_category: Health category from reflexion phase
            telemetry: Biofeedback signals from telemetry evaluator

        Returns:
            CycleSummary instance
        """
        return cls(
            cycle_number=cycle,
            thought_snippet=thought[:500] if thought else "",
            curation_insights=curation_insights or [],
            emotional_state=emotional_state or {},
            metrics=metrics or {},
            hypothesis_activity=hypothesis_activity or {},
            reflexion_category=reflexion_category,
            telemetry=telemetry or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "thought_snippet": self.thought_snippet,
            "curation_insights": self.curation_insights,
            "emotional_state": self.emotional_state,
            "metrics": self.metrics,
            "hypothesis_activity": self.hypothesis_activity,
            "reflexion_category": self.reflexion_category,
            "telemetry": self.telemetry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleSummary:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            cycle_number=data.get("cycle_number", 0),
            timestamp=timestamp,
            thought_snippet=data.get("thought_snippet", ""),
            curation_insights=data.get("curation_insights", []),
            emotional_state=data.get("emotional_state", {}),
            metrics=data.get("metrics", {}),
            hypothesis_activity=data.get("hypothesis_activity", {}),
            reflexion_category=data.get("reflexion_category", "STABLE"),
            telemetry=data.get("telemetry", {}),
        )


class MetacognitionBuffer:
    """Rolling buffer of cycle summaries for metacognitive analysis.

    Maintains a fixed-size deque of CycleSummary objects, automatically
    dropping the oldest when the buffer is full.
    """

    def __init__(self, maxlen: int = 5):
        """Initialize the buffer.

        Args:
            maxlen: Maximum number of cycle summaries to retain (default: 5)
        """
        self._buffer: deque[CycleSummary] = deque(maxlen=maxlen)
        self._maxlen = maxlen

    @property
    def maxlen(self) -> int:
        """Maximum number of cycle summaries the buffer can hold."""
        return self._maxlen

    def append(self, summary: CycleSummary) -> None:
        """Add a cycle summary to the buffer.

        Args:
            summary: CycleSummary to add
        """
        self._buffer.append(summary)

    def __len__(self) -> int:
        """Return the number of summaries in the buffer."""
        return len(self._buffer)

    def __iter__(self):
        """Iterate over summaries in chronological order."""
        return iter(self._buffer)

    @property
    def is_warmed_up(self) -> bool:
        """Check if buffer has minimum cycles for meaningful analysis."""
        return len(self._buffer) >= 3

    def to_prompt(self) -> str:
        """Format buffer as structured input for metacognition.

        Returns:
            Formatted string with all cycle summaries for the LLM prompt.
        """
        if not self._buffer:
            return "No cycle data available yet."

        sections = []
        for summary in self._buffer:
            # Format emotional state
            valence = summary.emotional_state.get("valence", 0)
            arousal = summary.emotional_state.get("arousal", 0)
            dominant = summary.emotional_state.get("dominant_affect", "neutral")

            # Format metrics
            pred_rate = summary.metrics.get("prediction_rate", 0)
            integration = summary.metrics.get("integration_success", 0)

            # Format hypothesis activity
            new_hyp = summary.hypothesis_activity.get("new_count", 0)
            verified = summary.hypothesis_activity.get("verified_count", 0)
            failed = summary.hypothesis_activity.get("failed_count", 0)

            # Format telemetry if available
            telemetry_str = ""
            if summary.telemetry.get("available"):
                conf = summary.telemetry.get("confidence_score", -1)
                strain = summary.telemetry.get("strain_score", -1)
                if conf >= 0 and strain >= 0:
                    telemetry_str = f"\n**Biofeedback:** confidence={conf:.2f}, strain={strain:.2f}"

            sections.append(f"""## Cycle {summary.cycle_number}
**Thought:** {summary.thought_snippet[:200]}{"..." if len(summary.thought_snippet) > 200 else ""}
**Insights:** {', '.join(summary.curation_insights[:3]) if summary.curation_insights else 'None'}
**Emotional State:** valence={valence:.2f}, arousal={arousal:.2f}, affect={dominant}
**Health:** {summary.reflexion_category}
**Metrics:** prediction_rate={pred_rate:.2f}, integration={integration:.2f}
**Hypotheses:** new={new_hyp}, verified={verified}, failed={failed}{telemetry_str}
""")

        return "\n".join(sections)

    def get_trend_summary(self) -> dict[str, Any]:
        """Compute simple trend indicators across the buffer.

        Returns:
            Dict with trend indicators for key metrics.
        """
        if len(self._buffer) < 2:
            return {"has_trend": False}

        summaries = list(self._buffer)

        # Extract metric sequences
        pred_rates = [s.metrics.get("prediction_rate", 0) for s in summaries]
        integration_rates = [s.metrics.get("integration_success", 0) for s in summaries]
        valences = [s.emotional_state.get("valence", 0) for s in summaries]
        arousals = [s.emotional_state.get("arousal", 0) for s in summaries]

        def trend_direction(values: list[float]) -> str:
            """Compute simple trend direction."""
            if len(values) < 2:
                return "stable"
            first_half = sum(values[: len(values) // 2]) / max(1, len(values) // 2)
            second_half = sum(values[len(values) // 2 :]) / max(
                1, len(values) - len(values) // 2
            )
            diff = second_half - first_half
            if diff > 0.1:
                return "increasing"
            elif diff < -0.1:
                return "decreasing"
            return "stable"

        # Count health categories
        health_counts: dict[str, int] = {}
        for s in summaries:
            cat = s.reflexion_category
            health_counts[cat] = health_counts.get(cat, 0) + 1

        return {
            "has_trend": True,
            "prediction_trend": trend_direction(pred_rates),
            "integration_trend": trend_direction(integration_rates),
            "valence_trend": trend_direction(valences),
            "arousal_trend": trend_direction(arousals),
            "health_distribution": health_counts,
            "cycles_analyzed": len(summaries),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize buffer to dictionary."""
        return {
            "maxlen": self._maxlen,
            "summaries": [s.to_dict() for s in self._buffer],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetacognitionBuffer:
        """Deserialize buffer from dictionary."""
        maxlen = data.get("maxlen", 5)
        buffer = cls(maxlen=maxlen)
        for summary_data in data.get("summaries", []):
            buffer.append(CycleSummary.from_dict(summary_data))
        return buffer

    def save(self, path: Path | None = None) -> None:
        """Save buffer to JSON file.

        Args:
            path: Optional custom path. Uses DEFAULT_BUFFER_PATH if not provided.
        """
        save_path = path or DEFAULT_BUFFER_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
            logger.debug("Metacognition buffer saved to %s (%d cycles)", save_path, len(self))
        except Exception as e:
            logger.error("Failed to save metacognition buffer: %s", e)

    @classmethod
    def load(cls, path: Path | None = None, maxlen: int = 5) -> MetacognitionBuffer:
        """Load buffer from JSON file.

        Args:
            path: Optional custom path. Uses DEFAULT_BUFFER_PATH if not provided.
            maxlen: Maximum buffer size (used if file doesn't exist)

        Returns:
            MetacognitionBuffer instance (empty if file doesn't exist)
        """
        load_path = path or DEFAULT_BUFFER_PATH

        if not load_path.exists():
            logger.info("No metacognition buffer found, starting fresh")
            return cls(maxlen=maxlen)

        try:
            data = json.loads(load_path.read_text())
            buffer = cls.from_dict(data)
            logger.info("Metacognition buffer loaded: %d cycles from %s", len(buffer), load_path)
            return buffer
        except json.JSONDecodeError as e:
            logger.error("Failed to parse metacognition buffer, starting fresh: %s", e)
            return cls(maxlen=maxlen)
        except (IOError, OSError, KeyError, TypeError) as e:
            logger.error("Failed to load metacognition buffer, starting fresh: %s", e)
            return cls(maxlen=maxlen)
