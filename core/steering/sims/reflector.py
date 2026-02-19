"""SIMS Reflector - pattern analysis and adjustment planning.

The Reflector analyzes observation data to identify patterns
and propose steering vector adjustments. It examines:
- High-surprise observations from SteeringObserver
- Identity coherence via vector similarity
- Activation drift patterns
- Psyche-stored contrastive pairs

It produces ReflectionResult with proposed VectorAdjustments
that the Executor can apply.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

try:
    import torch  # noqa: F401 - availability check
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.state_machine import SIMSContext

logger = logging.getLogger(__name__)


class AdjustmentType(Enum):
    """Types of steering vector adjustments."""

    STRENGTHEN = "strengthen"  # Increase vector magnitude
    WEAKEN = "weaken"  # Decrease vector magnitude
    ADD = "add"  # Add new vector
    REMOVE = "remove"  # Remove vector entirely


@dataclass
class VectorAdjustment:
    """Proposed adjustment to a steering vector.

    Attributes:
        vector_name: Name of the vector to adjust
        adjustment_type: Type of adjustment to make
        magnitude: Strength of adjustment (0-1)
        reason: Why this adjustment is proposed
        new_vector: For ADD type, the new vector to add
    """

    vector_name: str
    adjustment_type: AdjustmentType
    magnitude: float
    reason: str
    new_vector: Optional[Any] = None  # torch.Tensor when available

    def to_dict(self) -> dict:
        """Serialize for logging."""
        result = {
            "vector_name": self.vector_name,
            "adjustment_type": self.adjustment_type.value,
            "magnitude": self.magnitude,
            "reason": self.reason,
        }
        if self.new_vector is not None:
            result["has_new_vector"] = True
        return result


@dataclass
class ReflectionResult:
    """Result of SIMS reflection phase.

    Attributes:
        adjustments: List of proposed vector adjustments
        analysis_summary: Human-readable summary of analysis
        confidence: Confidence in proposed adjustments (0-1)
        timestamp: When reflection occurred
    """

    adjustments: list[VectorAdjustment]
    analysis_summary: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def should_apply(self, threshold: float = 0.5) -> bool:
        """Check if adjustments should be applied.

        Args:
            threshold: Minimum confidence required

        Returns:
            True if confidence exceeds threshold and there are adjustments
        """
        return self.confidence >= threshold and len(self.adjustments) > 0

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "adjustments": [adj.to_dict() for adj in self.adjustments],
            "analysis_summary": self.analysis_summary,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "should_apply": self.should_apply(),
        }


class SIMSReflector:
    """Analyzes patterns and proposes steering vector adjustments.

    The Reflector examines observations from SteeringObserver,
    queries Psyche for relevant context, and proposes adjustments
    to steering vectors based on detected patterns.

    Attributes:
        observer: SteeringObserver for accessing observations
        psyche: PsycheClient for graph queries
        hidden_size: Model hidden dimension
        identity_hooks: Optional IdentityHooks for identity analysis
    """

    def __init__(
        self,
        observer: Any,
        psyche: Any,
        hidden_size: int,
        identity_hooks: Optional[Any] = None,
    ):
        """Initialize the reflector.

        Args:
            observer: SteeringObserver instance
            psyche: PsycheClient instance
            hidden_size: Model hidden dimension (e.g., 768)
            identity_hooks: Optional IdentityHooks for identity coherence analysis
        """
        self.observer = observer
        self.psyche = psyche
        self.hidden_size = hidden_size
        self.identity_hooks = identity_hooks

    async def reflect(self, context: SIMSContext) -> ReflectionResult:
        """Analyze patterns and propose adjustments.

        Args:
            context: SIMS execution context with surprise level and metadata

        Returns:
            ReflectionResult with proposed adjustments
        """
        adjustments: list[VectorAdjustment] = []
        analysis_parts: list[str] = []
        confidence = 0.0

        # Get observation statistics
        stats = self.observer.get_stats()
        high_surprise_obs = self.observer.get_high_surprise_observations()

        # Analyze based on surprise level
        surprise_level = context.surprise_level

        if surprise_level < 0.3:
            # Low surprise - system is well-calibrated
            analysis_parts.append("Low surprise detected, system well-calibrated.")
            confidence = 0.2

        elif surprise_level < 0.7:
            # Moderate surprise - may need minor adjustments
            analysis_parts.append("Moderate surprise detected.")
            confidence = 0.5

            # Check for activation drift
            if context.metadata.get("activation_drift", 0) > 0.3:
                analysis_parts.append("Activation drift detected.")

        else:
            # High surprise - likely need adjustments
            analysis_parts.append("High surprise detected, analyzing patterns.")
            confidence = 0.7

            # Analyze high-surprise observations
            if high_surprise_obs:
                analysis_parts.append(
                    f"Found {len(high_surprise_obs)} high-surprise observations."
                )

                # Check for identity-related patterns
                if self.identity_hooks and context.current_vectors.get("identity"):
                    analysis_parts.append("Analyzing identity coherence.")
                    # Could add identity-strengthening adjustment here

        # Check for contrastive pair suggestions
        if hasattr(self.observer, "generate_contrastive_pairs"):
            pairs = await self.observer.generate_contrastive_pairs()
            if pairs:
                analysis_parts.append(f"Found {len(pairs)} contrastive pairs.")

        # Query Psyche for relevant steering vectors
        try:
            psyche_results = await self.psyche.query("MATCH (sv:SteeringVector) RETURN sv LIMIT 10")
            if psyche_results:
                analysis_parts.append(
                    f"Retrieved {len(psyche_results)} Psyche results."
                )
        except Exception as e:
            logger.debug(f"Psyche query failed: {e}")

        # Ensure constitutional vector is never removed
        for adj in adjustments:
            if adj.vector_name == "constitutional":
                if adj.adjustment_type == AdjustmentType.REMOVE:
                    adjustments.remove(adj)
                    analysis_parts.append(
                        "Blocked removal of constitutional vector (protected)."
                    )

        return ReflectionResult(
            adjustments=adjustments,
            analysis_summary=" ".join(analysis_parts) if analysis_parts else "No analysis performed.",
            confidence=confidence,
        )
