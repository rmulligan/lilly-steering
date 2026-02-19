"""SIMS Executor - applies steering vector adjustments.

The Executor takes ReflectionResult with proposed VectorAdjustments
and applies them to the actual steering vectors. It handles:
- STRENGTHEN: Increase vector magnitude
- WEAKEN: Decrease vector magnitude
- ADD: Register new steering vector
- REMOVE: Remove vector (except constitutional which is protected)

Results are tracked and optionally persisted to Psyche.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import torch  # noqa: F401 - availability check
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.sims.reflector import (
    AdjustmentType,
    VectorAdjustment,
    ReflectionResult,
)
from core.steering.sims.state_machine import SIMSContext

logger = logging.getLogger(__name__)


# Protected vectors that cannot be removed
PROTECTED_VECTORS = {"constitutional"}


@dataclass
class AppliedAdjustment:
    """Record of an applied adjustment.

    Attributes:
        vector_name: Name of the adjusted vector
        adjustment_type: Type of adjustment applied
        original_magnitude: Magnitude before adjustment
        new_magnitude: Magnitude after adjustment
        success: Whether the adjustment was applied successfully
        error: Error message if adjustment failed
    """

    vector_name: str
    adjustment_type: AdjustmentType
    original_magnitude: float
    new_magnitude: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for logging."""
        result = {
            "vector_name": self.vector_name,
            "adjustment_type": self.adjustment_type.value,
            "original_magnitude": self.original_magnitude,
            "new_magnitude": self.new_magnitude,
            "success": self.success,
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class ExecutionResult:
    """Result of SIMS execution phase.

    Attributes:
        applied_adjustments: List of applied adjustments
        total_requested: Number of adjustments requested
        total_applied: Number of adjustments successfully applied
        timestamp: When execution occurred
    """

    applied_adjustments: list[AppliedAdjustment]
    total_requested: int
    total_applied: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        """Calculate success rate of adjustments."""
        if self.total_requested == 0:
            return 1.0
        return self.total_applied / self.total_requested

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "applied_adjustments": [adj.to_dict() for adj in self.applied_adjustments],
            "total_requested": self.total_requested,
            "total_applied": self.total_applied,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp.isoformat(),
        }


class SIMSExecutor:
    """Applies steering vector adjustments from reflection results.

    The Executor handles the actual modification of steering vectors
    based on proposals from the Reflector. It ensures safety constraints
    (e.g., constitutional vector protection) and tracks all applied changes.

    Attributes:
        identity_hooks: IdentityHooks for vector manipulation
        psyche: PsycheClient for persistence
    """

    def __init__(
        self,
        identity_hooks: Any,
        psyche: Any,
    ):
        """Initialize the executor.

        Args:
            identity_hooks: IdentityHooks instance for vector operations
            psyche: PsycheClient instance for persistence
        """
        self.identity_hooks = identity_hooks
        self.psyche = psyche

    async def execute(
        self,
        context: SIMSContext,
        persist: bool = False,
    ) -> ExecutionResult:
        """Apply proposed adjustments from reflection result.

        Args:
            context: SIMS context containing reflect_result in metadata
            persist: Whether to persist changes to Psyche

        Returns:
            ExecutionResult with details of applied adjustments
        """
        applied: list[AppliedAdjustment] = []

        # Get reflection result from context
        reflection: Optional[ReflectionResult] = context.metadata.get("reflect_result")

        if reflection is None:
            logger.debug("No reflection result in context, skipping execution")
            return ExecutionResult(
                applied_adjustments=[],
                total_requested=0,
                total_applied=0,
            )

        total_requested = len(reflection.adjustments)

        for adjustment in reflection.adjustments:
            result = await self._apply_adjustment(adjustment)
            applied.append(result)

            # Persist if requested and successful
            if persist and result.success:
                await self._persist_adjustment(adjustment, result)

        total_applied = sum(1 for a in applied if a.success)

        logger.info(
            f"SIMS Executor: Applied {total_applied}/{total_requested} adjustments"
        )

        return ExecutionResult(
            applied_adjustments=applied,
            total_requested=total_requested,
            total_applied=total_applied,
        )

    async def _apply_adjustment(
        self, adjustment: VectorAdjustment
    ) -> AppliedAdjustment:
        """Apply a single adjustment.

        Args:
            adjustment: The adjustment to apply

        Returns:
            AppliedAdjustment with result details
        """
        vector_name = adjustment.vector_name
        adj_type = adjustment.adjustment_type

        # Check for protected vectors
        if adj_type == AdjustmentType.REMOVE and vector_name in PROTECTED_VECTORS:
            logger.warning(
                f"Blocked removal of protected vector: {vector_name}"
            )
            return AppliedAdjustment(
                vector_name=vector_name,
                adjustment_type=adj_type,
                original_magnitude=1.0,
                new_magnitude=1.0,
                success=False,
                error=f"Cannot remove protected vector: {vector_name}",
            )

        try:
            if adj_type == AdjustmentType.STRENGTHEN:
                self.identity_hooks.strengthen_vector(vector_name, adjustment.magnitude)
                return AppliedAdjustment(
                    vector_name=vector_name,
                    adjustment_type=adj_type,
                    original_magnitude=1.0,
                    new_magnitude=1.0 + adjustment.magnitude,
                    success=True,
                )

            elif adj_type == AdjustmentType.WEAKEN:
                self.identity_hooks.weaken_vector(vector_name, adjustment.magnitude)
                return AppliedAdjustment(
                    vector_name=vector_name,
                    adjustment_type=adj_type,
                    original_magnitude=1.0,
                    new_magnitude=max(0.0, 1.0 - adjustment.magnitude),
                    success=True,
                )

            elif adj_type == AdjustmentType.ADD:
                if adjustment.new_vector is None:
                    return AppliedAdjustment(
                        vector_name=vector_name,
                        adjustment_type=adj_type,
                        original_magnitude=0.0,
                        new_magnitude=0.0,
                        success=False,
                        error="No new_vector provided for ADD adjustment",
                    )
                self.identity_hooks.add_vector(vector_name, adjustment.new_vector)
                return AppliedAdjustment(
                    vector_name=vector_name,
                    adjustment_type=adj_type,
                    original_magnitude=0.0,
                    new_magnitude=adjustment.magnitude,
                    success=True,
                )

            elif adj_type == AdjustmentType.REMOVE:
                self.identity_hooks.remove_vector(vector_name)
                return AppliedAdjustment(
                    vector_name=vector_name,
                    adjustment_type=adj_type,
                    original_magnitude=1.0,
                    new_magnitude=0.0,
                    success=True,
                )

            else:
                return AppliedAdjustment(
                    vector_name=vector_name,
                    adjustment_type=adj_type,
                    original_magnitude=1.0,
                    new_magnitude=1.0,
                    success=False,
                    error=f"Unknown adjustment type: {adj_type}",
                )

        except Exception as e:
            logger.error(f"Failed to apply adjustment to {vector_name}: {e}")
            return AppliedAdjustment(
                vector_name=vector_name,
                adjustment_type=adj_type,
                original_magnitude=1.0,
                new_magnitude=1.0,
                success=False,
                error=str(e),
            )

    async def _persist_adjustment(
        self,
        adjustment: VectorAdjustment,
        result: AppliedAdjustment,
    ) -> None:
        """Persist an applied adjustment to Psyche.

        Args:
            adjustment: The original adjustment request
            result: The applied adjustment result
        """
        try:
            await self.psyche.upsert_steering_vector({
                "name": adjustment.vector_name,
                "adjustment_type": adjustment.adjustment_type.value,
                "magnitude": result.new_magnitude,
                "reason": adjustment.reason,
                "timestamp": result.to_dict().get("timestamp", datetime.now(timezone.utc).isoformat()),
            })
            logger.debug(f"Persisted adjustment for {adjustment.vector_name} to Psyche")
        except Exception as e:
            logger.warning(f"Failed to persist adjustment to Psyche: {e}")
