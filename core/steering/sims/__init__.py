"""SIMS (Self-Improvement through Model Steering) loop components.

The SIMS loop is the core self-steering mechanism:
1. OBSERVING: Monitor activations, detect surprise/drift
2. REFLECTING: Analyze patterns, identify needed adjustments
3. EXECUTING: Apply steering vector modifications
4. VALIDATING: Verify changes improve alignment with goals

Each state has handlers that can be customized for different
steering strategies.
"""

from core.steering.sims.state_machine import (
    SIMSState,
    SIMSStateMachine,
    StateTransition,
    SIMSContext,
)
from core.steering.sims.reflector import (
    AdjustmentType,
    VectorAdjustment,
    ReflectionResult,
    SIMSReflector,
)
from core.steering.sims.executor import (
    AppliedAdjustment,
    ExecutionResult,
    SIMSExecutor,
)
from core.steering.sims.validator import (
    ValidationOutcome,
    ValidationMetric,
    ValidationResult,
    SIMSValidator,
)

__all__ = [
    "SIMSState",
    "SIMSStateMachine",
    "StateTransition",
    "SIMSContext",
    "AdjustmentType",
    "VectorAdjustment",
    "ReflectionResult",
    "SIMSReflector",
    "AppliedAdjustment",
    "ExecutionResult",
    "SIMSExecutor",
    "ValidationOutcome",
    "ValidationMetric",
    "ValidationResult",
    "SIMSValidator",
]
