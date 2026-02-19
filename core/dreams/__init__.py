"""Dream cycles: tiered consolidation (micro, nap, full, deep)."""

from core.dreams.base import (
    DreamDepth,
    DreamContext,
    DreamInsight,
    DreamResult,
    BaseDream,
    DreamCycleError,
    SURPRISE_THRESHOLD_MICRO,
    SURPRISE_THRESHOLD_HIGH,
    PATTERN_DETECTION_MIN_SAMPLES,
)
from core.dreams.micro_dream import (
    MicroDream,
    trigger_micro_dream,
)

__all__ = [
    # Base types
    "DreamDepth",
    "DreamContext",
    "DreamInsight",
    "DreamResult",
    "BaseDream",
    "DreamCycleError",
    # Thresholds
    "SURPRISE_THRESHOLD_MICRO",
    "SURPRISE_THRESHOLD_HIGH",
    "PATTERN_DETECTION_MIN_SAMPLES",
    # Micro-dream
    "MicroDream",
    "trigger_micro_dream",
]
