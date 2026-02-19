"""Hypothesis lifecycle management.

Implements automated lifecycle transitions and confidence calibration
for hypotheses based on prediction verification outcomes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.cognitive.simulation.schemas import Hypothesis

from core.cognitive.simulation.schemas import HypothesisStatus

logger = logging.getLogger(__name__)

# Lifecycle thresholds
VALIDATION_THRESHOLD = 0.70  # >= 70% verified to validate
INVALIDATION_THRESHOLD = 0.25  # <= 25% verified to invalidate

# Calibration parameters
CALIBRATION_ALPHA = 0.3  # EMA learning rate for calibration error
CONFIDENCE_ADJUSTMENT_RATE = 0.1  # Rate at which confidence moves toward actual rate
CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 0.95


def get_min_predictions(total_cycles: int) -> int:
    """Get minimum predictions required for lifecycle decisions.

    Scales with system maturity - early systems need less evidence,
    mature systems require more rigorous verification.

    Args:
        total_cycles: Total cognitive cycles completed

    Returns:
        Minimum predictions required before lifecycle transitions
    """
    EARLY_STAGE_CYCLES = 50
    MATURE_STAGE_CYCLES = 200
    # Thresholds calibrated to actual prediction generation rates (~2.3 avg, 3 max per hypothesis)
    MIN_PREDICTIONS_EARLY = 2
    MIN_PREDICTIONS_MID = 3
    MIN_PREDICTIONS_MATURE = 5

    if total_cycles < EARLY_STAGE_CYCLES:
        return MIN_PREDICTIONS_EARLY  # Early stage - low bar
    elif total_cycles < MATURE_STAGE_CYCLES:
        return MIN_PREDICTIONS_MID  # Mid stage - moderate evidence
    else:
        return MIN_PREDICTIONS_MATURE  # Mature - strong evidence required


def evaluate_lifecycle_transition(
    hypothesis: "Hypothesis",
    total_cycles: int,
) -> HypothesisStatus:
    """Evaluate if hypothesis should transition to a new status.

    Args:
        hypothesis: The hypothesis to evaluate
        total_cycles: Total cognitive cycles for adaptive thresholds

    Returns:
        The new status (may be same as current)
    """
    total_evaluated = hypothesis.verified_count + hypothesis.falsified_count
    min_required = get_min_predictions(total_cycles)

    # Check for PROPOSED -> ACTIVE transition
    if hypothesis.status == HypothesisStatus.PROPOSED and total_evaluated > 0:
        return HypothesisStatus.ACTIVE

    # Not enough data for further transitions
    if total_evaluated < min_required:
        return hypothesis.status

    verification_rate = hypothesis.verified_count / total_evaluated

    # Check for validation
    if verification_rate >= VALIDATION_THRESHOLD:
        return HypothesisStatus.VERIFIED

    # Check for invalidation
    if verification_rate <= INVALIDATION_THRESHOLD:
        return HypothesisStatus.FALSIFIED

    # Still evaluating
    return HypothesisStatus.ACTIVE


def update_calibration(
    hypothesis: "Hypothesis",
    verified: bool,
) -> None:
    """Update hypothesis confidence calibration after verification.

    Updates:
    - verified_count or falsified_count
    - calibration_error (EMA of |confidence - actual_rate|)
    - confidence (moves toward actual_rate)

    Args:
        hypothesis: The hypothesis to update (modified in place)
        verified: True if prediction was verified, False if falsified
    """
    # Update counts
    if verified:
        hypothesis.verified_count += 1
    else:
        hypothesis.falsified_count += 1

    total = hypothesis.verified_count + hypothesis.falsified_count
    if total == 0:
        return

    # Calculate actual verification rate
    actual_rate = hypothesis.verified_count / total

    # Update calibration error with EMA
    new_error = abs(hypothesis.confidence - actual_rate)
    hypothesis.calibration_error = (
        CALIBRATION_ALPHA * new_error
        + (1 - CALIBRATION_ALPHA) * hypothesis.calibration_error
    )

    # Adjust confidence toward actual rate
    hypothesis.confidence += CONFIDENCE_ADJUSTMENT_RATE * (
        actual_rate - hypothesis.confidence
    )

    # Clamp to valid range
    hypothesis.confidence = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, hypothesis.confidence))

    logger.debug(
        f"Calibration update for {hypothesis.uid}: "
        f"confidence={hypothesis.confidence:.3f}, "
        f"calibration_error={hypothesis.calibration_error:.3f}, "
        f"actual_rate={actual_rate:.3f}"
    )
