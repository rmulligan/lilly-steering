"""Outcome assessment for autonomous decisions."""

import logging
from core.cognitive.reflexion.consequence_schemas import OutcomeAssessment
from core.cognitive.reflexion.schemas import HealthCategory
from core.psyche.schema import AutonomousDecision

logger = logging.getLogger(__name__)


class OutcomeAssessor:
    """Assesses the outcome of autonomous decisions by comparing health states.

    Success = health improved or stayed STABLE/THRIVING
    Failure = health degraded
    """

    def assess(
        self,
        decision: AutonomousDecision,
        health_before: HealthCategory,
        current_health: HealthCategory,
    ) -> OutcomeAssessment:
        """Assess decision outcome by comparing health states.

        Args:
            decision: The autonomous decision being assessed
            health_before: Health state when decision was made
            current_health: Current health state (after observation window)

        Returns:
            OutcomeAssessment with success classification
        """
        # Compare severity levels (lower is better)
        severity_before = health_before.severity
        severity_after = current_health.severity

        # Success if health improved or stayed same (at good levels)
        success = (
            severity_after < severity_before
            or (
                severity_after == severity_before
                and health_before in {HealthCategory.STABLE, HealthCategory.THRIVING}
            )
        )

        logger.debug(
            f"Assessed decision {decision.id}: "
            f"{health_before.value} → {current_health.value} = "
            f"{'SUCCESS' if success else 'FAILURE'}"
        )

        return OutcomeAssessment(
            success=success,
            health_before=health_before,
            health_after=current_health,
        )
