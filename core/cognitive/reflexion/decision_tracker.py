"""Decision tracking for consequence learning."""

import logging
from typing import TYPE_CHECKING

from core.cognitive.reflexion.schemas import HealthCategory
from core.psyche.schema import AutonomousDecision

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


class DecisionTracker:
    """Manages observation windows and graph queries for consequence learning.

    Queries the graph for decisions that are ready for assessment (10 cycles old),
    retrieves historical health states, and updates decision outcomes.
    """

    def __init__(self, psyche: "PsycheClient"):
        """Initialize with PsycheClient dependency.

        Args:
            psyche: PsycheClient for graph operations
        """
        self._psyche = psyche

    async def get_pending_decisions(
        self,
        current_cycle: int,
        offset: int = 10,
        limit: int = 5,
    ) -> list[AutonomousDecision]:
        """Get decisions ready for assessment.

        Queries for decisions at least `offset` cycles old that don't
        have outcomes yet. Uses age-based querying for robustness.

        Args:
            current_cycle: Current cycle number
            offset: Minimum observation window in cycles (default 10)
            limit: Maximum decisions to return (default 5)

        Returns:
            List of AutonomousDecision nodes ready for assessment
        """
        logger.debug(
            f"Querying for pending decisions (current: {current_cycle}, "
            f"min age: {offset} cycles)"
        )

        return await self._psyche.get_pending_decisions(
            current_cycle=current_cycle,
            offset=offset,
            limit=limit,
        )

    async def get_health_at_cycle(self, cycle_id: str) -> HealthCategory | None:
        """Get health state from a specific cycle.

        Args:
            cycle_id: Cycle identifier (e.g., "cycle_100")

        Returns:
            HealthCategory if journal entry exists, None otherwise
        """
        return await self._psyche.get_health_at_cycle(cycle_id)

    async def update_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        lesson: str,
        success: bool,
        cycle_count_assessed: int,
    ) -> None:
        """Update decision node with outcome assessment.

        Args:
            decision_id: Decision node ID
            outcome: Outcome classification
            lesson: Extracted lesson text
            success: Whether the decision led to successful outcome
            cycle_count_assessed: Cycle number when outcome was assessed
        """
        await self._psyche.update_decision_outcome(
            decision_id=decision_id,
            outcome=outcome,
            lesson=lesson,
            success=success,
            cycle_count_assessed=cycle_count_assessed,
        )

    def _calculate_target_cycle_id(self, current_cycle: int, offset: int) -> str:
        """Calculate target cycle ID for observation window.

        Args:
            current_cycle: Current cycle number
            offset: Observation window in cycles

        Returns:
            Cycle ID string (e.g., "cycle_100")
        """
        target_cycle = current_cycle - offset
        return f"cycle_{target_cycle}"
