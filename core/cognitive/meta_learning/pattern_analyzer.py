"""Pattern analysis for autonomous decision history."""

import logging
from typing import TYPE_CHECKING

from core.cognitive.meta_learning.schemas import PatternStats

if TYPE_CHECKING:
    from core.psyche.schema import AutonomousDecision

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Analyzes decision history to identify patterns by action type.

    Groups decisions by action_type extracted from action dict,
    calculates success rates, and produces PatternStats for
    heuristic extraction.
    """

    def analyze_by_action_type(
        self, decisions: list["AutonomousDecision"]
    ) -> dict[str, PatternStats]:
        """Group decisions by action type, calculate success rates.

        Only includes decisions with success != None (resolved decisions).

        Args:
            decisions: List of AutonomousDecision objects to analyze

        Returns:
            dict mapping action_type -> PatternStats
        """
        # Group by action type
        grouped: dict[str, list["AutonomousDecision"]] = {}

        for decision in decisions:
            # Skip pending decisions (success = None)
            if decision.success is None:
                continue

            # Extract action type from action dict
            action_type = decision.action.get("type", "unknown")

            if action_type not in grouped:
                grouped[action_type] = []
            grouped[action_type].append(decision)

        # Calculate stats for each group
        patterns: dict[str, PatternStats] = {}

        for action_type, group_decisions in grouped.items():
            count = len(group_decisions)
            successes = sum(1 for d in group_decisions if d.success)
            success_rate = successes / count if count > 0 else 0.0
            example_ids = [d.id for d in group_decisions]

            patterns[action_type] = PatternStats(
                action_type=action_type,
                count=count,
                success_rate=success_rate,
                example_ids=example_ids,
            )

        logger.info(f"Analyzed {len(decisions)} decisions into {len(patterns)} patterns")
        return patterns
