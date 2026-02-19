"""Heuristic extraction from pattern analysis."""

import logging

from core.cognitive.meta_learning.schemas import PatternStats

logger = logging.getLogger(__name__)

# Default quality thresholds for heuristic extraction
DEFAULT_MIN_EXAMPLES = 5
DEFAULT_MIN_SUCCESS_RATE = 0.7


class HeuristicExtractor:
    """Extracts actionable heuristics from patterns by filtering on quality thresholds.

    Converts high-quality patterns (sufficient examples, high success rate)
    into template-based heuristic text with supporting evidence IDs.
    """

    def extract_heuristics(
        self,
        patterns: dict[str, PatternStats],
        min_examples: int = DEFAULT_MIN_EXAMPLES,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ) -> list[tuple[str, list[str]]]:
        """Filter patterns by quality thresholds and generate heuristic text.

        Args:
            patterns: Dictionary mapping action_type to PatternStats
            min_examples: Minimum number of examples required (default: 5)
            min_success_rate: Minimum success rate required (default: 0.7)

        Returns:
            List of (heuristic_text, evidence_ids) tuples for patterns meeting thresholds
        """
        heuristics: list[tuple[str, list[str]]] = []

        for action_type, stats in patterns.items():
            # Apply quality thresholds
            if stats.count < min_examples:
                logger.debug(
                    f"Excluding {action_type}: count {stats.count} < min {min_examples}"
                )
                continue

            if stats.success_rate < min_success_rate:
                logger.debug(
                    f"Excluding {action_type}: success rate {stats.success_rate:.2f} < min {min_success_rate:.2f}"
                )
                continue

            # Generate heuristic text from template (inline, matching plan spec)
            success_pct = int(stats.success_rate * 100)
            heuristic = (
                f"Action type '{action_type}' tends to succeed "
                f"({success_pct}% success over {stats.count} examples)"
            )
            heuristics.append((heuristic, stats.example_ids))

        logger.info(
            f"Extracted {len(heuristics)} heuristics from {len(patterns)} patterns "
            f"(thresholds: {min_examples} examples, {min_success_rate:.0%} success)"
        )
        return heuristics
