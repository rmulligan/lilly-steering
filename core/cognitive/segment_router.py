"""Segment routing and saturation detection for episodes.

Handles transitions between segments within an episode, detecting when
a segment has reached saturation and selecting the next segment.
"""

import random
from dataclasses import dataclass

from core.cognitive.episode import Episode, EpisodeConfig, SegmentType, EPISODE_CONFIGS


@dataclass
class SegmentOutput:
    """Output from a segment's cognitive processing.

    Attributes:
        thought: The generated thought/content
        token_count: Number of tokens in the output
        semantic_density: Measure of information density (0.0-1.0)
        repetition_score: How much content repeats previous (0.0-1.0)
        question_emerged: Whether a new question emerged
        insight_emerged: Whether an insight was generated
    """

    thought: str
    token_count: int = 0
    semantic_density: float = 0.5
    repetition_score: float = 0.0
    question_emerged: bool = False
    insight_emerged: bool = False


class SegmentSaturation:
    """Detects when a segment has reached saturation.

    A segment is saturated when:
    - It has generated enough content (token threshold)
    - Content is becoming repetitive (repetition threshold)
    - Semantic density is dropping (diminishing returns)
    - An insight or question naturally concludes the segment

    The saturation check uses a combination of these signals to
    determine when to transition to the next segment.
    """

    # Thresholds for saturation detection
    MIN_TOKENS = 50  # Minimum tokens before saturation possible
    MAX_TOKENS = 500  # Maximum tokens before forced saturation
    REPETITION_THRESHOLD = 0.6  # Above this triggers saturation
    LOW_DENSITY_THRESHOLD = 0.3  # Below this suggests saturation

    def is_saturated(self, output: SegmentOutput, segment_count: int = 1) -> bool:
        """Check if a segment has reached saturation.

        Args:
            output: The segment's output to evaluate
            segment_count: How many outputs have been generated in this segment

        Returns:
            True if the segment is saturated and should transition
        """
        # Natural conclusion if insight or question emerged
        if output.insight_emerged or output.question_emerged:
            return True

        # Not saturated if below minimum tokens
        if output.token_count < self.MIN_TOKENS:
            return False

        # Forced saturation at maximum tokens
        if output.token_count >= self.MAX_TOKENS:
            return True

        # Check for repetition saturation
        if output.repetition_score >= self.REPETITION_THRESHOLD:
            return True

        # Check for low semantic density (diminishing returns)
        if output.semantic_density <= self.LOW_DENSITY_THRESHOLD:
            return True

        # After multiple outputs, lower thresholds
        if segment_count >= 3:
            if output.repetition_score >= 0.4 or output.semantic_density <= 0.4:
                return True

        return False

    def saturation_score(self, output: SegmentOutput) -> float:
        """Calculate a saturation score for the output.

        Args:
            output: The segment's output to evaluate

        Returns:
            Score from 0.0 (fresh) to 1.0 (saturated)
        """
        score = 0.0

        # Token contribution (0-0.3)
        token_ratio = min(output.token_count / self.MAX_TOKENS, 1.0)
        score += token_ratio * 0.3

        # Repetition contribution (0-0.3)
        score += output.repetition_score * 0.3

        # Inverse density contribution (0-0.3)
        inverse_density = 1.0 - output.semantic_density
        score += inverse_density * 0.3

        # Natural conclusion bonus (0.1)
        if output.insight_emerged or output.question_emerged:
            score += 0.1

        return min(score, 1.0)


class SegmentRouter:
    """Routes to the next segment within an episode.

    Decides which segment to run next based on:
    - The episode's toolkit (available segments)
    - Which segments have already been completed
    - The episode's progress (min/max segment constraints)

    The router ensures variety by avoiding immediate repetition
    and respecting the episode's configured segment limits.
    """

    def __init__(self, config: EpisodeConfig | None = None):
        """Initialize the router.

        Args:
            config: Episode configuration (can be set later via route())
        """
        self._config = config

    def route(self, episode: Episode) -> SegmentType:
        """Select the next segment for an episode.

        Args:
            episode: The current episode state

        Returns:
            The next SegmentType to execute
        """
        config = episode.get_config()

        # Count method segments completed (excluding universal segments)
        method_segments_done = [
            s for s in episode.segments_completed
            if s not in (SegmentType.OPENING, SegmentType.SYNTHESIS, SegmentType.CLOSING)
        ]
        method_count = len(method_segments_done)

        # If opening hasn't happened, do opening
        if SegmentType.OPENING not in episode.segments_completed:
            return SegmentType.OPENING

        # If we've hit max segments, go to synthesis
        if method_count >= config.default_max_segments:
            if SegmentType.SYNTHESIS not in episode.segments_completed:
                return SegmentType.SYNTHESIS
            return SegmentType.CLOSING

        # If we've hit min segments and have insights, can go to synthesis
        if method_count >= config.default_min_segments:
            # 30% chance to transition to synthesis after min segments
            if random.random() < 0.3:
                if SegmentType.SYNTHESIS not in episode.segments_completed:
                    return SegmentType.SYNTHESIS

        # Select from toolkit, avoiding recent segment
        return self._select_from_toolkit(config.toolkit, episode.current_segment)

    def _select_from_toolkit(
        self,
        toolkit: set[SegmentType],
        current: SegmentType,
    ) -> SegmentType:
        """Select a segment from the toolkit, avoiding the current one.

        Args:
            toolkit: Available segment types
            current: The current segment (to avoid immediate repetition)

        Returns:
            Selected segment type from the toolkit
        """
        available = [s for s in toolkit if s != current]
        if not available:
            available = list(toolkit)
        return random.choice(available)

    def should_close(self, episode: Episode) -> bool:
        """Check if the episode should transition to closing.

        Args:
            episode: The current episode state

        Returns:
            True if the episode should close
        """
        if SegmentType.SYNTHESIS not in episode.segments_completed:
            return False

        # Close after synthesis is complete
        return True
