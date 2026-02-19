"""Episode-level saturation detection.

Detects when an entire episode has reached saturation and should
transition to a new episode, based on segment completion, time,
and content quality signals.
"""

from datetime import datetime, timezone

from core.cognitive.episode import Episode, SegmentType


class EpisodeSaturation:
    """Detects when an episode has reached saturation.

    An episode is saturated when:
    - All required phases are complete (OPENING, method segments, SYNTHESIS, CLOSING)
    - Maximum time has elapsed
    - Content quality signals diminishing returns

    The saturation check ensures episodes have proper structure
    while allowing natural variation in length.
    """

    # Time thresholds in seconds
    MIN_EPISODE_DURATION = 60.0  # 1 minute minimum
    MAX_EPISODE_DURATION = 600.0  # 10 minutes maximum

    def is_saturated(self, episode: Episode, now: datetime | None = None) -> bool:
        """Check if an episode has reached saturation.

        Args:
            episode: The current episode state
            now: Current time (defaults to now)

        Returns:
            True if the episode is saturated and should transition
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Not saturated if opening hasn't happened
        if SegmentType.OPENING not in episode.segments_completed:
            return False

        # Not saturated below minimum duration
        duration = (now - episode.started_at).total_seconds()
        if duration < self.MIN_EPISODE_DURATION:
            return False

        # Forced saturation at maximum duration
        if duration >= self.MAX_EPISODE_DURATION:
            return True

        # Saturated if closing is complete
        if SegmentType.CLOSING in episode.segments_completed:
            return True

        # Saturated if synthesis is complete and enough time has passed
        if SegmentType.SYNTHESIS in episode.segments_completed:
            if duration >= self.MIN_EPISODE_DURATION * 2:  # 2 minutes
                return True

        return False

    def saturation_progress(self, episode: Episode, now: datetime | None = None) -> float:
        """Calculate progress toward episode saturation.

        Args:
            episode: The current episode state
            now: Current time (defaults to now)

        Returns:
            Score from 0.0 (just started) to 1.0 (saturated)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        progress = 0.0
        config = episode.get_config()

        # Time contribution (0-0.4)
        duration = (now - episode.started_at).total_seconds()
        time_ratio = min(duration / self.MAX_EPISODE_DURATION, 1.0)
        progress += time_ratio * 0.4

        # Phase completion contribution (0-0.4)
        phases_complete = 0.0
        if SegmentType.OPENING in episode.segments_completed:
            phases_complete += 0.25
        if SegmentType.SYNTHESIS in episode.segments_completed:
            phases_complete += 0.25
        if SegmentType.CLOSING in episode.segments_completed:
            phases_complete += 0.25

        # Method segments contribution
        method_segments = [
            s for s in episode.segments_completed
            if s not in (SegmentType.OPENING, SegmentType.SYNTHESIS, SegmentType.CLOSING)
        ]
        method_ratio = min(len(method_segments) / config.default_max_segments, 1.0)
        phases_complete += method_ratio * 0.25

        progress += phases_complete * 0.4

        # Content volume contribution (0-0.2)
        output_count = len(episode.segment_outputs)
        output_ratio = min(output_count / 10, 1.0)  # Max 10 outputs
        progress += output_ratio * 0.2

        return min(progress, 1.0)

    def should_force_synthesis(self, episode: Episode, now: datetime | None = None) -> bool:
        """Check if episode should be forced to synthesis phase.

        Args:
            episode: The current episode state
            now: Current time (defaults to now)

        Returns:
            True if episode should transition to synthesis
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Already did synthesis
        if SegmentType.SYNTHESIS in episode.segments_completed:
            return False

        # Force synthesis at 80% of max duration
        duration = (now - episode.started_at).total_seconds()
        if duration >= self.MAX_EPISODE_DURATION * 0.8:
            return True

        # Force synthesis at max method segments
        config = episode.get_config()
        method_segments = [
            s for s in episode.segments_completed
            if s not in (SegmentType.OPENING, SegmentType.SYNTHESIS, SegmentType.CLOSING)
        ]
        if len(method_segments) >= config.default_max_segments:
            return True

        return False
