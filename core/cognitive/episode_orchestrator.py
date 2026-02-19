"""Episode orchestration for the cognitive loop.

Coordinates episode lifecycle: selection, segment routing, transitions,
and saturation detection.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from core.cognitive.episode import Episode, EpisodeType, SegmentType
from core.cognitive.episode_saturation import EpisodeSaturation
from core.cognitive.episode_selector import EmotionalSignals, EpisodeSelector
from core.cognitive.segment_prompts import build_segment_prompt
from core.cognitive.segment_router import SegmentRouter
from core.cognitive.signal_collector import collect_signals

if TYPE_CHECKING:
    from core.cognitive.state import CognitiveState
    from core.self_model.affective_system import AffectiveState


@dataclass
class EpisodeHistory:
    """Record of a completed episode.

    Attributes:
        episode_type: The type of the episode
        started_at: When the episode began
        ended_at: When the episode concluded
        segment_count: Number of segments completed
    """

    episode_type: EpisodeType
    started_at: datetime
    ended_at: datetime
    segment_count: int


class EpisodeOrchestrator:
    """Orchestrates episode lifecycle within the cognitive loop.

    Responsibilities:
    - Detecting when a new episode should start
    - Selecting the next episode type based on cognitive signals
    - Routing between segments within an episode
    - Detecting episode saturation
    - Managing episode transitions

    The orchestrator is stateful, tracking episode history for timing
    calculations used in episode selection.
    """

    def __init__(self):
        """Initialize the orchestrator with default components."""
        self._selector = EpisodeSelector()
        self._router = SegmentRouter()
        self._saturation = EpisodeSaturation()
        self._history: list[EpisodeHistory] = []

    @property
    def episode_count(self) -> int:
        """Return the number of completed episodes."""
        return len(self._history)

    def should_start_new_episode(
        self,
        state: "CognitiveState",
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if a new episode should start.

        Returns True if:
        - No episode is currently active
        - Current episode is saturated

        Args:
            state: Current cognitive state
            now: Current time (defaults to now)

        Returns:
            True if a new episode should begin
        """
        if state.current_episode is None:
            return True

        return self._saturation.is_saturated(state.current_episode, now=now)

    def start_episode(
        self,
        state: "CognitiveState",
        opening_insight: str,
        episode_type: Optional[EpisodeType] = None,
        seed_entity: Optional[str] = None,
        affect_state: Optional["AffectiveState"] = None,
    ) -> tuple["CognitiveState", Episode]:
        """Start a new episode.

        Args:
            state: Current cognitive state
            opening_insight: The insight/thought that seeds the episode
            episode_type: Optional specific episode type (auto-selected if None)
            seed_entity: Optional entity from knowledge graph that seeds this episode
            affect_state: Optional affective state for emotion-modulated selection

        Returns:
            Tuple of (updated state, new episode)
        """
        # Select episode type if not specified
        if episode_type is None:
            signals = collect_signals(state, episode_tracker=self)
            # Build emotional signals from affective state if available
            emotional_signals = None
            if affect_state is not None:
                emotional_signals = EmotionalSignals(
                    is_bored=affect_state.is_bored(),
                    diversity_signal=affect_state.cognitive_diversity_signal(),
                    is_conflicted=len(affect_state.detect_conflicts()) > 0,
                )
            episode_type = self._selector.select(signals, emotional_signals)

        # Create the new episode
        episode = Episode(
            episode_type=episode_type,
            current_segment=SegmentType.OPENING,
            opening_insight=opening_insight,
            segments_completed=[],
            seed_entity=seed_entity,
        )

        # Update state with new episode
        new_state = state.with_update(
            thought=state.thought,
            vector=state.vector,
            current_episode=episode,
        )

        return new_state, episode

    def advance_segment(
        self,
        state: "CognitiveState",
        segment_output: str,
    ) -> tuple["CognitiveState", SegmentType]:
        """Advance to the next segment in the current episode.

        Records the output of the current segment, marks it complete,
        and routes to the next segment.

        Args:
            state: Current cognitive state
            segment_output: The generated output for the current segment

        Returns:
            Tuple of (updated state, next segment type)

        Raises:
            ValueError: If no episode is active
        """
        if state.current_episode is None:
            raise ValueError("No episode active to advance")

        episode = state.current_episode
        current = episode.current_segment

        # Record the output
        new_outputs = dict(episode.segment_outputs)
        new_outputs[current] = segment_output

        # Mark current segment as completed (if not already)
        new_completed = list(episode.segments_completed)
        if current not in new_completed:
            new_completed.append(current)

        # Route to next segment
        # Create a temporary episode with updated state for routing
        temp_episode = replace(
            episode,
            current_segment=current,
            segments_completed=new_completed,
            segment_outputs=new_outputs,
        )

        next_segment = self._router.route(temp_episode)

        # Create final updated episode by updating the current segment
        updated_episode = replace(temp_episode, current_segment=next_segment)

        # Update state with modified episode
        new_state = state.with_update(
            thought=state.thought,
            vector=state.vector,
            current_episode=updated_episode,
        )

        return new_state, next_segment

    def end_episode(
        self,
        state: "CognitiveState",
    ) -> tuple["CognitiveState", Optional[Episode]]:
        """End the current episode and record it in history.

        Args:
            state: Current cognitive state

        Returns:
            Tuple of (updated state with no episode, the ended episode)
        """
        ended_episode = state.current_episode

        # Record in history
        if ended_episode is not None:
            history_entry = EpisodeHistory(
                episode_type=ended_episode.episode_type,
                started_at=ended_episode.started_at,
                ended_at=datetime.now(timezone.utc),
                segment_count=len(ended_episode.segments_completed),
            )
            self._history.append(history_entry)

        # Clear episode from state
        new_state = state.with_update(
            thought=state.thought,
            vector=state.vector,
            current_episode=None,
        )

        return new_state, ended_episode

    def get_current_segment_prompt(
        self,
        state: "CognitiveState",
        concept: str,
        cycle: int = 0,
        include_flavor: bool = False,
    ) -> str:
        """Get the prompt for the current segment.

        Args:
            state: Current cognitive state
            concept: The concept to explore
            cycle: Cycle count for template rotation
            include_flavor: Whether to include tonal flavor

        Returns:
            Formatted prompt string for the segment

        Raises:
            ValueError: If no episode is active
        """
        if state.current_episode is None:
            raise ValueError("No episode active")

        return build_segment_prompt(
            state.current_episode,
            concept=concept,
            cycle=cycle,
            include_flavor=include_flavor,
        )

    def should_force_synthesis(
        self,
        state: "CognitiveState",
        now: Optional[datetime] = None,
    ) -> bool:
        """Check if the episode should be forced to synthesis.

        Args:
            state: Current cognitive state
            now: Current time (defaults to now)

        Returns:
            True if synthesis should be forced
        """
        if state.current_episode is None:
            return False

        return self._saturation.should_force_synthesis(state.current_episode, now=now)

    # EpisodeTracker protocol implementation for signal collection

    def time_since_last_synthesis(self) -> float:
        """Return seconds since last synthesis episode.

        Part of the EpisodeTracker protocol.
        """
        for entry in reversed(self._history):
            if entry.episode_type == EpisodeType.SYNTHESIS:
                return (datetime.now(timezone.utc) - entry.ended_at).total_seconds()
        return 0.0  # No synthesis episodes yet

    def time_since_last_reflection(self) -> float:
        """Return seconds since last meta-reflection episode.

        Part of the EpisodeTracker protocol.
        """
        for entry in reversed(self._history):
            if entry.episode_type == EpisodeType.META_REFLECTION:
                return (datetime.now(timezone.utc) - entry.ended_at).total_seconds()
        return 0.0  # No reflection episodes yet
