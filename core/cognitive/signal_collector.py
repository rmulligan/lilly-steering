"""Signal collection from cognitive state for episode selection.

Extracts SelectionSignals from CognitiveState to inform the EpisodeSelector
about the current cognitive context.
"""

from typing import TYPE_CHECKING, Any, Optional, Protocol

from core.cognitive.episode_selector import SelectionSignals

if TYPE_CHECKING:
    from core.cognitive.state import CognitiveState


class EpisodeTracker(Protocol):
    """Protocol for tracking episode history and timing.

    This protocol allows for different implementations of episode tracking,
    such as in-memory tracking or persistent storage via PsycheClient.
    """

    def time_since_last_synthesis(self) -> float:
        """Return seconds since last synthesis episode."""
        ...

    def time_since_last_reflection(self) -> float:
        """Return seconds since last meta-reflection episode."""
        ...


# Scaling factor for tension: consecutive_low_surprise of 10 = tension 1.0
TENSION_SCALE_FACTOR = 10.0


def collect_signals(
    state: "CognitiveState",
    episode_tracker: Optional[EpisodeTracker] = None,
) -> SelectionSignals:
    """Collect selection signals from cognitive state.

    Extracts the following signals:
    - open_questions: From state.current_question
    - entity_focus: Highest-salience concept from state.active_concepts
    - recent_insight_count: Length of state.recent_zettel_uids
    - tension_level: Derived from state.consecutive_low_surprise
    - last_episode_type: From state.current_episode (if any)
    - time_since_synthesis: From episode_tracker (or 0.0)
    - time_since_reflection: From episode_tracker (or 0.0)

    Args:
        state: The current cognitive state
        episode_tracker: Optional tracker for episode timing history

    Returns:
        SelectionSignals populated with current context
    """
    # Extract open questions
    open_questions = []
    if state.current_question:
        open_questions.append(state.current_question)

    # Extract entity focus from highest-salience active concept
    entity_focus = None
    if state.active_concepts:
        # active_concepts is list of (concept, salience) tuples
        # Get the concept with highest salience
        entity_focus = max(state.active_concepts, key=lambda x: x[1])[0]

    # Extract last episode type
    last_episode_type = None
    if state.current_episode is not None:
        last_episode_type = state.current_episode.episode_type.value

    # Calculate tension level from consecutive_low_surprise
    # Higher consecutive_low_surprise = exploration is stale = needs tension/change
    tension_level = min(state.consecutive_low_surprise / TENSION_SCALE_FACTOR, 1.0)

    # Count recent insights
    recent_insight_count = len(state.recent_zettel_uids)

    # Get timing from episode tracker if available
    time_since_synthesis = 0.0
    time_since_reflection = 0.0
    if episode_tracker is not None:
        time_since_synthesis = episode_tracker.time_since_last_synthesis()
        time_since_reflection = episode_tracker.time_since_last_reflection()

    return SelectionSignals(
        open_questions=open_questions,
        entity_focus=entity_focus,
        recent_insight_count=recent_insight_count,
        tension_level=tension_level,
        last_episode_type=last_episode_type,
        time_since_synthesis=time_since_synthesis,
        time_since_reflection=time_since_reflection,
    )
