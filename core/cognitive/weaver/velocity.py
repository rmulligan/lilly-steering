"""Cognitive Velocity Tracker: Modeling the Dynamics of User Focus.

This module implements the first-order derivative (velocity) of the user's
cognitive state, enabling prediction of their trajectory through the
knowledge graph.

Generalized Coordinates Background:
    In Continuous Time Active Inference, state is represented not just as
    position (x) but as generalized coordinates: {x, x', x'', ...}

    - x  (Position): Current focus topic/node
    - x' (Velocity): Rate of focus change (fast browsing vs. deep reading)
    - x'' (Acceleration): Change in velocity (speeding up vs. slowing down)

    By tracking velocity, the Weaver can *anticipate* where the user is going,
    not just react to where they are.

Key Metrics:
    - Focus Velocity: How fast is the user traversing topics?
    - Topic Drift: How far has focus moved from the starting point?
    - Engagement Depth: Is the user drilling down or browsing broadly?
    - Session Momentum: Is the pace increasing or decreasing?
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Momentum(Enum):
    """Direction of cognitive momentum.

    Maps to user behavior patterns:
    - ACCELERATING: Getting faster (exploring, hunting)
    - STEADY: Constant pace
    - DECELERATING: Slowing down (focusing, confusion, or fatigue)
    - STATIONARY: Not moving (deep reading or idle)
    """

    ACCELERATING = "accelerating"
    STEADY = "steady"
    DECELERATING = "decelerating"
    STATIONARY = "stationary"


class FocusMode(Enum):
    """Inferred cognitive mode based on velocity patterns.

    These modes inform the Weaver's intervention strategy:
    - SPRINTING: Rapid traversal, don't interrupt
    - EXPLORING: Moderate pace, light suggestions welcome
    - FOCUSING: Deep reading, save insights for later
    - STUCK: Very slow, possible confusion - offer help
    - DORMANT: No activity, pause tracking
    """

    SPRINTING = "sprinting"  # >2 topics/minute
    EXPLORING = "exploring"  # 0.5-2 topics/minute
    FOCUSING = "focusing"  # 0.1-0.5 topics/minute
    STUCK = "stuck"  # <0.1 topics/minute, short intervals
    DORMANT = "dormant"  # No observations


@dataclass
class FocusObservation:
    """A single observation of user focus.

    Attributes:
        topic: Topic or node UID the user is focused on
        timestamp: When the observation was made
        duration: Optional time spent on this topic (if known)
        depth: Optional depth indicator (0=surface, 1=deep)
        semantic_distance: Optional distance from previous topic
    """

    topic: str
    timestamp: datetime
    duration: Optional[float] = None  # seconds
    depth: float = 0.5  # 0-1 scale
    semantic_distance: Optional[float] = None  # from previous


@dataclass
class VelocityState:
    """Current cognitive velocity state.

    Attributes:
        velocity: Topics per minute (instantaneous)
        avg_velocity: Exponential moving average velocity
        momentum: Direction of velocity change
        focus_mode: Inferred cognitive mode
        topic_drift: Cumulative semantic distance traveled
        observation_count: Number of observations in window
        session_duration: Time since first observation
        last_observation: Most recent observation
        confidence: Confidence in velocity estimate (0-1)
    """

    velocity: float
    avg_velocity: float
    momentum: Momentum
    focus_mode: FocusMode
    topic_drift: float
    observation_count: int
    session_duration: float  # seconds
    last_observation: Optional[FocusObservation]
    confidence: float

    @property
    def is_fast(self) -> bool:
        """User is moving quickly through topics."""
        return self.focus_mode == FocusMode.SPRINTING

    @property
    def is_slow(self) -> bool:
        """User is moving slowly (focusing or stuck)."""
        return self.focus_mode in (FocusMode.FOCUSING, FocusMode.STUCK)

    @property
    def is_speeding_up(self) -> bool:
        """Velocity is increasing."""
        return self.momentum == Momentum.ACCELERATING

    @property
    def is_slowing_down(self) -> bool:
        """Velocity is decreasing."""
        return self.momentum == Momentum.DECELERATING

    def to_dict(self) -> dict:
        """Serialize for logging and persistence."""
        return {
            "velocity": self.velocity,
            "avg_velocity": self.avg_velocity,
            "momentum": self.momentum.value,
            "focus_mode": self.focus_mode.value,
            "topic_drift": self.topic_drift,
            "observation_count": self.observation_count,
            "session_duration": self.session_duration,
            "confidence": self.confidence,
            "last_topic": (
                self.last_observation.topic if self.last_observation else None
            ),
        }


# Velocity thresholds (topics per minute)
VELOCITY_SPRINT = 2.0  # >2 topics/min = sprinting
VELOCITY_EXPLORE = 0.5  # 0.5-2 = exploring
VELOCITY_FOCUS = 0.1  # 0.1-0.5 = focusing
VELOCITY_STUCK = 0.05  # <0.1 = stuck (if session is active)

# Momentum detection thresholds
MOMENTUM_THRESHOLD = 0.3  # 30% change in velocity = momentum shift

# Exponential moving average decay
EMA_ALPHA = 0.3  # Weight for new observations (0.3 = moderate smoothing)

# Maximum observation window
MAX_OBSERVATIONS = 100
OBSERVATION_WINDOW_SECONDS = 600  # 10 minutes


@dataclass
class CognitiveVelocityTracker:
    """Tracks the velocity of user focus through the knowledge graph.

    This tracker maintains a sliding window of focus observations and
    computes velocity metrics. It provides the x' (velocity) component
    of the generalized coordinates for Active Inference.

    Note on Resolution:
        Without real-time event streaming, velocity is computed from
        discrete observations. This means we detect patterns after they
        occur, not during. The EMA smoothing helps reduce noise but
        cannot eliminate the inherent latency.

    Attributes:
        observations: Sliding window of recent observations
        ema_velocity: Exponentially smoothed velocity
        prev_velocity: Previous velocity (for momentum calculation)
        session_start: When tracking began
    """

    observations: deque = field(
        default_factory=lambda: deque(maxlen=MAX_OBSERVATIONS)
    )
    ema_velocity: float = 0.0
    prev_velocity: float = 0.0
    session_start: Optional[datetime] = None
    cumulative_drift: float = 0.0
    _now_override: Optional[datetime] = None

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    def record_observation(
        self,
        topic: str,
        timestamp: Optional[datetime] = None,
        duration: Optional[float] = None,
        depth: float = 0.5,
        semantic_distance: Optional[float] = None,
    ) -> VelocityState:
        """Record a focus observation and update velocity state.

        Args:
            topic: Topic or node UID the user is focused on
            timestamp: When observation occurred (default: now)
            duration: Optional time spent on topic
            depth: Engagement depth (0=surface, 1=deep)
            semantic_distance: Distance from previous topic

        Returns:
            Updated VelocityState
        """
        if timestamp is None:
            timestamp = self._get_now()

        # Initialize session if needed
        if self.session_start is None:
            self.session_start = timestamp

        # Create observation
        observation = FocusObservation(
            topic=topic,
            timestamp=timestamp,
            duration=duration,
            depth=depth,
            semantic_distance=semantic_distance,
        )

        # Update cumulative drift if semantic distance provided
        if semantic_distance is not None:
            self.cumulative_drift += semantic_distance

        # Add to window
        self.observations.append(observation)

        # Prune old observations
        self._prune_old_observations(timestamp)

        # Compute new velocity using the observation timestamp
        return self.get_velocity(timestamp)

    def get_velocity(self, current_time: Optional[datetime] = None) -> VelocityState:
        """Compute current cognitive velocity state.

        Args:
            current_time: Reference time for calculations (default: now)

        Returns:
            VelocityState with all velocity metrics
        """
        if current_time is None:
            current_time = self._get_now()

        if len(self.observations) < 2:
            return VelocityState(
                velocity=0.0,
                avg_velocity=0.0,
                momentum=Momentum.STATIONARY,
                focus_mode=FocusMode.DORMANT,
                topic_drift=self.cumulative_drift,
                observation_count=len(self.observations),
                session_duration=self._get_session_duration(),
                last_observation=(
                    self.observations[-1] if self.observations else None
                ),
                confidence=0.3,
            )

        # Compute instantaneous velocity
        instant_velocity = self._compute_instantaneous_velocity()

        # Update EMA
        self.prev_velocity = self.ema_velocity
        self.ema_velocity = (
            EMA_ALPHA * instant_velocity + (1 - EMA_ALPHA) * self.ema_velocity
        )

        # Determine momentum
        momentum = self._compute_momentum(instant_velocity)

        # Determine focus mode
        focus_mode = self._determine_focus_mode(self.ema_velocity)

        # Compute confidence based on data quality
        confidence = self._compute_confidence(current_time)

        return VelocityState(
            velocity=instant_velocity,
            avg_velocity=self.ema_velocity,
            momentum=momentum,
            focus_mode=focus_mode,
            topic_drift=self.cumulative_drift,
            observation_count=len(self.observations),
            session_duration=self._get_session_duration(),
            last_observation=self.observations[-1] if self.observations else None,
            confidence=confidence,
        )

    def _compute_instantaneous_velocity(self) -> float:
        """Compute instantaneous velocity from recent observations.

        Velocity = (unique topics) / (time window in minutes)
        """
        if len(self.observations) < 2:
            return 0.0

        # Get observations in recent window
        recent = list(self.observations)[-10:]  # Last 10 observations

        if len(recent) < 2:
            return 0.0

        # Time span
        time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()

        if time_span < 1:  # Less than 1 second
            return 0.0

        # Count unique topics
        unique_topics = len(set(obs.topic for obs in recent))

        # Velocity in topics per minute
        velocity = (unique_topics / time_span) * 60

        return velocity

    def _compute_momentum(self, current_velocity: float) -> Momentum:
        """Determine momentum direction from velocity change."""
        if self.ema_velocity < 0.01:  # Near zero
            return Momentum.STATIONARY

        # Compute relative change
        if self.prev_velocity > 0:
            relative_change = (
                (current_velocity - self.prev_velocity) / self.prev_velocity
            )
        else:
            relative_change = 1.0 if current_velocity > 0.1 else 0.0

        if relative_change > MOMENTUM_THRESHOLD:
            return Momentum.ACCELERATING
        elif relative_change < -MOMENTUM_THRESHOLD:
            return Momentum.DECELERATING
        else:
            return Momentum.STEADY

    def _determine_focus_mode(self, velocity: float) -> FocusMode:
        """Infer cognitive mode from velocity."""
        if velocity >= VELOCITY_SPRINT:
            return FocusMode.SPRINTING
        elif velocity >= VELOCITY_EXPLORE:
            return FocusMode.EXPLORING
        elif velocity >= VELOCITY_FOCUS:
            return FocusMode.FOCUSING
        elif velocity >= VELOCITY_STUCK and len(self.observations) > 3:
            return FocusMode.STUCK
        else:
            return FocusMode.DORMANT

    def _compute_confidence(self, current_time: datetime) -> float:
        """Compute confidence in velocity estimate based on data quality.

        Args:
            current_time: Reference time for age calculations
        """
        # Base confidence from observation count
        obs_confidence = min(1.0, len(self.observations) / 10)

        # Recency factor
        if self.observations:
            last_obs = self.observations[-1]
            age = (current_time - last_obs.timestamp).total_seconds()
            recency = max(0, 1 - age / 300)  # Decays over 5 minutes
        else:
            recency = 0

        return 0.3 + 0.4 * obs_confidence + 0.3 * recency

    def _prune_old_observations(self, current_time: datetime) -> None:
        """Remove observations older than the window."""
        cutoff = current_time - timedelta(seconds=OBSERVATION_WINDOW_SECONDS)

        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

    def _get_session_duration(self) -> float:
        """Get session duration in seconds."""
        if not self.session_start:
            return 0.0

        if self.observations:
            latest = self.observations[-1].timestamp
        else:
            latest = self._get_now()

        return (latest - self.session_start).total_seconds()

    def reset(self) -> None:
        """Reset the tracker for a new session."""
        self.observations.clear()
        self.ema_velocity = 0.0
        self.prev_velocity = 0.0
        self.session_start = None
        self.cumulative_drift = 0.0

    def get_trajectory_summary(self) -> dict:
        """Get a summary of the cognitive trajectory for logging."""
        velocity_state = self.get_velocity()

        return {
            "session_duration_minutes": velocity_state.session_duration / 60,
            "observations": velocity_state.observation_count,
            "avg_velocity": velocity_state.avg_velocity,
            "current_mode": velocity_state.focus_mode.value,
            "momentum": velocity_state.momentum.value,
            "topic_drift": velocity_state.topic_drift,
            "confidence": velocity_state.confidence,
            "unique_topics": len(set(obs.topic for obs in self.observations)),
        }


def create_velocity_tracker(
    now: Optional[datetime] = None,
) -> CognitiveVelocityTracker:
    """Factory function to create a new velocity tracker.

    Usage:
        from core.cognitive.weaver import create_velocity_tracker

        tracker = create_velocity_tracker()
        tracker.record_observation("machine_learning")
        velocity = tracker.get_velocity()
    """
    tracker = CognitiveVelocityTracker()
    tracker._now_override = now
    return tracker


def interpret_velocity_for_weaver(velocity_state: VelocityState) -> str:
    """Generate guidance for the Weaver based on velocity state.

    This helps the Weaver decide when to intervene.
    """
    if velocity_state.focus_mode == FocusMode.SPRINTING:
        return "User is rapidly exploring. Defer all interventions until they slow down."

    if velocity_state.focus_mode == FocusMode.EXPLORING:
        if velocity_state.momentum == Momentum.ACCELERATING:
            return (
                "User is speeding up - likely hunting for something. "
                "Light suggestions only."
            )
        return "Moderate exploration pace. Ambient suggestions are appropriate."

    if velocity_state.focus_mode == FocusMode.FOCUSING:
        if velocity_state.momentum == Momentum.DECELERATING:
            return (
                "User is settling into deep reading. "
                "Queue insights for later surfacing."
            )
        return "Deep focus mode. Minimize interruptions."

    if velocity_state.focus_mode == FocusMode.STUCK:
        return (
            "User may be confused or overwhelmed. "
            "Consider offering a navigation hint."
        )

    return "Insufficient data to assess cognitive velocity."
