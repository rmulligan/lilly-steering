"""Weaver Control Loop: The Orchestration Layer.

This module implements the complete Active Inference control loop:
    SENSE -> THINK -> ACT -> LEARN

It's the "motor cortex" of the Weaver - converting perceptions into actions
while respecting the user's cognitive state and learning objectives.

Control Flow:
    1. SENSE: Gather observations (Discovery D, Velocity x')
    2. THINK: Select policy via EFE minimization
    3. DECIDE: Should we act? (Hysteresis, user state)
    4. ACT: Generate intervention (bridge, hypothesis, or wait)
    5. LEARN: Record outcome for future improvement

Intervention Surfacing:
    Interventions follow "Calm Technology" principles:
    - Non-intrusive: Side channel, not blocking
    - Contextual: Relevant to current focus
    - Deferrable: User can ignore without consequence
    - Learnable: System improves from acceptance/rejection
"""

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Coroutine, Any, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from services.event_bus import EventBus

from core.cognitive.types import InterventionType, WeaverIntervention, WeaverPolicy

from .discovery import DiscoveryParameter, DiscoveryResult, DiscoveryState
from .velocity import CognitiveVelocityTracker, VelocityState, FocusMode
from .feedback import FeedbackAggregator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Control loop timing
DEFAULT_HEARTBEAT_SECONDS = 30.0  # Fallback tick interval
MIN_INTERVENTION_GAP_SECONDS = 60.0  # Minimum time between interventions

# Truncation settings
SENTENCE_BOUNDARY_SEARCH_WINDOW = 60  # How far back to search for sentence boundaries
MIN_TRUNCATION_RATIO = 0.5  # Minimum ratio of max_length to accept a boundary


def _truncate_at_word_boundary(text: str, max_length: int) -> tuple[str, bool]:
    """Truncate text at a word boundary, returning (truncated_text, was_truncated).

    Prefers sentence boundaries (. ! ?) if available within the last 60 chars.
    Falls back to word boundary (space) if no sentence boundary found.
    """
    if len(text) <= max_length:
        return text, False

    # Look for sentence boundary in the truncation zone
    truncated = text[:max_length]
    sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

    # Search backwards from max_length for a sentence ending
    best_sentence_end = -1
    search_limit = max(0, max_length - SENTENCE_BOUNDARY_SEARCH_WINDOW)
    for i in range(max_length - 1, search_limit, -1):
        for ending in sentence_endings:
            if text[i:i+len(ending)] == ending:
                best_sentence_end = i + 1  # Include the punctuation
                break
        if best_sentence_end > 0:
            break

    min_acceptable_position = int(max_length * MIN_TRUNCATION_RATIO)
    if best_sentence_end > min_acceptable_position:
        # Found a good sentence boundary
        return text[:best_sentence_end].strip(), True

    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > min_acceptable_position:
        return text[:last_space].strip(), True

    # No good boundary found, hard truncate
    return truncated.strip(), True
MAX_PENDING_INTERVENTIONS = 5  # Queue limit

# Intervention thresholds
BRIDGE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for bridge proposals


@dataclass
class ControlLoopState:
    """Current state of the control loop.

    Attributes:
        last_tick: Last control loop tick
        last_intervention: Last intervention generated
        pending_interventions: Queue of pending interventions
        current_policy: Active policy
        discovery: Latest discovery result
        velocity: Latest velocity state
        is_running: Whether the loop is active
    """

    last_tick: Optional[datetime] = None
    last_intervention: Optional[datetime] = None
    pending_interventions: list[WeaverIntervention] = field(default_factory=list)
    current_policy: WeaverPolicy = WeaverPolicy.WAIT
    discovery: Optional[DiscoveryResult] = None
    velocity: Optional[VelocityState] = None
    is_running: bool = False


# Type for intervention callback
InterventionCallback = Callable[[WeaverIntervention], Coroutine[Any, Any, None]]


class WeaverControlLoop:
    """The complete Active Inference control loop.

    Orchestrates sensing, decision-making, and action generation
    for the Weaver system.

    Attributes:
        graph: PsycheClient for knowledge graph
        event_bus: EventBus for event handling
        heartbeat_seconds: Interval for heartbeat ticks
    """

    def __init__(
        self,
        graph: "PsycheClient",
        event_bus: Optional["EventBus"] = None,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        on_intervention: Optional[InterventionCallback] = None,
        now: Optional[datetime] = None,
    ):
        """Initialize the control loop.

        Args:
            graph: PsycheClient for queries
            event_bus: Optional event bus for subscriptions
            heartbeat_seconds: Heartbeat interval
            on_intervention: Callback when intervention is ready
            now: Optional datetime override for testing
        """
        self._now_override = now
        self.graph = graph
        self.event_bus = event_bus
        self.heartbeat_seconds = heartbeat_seconds
        self.on_intervention = on_intervention

        # Initialize components
        # Note: PolicySelector removed - policy is determined via simple state-to-policy mapping
        self.discovery_param = DiscoveryParameter(graph)
        self.velocity_tracker = CognitiveVelocityTracker()
        self.velocity_tracker._now_override = now
        self.feedback_aggregator = FeedbackAggregator(graph)

        # State
        self._state = ControlLoopState()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Lock to prevent concurrent tick() calls
        self._tick_lock = asyncio.Lock()

    def _get_now(self) -> datetime:
        """Get current datetime, using override if set (for testing)."""
        return self._now_override or datetime.now(timezone.utc)

    async def start(self) -> None:
        """Start the control loop."""
        if self._running:
            logger.warning("Control loop already running")
            return

        self._running = True
        self._state.is_running = True

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("Weaver control loop started")

    async def stop(self) -> None:
        """Stop the control loop."""
        self._running = False
        self._state.is_running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        logger.info("Weaver control loop stopped")

    async def _heartbeat_loop(self) -> None:
        """Background task that runs periodic ticks."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_seconds)
                if self._running:
                    await self.tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def tick(self) -> Optional[WeaverIntervention]:
        """Run a single iteration of the control loop.

        This is the main SENSE -> THINK -> ACT -> LEARN cycle.

        Returns:
            WeaverIntervention if one was generated, None otherwise
        """
        async with self._tick_lock:
            return await self._tick_internal()

    async def _tick_internal(self) -> Optional[WeaverIntervention]:
        """Internal tick implementation."""
        now = self._get_now()
        self._state.last_tick = now

        try:
            # =================================================================
            # SENSE: Gather observations
            # =================================================================
            discovery = await self._sense_discovery()
            velocity = self._sense_velocity()

            self._state.discovery = discovery
            self._state.velocity = velocity

            # =================================================================
            # THINK: Select policy
            # =================================================================
            policy = self._think_policy(discovery, velocity)
            self._state.current_policy = policy

            # =================================================================
            # DECIDE: Should we act?
            # =================================================================
            if not self._should_act(policy, velocity):
                return None

            # Check intervention gap (hysteresis)
            if self._state.last_intervention:
                gap = (now - self._state.last_intervention).total_seconds()
                if gap < MIN_INTERVENTION_GAP_SECONDS:
                    logger.debug(
                        f"Skipping intervention, gap {gap:.0f}s < {MIN_INTERVENTION_GAP_SECONDS}s"
                    )
                    return None

            # =================================================================
            # ACT: Generate intervention
            # =================================================================
            intervention = await self._act_generate(policy, discovery)

            if intervention:
                self._state.last_intervention = now
                self._state.pending_interventions.append(intervention)

                # Trim queue if too long
                while len(self._state.pending_interventions) > MAX_PENDING_INTERVENTIONS:
                    self._state.pending_interventions.pop(0)

                # Fire callback
                if self.on_intervention:
                    await self.on_intervention(intervention)

                # Emit event
                if self.event_bus:
                    await self.event_bus.emit(
                        "weaver.intervention.generated",
                        {"intervention": intervention.to_dict()},
                    )

                logger.info(
                    f"Generated intervention: {intervention.intervention_type.value}"
                )

            return intervention

        except Exception as e:
            logger.error(f"Control loop tick error: {e}")
            return None

    # =========================================================================
    # SENSE Phase
    # =========================================================================

    async def _sense_discovery(self) -> DiscoveryResult:
        """Sense the current discovery state of the knowledge graph."""
        try:
            return await self.discovery_param.compute()
        except Exception as e:
            logger.warning(f"Failed to compute discovery: {e}")
            return DiscoveryResult.for_error(f"Failed to compute discovery parameter: {e}")

    def _sense_velocity(self) -> VelocityState:
        """Get current velocity state from tracker."""
        return self.velocity_tracker.get_velocity(self._get_now())

    # =========================================================================
    # THINK Phase
    # =========================================================================

    def _think_policy(
        self,
        discovery: DiscoveryResult,
        velocity: VelocityState,
    ) -> WeaverPolicy:
        """Select policy based on discovery and velocity.

        Uses the PolicySelector with discovery and velocity context.
        """
        # Map discovery state to policy recommendation
        discovery_policy = self._discovery_to_policy(discovery)

        # Adjust for velocity (don't interrupt fast exploration)
        if velocity.focus_mode == FocusMode.SPRINTING:
            return WeaverPolicy.WAIT
        if velocity.focus_mode == FocusMode.FOCUSING:
            # User is deep reading, be gentle
            if discovery_policy == WeaverPolicy.BRIDGE:
                return WeaverPolicy.WAIT  # Defer bridging

        return discovery_policy

    def _discovery_to_policy(self, discovery: DiscoveryResult) -> WeaverPolicy:
        """Map discovery state to weaver policy."""
        state_to_policy = {
            DiscoveryState.EXPLORATION: WeaverPolicy.WAIT,
            DiscoveryState.TENSION: WeaverPolicy.BRIDGE,
            DiscoveryState.STAGNATION: WeaverPolicy.HYPOTHESIS,  # Was TEACHBACK, now HYPOTHESIS to break stagnation
            DiscoveryState.ISOMORPHISM: WeaverPolicy.WAIT,
            DiscoveryState.UNKNOWN: WeaverPolicy.WAIT,
        }
        return state_to_policy.get(discovery.state, WeaverPolicy.WAIT)

    # =========================================================================
    # DECIDE Phase
    # =========================================================================

    def _should_act(
        self,
        policy: WeaverPolicy,
        velocity: VelocityState,
    ) -> bool:
        """Decide whether to generate an intervention.

        Returns True if conditions favor intervention.
        """
        # Never act during WAIT policy
        if policy == WeaverPolicy.WAIT:
            return False

        # Don't act during sprinting
        if velocity.focus_mode == FocusMode.SPRINTING:
            return False

        # Don't act with low confidence velocity
        if velocity.confidence < 0.4:
            return True  # Act when we don't know user state

        # Act during exploring or when stuck
        if velocity.focus_mode in (FocusMode.EXPLORING, FocusMode.STUCK):
            return True

        # Act during dormant with non-wait policy
        if velocity.focus_mode == FocusMode.DORMANT:
            return True

        return False

    # =========================================================================
    # ACT Phase
    # =========================================================================

    async def _act_generate(
        self,
        policy: WeaverPolicy,
        discovery: DiscoveryResult,
    ) -> Optional[WeaverIntervention]:
        """Generate an intervention based on policy.

        Args:
            policy: Selected policy
            discovery: Current discovery state

        Returns:
            WeaverIntervention or None
        """
        if policy == WeaverPolicy.BRIDGE:
            return await self._generate_bridge_intervention(discovery)
        elif policy == WeaverPolicy.HYPOTHESIS:
            return await self._generate_hypothesis_intervention(discovery)
        else:
            return None

    async def _generate_bridge_intervention(
        self,
        discovery: DiscoveryResult,
    ) -> Optional[WeaverIntervention]:
        """Generate a bridge proposal intervention."""
        # Find clusters that could be bridged
        try:
            query = """
                MATCH (a:Fragment), (b:Fragment)
                WHERE a <> b
                AND NOT (a)-[:RELATES_TO]-(b)
                AND a.embedding IS NOT NULL
                AND b.embedding IS NOT NULL
                WITH a, b, rand() as r
                ORDER BY r
                LIMIT 1
                RETURN a.uid as uid_a, a.content as content_a,
                       b.uid as uid_b, b.content as content_b
            """
            result = await self.graph.query(query)

            if not result:
                return None

            row = result[0]
            raw_a = row.get("content_a", "")
            raw_b = row.get("content_b", "")
            content_a, truncated_a = _truncate_at_word_boundary(raw_a, 100)
            content_b, truncated_b = _truncate_at_word_boundary(raw_b, 100)
            ellipsis_a = "..." if truncated_a else ""
            ellipsis_b = "..." if truncated_b else ""

            prompt = (
                f"I notice two areas that might connect: "
                f"'{content_a}{ellipsis_a}' and '{content_b}{ellipsis_b}'. "
                f"Do you see a relationship between them?"
            )

            return WeaverIntervention(
                uid=str(uuid.uuid4()),
                intervention_type=InterventionType.BRIDGE_PROPOSAL,
                policy=WeaverPolicy.BRIDGE,
                prompt=prompt,
                priority=0.6,
                metadata={
                    "uid_a": row.get("uid_a"),
                    "uid_b": row.get("uid_b"),
                    "discovery_state": discovery.state.value,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to generate bridge intervention: {e}")
            return None

    async def _generate_hypothesis_intervention(
        self,
        discovery: DiscoveryResult,
    ) -> Optional[WeaverIntervention]:
        """Generate a hypothesis testing intervention."""
        try:
            # Find an edge fragment for hypothesis testing
            query = """
                MATCH (f:Fragment)
                WHERE f.content IS NOT NULL
                WITH f, rand() as r
                ORDER BY r
                LIMIT 1
                RETURN f.uid as uid, f.content as content
            """
            result = await self.graph.query(query)

            if not result:
                return None

            row = result[0]
            raw_content = row.get("content", "")
            content, was_truncated = _truncate_at_word_boundary(raw_content, 200)
            ellipsis = "..." if was_truncated else ""

            prompt = (
                f"Here's a thought to consider: {content}{ellipsis} "
                f"What implications might this have?"
            )

            return WeaverIntervention(
                uid=str(uuid.uuid4()),
                intervention_type=InterventionType.HYPOTHESIS_TEST,
                policy=WeaverPolicy.HYPOTHESIS,
                prompt=prompt,
                priority=0.4,
                metadata={
                    "fragment_uid": row.get("uid"),
                    "discovery_state": discovery.state.value,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to generate hypothesis intervention: {e}")
            return None

    # =========================================================================
    # LEARN Phase
    # =========================================================================

    async def record_outcome(
        self,
        intervention_uid: str,
        accepted: bool,
        feedback: Optional[str] = None,
    ) -> None:
        """Record the outcome of an intervention for learning.

        Args:
            intervention_uid: UID of the intervention
            accepted: Whether user accepted/engaged with intervention
            feedback: Optional user feedback
        """
        finding = {
            "intervention_uid": intervention_uid,
            "accepted": accepted,
            "feedback": feedback,
            "outcome": "success" if accepted else "rejected",
        }

        await self.feedback_aggregator.aggregate_and_persist(
            findings=[finding],
            goal="intervention_learning",
            source="weaver_control_loop",
        )

        logger.info(
            f"Recorded outcome for intervention {intervention_uid}: "
            f"accepted={accepted}"
        )

    # =========================================================================
    # Observation Recording
    # =========================================================================

    def record_focus(
        self,
        topic: str,
        timestamp: Optional[datetime] = None,
        semantic_distance: Optional[float] = None,
    ) -> None:
        """Record a focus observation for velocity tracking.

        Args:
            topic: Topic or node UID the user is focused on
            timestamp: When observation occurred (default: now)
            semantic_distance: Distance from previous topic
        """
        self.velocity_tracker.record_observation(
            topic=topic,
            timestamp=timestamp or self._get_now(),
            semantic_distance=semantic_distance,
        )

    # =========================================================================
    # State Access
    # =========================================================================

    def get_state(self) -> ControlLoopState:
        """Get current control loop state."""
        return self._state

    def get_pending_intervention(self) -> Optional[WeaverIntervention]:
        """Get the next pending intervention if any."""
        if self._state.pending_interventions:
            return self._state.pending_interventions[0]
        return None

    def pop_pending_intervention(self) -> Optional[WeaverIntervention]:
        """Pop and return the next pending intervention."""
        if self._state.pending_interventions:
            return self._state.pending_interventions.pop(0)
        return None

    def to_dict(self) -> dict:
        """Serialize state for logging/debugging."""
        return {
            "is_running": self._state.is_running,
            "last_tick": (
                self._state.last_tick.isoformat() if self._state.last_tick else None
            ),
            "last_intervention": (
                self._state.last_intervention.isoformat()
                if self._state.last_intervention
                else None
            ),
            "current_policy": self._state.current_policy.value,
            "pending_count": len(self._state.pending_interventions),
            "discovery": (
                self._state.discovery.to_dict() if self._state.discovery else None
            ),
            "velocity": (
                self._state.velocity.to_dict() if self._state.velocity else None
            ),
        }
