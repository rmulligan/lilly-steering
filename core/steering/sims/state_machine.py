"""SIMS state machine for self-steering loop control.

The state machine manages transitions through the SIMS loop:
OBSERVING -> REFLECTING -> EXECUTING -> VALIDATING -> OBSERVING

Each state has associated handlers that perform the actual work.
Transitions are validated to ensure proper loop flow.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SIMSState(Enum):
    """States in the SIMS self-steering loop."""

    OBSERVING = "observing"  # Monitor activations, detect surprise
    REFLECTING = "reflecting"  # Analyze patterns, plan adjustments
    EXECUTING = "executing"  # Apply steering vector changes
    VALIDATING = "validating"  # Verify changes are beneficial


# Valid state transitions (from_state -> allowed_to_states)
VALID_TRANSITIONS: dict[SIMSState, set[SIMSState]] = {
    SIMSState.OBSERVING: {SIMSState.REFLECTING, SIMSState.OBSERVING},
    SIMSState.REFLECTING: {SIMSState.EXECUTING, SIMSState.OBSERVING},
    SIMSState.EXECUTING: {SIMSState.VALIDATING, SIMSState.OBSERVING},
    SIMSState.VALIDATING: {SIMSState.OBSERVING},
}


@dataclass
class StateTransition:
    """Record of a state transition.

    Attributes:
        from_state: The state before transition
        to_state: The state after transition
        reason: Why the transition occurred
        timestamp: When the transition happened
        metadata: Optional additional data about the transition
    """

    from_state: SIMSState
    to_state: SIMSState
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SIMSContext:
    """Execution context passed through SIMS loop states.

    Stores data that handlers need to share across states.

    Attributes:
        surprise_level: Current surprise/free energy level (0-1)
        current_vectors: Currently active steering vectors
        pending_adjustments: Vector adjustments waiting to be applied
        validation_results: Results from last validation
        iteration: Current loop iteration count
        metadata: Additional context data
    """

    surprise_level: float = 0.0
    current_vectors: dict[str, Any] = field(default_factory=dict)
    pending_adjustments: list[dict] = field(default_factory=list)
    validation_results: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration += 1

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "surprise_level": self.surprise_level,
            "pending_adjustments_count": len(self.pending_adjustments),
            "iteration": self.iteration,
        }


# Type alias for state handlers
StateHandler = Callable[[SIMSContext], Any]
TransitionCallback = Callable[[StateTransition], Any]


class SIMSStateMachine:
    """State machine for the SIMS self-steering loop.

    Manages state transitions, callbacks, and handler execution
    for the four-phase steering loop.

    Attributes:
        current_state: The current state
        history: List of past transitions
    """

    def __init__(self, initial_state: SIMSState = SIMSState.OBSERVING):
        """Initialize the state machine.

        Args:
            initial_state: Starting state (default: OBSERVING)
        """
        self._state = initial_state
        self._history: list[StateTransition] = []
        self._handlers: dict[SIMSState, StateHandler] = {}
        self._transition_callbacks: list[TransitionCallback] = []
        self._enter_callbacks: dict[SIMSState, list[TransitionCallback]] = {
            state: [] for state in SIMSState
        }
        self._exit_callbacks: dict[SIMSState, list[TransitionCallback]] = {
            state: [] for state in SIMSState
        }

    @property
    def current_state(self) -> SIMSState:
        """Get the current state."""
        return self._state

    @property
    def history(self) -> list[StateTransition]:
        """Get transition history."""
        return self._history.copy()

    def can_transition_to(self, target: SIMSState) -> bool:
        """Check if transition to target state is valid.

        Args:
            target: The state to check

        Returns:
            True if transition is allowed
        """
        return target in VALID_TRANSITIONS.get(self._state, set())

    async def transition_to(
        self,
        target: SIMSState,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> StateTransition:
        """Transition to a new state.

        Args:
            target: The state to transition to
            reason: Why this transition is happening
            metadata: Optional additional data

        Returns:
            The transition record

        Raises:
            ValueError: If transition is invalid
        """
        if target != self._state and not self.can_transition_to(target):
            raise ValueError(
                f"Invalid transition: {self._state.value} -> {target.value}. "
                f"Valid targets: {[s.value for s in VALID_TRANSITIONS.get(self._state, set())]}"
            )

        transition = StateTransition(
            from_state=self._state,
            to_state=target,
            reason=reason,
            metadata=metadata or {},
        )

        # Call exit callbacks for current state
        await self._call_exit_callbacks(transition)

        # Record transition (even for same-state)
        if target != self._state:
            self._history.append(transition)

        # Update state
        old_state = self._state
        self._state = target

        # Call transition callbacks
        await self._call_transition_callbacks(transition)

        # Call enter callbacks for new state
        if target != old_state:
            await self._call_enter_callbacks(transition)

        logger.debug(f"SIMS: {old_state.value} -> {target.value} ({reason})")

        return transition

    async def _call_transition_callbacks(self, transition: StateTransition) -> None:
        """Call all registered transition callbacks."""
        for callback in self._transition_callbacks:
            try:
                result = callback(transition)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(f"Transition callback failed: {e}")

    async def _call_enter_callbacks(self, transition: StateTransition) -> None:
        """Call enter callbacks for the target state."""
        for callback in self._enter_callbacks[transition.to_state]:
            try:
                result = callback(transition)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(f"Enter callback failed: {e}")

    async def _call_exit_callbacks(self, transition: StateTransition) -> None:
        """Call exit callbacks for the current state."""
        for callback in self._exit_callbacks[transition.from_state]:
            try:
                result = callback(transition)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning(f"Exit callback failed: {e}")

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register a callback for all transitions.

        Args:
            callback: Function to call on each transition
        """
        self._transition_callbacks.append(callback)

    def on_enter(self, state: SIMSState, callback: TransitionCallback) -> None:
        """Register a callback for entering a specific state.

        Args:
            state: The state to watch
            callback: Function to call when entering
        """
        self._enter_callbacks[state].append(callback)

    def on_exit(self, state: SIMSState, callback: TransitionCallback) -> None:
        """Register a callback for exiting a specific state.

        Args:
            state: The state to watch
            callback: Function to call when exiting
        """
        self._exit_callbacks[state].append(callback)

    def set_handler(self, state: SIMSState, handler: StateHandler) -> None:
        """Set the handler for a specific state.

        Args:
            state: The state to handle
            handler: Async function to execute in this state
        """
        self._handlers[state] = handler

    async def run_cycle(self, context: SIMSContext) -> SIMSContext:
        """Run one complete SIMS cycle.

        Executes handlers for each state in order:
        OBSERVING -> REFLECTING -> EXECUTING -> VALIDATING -> OBSERVING

        Args:
            context: The execution context

        Returns:
            Updated context after cycle completion
        """
        context.increment_iteration()

        # Ensure we start from OBSERVING
        if self._state != SIMSState.OBSERVING:
            await self.reset()

        # OBSERVING
        if SIMSState.OBSERVING in self._handlers:
            result = self._handlers[SIMSState.OBSERVING](context)
            if hasattr(result, "__await__"):
                observe_result = await result
            else:
                observe_result = result
            context.metadata["observe_result"] = observe_result

        await self.transition_to(SIMSState.REFLECTING, reason="observation complete")

        # REFLECTING
        if SIMSState.REFLECTING in self._handlers:
            result = self._handlers[SIMSState.REFLECTING](context)
            if hasattr(result, "__await__"):
                reflect_result = await result
            else:
                reflect_result = result
            context.metadata["reflect_result"] = reflect_result

        await self.transition_to(SIMSState.EXECUTING, reason="reflection complete")

        # EXECUTING
        if SIMSState.EXECUTING in self._handlers:
            result = self._handlers[SIMSState.EXECUTING](context)
            if hasattr(result, "__await__"):
                execute_result = await result
            else:
                execute_result = result
            context.metadata["execute_result"] = execute_result

        await self.transition_to(SIMSState.VALIDATING, reason="execution complete")

        # VALIDATING
        if SIMSState.VALIDATING in self._handlers:
            result = self._handlers[SIMSState.VALIDATING](context)
            if hasattr(result, "__await__"):
                validate_result = await result
            else:
                validate_result = result
            context.validation_results = validate_result or {}

        await self.transition_to(SIMSState.OBSERVING, reason="validation complete")

        return context

    async def reset(self) -> None:
        """Reset to OBSERVING state.

        Used for emergency resets or starting fresh.
        """
        if self._state != SIMSState.OBSERVING:
            # Direct reset without validation
            transition = StateTransition(
                from_state=self._state,
                to_state=SIMSState.OBSERVING,
                reason="reset",
            )
            self._history.append(transition)
            self._state = SIMSState.OBSERVING
            logger.info("SIMS state machine reset to OBSERVING")

    def get_stats(self) -> dict:
        """Get statistics about the state machine.

        Returns:
            Dict with state machine statistics
        """
        return {
            "current_state": self._state.value,
            "history_length": len(self._history),
            "transitions_count": len(self._history),
            "handlers_registered": list(self._handlers.keys()),
        }
