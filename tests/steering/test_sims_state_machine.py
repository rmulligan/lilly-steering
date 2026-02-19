"""Tests for SIMS (Self-Improvement through Model Steering) state machine."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from core.steering.sims.state_machine import (
    SIMSState,
    SIMSStateMachine,
    StateTransition,
    SIMSContext,
)


class TestSIMSState:
    """Tests for SIMS state enumeration."""

    def test_all_states_defined(self):
        """Should have all four SIMS loop states."""
        assert SIMSState.OBSERVING
        assert SIMSState.REFLECTING
        assert SIMSState.EXECUTING
        assert SIMSState.VALIDATING

    def test_states_are_strings(self):
        """States should have string values for logging."""
        assert SIMSState.OBSERVING.value == "observing"
        assert SIMSState.REFLECTING.value == "reflecting"
        assert SIMSState.EXECUTING.value == "executing"
        assert SIMSState.VALIDATING.value == "validating"


class TestStateTransition:
    """Tests for state transition records."""

    def test_transition_stores_states(self):
        """Should store from/to states."""
        transition = StateTransition(
            from_state=SIMSState.OBSERVING,
            to_state=SIMSState.REFLECTING,
            reason="High surprise detected",
        )

        assert transition.from_state == SIMSState.OBSERVING
        assert transition.to_state == SIMSState.REFLECTING
        assert transition.reason == "High surprise detected"

    def test_transition_has_timestamp(self):
        """Should record when transition occurred."""
        transition = StateTransition(
            from_state=SIMSState.OBSERVING,
            to_state=SIMSState.REFLECTING,
            reason="test",
        )

        assert transition.timestamp is not None
        assert isinstance(transition.timestamp, datetime)

    def test_to_dict_serialization(self):
        """Should serialize for logging."""
        transition = StateTransition(
            from_state=SIMSState.OBSERVING,
            to_state=SIMSState.REFLECTING,
            reason="test",
        )

        result = transition.to_dict()
        assert "from_state" in result
        assert "to_state" in result
        assert "reason" in result
        assert "timestamp" in result


class TestSIMSContext:
    """Tests for SIMS execution context."""

    def test_context_stores_data(self):
        """Should store context data for state handlers."""
        context = SIMSContext(
            surprise_level=0.8,
            current_vectors={"identity": [0.1] * 768},
            pending_adjustments=[],
        )

        assert context.surprise_level == 0.8
        assert "identity" in context.current_vectors

    def test_context_has_iteration_count(self):
        """Should track iteration count."""
        context = SIMSContext()
        assert context.iteration == 0

    def test_context_increment_iteration(self):
        """Should increment iteration count."""
        context = SIMSContext()
        context.increment_iteration()
        assert context.iteration == 1


class TestSIMSStateMachine:
    """Tests for the SIMS state machine."""

    def test_initial_state_is_observing(self):
        """Should start in OBSERVING state."""
        machine = SIMSStateMachine()
        assert machine.current_state == SIMSState.OBSERVING

    def test_custom_initial_state(self):
        """Should accept custom initial state."""
        machine = SIMSStateMachine(initial_state=SIMSState.REFLECTING)
        assert machine.current_state == SIMSState.REFLECTING

    @pytest.mark.asyncio
    async def test_transition_changes_state(self):
        """Should change state on valid transition."""
        machine = SIMSStateMachine()

        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        assert machine.current_state == SIMSState.REFLECTING

    @pytest.mark.asyncio
    async def test_transition_records_history(self):
        """Should record transition in history."""
        machine = SIMSStateMachine()

        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        assert len(machine.history) == 1
        assert machine.history[0].from_state == SIMSState.OBSERVING
        assert machine.history[0].to_state == SIMSState.REFLECTING

    @pytest.mark.asyncio
    async def test_valid_transitions(self):
        """Should allow valid SIMS loop transitions."""
        machine = SIMSStateMachine()

        # Full SIMS loop: OBSERVING -> REFLECTING -> EXECUTING -> VALIDATING -> OBSERVING
        await machine.transition_to(SIMSState.REFLECTING, reason="surprise")
        assert machine.current_state == SIMSState.REFLECTING

        await machine.transition_to(SIMSState.EXECUTING, reason="adjustment ready")
        assert machine.current_state == SIMSState.EXECUTING

        await machine.transition_to(SIMSState.VALIDATING, reason="applied")
        assert machine.current_state == SIMSState.VALIDATING

        await machine.transition_to(SIMSState.OBSERVING, reason="validated")
        assert machine.current_state == SIMSState.OBSERVING

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        """Should raise on invalid transition."""
        machine = SIMSStateMachine()

        # Can't go from OBSERVING directly to VALIDATING
        with pytest.raises(ValueError, match="Invalid transition"):
            await machine.transition_to(SIMSState.VALIDATING, reason="invalid")

    @pytest.mark.asyncio
    async def test_transition_callback(self):
        """Should call registered callbacks on transition."""
        machine = SIMSStateMachine()
        callback = AsyncMock()

        machine.on_transition(callback)
        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_entry_callback(self):
        """Should call state-specific entry callback."""
        machine = SIMSStateMachine()
        on_reflect = AsyncMock()

        machine.on_enter(SIMSState.REFLECTING, on_reflect)
        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        on_reflect.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_exit_callback(self):
        """Should call state-specific exit callback."""
        machine = SIMSStateMachine()
        on_exit_observe = AsyncMock()

        machine.on_exit(SIMSState.OBSERVING, on_exit_observe)
        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        on_exit_observe.assert_called_once()

    def test_can_transition_to(self):
        """Should check if transition is valid."""
        machine = SIMSStateMachine()

        assert machine.can_transition_to(SIMSState.REFLECTING)
        assert not machine.can_transition_to(SIMSState.VALIDATING)

    @pytest.mark.asyncio
    async def test_reset_returns_to_observing(self):
        """Reset should return to OBSERVING state."""
        machine = SIMSStateMachine()
        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        await machine.reset()

        assert machine.current_state == SIMSState.OBSERVING

    @pytest.mark.asyncio
    async def test_run_until_complete(self):
        """Should run full SIMS loop with handlers."""
        machine = SIMSStateMachine()

        observe_handler = AsyncMock(return_value={"surprise": 0.8})
        reflect_handler = AsyncMock(return_value={"adjustment": [0.1] * 768})
        execute_handler = AsyncMock(return_value={"applied": True})
        validate_handler = AsyncMock(return_value={"valid": True})

        machine.set_handler(SIMSState.OBSERVING, observe_handler)
        machine.set_handler(SIMSState.REFLECTING, reflect_handler)
        machine.set_handler(SIMSState.EXECUTING, execute_handler)
        machine.set_handler(SIMSState.VALIDATING, validate_handler)

        context = SIMSContext()
        await machine.run_cycle(context)

        # Should have run through all states
        observe_handler.assert_called_once()
        reflect_handler.assert_called_once()
        execute_handler.assert_called_once()
        validate_handler.assert_called_once()

        # Should be back to OBSERVING
        assert machine.current_state == SIMSState.OBSERVING

    def test_get_stats(self):
        """Should return statistics about state machine."""
        machine = SIMSStateMachine()

        stats = machine.get_stats()

        assert "current_state" in stats
        assert "history_length" in stats
        assert "transitions_count" in stats


class TestSIMSStateMachineEdgeCases:
    """Edge case tests for robustness."""

    @pytest.mark.asyncio
    async def test_transition_to_same_state(self):
        """Should handle transition to same state gracefully."""
        machine = SIMSStateMachine()

        # Transitioning to same state should be allowed (idempotent)
        await machine.transition_to(SIMSState.OBSERVING, reason="refresh")

        assert machine.current_state == SIMSState.OBSERVING

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_block_transition(self):
        """Callback exceptions shouldn't prevent state change."""
        machine = SIMSStateMachine()

        async def failing_callback(transition):
            raise RuntimeError("Callback failed")

        machine.on_transition(failing_callback)

        # Should still transition despite callback failure
        await machine.transition_to(SIMSState.REFLECTING, reason="test")
        assert machine.current_state == SIMSState.REFLECTING

    @pytest.mark.asyncio
    async def test_skip_to_observing_allowed(self):
        """Should allow skipping back to OBSERVING from any state."""
        machine = SIMSStateMachine()
        await machine.transition_to(SIMSState.REFLECTING, reason="test")

        # Emergency reset to OBSERVING should be allowed
        await machine.reset()
        assert machine.current_state == SIMSState.OBSERVING
