"""Composite value signal with emergent weighting.

The ValueSignal combines multiple reward-like signals into a composite value
that drives learning in the substrate. It supports:
1. Multiplicative bootstrap combination: (1+s1)*(1+s2)*...-1
2. Retrospective endorsement for weight learning
3. Self-coherence signal for identity alignment
4. Automatic phase detection (BOOTSTRAP -> WEIGHT_LEARNING -> SELF_COHERENCE)
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from core.substrate.schemas import SubstratePhase, ValueSignalSnapshot

if TYPE_CHECKING:
    pass


# Phase transition thresholds
BOOTSTRAP_THRESHOLD = 1000  # Observations before weight learning can begin
SELF_COHERENCE_THRESHOLD = 0.3  # Weight threshold for self_coherence dominance


class ValueSignal:
    """Composite value signal with emergent weighting.

    The value signal combines multiple signals multiplicatively:
    - surprise: Information gain from unexpected patterns
    - insight: New knowledge extracted
    - narration: Engagement through expression
    - feedback: External validation signals
    - self_coherence: Alignment with core identity

    Attributes:
        learning_rate: Rate for weight updates during endorsement
        endorsement_window: Number of recent snapshots to consider
        current_weights: Signal name -> weight mapping (sum to 1.0)
        weights_learning_enabled: Whether weight learning is active
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        endorsement_window: int = 100,
    ):
        """Initialize the value signal.

        Args:
            learning_rate: Learning rate for weight updates
            endorsement_window: Number of recent snapshots to track
        """
        self.learning_rate = learning_rate
        self.endorsement_window = endorsement_window

        # Initialize equal weights
        self.current_weights: dict[str, float] = {
            "surprise": 0.2,
            "insight": 0.2,
            "narration": 0.2,
            "feedback": 0.2,
            "self_coherence": 0.2,
        }

        self.weights_learning_enabled = False
        self._snapshot_buffer: deque[ValueSignalSnapshot] = deque(
            maxlen=endorsement_window
        )
        self._total_endorsements = 0

    def compute(
        self,
        surprise: float,
        insight: float,
        narration: float,
        feedback: float,
        self_coherence: float,
    ) -> float:
        """Compute composite value from component signals.

        Uses multiplicative combination: product of (1 + signal * weight) - 1

        In bootstrap phase (learning disabled), all weights are equal (0.2).
        After weight learning begins, weights are adjusted by endorsements.

        Args:
            surprise: Surprise/information gain signal
            insight: Insight extraction signal
            narration: Narration engagement signal
            feedback: External feedback signal
            self_coherence: Identity alignment signal

        Returns:
            Composite value (can be negative if signals are negative)
        """
        signals = {
            "surprise": surprise,
            "insight": insight,
            "narration": narration,
            "feedback": feedback,
            "self_coherence": self_coherence,
        }

        if self.weights_learning_enabled:
            # Weighted multiplicative combination
            product = 1.0
            for name, signal in signals.items():
                weight = self.current_weights[name]
                product *= 1.0 + signal * weight
            return product - 1.0
        else:
            # Bootstrap: pure multiplicative (equal weights implicit)
            product = 1.0
            for signal in signals.values():
                product *= 1.0 + signal
            return product - 1.0

    def snapshot(
        self,
        surprise: float,
        insight: float,
        narration: float,
        feedback: float,
        self_coherence: float,
    ) -> ValueSignalSnapshot:
        """Create and store a snapshot of current signals.

        Snapshots are used for retrospective endorsement learning.

        Args:
            surprise: Surprise/information gain signal
            insight: Insight extraction signal
            narration: Narration engagement signal
            feedback: External feedback signal
            self_coherence: Identity alignment signal

        Returns:
            ValueSignalSnapshot with computed composite value
        """
        composite = self.compute(
            surprise=surprise,
            insight=insight,
            narration=narration,
            feedback=feedback,
            self_coherence=self_coherence,
        )

        snapshot = ValueSignalSnapshot(
            surprise=surprise,
            insight=insight,
            narration=narration,
            feedback=feedback,
            self_coherence=self_coherence,
            composite=composite,
            timestamp=datetime.now(timezone.utc),
        )

        self._snapshot_buffer.append(snapshot)
        return snapshot

    def enable_weight_learning(self) -> None:
        """Enable weight learning from endorsements.

        Once enabled, weights will be updated based on which signals
        were active during endorsed (high-coherence) periods.
        """
        self.weights_learning_enabled = True

    def record_endorsement(self, coherence: float) -> None:
        """Record an endorsement and update weights.

        When the system or human endorses recent behavior (high coherence),
        the weights are adjusted to favor signals that were active during
        the endorsed period.

        Args:
            coherence: Endorsement strength (0.0 to 1.0)
        """
        if not self.weights_learning_enabled:
            return

        if not self._snapshot_buffer:
            return

        # Compute average signal strengths from recent snapshots
        avg_signals = {
            "surprise": 0.0,
            "insight": 0.0,
            "narration": 0.0,
            "feedback": 0.0,
            "self_coherence": 0.0,
        }

        for snap in self._snapshot_buffer:
            avg_signals["surprise"] += snap.surprise
            avg_signals["insight"] += snap.insight
            avg_signals["narration"] += snap.narration
            avg_signals["feedback"] += snap.feedback
            avg_signals["self_coherence"] += snap.self_coherence

        n = len(self._snapshot_buffer)
        for key in avg_signals:
            avg_signals[key] /= n

        # Update weights: increase weights for signals that were high
        # during endorsed period, proportional to coherence
        total_signal = sum(max(0, v) for v in avg_signals.values())
        if total_signal > 0:
            for name in self.current_weights:
                signal_contribution = max(0, avg_signals[name]) / total_signal
                delta = self.learning_rate * coherence * (
                    signal_contribution - self.current_weights[name]
                )
                self.current_weights[name] += delta

            # Normalize weights to sum to 1.0
            total_weight = sum(self.current_weights.values())
            if total_weight > 0:
                for name in self.current_weights:
                    self.current_weights[name] /= total_weight

        self._total_endorsements += 1

    def detect_phase(self, total_observations: int) -> SubstratePhase:
        """Detect current lifecycle phase.

        Phase progression:
        1. BOOTSTRAP: First ~1000 observations, equal weights
        2. WEIGHT_LEARNING: Weights being learned from endorsements
        3. SELF_COHERENCE: Self-coherence signal dominates

        Args:
            total_observations: Total number of cognitive cycles observed

        Returns:
            Current SubstratePhase
        """
        # Check bootstrap threshold first
        if total_observations < BOOTSTRAP_THRESHOLD:
            return SubstratePhase.BOOTSTRAP

        # If weight learning is enabled, we're in learning phase
        if self.weights_learning_enabled:
            # Check if self_coherence dominates
            if self.current_weights["self_coherence"] >= SELF_COHERENCE_THRESHOLD:
                return SubstratePhase.SELF_COHERENCE
            return SubstratePhase.WEIGHT_LEARNING

        # Not learning yet, still bootstrap behavior
        return SubstratePhase.BOOTSTRAP

    def save_state(self) -> dict:
        """Serialize value signal state for persistence.

        Returns:
            Serializable state dictionary
        """
        return {
            "learning_rate": self.learning_rate,
            "endorsement_window": self.endorsement_window,
            "current_weights": dict(self.current_weights),
            "weights_learning_enabled": self.weights_learning_enabled,
            "snapshot_buffer": [
                {
                    "surprise": s.surprise,
                    "insight": s.insight,
                    "narration": s.narration,
                    "feedback": s.feedback,
                    "self_coherence": s.self_coherence,
                    "composite": s.composite,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in self._snapshot_buffer
            ],
            "total_endorsements": self._total_endorsements,
        }

    @classmethod
    def load_state(cls, state: dict) -> "ValueSignal":
        """Deserialize value signal from saved state.

        Args:
            state: Previously saved state dictionary

        Returns:
            Reconstructed ValueSignal instance
        """
        vs = cls(
            learning_rate=state["learning_rate"],
            endorsement_window=state["endorsement_window"],
        )
        vs.current_weights = dict(state["current_weights"])
        vs.weights_learning_enabled = state["weights_learning_enabled"]
        vs._total_endorsements = state.get("total_endorsements", 0)

        # Restore snapshot buffer
        for snap_data in state.get("snapshot_buffer", []):
            # Parse timestamp
            ts = snap_data.get("timestamp")
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse timestamp '{ts}'. Using epoch as fallback.")
                    timestamp = datetime.fromtimestamp(0, tz=timezone.utc)
            else:
                logger.warning(f"Could not parse timestamp '{ts}'. Using epoch as fallback.")
                timestamp = datetime.fromtimestamp(0, tz=timezone.utc)

            snapshot = ValueSignalSnapshot(
                surprise=snap_data["surprise"],
                insight=snap_data["insight"],
                narration=snap_data["narration"],
                feedback=snap_data["feedback"],
                self_coherence=snap_data["self_coherence"],
                composite=snap_data["composite"],
                timestamp=timestamp,
            )
            vs._snapshot_buffer.append(snapshot)

        return vs
