"""Translate recognition signals to valence events for preference learning.

Maps Ryan's signals to affective dimensions:
- APPROVE: Strong positive valence (0.85), moderate arousal (0.6)
- DISAPPROVE: Strong negative valence (-0.75), higher arousal (0.7)
- CURIOUS: Slight positive valence (0.2), high arousal (0.8)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.recognition.schema import RecognitionSignal, SignalType

if TYPE_CHECKING:
    from core.self_model.preference_learner import ValenceEvent

# Signal → Valence mapping
# These values calibrated for preference learning impact
SIGNAL_VALENCE_MAP: dict[SignalType, float] = {
    SignalType.APPROVE: 0.85,  # Strong positive - genuine/authentic
    SignalType.DISAPPROVE: -0.75,  # Strong negative - performative/off
    SignalType.CURIOUS: 0.2,  # Slight positive - engaging novelty
}

# Signal → Arousal mapping
# Arousal affects learning rate in preference system
SIGNAL_AROUSAL_MAP: dict[SignalType, float] = {
    SignalType.APPROVE: 0.6,  # Calm approval
    SignalType.DISAPPROVE: 0.7,  # Active rejection
    SignalType.CURIOUS: 0.8,  # High engagement
}


def translate_to_valence_event(signal: RecognitionSignal) -> "ValenceEvent":
    """Convert recognition signal to valence event for preference learning.

    The valence event captures:
    - Context: Reference to the thought that was recognized
    - Action: The thought generation action
    - Outcome: Ryan's signal type
    - Valence: Emotional weight (-1 to +1)
    - Sources: Attribution to relational dimension (100% from Ryan)

    Args:
        signal: Recognition signal from Ryan

    Returns:
        ValenceEvent suitable for PreferenceLearner.process_experience()
    """
    # Import here to avoid circular dependency
    from core.self_model.preference_learner import ValenceEvent

    valence = SIGNAL_VALENCE_MAP[signal.signal_type]
    arousal = SIGNAL_AROUSAL_MAP[signal.signal_type]

    return ValenceEvent(
        context=f"thought:{signal.thought_uid}",
        action_taken="generated_thought",
        outcome=f"ryan_{signal.signal_type.value}",
        valence=valence,
        valence_sources={
            # 100% relational - this is direct feedback from Ryan
            "relational": 1.0,
            # Metadata for tracking
            "signal_type": signal.signal_type.value,
            "signal_confidence": signal.confidence,
            "arousal": arousal,
        },
        timestamp=signal.timestamp,
        uid=f"ve_{signal.uid}",
    )


def get_signal_weight(signal: RecognitionSignal) -> float:
    """Get learning weight for a signal.

    Higher confidence and arousal = stronger learning.

    Args:
        signal: Recognition signal

    Returns:
        Weight multiplier for learning (0.0 to 1.0)
    """
    arousal = SIGNAL_AROUSAL_MAP[signal.signal_type]
    return signal.confidence * arousal
