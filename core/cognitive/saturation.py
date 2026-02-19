"""Saturation detection for stage transitions in progressive thinking."""
from dataclasses import dataclass

from core.cognitive.goal import InquiryGoal
from core.cognitive.stage import CognitiveStage, StageConfig


@dataclass
class SaturationSignal:
    """Signals indicating potential stage saturation.

    The key insight from semantic collapse analysis: repetition itself is a signal
    that exploration is exhausted and synthesis should begin. We track both
    consecutive AND cumulative low-surprise to avoid oscillation bugs where
    a single high-surprise cycle resets the consecutive counter.
    """

    surprise_declining: bool = False  # Surprise dropped for N consecutive cycles
    cumulative_low_surprise: int = 0  # Total low-surprise cycles in this stage
    repetition_detected: bool = False  # Opening tracker flagged repeat
    repetition_count: int = 0  # Number of repetitions in this stage
    insight_stale: bool = False  # Same insight extracted multiple times
    cycles_in_stage: int = 0  # Time spent in current stage


def check_saturation(
    goal: InquiryGoal,
    signal: SaturationSignal,
    config: StageConfig,
) -> bool:
    """Check if current stage is saturated and ready to advance.

    Each stage has different saturation rules:
    - QUESTION: Saturates on repetition OR max cycles
    - EXPLORE: Saturates on declining surprise
    - SYNTHESIZE: Saturates on stale insights OR max cycles
    - COMMIT: Always saturates after min cycle (ready to complete)
    """
    # Minimum cycles required (prevents premature advancement)
    if signal.cycles_in_stage < config.min_cycles:
        return False

    # Stage-specific saturation rules
    if goal.stage == CognitiveStage.QUESTION:
        # Question stage: saturates on repetition or reaching max cycles
        return signal.repetition_detected or signal.cycles_in_stage >= config.max_cycles

    elif goal.stage == CognitiveStage.EXPLORE:
        # Explore stage: saturates when exploration is exhausted
        # Key insight from semantic collapse analysis:
        # 1. Cumulative low-surprise (not just consecutive) prevents oscillation bugs
        # 2. Repetition IS the signal that we're ready to synthesize
        # 3. "You're repeating yourself = you've run out of new things to explore"
        return (
            signal.cumulative_low_surprise >= 3 or  # Cumulative prevents reset bug
            signal.repetition_count >= 2 or         # Repetition = ready to synthesize
            signal.surprise_declining or            # Keep legacy consecutive check
            signal.cycles_in_stage >= config.max_cycles
        )

    elif goal.stage == CognitiveStage.SYNTHESIZE:
        # Synthesize stage: saturates when insights become stale
        return signal.insight_stale or signal.cycles_in_stage >= config.max_cycles

    elif goal.stage == CognitiveStage.COMMIT:
        # Commit stage: always ready after min cycle
        return True

    return False


# Stage advancement mapping
STAGE_PROGRESSION = {
    CognitiveStage.QUESTION: CognitiveStage.EXPLORE,
    CognitiveStage.EXPLORE: CognitiveStage.SYNTHESIZE,
    CognitiveStage.SYNTHESIZE: CognitiveStage.COMMIT,
    CognitiveStage.COMMIT: CognitiveStage.COMMIT,  # No advancement past COMMIT
}


def advance_stage(goal: InquiryGoal) -> InquiryGoal:
    """Move to next stage, preserving accumulated insights.

    Returns a new InquiryGoal with the next stage and reset cycle counter.
    """
    next_stage = STAGE_PROGRESSION[goal.stage]
    return goal.with_update(
        stage=next_stage,
        stage_cycles=0,  # Reset counter for new stage
    )
