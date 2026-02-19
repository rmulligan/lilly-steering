"""Tier 1 highlight content for priority narrations.

Highlights represent interesting moments that should be narrated immediately
when they occur, taking priority over both process narration (tier 2) and
memory fallback (tier 3).

Highlight types in priority order:
1. PREDICTION_RESULT - Most engaging for listeners (verified/falsified)
2. ZETTEL_CREATED - Insight crystallization
3. EMOTIONAL_SHIFT - Affective state changes
4. STEERING_ALIGNMENT - How well output matched steering intent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class HighlightType(Enum):
    """Types of highlight content in priority order."""

    PREDICTION_RESULT = "prediction_result"  # Priority 1 (most interesting)
    ZETTEL_CREATED = "zettel_created"  # Priority 2
    EMOTIONAL_SHIFT = "emotional_shift"  # Priority 3
    STEERING_ALIGNMENT = "steering_alignment"  # Priority 4


# Priority values for sorting (lower = higher priority)
HIGHLIGHT_PRIORITY: dict[HighlightType, int] = {
    HighlightType.PREDICTION_RESULT: 1,
    HighlightType.ZETTEL_CREATED: 2,
    HighlightType.EMOTIONAL_SHIFT: 3,
    HighlightType.STEERING_ALIGNMENT: 4,
}


@dataclass
class HighlightContent:
    """Content for Tier 1 highlight narrations.

    Highlights are queued and narrated with priority over other content
    types when filling silence gaps.
    """

    highlight_type: HighlightType
    text: str
    priority: int = 1  # Lower = higher priority
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        # Auto-set priority from type if not explicitly provided
        if self.priority == 1:
            self.priority = HIGHLIGHT_PRIORITY.get(self.highlight_type, 99)


def _describe_valence(valence: float) -> str:
    """Map valence value to human-readable description."""
    if valence < 0.2:
        return "very negative"
    elif valence < 0.4:
        return "negative"
    elif valence < 0.6:
        return "neutral"
    elif valence < 0.8:
        return "positive"
    else:
        return "very positive"


def _describe_intensity(intensity: float) -> str:
    """Map intensity value to human-readable description."""
    if intensity < 0.3:
        return "subdued"
    elif intensity < 0.5:
        return "moderate"
    elif intensity < 0.7:
        return "heightened"
    elif intensity < 0.85:
        return "intense"
    else:
        return "very intense"


def build_prediction_highlight(
    outcome: str,
    hypothesis: str,
    confidence: Optional[float] = None,
) -> HighlightContent:
    """Build a highlight for prediction verification results.

    Args:
        outcome: "verified", "falsified", or "inconclusive"
        hypothesis: The hypothesis statement being tested
        confidence: Optional confidence level of the prediction

    Returns:
        HighlightContent ready for narration
    """
    # Truncate hypothesis for TTS readability
    brief_hypothesis = hypothesis[:60] + "..." if len(hypothesis) > 60 else hypothesis

    if outcome == "verified":
        text = f"A prediction was just verified. The hypothesis about {brief_hypothesis} held true."
    elif outcome == "falsified":
        text = f"A prediction was falsified. The hypothesis about {brief_hypothesis} did not hold."
    else:
        text = f"A prediction outcome remains inconclusive. Still testing: {brief_hypothesis}"

    if confidence is not None and confidence > 0.8:
        text += f" This was a high-confidence prediction."

    return HighlightContent(
        highlight_type=HighlightType.PREDICTION_RESULT,
        text=text,
    )


def build_zettel_highlight(
    title: str,
    lineage_concept: Optional[str] = None,
) -> HighlightContent:
    """Build a highlight for zettel (insight) crystallization.

    Args:
        title: The zettel title
        lineage_concept: Optional parent concept from which this emerged

    Returns:
        HighlightContent ready for narration
    """
    # Truncate title for TTS
    brief_title = title[:50] + "..." if len(title) > 50 else title

    if lineage_concept:
        text = f"An insight just crystallized: {brief_title}. It emerged from the exploration of {lineage_concept}."
    else:
        text = f"An insight just crystallized: {brief_title}."

    return HighlightContent(
        highlight_type=HighlightType.ZETTEL_CREATED,
        text=text,
    )


def build_emotional_highlight(
    old_valence: float,
    new_valence: float,
    old_intensity: Optional[float] = None,
    new_intensity: Optional[float] = None,
) -> Optional[HighlightContent]:
    """Build a highlight for significant emotional field shifts.

    Only creates a highlight if the shift is significant enough
    (delta > 0.15 for valence or intensity).

    Args:
        old_valence: Previous valence value
        new_valence: New valence value
        old_intensity: Optional previous intensity
        new_intensity: Optional new intensity

    Returns:
        HighlightContent if shift is significant, None otherwise
    """
    valence_delta = abs(new_valence - old_valence)
    intensity_delta = (
        abs(new_intensity - old_intensity)
        if old_intensity is not None and new_intensity is not None
        else 0.0
    )

    # Only highlight significant shifts
    if valence_delta < 0.15 and intensity_delta < 0.15:
        return None

    # Describe the shift
    if valence_delta >= intensity_delta:
        # Valence shift is more significant
        direction = "lifted" if new_valence > old_valence else "shifted"
        old_desc = _describe_valence(old_valence)
        new_desc = _describe_valence(new_valence)
        text = f"The emotional field has {direction}. The valence moved from {old_desc} to {new_desc}."
    else:
        # Intensity shift is more significant
        direction = "intensified" if new_intensity > old_intensity else "calmed"
        old_desc = _describe_intensity(old_intensity)
        new_desc = _describe_intensity(new_intensity)
        text = f"The emotional field has {direction}. The intensity moved from {old_desc} to {new_desc}."

    return HighlightContent(
        highlight_type=HighlightType.EMOTIONAL_SHIFT,
        text=text,
    )


def build_steering_alignment_highlight(
    alignment: float,
    steering_zone: str = "exploration",
) -> Optional[HighlightContent]:
    """Build a highlight for steering alignment results.

    Only creates a highlight for notable alignment values
    (very high alignment > 0.7 or misalignment < -0.2).

    Args:
        alignment: Alignment score in [-1, 1]
        steering_zone: Name of the steering zone being measured

    Returns:
        HighlightContent if alignment is notable, None otherwise
    """
    # Only highlight notable alignments
    if -0.2 <= alignment <= 0.7:
        return None

    if alignment > 0.7:
        text = f"Strong alignment detected. The {steering_zone} steering vector guided output effectively."
    elif alignment < -0.2:
        text = f"Unexpected divergence. The output moved opposite to the {steering_zone} steering direction."
    else:
        return None

    return HighlightContent(
        highlight_type=HighlightType.STEERING_ALIGNMENT,
        text=text,
    )
