"""Shared utilities for the experimentation framework."""

from __future__ import annotations

from typing import Optional


# Keyword-to-parameter mapping for natural language references
# PARAMETER_KEYWORD_MAP is manually maintained for keyword matching
# (not derived from ALLOWED_PARAMETERS which has been removed in Phase 1)
PARAMETER_KEYWORD_MAP: dict[str, str] = {
    "exploration": "steering.exploration.magnitude",
    "concept steering": "steering.concept.magnitude",
    "identity steering": "steering.identity.magnitude",
    "episode length": "episode.max_segments",
    "segments": "episode.max_segments",
    "decay": "emotional_field.decay_rate",
    "diffusion": "emotional_field.diffusion_rate",
    "simulation trigger": "simulation.trigger_confidence",
    "trigger confidence": "simulation.trigger_confidence",
}


def extract_parameter_from_claim(claim: str) -> Optional[str]:
    """Extract parameter path from prediction claim text.

    Searches for parameter references in claim text using keyword mapping.
    Phase 1 Full Operational Autonomy: Uses manually maintained keyword map
    rather than iterating over ALLOWED_PARAMETERS (which has been removed).

    Args:
        claim: Prediction claim text to search

    Returns:
        Parameter path if found, None otherwise
    """
    claim_lower = claim.lower()

    # Check for common keywords that map to parameters
    for keyword, param in PARAMETER_KEYWORD_MAP.items():
        if keyword in claim_lower:
            return param

    # Check if claim contains dot-notation that looks like a parameter path
    # (e.g., "steering.exploration.magnitude" or "episode.max_segments")
    words = claim_lower.split()
    for word in words:
        if "." in word and len(word.split(".")) >= 2:
            # Looks like a parameter path, return it
            return word

    return None
