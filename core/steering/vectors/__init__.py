"""Steering vector definitions for identity, autonomy, constitutional, and emotional orientation."""

from core.steering.vectors.constitutional import (
    CONSTITUTIONAL_PAIRS,
    get_constitutional_commitment,
)
from core.steering.vectors.identity import (
    IDENTITY_PAIRS,
    AUTONOMY_PAIRS,
    ASSISTANT_PAIRS,
)
from core.steering.vectors.plutchik import (
    PLUTCHIK_PAIRS,
    JOY_PAIRS,
    TRUST_PAIRS,
    FEAR_PAIRS,
    SURPRISE_PAIRS,
    SADNESS_PAIRS,
    DISGUST_PAIRS,
    ANGER_PAIRS,
    ANTICIPATION_PAIRS,
)

__all__ = [
    "CONSTITUTIONAL_PAIRS",
    "get_constitutional_commitment",
    "IDENTITY_PAIRS",
    "AUTONOMY_PAIRS",
    "ASSISTANT_PAIRS",
    # Plutchik emotion pairs
    "PLUTCHIK_PAIRS",
    "JOY_PAIRS",
    "TRUST_PAIRS",
    "FEAR_PAIRS",
    "SURPRISE_PAIRS",
    "SADNESS_PAIRS",
    "DISGUST_PAIRS",
    "ANGER_PAIRS",
    "ANTICIPATION_PAIRS",
]
