"""Quality Diversity (QD) metrics for EvalatisSteerer.

This module implements structured scoring based on four metrics:
- COHERENCE (20%): Quality floor via embedding alignment
- NOVELTY (35%): Diversity driver via inverse similarity to recent selections
- SURPRISE (25%): Emergence signal from activation surprise
- PRESENCE (20%): Authenticity from human approval patterns

Usage:
    from core.steering.qd import QDConfig, QDScorer

    config = QDConfig()
    scorer = QDScorer(config, embedding_service, feature_tracker)

    score = scorer.score(crystal, context)
    if score.passed_floor:
        # Crystal passes quality floor
        print(f"Total QD score: {score.total}")
"""

from core.steering.qd.config import QDConfig
from core.steering.qd.scorer import QDContext, QDScore, QDScorer

__all__ = [
    "QDConfig",
    "QDContext",
    "QDScore",
    "QDScorer",
]
