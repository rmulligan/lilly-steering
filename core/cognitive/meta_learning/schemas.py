"""Schemas for Phase 4 meta-learning."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PatternStats(BaseModel):
    """Statistics for a pattern identified in decision history.

    Groups decisions by action_type and calculates success metrics
    to identify patterns worth extracting as heuristics.
    """

    action_type: str = Field(..., description="Type of action (e.g., 'steering_adjustment')")
    count: int = Field(..., description="Number of examples of this action type", ge=0)
    success_rate: float = Field(..., description="Fraction of successful decisions (0.0-1.0)", ge=0.0, le=1.0)
    example_ids: list[str] = Field(
        default_factory=list, description="Decision IDs used to compute this pattern"
    )


class MetaLearningMetrics(BaseModel):
    """Metrics from a JUDGMENT_REVIEW episode.

    Tracks heuristic formation rate and diversity for research analysis.
    """

    heuristics_created: int = Field(..., description="Number of heuristics extracted", ge=0)
    action_types_covered: int = Field(..., description="Number of distinct action types", ge=0)
    avg_success_rate: float = Field(..., description="Average success rate across patterns", ge=0.0, le=1.0)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When metrics were captured",
    )
