"""SOAR Curriculum Learning for Lilly.

This module implements Self-Optimization via Asymmetric Reinforcement Learning (SOAR)
for skill-based curriculum learning. Key components:

- hard_problems: Defines cognitive challenges where skills should help
- skill_tracker: Tracks skill effectiveness via grounded student improvement
- promotion: Accumulates proven skills into curriculum (𝒟ᵦₑₛₜ)
- teacher: Optimizes skill generation for student improvement

Reference: https://arxiv.org/html/2601.18778v1
"""

from core.cognitive.curriculum.hard_problems import (
    HardProblem,
    HardProblemRegistry,
    HardProblemType,
    is_hard_problem_condition,
)
from core.cognitive.curriculum.promotion import (
    PromotionConfig,
    PromotionQueue,
    SkillPerformanceRecord,
    calculate_retrieval_boost,
)
from core.cognitive.curriculum.skill_tracker import (
    SkillEffectivenessTracker,
    SkillInfluencedCycle,
)
from core.cognitive.curriculum.teacher import (
    TeacherConfig,
    TeacherPolicy,
    WeakSpot,
    filter_structural_patterns,
)

__all__ = [
    # Hard problems
    "HardProblem",
    "HardProblemRegistry",
    "HardProblemType",
    "is_hard_problem_condition",
    # Promotion
    "PromotionConfig",
    "PromotionQueue",
    "SkillPerformanceRecord",
    "calculate_retrieval_boost",
    # Skill tracking
    "SkillEffectivenessTracker",
    "SkillInfluencedCycle",
    # Teacher policy
    "TeacherConfig",
    "TeacherPolicy",
    "WeakSpot",
    "filter_structural_patterns",
]
