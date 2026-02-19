"""Generate compact skills from verified hypotheses.

This module implements the UPSKILL pattern enhanced with SOAR curriculum learning:
auto-generating portable skill packages (~500 tokens) from hypotheses that reach
VERIFIED status. Skills capture structural patterns that can be injected into
generation prompts to improve cognitive quality.

SOAR Enhancement: Teacher policy optimization prioritizes structural patterns
over conclusions, and targets cognitive weak spots for curriculum improvement.

Inspired by:
- HuggingFace UPSKILL: https://huggingface.co/blog/upskill
- SOAR: https://arxiv.org/html/2601.18778v1
"""

import logging
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from core.cognitive.curriculum.teacher import STRUCTURAL_KEYWORDS

if TYPE_CHECKING:
    from core.cognitive.curriculum.teacher import TeacherPolicy
    from core.cognitive.simulation.schemas import Hypothesis
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.schema import LearnedSkill

logger = logging.getLogger(__name__)


async def generate_skill_from_hypothesis(
    hypothesis: "Hypothesis",
    embedding_service: "TieredEmbeddingService",
    teacher_policy: Optional["TeacherPolicy"] = None,
    weak_spot_operations: Optional[list[str]] = None,
) -> Optional["LearnedSkill"]:
    """Generate skill when hypothesis reaches VERIFIED status.

    Extracts patterns, examples, and context triggers from a verified hypothesis
    to create a compact skill that can improve future cognitive cycles.

    SOAR Enhancement: When teacher_policy is provided, emphasizes structural
    patterns over conclusions and tracks weak spot targeting.

    Args:
        hypothesis: A hypothesis that has reached VERIFIED status
        embedding_service: Service for generating retrieval embeddings
        teacher_policy: Optional SOAR teacher policy for structural emphasis
        weak_spot_operations: Optional list of weak spot cognitive operations

    Returns:
        LearnedSkill if hypothesis has sufficient content, None otherwise
    """
    from core.cognitive.curriculum.teacher import filter_structural_patterns
    from core.embedding.service import EmbeddingTier
    from core.psyche.schema import LearnedSkill

    # Skip if missing required content
    if not hypothesis.cognitive_operation or not hypothesis.patterns_extracted:
        logger.debug(
            f"Skipping skill generation for {hypothesis.uid}: "
            f"missing cognitive_operation or patterns"
        )
        return None

    uid = f"skill_{uuid4().hex[:12]}"
    name = hypothesis.cognitive_operation.replace(" ", "_").lower()

    # SOAR: Prioritize structural patterns over conclusions
    patterns = hypothesis.patterns_extracted
    if teacher_policy is not None:
        structural_patterns = filter_structural_patterns(patterns)
        structural_count = _count_structural_patterns(patterns)
    else:
        structural_patterns = patterns
        structural_count = 0

    # Top 3 patterns, pipe-delimited (structural first if SOAR enabled)
    pattern_summary = " | ".join(structural_patterns[:3])

    # Extract context trigger from brainstorm
    when_to_apply = _extract_context_trigger(hypothesis.brainstorm_trace)

    # Determine if targeting a weak spot
    is_weak_spot = False
    if weak_spot_operations:
        is_weak_spot = hypothesis.cognitive_operation in weak_spot_operations

    # Build embedding from skill content
    skill_text = f"{name}: {pattern_summary} {when_to_apply}"
    try:
        embedding_result = await embedding_service.encode(
            skill_text, tier=EmbeddingTier.RETRIEVAL
        )
        embedding = embedding_result.to_list()
    except Exception as e:
        logger.warning(f"Failed to generate embedding for skill: {e}")
        embedding = None

    skill = LearnedSkill(
        uid=uid,
        name=name,
        description=hypothesis.synthesis_narrative[:300] if hypothesis.synthesis_narrative else "",
        source_hypothesis_uid=hypothesis.uid,
        cognitive_operation=hypothesis.cognitive_operation,
        pattern_summary=pattern_summary,
        when_to_apply=when_to_apply,
        positive_example=hypothesis.positive_example[:200] if hypothesis.positive_example else "",
        negative_example=hypothesis.negative_example[:200] if hypothesis.negative_example else "",
        embedding=embedding,
        created_cycle=hypothesis.cycle_generated,
        # SOAR fields
        is_weak_spot_targeted=is_weak_spot,
        structural_pattern_count=structural_count,
    )

    logger.info(
        f"[TEACHER] Generated skill '{name}' from hypothesis {hypothesis.uid} "
        f"(structural_patterns={structural_count}, weak_spot={is_weak_spot})"
    )

    return skill


def _count_structural_patterns(patterns: list[str]) -> int:
    """Count patterns that emphasize structure over conclusions.

    SOAR finding: Question structure matters more than answer correctness.

    Args:
        patterns: List of pattern strings

    Returns:
        Count of structural patterns
    """
    count = 0
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if any(kw in pattern_lower for kw in STRUCTURAL_KEYWORDS):
            count += 1

    return count


def _extract_context_trigger(brainstorm_trace: str) -> str:
    """Extract ~100 char context trigger from brainstorm trace.

    Looks for explicit "Context:" markers in the brainstorm, falling back
    to the first 100 characters if not found.

    Args:
        brainstorm_trace: Full brainstorm block from simulation

    Returns:
        Context trigger string (max 100 chars)
    """
    if not brainstorm_trace:
        return ""

    # Look for explicit Context: marker
    if "Context:" in brainstorm_trace:
        # Take everything after "Context:" until newline
        context_section = brainstorm_trace.split("Context:")[-1]
        first_line = context_section.split("\n")[0].strip()
        return first_line[:100]

    # Fallback: use first 100 chars of brainstorm
    return brainstorm_trace.strip()[:100]
