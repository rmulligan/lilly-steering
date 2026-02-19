"""Lesson extraction for autonomous decisions using curator."""

import logging
import json
from typing import TYPE_CHECKING

from core.cognitive.reflexion.consequence_schemas import OutcomeAssessment, Lesson
from core.psyche.schema import AutonomousDecision

if TYPE_CHECKING:
    from core.model.curator_model import CuratorModel

logger = logging.getLogger(__name__)


LESSON_EXTRACTION_PROMPT = """You are assessing an autonomous decision Lilly made 10 cycles ago.

DECISION CONTEXT:
Question: {question}
Judgment: {judgment}
Action: {action}
Expectation: {expectation}

OUTCOME:
Health before: {health_before}
Health after: {health_after}
Success: {success}

TASK:
1. Extract a concise lesson (2-3 sentences) from this decision and its outcome.
2. Rate the lesson's significance (0.0-1.0) based on:
   - Was the outcome surprising or unexpected?
   - Is the lesson generalizable to other situations?
   - Should this be permanently remembered?

Return JSON:
{{
  "lesson": "...",
  "significance": 0.0-1.0
}}
"""


class LessonExtractor:
    """Extracts lessons from decision outcomes using the curator model.

    Uses vLLM curator to analyze decision context and outcome, generating
    rich natural language lessons with significance scoring. Falls back to
    simple templates if curator fails.
    """

    def __init__(self, curator: "CuratorModel"):
        """Initialize with CuratorModel dependency.

        Args:
            curator: CuratorModel for lesson generation
        """
        self._curator = curator

    async def extract(
        self,
        decision: AutonomousDecision,
        assessment: OutcomeAssessment,
    ) -> Lesson:
        """Extract lesson from decision outcome.

        Args:
            decision: The autonomous decision being assessed
            assessment: Outcome assessment with health comparison

        Returns:
            Lesson with text and significance score
        """
        try:
            # Build prompt from decision and assessment
            prompt = LESSON_EXTRACTION_PROMPT.format(
                question=decision.question,
                judgment=decision.judgment,
                action=json.dumps(decision.action),
                expectation=decision.expectation,
                health_before=assessment.health_before.value,
                health_after=assessment.health_after.value,
                success="Success" if assessment.success else "Failure",
            )

            # Call curator
            logger.debug(f"Extracting lesson for decision {decision.id}")
            response = await self._curator.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,  # More deterministic for lesson extraction
            )

            # Parse JSON response
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response

            # Validate required fields
            if "lesson" not in parsed or "significance" not in parsed:
                logger.warning(f"Curator response missing required fields: {parsed}")
                return self._fallback_lesson(decision, assessment)

            return Lesson(
                text=parsed["lesson"],
                significance=float(parsed["significance"]),
            )

        except (TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Curator extraction failed: {e}, using template fallback")
            return self._fallback_lesson(decision, assessment)
        except Exception as e:
            logger.error(f"Unexpected error during lesson extraction: {e}")
            return self._fallback_lesson(decision, assessment)

    def _fallback_lesson(
        self,
        decision: AutonomousDecision,
        assessment: OutcomeAssessment,
    ) -> Lesson:
        """Generate simple template lesson as fallback.

        Args:
            decision: The autonomous decision
            assessment: Outcome assessment

        Returns:
            Lesson with template text and zero significance
        """
        # Extract parameter name from action if available
        parameter = decision.action.get("parameter", "parameter")

        if assessment.success:
            text = (
                f"Modifying {parameter} improved health from "
                f"{assessment.health_before.value} to {assessment.health_after.value}."
            )
        else:
            text = (
                f"Modifying {parameter} degraded health from "
                f"{assessment.health_before.value} to {assessment.health_after.value}."
            )

        # Template lessons get zero significance (won't create zettel)
        return Lesson(text=text, significance=0.0)
