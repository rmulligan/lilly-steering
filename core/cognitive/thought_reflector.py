"""LLM-based extraction of insights and questions from thoughts.

Uses a lightweight CPU model to extract structured insight/question pairs
from cognitive loop thoughts, running in parallel with GPU generation.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


class TextGenerator(Protocol):
    """Protocol for text generation models."""

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from prompt."""
        ...


# Prompt for extracting insight and question from a thought
THOUGHT_REFLECTION_PROMPT = """Analyze this thought and extract its core insight and driving question.

Thought:
{thought}

Extract ONE key insight (the most important realization or understanding) and ONE driving question (what curiosity emerges for further exploration).

Output JSON with this exact format:
{{"insight": "...", "question": "..."}}

Rules:
- insight: A clear, complete statement capturing the key realization (1-2 sentences)
- question: A genuine open question that emerges from this thinking (must end with ?)
- If no clear insight, use the most substantive claim from the thought
- If no clear question, identify what remains unresolved or worth exploring
- Do NOT include "I think" or "I wonder" - state the insight/question directly

JSON:"""


@dataclass
class ThoughtReflection:
    """Extracted reflection from a thought."""

    insight: str
    question: str

    @property
    def has_insight(self) -> bool:
        """Check if insight was extracted."""
        return bool(self.insight and len(self.insight) > 10)

    @property
    def has_question(self) -> bool:
        """Check if question was extracted."""
        return bool(self.question and self.question.endswith("?"))


class ThoughtReflector:
    """
    Extract insights and questions from thoughts using LLM.

    Uses the CuratorModel (vLLM-based) to analyze thoughts and extract
    structured insight/question pairs. The model is wired via the
    TextGenerator protocol adapter.

    Example:
        reflector = ThoughtReflector(text_generator)
        reflection = await reflector.reflect(thought_text)
        print(reflection.insight, reflection.question)
    """

    def __init__(self, model: Optional[TextGenerator] = None):
        """
        Initialize the thought reflector.

        Args:
            model: Text generation model (optional, can set later)
        """
        self._model = model

    def set_model(self, model: TextGenerator) -> None:
        """Set the model for reflection."""
        self._model = model

    async def reflect(self, thought: str) -> ThoughtReflection:
        """
        Extract insight and question from a thought using LLM.

        Args:
            thought: The thought text to analyze

        Returns:
            ThoughtReflection with extracted insight and question
        """
        if not self._model:
            logger.warning("No model set for thought reflection, using empty result")
            return ThoughtReflection(insight="", question="")

        # Truncate very long thoughts to avoid context issues
        if len(thought) > 2000:
            thought = thought[:2000] + "..."

        prompt = THOUGHT_REFLECTION_PROMPT.format(thought=thought)

        try:
            response = await self._model.generate(prompt, max_tokens=256)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"Thought reflection failed: {e}")
            return ThoughtReflection(insight="", question="")

    def _parse_response(self, response: str) -> ThoughtReflection:
        """Parse LLM response to extract insight and question."""
        # Try to find JSON in response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            logger.debug(f"No JSON found in reflection response: {response[:200]}")
            return ThoughtReflection(insight="", question="")

        json_str = json_match.group(0)

        # Clean common JSON issues
        json_str = json_str.replace("\n", " ")
        json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas

        try:
            data = json.loads(json_str)

            insight = data.get("insight", "").strip()
            question = data.get("question", "").strip()

            # Ensure question ends with ?
            if question and not question.endswith("?"):
                question += "?"

            # Truncate if too long
            if len(insight) > 250:
                insight = insight[:250].rsplit(" ", 1)[0] + "..."
            if len(question) > 200:
                question = question[:200].rsplit(" ", 1)[0] + "?"

            return ThoughtReflection(insight=insight, question=question)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reflection JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:200]}")
            return ThoughtReflection(insight="", question="")


# Singleton instance for use across the cognitive loop
_reflector: Optional[ThoughtReflector] = None


def get_thought_reflector() -> ThoughtReflector:
    """Get or create the singleton ThoughtReflector instance."""
    global _reflector
    if _reflector is None:
        _reflector = ThoughtReflector()
    return _reflector


def set_thought_reflector_model(model: TextGenerator) -> None:
    """Set the model for the singleton ThoughtReflector."""
    get_thought_reflector().set_model(model)
