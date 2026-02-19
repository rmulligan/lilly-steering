"""
Knowledge extractor for Lilly's cognitive system.

Extracts structured knowledge (triples, entities) from text fragments
using LLM prompting with structured output.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from core.psyche.schema import Entity, Triple

logger = logging.getLogger(__name__)


def _sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string by fixing common LLM output errors.

    Handles:
    - Python None -> JSON null (with word boundaries to avoid corrupting strings)
    - Python True -> JSON true
    - Python False -> JSON false
    - Trailing commas before ] which are invalid JSON

    Args:
        json_str: Raw JSON string that may contain Python literals

    Returns:
        Sanitized JSON string ready for parsing
    """
    # Use word boundaries to avoid corrupting strings like "This is None of your business"
    json_str = re.sub(r'\bNone\b', 'null', json_str)
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    # Remove trailing commas before ] which are invalid JSON
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


def extract_json_array(text: str) -> Optional[str]:
    """
    Extract the first complete JSON array from text.

    This function properly handles nested brackets by counting bracket depth
    rather than using a greedy regex that would match from first '[' to last ']'.

    Also handles common LLM output formats like markdown code blocks.

    Args:
        text: Text that may contain a JSON array

    Returns:
        The extracted JSON array string, or None if not found
    """
    # Strip markdown code blocks if present
    # Handle ```json ... ```, ``` ... ```, or `json ... `
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Also try single backticks for inline code
    if text.startswith('`') and text.endswith('`'):
        text = text[1:-1].strip()

    start_idx = text.find('[')
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]

    # No matching closing bracket found
    return None


class TextGenerator(Protocol):
    """Protocol for text generation models."""

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt."""
        ...


# Prompt templates for knowledge extraction
TRIPLE_EXTRACTION_PROMPT = """Extract factual relationships from the following text as subject-predicate-object triples.

Text:
{text}

Extract up to {max_triples} triples. Output JSON array with this format:
[{{"subject": "...", "predicate": "...", "object": "..."}}, ...]

Rules:
- Subject and object should be specific named entities or concepts
- Predicate should be a verb or relationship phrase
- Only extract explicit facts, not inferences
- If no clear triples, return empty array []

JSON:"""

ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Text:
{text}

Extract up to {max_entities} entities. Output JSON array with this format:
[{{"name": "...", "type": "...", "description": "..."}}, ...]

Entity types: PERSON, ORGANIZATION, CONCEPT, PLACE, THING, EVENT, WORK

Rules:
- Include specific names, organizations, concepts mentioned
- Type should be one of the listed types
- Description should be brief (1 sentence max) if context provides it
- If no clear entities, return empty array []
- EXCLUDE document structure artifacts (page numbers, section headers, "page 1", "chapter 2", etc.)

JSON:"""

INSIGHT_EXTRACTION_PROMPT = """Extract distilled insights and open questions from the following text.

Text:
{text}

Extract up to {max_insights} insights. Each insight should be a condensed, reusable piece of knowledge that could inform future thinking. Output JSON array with this format:
[{{"insight_text": "...", "question_text": "...", "epistemic_status": "..."}}, ...]

Epistemic status values:
- "claim": A direct assertion or argument made by the author
- "observation": An empirical observation or fact presented
- "speculation": A tentative idea or hypothesis worth exploring
- "question": An open question that invites further investigation

Rules:
- insight_text: A clear, self-contained statement (1-2 sentences max)
- question_text: Optional open question that emerges from the insight (null if none)
- Capture the essence of ideas, not surface details
- Preserve nuance - don't oversimplify complex ideas
- If no clear insights, return empty array []

JSON:"""


@dataclass
class ExtractedInsight:
    """An insight extracted from text."""
    insight_text: str
    question_text: Optional[str]
    epistemic_status: Literal["claim", "observation", "speculation", "question"]


class TripleExtractor:
    """
    Extract subject-predicate-object triples from text using LLM.

    Uses structured prompting to extract factual relationships
    that can be stored in the knowledge graph.
    """

    def __init__(
        self,
        model: Optional[TextGenerator] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the triple extractor.

        Args:
            model: Text generation model (optional, can set later)
            confidence_threshold: Minimum confidence for extracted triples
        """
        self._model = model
        self.confidence_threshold = confidence_threshold

    def set_model(self, model: TextGenerator) -> None:
        """Set the model for extraction."""
        self._model = model

    async def extract(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
        max_triples: int = 10,
    ) -> list[Triple]:
        """
        Extract triples from text using the model.

        Args:
            text: Text to extract triples from
            source_fragment_uid: UID of the source fragment
            max_triples: Maximum number of triples to extract

        Returns:
            List of Triple objects
        """
        if not self._model:
            logger.warning("No model set for triple extraction")
            return []

        prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text, max_triples=max_triples)

        try:
            response = await self._model.generate(prompt, max_tokens=1024)
            triples = self._parse_triples_response(response, source_fragment_uid)
            logger.info(f"Extracted {len(triples)} triples from text")
            return triples
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []

    def _parse_triples_response(
        self, response: str, source_fragment_uid: Optional[str] = None
    ) -> list[Triple]:
        """Parse LLM response to extract triple objects."""
        # Find JSON array in response using bracket-aware extraction
        json_str = extract_json_array(response)
        if not json_str:
            logger.debug(f"No JSON array found in response: {response[:200]}")
            return []

        json_str = _sanitize_json_string(json_str)

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                logger.debug(f"Parsed JSON is not a list: {type(data)}")
                return []

            triples = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                subject = item.get("subject", "").strip()
                predicate = item.get("predicate", "").strip()
                obj = item.get("object", "").strip()

                if subject and predicate and obj:
                    triple = Triple(
                        uid=f"triple_{uuid.uuid4().hex[:12]}",
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.7,  # Default confidence for LLM extraction
                        source_fragment_uid=source_fragment_uid,
                        created_at=datetime.now(timezone.utc),
                    )
                    triples.append(triple)

            return triples

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse triple JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:200]}")
            return []

    def extract_sync(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> list[Triple]:
        """
        Extract triples using simple heuristics (no LLM required).

        This is a fallback method that extracts basic relationships
        using pattern matching. Less accurate but works without a model.

        Args:
            text: Text to extract from
            source_fragment_uid: Source fragment UID

        Returns:
            List of Triple objects
        """
        triples = []

        # Simple pattern: "X is/are Y" or "X was/were Y"
        is_pattern = re.compile(
            r"([A-Z][a-zA-Z\s]+)\s+(is|are|was|were)\s+(an?\s+)?([a-zA-Z\s]+[a-zA-Z])",
            re.IGNORECASE,
        )

        for match in is_pattern.finditer(text):
            subject = match.group(1).strip()
            predicate = match.group(2).strip()
            obj = match.group(4).strip()

            if len(subject) > 2 and len(obj) > 2:
                triple = Triple(
                    uid=f"triple_{uuid.uuid4().hex[:12]}",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=0.5,  # Lower confidence for heuristic extraction
                    source_fragment_uid=source_fragment_uid,
                    created_at=datetime.now(timezone.utc),
                )
                triples.append(triple)

        return triples[:10]  # Limit results


class EntityExtractor:
    """
    Extract named entities from text using LLM.

    Identifies people, organizations, concepts, and other named
    entities that should be tracked in the knowledge graph.
    """

    def __init__(
        self,
        model: Optional[TextGenerator] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the entity extractor.

        Args:
            model: Text generation model (optional, can set later)
            confidence_threshold: Minimum confidence for extracted entities
        """
        self._model = model
        self.confidence_threshold = confidence_threshold

    def set_model(self, model: TextGenerator) -> None:
        """Set the model for extraction."""
        self._model = model

    async def extract(
        self,
        text: str,
        max_entities: int = 10,
    ) -> list[Entity]:
        """
        Extract entities from text using the model.

        Args:
            text: Text to extract entities from
            max_entities: Maximum number of entities to extract

        Returns:
            List of Entity objects
        """
        if not self._model:
            logger.warning("No model set for entity extraction")
            return []

        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text, max_entities=max_entities)

        try:
            response = await self._model.generate(prompt, max_tokens=1024)
            entities = self._parse_entities_response(response)
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _parse_entities_response(self, response: str) -> list[Entity]:
        """Parse LLM response to extract entity objects."""
        # Find JSON array in response using bracket-aware extraction
        json_str = extract_json_array(response)
        if not json_str:
            logger.debug(f"No JSON array found in response: {response[:200]}")
            return []

        json_str = _sanitize_json_string(json_str)

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                logger.debug(f"Parsed JSON is not a list: {type(data)}")
                return []

            entities = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                entity_type = item.get("type", "THING").strip().upper()
                description = item.get("description", "").strip() or None

                # Filter out document structure artifacts
                if self._is_document_artifact(name):
                    logger.debug(f"Filtered document artifact entity: {name}")
                    continue

                if name:
                    entity = Entity(
                        uid=f"entity_{uuid.uuid4().hex[:12]}",
                        name=name,
                        entity_type=entity_type,
                        description=description,
                        created_at=datetime.now(timezone.utc),
                    )
                    entities.append(entity)

            return entities

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:200]}")
            return []

    def _is_document_artifact(self, name: str) -> bool:
        """Check if entity name looks like a document structure artifact."""
        import re
        name_lower = name.lower().strip()
        # Match patterns like "page 1", "chapter 2", "section 3", "pages 2-5"
        artifact_patterns = [
            r'^page\s*\d+$',
            r'^pages?\s*\d+',
            r'^chapter\s*\d+$',
            r'^section\s*\d+$',
            r'^figure\s*\d+$',
            r'^table\s*\d+$',
            r'^appendix\s*[a-z\d]+$',
            r'^\d+$',  # Just a number
            # PDF bounding boxes: "0 0 514 685", "0.0 0.0 100.5 100.5"
            r'^[\d\s\.]+$',
            # PDF object references: "annotations 6", "contents 7", "resources 8"
            r'^(annotations|contents|resources|mediabox|cropbox|trimbox|bleedbox|artbox)\s*\d+$',
            # PDF timestamps: "d:20260111014332"
            r'^d:\d+$',
            # Version numbers: "1.7", "2.0"
            r'^\d+\.\d+$',
        ]
        # Known PDF library/tool names
        pdf_artifacts = {'pdfium', 'pypdf', 'pymupdf', 'pdfminer', 'poppler'}
        if name_lower in pdf_artifacts:
            return True
        for pattern in artifact_patterns:
            if re.match(pattern, name_lower):
                return True
        return False

    def extract_sync(self, text: str) -> list[Entity]:
        """
        Extract entities using simple heuristics (no LLM required).

        Uses capitalization patterns to find potential named entities.
        Less accurate but works without a model.

        Args:
            text: Text to extract from

        Returns:
            List of Entity objects
        """
        entities = []
        seen_names: set[str] = set()

        # Pattern: Capitalized words (potential proper nouns)
        # Exclude common sentence starters
        skip_words = {
            "the", "a", "an", "this", "that", "these", "those", "i", "you",
            "he", "she", "it", "we", "they", "my", "your", "his", "her",
            "its", "our", "their", "what", "which", "who", "where", "when",
            "why", "how", "if", "but", "and", "or", "so", "yet", "for",
        }

        # Find capitalized words/phrases
        cap_pattern = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")

        for match in cap_pattern.finditer(text):
            name = match.group(1).strip()
            name_lower = name.lower()

            # Skip if common word or already seen
            if name_lower in skip_words or name_lower in seen_names:
                continue

            # Skip single short words at sentence start
            if len(name) < 3:
                continue

            seen_names.add(name_lower)

            # Guess entity type from context
            entity_type = self._guess_entity_type(name, text)

            entity = Entity(
                uid=f"entity_{uuid.uuid4().hex[:12]}",
                name=name,
                entity_type=entity_type,
                description=None,
                created_at=datetime.now(timezone.utc),
            )
            entities.append(entity)

        return entities[:10]  # Limit results

    def _guess_entity_type(self, name: str, context: str) -> str:
        """Guess entity type based on name and context."""
        name_lower = name.lower()
        context_lower = context.lower()

        # Check for person indicators
        person_titles = ["mr.", "mrs.", "ms.", "dr.", "prof."]
        for title in person_titles:
            if title in context_lower and name_lower in context_lower:
                return "PERSON"

        # Check for organization indicators
        org_suffixes = ["inc", "corp", "llc", "ltd", "company", "organization"]
        for suffix in org_suffixes:
            if suffix in name_lower:
                return "ORGANIZATION"

        # Default to CONCEPT for abstract things, THING for concrete
        return "CONCEPT"


class KnowledgeExtractor:
    """
    Combined extractor for triples and entities.

    Convenience class that runs both extractions and can link
    entities to their mentions in the knowledge graph.
    """

    def __init__(self, model: Optional[TextGenerator] = None):
        """
        Initialize the knowledge extractor.

        Args:
            model: Text generation model (optional)
        """
        self.triple_extractor = TripleExtractor(model)
        self.entity_extractor = EntityExtractor(model)

    def set_model(self, model: TextGenerator) -> None:
        """Set the model for both extractors."""
        self.triple_extractor.set_model(model)
        self.entity_extractor.set_model(model)

    async def extract_all(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> tuple[list[Triple], list[Entity]]:
        """
        Extract both triples and entities from text.

        Args:
            text: Text to extract from
            source_fragment_uid: Source fragment UID for triples

        Returns:
            Tuple of (triples, entities)
        """
        triples = await self.triple_extractor.extract(text, source_fragment_uid)
        entities = await self.entity_extractor.extract(text)
        return triples, entities

    def extract_all_sync(
        self,
        text: str,
        source_fragment_uid: Optional[str] = None,
    ) -> tuple[list[Triple], list[Entity]]:
        """
        Extract both triples and entities using heuristics (no LLM).

        Args:
            text: Text to extract from
            source_fragment_uid: Source fragment UID for triples

        Returns:
            Tuple of (triples, entities)
        """
        triples = self.triple_extractor.extract_sync(text, source_fragment_uid)
        entities = self.entity_extractor.extract_sync(text)
        return triples, entities


class InsightExtractor:
    """
    Extract distilled insights from text using LLM.

    Extracts condensed, reusable pieces of knowledge that can be stored
    in the Zettel library for compounding knowledge across cognitive cycles.
    """

    def __init__(
        self,
        model: Optional[TextGenerator] = None,
    ):
        """
        Initialize the insight extractor.

        Args:
            model: Text generation model (optional, can set later)
        """
        self._model = model

    def set_model(self, model: TextGenerator) -> None:
        """Set the model for extraction."""
        self._model = model

    async def extract(
        self,
        text: str,
        max_insights: int = 5,
    ) -> list[ExtractedInsight]:
        """
        Extract insights from text using the model.

        Args:
            text: Text to extract insights from
            max_insights: Maximum number of insights to extract

        Returns:
            List of ExtractedInsight objects
        """
        if not self._model:
            logger.warning("No model set for insight extraction")
            return []

        prompt = INSIGHT_EXTRACTION_PROMPT.format(text=text, max_insights=max_insights)

        try:
            response = await self._model.generate(prompt, max_tokens=1024)
            insights = self._parse_insights_response(response)
            logger.info(f"Extracted {len(insights)} insights from text")
            return insights
        except Exception as e:
            logger.error(f"Insight extraction failed: {e}")
            return []

    def _parse_insights_response(self, response: str) -> list[ExtractedInsight]:
        """Parse LLM response to extract insight objects."""
        # Find JSON array in response using bracket-aware extraction
        json_str = extract_json_array(response)
        if not json_str:
            logger.debug(f"No JSON array found in response: {response[:200]}")
            return []

        json_str = _sanitize_json_string(json_str)

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                logger.debug(f"Parsed JSON is not a list: {type(data)}")
                return []

            insights = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                insight_text = item.get("insight_text", "").strip()
                question_text = item.get("question_text")
                if question_text:
                    question_text = question_text.strip() or None
                epistemic_status = item.get("epistemic_status", "observation").strip().lower()

                # Validate epistemic status
                valid_statuses = {"claim", "observation", "speculation", "question"}
                if epistemic_status not in valid_statuses:
                    epistemic_status = "observation"

                if insight_text:
                    insight = ExtractedInsight(
                        insight_text=insight_text,
                        question_text=question_text,
                        epistemic_status=epistemic_status,  # type: ignore[arg-type]
                    )
                    insights.append(insight)

            return insights

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse insight JSON: {e}")
            logger.debug(f"JSON string was: {json_str[:200]}")
            return []

    def extract_sync(self, text: str, max_insights: int = 5) -> list[ExtractedInsight]:
        """
        Extract insights using simple heuristics (no LLM required).

        Uses sentence structure patterns to find potential insights.
        Less accurate but works without a model.

        Args:
            text: Text to extract from
            max_insights: Maximum insights to extract

        Returns:
            List of ExtractedInsight objects
        """
        insights = []

        # Pattern 1: Sentences starting with insight markers
        insight_markers = [
            r"The key (insight|finding|point) is",
            r"(Importantly|Notably|Crucially)",
            r"This (suggests|implies|reveals|shows)",
            r"What (this|we) (means|learn)",
        ]

        # Pattern 2: Question sentences
        question_pattern = re.compile(r"([^.!?]+\?)", re.IGNORECASE)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            # Check for insight markers
            is_insight = False
            for marker in insight_markers:
                if re.search(marker, sentence, re.IGNORECASE):
                    is_insight = True
                    break

            if is_insight:
                insight = ExtractedInsight(
                    insight_text=sentence,
                    question_text=None,
                    epistemic_status="claim",
                )
                insights.append(insight)

            # Check for questions
            elif sentence.endswith("?"):
                insight = ExtractedInsight(
                    insight_text=f"Question raised: {sentence}",
                    question_text=sentence,
                    epistemic_status="question",
                )
                insights.append(insight)

            if len(insights) >= max_insights:
                break

        return insights
