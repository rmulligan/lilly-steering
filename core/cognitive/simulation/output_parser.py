"""Parser for Graph-Preflexor structured output.

Graph-Preflexor produces output with sentinel blocks:
- <brainstorm>...</brainstorm> - Divergent exploration
- <graph>...</graph> - Human-readable graph description
- <graph_json>...</graph_json> - Machine-parseable JSON graph
- <patterns>...</patterns> - Discovered abstractions
- <synthesis>...</synthesis> - Final conclusions with structured fields

The synthesis block supports structured fields for outcome-based steering:
- hypothesis: The hypothesis statement
- cognitive_operation: One of bridge-building, tension-seeking, assumption-questioning,
  scale-shifting, pattern-recognition
- confidence: Confidence level 0.0-1.0
- positive_exemplar: Example text demonstrating the desired cognitive pattern
- negative_exemplar: Example text demonstrating what to avoid
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Valid cognitive operations for outcome-based steering
VALID_COGNITIVE_OPERATIONS = frozenset({
    "bridge-building",
    "tension-seeking",
    "assumption-questioning",
    "scale-shifting",
    "pattern-recognition",
})


@dataclass
class SynthesisFields:
    """Structured fields extracted from the synthesis block.

    Used for outcome-based steering vector extraction.

    Attributes:
        hypothesis: The hypothesis statement being tested
        cognitive_operation: The cognitive operation (e.g., bridge-building)
        confidence: Confidence level 0.0-1.0
        positive_exemplar: Example text demonstrating the desired pattern
        negative_exemplar: Example text demonstrating what to avoid
        raw_text: The full synthesis text (for backwards compatibility)
    """

    hypothesis: str = ""
    cognitive_operation: str = ""
    confidence: float = 0.5
    positive_exemplar: str = ""
    negative_exemplar: str = ""
    raw_text: str = ""

    @property
    def has_contrastive_pair(self) -> bool:
        """Check if both positive and negative exemplars are present."""
        return bool(self.positive_exemplar and self.negative_exemplar)

    @property
    def has_valid_cognitive_operation(self) -> bool:
        """Check if cognitive operation is valid."""
        return self.cognitive_operation in VALID_COGNITIVE_OPERATIONS


@dataclass
class ParsedPreflexorOutput:
    """Structured output from Graph-Preflexor generation.

    Attributes:
        brainstorm: Content from <brainstorm> block (divergent exploration)
        graph_readable: Content from <graph> block (human-readable)
        graph_json: Parsed JSON from <graph_json> block (machine-parseable)
        patterns: List of patterns from <patterns> block
        hypotheses_json: List of hypotheses from <hypotheses_json> block
        predictions_json: List of predictions from <predictions_json> block
        synthesis: Content from <synthesis> block (conclusions)
        synthesis_fields: Structured fields parsed from synthesis block
        thinking_trace: Any content outside sentinel blocks (thinking)
        raw_output: The complete raw output for debugging
    """

    brainstorm: str = ""
    graph_readable: str = ""
    graph_json: dict = field(default_factory=dict)
    patterns: list[str] = field(default_factory=list)
    hypotheses_json: list[dict] = field(default_factory=list)
    predictions_json: list[dict] = field(default_factory=list)
    synthesis: str = ""
    synthesis_fields: Optional[SynthesisFields] = None
    thinking_trace: str = ""
    raw_output: str = ""

    @property
    def has_valid_graph(self) -> bool:
        """Check if parsed output contains a valid graph structure."""
        return bool(self.graph_json.get("nodes"))

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return len(self.graph_json.get("nodes", []))

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return len(self.graph_json.get("edges", []))

    def get_nodes(self) -> list[dict]:
        """Get list of graph nodes."""
        return self.graph_json.get("nodes", [])

    def get_edges(self) -> list[dict]:
        """Get list of graph edges."""
        return self.graph_json.get("edges", [])

    @property
    def has_structured_hypotheses(self) -> bool:
        """Check if parsed output contains structured hypotheses."""
        return bool(self.hypotheses_json)

    @property
    def has_structured_predictions(self) -> bool:
        """Check if parsed output contains structured predictions."""
        return bool(self.predictions_json)

    @property
    def hypothesis_count(self) -> int:
        """Number of structured hypotheses."""
        return len(self.hypotheses_json)

    @property
    def prediction_count(self) -> int:
        """Number of structured predictions."""
        return len(self.predictions_json)

    def get_hypotheses(self) -> list[dict]:
        """Get list of structured hypotheses."""
        return self.hypotheses_json

    def get_predictions(self) -> list[dict]:
        """Get list of structured predictions."""
        return self.predictions_json

    @property
    def has_contrastive_pair(self) -> bool:
        """Check if synthesis has both positive and negative exemplars."""
        return self.synthesis_fields is not None and self.synthesis_fields.has_contrastive_pair


class PreflexorOutputParser:
    """Parser for Graph-Preflexor sentinel block output.

    Extracts structured content from the model's <brainstorm>, <graph>,
    <graph_json>, <patterns>, and <synthesis> blocks.
    """

    # Regex patterns for sentinel blocks (DOTALL for multiline content)
    BLOCK_PATTERNS = {
        "brainstorm": re.compile(r"<brainstorm>(.*?)</brainstorm>", re.DOTALL),
        "graph": re.compile(r"<graph>(.*?)</graph>", re.DOTALL),
        "graph_json": re.compile(r"<graph_json>(.*?)</graph_json>", re.DOTALL),
        "patterns": re.compile(r"<patterns>(.*?)</patterns>", re.DOTALL),
        "hypotheses_json": re.compile(r"<hypotheses_json>(.*?)</hypotheses_json>", re.DOTALL),
        "predictions_json": re.compile(r"<predictions_json>(.*?)</predictions_json>", re.DOTALL),
        "synthesis": re.compile(r"<synthesis>(.*?)</synthesis>", re.DOTALL),
    }

    # Pattern for individual list items in patterns block
    PATTERN_ITEM_RE = re.compile(r"^\s*[-*]\s*(.+)$", re.MULTILINE)

    # Pattern for extracting predictions with optional goal fields
    # Format: PREDICTION: claim | condition | goal:X | +delta
    # Parts 3 (goal) and 4 (delta) are optional
    PREDICTION_PATTERN = re.compile(
        r"^[ \t]*PREDICTION:\s*"  # Line starts with PREDICTION:
        r"([^|]+?)"  # Group 1: claim (everything up to first |)
        r"\s*\|\s*"  # First pipe separator
        r"([^|]+?)"  # Group 2: condition (everything up to second | or end)
        r"(?:\s*\|\s*(goal:[^\s|]+))?"  # Group 3 (optional): goal:uid
        r"(?:\s*\|\s*([+-]?\d+\.?\d*))?"  # Group 4 (optional): delta (e.g., +0.15 or -0.20)
        r"\s*$",  # End of line
        re.MULTILINE | re.IGNORECASE
    )

    # Common abbreviations that should NOT be treated as sentence endings
    # These cause false truncation when using simple `.+?\.` patterns
    ABBREVIATIONS = {
        "e.g", "i.e", "etc", "vs", "cf", "al", "no", "fig", "dr", "mr",
        "mrs", "ms", "st", "vol", "pp", "ed", "eds", "et", "approx",
    }

    def parse(self, raw_output: str) -> ParsedPreflexorOutput:
        """Parse Graph-Preflexor output into structured format.

        Args:
            raw_output: Raw text output from Graph-Preflexor

        Returns:
            ParsedPreflexorOutput with extracted content
        """
        result = ParsedPreflexorOutput(raw_output=raw_output)

        # Extract brainstorm block
        brainstorm_match = self.BLOCK_PATTERNS["brainstorm"].search(raw_output)
        if brainstorm_match:
            result.brainstorm = brainstorm_match.group(1).strip()

        # Extract graph block (human-readable)
        graph_match = self.BLOCK_PATTERNS["graph"].search(raw_output)
        if graph_match:
            result.graph_readable = graph_match.group(1).strip()

        # Extract and parse graph_json block
        graph_json_match = self.BLOCK_PATTERNS["graph_json"].search(raw_output)
        if graph_json_match:
            result.graph_json = self._parse_graph_json(graph_json_match.group(1))

        # Extract patterns block and parse into list
        patterns_match = self.BLOCK_PATTERNS["patterns"].search(raw_output)
        if patterns_match:
            result.patterns = self._parse_patterns(patterns_match.group(1))

        # Extract and parse hypotheses_json block (structured hypotheses)
        hypotheses_match = self.BLOCK_PATTERNS["hypotheses_json"].search(raw_output)
        if hypotheses_match:
            result.hypotheses_json = self._parse_json_list(
                hypotheses_match.group(1), "hypotheses"
            )

        # Extract and parse predictions_json block (structured predictions)
        predictions_match = self.BLOCK_PATTERNS["predictions_json"].search(raw_output)
        if predictions_match:
            result.predictions_json = self._parse_json_list(
                predictions_match.group(1), "predictions"
            )

        # Extract synthesis block and parse structured fields
        synthesis_match = self.BLOCK_PATTERNS["synthesis"].search(raw_output)
        if synthesis_match:
            synthesis_content = synthesis_match.group(1).strip()
            result.synthesis = synthesis_content
            result.synthesis_fields = self._parse_synthesis_block(synthesis_content)

        # Extract thinking trace (content outside sentinel blocks)
        result.thinking_trace = self._extract_thinking_trace(raw_output)

        # Log parsing results
        logger.debug(
            f"Parsed Preflexor output: brainstorm={len(result.brainstorm)} chars, "
            f"graph_nodes={result.node_count}, patterns={len(result.patterns)}, "
            f"hypotheses={len(result.hypotheses_json)}, predictions={len(result.predictions_json)}, "
            f"synthesis={len(result.synthesis)} chars"
        )

        return result

    def _parse_graph_json(self, json_content: str) -> dict:
        """Parse JSON graph structure.

        Args:
            json_content: Raw JSON string from <graph_json> block

        Returns:
            Parsed dict with nodes and edges, or empty dict on failure
        """
        json_content = json_content.strip()
        if not json_content:
            return {}

        try:
            parsed = json.loads(json_content)

            # Validate expected structure
            if not isinstance(parsed, dict):
                logger.warning("Graph JSON is not a dict, wrapping")
                return {"nodes": [], "edges": [], "data": parsed}

            # Ensure nodes and edges keys exist
            if "nodes" not in parsed:
                parsed["nodes"] = []
            if "edges" not in parsed:
                parsed["edges"] = []

            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse graph JSON: {e}")
            # Try to extract any JSON-like structure
            return self._fallback_json_parse(json_content)

    def _fallback_json_parse(self, content: str) -> dict:
        """Attempt fallback JSON parsing with common fixes.

        Args:
            content: JSON-like content that failed standard parsing

        Returns:
            Parsed dict or empty dict
        """
        # Try common fixes
        fixes = [
            # Remove trailing commas
            (r",\s*([}\]])", r"\1"),
            # Fix single quotes to double quotes
            (r"'", '"'),
            # Remove JavaScript comments
            (r"//[^\n]*\n", "\n"),
        ]

        fixed = content
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.warning("Fallback JSON parsing failed")
            return {}

    def _parse_json_list(self, json_content: str, block_name: str) -> list[dict]:
        """Parse a JSON array from a sentinel block.

        Args:
            json_content: Raw JSON string (expected to be an array)
            block_name: Name of the block for logging purposes

        Returns:
            List of parsed dicts, or empty list on failure
        """
        json_content = json_content.strip()
        if not json_content:
            return []

        try:
            parsed = json.loads(json_content)

            if isinstance(parsed, list):
                # Validate each item is a dict
                return [item for item in parsed if isinstance(item, dict)]
            elif isinstance(parsed, dict):
                # Single item wrapped in object - return as single-item list
                logger.warning(f"{block_name} JSON was object, expected array")
                return [parsed]
            else:
                logger.warning(f"{block_name} JSON has unexpected type: {type(parsed)}")
                return []

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {block_name} JSON: {e}")
            # Try fallback parsing
            return self._fallback_json_list_parse(json_content, block_name)

    def _fallback_json_list_parse(self, content: str, block_name: str) -> list[dict]:
        """Attempt fallback JSON list parsing with common fixes.

        Args:
            content: JSON-like content that failed standard parsing
            block_name: Name of the block for logging

        Returns:
            List of parsed dicts or empty list
        """
        # Apply common fixes
        fixes = [
            (r",\s*([}\]])", r"\1"),  # Remove trailing commas
            (r"'", '"'),  # Fix single quotes
            (r"//[^\n]*\n", "\n"),  # Remove JS comments
        ]

        fixed = content
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, fixed)

        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            return []
        except json.JSONDecodeError:
            logger.warning(f"Fallback {block_name} JSON parsing failed")
            return []

    def _parse_patterns(self, patterns_content: str) -> list[str]:
        """Parse patterns block into list of pattern strings.

        Args:
            patterns_content: Raw content from <patterns> block

        Returns:
            List of individual pattern strings
        """
        patterns_content = patterns_content.strip()
        if not patterns_content:
            return []

        # Try to parse as list items (markdown-style bullets)
        items = self.PATTERN_ITEM_RE.findall(patterns_content)
        if items:
            return [item.strip() for item in items if item.strip()]

        # Try numbered list format
        numbered_items = re.findall(r"^\s*\d+[.)]\s*(.+)$", patterns_content, re.MULTILINE)
        if numbered_items:
            return [item.strip() for item in numbered_items if item.strip()]

        # Fall back to splitting by newlines
        lines = [line.strip() for line in patterns_content.split("\n") if line.strip()]
        return lines

    def _parse_synthesis_block(self, synthesis_content: str) -> SynthesisFields:
        """Parse structured fields from synthesis block.

        The synthesis block can contain YAML-like structured fields for
        outcome-based steering:

        <synthesis>
        hypothesis: Bridge-building between concepts yields insight
        cognitive_operation: bridge-building
        confidence: 0.8

        positive_exemplar: |
          I notice that consciousness shares a deep pattern with emergence...

        negative_exemplar: |
          Consciousness is defined as subjective experience...
        </synthesis>

        Args:
            synthesis_content: Raw content from <synthesis> block

        Returns:
            SynthesisFields with extracted structured data
        """
        fields = SynthesisFields(raw_text=synthesis_content)

        if not synthesis_content:
            return fields

        # Regex patterns for field extraction
        # Simple single-line fields: "field_name: value"
        simple_field_re = re.compile(
            r"^(hypothesis|cognitive_operation|confidence):\s*(.+?)$",
            re.MULTILINE
        )

        # Multi-line fields with YAML-like block scalar: "field_name: |"
        # followed by indented content
        multiline_field_re = re.compile(
            r"^(positive_exemplar|negative_exemplar):\s*\|?\s*\n((?:[ \t]+.+\n?)+)",
            re.MULTILINE
        )

        # Extract simple fields
        for match in simple_field_re.finditer(synthesis_content):
            field_name = match.group(1)
            value = match.group(2).strip()

            if field_name == "hypothesis":
                fields.hypothesis = value
            elif field_name == "cognitive_operation":
                # Normalize to lowercase with hyphens
                normalized = value.lower().strip().replace("_", "-")
                if normalized in VALID_COGNITIVE_OPERATIONS:
                    fields.cognitive_operation = normalized
                else:
                    logger.warning(
                        f"Invalid cognitive_operation '{value}', "
                        f"expected one of {VALID_COGNITIVE_OPERATIONS}"
                    )
            elif field_name == "confidence":
                try:
                    conf = float(value)
                    fields.confidence = max(0.0, min(1.0, conf))
                except ValueError:
                    logger.warning(f"Invalid confidence value: {value}")

        # Extract multi-line exemplar fields
        for match in multiline_field_re.finditer(synthesis_content):
            field_name = match.group(1)
            # Dedent the multi-line content
            raw_value = match.group(2)
            # Find minimum indentation and strip it
            lines = raw_value.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                dedented = "\n".join(
                    line[min_indent:] if len(line) > min_indent else line.lstrip()
                    for line in lines
                ).strip()
            else:
                dedented = ""

            if field_name == "positive_exemplar":
                fields.positive_exemplar = dedented
            elif field_name == "negative_exemplar":
                fields.negative_exemplar = dedented

        # Log extraction results
        if fields.has_contrastive_pair:
            logger.debug(
                f"Extracted contrastive pair: operation={fields.cognitive_operation}, "
                f"positive={len(fields.positive_exemplar)} chars, "
                f"negative={len(fields.negative_exemplar)} chars"
            )

        return fields

    def _extract_thinking_trace(self, raw_output: str) -> str:
        """Extract content outside of sentinel blocks as thinking trace.

        Args:
            raw_output: Raw output to extract thinking from

        Returns:
            Text content outside sentinel blocks
        """
        # Remove all sentinel blocks
        cleaned = raw_output
        for pattern in self.BLOCK_PATTERNS.values():
            cleaned = pattern.sub("", cleaned)

        # Clean up whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_full_sentence(self, text: str, start_pos: int) -> str:
        """Extract a complete sentence from text, handling abbreviations.

        Finds the true sentence boundary by skipping periods that are part of
        common abbreviations (e.g., i.e., etc., vs.) and stopping only at
        periods followed by whitespace + capital letter, end of text, or newline.

        Args:
            text: The text to extract from
            start_pos: Position to start extracting

        Returns:
            The complete sentence content (without trailing period)
        """
        if start_pos >= len(text):
            return ""

        # Track position in text
        pos = start_pos
        while pos < len(text):
            # Find next period
            period_idx = text.find(".", pos)
            if period_idx == -1:
                # No more periods - return rest of text
                return text[start_pos:].strip()

            # Check if this period is part of an abbreviation
            # Look back to find the word before the period
            word_start = period_idx
            while word_start > 0 and text[word_start - 1].isalpha():
                word_start -= 1
            word_before = text[word_start:period_idx].lower()

            if word_before in self.ABBREVIATIONS:
                # This is an abbreviation - skip this period
                pos = period_idx + 1
                continue

            # Check what follows the period
            after_period = text[period_idx + 1:period_idx + 3] if period_idx + 1 < len(text) else ""

            # True sentence boundary: period followed by:
            # - End of text
            # - Newline
            # - Whitespace + capital letter
            # - Whitespace + quote/paren (new sentence in quotes)
            if not after_period:
                # End of text
                return text[start_pos:period_idx].strip()

            if after_period[0] in "\n\r":
                # Newline after period
                return text[start_pos:period_idx].strip()

            if after_period[0] in " \t":
                # Check if followed by capital letter, quote, or opening paren
                if len(after_period) > 1 and (
                    after_period[1].isupper() or after_period[1] in '"\'([)'
                ):
                    return text[start_pos:period_idx].strip()

            # Not a clear sentence boundary - continue looking
            pos = period_idx + 1

        # Reached end of text
        return text[start_pos:].strip()

    def extract_hypotheses_from_synthesis(
        self, synthesis: str, confidence: float = 0.5
    ) -> list[dict]:
        """Extract potential hypothesis statements from synthesis text.

        Looks for patterns like:
        - "This suggests that..."
        - "I hypothesize that..."
        - "It follows that..."
        - "We can conclude that..."

        Args:
            synthesis: The synthesis text to parse
            confidence: Default confidence for extracted hypotheses

        Returns:
            List of dicts with 'statement' and 'confidence' keys
        """
        if not synthesis:
            return []

        # Patterns to find the START of hypothesis statements
        # We capture everything after the trigger phrase and post-process for sentence boundary
        hypothesis_start_patterns = [
            r"(?:This|It)\s+suggests?\s+that\s+",
            r"I\s+hypothesize\s+that\s+",
            r"(?:It|This)\s+follows\s+that\s+",
            r"We\s+can\s+conclude\s+that\s+",
            r"(?:The|This)\s+(?:evidence|data|pattern)s?\s+(?:indicate|show|suggest)s?\s+that\s+",
            r"(?:If|Assuming)\s+this\s+holds?,?\s+then\s+",
        ]

        hypotheses = []
        seen_statements = set()  # Deduplicate

        for pattern in hypothesis_start_patterns:
            for match in re.finditer(pattern, synthesis, re.IGNORECASE):
                # Get position after the trigger phrase
                start_pos = match.end()
                # Extract full sentence from this position
                statement = self._extract_full_sentence(synthesis, start_pos)

                if statement and len(statement) > 20 and statement not in seen_statements:
                    seen_statements.add(statement)
                    hypotheses.append({
                        "statement": statement,
                        "confidence": confidence,
                    })

        return hypotheses

    def extract_predictions_from_patterns(
        self, patterns: list[str], synthesis: str
    ) -> list[dict]:
        """Extract testable predictions from patterns and synthesis.

        Looks for:
        - Future tense statements ("will occur", "should appear")
        - Conditional predictions ("if X, then Y")
        - Expectation statements ("expect to see", "anticipate that")

        Args:
            patterns: List of pattern strings
            synthesis: Synthesis text

        Returns:
            List of dicts with prediction data
        """
        predictions = []
        combined_text = "\n".join(patterns) + "\n" + synthesis
        seen_claims = set()  # Deduplicate

        # Patterns to find the START of prediction statements
        prediction_start_patterns = [
            r"(?:I|We)\s+(?:expect|anticipate|predict)\s+(?:to\s+see\s+)?that?\s+",
            r"(?:This|It)\s+(?:should|will|would)\s+(?:lead\s+to|result\s+in|cause)\s+",
            r"(?:The|This)\s+implies\s+that\s+",
        ]

        for pattern in prediction_start_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                start_pos = match.end()
                claim = self._extract_full_sentence(combined_text, start_pos)

                if claim and len(claim) > 20 and claim not in seen_claims:
                    seen_claims.add(claim)
                    predictions.append({
                        "claim": claim,
                        "confidence": 0.5,
                        "condition_type": "time_based",
                        "condition_value": "5",  # Default to 5 cycles
                    })

        # Special handling for "If X, then Y" conditionals
        if_then_pattern = r"If\s+(.+?),\s+then\s+"
        for match in re.finditer(if_then_pattern, combined_text, re.IGNORECASE):
            condition = match.group(1).strip()
            consequent_start = match.end()
            consequent = self._extract_full_sentence(combined_text, consequent_start)

            if condition and consequent:
                claim = f"If {condition}, then {consequent}"
                if len(claim) > 20 and claim not in seen_claims:
                    seen_claims.add(claim)
                    predictions.append({
                        "claim": claim,
                        "confidence": 0.5,
                        "condition_type": "time_based",
                        "condition_value": "5",  # Default to 5 cycles
                    })

        return predictions

    def _extract_predictions(self, text: str) -> list[dict]:
        """Extract predictions with goal fields from structured prediction lines.

        Parses PREDICTION lines in the format:
            PREDICTION: claim | condition | goal:X | +delta

        The goal and delta fields are optional. Examples:
            PREDICTION: increased integration | after 5 cycles | goal:understand_my_nature | +0.15
            PREDICTION: pattern stabilizes | after 3 cycles
            PREDICTION: coherence increases | after 10 cycles | goal:epistemic_growth

        Args:
            text: Text containing PREDICTION lines to parse

        Returns:
            List of dicts with keys:
                - claim: The prediction claim
                - condition: Verification condition (e.g., "after 5 cycles")
                - target_goal_uid: Optional goal UID (e.g., "goal:understand_my_nature")
                - expected_goal_delta: Expected alignment change (default 0.0)
        """
        if not text:
            return []

        predictions = []
        seen_claims = set()  # Deduplicate identical predictions

        for match in self.PREDICTION_PATTERN.finditer(text):
            claim = match.group(1).strip()
            condition = match.group(2).strip()
            goal_uid = match.group(3)  # May be None
            delta_str = match.group(4)  # May be None

            # Skip duplicates
            if claim in seen_claims:
                continue
            seen_claims.add(claim)

            # Parse delta, defaulting to 0.0 if not provided
            expected_delta = 0.0
            if delta_str:
                try:
                    expected_delta = float(delta_str)
                except ValueError:
                    logger.warning(f"Invalid delta value in prediction: {delta_str}")

            prediction = {
                "claim": claim,
                "condition": condition,
                "target_goal_uid": goal_uid,  # None if not provided
                "expected_goal_delta": expected_delta,
            }
            predictions.append(prediction)

        logger.debug(f"Extracted {len(predictions)} predictions with goal fields from text")
        return predictions
