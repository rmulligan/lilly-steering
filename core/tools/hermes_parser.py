"""Hermes tool call parser for TransformerLens.

Parses tool calls from Qwen model output in Hermes format.

Hermes Format:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "value1"}}
    </tool_call>

Also handles think blocks:
    <think>
    Internal reasoning...
    </think>

Usage:
    parser = HermesToolParser()
    result = parser.parse(model_output)

    if result.has_tool_calls:
        for call in result.tool_calls:
            tool_result = await execute_tool(call.name, call.arguments)
            continuation = parser.format_tool_result(call, tool_result)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A parsed tool call from model output."""

    name: str
    arguments: dict[str, Any]
    raw_json: str

    call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging/storage."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ThinkBlock:
    """A parsed think block from model output."""

    content: str
    start_pos: int
    end_pos: int


@dataclass
class ParseResult:
    """Result of parsing model output."""

    raw_output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    think_blocks: list[ThinkBlock] = field(default_factory=list)
    response_text: str = ""
    needs_continuation: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_thinking(self) -> bool:
        return len(self.think_blocks) > 0

    @property
    def thinking_content(self) -> str:
        """Combined content of all think blocks."""
        return "\n".join(block.content for block in self.think_blocks)


class HermesToolParser:
    """Parser for Hermes-format tool calls in model output."""

    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )

    THINK_PATTERN = re.compile(
        r"<think>\s*(.*?)\s*</think>",
        re.DOTALL | re.IGNORECASE,
    )

    INCOMPLETE_TOOL_CALL = re.compile(
        r"<tool_call>\s*(?:(?!</tool_call>).)*$",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, strict: bool = False):
        """Initialize the parser.

        Args:
            strict: If True, raise on malformed tool calls. If False, skip them.
        """
        self.strict = strict
        self._call_counter = 0

    def parse(self, output: str) -> ParseResult:
        """Parse model output for tool calls and think blocks."""
        result = ParseResult(raw_output=output)

        result.think_blocks = self._extract_think_blocks(output)
        result.tool_calls = self._extract_tool_calls(output)

        if self.INCOMPLETE_TOOL_CALL.search(output):
            result.needs_continuation = True
            logger.debug("Detected incomplete tool call, needs continuation")

        if result.tool_calls:
            result.needs_continuation = True

        result.response_text = self._extract_response_text(output)

        return result

    def _extract_think_blocks(self, output: str) -> list[ThinkBlock]:
        """Extract all think blocks from output."""
        blocks = []

        for match in self.THINK_PATTERN.finditer(output):
            blocks.append(
                ThinkBlock(
                    content=match.group(1).strip(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return blocks

    def _extract_tool_calls(self, output: str) -> list[ToolCall]:
        """Extract all tool calls from output."""
        calls = []

        for match in self.TOOL_CALL_PATTERN.finditer(output):
            raw_json = match.group(1).strip()

            try:
                data = json.loads(raw_json)

                if "name" not in data:
                    if self.strict:
                        raise ValueError("Tool call missing 'name' field")
                    logger.warning(f"Skipping tool call without name: {raw_json[:100]}")
                    continue

                self._call_counter += 1
                call = ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {}),
                    raw_json=raw_json,
                    call_id=f"call_{self._call_counter}",
                )
                calls.append(call)

            except json.JSONDecodeError as e:
                if self.strict:
                    raise ValueError(f"Invalid JSON in tool call: {e}")
                logger.warning(f"Skipping malformed tool call JSON: {raw_json[:100]}")
                continue

        return calls

    def _extract_response_text(self, output: str) -> str:
        """Extract response text (everything outside special blocks)."""
        text = self.TOOL_CALL_PATTERN.sub("", output)
        text = self.THINK_PATTERN.sub("", text)
        # Also remove incomplete tool calls (opening tag without closing)
        text = self.INCOMPLETE_TOOL_CALL.sub("", text)
        # Remove any stray tool_call tags
        text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def format_tool_result(
        self,
        call: ToolCall,
        result: Any,
        error: Optional[str] = None,
    ) -> str:
        """Format a tool result for injection back into the conversation."""
        if error:
            return (
                f"\n<tool_response>\n"
                f'{{"call_id": "{call.call_id}", "name": "{call.name}", "error": "{error}"}}\n'
                f"</tool_response>\n"
            )

        if isinstance(result, str):
            result_str = result
        else:
            try:
                result_str = json.dumps(result, indent=2, default=str)
            except (TypeError, ValueError):
                result_str = str(result)

        return (
            f"\n<tool_response>\n"
            f'{{"call_id": "{call.call_id}", "name": "{call.name}", "result": {json.dumps(result_str)}}}\n'
            f"</tool_response>\n"
        )

    def format_tool_definitions(self, tools: list[dict]) -> str:
        """Format tool definitions for inclusion in system prompt."""
        if not tools:
            return ""

        lines = ["You have access to the following tools:\n"]

        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "No description")
            params = func.get("parameters", {})

            lines.append(f"### {name}")
            lines.append(f"{desc}\n")

            props = params.get("properties", {})
            required = params.get("required", [])

            if props:
                lines.append("Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = " (required)" if param_name in required else ""
                    lines.append(f"  - {param_name}: {param_type}{req_marker} - {param_desc}")
                lines.append("")

        # Use the first actual tool name in the example to avoid model taking placeholder literally
        example_tool = tools[0].get("function", tools[0]).get("name", "search_knowledge") if tools else "search_knowledge"

        lines.append(
            f"""
To use a tool, output a tool call in this exact format:
<tool_call>
{{"name": "{example_tool}", "arguments": {{"subject": "example"}}}}
</tool_call>

You can use multiple tool calls in one response. Wait for tool results before continuing.
"""
        )

        return "\n".join(lines)


def parse_hermes_output(output: str) -> ParseResult:
    """Parse model output using default parser."""
    parser = HermesToolParser()
    return parser.parse(output)
