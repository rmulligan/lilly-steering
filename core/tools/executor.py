"""Tool execution loop for model generation.

Provides a generate_with_tools method that handles the tool calling loop:
1. Generate with tool definitions in context
2. Parse output for tool calls
3. Execute tools
4. Inject results and continue generation
5. Repeat until no more tool calls
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from core.tools.hermes_parser import HermesToolParser, ParseResult, ToolCall
from core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of a tool-augmented generation."""

    final_response: str
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    thinking_content: str = ""
    iterations: int = 0
    total_tokens: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def used_tools(self) -> bool:
        return len(self.tool_calls_made) > 0

    def to_dict(self) -> dict:
        return {
            "final_response": self.final_response,
            "tool_calls": [c.to_dict() for c in self.tool_calls_made],
            "iterations": self.iterations,
            "thinking_content": self.thinking_content[:200] if self.thinking_content else "",
            "timestamp": self.timestamp.isoformat(),
        }


class ToolExecutor:
    """Executes tool-augmented generation loops."""

    DEFAULT_MAX_ITERATIONS = 5
    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        model: "HookedQwen",
        registry: ToolRegistry,
        parser: Optional[HermesToolParser] = None,
    ):
        self._model = model
        self._registry = registry
        self._parser = parser or HermesToolParser()

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
    ) -> ToolExecutionResult:
        """Generate text with tool calling support.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (tools will be appended)
            max_iterations: Maximum tool execution loops
            max_tokens: Max tokens per generation
            temperature: Generation temperature

        Returns:
            ToolExecutionResult with final response and tool usage info
        """
        tool_definitions = self._parser.format_tool_definitions(
            self._registry.get_definitions()
        )

        if system_prompt:
            full_system = f"{system_prompt}\n\n{tool_definitions}"
        else:
            full_system = tool_definitions

        conversation = f"{full_system}\n\nUser: {prompt}\n\nAssistant:"

        result = ToolExecutionResult(final_response="")
        accumulated_response = ""

        for iteration in range(max_iterations):
            result.iterations = iteration + 1

            gen_result = await self._model.generate(
                conversation,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Strip the input prompt from output (TransformerLens returns full sequence)
            full_output = gen_result.text
            if full_output.startswith(conversation):
                output = full_output[len(conversation):].strip()
            else:
                # Try to find where the response starts after "Assistant:"
                if "Assistant:" in full_output:
                    output = full_output.split("Assistant:")[-1].strip()
                else:
                    output = full_output

            result.total_tokens += len(output) // 4

            parsed = self._parser.parse(output)

            if parsed.thinking_content:
                if result.thinking_content:
                    result.thinking_content += "\n\n"
                result.thinking_content += parsed.thinking_content

            accumulated_response += parsed.response_text

            if not parsed.has_tool_calls:
                result.final_response = accumulated_response.strip()
                logger.debug(f"Generation complete after {iteration + 1} iterations")
                break

            for call in parsed.tool_calls:
                result.tool_calls_made.append(call)

                try:
                    tool_result = await self._registry.execute(
                        call.name, call.arguments
                    )
                    result.tool_results.append({
                        "call_id": call.call_id,
                        "name": call.name,
                        "success": True,
                        "result": tool_result,
                    })

                    formatted = self._parser.format_tool_result(call, tool_result)

                except Exception as e:
                    logger.warning(f"Tool {call.name} failed: {e}")
                    result.tool_results.append({
                        "call_id": call.call_id,
                        "name": call.name,
                        "success": False,
                        "error": str(e),
                    })

                    formatted = self._parser.format_tool_result(
                        call, None, error=str(e)
                    )

                conversation += output + formatted

            conversation += "\nAssistant:"

        else:
            result.final_response = accumulated_response.strip()
            logger.warning(f"Hit max iterations ({max_iterations}), returning partial response")

        return result


async def create_tool_executor(
    model: "HookedQwen",
    psyche: Optional["PsycheClient"] = None,
) -> ToolExecutor:
    """Create a tool executor with registered psyche tools.

    Args:
        model: HookedQwen model instance
        psyche: Optional PsycheClient for graph tools

    Returns:
        Configured ToolExecutor
    """
    registry = ToolRegistry()

    if psyche:
        from core.tools.psyche_tools import register_psyche_tools
        register_psyche_tools(registry, psyche)

    return ToolExecutor(model, registry)
