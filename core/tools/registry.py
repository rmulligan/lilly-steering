"""Tool registry for managing available tools.

Provides registration, lookup, and execution of tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

ToolFunction = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class Tool:
    """A registered tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction
    required_params: list[str] = field(default_factory=list)

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }


class ToolRegistry:
    """Registry for managing tools available to the model."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: ToolFunction,
        required: Optional[list[str]] = None,
    ) -> Tool:
        """Register a new tool.

        Args:
            name: Tool name (must be unique)
            description: Human-readable description
            parameters: Parameter schema (OpenAI format)
            function: Async function to execute
            required: List of required parameter names

        Returns:
            The registered Tool instance
        """
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required_params=required or [],
        )
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[dict]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_openai_format() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with arguments.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            KeyError: If tool not found
            TypeError: If required arguments missing
        """
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool not found: {name}")

        missing = [p for p in tool.required_params if p not in arguments]
        if missing:
            raise TypeError(f"Missing required arguments for {name}: {missing}")

        logger.debug(f"Executing tool {name} with args: {arguments}")
        return await tool.function(**arguments)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False
