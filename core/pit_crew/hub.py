"""Letta hub client for pit crew lead agent.

The pit crew lead agent maintains canonical memory about Lilly's subsystems
and coordinates specialist agents for development issues.

Memory blocks (8 total):
- core_directives: Role as development hub coordinator
- subsystem_status: Current health of each subsystem (steering, simulation, substrate, etc.)
- active_tickets: Open tickets and their resolution state
- specialist_roster: Available specialists and their domains
- cross_agent_questions: Questions requiring coordination across domains
- recent_findings: Rolling window of specialist discoveries
- pr_pipeline: Proposed changes and their implementation status
- guidance_queue: Prepared context for developer queries

Data flow:
1. create_ticket() - Register new development issue
2. dispatch_specialists() - Spawn relevant specialists in parallel
3. synthesize_reports() - Integrate specialist findings
4. update_hub() - Persist discoveries to memory blocks
5. get_recommendations() - Provide actionable next steps
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

# Lazy import for letta_client
_LETTA_CLIENT_AVAILABLE = False
try:
    from letta_client import Letta

    _LETTA_CLIENT_AVAILABLE = True
except ImportError:
    Letta = None  # type: ignore

if TYPE_CHECKING:
    from letta_client import Letta as LettaClient

    from core.pit_crew.schemas import SpecialistReport


@dataclass
class HubTicket:
    """Development ticket in the hub.

    Represents a single issue or task tracked by the Lead agent.
    """

    ticket_id: str
    title: str
    subsystems: list[str]  # Which subsystems affected
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "open"  # open, investigating, implementing, resolved
    priority: str = "normal"  # low, normal, high, critical
    specialists_consulted: list[str] = field(default_factory=list)
    pr_numbers: list[str] = field(default_factory=list)

    def to_message(self) -> str:
        """Format as natural language message for Letta agent."""
        parts = [
            f"Ticket {self.ticket_id}: {self.title}",
            f"- Subsystems: {', '.join(self.subsystems)}",
            f"- Status: {self.status}",
            f"- Priority: {self.priority}",
        ]
        if self.specialists_consulted:
            parts.append(f"- Specialists: {', '.join(self.specialists_consulted)}")
        if self.pr_numbers:
            parts.append(f"- PRs: {', '.join(self.pr_numbers)}")
        return "\n".join(parts)


@dataclass
class SpecialistSynthesis:
    """Synthesis of multiple specialist reports.

    Integrates findings from specialists into coherent recommendations.
    """

    ticket_id: str
    specialists: list[str]
    common_findings: list[str] = field(default_factory=list)
    conflicting_findings: list[str] = field(default_factory=list)
    cross_agent_questions: list[str] = field(default_factory=list)
    recommended_prs: list[dict] = field(default_factory=list)  # {title, risk, files}
    confidence: str = "medium"  # low, medium, high

    def to_message(self) -> str:
        """Format as natural language message for Letta agent."""
        parts = [
            f"Synthesis for {self.ticket_id}:",
            f"- Consulted: {', '.join(self.specialists)}",
            f"- Confidence: {self.confidence}",
        ]

        if self.common_findings:
            parts.append("\nCommon findings:")
            parts.extend(f"  - {finding}" for finding in self.common_findings)

        if self.conflicting_findings:
            parts.append("\nConflicting findings:")
            parts.extend(f"  - {finding}" for finding in self.conflicting_findings)

        if self.cross_agent_questions:
            parts.append("\nCross-agent questions:")
            parts.extend(f"  - {q}" for q in self.cross_agent_questions)

        if self.recommended_prs:
            parts.append(f"\nRecommended PRs: {len(self.recommended_prs)}")

        return "\n".join(parts)


@dataclass
class HubGuidance:
    """Guidance from the Lead agent for developer queries.

    Provides context-aware recommendations based on hub state.
    """

    recommendations: list[str] = field(default_factory=list)  # Actionable steps
    warnings: list[str] = field(default_factory=list)  # Potential risks
    context: str = ""  # Background from hub memory
    confidence: float = 0.5  # How confident (0-1)

    @classmethod
    def empty(cls) -> "HubGuidance":
        """Return empty guidance for fallback cases."""
        return cls(recommendations=[], warnings=[], context="", confidence=0.0)


class PitCrewHubClient:
    """Client for Letta pit crew lead agent.

    Provides async methods for:
    - create_ticket(): Register new development issue
    - update_ticket(): Add specialist reports to ticket
    - synthesize_reports(): Get integrated findings
    - get_guidance(): Query for development recommendations

    Implements graceful degradation - if Letta is unavailable or times out,
    returns None/empty and allows operations to continue.
    """

    def __init__(
        self,
        timeout: float = 60.0,
        enabled: bool = True,
        agent_id: Optional[str] = None,
    ):
        """Initialize the pit crew hub client.

        Args:
            timeout: Timeout for requests in seconds (default 60s)
            enabled: Feature flag to enable/disable Letta integration
            agent_id: Letta agent ID (defaults to LETTA_PIT_CREW_AGENT_ID env var)
        """
        self._timeout = timeout
        self._enabled = enabled
        self._agent_id = agent_id or os.getenv("LETTA_PIT_CREW_AGENT_ID", "")
        self._client: Optional["LettaClient"] = None
        self._initialized = False

    def _ensure_client(self) -> bool:
        """Lazily initialize the Letta client.

        Returns:
            True if client is available and ready, False otherwise.
        """
        if not self._enabled:
            return False

        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not _LETTA_CLIENT_AVAILABLE:
            logger.warning("letta-client package not installed - Pit crew hub disabled")
            return False

        if not self._agent_id:
            logger.warning("No LETTA_PIT_CREW_AGENT_ID set - Pit crew hub disabled")
            return False

        api_key = os.getenv("LETTA_API_KEY", "")
        if not api_key:
            logger.warning("No LETTA_API_KEY set - Pit crew hub disabled")
            return False

        try:
            self._client = Letta()
            logger.info(f"Pit crew hub client initialized for agent {self._agent_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Letta client: {e}")
            return False

    async def create_ticket(self, ticket: HubTicket) -> Optional[str]:
        """Create a new ticket in the hub.

        Args:
            ticket: Ticket to create

        Returns:
            Response from Lead agent, or None if unavailable.
        """
        if not self._enabled:
            return None

        if not self._ensure_client():
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._create_ticket_sync, ticket
                ),
                timeout=self._timeout,
            )
            logger.info(f"Created hub ticket: {ticket.ticket_id}")
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Hub ticket creation timed out after {self._timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Hub ticket creation failed: {e}")
            return None

    def _create_ticket_sync(self, ticket: HubTicket) -> Optional[str]:
        """Synchronous ticket creation (called from thread pool)."""
        if not self._client:
            return None

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": f"New ticket:\n{ticket.to_message()}",
                    }
                ],
            )

            # Extract the assistant's response
            # Note: Letta API uses message_type="assistant_message" instead of role="assistant"
            for msg in response.messages:
                msg_type = getattr(msg, "message_type", None)
                if (
                    msg_type == "assistant_message"
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    return msg.content

            return None
        except Exception:
            raise

    async def update_ticket(
        self, ticket_id: str, synthesis: SpecialistSynthesis
    ) -> Optional[str]:
        """Update ticket with specialist synthesis.

        Args:
            ticket_id: Ticket to update
            synthesis: Integrated specialist findings

        Returns:
            Response from Lead agent, or None if unavailable.
        """
        if not self._enabled:
            return None

        if not self._ensure_client():
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._update_ticket_sync, ticket_id, synthesis
                ),
                timeout=self._timeout,
            )
            logger.info(f"Updated hub ticket {ticket_id} with synthesis")
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Hub ticket update timed out after {self._timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Hub ticket update failed: {e}")
            return None

    def _update_ticket_sync(
        self, ticket_id: str, synthesis: SpecialistSynthesis
    ) -> Optional[str]:
        """Synchronous ticket update (called from thread pool)."""
        if not self._client:
            return None

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": f"Update for {ticket_id}:\n{synthesis.to_message()}",
                    }
                ],
            )

            # Extract the assistant's response
            # Note: Letta API uses message_type="assistant_message" instead of role="assistant"
            for msg in response.messages:
                msg_type = getattr(msg, "message_type", None)
                if (
                    msg_type == "assistant_message"
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    return msg.content

            return None
        except Exception:
            raise

    async def get_guidance(self, query: str) -> Optional[HubGuidance]:
        """Get development guidance from the hub.

        Args:
            query: Developer question or context

        Returns:
            Guidance with recommendations, or None if unavailable.
        """
        if not self._enabled:
            return None

        if not self._ensure_client():
            return None

        try:
            guidance = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._get_guidance_sync, query
                ),
                timeout=self._timeout,
            )
            if guidance:
                logger.info(f"Hub guidance received for: {query[:50]}...")
            return guidance
        except asyncio.TimeoutError:
            logger.warning(f"Hub guidance timed out after {self._timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Hub guidance failed: {e}")
            return None

    def _get_guidance_sync(self, query: str) -> Optional[HubGuidance]:
        """Synchronous guidance fetch (called from thread pool)."""
        if not self._client:
            return None

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": f"Developer query: {query}",
                    }
                ],
            )

            # Extract and parse the assistant's response
            # Note: Letta API uses message_type="assistant_message" instead of role="assistant"
            for msg in response.messages:
                msg_type = getattr(msg, "message_type", None)
                if (
                    msg_type == "assistant_message"
                    and hasattr(msg, "content")
                    and msg.content
                ):
                    return self._parse_guidance(msg.content)

            return None
        except Exception:
            raise

    def _guidance_from_dict(self, data: dict) -> HubGuidance:
        """Helper to create HubGuidance from a dictionary.

        Args:
            data: Dictionary with guidance fields

        Returns:
            HubGuidance instance
        """
        return HubGuidance(
            recommendations=data.get("recommendations", []),
            warnings=data.get("warnings", []),
            context=data.get("context", ""),
            confidence=float(data.get("confidence", 0.5)),
        )

    def _parse_guidance(self, content: str) -> HubGuidance:
        """Parse Lead agent response into HubGuidance.

        Attempts JSON parsing first, falls back to plain text.
        """
        # Try JSON markdown blocks
        json_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        match = json_block_pattern.search(content)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._guidance_from_dict(data)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning("Failed to parse JSON from markdown block: %s", e)

        # Try embedded JSON
        decoder = json.JSONDecoder()
        search_start = 0
        while search_start < len(content):
            brace_pos = content.find("{", search_start)
            if brace_pos == -1:
                break
            try:
                data, end_pos = decoder.raw_decode(content, brace_pos)
                if isinstance(data, dict):
                    return self._guidance_from_dict(data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Failed to parse embedded JSON starting at position %d: %s",
                    brace_pos,
                    e,
                )
            search_start = brace_pos + 1

        # Fallback: treat as plain text context
        return HubGuidance(
            recommendations=[],
            warnings=[],
            context=content.strip(),
            confidence=0.5,
        )
