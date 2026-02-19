"""Letta continuity client for developmental memory integration.

Provides a client for the lilly-continuity Letta agent that augments
Mox synthesis with long-term pattern tracking across cognitive cycles.

The Letta agent maintains 8 memory blocks:
- core_directives: Role and behavioral guidelines
- developmental_trajectory: Long-term growth patterns
- metric_trends: Rolling metric windows
- hypothesis_history: Domain success/failure tracking
- emotional_patterns: Affect trajectory tracking
- experiment_history: Self-experimentation outcomes
- active_concerns: Current developmental risks
- guidance_queue: Prepared context for Mox

Data flow per cycle:
1. get_guidance() - Fetch developmental context before Mox (sync, 60s timeout)
2. Mox synthesizes with Letta's guidance as context
3. send_cycle_summary() - Update Letta with cycle data and wait for feedback (sync, 60s timeout)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


@dataclass
class CycleSummaryForLetta:
    """Minimal cycle data for developmental tracking.

    Contains only outcomes and metrics - not raw thought content.
    Designed to be token-efficient while enabling pattern detection.
    """

    cycle_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Outcomes (not raw content)
    discoveries_count: int = 0
    insights_formed: list[str] = field(default_factory=list)  # Zettel titles only
    hypothesis_outcomes: list[tuple[str, str]] = field(
        default_factory=list
    )  # (domain, VERIFIED/FALSIFIED/PENDING)
    beliefs_changed: int = 0

    # Health & affect
    health_status: str = "STABLE"  # THRIVING/STABLE/STRESSED/CRITICAL
    valence: float = 0.0  # -1 to 1
    arousal: float = 0.5  # 0 to 1

    # Metrics
    discovery_parameter: float = 0.0
    semantic_entropy: float = 0.0
    verification_rate: float = 0.0  # Recent window

    # Experiments
    active_experiment: Optional[str] = None  # Parameter being tested
    experiment_outcome: Optional[str] = None  # If one completed this cycle

    def to_message(self) -> str:
        """Format as natural language message for Letta agent."""
        parts = [f"Cycle {self.cycle_number} summary ({self.timestamp.isoformat()}):"]

        # Outcomes
        parts.append(f"- Discoveries: {self.discoveries_count}")
        if self.insights_formed:
            parts.append(f"- Insights: {', '.join(self.insights_formed[:3])}")
        if self.hypothesis_outcomes:
            outcomes_str = "; ".join(
                f"{domain}: {outcome}" for domain, outcome in self.hypothesis_outcomes[:3]
            )
            parts.append(f"- Hypothesis outcomes: {outcomes_str}")
        if self.beliefs_changed:
            parts.append(f"- Beliefs changed: {self.beliefs_changed}")

        # Health & affect
        parts.append(f"- Health: {self.health_status}")
        parts.append(f"- Affect: valence={self.valence:.2f}, arousal={self.arousal:.2f}")

        # Metrics
        parts.append(
            f"- Metrics: D={self.discovery_parameter:.3f}, "
            f"H_sem={self.semantic_entropy:.3f}, "
            f"verify_rate={self.verification_rate:.2f}"
        )

        # Experiments
        if self.active_experiment:
            parts.append(f"- Active experiment: {self.active_experiment}")
        if self.experiment_outcome:
            parts.append(f"- Experiment outcome: {self.experiment_outcome}")

        return "\n".join(parts)


@dataclass
class DevelopmentalGuidance:
    """Letta's context injection for Mox synthesis.

    Contains 2-3 sentences of developmental context based on
    long-term pattern observation across cycles.
    """

    guidance: str  # 2-3 sentences, natural language
    active_concerns: list[str] = field(default_factory=list)  # 0-3 current concerns
    confidence: float = 0.5  # How confident Letta is in this guidance (0-1)

    @classmethod
    def empty(cls) -> "DevelopmentalGuidance":
        """Return empty guidance for fallback cases."""
        return cls(guidance="", active_concerns=[], confidence=0.0)


class LettaContinuityClient:
    """Client for Letta developmental continuity agent.

    Provides async methods for:
    - get_guidance(): Fetch developmental context before Mox synthesis
    - send_cycle_summary(): Update Letta with cycle data after synthesis

    Implements graceful degradation - if Letta is unavailable or times out,
    returns None/empty and allows the cognitive cycle to continue.
    """

    def __init__(
        self,
        timeout: float = 60.0,
        enabled: bool = True,
        agent_id: Optional[str] = None,
    ):
        """Initialize the Letta continuity client.

        Args:
            timeout: Timeout for guidance requests in seconds (default 60s)
            enabled: Feature flag to enable/disable Letta integration
            agent_id: Letta agent ID (defaults to LETTA_CONTINUITY_AGENT_ID env var)
        """
        self._timeout = timeout
        self._enabled = enabled
        self._agent_id = agent_id or os.getenv("LETTA_CONTINUITY_AGENT_ID", "")
        self._client: Optional["LettaClient"] = None
        self._initialized = False

    def _ensure_client(self) -> bool:
        """Lazily initialize the Letta client.

        Returns:
            True if client is available and ready, False otherwise.
        """
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not _LETTA_CLIENT_AVAILABLE:
            logger.warning("letta-client package not installed - Letta continuity disabled")
            return False

        if not self._agent_id:
            logger.warning("No LETTA_CONTINUITY_AGENT_ID set - Letta continuity disabled")
            return False

        api_key = os.getenv("LETTA_API_KEY", "")
        if not api_key:
            logger.warning("No LETTA_API_KEY set - Letta continuity disabled")
            return False

        try:
            self._client = Letta()
            logger.info(f"Letta continuity client initialized for agent {self._agent_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Letta client: {e}")
            return False

    async def get_guidance(self) -> Optional[DevelopmentalGuidance]:
        """Fetch developmental guidance for Mox synthesis.

        Sends a request to the Letta agent asking for developmental context
        based on accumulated cycle data. Returns None on timeout or failure
        (graceful degradation).

        Returns:
            DevelopmentalGuidance with 2-3 sentences of context, or None if unavailable.
        """
        if not self._enabled:
            return None

        if not self._ensure_client():
            return None

        try:
            # Run the blocking Letta call in a thread pool with timeout
            guidance = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_guidance_sync
                ),
                timeout=self._timeout,
            )

            if guidance:
                logger.info(
                    f"Letta guidance received ({len(guidance.guidance)} chars, "
                    f"confidence={guidance.confidence:.2f})"
                )
            return guidance

        except asyncio.TimeoutError:
            logger.warning(f"Letta guidance timed out after {self._timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Letta guidance failed: {e}")
            return None

    def _fetch_guidance_sync(self) -> Optional[DevelopmentalGuidance]:
        """Synchronous guidance fetch (called from thread pool)."""
        if not self._client:
            return None

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": "Provide developmental guidance for the upcoming synthesis.",
                    }
                ],
            )

            # Extract the assistant's response content
            for msg in response.messages:
                if getattr(msg, "role", None) == "assistant" and hasattr(msg, "content") and msg.content:
                    return self._parse_guidance(msg.content)

            return None

        except Exception as e:
            logger.debug(f"Letta guidance fetch error: {e}")
            return None

    def _parse_guidance(self, content: str) -> DevelopmentalGuidance:
        """Parse Letta agent response into DevelopmentalGuidance.

        Attempts to parse JSON format first, falls back to treating
        the entire response as natural language guidance.

        Parsing strategy:
        1. Look for ```json markdown blocks via regex
        2. Use json.JSONDecoder().raw_decode() for embedded JSON objects
        3. Fall back to treating entire response as plain text guidance
        """
        import re

        # Strategy 1: Look for ```json markdown blocks via regex
        json_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        match = json_block_pattern.search(content)
        if match:
            try:
                data = json.loads(match.group(1))
                return DevelopmentalGuidance(
                    guidance=data.get("guidance", content.strip()),
                    active_concerns=data.get("active_concerns", []),
                    confidence=float(data.get("confidence", 0.5)),
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"Could not parse JSON from markdown block: {e}")

        # Strategy 2: Use raw_decode to find embedded JSON objects
        # This handles stray braces better than index/rindex
        decoder = json.JSONDecoder()
        search_start = 0
        while search_start < len(content):
            # Find the next '{' character
            brace_pos = content.find("{", search_start)
            if brace_pos == -1:
                break
            try:
                data, end_pos = decoder.raw_decode(content, brace_pos)
                if isinstance(data, dict):
                    return DevelopmentalGuidance(
                        guidance=data.get("guidance", content.strip()),
                        active_concerns=data.get("active_concerns", []),
                        confidence=float(data.get("confidence", 0.5)),
                    )
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON starting here, try next brace
                pass
            search_start = brace_pos + 1

        # Strategy 3: Fall back to treating entire response as plain text guidance
        return DevelopmentalGuidance(
            guidance=content.strip(),
            active_concerns=[],
            confidence=0.5,
        )

    async def send_cycle_summary(
        self, summary: CycleSummaryForLetta
    ) -> Optional[DevelopmentalGuidance]:
        """Send cycle summary to Letta and wait for response.

        Updates the Letta agent with cycle data for pattern tracking and
        waits for Letta's feedback to apply to the current cycle.

        Args:
            summary: Cycle data to send to Letta

        Returns:
            DevelopmentalGuidance with Letta's feedback, or None if unavailable.
        """
        if not self._enabled:
            return None

        if not self._ensure_client():
            return None

        try:
            # Wait for Letta's response with timeout
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._send_summary_sync, summary
                ),
                timeout=self._timeout,
            )
            logger.info(
                f"Letta cycle summary sent: cycle={summary.cycle_number}, "
                f"health={summary.health_status}"
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Letta summary timed out after {self._timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Letta summary send failed: {e}")
            return None

    def _send_summary_sync(
        self, summary: CycleSummaryForLetta
    ) -> Optional[DevelopmentalGuidance]:
        """Synchronous summary send (called from thread pool).

        Returns:
            Parsed guidance from Letta's response, or None.
        """
        if not self._client:
            return None

        try:
            response = self._client.agents.messages.create(
                agent_id=self._agent_id,
                messages=[
                    {
                        "role": "user",
                        "content": summary.to_message(),
                    }
                ],
            )

            # Extract and parse the assistant's response
            for msg in response.messages:
                if getattr(msg, "role", None) == "assistant" and hasattr(msg, "content") and msg.content:
                    return self._parse_guidance(msg.content)

            return None
        except Exception:
            raise
