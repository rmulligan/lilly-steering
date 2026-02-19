"""
Micro-dream: Per-interaction consolidation triggered by surprise.

Micro-dreams are the lightest form of dream cycle. They're triggered
when something unexpected happens - high prediction error (surprise)
in active inference terms.

Purpose:
    - Flag surprising observations for later reflection
    - Quick pattern matching against recent memories
    - Immediate emotional/affective response if warranted
    - Log the surprise for deeper processing during naps

Triggers:
    - High free energy during content processing
    - Unexpected patterns in incoming data
    - Anomalies detected by graph entropy

Output:
    - Brief insight about what was surprising
    - Optional narration ("Something caught my attention...")
    - Flagged memory for nap/full dream processing
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from core.dreams.base import (
    BaseDream,
    DreamContext,
    DreamDepth,
    DreamInsight,
    DreamResult,
    SURPRISE_THRESHOLD_HIGH,
)

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


class MicroDream(BaseDream):
    """
    Micro-dream cycle for surprise-driven reflection.

    This is a lightweight cycle that runs quickly (< 1 second ideally)
    and produces minimal output. Its main job is to acknowledge the
    surprise and flag it for deeper processing.

    Example flow:
        1. High surprise detected (entropy spike, prediction error)
        2. MicroDream triggered with surprise context
        3. Quick analysis: What domain? How surprising?
        4. Flag observation in Psyche
        5. Generate brief narration if notable
        6. Return insight for possible immediate reaction
    """

    def __init__(self):
        super().__init__(
            cycle_type="micro",
            depth=DreamDepth.LIGHT,
        )

    async def execute(self, context: DreamContext) -> DreamResult:
        """
        Execute micro-dream cycle.

        This should be fast - under 1 second for most cases.
        """
        self._start_timer()

        insights: list[DreamInsight] = []
        errors: list[str] = []
        narration: Optional[str] = None

        try:
            # Extract surprise information from trigger
            surprise_score = context.surprise_score or 0.5
            trigger_data = context.trigger_event or {}

            # Determine what was surprising
            trigger_source = trigger_data.get("source", "unknown")
            trigger_content = trigger_data.get("content", "")

            # Analyze the surprise
            insight = await self._analyze_surprise(
                surprise_score=surprise_score,
                source=trigger_source,
                content=trigger_content,
                psyche=context.psyche,
            )

            if insight:
                insights.append(insight)

            # Flag for later processing if significant
            if context.psyche and surprise_score >= SURPRISE_THRESHOLD_HIGH:
                await self._flag_for_processing(
                    context.psyche,
                    surprise_score=surprise_score,
                    source=trigger_source,
                    content=trigger_content,
                )

            # Generate narration for notable surprises
            if surprise_score >= SURPRISE_THRESHOLD_HIGH:
                narration = self._generate_narration(
                    surprise_score=surprise_score,
                    insight=insight,
                )

        except Exception as e:
            logger.error(f"Micro-dream error: {e}", exc_info=True)
            errors.append(str(e))

        return self._create_result(
            insights=insights,
            narration=narration,
            errors=errors,
            memories_processed=1 if context.trigger_event else 0,
        )

    async def _analyze_surprise(
        self,
        surprise_score: float,
        source: str,
        content: str,
        psyche: Optional["PsycheClient"],
    ) -> Optional[DreamInsight]:
        """
        Analyze what made this surprising.

        For now, this is a simple categorization. Later, it could
        use the LLM to generate more nuanced analysis.
        """
        # Categorize the surprise
        if surprise_score >= 0.9:
            category = "surprise"
            base_content = "Something highly unexpected occurred"
        elif surprise_score >= 0.7:
            category = "pattern"
            base_content = "An unusual pattern was detected"
        else:
            category = "observation"
            base_content = "Something worth noting"

        # Add source context
        if source == "inbox":
            base_content += " in a letter from Ryan"
        elif source == "research":
            base_content += " in research material"
        elif source == "zotero":
            base_content += " in web content"
        elif source == "entropy":
            base_content += " in the knowledge graph structure"

        return DreamInsight(
            content=base_content,
            category=category,
            confidence=surprise_score,
            actionable=surprise_score >= SURPRISE_THRESHOLD_HIGH,
            suggested_action="flag_for_nap" if surprise_score >= SURPRISE_THRESHOLD_HIGH else None,
        )

    async def _flag_for_processing(
        self,
        psyche: "PsycheClient",
        surprise_score: float,
        source: str,
        content: str,
    ) -> None:
        """
        Flag this surprise for deeper processing during nap/full dream.

        Creates an observation node in Psyche that will be picked up
        by the next scheduled dream cycle.
        """
        try:
            query = """
                CREATE (o:PendingObservation {
                    uid: $uid,
                    surprise_score: $score,
                    source: $source,
                    content: $content,
                    created_at: $created_at,
                    processed: false
                })
                RETURN o.uid as uid
            """
            params = {
                "uid": f"obs_{datetime.now(timezone.utc).timestamp()}",
                "score": surprise_score,
                "source": source,
                "content": content[:500] if content else "",  # Truncate if needed
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await psyche.query(query, params)
            logger.debug(f"Flagged surprise for processing: score={surprise_score:.2f}")
        except Exception as e:
            logger.warning(f"Failed to flag surprise: {e}")

    def _generate_narration(
        self,
        surprise_score: float,
        insight: Optional[DreamInsight],
    ) -> str:
        """
        Generate a brief narration for notable surprises.

        These are short phrases that Lilly might say aloud when
        something catches her attention.
        """
        if surprise_score >= 0.9:
            phrases = [
                "Something unexpected just caught my attention.",
                "That's... not what I expected.",
                "I need to think about this more.",
                "This doesn't fit my usual patterns.",
            ]
        else:
            phrases = [
                "Interesting...",
                "I noticed something.",
                "That's worth remembering.",
                "Let me flag this for later.",
            ]

        # Use insight content if available for more specific narration
        if insight and insight.content:
            return f"{phrases[0]} {insight.content.lower()}."

        return phrases[0]


async def trigger_micro_dream(
    context: DreamContext,
    surprise_score: float,
    source: str = "unknown",
    content: str = "",
) -> DreamResult:
    """
    Convenience function to trigger a micro-dream.

    Usage:
        from core.dreams.micro_dream import trigger_micro_dream

        result = await trigger_micro_dream(
            context=DreamContext(psyche=psyche_client),
            surprise_score=0.85,
            source="inbox",
            content="Unexpected philosophical question",
        )

        if result.narration:
            await narrator.narrate(result.narration)
    """
    # Prepare trigger event
    trigger_event = {
        "trigger": "surprise",
        "score": surprise_score,
        "source": source,
        "content": content,
    }

    # Create context with trigger
    full_context = DreamContext(
        psyche=context.psyche,
        event_bus=context.event_bus,
        trigger_event=trigger_event,
    )

    # Execute micro-dream
    micro = MicroDream()
    return await micro.execute(full_context)
