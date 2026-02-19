"""Feedback Aggregator: Collects and persists simulation learnings.

After cognitive processing completes, this aggregates findings and stores them
via the knowledge ingestion pipeline for HippoRAG processing.

This closes the Active Inference loop:
    Simulation -> Findings -> Persistence -> HippoRAG -> Enriched Graph -> Better Simulation

Architecture:
    1. Cognitive processing produces findings (paths, gaps, outcomes)
    2. FeedbackAggregator converts to SimulationLearning objects
    3. Learnings are stored to the knowledge graph
    4. Future processing benefits from enriched knowledge
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to slug for tagging."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:50].strip("-")


@dataclass
class SimulationLearning:
    """A discrete learning from simulation."""

    content: str
    summary: str
    learning_type: str  # "path", "gap", "outcome", "insight"
    source: str
    goal: str
    metadata: dict = field(default_factory=dict)

    @property
    def tags(self) -> list[str]:
        """Generate tags for this learning."""
        return [
            "simulation",
            self.learning_type,
            _slugify(self.goal),
            f"source:{self.source}",
        ]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "summary": self.summary,
            "learning_type": self.learning_type,
            "source": self.source,
            "goal": self.goal,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class FeedbackAggregator:
    """Aggregates cognitive processing findings and persists to knowledge graph.

    Converts raw findings (paths, gaps, outcomes) into structured learnings
    and stores them for future retrieval.
    """

    # Maximum content length for single learning
    MAX_CONTENT_LENGTH = 2000

    def __init__(
        self,
        graph: Optional["PsycheClient"] = None,
        embeddings: Optional[EmbeddingProvider] = None,
    ):
        """Initialize the feedback aggregator.

        Args:
            graph: Optional PsycheClient for storage
            embeddings: Optional embedding provider for semantic storage
        """
        self.graph = graph
        self.embeddings = embeddings
        self._learnings_persisted = 0

    async def aggregate_and_persist(
        self,
        findings: list[dict],
        goal: str,
        source: str,
    ) -> int:
        """Process findings from cognitive processing.

        Args:
            findings: List of finding dicts
            goal: Original goal being pursued
            source: Which component produced these

        Returns:
            Number of learnings persisted
        """
        if not findings:
            return 0

        learnings = self._convert_to_learnings(findings, goal, source)

        if not learnings:
            return 0

        try:
            for learning in learnings:
                await self._store_learning(learning)

            self._learnings_persisted += len(learnings)

            logger.info(
                f"Persisted {len(learnings)} learnings from {source} "
                f"for goal: {goal[:50]}..."
            )

            return len(learnings)

        except Exception as e:
            logger.error(f"Failed to persist learnings: {e}")
            return 0

    async def _store_learning(self, learning: SimulationLearning) -> None:
        """Store a single learning to the graph."""
        if not self.graph:
            logger.warning("No graph client available for storage")
            return

        try:
            # Generate embedding if provider available
            embedding = None
            if self.embeddings:
                embedding = await self.embeddings.generate(learning.content)

            # Store as a fragment via Cypher query
            query = """
                CREATE (f:Fragment {
                    uid: $uid,
                    content: $content,
                    summary: $summary,
                    source: $source,
                    tags: $tags,
                    created_at: $created_at,
                    embedding: $embedding
                })
                RETURN f.uid as uid
            """
            params = {
                "uid": f"learning_{uuid.uuid4().hex[:8]}",
                "content": learning.content,
                "summary": learning.summary,
                "source": f"simulation:{learning.source}",
                "tags": learning.tags,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding": embedding,
            }

            await self.graph.query(query, params)

        except Exception as e:
            logger.warning(f"Failed to store learning: {e}")

    def _convert_to_learnings(
        self,
        findings: list[dict],
        goal: str,
        source: str,
    ) -> list[SimulationLearning]:
        """Convert raw findings to SimulationLearning objects.

        Handles various finding formats:
        - Path findings with steps and confidence
        - Gap findings with severity and description
        - Outcome findings with success/failure
        - Insight findings with content
        """
        learnings = []

        for f in findings:
            learning = self._convert_single_finding(f, goal, source)
            if learning:
                learnings.append(learning)

        return learnings

    def _convert_single_finding(
        self,
        finding: dict,
        goal: str,
        source: str,
    ) -> Optional[SimulationLearning]:
        """Convert a single finding to SimulationLearning."""
        # Determine learning type
        if "steps" in finding or "path_length" in finding:
            learning_type = "path"
            content = self._format_path_finding(finding)
            summary = f"Path to {goal[:50]} via {len(finding.get('steps', []))} steps"

        elif "severity" in finding or "gap_type" in finding:
            learning_type = "gap"
            content = self._format_gap_finding(finding)
            summary = finding.get("description", f"Gap identified for {goal[:50]}")[:100]

        elif "success" in finding or "outcome" in finding:
            learning_type = "outcome"
            success = finding.get("success", finding.get("outcome") == "success")
            content = finding.get("content", f"Outcome: {'success' if success else 'failure'}")
            summary = f"Outcome for {goal[:50]}: {'success' if success else 'failure'}"

        elif "confidence" in finding or "total_confidence" in finding:
            learning_type = "insight"
            conf = finding.get("confidence", finding.get("total_confidence", 0))
            content = f"Confidence: {conf:.0%} for path to {goal}"
            summary = f"Confidence observation: {conf:.0%}"

        else:
            learning_type = "insight"
            content = finding.get("content", finding.get("description", str(finding)))
            summary = finding.get("summary", content[:100])

        # Truncate content if too long
        if len(content) > self.MAX_CONTENT_LENGTH:
            content = content[: self.MAX_CONTENT_LENGTH] + "..."

        return SimulationLearning(
            content=content,
            summary=summary,
            learning_type=learning_type,
            source=source,
            goal=goal,
            metadata=finding.get("metadata", {}),
        )

    def _format_path_finding(self, finding: dict) -> str:
        """Format a path finding as content."""
        steps = finding.get("steps", [])
        confidence = finding.get("confidence", finding.get("total_confidence", 0))

        content = f"Path discovered with confidence {confidence:.0%}.\n\n"

        if steps:
            content += "Steps:\n"
            for i, step in enumerate(steps[:10], 1):  # Limit to 10 steps
                if isinstance(step, dict):
                    step_content = step.get("content", str(step))[:100]
                    step_conf = step.get("confidence", "N/A")
                    content += f"{i}. {step_content} (conf: {step_conf})\n"
                else:
                    content += f"{i}. {str(step)[:100]}\n"

        return content

    def _format_gap_finding(self, finding: dict) -> str:
        """Format a gap finding as content."""
        description = finding.get("description", "Unknown gap")
        severity = finding.get("severity", "MEDIUM")
        gap_type = finding.get("gap_type", "unknown")
        location = finding.get("location_hint", "unknown location")

        content = f"Knowledge gap ({severity} severity).\n\n"
        content += f"Type: {gap_type}\n"
        content += f"Location: {location}\n"
        content += f"Description: {description}\n"

        if finding.get("suggested_question"):
            content += f"\nSuggested question: {finding['suggested_question']}\n"

        return content

    @property
    def total_persisted(self) -> int:
        """Total learnings persisted since initialization."""
        return self._learnings_persisted


async def aggregate_findings(
    findings: list[dict],
    goal: str,
    source: str,
    graph: Optional["PsycheClient"] = None,
    embeddings: Optional[EmbeddingProvider] = None,
) -> int:
    """Convenience function to aggregate findings.

    Args:
        findings: List of finding dicts
        goal: Goal description
        source: Source identifier
        graph: Optional PsycheClient
        embeddings: Optional embedding provider

    Returns:
        Number of learnings persisted
    """
    aggregator = FeedbackAggregator(
        graph=graph,
        embeddings=embeddings,
    )

    return await aggregator.aggregate_and_persist(
        findings=findings,
        goal=goal,
        source=source,
    )
