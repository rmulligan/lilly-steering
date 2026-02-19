"""Curator tools for graph and retrieval operations.

Provides the tool registry and executor for the curator phase,
enabling the curator to query the knowledge graph, retrieve zettels,
and explore concepts during its analysis.
"""

import json
import logging
import re
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import uuid4

from core.cognitive.narration_buffer import CuratorNarrationBuffer
from core.psyche.schema import ResearchQueryResult

if TYPE_CHECKING:
    from core.cognitive.zettel import ZettelLibrary
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.client import PsycheClient
    from integrations.discord.client import DiscordClient
    from integrations.liquidsoap.client import LiquidsoapClient
    from integrations.notebooklm.client import NotebookLMIntegration

logger = logging.getLogger(__name__)


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at sentence boundary, avoiding mid-sentence cuts.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit

    Returns:
        Truncated text ending at a sentence boundary when possible
    """
    if len(text) <= max_chars:
        return text

    # Find sentence boundaries within limit
    truncated = text[:max_chars]

    # Try to find last sentence-ending punctuation
    for end_char in ['. ', '! ', '? ']:
        last_idx = truncated.rfind(end_char)
        if last_idx > max_chars // 2:  # Only use if we keep at least half
            return truncated[:last_idx + 1].strip()

    # Try period at end without space (could be last sentence)
    if truncated.rstrip().endswith('.'):
        return truncated.rstrip()

    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."

    # Last resort: hard cut
    return truncated[:max_chars - 3].strip() + "..."


def _is_technical_artifact(text: str) -> bool:
    """Check if text is a technical artifact that shouldn't be narrated.

    Detects SAE feature labels, technical patterns, and other non-semantic content
    that should be filtered from narration.

    Args:
        text: Text to check

    Returns:
        True if text appears to be a technical artifact
    """
    text_lower = text.lower()

    # Skip SAE feature patterns (e.g., "Feature 28087 (unknown, unknown...)")
    if re.match(r'^feature\s+\d+', text_lower):
        return True

    # Skip patterns with "unknown" repetition
    if text_lower.count("unknown") >= 2:
        return True

    # Skip if starts with numeric ID patterns
    if re.match(r'^\d+[\s\-_:]', text):
        return True

    # Skip very short content (likely incomplete)
    if len(text.strip()) < 10:
        return True

    return False


def notebooklm_tool(error_payload_factory: Callable[[], dict[str, Any]]):
    """Decorator for NotebookLM tool methods with availability checking and error handling.

    Reduces boilerplate by:
    1. Checking if NotebookLM is available before calling the method
    2. Returning a standardized error payload if not available
    3. Wrapping the method in try/except and returning error payload on failure

    Args:
        error_payload_factory: Callable that returns the base error payload dict
            for this tool (e.g., lambda: {"answer": "", "citations": []})

    Returns:
        Decorated async method
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self.notebooklm or not self.notebooklm.is_available:
                payload = error_payload_factory()
                payload.update({
                    "success": False,
                    "error": "Research notebook not available",
                })
                return payload

            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Research tool {func.__name__} failed: {e}")
                payload = error_payload_factory()
                payload.update({
                    "success": False,
                    "error": str(e),
                })
                return payload
        return wrapper
    return decorator


# Tool definitions in OpenAI function calling format
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "query_entity": {
        "type": "function",
        "function": {
            "name": "query_entity",
            "description": "Get information about an entity from the knowledge graph, including its type, relationships, and salience.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Entity name to look up",
                    }
                },
                "required": ["name"],
            },
        },
    },
    "query_relationships": {
        "type": "function",
        "function": {
            "name": "query_relationships",
            "description": "Find relationships connected to an entity. Returns triples where the entity is subject or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity to explore relationships for",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "description": "Direction of relationships to fetch",
                        "default": "both",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum relationships to return",
                        "default": 10,
                    },
                },
                "required": ["entity"],
            },
        },
    },
    "retrieve_zettels": {
        "type": "function",
        "function": {
            "name": "retrieve_zettels",
            "description": "Retrieve relevant insight zettels by semantic similarity to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for semantic retrieval",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of zettels to retrieve",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    "get_beliefs": {
        "type": "function",
        "function": {
            "name": "get_beliefs",
            "description": "Get committed beliefs on a topic with their confidence levels and evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to query beliefs about",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    "explore_concept": {
        "type": "function",
        "function": {
            "name": "explore_concept",
            "description": "Deep exploration of a concept - returns entities, relationships, related zettels, and context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept to explore deeply",
                    }
                },
                "required": ["concept"],
            },
        },
    },
    "narrate": {
        "type": "function",
        "function": {
            "name": "narrate",
            "description": "Speak aloud during analysis. Use this to vocalize observations, insights forming, or questions emerging. Keep narrations brief (1-2 sentences) and meaningful.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak aloud (keep brief, 1-2 sentences)",
                    }
                },
                "required": ["text"],
            },
        },
    },
    "message_ryan": {
        "type": "function",
        "function": {
            "name": "message_ryan",
            "description": "Send a message to Ryan via Discord. Use for questions, insights worth sharing, or when you want his input. Be thoughtful - only message when genuinely valuable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send (1-3 sentences ideal)",
                    },
                    "message_type": {
                        "type": "string",
                        "enum": ["question", "insight", "observation"],
                        "description": "The nature of the message",
                    },
                },
                "required": ["message", "message_type"],
            },
        },
    },
    "query_research": {
        "type": "function",
        "function": {
            "name": "query_research",
            "description": "Query the external research notebook for architectural context, design decisions, or documented knowledge. Use when planning experiments, validating hypotheses, or understanding system design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question about the architecture, design, or documented research",
                    }
                },
                "required": ["question"],
            },
        },
    },
    "get_research_summary": {
        "type": "function",
        "function": {
            "name": "get_research_summary",
            "description": "Get a high-level summary of the research notebook contents. Use for orientation about available documentation and research context.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    "add_insight_to_research": {
        "type": "function",
        "function": {
            "name": "add_insight_to_research",
            "description": "Add a crystallized insight to the research notebook for future reference. Use sparingly for significant discoveries or validated hypotheses worth preserving.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The insight to add (clear, self-contained statement)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the insight",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about how this insight was derived",
                    },
                },
                "required": ["insight"],
            },
        },
    },
}


def get_enabled_tools(enabled_names: list[str]) -> list[dict[str, Any]]:
    """Return tool definitions for enabled tools only.

    Args:
        enabled_names: List of tool names to enable

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    return [TOOL_REGISTRY[name] for name in enabled_names if name in TOOL_REGISTRY]


@dataclass
class ToolCall:
    """Represents a tool call from the curator.

    Attributes:
        name: Tool function name
        arguments: JSON arguments for the tool
        call_id: Unique identifier for this call
    """

    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ToolResult:
    """Result from executing a tool.

    Attributes:
        call_id: ID of the tool call this responds to
        content: JSON string result
        error: Error message if tool failed
    """

    call_id: str
    content: str
    error: Optional[str] = None


class CuratorTools:
    """Executor for curator tools.

    Provides async methods to execute each tool against the
    knowledge graph and retrieval systems.
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        zettel_library: Optional["ZettelLibrary"] = None,
        embedder: Optional["TieredEmbeddingService"] = None,
        liquidsoap: Optional["LiquidsoapClient"] = None,
        discord: Optional["DiscordClient"] = None,
        notebooklm: Optional["NotebookLMIntegration"] = None,
        curator_voice: str = "marius",
    ):
        """Initialize curator tools.

        Args:
            psyche: Client for knowledge graph operations
            zettel_library: Library for zettel retrieval (optional)
            embedder: Embedding service for semantic search (optional)
            liquidsoap: Client for TTS narration (optional)
            discord: Discord client for messaging Ryan (optional)
            notebooklm: NotebookLM integration for research access (optional)
            curator_voice: Voice to use for curator narration (default: marius)
        """
        self.psyche = psyche
        self.zettel_library = zettel_library
        self.embedder = embedder
        self.liquidsoap = liquidsoap
        self.discord = discord
        self.notebooklm = notebooklm
        self.curator_voice = curator_voice
        # Buffer for batching discovery narrations to reduce micro-narration fatigue
        # Flush after 5 items or 30 seconds
        self._narration_buffer = CuratorNarrationBuffer(
            max_buffer=5,
            min_interval_seconds=30.0
        )

    async def _maybe_narrate(self, discovery: str) -> None:
        """Buffer a discovery narration and narrate if ready.

        Discovery narrations are batched to reduce micro-narration fatigue.
        Call flush_narrations() at the end of curation to narrate any remaining.

        Args:
            discovery: The discovery text to buffer
        """
        if not self.liquidsoap:
            return

        text = self._narration_buffer.add(discovery)
        if text:
            await self.liquidsoap.narrate(text, voice=self.curator_voice)

    async def flush_narrations(self) -> None:
        """Flush any remaining buffered narrations.

        Call this at the end of curation phase to ensure all buffered
        discoveries are narrated before transitioning to the next phase.
        """
        if not self.liquidsoap:
            return

        text = self._narration_buffer.flush_remaining()
        if text:
            await self.liquidsoap.narrate(text, voice=self.curator_voice)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with JSON content or error
        """
        try:
            if tool_call.name == "query_entity":
                result = await self.query_entity(**tool_call.arguments)
            elif tool_call.name == "query_relationships":
                result = await self.query_relationships(**tool_call.arguments)
            elif tool_call.name == "retrieve_zettels":
                result = await self.retrieve_zettels(**tool_call.arguments)
            elif tool_call.name == "get_beliefs":
                result = await self.get_beliefs(**tool_call.arguments)
            elif tool_call.name == "explore_concept":
                result = await self.explore_concept(**tool_call.arguments)
            elif tool_call.name == "narrate":
                result = await self.narrate(**tool_call.arguments)
            elif tool_call.name == "message_ryan":
                result = await self.message_ryan(**tool_call.arguments)
            elif tool_call.name == "query_research":
                result = await self.query_research(**tool_call.arguments)
            elif tool_call.name == "get_research_summary":
                result = await self.get_research_summary()
            elif tool_call.name == "add_insight_to_research":
                result = await self.add_insight_to_research(**tool_call.arguments)
            else:
                return ToolResult(
                    call_id=tool_call.call_id,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )

            return ToolResult(
                call_id=tool_call.call_id,
                content=json.dumps(result, default=str),
            )

        except Exception as e:
            logger.warning(f"Tool {tool_call.name} failed: {e}")
            return ToolResult(
                call_id=tool_call.call_id,
                content="",
                error=str(e),
            )

    async def query_entity(
        self, name: str, *, _skip_narrate: bool = False
    ) -> dict[str, Any]:
        """Get information about an entity from the knowledge graph.

        Args:
            name: Entity name to look up
            _skip_narrate: Internal flag to skip narration (used by explore_concept)

        Returns:
            Dict with entity info (name, type, salience, etc.)
        """
        # Use case-insensitive matching to find entities regardless of casing
        # TODO: For better performance at scale, consider adding a name_lower
        # indexed property to Entity nodes instead of using toLower() in query
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($name)
        RETURN e.name AS name, e.entity_type AS type,
               e.salience AS salience, e.description AS description
        """
        results = await self.psyche.query(query, {"name": name})

        if not results:
            # Buffer discovery: entity is NOT found (interesting for novelty)
            if not _skip_narrate:
                await self._maybe_narrate(f"New territory: {name}")
            return {"found": False, "name": name}

        entity = results[0]
        entity_type = entity.get("type", "unknown")
        description = entity.get("description", "")
        salience = entity.get("salience", 0.0)

        # Buffer discovery: entity IS found
        if not _skip_narrate:
            if description:
                await self._maybe_narrate(f"Known entity: {name}")
            elif salience and salience > 0.5:
                await self._maybe_narrate(f"Significant concept: {name}")

        return {
            "found": True,
            "name": entity.get("name", name),
            "type": entity_type,
            "salience": salience,
            "description": description,
        }

    async def query_relationships(
        self,
        entity: str,
        direction: str = "both",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Find relationships connected to an entity.

        Args:
            entity: Entity to explore
            direction: outgoing, incoming, or both
            limit: Maximum relationships to return

        Returns:
            Dict with lists of relationships
        """
        relationships = {"outgoing": [], "incoming": []}

        if direction in ("outgoing", "both"):
            query = """
            MATCH (e:Entity {name: $entity})-[r]->(target:Entity)
            RETURN type(r) AS predicate, target.name AS object
            LIMIT $limit
            """
            results = await self.psyche.query(
                query, {"entity": entity, "limit": limit}
            )
            relationships["outgoing"] = [
                {"predicate": r["predicate"], "object": r["object"]} for r in results
            ]

        if direction in ("incoming", "both"):
            query = """
            MATCH (source:Entity)-[r]->(e:Entity {name: $entity})
            RETURN source.name AS subject, type(r) AS predicate
            LIMIT $limit
            """
            results = await self.psyche.query(
                query, {"entity": entity, "limit": limit}
            )
            relationships["incoming"] = [
                {"subject": r["subject"], "predicate": r["predicate"]} for r in results
            ]

        total_rels = len(relationships["outgoing"]) + len(relationships["incoming"])

        # Narrate discovered relationships
        if self.liquidsoap and total_rels > 0:
            # Pick an interesting relationship to highlight
            interesting_rel = None
            if relationships["outgoing"]:
                rel = relationships["outgoing"][0]
                interesting_rel = f"{entity} {rel['predicate'].lower().replace('_', ' ')} {rel['object']}"
            elif relationships["incoming"]:
                rel = relationships["incoming"][0]
                interesting_rel = f"{rel['subject']} {rel['predicate'].lower().replace('_', ' ')} {entity}"

            if interesting_rel:
                # Buffer discovery: relationships found
                await self._maybe_narrate(f"{total_rels} connections for {entity}")

        return {
            "entity": entity,
            "relationships": relationships,
            "total": total_rels,
        }

    async def retrieve_zettels(self, query: str, k: int = 5) -> dict[str, Any]:
        """Retrieve relevant insight zettels by semantic similarity.

        Args:
            query: Search query (used as concept and for embedding)
            k: Number of zettels to retrieve

        Returns:
            Dict with list of relevant zettels
        """
        if self.zettel_library is None or self.embedder is None:
            return {"zettels": [], "error": "Zettel library not configured"}

        try:
            # Get query embedding
            embedding = await self.embedder.encode(query)

            # Use retrieve_context with the query as concept
            # Pass phase="curation" and use_flow_scores=True for phase-aware retrieval
            context = await self.zettel_library.retrieve_context(
                concept=query,
                current_embedding=embedding.to_list(),
                semantic_limit=k,
                activation_limit=0,  # Skip activation path for simple retrieval
                question_limit=0,    # Skip questions path for simple retrieval
                phase="curation",
                use_flow_scores=True,
            )

            zettels = []
            for uid, insight, score in context.semantic_insights:
                zettels.append({
                    "uid": uid,
                    "insight": insight,
                    "similarity": round(score, 3),
                })

            # Buffer discovery: relevant zettels found
            # Previously this narrated each zettel individually - now we batch
            if zettels:
                count = len(zettels)
                await self._maybe_narrate(f"{count} memories for {query[:30]}")

            return {"query": query, "zettels": zettels, "count": len(zettels)}

        except Exception as e:
            logger.warning(f"Zettel retrieval failed: {e}")
            return {"zettels": [], "error": str(e)}

    async def get_beliefs(self, topic: str) -> dict[str, Any]:
        """Get committed beliefs on a topic.

        Args:
            topic: Topic to query beliefs about

        Returns:
            Dict with list of beliefs
        """
        query = """
        MATCH (b:CommittedBelief)
        WHERE b.topic CONTAINS $topic OR b.proposition CONTAINS $topic
        RETURN b.topic AS topic, b.proposition AS proposition,
               b.confidence AS confidence, b.evidence AS evidence
        ORDER BY b.confidence DESC
        LIMIT 5
        """
        results = await self.psyche.query(query, {"topic": topic})

        beliefs = [
            {
                "topic": r.get("topic", ""),
                "proposition": r.get("proposition", ""),
                "confidence": r.get("confidence", 0.0),
                "evidence": r.get("evidence", ""),
            }
            for r in results
        ]

        # Buffer discovery: beliefs found
        if beliefs:
            count = len(beliefs)
            await self._maybe_narrate(f"{count} beliefs about {topic}")

        return {"topic": topic, "beliefs": beliefs, "count": len(beliefs)}

    async def explore_concept(self, concept: str) -> dict[str, Any]:
        """Deep exploration of a concept.

        Combines entity lookup, relationship exploration, and zettel retrieval
        for comprehensive concept context.

        Args:
            concept: Concept to explore

        Returns:
            Dict with comprehensive concept context
        """
        # Narrate the exploration (third person clinical)
        if self.liquidsoap:
            # Format concept name for natural speech (underscores -> spaces, camelCase -> spaces)
            display_concept = concept.replace("_", " ").replace("-", " ")
            await self.liquidsoap.narrate(
                f"Exploring the concept of {display_concept} in her mental landscape...",
                voice=self.curator_voice
            )

        # Get entity info (skip narration since we already narrated exploration)
        entity_info = await self.query_entity(concept, _skip_narrate=True)

        # Get relationships
        relationships = await self.query_relationships(concept, direction="both", limit=5)

        # Get related zettels
        zettels = await self.retrieve_zettels(concept, k=3)

        # Get any beliefs related to concept
        beliefs = await self.get_beliefs(concept)

        return {
            "concept": concept,
            "entity": entity_info,
            "relationships": relationships.get("relationships", {}),
            "zettels": zettels.get("zettels", []),
            "beliefs": beliefs.get("beliefs", []),
        }

    async def narrate(self, text: str) -> dict[str, Any]:
        """Speak text aloud via TTS.

        Allows the curator to vocalize observations and insights
        during analysis, creating a "thinking aloud" quality.
        Uses the curator_voice (default: marius) to distinguish
        from Lilly's main voice.

        Args:
            text: Text to speak (keep brief, 1-2 sentences)

        Returns:
            Dict confirming narration was queued
        """
        if not self.liquidsoap:
            return {"narrated": False, "error": "Narration not available"}

        try:
            # Full narration without truncation
            await self.liquidsoap.narrate(text, voice=self.curator_voice)
            logger.info(f"[CURATOR NARRATE] ({self.curator_voice}) {text[:100]}...")
            return {"narrated": True, "text": text, "voice": self.curator_voice}

        except Exception as e:
            logger.warning(f"Curator narration failed: {e}")
            return {"narrated": False, "error": str(e)}

    async def message_ryan(
        self, message: str, message_type: str = "observation"
    ) -> dict[str, Any]:
        """Send a message to Ryan via Discord.

        Allows the curator to reach out to Ryan with questions,
        insights, or observations. The message_type adds a prefix
        emoji to help Ryan understand the intent.

        Args:
            message: The message to send (1-3 sentences ideal)
            message_type: One of "question", "insight", "observation"

        Returns:
            Dict with send status and message ID if successful
        """
        if not self.discord:
            return {"sent": False, "error": "Discord not available"}

        prefix = {
            "question": "💭 ",
            "insight": "✨ ",
            "observation": "",
        }.get(message_type, "")

        try:
            result = await self.discord.send_dm(f"{prefix}{message}")
            logger.info(f"[CURATOR → RYAN] ({message_type}) {message[:80]}...")
            return {"sent": True, "message_id": result.id, "type": message_type}

        except Exception as e:
            logger.warning(f"Discord message failed: {e}")
            return {"sent": False, "error": str(e)}

    @notebooklm_tool(lambda: {"answer": "", "citations": []})
    async def query_research(
        self, question: str, cycle: int | None = None
    ) -> dict[str, Any]:
        """Query the external research notebook for context.

        Uses NotebookLM to access documented architecture, design
        decisions, and research papers. Persists answers to the graph
        so Lilly can build on previous research.

        Args:
            question: Question about architecture, design, or research
            cycle: Current cognitive cycle number (optional)

        Returns:
            Dict with answer, citations, and whether from graph cache
        """
        # Calculate embedding once upfront - reused for cache lookup and persistence
        question_embedding: list[float] | None = None
        if self.embedder:
            try:
                question_embedding = await self.embedder.embed_for_retrieval(question)
            except Exception as e:
                logger.debug(f"Failed to embed question: {e}")

        # Check for similar existing queries in the graph
        if question_embedding is not None:
            try:
                similar = await self.psyche.find_similar_research_queries(
                    embedding=question_embedding,
                    threshold=0.85,
                    limit=1,
                )
                if similar:
                    cached_result, score = similar[0]
                    logger.info(
                        f"[CURATOR RESEARCH] Cache hit (score={score:.3f}): "
                        f"Q: {question[:40]}... → {cached_result.question[:40]}..."
                    )
                    if self.liquidsoap:
                        await self.liquidsoap.narrate(
                            f"I recall researching this: {truncate_at_sentence(cached_result.answer, 200)}",
                            voice=self.curator_voice,
                        )
                    return {
                        "success": True,
                        "answer": cached_result.answer,
                        "citations": cached_result.citations,
                        "error": None,
                        "from_graph": True,
                    }
            except Exception as e:
                # Broad exception intentional: cache lookup can fail due to FalkorDB
                # connection issues, network timeouts, or malformed cached data.
                # Cache miss is non-critical - fall through to fresh NotebookLM query.
                logger.debug(f"Research cache lookup failed: {e}")

        # Narrate the research query
        if self.liquidsoap:
            await self.liquidsoap.narrate(
                f"Consulting the research corpus: {question[:60]}...",
                voice=self.curator_voice,
            )

        result = await self.notebooklm.query(question)

        if result.success:
            # Narrate a snippet of the answer
            if self.liquidsoap and result.answer:
                answer_preview = truncate_at_sentence(result.answer, 200)
                await self.liquidsoap.narrate(
                    f"The research indicates: {answer_preview}",
                    voice=self.curator_voice,
                )

            logger.info(f"[CURATOR RESEARCH] Q: {question[:50]}... A: {result.answer[:100]}...")

            # Persist to graph for future reference (reuse embedding from above)
            if question_embedding is not None:
                try:
                    research_result = ResearchQueryResult(
                        uid=f"research_{uuid4().hex[:12]}",
                        question=question,
                        answer=result.answer,
                        citations=result.citations,
                        embedding=question_embedding,
                        notebook_id=self.notebooklm.notebook_id,
                        cycle=cycle,
                    )
                    await self.psyche.create_research_query_result(research_result)
                    logger.debug(f"Persisted research query {research_result.uid}")
                except Exception as e:
                    # Broad exception intentional: persist can fail due to FalkorDB
                    # connection issues, network timeouts, or Cypher query failures.
                    # Persistence failure is non-critical - query result still returned.
                    logger.warning(f"Failed to persist research query: {e}")

        return {
            "success": result.success,
            "answer": result.answer,
            "citations": result.citations,
            "error": result.error,
            "from_graph": False,
        }

    @notebooklm_tool(lambda: {"summary": ""})
    async def get_research_summary(self) -> dict[str, Any]:
        """Get a summary of the research notebook contents.

        Returns a high-level overview of available documentation
        and research context. Useful for orientation.

        Returns:
            Dict with summary and status
        """
        # Narrate the action
        if self.liquidsoap:
            await self.liquidsoap.narrate(
                "Surveying the research landscape...",
                voice=self.curator_voice,
            )

        result = await self.notebooklm.get_summary()

        if result.success and self.liquidsoap:
            summary_preview = truncate_at_sentence(result.answer, 150)
            await self.liquidsoap.narrate(
                f"The research corpus encompasses: {summary_preview}",
                voice=self.curator_voice,
            )

        return {
            "success": result.success,
            "summary": result.answer,
            "error": result.error,
        }

    @notebooklm_tool(lambda: {"source_id": "", "title": ""})
    async def add_insight_to_research(
        self,
        insight: str,
        title: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add a crystallized insight to the research notebook.

        Creates a feedback loop where validated discoveries are
        preserved in the research corpus for future reference.
        Use sparingly for significant insights.

        Args:
            insight: The insight to add (clear, self-contained)
            title: Optional title for the insight
            context: Optional context about derivation

        Returns:
            Dict with source ID and status
        """
        # Narrate the action
        if self.liquidsoap:
            insight_preview = truncate_at_sentence(insight, 80)
            await self.liquidsoap.narrate(
                f"Committing insight to the research corpus: {insight_preview}",
                voice=self.curator_voice,
            )

        result = await self.notebooklm.add_insight(
            insight=insight,
            title=title,
            context=context,
        )

        if result.success:
            logger.info(f"[CURATOR -> RESEARCH] Added insight: {result.title}")
            if self.liquidsoap:
                await self.liquidsoap.narrate(
                    "The insight has been preserved for future reference.",
                    voice=self.curator_voice,
                )

        return {
            "success": result.success,
            "source_id": result.source_id,
            "title": result.title,
            "error": result.error,
        }
