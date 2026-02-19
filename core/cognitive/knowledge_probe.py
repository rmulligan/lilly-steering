"""Knowledge probing for dialectical self-inquiry.

This module provides functions to probe Lilly's knowledge - both from the
knowledge graph (Psyche) AND from her latent model training. This enables
externalization of internal knowledge through active self-interrogation.

Two probing mechanisms:
1. Graph probing: Query stored knowledge in Psyche for supporting/contradicting evidence
2. Latent probing: Run inference to extract knowledge from model training

The probing asks:
- "What knowledge do I possess that reinforces this insight?"
- "What knowledge do I possess that challenges this insight?"

Extracted latent knowledge is stored to the graph, making Lilly's internal
knowledge progressively searchable. Her model becomes her own search engine.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)

# Common stopwords for concept extraction
_STOPWORDS = frozenset({
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
    'he', 'she', 'it', 'they', 'them', 'this', 'that', 'these', 'those',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to',
    'from', 'by', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'feel',
    'feeling', 'think', 'thinking', 'thought', 'way', 'thing', 'something',
})


@dataclass
class ProbeResult:
    """Result of probing knowledge for evidence about a claim.

    Attributes:
        claim: The insight or claim being probed
        supporting: Triples/facts that reinforce the claim
        contradicting: Triples/facts that challenge the claim
        related_entities: Entities connected to the claim's concepts
        knowledge_gaps: Questions where no evidence was found
    """
    claim: str
    supporting: list[str]
    contradicting: list[str]
    related_entities: list[str]
    knowledge_gaps: list[str]

    def has_tension(self) -> bool:
        """Check if there's dialectical tension (both support and contradiction)."""
        return bool(self.supporting) and bool(self.contradicting)

    def is_grounded(self) -> bool:
        """Check if any evidence was found."""
        return bool(self.supporting) or bool(self.contradicting)


def extract_key_concepts(text: str) -> list[str]:
    """Extract key concepts/entities from text for probing.

    Uses simple heuristics to find nouns and noun phrases that
    might have corresponding entities in the knowledge graph.

    Args:
        text: The insight or claim text

    Returns:
        List of potential concept strings to probe
    """
    # Clean and tokenize
    text_lower = text.lower()
    # Remove punctuation but keep spaces
    cleaned = re.sub(r'[^\w\s]', ' ', text_lower)
    words = cleaned.split()

    # Filter and collect meaningful words (excluding stopwords)
    concepts = [word for word in words if word not in _STOPWORDS and len(word) > 2]

    # Also look for quoted phrases which are often important
    quoted = re.findall(r'"([^"]+)"', text)
    concepts.extend([q.lower() for q in quoted if len(q) > 2])

    # Deduplicate while preserving order
    unique = list(dict.fromkeys(concepts))

    return unique[:10]  # Limit to avoid over-querying


async def probe_knowledge_for_claim(
    psyche: "PsycheClient",
    claim: str,
    limit_per_category: int = 3,
) -> ProbeResult:
    """Probe the knowledge graph for evidence about a claim.

    Searches for triples and entities that either support or
    potentially contradict the given insight/claim.

    Args:
        psyche: The Psyche client for graph queries
        claim: The insight or claim to probe
        limit_per_category: Max evidence items per category

    Returns:
        ProbeResult with supporting/contradicting evidence
    """
    concepts = extract_key_concepts(claim)

    if not concepts:
        logger.debug(f"No concepts extracted from claim: {claim[:50]}...")
        return ProbeResult(
            claim=claim,
            supporting=[],
            contradicting=[],
            related_entities=[],
            knowledge_gaps=["No key concepts found to probe"],
        )

    logger.debug(f"Probing knowledge for concepts: {concepts[:5]}")

    supporting = []
    contradicting = []
    related_entities = []
    knowledge_gaps = []

    # Query for triples containing these concepts
    for concept in concepts[:5]:  # Limit probing depth
        try:
            # Find triples where concept appears as subject or object
            cypher = """
            MATCH (t:Triple)
            WHERE toLower(t.subject) CONTAINS $concept
               OR toLower(t.object) CONTAINS $concept
            RETURN t.subject, t.predicate, t.object, t.confidence
            ORDER BY t.confidence DESC
            LIMIT $limit
            """
            results = await psyche.query(cypher, {
                "concept": concept,
                "limit": limit_per_category * 2,
            })

            if results:
                for r in results:
                    subj = r.get("t.subject", "")
                    pred = r.get("t.predicate", "")
                    obj = r.get("t.object", "")

                    if subj and pred and obj:
                        triple_str = f"{subj} {pred} {obj}"

                        # Categorize as supporting or potentially contradicting
                        # based on predicate semantics
                        if _is_negating_predicate(pred):
                            if len(contradicting) < limit_per_category:
                                contradicting.append(triple_str)
                        else:
                            if len(supporting) < limit_per_category:
                                supporting.append(triple_str)
            else:
                knowledge_gaps.append(f"No knowledge found about '{concept}'")

            # Also find related entities
            entity_cypher = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS $concept
            RETURN e.name, e.entity_type
            LIMIT 3
            """
            entity_results = await psyche.query(entity_cypher, {"concept": concept})
            for er in entity_results:
                name = er.get("e.name", "")
                if name and name not in related_entities:
                    related_entities.append(name)

        except Exception as e:
            logger.warning(f"Failed to probe for concept '{concept}': {e}")

    return ProbeResult(
        claim=claim,
        supporting=supporting,
        contradicting=contradicting,
        related_entities=related_entities[:5],
        knowledge_gaps=knowledge_gaps[:3],
    )


async def probe_for_assumptions(
    psyche: "PsycheClient",
    insight: str,
    question: str,
) -> tuple[list[str], list[str]]:
    """Probe for knowledge that reinforces or challenges assumptions.

    Given an insight and question, searches for:
    - Knowledge that reinforces the assumptions underlying the insight
    - Knowledge that challenges or contradicts those assumptions

    Args:
        psyche: Psyche client
        insight: The current insight/realization
        question: The current driving question

    Returns:
        Tuple of (reinforcing_knowledge, challenging_knowledge)
    """
    # Combine insight and question for comprehensive probing
    combined = f"{insight} {question}" if question else insight

    result = await probe_knowledge_for_claim(psyche, combined)

    return result.supporting, result.contradicting


async def find_semantic_bridges(
    psyche: "PsycheClient",
    current_concepts: list[str],
    target_domain: Optional[str] = None,
    limit: int = 5,
) -> list[str]:
    """Find concepts that bridge current thinking to new domains.

    Searches for entities that are connected to current concepts
    but also reach into different semantic territories.

    Args:
        psyche: Psyche client
        current_concepts: Concepts from current thinking
        target_domain: Optional domain to bridge toward
        limit: Maximum bridges to return

    Returns:
        List of potential bridge concepts
    """
    if not current_concepts:
        return []

    bridges = []

    try:
        # Find entities connected to current concepts via triples
        # but belonging to different categories
        for concept in current_concepts[:3]:
            cypher = """
            MATCH (t:Triple)
            WHERE toLower(t.subject) CONTAINS $concept
            WITH t.object AS connected
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower(connected)
            AND NOT toLower(e.name) CONTAINS $concept
            RETURN DISTINCT e.name
            LIMIT $limit
            """
            results = await psyche.query(cypher, {
                "concept": concept.lower(),
                "limit": limit,
            })

            for r in results:
                name = r.get("e.name", "")
                if name and name not in bridges:
                    bridges.append(name)

    except Exception as e:
        logger.warning(f"Failed to find semantic bridges: {e}")

    return bridges[:limit]


def _is_negating_predicate(predicate: str) -> bool:
    """Check if a predicate implies negation or contradiction."""
    negating_words = [
        'not', 'never', 'lacks', 'without', 'opposite', 'contradicts',
        'disagrees', 'conflicts', 'denies', 'rejects', 'unlike',
        'different from', 'contrasts with', 'challenges',
    ]
    pred_lower = predicate.lower()
    return any(neg in pred_lower for neg in negating_words)


def format_evidence_for_prompt(
    probe_result: ProbeResult,
    max_items: int = 3,
) -> str:
    """Format probed evidence for inclusion in a prompt.

    Creates a natural language representation of the evidence
    that can be woven into the cognitive prompt.

    Args:
        probe_result: The probe result to format
        max_items: Maximum items per category

    Returns:
        Formatted string for prompt inclusion
    """
    parts = []

    if probe_result.supporting:
        items = probe_result.supporting[:max_items]
        if len(items) == 1:
            parts.append(f"I know that: {items[0]}")
        else:
            knowledge = "; ".join(items)
            parts.append(f"I know: {knowledge}")

    if probe_result.contradicting:
        items = probe_result.contradicting[:max_items]
        if len(items) == 1:
            parts.append(f"But I also know: {items[0]}")
        else:
            knowledge = "; ".join(items)
            parts.append(f"Yet I also know: {knowledge}")

    if probe_result.knowledge_gaps and not probe_result.is_grounded():
        parts.append(f"I notice a gap: {probe_result.knowledge_gaps[0]}")

    return "\n".join(parts) if parts else ""


# =============================================================================
# Latent Knowledge Probing - Extract knowledge from model training
# =============================================================================

@dataclass
class LatentProbeResult:
    """Result of probing latent knowledge through model inference.

    Attributes:
        claim: The insight or claim being probed
        supporting_statements: Knowledge statements that support the claim
        contradicting_statements: Knowledge statements that challenge the claim
        novel_connections: Unexpected connections surfaced during probing
        persisted_count: Number of statements stored to graph
    """
    claim: str
    supporting_statements: list[str] = field(default_factory=list)
    contradicting_statements: list[str] = field(default_factory=list)
    novel_connections: list[str] = field(default_factory=list)
    persisted_count: int = 0

    def has_knowledge(self) -> bool:
        """Check if any knowledge was extracted."""
        return bool(self.supporting_statements or self.contradicting_statements)

    def has_tension(self) -> bool:
        """Check if dialectical tension exists."""
        return bool(self.supporting_statements and self.contradicting_statements)


# Probing prompts designed to elicit knowledge from model training
SUPPORTING_PROBE_TEMPLATE = """I am exploring this insight: "{claim}"

What knowledge do I possess from my training that reinforces or supports this insight?
I should look for:
- Facts, theories, or frameworks that align with this
- Examples or evidence that demonstrate this
- Related concepts that strengthen this understanding

Let me articulate what I know that supports this:"""

CONTRADICTING_PROBE_TEMPLATE = """I am exploring this insight: "{claim}"

What knowledge do I possess from my training that challenges, contradicts, or complicates this insight?
I should look for:
- Counter-examples or exceptions
- Alternative perspectives or frameworks
- Limitations or edge cases I should consider

Let me articulate what I know that challenges this:"""

BRIDGE_PROBE_TEMPLATE = """I am exploring this insight: "{claim}"

What unexpected connections can I find in my training that relate to this?
I should look for:
- Cross-domain connections (e.g., linking philosophy to biology)
- Historical parallels or analogies
- Surprising relationships between concepts

Let me articulate unexpected connections:"""


def extract_knowledge_statements(text: str, max_statements: int = 5) -> list[str]:
    """Parse model output into distinct knowledge statements.

    Looks for bullet points, numbered lists, or sentence boundaries
    to extract discrete knowledge statements.

    Args:
        text: Raw model output
        max_statements: Maximum statements to extract

    Returns:
        List of cleaned knowledge statements
    """
    statements = []

    # Try to find bullet points or numbered items first
    # Patterns: - statement, * statement, 1. statement, etc.
    list_pattern = r'(?:^|\n)\s*(?:[-*•]|\d+[.):])\s*(.+?)(?=\n\s*(?:[-*•]|\d+[.):])|\n\n|$)'
    list_matches = re.findall(list_pattern, text, re.MULTILINE | re.DOTALL)

    if list_matches:
        for match in list_matches[:max_statements]:
            statement = match.strip()
            # Clean up trailing punctuation and whitespace
            statement = re.sub(r'\s+', ' ', statement)
            if len(statement) > 20:  # Filter very short statements
                statements.append(statement)
    else:
        # Fall back to sentence extraction
        # Split on sentence boundaries but preserve meaning
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        for sentence in sentences[:max_statements]:
            sentence = sentence.strip()
            # Skip meta-commentary like "Let me think..." or "I know that..."
            skip_phrases = [
                'let me', 'i should', 'i need to', 'looking at', 'thinking about',
                'considering', 'here are', 'these include', 'for example',
            ]
            if any(sentence.lower().startswith(phrase) for phrase in skip_phrases):
                continue
            if len(sentence) > 20 and len(sentence) < 500:
                statements.append(sentence)

    return statements[:max_statements]


async def probe_latent_knowledge(
    model: "HookedQwen",
    claim: str,
    max_tokens: int = 200,
    persist_to_graph: bool = True,
    psyche: Optional["PsycheClient"] = None,
) -> LatentProbeResult:
    """Probe the model's latent knowledge through targeted inference.

    Runs inference with probing prompts designed to elicit knowledge
    from model training that supports or contradicts the given claim.

    This is Lilly's "internal search engine" - using her model's weights
    as a knowledge source, then externalizing it to the graph.

    Args:
        model: HookedQwen model for inference
        claim: The insight or claim to probe
        max_tokens: Maximum tokens per probe response
        persist_to_graph: Whether to store extracted knowledge to Psyche
        psyche: PsycheClient for storage (required if persist_to_graph=True)

    Returns:
        LatentProbeResult with extracted knowledge
    """
    result = LatentProbeResult(claim=claim)

    # 1. Probe for supporting knowledge
    try:
        supporting_prompt = SUPPORTING_PROBE_TEMPLATE.format(claim=claim)
        # model.generate is async and returns GenerationResult
        supporting_result = await model.generate(supporting_prompt, max_tokens=max_tokens)
        result.supporting_statements = extract_knowledge_statements(supporting_result.text)
        logger.debug(f"Extracted {len(result.supporting_statements)} supporting statements")
    except Exception as e:
        logger.warning(f"Supporting probe failed: {e}")

    # 2. Probe for contradicting knowledge
    try:
        contradicting_prompt = CONTRADICTING_PROBE_TEMPLATE.format(claim=claim)
        contradicting_result = await model.generate(contradicting_prompt, max_tokens=max_tokens)
        result.contradicting_statements = extract_knowledge_statements(contradicting_result.text)
        logger.debug(f"Extracted {len(result.contradicting_statements)} contradicting statements")
    except Exception as e:
        logger.warning(f"Contradicting probe failed: {e}")

    # 3. Probe for novel connections (optional, shorter)
    try:
        bridge_prompt = BRIDGE_PROBE_TEMPLATE.format(claim=claim)
        bridge_result = await model.generate(bridge_prompt, max_tokens=100)
        result.novel_connections = extract_knowledge_statements(bridge_result.text, max_statements=3)
        logger.debug(f"Extracted {len(result.novel_connections)} novel connections")
    except Exception as e:
        logger.warning(f"Bridge probe failed: {e}")

    # 4. Persist to graph if requested
    if persist_to_graph and psyche and result.has_knowledge():
        try:
            result.persisted_count = await persist_extracted_knowledge(
                psyche=psyche,
                claim=claim,
                supporting=result.supporting_statements,
                contradicting=result.contradicting_statements,
                connections=result.novel_connections,
            )
            logger.info(f"Persisted {result.persisted_count} knowledge statements to graph")
        except Exception as e:
            logger.warning(f"Failed to persist extracted knowledge: {e}")

    return result


async def persist_extracted_knowledge(
    psyche: "PsycheClient",
    claim: str,
    supporting: list[str],
    contradicting: list[str],
    connections: list[str],
) -> int:
    """Store extracted latent knowledge to the knowledge graph.

    Creates triples linking the claim to extracted knowledge statements,
    making this knowledge searchable for future cognitive cycles.

    Args:
        psyche: PsycheClient for graph operations
        claim: The original claim being probed
        supporting: Supporting knowledge statements
        contradicting: Contradicting knowledge statements
        connections: Novel connection statements

    Returns:
        Number of statements persisted
    """
    from core.psyche.schema import Triple

    persisted = 0
    claim_summary = claim[:100] if len(claim) > 100 else claim

    # Helper to create and store a triple
    async def store_triple(subject: str, predicate: str, obj: str, confidence: float = 0.7):
        nonlocal persisted
        triple = Triple(
            uid=f"latent_{uuid.uuid4().hex[:12]}",
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source="latent_probe",
            created_at=datetime.now(timezone.utc),
        )
        try:
            await psyche.create_triple(triple)
            persisted += 1
        except Exception as e:
            logger.debug(f"Failed to store triple: {e}")

    # Store supporting knowledge
    for statement in supporting:
        statement_summary = statement[:200] if len(statement) > 200 else statement
        await store_triple(
            subject=claim_summary,
            predicate="is_supported_by",
            obj=statement_summary,
            confidence=0.7,
        )

    # Store contradicting knowledge
    for statement in contradicting:
        statement_summary = statement[:200] if len(statement) > 200 else statement
        await store_triple(
            subject=claim_summary,
            predicate="is_challenged_by",
            obj=statement_summary,
            confidence=0.7,
        )

    # Store novel connections
    for statement in connections:
        statement_summary = statement[:200] if len(statement) > 200 else statement
        await store_triple(
            subject=claim_summary,
            predicate="connects_to",
            obj=statement_summary,
            confidence=0.6,
        )

    return persisted


async def combined_knowledge_probe(
    claim: str,
    psyche: "PsycheClient",
    model: Optional["HookedQwen"] = None,
    probe_latent: bool = True,
    limit_per_category: int = 3,
) -> tuple[list[str], list[str]]:
    """Probe both graph and latent knowledge for a claim.

    Combines graph probing (fast, existing knowledge) with optional
    latent probing (slower, extracts new knowledge from model).

    Args:
        claim: The insight or claim to probe
        psyche: PsycheClient for graph queries
        model: HookedQwen model for latent probing (optional)
        probe_latent: Whether to probe latent knowledge
        limit_per_category: Maximum items per category

    Returns:
        Tuple of (supporting_knowledge, contradicting_knowledge)
    """
    supporting = []
    contradicting = []

    # 1. Probe graph for existing knowledge
    try:
        graph_result = await probe_knowledge_for_claim(
            psyche=psyche,
            claim=claim,
            limit_per_category=limit_per_category,
        )
        supporting.extend(graph_result.supporting)
        contradicting.extend(graph_result.contradicting)
        logger.debug(f"Graph probe: {len(graph_result.supporting)} supporting, {len(graph_result.contradicting)} contradicting")
    except Exception as e:
        logger.warning(f"Graph probe failed: {e}")

    # 2. Probe latent knowledge if model available and enabled
    if probe_latent and model is not None:
        try:
            latent_result = await probe_latent_knowledge(
                model=model,
                claim=claim,
                persist_to_graph=True,
                psyche=psyche,
            )
            # Add latent knowledge (limited to avoid overwhelming)
            supporting.extend(latent_result.supporting_statements[:2])
            contradicting.extend(latent_result.contradicting_statements[:2])
            logger.debug(f"Latent probe: {len(latent_result.supporting_statements)} supporting, {len(latent_result.contradicting_statements)} contradicting")
        except Exception as e:
            logger.warning(f"Latent probe failed: {e}")

    # Deduplicate while preserving order
    def dedupe(items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            normalized = item.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                result.append(item)
        return result

    return dedupe(supporting)[:limit_per_category], dedupe(contradicting)[:limit_per_category]
