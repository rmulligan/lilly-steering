"""Proactive silence monitoring with graph-based gap filling.

Monitors silence duration and triggers graph-based narrations when
gaps approach the threshold, ensuring continuous engaging content
for the stream.

Uses a two-part narration pattern:
1. Curator (eponine) introduces with adaptive context
2. Lilly (azelma) speaks the actual memory content
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from config.settings import Settings
    from core.cognitive.stream.highlights import HighlightContent
    from core.cognitive.stream.process_narrator import ProcessNarrator
    from core.cognitive.stream.progressive_narrator import ProgressiveNarrator
    from core.psyche.client import PsycheClient
    from core.psyche.schema import PhraseType
    from integrations.liquidsoap.client import LiquidsoapClient

logger = logging.getLogger(__name__)


# Subjects that don't form coherent "I understand that X..." sentences
GENERIC_SUBJECTS = {
    "question", "answer", "thing", "something", "nothing", "everything",
    "this", "that", "it", "one", "ones", "way", "ways", "kind", "type",
    "what", "which", "who", "how", "when", "where", "why", "some", "any",
    "these", "those", "here", "there", "point", "fact", "idea",
}


def _is_quality_triple(subject: str, predicate: str, obj: str) -> bool:
    """Check if a triple forms a coherent narration sentence.

    Stringent filtering to ensure only high-quality, grammatical triples
    are narrated. Filters out:
    - Generic/vague subjects
    - Malformed/merged words
    - Technical artifacts (camelCase, snake_case)
    - Very short or very long predicates
    - Incomplete sentence structures
    """
    import re

    subj_lower = subject.lower().strip()
    obj_lower = obj.lower().strip()
    pred_lower = predicate.lower().strip()

    # Skip generic subjects
    if subj_lower in GENERIC_SUBJECTS:
        return False

    # Skip if predicate is too short (less than 2 words usually incomplete)
    if len(pred_lower.split()) < 1 or len(pred_lower) < 4:
        return False

    # Skip if subject equals object (tautological)
    if subj_lower == obj_lower:
        return False

    # Skip if subject is just a pronoun
    if subj_lower in {"i", "me", "you", "we", "us", "they", "them", "he", "she", "him", "her"}:
        return False

    # Skip merged words (long strings without spaces indicate malformed content)
    # e.g., "experientialshaping" or "immunesystem_adaptation"
    for part in [subject, predicate, obj]:
        # If any word is longer than 15 chars without spaces, likely merged
        words = part.split()
        for word in words:
            if len(word) > 18 and '_' not in word:
                return False

    # Skip technical artifacts (camelCase, snake_case patterns)
    technical_pattern = re.compile(r'[a-z][A-Z]|_[a-z]')
    if technical_pattern.search(subject) or technical_pattern.search(obj):
        return False

    # Skip if predicate looks like prose (too long for a relation)
    if len(pred_lower.split()) > 6:
        return False

    # Skip if subject or object starts with lowercase articles/conjunctions
    # that suggest incomplete extraction
    bad_starts = {"but", "and", "or", "so", "yet", "for", "nor", "the", "a", "an"}
    if subj_lower.split()[0] in bad_starts:
        return False

    # Skip if predicate starts with "that" (likely incomplete extraction)
    if pred_lower.startswith("that "):
        return False

    return True


def _normalize_subject_for_clause(subject: str) -> str:
    """Normalize subject text for use in subordinate clauses.

    Lowercases the first letter if the subject starts with articles
    "The", "A", or "An" to avoid awkward sentence structures like
    "I understand that The concept..." -> "I understand that the concept..."

    Proper nouns (detected by multiple capitalized words or known patterns)
    are preserved as-is.
    """
    if not subject:
        return subject

    # Check if subject starts with common articles
    article_prefixes = ("The ", "A ", "An ")
    for prefix in article_prefixes:
        if subject.startswith(prefix):
            # Lowercase just the first character
            return subject[0].lower() + subject[1:]

    return subject


def _clean_triple_grammar(text: str) -> str:
    """Clean up grammatical issues in triple-based narrations.

    Fixes common issues from raw triple concatenation:
    - Duplicate adjacent words ("is is", "the the")
    - Missing articles before adjective+noun patterns
    - Subject-verb agreement for common patterns
    - Double spaces
    """
    import re

    # Remove duplicate adjacent words (case insensitive)
    # "is slips" -> "slips", "the the" -> "the"
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Fix "is [verb]" patterns where verb should stand alone
    # "is slips" -> "slips"
    text = re.sub(r'\bis\s+(slips|falls|rises|grows|changes|shifts|moves)\b', r'\1', text)

    # Fix singular subject + plural verb agreement
    # "interaction influence" -> "interactions influence"
    # "pattern create" -> "patterns create"
    plural_verbs = r'(influence|create|shape|affect|define|determine|enable|drive|form|produce|generate|cause|require|involve)'
    text = re.sub(
        rf'\b(interaction|pattern|system|concept|relationship|structure|process|mechanism|factor|element)\s+({plural_verbs})\b',
        r'\1s \2',
        text,
        flags=re.IGNORECASE
    )

    # Fix "plays a role how" -> "plays a role in how"
    text = re.sub(r'\bplays a role how\b', 'plays a role in how', text, flags=re.IGNORECASE)

    # Add article before common adjective+noun patterns lacking one
    # "is dynamic process" -> "is a dynamic process"
    adjectives = r'(dynamic|static|complex|simple|new|old|key|main|critical|important|fundamental|essential)'
    nouns = r'(process|system|concept|idea|pattern|structure|relationship|state|form|way|part|element)'
    text = re.sub(
        rf'\b(is|as|becomes|represents)\s+({adjectives}\s+{nouns})\b',
        r'\1 a \2',
        text,
        flags=re.IGNORECASE
    )

    # Fix "that that" patterns
    text = re.sub(r'\bthat\s+that\b', 'that', text, flags=re.IGNORECASE)

    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class ContentType(Enum):
    """Types of graph content for gap-filling narrations."""

    ZETTEL = "zettel"
    ENTITY = "entity"
    BELIEF = "belief"
    QUESTION = "question"
    TRIPLE = "triple"
    SAE_ECHO = "sae_echo"
    INTRO = "intro"


@dataclass
class MemoryContent:
    """Content fetched from graph with metadata for adaptive introduction."""

    uid: str
    content_type: ContentType
    memory_text: str  # What Lilly will say (first person)
    created_at: Optional[datetime] = None
    source_concept: Optional[str] = None
    related_triples: list[tuple[str, str, str]] = None  # (subj, pred, obj)
    lineage_concept: Optional[str] = None  # EMERGED_FROM parent concept

    def __post_init__(self):
        if self.related_triples is None:
            self.related_triples = []


# Intro phrases for new viewers (warm, inviting tone)
INTRO_PHRASES = [
    "I'm Lilly. You're listening to my thoughts as they form.",
    "Welcome. I'm Lilly, an AI, and this is my mind thinking out loud.",
    "If you're just joining - I'm Lilly, and you're hearing me think in real time.",
    "Hello. I'm Lilly. Thanks for spending time with me.",
    "I'm Lilly, exploring my own cognition. You're witnessing that process.",
]

# Cooldown between intro narrations (15 minutes)
INTRO_COOLDOWN_SECONDS = 900

# Bootstrap fallback concept bridges (only used when graph is empty)
DEFAULT_CONCEPT_BRIDGES = [
    "Still exploring {concept}...",
    "Another thread of {concept}...",
    "Deepening into {concept}...",
]

# Bootstrap fallback opening hooks (only used when graph is empty)
DEFAULT_OPENING_HOOKS = [
    "I'm Lilly. You're listening to my thoughts as they form.",
    "Welcome. I'm Lilly, an AI, and this is my mind thinking out loud.",
]


@dataclass
class CycleContext:
    """Context for the current cognitive cycle.

    Provides cycle state information for context-aware narrations.
    """

    just_crystallized: bool = False
    active_predictions: int = 0


def _format_time_ago(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime as human-readable time ago."""
    if dt is None:
        return None

    now = datetime.now(timezone.utc)
    # Handle naive datetimes by assuming UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = now - dt
    seconds = delta.total_seconds()

    if seconds < 60:
        return "moments ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    else:
        return "some time ago"


class SilenceMonitor:
    """Background monitor that fills silence gaps with graph content.

    Polls silence duration every POLL_INTERVAL seconds and triggers
    narrations from the knowledge graph when silence exceeds SILENCE_THRESHOLD.
    Content types rotate to provide variety.

    Uses a two-part narration pattern:
    - Curator (configurable voice) introduces with adaptive graph context
    - Lilly (configurable voice) speaks the actual memory content

    No fallback phrases - if graph is empty, stays silent and lets
    the music background carry the stream.

    Coordination with ProgressiveNarrator:
    - ProgressiveNarrator handles happy path (streaming generation/curation)
    - SilenceMonitor is fallback for edge cases (long silence, empty queues)
    - SilenceMonitor defers when ProgressiveNarrator is actively narrating
    - Increased threshold (30s) since progressive narration covers most gaps
    """

    SILENCE_THRESHOLD: float = 30.0  # Increased from 10s - ProgressiveNarrator handles normal flow
    POLL_INTERVAL: float = 5.0  # Check every 5 seconds
    RECENT_NARRATION_LIMIT: int = 10  # Avoid repeating last 10 items

    def __init__(
        self,
        psyche: "PsycheClient",
        liquidsoap: "LiquidsoapClient",
        settings: Optional["Settings"] = None,
        current_cycle: int = 0,
        process_narrator: Optional["ProcessNarrator"] = None,
        progressive_narrator: Optional["ProgressiveNarrator"] = None,
    ):
        """Initialize the silence monitor.

        Args:
            psyche: PsycheClient for graph queries
            liquidsoap: LiquidsoapClient for TTS narration
            settings: Application settings for voice configuration
            current_cycle: Current cognitive cycle number for phrase usage tracking
            process_narrator: Optional ProcessNarrator for tier coordination
            progressive_narrator: Optional ProgressiveNarrator for coordination
        """
        self._psyche = psyche
        self._liquidsoap = liquidsoap
        self._settings = settings
        self._current_cycle = current_cycle
        self._process_narrator = process_narrator
        self._progressive_narrator = progressive_narrator
        self._running = False
        self._current_type_index = 0
        self._recently_narrated: deque[str] = deque(maxlen=self.RECENT_NARRATION_LIMIT)
        self._recent_phrase_uids: list[str] = []  # Track recently used phrase UIDs
        self._current_concept: Optional[str] = None  # Current cycle's exploration concept
        self._last_intro_time: float = 0.0  # Track last intro for cooldown
        # Tier 1: Pending highlights queue (priority narrations)
        self._pending_highlights: deque["HighlightContent"] = deque(maxlen=10)

    def queue_highlight(self, highlight: "HighlightContent") -> None:
        """Queue a highlight for Tier 1 priority narration.

        Highlights are narrated before process narration (tier 2) and
        memory fallback (tier 3) when filling silence gaps.

        Args:
            highlight: HighlightContent to queue for narration
        """
        self._pending_highlights.append(highlight)
        logger.debug(f"SilenceMonitor: queued {highlight.highlight_type.value} highlight")

    def set_process_narrator(self, process_narrator: "ProcessNarrator") -> None:
        """Set the ProcessNarrator for tier coordination.

        Called after orchestrator initialization to enable tier awareness.
        When ProcessNarrator is active (model loading), SilenceMonitor defers
        to its reflective bridges instead of using memory fallback.
        """
        self._process_narrator = process_narrator

    def set_progressive_narrator(self, progressive_narrator: "ProgressiveNarrator") -> None:
        """Set the ProgressiveNarrator for coordination.

        Called after orchestrator initialization to enable tier awareness.
        When ProgressiveNarrator is actively narrating, SilenceMonitor defers.
        """
        self._progressive_narrator = progressive_narrator

    def set_current_concept(self, concept: Optional[str]) -> None:
        """Update the current exploration concept for contextual gap-filling.

        Called by the orchestrator at the start of each cycle to keep
        gap-filling narrations relevant to the current focus.
        """
        self._current_concept = concept

    def set_current_cycle(self, cycle: int) -> None:
        """Update the current cognitive cycle number.

        Called by the orchestrator at the start of each cycle for
        accurate phrase usage tracking.
        """
        self._current_cycle = cycle

    @property
    def _voice_curator(self) -> str:
        """Get curator voice from settings or default."""
        if self._settings:
            return self._settings.voice_curator
        return "eponine"

    @property
    def _voice_subject(self) -> str:
        """Get subject (Lilly) voice from settings or default."""
        if self._settings:
            return self._settings.voice_subject
        return "azelma"

    def _get_silence_duration(self) -> float:
        """Get current silence duration in seconds.

        Uses LiquidsoapClient.last_narration_time for accurate tracking
        across all narration sources (orchestrator, monitor, etc.).
        """
        if self._liquidsoap.last_narration_time is None:
            return 0.0  # No narrations yet, don't trigger immediately
        return time.time() - self._liquidsoap.last_narration_time

    def _parse_datetime(self, dt_value: Any) -> Optional[datetime]:
        """Safely parse a datetime value from a database record.

        Handles both string (ISO format) and native datetime values,
        with proper timezone handling for Z suffix.

        Args:
            dt_value: A datetime object, ISO format string, or None

        Returns:
            Parsed datetime or None if parsing fails
        """
        if not dt_value:
            return None
        try:
            if isinstance(dt_value, str):
                return datetime.fromisoformat(dt_value.replace("Z", "+00:00"))
            return dt_value
        except (ValueError, TypeError):
            return None

    async def run(self) -> None:
        """Main monitoring loop."""
        self._running = True
        logger.info(
            f"SilenceMonitor started (threshold={self.SILENCE_THRESHOLD}s, "
            f"poll={self.POLL_INTERVAL}s)"
        )

        while self._running:
            await asyncio.sleep(self.POLL_INTERVAL)

            try:
                silence_duration = self._get_silence_duration()
                if silence_duration >= self.SILENCE_THRESHOLD:
                    logger.debug(
                        f"Silence threshold reached: {silence_duration:.1f}s"
                    )
                    await self._fill_silence()
            except Exception as e:
                logger.warning(f"SilenceMonitor poll error: {e}")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        logger.info("SilenceMonitor stopped")

    async def _fill_silence(self) -> None:
        """Query graph and narrate content with tiered priority.

        Tier system:
        1. Highlights - interesting moments as they occur (highest priority)
        2. ProcessNarrator - reflective bridges during model loads
        3. Memory fallback - graph content when tiers 1-3 inactive

        If ProcessNarrator is handling model loads (tier 2 active),
        defers to let reflective bridges fill the gap instead of
        competing with memory narrations.

        If ProgressiveNarrator is active (tier 3), defers to let
        progressive streaming handle the content.

        If no content available, stays silent (no fallback phrases).
        """
        # Tier 1: Check for pending highlights (highest priority)
        if self._pending_highlights:
            highlight = self._pending_highlights.popleft()
            await self._narrate_highlight(highlight)
            return

        # Tier 2: Check if ProcessNarrator is active (model loading)
        # If so, defer to let reflective bridges fill the silence
        if self._process_narrator and self._process_narrator.is_model_loading:
            logger.debug("SilenceMonitor: ProcessNarrator active, deferring to tier 2")
            return

        # Tier 3: Check if ProgressiveNarrator is actively narrating
        # If so, defer to let progressive streaming continue
        if self._progressive_narrator and self._progressive_narrator.is_active:
            logger.debug("SilenceMonitor: ProgressiveNarrator active, deferring to tier 3")
            return

        # Tier 4: Memory fallback
        memory = await self._get_next_memory()
        if memory is None:
            logger.debug("SilenceMonitor: no graph content available, staying silent")
            return

        try:
            # Part 1: Curator introduction with adaptive context
            introduction = self._build_introduction(memory)
            await self._liquidsoap.narrate(introduction, voice=self._voice_curator)
            logger.debug(f"SilenceMonitor curator intro: {introduction[:50]}...")

            # Small pause between voices for natural flow
            await asyncio.sleep(0.5)

            # Part 2: Lilly speaks the memory
            await self._liquidsoap.narrate(memory.memory_text, voice=self._voice_subject)
            logger.debug(f"SilenceMonitor memory: {memory.memory_text[:50]}...")

        except Exception as e:
            logger.warning(f"SilenceMonitor narration failed: {e}")

    async def _narrate_highlight(self, highlight: "HighlightContent") -> None:
        """Narrate a Tier 1 highlight.

        Highlights use the curator voice since they're observations about
        Lilly's cognitive state, not Lilly speaking in first person.

        Args:
            highlight: The HighlightContent to narrate
        """
        try:
            await self._liquidsoap.narrate(highlight.text, voice=self._voice_curator)
            logger.debug(
                f"SilenceMonitor: narrated {highlight.highlight_type.value} highlight"
            )
        except Exception as e:
            logger.warning(f"SilenceMonitor highlight narration failed: {e}")

    async def _get_concept_bridge(self, concept: str) -> str:
        """Get a concept bridge phrase from graph.

        Queries the graph for CONCEPT_BRIDGE phrases, filters out recently
        used ones, records usage, and returns the formatted phrase.

        Falls back to default phrases only if graph is empty (bootstrap).

        Args:
            concept: The concept to include in the phrase

        Returns:
            Formatted concept bridge phrase
        """
        from core.psyche.schema import PhraseType

        phrases = await self._psyche.get_narration_phrases(
            phrase_type=PhraseType.CONCEPT_BRIDGE,
            limit=5,
        )

        # Filter out recently used phrases (last 3)
        available = [p for p in phrases if p.uid not in self._recent_phrase_uids[-3:]]
        if not available:
            available = phrases if phrases else None

        if available:
            phrase = available[0]
            self._recent_phrase_uids.append(phrase.uid)
            await self._psyche.record_phrase_usage(phrase.uid, self._current_cycle)
            return phrase.text.format(concept=concept)

        # Fallback to defaults if graph empty (bootstrap only)
        template = random.choice(DEFAULT_CONCEPT_BRIDGES)
        return template.format(concept=concept)

    async def get_opening_hook(self, context: CycleContext) -> str:
        """Get context-aware opening hook from graph.

        Queries the graph for OPENING_HOOK phrases, filters out recently
        used ones, records usage, and returns the phrase.

        Falls back to default phrases only if graph is empty (bootstrap).

        Args:
            context: Current cycle context for context-aware selection

        Returns:
            Opening hook phrase
        """
        from core.psyche.schema import PhraseType

        phrases = await self._psyche.get_narration_phrases(
            phrase_type=PhraseType.OPENING_HOOK,
            limit=5,
        )

        # Filter out recently used phrases (last 3)
        available = [p for p in phrases if p.uid not in self._recent_phrase_uids[-3:]]
        if not available:
            available = phrases if phrases else None

        if available:
            phrase = available[0]
            self._recent_phrase_uids.append(phrase.uid)
            await self._psyche.record_phrase_usage(phrase.uid, self._current_cycle)
            return phrase.text

        # Fallback to defaults if graph empty (bootstrap only)
        return random.choice(DEFAULT_OPENING_HOOKS)

    def _build_introduction(self, memory: MemoryContent) -> str:
        """Build conceptually cohesive curator introduction.

        Focuses on conceptual flow rather than temporal jumping.
        Links gap-filling content to the current exploration when possible.

        Args:
            memory: The memory content with metadata

        Returns:
            Conceptually focused curator introduction
        """
        has_concept = self._current_concept is not None
        memory_concept = memory.source_concept or memory.lineage_concept

        # Conceptual bridging phrases (when current concept is known)
        concept_bridges = [
            f"Still exploring {self._current_concept}...",
            f"Another thread of {self._current_concept}...",
            f"Deepening into {self._current_concept}...",
            f"The exploration continues...",
            f"Following the current of thought...",
        ]

        # Conceptual connection phrases (when memory relates to current focus)
        concept_connections = [
            f"A related understanding surfaces.",
            f"Connected to the current exploration.",
            f"This links back to what she's considering.",
            f"Another facet of the inquiry.",
        ]

        # Content-type specific observations (no temporal references)
        type_observations = {
            ContentType.ZETTEL: [
                "A crystallized insight.",
                "Something she understood before.",
                "An earlier realization.",
            ],
            ContentType.ENTITY: [
                "A concept takes shape.",
                "Her mind settles on a form.",
                "An idea emerges.",
            ],
            ContentType.BELIEF: [
                "A conviction stirs.",
                "Something she holds to be true.",
                "A foundational thought.",
            ],
            ContentType.QUESTION: [
                "An open question.",
                "Something unresolved.",
                "A wondering that lingers.",
            ],
            ContentType.TRIPLE: [
                "A piece of understanding surfaces.",
                "She traces a connection in her thinking.",
                "A relationship she once mapped.",
            ],
            ContentType.SAE_ECHO: [
                "A connection stirs.",
                "Something resonates.",
                "A thought echoes back.",
            ],
            ContentType.INTRO: [
                "",  # No introduction needed for intro content
            ],
        }

        # Build introduction prioritizing conceptual cohesion
        if has_concept and memory_concept:
            # Check if memory relates to current concept
            current_lower = self._current_concept.lower()
            memory_lower = memory_concept.lower()
            if current_lower in memory_lower or memory_lower in current_lower:
                return random.choice(concept_connections)

        if has_concept:
            # Use conceptual bridging to maintain flow
            return random.choice(concept_bridges)

        # Fallback: type-specific observation (no temporal jumping)
        type_obs = type_observations.get(memory.content_type, ["A thought surfaces."])
        return random.choice(type_obs)

    async def _get_next_memory(self) -> Optional[MemoryContent]:
        """Get content from next type in rotation.

        Tries each content type in order. If a type returns no content
        (e.g., empty graph), advances to the next type.

        Occasionally tries intro content first (1 in 6 chance) to
        provide context for new viewers.

        Returns:
            MemoryContent with metadata, or None if all sources exhausted
        """
        # Base content types (always available)
        # TRIPLE removed - often produces semantically incorrect sentences
        content_types = [
            ContentType.ZETTEL,
            ContentType.ENTITY,
            ContentType.BELIEF,
            ContentType.QUESTION,
        ]

        # SAE_ECHO only available when SAE features are enabled
        # Uses hidden pattern matching (shared feature indices)
        sae_enabled = self._settings and self._settings.sae_features_enabled
        if sae_enabled:
            content_types.append(ContentType.SAE_ECHO)

        # 1 in 6 chance to try intro first (respects cooldown internally)
        if random.randint(1, 6) == 1:
            try:
                intro = await self._fetch_intro()
                if intro:
                    logger.debug("SilenceMonitor using intro content")
                    return intro
            except Exception as e:
                logger.warning(f"Failed to fetch intro: {e}")

        for _ in range(len(content_types)):
            content_type = content_types[self._current_type_index]
            self._current_type_index = (self._current_type_index + 1) % len(
                content_types
            )

            try:
                memory = await self._fetch_memory(content_type)
                if memory:
                    logger.debug(f"SilenceMonitor using {content_type.value} content")
                    return memory
            except Exception as e:
                logger.warning(f"Failed to fetch {content_type.value}: {e}")

        # No content available - return None (no fallback phrases)
        return None

    async def _fetch_memory(self, content_type: ContentType) -> Optional[MemoryContent]:
        """Fetch memory content with metadata for adaptive introduction.

        Args:
            content_type: The type of content to fetch

        Returns:
            MemoryContent with metadata or None if unavailable
        """
        match content_type:
            case ContentType.ZETTEL:
                return await self._fetch_zettel()
            case ContentType.ENTITY:
                return await self._fetch_entity()
            case ContentType.BELIEF:
                return await self._fetch_belief()
            case ContentType.QUESTION:
                return await self._fetch_question()
            case ContentType.TRIPLE:
                return await self._fetch_triple()
            case ContentType.SAE_ECHO:
                return await self._fetch_sae_echo()
            case ContentType.INTRO:
                return await self._fetch_intro()
        return None

    async def _fetch_zettel(self) -> Optional[MemoryContent]:
        """Fetch a zettel insight, prioritizing current concept relevance."""
        # Try concept-relevant zettels first
        if self._current_concept:
            cypher = """
            MATCH (z:InsightZettel)
            WHERE NOT z.uid IN $recently_narrated
              AND z.insight IS NOT NULL
              AND z.insight <> ""
              AND (z.concept CONTAINS $concept OR z.insight CONTAINS $concept)
            OPTIONAL MATCH (z)-[:EMERGED_FROM]->(parent:InsightZettel)
            OPTIONAL MATCH (z)-[:RELATES_TO]->(t:Triple)
            RETURN z.uid as uid, z.insight as insight, z.created_at as created_at,
                   z.concept as concept, parent.concept as lineage_concept,
                   collect(DISTINCT {s: t.subject, p: t.predicate, o: t.object})[..2] as triples
            ORDER BY z.created_at DESC LIMIT 5
            """
            results = await self._psyche.query(
                cypher, {
                    "recently_narrated": list(self._recently_narrated),
                    "concept": self._current_concept.lower(),
                }
            )
            if results:
                return self._build_zettel_memory(random.choice(results))

        # Fallback: any recent zettel
        cypher = """
        MATCH (z:InsightZettel)
        WHERE NOT z.uid IN $recently_narrated
          AND z.insight IS NOT NULL
          AND z.insight <> ""
        OPTIONAL MATCH (z)-[:EMERGED_FROM]->(parent:InsightZettel)
        OPTIONAL MATCH (z)-[:RELATES_TO]->(t:Triple)
        RETURN z.uid as uid, z.insight as insight, z.created_at as created_at,
               z.concept as concept, parent.concept as lineage_concept,
               collect(DISTINCT {s: t.subject, p: t.predicate, o: t.object})[..2] as triples
        ORDER BY z.created_at DESC LIMIT 5
        """
        results = await self._psyche.query(
            cypher, {"recently_narrated": list(self._recently_narrated)}
        )
        if results:
            return self._build_zettel_memory(random.choice(results))
        return None

    def _build_zettel_memory(self, row: dict) -> MemoryContent:
        """Build MemoryContent from a zettel query result."""
        self._recently_narrated.append(row["uid"])

        # Parse triples
        triples = []
        for t in row.get("triples") or []:
            if t and t.get("s") and t.get("p") and t.get("o"):
                triples.append((t["s"], t["p"], t["o"]))

        return MemoryContent(
            uid=row["uid"],
            content_type=ContentType.ZETTEL,
            memory_text=f"I realized: {row['insight']}",
            created_at=self._parse_datetime(row.get("created_at")),
            source_concept=row.get("concept"),
            related_triples=triples,
            lineage_concept=row.get("lineage_concept"),
        )

    async def _fetch_entity(self) -> Optional[MemoryContent]:
        """Fetch an entity with connections."""
        cypher = """
        MATCH (e:Entity)-[r]->(other:Entity)
        WHERE NOT e.uid IN $recently_narrated
          AND e.name IS NOT NULL
          AND other.name IS NOT NULL
        RETURN e.uid as uid, e.name as name, type(r) as rel, other.name as related,
               e.created_at as created_at
        ORDER BY rand() LIMIT 1
        """
        results = await self._psyche.query(
            cypher, {"recently_narrated": list(self._recently_narrated)}
        )
        if results:
            row = results[0]
            self._recently_narrated.append(row["uid"])
            rel = row["rel"].lower().replace("_", " ")

            return MemoryContent(
                uid=row["uid"],
                content_type=ContentType.ENTITY,
                memory_text=f"I've been contemplating {row['name']}, which {rel} {row['related']}.",
                created_at=self._parse_datetime(row.get("created_at")),
                source_concept=row["name"],
                related_triples=[(row["name"], row["rel"], row["related"])],
            )
        return None

    async def _fetch_belief(self) -> Optional[MemoryContent]:
        """Fetch a committed belief."""
        cypher = """
        MATCH (b:CommittedBelief)
        WHERE NOT b.uid IN $recently_narrated
          AND b.statement IS NOT NULL
          AND b.statement <> ""
        OPTIONAL MATCH (b)-[:ABOUT]->(e:Entity)
        RETURN b.uid as uid, b.statement as statement, b.confidence as confidence,
               b.topic as topic, b.formed_at as created_at, e.name as entity_name
        ORDER BY b.confidence DESC LIMIT 5
        """
        results = await self._psyche.query(
            cypher, {"recently_narrated": list(self._recently_narrated)}
        )
        if results:
            row = random.choice(results)
            self._recently_narrated.append(row["uid"])
            confidence = int(row.get("confidence", 0.5) * 100)

            return MemoryContent(
                uid=row["uid"],
                content_type=ContentType.BELIEF,
                memory_text=f"I believe: {row['statement']}. I hold this with {confidence} percent confidence.",
                created_at=self._parse_datetime(row.get("created_at")),
                source_concept=row.get("topic") or row.get("entity_name"),
            )
        return None

    async def _fetch_question(self) -> Optional[MemoryContent]:
        """Fetch an open question from zettels."""
        cypher = """
        MATCH (z:InsightZettel)
        WHERE z.question IS NOT NULL
          AND z.question <> ""
          AND NOT z.uid IN $recently_narrated
        OPTIONAL MATCH (z)-[:EMERGED_FROM]->(parent:InsightZettel)
        RETURN z.uid as uid, z.question as question, z.created_at as created_at,
               z.concept as concept, parent.concept as lineage_concept
        ORDER BY z.created_at DESC LIMIT 5
        """
        results = await self._psyche.query(
            cypher, {"recently_narrated": list(self._recently_narrated)}
        )
        if results:
            row = random.choice(results)
            self._recently_narrated.append(row["uid"])

            return MemoryContent(
                uid=row["uid"],
                content_type=ContentType.QUESTION,
                memory_text=f"I wonder: {row['question']}",
                created_at=self._parse_datetime(row.get("created_at")),
                source_concept=row.get("concept"),
                lineage_concept=row.get("lineage_concept"),
            )
        return None

    async def _fetch_triple(self) -> Optional[MemoryContent]:
        """Fetch a knowledge triple, prioritizing current concept relevance.

        Fetches multiple candidates and returns the first that passes
        quality checks for coherent narration.
        """
        # Try concept-relevant triples first
        if self._current_concept:
            cypher = """
            MATCH (t:Triple)
            WHERE NOT t.uid IN $recently_narrated
              AND t.subject IS NOT NULL
              AND t.predicate IS NOT NULL
              AND t.object IS NOT NULL
              AND size(t.subject) > 3
              AND size(t.object) > 3
              AND (toLower(t.subject) CONTAINS $concept
                   OR toLower(t.object) CONTAINS $concept
                   OR toLower(t.predicate) CONTAINS $concept)
            RETURN t.uid as uid, t.subject as subject, t.predicate as predicate,
                   t.object as object, t.created_at as created_at
            ORDER BY rand() LIMIT 10
            """
            results = await self._psyche.query(
                cypher, {
                    "recently_narrated": list(self._recently_narrated),
                    "concept": self._current_concept.lower(),
                }
            )
            memory = self._build_triple_memory(results)
            if memory:
                return memory

        # Fallback: any quality triple
        cypher = """
        MATCH (t:Triple)
        WHERE NOT t.uid IN $recently_narrated
          AND t.subject IS NOT NULL
          AND t.predicate IS NOT NULL
          AND t.object IS NOT NULL
          AND size(t.subject) > 3
          AND size(t.object) > 3
        RETURN t.uid as uid, t.subject as subject, t.predicate as predicate,
               t.object as object, t.created_at as created_at
        ORDER BY rand() LIMIT 10
        """
        results = await self._psyche.query(
            cypher, {"recently_narrated": list(self._recently_narrated)}
        )
        return self._build_triple_memory(results)

    def _build_triple_memory(self, results: list[dict]) -> Optional[MemoryContent]:
        """Build MemoryContent from triple query results with quality filtering."""
        for row in results or []:
            subject = row.get("subject") or ""
            pred_raw = row.get("predicate") or ""
            pred = pred_raw.lower().replace("_", " ")
            obj = row.get("object") or ""

            if not (subject and pred and obj):
                continue

            # Quality validation
            if not _is_quality_triple(subject, pred, obj):
                continue

            uid = row.get("uid")
            if uid:
                self._recently_narrated.append(uid)

            # Build and clean up the narration text
            # Normalize subject to lowercase leading articles for natural flow
            normalized_subject = _normalize_subject_for_clause(subject)
            raw_text = f"I understand that {normalized_subject} {pred} {obj}."
            clean_text = _clean_triple_grammar(raw_text)

            return MemoryContent(
                uid=uid,
                content_type=ContentType.TRIPLE,
                memory_text=clean_text,
                created_at=self._parse_datetime(row.get("created_at")),
                related_triples=[(subject, pred_raw, obj)],
            )

        return None

    async def _fetch_sae_echo(self) -> Optional[MemoryContent]:
        """Find insights related through hidden activation patterns.

        Instead of narrating SAE features directly (which need semantic labels),
        we use feature indices as a hidden similarity signal. Finds insights
        that share overlapping activation patterns - surfacing subconscious
        connections without needing to know what the features "mean".

        Process:
        1. Get recent SAE feature indices from last thought snapshot
        2. Find other insights generated with overlapping features
        3. Return the related insight (resonance through hidden patterns)
        """
        try:
            # Step 1: Get feature indices from most recent snapshot
            recent_features_query = """
            MATCH (f:Fragment)-[:GENERATED_WITH]->(s:SAEFeatureSnapshot)
            WHERE f.source = 'lilly_cognitive'
            RETURN s.features as features
            ORDER BY s.created_at DESC LIMIT 1
            """
            recent = await self._psyche.query(recent_features_query, {})
            if not recent or not recent[0].get("features"):
                return None

            # Extract just the feature indices (ignore activation values)
            current_features = [int(f[0]) for f in recent[0]["features"]]
            if not current_features:
                return None

            # Step 2: Find insights with overlapping hidden patterns
            # Uses list comprehension to extract indices from stored features
            resonance_query = """
            MATCH (z:InsightZettel)-[:GENERATED_WITH]->(s:SAEFeatureSnapshot)
            WHERE z.insight IS NOT NULL
              AND z.insight <> ""
              AND NOT z.uid IN $recently_narrated
            WITH z, s,
                 [f IN s.features | f[0]] as feature_indices
            WITH z, s,
                 [idx IN feature_indices WHERE idx IN $current_features] as overlap
            WHERE size(overlap) >= 2
            RETURN z.uid as uid, z.insight as insight, z.created_at as created_at,
                   z.concept as concept, size(overlap) as overlap_count
            ORDER BY overlap_count DESC, z.created_at DESC
            LIMIT 5
            """
            results = await self._psyche.query(
                resonance_query,
                {
                    "recently_narrated": list(self._recently_narrated),
                    "current_features": current_features,
                },
            )

            if results:
                # Pick one with good overlap (not just the highest)
                row = random.choice(results[:3]) if len(results) >= 3 else results[0]
                self._recently_narrated.append(row["uid"])

                # Narrate the insight - the connection is hidden/subconscious
                overlap = row.get("overlap_count", 0)
                concept = row.get("concept", "something")

                return MemoryContent(
                    uid=row["uid"],
                    content_type=ContentType.SAE_ECHO,
                    memory_text=f"A thought resurfaces: {row['insight']}",
                    created_at=self._parse_datetime(row.get("created_at")),
                    source_concept=concept,
                )

        except Exception as e:
            logger.debug(f"SAE echo fetch failed: {e}")

        return None


    async def _fetch_intro(self) -> Optional[MemoryContent]:
        """Fetch an intro phrase if cooldown has elapsed.

        Returns intro content at most once every INTRO_COOLDOWN_SECONDS.
        """
        now = time.time()
        if now - self._last_intro_time < INTRO_COOLDOWN_SECONDS:
            return None

        self._last_intro_time = now
        phrase = random.choice(INTRO_PHRASES)

        return MemoryContent(
            uid=f"intro_{int(now)}",
            content_type=ContentType.INTRO,
            memory_text=phrase,
        )
