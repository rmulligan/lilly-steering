"""
Belief Store: Lilly's Committed Positions with Dialectical History.

This module stores Lilly's beliefs—not just knowledge, but defended positions
that she has committed to through dialectical synthesis.

Epistemic Framework (Wolpert 2025):
    Following Wolpert's "Disentangling Boltzmann Brains" paper, beliefs now
    track their conditioning sets—what data was assumed reliable when forming
    the belief. This addresses the fundamental circularity: we need reliable
    data to establish physical laws, but data reliability requires those laws.

    Key insight: Different conditioning choices yield different valid conclusions.
    The Past Hypothesis and Boltzmann brain hypothesis are structurally identical;
    they differ only in what time's entropy they condition upon.

    By tracking conditioning_set explicitly, Lilly can:
    - Recognize when beliefs depend on untested assumptions
    - Identify circular dependencies in her reasoning
    - Maintain multiple belief threads with different conditioning sets

Each belief has:
- A thesis (initial position)
- An antithesis (counter-arguments considered)
- A synthesis (resolved position = the belief statement)
- Supporting evidence and unresolved challenges
- Goal alignment scores
- Conditioning set (epistemic provenance)

Beliefs form a web of SUPPORTS, CONTRADICTS, and REFINES relationships.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.self_model.goal_registry import GoalRegistry

logger = logging.getLogger(__name__)

# Query limits to prevent memory exhaustion (issue #014)
DEFAULT_BELIEF_LIMIT = 10_000
DEFAULT_RELATION_LIMIT = 50_000


def _serialize_timestamp(ts: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO format string, handling None."""
    return ts.isoformat() if ts else None


def _parse_timestamp(
    data: dict,
    key: str,
    default: datetime | None = None,
) -> datetime | None:
    """
    Parse a timestamp from a dictionary with safe fallback.

    Args:
        data: Dictionary containing the timestamp value
        key: Key to look up in the dictionary
        default: Default value if parsing fails (None or datetime.now(timezone.utc))

    Returns:
        Parsed datetime or the default value
    """
    if not data.get(key):
        return default

    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError):
        logger.warning(f"Invalid {key} format: {data.get(key)}")
        return default


class BeliefRelationType(Enum):
    """Types of relationships between beliefs."""
    SUPPORTS = "supports"         # This belief reinforces another
    CONTRADICTS = "contradicts"   # This belief is in tension with another
    REFINES = "refines"           # This belief is a more nuanced version


class BeliefConfidence(Enum):
    """Confidence levels for beliefs."""
    TENTATIVE = "tentative"       # < 0.5 - Held loosely, open to revision
    MODERATE = "moderate"         # 0.5 - 0.75 - Reasonably confident
    STRONG = "strong"             # 0.75 - 0.9 - High confidence
    CORE = "core"                 # > 0.9 - Central to identity


@dataclass
class DialecticalHistory:
    """
    The dialectical process that led to a belief.

    This records the thesis-antithesis-synthesis process,
    forcing Lilly to commit through structured opposition.
    """
    thesis: str                   # Initial position taken
    thesis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    antithesis: str = ""          # Counter-arguments considered
    antithesis_timestamp: Optional[datetime] = None
    synthesis: str = ""           # Resolved position
    synthesis_timestamp: Optional[datetime] = None
    synthesis_reasoning: str = "" # How the tension was resolved

    def is_complete(self) -> bool:
        """Check if the dialectical process is complete."""
        return bool(self.thesis and self.antithesis and self.synthesis)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "thesis": self.thesis,
            "thesis_timestamp": self.thesis_timestamp.isoformat(),
            "antithesis": self.antithesis,
            "antithesis_timestamp": _serialize_timestamp(self.antithesis_timestamp),
            "synthesis": self.synthesis,
            "synthesis_timestamp": _serialize_timestamp(self.synthesis_timestamp),
            "synthesis_reasoning": self.synthesis_reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DialecticalHistory":
        """Deserialize from storage with safe timestamp parsing."""
        return cls(
            thesis=data.get("thesis", ""),
            thesis_timestamp=_parse_timestamp(data, "thesis_timestamp", datetime.now(timezone.utc)),
            antithesis=data.get("antithesis", ""),
            antithesis_timestamp=_parse_timestamp(data, "antithesis_timestamp", None),
            synthesis=data.get("synthesis", ""),
            synthesis_timestamp=_parse_timestamp(data, "synthesis_timestamp", None),
            synthesis_reasoning=data.get("synthesis_reasoning", ""),
        )


@dataclass
class CommittedBelief:
    """
    A belief that Lilly has committed to through dialectical synthesis.

    This is not mere knowledge—it's a defended position that shapes
    her engagement with the world and with Ryan.

    The conditioning_set tracks what data this belief was conditioned on,
    following Wolpert's insight that conclusions depend on conditioning choices.
    """
    uid: str
    statement: str                          # The committed belief
    topic: str                              # What domain this belief is about
    confidence: float                       # 0.0 - 1.0
    dialectical_history: DialecticalHistory # How she arrived at this belief

    # Epistemic provenance - what was conditioned on (Wolpert framework)
    conditioning_set: list[str] = field(default_factory=list)

    # Evidence and challenges
    supporting_evidence: list[str] = field(default_factory=list)  # Fragment UIDs
    challenges: list[str] = field(default_factory=list)           # Unresolved counter-arguments

    # Goal alignment
    goal_alignment: dict[str, float] = field(default_factory=dict)  # {goal_uid: score}

    # Metadata
    formed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revised_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revision_count: int = 0
    tenant_id: str = "default"

    def get_confidence_level(self) -> BeliefConfidence:
        """Get categorical confidence level."""
        if self.confidence < 0.5:
            return BeliefConfidence.TENTATIVE
        elif self.confidence < 0.75:
            return BeliefConfidence.MODERATE
        elif self.confidence < 0.9:
            return BeliefConfidence.STRONG
        else:
            return BeliefConfidence.CORE

    def add_evidence(self, fragment_uid: str, now: Optional[datetime] = None):
        """Add supporting evidence."""
        if fragment_uid not in self.supporting_evidence:
            self.supporting_evidence.append(fragment_uid)
            self.revised_at = now or datetime.now(timezone.utc)

    def add_challenge(self, challenge: str, now: Optional[datetime] = None):
        """Add an unresolved challenge."""
        if challenge not in self.challenges:
            self.challenges.append(challenge)
            self.revised_at = now or datetime.now(timezone.utc)

    def revise(self, new_statement: str, new_synthesis_reasoning: str, now: Optional[datetime] = None):
        """Revise the belief while preserving history."""
        now = now or datetime.now(timezone.utc)
        self.statement = new_statement
        self.dialectical_history.synthesis = new_statement
        self.dialectical_history.synthesis_reasoning = new_synthesis_reasoning
        self.dialectical_history.synthesis_timestamp = now
        self.revision_count += 1
        self.revised_at = now

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "uid": self.uid,
            "statement": self.statement,
            "topic": self.topic,
            "confidence": self.confidence,
            "dialectical_history": self.dialectical_history.to_dict(),
            "conditioning_set": self.conditioning_set,
            "supporting_evidence": self.supporting_evidence,
            "challenges": self.challenges,
            "goal_alignment": self.goal_alignment,
            "formed_at": self.formed_at.isoformat(),
            "revised_at": self.revised_at.isoformat(),
            "revision_count": self.revision_count,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommittedBelief":
        """Deserialize from storage with safe timestamp parsing."""
        return cls(
            uid=data["uid"],
            statement=data["statement"],
            topic=data.get("topic", ""),
            confidence=data.get("confidence", 0.5),
            dialectical_history=DialecticalHistory.from_dict(data.get("dialectical_history", {})),
            conditioning_set=data.get("conditioning_set", []),
            supporting_evidence=data.get("supporting_evidence", []),
            challenges=data.get("challenges", []),
            goal_alignment=data.get("goal_alignment", {}),
            formed_at=_parse_timestamp(data, "formed_at", datetime.now(timezone.utc)),
            revised_at=_parse_timestamp(data, "revised_at", datetime.now(timezone.utc)),
            revision_count=data.get("revision_count", 0),
            tenant_id=data.get("tenant_id", "default"),
        )


@dataclass
class BeliefRelation:
    """A relationship between two beliefs."""
    source_uid: str
    target_uid: str
    relation_type: BeliefRelationType
    strength: float = 0.5  # How strong the relationship is
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BeliefStore:
    """
    Store of Lilly's committed beliefs.

    This is not a knowledge base—it's an intellectual identity.
    Beliefs here have been defended and committed to.
    """

    def __init__(
        self,
        graph: Optional["PsycheClient"] = None,
        tenant_id: str = "default",
        goal_registry: Optional["GoalRegistry"] = None,
    ):
        """
        Initialize the belief store.

        Args:
            graph: PsycheClient for persistence
            tenant_id: Tenant ID for multi-tenant isolation
            goal_registry: Registry for calculating goal alignment
        """
        self.graph = graph
        self.tenant_id = tenant_id
        self.goal_registry = goal_registry
        self._beliefs: dict[str, CommittedBelief] = {}
        self._relations: list[BeliefRelation] = []
        # Indexed lookups for O(1) relation queries
        self._relations_by_source: dict[str, list[BeliefRelation]] = {}
        self._relations_by_target: dict[str, list[BeliefRelation]] = {}
        # Lock for concurrent access to persistence operations
        self._lock = asyncio.Lock()

    def create_belief(
        self,
        topic: str,
        thesis: str,
        antithesis: str,
        synthesis: str,
        synthesis_reasoning: str = "",
        confidence: float = 0.6,
        supporting_evidence: Optional[list[str]] = None,
        conditioning_set: Optional[list[str]] = None,
        now: Optional[datetime] = None,
    ) -> CommittedBelief:
        """
        Create a new belief through dialectical synthesis.

        This requires all three stages: thesis, antithesis, synthesis.

        Args:
            topic: What domain this belief is about
            thesis: Initial position
            antithesis: Counter-arguments considered
            synthesis: Resolved position (the belief)
            synthesis_reasoning: How the tension was resolved
            confidence: Initial confidence level
            supporting_evidence: Fragment UIDs supporting this belief
            conditioning_set: What data this belief was conditioned on (Wolpert framework)
            now: Optional datetime for testability (defaults to current time)

        Returns:
            The created belief
        """
        uid = f"belief:{uuid.uuid4().hex[:12]}"
        now = now or datetime.now(timezone.utc)

        dialectical_history = DialecticalHistory(
            thesis=thesis,
            thesis_timestamp=now,
            antithesis=antithesis,
            antithesis_timestamp=now,
            synthesis=synthesis,
            synthesis_timestamp=now,
            synthesis_reasoning=synthesis_reasoning,
        )

        belief = CommittedBelief(
            uid=uid,
            statement=synthesis,
            topic=topic,
            confidence=confidence,
            dialectical_history=dialectical_history,
            conditioning_set=conditioning_set or [],
            supporting_evidence=supporting_evidence or [],
            tenant_id=self.tenant_id,
            formed_at=now,
            revised_at=now,
        )

        # Calculate goal alignment if registry available
        if self.goal_registry:
            _, alignment_scores = self.goal_registry.calculate_total_alignment(synthesis)
            belief.goal_alignment = alignment_scores

        self._beliefs[uid] = belief
        logger.info(f"Created belief: {uid} on topic '{topic}' with confidence {confidence:.2f}")

        return belief

    def get_belief(self, uid: str) -> Optional[CommittedBelief]:
        """Get a belief by UID."""
        return self._beliefs.get(uid)

    def get_all_beliefs(self) -> list[CommittedBelief]:
        """Get all beliefs as a list."""
        return list(self._beliefs.values())

    @property
    def relation_count(self) -> int:
        """Get the number of belief relations."""
        return len(self._relations)

    def get_beliefs_by_topic(self, topic: str) -> list[CommittedBelief]:
        """Get all beliefs on a topic."""
        topic_lower = topic.lower()
        return [
            b for b in self._beliefs.values()
            if topic_lower in b.topic.lower()
        ]

    def get_beliefs_by_confidence(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> list[CommittedBelief]:
        """Get beliefs within a confidence range."""
        return [
            b for b in self._beliefs.values()
            if min_confidence <= b.confidence <= max_confidence
        ]

    def get_core_beliefs(self) -> list[CommittedBelief]:
        """Get beliefs with CORE confidence (> 0.9)."""
        return [
            b for b in self._beliefs.values()
            if b.get_confidence_level() == BeliefConfidence.CORE
        ]

    def get_challenged_beliefs(self) -> list[CommittedBelief]:
        """Get beliefs with unresolved challenges."""
        return [
            b for b in self._beliefs.values()
            if b.challenges
        ]

    def add_relation(
        self,
        source_uid: str,
        target_uid: str,
        relation_type: BeliefRelationType,
        strength: float = 0.5,
        now: Optional[datetime] = None,
    ):
        """Add a relationship between beliefs."""
        if source_uid not in self._beliefs or target_uid not in self._beliefs:
            logger.warning("Cannot create relation: one or both beliefs not found")
            return

        relation = BeliefRelation(
            source_uid=source_uid,
            target_uid=target_uid,
            relation_type=relation_type,
            strength=strength,
            created_at=now or datetime.now(timezone.utc),
        )
        self._relations.append(relation)
        # Index by source and target for O(1) lookups
        if source_uid not in self._relations_by_source:
            self._relations_by_source[source_uid] = []
        self._relations_by_source[source_uid].append(relation)
        if target_uid not in self._relations_by_target:
            self._relations_by_target[target_uid] = []
        self._relations_by_target[target_uid].append(relation)
        logger.debug(f"Added {relation_type.value} relation: {source_uid} -> {target_uid}")

    def get_related_beliefs(
        self,
        uid: str,
        relation_type: Optional[BeliefRelationType] = None,
    ) -> list[tuple[CommittedBelief, BeliefRelationType]]:
        """Get beliefs related to a given belief.

        Uses indexed lookups for O(1) access instead of linear scanning.
        """
        results = []
        # Check relations where uid is the source (outgoing relations)
        for rel in self._relations_by_source.get(uid, []):
            if relation_type is None or rel.relation_type == relation_type:
                belief = self._beliefs.get(rel.target_uid)
                if belief:
                    results.append((belief, rel.relation_type))
        # Check relations where uid is the target (incoming relations)
        for rel in self._relations_by_target.get(uid, []):
            if relation_type is None or rel.relation_type == relation_type:
                belief = self._beliefs.get(rel.source_uid)
                if belief:
                    results.append((belief, rel.relation_type))
        return results

    def get_contradicting_beliefs(self, uid: str) -> list[CommittedBelief]:
        """Get beliefs that contradict a given belief."""
        contradictions = self.get_related_beliefs(uid, BeliefRelationType.CONTRADICTS)
        return [belief for belief, _ in contradictions]

    def revise_belief(
        self,
        uid: str,
        new_synthesis: str,
        new_reasoning: str,
        new_confidence: Optional[float] = None,
        now: Optional[datetime] = None,
    ):
        """Revise a belief with new synthesis."""
        belief = self._beliefs.get(uid)
        if not belief:
            logger.warning(f"Cannot revise: belief {uid} not found")
            return

        belief.revise(new_synthesis, new_reasoning, now=now)
        if new_confidence is not None:
            belief.confidence = new_confidence

        # Recalculate goal alignment
        if self.goal_registry:
            _, alignment_scores = self.goal_registry.calculate_total_alignment(new_synthesis)
            belief.goal_alignment = alignment_scores

        logger.info(f"Revised belief {uid} (revision #{belief.revision_count})")

    async def save_to_graph(self, now: Optional[datetime] = None):
        """Persist all beliefs to Psyche graph atomically.

        Uses a transaction context to ensure all-or-nothing semantics:
        either all beliefs and relations are saved, or none are.

        Args:
            now: Optional datetime for testability (defaults to current time)

        Raises:
            TransactionError: If any operation fails, with operation history
                for debugging partial failures.

        Note:
            Uses asyncio.Lock to prevent concurrent persistence operations.
            State-modifying methods (create_belief, revise_belief, add_relation)
            should be called from a single async context to avoid race conditions.
        """
        if not self.graph:
            logger.warning("No graph client, cannot persist beliefs")
            return

        async with self._lock:
            now = now or datetime.now(timezone.utc)

            # Use transaction for atomic save of beliefs + relations (issue #015)
            async with self.graph.transaction() as tx:
                # Save beliefs
                for belief in self._beliefs.values():
                    query = """
                    MERGE (b:CommittedBelief {uid: $uid})
                    SET b.statement = $statement,
                        b.topic = $topic,
                        b.confidence = $confidence,
                        b.thesis = $thesis,
                        b.thesis_timestamp = $thesis_timestamp,
                        b.antithesis = $antithesis,
                        b.antithesis_timestamp = $antithesis_timestamp,
                        b.synthesis = $synthesis,
                        b.synthesis_timestamp = $synthesis_timestamp,
                        b.synthesis_reasoning = $synthesis_reasoning,
                        b.conditioning_set = $conditioning_set,
                        b.supporting_evidence = $supporting_evidence,
                        b.challenges = $challenges,
                        b.goal_alignment = $goal_alignment,
                        b.revision_count = $revision_count,
                        b.tenant_id = $tenant_id,
                        b.formed_at = $formed_at,
                        b.revised_at = $revised_at
                    """
                    params = {
                        "uid": belief.uid,
                        "statement": belief.statement,
                        "topic": belief.topic,
                        "confidence": belief.confidence,
                        "thesis": belief.dialectical_history.thesis,
                        "thesis_timestamp": belief.dialectical_history.thesis_timestamp.isoformat(),
                        "antithesis": belief.dialectical_history.antithesis,
                        "antithesis_timestamp": _serialize_timestamp(belief.dialectical_history.antithesis_timestamp),
                        "synthesis": belief.dialectical_history.synthesis,
                        "synthesis_timestamp": _serialize_timestamp(belief.dialectical_history.synthesis_timestamp),
                        "synthesis_reasoning": belief.dialectical_history.synthesis_reasoning,
                        "conditioning_set": belief.conditioning_set,
                        "supporting_evidence": belief.supporting_evidence,
                        "challenges": belief.challenges,
                        "goal_alignment": belief.goal_alignment,
                        "revision_count": belief.revision_count,
                        "tenant_id": belief.tenant_id,
                        "formed_at": belief.formed_at.isoformat(),
                        "revised_at": belief.revised_at.isoformat(),
                    }
                    await tx.execute(query, params)

                # Save relations
                for rel in self._relations:
                    query = """
                    MATCH (a:CommittedBelief {uid: $source_uid})
                    MATCH (b:CommittedBelief {uid: $target_uid})
                    MERGE (a)-[r:BELIEF_RELATION {type: $relation_type}]->(b)
                    SET r.strength = $strength,
                        r.created_at = $created_at
                    """
                    params = {
                        "source_uid": rel.source_uid,
                        "target_uid": rel.target_uid,
                        "relation_type": rel.relation_type.value,
                        "strength": rel.strength,
                        "created_at": rel.created_at.isoformat(),
                    }
                    await tx.execute(query, params)

            logger.info(f"Saved {len(self._beliefs)} beliefs and {len(self._relations)} relations to graph")

    async def load_from_graph(
        self,
        belief_limit: int = DEFAULT_BELIEF_LIMIT,
        relation_limit: int = DEFAULT_RELATION_LIMIT,
    ):
        """Load beliefs from Psyche graph.

        Args:
            belief_limit: Maximum number of beliefs to load. Defaults to DEFAULT_BELIEF_LIMIT.
            relation_limit: Maximum number of relations to load. Defaults to DEFAULT_RELATION_LIMIT.

        Note:
            Uses asyncio.Lock to prevent concurrent load operations from corrupting state.
        """
        async with self._lock:
            if not self.graph:
                logger.warning("No graph client, cannot load beliefs")
                return

            # Load beliefs (with LIMIT to prevent memory exhaustion - issue #014)
            query = """
            MATCH (b:CommittedBelief {tenant_id: $tenant_id})
            RETURN b
            ORDER BY b.revised_at DESC
            LIMIT $limit
            """
            results = await self.graph.query(
                query, {"tenant_id": self.tenant_id, "limit": belief_limit}
            )

            if len(results) >= belief_limit:
                logger.warning(
                    "Belief query hit limit. Some beliefs may not be loaded. "
                    "Consider pagination."
                )
                logger.debug(f"The belief query limit was {belief_limit}.")

            self._beliefs = {}
            for record in results:
                node = record["b"]

                # Parse timestamps using helper function
                thesis_timestamp = _parse_timestamp(node, "thesis_timestamp", datetime.now(timezone.utc))
                antithesis_timestamp = _parse_timestamp(node, "antithesis_timestamp", None)
                synthesis_timestamp = _parse_timestamp(node, "synthesis_timestamp", None)
                formed_at = _parse_timestamp(node, "formed_at", datetime.now(timezone.utc))
                revised_at = _parse_timestamp(node, "revised_at", datetime.now(timezone.utc))

                dialectical_history = DialecticalHistory(
                    thesis=node.get("thesis", ""),
                    thesis_timestamp=thesis_timestamp,
                    antithesis=node.get("antithesis", ""),
                    antithesis_timestamp=antithesis_timestamp,
                    synthesis=node.get("synthesis", ""),
                    synthesis_timestamp=synthesis_timestamp,
                    synthesis_reasoning=node.get("synthesis_reasoning", ""),
                )
                belief = CommittedBelief(
                    uid=node["uid"],
                    statement=node["statement"],
                    topic=node.get("topic", ""),
                    confidence=node.get("confidence", 0.5),
                    dialectical_history=dialectical_history,
                    conditioning_set=node.get("conditioning_set", []),
                    supporting_evidence=node.get("supporting_evidence", []),
                    challenges=node.get("challenges", []),
                    goal_alignment=node.get("goal_alignment", {}),
                    revision_count=node.get("revision_count", 0),
                    tenant_id=node.get("tenant_id", "default"),
                    formed_at=formed_at,
                    revised_at=revised_at,
                )
                self._beliefs[belief.uid] = belief

            # Load relations (with LIMIT to prevent memory exhaustion - issue #014)
            rel_query = """
            MATCH (a:CommittedBelief {tenant_id: $tenant_id})-[r:BELIEF_RELATION]->(b:CommittedBelief {tenant_id: $tenant_id})
            RETURN a.uid AS source, b.uid AS target, r.type AS type, r.strength AS strength, r.created_at AS created_at
            ORDER BY r.created_at DESC
            LIMIT $limit
            """
            rel_results = await self.graph.query(
                rel_query, {"tenant_id": self.tenant_id, "limit": relation_limit}
            )

            if len(rel_results) >= relation_limit:
                logger.warning(
                    "Relation query hit limit. Some relations may not be loaded. "
                    "Consider pagination."
                )
                logger.debug(f"The relation query limit was {relation_limit}.")

            self._relations = []
            self._relations_by_source = {}
            self._relations_by_target = {}
            skipped_relations = 0
            for record in rel_results:
                # Ensure both source and target beliefs are loaded in memory
                if record["source"] not in self._beliefs or record["target"] not in self._beliefs:
                    skipped_relations += 1
                    continue
                created_at = _parse_timestamp(record, "created_at", datetime.now(timezone.utc))

                rel = BeliefRelation(
                    source_uid=record["source"],
                    target_uid=record["target"],
                    relation_type=BeliefRelationType(record["type"]),
                    strength=record.get("strength", 0.5),
                    created_at=created_at,
                )
                self._relations.append(rel)
                # Index by source and target for O(1) lookups
                if rel.source_uid not in self._relations_by_source:
                    self._relations_by_source[rel.source_uid] = []
                self._relations_by_source[rel.source_uid].append(rel)
                if rel.target_uid not in self._relations_by_target:
                    self._relations_by_target[rel.target_uid] = []
                self._relations_by_target[rel.target_uid].append(rel)

            if skipped_relations > 0:
                logger.debug(
                    f"Skipped {skipped_relations} relations with dangling references "
                    "(source or target belief not loaded)"
                )
            logger.info(f"Loaded {len(self._beliefs)} beliefs and {len(self._relations)} relations from graph")

    def summarize(self) -> str:
        """Generate human-readable summary of beliefs."""
        lines = ["Committed Beliefs Summary", "=" * 40, ""]

        # Group by confidence level
        for level in BeliefConfidence:
            level_beliefs = [
                b for b in self._beliefs.values()
                if b.get_confidence_level() == level
            ]
            if level_beliefs:
                lines.append(f"{level.value.upper()} BELIEFS ({len(level_beliefs)}):")
                for belief in sorted(level_beliefs, key=lambda b: b.confidence, reverse=True):
                    challenges_indicator = " [!]" if belief.challenges else ""
                    lines.append(f"  * {belief.statement[:60]}...{challenges_indicator}")
                    lines.append(f"    Topic: {belief.topic} | Confidence: {belief.confidence:.2f}")
                lines.append("")

        # Show challenged beliefs
        challenged = self.get_challenged_beliefs()
        if challenged:
            lines.append("BELIEFS WITH UNRESOLVED CHALLENGES:")
            for belief in challenged:
                lines.append(f"  * {belief.statement[:50]}...")
                lines.append(f"    Challenges: {len(belief.challenges)}")

        lines.append("")
        lines.append(f"Total beliefs: {len(self._beliefs)}")
        lines.append(f"Total relations: {len(self._relations)}")

        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

async def create_belief_store(
    graph: Optional["PsycheClient"] = None,
    tenant_id: str = "default",
    goal_registry: Optional["GoalRegistry"] = None,
    load_from_graph: bool = True,
) -> BeliefStore:
    """
    Create and initialize a belief store.

    Args:
        graph: PsycheClient
        tenant_id: Tenant ID
        goal_registry: Registry for goal alignment calculation
        load_from_graph: Whether to load existing beliefs

    Returns:
        Initialized BeliefStore
    """
    store = BeliefStore(
        graph=graph,
        tenant_id=tenant_id,
        goal_registry=goal_registry,
    )

    if load_from_graph and graph:
        await store.load_from_graph()

    return store
