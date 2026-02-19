"""FalkorDB client for the Psyche knowledge graph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, TYPE_CHECKING

from core.embedding.service import EmbeddingTier

if TYPE_CHECKING:
    from core.psyche.schema import (
        AutonomousDecision,
        CognitiveAnchor,
        CognitiveStateSnapshot,
        Entity,
        Fragment,
        FragmentState,
        HypothesisSteeringVector,
        InsightZettel,
        LearnedSkill,
        NarrationPhrase,
        PhraseType,
        PredictionPattern,
        Triple,
    )
    from core.cognitive.experimentation.schemas import (
        ExperimentDomain,
        ExperimentPhase,
        ExperimentMeasurement,
    )
    from core.cognitive.simulation.schemas import Hypothesis, Prediction

logger = logging.getLogger(__name__)

# Skill retirement thresholds
SKILL_MIN_USES_FOR_RETIREMENT = 10
SKILL_MIN_EFFECTIVENESS_FOR_RETIREMENT = 0.3

# Coactivation edge importance EMA parameters
_IMPORTANCE_EMA_DECAY = 0.95
_IMPORTANCE_EMA_LEARNING_RATE = 0.5

# Zettel ID generation
ZETTEL_ID_HEX_LENGTH = 12


def _truncate_params(params: Optional[dict], max_len: int = 200) -> str:
    """Truncate params dict string representation to avoid large log entries.

    This prevents logging excessively large values like embeddings or
    sensitive information that may be present in query parameters.

    Args:
        params: The parameters dictionary to truncate.
        max_len: Maximum length of the string representation.

    Returns:
        Truncated string representation with "..." suffix if truncated.
    """
    if params is None:
        return "None"
    params_str = str(params)
    if len(params_str) > max_len:
        return params_str[:max_len] + "..."
    return params_str


def _sanitize_string(value: str) -> str:
    """Remove null bytes from strings for FalkorDB compatibility.

    FalkorDB/Redis uses C strings which are null-terminated. Null bytes
    in string parameters cause query truncation and parsing errors.

    Args:
        value: String to sanitize.

    Returns:
        String with null bytes removed.
    """
    return value.replace("\x00", "") if value else value


def _normalize_entity_name(name: str) -> str:
    """Normalize entity names for consistent matching and deduplication.

    Converts various naming conventions to a canonical form:
    - Replaces underscores with spaces (snake_case → natural language)
    - Replaces hyphens with spaces (kebab-case → natural language)
    - Collapses multiple spaces
    - Strips leading/trailing whitespace
    - Converts to lowercase

    This prevents duplicate entities like:
    - "phenomenology_of_cognition" vs "phenomenology of cognition"
    - "self-awareness" vs "self awareness"

    Args:
        name: Raw entity name to normalize.

    Returns:
        Normalized entity name.
    """
    if not name:
        return ""

    # Replace underscores and hyphens with spaces
    result = name.replace("_", " ").replace("-", " ")

    # Collapse multiple spaces and strip
    result = re.sub(r"\s+", " ", result).strip()

    # Lowercase for consistent matching
    result = result.lower()

    return result


def _node_to_props(value: Any) -> Any:
    """Convert a FalkorDB Node to its properties dict.

    FalkorDB returns Node objects when queries use `RETURN h` for nodes.
    These Node objects have a `.properties` attribute containing the actual
    data as a dict. This helper normalizes the return value for consistent
    handling whether the query returns a Node or already a dict.

    Args:
        value: A value from a FalkorDB query result, which could be a
               Node object, a dict, or another type.

    Returns:
        The properties dict from a Node, or the value unchanged if it's not a Node.
    """
    return getattr(value, "properties", value)


def _parse_datetime_field(
    value: Any,
    field_name: str = "datetime",
    default: datetime | None = None,
) -> datetime:
    """Parse a datetime field from FalkorDB with consistent error handling.

    FalkorDB stores datetime fields as ISO format strings. This function
    provides consistent parsing with proper error logging and fallback
    behavior to avoid type inconsistencies (string vs datetime).

    Args:
        value: The raw value from FalkorDB (expected to be ISO string or None)
        field_name: Name of the field for error messages
        default: Default datetime to use if parsing fails or value is None.
                 If None, uses datetime.now(timezone.utc).

    Returns:
        Parsed datetime object, never a string.
    """
    if default is None:
        default = datetime.now(tz=timezone.utc)

    if not value:
        return default

    if isinstance(value, datetime):
        return value

    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Invalid {field_name} timestamp format: {value!r} - {e}"
        )
        return default


# Resource limit constants for graph query operations
DEFAULT_QUERY_TIMEOUT = 30.0  # seconds - prevents runaway queries
MAX_CONCURRENT_QUERIES = 10  # limits concurrent load on the database
EXPENSIVE_OPERATION_TIMEOUT = 60.0  # seconds - for operations like full graph scans

# Skill effectiveness tracking constants
SKILL_EFFECTIVENESS_EMA_ALPHA = 0.2  # EMA smoothing factor for skill effectiveness updates


@dataclass
class TransactionOperation:
    """Record of an operation executed within a transaction."""

    cypher: str
    params: dict
    result: Any = None
    success: bool = False


class Transaction:
    """
    Fail-fast operation sequence with tracking for multi-step graph operations.

    IMPORTANT: This class does NOT provide true ACID atomicity. FalkorDB (a graph
    database using Cypher) does not support multi-operation transactions like
    Redis pipelines or SQL BEGIN/COMMIT blocks. The FalkorDB Python client has
    no pipeline(), transaction(), MULTI, or EXEC methods.

    What this class provides:
    - Sequential execution of operations with fail-fast behavior
    - Operation tracking and history for debugging partial failures
    - Clear error reporting showing which operation failed and which succeeded
    - A consistent interface for grouping related operations

    What this class does NOT provide:
    - Rollback of previously executed operations on failure
    - Isolation from concurrent operations
    - True atomicity guarantees

    If an operation fails mid-sequence, earlier operations remain committed to
    the database. The TransactionError will contain the full operation history,
    allowing callers to implement compensating actions if needed.

    Usage:
        async with client.transaction() as tx:
            await tx.execute(CREATE_QUERY, params1)
            await tx.execute(UPDATE_QUERY, params2)
            # On success: all operations completed
            # On failure: TransactionError with operation history
    """

    def __init__(self, client: "PsycheClient"):
        self._client = client
        self._operations: list[TransactionOperation] = []
        self._committed = False

    @property
    def operations(self) -> list[TransactionOperation]:
        """Get list of operations executed in this transaction."""
        return self._operations

    async def _run_operation(
        self,
        client_method,
        cypher: str,
        params: Optional[dict],
        timeout: Optional[float] = None,
    ):
        """Execute an operation within the transaction, tracking success/failure."""
        op = TransactionOperation(cypher=cypher, params=params or {})
        self._operations.append(op)
        try:
            result = await client_method(cypher, params, timeout=timeout)
            op.result = result
            op.success = True
            return result
        except Exception as e:
            op.result = str(e)
            op.success = False
            raise TransactionError(
                f"Transaction failed at operation {len(self._operations)}: {e}",
                operations=self._operations,
            ) from e

    async def execute(
        self,
        cypher: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Execute a write operation within the transaction."""
        return await self._run_operation(self._client.execute, cypher, params, timeout)

    async def query(
        self,
        cypher: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> list[dict]:
        """Execute a read operation within the transaction."""
        return await self._run_operation(self._client.query, cypher, params, timeout)

    def _mark_committed(self) -> None:
        """Mark the transaction as successfully committed."""
        self._committed = True


class TransactionError(Exception):
    """Error raised when a transaction fails, with operation history for debugging."""

    def __init__(self, message: str, operations: list[TransactionOperation]):
        super().__init__(message)
        self.operations = operations

    def get_failed_operation(self) -> Optional[TransactionOperation]:
        """Get the operation that caused the failure."""
        for op in self.operations:
            if not op.success:
                return op
        return None

    def get_successful_operations(self) -> list[TransactionOperation]:
        """Get operations that completed before the failure."""
        return [op for op in self.operations if op.success]


class EntityNotFoundError(Exception):
    """Error raised when a referenced entity does not exist.

    This provides clear feedback for foreign key validation failures,
    preventing silent creation of orphaned relationships.
    """

    def __init__(self, entity_type: str, uid: str):
        super().__init__(f"{entity_type} with uid '{uid}' not found")
        self.entity_type = entity_type
        self.uid = uid


# Node types that support FK validation
VALIDATABLE_NODE_TYPES = frozenset({"Fragment", "Entity", "Triple", "CommittedBelief"})

# Lazy import for FalkorDB
FalkorDB = None
FALKORDB_AVAILABLE = False

try:
    from falkordb import FalkorDB
    FALKORDB_AVAILABLE = True
except ImportError:
    logger.warning("falkordb not installed, PsycheClient will be disabled")


class PsycheClient:
    """
    Async client for FalkorDB knowledge graph.

    This is Lilly's persistent psyche - all memories, knowledge,
    steering vectors, and introspective entries live here.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = "",
        database: str = "lilly",
        query_timeout: float = DEFAULT_QUERY_TIMEOUT,
        max_concurrent_queries: int = MAX_CONCURRENT_QUERIES,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.database = database
        self.query_timeout = query_timeout
        self._client = None
        self._graph = None
        self._connected = False
        # Semaphore to limit concurrent queries
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def connect(self) -> None:
        """Establish connection to FalkorDB."""
        if not FALKORDB_AVAILABLE:
            raise RuntimeError("falkordb package not installed")

        def _sync_connect():
            client = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password or None,
            )
            graph = client.select_graph(self.database)
            return client, graph

        try:
            self._client, self._graph = await asyncio.to_thread(_sync_connect)
            self._connected = True
            logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection to FalkorDB."""
        if self._client:
            self._connected = False
            self._client = None
            self._graph = None
            logger.info("Disconnected from FalkorDB")

    async def health_check(self) -> dict[str, Any]:
        """Check database health."""
        if not self._connected:
            return {"status": "disconnected"}

        try:
            await asyncio.to_thread(self._graph.query, "RETURN 1 as health")
            return {
                "status": "healthy",
                "host": self.host,
                "port": self.port,
                "database": self.database,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def _run_with_timeout_and_semaphore(
        self,
        cypher: str,
        params: Optional[dict],
        timeout: Optional[float],
        operation_name: str,
    ):
        """
        Execute a Cypher query with timeout and semaphore protection.

        This private helper consolidates the shared logic between query() and
        execute() methods: connection checking, timeout determination, semaphore
        acquisition, and exception handling with truncated logging.

        Args:
            cypher: Cypher query string
            params: Optional parameters for the query
            timeout: Optional timeout override (uses self.query_timeout if not specified)
            operation_name: Name for logging (e.g., "Query" or "Execute")

        Returns:
            The raw FalkorDB query result object

        Raises:
            RuntimeError: If not connected to FalkorDB
            TimeoutError: If query exceeds timeout
        """
        if not self._connected:
            raise RuntimeError("Not connected to FalkorDB")

        effective_timeout = timeout if timeout is not None else self.query_timeout

        async with self._query_semaphore:
            try:
                async with asyncio.timeout(effective_timeout):
                    return await asyncio.to_thread(
                        self._graph.query, cypher, params or {}
                    )
            except TimeoutError:
                logger.error(
                    f"{operation_name} timeout after {effective_timeout}s\n"
                    f"Query: {cypher[:200]}{'...' if len(cypher) > 200 else ''}\n"
                    f"Params: {_truncate_params(params)}"
                )
                raise
            except Exception as e:
                logger.error(
                    f"{operation_name} failed: {e}\n"
                    f"Query: {cypher[:200]}{'...' if len(cypher) > 200 else ''}"
                )
                raise

    async def query(
        self,
        cypher: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> list[dict]:
        """Execute Cypher query and return results.

        Args:
            cypher: Cypher query string
            params: Optional parameters for the query
            timeout: Optional timeout override (uses self.query_timeout if not specified)

        Returns:
            List of result dictionaries

        Raises:
            RuntimeError: If not connected to FalkorDB
            TimeoutError: If query exceeds timeout
        """
        result = await self._run_with_timeout_and_semaphore(
            cypher, params, timeout, "Query"
        )
        if result.result_set:
            # FalkorDB headers can be either plain strings or lists with type info
            # Format varies: ['name'], [1, 'name'], or just 'name'
            # Extract the string element which is the column name
            def extract_header_name(h):
                if isinstance(h, str):
                    return h
                if isinstance(h, (list, tuple)):
                    # Find the string element (column name) in the list
                    for item in h:
                        if isinstance(item, str):
                            return item
                    # Fallback to first element if no string found
                    logger.warning(f"Unexpected header format (no string found): {h}")
                    return str(h[0]) if h else "unknown"
                return str(h)

            headers = [extract_header_name(h) for h in result.header]
            return [dict(zip(headers, row)) for row in result.result_set]
        return []

    async def execute(
        self,
        cypher: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Execute Cypher write query and return affected count.

        Args:
            cypher: Cypher query string
            params: Optional parameters for the query
            timeout: Optional timeout override (uses self.query_timeout if not specified)

        Returns:
            Number of affected nodes/relationships/properties

        Raises:
            RuntimeError: If not connected to FalkorDB
            TimeoutError: If query exceeds timeout
        """
        result = await self._run_with_timeout_and_semaphore(
            cypher, params, timeout, "Execute"
        )
        # FalkorDB client exposes stats as direct attributes on QueryResult
        return (
            (result.nodes_created or 0) +
            (result.relationships_created or 0) +
            (result.properties_set or 0)
        )

    @asynccontextmanager
    async def transaction(self):
        """
        Create a transaction context for atomic multi-step operations.

        All operations within the context are executed sequentially.
        If any operation fails, subsequent operations are not executed
        and a TransactionError is raised with operation history.

        Usage:
            async with client.transaction() as tx:
                await tx.execute(QUERY1, params1)
                await tx.execute(QUERY2, params2)
                # Success: all operations completed
                # Failure: TransactionError with operation history

        Note:
            This provides application-level transaction semantics.
            For true database-level ACID transactions, FalkorDB Enterprise
            or a different graph database may be required.
        """
        tx = Transaction(self)
        try:
            yield tx
            tx._mark_committed()
            logger.debug(f"Transaction committed with {len(tx.operations)} operations")
        except TransactionError:
            # Already logged at operation level
            raise
        except Exception as e:
            logger.error(f"Transaction failed unexpectedly: {e}")
            raise TransactionError(
                f"Transaction failed: {e}",
                operations=tx.operations,
            ) from e

    # === Fragment CRUD Operations ===

    async def create_fragment(self, fragment: "Fragment") -> bool:
        """Create a Fragment node in the graph."""
        # Build embedding clause - must use vecf32() for vector index compatibility
        if fragment.embedding is not None:
            embedding_str = ", ".join(str(x) for x in fragment.embedding)
            embedding_clause = f"vecf32([{embedding_str}])"
        else:
            embedding_clause = "null"

        cypher = f"""
        CREATE (f:Fragment {{
            uid: $uid,
            content: $content,
            source: $source,
            state: $state,
            resonance: $resonance,
            confidence: $confidence,
            created_at: $created_at,
            last_accessed: $last_accessed,
            embedding: {embedding_clause}
        }})
        """
        params = {
            "uid": _sanitize_string(fragment.uid),
            "content": _sanitize_string(fragment.content),
            "source": _sanitize_string(fragment.source),
            "state": fragment.state.value,
            "resonance": fragment.resonance,
            "confidence": fragment.confidence,
            "created_at": fragment.created_at.isoformat(),
            "last_accessed": fragment.last_accessed.isoformat(),
        }

        affected = await self.execute(cypher, params)
        return affected > 0

    async def update_fragment_embedding(
        self, uid: str, embedding: list[float]
    ) -> bool:
        """Update a fragment's embedding in the graph.

        Args:
            uid: Fragment unique identifier
            embedding: Embedding vector to store

        Returns:
            True if fragment was updated
        """
        embedding_str = ", ".join(str(x) for x in embedding)

        cypher = f"""
        MATCH (f:Fragment {{uid: $uid}})
        SET f.embedding = vecf32([{embedding_str}])
        """
        params = {"uid": uid}

        affected = await self.execute(cypher, params)
        return affected > 0

    async def get_fragment(self, uid: str) -> Optional["Fragment"]:
        """Retrieve a Fragment by UID."""
        from datetime import datetime

        from core.psyche.schema import Fragment, FragmentState

        cypher = """
        MATCH (f:Fragment {uid: $uid})
        RETURN f.uid, f.content, f.source, f.state, f.resonance,
               f.confidence, f.created_at, f.last_accessed
        """

        results = await self.query(cypher, {"uid": uid})
        if not results:
            return None

        row = results[0]
        return Fragment(
            uid=row["f.uid"],
            content=row["f.content"],
            source=row["f.source"],
            state=FragmentState(row["f.state"]),
            resonance=row["f.resonance"],
            confidence=row["f.confidence"],
            created_at=datetime.fromisoformat(row["f.created_at"]),
            last_accessed=datetime.fromisoformat(row["f.last_accessed"]),
        )

    async def update_fragment_resonance(self, uid: str, resonance: float) -> bool:
        """Update a Fragment's resonance score."""
        cypher = """
        MATCH (f:Fragment {uid: $uid})
        SET f.resonance = $resonance, f.last_accessed = $now
        RETURN f.uid
        """

        results = await self.query(cypher, {
            "uid": uid,
            "resonance": resonance,
            "now": datetime.now(timezone.utc).isoformat(),
        })
        return len(results) > 0

    async def update_fragment_state(self, uid: str, state: "FragmentState") -> bool:
        """Update a Fragment's state.

        Args:
            uid: Fragment unique identifier
            state: New state for the fragment

        Returns:
            True if fragment was updated
        """
        cypher = """
        MATCH (f:Fragment {uid: $uid})
        SET f.state = $state, f.last_accessed = $now
        """

        affected = await self.execute(cypher, {
            "uid": uid,
            "state": state.value,
            "now": datetime.now(timezone.utc).isoformat(),
        })
        return affected > 0

    # === Triple CRUD Operations ===

    async def create_triple(self, triple: "Triple") -> bool:
        """Create a Triple node in the graph."""

        cypher = """
        CREATE (t:Triple {
            uid: $uid,
            subject: $subject,
            predicate: $predicate,
            object: $object,
            confidence: $confidence,
            source_fragment_uid: $source_fragment_uid,
            created_at: $created_at
        })
        """
        params = {
            "uid": _sanitize_string(triple.uid),
            "subject": _sanitize_string(triple.subject),
            "predicate": _sanitize_string(triple.predicate),
            "object": _sanitize_string(triple.object),
            "confidence": triple.confidence,
            "source_fragment_uid": _sanitize_string(triple.source_fragment_uid),
            "created_at": triple.created_at.isoformat(),
        }

        affected = await self.execute(cypher, params)
        return affected > 0

    async def get_triples_for_fragment(self, fragment_uid: str) -> list["Triple"]:
        """Get all triples sourced from a fragment."""
        from core.psyche.schema import Triple

        cypher = """
        MATCH (t:Triple {source_fragment_uid: $fragment_uid})
        RETURN t.uid, t.subject, t.predicate, t.object, t.confidence,
               t.source_fragment_uid, t.created_at
        """

        results = await self.query(cypher, {"fragment_uid": fragment_uid})
        return [
            Triple(
                uid=r["t.uid"],
                subject=r["t.subject"],
                predicate=r["t.predicate"],
                object=r["t.object"],
                confidence=r["t.confidence"],
                source_fragment_uid=r["t.source_fragment_uid"],
                created_at=datetime.fromisoformat(r["t.created_at"]),
            )
            for r in results
        ]

    async def search_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        limit: int = 50,
    ) -> list["Triple"]:
        """Search triples by subject, predicate, or object (any combination)."""
        from core.psyche.schema import Triple

        conditions = []
        params: dict[str, Any] = {"limit": limit}

        if subject:
            conditions.append("t.subject CONTAINS $subject")
            params["subject"] = subject
        if predicate:
            conditions.append("t.predicate = $predicate")
            params["predicate"] = predicate
        if obj:
            conditions.append("t.object CONTAINS $object")
            params["object"] = obj

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        cypher = f"""
        MATCH (t:Triple)
        WHERE {where_clause}
        RETURN t.uid, t.subject, t.predicate, t.object, t.confidence,
               t.source_fragment_uid, t.created_at
        ORDER BY t.created_at DESC
        LIMIT $limit
        """

        results = await self.query(cypher, params)
        return [
            Triple(
                uid=r["t.uid"],
                subject=r["t.subject"],
                predicate=r["t.predicate"],
                object=r["t.object"],
                confidence=r["t.confidence"],
                source_fragment_uid=r["t.source_fragment_uid"],
                created_at=datetime.fromisoformat(r["t.created_at"]),
            )
            for r in results
        ]

    # === Entity CRUD Operations ===

    async def create_entity(self, entity: "Entity") -> bool:
        """Create an Entity node in the graph."""

        cypher = """
        CREATE (e:Entity {
            uid: $uid,
            name: $name,
            entity_type: $entity_type,
            description: $description,
            created_at: $created_at
        })
        """
        params = {
            "uid": _sanitize_string(entity.uid),
            "name": _sanitize_string(entity.name),
            "entity_type": _sanitize_string(entity.entity_type),
            "description": _sanitize_string(entity.description),
            "created_at": entity.created_at.isoformat(),
        }

        affected = await self.execute(cypher, params)
        return affected > 0

    async def upsert_entity(self, entity: "Entity") -> bool:
        """Create or update an Entity node in the graph.

        Uses MERGE on name to avoid duplicates. If the entity exists,
        updates description and entity_type.
        """
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.uid = $uid,
            e.entity_type = $entity_type,
            e.description = $description,
            e.created_at = $created_at
        ON MATCH SET
            e.entity_type = COALESCE($entity_type, e.entity_type),
            e.description = COALESCE($description, e.description)
        RETURN e
        """
        params = {
            "uid": _sanitize_string(entity.uid),
            "name": _sanitize_string(entity.name),
            "entity_type": _sanitize_string(entity.entity_type),
            "description": _sanitize_string(entity.description),
            "created_at": entity.created_at.isoformat(),
        }

        result = await self.query(cypher, params)
        return len(result) > 0

    async def get_entity(self, uid: str) -> Optional["Entity"]:
        """Retrieve an Entity by UID."""
        from core.psyche.schema import Entity

        cypher = """
        MATCH (e:Entity {uid: $uid})
        RETURN e.uid, e.name, e.entity_type, e.description, e.created_at
        """

        results = await self.query(cypher, {"uid": uid})
        if not results:
            return None

        r = results[0]
        return Entity(
            uid=r["e.uid"],
            name=r["e.name"],
            entity_type=r["e.entity_type"],
            description=r["e.description"],
            created_at=datetime.fromisoformat(r["e.created_at"]),
        )

    async def find_entity_by_name(self, name: str) -> Optional["Entity"]:
        """Find an entity by name (case-insensitive)."""
        from core.psyche.schema import Entity

        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($name)
        RETURN e.uid, e.name, e.entity_type, e.description, e.created_at
        """

        results = await self.query(cypher, {"name": name})
        if not results:
            return None

        r = results[0]
        return Entity(
            uid=r["e.uid"],
            name=r["e.name"],
            entity_type=r["e.entity_type"],
            description=r["e.description"],
            created_at=datetime.fromisoformat(r["e.created_at"]),
        )

    async def update_entity_salience(
        self, name: str, salience_delta: float
    ) -> bool:
        """Update an entity's salience, creating the entity if it doesn't exist.

        Uses MERGE to upsert the entity, then updates salience.
        Salience is clamped to [0.0, 1.0].

        Entity names are normalized before matching to prevent duplicates:
        - snake_case → "natural language" (underscores become spaces)
        - kebab-case → "natural language" (hyphens become spaces)
        - Multiple spaces collapsed, lowercase

        Args:
            name: Entity name (will be normalized)
            salience_delta: Amount to add to current salience

        Returns:
            True if updated successfully
        """
        # Normalize entity name to prevent duplicates like
        # "phenomenology_of_cognition" vs "phenomenology of cognition"
        normalized_name = _normalize_entity_name(name)

        if not normalized_name or len(normalized_name) < 2:
            logger.debug(f"Skipping entity with too-short name: '{name}'")
            return False

        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.uid = $uid,
            e.entity_type = 'concept',
            e.salience = $initial_salience,
            e.created_at = $created_at
        ON MATCH SET
            e.salience = CASE
                WHEN e.salience IS NULL THEN $salience_delta
                WHEN e.salience + $salience_delta < 0.0 THEN 0.0
                WHEN e.salience + $salience_delta > 1.0 THEN 1.0
                ELSE e.salience + $salience_delta
            END
        RETURN e.name
        """

        # Generate uid for potential creation
        import uuid as uuid_module
        uid = f"entity_{uuid_module.uuid4().hex[:12]}"
        initial_salience = max(0.0, min(1.0, 0.5 + salience_delta))

        params = {
            "name": _sanitize_string(normalized_name),
            "uid": uid,
            "salience_delta": salience_delta,
            "initial_salience": initial_salience,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = await self.query(cypher, params)
            if result:
                logger.debug(f"Updated entity salience: {normalized_name} ({salience_delta:+.2f})")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to update entity salience for {normalized_name}: {e}")
            return False

    # === Validation Helpers ===

    async def _validate_node_exists(
        self, node_type: str, uid: str, raise_error: bool = True
    ) -> bool:
        """Check if a node with the given type and uid exists.

        Args:
            node_type: The type of node (Fragment, Entity, Triple, etc.)
            uid: The unique identifier of the node
            raise_error: If True, raise EntityNotFoundError when not found

        Returns:
            True if the node exists, False otherwise (only if raise_error=False)

        Raises:
            EntityNotFoundError: If the node doesn't exist and raise_error=True
            ValueError: If the node_type is not in VALIDATABLE_NODE_TYPES
        """
        if node_type not in VALIDATABLE_NODE_TYPES:
            raise ValueError(
                f"Invalid node type '{node_type}'. "
                f"Must be one of: {', '.join(sorted(VALIDATABLE_NODE_TYPES))}"
            )

        cypher = f"MATCH (n:{node_type} {{uid: $uid}}) RETURN count(n) as count"
        results = await self.query(cypher, {"uid": uid})
        exists = bool(results and results[0].get("count", 0) > 0)

        if not exists and raise_error:
            raise EntityNotFoundError(node_type, uid)

        return exists

    # === Relationship Operations ===

    async def link_fragment_entity(
        self, fragment_uid: str, entity_uid: str, validate: bool = False
    ) -> bool:
        """Create a MENTIONS relationship between fragment and entity.

        Args:
            fragment_uid: The uid of the Fragment node
            entity_uid: The uid of the Entity node
            validate: If True, verify both nodes exist before creating relationship

        Returns:
            True if the relationship was created or already exists

        Raises:
            EntityNotFoundError: If validate=True and either node doesn't exist
        """
        if validate:
            await asyncio.gather(
                self._validate_node_exists("Fragment", fragment_uid),
                self._validate_node_exists("Entity", entity_uid),
            )

        cypher = """
        MATCH (f:Fragment {uid: $fragment_uid})
        MATCH (e:Entity {uid: $entity_uid})
        MERGE (f)-[r:MENTIONS]->(e)
        RETURN count(r) as count
        """

        results = await self.query(cypher, {
            "fragment_uid": fragment_uid,
            "entity_uid": entity_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    async def link_triple_to_fragment(
        self, triple_uid: str, fragment_uid: str, validate: bool = False
    ) -> bool:
        """Create a SOURCED_FROM relationship between triple and fragment.

        Args:
            triple_uid: The uid of the Triple node
            fragment_uid: The uid of the Fragment node
            validate: If True, verify both nodes exist before creating relationship

        Returns:
            True if the relationship was created or already exists

        Raises:
            EntityNotFoundError: If validate=True and either node doesn't exist
        """
        if validate:
            await asyncio.gather(
                self._validate_node_exists("Triple", triple_uid),
                self._validate_node_exists("Fragment", fragment_uid),
            )

        cypher = """
        MATCH (t:Triple {uid: $triple_uid})
        MATCH (f:Fragment {uid: $fragment_uid})
        MERGE (t)-[r:SOURCED_FROM]->(f)
        RETURN count(r) as count
        """

        results = await self.query(cypher, {
            "triple_uid": triple_uid,
            "fragment_uid": fragment_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    # === Query Operations ===

    async def get_related_fragments(
        self, entity_uid: str, limit: int = 10
    ) -> list["Fragment"]:
        """Get fragments that mention an entity."""
        from core.psyche.schema import Fragment, FragmentState

        cypher = """
        MATCH (f:Fragment)-[:MENTIONS]->(e:Entity {uid: $entity_uid})
        RETURN f.uid, f.content, f.source, f.state, f.resonance,
               f.confidence, f.created_at, f.last_accessed
        ORDER BY f.resonance DESC
        LIMIT $limit
        """

        results = await self.query(cypher, {"entity_uid": entity_uid, "limit": limit})
        return [
            Fragment(
                uid=r["f.uid"],
                content=r["f.content"],
                source=r["f.source"],
                state=FragmentState(r["f.state"]),
                resonance=r["f.resonance"],
                confidence=r["f.confidence"],
                created_at=datetime.fromisoformat(r["f.created_at"]),
                last_accessed=datetime.fromisoformat(r["f.last_accessed"]),
            )
            for r in results
        ]

    async def get_recent_fragments(self, limit: int = 10) -> list["Fragment"]:
        """Get most recently created fragments."""
        from core.psyche.schema import Fragment, FragmentState

        cypher = """
        MATCH (f:Fragment)
        RETURN f.uid, f.content, f.source, f.state, f.resonance,
               f.confidence, f.created_at, f.last_accessed
        ORDER BY f.created_at DESC
        LIMIT $limit
        """

        results = await self.query(cypher, {"limit": limit})
        if not results:
            return []

        # Debug: log first result keys if unexpected structure
        if results and "f.uid" not in results[0]:
            logger.warning(f"Unexpected result keys in get_recent_fragments: {list(results[0].keys())}")
            return []

        return [
            Fragment(
                uid=r["f.uid"],
                content=r["f.content"],
                source=r["f.source"],
                state=FragmentState(r["f.state"]),
                resonance=r["f.resonance"],
                confidence=r["f.confidence"],
                created_at=datetime.fromisoformat(r["f.created_at"]),
                last_accessed=datetime.fromisoformat(r["f.last_accessed"]),
            )
            for r in results
        ]

    async def get_newest_unprocessed(
        self,
        exclude_sources: Optional[list[str]] = None,
    ) -> Optional["Fragment"]:
        """Get the newest unprocessed (LIMBO state) fragment.

        Args:
            exclude_sources: List of sources to exclude (e.g., ['lilly_cognitive'])

        Returns:
            The newest unprocessed Fragment, or None if none found
        """
        from core.psyche.schema import Fragment, FragmentState

        # Build WHERE clause for source exclusion
        where_clause = "f.state = 'LIMBO'"
        params: dict = {}

        if exclude_sources:
            where_clause += " AND NOT f.source IN $exclude_sources"
            params["exclude_sources"] = exclude_sources

        cypher = f"""
        MATCH (f:Fragment)
        WHERE {where_clause}
        RETURN f.uid, f.content, f.source, f.state, f.resonance,
               f.confidence, f.created_at, f.last_accessed
        ORDER BY f.created_at DESC
        LIMIT 1
        """

        results = await self.query(cypher, params)
        if not results:
            return None

        r = results[0]
        if "f.uid" not in r:
            logger.warning(f"Unexpected result keys in get_newest_unprocessed: {list(r.keys())}")
            return None

        return Fragment(
            uid=r["f.uid"],
            content=r["f.content"],
            source=r["f.source"],
            state=FragmentState(r["f.state"]),
            resonance=r["f.resonance"],
            confidence=r["f.confidence"],
            created_at=datetime.fromisoformat(r["f.created_at"]),
            last_accessed=datetime.fromisoformat(r["f.last_accessed"]),
        )

    async def semantic_search(
        self, embedding: list[float], limit: int = 5
    ) -> list[tuple["Fragment", float]]:
        """
        Find fragments by embedding similarity using FalkorDB's native vector index.

        This method uses FalkorDB's built-in vector similarity search via the
        db.idx.vector.queryNodes procedure, which performs efficient KNN search
        using the HNSW algorithm.

        Dynamically selects the property to search based on query dimension:
        - 4096-dim queries (golden): use `embedding` property
        - 1024-dim queries (retrieval): use `embedding_retrieval` property

        This enables:
        - Full 4096-dim search at startup (before LLM loads)
        - Fast 1024-dim search at runtime (MRL-compatible)

        Args:
            embedding: Query vector to find similar fragments for.
            limit: Maximum number of results to return (default 5).

        Returns:
            List of (Fragment, score) tuples sorted by similarity descending.
            Score represents cosine similarity (higher is more similar).
        """
        from core.psyche.schema import Fragment, FragmentState

        # Determine which property to query based on embedding dimension
        query_dim = len(embedding)
        if query_dim >= 4096:
            property_name = "embedding"
            logger.debug(f"Using full embedding index for {query_dim}-dim query")
        else:
            property_name = "embedding_retrieval"
            logger.debug(f"Using retrieval embedding index for {query_dim}-dim query")

        # Convert embedding list to FalkorDB vector format
        embedding_str = ", ".join(str(x) for x in embedding)

        # Use FalkorDB's native vector similarity search
        # Return all Fragment fields to preserve metadata
        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'Fragment',
            '{property_name}',
            $limit,
            vecf32([{embedding_str}])
        ) YIELD node, score
        RETURN node.uid as uid,
               node.content as content,
               node.source as source,
               node.state as state,
               node.resonance as resonance,
               node.confidence as confidence,
               node.created_at as created_at,
               node.last_accessed as last_accessed,
               node.embedding as embedding,
               score
        """

        try:
            results = await self.query(cypher, {"limit": limit})
            if results:
                default_time = datetime.now(tz=timezone.utc)
                return [
                    (
                        Fragment(
                            uid=r["uid"],
                            content=r["content"],
                            source=r.get("source") or "unknown",
                            state=FragmentState(r["state"]) if r.get("state") else FragmentState.LIMBO,
                            resonance=r.get("resonance") if r.get("resonance") is not None else 0.5,
                            confidence=r.get("confidence") if r.get("confidence") is not None else 0.8,
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at", default_time),
                            last_accessed=_parse_datetime_field(r.get("last_accessed"), "last_accessed", default_time),
                            embedding=r.get("embedding"),
                        ),
                        r["score"],
                    )
                    for r in results
                ]
        except Exception as e:
            error_msg = str(e).lower()
            # Note: FalkorDB doesn't provide specific exception types for vector
            # operations (e.g., DimensionMismatchError, IndexNotFoundError).
            # String matching on error messages is the pragmatic approach here.
            # This may need updating if FalkorDB changes its error messages.
            if "dimension" in error_msg or "expected" in error_msg:
                logger.warning(
                    f"Vector dimension mismatch in semantic_search: {e}. "
                    "Query embeddings may not match stored embedding dimensions. "
                    "Returning empty results."
                )
                return []
            if "no such index" in error_msg or "index" in error_msg:
                logger.warning(
                    f"Vector index not found for {property_name}: {e}. "
                    "Create index with ensure_vector_index() or ensure_retrieval_vector_index()."
                )
                return []

            logger.warning(
                f"Vector index query failed, falling back to in-memory search: {e}"
            )

        # Fallback: in-memory similarity computation (less efficient)
        return await self._semantic_search_fallback(embedding, limit)

    async def _semantic_search_fallback(
        self, embedding: list[float], limit: int
    ) -> list[tuple["Fragment", float]]:
        """
        Fallback semantic search using in-memory cosine similarity.

        WARNING: This method loads all fragments with embeddings into memory
        and computes similarity in Python. It will not scale well and should
        only be used when the vector index is unavailable.

        For production use, ensure the vector index is created via
        ensure_vector_index().
        """
        from core.psyche.schema import Fragment, FragmentState
        import math

        # This fallback is expensive and may fail on schema mismatches.
        # Return empty list gracefully if it fails.
        try:
            # Return all Fragment fields to preserve metadata
            cypher = """
            MATCH (f:Fragment)
            WHERE f.embedding IS NOT NULL
            RETURN f.uid as uid,
                   f.content as content,
                   f.source as source,
                   f.state as state,
                   f.resonance as resonance,
                   f.confidence as confidence,
                   f.created_at as created_at,
                   f.last_accessed as last_accessed,
                   f.embedding as embedding
            LIMIT 1000
            """

            results = await self.query(cypher)
            if not results:
                return []

            def cosine_similarity(a: list[float], b: list[float]) -> float:
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)

            scored = []
            default_time = datetime.now(tz=timezone.utc)
            for r in results:
                frag_embedding = r.get("embedding")
                if frag_embedding and len(frag_embedding) == len(embedding):
                    score = cosine_similarity(embedding, frag_embedding)
                    # Use actual values from database, with sensible fallbacks
                    fragment = Fragment(
                        uid=r["uid"],
                        content=r["content"],
                        source=r.get("source") or "unknown",
                        state=FragmentState(r["state"]) if r.get("state") else FragmentState.LIMBO,
                        resonance=r.get("resonance") if r.get("resonance") is not None else 0.5,
                        confidence=r.get("confidence") if r.get("confidence") is not None else 0.8,
                        created_at=_parse_datetime_field(r.get("created_at"), "created_at", default_time),
                        last_accessed=_parse_datetime_field(r.get("last_accessed"), "last_accessed", default_time),
                        embedding=frag_embedding,
                    )
                    scored.append((fragment, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        except Exception as e:
            logger.warning(f"Fallback semantic search failed: {e}")
            return []

    async def count_nodes(self, label: str = "Fragment") -> int:
        """Count nodes of a given label."""
        cypher = f"MATCH (n:{label}) RETURN count(n) as count"
        results = await self.query(cypher)
        return results[0]["count"] if results else 0

    # === Schema and Index Operations ===

    async def ensure_indexes(self) -> dict[str, bool]:
        """
        Create indexes for efficient querying.

        Creates indexes on:
        - Fragment.uid, Fragment.source, Fragment.state
        - Triple.uid, Triple.subject, Triple.predicate
        - Entity.uid, Entity.name, Entity.entity_type
        - CommittedBelief.uid, CommittedBelief.tenant_id, CommittedBelief.topic
        - CommittedBelief.(tenant_id, created_at) [composite]
        - SteeringVector.name

        Supports both single-property indexes and composite indexes.
        For composite indexes, pass a tuple of property names as the second element:
            ("Label", ("prop1", "prop2"))

        Returns:
            Dict mapping index name to creation success
        """
        indexes: list[tuple[str, str | tuple[str, ...]]] = [
            ("Fragment", "uid"),
            ("Fragment", "source"),
            ("Fragment", "state"),
            ("Triple", "uid"),
            ("Triple", "subject"),
            ("Triple", "predicate"),
            ("Triple", "object_"),  # Needed for efficient degree queries
            ("Entity", "uid"),
            ("Entity", "name"),
            ("Entity", "entity_type"),
            # Active Inference indexes (issue #016)
            ("CommittedBelief", "uid"),
            ("CommittedBelief", "tenant_id"),
            ("CommittedBelief", "topic"),
            # Composite index for efficient tenant + time range queries
            ("CommittedBelief", ("tenant_id", "created_at")),
            ("SteeringVector", "name"),
        ]

        results = {}
        for label, props in indexes:
            if isinstance(props, tuple):
                # Handle composite index
                prop_list = ", ".join(f"n.{p}" for p in props)
                index_name = f"idx_{label}_{'_'.join(props)}"
                cypher = f"CREATE INDEX FOR (n:{label}) ON ({prop_list})"
            else:
                # Handle single property index
                index_name = f"idx_{label}_{props}"
                cypher = f"CREATE INDEX FOR (n:{label}) ON (n.{props})"

            try:
                await self.execute(cypher)
                results[index_name] = True
                logger.info(f"Created index: {index_name}")
            except Exception as e:
                # Index might already exist
                if "already exists" in str(e).lower():
                    results[index_name] = True
                    logger.debug(f"Index already exists: {index_name}")
                else:
                    results[index_name] = False
                    logger.warning(f"Failed to create index {index_name}: {e}")

        # Also ensure InsightZettel vector index for zettel retrieval
        # Uses 1024-dim (retrieval tier) since store_zettel uses EmbeddingTier.RETRIEVAL
        zettel_idx_success = await self.ensure_zettel_vector_index(dimension=1024)
        results["zettel_embedding_idx"] = zettel_idx_success

        # Ensure ResearchQueryResult vector index for research answer retrieval
        research_idx_success = await self.ensure_research_query_vector_index(dimension=1024)
        results["research_query_embedding_idx"] = research_idx_success

        return results

    async def ensure_vector_index(
        self,
        dimension: int = 4096,
        similarity_function: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
    ) -> bool:
        """
        Create a vector index on Fragment.embedding for efficient similarity search.

        This enables FalkorDB's native vector similarity search using the HNSW
        algorithm, which is essential for scalable semantic_search operations.

        Args:
            dimension: Dimensionality of the embedding vectors (default 4096 for
                Qwen3-Embedding-8B golden embeddings).
            similarity_function: Distance metric - "cosine" or "euclidean"
                (default "cosine").
            m: HNSW parameter controlling the number of bi-directional links
                created for each element (higher = better recall, more memory).
            ef_construction: HNSW parameter controlling index build quality
                (higher = better quality, slower build).

        Returns:
            True if index was created or already exists, False on failure.

        Note:
            This method should be called once during database initialization,
            typically after ensure_indexes(). The index creation is idempotent.
        """
        # FalkorDB vector index creation with HNSW parameters
        cypher = f"""
        CREATE VECTOR INDEX FOR (f:Fragment) ON (f.embedding)
        OPTIONS {{
            dimension: {dimension},
            similarityFunction: '{similarity_function}',
            M: {m},
            efConstruction: {ef_construction}
        }}
        """

        try:
            await self.execute(cypher)
            logger.info(
                f"Created vector index on Fragment.embedding "
                f"(dim={dimension}, similarity={similarity_function})"
            )
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "already indexed" in error_msg:
                logger.debug("Vector index on Fragment.embedding already exists")
                return True
            else:
                logger.warning(f"Failed to create vector index: {e}")
                return False

    async def ensure_retrieval_vector_index(
        self,
        dimension: int = 1024,
        similarity_function: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
    ) -> bool:
        """
        Create a vector index on Fragment.embedding_retrieval for runtime search.

        This index enables fast 1024-dim semantic search using MRL-truncated
        embeddings while the main LLM occupies GPU memory.

        Args:
            dimension: Dimensionality of the retrieval vectors (default 1024).
            similarity_function: Distance metric - "cosine" or "euclidean".
            m: HNSW parameter for bi-directional links per element.
            ef_construction: HNSW parameter for index build quality.

        Returns:
            True if index was created or already exists, False on failure.
        """
        cypher = f"""
        CREATE VECTOR INDEX FOR (f:Fragment) ON (f.embedding_retrieval)
        OPTIONS {{
            dimension: {dimension},
            similarityFunction: '{similarity_function}',
            M: {m},
            efConstruction: {ef_construction}
        }}
        """

        try:
            await self.execute(cypher)
            logger.info(
                f"Created vector index on Fragment.embedding_retrieval "
                f"(dim={dimension}, similarity={similarity_function})"
            )
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "already indexed" in error_msg:
                logger.debug("Vector index on Fragment.embedding_retrieval already exists")
                return True
            else:
                logger.warning(f"Failed to create retrieval vector index: {e}")
                return False

    async def drop_vector_index(self) -> bool:
        """
        Drop the vector index on Fragment.embedding.

        Use this before calling ensure_vector_index() with a new dimension
        if the index was created with a different dimension.

        Returns:
            True if index was dropped or didn't exist, False on failure.
        """
        # Note: FalkorDB syntax mirrors CREATE syntax
        cypher = "DROP VECTOR INDEX FOR (f:Fragment) ON (f.embedding)"

        try:
            await self.execute(cypher)
            logger.info("Dropped vector index on Fragment.embedding")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "no such index" in error_msg or "does not exist" in error_msg:
                logger.debug("Vector index on Fragment.embedding does not exist")
                return True
            else:
                logger.warning(f"Failed to drop vector index: {e}")
                return False

    async def recreate_vector_index(
        self,
        dimension: int = 4096,
        similarity_function: str = "cosine",
    ) -> bool:
        """
        Drop and recreate the vector index with the specified dimension.

        Use this when the embedding dimension has changed.

        Args:
            dimension: Dimensionality of the embedding vectors.
            similarity_function: Distance metric - "cosine" or "euclidean".

        Returns:
            True if index was successfully recreated, False on failure.
        """
        dropped = await self.drop_vector_index()
        if not dropped:
            return False

        return await self.ensure_vector_index(
            dimension=dimension,
            similarity_function=similarity_function,
        )

    async def get_schema_stats(self) -> dict[str, Any]:
        """
        Get statistics about the graph schema.

        Returns:
            Dict with node counts, relationship counts, etc.
        """
        stats = {}

        # Count nodes by label
        for label in ["Fragment", "Triple", "Entity", "SteeringVector"]:
            try:
                count = await self.count_nodes(label)
                stats[f"{label.lower()}_count"] = count
            except Exception:
                stats[f"{label.lower()}_count"] = 0

        # Count relationships
        try:
            cypher = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
            results = await self.query(cypher)
            stats["relationships"] = {r["type"]: r["count"] for r in results}
        except Exception:
            stats["relationships"] = {}

        return stats

    # === Steering Vector Operations ===

    async def upsert_steering_vector(self, data: dict) -> dict:
        """
        Create or update a SteeringVector node in the graph.

        This method is used by the SIMS Executor to persist steering vector
        adjustments and by PlutchikExtractor to store Plutchik emotion vectors.
        It uses MERGE to create the node if it doesn't exist or update it if it does.

        Args:
            data: Dict containing steering vector data:
                - name (required): Unique name for the vector
                - adjustment_type: Type of adjustment (strengthen, weaken, add, remove)
                - magnitude: Current magnitude/coefficient of the vector
                - reason: Reason for the adjustment
                - timestamp: ISO format timestamp of the adjustment
                - vector_data: List of floats representing the steering direction
                - layer: Target layer for the vector
                - coefficient: Scaling coefficient for the vector
                - active: Whether the vector is currently active
                - pairs_hash: Hash of contrastive pairs (for cache invalidation)
                - emotion_type: Type of emotion (e.g., "plutchik_primary")

        Returns:
            Dict with the created/updated node properties including uid.

        Raises:
            RuntimeError: If not connected to FalkorDB.
        """
        name = data.get("name")
        if not name:
            raise ValueError("SteeringVector name is required")

        # Generate a deterministic UID based on name for consistent upserts
        uid = f"sv_{name}"

        adjustment_type = data.get("adjustment_type", "unknown")
        magnitude = data.get("magnitude", 1.0)
        reason = data.get("reason", "")
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())

        # Plutchik-specific fields
        vector_data = data.get("vector_data")  # List of floats
        layer = data.get("layer")  # Target layer
        coefficient = data.get("coefficient", 1.0)
        active = data.get("active", True)
        pairs_hash = data.get("pairs_hash", "")
        emotion_type = data.get("emotion_type", "")

        # Serialize vector_data as JSON string for storage
        vector_data_json = json.dumps(vector_data) if vector_data else None

        # Use MERGE to create or update the node
        cypher = """
        MERGE (sv:SteeringVector {name: $name})
        ON CREATE SET
            sv.uid = $uid,
            sv.adjustment_type = $adjustment_type,
            sv.magnitude = $magnitude,
            sv.reason = $reason,
            sv.vector_data = $vector_data,
            sv.layer = $layer,
            sv.coefficient = $coefficient,
            sv.active = $active,
            sv.pairs_hash = $pairs_hash,
            sv.emotion_type = $emotion_type,
            sv.created_at = $timestamp,
            sv.updated_at = $timestamp
        ON MATCH SET
            sv.adjustment_type = $adjustment_type,
            sv.magnitude = $magnitude,
            sv.reason = $reason,
            sv.vector_data = $vector_data,
            sv.layer = $layer,
            sv.coefficient = $coefficient,
            sv.active = $active,
            sv.pairs_hash = $pairs_hash,
            sv.emotion_type = $emotion_type,
            sv.updated_at = $timestamp
        RETURN sv.uid as uid, sv.name as name, sv.adjustment_type as adjustment_type,
               sv.magnitude as magnitude, sv.reason as reason,
               sv.vector_data as vector_data, sv.layer as layer,
               sv.coefficient as coefficient, sv.active as active,
               sv.pairs_hash as pairs_hash, sv.emotion_type as emotion_type,
               sv.created_at as created_at, sv.updated_at as updated_at
        """

        params = {
            "uid": uid,
            "name": name,
            "adjustment_type": adjustment_type,
            "magnitude": magnitude,
            "reason": reason,
            "vector_data": vector_data_json,
            "layer": layer,
            "coefficient": coefficient,
            "active": active,
            "pairs_hash": pairs_hash,
            "emotion_type": emotion_type,
            "timestamp": timestamp,
        }

        results = await self.query(cypher, params)
        if results:
            logger.debug(f"Upserted SteeringVector: {name}")
            return results[0]

        # Fallback return if query returns empty (shouldn't happen with MERGE)
        return {
            "uid": uid,
            "name": name,
            "adjustment_type": adjustment_type,
            "magnitude": magnitude,
            "reason": reason,
            "vector_data": vector_data_json,
            "layer": layer,
            "coefficient": coefficient,
            "active": active,
            "pairs_hash": pairs_hash,
            "emotion_type": emotion_type,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    async def get_steering_vector(self, name: str) -> Optional[dict]:
        """
        Retrieve a SteeringVector by name.

        Args:
            name: The name of the steering vector.

        Returns:
            Dict with vector properties or None if not found.
            If vector_data exists, it will be parsed from JSON string to list.
        """
        cypher = """
        MATCH (sv:SteeringVector {name: $name})
        RETURN sv.uid as uid, sv.name as name, sv.adjustment_type as adjustment_type,
               sv.magnitude as magnitude, sv.reason as reason,
               sv.vector_data as vector_data, sv.layer as layer,
               sv.coefficient as coefficient, sv.active as active,
               sv.pairs_hash as pairs_hash, sv.emotion_type as emotion_type,
               sv.created_at as created_at, sv.updated_at as updated_at
        """

        results = await self.query(cypher, {"name": name})
        if results:
            result = results[0]
            # Parse vector_data from JSON string if present
            if result.get("vector_data"):
                try:
                    result["vector_data"] = json.loads(result["vector_data"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse vector_data for steering vector '{name}': {e}")
                    pass  # Keep as-is if already a list or invalid
            return result
        return None

    async def list_steering_vectors(self, limit: int = 50) -> list[dict]:
        """
        List all steering vectors.

        Args:
            limit: Maximum number of vectors to return.

        Returns:
            List of dicts with vector properties.
        """
        cypher = """
        MATCH (sv:SteeringVector)
        RETURN sv.uid as uid, sv.name as name, sv.adjustment_type as adjustment_type,
               sv.magnitude as magnitude, sv.reason as reason,
               sv.created_at as created_at, sv.updated_at as updated_at
        ORDER BY sv.updated_at DESC
        LIMIT $limit
        """

        return await self.query(cypher, {"limit": limit})

    # === Graph Topology Operations ===

    async def get_top_hubs(
        self,
        tenant_id: str,
        min_degree: int = 3,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get top hub nodes by degree centrality.

        Hub nodes are central concepts that stabilize the knowledge web.
        They connect multiple blooms and are critical for memory retention.

        Args:
            tenant_id: Tenant identifier.
            min_degree: Minimum total degree to qualify as a hub.
            limit: Maximum number of hubs to return.

        Returns:
            List of dicts with uid, degree, in_degree, out_degree, title, content.
        """
        cypher = """
            MATCH (n {tenant_id: $tenant_id})
            WHERE n:Zettel OR n:Fragment
            WITH n,
                 SIZE([(n)-[]->() | 1]) AS out_deg,
                 SIZE([()-[]->(n) | 1]) AS in_deg
            WITH n, out_deg, in_deg, out_deg + in_deg AS total_degree
            WHERE total_degree >= $min_degree
            RETURN n.uid AS uid,
                   n.title AS title,
                   SUBSTRING(n.content, 0, 200) AS content_preview,
                   out_deg AS out_degree,
                   in_deg AS in_degree,
                   total_degree AS degree
            ORDER BY total_degree DESC
            LIMIT $limit
        """

        try:
            results = await self.query(cypher, {
                "tenant_id": tenant_id,
                "min_degree": min_degree,
                "limit": limit,
            })

            return [
                {
                    "uid": row.get("uid"),
                    "title": row.get("title"),
                    "content_preview": row.get("content_preview"),
                    "degree": row.get("degree", 0),
                    "in_degree": row.get("in_degree", 0),
                    "out_degree": row.get("out_degree", 0),
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to get top hubs: {e}")
            return []

    # === Prompt Component Operations ===

    async def create_prompt_component(
        self,
        component: dict,
    ) -> bool:
        """
        Create a PromptComponent node in the graph.

        Args:
            component: Dict with PromptComponent data including:
                - uid, component_type, content, state, layer
                - version, supersedes_uid (optional)
                - thesis, antithesis, synthesis, synthesis_reasoning
                - origin, source_uid (optional)
                - confidence, usage_count
                - created_at, modified_at

        Returns:
            True if created successfully.
        """
        cypher = """
        CREATE (p:PromptComponent {
            uid: $uid,
            component_type: $component_type,
            content: $content,
            state: $state,
            layer: $layer,
            version: $version,
            supersedes_uid: $supersedes_uid,
            thesis: $thesis,
            antithesis: $antithesis,
            synthesis: $synthesis,
            synthesis_reasoning: $synthesis_reasoning,
            origin: $origin,
            source_uid: $source_uid,
            confidence: $confidence,
            usage_count: $usage_count,
            created_at: $created_at,
            modified_at: $modified_at
        })
        """
        params = {
            "uid": _sanitize_string(component.get("uid", "")),
            "component_type": component.get("component_type", "instruction"),
            "content": _sanitize_string(component.get("content", "")),
            "state": component.get("state", "active"),
            "layer": component.get("layer", 5),
            "version": component.get("version", 1),
            "supersedes_uid": component.get("supersedes_uid"),
            "thesis": _sanitize_string(component.get("thesis", "")),
            "antithesis": _sanitize_string(component.get("antithesis", "")),
            "synthesis": _sanitize_string(component.get("synthesis", "")),
            "synthesis_reasoning": _sanitize_string(
                component.get("synthesis_reasoning", "")
            ),
            "origin": component.get("origin", "inherited"),
            "source_uid": component.get("source_uid"),
            "confidence": component.get("confidence", 0.8),
            "usage_count": component.get("usage_count", 0),
            "created_at": component.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "modified_at": component.get(
                "modified_at", datetime.now(timezone.utc).isoformat()
            ),
        }

        affected = await self.execute(cypher, params)
        return affected > 0

    async def get_prompt_component(self, uid: str) -> Optional[dict]:
        """Get a PromptComponent by UID."""
        cypher = """
        MATCH (p:PromptComponent {uid: $uid})
        RETURN p.uid as uid, p.component_type as component_type,
               p.content as content, p.state as state, p.layer as layer,
               p.version as version, p.supersedes_uid as supersedes_uid,
               p.thesis as thesis, p.antithesis as antithesis,
               p.synthesis as synthesis, p.synthesis_reasoning as synthesis_reasoning,
               p.origin as origin, p.source_uid as source_uid,
               p.confidence as confidence, p.usage_count as usage_count,
               p.created_at as created_at, p.modified_at as modified_at
        """
        results = await self.query(cypher, {"uid": uid})
        return results[0] if results else None

    async def get_active_prompt_components(self) -> list[dict]:
        """Get all active PromptComponents ordered by layer."""
        cypher = """
        MATCH (p:PromptComponent {state: 'active'})
        RETURN p.uid as uid, p.component_type as component_type,
               p.content as content, p.state as state, p.layer as layer,
               p.version as version, p.supersedes_uid as supersedes_uid,
               p.thesis as thesis, p.antithesis as antithesis,
               p.synthesis as synthesis, p.synthesis_reasoning as synthesis_reasoning,
               p.origin as origin, p.source_uid as source_uid,
               p.confidence as confidence, p.usage_count as usage_count,
               p.created_at as created_at, p.modified_at as modified_at
        ORDER BY p.layer ASC, p.created_at ASC
        """
        return await self.query(cypher, {})

    async def get_prompt_component_history(self, uid: str) -> list[dict]:
        """Get version history for a PromptComponent by following supersedes chain."""
        cypher = """
        MATCH (p:PromptComponent {uid: $uid})
        OPTIONAL MATCH path = (p)-[:SUPERSEDES*0..]->(older:PromptComponent)
        WITH older ORDER BY older.version DESC
        RETURN older.uid as uid, older.component_type as component_type,
               older.content as content, older.state as state, older.layer as layer,
               older.version as version, older.supersedes_uid as supersedes_uid,
               older.thesis as thesis, older.antithesis as antithesis,
               older.synthesis as synthesis, older.synthesis_reasoning as synthesis_reasoning,
               older.origin as origin, older.confidence as confidence,
               older.created_at as created_at
        """
        return await self.query(cypher, {"uid": uid})

    async def update_prompt_component_state(
        self,
        uid: str,
        state: str,
        modified_at: Optional[str] = None,
    ) -> bool:
        """Update the state of a PromptComponent."""
        cypher = """
        MATCH (p:PromptComponent {uid: $uid})
        SET p.state = $state,
            p.modified_at = $modified_at
        RETURN p.uid
        """
        params = {
            "uid": uid,
            "state": state,
            "modified_at": modified_at or datetime.now(timezone.utc).isoformat(),
        }
        results = await self.query(cypher, params)
        return len(results) > 0

    async def increment_prompt_usage(self, uid: str) -> bool:
        """Increment usage count for a PromptComponent."""
        cypher = """
        MATCH (p:PromptComponent {uid: $uid})
        SET p.usage_count = p.usage_count + 1,
            p.modified_at = $modified_at
        RETURN p.uid
        """
        results = await self.query(
            cypher,
            {"uid": uid, "modified_at": datetime.now(timezone.utc).isoformat()},
        )
        return len(results) > 0

    async def create_prompt_reflection(self, reflection: dict) -> bool:
        """
        Create a PromptReflection node in the graph.

        Args:
            reflection: Dict with PromptReflection data.

        Returns:
            True if created successfully.
        """
        cypher = """
        CREATE (r:PromptReflection {
            uid: $uid,
            component_uid: $component_uid,
            reflection_type: $reflection_type,
            content: $content,
            valence: $valence,
            resonance_score: $resonance_score,
            action_taken: $action_taken,
            cycle_type: $cycle_type,
            created_at: $created_at
        })
        """
        params = {
            "uid": _sanitize_string(reflection.get("uid", "")),
            "component_uid": reflection.get("component_uid", ""),
            "reflection_type": reflection.get("reflection_type", "resonance"),
            "content": _sanitize_string(reflection.get("content", "")),
            "valence": reflection.get("valence", 0.0),
            "resonance_score": reflection.get("resonance_score", 0.5),
            "action_taken": reflection.get("action_taken"),
            "cycle_type": reflection.get("cycle_type", "nap"),
            "created_at": reflection.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
        }

        affected = await self.execute(cypher, params)
        return affected > 0

    async def get_reflections_for_component(
        self,
        component_uid: str,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent reflections for a PromptComponent."""
        cypher = """
        MATCH (r:PromptReflection {component_uid: $component_uid})
        RETURN r.uid as uid, r.component_uid as component_uid,
               r.reflection_type as reflection_type, r.content as content,
               r.valence as valence, r.resonance_score as resonance_score,
               r.action_taken as action_taken, r.cycle_type as cycle_type,
               r.created_at as created_at
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        return await self.query(cypher, {"component_uid": component_uid, "limit": limit})

    async def link_prompt_supersession(
        self,
        new_uid: str,
        old_uid: str,
    ) -> bool:
        """Create SUPERSEDES relationship between prompt component versions."""
        cypher = """
        MATCH (new:PromptComponent {uid: $new_uid})
        MATCH (old:PromptComponent {uid: $old_uid})
        MERGE (new)-[:SUPERSEDES]->(old)
        RETURN new.uid
        """
        results = await self.query(cypher, {"new_uid": new_uid, "old_uid": old_uid})
        return len(results) > 0

    # === Coactivation Edge Operations ===

    async def upsert_coactivation_edge(
        self,
        entity_a: str,
        entity_b: str,
        strength: float,
    ) -> bool:
        """Create or update weighted edge between entities.

        Uses atomic MERGE to avoid race conditions. Stores latest weight,
        increments observation count, and tracks importance using EMA.

        Importance formula (EMA with recency decay):
        - ON CREATE: importance = strength
        - ON MATCH: importance = importance * 0.95 + strength * 0.5

        This means frequently co-activated edges accumulate importance,
        but with decay to favor recent activations.

        Args:
            entity_a: First entity name
            entity_b: Second entity name
            strength: Coactivation strength from this observation

        Returns:
            True if edge was created/updated
        """
        # Use MERGE for atomic upsert - no race condition
        # The undirected pattern (a)-[r:COACTIVATED]-(b) ensures
        # the same edge is matched regardless of query direction
        cypher = """
        MATCH (a:Entity {name: $a}), (b:Entity {name: $b})
        MERGE (a)-[r:COACTIVATED]-(b)
        ON CREATE SET
            r.weight = $strength,
            r.observations = 1,
            r.importance = $strength,
            r.created_at = $now,
            r.updated_at = $now
        ON MATCH SET
            r.weight = $strength,
            r.observations = r.observations + 1,
            r.importance = r.importance * $decay + $strength * $learning_rate,
            r.updated_at = $now
        """
        affected = await self.execute(cypher, {
            "a": _sanitize_string(entity_a),
            "b": _sanitize_string(entity_b),
            "strength": strength,
            "decay": _IMPORTANCE_EMA_DECAY,
            "learning_rate": _IMPORTANCE_EMA_LEARNING_RATE,
            "now": datetime.now(timezone.utc).isoformat(),
        })
        return affected > 0

    async def get_coactivation_edge(
        self,
        entity_a: str,
        entity_b: str,
    ) -> Optional[dict]:
        """Get edge details between two entities.

        Args:
            entity_a: First entity name
            entity_b: Second entity name

        Returns:
            Dict with edge properties (weight, observations, importance,
            created_at, updated_at) or None if no edge exists
        """
        # Undirected pattern matches edge regardless of direction
        results = await self.query("""
            MATCH (a:Entity {name: $a})-[r:COACTIVATED]-(b:Entity {name: $b})
            RETURN r.weight as weight, r.observations as observations,
                   r.importance as importance,
                   r.created_at as created_at, r.updated_at as updated_at
        """, {"a": _sanitize_string(entity_a), "b": _sanitize_string(entity_b)})
        return results[0] if results else None

    async def get_entity_neighborhood(
        self,
        entity_name: str,
        limit: int = 10,
        edge_types: list[str] | None = None,
        order_by: str = "weight",
    ) -> dict:
        """Get local neighborhood around an entity.

        Retrieves entities connected to the center entity via specified edge types,
        ordered by the specified criterion.

        Args:
            entity_name: Center entity name
            limit: Max neighbors to return
            edge_types: Filter to specific edge types (default: COACTIVATED, CO_MENTIONED)
            order_by: Ordering criterion for neighbors:
                - "weight": Order by edge weight (default)
                - "importance": Order by importance (frequency + recency)
                - "observations": Order by observation count

        Returns:
            dict with "nodes" and "edges" lists:
            - nodes: List of entity dicts with name, entity_type, salience, is_center
            - edges: List of edge dicts with source, target, type, weight, importance
        """
        edge_types = edge_types or ["COACTIVATED", "CO_MENTIONED"]
        edge_pattern = "|".join(edge_types)

        # Map order_by to actual property expression
        order_prop = {
            "weight": "r.weight",
            "importance": "COALESCE(r.importance, r.weight)",
            "observations": "COALESCE(r.observations, 1)",
        }.get(order_by, "r.weight")

        # Query neighbors with ordering
        # Note: FalkorDB doesn't support parameterized edge types, so we use f-string
        # for the edge pattern (which is safe since we control the allowed values)
        query = f"""
        MATCH (center:Entity {{name: $name}})-[r:{edge_pattern}]-(neighbor:Entity)
        RETURN neighbor.name AS name,
               neighbor.entity_type AS entity_type,
               neighbor.salience AS salience,
               type(r) AS edge_type,
               r.weight AS weight,
               COALESCE(r.importance, r.weight) AS importance,
               COALESCE(r.observations, 1) AS observations
        ORDER BY {order_prop} DESC
        LIMIT $limit
        """

        results = await self.query(query, {
            "name": _sanitize_string(entity_name),
            "limit": limit,
        })

        # Fetch center node's actual properties
        center_query = """
        MATCH (center:Entity {name: $name})
        RETURN COALESCE(center.entity_type, 'concept') AS entity_type,
               COALESCE(center.salience, 0.5) AS salience
        """
        center_results = await self.query(center_query, {
            "name": _sanitize_string(entity_name),
        })

        # Extract center node properties (use defaults if not found)
        if center_results:
            center_entity_type = center_results[0].get("entity_type", "concept")
            center_salience = center_results[0].get("salience", 0.5)
        else:
            center_entity_type = "concept"
            center_salience = 0.5

        # Build response structure - center node first
        nodes = [{
            "name": entity_name,
            "entity_type": center_entity_type,
            "salience": center_salience,
            "is_center": True,
        }]
        edges = []

        for row in results:
            nodes.append({
                "name": row["name"],
                "entity_type": row.get("entity_type", "concept"),
                "salience": row.get("salience", 0.5),
                "is_center": False,
            })
            edges.append({
                "source": entity_name,
                "target": row["name"],
                "type": row["edge_type"],
                "weight": row.get("weight", 1.0),
                "importance": row.get("importance", row.get("weight", 1.0)),
            })

        return {"nodes": nodes, "edges": edges}


    # ================================================================
    # SAE Feature Snapshot & Evocation Methods
    # ================================================================

    async def store_sae_snapshot(
        self,
        fragment_uid: str,
        features: list[tuple[int, float]],
        cycle: int,
    ) -> bool:
        """Store SAE feature snapshot linked to a thought fragment.

        Creates a SAEFeatureSnapshot node and links it to the fragment via
        GENERATED_WITH relationship. This captures the internal activation
        state that produced the thought.

        Args:
            fragment_uid: UID of the thought fragment
            features: List of (feature_idx, activation) tuples (top 10)
            cycle: Cognitive cycle number

        Returns:
            True if snapshot was stored successfully
        """
        # Convert features list to JSON-compatible format
        features_data = [[idx, activation] for idx, activation in features]

        cypher = """
        MATCH (f:Fragment {uid: $fragment_uid})
        CREATE (s:SAEFeatureSnapshot {
            fragment_uid: $fragment_uid,
            features: $features,
            cycle: $cycle,
            created_at: $now
        })
        CREATE (f)-[:GENERATED_WITH]->(s)
        """
        try:
            affected = await self.execute(cypher, {
                "fragment_uid": fragment_uid,
                "features": features_data,
                "cycle": cycle,
                "now": datetime.now(timezone.utc).isoformat(),
            })
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to store SAE snapshot for {fragment_uid}: {e}")
            return False

    async def get_sae_snapshot_for_fragment(
        self,
        fragment_uid: str,
    ) -> Optional[list[tuple[int, float]]]:
        """Retrieve SAE features that were active when fragment was generated.

        Args:
            fragment_uid: UID of the thought fragment

        Returns:
            List of (feature_idx, activation) tuples or None if no snapshot
        """
        results = await self.query("""
            MATCH (f:Fragment {uid: $fragment_uid})-[:GENERATED_WITH]->(s:SAEFeatureSnapshot)
            RETURN s.features as features
        """, {"fragment_uid": fragment_uid})

        if not results or not results[0].get("features"):
            return None

        # Convert back to list of tuples
        features = results[0]["features"]
        return [(int(f[0]), float(f[1])) for f in features]

    async def upsert_evocation_edge(
        self,
        feature_idx: int,
        entity_uid: str,
        weight: float,
        observation_count: int,
    ) -> bool:
        """Create/update EVOKES edge between SAE feature and Entity.

        Uses atomic MERGE to avoid race conditions. The weight should be
        pre-computed via EMA by the caller.

        Args:
            feature_idx: SAE feature index
            entity_uid: Entity UID
            weight: EMA-updated weight
            observation_count: Total observations

        Returns:
            True if edge was created/updated
        """
        cypher = """
        MERGE (f:SAEFeature {index: $feature_idx})
        ON CREATE SET f.created_at = $now
        WITH f
        MATCH (e:Entity {uid: $entity_uid})
        MERGE (f)-[r:EVOKES]->(e)
        ON CREATE SET
            r.weight = $weight,
            r.observation_count = $count,
            r.created_at = $now,
            r.updated_at = $now
        ON MATCH SET
            r.weight = $weight,
            r.observation_count = $count,
            r.updated_at = $now
        """
        try:
            affected = await self.execute(cypher, {
                "feature_idx": feature_idx,
                "entity_uid": _sanitize_string(entity_uid),
                "weight": weight,
                "count": observation_count,
                "now": datetime.now(timezone.utc).isoformat(),
            })
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to upsert evocation edge {feature_idx}->{entity_uid}: {e}")
            return False

    async def get_evoked_entities(
        self,
        feature_indices: list[int],
        min_weight: float = 0.05,
        limit: int = 10,
    ) -> list[dict]:
        """Query entities evoked by active features, summing weights.

        For each entity evoked by any of the given features, sum the weighted
        contributions and return the top entities by total evocation strength.

        Args:
            feature_indices: List of active SAE feature indices
            min_weight: Minimum edge weight to consider
            limit: Maximum entities to return

        Returns:
            List of dicts with entity info and total evocation weight:
            [{"uid": "...", "name": "...", "description": "...", "total_weight": 0.5}, ...]
        """
        if not feature_indices:
            return []

        cypher = """
        MATCH (f:SAEFeature)-[r:EVOKES]->(e:Entity)
        WHERE f.index IN $features AND r.weight >= $min_weight
        WITH e, SUM(r.weight) as total_weight
        RETURN e.uid as uid, e.name as name, e.description as description,
               e.entity_type as entity_type, total_weight
        ORDER BY total_weight DESC
        LIMIT $limit
        """
        results = await self.query(cypher, {
            "features": feature_indices,
            "min_weight": min_weight,
            "limit": limit,
        })
        return results

    async def get_unlinked_thought_fragments(
        self,
        limit: int = 50,
    ) -> list[tuple[str, list[tuple[int, float]]]]:
        """Get fragments with SAE snapshots but no EVOKES edges yet.

        Used by background processing to link SAE features to entities
        after HippoRAG has extracted entities from the thoughts.

        Args:
            limit: Maximum fragments to return

        Returns:
            List of (fragment_uid, features) tuples for processing
        """
        cypher = """
        MATCH (f:Fragment)-[:GENERATED_WITH]->(s:SAEFeatureSnapshot)
        WHERE f.source = 'lilly_cognitive'
          AND NOT EXISTS {
            MATCH (e:Entity)<-[:MENTIONS]-(f)
            MATCH (sae:SAEFeature)-[:EVOKES]->(e)
            WHERE sae.index IN [x IN s.features | x[0]]
          }
        RETURN f.uid as uid, s.features as features
        ORDER BY s.created_at DESC
        LIMIT $limit
        """
        results = await self.query(cypher, {"limit": limit})

        return [
            (r["uid"], [(int(f[0]), float(f[1])) for f in r["features"]])
            for r in results
            if r.get("features")
        ]

    async def get_entities_for_fragment(
        self,
        fragment_uid: str,
    ) -> list[dict]:
        """Get entities mentioned in or extracted from a fragment.

        Used to find entities that should be linked to SAE features
        from the fragment's snapshot.

        Args:
            fragment_uid: UID of the fragment

        Returns:
            List of entity dicts: [{"uid": "...", "name": "...", "entity_type": "..."}, ...]
        """
        cypher = """
        MATCH (f:Fragment {uid: $uid})-[:MENTIONS]->(e:Entity)
        RETURN e.uid as uid, e.name as name, e.entity_type as entity_type
        """
        results = await self.query(cypher, {"uid": fragment_uid})
        return results

    # ================================================================
    # InsightZettel CRUD Operations
    # ================================================================

    async def create_zettel(self, zettel: "InsightZettel") -> bool:
        """Create an InsightZettel node in the graph.

        Also creates a vector index entry if embedding is provided.
        Uses vecf32() function for vector storage (same pattern as Fragment embeddings).

        Args:
            zettel: InsightZettel model instance

        Returns:
            True if created successfully
        """
        # Create node without embedding first, then set vector property separately
        # This avoids f-string interpolation of the embedding vector
        cypher_create = """
        CREATE (z:InsightZettel {
            uid: $uid,
            insight_text: $insight_text,
            question_text: $question_text,
            question_status: $question_status,
            source_type: $source_type,
            source_uid: $source_uid,
            concept: $concept,
            cycle: $cycle,
            sae_feature_indices: $sae_feature_indices,
            created_at: $created_at,
            novelty_score: $novelty_score,
            is_refinement: $is_refinement,
            refines_uid: $refines_uid
        })
        """
        params = {
            "uid": _sanitize_string(zettel.uid),
            "insight_text": _sanitize_string(zettel.insight_text),
            "question_text": _sanitize_string(zettel.question_text) if zettel.question_text else None,
            "question_status": zettel.question_status.value,
            "source_type": zettel.source_type.value,
            "source_uid": _sanitize_string(zettel.source_uid),
            "concept": _sanitize_string(zettel.concept),
            "cycle": zettel.cycle,
            "sae_feature_indices": zettel.sae_feature_indices,
            "created_at": zettel.created_at.isoformat(),
            "novelty_score": zettel.novelty_score,
            "is_refinement": zettel.is_refinement,
            "refines_uid": _sanitize_string(zettel.refines_uid) if zettel.refines_uid else None,
        }

        try:
            affected = await self.execute(cypher_create, params)
            if affected <= 0:
                return False

            # Set embedding using vecf32() function (same pattern as Fragment embeddings)
            if zettel.embedding is not None:
                embedding_str = ", ".join(str(x) for x in zettel.embedding)
                cypher_embedding = f"""
                MATCH (z:InsightZettel {{uid: $uid}})
                SET z.embedding = vecf32([{embedding_str}])
                """
                await self.execute(cypher_embedding, {
                    "uid": _sanitize_string(zettel.uid),
                })

            return True
        except Exception as e:
            logger.warning(f"Failed to create InsightZettel {zettel.uid}: {e}")
            return False

    async def get_zettel(self, uid: str) -> Optional["InsightZettel"]:
        """Retrieve an InsightZettel by UID.

        Args:
            uid: Zettel unique identifier

        Returns:
            InsightZettel instance or None if not found
        """
        from core.psyche.schema import InsightZettel, InsightSourceType, QuestionStatus

        cypher = """
        MATCH (z:InsightZettel {uid: $uid})
        RETURN z.uid as uid, z.insight_text as insight_text,
               z.question_text as question_text, z.question_status as question_status,
               z.source_type as source_type, z.source_uid as source_uid,
               z.concept as concept, z.cycle as cycle,
               z.sae_feature_indices as sae_feature_indices,
               z.created_at as created_at
        """
        results = await self.query(cypher, {"uid": uid})
        if not results:
            return None

        r = results[0]
        return InsightZettel(
            uid=r["uid"],
            insight_text=r["insight_text"],
            question_text=r["question_text"],
            question_status=QuestionStatus(r["question_status"]),
            source_type=InsightSourceType(r["source_type"]),
            source_uid=r["source_uid"],
            concept=r["concept"],
            cycle=r.get("cycle"),
            sae_feature_indices=r.get("sae_feature_indices") or [],
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    async def update_zettel_question_status(
        self,
        uid: str,
        status: str,
    ) -> bool:
        """Update the question status of an InsightZettel.

        Args:
            uid: Zettel UID
            status: New status ("open", "addressed", "dissolved")

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (z:InsightZettel {uid: $uid})
        SET z.question_status = $status
        RETURN z.uid
        """
        results = await self.query(cypher, {"uid": uid, "status": status})
        return len(results) > 0

    async def link_zettel_to_fragment(
        self,
        zettel_uid: str,
        fragment_uid: str,
    ) -> bool:
        """Create SOURCED_FROM relationship between zettel and fragment.

        Args:
            zettel_uid: InsightZettel UID
            fragment_uid: Fragment UID (the source thought/document)

        Returns:
            True if relationship created
        """
        cypher = """
        MATCH (z:InsightZettel {uid: $zettel_uid})
        MATCH (f:Fragment {uid: $fragment_uid})
        MERGE (z)-[r:SOURCED_FROM]->(f)
        RETURN count(r) as count
        """
        results = await self.query(cypher, {
            "zettel_uid": zettel_uid,
            "fragment_uid": fragment_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    async def link_zettel_to_sae_snapshot(
        self,
        zettel_uid: str,
        fragment_uid: str,
    ) -> bool:
        """Create GENERATED_WITH relationship between zettel and SAE snapshot.

        Links the zettel to the SAE feature snapshot via the fragment.

        Args:
            zettel_uid: InsightZettel UID
            fragment_uid: Fragment UID whose snapshot to link

        Returns:
            True if relationship created
        """
        cypher = """
        MATCH (z:InsightZettel {uid: $zettel_uid})
        MATCH (f:Fragment {uid: $fragment_uid})-[:GENERATED_WITH]->(s:SAEFeatureSnapshot)
        MERGE (z)-[r:GENERATED_WITH]->(s)
        RETURN count(r) as count
        """
        results = await self.query(cypher, {
            "zettel_uid": zettel_uid,
            "fragment_uid": fragment_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    async def link_zettel_emerged_from(
        self,
        child_uid: str,
        parent_uid: str,
    ) -> bool:
        """Create EMERGED_FROM relationship for zettel lineage.

        Tracks which retrieved insight(s) influenced this insight's generation.

        Args:
            child_uid: The new InsightZettel UID
            parent_uid: The retrieved InsightZettel UID that influenced it

        Returns:
            True if relationship created
        """
        cypher = """
        MATCH (child:InsightZettel {uid: $child_uid})
        MATCH (parent:InsightZettel {uid: $parent_uid})
        MERGE (child)-[r:EMERGED_FROM]->(parent)
        RETURN count(r) as count
        """
        results = await self.query(cypher, {
            "child_uid": child_uid,
            "parent_uid": parent_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    async def link_zettel_addressed_by(
        self,
        question_uid: str,
        answer_uid: str,
    ) -> bool:
        """Create ADDRESSED_BY relationship when question gets answered.

        Also updates the question's status to "addressed".

        Args:
            question_uid: The InsightZettel with open question
            answer_uid: The InsightZettel that addresses it

        Returns:
            True if relationship created and status updated
        """
        cypher = """
        MATCH (question:InsightZettel {uid: $question_uid})
        MATCH (answer:InsightZettel {uid: $answer_uid})
        MERGE (question)-[r:ADDRESSED_BY]->(answer)
        SET question.question_status = 'addressed'
        RETURN count(r) as count
        """
        results = await self.query(cypher, {
            "question_uid": question_uid,
            "answer_uid": answer_uid,
        })
        return bool(results and results[0].get("count", 0) > 0)

    async def query_zettels_by_embedding(
        self,
        embedding: list[float],
        limit: int = 3,
    ) -> list[tuple["InsightZettel", float]]:
        """Query InsightZettels by embedding similarity.

        Uses FalkorDB's native vector similarity search to find semantically
        related insights.

        Args:
            embedding: Query vector to find similar zettels for
            limit: Maximum number of results to return (default 3)

        Returns:
            List of (InsightZettel, score) tuples sorted by similarity descending
        """
        from core.psyche.schema import InsightZettel, QuestionStatus, InsightSourceType

        # Convert embedding to string for f-string interpolation
        # Note: FalkorDB doesn't support parameterized vectors in db.idx.vector.queryNodes
        embedding_str = ", ".join(str(x) for x in embedding)

        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'InsightZettel',
            'embedding',
            $limit,
            vecf32([{embedding_str}])
        ) YIELD node, score
        RETURN node.uid as uid,
               node.insight_text as insight_text,
               node.question_text as question_text,
               node.question_status as question_status,
               node.source_type as source_type,
               node.source_uid as source_uid,
               node.concept as concept,
               node.cycle as cycle,
               node.sae_feature_indices as sae_feature_indices,
               node.created_at as created_at,
               score
        ORDER BY score DESC
        """

        try:
            results = await self.query(cypher, {"limit": limit})
            if results:
                zettels = []
                for r in results:
                    try:
                        zettel = InsightZettel(
                            uid=r["uid"],
                            insight_text=r["insight_text"],
                            question_text=r.get("question_text"),
                            question_status=QuestionStatus(r.get("question_status", "open")),
                            source_type=InsightSourceType(r["source_type"]),
                            source_uid=r["source_uid"],
                            concept=r["concept"],
                            cycle=r.get("cycle"),
                            sae_feature_indices=r.get("sae_feature_indices") or [],
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at"),
                        )
                        zettels.append((zettel, float(r.get("score", 1.0))))
                    except Exception as e:
                        logger.warning(f"Failed to parse zettel from result: {e}")
                        continue
                return zettels
        except Exception as e:
            error_msg = str(e).lower()
            if "no such index" in error_msg or "index" in error_msg:
                logger.warning(
                    f"InsightZettel vector index not found: {e}. "
                    "Falling back to empty results."
                )
                return []
            logger.warning(f"Zettel embedding query failed: {e}")

        return []

    async def query_zettels_by_sae_features(
        self,
        feature_indices: list[int],
        limit: int = 3,
    ) -> list[tuple["InsightZettel", float]]:
        """Query InsightZettels by SAE feature overlap.

        Finds zettels whose sae_feature_indices overlap with the given features.
        Returns zettels ranked by overlap count.

        Args:
            feature_indices: List of active SAE feature indices
            limit: Maximum number of results to return (default 3)

        Returns:
            List of (InsightZettel, overlap_count) tuples sorted by overlap descending
        """
        from core.psyche.schema import InsightZettel, QuestionStatus, InsightSourceType

        if not feature_indices:
            return []

        cypher = """
        MATCH (z:InsightZettel)
        WHERE z.sae_feature_indices IS NOT NULL
        WITH z, [idx IN z.sae_feature_indices WHERE idx IN $feature_indices] AS overlap
        WHERE size(overlap) > 0
        RETURN z.uid as uid,
               z.insight_text as insight_text,
               z.question_text as question_text,
               z.question_status as question_status,
               z.source_type as source_type,
               z.source_uid as source_uid,
               z.concept as concept,
               z.cycle as cycle,
               z.sae_feature_indices as sae_feature_indices,
               z.created_at as created_at,
               size(overlap) as overlap_count
        ORDER BY overlap_count DESC
        LIMIT $limit
        """

        try:
            results = await self.query(cypher, {
                "feature_indices": feature_indices,
                "limit": limit,
            })
            if results:
                zettels = []
                for r in results:
                    try:
                        zettel = InsightZettel(
                            uid=r["uid"],
                            insight_text=r["insight_text"],
                            question_text=r.get("question_text"),
                            question_status=QuestionStatus(r.get("question_status", "open")),
                            source_type=InsightSourceType(r["source_type"]),
                            source_uid=r["source_uid"],
                            concept=r["concept"],
                            cycle=r.get("cycle"),
                            sae_feature_indices=r.get("sae_feature_indices") or [],
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at"),
                        )
                        zettels.append((zettel, float(r["overlap_count"])))
                    except Exception as e:
                        logger.warning(f"Failed to parse zettel from result: {e}")
                        continue
                return zettels
        except Exception as e:
            logger.warning(f"SAE feature zettel query failed: {e}")

        return []

    async def query_open_questions(
        self,
        embedding: list[float],
        limit: int = 2,
    ) -> list[tuple["InsightZettel", float]]:
        """Query open questions by embedding similarity.

        Finds InsightZettels with unresolved questions that are semantically
        related to the current context. The embedding is passed as a parameter
        using vecf32($embedding) syntax to avoid f-string injection risks.

        Args:
            embedding: Query vector to find related open questions
            limit: Maximum number of results to return (default 2)

        Returns:
            List of (InsightZettel, score) tuples with open questions
        """
        from core.psyche.schema import InsightZettel, QuestionStatus, InsightSourceType

        # Query more than limit since we filter by question_status
        fetch_limit = limit * 3

        # Convert embedding to string for f-string interpolation
        # Note: FalkorDB doesn't support parameterized vectors in db.idx.vector.queryNodes
        embedding_str = ", ".join(str(x) for x in embedding)

        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'InsightZettel',
            'embedding',
            $fetch_limit,
            vecf32([{embedding_str}])
        ) YIELD node, score
        WHERE node.question_text IS NOT NULL
          AND node.question_status = 'open'
        RETURN node.uid as uid,
               node.insight_text as insight_text,
               node.question_text as question_text,
               node.question_status as question_status,
               node.source_type as source_type,
               node.source_uid as source_uid,
               node.concept as concept,
               node.cycle as cycle,
               node.sae_feature_indices as sae_feature_indices,
               node.created_at as created_at,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            results = await self.query(cypher, {
                "fetch_limit": fetch_limit,
                "limit": limit,
            })
            if results:
                zettels = []
                for r in results:
                    try:
                        zettel = InsightZettel(
                            uid=r["uid"],
                            insight_text=r["insight_text"],
                            question_text=r.get("question_text"),
                            question_status=QuestionStatus(r.get("question_status", "open")),
                            source_type=InsightSourceType(r["source_type"]),
                            source_uid=r["source_uid"],
                            concept=r["concept"],
                            cycle=r.get("cycle"),
                            sae_feature_indices=r.get("sae_feature_indices") or [],
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at"),
                        )
                        zettels.append((zettel, float(r.get("score", 1.0))))
                    except Exception as e:
                        logger.warning(f"Failed to parse open question zettel: {e}")
                        continue
                return zettels
        except Exception as e:
            error_msg = str(e).lower()
            if "no such index" in error_msg or "index" in error_msg:
                logger.warning(
                    f"InsightZettel vector index not found: {e}. "
                    "Falling back to recency-based query."
                )
                # Fallback to non-vector query
                return await self._query_open_questions_fallback(limit)
            logger.warning(f"Open questions query failed: {e}")

        return []

    async def _query_open_questions_fallback(
        self,
        limit: int = 2,
    ) -> list[tuple["InsightZettel", float]]:
        """Fallback query for open questions without vector index."""
        from core.psyche.schema import InsightZettel, QuestionStatus, InsightSourceType

        cypher = """
        MATCH (z:InsightZettel)
        WHERE z.question_text IS NOT NULL
          AND z.question_status = 'open'
        RETURN z.uid as uid,
               z.insight_text as insight_text,
               z.question_text as question_text,
               z.question_status as question_status,
               z.source_type as source_type,
               z.source_uid as source_uid,
               z.concept as concept,
               z.cycle as cycle,
               z.sae_feature_indices as sae_feature_indices,
               z.created_at as created_at
        ORDER BY z.created_at DESC
        LIMIT $limit
        """

        try:
            results = await self.query(cypher, {"limit": limit})
            if results:
                zettels = []
                for r in results:
                    try:
                        zettel = InsightZettel(
                            uid=r["uid"],
                            insight_text=r["insight_text"],
                            question_text=r.get("question_text"),
                            question_status=QuestionStatus(r.get("question_status", "open")),
                            source_type=InsightSourceType(r["source_type"]),
                            source_uid=r["source_uid"],
                            concept=r["concept"],
                            cycle=r.get("cycle"),
                            sae_feature_indices=r.get("sae_feature_indices") or [],
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at"),
                        )
                        zettels.append((zettel, 1.0))  # Fallback uses default score
                    except Exception as e:
                        logger.warning(f"Failed to parse open question zettel: {e}")
                        continue
                return zettels
        except Exception as e:
            logger.warning(f"Open questions fallback query failed: {e}")

        return []

    async def get_recent_zettels(
        self,
        limit: int = 5,
    ) -> list["InsightZettel"]:
        """Get most recently created InsightZettels.

        Used for breadcrumb trail context showing recent thought progression.

        Args:
            limit: Maximum number of results to return (default 5)

        Returns:
            List of InsightZettel objects sorted by created_at descending
        """
        from core.psyche.schema import InsightZettel, QuestionStatus, InsightSourceType

        cypher = """
        MATCH (z:InsightZettel)
        RETURN z.uid as uid,
               z.insight_text as insight_text,
               z.question_text as question_text,
               z.question_status as question_status,
               z.source_type as source_type,
               z.source_uid as source_uid,
               z.concept as concept,
               z.cycle as cycle,
               z.sae_feature_indices as sae_feature_indices,
               z.created_at as created_at
        ORDER BY z.created_at DESC
        LIMIT $limit
        """

        try:
            results = await self.query(cypher, {"limit": limit})
            if results:
                zettels = []
                for r in results:
                    try:
                        zettel = InsightZettel(
                            uid=r["uid"],
                            insight_text=r["insight_text"],
                            question_text=r.get("question_text"),
                            question_status=QuestionStatus(r.get("question_status", "open")),
                            source_type=InsightSourceType(r["source_type"]),
                            source_uid=r["source_uid"],
                            concept=r["concept"],
                            cycle=r.get("cycle"),
                            sae_feature_indices=r.get("sae_feature_indices") or [],
                            created_at=_parse_datetime_field(r.get("created_at"), "created_at"),
                        )
                        zettels.append(zettel)
                    except Exception as e:
                        logger.warning(f"Failed to parse recent zettel: {e}")
                        continue
                return zettels
        except Exception as e:
            logger.warning(f"Recent zettels query failed: {e}")

        return []

    async def discover_related_zettels(
        self,
        zettel_uid: str,
        embedding: list[float],
        similarity_threshold: float = 0.78,
        max_connections: int = 3,
    ) -> list[tuple["InsightZettel", float]]:
        """Discover zettels semantically related to the given zettel.

        Used for A-MEM-style retroactive link discovery - finds existing
        zettels that should be connected but weren't at creation time.

        Args:
            zettel_uid: UID of the new zettel (to exclude from results)
            embedding: Embedding vector for similarity search
            similarity_threshold: Minimum similarity score (default 0.78)
            max_connections: Maximum number of connections to return

        Returns:
            List of (InsightZettel, similarity_score) tuples
        """
        from core.psyche.schema import InsightZettel, InsightSourceType, QuestionStatus

        # Use vector index for similarity search
        cypher = """
        CALL db.idx.vector.queryNodes(
            'InsightZettel', 'embedding', $limit, vecf32($embedding)
        ) YIELD node, score
        WHERE node.uid <> $exclude_uid AND score >= $threshold
        RETURN node, score
        ORDER BY score DESC
        """

        try:
            results = await self.query(cypher, {
                "exclude_uid": zettel_uid,
                "embedding": embedding,
                "limit": max_connections * 2,  # Request extra for filtering
                "threshold": similarity_threshold,
            })

            zettels = []
            for record in results[:max_connections]:
                node = record.get("node", {})
                score = record.get("score", 0.0)

                # Parse source_type with fallback
                source_type_str = node.get("source_type", "cognitive")
                try:
                    source_type = InsightSourceType(source_type_str)
                except ValueError:
                    source_type = InsightSourceType.COGNITIVE

                # Parse question_status with fallback
                question_status_str = node.get("question_status", "dissolved")
                try:
                    question_status = QuestionStatus(question_status_str)
                except ValueError:
                    question_status = QuestionStatus.DISSOLVED

                zettel = InsightZettel(
                    uid=node.get("uid", ""),
                    insight_text=node.get("insight_text", ""),
                    question_text=node.get("question_text"),
                    question_status=question_status,
                    source_type=source_type,
                    source_uid=node.get("source_uid", ""),
                    concept=node.get("concept", ""),
                    cycle=node.get("cycle"),
                    sae_feature_indices=node.get("sae_feature_indices", []),
                )
                zettels.append((zettel, score))

            return zettels

        except Exception as e:
            logger.warning(f"discover_related_zettels failed: {e}")
            return []

    async def link_zettel_relates_to(
        self,
        zettel_uid: str,
        related_uid: str,
        score: float = 0.0,
        relationship_type: str = "semantic",
    ) -> bool:
        """Create RELATES_TO relationship between zettels.

        Unlike EMERGED_FROM (directional lineage), RELATES_TO captures
        discovered semantic connections found through retroactive analysis.

        Args:
            zettel_uid: Source zettel UID
            related_uid: Related zettel UID
            score: Similarity score that triggered the connection
            relationship_type: Type of relationship (default "semantic")

        Returns:
            True if relationship created
        """
        cypher = """
        MATCH (a:InsightZettel {uid: $uid_a})
        MATCH (b:InsightZettel {uid: $uid_b})
        MERGE (a)-[r:RELATES_TO]->(b)
        SET r.score = $score,
            r.type = $rel_type,
            r.discovered_at = $timestamp
        RETURN count(r) as count
        """
        try:
            results = await self.query(cypher, {
                "uid_a": zettel_uid,
                "uid_b": related_uid,
                "score": score,
                "rel_type": relationship_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return bool(results and results[0].get("count", 0) > 0)
        except Exception as e:
            logger.warning(f"link_zettel_relates_to failed: {e}")
            return False

    async def link_zettel_refines(
        self,
        zettel_uid: str,
        parent_uid: str,
        novelty_score: float,
    ) -> bool:
        """Create REFINES edge from refinement zettel to parent.

        When a zettel is identified as a refinement of existing knowledge
        (novelty_score < threshold), this method creates a REFINES edge
        to track the lineage.

        Args:
            zettel_uid: UID of the refinement (child) zettel
            parent_uid: UID of the parent zettel being refined
            novelty_score: The child's novelty score (0-1)

        Returns:
            True if edge was created, False if nodes not found
        """
        cypher = """
        MATCH (child:InsightZettel {uid: $child_uid})
        MATCH (parent:InsightZettel {uid: $parent_uid})
        MERGE (child)-[r:REFINES]->(parent)
        SET r.novelty_score = $novelty_score,
            r.similarity = $similarity,
            r.created_at = $created_at
        RETURN count(r) as count
        """
        try:
            results = await self.query(cypher, {
                "child_uid": zettel_uid,
                "parent_uid": parent_uid,
                "novelty_score": novelty_score,
                "similarity": 1.0 - novelty_score,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            return bool(results and results[0].get("count", 0) > 0)
        except Exception as e:
            logger.warning(f"link_zettel_refines failed: {e}")
            return False

    async def add_zettel_keyword(
        self,
        zettel_uid: str,
        keyword: str,
    ) -> bool:
        """Add a keyword tag to an existing zettel.

        Enables A-MEM-style dynamic attribute updates where new insights
        can enrich existing zettels with additional context.

        Args:
            zettel_uid: UID of the zettel to update
            keyword: Keyword to add

        Returns:
            True if keyword added
        """
        # FalkorDB doesn't have native array operations, so we need to handle this carefully
        cypher = """
        MATCH (z:InsightZettel {uid: $uid})
        SET z.keywords = CASE
            WHEN z.keywords IS NULL THEN [$keyword]
            WHEN NOT $keyword IN z.keywords THEN z.keywords + $keyword
            ELSE z.keywords
        END
        RETURN count(z) as count
        """
        try:
            results = await self.query(cypher, {
                "uid": zettel_uid,
                "keyword": keyword,
            })
            return bool(results and results[0].get("count", 0) > 0)
        except Exception as e:
            logger.warning(f"add_zettel_keyword failed: {e}")
            return False

    async def get_zettel_relations(
        self,
        zettel_uid: str,
    ) -> list[tuple[str, str, float]]:
        """Get all RELATES_TO connections for a zettel.

        Args:
            zettel_uid: UID of the zettel

        Returns:
            List of (related_uid, relationship_type, score) tuples
        """
        cypher = """
        MATCH (z:InsightZettel {uid: $uid})-[r:RELATES_TO]->(other:InsightZettel)
        RETURN other.uid as related_uid, r.type as rel_type, r.score as score
        """
        try:
            results = await self.query(cypher, {"uid": zettel_uid})
            return [
                (r.get("related_uid", ""), r.get("rel_type", ""), r.get("score", 0.0))
                for r in results
            ]
        except Exception as e:
            logger.warning(f"get_zettel_relations failed: {e}")
            return []

    async def ensure_zettel_vector_index(
        self,
        dimension: int = 1024,
        similarity_function: str = "cosine",
    ) -> bool:
        """Create vector index on InsightZettel.embedding for similarity search.

        Args:
            dimension: Vector dimension (default 1024 for retrieval embeddings)
            similarity_function: Similarity function (default 'cosine')

        Returns:
            True if index created or already exists
        """
        cypher = f"""
        CREATE VECTOR INDEX FOR (z:InsightZettel) ON (z.embedding)
        OPTIONS {{
            dimension: {dimension},
            similarityFunction: '{similarity_function}'
        }}
        """

        try:
            await self.execute(cypher)
            logger.info("Created InsightZettel vector index")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "already exist" in error_msg or "duplicate" in error_msg:
                logger.debug("InsightZettel vector index already exists")
                return True
            logger.error(f"Failed to create InsightZettel vector index: {e}")
            return False

    # === Research Query Methods (NotebookLM Persistence) ===

    async def create_research_query_result(
        self, result: "ResearchQueryResult"
    ) -> bool:
        """Create a ResearchQueryResult node in the graph.

        Persists NotebookLM research answers so Lilly can build on
        previous architectural inquiries.

        Args:
            result: ResearchQueryResult model instance

        Returns:
            True if created successfully
        """
        # Build embedding clause - must use vecf32() for vector index compatibility
        embedding_clause = "null"
        if result.embedding is not None:
            embedding_str = ", ".join(str(x) for x in result.embedding)
            embedding_clause = f"vecf32([{embedding_str}])"

        cypher_create = f"""
        CREATE (r:ResearchQueryResult {{
            uid: $uid,
            question: $question,
            answer: $answer,
            citations: $citations,
            notebook_id: $notebook_id,
            cycle: $cycle,
            created_at: $created_at,
            embedding: {embedding_clause}
        }})
        """
        params = {
            "uid": _sanitize_string(result.uid),
            "question": _sanitize_string(result.question),
            "answer": _sanitize_string(result.answer),
            "citations": result.citations,
            "notebook_id": _sanitize_string(result.notebook_id),
            "cycle": result.cycle,
            "created_at": result.created_at.isoformat(),
        }

        try:
            affected = await self.execute(cypher_create, params)
            logger.info(f"Created ResearchQueryResult {result.uid}")
            return affected > 0
        except Exception as e:
            # Broad exception is intentional: FalkorDB can raise various error types
            # (connection, query syntax, data format) and we want graceful degradation
            # with appropriate logging rather than crashing the cognitive cycle.
            logger.warning(f"Failed to create ResearchQueryResult {result.uid}: {e}")
            return False

    async def find_similar_research_queries(
        self,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 3,
    ) -> list[tuple["ResearchQueryResult", float]]:
        """Find semantically similar past research queries.

        Uses FalkorDB's vector similarity search to find related
        research queries that Lilly has asked before.

        Args:
            embedding: Query vector for similarity search
            threshold: Minimum similarity score (default 0.85)
            limit: Maximum results to return (default 3)

        Returns:
            List of (ResearchQueryResult, score) tuples sorted by similarity
        """
        from core.psyche.schema import ResearchQueryResult

        # NOTE: FalkorDB's db.idx.vector.queryNodes does not support parameterized
        # vectors - the vecf32() function requires a literal array in the query.
        # This is a known FalkorDB limitation. The embedding is safe from injection
        # because it's typed as list[float] and we convert each element via str(),
        # which for floats produces only numeric characters, decimal points, and
        # optional minus signs/exponents (e.g., "-1.5e-10").
        embedding_str = ", ".join(str(x) for x in embedding)

        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'ResearchQueryResult',
            'embedding',
            $limit,
            vecf32([{embedding_str}])
        ) YIELD node, score
        WHERE score >= $threshold
        RETURN node.uid as uid,
               node.question as question,
               node.answer as answer,
               node.citations as citations,
               node.notebook_id as notebook_id,
               node.cycle as cycle,
               node.created_at as created_at,
               score
        ORDER BY score DESC
        """

        try:
            results = await self.query(cypher, {"limit": limit, "threshold": threshold})
            return [
                (
                    ResearchQueryResult(
                        uid=r["uid"],
                        question=r["question"],
                        answer=r["answer"],
                        citations=r.get("citations") or [],
                        notebook_id=r["notebook_id"],
                        cycle=r.get("cycle"),
                        created_at=datetime.fromisoformat(r["created_at"]),
                    ),
                    r["score"],
                )
                for r in results
            ]
        except Exception as e:
            # Broad exception is intentional: FalkorDB can raise various error types
            # (connection, query syntax, missing index) and we want graceful degradation.
            # Index may not exist yet - return empty list
            if "no such index" in str(e).lower():
                logger.debug("ResearchQueryResult vector index not yet created")
                return []
            logger.warning(f"Error finding similar research queries: {e}")
            return []

    async def ensure_research_query_vector_index(
        self,
        dimension: int = 1024,
        similarity_function: str = "cosine",
    ) -> bool:
        """Create vector index for ResearchQueryResult embeddings.

        Args:
            dimension: Vector dimension (default 1024 for retrieval embeddings)
            similarity_function: Similarity function (default 'cosine')

        Returns:
            True if index created or already exists
        """
        cypher = f"""
        CREATE VECTOR INDEX FOR (r:ResearchQueryResult) ON (r.embedding)
        OPTIONS {{
            dimension: {dimension},
            similarityFunction: '{similarity_function}'
        }}
        """

        try:
            await self.execute(cypher)
            logger.info("Created ResearchQueryResult vector index")
            return True
        except Exception as e:
            # Broad exception is intentional: FalkorDB can raise various error types
            # (connection, permissions, index already exists) and we handle gracefully.
            error_msg = str(e).lower()
            if "already exist" in error_msg or "duplicate" in error_msg:
                logger.debug("ResearchQueryResult vector index already exists")
                return True
            logger.error(f"Failed to create ResearchQueryResult vector index: {e}")
            return False

    # === Narration Phrase Methods ===

    async def create_narration_phrase(self, phrase: "NarrationPhrase") -> str:
        """Store a new narration phrase.

        Args:
            phrase: NarrationPhrase model instance

        Returns:
            The UID of the created phrase
        """
        import uuid
        from core.psyche.schema import NarrationPhrase

        uid = phrase.uid or str(uuid.uuid4())

        query = """
        CREATE (p:NarrationPhrase {
            uid: $uid,
            text: $text,
            phrase_type: $phrase_type,
            usage_count: 0,
            created_cycle: $created_cycle,
            retired: false
        })
        RETURN p.uid as uid
        """
        params = {
            "uid": uid,
            "text": _sanitize_string(phrase.text),
            "phrase_type": phrase.phrase_type.value,
            "created_cycle": phrase.created_cycle,
        }

        try:
            result = await self.query(query, params)
            logger.info(f"Created NarrationPhrase {uid}")
            return result[0]["uid"] if result else uid
        except Exception as e:
            logger.warning(f"Failed to create NarrationPhrase: {e}")
            return uid

    async def get_narration_phrases(
        self,
        phrase_type: "PhraseType",
        limit: int = 10,
        exclude_recent: int = 3,
    ) -> list["NarrationPhrase"]:
        """Get active phrases of a type, excluding recently used.

        Args:
            phrase_type: Type of phrase to retrieve
            limit: Maximum number of phrases to return
            exclude_recent: Number of most recently used phrases to exclude

        Returns:
            List of NarrationPhrase models ordered by usage count (least used first)
        """
        from core.psyche.schema import NarrationPhrase, PhraseType

        query = """
        MATCH (p:NarrationPhrase)
        WHERE p.phrase_type = $phrase_type
          AND p.retired = false
          AND (p.usage_count < 20)
        RETURN p.uid as uid,
               p.text as text,
               p.phrase_type as phrase_type,
               p.usage_count as usage_count,
               p.created_cycle as created_cycle,
               p.last_used_cycle as last_used_cycle,
               p.retired as retired
        ORDER BY p.usage_count ASC, rand()
        LIMIT $limit
        """
        params = {
            "phrase_type": phrase_type.value,
            "limit": limit,
        }

        try:
            results = await self.query(query, params)
            phrases = []
            for r in results:
                phrases.append(NarrationPhrase(
                    uid=r.get("uid"),
                    text=r.get("text"),
                    phrase_type=PhraseType(r.get("phrase_type")),
                    usage_count=r.get("usage_count", 0),
                    created_cycle=r.get("created_cycle", 0),
                    last_used_cycle=r.get("last_used_cycle"),
                    retired=r.get("retired", False),
                ))
            return phrases
        except Exception as e:
            logger.warning(f"Failed to get narration phrases: {e}")
            return []

    async def record_phrase_usage(self, uid: str, cycle: int) -> None:
        """Increment usage count and update last_used_cycle.

        Args:
            uid: Phrase UID to update
            cycle: Current cognitive cycle number
        """
        query = """
        MATCH (p:NarrationPhrase {uid: $uid})
        SET p.usage_count = p.usage_count + 1,
            p.last_used_cycle = $cycle
        """
        try:
            await self.execute(query, {"uid": uid, "cycle": cycle})
            logger.debug(f"Recorded usage for phrase {uid}")
        except Exception as e:
            logger.warning(f"Failed to record phrase usage: {e}")

    async def retire_overused_phrases(self, max_uses: int = 20) -> int:
        """Mark phrases as retired after exceeding usage threshold.

        Args:
            max_uses: Maximum usage count before retirement (default 20)

        Returns:
            Number of phrases retired
        """
        query = """
        MATCH (p:NarrationPhrase)
        WHERE p.usage_count >= $max_uses AND p.retired = false
        SET p.retired = true
        RETURN count(p) as retired_count
        """
        try:
            result = await self.query(query, {"max_uses": max_uses})
            count = result[0]["retired_count"] if result else 0
            if count > 0:
                logger.info(f"Retired {count} overused narration phrases")
            return count
        except Exception as e:
            logger.warning(f"Failed to retire overused phrases: {e}")
            return 0

    # === Learned Skill Methods (UPSKILL pattern) ===

    async def create_learned_skill(self, skill: "LearnedSkill") -> bool:
        """Store a learned skill with embedding for vector search.

        Args:
            skill: LearnedSkill model instance

        Returns:
            True if created successfully, False otherwise
        """
        from core.psyche.schema import LearnedSkill

        timestamp = datetime.now(timezone.utc).isoformat()

        # Create skill node
        cypher = """
        CREATE (s:LearnedSkill {
            uid: $uid,
            name: $name,
            description: $description,
            source_hypothesis_uid: $source_hypothesis_uid,
            cognitive_operation: $cognitive_operation,
            pattern_summary: $pattern_summary,
            when_to_apply: $when_to_apply,
            positive_example: $positive_example,
            negative_example: $negative_example,
            usage_count: 0,
            success_count: 0,
            effectiveness_score: 0.5,
            created_cycle: $created_cycle,
            retired: false,
            created_at: $timestamp
        })
        RETURN s.uid as uid
        """
        params = {
            "uid": skill.uid,
            "name": skill.name,
            "description": skill.description,
            "source_hypothesis_uid": skill.source_hypothesis_uid,
            "cognitive_operation": skill.cognitive_operation,
            "pattern_summary": skill.pattern_summary,
            "when_to_apply": skill.when_to_apply,
            "positive_example": skill.positive_example,
            "negative_example": skill.negative_example,
            "created_cycle": skill.created_cycle,
            "timestamp": timestamp,
        }

        try:
            result = await self.query(cypher, params)
            if not result:
                return False

            # Set embedding separately using vecf32()
            if skill.embedding:
                emb_cypher = """
                MATCH (s:LearnedSkill {uid: $uid})
                SET s.embedding = vecf32($embedding)
                """
                await self.execute(emb_cypher, {"uid": skill.uid, "embedding": skill.embedding})

            logger.info(f"Created LearnedSkill {skill.uid} ({skill.name})")
            return True
        except Exception as e:
            logger.warning(f"Failed to create LearnedSkill: {e}")
            return False

    async def query_skills_by_embedding(
        self, embedding: list[float], limit: int = 3
    ) -> list[tuple["LearnedSkill", float]]:
        """Find skills similar to context embedding.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of skills to return

        Returns:
            List of (LearnedSkill, similarity) tuples ordered by similarity
        """
        from core.psyche.schema import LearnedSkill

        cypher = """
        MATCH (s:LearnedSkill)
        WHERE s.retired = false AND s.embedding IS NOT NULL
        WITH s, vec.euclideanDistance(s.embedding, vecf32($embedding)) AS distance
        ORDER BY distance ASC
        LIMIT $limit
        RETURN s.uid as uid,
               s.name as name,
               s.description as description,
               s.source_hypothesis_uid as source_hypothesis_uid,
               s.cognitive_operation as cognitive_operation,
               s.pattern_summary as pattern_summary,
               s.when_to_apply as when_to_apply,
               s.positive_example as positive_example,
               s.negative_example as negative_example,
               s.usage_count as usage_count,
               s.success_count as success_count,
               s.effectiveness_score as effectiveness_score,
               s.created_cycle as created_cycle,
               s.last_used_cycle as last_used_cycle,
               s.retired as retired,
               s.created_at as created_at,
               1.0 / (1.0 + distance) AS similarity
        """

        try:
            results = await self.query(cypher, {"embedding": embedding, "limit": limit})
            skills = []
            for r in results or []:
                skill = LearnedSkill(
                    uid=r["uid"],
                    name=r["name"],
                    description=r["description"] or "",
                    source_hypothesis_uid=r["source_hypothesis_uid"],
                    cognitive_operation=r["cognitive_operation"],
                    pattern_summary=r["pattern_summary"] or "",
                    when_to_apply=r["when_to_apply"] or "",
                    positive_example=r["positive_example"] or "",
                    negative_example=r["negative_example"] or "",
                    usage_count=r["usage_count"] or 0,
                    success_count=r["success_count"] or 0,
                    effectiveness_score=r["effectiveness_score"] or 0.5,
                    created_cycle=r["created_cycle"] or 0,
                    last_used_cycle=r["last_used_cycle"],
                    retired=r["retired"] or False,
                )
                skills.append((skill, r["similarity"]))
            return skills
        except Exception as e:
            logger.warning(f"Failed to query skills by embedding: {e}")
            return []

    async def record_skill_usage(self, skill_uid: str, cycle: int, verified: bool) -> None:
        """Update usage count and effectiveness score (EMA alpha=0.2).

        Args:
            skill_uid: UID of the skill used
            cycle: Current cognitive cycle
            verified: Whether any prediction was verified this cycle
        """
        target = 1.0 if verified else 0.0
        cypher = """
        MATCH (s:LearnedSkill {uid: $uid})
        SET s.usage_count = s.usage_count + 1,
            s.last_used_cycle = $cycle,
            s.success_count = CASE WHEN $verified THEN s.success_count + 1 ELSE s.success_count END,
            s.effectiveness_score = s.effectiveness_score + $alpha * ($target - s.effectiveness_score)
        """

        try:
            await self.execute(
                cypher,
                {
                    "uid": skill_uid,
                    "cycle": cycle,
                    "verified": verified,
                    "target": target,
                    "alpha": SKILL_EFFECTIVENESS_EMA_ALPHA,
                },
            )
            logger.debug(f"Recorded usage for skill {skill_uid} (verified={verified})")
        except Exception as e:
            logger.warning(f"Failed to record skill usage: {e}")

    async def retire_ineffective_skills(
        self,
        min_uses: int = SKILL_MIN_USES_FOR_RETIREMENT,
        min_effectiveness: float = SKILL_MIN_EFFECTIVENESS_FOR_RETIREMENT,
    ) -> int:
        """Mark underperforming skills as retired.

        Args:
            min_uses: Minimum usage count before considering retirement
            min_effectiveness: Skills below this effectiveness get retired

        Returns:
            Number of skills retired
        """
        cypher = """
        MATCH (s:LearnedSkill)
        WHERE s.usage_count >= $min_uses
          AND s.effectiveness_score < $min_effectiveness
          AND s.retired = false
        SET s.retired = true
        RETURN count(s) as count
        """

        try:
            result = await self.query(
                cypher,
                {"min_uses": min_uses, "min_effectiveness": min_effectiveness}
            )
            count = result[0]["count"] if result else 0
            if count > 0:
                logger.info(f"Retired {count} ineffective skills")
            return count
        except Exception as e:
            logger.warning(f"Failed to retire ineffective skills: {e}")
            return 0

    # === Crystal Vector Methods (EvalatisSteerer) ===

    async def upsert_crystal_vector(
        self,
        uid: str,
        name: str,
        zone: str,
        vector_data: list[float],
        parent_names: list[str],
        birth_cycle: int,
        birth_surprise: float,
        selection_count: int = 0,
        total_surprise: float = 0.0,
        staleness: float = 0.0,
        children_spawned: int = 0,
        retired: bool = False,
    ) -> dict:
        """Create or update a CrystalVector node.

        Crystals are frozen steering vectors from EvalatisSteerer that proved
        their worth through sustained surprise. They persist across sessions.

        Args:
            uid: Unique identifier
            name: Crystal name (e.g., "exp_01191432_042")
            zone: Zone name (exploration, concept, identity)
            vector_data: The frozen steering vector
            parent_names: Names of parent crystals (empty if from emergence)
            birth_cycle: Cycle number when crystallized
            birth_surprise: Surprise value at crystallization
            selection_count: Times selected for steering
            total_surprise: Cumulative surprise when selected
            staleness: Current staleness score
            children_spawned: Number of children spawned
            retired: Whether crystal has been pruned

        Returns:
            Dict with the created/updated node properties.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (c:CrystalVector {uid: $uid})
        ON CREATE SET
            c.name = $name,
            c.zone = $zone,
            c.vector_data = $vector_data,
            c.parent_names = $parent_names,
            c.birth_cycle = $birth_cycle,
            c.birth_surprise = $birth_surprise,
            c.selection_count = $selection_count,
            c.total_surprise = $total_surprise,
            c.staleness = $staleness,
            c.children_spawned = $children_spawned,
            c.retired = $retired,
            c.created_at = $timestamp,
            c.updated_at = $timestamp
        ON MATCH SET
            c.selection_count = $selection_count,
            c.total_surprise = $total_surprise,
            c.staleness = $staleness,
            c.children_spawned = $children_spawned,
            c.retired = $retired,
            c.updated_at = $timestamp
        RETURN c.uid as uid, c.name as name, c.zone as zone,
               c.birth_cycle as birth_cycle, c.birth_surprise as birth_surprise,
               c.selection_count as selection_count, c.total_surprise as total_surprise,
               c.retired as retired
        """

        params = {
            "uid": uid,
            "name": name,
            "zone": zone,
            "vector_data": vector_data,
            "parent_names": parent_names or [],
            "birth_cycle": birth_cycle,
            "birth_surprise": birth_surprise,
            "selection_count": selection_count,
            "total_surprise": total_surprise,
            "staleness": staleness,
            "children_spawned": children_spawned,
            "retired": retired,
            "timestamp": timestamp,
        }

        results = await self.query(cypher, params)
        if results:
            logger.debug(f"Upserted CrystalVector: {name} in zone {zone}")
            return results[0]

        return {"uid": uid, "name": name, "zone": zone}

    async def get_crystal_vectors_for_zone(self, zone: str) -> list[dict]:
        """Get all active (non-retired) crystal vectors for a zone.

        Args:
            zone: Zone name (exploration, concept, identity)

        Returns:
            List of crystal vector dicts with all properties.
        """
        cypher = """
        MATCH (c:CrystalVector {zone: $zone, retired: false})
        RETURN c.uid as uid, c.name as name, c.zone as zone,
               c.vector_data as vector_data, c.parent_names as parent_names,
               c.birth_cycle as birth_cycle, c.birth_surprise as birth_surprise,
               c.selection_count as selection_count, c.total_surprise as total_surprise,
               c.staleness as staleness, c.children_spawned as children_spawned,
               c.created_at as created_at
        ORDER BY c.birth_cycle DESC
        """

        results = await self.query(cypher, {"zone": zone})
        # Ensure parent_names has a default value
        for r in results or []:
            if not r.get("parent_names"):
                r["parent_names"] = []
        return results or []

    async def get_all_crystal_vectors(self, include_retired: bool = False) -> list[dict]:
        """Get all crystal vectors across all zones.

        Args:
            include_retired: Whether to include retired crystals

        Returns:
            List of crystal vector dicts.
        """
        if include_retired:
            cypher = """
            MATCH (c:CrystalVector)
            RETURN c.uid as uid, c.name as name, c.zone as zone,
                   c.vector_data as vector_data, c.parent_names as parent_names,
                   c.birth_cycle as birth_cycle, c.birth_surprise as birth_surprise,
                   c.selection_count as selection_count, c.total_surprise as total_surprise,
                   c.staleness as staleness, c.children_spawned as children_spawned,
                   c.retired as retired, c.created_at as created_at
            ORDER BY c.zone, c.birth_cycle DESC
            """
        else:
            cypher = """
            MATCH (c:CrystalVector {retired: false})
            RETURN c.uid as uid, c.name as name, c.zone as zone,
                   c.vector_data as vector_data, c.parent_names as parent_names,
                   c.birth_cycle as birth_cycle, c.birth_surprise as birth_surprise,
                   c.selection_count as selection_count, c.total_surprise as total_surprise,
                   c.staleness as staleness, c.children_spawned as children_spawned,
                   c.retired as retired, c.created_at as created_at
            ORDER BY c.zone, c.birth_cycle DESC
            """

        results = await self.query(cypher, {})
        # Ensure parent_names has a default value
        for r in results or []:
            if not r.get("parent_names"):
                r["parent_names"] = []
        return results or []

    async def retire_crystal_vector(self, uid: str) -> bool:
        """Mark a crystal vector as retired.

        Args:
            uid: Crystal vector UID

        Returns:
            True if updated, False if not found.
        """
        cypher = """
        MATCH (c:CrystalVector {uid: $uid})
        SET c.retired = true, c.updated_at = $timestamp
        RETURN c.uid as uid
        """

        timestamp = datetime.now(timezone.utc).isoformat()
        results = await self.query(cypher, {"uid": uid, "timestamp": timestamp})
        return len(results) > 0 if results else False

    # === Belief Operations ===

    async def update_belief_confidence(
        self,
        topic: str,
        confidence_delta: float,
        evidence: str,
        tenant_id: str = "default",
    ) -> list[tuple[str, float]]:
        """Update the confidence of beliefs matching a topic.

        This method finds beliefs by topic and adjusts their confidence
        by the given delta, clamping to [0.0, 1.0]. It also appends
        the evidence to the belief's supporting evidence list.

        Args:
            topic: Topic to match beliefs against
            confidence_delta: Change in confidence (-1.0 to 1.0)
            evidence: New evidence supporting the update
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            List of (uid, new_confidence) tuples for updated beliefs.
            Empty list if no beliefs were updated (falsy for boolean checks).
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Update beliefs matching the topic, adjusting confidence within bounds
        cypher = """
        MATCH (b:CommittedBelief {topic: $topic, tenant_id: $tenant_id})
        SET b.confidence = CASE
            WHEN b.confidence + $delta > 1.0 THEN 1.0
            WHEN b.confidence + $delta < 0.0 THEN 0.0
            ELSE b.confidence + $delta
        END,
        b.revised_at = $timestamp,
        b.revision_count = COALESCE(b.revision_count, 0) + 1,
        b.supporting_evidence = CASE
            WHEN b.supporting_evidence IS NULL THEN [$evidence]
            ELSE b.supporting_evidence + [$evidence]
        END
        RETURN b.uid as uid, b.confidence as confidence
        """

        results = await self.query(cypher, {
            "topic": topic,
            "delta": confidence_delta,
            "evidence": evidence,
            "timestamp": timestamp,
            "tenant_id": tenant_id,
        })

        updated: list[tuple[str, float]] = []
        if results:
            for r in results:
                uid = r.get("uid", "")
                confidence = float(r.get("confidence", 0.0))
                logger.debug(
                    f"Updated belief {uid}: confidence now {confidence:.2f}"
                )
                updated.append((uid, confidence))
        return updated

    async def delete_retired_crystals(self, zone: Optional[str] = None) -> int:
        """Permanently delete retired crystal vectors.

        Args:
            zone: Optional zone to limit deletion to

        Returns:
            Count of deleted crystals.
        """
        if zone:
            cypher = """
            MATCH (c:CrystalVector {zone: $zone, retired: true})
            WITH c, c.uid as uid
            DELETE c
            RETURN count(uid) as deleted
            """
            params = {"zone": zone}
        else:
            cypher = """
            MATCH (c:CrystalVector {retired: true})
            WITH c, c.uid as uid
            DELETE c
            RETURN count(uid) as deleted
            """
            params = {}

        results = await self.query(cypher, params)
        if results and len(results) > 0:
            return results[0].get("deleted", 0)
        return 0

    # === Cognitive State Persistence ===

    async def save_cognitive_state(
        self,
        curated_prompt: str,
        last_concept: str,
        current_insight: str = "",
        current_question: str = "",
        cycle_count: int = 0,
        recent_concepts: Optional[list[str]] = None,
    ) -> bool:
        """Save cognitive state for persistence across restarts.

        Uses MERGE to ensure only one snapshot exists at a time.
        Each save overwrites the previous state.

        Args:
            curated_prompt: Full prompt for next generation cycle
            last_concept: Most recent exploration concept
            current_insight: Insight driving forward momentum
            current_question: Question driving exploration
            cycle_count: Current cycle number
            recent_concepts: List of recently explored concepts

        Returns:
            True if saved successfully
        """
        uid = "cognitive_state_current"
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (s:CognitiveStateSnapshot {uid: $uid})
        ON CREATE SET
            s.curated_prompt = $curated_prompt,
            s.last_concept = $last_concept,
            s.current_insight = $current_insight,
            s.current_question = $current_question,
            s.cycle_count = $cycle_count,
            s.recent_concepts = $recent_concepts,
            s.updated_at = $updated_at
        ON MATCH SET
            s.curated_prompt = $curated_prompt,
            s.last_concept = $last_concept,
            s.current_insight = $current_insight,
            s.current_question = $current_question,
            s.cycle_count = $cycle_count,
            s.recent_concepts = $recent_concepts,
            s.updated_at = $updated_at
        """

        params = {
            "uid": uid,
            "curated_prompt": _sanitize_string(curated_prompt),
            "last_concept": _sanitize_string(last_concept),
            "current_insight": _sanitize_string(current_insight),
            "current_question": _sanitize_string(current_question),
            "cycle_count": cycle_count,
            "recent_concepts": recent_concepts or [],
            "updated_at": timestamp,
        }

        try:
            await self.execute(cypher, params)
            logger.debug(f"Saved cognitive state: concept={last_concept}, cycle={cycle_count}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cognitive state: {e}")
            return False

    async def load_cognitive_state(self) -> Optional["CognitiveStateSnapshot"]:
        """Load persisted cognitive state.

        Returns:
            CognitiveStateSnapshot if found, None otherwise
        """
        from core.psyche.schema import CognitiveStateSnapshot

        cypher = """
        MATCH (s:CognitiveStateSnapshot {uid: 'cognitive_state_current'})
        RETURN s.curated_prompt as curated_prompt,
               s.last_concept as last_concept,
               s.current_insight as current_insight,
               s.current_question as current_question,
               s.cycle_count as cycle_count,
               s.recent_concepts as recent_concepts,
               s.updated_at as updated_at
        """

        try:
            results = await self.query(cypher)
            if not results:
                return None

            r = results[0]
            return CognitiveStateSnapshot(
                curated_prompt=r.get("curated_prompt", ""),
                last_concept=r.get("last_concept", ""),
                current_insight=r.get("current_insight", ""),
                current_question=r.get("current_question", ""),
                cycle_count=r.get("cycle_count", 0),
                recent_concepts=r.get("recent_concepts") or [],
                updated_at=datetime.fromisoformat(r["updated_at"]) if r.get("updated_at") else datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning(f"Failed to load cognitive state: {e}")
            return None

    # === Emotional Field Persistence ===

    async def save_emotional_field(self, field_data: dict) -> bool:
        """Persist emotional field state to the graph.

        Stores the complete emotional field state (wave packets, global valence,
        arousal, etc.) as a JSON blob in a dedicated EmotionalField node.
        Uses MERGE to ensure only one field state exists at a time.

        Args:
            field_data: Dictionary containing full emotional field state

        Returns:
            True if saved successfully
        """
        uid = "emotional_field_current"
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (f:EmotionalField {uid: $uid})
        SET f.data = $data, f.updated_at = $updated_at
        """

        params = {
            "uid": uid,
            "data": json.dumps(field_data),
            "updated_at": timestamp,
        }

        try:
            await self.execute(cypher, params)
            logger.debug("Saved emotional field state")
            return True
        except Exception as e:
            logger.warning(f"Failed to save emotional field: {e}")
            return False

    async def load_emotional_field(self) -> Optional[dict]:
        """Load persisted emotional field state.

        Returns:
            Dictionary containing emotional field state if found, None otherwise
        """
        cypher = """
        MATCH (f:EmotionalField {uid: 'emotional_field_current'})
        RETURN f.data as data
        """

        try:
            results = await self.query(cypher)
            if not results:
                return None

            data = results[0].get("data")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Failed to load emotional field: {e}")
            return None

    # ==========================================
    # Simulation Phase: Hypothesis & Prediction
    # ==========================================

    async def create_hypothesis(self, hypothesis: "Hypothesis") -> bool:
        """Create a Hypothesis node in the graph.

        Args:
            hypothesis: Hypothesis model instance from simulation schemas

        Returns:
            True if created successfully
        """
        cypher = """
        CREATE (h:Hypothesis {
            uid: $uid,
            statement: $statement,
            source_zettel_uid: $source_zettel_uid,
            source_belief_uid: $source_belief_uid,
            source_thought: $source_thought,
            status: $status,
            confidence: $confidence,
            brainstorm_trace: $brainstorm_trace,
            synthesis_narrative: $synthesis_narrative,
            cognitive_operation: $cognitive_operation,
            positive_example: $positive_example,
            negative_example: $negative_example,
            steering_vector_uid: $steering_vector_uid,
            cycle_generated: $cycle_generated,
            predictions_count: $predictions_count,
            verified_count: $verified_count,
            falsified_count: $falsified_count,
            created_at: $created_at,
            parent_hypothesis_uid: $parent_hypothesis_uid,
            lineage_verified_count: $lineage_verified_count,
            lineage_falsified_count: $lineage_falsified_count,
            calibration_error: $calibration_error,
            last_evaluated_cycle: $last_evaluated_cycle,
            pending_count: $pending_count,
            falsification_condition: $falsification_condition,
            novelty_statement: $novelty_statement,
            last_follow_up_cycle: $last_follow_up_cycle,
            is_experiment: $is_experiment,
            experiment_domain: $experiment_domain,
            parameter_path: $parameter_path,
            control_value: $control_value,
            treatment_value: $treatment_value,
            baseline_cycles: $baseline_cycles,
            treatment_cycles: $treatment_cycles,
            washout_cycles: $washout_cycles,
            current_phase: $current_phase,
            phase_start_cycle: $phase_start_cycle,
            target_metric: $target_metric,
            expected_direction: $expected_direction,
            min_effect_size: $min_effect_size,
            rollback_trigger: $rollback_trigger,
            baseline_mean: $baseline_mean,
            treatment_mean: $treatment_mean,
            effect_size: $effect_size,
            recommendation: $recommendation
        })
        """
        params = {
            "uid": _sanitize_string(hypothesis.uid),
            "statement": _sanitize_string(hypothesis.statement),
            "source_zettel_uid": _sanitize_string(hypothesis.source_zettel_uid) if hypothesis.source_zettel_uid else None,
            "source_belief_uid": _sanitize_string(hypothesis.source_belief_uid) if hypothesis.source_belief_uid else None,
            "source_thought": _sanitize_string(hypothesis.source_thought[:500]) if hypothesis.source_thought else "",
            "status": hypothesis.status.value,
            "confidence": hypothesis.confidence,
            "brainstorm_trace": _sanitize_string(hypothesis.brainstorm_trace[:2000]) if hypothesis.brainstorm_trace else "",
            "synthesis_narrative": _sanitize_string(hypothesis.synthesis_narrative[:2000]) if hypothesis.synthesis_narrative else "",
            "cognitive_operation": (
                _sanitize_string(hypothesis.cognitive_operation)
                if hypothesis.cognitive_operation else None
            ),
            "positive_example": (
                _sanitize_string(hypothesis.positive_example[:1000])
                if hypothesis.positive_example else ""
            ),
            "negative_example": (
                _sanitize_string(hypothesis.negative_example[:1000])
                if hypothesis.negative_example else ""
            ),
            "steering_vector_uid": hypothesis.steering_vector_uid,
            "cycle_generated": hypothesis.cycle_generated,
            "predictions_count": hypothesis.predictions_count,
            "verified_count": hypothesis.verified_count,
            "falsified_count": hypothesis.falsified_count,
            "created_at": hypothesis.created_at.isoformat(),
            "parent_hypothesis_uid": hypothesis.parent_hypothesis_uid,
            "lineage_verified_count": hypothesis.lineage_verified_count,
            "lineage_falsified_count": hypothesis.lineage_falsified_count,
            "calibration_error": hypothesis.calibration_error,
            "last_evaluated_cycle": hypothesis.last_evaluated_cycle,
            "pending_count": hypothesis.pending_count,
            # Additional fields
            "falsification_condition": (
                _sanitize_string(hypothesis.falsification_condition)
                if hypothesis.falsification_condition else None
            ),
            "novelty_statement": (
                _sanitize_string(hypothesis.novelty_statement)
                if hypothesis.novelty_statement else None
            ),
            "last_follow_up_cycle": hypothesis.last_follow_up_cycle,
            # Experiment extension fields
            "is_experiment": hypothesis.is_experiment,
            "experiment_domain": hypothesis.experiment_domain,
            "parameter_path": hypothesis.parameter_path,
            "control_value": hypothesis.control_value,
            "treatment_value": hypothesis.treatment_value,
            "baseline_cycles": hypothesis.baseline_cycles,
            "treatment_cycles": hypothesis.treatment_cycles,
            "washout_cycles": hypothesis.washout_cycles,
            "current_phase": hypothesis.current_phase,
            "phase_start_cycle": hypothesis.phase_start_cycle,
            "target_metric": hypothesis.target_metric,
            "expected_direction": hypothesis.expected_direction,
            "min_effect_size": hypothesis.min_effect_size,
            "rollback_trigger": hypothesis.rollback_trigger,
            "baseline_mean": hypothesis.baseline_mean,
            "treatment_mean": hypothesis.treatment_mean,
            "effect_size": hypothesis.effect_size,
            "recommendation": hypothesis.recommendation,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to create Hypothesis {hypothesis.uid}: {e}")
            return False

    async def get_hypothesis(self, uid: str) -> Optional["Hypothesis"]:
        """Retrieve a Hypothesis by UID.

        Args:
            uid: Hypothesis unique identifier

        Returns:
            Hypothesis instance or None if not found
        """
        from core.cognitive.simulation.schemas import Hypothesis, HypothesisStatus

        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        RETURN h.uid as uid, h.statement as statement,
               h.source_zettel_uid as source_zettel_uid,
               h.source_belief_uid as source_belief_uid,
               h.source_thought as source_thought,
               h.status as status, h.confidence as confidence,
               h.brainstorm_trace as brainstorm_trace,
               h.synthesis_narrative as synthesis_narrative,
               h.cognitive_operation as cognitive_operation,
               h.positive_example as positive_example,
               h.negative_example as negative_example,
               h.cycle_generated as cycle_generated,
               h.predictions_count as predictions_count,
               h.verified_count as verified_count,
               h.falsified_count as falsified_count,
               h.created_at as created_at,
               h.steering_vector_uid as steering_vector_uid,
               h.last_follow_up_cycle as last_follow_up_cycle,
               h.parent_hypothesis_uid as parent_hypothesis_uid,
               h.lineage_verified_count as lineage_verified_count,
               h.lineage_falsified_count as lineage_falsified_count,
               h.calibration_error as calibration_error,
               h.last_evaluated_cycle as last_evaluated_cycle,
               h.pending_count as pending_count,
               h.falsification_condition as falsification_condition,
               h.novelty_statement as novelty_statement,
               h.is_experiment as is_experiment,
               h.experiment_domain as experiment_domain,
               h.parameter_path as parameter_path,
               h.control_value as control_value,
               h.treatment_value as treatment_value,
               h.baseline_cycles as baseline_cycles,
               h.treatment_cycles as treatment_cycles,
               h.washout_cycles as washout_cycles,
               h.current_phase as current_phase,
               h.phase_start_cycle as phase_start_cycle,
               h.target_metric as target_metric,
               h.expected_direction as expected_direction,
               h.min_effect_size as min_effect_size,
               h.rollback_trigger as rollback_trigger,
               h.baseline_mean as baseline_mean,
               h.treatment_mean as treatment_mean,
               h.effect_size as effect_size,
               h.recommendation as recommendation
        """
        results = await self.query(cypher, {"uid": uid})
        if not results:
            return None

        r = results[0]
        return Hypothesis(
            uid=r["uid"],
            statement=r["statement"],
            source_zettel_uid=r.get("source_zettel_uid"),
            source_belief_uid=r.get("source_belief_uid"),
            source_thought=r.get("source_thought", ""),
            status=HypothesisStatus(r["status"]),
            confidence=r.get("confidence", 0.5),
            brainstorm_trace=r.get("brainstorm_trace", ""),
            synthesis_narrative=r.get("synthesis_narrative", ""),
            cognitive_operation=r.get("cognitive_operation", ""),
            positive_example=r.get("positive_example", ""),
            negative_example=r.get("negative_example", ""),
            cycle_generated=r.get("cycle_generated", 0),
            predictions_count=r.get("predictions_count", 0),
            verified_count=r.get("verified_count", 0),
            falsified_count=r.get("falsified_count", 0),
            created_at=datetime.fromisoformat(r["created_at"]),
            steering_vector_uid=r.get("steering_vector_uid"),
            last_follow_up_cycle=r.get("last_follow_up_cycle"),
            parent_hypothesis_uid=r.get("parent_hypothesis_uid"),
            lineage_verified_count=r.get("lineage_verified_count", 0),
            lineage_falsified_count=r.get("lineage_falsified_count", 0),
            calibration_error=r.get("calibration_error", 0.0),
            last_evaluated_cycle=r.get("last_evaluated_cycle", 0),
            pending_count=r.get("pending_count", 0),
            falsification_condition=r.get("falsification_condition"),
            novelty_statement=r.get("novelty_statement"),
            # Experiment extension fields
            is_experiment=r.get("is_experiment", False),
            experiment_domain=r.get("experiment_domain"),
            parameter_path=r.get("parameter_path"),
            control_value=r.get("control_value"),
            treatment_value=r.get("treatment_value"),
            baseline_cycles=r.get("baseline_cycles", 5),
            treatment_cycles=r.get("treatment_cycles", 10),
            washout_cycles=r.get("washout_cycles", 3),
            current_phase=r.get("current_phase", "pending"),
            phase_start_cycle=r.get("phase_start_cycle"),
            target_metric=r.get("target_metric"),
            expected_direction=r.get("expected_direction", "increase"),
            min_effect_size=r.get("min_effect_size", 0.1),
            rollback_trigger=r.get("rollback_trigger"),
            baseline_mean=r.get("baseline_mean"),
            treatment_mean=r.get("treatment_mean"),
            effect_size=r.get("effect_size"),
            recommendation=r.get("recommendation"),
        )

    async def update_hypothesis(self, hypothesis: "Hypothesis") -> bool:
        """Update an existing Hypothesis.

        Args:
            hypothesis: Updated Hypothesis instance

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.status = $status,
            h.confidence = $confidence,
            h.predictions_count = $predictions_count,
            h.verified_count = $verified_count,
            h.falsified_count = $falsified_count,
            h.calibration_error = $calibration_error,
            h.last_evaluated_cycle = $last_evaluated_cycle,
            h.pending_count = $pending_count
        """
        params = {
            "uid": hypothesis.uid,
            "status": hypothesis.status.value,
            "confidence": hypothesis.confidence,
            "predictions_count": hypothesis.predictions_count,
            "verified_count": hypothesis.verified_count,
            "falsified_count": hypothesis.falsified_count,
            "calibration_error": hypothesis.calibration_error,
            "last_evaluated_cycle": hypothesis.last_evaluated_cycle,
            "pending_count": hypothesis.pending_count,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update Hypothesis {hypothesis.uid}: {e}")
            return False

    async def get_active_hypotheses(self, limit: int = 10) -> list["Hypothesis"]:
        """Get active hypotheses with pending predictions.

        Args:
            limit: Maximum number of hypotheses to return

        Returns:
            List of active Hypothesis instances
        """
        from core.cognitive.simulation.schemas import Hypothesis, HypothesisStatus

        cypher = """
        MATCH (h:Hypothesis)
        WHERE h.status IN ['proposed', 'active']
        RETURN h.uid as uid, h.statement as statement,
               h.source_zettel_uid as source_zettel_uid,
               h.status as status, h.confidence as confidence,
               h.cycle_generated as cycle_generated,
               h.predictions_count as predictions_count,
               h.verified_count as verified_count,
               h.falsified_count as falsified_count,
               h.created_at as created_at,
               h.calibration_error as calibration_error,
               h.last_evaluated_cycle as last_evaluated_cycle,
               h.pending_count as pending_count
        ORDER BY h.created_at DESC
        LIMIT $limit
        """
        results = await self.query(cypher, {"limit": limit})

        hypotheses = []
        for r in results:
            hypotheses.append(Hypothesis(
                uid=r["uid"],
                statement=r["statement"],
                source_zettel_uid=r.get("source_zettel_uid"),
                status=HypothesisStatus(r["status"]),
                confidence=r.get("confidence", 0.5),
                cycle_generated=r.get("cycle_generated", 0),
                predictions_count=r.get("predictions_count", 0),
                verified_count=r.get("verified_count", 0),
                falsified_count=r.get("falsified_count", 0),
                created_at=datetime.fromisoformat(r["created_at"]),
                calibration_error=r.get("calibration_error", 0.0),
                last_evaluated_cycle=r.get("last_evaluated_cycle", 0),
                pending_count=r.get("pending_count", 0),
            ))
        return hypotheses

    async def get_hypotheses_needing_vector_extraction(
        self, limit: int = 5
    ) -> list["Hypothesis"]:
        """Get hypotheses that have contrastive pairs but no steering vector.

        These are hypotheses where vector extraction was deferred because
        the model wasn't loaded at creation time.

        Args:
            limit: Maximum number of hypotheses to return

        Returns:
            List of Hypothesis instances needing vector extraction
        """
        from core.cognitive.simulation.schemas import Hypothesis, HypothesisStatus

        cypher = """
        MATCH (h:Hypothesis)
        WHERE h.positive_example IS NOT NULL
          AND h.positive_example <> ''
          AND h.negative_example IS NOT NULL
          AND h.negative_example <> ''
          AND (h.steering_vector_uid IS NULL OR h.steering_vector_uid = '')
          AND h.status IN ['proposed', 'active']
        RETURN h.uid as uid, h.statement as statement,
               h.cognitive_operation as cognitive_operation,
               h.positive_example as positive_example,
               h.negative_example as negative_example,
               h.status as status, h.confidence as confidence,
               h.cycle_generated as cycle_generated,
               h.created_at as created_at
        ORDER BY h.created_at ASC
        LIMIT $limit
        """
        results = await self.query(cypher, {"limit": limit})

        hypotheses = []
        for r in results:
            hypotheses.append(Hypothesis(
                uid=r["uid"],
                statement=r["statement"],
                cognitive_operation=r.get("cognitive_operation") or "",
                positive_example=r.get("positive_example", ""),
                negative_example=r.get("negative_example", ""),
                status=HypothesisStatus(r["status"]),
                confidence=r.get("confidence", 0.5),
                cycle_generated=r.get("cycle_generated", 0),
                created_at=datetime.fromisoformat(r["created_at"]),
            ))
        return hypotheses

    async def set_hypothesis_steering_vector_uid(
        self, hypothesis_uid: str, steering_vector_uid: str
    ) -> bool:
        """Link a hypothesis to its extracted steering vector.

        Called after vector extraction to record the relationship.

        Args:
            hypothesis_uid: UID of the hypothesis
            steering_vector_uid: UID of the extracted steering vector

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $hypothesis_uid})
        SET h.steering_vector_uid = $steering_vector_uid
        """
        try:
            affected = await self.execute(cypher, {
                "hypothesis_uid": hypothesis_uid,
                "steering_vector_uid": steering_vector_uid,
            })
            return affected > 0
        except Exception as e:
            logger.warning(
                f"Failed to set steering_vector_uid for hypothesis "
                f"{hypothesis_uid}: {e}"
            )
            return False

    async def create_prediction(self, prediction: "Prediction") -> bool:
        """Create a Prediction node in the graph and link it to its Hypothesis.

        Args:
            prediction: Prediction model instance from simulation schemas

        Returns:
            True if created successfully
        """
        # Serialize baseline_metrics to JSON if present
        baseline_metrics_json = None
        if prediction.baseline_metrics:
            baseline_metrics_json = json.dumps(prediction.baseline_metrics)

        cypher = """
        MATCH (h:Hypothesis {uid: $hypothesis_uid})
        CREATE (p:Prediction {
            uid: $uid,
            hypothesis_uid: $hypothesis_uid,
            claim: $claim,
            condition_type: $condition_type,
            condition_value: $condition_value,
            status: $status,
            confidence: $confidence,
            earliest_verify_cycle: $earliest_verify_cycle,
            expiry_cycle: $expiry_cycle,
            baseline_metrics: $baseline_metrics,
            baseline_cycle: $baseline_cycle,
            cycle_created: $cycle_created,
            target_goal_uid: $target_goal_uid,
            expected_goal_delta: $expected_goal_delta,
            goal_snapshot_before: $goal_snapshot_before,
            created_at: $created_at
        })-[:TESTS]->(h)
        """
        params = {
            "uid": _sanitize_string(prediction.uid),
            "hypothesis_uid": _sanitize_string(prediction.hypothesis_uid),
            "claim": _sanitize_string(prediction.claim),
            "condition_type": prediction.condition_type.value,
            "condition_value": prediction.condition_value,
            "status": prediction.status.value,
            "confidence": prediction.confidence,
            "earliest_verify_cycle": prediction.earliest_verify_cycle,
            "expiry_cycle": prediction.expiry_cycle,
            "baseline_metrics": baseline_metrics_json,
            "baseline_cycle": prediction.baseline_cycle,
            "cycle_created": prediction.cycle_created,
            "target_goal_uid": prediction.target_goal_uid,
            "expected_goal_delta": prediction.expected_goal_delta,
            "goal_snapshot_before": prediction.goal_snapshot_before,
            "created_at": prediction.created_at.isoformat(),
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to create Prediction {prediction.uid}: {e}")
            return False

    async def get_pending_predictions(self, max_cycle: int) -> list["Prediction"]:
        """Get pending predictions that can be verified at or before given cycle.

        Args:
            max_cycle: Maximum cycle number for verification eligibility

        Returns:
            List of pending Prediction instances
        """
        from core.cognitive.simulation.schemas import (
            Prediction,
            PredictionConditionType,
            PredictionStatus,
        )

        cypher = """
        MATCH (p:Prediction)
        WHERE p.status = 'pending'
          AND (p.earliest_verify_cycle IS NULL OR p.earliest_verify_cycle <= $max_cycle)
          AND (p.expiry_cycle IS NULL OR p.expiry_cycle >= $max_cycle)
        RETURN p.uid as uid, p.hypothesis_uid as hypothesis_uid,
               p.claim as claim, p.condition_type as condition_type,
               p.condition_value as condition_value,
               p.status as status, p.confidence as confidence,
               p.earliest_verify_cycle as earliest_verify_cycle,
               p.expiry_cycle as expiry_cycle,
               p.baseline_metrics as baseline_metrics,
               p.baseline_cycle as baseline_cycle,
               p.target_goal_uid as target_goal_uid,
               p.expected_goal_delta as expected_goal_delta,
               p.goal_snapshot_before as goal_snapshot_before,
               p.created_at as created_at,
               p.cycle_created as cycle_created
        """
        results = await self.query(cypher, {"max_cycle": max_cycle})

        predictions = []
        for r in results:
            # Deserialize baseline_metrics from JSON if present
            baseline_metrics = None
            if r.get("baseline_metrics"):
                try:
                    baseline_metrics = json.loads(r["baseline_metrics"])
                except (json.JSONDecodeError, TypeError):
                    baseline_metrics = None

            predictions.append(Prediction(
                uid=r["uid"],
                hypothesis_uid=r["hypothesis_uid"],
                claim=r["claim"],
                condition_type=PredictionConditionType(r["condition_type"]),
                condition_value=r.get("condition_value", ""),
                status=PredictionStatus(r["status"]),
                confidence=r.get("confidence", 0.5),
                earliest_verify_cycle=r.get("earliest_verify_cycle"),
                expiry_cycle=r.get("expiry_cycle"),
                baseline_metrics=baseline_metrics,
                baseline_cycle=r.get("baseline_cycle"),
                target_goal_uid=r.get("target_goal_uid"),
                expected_goal_delta=r.get("expected_goal_delta", 0.0),
                goal_snapshot_before=r.get("goal_snapshot_before"),
                created_at=datetime.fromisoformat(r["created_at"]),
                cycle_created=r.get("cycle_created"),
            ))
        return predictions

    async def update_prediction(self, prediction: "Prediction") -> bool:
        """Update a Prediction after verification.

        Args:
            prediction: Updated Prediction instance

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (p:Prediction {uid: $uid})
        SET p.status = $status,
            p.outcome = $outcome,
            p.accuracy_score = $accuracy_score,
            p.verification_cycle = $verification_cycle
        """
        params = {
            "uid": prediction.uid,
            "status": prediction.status.value,
            "outcome": prediction.outcome,
            "accuracy_score": prediction.accuracy_score,
            "verification_cycle": prediction.verification_cycle,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update Prediction {prediction.uid}: {e}")
            return False

    async def get_recent_verified_predictions(self, limit: int = 5) -> list[dict]:
        """Get recently verified or falsified predictions for external context.

        Returns predictions that have been resolved (verified or falsified) to
        provide empirical grounding in generation prompts. Results are ordered
        by verification_cycle descending (most recent first).

        Args:
            limit: Maximum number of predictions to return

        Returns:
            List of dicts with claim, status, and outcome fields
        """
        cypher = """
        MATCH (p:Prediction)
        WHERE p.status IN ['verified', 'falsified']
          AND p.outcome IS NOT NULL
        RETURN p.claim as claim, p.status as status, p.outcome as outcome
        ORDER BY p.verification_cycle DESC
        LIMIT $limit
        """
        try:
            result = await self.query(cypher, {"limit": limit})
            return [
                {
                    "claim": row.get("claim", ""),
                    "status": row.get("status", ""),
                    "outcome": row.get("outcome", ""),
                }
                for row in result
            ]
        except Exception as e:
            logger.warning(f"Failed to get recent verified predictions: {e}")
            return []

    async def get_verification_rate(self) -> float:
        """Calculate overall verification rate from all resolved predictions.

        Returns:
            Verification rate as verified / (verified + falsified), or 0.0 if no
            resolved predictions exist.
        """
        cypher = """
        MATCH (p:Prediction)
        WHERE p.status IN ['verified', 'falsified']
        RETURN p.status as status, COUNT(*) as count
        """
        try:
            result = await self.query(cypher)
            counts = {row.get("status", ""): row.get("count", 0) for row in result}
            verified = counts.get("verified", 0)
            falsified = counts.get("falsified", 0)

            total = verified + falsified
            if total == 0:
                return 0.0
            return verified / total
        except Exception as e:
            logger.warning(f"Failed to get verification rate: {e}")
            return 0.0

    async def update_hypothesis_stats(
        self, uid: str, verified_delta: int, falsified_delta: int
    ) -> bool:
        """Update hypothesis verification statistics.

        Args:
            uid: Hypothesis UID
            verified_delta: Change in verified count
            falsified_delta: Change in falsified count

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.verified_count = coalesce(h.verified_count, 0) + $verified_delta,
            h.falsified_count = coalesce(h.falsified_count, 0) + $falsified_delta
        """
        params = {
            "uid": uid,
            "verified_delta": verified_delta,
            "falsified_delta": falsified_delta,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update hypothesis stats {uid}: {e}")
            return False

    # ==========================================
    # Hypothesis Steering Vector CRUD
    # ==========================================

    async def save_hypothesis_steering_vector(
        self, vector: "HypothesisSteeringVector"
    ) -> bool:
        """Save or update a HypothesisSteeringVector in the graph.

        Uses MERGE to create or update the vector node. The vector is linked
        to its parent hypothesis via a STEERS relationship.

        Args:
            vector: HypothesisSteeringVector instance to save

        Returns:
            True if saved successfully
        """
        # Serialize vector_data as JSON string for storage
        vector_data_json = json.dumps(vector.vector_data)

        cypher = """
        MERGE (sv:HypothesisSteeringVector {uid: $uid})
        ON CREATE SET
            sv.hypothesis_uid = $hypothesis_uid,
            sv.cognitive_operation = $cognitive_operation,
            sv.vector_data = $vector_data,
            sv.layer = $layer,
            sv.effectiveness_score = $effectiveness_score,
            sv.application_count = $application_count,
            sv.verified_count = $verified_count,
            sv.falsified_count = $falsified_count,
            sv.measured_capacity = $measured_capacity,
            sv.created_at = $created_at,
            sv.last_applied = $last_applied
        ON MATCH SET
            sv.effectiveness_score = $effectiveness_score,
            sv.application_count = $application_count,
            sv.verified_count = $verified_count,
            sv.falsified_count = $falsified_count,
            sv.measured_capacity = $measured_capacity,
            sv.last_applied = $last_applied
        """
        params = {
            "uid": _sanitize_string(vector.uid),
            "hypothesis_uid": _sanitize_string(vector.hypothesis_uid),
            "cognitive_operation": _sanitize_string(vector.cognitive_operation),
            "vector_data": vector_data_json,
            "layer": vector.layer,
            "effectiveness_score": vector.effectiveness_score,
            "application_count": vector.application_count,
            "verified_count": vector.verified_count,
            "falsified_count": vector.falsified_count,
            "measured_capacity": vector.measured_capacity,
            "created_at": vector.created_at.isoformat(),
            "last_applied": vector.last_applied.isoformat()
            if vector.last_applied
            else None,
        }

        try:
            affected = await self.execute(cypher, params)
            if affected > 0:
                # Create relationship to hypothesis if it exists
                link_cypher = """
                MATCH (sv:HypothesisSteeringVector {uid: $sv_uid})
                MATCH (h:Hypothesis {uid: $hyp_uid})
                MERGE (sv)-[:STEERS]->(h)
                """
                await self.execute(
                    link_cypher,
                    {"sv_uid": vector.uid, "hyp_uid": vector.hypothesis_uid},
                )
            return affected > 0
        except Exception as e:
            logger.warning(
                f"Failed to save HypothesisSteeringVector {vector.uid}: {e}"
            )
            return False

    async def get_hypothesis_steering_vector(
        self, uid: str
    ) -> Optional["HypothesisSteeringVector"]:
        """Retrieve a HypothesisSteeringVector by UID.

        Args:
            uid: Steering vector unique identifier

        Returns:
            HypothesisSteeringVector instance or None if not found
        """
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        cypher = """
        MATCH (sv:HypothesisSteeringVector {uid: $uid})
        RETURN sv.uid as uid, sv.hypothesis_uid as hypothesis_uid,
               sv.cognitive_operation as cognitive_operation,
               sv.vector_data as vector_data, sv.layer as layer,
               sv.effectiveness_score as effectiveness_score,
               sv.application_count as application_count,
               sv.verified_count as verified_count,
               sv.falsified_count as falsified_count,
               sv.measured_capacity as measured_capacity,
               sv.created_at as created_at, sv.last_applied as last_applied
        """
        results = await self.query(cypher, {"uid": uid})
        if not results:
            return None

        r = results[0]

        # Parse vector_data from JSON string
        vector_data = json.loads(r["vector_data"]) if r.get("vector_data") else []

        # Parse timestamps
        try:
            created_at = datetime.fromisoformat(r["created_at"])
        except (ValueError, KeyError, TypeError):
            created_at = datetime.now(timezone.utc)

        last_applied = None
        if r.get("last_applied"):
            try:
                last_applied = datetime.fromisoformat(r["last_applied"])
            except (ValueError, TypeError):
                pass

        return HypothesisSteeringVector(
            uid=r["uid"],
            hypothesis_uid=r["hypothesis_uid"],
            cognitive_operation=r.get("cognitive_operation") or "",
            vector_data=vector_data,
            layer=r.get("layer", 0),
            effectiveness_score=r.get("effectiveness_score", 0.5),
            application_count=r.get("application_count", 0),
            verified_count=r.get("verified_count", 0),
            falsified_count=r.get("falsified_count", 0),
            measured_capacity=r.get("measured_capacity", 2.0),
            created_at=created_at,
            last_applied=last_applied,
        )

    async def delete_hypothesis_steering_vector(self, uid: str) -> bool:
        """Delete a HypothesisSteeringVector by UID.

        Also removes any relationships to hypotheses.

        Args:
            uid: Steering vector unique identifier

        Returns:
            True if deleted successfully
        """
        cypher = """
        MATCH (sv:HypothesisSteeringVector {uid: $uid})
        DETACH DELETE sv
        """
        try:
            affected = await self.execute(cypher, {"uid": uid})
            return affected > 0
        except Exception as e:
            logger.warning(
                f"Failed to delete HypothesisSteeringVector {uid}: {e}"
            )
            return False

    async def get_vectors_for_hypothesis(
        self, hypothesis_uid: str
    ) -> list["HypothesisSteeringVector"]:
        """Get all steering vectors associated with a hypothesis.

        Args:
            hypothesis_uid: UID of the hypothesis

        Returns:
            List of HypothesisSteeringVector instances
        """
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        cypher = """
        MATCH (sv:HypothesisSteeringVector {hypothesis_uid: $hypothesis_uid})
        RETURN sv.uid as uid, sv.hypothesis_uid as hypothesis_uid,
               sv.cognitive_operation as cognitive_operation,
               sv.vector_data as vector_data, sv.layer as layer,
               sv.effectiveness_score as effectiveness_score,
               sv.application_count as application_count,
               sv.verified_count as verified_count,
               sv.falsified_count as falsified_count,
               sv.measured_capacity as measured_capacity,
               sv.created_at as created_at, sv.last_applied as last_applied
        ORDER BY sv.created_at DESC
        """
        results = await self.query(cypher, {"hypothesis_uid": hypothesis_uid})

        vectors = []
        for r in results:
            # Parse vector_data from JSON string
            vector_data = json.loads(r["vector_data"]) if r.get("vector_data") else []

            # Parse timestamps
            try:
                created_at = datetime.fromisoformat(r["created_at"])
            except (ValueError, KeyError, TypeError):
                created_at = datetime.now(timezone.utc)

            last_applied = None
            if r.get("last_applied"):
                try:
                    last_applied = datetime.fromisoformat(r["last_applied"])
                except (ValueError, TypeError):
                    pass

            vectors.append(
                HypothesisSteeringVector(
                    uid=r["uid"],
                    hypothesis_uid=r["hypothesis_uid"],
                    cognitive_operation=r.get("cognitive_operation") or "",
                    vector_data=vector_data,
                    layer=r.get("layer", 0),
                    effectiveness_score=r.get("effectiveness_score", 0.5),
                    application_count=r.get("application_count", 0),
                    verified_count=r.get("verified_count", 0),
                    falsified_count=r.get("falsified_count", 0),
                    measured_capacity=r.get("measured_capacity", 2.0),
                    created_at=created_at,
                    last_applied=last_applied,
                )
            )
        return vectors

    async def get_effective_hypothesis_vectors(
        self,
        min_effectiveness: float = 0.4,
        min_applications: int = 3,
        limit: int = 5,
    ) -> list["HypothesisSteeringVector"]:
        """Retrieve hypothesis vectors proven effective through verification.

        Queries for steering vectors that meet minimum effectiveness and
        application thresholds, ordered by effectiveness descending.
        These vectors represent learned steering directions that have
        demonstrated positive outcomes through the hypothesis-prediction-
        verification feedback loop.

        Args:
            min_effectiveness: Minimum effectiveness_score (0.0-1.0).
                Default 0.4 filters for vectors that work more often than not.
            min_applications: Minimum application_count required.
                Default 3 ensures enough data to evaluate effectiveness.
            limit: Maximum number of vectors to return.
                Default 5 to avoid over-steering.

        Returns:
            List of effective HypothesisSteeringVector objects, ordered by
            effectiveness descending.
        """
        from core.steering.hypothesis_vectors import HypothesisSteeringVector

        cypher = """
        MATCH (sv:HypothesisSteeringVector)
        WHERE sv.effectiveness_score >= $min_eff
          AND sv.application_count >= $min_apps
        RETURN sv.uid as uid, sv.hypothesis_uid as hypothesis_uid,
               sv.cognitive_operation as cognitive_operation,
               sv.vector_data as vector_data, sv.layer as layer,
               sv.effectiveness_score as effectiveness_score,
               sv.application_count as application_count,
               sv.verified_count as verified_count,
               sv.falsified_count as falsified_count,
               sv.measured_capacity as measured_capacity,
               sv.created_at as created_at, sv.last_applied as last_applied
        ORDER BY sv.effectiveness_score DESC
        LIMIT $limit
        """

        results = await self.query(
            cypher,
            {"min_eff": min_effectiveness, "min_apps": min_applications, "limit": limit},
        )

        vectors = []
        for r in results:
            # Parse vector_data from JSON string
            vector_data = json.loads(r["vector_data"]) if r.get("vector_data") else []

            # Parse timestamps
            try:
                created_at = datetime.fromisoformat(r["created_at"]) if r.get("created_at") else datetime.now(timezone.utc)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid created_at timestamp for vector {r.get('uid')}: {e}, using current time"
                )
                created_at = datetime.now(timezone.utc)

            last_applied = None
            if r.get("last_applied"):
                try:
                    last_applied = datetime.fromisoformat(r["last_applied"])
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid last_applied timestamp for vector {r.get('uid')}: {e}"
                    )

            vectors.append(
                HypothesisSteeringVector(
                    uid=r["uid"],
                    hypothesis_uid=r["hypothesis_uid"],
                    cognitive_operation=r.get("cognitive_operation") or "",
                    vector_data=vector_data,
                    layer=r.get("layer", 0),
                    effectiveness_score=r.get("effectiveness_score", 0.5),
                    application_count=r.get("application_count", 0),
                    verified_count=r.get("verified_count", 0),
                    falsified_count=r.get("falsified_count", 0),
                    measured_capacity=r.get("measured_capacity", 2.0),
                    created_at=created_at,
                    last_applied=last_applied,
                )
            )

        return vectors

    async def update_hypothesis_vector_effectiveness(
        self,
        hypothesis_uid: str,
        delta: float,
    ) -> bool:
        """Update hypothesis vector effectiveness using delta adjustment.

        Adjusts the effectiveness_score by delta, clamping to [0.0, 1.0].
        Also increments application_count to track usage. This is called
        after prediction verification to update the vector's track record.

        Args:
            hypothesis_uid: The hypothesis UID whose vector should be updated
            delta: Effectiveness change (typically +0.1 for verified, -0.05 for falsified)

        Returns:
            True if update succeeded, False otherwise
        """
        cypher = """
        MATCH (sv:HypothesisSteeringVector {hypothesis_uid: $hyp_uid})
        SET sv.effectiveness_score = CASE
            WHEN sv.effectiveness_score + $delta > 1.0 THEN 1.0
            WHEN sv.effectiveness_score + $delta < 0.0 THEN 0.0
            ELSE sv.effectiveness_score + $delta
        END,
        sv.application_count = sv.application_count + 1
        """
        try:
            affected = await self.execute(cypher, {"hyp_uid": hypothesis_uid, "delta": delta})
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update hypothesis vector effectiveness: {e}")
            return False

    async def link_hypothesis_to_steering_vector(
        self,
        hypothesis_uid: str,
        steering_vector_uid: str,
    ) -> bool:
        """Link a hypothesis to its steering vector.

        Updates the hypothesis node with the steering_vector_uid so that
        the verifier can find the associated vector when updating effectiveness.

        Args:
            hypothesis_uid: UID of the hypothesis
            steering_vector_uid: UID of the HypothesisSteeringVector

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $hyp_uid})
        SET h.steering_vector_uid = $sv_uid
        """
        try:
            affected = await self.execute(
                cypher,
                {"hyp_uid": hypothesis_uid, "sv_uid": steering_vector_uid}
            )
            return affected > 0
        except Exception as e:
            logger.warning(
                f"Failed to link hypothesis {hypothesis_uid} to "
                f"steering vector {steering_vector_uid}: {e}"
            )
            return False

    async def update_hypothesis_status(
        self, uid: str, status: str
    ) -> bool:
        """Update hypothesis status.

        Used when abandoning hypotheses whose vectors are pruned.

        Args:
            uid: Hypothesis UID
            status: New status value (e.g., 'abandoned')

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.status = $status
        """
        try:
            affected = await self.execute(cypher, {"uid": uid, "status": status})
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update hypothesis status {uid}: {e}")
            return False

    async def update_hypothesis_last_follow_up_cycle(
        self, uid: str, cycle: int
    ) -> bool:
        """Update hypothesis last_follow_up_cycle for cooldown tracking.

        Used to prevent the same hypothesis from triggering follow-up
        simulation in consecutive cycles.

        Args:
            uid: Hypothesis UID
            cycle: Current cognitive cycle

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.last_follow_up_cycle = $cycle
        """
        try:
            affected = await self.execute(cypher, {"uid": uid, "cycle": cycle})
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update hypothesis last_follow_up_cycle {uid}: {e}")
            return False

    # ==========================================
    # Cycle Metrics Methods
    # ==========================================

    async def save_cycle_metrics(
        self,
        cycle: int,
        metrics: dict,
    ) -> bool:
        """Persist a MetricsSnapshot for a cognitive cycle.

        Creates a CycleMetrics node that stores graph and semantic entropy
        metrics for a given cycle. Used by hypothesis predictions with
        METRIC_THRESHOLD conditions.

        Args:
            cycle: Cognitive cycle number
            metrics: MetricsSnapshot.to_dict() data

        Returns:
            True if saved successfully
        """
        cypher = """
        MERGE (m:CycleMetrics {cycle: $cycle})
        SET m.timestamp = $timestamp,
            m.structural_entropy = $structural_entropy,
            m.cluster_entropy = $cluster_entropy,
            m.orphan_rate = $orphan_rate,
            m.hub_concentration = $hub_concentration,
            m.node_count = $node_count,
            m.edge_count = $edge_count,
            m.cluster_count = $cluster_count,
            m.semantic_entropy = $semantic_entropy,
            m.topic_concentration = $topic_concentration,
            m.effective_dimensions = $effective_dimensions,
            m.discovery_parameter = $discovery_parameter,
            m.metrics_json = $metrics_json
        """
        try:
            await self.execute(
                cypher,
                {
                    "cycle": cycle,
                    "timestamp": metrics.get("timestamp", ""),
                    "structural_entropy": metrics.get("structural_entropy", 0.0),
                    "cluster_entropy": metrics.get("cluster_entropy", 0.0),
                    "orphan_rate": metrics.get("orphan_rate", 0.0),
                    "hub_concentration": metrics.get("hub_concentration", 0.0),
                    "node_count": metrics.get("node_count", 0),
                    "edge_count": metrics.get("edge_count", 0),
                    "cluster_count": metrics.get("cluster_count", 0),
                    "semantic_entropy": metrics.get("semantic_entropy", 0.0),
                    "topic_concentration": metrics.get("topic_concentration", 0.0),
                    "effective_dimensions": metrics.get("effective_dimensions", 0.0),
                    "discovery_parameter": metrics.get("discovery_parameter", 0.0),
                    "metrics_json": json.dumps(metrics),
                },
            )
            logger.debug(f"Saved cycle metrics for cycle {cycle}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save cycle metrics for cycle {cycle}: {e}")
            return False

    async def get_cycle_metrics(self, cycle: int) -> dict | None:
        """Retrieve MetricsSnapshot for a specific cycle.

        Args:
            cycle: Cognitive cycle number

        Returns:
            Metrics dict or None if not found
        """
        cypher = """
        MATCH (m:CycleMetrics {cycle: $cycle})
        RETURN m.metrics_json as metrics_json
        """
        try:
            result = await self.query(cypher, {"cycle": cycle})
            if result and result[0].get("metrics_json"):
                return json.loads(result[0]["metrics_json"])
            return None
        except Exception as e:
            logger.warning(f"Failed to get cycle metrics for cycle {cycle}: {e}")
            return None

    async def get_recent_cycle_metrics(self, limit: int = 10) -> list[dict]:
        """Retrieve recent cycle metrics for trend analysis.

        Args:
            limit: Maximum number of recent cycles to return

        Returns:
            List of metrics dicts, ordered by cycle descending
        """
        cypher = """
        MATCH (m:CycleMetrics)
        RETURN m.cycle as cycle, m.metrics_json as metrics_json
        ORDER BY m.cycle DESC
        LIMIT $limit
        """
        try:
            result = await self.query(cypher, {"limit": limit})
            metrics = []
            for row in result:
                if row.get("metrics_json"):
                    data = json.loads(row["metrics_json"])
                    data["cycle"] = row.get("cycle")
                    metrics.append(data)
            return metrics
        except Exception as e:
            logger.warning(f"Failed to get recent cycle metrics: {e}")
            return []

    async def update_cycle_metrics(
        self,
        cycle_number: int,
        metrics: dict[str, float | int | None],
    ) -> bool:
        """Update or create CycleMetrics node with actual metric values.

        This method uses MERGE to create/update a CycleMetrics node by cycle_number,
        persisting all metrics fields including concept_count, zettel_count,
        thought_length, and sae_feature_count which are needed for trend analysis
        and METRIC_THRESHOLD prediction verification.

        Args:
            cycle_number: The cognitive cycle number these metrics are for
            metrics: Dict of metric names to values. Missing fields will be set to None.

        Returns:
            True if saved successfully, False on error.
        """
        from datetime import timezone

        cypher = """
        MERGE (cm:CycleMetrics {cycle_number: $cycle_number})
        SET cm.structural_entropy = $structural_entropy,
            cm.semantic_entropy = $semantic_entropy,
            cm.hub_concentration = $hub_concentration,
            cm.edge_count = $edge_count,
            cm.concept_count = $concept_count,
            cm.zettel_count = $zettel_count,
            cm.discovery_parameter = $discovery_parameter,
            cm.thought_length = $thought_length,
            cm.sae_feature_count = $sae_feature_count,
            cm.updated_at = $updated_at
        """

        params = {
            "cycle_number": cycle_number,
            "structural_entropy": metrics.get("structural_entropy"),
            "semantic_entropy": metrics.get("semantic_entropy"),
            "hub_concentration": metrics.get("hub_concentration"),
            "edge_count": metrics.get("edge_count"),
            "concept_count": metrics.get("concept_count"),
            "zettel_count": metrics.get("zettel_count"),
            "discovery_parameter": metrics.get("discovery_parameter"),
            "thought_length": metrics.get("thought_length"),
            "sae_feature_count": metrics.get("sae_feature_count"),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

        try:
            await self.execute(cypher, params)
            logger.debug(f"Updated cycle metrics for cycle {cycle_number}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update cycle metrics for cycle {cycle_number}: {e}")
            return False

    # ==========================================
    # Reflexion Methods
    # ==========================================

    async def create_reflexion_entry(
        self,
        uid: str,
        cycle_number: int,
        health_assessment_json: str,
        metrics_snapshot: dict[str, float],
        modifications_count: int,
        narrative_summary: str,
    ) -> int:
        """Create a ReflexionEntry node.

        Args:
            uid: Unique identifier for this entry
            cycle_number: Cognitive cycle number
            health_assessment_json: Serialized HealthAssessment
            metrics_snapshot: Key metrics at time of reflexion
            modifications_count: Number of parameters modified
            narrative_summary: Human-readable summary of reflexion

        Returns:
            Count of created nodes (1 on success, 0 on failure)
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        CREATE (e:ReflexionEntry {
            uid: $uid,
            cycle_number: $cycle_number,
            timestamp: $timestamp,
            health_assessment: $health_assessment,
            metrics_snapshot: $metrics_snapshot,
            modifications_count: $modifications_count,
            narrative_summary: $narrative_summary
        })
        RETURN count(e) AS created
        """
        params = {
            "uid": _sanitize_string(uid),
            "cycle_number": cycle_number,
            "timestamp": timestamp,
            "health_assessment": _sanitize_string(health_assessment_json),
            "metrics_snapshot": json.dumps(metrics_snapshot),
            "modifications_count": modifications_count,
            "narrative_summary": _sanitize_string(narrative_summary),
        }

        try:
            affected = await self.execute(cypher, params)
            return affected
        except Exception as e:
            logger.warning(f"Failed to create reflexion entry {uid}: {e}")
            return 0

    async def get_recent_reflexion_entries(self, limit: int = 10) -> list[dict]:
        """Get recent reflexion journal entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of reflexion entry dictionaries ordered by cycle_number DESC
        """
        cypher = """
        MATCH (e:ReflexionEntry)
        RETURN e.uid as uid,
               e.cycle_number as cycle_number,
               e.timestamp as timestamp,
               e.health_assessment as health_assessment,
               e.metrics_snapshot as metrics_snapshot,
               e.baseline_comparison as baseline_comparison,
               e.phenomenological as phenomenological,
               e.modifications_count as modifications_count,
               e.overall_coherence as overall_coherence,
               e.narrative_summary as narrative_summary
        ORDER BY e.cycle_number DESC
        LIMIT $limit
        """

        try:
            results = await self.query(cypher, {"limit": limit})
            entries = []
            for r in results:
                entry = {
                    "uid": r["uid"],
                    "cycle_number": r["cycle_number"],
                    "timestamp": r.get("timestamp"),
                    "health_assessment": r.get("health_assessment"),
                    "metrics_snapshot": r.get("metrics_snapshot"),
                    "baseline_comparison": r.get("baseline_comparison"),
                    "phenomenological": r.get("phenomenological"),
                    "modifications_count": r.get("modifications_count", 0),
                    "overall_coherence": r.get("overall_coherence", 0.0),
                    "narrative_summary": r.get("narrative_summary", ""),
                }
                entries.append(entry)
            return entries
        except Exception as e:
            logger.warning(f"Failed to get recent reflexion entries: {e}")
            return []

    async def upsert_cognitive_parameter(
        self,
        path: str,
        value: Any,
        value_type: str,
        rationale: Optional[str] = None,
    ) -> int:
        """Create or update a cognitive parameter.

        Uses MERGE on path to avoid duplicates. On update, stores
        the previous value before applying the new one.

        Args:
            path: Dot-notation path (e.g., "simulation.confidence_threshold")
            value: New value to set
            value_type: Type hint ("float", "int", "str", "bool")
            rationale: Explanation for the change

        Returns:
            Count of affected properties
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Generate uid for potential creation
        import uuid as uuid_module
        uid = f"param_{uuid_module.uuid4().hex[:12]}"

        # Serialize value for storage
        value_str = json.dumps(value) if not isinstance(value, str) else value

        cypher = """
        MERGE (p:CognitiveParameter {path: $path})
        ON CREATE SET
            p.uid = $uid,
            p.value = $value,
            p.value_type = $value_type,
            p.rationale = $rationale,
            p.created_at = $timestamp,
            p.updated_at = $timestamp
        ON MATCH SET
            p.previous_value = p.value,
            p.value = $value,
            p.value_type = $value_type,
            p.rationale = $rationale,
            p.updated_at = $timestamp
        RETURN p.uid as uid, p.path as path, p.value as value,
               p.value_type as value_type, p.previous_value as previous_value,
               p.rationale as rationale
        """
        params = {
            "uid": uid,
            "path": _sanitize_string(path),
            "value": value_str,
            "value_type": value_type,
            "rationale": _sanitize_string(rationale) if rationale else None,
            "timestamp": timestamp,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected
        except Exception as e:
            logger.warning(f"Failed to upsert cognitive parameter {path}: {e}")
            return 0

    async def get_cognitive_parameter(self, path: str) -> Optional[dict]:
        """Get a cognitive parameter by path.

        Args:
            path: Dot-notation path to the parameter

        Returns:
            Parameter dict or None if not found
        """
        cypher = """
        MATCH (p:CognitiveParameter {path: $path})
        RETURN p.uid as uid, p.path as path, p.value as value,
               p.value_type as value_type, p.previous_value as previous_value,
               p.rationale as rationale, p.created_at as created_at,
               p.updated_at as updated_at
        """

        try:
            results = await self.query(cypher, {"path": path})
            if not results:
                return None

            r = results[0]
            # Deserialize value if needed
            value = r["value"]
            value_type = r.get("value_type", "str")
            if value_type in ("float", "int", "bool") and isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass

            previous_value = r.get("previous_value")
            if previous_value and value_type in ("float", "int", "bool") and isinstance(previous_value, str):
                try:
                    previous_value = json.loads(previous_value)
                except (json.JSONDecodeError, ValueError):
                    pass

            return {
                "uid": r["uid"],
                "path": r["path"],
                "value": value,
                "value_type": value_type,
                "previous_value": previous_value,
                "rationale": r.get("rationale"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }
        except Exception as e:
            logger.warning(f"Failed to get cognitive parameter {path}: {e}")
            return None

    # === Attractor Persistence Methods (Feature Substrate) ===

    async def upsert_attractor(
        self,
        uid: str,
        attractor_type: str,
        pull_strength: float,
        pull_radius: float,
        value_weight: float,
        visit_count: int,
        source_uid: str,
        source_name: str = "",
        position: Optional[list[float]] = None,
    ) -> bool:
        """Upsert an attractor node.

        Attractors are stable configurations in feature space that pull
        nearby activations toward them. They represent concepts, memories,
        moods, or emergent patterns from the Feature Substrate.

        Args:
            uid: Attractor UID
            attractor_type: Type (entity, zettel, mood, identity, emergent)
            pull_strength: How strongly it attracts (0-1)
            pull_radius: Influence radius in embedding space
            value_weight: Contribution to value signal
            visit_count: Number of times activated
            source_uid: UID of source node (Entity, InsightZettel, etc.)
            source_name: Human-readable name of the source
            position: Optional position vector in embedding space

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build SET clauses dynamically
        set_clauses_create = [
            "a.attractor_type = $type",
            "a.pull_strength = $strength",
            "a.pull_radius = $radius",
            "a.value_weight = $weight",
            "a.visit_count = $visits",
            "a.source_uid = $source_uid",
            "a.source_name = $source_name",
            "a.created_at = $timestamp",
            "a.updated_at = $timestamp",
        ]
        set_clauses_match = [
            "a.pull_strength = $strength",
            "a.pull_radius = $radius",
            "a.value_weight = $weight",
            "a.visit_count = $visits",
            "a.updated_at = $timestamp",
        ]

        params = {
            "uid": uid,
            "type": attractor_type,
            "strength": pull_strength,
            "radius": pull_radius,
            "weight": value_weight,
            "visits": visit_count,
            "source_uid": source_uid,
            "source_name": source_name,
            "timestamp": timestamp,
        }

        # Conditionally add position if provided
        if position is not None:
            set_clauses_create.append("a.position = $position")
            set_clauses_match.append("a.position = $position")
            params["position"] = position

        create_clause_str = ",\n            ".join(set_clauses_create)
        match_clause_str = ",\n            ".join(set_clauses_match)
        cypher = f"""
        MERGE (a:Attractor {{uid: $uid}})
        ON CREATE SET
            {create_clause_str}
        ON MATCH SET
            {match_clause_str}
        """

        try:
            result = await self.execute(cypher, params)
            if result > 0:
                logger.debug(f"Upserted Attractor: {uid} ({attractor_type})")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to upsert attractor {uid}: {e}")
            return False

    async def get_attractors_by_type(
        self,
        attractor_type: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get attractors of a specific type.

        Args:
            attractor_type: Type to filter by (entity, zettel, mood, identity, emergent)
            limit: Maximum results

        Returns:
            List of attractor records
        """
        cypher = """
        MATCH (a:Attractor {attractor_type: $type})
        RETURN a.uid as uid,
               a.attractor_type as attractor_type,
               a.pull_strength as pull_strength,
               a.pull_radius as pull_radius,
               a.value_weight as value_weight,
               a.visit_count as visit_count,
               a.source_uid as source_uid,
               a.source_name as source_name
        ORDER BY a.visit_count DESC
        LIMIT $limit
        """

        try:
            return await self.query(cypher, {"type": attractor_type, "limit": limit})
        except Exception as e:
            logger.warning(f"Failed to get attractors by type {attractor_type}: {e}")
            return []

    async def get_all_attractors(self, limit: int = 500) -> list[dict]:
        """Get all attractors regardless of type.

        Args:
            limit: Maximum results

        Returns:
            List of attractor records
        """
        cypher = """
        MATCH (a:Attractor)
        RETURN a.uid as uid,
               a.attractor_type as attractor_type,
               a.pull_strength as pull_strength,
               a.pull_radius as pull_radius,
               a.value_weight as value_weight,
               a.visit_count as visit_count,
               a.source_uid as source_uid,
               a.source_name as source_name
        ORDER BY a.visit_count DESC
        LIMIT $limit
        """

        try:
            return await self.query(cypher, {"limit": limit})
        except Exception as e:
            logger.warning(f"Failed to get all attractors: {e}")
            return []

    # Valid node labels for attractor linking
    _VALID_ATTRACTOR_SOURCES = frozenset({
        "Entity",
        "InsightZettel",
        "SteeringVector",
        "CrystalVector",
        "CommittedBelief",
        "Episode",
        "Attractor",
    })

    async def link_attractor_to_source(
        self,
        attractor_uid: str,
        source_uid: str,
        source_label: str,
    ) -> bool:
        """Create MANIFESTS_AS edge from source to attractor.

        This links the original knowledge graph node (Entity, InsightZettel, etc.)
        to its attractor representation in feature space.

        Args:
            attractor_uid: Attractor UID
            source_uid: Source node UID
            source_label: Source node label (Entity, InsightZettel, etc.)

        Returns:
            True if successful

        Raises:
            ValueError: If source_label is not in the allowed whitelist
        """
        # Validate source_label against whitelist to prevent injection
        if source_label not in self._VALID_ATTRACTOR_SOURCES:
            raise ValueError(
                f"Invalid source_label '{source_label}'. "
                f"Allowed: {', '.join(sorted(self._VALID_ATTRACTOR_SOURCES))}"
            )

        # Labels cannot be parameterized in Cypher, but we've validated the input
        cypher = f"""
        MATCH (s:{source_label} {{uid: $source_uid}})
        MATCH (a:Attractor {{uid: $attractor_uid}})
        MERGE (s)-[:MANIFESTS_AS]->(a)
        """

        try:
            result = await self.execute(
                cypher,
                {"source_uid": source_uid, "attractor_uid": attractor_uid},
            )
            if result > 0:
                logger.debug(
                    f"Linked {source_label}:{source_uid} -> Attractor:{attractor_uid}"
                )
                return True
            return False
        except Exception as e:
            logger.warning(
                f"Failed to link attractor {attractor_uid} to {source_label}:{source_uid}: {e}"
            )
            return False

    async def update_attractor_visit(self, uid: str) -> bool:
        """Increment visit count and update last_visited timestamp.

        Args:
            uid: Attractor UID

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MATCH (a:Attractor {uid: $uid})
        SET a.visit_count = a.visit_count + 1,
            a.last_visited = $timestamp,
            a.updated_at = $timestamp
        RETURN a.uid as uid, a.visit_count as visit_count
        """

        try:
            results = await self.query(cypher, {"uid": uid, "timestamp": timestamp})
            if results:
                logger.debug(f"Updated attractor visit: {uid} -> {results[0]['visit_count']}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to update attractor visit {uid}: {e}")
            return False

    async def delete_attractor(self, uid: str) -> bool:
        """Delete an attractor and its relationships.

        Args:
            uid: Attractor UID

        Returns:
            True if successful
        """
        cypher = """
        MATCH (a:Attractor {uid: $uid})
        WITH a, a.uid as deleted_uid
        DETACH DELETE a
        RETURN count(deleted_uid) as deleted
        """

        try:
            results = await self.query(cypher, {"uid": uid})
            if results and results[0].get("deleted", 0) > 0:
                logger.debug(f"Deleted attractor: {uid}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete attractor {uid}: {e}")
            return False

    # === Prediction Pattern Methods ===

    async def save_prediction_pattern(self, pattern: "PredictionPattern") -> bool:
        """Save or update a prediction pattern.

        Uses MERGE to create or update the pattern node for a condition type.

        Args:
            pattern: PredictionPattern instance to save

        Returns:
            True if saved successfully
        """
        condition_type = _sanitize_string(pattern.condition_type)
        cypher = """
        MERGE (p:PredictionPattern {condition_type: $condition_type})
        SET p.success_count = $success_count,
            p.failure_count = $failure_count,
            p.dominant_failure = $dominant_failure,
            p.updated_at = $updated_at
        RETURN p
        """
        params = {
            "condition_type": condition_type,
            "success_count": pattern.success_count,
            "failure_count": pattern.failure_count,
            "dominant_failure": pattern.dominant_failure,
            "updated_at": pattern.updated_at.isoformat(),
        }

        try:
            result = await self.query(cypher, params)
            return len(result) > 0
        except Exception as e:
            logger.warning(f"Failed to save PredictionPattern {condition_type}: {e}")
            return False

    async def get_prediction_pattern(
        self, condition_type: str
    ) -> "PredictionPattern | None":
        """Get pattern for a condition type.

        Args:
            condition_type: The condition type to look up

        Returns:
            PredictionPattern instance or None if not found
        """
        from core.psyche.schema import PredictionPattern

        condition_type = _sanitize_string(condition_type)
        cypher = """
        MATCH (p:PredictionPattern {condition_type: $condition_type})
        RETURN p.condition_type as condition_type,
               p.success_count as success_count,
               p.failure_count as failure_count,
               p.dominant_failure as dominant_failure,
               p.updated_at as updated_at
        """

        try:
            result = await self.query(cypher, {"condition_type": condition_type})
            if not result:
                return None
            return PredictionPattern.from_dict(result[0])
        except Exception as e:
            logger.warning(f"Failed to get PredictionPattern {condition_type}: {e}")
            return None

    # =========================================================================
    # Experiment Persistence Methods
    # =========================================================================

    async def get_active_experiments(
        self,
        domain: Optional["ExperimentDomain"] = None,
    ) -> list["Hypothesis"]:
        """Get all active experiments, optionally filtered by domain.

        Experiments are Hypothesis nodes with is_experiment=true. Active experiments
        are those in phases: pending, baseline, treatment, or washout.

        Args:
            domain: Filter to specific experiment domain

        Returns:
            List of Hypothesis objects that are experiments in active phases
        """
        from core.cognitive.simulation.schemas import Hypothesis
        from core.cognitive.experimentation.schemas import ExperimentPhase

        active_phases = [
            ExperimentPhase.PENDING.value,
            ExperimentPhase.BASELINE.value,
            ExperimentPhase.TREATMENT.value,
            ExperimentPhase.WASHOUT.value,
        ]

        if domain:
            cypher = """
            MATCH (h:Hypothesis)
            WHERE h.is_experiment = true
              AND h.experiment_domain = $domain
              AND h.current_phase IN $phases
            RETURN h
            ORDER BY h.created_at DESC
            """
            params = {"domain": domain.value, "phases": active_phases}
        else:
            cypher = """
            MATCH (h:Hypothesis)
            WHERE h.is_experiment = true
              AND h.current_phase IN $phases
            RETURN h
            ORDER BY h.created_at DESC
            """
            params = {"phases": active_phases}

        results = await self.query(cypher, params)
        experiments = []
        for r in results:
            h = r["h"]
            experiments.append(Hypothesis.from_dict(_node_to_props(h)))
        return experiments

    async def record_experiment_measurement(
        self,
        measurement: "ExperimentMeasurement",
    ) -> None:
        """Record a measurement for an experiment.

        Creates an ExperimentMeasurement node linked to the experiment hypothesis
        via HAS_MEASUREMENT relationship.

        Args:
            measurement: The measurement to persist
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $experiment_uid})
        CREATE (m:ExperimentMeasurement {
            cycle: $cycle,
            phase: $phase,
            snapshot: $snapshot,
            recorded_at: $recorded_at
        })
        CREATE (h)-[:HAS_MEASUREMENT]->(m)
        """
        await self.execute(cypher, {
            "experiment_uid": measurement.experiment_uid,
            "cycle": measurement.cycle,
            "phase": measurement.phase.value,
            "snapshot": json.dumps(measurement.snapshot.to_dict()),
            "recorded_at": measurement.recorded_at.isoformat(),
        })

    async def get_experiment_measurements(
        self,
        experiment_uid: str,
        phase: Optional["ExperimentPhase"] = None,
    ) -> list["ExperimentMeasurement"]:
        """Get measurements for an experiment.

        Args:
            experiment_uid: The experiment hypothesis UID
            phase: Optional filter by phase

        Returns:
            List of measurements ordered by cycle ascending
        """
        from core.cognitive.experimentation.schemas import (
            ExperimentMeasurement,
            ExperimentPhase,
        )
        from core.cognitive.simulation.schemas import MetricsSnapshot

        if phase:
            cypher = """
            MATCH (h:Hypothesis {uid: $uid})-[:HAS_MEASUREMENT]->(m:ExperimentMeasurement)
            WHERE m.phase = $phase
            RETURN m
            ORDER BY m.cycle ASC
            """
            params = {"uid": experiment_uid, "phase": phase.value}
        else:
            cypher = """
            MATCH (h:Hypothesis {uid: $uid})-[:HAS_MEASUREMENT]->(m:ExperimentMeasurement)
            RETURN m
            ORDER BY m.cycle ASC
            """
            params = {"uid": experiment_uid}

        results = await self.query(cypher, params)
        measurements = []
        for r in results:
            m = _node_to_props(r["m"])
            snapshot_data = m.get("snapshot", {})
            # Handle string-encoded snapshot (FalkorDB may return JSON string)
            if isinstance(snapshot_data, str):
                snapshot_data = json.loads(snapshot_data)
            measurements.append(ExperimentMeasurement(
                experiment_uid=experiment_uid,
                cycle=m["cycle"],
                phase=ExperimentPhase(m["phase"]),
                snapshot=MetricsSnapshot.from_dict(snapshot_data),
                recorded_at=_parse_datetime_field(
                    m.get("recorded_at"), "recorded_at"
                ),
            ))
        return measurements

    async def get_last_experiment_in_domain(
        self,
        domain: "ExperimentDomain",
    ) -> Optional[tuple["Hypothesis", int]]:
        """Get most recent completed/aborted experiment in a domain.

        Used to check cooldown periods between experiments in the same domain.

        Args:
            domain: The experiment domain

        Returns:
            Tuple of (hypothesis, completed_cycle) or None if no experiments found
        """
        from core.cognitive.simulation.schemas import Hypothesis
        from core.cognitive.experimentation.schemas import ExperimentPhase

        cypher = """
        MATCH (h:Hypothesis)
        WHERE h.is_experiment = true
          AND h.experiment_domain = $domain
          AND h.current_phase IN $completed_phases
        RETURN h, h.phase_start_cycle as completed_cycle
        ORDER BY completed_cycle DESC
        LIMIT 1
        """
        results = await self.query(cypher, {
            "domain": domain.value,
            "completed_phases": [
                ExperimentPhase.COMPLETE.value,
                ExperimentPhase.ABORTED.value,
            ],
        })
        if not results:
            return None
        r = results[0]
        return (Hypothesis.from_dict(_node_to_props(r["h"])), r["completed_cycle"])

    async def update_experiment_phase(
        self,
        experiment_uid: str,
        new_phase: "ExperimentPhase",
        phase_start_cycle: Optional[int] = None,
    ) -> bool:
        """Update experiment phase and optionally the phase start cycle.

        Args:
            experiment_uid: The experiment hypothesis UID
            new_phase: The new experiment phase
            phase_start_cycle: Cycle when the new phase started (optional)

        Returns:
            True if updated successfully
        """
        if phase_start_cycle is not None:
            cypher = """
            MATCH (h:Hypothesis {uid: $uid})
            SET h.current_phase = $phase,
                h.phase_start_cycle = $phase_start_cycle
            """
            params = {
                "uid": experiment_uid,
                "phase": new_phase.value,
                "phase_start_cycle": phase_start_cycle,
            }
        else:
            cypher = """
            MATCH (h:Hypothesis {uid: $uid})
            SET h.current_phase = $phase
            """
            params = {
                "uid": experiment_uid,
                "phase": new_phase.value,
            }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update experiment phase for {experiment_uid}: {e}")
            return False

    async def update_experiment_outcome(
        self,
        experiment_uid: str,
        baseline_mean: float,
        treatment_mean: float,
        effect_size: float,
        recommendation: str,
    ) -> bool:
        """Update experiment outcome after analysis.

        Called when experiment reaches COMPLETE phase with analysis results.

        Args:
            experiment_uid: The experiment hypothesis UID
            baseline_mean: Mean metric value during baseline phase
            treatment_mean: Mean metric value during treatment phase
            effect_size: Computed effect size (treatment - baseline normalized)
            recommendation: One of "ADOPT", "REJECT", "INCONCLUSIVE"

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.baseline_mean = $baseline_mean,
            h.treatment_mean = $treatment_mean,
            h.effect_size = $effect_size,
            h.recommendation = $recommendation
        """
        params = {
            "uid": experiment_uid,
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "effect_size": effect_size,
            "recommendation": recommendation,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update experiment outcome for {experiment_uid}: {e}")
            return False

    async def update_experiment_baseline_mean(
        self,
        experiment_uid: str,
        baseline_mean: float,
    ) -> bool:
        """Persist baseline mean when transitioning from BASELINE to TREATMENT.

        This ensures baseline_mean survives restarts for rollback protection.

        Args:
            experiment_uid: The experiment hypothesis UID
            baseline_mean: Mean metric value during baseline phase

        Returns:
            True if updated successfully
        """
        cypher = """
        MATCH (h:Hypothesis {uid: $uid})
        SET h.baseline_mean = $baseline_mean
        """
        params = {
            "uid": experiment_uid,
            "baseline_mean": baseline_mean,
        }

        try:
            affected = await self.execute(cypher, params)
            return affected > 0
        except Exception as e:
            logger.warning(f"Failed to update baseline_mean for {experiment_uid}: {e}")
            return False

    async def get_parameter_prediction_stats(self) -> dict[str, dict]:
        """Get prediction outcome statistics grouped by parameter.

        Aggregates prediction verification rates for parameters that have
        been the subject of predictions (via claim text analysis).

        Returns:
            Dict mapping parameter_path to stats dict with:
            - total: Total predictions about this parameter
            - verified: Number verified
            - falsified: Number falsified
        """
        # Query predictions and extract parameter references from claims
        cypher = """
        MATCH (p:Prediction)
        WHERE p.status IN ['verified', 'falsified']
        RETURN p.claim as claim, p.status as status
        """
        results = await self.query(cypher, {})

        # Use shared parameter extraction utility
        from core.cognitive.experimentation.utils import extract_parameter_from_claim

        # Build stats by parameter
        stats: dict[str, dict] = {}

        for row in results:
            claim = row.get("claim", "")
            status = row.get("status", "")

            # Find parameter reference in claim using shared utility
            parameter_path = extract_parameter_from_claim(claim)
            if not parameter_path:
                continue

            if parameter_path not in stats:
                stats[parameter_path] = {"total": 0, "verified": 0, "falsified": 0}

            stats[parameter_path]["total"] += 1
            if status == "verified":
                stats[parameter_path]["verified"] += 1
            elif status == "falsified":
                stats[parameter_path]["falsified"] += 1

        return stats

    async def get_parameter_experiment_stats(self) -> dict[str, dict]:
        """Get experiment outcome statistics grouped by parameter.

        Aggregates experiment recommendations (ADOPT/REJECT/INCONCLUSIVE)
        for each parameter that has been the subject of experiments.

        Returns:
            Dict mapping parameter_path to stats dict with:
            - total: Total experiments on this parameter
            - adopted: Number with ADOPT recommendation
            - rejected: Number with REJECT recommendation
            - inconclusive: Number with INCONCLUSIVE recommendation
        """
        cypher = """
        MATCH (h:Hypothesis)
        WHERE h.is_experiment = true
          AND h.parameter_path IS NOT NULL
          AND h.recommendation IS NOT NULL
        RETURN h.parameter_path as parameter_path,
               h.recommendation as recommendation
        """
        results = await self.query(cypher, {})

        stats: dict[str, dict] = {}

        for row in results:
            parameter_path = row.get("parameter_path", "")
            recommendation = row.get("recommendation", "")

            if not parameter_path:
                continue

            if parameter_path not in stats:
                stats[parameter_path] = {
                    "total": 0,
                    "adopted": 0,
                    "rejected": 0,
                    "inconclusive": 0,
                }

            stats[parameter_path]["total"] += 1
            if recommendation == "ADOPT":
                stats[parameter_path]["adopted"] += 1
            elif recommendation == "REJECT":
                stats[parameter_path]["rejected"] += 1
            else:
                stats[parameter_path]["inconclusive"] += 1

        return stats

    # ==========================================================================
    # INQUIRY GOAL PERSISTENCE
    # ==========================================================================

    async def create_inquiry_goal(self, goal: "InquiryGoal") -> str:
        """Create InquiryGoal node in graph, return uid.

        Args:
            goal: The InquiryGoal to persist

        Returns:
            The uid of the created node
        """
        from uuid import uuid4

        uid = str(uuid4())
        cypher = """
        CREATE (g:InquiryGoal {
            uid: $uid,
            question: $question,
            emerged_from: $emerged_from,
            stage: $stage,
            stage_cycles: $stage_cycles,
            insights: $insights,
            created_at: $created_at
        })
        RETURN g.uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "question": _sanitize_string(goal.question),
            "emerged_from": _sanitize_string(goal.emerged_from),
            "stage": goal.stage.value,
            "stage_cycles": goal.stage_cycles,
            "insights": goal.insights,
            "created_at": goal.created_at.isoformat(),
        }
        results = await self.query(cypher, params)
        if not results or not results[0].get("g.uid"):
            raise RuntimeError(f"Failed to create InquiryGoal in database for UID {uid}")
        created_uid = results[0]["g.uid"]
        logger.debug(f"Created InquiryGoal: {created_uid}")
        return created_uid

    async def update_inquiry_goal(self, uid: str, goal: "InquiryGoal") -> bool:
        """Update existing InquiryGoal.

        Args:
            uid: The uid of the goal to update
            goal: The InquiryGoal with updated fields

        Returns:
            True if update succeeded
        """
        cypher = """
        MATCH (g:InquiryGoal {uid: $uid})
        SET g.question = $question,
            g.emerged_from = $emerged_from,
            g.stage = $stage,
            g.stage_cycles = $stage_cycles,
            g.insights = $insights
        RETURN g.uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "question": _sanitize_string(goal.question),
            "emerged_from": _sanitize_string(goal.emerged_from),
            "stage": goal.stage.value,
            "stage_cycles": goal.stage_cycles,
            "insights": goal.insights,
        }
        affected = await self.execute(cypher, params)
        return affected > 0

    async def get_inquiry_goal(self, uid: str) -> "Optional[InquiryGoal]":
        """Retrieve InquiryGoal by uid.

        Args:
            uid: The uid of the goal to retrieve

        Returns:
            The InquiryGoal if found, None otherwise
        """
        from core.cognitive.goal import InquiryGoal
        from core.cognitive.stage import CognitiveStage
        from datetime import datetime

        cypher = """
        MATCH (g:InquiryGoal {uid: $uid})
        RETURN g.uid as uid, g.question as question, g.emerged_from as emerged_from,
               g.stage as stage, g.stage_cycles as stage_cycles,
               g.insights as insights, g.created_at as created_at
        """
        results = await self.query(cypher, {"uid": _sanitize_string(uid)})
        if not results:
            return None

        row = results[0]
        return InquiryGoal(
            question=row.get("question", ""),
            emerged_from=row.get("emerged_from", ""),
            stage=CognitiveStage(row.get("stage", "question")),
            stage_cycles=row.get("stage_cycles", 0),
            insights=row.get("insights", []),
            created_at=datetime.fromisoformat(row.get("created_at", datetime.now().isoformat())),
            uid=row.get("uid"),
        )

    async def complete_inquiry_goal(self, uid: str) -> bool:
        """Mark an InquiryGoal as completed.

        Args:
            uid: The uid of the goal to complete

        Returns:
            True if update succeeded
        """
        from datetime import datetime, timezone

        cypher = """
        MATCH (g:InquiryGoal {uid: $uid})
        SET g.completed_at = $completed_at
        RETURN g.uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        affected = await self.execute(cypher, params)
        return affected > 0

    # ==========================================================================
    # EPISODE PERSISTENCE
    # ==========================================================================

    async def create_episode(self, episode: "Episode") -> str:
        """Create Episode node in graph, return uid.

        Args:
            episode: The Episode to persist

        Returns:
            The uid of the created node
        """
        from uuid import uuid4

        uid = str(uuid4())
        # Convert segment types to their string values
        segments_completed = [s.value for s in episode.segments_completed]

        cypher = """
        CREATE (e:Episode {
            uid: $uid,
            episode_type: $episode_type,
            current_segment: $current_segment,
            opening_insight: $opening_insight,
            segments_completed: $segments_completed,
            started_at: $started_at,
            seed_entity: $seed_entity
        })
        RETURN e.uid AS uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "episode_type": episode.episode_type.value,
            "current_segment": episode.current_segment.value,
            "opening_insight": _sanitize_string(episode.opening_insight[:500]),  # Truncate
            "segments_completed": segments_completed,
            "started_at": episode.started_at.isoformat(),
            "seed_entity": _sanitize_string(episode.seed_entity) if episode.seed_entity else None,
        }
        results = await self.query(cypher, params)
        if not results or not results[0].get("uid"):
            raise RuntimeError(f"Failed to create Episode in database for UID {uid}")
        created_uid = results[0]["uid"]
        logger.debug(f"Created Episode: {created_uid}")
        return created_uid

    async def update_episode(self, uid: str, episode: "Episode") -> bool:
        """Update existing Episode.

        Args:
            uid: The uid of the episode to update
            episode: The Episode with updated fields

        Returns:
            True if update succeeded
        """
        segments_completed = [s.value for s in episode.segments_completed]

        cypher = """
        MATCH (e:Episode {uid: $uid})
        SET e.current_segment = $current_segment,
            e.segments_completed = $segments_completed
        RETURN e.uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "current_segment": episode.current_segment.value,
            "segments_completed": segments_completed,
        }
        affected = await self.execute(cypher, params)
        return affected > 0

    async def complete_episode(self, uid: str) -> bool:
        """Mark an Episode as completed.

        Args:
            uid: The uid of the episode to complete

        Returns:
            True if update succeeded
        """
        from datetime import datetime, timezone

        cypher = """
        MATCH (e:Episode {uid: $uid})
        SET e.completed_at = $completed_at
        RETURN e.uid
        """
        params = {
            "uid": _sanitize_string(uid),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        affected = await self.execute(cypher, params)
        return affected > 0

    # ==========================================================================
    # Autonomous Decision Methods (Phase 1: Full Operational Autonomy)
    # ==========================================================================

    async def record_autonomous_decision(self, decision: "AutonomousDecision") -> None:
        """Record an autonomous decision to the graph.

        Args:
            decision: The autonomous decision to record
        """
        from core.psyche.schema import AutonomousDecision

        cypher = """
        CREATE (d:AutonomousDecision {
            id: $id,
            timestamp: $timestamp,
            cycle_id: $cycle_id,
            cycle_count_created: $cycle_count_created,
            health_created: $health_created,
            cycle_count_assessed: $cycle_count_assessed,
            question: $question,
            knowledge_synthesized: $knowledge_synthesized,
            judgment: $judgment,
            action: $action,
            expectation: $expectation,
            outcome: $outcome,
            lesson_learned: $lesson_learned,
            success: $success,
            related_hypothesis: $related_hypothesis,
            related_experiment: $related_experiment
        })
        RETURN d
        """

        params = {
            "id": _sanitize_string(decision.id),
            "timestamp": decision.timestamp.isoformat(),
            "cycle_id": _sanitize_string(decision.cycle_id),
            "cycle_count_created": decision.cycle_count_created,
            "health_created": decision.health_created.value,
            "cycle_count_assessed": decision.cycle_count_assessed,
            "question": _sanitize_string(decision.question),
            "knowledge_synthesized": json.dumps(decision.knowledge_synthesized),
            "judgment": _sanitize_string(decision.judgment),
            "action": json.dumps(decision.action),
            "expectation": _sanitize_string(decision.expectation),
            "outcome": _sanitize_string(decision.outcome) if decision.outcome else None,
            "lesson_learned": _sanitize_string(decision.lesson_learned) if decision.lesson_learned else None,
            "success": decision.success,
            "related_hypothesis": _sanitize_string(decision.related_hypothesis) if decision.related_hypothesis else None,
            "related_experiment": _sanitize_string(decision.related_experiment) if decision.related_experiment else None,
        }

        await self.execute(cypher, params)

    async def get_autonomous_decision(self, decision_id: str) -> "AutonomousDecision | None":
        """Get an autonomous decision by ID.

        Args:
            decision_id: The decision ID

        Returns:
            AutonomousDecision if found, None otherwise
        """
        from core.psyche.schema import AutonomousDecision

        cypher = """
        MATCH (d:AutonomousDecision {id: $id})
        RETURN d
        """

        results = await self.query(cypher, {"id": decision_id})
        if not results:
            return None

        data = _node_to_props(results[0]["d"])
        return AutonomousDecision.from_dict(data)

    async def get_recent_decisions(self, limit: int = 10) -> list["AutonomousDecision"]:
        """Get recent autonomous decisions.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of autonomous decisions, most recent first
        """
        from core.psyche.schema import AutonomousDecision

        cypher = """
        MATCH (d:AutonomousDecision)
        RETURN d
        ORDER BY d.timestamp DESC
        LIMIT $limit
        """

        results = await self.query(cypher, {"limit": limit})

        decisions = []
        for record in results:
            data = _node_to_props(record["d"])
            decisions.append(AutonomousDecision.from_dict(data))

        return decisions

    async def _do_synthesis(
        self,
        question: str,
        domain: str,
        limit: int,
    ) -> list[str]:
        """Internal helper to perform knowledge synthesis queries.

        Args:
            question: The decision question
            domain: Domain context
            limit: Maximum number of items

        Returns:
            List of plain-text knowledge summaries
        """
        results: list[str] = []

        # Query 1: Recent decisions in domain (last 20 cycles)
        recent_threshold = datetime.now(timezone.utc) - timedelta(hours=24)  # ~20 cycles

        decisions_query = """
        MATCH (d:AutonomousDecision)
        WHERE d.timestamp > $recent_threshold
        RETURN d
        ORDER BY d.timestamp DESC
        LIMIT 5
        """

        decisions = await self.query(
            decisions_query,
            {"recent_threshold": recent_threshold.isoformat()}
        )

        # Format decision summaries
        for record in decisions:
            d = _node_to_props(record.get("d", {}))
            action = d.get("action", {})
            param = action.get("parameter_path", "unknown")
            timestamp = d.get("timestamp", "")
            judgment = d.get("judgment", "")

            # Calculate cycles ago (rough estimate: 1 cycle = ~2 hours)
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                cycles_ago = int((datetime.now(timezone.utc) - dt).total_seconds() / 7200)
                time_label = f"{cycles_ago} cycles ago"
            except:
                time_label = "recently"

            summary = f"[decision {time_label}] {param}: {judgment}"
            results.append(summary)

        # Query 2: Relevant zettels via semantic search
        question_embedding = await self._generate_embedding(question)

        if question_embedding:
            zettels_query = """
            CALL db.idx.vector.queryNodes('zettel_embedding', 5, $embedding)
            YIELD node, score
            WHERE score > 0.7
            RETURN node, score
            ORDER BY score DESC
            """

            zettels = await self.query(
                zettels_query,
                {"embedding": question_embedding}
            )

            # Format zettel summaries
            for record in zettels:
                z = _node_to_props(record.get("node", {}))
                content = z.get("insight_text", "")
                zettel_id = z.get("uid", "unknown")
                score = record.get("score", 0)

                # Truncate content if too long
                if len(content) > 120:
                    content = content[:117] + "..."

                summary = f"[zettel {zettel_id[:8]}] {content}"
                results.append(summary)

        # Query 3: Active experiments (conditional on domain="experimentation")
        if domain == "experimentation":
            experiments_query = """
            MATCH (h:Hypothesis)
            WHERE h.is_experiment = true
            RETURN h
            ORDER BY h.created_at DESC
            LIMIT 3
            """

            experiments = await self.query(experiments_query)

            # Format experiment summaries
            for record in experiments:
                h = _node_to_props(record.get("h", {}))
                exp_id = h.get("uid", "unknown")
                phase = h.get("current_phase", "UNKNOWN")
                claim = h.get("statement", "")

                # Truncate claim if too long
                if len(claim) > 120:
                    claim = claim[:117] + "..."

                summary = f"[experiment {exp_id[:8]} {phase}] {claim}"
                results.append(summary)

        return results[:limit]

    async def synthesize_knowledge_for_decision(
        self,
        question: str,
        domain: str,
        limit: int = 10,
        timeout_ms: int = 500,
    ) -> list[str]:
        """Synthesize relevant knowledge for an autonomous decision.

        Queries three parallel sources:
        1. Recent AutonomousDecisions in the same domain (last 20 cycles)
        2. Relevant InsightZettels (semantic similarity via embeddings)
        3. Related experiments (if domain matches)

        Args:
            question: The decision question (e.g., "Should I modify exploration.magnitude?")
            domain: Domain context ("reflexion", "steering", "experimentation")
            limit: Maximum number of knowledge items to return
            timeout_ms: Query timeout in milliseconds (default 500ms)

        Returns:
            List of plain-text summaries of relevant past decisions,
            zettels, and experiments to inform judgment.
        """
        try:
            # Wrap synthesis in timeout
            result = await asyncio.wait_for(
                self._do_synthesis(question, domain, limit),
                timeout=timeout_ms / 1000
            )
            return result

        except asyncio.TimeoutError:
            logger.warning(
                f"Knowledge synthesis timed out after {timeout_ms}ms "
                f"for question: {question[:50]}..."
            )
            return []

        except Exception as e:
            logger.warning(f"Knowledge synthesis failed: {e}")
            return []

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for semantic search.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if embedding fails
        """
        try:
            # Use existing embedding generation (assumes self._embedder exists)
            if hasattr(self, "_embedder"):
                return await self._embedder.encode(text, tier=EmbeddingTier.RETRIEVAL)
            return None
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    async def get_pending_decisions(
        self,
        current_cycle: int,
        offset: int = 10,
        limit: int = 5,
    ) -> list[AutonomousDecision]:
        """Get decisions ready for consequence assessment.

        Uses age-based querying (>= offset cycles old) instead of exact
        cycle matching for robustness to cold starts and crashes.

        Args:
            current_cycle: Current cycle number
            offset: Minimum age in cycles for assessment (default 10)
            limit: Maximum decisions to return (default 5, rate limiting)

        Returns:
            List of AutonomousDecision objects ready for assessment
        """
        max_cycle_for_assessment = current_cycle - offset
        
        query = """
        MATCH (d:AutonomousDecision)
        WHERE d.cycle_count_created <= $max_cycle
          AND d.outcome IS NULL
          AND d.cycle_count_assessed IS NULL
        RETURN d
        ORDER BY d.cycle_count_created ASC
        LIMIT $limit
        """

        try:
            result = await self.query(
                query,
                params={
                    "max_cycle": max_cycle_for_assessment,
                    "limit": limit,
                },
            )

            decisions = []
            for record in result:
                node = record["d"]
                props = _node_to_props(node)
                decision = AutonomousDecision.from_dict(props)
                decisions.append(decision)

            return decisions

        except Exception as e:
            logger.error(f"Failed to get pending decisions: {e}")
            return []

    async def get_health_at_cycle(self, cycle_id: str) -> HealthCategory | None:
        """Get health state from a specific cycle's reflexion journal.

        Args:
            cycle_id: Cycle identifier (e.g., "cycle_100")

        Returns:
            HealthCategory if journal exists, None if not found
        """
        from core.cognitive.reflexion.schemas import HealthCategory

        query = """
        MATCH (j:ReflexionJournal)
        WHERE j.cycle_id = $cycle_id
        RETURN j.health_category as health
        """

        try:
            result = await self.query(query, params={"cycle_id": cycle_id})

            if not result:
                return None

            health_str = result[0].get("health")
            if not health_str:
                return None

            return HealthCategory(health_str)

        except Exception as e:
            logger.error(f"Failed to get health at cycle {cycle_id}: {e}")
            return None

    async def update_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        lesson: str,
        success: bool,
        cycle_count_assessed: int,
    ) -> None:
        """Update an AutonomousDecision node with outcome assessment.

        Args:
            decision_id: Decision node ID
            outcome: Outcome description (e.g., "success", "failure")
            lesson: Extracted lesson text
            success: Boolean success classification
            cycle_count_assessed: Cycle number when outcome was assessed (for idempotency)
        """
        query = """
        MATCH (d:AutonomousDecision {id: $decision_id})
        SET d.outcome = $outcome,
            d.lesson_learned = $lesson,
            d.success = $success,
            d.cycle_count_assessed = $cycle_count_assessed
        RETURN d
        """

        try:
            await self.execute(
                query,
                params={
                    "decision_id": _sanitize_string(decision_id),
                    "outcome": _sanitize_string(outcome),
                    "lesson": _sanitize_string(lesson),
                    "success": success,
                    "cycle_count_assessed": cycle_count_assessed,
                },
            )
        except Exception as e:
            logger.error(f"Failed to update decision outcome: {e}")
            raise

    async def create_zettel_from_lesson(
        self,
        lesson: str,
        decision_id: str,
    ) -> str:
        """Create an InsightZettel from an autonomous decision lesson.

        Creates a zettel tagged with "autonomous-learning" and "consequence-derived",
        linked to the source decision via EMERGED_FROM relationship.

        Args:
            lesson: Lesson text to store in zettel
            decision_id: Source AutonomousDecision node ID

        Returns:
            Created zettel UID
        """
        import uuid
        from datetime import datetime, timezone

        zettel_uid = f"zettel_{uuid.uuid4().hex[:12]}"

        query = """
        CREATE (z:InsightZettel {
            uid: $uid,
            insight: $lesson,
            timestamp: $timestamp,
            tags: $tags
        })
        WITH z
        MATCH (d:AutonomousDecision {id: $decision_id})
        CREATE (z)-[:EMERGED_FROM]->(d)
        RETURN z.uid as uid
        """

        try:
            await self.execute(
                query,
                params={
                    "uid": zettel_uid,
                    "lesson": _sanitize_string(lesson),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tags": ["autonomous-learning", "consequence-derived"],
                    "decision_id": _sanitize_string(decision_id),
                },
            )

            logger.info(f"Created zettel {zettel_uid} from decision {decision_id}")
            return zettel_uid

        except Exception as e:
            logger.error(f"Failed to create zettel from lesson: {e}")
            raise

    async def create_heuristic_zettel(
        self,
        content: str,
        evidence: list[str],
        action_type: str,
    ) -> str:
        """Create InsightZettel tagged as self-learned-heuristic.

        Args:
            content: Heuristic text describing the pattern
            evidence: Decision IDs that support this heuristic
            action_type: Type of action this heuristic applies to

        Returns:
            ID of created zettel
        """
        import uuid
        from datetime import datetime, timezone

        # Generate zettel ID with proper format (zettel_ prefix + hex chars)
        zettel_id = f"zettel_{uuid.uuid4().hex[:ZETTEL_ID_HEX_LENGTH]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Sanitize string parameters
        zettel_id = _sanitize_string(zettel_id)
        content = _sanitize_string(content)
        timestamp = _sanitize_string(timestamp)
        evidence = [_sanitize_string(eid) for eid in evidence]

        # Create zettel with self-learned-heuristic and action_type tags
        query = """
        CREATE (z:InsightZettel {
            uid: $zettel_id,
            insight_text: $content,
            source_type: 'REFLECTION',
            source_uid: '',
            concept: $action_type,
            created_at: $timestamp,
            tags: $tags
        })
        WITH z
        UNWIND $evidence AS decision_id
        MATCH (d:AutonomousDecision {id: decision_id})
        CREATE (z)-[:BASED_ON]->(d)
        RETURN z.uid AS uid
        """

        params = {
            "zettel_id": zettel_id,
            "content": content,
            "action_type": action_type,
            "timestamp": timestamp,
            "tags": ["self-learned-heuristic", action_type],
            "evidence": evidence,
        }

        try:
            await self.execute(query, params)
            logger.info(f"Created heuristic zettel {zettel_id} from {len(evidence)} decisions")
            return zettel_id
        except Exception as e:
            logger.error(f"Failed to create heuristic zettel: {e}")
            raise


    # === Individuation Dynamics ===

    async def upsert_identity_trajectory(
        self,
        uid: str,
        element_id: str,
        element_type: str,
        current_position: float,
        current_velocity: float,
        current_acceleration: float,
        current_phase: str,
        phase_stability: int,
        observation_count: int,
    ) -> bool:
        """Upsert an identity trajectory node.

        Tracks how identity elements (commitments, values, beliefs) evolve
        over time - velocity, acceleration, phase dynamics.

        Args:
            uid: Trajectory UID
            element_id: UID of the tracked element
            element_type: Type of element (commitment, value, belief)
            current_position: Current confidence/strength value
            current_velocity: Rate of change
            current_acceleration: Change in rate
            current_phase: Dynamics phase (nascent, stable, crystallizing, etc.)
            phase_stability: Cycles in current phase
            observation_count: Total observations

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (t:IdentityTrajectory {uid: $uid})
        ON CREATE SET
            t.created_at = $timestamp
        SET
            t.element_id = $element_id,
            t.element_type = $element_type,
            t.current_position = $position,
            t.current_velocity = $velocity,
            t.current_acceleration = $acceleration,
            t.current_phase = $phase,
            t.phase_stability = $stability,
            t.observation_count = $obs_count,
            t.updated_at = $timestamp
        """

        params = {
            "uid": uid,
            "element_id": element_id,
            "element_type": element_type,
            "position": current_position,
            "velocity": current_velocity,
            "acceleration": current_acceleration,
            "phase": current_phase,
            "stability": phase_stability,
            "obs_count": observation_count,
            "timestamp": timestamp,
        }

        try:
            result = await self.execute(cypher, params)
            if result > 0:
                logger.debug(f"Upserted IdentityTrajectory: {uid} ({element_type})")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to upsert identity trajectory {uid}: {e}")
            return False

    async def get_identity_trajectory(
        self, element_id: str
    ) -> Optional[dict]:
        """Get trajectory for an identity element.

        Args:
            element_id: UID of the tracked element

        Returns:
            Trajectory data dict or None if not found
        """
        cypher = """
        MATCH (t:IdentityTrajectory {element_id: $element_id})
        RETURN t
        """
        result = await self.query(cypher, {"element_id": element_id})
        if result:
            return _node_to_props(result[0]["t"])
        return None

    async def get_transforming_trajectories(self) -> list[dict]:
        """Get all trajectories in transforming phases.

        Returns trajectories that are crystallizing, dissolving, or volatile.
        """
        cypher = """
        MATCH (t:IdentityTrajectory)
        WHERE t.current_phase IN ['crystallizing', 'dissolving', 'volatile']
        RETURN t
        ORDER BY ABS(t.current_velocity) DESC
        LIMIT 20
        """
        result = await self.query(cypher, {})
        return [_node_to_props(r["t"]) for r in result]

    async def upsert_attractor_basin(
        self,
        uid: str,
        center_state_json: str,
        radius: float,
        visit_count: int,
        total_dwell_time: float,
        strength: float,
        element_ids_json: str,
    ) -> bool:
        """Upsert an attractor basin node.

        Attractor basins are stable configurations in identity space -
        states Lilly tends to return to.

        Args:
            uid: Basin UID
            center_state_json: JSON string of element_id -> confidence mapping
            radius: Basin radius
            visit_count: Number of visits
            total_dwell_time: Total time spent in basin (seconds)
            strength: Computed strength (from visits and dwell time)
            element_ids_json: JSON string of element IDs that define this basin

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (b:AttractorBasin {uid: $uid})
        ON CREATE SET
            b.formation_time = $timestamp
        SET
            b.center_state_json = $center_state,
            b.radius = $radius,
            b.visit_count = $visits,
            b.total_dwell_time = $dwell,
            b.strength = $strength,
            b.element_ids_json = $elements,
            b.last_visit = $timestamp
        """

        params = {
            "uid": uid,
            "center_state": center_state_json,
            "radius": radius,
            "visits": visit_count,
            "dwell": total_dwell_time,
            "strength": strength,
            "elements": element_ids_json,
            "timestamp": timestamp,
        }

        try:
            result = await self.execute(cypher, params)
            if result > 0:
                logger.debug(f"Upserted AttractorBasin: {uid}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to upsert attractor basin {uid}: {e}")
            return False

    async def get_attractor_basins(self, limit: int = 10) -> list[dict]:
        """Get strongest attractor basins.

        Args:
            limit: Maximum number of basins to return

        Returns:
            List of basin data dicts sorted by strength
        """
        cypher = """
        MATCH (b:AttractorBasin)
        RETURN b
        ORDER BY b.strength DESC
        LIMIT $limit
        """
        result = await self.query(cypher, {"limit": limit})
        return [_node_to_props(r["b"]) for r in result]

    async def create_individuation_transition(
        self,
        uid: str,
        from_phase: str,
        to_phase: str,
        trigger: str,
        trigger_element: str,
        affected_elements_json: str,
        energy_released: float,
        narrative: str,
    ) -> bool:
        """Create a phase transition record.

        Phase transitions mark significant shifts in identity dynamics -
        moments when how identity evolves changes, not just the content.

        Args:
            uid: Transition UID
            from_phase: Previous phase
            to_phase: New phase
            trigger: What caused this transition
            trigger_element: Element ID that triggered the shift
            affected_elements_json: JSON string of affected element IDs
            energy_released: Magnitude of the change
            narrative: Human-readable description

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        cypher = """
        CREATE (t:IndividuationTransition {
            uid: $uid,
            from_phase: $from_phase,
            to_phase: $to_phase,
            trigger: $trigger,
            trigger_element: $trigger_elem,
            affected_elements_json: $affected,
            energy_released: $energy,
            narrative: $narrative,
            timestamp: $timestamp
        })
        """

        params = {
            "uid": uid,
            "from_phase": from_phase,
            "to_phase": to_phase,
            "trigger": trigger,
            "trigger_elem": trigger_element,
            "affected": affected_elements_json,
            "energy": energy_released,
            "narrative": _sanitize_string(narrative),
            "timestamp": timestamp,
        }

        try:
            result = await self.execute(cypher, params)
            if result > 0:
                logger.debug(f"Created IndividuationTransition: {uid}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to create individuation transition {uid}: {e}")
            return False

    async def get_recent_individuation_transitions(
        self, limit: int = 10
    ) -> list[dict]:
        """Get recent phase transitions.

        Args:
            limit: Maximum number of transitions to return

        Returns:
            List of transition data dicts, most recent first
        """
        cypher = """
        MATCH (t:IndividuationTransition)
        RETURN t
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """
        result = await self.query(cypher, {"limit": limit})
        return [_node_to_props(r["t"]) for r in result]

    async def link_trajectory_to_commitment(
        self, trajectory_uid: str, commitment_uid: str
    ) -> bool:
        """Link a trajectory node to its source commitment.

        Args:
            trajectory_uid: UID of the trajectory
            commitment_uid: UID of the commitment

        Returns:
            True if successful
        """
        cypher = """
        MATCH (t:IdentityTrajectory {uid: $traj_uid})
        MATCH (c:Commitment {uid: $comm_uid})
        MERGE (t)-[:TRACKS]->(c)
        """
        try:
            await self.execute(
                cypher, {"traj_uid": trajectory_uid, "comm_uid": commitment_uid}
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to link trajectory {trajectory_uid} to commitment: {e}"
            )
            return False

    async def link_transition_to_trajectory(
        self, transition_uid: str, trajectory_uid: str
    ) -> bool:
        """Link a transition to the trajectory that triggered it.

        Args:
            transition_uid: UID of the transition
            trajectory_uid: UID of the trajectory

        Returns:
            True if successful
        """
        cypher = """
        MATCH (trans:IndividuationTransition {uid: $trans_uid})
        MATCH (traj:IdentityTrajectory {uid: $traj_uid})
        MERGE (trans)-[:TRIGGERED_BY]->(traj)
        """
        try:
            await self.execute(
                cypher, {"trans_uid": transition_uid, "traj_uid": trajectory_uid}
            )
            return True
        except Exception as e:
            logger.warning(
                f"Failed to link transition {transition_uid} to trajectory: {e}"
            )
            return False

    async def save_individuation_dynamics(
        self, dynamics_dict: dict
    ) -> bool:
        """Save full IndividuationDynamics state.

        Stores the complete dynamics state as a JSON blob for fast
        reload. Individual nodes are also created for graph queries.

        Args:
            dynamics_dict: Serialized IndividuationDynamics state

        Returns:
            True if successful
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        state_json = json.dumps(dynamics_dict)

        cypher = """
        MERGE (d:IndividuationDynamicsState {id: 'current'})
        SET d.state_json = $state,
            d.updated_at = $timestamp
        """

        try:
            await self.execute(
                cypher, {"state": state_json, "timestamp": timestamp}
            )
            logger.debug("Saved IndividuationDynamics state")
            return True
        except Exception as e:
            logger.warning(f"Failed to save individuation dynamics: {e}")
            return False

    async def load_individuation_dynamics(self) -> Optional[dict]:
        """Load IndividuationDynamics state.

        Returns:
            Deserialized dynamics state dict, or None if not found
        """
        cypher = """
        MATCH (d:IndividuationDynamicsState {id: 'current'})
        RETURN d.state_json AS state
        """

        try:
            result = await self.query(cypher, {})
            if result and result[0].get("state"):
                return json.loads(result[0]["state"])
            return None
        except Exception as e:
            logger.warning(f"Failed to load individuation dynamics: {e}")
            return None

    # === Cognitive Anchor Methods ===

    async def upsert_cognitive_anchor(
        self,
        anchor: "CognitiveAnchor",
    ) -> bool:
        """Persist or update a cognitive anchor.

        Uses MERGE on uid for idempotent upserts. Stores embedding
        as a JSON-encoded list for retrieval.

        Args:
            anchor: CognitiveAnchor model to persist

        Returns:
            True if successful, False otherwise
        """
        from core.psyche.schema import CognitiveAnchor  # noqa: F811

        timestamp = datetime.now(timezone.utc).isoformat()

        # Serialize embedding as JSON list
        embedding_json = (
            json.dumps(anchor.embedding) if anchor.embedding else None
        )

        cypher = """
        MERGE (a:CognitiveAnchor {uid: $uid})
        ON CREATE SET
            a.mode_name = $mode_name,
            a.anchor_text = $anchor_text,
            a.description = $description,
            a.embedding = $embedding,
            a.is_predefined = $is_predefined,
            a.discovered_at = $discovered_at,
            a.discovery_cycle = $discovery_cycle,
            a.source_thought_uid = $source_thought_uid,
            a.usage_count = $usage_count,
            a.confidence = $confidence,
            a.retired = $retired,
            a.created_at = $timestamp
        ON MATCH SET
            a.mode_name = $mode_name,
            a.anchor_text = $anchor_text,
            a.description = $description,
            a.embedding = $embedding,
            a.usage_count = $usage_count,
            a.confidence = $confidence,
            a.retired = $retired
        RETURN a.uid as uid
        """
        params = {
            "uid": anchor.uid,
            "mode_name": _sanitize_string(anchor.mode_name),
            "anchor_text": _sanitize_string(anchor.anchor_text),
            "description": _sanitize_string(anchor.description),
            "embedding": embedding_json,
            "is_predefined": anchor.is_predefined,
            "discovered_at": (
                anchor.discovered_at.isoformat() if anchor.discovered_at else None
            ),
            "discovery_cycle": anchor.discovery_cycle,
            "source_thought_uid": anchor.source_thought_uid,
            "usage_count": anchor.usage_count,
            "confidence": anchor.confidence,
            "retired": anchor.retired,
            "timestamp": timestamp,
        }

        try:
            result = await self.query(cypher, params)
            return bool(result)
        except Exception as e:
            logger.warning(f"Failed to upsert cognitive anchor {anchor.uid}: {e}")
            return False

    async def get_emergent_anchors(
        self,
        include_retired: bool = False,
    ) -> list["CognitiveAnchor"]:
        """Retrieve all discovered (emergent) anchors.

        Returns anchors that were discovered through orphan clustering,
        not the predefined 10 modes.

        Args:
            include_retired: If True, include retired anchors

        Returns:
            List of CognitiveAnchor models
        """
        from core.psyche.schema import CognitiveAnchor

        where_clause = "WHERE a.is_predefined = false"
        if not include_retired:
            where_clause += " AND (a.retired IS NULL OR a.retired = false)"

        cypher = f"""
        MATCH (a:CognitiveAnchor)
        {where_clause}
        RETURN a.uid as uid, a.mode_name as mode_name, a.anchor_text as anchor_text,
               a.description as description, a.embedding as embedding,
               a.is_predefined as is_predefined, a.discovered_at as discovered_at,
               a.discovery_cycle as discovery_cycle, a.source_thought_uid as source_thought_uid,
               a.usage_count as usage_count, a.confidence as confidence, a.retired as retired,
               a.created_at as created_at
        ORDER BY a.usage_count DESC, a.confidence DESC
        """

        try:
            results = await self.query(cypher, {})
            anchors = []
            for r in results:
                # Parse embedding from JSON
                embedding = None
                if r.get("embedding"):
                    try:
                        embedding = json.loads(r["embedding"])
                    except json.JSONDecodeError:
                        pass

                anchor = CognitiveAnchor(
                    uid=r["uid"],
                    mode_name=r.get("mode_name", ""),
                    anchor_text=r.get("anchor_text", ""),
                    description=r.get("description", ""),
                    embedding=embedding,
                    is_predefined=r.get("is_predefined", False),
                    discovered_at=_parse_datetime_field(r.get("discovered_at")),
                    discovery_cycle=r.get("discovery_cycle"),
                    source_thought_uid=r.get("source_thought_uid"),
                    usage_count=r.get("usage_count", 0),
                    confidence=r.get("confidence", 1.0),
                    retired=r.get("retired", False),
                    created_at=_parse_datetime_field(r.get("created_at")),
                )
                anchors.append(anchor)
            return anchors
        except Exception as e:
            logger.warning(f"Failed to get emergent anchors: {e}")
            return []

    async def increment_anchor_usage(self, uid: str) -> None:
        """Bump usage count when this mode is dominant.

        Also updates confidence using EMA: confidence = 0.95 * old + 0.05 * 1.0

        Args:
            uid: The anchor UID to update
        """
        cypher = """
        MATCH (a:CognitiveAnchor {uid: $uid})
        SET a.usage_count = COALESCE(a.usage_count, 0) + 1,
            a.confidence = 0.95 * COALESCE(a.confidence, 0.5) + 0.05
        RETURN a.uid as uid, a.usage_count as usage_count
        """

        try:
            await self.query(cypher, {"uid": uid})
        except Exception as e:
            logger.warning(f"Failed to increment anchor usage for {uid}: {e}")

    async def retire_anchor(self, uid: str) -> bool:
        """Mark an anchor as retired due to low usage.

        Retired anchors are not deleted but excluded from similarity
        computation by default.

        Args:
            uid: The anchor UID to retire

        Returns:
            True if successful
        """
        cypher = """
        MATCH (a:CognitiveAnchor {uid: $uid})
        SET a.retired = true
        RETURN a.uid as uid
        """

        try:
            result = await self.query(cypher, {"uid": uid})
            if result:
                logger.info(f"Retired cognitive anchor: {uid}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to retire anchor {uid}: {e}")
            return False
