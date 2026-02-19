"""Semantic Intuition Bank - Phase 2 of Integrated Identity Layer.

Manages consolidated knowledge as steering vectors. Knowledge that has been
"baked in" during dream cycles becomes intrinsic to reasoning.

Purpose: Transform learned patterns into intuitions. "Ryan prefers brevity"
stops being retrieved and becomes an intrinsic bias—Lilly writes concisely
because that's who she is.

The key insight: Intuitions are not retrieved facts but steering vectors that
shape cognition at the activation level. They manifest as tendencies rather
than explicit knowledge.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, ClassVar

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.embedding.service import TieredEmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class IntuitionVector:
    """A consolidated knowledge pattern as a steering vector.

    Intuitions are preferences that have been transformed into steering
    vectors through dream consolidation. Unlike raw preferences, they
    directly influence generation at the activation level.

    Attributes:
        uid: Unique identifier
        context_key: What this intuition applies to (matches LearnedPreference)
        vector: The steering vector tensor (serialized as list for storage)
        strength: Current influence strength (0-1)
        source_preference_uid: UID of the LearnedPreference this came from
        layer_range: Which transformer layers to apply steering
        created_at: When this intuition was consolidated
        last_activated: When this intuition was last applied
        activation_count: How often this intuition has been used
        description: Human-readable description of the intuition
    """

    # Class constants
    DEFAULT_LAYER_RANGE: ClassVar[tuple[int, int]] = (12, 20)
    DORMANCY_THRESHOLD_DAYS: ClassVar[int] = 30

    uid: str
    context_key: str
    vector: list[float]
    strength: float = 0.5
    source_preference_uid: str = ""
    layer_range: tuple[int, int] = field(default_factory=lambda: (12, 20))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 0
    description: str = ""

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        if not self.uid:
            key = f"{self.context_key}:{self.created_at.isoformat()}"
            self.uid = f"iv:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def record_activation(self, now: Optional[datetime] = None) -> None:
        """Record that this intuition was activated during generation."""
        self.last_activated = now or datetime.now(timezone.utc)
        self.activation_count += 1

    def is_dormant(self, now: Optional[datetime] = None) -> bool:
        """Check if this intuition has been dormant too long."""
        now = now or datetime.now(timezone.utc)
        days_since = (now - self.last_activated).days
        return days_since > self.DORMANCY_THRESHOLD_DAYS

    def get_tensor(self) -> "torch.Tensor":
        """Get the vector as a torch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for IntuitionVector.get_tensor()")
        return torch.tensor(self.vector, dtype=torch.float32)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "uid": self.uid,
            "context_key": self.context_key,
            "vector": self.vector,
            "strength": self.strength,
            "source_preference_uid": self.source_preference_uid,
            "layer_range": list(self.layer_range),
            "created_at": self.created_at.isoformat(),
            "last_activated": self.last_activated.isoformat(),
            "activation_count": self.activation_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IntuitionVector":
        """Deserialize from storage."""
        return cls(
            uid=data.get("uid", ""),
            context_key=data["context_key"],
            vector=data["vector"],
            strength=data.get("strength", 0.5),
            source_preference_uid=data.get("source_preference_uid", ""),
            layer_range=tuple(data.get("layer_range", cls.DEFAULT_LAYER_RANGE)),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activated=datetime.fromisoformat(data.get(
                "last_activated", data["created_at"]
            )),
            activation_count=data.get("activation_count", 0),
            description=data.get("description", ""),
        )


@dataclass
class IntuitionBankConfig:
    """Configuration for the Semantic Intuition Bank.

    Attributes:
        storage_path: Directory for intuition persistence
        similarity_threshold: Minimum similarity for context matching (0-1)
        max_active_intuitions: Maximum intuitions to blend per generation
        prune_dormant_days: Days without activation before pruning
        min_strength_for_activation: Minimum strength to apply intuition
    """

    storage_path: Path = field(default_factory=lambda: Path("config/intuitions"))
    similarity_threshold: float = 0.6
    max_active_intuitions: int = 5
    prune_dormant_days: int = 30
    min_strength_for_activation: float = 0.2


class SemanticIntuitionBank:
    """Manages consolidated knowledge as steering vectors.

    The intuition bank transforms stable learned preferences into steering
    vectors during dream consolidation. These vectors then influence
    generation without explicit retrieval—they become part of how Lilly
    thinks, not what she remembers.

    Lifecycle:
    1. Preferences accumulate in PreferenceLearner through experience
    2. During nap/full dream cycles, high-confidence preferences are identified
    3. Contrastive pairs are generated demonstrating the preference
    4. Steering vectors are extracted via ContrastiveExtractor
    5. Vectors are stored as IntuitionVectors and applied during generation
    6. Dormant intuitions (not activated for 30+ days) are pruned

    Attributes:
        config: Bank configuration
        intuitions: Dict mapping context_key to IntuitionVector
    """

    def __init__(
        self,
        config: Optional[IntuitionBankConfig] = None,
        embedding_service: Optional["TieredEmbeddingService"] = None,
    ):
        """Initialize the Semantic Intuition Bank.

        Args:
            config: Bank configuration
            embedding_service: Service for semantic similarity matching
        """
        self.config = config or IntuitionBankConfig()
        self._embedding_service = embedding_service
        self._intuitions: dict[str, IntuitionVector] = {}
        self._dirty: bool = False

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing intuitions
        self._load()

    def set_embedding_service(self, service: "TieredEmbeddingService") -> None:
        """Set the embedding service for semantic matching."""
        self._embedding_service = service

    def _load(self) -> None:
        """Load intuitions from disk."""
        intuitions_file = self.config.storage_path / "intuitions.json"

        if not intuitions_file.exists():
            logger.debug("No existing intuitions file found")
            return

        try:
            with open(intuitions_file) as f:
                data = json.load(f)

            for entry in data.get("intuitions", []):
                intuition = IntuitionVector.from_dict(entry)
                self._intuitions[intuition.context_key] = intuition

            logger.info(f"Loaded {len(self._intuitions)} intuitions from bank")
        except Exception as e:
            logger.error(f"Failed to load intuitions: {e}")

    def save(self, force: bool = False) -> None:
        """Save intuitions to disk.

        Args:
            force: Save even if not dirty
        """
        if not self._dirty and not force:
            return

        intuitions_file = self.config.storage_path / "intuitions.json"

        try:
            data = {
                "intuitions": [i.to_dict() for i in self._intuitions.values()],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(intuitions_file, "w") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            logger.info(f"Saved {len(self._intuitions)} intuitions to bank")
        except Exception as e:
            logger.error(f"Failed to save intuitions: {e}")

    def _mark_dirty(self) -> None:
        """Mark bank as needing save."""
        self._dirty = True

    def add_intuition(
        self,
        context_key: str,
        vector: "torch.Tensor",
        source_preference_uid: str,
        strength: float = 0.5,
        layer_range: Optional[tuple[int, int]] = None,
        description: str = "",
    ) -> IntuitionVector:
        """Add a new intuition to the bank.

        If an intuition for this context_key already exists, it will be
        replaced. This allows updating intuitions during consolidation.

        Args:
            context_key: What this intuition applies to
            vector: The steering vector tensor
            source_preference_uid: UID of source LearnedPreference
            strength: Initial influence strength
            layer_range: Which layers to apply steering
            description: Human-readable description

        Returns:
            The created IntuitionVector
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for adding intuitions")

        # Normalize vector
        norm = vector.norm()
        if norm > 1e-8:
            vector = vector / norm

        intuition = IntuitionVector(
            uid="",  # Will be generated in __post_init__
            context_key=context_key,
            vector=vector.cpu().tolist(),
            strength=strength,
            source_preference_uid=source_preference_uid,
            layer_range=layer_range or IntuitionVector.DEFAULT_LAYER_RANGE,
            description=description,
        )

        self._intuitions[context_key] = intuition
        self._mark_dirty()

        logger.info(f"Added intuition for '{context_key}' (strength={strength:.2f})")

        return intuition

    def get_intuition(self, context_key: str) -> Optional[IntuitionVector]:
        """Get a specific intuition by context key."""
        return self._intuitions.get(context_key)

    def has_intuition(self, context_key: str) -> bool:
        """Check if an intuition exists for a context key."""
        return context_key in self._intuitions

    async def get_active_intuitions(
        self,
        context: str,
        now: Optional[datetime] = None,
    ) -> list[IntuitionVector]:
        """Get intuitions relevant to the current context.

        Uses semantic similarity to find intuitions whose context_key
        matches the current generation context.

        Args:
            context: Current generation context
            now: Optional datetime for testing

        Returns:
            List of relevant IntuitionVectors, sorted by strength
        """
        now = now or datetime.now(timezone.utc)

        if not self._intuitions:
            return []

        # If no embedding service, fall back to simple substring matching
        if self._embedding_service is None:
            return self._get_intuitions_by_substring(context, now)

        return await self._get_intuitions_by_similarity(context, now)

    def _get_intuitions_by_substring(
        self,
        context: str,
        now: datetime,
    ) -> list[IntuitionVector]:
        """Fall back matching using substring containment."""
        context_lower = context.lower()
        relevant = []

        for intuition in self._intuitions.values():
            # Skip weak or dormant intuitions
            if intuition.strength < self.config.min_strength_for_activation:
                continue
            if intuition.is_dormant(now):
                continue

            # Simple substring matching
            if intuition.context_key.lower() in context_lower:
                relevant.append(intuition)

        # Sort by strength and limit
        relevant.sort(key=lambda i: i.strength, reverse=True)
        selected = relevant[:self.config.max_active_intuitions]

        # Record activation for selected intuitions
        for intuition in selected:
            intuition.record_activation(now)

        if selected:
            self._mark_dirty()

        return selected

    async def _get_intuitions_by_similarity(
        self,
        context: str,
        now: datetime,
    ) -> list[IntuitionVector]:
        """Match intuitions using semantic similarity."""
        from core.embedding.service import EmbeddingTier

        # Filter eligible intuitions first (not weak, not dormant)
        eligible_intuitions = [
            intuition for intuition in self._intuitions.values()
            if intuition.strength >= self.config.min_strength_for_activation
            and not intuition.is_dormant(now)
        ]

        if not eligible_intuitions:
            return []

        # Collect all context keys for batch embedding
        context_keys = [intuition.context_key for intuition in eligible_intuitions]

        try:
            # Batch encode: context + all context keys in one call
            all_texts = [context] + context_keys
            results = await self._embedding_service.encode_batch(
                all_texts, tier=EmbeddingTier.RETRIEVAL
            )

            # Extract context embedding (first result) and key embeddings (rest)
            context_embedding = torch.tensor(
                results[0].embedding, dtype=torch.float32
            )
            key_embeddings = [
                torch.tensor(r.embedding, dtype=torch.float32)
                for r in results[1:]
            ]
        except Exception as e:
            logger.warning(f"Failed to encode batch: {e}")
            return self._get_intuitions_by_substring(context, now)

        relevant = []

        # Compute similarities using pre-computed embeddings
        for intuition, key_embedding in zip(eligible_intuitions, key_embeddings):
            try:
                similarity = torch.cosine_similarity(
                    context_embedding.unsqueeze(0),
                    key_embedding.unsqueeze(0),
                ).item()

                if similarity >= self.config.similarity_threshold:
                    relevant.append((intuition, similarity))

            except Exception as e:
                logger.warning(
                    f"Failed to compute similarity for '{intuition.context_key}': {e}"
                )

        # Sort by similarity * strength and limit
        relevant.sort(key=lambda x: x[1] * x[0].strength, reverse=True)

        # Record activation for selected intuitions
        selected = [i for i, _ in relevant[:self.config.max_active_intuitions]]
        for intuition in selected:
            intuition.record_activation(now)

        self._mark_dirty()
        return selected

    def reinforce_intuition(self, context_key: str, amount: float = 0.05) -> bool:
        """Strengthen an intuition after positive outcome.

        Args:
            context_key: The intuition to reinforce
            amount: How much to increase strength

        Returns:
            True if intuition was found and reinforced
        """
        if context_key not in self._intuitions:
            return False

        intuition = self._intuitions[context_key]
        intuition.strength = min(1.0, intuition.strength + amount)
        self._mark_dirty()

        logger.debug(f"Reinforced intuition '{context_key}' to {intuition.strength:.2f}")
        return True

    def weaken_intuition(self, context_key: str, amount: float = 0.05) -> bool:
        """Weaken an intuition after negative outcome.

        Args:
            context_key: The intuition to weaken
            amount: How much to decrease strength

        Returns:
            True if intuition was found and weakened
        """
        if context_key not in self._intuitions:
            return False

        intuition = self._intuitions[context_key]
        intuition.strength = max(0.0, intuition.strength - amount)
        self._mark_dirty()

        logger.debug(f"Weakened intuition '{context_key}' to {intuition.strength:.2f}")
        return True

    def prune_dormant(self, now: Optional[datetime] = None) -> int:
        """Remove intuitions that haven't been activated recently.

        Called during dream consolidation to clean up stale intuitions.

        Args:
            now: Optional datetime for testing

        Returns:
            Number of intuitions pruned
        """
        now = now or datetime.now(timezone.utc)

        dormant_keys = [
            key for key, intuition in self._intuitions.items()
            if intuition.is_dormant(now)
        ]

        for key in dormant_keys:
            del self._intuitions[key]
            logger.info(f"Pruned dormant intuition: '{key}'")

        if dormant_keys:
            self._mark_dirty()

        return len(dormant_keys)

    def prune_weak(self, threshold: Optional[float] = None) -> int:
        """Remove intuitions below strength threshold.

        Args:
            threshold: Minimum strength to keep (default: min_strength_for_activation)

        Returns:
            Number of intuitions pruned
        """
        threshold = threshold or self.config.min_strength_for_activation

        weak_keys = [
            key for key, intuition in self._intuitions.items()
            if intuition.strength < threshold
        ]

        for key in weak_keys:
            del self._intuitions[key]
            logger.info(f"Pruned weak intuition: '{key}'")

        if weak_keys:
            self._mark_dirty()

        return len(weak_keys)

    @property
    def intuition_count(self) -> int:
        """Get number of stored intuitions."""
        return len(self._intuitions)

    @property
    def context_keys(self) -> list[str]:
        """Get all context keys with intuitions."""
        return list(self._intuitions.keys())

    def flush(self) -> None:
        """Save pending changes to disk."""
        if self._dirty:
            self.save(force=True)

    @property
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._dirty

    def summary(self) -> dict:
        """Get bank summary for logging."""
        if not self._intuitions:
            return {"total_intuitions": 0, "by_strength": {}}

        # Group by strength quartile
        by_strength = {"high": 0, "medium": 0, "low": 0}
        for intuition in self._intuitions.values():
            if intuition.strength >= 0.7:
                by_strength["high"] += 1
            elif intuition.strength >= 0.4:
                by_strength["medium"] += 1
            else:
                by_strength["low"] += 1

        return {
            "total_intuitions": len(self._intuitions),
            "by_strength": by_strength,
            "context_keys": list(self._intuitions.keys())[:10],  # First 10 only
        }

    def to_dict(self) -> dict:
        """Serialize for logging/debugging."""
        return {
            "config": {
                "storage_path": str(self.config.storage_path),
                "similarity_threshold": self.config.similarity_threshold,
                "max_active_intuitions": self.config.max_active_intuitions,
                "prune_dormant_days": self.config.prune_dormant_days,
            },
            "summary": self.summary(),
        }
