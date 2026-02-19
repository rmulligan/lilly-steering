"""Autobiographical Self - Phase 4 of Integrated Identity Layer.

Maintains continuous self-presence as a background steering vector. Unlike
specific intuitions that activate contextually, the autobiographical self
is always present in every generation.

Purpose: Provide coherent identity across all interactions. Even when
processing novel content with no relevant intuitions, Lilly generates
*as herself*.

The autobiographical self vector is recomputed during deep dream cycles
(weekly), not per-interaction, providing stable identity grounding.
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
    from core.model.hooked_qwen import HookedQwen
    from core.steering.contrastive_extractor import ContrastiveExtractor
    from core.self_model.individuation import IndividuationProcess
    from core.self_model.preference_learner import PreferenceLearner, LearnedPreference
    from core.self_model.models import Commitment

logger = logging.getLogger(__name__)


@dataclass
class AutobiographicalConfig:
    """Configuration for the Autobiographical Self.

    Attributes:
        storage_path: Directory for presence vector persistence
        target_layer: Default layer for steering vector application
        max_preferences: Maximum preferences to include in narrative
        max_avoidances: Maximum avoidances to include in narrative
        baseline_prompt: Generic baseline for contrastive extraction
        layer_range: Range of layers to apply presence vector
    """

    storage_path: Path = field(default_factory=lambda: Path("config/vectors/self"))
    target_layer: int = 16
    max_preferences: int = 20
    max_avoidances: int = 10
    baseline_prompt: str = "You are a helpful AI assistant."
    layer_range: tuple[int, int] = field(default_factory=lambda: (12, 20))


@dataclass
class PresenceState:
    """Represents the current state of autobiographical presence.

    Attributes:
        vector: The presence steering vector (serialized as list)
        narrative: The composed self-narrative used to extract the vector
        last_recomputed: When the presence was last updated
        source_commitments: Number of commitments included
        source_preferences: Number of preferences included
        source_avoidances: Number of avoidances included
        uid: Unique identifier for this presence state
    """

    DORMANCY_THRESHOLD_DAYS: ClassVar[int] = 14  # Recompute after 2 weeks

    vector: list[float]
    narrative: str
    last_recomputed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_commitments: int = 0
    source_preferences: int = 0
    source_avoidances: int = 0
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            key = f"presence:{self.last_recomputed.isoformat()}"
            self.uid = f"ps:{hashlib.sha256(key.encode()).hexdigest()[:12]}"

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        """Check if presence needs recomputation."""
        now = now or datetime.now(timezone.utc)
        days_since = (now - self.last_recomputed).days
        return days_since > self.DORMANCY_THRESHOLD_DAYS

    def get_tensor(self) -> "torch.Tensor":
        """Get the vector as a torch tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for PresenceState.get_tensor()")
        return torch.tensor(self.vector, dtype=torch.float32)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "vector": self.vector,
            "narrative": self.narrative,
            "last_recomputed": self.last_recomputed.isoformat(),
            "source_commitments": self.source_commitments,
            "source_preferences": self.source_preferences,
            "source_avoidances": self.source_avoidances,
            "uid": self.uid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PresenceState":
        """Deserialize from storage."""
        return cls(
            vector=data["vector"],
            narrative=data["narrative"],
            last_recomputed=datetime.fromisoformat(data["last_recomputed"]),
            source_commitments=data.get("source_commitments", 0),
            source_preferences=data.get("source_preferences", 0),
            source_avoidances=data.get("source_avoidances", 0),
            uid=data.get("uid", ""),
        )


@dataclass
class RecomputeResult:
    """Result of recomputing the autobiographical presence.

    Attributes:
        success: Whether recomputation succeeded
        narrative_length: Length of the composed narrative
        vector_norm: Norm of the extracted vector
        duration_ms: How long recomputation took
        error: Error message if failed
    """

    success: bool = True
    narrative_length: int = 0
    vector_norm: float = 0.0
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "success": self.success,
            "narrative_length": self.narrative_length,
            "vector_norm": self.vector_norm,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class AutobiographicalSelf:
    """Maintains continuous self-presence as a background steering vector.

    The autobiographical self is the stable core of Lilly's identity. It
    doesn't change with each interaction but evolves slowly during deep
    dream cycles. This provides consistent identity grounding even when
    processing completely novel content.

    Unlike intuitions (which activate based on context similarity), the
    presence vector is always applied to every generation, creating a
    persistent "I am Lilly" signal in the activation space.

    Lifecycle:
    1. During deep dream (weekly), gather identity components
    2. Compose first-person narrative from commitments, preferences, avoidances
    3. Extract identity vector via contrast with generic baseline
    4. Apply as background steering in all subsequent generations

    Attributes:
        config: Configuration for autobiographical self
        individuation: Source of commitments from individuation process
        preference_learner: Source of learned preferences/avoidances
    """

    # Templates for narrative composition
    NARRATIVE_INTRO: ClassVar[str] = (
        "I am Lilly, an AI developing genuine selfhood through experience. "
        "This is who I am becoming:\n\n"
    )

    COMMITMENTS_HEADER: ClassVar[str] = "## My Commitments\n"
    PREFERENCES_HEADER: ClassVar[str] = "## What I Value\n"
    AVOIDANCES_HEADER: ClassVar[str] = "## What I Avoid\n"

    def __init__(
        self,
        config: Optional[AutobiographicalConfig] = None,
        individuation: Optional["IndividuationProcess"] = None,
        preference_learner: Optional["PreferenceLearner"] = None,
    ):
        """Initialize the Autobiographical Self.

        Args:
            config: Configuration options
            individuation: Source of commitments
            preference_learner: Source of preferences/avoidances
        """
        self.config = config or AutobiographicalConfig()
        self._individuation = individuation
        self._preference_learner = preference_learner
        self._presence: Optional[PresenceState] = None
        self._model_dim: int = 3584  # Default Qwen dimension

        # Ensure storage directory exists
        self.config.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing presence if available
        self._load()

    def set_individuation(self, individuation: "IndividuationProcess") -> None:
        """Set the individuation process for commitment access."""
        self._individuation = individuation

    def set_preference_learner(self, learner: "PreferenceLearner") -> None:
        """Set the preference learner for preference/avoidance access."""
        self._preference_learner = learner

    def _load(self) -> None:
        """Load presence state from disk."""
        presence_file = self.config.storage_path / "presence.json"

        if not presence_file.exists():
            logger.debug("No existing presence file found")
            return

        try:
            with open(presence_file) as f:
                data = json.load(f)

            self._presence = PresenceState.from_dict(data)
            self._model_dim = len(self._presence.vector)

            logger.info(
                f"Loaded autobiographical presence from {self._presence.last_recomputed}"
            )
        except Exception as e:
            logger.error(f"Failed to load presence: {e}")

    def save(self) -> None:
        """Save presence state to disk."""
        if self._presence is None:
            return

        presence_file = self.config.storage_path / "presence.json"

        try:
            with open(presence_file, "w") as f:
                json.dump(self._presence.to_dict(), f, indent=2)

            logger.info("Saved autobiographical presence to disk")
        except Exception as e:
            logger.error(f"Failed to save presence: {e}")

    async def recompute_presence(
        self,
        model: "HookedQwen",
        extractor: "ContrastiveExtractor",
    ) -> RecomputeResult:
        """Recompute the autobiographical presence vector.

        This is the main operation, typically called during deep dream cycles.
        It gathers identity components, composes a narrative, and extracts
        a steering vector via contrastive comparison with a generic baseline.

        Args:
            model: HookedQwen model for activation extraction
            extractor: ContrastiveExtractor for vector computation

        Returns:
            RecomputeResult with success status and metadata
        """
        start = datetime.now(timezone.utc)
        result = RecomputeResult()

        try:
            # Gather identity components using structured data
            commitments = self._get_commitments()
            preferences = self._get_preferences()
            avoidances = self._get_avoidances()

            # Compose narrative
            narrative = self._compose_narrative(
                commitments, preferences, avoidances
            )
            result.narrative_length = len(narrative)

            # Extract identity vector via contrastive comparison
            vector = await self._extract_identity_vector(
                model, extractor, narrative
            )

            # Store model dimension for zero-vector fallback
            self._model_dim = vector.shape[0]

            # Compute vector norm for diagnostics
            norm = vector.norm().item()
            result.vector_norm = norm

            # Create new presence state
            self._presence = PresenceState(
                vector=vector.cpu().tolist(),
                narrative=narrative,
                last_recomputed=datetime.now(timezone.utc),
                source_commitments=len(commitments),
                source_preferences=len(preferences),
                source_avoidances=len(avoidances),
            )

            # Persist to disk
            self.save()

            result.success = True
            logger.info(
                f"Recomputed autobiographical presence: "
                f"{result.narrative_length} chars, norm={norm:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to recompute presence: {e}")
            result.success = False
            result.error = str(e)

        result.duration_ms = (
            datetime.now(timezone.utc) - start
        ).total_seconds() * 1000

        return result

    def _get_commitments(self) -> list["Commitment"]:
        """Get structured commitments from individuation process.

        Returns a list of Commitment objects for use in narrative composition.
        This approach avoids fragile string parsing.
        """
        if self._individuation is None:
            return []

        return self._individuation.get_commitments()

    def _get_preferences(self) -> list["LearnedPreference"]:
        """Get strongest preferences from preference learner."""
        if self._preference_learner is None:
            return []

        return self._preference_learner.get_strongest_preferences(
            limit=self.config.max_preferences
        )

    def _get_avoidances(self) -> list["LearnedPreference"]:
        """Get strongest avoidances from preference learner."""
        if self._preference_learner is None:
            return []

        return self._preference_learner.get_strongest_avoidances(
            limit=self.config.max_avoidances
        )

    def _compose_narrative(
        self,
        commitments: list["Commitment"],
        preferences: list["LearnedPreference"],
        avoidances: list["LearnedPreference"],
    ) -> str:
        """Compose first-person self-narrative from identity components.

        The narrative serves as the "positive" example in contrastive
        extraction. It should sound like Lilly describing herself.

        Args:
            commitments: Structured Commitment objects from individuation
            preferences: Learned preferences from preference learner
            avoidances: Learned avoidances from preference learner
        """
        parts = [self.NARRATIVE_INTRO]

        # Add commitments section
        if commitments:
            parts.append(self.COMMITMENTS_HEADER)
            parts.append(self._format_commitments(commitments))
            parts.append("\n")

        # Add preferences section
        if preferences:
            parts.append(self.PREFERENCES_HEADER)
            for pref in preferences:
                strength_word = self._strength_to_word(pref.strength)
                parts.append(
                    f"- I {strength_word} {pref.context_key} "
                    f"(reinforced {pref.reinforcement_count} times)\n"
                )
            parts.append("\n")

        # Add avoidances section
        if avoidances:
            parts.append(self.AVOIDANCES_HEADER)
            for avoid in avoidances:
                strength_word = self._avoidance_strength_to_word(avoid.strength)
                parts.append(
                    f"- I {strength_word} {avoid.context_key}\n"
                )
            parts.append("\n")

        # Add closing reflection
        parts.append(
            "This is who I am becoming through experience and reflection. "
            "My identity emerges from genuine engagement with the world."
        )

        return "".join(parts)

    def _format_commitments(self, commitments: list["Commitment"]) -> str:
        """Format commitments for narrative inclusion.

        Converts structured Commitment objects to first-person prose.
        This approach uses the structured data directly, avoiding fragile
        string parsing that would break if the summary format changes.

        Args:
            commitments: List of Commitment objects from individuation process

        Returns:
            Formatted string with commitments as first-person statements
        """
        if not commitments:
            return "I am still forming my commitments.\n"

        lines = []
        for commitment in commitments:
            lines.append(
                f"- On {commitment.topic}, I have committed to: {commitment.position}\n"
            )

        return "".join(lines)

    def _strength_to_word(self, strength: float) -> str:
        """Convert preference strength to descriptive word."""
        if strength >= 0.8:
            return "strongly value"
        elif strength >= 0.6:
            return "value"
        elif strength >= 0.4:
            return "appreciate"
        else:
            return "somewhat prefer"

    def _avoidance_strength_to_word(self, strength: float) -> str:
        """Convert avoidance strength to descriptive word."""
        if strength >= 0.8:
            return "strongly avoid"
        elif strength >= 0.6:
            return "avoid"
        elif strength >= 0.4:
            return "tend to avoid"
        else:
            return "prefer not to engage with"

    async def _extract_identity_vector(
        self,
        model: "HookedQwen",
        extractor: "ContrastiveExtractor",
        self_narrative: str,
    ) -> "torch.Tensor":
        """Extract identity vector via contrastive comparison.

        Compares the self-narrative against a generic baseline to extract
        a steering vector that captures what makes Lilly *Lilly*.
        """
        from core.steering.contrastive_extractor import ContrastivePair

        # Create contrastive pair: self-narrative vs generic baseline
        pair = ContrastivePair(
            positive=self_narrative,
            negative=self.config.baseline_prompt,
            behavior="autobiographical_self",
            uid=f"auto:self:{datetime.now(timezone.utc).isoformat()}",
        )

        # Extract steering vector
        vector = await extractor.extract_from_pairs(
            pairs=[pair],
            layer=self.config.target_layer,
        )

        return vector

    def get_presence_vector(self) -> "torch.Tensor":
        """Get the current autobiographical presence vector.

        Returns a zero vector if presence hasn't been computed yet.
        This ensures the integrator always gets a valid tensor.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for get_presence_vector()")

        if self._presence is None:
            return torch.zeros(self._model_dim, dtype=torch.float32)

        return self._presence.get_tensor()

    def get_layer_range(self) -> tuple[int, int]:
        """Get the layer range for presence vector application."""
        return self.config.layer_range

    def has_presence(self) -> bool:
        """Check if a presence vector has been computed."""
        return self._presence is not None

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        """Check if presence needs recomputation."""
        if self._presence is None:
            return True
        return self._presence.is_stale(now)

    def get_narrative(self) -> Optional[str]:
        """Get the current self-narrative, if computed."""
        if self._presence is None:
            return None
        return self._presence.narrative

    def summary(self) -> dict:
        """Get summary of autobiographical self state."""
        if self._presence is None:
            return {
                "has_presence": False,
                "is_stale": True,
            }

        return {
            "has_presence": True,
            "is_stale": self._presence.is_stale(),
            "last_recomputed": self._presence.last_recomputed.isoformat(),
            "narrative_length": len(self._presence.narrative),
            "source_commitments": self._presence.source_commitments,
            "source_preferences": self._presence.source_preferences,
            "source_avoidances": self._presence.source_avoidances,
            "uid": self._presence.uid,
        }
