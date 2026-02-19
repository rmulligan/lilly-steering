"""Synthesis layer for blending identity and Psyche vectors.

The synthesis layer continuously blends:
1. Static identity vectors from IdentityHooks (identity, autonomy, anti-assistant)
2. Dynamic vectors loaded from Psyche (commitments, beliefs, preferences)
3. Constitutional vector (always included with minimum floor)

Arousal modulation scales the overall blend strength based on affective state.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = None

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.steering.identity_hooks import IdentityHooks
    from core.identity.affective_resonator import AffectiveResonator
    from core.self_model.affective_system import AffectiveState

logger = logging.getLogger(__name__)


@dataclass
class BlendConfig:
    """Configuration for vector blending.

    Attributes:
        identity_weight: Base weight for identity layer vectors
        psyche_weight: Base weight for dynamically loaded Psyche vectors
        resonance_weight: Base weight for affective resonance vectors
        constitutional_floor: Minimum weight for constitutional vector (never zero)
        max_psyche_vectors: Maximum Psyche vectors to load per blend
    """

    identity_weight: float = 1.0
    psyche_weight: float = 0.5
    resonance_weight: float = 0.3
    constitutional_floor: float = 0.3
    max_psyche_vectors: int = 5


@dataclass
class BlendedVector:
    """Result of vector blending for a specific layer.

    Attributes:
        vector: The blended vector tensor
        sources: List of source names contributing to blend
        weights: Dict mapping source name to applied weight
        arousal_multiplier: The arousal modulation applied
        timestamp: When the blend was computed
    """

    vector: "Tensor"
    sources: list[str]
    weights: dict[str, float]
    arousal_multiplier: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Serialize for logging (excludes tensor)."""
        return {
            "sources": self.sources,
            "weights": self.weights,
            "arousal_multiplier": self.arousal_multiplier,
            "timestamp": self.timestamp.isoformat(),
        }


class ArousalModulator:
    """Modulates vector strength based on arousal level.

    Higher arousal increases vector strength (more intense steering),
    lower arousal decreases it (gentler steering).

    The modulation follows a linear scaling around neutral (0.5):
    - arousal=0.5 -> multiplier=1.0 (no change)
    - arousal=1.0 -> multiplier=max_multiplier
    - arousal=0.0 -> multiplier=min_multiplier
    """

    def __init__(
        self,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
        neutral_arousal: float = 0.5,
    ):
        """Initialize the arousal modulator.

        Args:
            min_multiplier: Minimum strength multiplier (at arousal=0)
            max_multiplier: Maximum strength multiplier (at arousal=1)
            neutral_arousal: Arousal level that produces no change (must be in (0, 1))

        Raises:
            ValueError: If neutral_arousal is not strictly between 0 and 1
        """
        if neutral_arousal <= 0.0 or neutral_arousal >= 1.0:
            raise ValueError(
                f"neutral_arousal must be strictly between 0 and 1, got {neutral_arousal}"
            )
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.neutral_arousal = neutral_arousal

    def modulate(self, base_strength: float, arousal: float) -> float:
        """Apply arousal modulation to base strength.

        Args:
            base_strength: The unmodulated strength
            arousal: Current arousal level (0-1)

        Returns:
            Modulated strength
        """
        # Clamp arousal to valid range
        arousal = max(0.0, min(1.0, arousal))

        # Linear interpolation
        if arousal >= self.neutral_arousal:
            # Scale up from neutral to max
            t = (arousal - self.neutral_arousal) / (1.0 - self.neutral_arousal)
            multiplier = 1.0 + t * (self.max_multiplier - 1.0)
        else:
            # Scale down from neutral to min
            t = (self.neutral_arousal - arousal) / self.neutral_arousal
            multiplier = 1.0 - t * (1.0 - self.min_multiplier)

        return base_strength * multiplier


class SynthesisBlender:
    """Blends identity, Psyche, and resonance vectors with arousal modulation.

    The synthesis layer sits between:
    - IdentityHooks (static identity vectors extracted at startup)
    - PsycheClient (dynamic vectors from knowledge graph)
    - AffectiveResonator (valence-weighted vectors from recent experiences)

    It produces a single blended vector per layer that combines all sources
    with appropriate weighting and arousal modulation.

    Attributes:
        identity_hooks: Source of identity layer vectors
        psyche: Source of dynamic Psyche vectors
        resonator: Source of affective resonance vectors (optional)
        hidden_size: Model hidden dimension
        config: Blend configuration
        modulator: Arousal modulator
    """

    def __init__(
        self,
        identity_hooks: "IdentityHooks",
        psyche: "PsycheClient",
        hidden_size: int,
        config: Optional[BlendConfig] = None,
        modulator: Optional[ArousalModulator] = None,
        resonator: Optional["AffectiveResonator"] = None,
    ):
        """Initialize the synthesis blender.

        Args:
            identity_hooks: IdentityHooks instance with extracted vectors
            psyche: PsycheClient for loading dynamic vectors
            hidden_size: Model hidden dimension
            config: Optional blend configuration
            modulator: Optional arousal modulator
            resonator: Optional AffectiveResonator for valence-based steering
        """
        self.identity_hooks = identity_hooks
        self.psyche = psyche
        self.hidden_size = hidden_size
        self.config = config or BlendConfig()
        self.modulator = modulator or ArousalModulator()
        self.resonator = resonator

    async def blend(
        self,
        layer: int,
        arousal: float = 0.5,
        context: Optional[str] = None,
        psyche_vectors: Optional[list[dict]] = None,
        affective_state: Optional["AffectiveState"] = None,
    ) -> BlendedVector:
        """Compute blended vector for a specific layer.

        Args:
            layer: The transformer layer index
            arousal: Current arousal level (0-1)
            context: Optional context for Psyche vector selection
            psyche_vectors: Pre-loaded Psyche vectors to avoid repeated DB queries.
                If None, vectors will be loaded from the database.
            affective_state: Current affective state for resonance computation.
                If provided with context, enables affective resonance blending.

        Returns:
            BlendedVector with combined sources
        """
        sources = []
        weights = {}
        vectors = []

        # Get identity vectors for this layer
        identity_vectors = self.identity_hooks.get_vectors_for_layer(layer)

        # Add identity vectors with modulated weights
        arousal_mult = self.modulator.modulate(1.0, arousal)

        for name, vec in identity_vectors.items():
            base_weight = self.config.identity_weight * arousal_mult
            # Enforce constitutional floor upfront
            if name == "constitutional":
                weight = max(base_weight, self.config.constitutional_floor)
            else:
                weight = base_weight
            sources.append(name)
            weights[name] = weight
            vectors.append(vec * weight)

        # Load dynamic vectors from Psyche (use pre-loaded if provided)
        if psyche_vectors is None:
            psyche_vectors = await self._load_psyche_vectors(context)
        for pv in psyche_vectors:
            name = pv.get("name", "psyche")
            vec = self._parse_embedding(pv.get("embedding"))
            if vec is not None:
                strength = pv.get("strength", 1.0)
                weight = self.config.psyche_weight * strength * arousal_mult
                sources.append(f"psyche:{name}")
                weights[f"psyche:{name}"] = weight
                vectors.append(vec * weight)

        # Compute affective resonance if configured
        if self.resonator is not None and context is not None and affective_state is not None:
            resonance_layers = self.resonator.get_target_layers()
            if resonance_layers[0] <= layer <= resonance_layers[1]:
                try:
                    resonance_result = await self.resonator.compute_resonance(
                        context=context,
                        psyche=self.psyche,
                        affective_state=affective_state,
                    )
                    if resonance_result.vector is not None:
                        weight = self.config.resonance_weight * arousal_mult
                        sources.append("resonance")
                        weights["resonance"] = weight
                        vectors.append(resonance_result.vector * weight)
                        logger.debug(
                            f"Layer {layer}: Added resonance vector "
                            f"(valence={resonance_result.aggregated_valence:.2f}, "
                            f"experiences={resonance_result.experience_count})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute resonance: {e}")

        # Combine all vectors
        if vectors:
            combined = sum(vectors)
        else:
            combined = torch.zeros(self.hidden_size)

        return BlendedVector(
            vector=combined,
            sources=sources,
            weights=weights,
            arousal_multiplier=arousal_mult,
        )

    async def _load_psyche_vectors(
        self,
        context: Optional[str] = None,
    ) -> list[dict]:
        """Load steering vectors from Psyche.

        Args:
            context: Optional context for filtering vectors

        Returns:
            List of vector records from Psyche
        """
        try:
            query = """
                MATCH (sv:SteeringVector)
                WHERE sv.active = true
                RETURN sv.uid as uid, sv.name as name, sv.embedding as embedding,
                       sv.strength as strength
                ORDER BY sv.strength DESC
                LIMIT $limit
            """
            result = await self.psyche.query(
                query,
                {"limit": self.config.max_psyche_vectors},
            )
            return result or []
        except Exception as e:
            logger.warning(f"Failed to load Psyche vectors: {e}")
            return []

    def _parse_embedding(self, embedding) -> Optional["Tensor"]:
        """Parse embedding from various formats.

        Args:
            embedding: Embedding in list, string, or tensor format

        Returns:
            Tensor or None if parsing fails
        """
        import json

        if embedding is None:
            return None

        try:
            if isinstance(embedding, torch.Tensor):
                return embedding

            if isinstance(embedding, str):
                embedding = json.loads(embedding)

            if isinstance(embedding, (list, tuple)):
                return torch.tensor(embedding, dtype=torch.float32)

            return None
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse embedding: {e}")
            return None

    async def create_blend_hook(
        self,
        layer: int,
        arousal: float = 0.5,
        psyche_vectors: Optional[list[dict]] = None,
    ):
        """Create a hook function with pre-blended vector.

        Args:
            layer: The layer to create hook for
            arousal: Current arousal level
            psyche_vectors: Pre-loaded Psyche vectors to avoid repeated DB queries.
                If None, vectors will be loaded from the database.

        Returns:
            Callable hook function
        """
        blended = await self.blend(layer, arousal, psyche_vectors=psyche_vectors)

        def hook_fn(activation: "Tensor", _hook) -> "Tensor":
            if blended.vector is None or len(blended.sources) == 0:
                return activation

            result = activation.clone()
            # Apply to last token position
            result[:, -1, :] = result[:, -1, :] + blended.vector

            return result

        return hook_fn

    async def get_all_blend_hooks(
        self,
        n_layers: int,
        arousal: float = 0.5,
    ) -> dict[int, callable]:
        """Get blend hooks for all layers that need injection.

        Args:
            n_layers: Total number of layers in the model
            arousal: Current arousal level

        Returns:
            Dict mapping layer index to hook function
        """
        # Load Psyche vectors once upfront to avoid N database queries
        psyche_vectors = await self._load_psyche_vectors()

        hooks = {}
        for layer in range(n_layers):
            # Check if any vectors apply to this layer
            identity_vectors = self.identity_hooks.get_vectors_for_layer(layer)
            if identity_vectors:
                hook = await self.create_blend_hook(
                    layer, arousal, psyche_vectors=psyche_vectors
                )
                hooks[layer] = hook

        return hooks
