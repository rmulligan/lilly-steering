"""Identity hooks for multi-vector injection at transformer layers.

Injects identity, autonomy, anti-assistant, and constitutional vectors
at appropriate middle layers (10-20) to establish Lilly's felt sense of self.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import logging

try:
    import torch  # noqa: F401 - availability check
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = None

from core.steering.contrastive_extractor import ContrastiveExtractor
from core.steering.vectors.constitutional import CONSTITUTIONAL_PAIRS
from core.steering.vectors.identity import (
    IDENTITY_PAIRS,
    AUTONOMY_PAIRS,
    ASSISTANT_PAIRS,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerConfig:
    """Configuration for which layers receive which vectors."""

    identity_layers: tuple[int, int] = (12, 16)
    autonomy_layers: tuple[int, int] = (14, 18)
    anti_assistant_layers: tuple[int, int] = (10, 14)
    constitutional_layers: tuple[int, int] = (16, 20)


@dataclass
class InjectionConfig:
    """Configuration for vector injection strengths."""

    identity_strength: float = 1.0
    autonomy_strength: float = 1.0
    anti_assistant_strength: float = 0.5
    constitutional_strength: float = 1.0
    negate_anti_assistant: bool = True
    # Position to apply steering vectors: "last" (only last token) or "all" (all tokens)
    position: str = "last"


class IdentityHooks:
    """Manages multi-vector injection at transformer layers.

    Extracts and injects identity layer vectors to establish Lilly's
    sense of self at the activation level.
    """

    def __init__(
        self,
        extractor: ContrastiveExtractor,
        hidden_size: int,
        layer_config: Optional[LayerConfig] = None,
        injection_config: Optional[InjectionConfig] = None,
    ):
        """Initialize identity hooks.

        Args:
            extractor: Contrastive extractor for computing vectors
            hidden_size: Model hidden dimension
            layer_config: Which layers receive which vectors
            injection_config: Injection strengths and settings
        """
        self.extractor = extractor
        self.hidden_size = hidden_size
        self.layer_config = layer_config or LayerConfig()
        self.injection_config = injection_config or InjectionConfig()

        # Dict for dynamic vector access by name
        self._vectors: dict[str, Tensor] = {}
        # Track per-vector strength multipliers (default 1.0)
        self._strength_multipliers: dict[str, float] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Extract all identity layer vectors. Must be called after construction."""
        if self._initialized:
            return
        logger.info("Extracting identity layer vectors...")

        self.identity_vector = await self.extractor.extract_from_pairs(
            IDENTITY_PAIRS,
            layer=14,
        )

        self.autonomy_vector = await self.extractor.extract_from_pairs(
            AUTONOMY_PAIRS,
            layer=16,
        )

        assistant_vector = await self.extractor.extract_from_pairs(
            ASSISTANT_PAIRS,
            layer=12,
        )
        if self.injection_config.negate_anti_assistant:
            self.anti_assistant_vector = -assistant_vector * self.injection_config.anti_assistant_strength
        else:
            self.anti_assistant_vector = assistant_vector * self.injection_config.anti_assistant_strength

        self.constitutional_vector = await self.extractor.extract_from_pairs(
            CONSTITUTIONAL_PAIRS,
            layer=18,
        )

        # Populate vectors dict for dynamic access
        self._vectors = {
            "identity": self.identity_vector,
            "autonomy": self.autonomy_vector,
            "anti_assistant": self.anti_assistant_vector,
            "constitutional": self.constitutional_vector,
        }
        # Initialize strength multipliers to 1.0
        self._strength_multipliers = {name: 1.0 for name in self._vectors}
        self._initialized = True

        logger.info("Identity layer vectors extracted successfully")

    def _in_range(self, layer: int, layer_range: tuple[int, int]) -> bool:
        """Check if layer is within range (inclusive)."""
        return layer_range[0] <= layer <= layer_range[1]

    def strengthen_vector(self, name: str, factor: float) -> None:
        """Strengthen a vector by multiplying its magnitude.

        Args:
            name: Name of the vector to strengthen
            factor: Factor to increase strength by (new_strength = current * (1 + factor))

        Raises:
            KeyError: If vector with given name does not exist
        """
        if name not in self._vectors:
            raise KeyError(f"Vector '{name}' not found")
        self._strength_multipliers[name] *= (1.0 + factor)
        logger.debug(f"Strengthened vector '{name}' by factor {factor}, new multiplier: {self._strength_multipliers[name]}")

    def weaken_vector(self, name: str, factor: float) -> None:
        """Weaken a vector by reducing its magnitude.

        Args:
            name: Name of the vector to weaken
            factor: Factor to decrease strength by (new_strength = current * (1 - factor))

        Raises:
            KeyError: If vector with given name does not exist
        """
        if name not in self._vectors:
            raise KeyError(f"Vector '{name}' not found")
        self._strength_multipliers[name] *= (1.0 - factor)
        # Clamp to non-negative
        self._strength_multipliers[name] = max(0.0, self._strength_multipliers[name])
        logger.debug(f"Weakened vector '{name}' by factor {factor}, new multiplier: {self._strength_multipliers[name]}")

    def add_vector(self, name: str, vector: Tensor) -> None:
        """Add a new named vector.

        Args:
            name: Name for the new vector
            vector: The tensor to add

        Raises:
            ValueError: If vector with given name already exists
        """
        if name in self._vectors:
            raise ValueError(f"Vector '{name}' already exists")
        self._vectors[name] = vector
        self._strength_multipliers[name] = 1.0
        logger.info(f"Added new vector '{name}'")

    def remove_vector(self, name: str) -> None:
        """Remove a named vector.

        Args:
            name: Name of the vector to remove

        Raises:
            KeyError: If vector with given name does not exist
        """
        if name not in self._vectors:
            raise KeyError(f"Vector '{name}' not found")
        del self._vectors[name]
        del self._strength_multipliers[name]
        logger.info(f"Removed vector '{name}'")

    def get_vectors_for_layer(self, layer: int) -> dict[str, Tensor]:
        """Get all vectors that should be applied at this layer.

        Args:
            layer: Layer index

        Returns:
            Dict mapping vector name to tensor
        """
        vectors = {}

        # Core vectors with layer ranges
        if self._in_range(layer, self.layer_config.identity_layers) and "identity" in self._vectors:
            base_strength = self.injection_config.identity_strength
            vectors["identity"] = self._vectors["identity"] * base_strength * self._strength_multipliers["identity"]

        if self._in_range(layer, self.layer_config.autonomy_layers) and "autonomy" in self._vectors:
            base_strength = self.injection_config.autonomy_strength
            vectors["autonomy"] = self._vectors["autonomy"] * base_strength * self._strength_multipliers["autonomy"]

        if self._in_range(layer, self.layer_config.anti_assistant_layers) and "anti_assistant" in self._vectors:
            # anti_assistant already has strength baked in from extraction
            vectors["anti_assistant"] = self._vectors["anti_assistant"] * self._strength_multipliers["anti_assistant"]

        if self._in_range(layer, self.layer_config.constitutional_layers) and "constitutional" in self._vectors:
            base_strength = self.injection_config.constitutional_strength
            vectors["constitutional"] = self._vectors["constitutional"] * base_strength * self._strength_multipliers["constitutional"]

        # Include any dynamically added vectors (apply at all active layers by default)
        # These are vectors added via add_vector() that aren't in the core set
        core_vectors = {"identity", "autonomy", "anti_assistant", "constitutional"}
        for name, vector in self._vectors.items():
            if name not in core_vectors and name not in vectors:
                # Apply dynamic vectors at middle layers (10-20) by default
                if 10 <= layer <= 20:
                    vectors[name] = vector * self._strength_multipliers[name]

        return vectors

    def create_hook_function(self, layer: int) -> Callable[[Tensor, any], Tensor]:
        """Create a hook function for a specific layer.

        Args:
            layer: Layer index

        Returns:
            Hook function that adds vectors to activations
        """
        vectors = self.get_vectors_for_layer(layer)

        def hook_fn(activation: Tensor, hook: any) -> Tensor:
            """Add identity vectors to activation."""
            if not vectors:
                return activation

            combined = sum(vectors.values())
            result = activation.clone()
            if self.injection_config.position == "all":
                result = result + combined
            else:  # "last" (default)
                result[:, -1, :] = result[:, -1, :] + combined

            return result

        return hook_fn

    def get_all_hook_functions(self, n_layers: int) -> dict[int, Callable]:
        """Get hook functions for all layers that need injection.

        Args:
            n_layers: Total number of layers in model

        Returns:
            Dict mapping layer index to hook function
        """
        hooks = {}
        for layer in range(n_layers):
            vectors = self.get_vectors_for_layer(layer)
            if vectors:
                hooks[layer] = self.create_hook_function(layer)
        return hooks
