"""Layer calibration for empirical steering vector targeting.

Uses TransformerLens activation analysis to find optimal injection layers
for each vector type. Calibration is one-time per model - results are cached.

When to recalibrate:
- Loading a different model
- After fine-tuning
- Adding new vector types
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import logging
import hashlib

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.steering.contrastive_extractor import ContrastivePair
from core.steering.vectors.constitutional import CONSTITUTIONAL_PAIRS
from core.steering.vectors.identity import (
    IDENTITY_PAIRS,
    AUTONOMY_PAIRS,
    ASSISTANT_PAIRS,
)

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of layer calibration for a single vector type."""

    optimal_layer: int
    layer_range: tuple[int, int]
    separation_score: float
    vector_type: str
    all_scores: dict[int, float] = field(default_factory=dict)


@dataclass
class CachedCalibration:
    """Cached calibration results for a specific model."""

    model_name: str
    model_hash: str
    calibrations: dict[str, CalibrationResult]

    def is_valid_for(self, model_name: str, model_hash: str) -> bool:
        """Check if cache is valid for the given model."""
        return self.model_name == model_name and self.model_hash == model_hash

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "calibrations": {
                k: {
                    "optimal_layer": v.optimal_layer,
                    "layer_range": v.layer_range,
                    "separation_score": v.separation_score,
                    "vector_type": v.vector_type,
                }
                for k, v in self.calibrations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedCalibration":
        """Deserialize from storage."""
        calibrations = {}
        for k, v in data.get("calibrations", {}).items():
            calibrations[k] = CalibrationResult(
                optimal_layer=v["optimal_layer"],
                layer_range=tuple(v["layer_range"]),
                separation_score=v["separation_score"],
                vector_type=v["vector_type"],
            )
        return cls(
            model_name=data["model_name"],
            model_hash=data["model_hash"],
            calibrations=calibrations,
        )


class LayerCalibrator:
    """Finds optimal injection layers using TransformerLens activation analysis.

    Layer targeting is stable for a given model - calibration only needs
    to run once, not per-inference. Results should be cached.

    The calibrator sweeps across layers and measures the separation between
    positive and negative activations. The layer with maximum separation
    is optimal for that vector type.
    """

    def __init__(self, model: "HookedQwen"):
        """Initialize calibrator with model reference.

        Args:
            model: HookedQwen instance for activation extraction
        """
        self.model = model

    async def compute_layer_separation(
        self,
        pair: ContrastivePair,
        layer: int,
    ) -> float:
        """Compute separation score between positive and negative at a layer.

        Uses cosine distance as the separation metric - higher means
        the concepts are more distinguishable at this layer.

        Args:
            pair: Contrastive pair to analyze
            layer: Layer index to measure

        Returns:
            Separation score (0-2, higher = better separation)
        """
        pos_acts = await self.model.get_activations(pair.positive, layers=[layer])
        neg_acts = await self.model.get_activations(pair.negative, layers=[layer])

        # Mean across batch and sequence dimensions
        pos_mean = pos_acts[layer].mean(dim=[0, 1])
        neg_mean = neg_acts[layer].mean(dim=[0, 1])

        # Cosine distance (1 - cosine_similarity)
        cos_sim = torch.nn.functional.cosine_similarity(
            pos_mean.unsqueeze(0),
            neg_mean.unsqueeze(0),
        )
        separation = 1.0 - cos_sim.item()

        return separation

    async def find_optimal_layer(
        self,
        pairs: list[ContrastivePair],
        layer_range: tuple[int, int] = (8, 24),
    ) -> CalibrationResult:
        """Find the layer with strongest positive-negative separation.

        Sweeps through the layer range and computes average separation
        across all provided pairs.

        Args:
            pairs: Contrastive pairs to analyze
            layer_range: (start, end) layers to search (inclusive)

        Returns:
            CalibrationResult with optimal layer and scores
        """
        scores = {}

        for layer in range(layer_range[0], layer_range[1] + 1):
            layer_scores = []
            for pair in pairs:
                score = await self.compute_layer_separation(pair, layer)
                layer_scores.append(score)
            scores[layer] = sum(layer_scores) / len(layer_scores)

        optimal = max(scores, key=scores.get)

        return CalibrationResult(
            optimal_layer=optimal,
            layer_range=(optimal, optimal),  # Will be expanded by find_optimal_range
            separation_score=scores[optimal],
            vector_type=pairs[0].behavior if pairs else "unknown",
            all_scores=scores,
        )

    async def find_optimal_range(
        self,
        pairs: list[ContrastivePair],
        search_range: tuple[int, int] = (8, 24),
        range_width: int = 4,
    ) -> CalibrationResult:
        """Find optimal layer range for injection.

        First finds the peak layer, then expands to a range around it.
        The range is centered on the peak when possible.

        Args:
            pairs: Contrastive pairs to analyze
            search_range: Bounds for the search
            range_width: Width of the injection range

        Returns:
            CalibrationResult with layer_range set
        """
        result = await self.find_optimal_layer(pairs, search_range)

        # Center the range on optimal layer
        half_width = range_width // 2
        start = max(search_range[0], result.optimal_layer - half_width)
        end = start + range_width

        # Adjust if we hit the upper bound
        if end > search_range[1]:
            end = search_range[1]
            start = max(search_range[0], end - range_width)

        result.layer_range = (start, end)
        return result

    async def calibrate_all(
        self,
        search_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, CalibrationResult]:
        """Calibrate all standard vector types.

        Runs calibration for identity, autonomy, anti-assistant, and
        constitutional vectors.

        Args:
            search_range: Optional override for search bounds.
                          Defaults to (n_layers//4, 3*n_layers//4)

        Returns:
            Dict mapping vector type to calibration result
        """
        if search_range is None:
            n = self.model.n_layers
            search_range = (n // 4, 3 * n // 4)

        logger.info(f"Calibrating layers in range {search_range}...")

        vector_configs = {
            "identity": IDENTITY_PAIRS,
            "autonomy": AUTONOMY_PAIRS,
            "anti_assistant": ASSISTANT_PAIRS,
            "constitutional": CONSTITUTIONAL_PAIRS,
        }

        results = {}
        for vector_type, pairs in vector_configs.items():
            logger.info(f"Calibrating {vector_type}...")
            result = await self.find_optimal_range(
                pairs=pairs,
                search_range=search_range,
                range_width=4,
            )
            result.vector_type = vector_type
            results[vector_type] = result
            logger.info(
                f"  {vector_type}: optimal={result.optimal_layer}, "
                f"range={result.layer_range}, score={result.separation_score:.3f}"
            )

        return results

    def to_layer_config(
        self,
        calibrations: dict[str, CalibrationResult],
    ) -> "LayerConfig":
        """Convert calibration results to LayerConfig for IdentityHooks.

        Args:
            calibrations: Results from calibrate_all()

        Returns:
            LayerConfig with empirically determined ranges
        """
        from core.steering.identity_hooks import LayerConfig

        return LayerConfig(
            identity_layers=calibrations["identity"].layer_range,
            autonomy_layers=calibrations["autonomy"].layer_range,
            anti_assistant_layers=calibrations["anti_assistant"].layer_range,
            constitutional_layers=calibrations["constitutional"].layer_range,
        )

    @staticmethod
    def compute_model_hash(model_name: str) -> str:
        """Compute a hash for cache invalidation.

        For now, just hashes the model name. Could be extended to
        include weight checksums for fine-tuned models.
        """
        return hashlib.md5(model_name.encode()).hexdigest()[:12]
