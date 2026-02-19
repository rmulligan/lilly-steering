"""Auto-Extractor for discovering new steering dimensions.

Periodically extracts new vectors from accumulated experiences,
expanding Lilly's personality dimensionality over time.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch

from .vector_library import VectorLibrary
from .contrastive_extractor import ContrastivePair

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)


class AutoExtractor:
    """Automatically extracts new steering vectors from experiences.

    Monitors accumulated contrastive pairs and extracts new vectors
    when enough data is available. Adds discovered dimensions to
    the vector library.

    Attributes:
        library: Vector library to add new vectors to
        extraction_threshold: Pairs needed to trigger extraction
        default_layers: Layer range for new vectors
    """

    def __init__(
        self,
        library: VectorLibrary,
        extraction_threshold: int = 20,
        default_layers: tuple[int, int] = (18, 27),
    ):
        """Initialize AutoExtractor.

        Args:
            library: Vector library for storing extracted vectors
            extraction_threshold: Contrastive pairs needed for extraction
            default_layers: Default layer range for new vectors
        """
        self.library = library
        self.extraction_threshold = extraction_threshold
        self.default_layers = default_layers

        self._pending_pairs: list[ContrastivePair] = []
        # Derive extraction count from existing auto-extracted vectors to avoid duplicates
        self._extraction_count = self._get_max_extraction_count()

    def _get_max_extraction_count(self) -> int:
        """Get the highest extraction count from existing auto-extracted vectors."""
        max_count = 0
        for name in self.library.vector_names:
            if name.startswith("auto_"):
                try:
                    # Extract number from "auto_XXX_YYYYMMDD" format
                    parts = name.split("_")
                    if len(parts) >= 2:
                        count = int(parts[1])
                        max_count = max(max_count, count)
                except (ValueError, IndexError):
                    pass
        return max_count

    def add_pair(self, pair: ContrastivePair) -> None:
        """Add a contrastive pair for potential extraction.

        Args:
            pair: Contrastive pair (positive/negative examples)
        """
        self._pending_pairs.append(pair)
        logger.debug(f"Added pair for extraction ({len(self._pending_pairs)} pending)")

    @property
    def ready_for_extraction(self) -> bool:
        """Check if enough pairs for extraction."""
        return len(self._pending_pairs) >= self.extraction_threshold

    async def extract_if_ready(
        self,
        model: "HookedQwen",
        category: str = "auto_extracted",
        description: str = "Auto-extracted from experiences",
    ) -> str | None:
        """Extract a new vector if threshold reached.

        Args:
            model: HookedQwen for activation capture
            category: Category for new vector
            description: Description for new vector

        Returns:
            Name of extracted vector, or None if not ready
        """
        if not self.ready_for_extraction:
            return None

        logger.info(f"Extracting new vector from {len(self._pending_pairs)} pairs")

        # Use pairs for extraction
        pairs_to_use = self._pending_pairs[:self.extraction_threshold]

        try:
            vector = await self._extract_vector(model, pairs_to_use)

            # Generate name
            self._extraction_count += 1
            name = f"auto_{self._extraction_count:03d}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

            # Add to library
            self.library.add_vector(
                name=name,
                vector=vector,
                category=category,
                description=description,
                source="auto_extracted",
                layer_range=self.default_layers,
                coefficient=1.0,  # Start at default
            )

            # Clear used pairs
            self._pending_pairs = self._pending_pairs[self.extraction_threshold:]

            logger.info(f"Extracted new vector '{name}'")
            return name

        except Exception as e:
            logger.error(f"Vector extraction failed: {e}")
            return None

    async def _extract_vector(
        self,
        model: "HookedQwen",
        pairs: list[ContrastivePair],
    ) -> torch.Tensor:
        """Extract a steering vector from contrastive pairs.

        Args:
            model: HookedQwen for activation capture
            pairs: Contrastive pairs to use

        Returns:
            Extracted steering vector
        """
        layers = list(range(self.default_layers[0], self.default_layers[1] + 1))

        async def capture_pair_activations(pair: ContrastivePair):
            """Capture activations for a single pair."""
            pos_result = await model.generate(
                prompt=pair.positive,
                max_tokens=1,
                temperature=0.1,
                capture_activations=True,
                capture_layers=layers,
                use_chat_template=True,
                enable_thinking=False,
            )

            neg_result = await model.generate(
                prompt=pair.negative,
                max_tokens=1,
                temperature=0.1,
                capture_activations=True,
                capture_layers=layers,
                use_chat_template=True,
                enable_thinking=False,
            )

            pos_acts = [s.activations.mean(dim=[0, 1]) for s in pos_result.snapshots]
            neg_acts = [s.activations.mean(dim=[0, 1]) for s in neg_result.snapshots]
            return pos_acts, neg_acts

        # Process pairs in parallel
        results = await asyncio.gather(*[capture_pair_activations(p) for p in pairs])

        positive_activations = []
        negative_activations = []
        for pos_acts, neg_acts in results:
            positive_activations.extend(pos_acts)
            negative_activations.extend(neg_acts)

        # Compute mean difference
        positive_mean = torch.stack(positive_activations).mean(dim=0)
        negative_mean = torch.stack(negative_activations).mean(dim=0)

        vector = positive_mean - negative_mean

        # Normalize
        norm = vector.norm()
        if norm > 1e-8:
            vector = vector / norm

        return vector

    def get_pending_count(self) -> int:
        """Get number of pending pairs."""
        return len(self._pending_pairs)

    def clear_pending(self) -> None:
        """Clear all pending pairs."""
        self._pending_pairs = []
