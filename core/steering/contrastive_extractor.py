"""Contrastive pair extraction for steering vector computation.

This module implements the Contrastive Activation Addition (CAA) approach
for computing steering vectors. Given positive and negative example pairs,
we compute:

    steering_vector = mean(positive_activations) - mean(negative_activations)

This is optimal under the squared distance objective, as shown in
"Steering Language Models With Activation Engineering" (Turner et al., 2023).

Architecture:
    ContrastivePair: Dataclass holding positive/negative text examples
    ContrastiveExtractor: Computes steering vectors using a HookedQwen model
    extract_steering_vector: Convenience function for quick extraction

Usage:
    extractor = ContrastiveExtractor(model=qwen)

    pair = ContrastivePair(
        positive="Let me think carefully about this...",
        negative="I'll just guess...",
        behavior="deliberate_reasoning",
    )

    vector = await extractor.extract_vector(pair, layer=15)
    # Apply vector during generation with model.set_steering_vector()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen

logger = logging.getLogger(__name__)


@dataclass
class ContrastivePair:
    """A positive/negative example pair for steering vector extraction.

    Contrastive pairs encode a target behavior by providing examples of
    text that exhibits the behavior (positive) and text that exhibits
    the opposite (negative). The steering vector computed from these
    pairs can then be added to model activations to shift generation
    toward the positive behavior.

    Attributes:
        positive: Text example exhibiting the desired behavior
        negative: Text example exhibiting the undesired behavior
        behavior: Name/description of the target behavior
        uid: Optional unique identifier for the pair
    """

    positive: str
    negative: str
    behavior: str
    uid: Optional[str] = None


class ContrastiveExtractor:
    """
    Extracts steering vectors from contrastive pairs.

    Uses the mean-of-differences approach (CAA):
        vector = mean(positive_activations) - mean(negative_activations)

    This is optimal under the squared distance objective. The resulting
    vectors can be applied during inference to shift model behavior.

    The extractor works with a HookedQwen model to capture activations
    at specific layers. For best results:
    - Use middle layers (around layer 15 for a 32-layer model)
    - Provide multiple pairs to reduce variance
    - Ensure positive and negative examples have similar structure

    Attributes:
        model: HookedQwen instance for activation extraction
    """

    def __init__(self, model: "HookedQwen"):
        """Initialize ContrastiveExtractor.

        Args:
            model: HookedQwen model for activation extraction
        """
        self.model = model

    async def extract_vector(
        self,
        pair: ContrastivePair,
        layer: int,
    ) -> "torch.Tensor":
        """
        Extract steering vector from a single contrastive pair.

        Computes the difference between mean positive and mean negative
        activations, then normalizes to unit length.

        This method delegates to extract_from_pairs with a single-element
        list to avoid code duplication.

        Args:
            pair: Positive/negative text pair
            layer: Which layer to extract activations from

        Returns:
            Normalized steering vector with shape (d_model,)
        """
        return await self.extract_from_pairs([pair], layer)

    async def extract_from_pairs(
        self,
        pairs: list[ContrastivePair],
        layer: int,
    ) -> "torch.Tensor":
        """
        Extract steering vector from multiple contrastive pairs.

        Averaging across multiple pairs reduces variance and produces
        a more robust steering vector. Each pair contributes equally
        to the final vector.

        Args:
            pairs: List of positive/negative pairs
            layer: Which layer to extract activations from

        Returns:
            Normalized steering vector with shape (d_model,)

        Raises:
            ValueError: If pairs list is empty
        """
        if not pairs:
            raise ValueError("At least one pair required")

        # Build list of all activation extraction tasks for parallel execution
        tasks = []
        for pair in pairs:
            tasks.append(self.model.get_activations(pair.positive, layers=[layer]))
            tasks.append(self.model.get_activations(pair.negative, layers=[layer]))

        # Execute all activation extractions in parallel
        all_activations = await asyncio.gather(*tasks)

        # Process results pairwise (pos, neg, pos, neg, ...)
        all_diffs = []
        for i in range(0, len(all_activations), 2):
            pos_acts = all_activations[i]
            neg_acts = all_activations[i + 1]

            # Compute mean activations
            pos_mean = pos_acts[layer].mean(dim=[0, 1])
            neg_mean = neg_acts[layer].mean(dim=[0, 1])

            # Compute difference for this pair
            diff = pos_mean - neg_mean
            all_diffs.append(diff)

        # Stack and mean across pairs
        stacked = torch.stack(all_diffs, dim=0)  # (n_pairs, d_model)
        vector = stacked.mean(dim=0)  # (d_model,)

        # Normalize to unit length (with zero-vector protection)
        norm = vector.norm()
        if norm > 1e-8:
            vector = vector / norm
        else:
            behaviors = [p.behavior for p in pairs]
            logger.warning(
                f"Near-zero norm ({norm:.2e}) for pairs {behaviors} - "
                "positive and negative activations may be too similar"
            )

        return vector


async def extract_steering_vector(
    model: "HookedQwen",
    pairs: list[ContrastivePair],
    layer: int,
) -> "torch.Tensor":
    """
    Convenience function to extract steering vector from contrastive pairs.

    Creates a ContrastiveExtractor and extracts a steering vector from
    the provided pairs. For repeated extractions, prefer creating an
    extractor instance directly.

    Args:
        model: HookedQwen model for activation extraction
        pairs: List of contrastive pairs defining the target behavior
        layer: Target layer for activation extraction

    Returns:
        Normalized steering vector with shape (d_model,)
    """
    extractor = ContrastiveExtractor(model)
    return await extractor.extract_from_pairs(pairs, layer)
