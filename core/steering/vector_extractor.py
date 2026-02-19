"""Vector extraction from hypothesis contrastive pairs.

This module provides VectorExtractor for extracting steering vectors from
hypothesis contrastive pairs (positive/negative examples). It uses the
Contrastive Activation Addition (CAA) approach:

    steering_vector = mean(positive_activations) - mean(negative_activations)

The extracted vectors are wrapped in HypothesisSteeringVector objects for
tracking effectiveness through the outcome-based self-steering loop.

Architecture:
    VectorExtractor: Extracts steering vectors from hypothesis contrastive pairs
    extract_hypothesis_vector: Convenience function for quick extraction

Usage:
    extractor = VectorExtractor(model=qwen, target_layer=15)

    hypothesis = Hypothesis(
        uid="hyp_001",
        statement="Exploring emergence leads to insight",
        cognitive_operation="explore_emergence",
        positive_example="Let me trace how this pattern emerged...",
        negative_example="I'll just accept this as given...",
    )

    vector = await extractor.extract_vector(hypothesis)
    if vector:
        # Apply vector during generation
        model.set_steering_vector(layer=15, vector=torch.tensor(vector.vector_data))
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen
    from core.cognitive.simulation.schemas import Hypothesis

from core.steering.hypothesis_vectors import HypothesisSteeringVector

logger = logging.getLogger(__name__)


class VectorExtractor:
    """
    Extracts steering vectors from hypothesis contrastive pairs.

    Uses the mean-of-differences approach (CAA):
        vector = mean(positive_activations) - mean(negative_activations)

    The resulting vector is wrapped in a HypothesisSteeringVector that
    tracks effectiveness through verification feedback.

    Attributes:
        model: HookedQwen instance for activation extraction
        target_layer: Default layer for activation extraction
    """

    def __init__(
        self,
        model: "HookedQwen",
        target_layer: int = 15,
    ):
        """Initialize VectorExtractor.

        Args:
            model: HookedQwen model for activation extraction
            target_layer: Default layer for activation extraction (default: 15,
                middle layer for most models)
        """
        self.model = model
        self.target_layer = target_layer

    def _has_contrastive_pair(self, hypothesis: "Hypothesis") -> bool:
        """Check if hypothesis has valid contrastive pair.

        A valid contrastive pair requires both positive_example and
        negative_example to be non-empty strings.

        Args:
            hypothesis: The hypothesis to check

        Returns:
            True if hypothesis has valid contrastive pair
        """
        return bool(
            hypothesis.positive_example
            and hypothesis.positive_example.strip()
            and hypothesis.negative_example
            and hypothesis.negative_example.strip()
        )

    async def extract_vector(
        self,
        hypothesis: "Hypothesis",
        layer: Optional[int] = None,
    ) -> Optional[HypothesisSteeringVector]:
        """
        Extract steering vector from a hypothesis contrastive pair.

        Computes the difference between mean positive and mean negative
        activations, then normalizes to unit length. Returns None if the
        hypothesis lacks a contrastive pair.

        Args:
            hypothesis: Hypothesis with positive/negative examples
            layer: Which layer to extract activations from (default: target_layer)

        Returns:
            HypothesisSteeringVector if extraction succeeded, None if hypothesis
            lacks contrastive pair or extraction failed
        """
        # Check for contrastive pair
        if not self._has_contrastive_pair(hypothesis):
            logger.debug(
                f"Hypothesis {hypothesis.uid} lacks contrastive pair, skipping extraction"
            )
            return None

        target = layer if layer is not None else self.target_layer

        try:
            # Get activations for both examples
            pos_acts = await self.model.get_activations(
                hypothesis.positive_example,
                layers=[target],
            )
            neg_acts = await self.model.get_activations(
                hypothesis.negative_example,
                layers=[target],
            )

            # Compute mean activations
            # Activations shape: [batch, seq, d_model]
            pos_mean = pos_acts[target].mean(dim=[0, 1])  # (d_model,)
            neg_mean = neg_acts[target].mean(dim=[0, 1])  # (d_model,)

            # Compute difference
            diff = pos_mean - neg_mean

            # Normalize to unit length (with zero-vector protection)
            norm = diff.norm()
            if norm > 1e-8:
                vector = diff / norm
            else:
                logger.warning(
                    f"Near-zero norm ({norm:.2e}) for hypothesis {hypothesis.uid} - "
                    "positive and negative activations may be too similar"
                )
                vector = diff  # Keep zero vector

            # Convert to list for HypothesisSteeringVector
            vector_data = vector.tolist()

            # Create HypothesisSteeringVector
            steering_vector = HypothesisSteeringVector(
                uid=f"hsv_{uuid4().hex[:8]}",
                hypothesis_uid=hypothesis.uid,
                cognitive_operation=hypothesis.cognitive_operation or "unknown",
                vector_data=vector_data,
                layer=target,
            )

            logger.info(
                f"Extracted steering vector {steering_vector.uid} from hypothesis "
                f"{hypothesis.uid} (operation: {steering_vector.cognitive_operation})"
            )

            return steering_vector

        except Exception as e:
            logger.error(
                f"Failed to extract vector from hypothesis {hypothesis.uid}: {e}"
            )
            return None

    async def extract_vectors(
        self,
        hypotheses: list["Hypothesis"],
        layer: Optional[int] = None,
    ) -> list[HypothesisSteeringVector]:
        """
        Extract steering vectors from multiple hypotheses.

        Processes each hypothesis and returns a list of successfully extracted
        vectors. Hypotheses without contrastive pairs are skipped.

        Args:
            hypotheses: List of hypotheses to extract from
            layer: Which layer to extract activations from (default: target_layer)

        Returns:
            List of successfully extracted HypothesisSteeringVector objects
        """
        vectors = []
        for hypothesis in hypotheses:
            vector = await self.extract_vector(hypothesis, layer=layer)
            if vector is not None:
                vectors.append(vector)

        logger.info(
            f"Extracted {len(vectors)} vectors from {len(hypotheses)} hypotheses"
        )
        return vectors


async def extract_hypothesis_vector(
    model: "HookedQwen",
    hypothesis: "Hypothesis",
    layer: int = 15,
) -> Optional[HypothesisSteeringVector]:
    """
    Convenience function to extract steering vector from a hypothesis.

    Creates a VectorExtractor and extracts a steering vector from the
    provided hypothesis. For repeated extractions, prefer creating an
    extractor instance directly.

    Args:
        model: HookedQwen model for activation extraction
        hypothesis: Hypothesis with contrastive pair
        layer: Target layer for activation extraction (default: 15)

    Returns:
        HypothesisSteeringVector if extraction succeeded, None otherwise
    """
    extractor = VectorExtractor(model, target_layer=layer)
    return await extractor.extract_vector(hypothesis, layer=layer)
