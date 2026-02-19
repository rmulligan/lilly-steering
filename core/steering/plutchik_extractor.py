"""Plutchik emotion steering vector extraction and persistence.

This module extracts CAA steering vectors for Plutchik's 8 primary emotions
using contrastive pairs defined in core/steering/vectors/plutchik.py.

The extracted vectors can be used to steer the model toward specific
emotional registers during generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from core.steering.contrastive_extractor import ContrastiveExtractor
from core.steering.vectors.plutchik import PLUTCHIK_PAIRS

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

# Target layer for extraction - mid-exploration zone for Qwen 32-layer model
DEFAULT_TARGET_LAYER = 6


class PlutchikExtractor:
    """Extracts and manages Plutchik emotion steering vectors.

    Uses the ContrastiveExtractor to compute steering vectors for each of
    Plutchik's 8 primary emotions, then persists them to Psyche for use
    during cognitive cycles.

    Attributes:
        model: HookedQwen model for activation extraction
        psyche: PsycheClient for vector persistence
        extractor: ContrastiveExtractor instance
        target_layer: Which layer to extract activations from
    """

    def __init__(
        self,
        model: "HookedQwen",
        psyche: "PsycheClient",
        target_layer: int = DEFAULT_TARGET_LAYER,
    ):
        """Initialize PlutchikExtractor.

        Args:
            model: HookedQwen model for activation extraction
            psyche: PsycheClient for vector persistence
            target_layer: Target layer for extraction (default: 6)
        """
        self.model = model
        self.psyche = psyche
        self.extractor = ContrastiveExtractor(model)
        self.target_layer = target_layer

    async def extract_all(self) -> dict[str, "torch.Tensor"]:
        """Extract steering vectors for all 8 primary emotions.

        Returns:
            Dict mapping emotion name to normalized steering vector
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for vector extraction")

        vectors: dict[str, torch.Tensor] = {}

        for emotion, pairs in PLUTCHIK_PAIRS.items():
            logger.info(f"[PLUTCHIK] Extracting {emotion} vector from {len(pairs)} pairs...")
            vector = await self.extractor.extract_from_pairs(pairs, self.target_layer)
            vectors[emotion] = vector
            logger.info(
                f"[PLUTCHIK] Extracted {emotion} vector: "
                f"norm={vector.norm():.3f}, shape={vector.shape}"
            )

        return vectors

    async def extract_single(self, emotion: str) -> "torch.Tensor":
        """Extract steering vector for a single emotion.

        Args:
            emotion: One of the 8 Plutchik emotions

        Returns:
            Normalized steering vector

        Raises:
            ValueError: If emotion is not a valid Plutchik emotion
        """
        if emotion not in PLUTCHIK_PAIRS:
            valid = list(PLUTCHIK_PAIRS.keys())
            raise ValueError(f"Unknown emotion '{emotion}'. Valid: {valid}")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for vector extraction")

        pairs = PLUTCHIK_PAIRS[emotion]
        vector = await self.extractor.extract_from_pairs(pairs, self.target_layer)
        logger.info(
            f"[PLUTCHIK] Extracted {emotion} vector: "
            f"norm={vector.norm():.3f}, shape={vector.shape}"
        )
        return vector

    async def persist(self, vectors: dict[str, "torch.Tensor"]) -> dict[str, str]:
        """Save extracted vectors to Psyche as SteeringVector nodes.

        Args:
            vectors: Dict mapping emotion name to steering vector

        Returns:
            Dict mapping emotion name to created SteeringVector UID
        """
        uids: dict[str, str] = {}
        timestamp = datetime.now(timezone.utc).isoformat()

        for emotion, vector in vectors.items():
            pairs_hash = self._hash_pairs(emotion)
            name = f"plutchik_{emotion}"

            data = {
                "name": name,
                "layer": self.target_layer,
                "vector_data": vector.tolist(),
                "coefficient": 1.0,
                "active": True,
                "pairs_hash": pairs_hash,
                "emotion_type": "plutchik_primary",
                "timestamp": timestamp,
            }

            result = await self.psyche.upsert_steering_vector(data)
            uid = result.get("uid", name)
            uids[emotion] = uid
            logger.info(f"[PLUTCHIK] Persisted {emotion} vector as {uid}")

        return uids

    async def persist_single(self, emotion: str, vector: "torch.Tensor") -> str:
        """Save a single extracted vector to Psyche.

        Args:
            emotion: The emotion name
            vector: The steering vector to persist

        Returns:
            The created SteeringVector UID
        """
        result = await self.persist({emotion: vector})
        return result[emotion]

    async def verify(
        self, vectors: dict[str, "torch.Tensor"]
    ) -> dict[str, dict[str, float]]:
        """Verify extracted vectors by testing positive > negative projection.

        For each emotion, computes the vector's projection onto the positive
        and negative examples. A valid vector should project more strongly
        onto positive examples.

        Args:
            vectors: Dict mapping emotion name to steering vector

        Returns:
            Dict with verification results per emotion:
            {
                "joy": {
                    "positive_proj": 0.82,
                    "negative_proj": -0.15,
                    "margin": 0.97,
                    "valid": True
                },
                ...
            }
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for verification")

        results: dict[str, dict[str, float]] = {}

        for emotion, vector in vectors.items():
            pairs = PLUTCHIK_PAIRS[emotion]

            # Aggregate projections across all pairs for robust verification
            pos_projections: list[float] = []
            neg_projections: list[float] = []

            for pair in pairs:
                pos_acts = await self.model.get_activations(
                    pair.positive, layers=[self.target_layer]
                )
                neg_acts = await self.model.get_activations(
                    pair.negative, layers=[self.target_layer]
                )

                # Mean activations
                pos_mean = pos_acts[self.target_layer].mean(dim=[0, 1])
                neg_mean = neg_acts[self.target_layer].mean(dim=[0, 1])

                # Project onto steering vector
                pos_projections.append(torch.dot(pos_mean, vector).item())
                neg_projections.append(torch.dot(neg_mean, vector).item())

            # Average projections across all pairs
            pos_proj = sum(pos_projections) / len(pos_projections)
            neg_proj = sum(neg_projections) / len(neg_projections)
            margin = pos_proj - neg_proj

            results[emotion] = {
                "positive_proj": pos_proj,
                "negative_proj": neg_proj,
                "margin": margin,
                "valid": margin > 0,
                "num_pairs": len(pairs),
            }

            status = "PASS" if margin > 0 else "FAIL"
            logger.info(
                f"[PLUTCHIK] Verify {emotion}: {status} "
                f"(pos={pos_proj:.3f}, neg={neg_proj:.3f}, margin={margin:.3f}, "
                f"pairs={len(pairs)})"
            )

        return results

    async def load_from_psyche(self) -> dict[str, "torch.Tensor"]:
        """Load previously extracted vectors from Psyche.

        Returns:
            Dict mapping emotion name to steering vector, empty if not found
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for loading vectors")

        vectors: dict[str, torch.Tensor] = {}

        for emotion in PLUTCHIK_PAIRS.keys():
            name = f"plutchik_{emotion}"
            result = await self.psyche.get_steering_vector(name)

            if result and result.get("vector_data"):
                vector_data = result["vector_data"]
                # Handle JSON-encoded vector data
                if isinstance(vector_data, str):
                    vector_data = json.loads(vector_data)
                vectors[emotion] = torch.tensor(vector_data)
                logger.debug(f"[PLUTCHIK] Loaded {emotion} vector from Psyche")

        if vectors:
            logger.info(f"[PLUTCHIK] Loaded {len(vectors)}/8 vectors from Psyche")
        else:
            logger.info("[PLUTCHIK] No vectors found in Psyche")

        return vectors

    def _hash_pairs(self, emotion: str) -> str:
        """Compute hash of contrastive pairs for cache invalidation.

        Args:
            emotion: The emotion to hash pairs for

        Returns:
            Hex digest of pairs content
        """
        pairs = PLUTCHIK_PAIRS[emotion]
        content = json.dumps(
            [(p.positive, p.negative, p.behavior) for p in pairs],
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def needs_update(self, emotion: str) -> bool:
        """Check if an emotion's vector needs re-extraction.

        Compares stored pairs_hash with current pairs hash to detect
        when contrastive pairs have changed.

        Args:
            emotion: The emotion to check

        Returns:
            True if vector needs re-extraction
        """
        name = f"plutchik_{emotion}"
        result = await self.psyche.get_steering_vector(name)

        if not result:
            return True

        stored_hash = result.get("pairs_hash", "")
        current_hash = self._hash_pairs(emotion)

        return stored_hash != current_hash


async def extract_and_persist_plutchik_vectors(
    model: "HookedQwen",
    psyche: "PsycheClient",
    target_layer: int = DEFAULT_TARGET_LAYER,
    force: bool = False,
) -> dict[str, str]:
    """Convenience function to extract and persist all Plutchik vectors.

    Args:
        model: HookedQwen model for activation extraction
        psyche: PsycheClient for vector persistence
        target_layer: Target layer for extraction
        force: If True, re-extract even if vectors exist

    Returns:
        Dict mapping emotion name to SteeringVector UID
    """
    extractor = PlutchikExtractor(model, psyche, target_layer)

    # Check which vectors need updating
    to_extract: list[str] = []
    if force:
        to_extract = list(PLUTCHIK_PAIRS.keys())
    else:
        for emotion in PLUTCHIK_PAIRS.keys():
            if await extractor.needs_update(emotion):
                to_extract.append(emotion)
                logger.info(f"[PLUTCHIK] {emotion} needs extraction")
            else:
                logger.info(f"[PLUTCHIK] {emotion} up to date")

    if not to_extract:
        logger.info("[PLUTCHIK] All vectors up to date")
        # Return existing UIDs
        return {f"plutchik_{e}": f"plutchik_{e}" for e in PLUTCHIK_PAIRS.keys()}

    # Extract needed vectors
    if len(to_extract) == 8:
        vectors = await extractor.extract_all()
    else:
        results = await asyncio.gather(*(extractor.extract_single(e) for e in to_extract))
        vectors = dict(zip(to_extract, results))

    # Verify extraction quality
    verification = await extractor.verify(vectors)
    failed = [e for e, v in verification.items() if not v["valid"]]
    if failed:
        logger.warning(f"[PLUTCHIK] Verification failed for: {failed}")

    # Persist to Psyche
    uids = await extractor.persist(vectors)

    return uids
