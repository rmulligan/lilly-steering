"""Neuronpedia client for fetching SAE feature interpretations.

This module provides access to human-readable interpretations of SAE features
from the Neuronpedia API. Interpretations are cached to avoid repeated API calls.

Usage:
    from core.sae.neuronpedia import get_feature_interpretation

    # Get interpretation for a specific feature
    interpretation = await get_feature_interpretation(49431)
    # Returns: "mathematical reasoning" or similar human-readable label
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import aiohttp for async HTTP
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.debug("aiohttp not available - Neuronpedia API disabled")

# Configuration
# NOTE: As of Jan 2026, Qwen3-8B transcoders are NOT indexed on Neuronpedia.
# The mwhanna-qwen3-8b-transcoders from SAELens don't have interpretations available.
# This client will gracefully fall back to generic labels until interpretations are available.
NEURONPEDIA_API_BASE = "https://www.neuronpedia.org/api/feature"
MODEL_ID = "qwen3-8b"
SAE_ID = "16-transcoder-hp"  # Layer 16 transcoder (high-precision)

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "lilly" / "neuronpedia"
CACHE_FILE = CACHE_DIR / f"{MODEL_ID}_{SAE_ID}.json"

# In-memory cache (populated from disk on first access)
_interpretation_cache: dict[int, str] = {}
_cache_loaded: bool = False


@dataclass
class FeatureInfo:
    """Information about an SAE feature from Neuronpedia."""

    index: int
    interpretation: str  # Human-readable interpretation
    description: Optional[str] = None  # Longer description if available
    max_activation_examples: list[str] = None  # Examples that maximally activate this feature


def _load_cache() -> dict[int, str]:
    """Load interpretation cache from disk."""
    global _interpretation_cache, _cache_loaded

    if _cache_loaded:
        return _interpretation_cache

    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                # Convert string keys back to int
                _interpretation_cache = {int(k): v for k, v in data.items()}
                logger.debug(f"Loaded {len(_interpretation_cache)} cached Neuronpedia interpretations")
        except Exception as e:
            logger.warning(f"Failed to load Neuronpedia cache: {e}")
            _interpretation_cache = {}

    _cache_loaded = True
    return _interpretation_cache


def _save_cache() -> None:
    """Save interpretation cache to disk."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in _interpretation_cache.items()}, f)
    except Exception as e:
        logger.warning(f"Failed to save Neuronpedia cache: {e}")


async def fetch_feature_info(feature_index: int) -> Optional[FeatureInfo]:
    """Fetch feature information from Neuronpedia API.

    Args:
        feature_index: The SAE feature index to look up

    Returns:
        FeatureInfo if successful, None if not available
    """
    if not AIOHTTP_AVAILABLE:
        return None

    url = f"{NEURONPEDIA_API_BASE}/{MODEL_ID}/{SAE_ID}/{feature_index}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract interpretation from response
                    # Neuronpedia API returns various fields, we want the autointerp description
                    interpretation = (
                        data.get("autointerp", {}).get("description")
                        or data.get("description")
                        or data.get("interpretation")
                        or f"feature {feature_index}"  # Fallback
                    )

                    # Clean up the interpretation
                    interpretation = _clean_interpretation(interpretation)

                    return FeatureInfo(
                        index=feature_index,
                        interpretation=interpretation,
                        description=data.get("long_description"),
                        max_activation_examples=data.get("activations", [])[:3],
                    )
                elif response.status == 404:
                    logger.debug(f"Feature {feature_index} not found on Neuronpedia")
                    return None
                else:
                    logger.warning(f"Neuronpedia API returned {response.status} for feature {feature_index}")
                    return None

    except asyncio.TimeoutError:
        logger.debug(f"Neuronpedia API timeout for feature {feature_index}")
        return None
    except Exception as e:
        logger.debug(f"Neuronpedia API error for feature {feature_index}: {e}")
        return None


def _clean_interpretation(interpretation: str) -> str:
    """Clean up interpretation text for narration.

    Args:
        interpretation: Raw interpretation from Neuronpedia

    Returns:
        Cleaned interpretation suitable for TTS narration
    """
    if not interpretation:
        return ""

    # Remove common prefixes
    prefixes_to_remove = [
        "This feature activates for ",
        "This feature fires for ",
        "Activates for ",
        "Fires for ",
        "This neuron represents ",
        "Represents ",
    ]

    result = interpretation
    for prefix in prefixes_to_remove:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):]
            break

    # Truncate if too long (for TTS)
    if len(result) > 60:
        # Try to find a natural break point
        break_points = [". ", ", ", " - ", " – "]
        for bp in break_points:
            idx = result.find(bp)
            if 20 < idx < 60:
                result = result[:idx]
                break
        else:
            result = result[:57] + "..."

    return result.strip()


async def get_feature_interpretation(feature_index: int) -> str:
    """Get human-readable interpretation for an SAE feature.

    Checks cache first, fetches from Neuronpedia if not cached.

    Args:
        feature_index: The SAE feature index

    Returns:
        Human-readable interpretation string
    """
    cache = _load_cache()

    # Check cache first
    if feature_index in cache:
        return cache[feature_index]

    # Fetch from Neuronpedia
    info = await fetch_feature_info(feature_index)

    if info and info.interpretation:
        # Cache the result
        _interpretation_cache[feature_index] = info.interpretation
        _save_cache()
        return info.interpretation

    # Fallback: return feature index as readable text
    return f"feature {feature_index}"


async def batch_get_interpretations(feature_indices: list[int]) -> dict[int, str]:
    """Get interpretations for multiple features.

    Fetches from cache first, then batches API calls for uncached features.

    Args:
        feature_indices: List of feature indices to look up

    Returns:
        Dict mapping feature index to interpretation
    """
    cache = _load_cache()
    results = {}
    uncached = []

    # Check cache
    for idx in feature_indices:
        if idx in cache:
            results[idx] = cache[idx]
        else:
            uncached.append(idx)

    # Fetch uncached (with rate limiting)
    if uncached and AIOHTTP_AVAILABLE:
        # Limit concurrent requests to avoid rate limiting
        semaphore = asyncio.Semaphore(3)

        async def fetch_one(idx: int) -> tuple[int, str]:
            async with semaphore:
                info = await fetch_feature_info(idx)
                interpretation = info.interpretation if info else f"feature {idx}"
                return idx, interpretation

        # Fetch in parallel
        tasks = [fetch_one(idx) for idx in uncached[:10]]  # Limit to 10 at a time
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetched:
            if isinstance(result, tuple):
                idx, interp = result
                results[idx] = interp
                _interpretation_cache[idx] = interp

        # Save updated cache
        _save_cache()

    # Fill in any remaining with fallbacks
    for idx in feature_indices:
        if idx not in results:
            results[idx] = f"feature {idx}"

    return results


def clear_cache() -> None:
    """Clear the interpretation cache."""
    global _interpretation_cache, _cache_loaded
    _interpretation_cache = {}
    _cache_loaded = False
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
