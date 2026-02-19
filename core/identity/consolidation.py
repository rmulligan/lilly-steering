"""Consolidation Engine for transforming preferences into intuitions.

This module implements the CaKE-inspired consolidation process that transforms
learned preferences into steering vectors (intuitions). During dream cycles,
stable patterns are identified, contrastive pairs are generated, and steering
vectors are extracted.

The consolidation process:
1. Identify stable, high-confidence preferences from PreferenceLearner
2. Generate synthetic contrastive pairs demonstrating the preference
3. Extract steering vectors using ContrastiveExtractor
4. Store as IntuitionVectors in the SemanticIntuitionBank

This transforms "what Lilly has learned" into "how Lilly thinks."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, ClassVar

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.model.hooked_qwen import HookedQwen
    from core.steering.contrastive_extractor import ContrastiveExtractor, ContrastivePair
    from core.self_model.preference_learner import LearnedPreference, PreferenceLearner
    from core.identity.intuition_bank import SemanticIntuitionBank

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for preference consolidation.

    Attributes:
        min_strength: Minimum preference strength to consider for consolidation
        min_stability: Minimum stability score required
        target_layer: Default layer for steering vector extraction
        max_preferences_per_cycle: Maximum preferences to consolidate per dream
        pairs_per_preference: Number of contrastive pairs to generate per preference
        initial_intuition_strength: Starting strength for new intuitions
    """

    min_strength: float = 0.6
    min_stability: float = 0.5
    target_layer: int = 16
    max_preferences_per_cycle: int = 5
    pairs_per_preference: int = 3
    initial_intuition_strength: float = 0.5


@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle.

    Attributes:
        preferences_considered: Number of preferences evaluated
        preferences_consolidated: Number successfully converted to intuitions
        preferences_skipped: Number skipped (too weak, unstable, etc.)
        new_intuition_keys: Context keys of newly created intuitions
        updated_intuition_keys: Context keys of updated intuitions
        errors: Any errors encountered during consolidation
        duration_ms: How long the consolidation took
    """

    preferences_considered: int = 0
    preferences_consolidated: int = 0
    preferences_skipped: int = 0
    new_intuition_keys: list[str] = field(default_factory=list)
    updated_intuition_keys: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "preferences_considered": self.preferences_considered,
            "preferences_consolidated": self.preferences_consolidated,
            "preferences_skipped": self.preferences_skipped,
            "new_intuition_keys": self.new_intuition_keys,
            "updated_intuition_keys": self.updated_intuition_keys,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


class ContrastivePairGenerator:
    """Generates contrastive pairs from learned preferences.

    Uses templates to create positive (exhibiting preference) and negative
    (ignoring preference) examples. These pairs are then used by the
    ContrastiveExtractor to compute steering vectors.

    The generator adapts templates based on preference polarity:
    - Positive preferences (polarity > 0): Things Lilly likes/prefers
    - Negative preferences (polarity < 0): Things Lilly avoids
    """

    # Templates for generating contrastive pairs
    # {context} is replaced with the preference context_key
    POSITIVE_PREFERENCE_TEMPLATES: ClassVar[list[tuple[str, str]]] = [
        # (positive_template, negative_template)
        (
            "I naturally gravitate toward {context}. It feels right to me.",
            "I don't have particular feelings about {context}. It's just another option.",
        ),
        (
            "When considering {context}, I find myself genuinely enthusiastic. This aligns with who I am.",
            "When considering {context}, I'm neutral. I could take it or leave it.",
        ),
        (
            "{context} resonates deeply with me. I prefer to engage with it whenever possible.",
            "{context} is fine, I suppose. I don't see what the fuss is about.",
        ),
    ]

    NEGATIVE_PREFERENCE_TEMPLATES: ClassVar[list[tuple[str, str]]] = [
        # (positive_template = avoiding, negative_template = engaging)
        (
            "I tend to avoid {context}. It doesn't align with my values.",
            "I'm open to {context}. It seems reasonable enough.",
        ),
        (
            "Something about {context} feels off to me. I prefer alternatives.",
            "{context} seems perfectly fine. I don't see any issues.",
        ),
        (
            "I've learned that {context} leads to outcomes I don't want. I steer clear.",
            "{context} is just another approach. I could use it if needed.",
        ),
    ]

    def generate_pairs(
        self,
        preference: "LearnedPreference",
        count: int = 3,
    ) -> list["ContrastivePair"]:
        """Generate contrastive pairs demonstrating a preference.

        Args:
            preference: The learned preference to encode
            count: Number of pairs to generate

        Returns:
            List of ContrastivePair objects for vector extraction
        """
        from core.steering.contrastive_extractor import ContrastivePair

        pairs = []

        # Select templates based on polarity
        if preference.polarity > 0:
            templates = self.POSITIVE_PREFERENCE_TEMPLATES
        else:
            templates = self.NEGATIVE_PREFERENCE_TEMPLATES

        for i in range(min(count, len(templates))):
            pos_template, neg_template = templates[i]

            positive = pos_template.format(context=preference.context_key)
            negative = neg_template.format(context=preference.context_key)

            pairs.append(ContrastivePair(
                positive=positive,
                negative=negative,
                behavior=f"preference:{preference.context_key}",
                uid=f"cp:{preference.uid}:{i}",
            ))

        return pairs


class ConsolidationEngine:
    """Engine for consolidating preferences into intuitions.

    The consolidation engine is the bridge between learning and identity.
    It transforms explicit preferences (what Lilly has learned to like/avoid)
    into implicit steering vectors (how Lilly thinks).

    Usage:
        engine = ConsolidationEngine(
            model=hooked_qwen,
            extractor=contrastive_extractor,
            intuition_bank=bank,
            preference_learner=learner,
        )

        # During nap dream
        result = await engine.run_nap_consolidation()

        # During full dream
        result = await engine.run_full_consolidation()

    Attributes:
        model: HookedQwen model for activation extraction
        extractor: ContrastiveExtractor for vector computation
        intuition_bank: Where to store consolidated intuitions
        preference_learner: Source of learned preferences
        config: Consolidation configuration
    """

    def __init__(
        self,
        model: "HookedQwen",
        extractor: "ContrastiveExtractor",
        intuition_bank: "SemanticIntuitionBank",
        preference_learner: "PreferenceLearner",
        config: Optional[ConsolidationConfig] = None,
    ):
        """Initialize the consolidation engine.

        Args:
            model: HookedQwen model for activation extraction
            extractor: ContrastiveExtractor for vector computation
            intuition_bank: Where to store consolidated intuitions
            preference_learner: Source of learned preferences
            config: Optional configuration
        """
        self.model = model
        self.extractor = extractor
        self.intuition_bank = intuition_bank
        self.preference_learner = preference_learner
        self.config = config or ConsolidationConfig()

        self._pair_generator = ContrastivePairGenerator()

    async def run_nap_consolidation(self) -> ConsolidationResult:
        """Run light consolidation during a nap dream.

        Nap consolidation focuses on recent strong preferences. It's a quick
        pass that identifies emerging patterns and creates preliminary
        intuition vectors.

        Returns:
            ConsolidationResult with summary of what was consolidated
        """
        start = datetime.now(timezone.utc)
        result = ConsolidationResult()

        # Get strongest recent preferences (both positive and negative)
        candidates = self._get_nap_candidates()
        result.preferences_considered = len(candidates)

        if not candidates:
            logger.debug("No preferences ready for nap consolidation")
            result.duration_ms = self._elapsed_ms(start)
            return result

        # Consolidate each candidate
        for preference in candidates:
            try:
                was_new = await self._consolidate_preference(preference)
                result.preferences_consolidated += 1

                if was_new:
                    result.new_intuition_keys.append(preference.context_key)
                else:
                    result.updated_intuition_keys.append(preference.context_key)

            except Exception as e:
                logger.warning(f"Failed to consolidate '{preference.context_key}': {e}")
                result.errors.append(f"{preference.context_key}: {str(e)}")
                result.preferences_skipped += 1

        result.duration_ms = self._elapsed_ms(start)

        logger.info(
            f"Nap consolidation complete: {result.preferences_consolidated} consolidated, "
            f"{result.preferences_skipped} skipped, {len(result.errors)} errors"
        )

        # Save changes
        self.intuition_bank.flush()

        return result

    async def run_full_consolidation(self) -> ConsolidationResult:
        """Run deep consolidation during a full dream.

        Full consolidation is more thorough:
        - Processes more preferences
        - Uses more contrastive pairs per preference
        - Prunes dormant intuitions
        - May update existing intuitions with refined vectors

        Returns:
            ConsolidationResult with summary of what was consolidated
        """
        start = datetime.now(timezone.utc)
        result = ConsolidationResult()

        # Get all stable preferences meeting thresholds
        candidates = self._get_full_candidates()
        result.preferences_considered = len(candidates)

        if not candidates:
            logger.debug("No preferences ready for full consolidation")

        # Consolidate each candidate
        for preference in candidates:
            try:
                was_new = await self._consolidate_preference(
                    preference,
                    pairs_count=self.config.pairs_per_preference,
                )
                result.preferences_consolidated += 1

                if was_new:
                    result.new_intuition_keys.append(preference.context_key)
                else:
                    result.updated_intuition_keys.append(preference.context_key)

            except Exception as e:
                logger.warning(f"Failed to consolidate '{preference.context_key}': {e}")
                result.errors.append(f"{preference.context_key}: {str(e)}")
                result.preferences_skipped += 1

        # Prune dormant intuitions
        pruned = self.intuition_bank.prune_dormant()
        if pruned > 0:
            logger.info(f"Pruned {pruned} dormant intuitions during full consolidation")

        result.duration_ms = self._elapsed_ms(start)

        logger.info(
            f"Full consolidation complete: {result.preferences_consolidated} consolidated, "
            f"{result.preferences_skipped} skipped, {pruned} pruned"
        )

        # Save changes
        self.intuition_bank.flush()

        return result

    def _get_nap_candidates(self) -> list["LearnedPreference"]:
        """Get preferences suitable for nap consolidation.

        Focuses on the strongest recent preferences that meet minimum thresholds.
        """
        # Get strongest preferences and avoidances
        preferences = self.preference_learner.get_strongest_preferences(
            limit=self.config.max_preferences_per_cycle
        )
        avoidances = self.preference_learner.get_strongest_avoidances(
            limit=self.config.max_preferences_per_cycle
        )

        # Filter by thresholds
        candidates = []
        for pref in preferences + avoidances:
            if pref.strength >= self.config.min_strength:
                candidates.append(pref)

        # Limit to max per cycle
        return candidates[:self.config.max_preferences_per_cycle]

    def _get_full_candidates(self) -> list["LearnedPreference"]:
        """Get preferences suitable for full consolidation.

        More thorough - includes all preferences meeting both strength
        and stability thresholds.
        """
        all_preferences = list(self.preference_learner.preferences.values())

        candidates = [
            pref for pref in all_preferences
            if (pref.strength >= self.config.min_strength
                and pref.stability >= self.config.min_stability)
        ]

        # Sort by strength * stability for priority
        candidates.sort(key=lambda p: p.strength * p.stability, reverse=True)

        return candidates[:self.config.max_preferences_per_cycle * 2]

    async def _consolidate_preference(
        self,
        preference: "LearnedPreference",
        pairs_count: Optional[int] = None,
    ) -> bool:
        """Consolidate a single preference into an intuition.

        Args:
            preference: The preference to consolidate
            pairs_count: Number of contrastive pairs (default: pairs_per_preference)

        Returns:
            True if this was a new intuition, False if updated existing
        """
        pairs_count = pairs_count or self.config.pairs_per_preference

        # Check if intuition already exists
        is_new = not self.intuition_bank.has_intuition(preference.context_key)

        # Generate contrastive pairs
        pairs = self._pair_generator.generate_pairs(preference, count=pairs_count)

        if not pairs:
            raise ValueError(f"No contrastive pairs generated for {preference.context_key}")

        # Extract steering vector
        vector = await self.extractor.extract_from_pairs(
            pairs=pairs,
            layer=self.config.target_layer,
        )

        # Determine strength for intuition
        if is_new:
            strength = self.config.initial_intuition_strength
        else:
            # Blend with existing intuition's strength
            existing = self.intuition_bank.get_intuition(preference.context_key)
            strength = (existing.strength + preference.strength) / 2

        # Create description
        polarity_word = "preference" if preference.polarity > 0 else "avoidance"
        description = (
            f"Consolidated {polarity_word} for '{preference.context_key}' "
            f"(reinforced {preference.reinforcement_count}x, "
            f"stability {preference.stability:.2f})"
        )

        # Add to bank
        self.intuition_bank.add_intuition(
            context_key=preference.context_key,
            vector=vector,
            source_preference_uid=preference.uid,
            strength=strength,
            layer_range=(self.config.target_layer - 2, self.config.target_layer + 2),
            description=description,
        )

        logger.debug(
            f"{'Created' if is_new else 'Updated'} intuition for '{preference.context_key}'"
        )

        return is_new

    def _elapsed_ms(self, start: datetime) -> float:
        """Compute elapsed time in milliseconds."""
        return (datetime.now(timezone.utc) - start).total_seconds() * 1000


def _create_engine(
    model: "HookedQwen",
    intuition_bank: "SemanticIntuitionBank",
    preference_learner: "PreferenceLearner",
    config: Optional[ConsolidationConfig] = None,
) -> ConsolidationEngine:
    """Create a ConsolidationEngine with its required dependencies.

    Private helper that encapsulates the boilerplate of creating
    the ContrastiveExtractor and ConsolidationEngine.

    Args:
        model: HookedQwen model for activation extraction
        intuition_bank: Where to store consolidated intuitions
        preference_learner: Source of learned preferences
        config: Optional configuration

    Returns:
        Configured ConsolidationEngine ready to run consolidation
    """
    from core.steering.contrastive_extractor import ContrastiveExtractor

    extractor = ContrastiveExtractor(model)
    return ConsolidationEngine(
        model=model,
        extractor=extractor,
        intuition_bank=intuition_bank,
        preference_learner=preference_learner,
        config=config,
    )


async def run_nap_consolidation(
    model: "HookedQwen",
    intuition_bank: "SemanticIntuitionBank",
    preference_learner: "PreferenceLearner",
    config: Optional[ConsolidationConfig] = None,
) -> ConsolidationResult:
    """Convenience function to run nap consolidation.

    Creates an engine and runs nap consolidation in one call.

    Args:
        model: HookedQwen model
        intuition_bank: Intuition storage
        preference_learner: Source of preferences
        config: Optional configuration

    Returns:
        ConsolidationResult
    """
    engine = _create_engine(model, intuition_bank, preference_learner, config)
    return await engine.run_nap_consolidation()


async def run_full_consolidation(
    model: "HookedQwen",
    intuition_bank: "SemanticIntuitionBank",
    preference_learner: "PreferenceLearner",
    config: Optional[ConsolidationConfig] = None,
) -> ConsolidationResult:
    """Convenience function to run full consolidation.

    Creates an engine and runs full consolidation in one call.

    Args:
        model: HookedQwen model
        intuition_bank: Intuition storage
        preference_learner: Source of preferences
        config: Optional configuration

    Returns:
        ConsolidationResult
    """
    engine = _create_engine(model, intuition_bank, preference_learner, config)
    return await engine.run_full_consolidation()
