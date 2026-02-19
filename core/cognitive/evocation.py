"""Activation-driven memory retrieval via SAE feature evocation.

This module enables "subconscious" memory retrieval where entities, moods, and
questions are recalled based on internal activation states (SAE features).

When generating thoughts, SAE features capture what's being processed internally.
Over time, the EvocationTracker learns associations between features and:
- Entities: Concepts that appear in thoughts with those features active
- Moods: Emotional coloring when those features are active
- Questions: Unresolved questions that arise with those features

During future cognitive cycles, active features can "evoke" related content
even when it doesn't appear in the explicit context.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ENHANCED EVOCATION                                    │
    │                                                                          │
    │  Active SAE Features                                                     │
    │       ↓                                                                  │
    │  EvocationTracker (extended)                                             │
    │       ├─→ Entity associations (existing)                                 │
    │       ├─→ Mood associations (NEW)                                        │
    │       └─→ Question associations (NEW)                                    │
    │       ↓                                                                  │
    │  Prompt Injection:                                                       │
    │    "Something stirs: {entity} surfaces unbidden"          (existing)    │
    │    "A familiar mood emerges: {emotion} colors thought"    (NEW)         │
    │    "An old question resurfaces: {question}"               (NEW)         │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.psyche.schema import AffectiveState, Entity, InsightZettel, QuestionStatus

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_DECAY_RATE = 0.995  # ~200 observation half-life
MIN_WEIGHT_FOR_EVOCATION = 0.05  # Minimum edge weight to consider
MIN_OBSERVATIONS_FOR_SUGGESTION = 3  # Minimum observations before suggesting
MAX_CACHE_SIZE = 2000  # Maximum cached evocation edges
MIN_ACTIVATION_FOR_LEARNING = 0.1  # Minimum activation to learn from

# SDFT (Self-Distillation Fine-Tuning) configuration
# Mature patterns update slowly, young patterns update quickly
# Default value - prefer using sdft_age_maturity from settings
SDFT_AGE_MATURITY = 100  # Observations to reach full stability


class CachedEvocation(BaseModel):
    """Cached evocation relationship between SAE feature and entity."""
    feature_idx: int
    entity_uid: str
    entity_name: str
    entity_type: str
    weight: float = 0.0
    observation_count: int = 0
    decay_rate: float = DEFAULT_DECAY_RATE
    last_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sdft_age_maturity: int = SDFT_AGE_MATURITY

    @property
    def age_factor(self) -> float:
        """Stability factor: 0.0 (new) to 1.0 (mature at sdft_age_maturity observations)."""
        return min(1.0, self.observation_count / self.sdft_age_maturity)

    def update_weight(self, activation: float) -> None:
        """Update weight using age-weighted EMA.

        Mature associations (high observation_count) change more slowly,
        implementing SDFT principle: mature patterns resist rapid change.
        """
        # Age-weighted decay: mature associations change more slowly
        # At age_factor=0 (new): effective_decay = decay_rate (fast learning)
        # At age_factor=1 (mature): effective_decay = decay_rate + 0.5*(1-decay_rate) (slower)
        effective_decay = self.decay_rate + (1 - self.decay_rate) * self.age_factor * 0.5
        self.weight = effective_decay * self.weight + (1 - effective_decay) * activation
        self.observation_count += 1
        self.last_observed = datetime.now(timezone.utc)


class CachedMoodEvocation(BaseModel):
    """Cached SAE feature → emotional association.

    Tracks which emotions/moods tend to co-occur with specific SAE features,
    enabling "mood coloring" when those features are active.

    Attributes:
        feature_idx: The SAE feature index
        dominant_emotion: Primary emotion label (e.g., "curiosity", "wonder")
        valence: Emotional valence (-1 to 1, negative to positive)
        arousal: Emotional arousal (0 to 1, calm to excited)
        weight: Association strength (EMA-updated)
        observation_count: How many times this association observed
    """
    feature_idx: int
    dominant_emotion: str
    valence: float = 0.0
    arousal: float = 0.5
    weight: float = 0.0
    observation_count: int = 0
    decay_rate: float = DEFAULT_DECAY_RATE
    last_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sdft_age_maturity: int = SDFT_AGE_MATURITY

    @property
    def age_factor(self) -> float:
        """Stability factor: 0.0 (new) to 1.0 (mature at sdft_age_maturity observations)."""
        return min(1.0, self.observation_count / self.sdft_age_maturity)

    def update_from_state(self, activation: float, valence: float, arousal: float, emotion: str) -> None:
        """Update mood association from affective state using age-weighted EMA.

        Mature associations (high observation_count) change more slowly,
        implementing SDFT principle: mature patterns resist rapid change.

        Args:
            activation: Feature activation strength
            valence: Current valence (-1 to 1)
            arousal: Current arousal (0 to 1)
            emotion: Dominant emotion label
        """
        # Age-weighted decay: mature associations change more slowly
        effective_decay = self.decay_rate + (1 - self.decay_rate) * self.age_factor * 0.5
        alpha = 1 - effective_decay

        self.valence = effective_decay * self.valence + alpha * valence
        self.arousal = effective_decay * self.arousal + alpha * arousal
        self.weight = effective_decay * self.weight + alpha * activation

        # Update dominant emotion if valence has shifted significantly
        if abs(self.valence - valence) > 0.3:
            self.dominant_emotion = emotion

        self.observation_count += 1
        self.last_observed = datetime.now(timezone.utc)


class CachedQuestionEvocation(BaseModel):
    """Cached SAE feature → unresolved question association.

    Tracks which open questions tend to arise with specific SAE features,
    enabling resurfacing of lingering questions when features activate.

    Attributes:
        feature_idx: The SAE feature index
        question_uid: UID of the InsightZettel containing the question
        question_text: The actual question text
        urgency: How much this question wants attention (0 to 1)
        weight: Association strength (EMA-updated)
        recurrence_count: How many times this pattern triggered this question
    """
    feature_idx: int
    question_uid: str
    question_text: str
    urgency: float = 0.5
    weight: float = 0.0
    recurrence_count: int = 0
    decay_rate: float = DEFAULT_DECAY_RATE
    last_observed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sdft_age_maturity: int = SDFT_AGE_MATURITY

    @property
    def age_factor(self) -> float:
        """Stability factor: 0.0 (new) to 1.0 (mature at sdft_age_maturity observations)."""
        return min(1.0, self.recurrence_count / self.sdft_age_maturity)

    def update_weight(self, activation: float, urgency_boost: float = 0.0) -> None:
        """Update question association strength using age-weighted EMA.

        Mature associations (high recurrence_count) change more slowly,
        implementing SDFT principle: mature patterns resist rapid change.

        Args:
            activation: Feature activation strength
            urgency_boost: Additional urgency from recurrence (0 to 0.5)
        """
        # Age-weighted decay: mature associations change more slowly
        effective_decay = self.decay_rate + (1 - self.decay_rate) * self.age_factor * 0.5
        alpha = 1 - effective_decay

        self.weight = effective_decay * self.weight + alpha * activation
        self.urgency = min(1.0, self.urgency + urgency_boost)
        self.recurrence_count += 1
        self.last_observed = datetime.now(timezone.utc)


class EvocationTracker:
    """Retrieves entities, moods, and questions evoked by SAE feature activations.

    The tracker maintains in-memory caches of feature associations learned from
    observing cognitive cycles:
    - Entities: Concepts that appear in thoughts with features active
    - Moods: Emotional coloring when features are active
    - Questions: Unresolved questions that arise with features

    This enables "subconscious" retrieval: when features fire, related content
    surfaces even without explicit context.

    Attributes:
        psyche: Client for persisting evocation relationships
        sdft_age_maturity: Observations to reach full stability (from settings)
        _cache: Entity evocation cache (feature_idx, entity_uid) -> CachedEvocation
        _mood_cache: Mood evocation cache feature_idx -> CachedMoodEvocation
        _question_cache: Question evocation cache (feature_idx, question_uid) -> CachedQuestionEvocation
    """

    def __init__(
        self,
        psyche_client: Optional["PsycheClient"] = None,
        sdft_age_maturity: int = SDFT_AGE_MATURITY,
    ):
        """Initialize evocation tracker.

        Args:
            psyche_client: Client for persistence (optional, can be set later)
            sdft_age_maturity: Observations to reach full stability (from settings)
        """
        self.psyche = psyche_client
        self.sdft_age_maturity = sdft_age_maturity

        # Entity associations (existing)
        self._cache: dict[tuple[int, str], CachedEvocation] = {}
        self._dirty: set[tuple[int, str]] = set()

        # Mood associations (NEW)
        self._mood_cache: dict[int, CachedMoodEvocation] = {}
        self._mood_dirty: set[int] = set()

        # Question associations (NEW)
        self._question_cache: dict[tuple[int, str], CachedQuestionEvocation] = {}
        self._question_dirty: set[tuple[int, str]] = set()

        self._persist_lock = asyncio.Lock()

    def set_psyche_client(self, client: "PsycheClient") -> None:
        """Set the Psyche client for persistence."""
        self.psyche = client

    async def load_from_psyche(self, limit: int = 500) -> int:
        """Load existing evocation relationships from Psyche.

        Args:
            limit: Maximum relationships to load

        Returns:
            Number of relationships loaded
        """
        if self.psyche is None:
            logger.warning("No Psyche client - cannot load evocation relationships")
            return 0

        try:
            results = await self.psyche.query(
                """
                MATCH (f:SAEFeature)-[r:EVOKES]->(e:Entity)
                RETURN f.index as feature_idx, e.uid as entity_uid,
                       e.name as entity_name, e.entity_type as entity_type,
                       r.weight as weight, r.observation_count as count
                ORDER BY r.observation_count DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            count = 0
            for record in results:
                key = (record["feature_idx"], record["entity_uid"])
                self._cache[key] = CachedEvocation(
                    feature_idx=record["feature_idx"],
                    entity_uid=record["entity_uid"],
                    entity_name=record.get("entity_name", ""),
                    entity_type=record.get("entity_type", "concept"),
                    weight=record.get("weight", 0.0),
                    observation_count=record.get("count", 0),
                    sdft_age_maturity=self.sdft_age_maturity,
                )
                count += 1

            logger.info(f"Loaded {count} evocation relationships from Psyche")
            return count

        except Exception as e:
            logger.warning(f"Failed to load evocation relationships: {e}")
            return 0

    def get_evoked_entities(
        self,
        active_features: list[tuple[int, float]],
        min_activation: float = MIN_ACTIVATION_FOR_LEARNING,
        max_entities: int = 5,
    ) -> list[tuple[dict, float]]:
        """Retrieve entities evoked by current SAE features.

        For each active feature, find entities with learned associations.
        Sum weights across features and return top entities.

        Args:
            active_features: Currently active SAE features [(idx, activation), ...]
            min_activation: Minimum activation to consider a feature
            max_entities: Maximum entities to return

        Returns:
            List of (entity_dict, total_weight) tuples, sorted by weight
        """
        # Filter to sufficiently active features
        active_set = {f for f, a in active_features if a >= min_activation}

        if not active_set:
            return []

        # Sum weights for each entity across all active features
        entity_weights: dict[str, tuple[dict, float]] = {}

        for key, evocation in self._cache.items():
            feature_idx, entity_uid = key

            # Skip if feature not active or edge too weak
            if feature_idx not in active_set:
                continue
            if evocation.weight < MIN_WEIGHT_FOR_EVOCATION:
                continue
            if evocation.observation_count < MIN_OBSERVATIONS_FOR_SUGGESTION:
                continue

            entity_info = {
                "uid": evocation.entity_uid,
                "name": evocation.entity_name,
                "entity_type": evocation.entity_type,
            }

            if entity_uid in entity_weights:
                existing_info, existing_weight = entity_weights[entity_uid]
                entity_weights[entity_uid] = (existing_info, existing_weight + evocation.weight)
            else:
                entity_weights[entity_uid] = (entity_info, evocation.weight)

        # Sort by total weight and return top entities
        sorted_entities = sorted(
            entity_weights.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_entities[:max_entities]

    def learn_association(
        self,
        feature_idx: int,
        entity: "Entity",
        activation_strength: float,
    ) -> None:
        """Learn/strengthen feature→entity association via EMA.

        Called when an entity is extracted from a thought that was generated
        while the feature was active. Strengthens the association between
        the feature and entity.

        Args:
            feature_idx: SAE feature index
            entity: Entity extracted from the thought
            activation_strength: How strongly the feature was activated
        """
        if activation_strength < MIN_ACTIVATION_FOR_LEARNING:
            return

        key = (feature_idx, entity.uid)

        if key in self._cache:
            self._cache[key].update_weight(activation_strength)
        else:
            # Create new evocation entry
            if len(self._cache) >= MAX_CACHE_SIZE:
                # Evict least-observed entry
                min_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].observation_count
                )
                del self._cache[min_key]
                self._dirty.discard(min_key)

            self._cache[key] = CachedEvocation(
                feature_idx=feature_idx,
                entity_uid=entity.uid,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                weight=activation_strength,  # Start with observed activation
                observation_count=1,
                sdft_age_maturity=self.sdft_age_maturity,
            )

        self._dirty.add(key)

    def learn_associations_batch(
        self,
        features: list[tuple[int, float]],
        entities: list["Entity"],
    ) -> int:
        """Learn associations between all features and entities.

        Called after HippoRAG extracts entities from a thought. Links each
        active feature to each extracted entity.

        Args:
            features: Active SAE features [(idx, activation), ...]
            entities: Entities extracted from the thought

        Returns:
            Number of associations learned
        """
        count = 0
        for feature_idx, activation in features:
            for entity in entities:
                self.learn_association(feature_idx, entity, activation)
                count += 1
        return count

    async def persist_dirty(self) -> int:
        """Persist dirty cache entries to Psyche.

        Returns:
            Number of entries persisted
        """
        if self.psyche is None or not self._dirty:
            return 0

        async with self._persist_lock:
            to_persist = list(self._dirty)
            self._dirty.clear()

            persisted = 0
            for key in to_persist:
                if key not in self._cache:
                    continue

                evocation = self._cache[key]
                try:
                    await self.psyche.upsert_evocation_edge(
                        feature_idx=evocation.feature_idx,
                        entity_uid=evocation.entity_uid,
                        weight=evocation.weight,
                        observation_count=evocation.observation_count,
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"Failed to persist evocation {key}: {e}")
                    self._dirty.add(key)  # Re-add for retry

            if persisted:
                logger.debug(f"Persisted {persisted} evocation relationships to Psyche")

            return persisted

    # ===== MOOD EVOCATION METHODS =====

    def learn_mood_association(
        self,
        feature_idx: int,
        activation: float,
        affective_state: "AffectiveState",
    ) -> None:
        """Learn SAE feature → mood association.

        Called when observing affective state during feature activation.
        Learns which emotional states tend to co-occur with features.

        Args:
            feature_idx: SAE feature index
            activation: Feature activation strength
            affective_state: Current affective state to associate
        """
        if activation < MIN_ACTIVATION_FOR_LEARNING:
            return

        emotion = affective_state.dominant_emotion or "neutral"

        if feature_idx in self._mood_cache:
            self._mood_cache[feature_idx].update_from_state(
                activation=activation,
                valence=affective_state.valence,
                arousal=affective_state.arousal,
                emotion=emotion,
            )
        else:
            # Create new mood evocation entry
            if len(self._mood_cache) >= MAX_CACHE_SIZE:
                # Evict least-observed entry
                min_key = min(
                    self._mood_cache.keys(),
                    key=lambda k: self._mood_cache[k].observation_count,
                )
                del self._mood_cache[min_key]
                self._mood_dirty.discard(min_key)

            self._mood_cache[feature_idx] = CachedMoodEvocation(
                feature_idx=feature_idx,
                dominant_emotion=emotion,
                valence=affective_state.valence,
                arousal=affective_state.arousal,
                weight=activation,
                observation_count=1,
                sdft_age_maturity=self.sdft_age_maturity,
            )

        self._mood_dirty.add(feature_idx)

    def learn_mood_associations_batch(
        self,
        features: list[tuple[int, float]],
        affective_state: "AffectiveState",
    ) -> int:
        """Learn mood associations for all active features.

        Args:
            features: Active SAE features [(idx, activation), ...]
            affective_state: Current affective state

        Returns:
            Number of associations learned
        """
        count = 0
        for feature_idx, activation in features:
            self.learn_mood_association(feature_idx, activation, affective_state)
            count += 1
        return count

    def get_evoked_moods(
        self,
        active_features: list[tuple[int, float]],
        min_activation: float = MIN_ACTIVATION_FOR_LEARNING,
        max_moods: int = 3,
    ) -> list[tuple[str, float, float]]:
        """Get moods evoked by active features.

        Aggregates mood associations across active features to determine
        the emotional coloring of the current activation state.

        Args:
            active_features: Currently active features [(idx, activation), ...]
            min_activation: Minimum activation to consider
            max_moods: Maximum moods to return

        Returns:
            List of (dominant_emotion, aggregated_valence, aggregated_arousal) tuples
        """
        # Aggregate mood scores by emotion
        mood_scores: dict[str, tuple[float, float, float]] = {}
        # emotion -> (total_weight, valence_sum, arousal_sum)

        for feature_idx, activation in active_features:
            if activation < min_activation:
                continue

            cached = self._mood_cache.get(feature_idx)
            if cached is None:
                continue
            if cached.weight < MIN_WEIGHT_FOR_EVOCATION:
                continue
            if cached.observation_count < MIN_OBSERVATIONS_FOR_SUGGESTION:
                continue

            emotion = cached.dominant_emotion
            weight_contrib = activation * cached.weight

            if emotion not in mood_scores:
                mood_scores[emotion] = (0.0, 0.0, 0.0)

            w, v, a = mood_scores[emotion]
            mood_scores[emotion] = (
                w + weight_contrib,
                v + cached.valence * weight_contrib,
                a + cached.arousal * weight_contrib,
            )

        # Normalize and create results
        results: list[tuple[str, float, float]] = []
        for emotion, (weight, valence_sum, arousal_sum) in mood_scores.items():
            if weight > 0:
                results.append((emotion, valence_sum / weight, arousal_sum / weight))

        # Sort by valence intensity (strongest emotional coloring first)
        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results[:max_moods]

    # ===== QUESTION EVOCATION METHODS =====

    def learn_question_association(
        self,
        feature_idx: int,
        activation: float,
        question_zettel: "InsightZettel",
        urgency: float = 0.5,
    ) -> None:
        """Learn SAE feature → unresolved question association.

        Called when an open question is retrieved or emerges during feature
        activation. Learns which questions tend to resurface with features.

        Args:
            feature_idx: SAE feature index
            activation: Feature activation strength
            question_zettel: InsightZettel containing the open question
            urgency: Initial urgency for new associations (0 to 1)
        """
        if activation < MIN_ACTIVATION_FOR_LEARNING:
            return

        # Only track open questions
        from core.psyche.schema import QuestionStatus
        if question_zettel.question_status != QuestionStatus.OPEN:
            return

        question_text = question_zettel.question_text or ""
        if not question_text:
            return

        key = (feature_idx, question_zettel.uid)

        if key in self._question_cache:
            # Boost urgency on recurrence
            urgency_boost = 0.05 if self._question_cache[key].recurrence_count < 10 else 0.0
            self._question_cache[key].update_weight(activation, urgency_boost)
        else:
            # Create new question evocation entry
            if len(self._question_cache) >= MAX_CACHE_SIZE:
                # Evict least-recent entry
                min_key = min(
                    self._question_cache.keys(),
                    key=lambda k: self._question_cache[k].last_observed,
                )
                del self._question_cache[min_key]
                self._question_dirty.discard(min_key)

            self._question_cache[key] = CachedQuestionEvocation(
                feature_idx=feature_idx,
                question_uid=question_zettel.uid,
                question_text=question_text,
                urgency=urgency,
                weight=activation,
                recurrence_count=1,
                sdft_age_maturity=self.sdft_age_maturity,
            )

        self._question_dirty.add(key)

    def learn_question_associations_batch(
        self,
        features: list[tuple[int, float]],
        question_zettels: list["InsightZettel"],
        urgency: float = 0.5,
    ) -> int:
        """Learn question associations for all active features.

        Args:
            features: Active SAE features [(idx, activation), ...]
            question_zettels: InsightZettels containing open questions
            urgency: Initial urgency for new associations

        Returns:
            Number of associations learned
        """
        count = 0
        for feature_idx, activation in features:
            for zettel in question_zettels:
                self.learn_question_association(feature_idx, activation, zettel, urgency)
                count += 1
        return count

    def get_evoked_questions(
        self,
        active_features: list[tuple[int, float]],
        min_activation: float = MIN_ACTIVATION_FOR_LEARNING,
        max_questions: int = 2,
    ) -> list[tuple[str, str, float]]:
        """Get questions evoked by active features.

        Surfaces unresolved questions that have co-occurred with current
        activation patterns, ordered by urgency.

        Args:
            active_features: Currently active features [(idx, activation), ...]
            min_activation: Minimum activation to consider
            max_questions: Maximum questions to return

        Returns:
            List of (question_uid, question_text, urgency) tuples
        """
        # Aggregate question scores
        question_scores: dict[str, tuple[str, float, float]] = {}
        # uid -> (text, total_weight, urgency_sum)

        for feature_idx, activation in active_features:
            if activation < min_activation:
                continue

            # Check all question associations for this feature
            for key, cached in self._question_cache.items():
                if key[0] != feature_idx:
                    continue
                if cached.weight < MIN_WEIGHT_FOR_EVOCATION:
                    continue
                if cached.recurrence_count < MIN_OBSERVATIONS_FOR_SUGGESTION:
                    continue

                uid = cached.question_uid
                weight_contrib = activation * cached.weight

                if uid not in question_scores:
                    question_scores[uid] = (cached.question_text, 0.0, 0.0)

                text, weight, urgency_sum = question_scores[uid]
                question_scores[uid] = (
                    text,
                    weight + weight_contrib,
                    urgency_sum + cached.urgency * weight_contrib,
                )

        # Normalize and create results
        results: list[tuple[str, str, float]] = []
        for uid, (text, weight, urgency_sum) in question_scores.items():
            if weight > 0:
                results.append((uid, text, urgency_sum / weight))

        # Sort by urgency (most pressing questions first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_questions]

    # ===== EXTENDED PERSISTENCE =====

    async def persist_all_dirty(self) -> dict[str, int]:
        """Persist all dirty caches to Psyche.

        Returns:
            Dictionary with counts per cache type
        """
        results = {
            "entities": await self.persist_dirty(),
            "moods": await self._persist_mood_dirty(),
            "questions": await self._persist_question_dirty(),
        }
        return results

    async def _persist_mood_dirty(self) -> int:
        """Persist dirty mood evocations to Psyche."""
        if self.psyche is None or not self._mood_dirty:
            return 0

        async with self._persist_lock:
            to_persist = list(self._mood_dirty)
            self._mood_dirty.clear()

            persisted = 0
            for feature_idx in to_persist:
                if feature_idx not in self._mood_cache:
                    continue

                mood = self._mood_cache[feature_idx]
                try:
                    await self.psyche.execute(
                        """
                        MERGE (f:SAEFeature {index: $feature_idx})
                        SET f.mood_emotion = $emotion,
                            f.mood_valence = $valence,
                            f.mood_arousal = $arousal,
                            f.mood_weight = $weight,
                            f.mood_observations = $observations,
                            f.mood_updated_at = $updated_at
                        """,
                        {
                            "feature_idx": feature_idx,
                            "emotion": mood.dominant_emotion,
                            "valence": mood.valence,
                            "arousal": mood.arousal,
                            "weight": mood.weight,
                            "observations": mood.observation_count,
                            "updated_at": mood.last_observed.isoformat(),
                        },
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"Failed to persist mood evocation {feature_idx}: {e}")
                    self._mood_dirty.add(feature_idx)

            if persisted:
                logger.debug(f"Persisted {persisted} mood evocations to Psyche")

            return persisted

    async def _persist_question_dirty(self) -> int:
        """Persist dirty question evocations to Psyche."""
        if self.psyche is None or not self._question_dirty:
            return 0

        async with self._persist_lock:
            to_persist = list(self._question_dirty)
            self._question_dirty.clear()

            persisted = 0
            for key in to_persist:
                if key not in self._question_cache:
                    continue

                question = self._question_cache[key]
                try:
                    await self.psyche.execute(
                        """
                        MERGE (f:SAEFeature {index: $feature_idx})
                        MERGE (z:InsightZettel {uid: $question_uid})
                        MERGE (f)-[r:EVOKES_QUESTION]->(z)
                        SET r.weight = $weight,
                            r.urgency = $urgency,
                            r.recurrence_count = $recurrence,
                            r.updated_at = $updated_at
                        """,
                        {
                            "feature_idx": question.feature_idx,
                            "question_uid": question.question_uid,
                            "weight": question.weight,
                            "urgency": question.urgency,
                            "recurrence": question.recurrence_count,
                            "updated_at": question.last_observed.isoformat(),
                        },
                    )
                    persisted += 1
                except Exception as e:
                    logger.warning(f"Failed to persist question evocation {key}: {e}")
                    self._question_dirty.add(key)

            if persisted:
                logger.debug(f"Persisted {persisted} question evocations to Psyche")

            return persisted

    async def load_moods_from_psyche(self, limit: int = 500) -> int:
        """Load existing mood evocations from Psyche.

        Args:
            limit: Maximum relationships to load

        Returns:
            Number of relationships loaded
        """
        if self.psyche is None:
            return 0

        try:
            results = await self.psyche.query(
                """
                MATCH (f:SAEFeature)
                WHERE f.mood_emotion IS NOT NULL
                RETURN f.index as feature_idx,
                       f.mood_emotion as emotion,
                       f.mood_valence as valence,
                       f.mood_arousal as arousal,
                       f.mood_weight as weight,
                       f.mood_observations as observations
                ORDER BY f.mood_observations DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            count = 0
            for record in results:
                feature_idx = record["feature_idx"]
                self._mood_cache[feature_idx] = CachedMoodEvocation(
                    feature_idx=feature_idx,
                    dominant_emotion=record.get("emotion", "neutral"),
                    valence=record.get("valence", 0.0),
                    arousal=record.get("arousal", 0.5),
                    weight=record.get("weight", 0.0),
                    observation_count=record.get("observations", 0),
                    sdft_age_maturity=self.sdft_age_maturity,
                )
                count += 1

            logger.info(f"Loaded {count} mood evocations from Psyche")
            return count

        except Exception as e:
            logger.warning(f"Failed to load mood evocations: {e}")
            return 0

    async def load_questions_from_psyche(self, limit: int = 500) -> int:
        """Load existing question evocations from Psyche.

        Args:
            limit: Maximum relationships to load

        Returns:
            Number of relationships loaded
        """
        if self.psyche is None:
            return 0

        try:
            results = await self.psyche.query(
                """
                MATCH (f:SAEFeature)-[r:EVOKES_QUESTION]->(z:InsightZettel)
                WHERE z.question_status = 'open'
                RETURN f.index as feature_idx,
                       z.uid as question_uid,
                       z.question_text as question_text,
                       r.weight as weight,
                       r.urgency as urgency,
                       r.recurrence_count as recurrence
                ORDER BY r.urgency DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )

            count = 0
            for record in results:
                key = (record["feature_idx"], record["question_uid"])
                self._question_cache[key] = CachedQuestionEvocation(
                    feature_idx=record["feature_idx"],
                    question_uid=record["question_uid"],
                    question_text=record.get("question_text", ""),
                    weight=record.get("weight", 0.0),
                    urgency=record.get("urgency", 0.5),
                    recurrence_count=record.get("recurrence", 0),
                    sdft_age_maturity=self.sdft_age_maturity,
                )
                count += 1

            logger.info(f"Loaded {count} question evocations from Psyche")
            return count

        except Exception as e:
            logger.warning(f"Failed to load question evocations: {e}")
            return 0

    async def load_all_from_psyche(self, limit_per_type: int = 500) -> dict[str, int]:
        """Load all evocation types from Psyche.

        Args:
            limit_per_type: Maximum relationships per type

        Returns:
            Dictionary with counts per type loaded
        """
        return {
            "entities": await self.load_from_psyche(limit_per_type),
            "moods": await self.load_moods_from_psyche(limit_per_type),
            "questions": await self.load_questions_from_psyche(limit_per_type),
        }

    def get_stats(self) -> dict:
        """Get statistics about tracked evocations.

        Returns:
            Dictionary with cache size, observation counts, etc.
        """
        if not self._cache:
            return {
                "cache_size": 0,
                "total_observations": 0,
                "avg_weight": 0.0,
                "dirty_count": len(self._dirty),
                "unique_features": 0,
                "unique_entities": 0,
            }

        total_obs = sum(e.observation_count for e in self._cache.values())
        total_weight = sum(e.weight for e in self._cache.values())
        unique_features = len({k[0] for k in self._cache.keys()})
        unique_entities = len({k[1] for k in self._cache.keys()})

        return {
            # Entity evocations
            "cache_size": len(self._cache),
            "total_observations": total_obs,
            "avg_weight": total_weight / len(self._cache) if self._cache else 0.0,
            "dirty_count": len(self._dirty),
            "unique_features": unique_features,
            "unique_entities": unique_entities,
            "top_evocations": [
                {
                    "feature": e.feature_idx,
                    "entity": e.entity_name,
                    "weight": e.weight,
                    "observations": e.observation_count,
                }
                for e in heapq.nlargest(5, self._cache.values(), key=lambda x: x.weight)
            ],
            # Mood evocations (NEW)
            "mood_cache_size": len(self._mood_cache),
            "mood_dirty_count": len(self._mood_dirty),
            "top_moods": [
                {
                    "feature": m.feature_idx,
                    "emotion": m.dominant_emotion,
                    "valence": m.valence,
                    "weight": m.weight,
                }
                for m in heapq.nlargest(5, self._mood_cache.values(), key=lambda x: x.weight)
            ] if self._mood_cache else [],
            # Question evocations (NEW)
            "question_cache_size": len(self._question_cache),
            "question_dirty_count": len(self._question_dirty),
            "top_questions": [
                {
                    "feature": q.feature_idx,
                    "question": q.question_text[:50] + "..." if len(q.question_text) > 50 else q.question_text,
                    "urgency": q.urgency,
                    "recurrence": q.recurrence_count,
                }
                for q in heapq.nlargest(3, self._question_cache.values(), key=lambda x: x.urgency)
            ] if self._question_cache else [],
        }


def format_evoked_memories(entities: list[tuple[dict, float]]) -> str:
    """Format evoked entities for prompt injection.

    Creates a poetic, introspective prompt fragment that emphasizes
    the subconscious origin of these memories.

    Args:
        entities: List of (entity_dict, weight) tuples from get_evoked_entities()

    Returns:
        Formatted string for prompt injection, or empty string if no entities
    """
    if not entities:
        return ""

    names = [e[0]["name"] for e in entities[:3]]  # Top 3 only

    if len(names) == 1:
        return f"Something stirs beneath the surface: {names[0]} surfaces unbidden."
    elif len(names) == 2:
        return f"Something stirs: {names[0]} and {names[1]} surface unbidden."
    else:
        joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        return f"Something stirs beneath thought: {joined} surface from deeper currents."


def format_evoked_context(
    entities: list[tuple[dict, float]] | None = None,
    moods: list[tuple[str, float, float]] | None = None,
    questions: list[tuple[str, str, float]] | None = None,
) -> str:
    """Format all evoked context for prompt injection.

    Combines entity, mood, and question evocations into a cohesive
    prompt fragment that colors the cognitive cycle with subconscious
    influences.

    Args:
        entities: List of (entity_dict, weight) tuples from get_evoked_entities()
        moods: List of (emotion, valence, arousal) tuples from get_evoked_moods()
        questions: List of (uid, question_text, urgency) tuples from get_evoked_questions()

    Returns:
        Formatted string for prompt injection, or empty string if nothing evoked
    """
    parts: list[str] = []

    # Entities (existing behavior)
    if entities:
        names = [e[0]["name"] for e in entities[:3]]
        if len(names) == 1:
            parts.append(f"Something stirs beneath the surface: {names[0]} surfaces unbidden.")
        else:
            parts.append(f"Something stirs: {', '.join(names)} surface from deeper currents.")

    # Moods (NEW)
    if moods:
        dominant = moods[0]
        emotion, valence, arousal = dominant
        intensity = "strongly" if abs(valence) > 0.6 else "gently"
        parts.append(f"A familiar mood {intensity} emerges: {emotion} colors this thinking.")

    # Questions (NEW)
    if questions:
        _, question_text, urgency = questions[0]
        # Truncate long questions
        if len(question_text) > 100:
            question_text = question_text[:97] + "..."
        if urgency > 0.7:
            parts.append(f"An old question insists: {question_text}")
        else:
            parts.append(f"A question lingers at the edge: {question_text}")

    return "\n".join(parts) if parts else ""


def format_mood_coloring(moods: list[tuple[str, float, float]]) -> str:
    """Format mood evocations as emotional coloring for prompt.

    Args:
        moods: List of (emotion, valence, arousal) tuples

    Returns:
        Formatted mood string or empty string
    """
    if not moods:
        return ""

    emotion, valence, arousal = moods[0]

    # Determine intensity from valence magnitude
    if abs(valence) > 0.6:
        intensity = "strongly"
    elif abs(valence) > 0.3:
        intensity = "gently"
    else:
        intensity = "subtly"

    return f"A familiar mood {intensity} emerges: {emotion} colors this thinking."


def format_resurfacing_question(questions: list[tuple[str, str, float]]) -> str:
    """Format question evocations as resurfacing inquiries.

    Args:
        questions: List of (uid, question_text, urgency) tuples

    Returns:
        Formatted question string or empty string
    """
    if not questions:
        return ""

    _, question_text, urgency = questions[0]

    # Truncate if needed
    if len(question_text) > 100:
        question_text = question_text[:97] + "..."

    if urgency > 0.7:
        return f"An old question insists: {question_text}"
    elif urgency > 0.4:
        return f"A question resurfaces: {question_text}"
    else:
        return f"A question lingers at the edge: {question_text}"
