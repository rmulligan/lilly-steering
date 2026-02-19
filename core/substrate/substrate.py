"""Main FeatureSubstrate orchestrator for emergent cognition.

The FeatureSubstrate coordinates all substrate components:
- ActivationBuffer: Rolling window of recent feature activations
- TraceMatrix: Hebbian co-activation patterns
- EmbeddingSpace: Dense feature embeddings with attractors
- ValueSignal: Composite value for learning modulation
- ConsolidationManager: Dream-cycle memory consolidation

This is the primary interface for the cognitive loop to interact
with the substrate layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from core.substrate.activation_buffer import ActivationBuffer
from core.substrate.consolidation import ConsolidationManager
from core.substrate.embedding_space import EmbeddingSpace
from core.substrate.schemas import (
    Attractor,
    AttractorType,
    DreamCycleType,
    EvokedContext,
    EvokedEntity,
    EvokedMood,
    EvokedQuestion,
    EvokedZettel,
    FeatureActivation,
    PhaseTransition,
    SubstrateHealth,
    SubstratePhase,
)
from core.substrate.trace_matrix import TraceMatrix
from core.substrate.value_signal import ValueSignal

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from config.settings import Settings

logger = logging.getLogger(__name__)


class FeatureSubstrate:
    """Main orchestrator coordinating all substrate layers.

    The substrate provides:
    1. Associative memory surfacing via attractors
    2. Self-organizing attention via trace patterns
    3. Feature-based personality traits via value weighting
    4. Predictive feature anticipation via priming

    Lifecycle phases:
    - BOOTSTRAP: First ~1000 cycles, equal value weights
    - WEIGHT_LEARNING: Cycles ~1000-5000, weights learned from endorsements
    - SELF_COHERENCE: Ongoing, self-coherence signal dominates

    Attributes:
        buffer: Rolling window of recent feature activations
        trace: Hebbian co-activation matrix
        embeddings: Dense feature embedding space with attractors
        value: Composite value signal for learning
        consolidation: Dream-cycle memory consolidation manager
        psyche: FalkorDB client for graph operations
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        settings: "Settings",
    ):
        """Initialize the substrate with all components.

        Args:
            psyche: PsycheClient for graph operations
            settings: Application settings with substrate configuration
        """
        self.psyche = psyche
        self.settings = settings

        # Extract configuration with defaults
        n_features = getattr(settings, "substrate_n_features", 163840)
        embed_dim = getattr(settings, "substrate_embed_dim", 64)
        buffer_capacity = getattr(settings, "substrate_buffer_capacity", 10)

        # Initialize all components
        self.buffer = ActivationBuffer(
            capacity=buffer_capacity,
            n_features=n_features,
        )
        self.trace = TraceMatrix(n_features=n_features)
        self.embeddings = EmbeddingSpace(
            n_features=n_features,
            embed_dim=embed_dim,
        )
        self.value = ValueSignal()
        self.consolidation = ConsolidationManager(
            buffer=self.buffer,
            trace=self.trace,
            embeddings=self.embeddings,
        )

        # State tracking
        self._total_observations = 0
        self._last_surprise = 0.0
        self._last_value = 0.0
        self._phase_history: list[PhaseTransition] = []

        logger.info(
            "FeatureSubstrate initialized: n_features=%d, embed_dim=%d",
            n_features,
            embed_dim,
        )

    @property
    def total_observations(self) -> int:
        """Return total number of observations processed."""
        return self._total_observations

    @property
    def phase(self) -> SubstratePhase:
        """Return current lifecycle phase."""
        return self.value.detect_phase(self._total_observations)

    async def observe(
        self,
        features: list[FeatureActivation],
        surprise: float = 0.0,
        insight: float = 0.0,
        narration: float = 0.0,
        feedback: float = 0.0,
    ) -> float:
        """Process an observation of feature activations.

        This is called each cognitive cycle with the current SAE features.
        It updates all substrate components and computes the value signal.

        Args:
            features: Currently active features with activations
            surprise: Surprise/information gain signal (0-1)
            insight: Insight extraction signal (0-1)
            narration: Narration engagement signal (0-1)
            feedback: External feedback signal (0-1)

        Returns:
            The computed composite value signal
        """
        # Add to activation buffer
        self.buffer.add(features)

        # Compute self-coherence from attractor alignment
        self_coherence = self._compute_self_coherence(features)

        # Compute and record value signal
        value = self.value.compute(
            surprise=surprise,
            insight=insight,
            narration=narration,
            feedback=feedback,
            self_coherence=self_coherence,
        )

        # Record snapshot for endorsement learning
        self.value.snapshot(
            surprise=surprise,
            insight=insight,
            narration=narration,
            feedback=feedback,
            self_coherence=self_coherence,
        )

        # Update trace with Hebbian learning
        if len(features) >= 2 and value > 0:
            self.trace.hebbian_update(features, value)

        # Apply attractor pull to active features
        self.embeddings.apply_attractor_pull(features)

        # Update state
        self._last_surprise = surprise
        self._last_value = value
        self._total_observations += 1

        # Check for phase transitions
        self._check_phase_transition()

        return value

    async def get_evoked(
        self,
        active_features: list[FeatureActivation],
        threshold: float = 0.5,
    ) -> EvokedContext:
        """Get memories evoked by current feature activations.

        Finds attractors near active features and returns associated
        entities, zettels, moods, and questions.

        Args:
            active_features: Currently active features
            threshold: Minimum activation to include in result

        Returns:
            EvokedContext with surfaced memories
        """
        entities: list[EvokedEntity] = []
        zettels: list[EvokedZettel] = []
        moods: list[EvokedMood] = []
        questions: list[EvokedQuestion] = []

        if not active_features:
            return EvokedContext()

        # Get attractor activations
        attractor_acts = self.embeddings.get_attractor_activations(active_features)

        for uid, activation in attractor_acts.items():
            if activation < threshold:
                continue

            attractor = self.embeddings.get_attractor(uid)
            if not attractor:
                continue

            # Update attractor visit count
            attractor.visit_count += 1

            if attractor.attractor_type == AttractorType.ENTITY:
                entities.append(
                    EvokedEntity(
                        uid=attractor.source_uid,
                        name=attractor.source_name,
                        entity_type="concept",  # Could be enriched from graph
                        activation=activation,
                    )
                )
            elif attractor.attractor_type == AttractorType.ZETTEL:
                zettels.append(
                    EvokedZettel(
                        uid=attractor.source_uid,
                        insight=attractor.source_name,
                        activation=activation,
                    )
                )
            elif attractor.attractor_type == AttractorType.MOOD:
                # Parse mood info from source_name (format: "name:valence:arousal")
                # Use rsplit to handle mood names that may contain colons
                try:
                    parts = attractor.source_name.rsplit(":", maxsplit=2)
                    if len(parts) == 3:
                        # Full format: "name:valence:arousal"
                        name = parts[0] if parts[0] else "unknown"
                        valence = float(parts[1])
                        arousal = float(parts[2])
                    elif len(parts) == 2:
                        # Partial format: "name:valence" (arousal defaults to 0.0)
                        name = parts[0] if parts[0] else "unknown"
                        valence = float(parts[1])
                        arousal = 0.0
                    else:
                        # Just name or empty string
                        name = parts[0] if parts and parts[0] else "unknown"
                        valence = 0.0
                        arousal = 0.0
                except (ValueError, IndexError, AttributeError) as e:
                    # Fallback for malformed data
                    logger.warning(
                        "Failed to parse mood from source_name '%s': %s",
                        attractor.source_name,
                        e,
                    )
                    name = attractor.source_name or "unknown"
                    valence = 0.0
                    arousal = 0.0
                moods.append(
                    EvokedMood(
                        name=name,
                        valence=valence,
                        arousal=arousal,
                        activation=activation,
                    )
                )
            elif attractor.attractor_type == AttractorType.EMERGENT:
                # Emergent attractors can evoke questions
                questions.append(
                    EvokedQuestion(
                        uid=attractor.source_uid,
                        question=attractor.source_name,
                        activation=activation,
                    )
                )

        return EvokedContext(
            entities=entities,
            zettels=zettels,
            moods=moods,
            questions=questions,
        )

    async def get_primed_features(
        self,
        active_indices: list[int],
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Get features primed (predicted) by currently active features.

        Uses trace matrix associations to predict which features
        are likely to activate next.

        Args:
            active_indices: Indices of currently active features
            top_k: Maximum number of primed features to return

        Returns:
            List of (feature_idx, priming_strength) tuples
        """
        return self.trace.get_primed_features(active_indices, top_k=top_k)

    async def consolidate(
        self,
        cycle_type: DreamCycleType,
        surprise: float | None = None,
    ) -> dict:
        """Run consolidation for a dream cycle.

        Transfers patterns between substrate layers at different
        timescales, modulated by surprise level.

        Args:
            cycle_type: Type of dream cycle (MICRO, NAP, FULL, DEEP)
            surprise: Override surprise level (uses last observed if None)

        Returns:
            Dict with consolidation statistics
        """
        if surprise is None:
            surprise = self._last_surprise

        stats = self.consolidation.consolidate(
            cycle_type=cycle_type,
            surprise=surprise,
            value=self._last_value,
        )

        # For FULL and DEEP cycles, persist attractors
        if cycle_type in (DreamCycleType.FULL, DreamCycleType.DEEP):
            await self._persist_attractors()

        return stats

    def health(self) -> SubstrateHealth:
        """Get health metrics for the substrate.

        Returns:
            SubstrateHealth with diagnostic metrics
        """
        return SubstrateHealth(
            attractor_count=len(self.embeddings.attractors),
            mean_attractor_strength=self._mean_attractor_strength(),
            feature_coverage=self._feature_coverage(),
            trace_sparsity=self.trace.sparsity,
            embedding_variance=self.embeddings.variance(),
            phase=self.phase,
            total_observations=self._total_observations,
        )

    async def seed_attractors_from_graph(self) -> int:
        """Bootstrap attractors from existing entities and zettels.

        Queries the graph for important entities and creates
        attractors for them in the embedding space.

        Returns:
            Number of attractors created
        """
        count = 0

        # Seed from entities
        try:
            entities = await self.psyche.query(
                """
                MATCH (e:Entity)
                WHERE e.importance > 0.5 OR e.entity_type = 'concept'
                RETURN e.uid AS uid, e.name AS name, e.entity_type AS entity_type
                LIMIT 100
                """
            )

            for entity in entities:
                attractor = self._create_attractor_for_entity(entity)
                if attractor:
                    self.embeddings.add_attractor(attractor)
                    count += 1

        except Exception as e:
            logger.warning("Failed to seed attractors from entities: %s", e)

        # Seed from zettels
        try:
            zettels = await self.psyche.query(
                """
                MATCH (z:InsightZettel)
                WHERE z.confidence > 0.7
                RETURN z.uid AS uid, z.insight AS insight
                LIMIT 50
                """
            )

            for zettel in zettels:
                attractor = self._create_attractor_for_zettel(zettel)
                if attractor:
                    self.embeddings.add_attractor(attractor)
                    count += 1

        except Exception as e:
            logger.warning("Failed to seed attractors from zettels: %s", e)

        logger.info("Seeded %d attractors from graph", count)
        return count

    def record_endorsement(self, coherence: float) -> None:
        """Record a self-coherence endorsement for weight learning.

        Called when system or human endorses recent behavior.
        Updates value signal weights based on active signals.

        Args:
            coherence: Endorsement strength (0.0 to 1.0)
        """
        self.value.record_endorsement(coherence)

    def save_state(self) -> dict:
        """Serialize substrate state for persistence.

        Returns:
            Serializable state dictionary
        """
        return {
            "total_observations": self._total_observations,
            "last_surprise": self._last_surprise,
            "last_value": self._last_value,
            "trace": self.trace.save_state(),
            "embeddings": self.embeddings.save_state(),
            "value": self.value.save_state(),
            "phase_history": [
                {
                    "from_phase": t.from_phase.value,
                    "to_phase": t.to_phase.value,
                    "timestamp": t.timestamp.isoformat(),
                    "weights_snapshot": t.weights_snapshot,
                    "observation_count": t.observation_count,
                }
                for t in self._phase_history
            ],
        }

    @classmethod
    def load_state(
        cls,
        state: dict,
        psyche: "PsycheClient",
        settings: "Settings",
    ) -> "FeatureSubstrate":
        """Deserialize substrate from saved state.

        Args:
            state: Previously saved state dictionary
            psyche: PsycheClient for graph operations
            settings: Application settings

        Returns:
            Reconstructed FeatureSubstrate instance
        """
        substrate = cls(psyche=psyche, settings=settings)

        # Restore counters
        substrate._total_observations = state.get("total_observations", 0)
        substrate._last_surprise = state.get("last_surprise", 0.0)
        substrate._last_value = state.get("last_value", 0.0)

        # Restore components
        if "trace" in state:
            substrate.trace = TraceMatrix.load_state(state["trace"])

        if "embeddings" in state:
            substrate.embeddings = EmbeddingSpace.load_state(state["embeddings"])

        if "value" in state:
            substrate.value = ValueSignal.load_state(state["value"])

        # Rebuild consolidation manager with restored components
        substrate.consolidation = ConsolidationManager(
            buffer=substrate.buffer,
            trace=substrate.trace,
            embeddings=substrate.embeddings,
        )

        # Restore phase history
        from datetime import datetime, timezone

        for t in state.get("phase_history", []):
            try:
                ts = datetime.fromisoformat(t["timestamp"])
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            substrate._phase_history.append(
                PhaseTransition(
                    from_phase=SubstratePhase(t["from_phase"]),
                    to_phase=SubstratePhase(t["to_phase"]),
                    timestamp=ts,
                    weights_snapshot=t.get("weights_snapshot", {}),
                    observation_count=t.get("observation_count", 0),
                )
            )

        logger.info(
            "FeatureSubstrate loaded: observations=%d, phase=%s",
            substrate._total_observations,
            substrate.phase.value,
        )
        return substrate

    def _compute_self_coherence(
        self, features: list[FeatureActivation]
    ) -> float:
        """Compute self-coherence signal from attractor alignment.

        Self-coherence measures how well current features align with
        identity-related attractors.

        Args:
            features: Currently active features

        Returns:
            Self-coherence score (0-1)
        """
        if not features or not self.embeddings.attractors:
            return 0.0

        # Find identity attractors
        identity_attractors = [
            a for a in self.embeddings.attractors.values()
            if a.attractor_type == AttractorType.IDENTITY
        ]

        if not identity_attractors:
            return 0.5  # Neutral if no identity attractors

        # Compute alignment with identity attractors
        total_alignment = 0.0
        total_activation = 0.0

        for feat in features:
            idx = feat.feature_idx
            activation = feat.activation
            emb = self.embeddings.embeddings[idx]

            for attractor in identity_attractors:
                attr_pos = np.array(attractor.position, dtype=np.float32)
                dist = np.linalg.norm(emb - attr_pos)

                # Convert distance to alignment (closer = higher alignment)
                if dist < attractor.pull_radius:
                    alignment = (1.0 - dist / attractor.pull_radius) * activation
                    total_alignment += alignment * attractor.value_weight

            total_activation += activation

        if total_activation == 0:
            return 0.5

        # Normalize to 0-1
        return min(1.0, total_alignment / (total_activation * len(identity_attractors)))

    def _check_phase_transition(self) -> None:
        """Check and record phase transitions.

        Monitors for transitions between BOOTSTRAP, WEIGHT_LEARNING,
        and SELF_COHERENCE phases.
        """
        current_phase = self.phase

        # Determine previous phase
        if self._phase_history:
            prev_phase = self._phase_history[-1].to_phase
        else:
            prev_phase = SubstratePhase.BOOTSTRAP

        # Record transition if phase changed
        if current_phase != prev_phase:
            transition = PhaseTransition(
                from_phase=prev_phase,
                to_phase=current_phase,
                weights_snapshot=dict(self.value.current_weights),
                observation_count=self._total_observations,
            )
            self._phase_history.append(transition)

            logger.info(
                "Substrate phase transition: %s -> %s at observation %d",
                prev_phase.value,
                current_phase.value,
                self._total_observations,
            )

    async def _persist_attractors(self) -> int:
        """Persist attractors to the graph.

        Creates or updates Attractor nodes in FalkorDB using a batched UNWIND query.

        Returns:
            Number of attractors persisted
        """
        if not self.embeddings.attractors:
            return 0

        # Build list of attractor dictionaries for batched insert
        attractor_list = [
            {
                "uid": uid,
                "attractor_type": attractor.attractor_type.value,
                "source_uid": attractor.source_uid,
                "source_name": attractor.source_name,
                "pull_strength": attractor.pull_strength,
                "pull_radius": attractor.pull_radius,
                "value_weight": attractor.value_weight,
                "visit_count": attractor.visit_count,
                "position": attractor.position,
            }
            for uid, attractor in self.embeddings.attractors.items()
        ]

        try:
            await self.psyche.execute(
                """
                UNWIND $attractors AS attr
                MERGE (a:Attractor {uid: attr.uid})
                SET a.attractor_type = attr.attractor_type,
                    a.source_uid = attr.source_uid,
                    a.source_name = attr.source_name,
                    a.pull_strength = attr.pull_strength,
                    a.pull_radius = attr.pull_radius,
                    a.value_weight = attr.value_weight,
                    a.visit_count = attr.visit_count,
                    a.position = attr.position
                """,
                {"attractors": attractor_list},
            )
            return len(attractor_list)
        except Exception as e:
            logger.warning("Failed to persist attractors: %s", e)
            return 0

    def _create_attractor_for_entity(
        self, entity: dict
    ) -> Attractor | None:
        """Create an attractor from an entity record.

        Args:
            entity: Entity dict from graph query

        Returns:
            Attractor or None if creation fails
        """
        uid = entity.get("uid")
        name = entity.get("name", "")

        if not uid:
            return None

        # Initialize with random position
        rng = np.random.default_rng(hash(uid) % (2**32))
        position = rng.normal(0, 1, self.embeddings.embed_dim).astype(np.float32)
        position = position / np.linalg.norm(position)

        return Attractor(
            uid=f"attr_entity_{uid}",
            attractor_type=AttractorType.ENTITY,
            position=position.tolist(),
            source_uid=uid,
            source_name=name,
        )

    def _create_attractor_for_zettel(
        self, zettel: dict
    ) -> Attractor | None:
        """Create an attractor from a zettel record.

        Args:
            zettel: Zettel dict from graph query

        Returns:
            Attractor or None if creation fails
        """
        uid = zettel.get("uid")
        insight = zettel.get("insight", "")

        if not uid:
            return None

        # Initialize with random position
        rng = np.random.default_rng(hash(uid) % (2**32))
        position = rng.normal(0, 1, self.embeddings.embed_dim).astype(np.float32)
        position = position / np.linalg.norm(position)

        return Attractor(
            uid=f"attr_zettel_{uid}",
            attractor_type=AttractorType.ZETTEL,
            position=position.tolist(),
            source_uid=uid,
            source_name=insight[:100],  # Truncate for name
        )

    def _mean_attractor_strength(self) -> float:
        """Compute mean pull strength of all attractors.

        Returns:
            Mean pull_strength or 0.0 if no attractors
        """
        if not self.embeddings.attractors:
            return 0.0

        total = sum(a.pull_strength for a in self.embeddings.attractors.values())
        return total / len(self.embeddings.attractors)

    def _feature_coverage(self) -> float:
        """Compute fraction of features near an attractor.

        Returns:
            Fraction in [0, 1]
        """
        if not self.embeddings.attractors:
            return 0.0

        # Sample features to estimate coverage
        sample_size = min(1000, self.embeddings.n_features)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(
            self.embeddings.n_features,
            size=sample_size,
            replace=False,
        )

        near_count = 0
        for idx in sample_indices:
            emb = self.embeddings.embeddings[idx]

            for attractor in self.embeddings.attractors.values():
                attr_pos = np.array(attractor.position, dtype=np.float32)
                dist = np.linalg.norm(emb - attr_pos)

                if dist < attractor.pull_radius:
                    near_count += 1
                    break  # Count each feature only once

        return near_count / sample_size
