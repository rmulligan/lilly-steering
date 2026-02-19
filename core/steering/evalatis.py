"""Hybrid Emergence-Selection Steering (EvalatisSteerer).

Combines continuous emergence (EMA from activations) with evolutionary
selection (crystallized vectors competing based on affinity and staleness).
Vectors can crystallize, spawn children, and be pruned based on performance.

Architecture:
    Per Zone:
        EmergentSlot: Live vector evolving from activations
        CrystalPopulation: Frozen high-performers competing for selection

    Selection Arena:
        Emergent competes with crystals based on surprise_ema vs affinity×staleness
        Recognition signals from human feedback influence selection and lifecycle

    Lifecycle Events:
        - Crystallization: Emergent → Population (boosted by approval)
        - Spawning: Parent pair → Child
        - Pruning: Weak crystal → Retired (protected by approval)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypedDict

import numpy as np

from core.steering.config import HierarchicalSteeringConfig, SteeringZone
from core.steering.crystal import (
    CrystalEntry,
    CrystallizationConfig,
    blend_vectors_with_mutation,
    generate_crystal_name,
)
from core.steering.emergent import EmergentSlot, create_emergent_slot
from core.steering.utils import cosine_similarity

if TYPE_CHECKING:
    from core.recognition.feature_tracker import ApprovedFeatureTracker
    from core.steering.qd.scorer import QDContext, QDScore, QDScorer

logger = logging.getLogger(__name__)

# Phase modulation factors for EMA rates
# Based on brain research: slower processing during consolidation phases
# See: Honey et al. 2012 - temporal hierarchy in neural processing
PHASE_MODULATION_FACTORS: dict[str, float] = {
    "generation": 1.0,      # Full speed - exploring new directions
    "curation": 0.75,       # Slightly slower - analyzing discoveries
    "simulation": 0.5,      # Slower - rigorous hypothesis testing
    "integration": 0.5,     # Slower - consolidating into memory
    "reflexion": 0.25,      # Much slower - meta-analysis and self-review
    "continuity": 0.25,     # Much slower - synthesis and persistence
}


class CycleEvents(TypedDict, total=False):
    """Events returned from update_from_cycle."""

    crystallized: Optional[CrystalEntry]
    spawned: Optional[CrystalEntry]
    pruned: Optional[str]  # Name of pruned crystal
    selected_name: str
    selected_is_emergent: bool


@dataclass
class ZoneState:
    """Per-zone state containing emergent slot and crystal population."""

    emergent: EmergentSlot
    crystals: List[CrystalEntry] = field(default_factory=list)
    current_selection: Optional[str] = None  # Name of currently selected vector
    current_is_emergent: bool = True
    last_sae_features: List[Tuple[int, float]] = field(default_factory=list)  # For approval tracking
    last_approval_bonus: float = 0.0  # Cached approval bonus from last cycle


class EvalatisSteerer:
    """Hybrid emergence-selection steerer with crystallization lifecycle.

    Manages steering vectors across multiple layer zones like HierarchicalSteerer,
    but adds:
    - Emergent slot per zone for live vector evolution
    - Crystal population per zone for proven high-performers
    - Selection arena where emergent competes with crystals
    - Crystallization when emergent sustains high surprise
    - Spawning when high-affinity crystal pairs combine
    - Pruning when crystals underperform

    Attributes:
        config: Hierarchical steering configuration (zones, layers).
        crystal_config: Crystallization/spawning/pruning thresholds.
        d_model: Model dimension (vector size).
        zones: Dict mapping zone names to ZoneState.
        global_cycle: Total cycles processed.
    """

    def __init__(
        self,
        config: HierarchicalSteeringConfig,
        d_model: int,
        crystal_config: Optional[CrystallizationConfig] = None,
        feature_tracker: Optional["ApprovedFeatureTracker"] = None,
        qd_scorer: Optional["QDScorer"] = None,
    ):
        """Initialize steerer with emergent slots and empty populations.

        Args:
            config: Configuration specifying zones and their parameters.
            d_model: Model hidden dimension (steering vector size).
            crystal_config: Optional crystallization configuration.
            feature_tracker: Optional ApprovedFeatureTracker for recognition signal integration.
                When provided, selection, crystallization, and pruning decisions
                factor in Ryan's approval patterns for SAE features.
            qd_scorer: Optional QDScorer for Quality Diversity-based selection.
                When provided, selection uses QD metrics (coherence, novelty,
                surprise, presence) instead of affinity×freshness.
        """
        self.config = config
        self.d_model = d_model
        self.crystal_config = crystal_config or CrystallizationConfig()
        self.feature_tracker = feature_tracker
        self.qd_scorer = qd_scorer

        # Initialize per-zone state
        self.zones: Dict[str, ZoneState] = {}
        for zone in config.zones:
            self.zones[zone.name] = ZoneState(
                emergent=create_emergent_slot(d_model),
            )

        self.global_cycle = 0

    def set_feature_tracker(self, tracker: "ApprovedFeatureTracker") -> None:
        """Set the feature tracker for recognition signal integration.

        This allows wiring the tracker after initialization, which is useful
        when the orchestrator sets up components incrementally.

        Args:
            tracker: The ApprovedFeatureTracker to use for approval scoring.
        """
        self.feature_tracker = tracker
        logger.info("EvalatisSteerer: Recognition signal integration enabled")

    def set_qd_scorer(self, scorer: "QDScorer") -> None:
        """Set the QD scorer for Quality Diversity-based selection.

        This allows wiring the scorer after initialization, which is useful
        when the orchestrator sets up components incrementally.

        Args:
            scorer: The QDScorer to use for selection.
        """
        self.qd_scorer = scorer
        logger.info("EvalatisSteerer: QD scoring enabled")

    def _get_approval_bonus(self, sae_features: List[Tuple[int, float]]) -> float:
        """Get approval bonus from feature tracker.

        Args:
            sae_features: Current SAE feature activations [(idx, activation), ...]

        Returns:
            Approval bonus in [-1, 1] range, or 0 if no tracker.
        """
        if self.feature_tracker is None or not sae_features:
            return 0.0
        return self.feature_tracker.get_approval_bonus(sae_features)

    def get_vector(self, layer: int) -> Optional[np.ndarray]:
        """Get steering vector for a layer, or None if no steering.

        Compatible with HierarchicalSteerer interface. Returns the currently
        selected vector (emergent or crystal) for the zone containing this layer.

        Args:
            layer: The transformer layer index.

        Returns:
            The steering vector for this layer (magnitude-capped),
            or None if this layer should not be steered.
        """
        if layer >= self.config.observation_layer:
            return None

        zone_config = self.config.get_zone(layer)
        if zone_config is None:
            return None

        zone = self.zones.get(zone_config.name)
        if zone is None:
            return None

        # Get currently selected vector
        if zone.current_is_emergent:
            vector = zone.emergent.vector.copy()
        else:
            crystal = self._get_crystal_by_name(zone_config.name, zone.current_selection)
            if crystal is None:
                vector = zone.emergent.vector.copy()
            else:
                vector = crystal.vector.copy()

        # Cap magnitude
        magnitude = np.linalg.norm(vector)
        if magnitude > zone_config.max_magnitude:
            vector = vector * (zone_config.max_magnitude / magnitude)

        return vector

    def update_vector(
        self,
        zone_name: str,
        new_direction: np.ndarray,
        scale: float = 1.0,
    ) -> None:
        """Update a zone's emergent vector (compatibility method).

        Provides backward compatibility with HierarchicalSteerer interface.
        For full lifecycle management, use update_from_cycle() instead.

        Args:
            zone_name: Name of the zone to update.
            new_direction: Direction vector (will be normalized).
            scale: Scale factor (used as surprise proxy for EMA update).
        """
        zone = self.zones.get(zone_name)
        zone_config = self._get_zone_config(zone_name)
        if zone is None or zone_config is None:
            return

        # Use scale as a crude surprise proxy
        surprise = scale * 50.0  # Convert to surprise range
        zone.emergent.update(new_direction, surprise, zone_config.ema_alpha)

    def adjust_zone_magnitude(self, zone_name: str, boost: float) -> None:
        """Adjust a zone's emergent vector magnitude by a boost factor.

        Used by affect steering to modulate zone strength based on
        emotional state (e.g., high arousal/curiosity boosts exploration).

        Args:
            zone_name: Name of the zone to adjust.
            boost: Additive magnitude adjustment (typically -0.05 to +0.05).
        """
        logger.info(f"[AFFECT] adjust_zone_magnitude called: zone={zone_name}, boost={boost:+.4f}")

        zone = self.zones.get(zone_name)
        if zone is None:
            logger.debug(f"[AFFECT] Zone '{zone_name}' not found in EvalatisSteerer")
            return

        vector = zone.emergent.vector
        current_mag = np.linalg.norm(vector)

        if current_mag < 1e-6:
            # No vector to scale - expected if steering hasn't been set yet
            logger.debug(f"[AFFECT] Zone '{zone_name}' emergent has zero magnitude, cannot adjust (boost={boost:+.4f})")
            return

        # Compute new magnitude (clamped to non-negative)
        new_mag = max(0.0, current_mag + boost)

        # Scale vector to new magnitude
        zone.emergent.vector = vector * (new_mag / current_mag)

        logger.debug(
            f"[AFFECT] {zone_name} magnitude adjusted: {current_mag:.3f} -> {new_mag:.3f} (boost={boost:+.4f})"
        )

    def update_from_cycle(
        self,
        zone_name: str,
        activations: np.ndarray,
        surprise: float,
        context_embedding: Optional[np.ndarray] = None,
        current_embedding: Optional[np.ndarray] = None,
        sae_features: Optional[List[Tuple[int, float]]] = None,
        hypothesis_effectiveness: Optional[float] = None,
        phase: str = "generation",
    ) -> CycleEvents:
        """Full cycle update with selection, crystallization, spawning, pruning.

        This is the primary update method for EvalatisSteerer. It:
        1. Computes approval bonus from recognition signals (if tracker configured)
        2. Selects between emergent and crystals based on scores + approval
        3. Updates the selected vector with new activations (using phase-modulated EMA)
        4. Checks for crystallization triggers (boosted by approval)
        5. Checks for spawning opportunities
        6. Prunes weak crystals if at population limit (protected by approval)

        Args:
            zone_name: Name of the zone to update.
            activations: Current activation vector for steering.
            surprise: Current cycle's surprise value.
            context_embedding: Optional embedding for affinity calculation.
            current_embedding: Optional embedding of current thought for latent
                coherence scoring (ATP-Latent inspired proactive diversity).
            sae_features: Optional SAE feature activations for approval scoring.
                When provided with a feature_tracker, enables recognition signal
                integration where Ryan's approval patterns influence steering.
            hypothesis_effectiveness: Optional average effectiveness score of active
                hypotheses in [0, 1]. When provided, blends with approval_bonus to
                enable learning from prediction verification (SS-401).
            phase: Current cognitive phase (generation, curation, simulation,
                integration, reflexion, continuity). Used to modulate EMA rates
                for brain-inspired cross-timescale integration.

        Returns:
            CycleEvents dict with any lifecycle events that occurred.
        """
        events: CycleEvents = {}
        zone = self.zones.get(zone_name)
        zone_config = self._get_zone_config(zone_name)

        if zone is None or zone_config is None:
            return events

        self.global_cycle += 1

        # 0. Compute approval bonus from recognition signals
        approval_bonus = 0.0
        cfg = self.crystal_config
        if sae_features:
            zone.last_sae_features = list(sae_features)
            approval_bonus = self._get_approval_bonus(sae_features)
            zone.last_approval_bonus = approval_bonus
            if abs(approval_bonus) > cfg.approval_logging_threshold:
                logger.debug(
                    f"Recognition signal: approval_bonus={approval_bonus:.3f} "
                    f"for {len(sae_features)} features"
                )

        # 0b. Blend hypothesis effectiveness with approval bonus (SS-401)
        # Enables learning from prediction verification outcomes
        if hypothesis_effectiveness is not None:
            # Convert effectiveness [0, 1] to bonus [-1, 1]
            hyp_bonus = (hypothesis_effectiveness - 0.5) * 2.0
            # Average both signals, clamp to [-1, 1]
            approval_bonus = np.clip((approval_bonus + hyp_bonus) / 2.0, -1.0, 1.0)
            logger.debug(
                f"Hypothesis effectiveness blended: hyp_eff={hypothesis_effectiveness:.3f}, "
                f"combined_approval={approval_bonus:.3f}"
            )

        # 1. Selection: Choose between emergent and crystals (approval influences scoring)
        selected_name, is_emergent = self._select_vector(
            zone_name, context_embedding, approval_bonus, sae_features, current_embedding
        )
        zone.current_selection = selected_name
        zone.current_is_emergent = is_emergent
        events["selected_name"] = selected_name
        events["selected_is_emergent"] = is_emergent

        # 2. Update the emergent slot (always evolving)
        # Use phase-modulated EMA alpha for brain-inspired cross-timescale integration
        effective_alpha = self.get_effective_ema_alpha(zone_name, phase)
        baseline = self._get_baseline_for_zone(zone_name)
        direction = activations - baseline if baseline is not None else activations
        zone.emergent.update(direction, surprise, effective_alpha)

        # 3. If a crystal was selected, record its selection
        if not is_emergent:
            crystal = self._get_crystal_by_name(zone_name, selected_name)
            if crystal:
                crystal.record_selection(surprise)

        # 4. Update staleness for all crystals
        for crystal in zone.crystals:
            if not crystal.retired:
                crystal.update_staleness(self.crystal_config.staleness_max_cycles)

        # 5. Check crystallization (approval boosts likelihood)
        crystallized = self._maybe_crystallize(zone_name, approval_bonus)
        if crystallized:
            events["crystallized"] = crystallized
            logger.info(
                f"Crystallized {crystallized.name} in {zone_name} "
                f"(surprise_ema={zone.emergent.surprise_ema:.1f})"
            )

        # 6. Check spawning
        spawned = self._maybe_spawn(zone_name)
        if spawned:
            events["spawned"] = spawned
            logger.info(
                f"Spawned {spawned.name} in {zone_name} from "
                f"{spawned.parent_names}"
            )

        # 7. Check pruning (if at population limit, approval protects crystals)
        pruned = self._maybe_prune(zone_name, approval_bonus)
        if pruned:
            events["pruned"] = pruned
            logger.info(f"Pruned {pruned} from {zone_name}")

        return events

    def _select_vector(
        self,
        zone_name: str,
        context_embedding: Optional[np.ndarray] = None,
        approval_bonus: float = 0.0,
        sae_features: Optional[List[Tuple[int, float]]] = None,
        current_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, bool]:
        """Select between emergent and crystals based on scores.

        When qd_scorer is configured, uses Quality Diversity metrics for
        selection. Otherwise falls back to affinity×freshness scoring.

        When approval_bonus is positive, the emergent slot gets a boost,
        encouraging exploration of approved-feature directions. When negative,
        crystals are favored to avoid disapproved patterns.

        Args:
            zone_name: Zone to select from.
            context_embedding: Optional context for affinity/coherence calculation.
            approval_bonus: Recognition signal bonus [-1, 1]. Positive = features approved,
                negative = features disapproved.
            sae_features: Optional SAE features for QD presence scoring.
            current_embedding: Optional embedding of current thought for latent coherence.

        Returns:
            Tuple of (selected_name, is_emergent).
        """
        # Use QD scoring if scorer is configured
        if self.qd_scorer is not None:
            return self._select_vector_with_qd(
                zone_name, context_embedding, approval_bonus, sae_features, current_embedding
            )

        zone = self.zones.get(zone_name)
        if zone is None:
            return ("emergent", True)

        cfg = self.crystal_config

        # Score emergent: surprise_ema normalized, with bonus
        emergent_score = (zone.emergent.surprise_ema / 50.0) * cfg.emergent_bonus

        # Apply approval influence to emergent score
        # Positive approval → boost emergent (explore approved directions)
        # Negative approval → suppress emergent (fall back to proven crystals)
        if approval_bonus != 0.0:
            approval_factor = 1.0 + (approval_bonus * cfg.approval_bonus_weight)
            emergent_score *= approval_factor

        # Score crystals
        best_crystal: Optional[CrystalEntry] = None
        best_crystal_score = 0.0

        for crystal in zone.crystals:
            if crystal.retired:
                continue

            # Affinity: cosine similarity to context (or avg_surprise if no context)
            if context_embedding is not None:
                affinity = cosine_similarity(crystal.vector, context_embedding)
                affinity = max(0.0, affinity)  # Clamp negative
            else:
                # Fallback: use normalized avg_surprise as proxy
                affinity = crystal.avg_surprise / 50.0

            # Staleness penalty
            staleness_penalty = crystal.staleness * cfg.staleness_penalty_scale

            # Freshness bonus (inverse of staleness)
            freshness = 1.0 + (1.0 - crystal.staleness) * (cfg.freshness_bonus_max - 1.0)

            # Combined score
            score = affinity * (1.0 - staleness_penalty) * freshness

            # Negative approval → boost crystals (retreat from disapproved emergent)
            if approval_bonus < 0:
                score *= 1.0 + (abs(approval_bonus) * cfg.approval_bonus_weight)

            if score > best_crystal_score:
                best_crystal_score = score
                best_crystal = crystal

        # Selection: emergent wins if higher score
        if emergent_score >= best_crystal_score or best_crystal is None:
            return ("emergent", True)
        else:
            return (best_crystal.name, False)

    def _select_vector_with_qd(
        self,
        zone_name: str,
        context_embedding: Optional[np.ndarray] = None,
        approval_bonus: float = 0.0,
        sae_features: Optional[List[Tuple[int, float]]] = None,
        current_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[str, bool]:
        """Select between emergent and crystals using QD scoring.

        Uses Quality Diversity metrics (coherence, novelty, surprise, presence,
        latent_coherence) for selection instead of affinity×freshness.

        Args:
            zone_name: Zone to select from.
            context_embedding: Optional context for coherence calculation.
            approval_bonus: Recognition signal bonus (passed to context).
            sae_features: Optional SAE features for presence scoring.
            current_embedding: Optional embedding of current thought for latent coherence.

        Returns:
            Tuple of (selected_name, is_emergent).
        """
        from core.steering.qd.scorer import QDContext

        zone = self.zones.get(zone_name)
        if zone is None:
            return ("emergent", True)

        # Build QD context
        context = QDContext(
            zone_name=zone_name,
            current_cycle=self.global_cycle,
            context_embedding=context_embedding,
            current_embedding=current_embedding,
            sae_features=sae_features,
            approval_bonus=approval_bonus,
        )

        # Score all non-retired crystals
        candidates: List[Tuple[str, float, bool, Optional[np.ndarray]]] = []

        for crystal in zone.crystals:
            if crystal.retired:
                continue

            score = self.qd_scorer.score(crystal, context)
            if score.passed_floor:
                candidates.append((crystal.name, score.total, False, crystal.vector))

        # Score emergent slot
        if zone.emergent.vector is not None and np.linalg.norm(zone.emergent.vector) > 0:
            emergent_score = self.qd_scorer.score_emergent(zone.emergent, context)
            if emergent_score.passed_floor:
                candidates.append(("emergent", emergent_score.total, True, zone.emergent.vector))

        if not candidates:
            # No candidates passed floor - fall back to emergent
            return ("emergent", True)

        # Select best by total QD score
        best = max(candidates, key=lambda x: x[1])
        selected_name, _, is_emergent, selected_vector = best

        # Record selection for novelty tracking
        if selected_vector is not None:
            self.qd_scorer.record_selection(selected_vector)

        return (selected_name, is_emergent)

    def _maybe_crystallize(
        self, zone_name: str, approval_bonus: float = 0.0
    ) -> Optional[CrystalEntry]:
        """Check if emergent should crystallize into population.

        High approval lowers the thresholds, making crystallization easier
        when Ryan is approving the current direction. This captures "good"
        directions earlier.

        Args:
            zone_name: Zone to check.
            approval_bonus: Recognition signal bonus [-1, 1].

        Returns:
            New CrystalEntry if crystallized, None otherwise.
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        cfg = self.crystal_config
        em = zone.emergent

        # Apply approval boost to thresholds
        # High approval → lower thresholds (easier to crystallize)
        # This is the "recognition signal → preserve direction" pathway
        threshold_multiplier = 1.0
        if approval_bonus > 0:
            # Reduce thresholds by up to approval_crystallize_boost (e.g., 0.8 = 20% lower)
            threshold_multiplier = 1.0 - (approval_bonus * (1.0 - cfg.approval_crystallize_boost))

        # Check thresholds (adjusted by approval)
        min_cycles = int(cfg.min_cycles_for_crystallize * threshold_multiplier)
        min_surprise_ema = cfg.min_surprise_ema * threshold_multiplier
        min_cumulative = cfg.min_cumulative_surprise * threshold_multiplier

        # Log crystallization check for threshold calibration (INFO level for empirical tuning)
        logger.info(
            f"[CRYSTALLIZE] Check [{zone_name}]: "
            f"cycles={em.cycles_since_crystallize}/{min_cycles}, "
            f"surprise_ema={em.surprise_ema:.3f} (threshold={min_surprise_ema:.3f}), "
            f"cumulative={em.cumulative_surprise:.3f} (threshold={min_cumulative:.3f})"
        )

        if em.cycles_since_crystallize < min_cycles:
            logger.info(f"[CRYSTALLIZE] [{zone_name}] BLOCKED: need {min_cycles - em.cycles_since_crystallize} more cycles")
            return None
        if em.surprise_ema < min_surprise_ema:
            logger.info(f"[CRYSTALLIZE] [{zone_name}] BLOCKED: surprise_ema {em.surprise_ema:.3f} < {min_surprise_ema:.3f}")
            return None
        if em.cumulative_surprise < min_cumulative:
            logger.info(f"[CRYSTALLIZE] [{zone_name}] BLOCKED: cumulative {em.cumulative_surprise:.3f} < {min_cumulative:.3f}")
            return None

        # Use peak_vector if available, otherwise current
        vector = em.peak_vector if em.peak_vector is not None else em.vector.copy()

        # Create crystal
        crystal = CrystalEntry(
            name=generate_crystal_name(zone_name, self.global_cycle),
            vector=vector.copy(),
            birth_cycle=self.global_cycle,
            birth_surprise=em.peak_surprise,
        )

        zone.crystals.append(crystal)

        # Log approval influence if significant
        if approval_bonus > self.crystal_config.approval_logging_threshold:
            logger.info(
                f"Approval-boosted crystallization: {crystal.name} "
                f"(approval={approval_bonus:.2f}, threshold_mult={threshold_multiplier:.2f})"
            )

        # Reset emergent for fresh emergence
        em.reset_for_new_emergence(self.d_model)

        return crystal

    def _maybe_spawn(self, zone_name: str) -> Optional[CrystalEntry]:
        """Check if high-affinity crystal pair should spawn child.

        Args:
            zone_name: Zone to check.

        Returns:
            New CrystalEntry if spawned, None otherwise.
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        cfg = self.crystal_config

        # Need at least 2 crystals for spawning
        eligible = [
            c for c in zone.crystals
            if not c.retired and c.can_spawn(cfg, self.global_cycle)
        ]
        if len(eligible) < 2:
            return None

        # Check population limit
        active_count = sum(1 for c in zone.crystals if not c.retired)
        if active_count >= cfg.max_crystals_per_zone:
            return None

        # Find best parent pair by combined avg_surprise
        best_pair: Optional[Tuple[CrystalEntry, CrystalEntry]] = None
        best_combined = 0.0

        for i, c1 in enumerate(eligible):
            for c2 in eligible[i + 1:]:
                # Affinity between parents
                affinity = cosine_similarity(c1.vector, c2.vector)
                # Combined score: both must be good performers
                combined = (c1.avg_surprise + c2.avg_surprise) / 100.0 * (1.0 + affinity)

                if combined > best_combined:
                    best_combined = combined
                    best_pair = (c1, c2)

        if best_pair is None or best_combined < cfg.spawn_affinity_threshold:
            return None

        p1, p2 = best_pair

        # Blend vectors with mutation
        child_vector = blend_vectors_with_mutation(
            p1.vector, p2.vector, cfg.spawn_mutation_scale
        )

        # Create child crystal
        child = CrystalEntry(
            name=generate_crystal_name(zone_name, self.global_cycle),
            vector=child_vector,
            parent_names=[p1.name, p2.name],
            birth_cycle=self.global_cycle,
            birth_surprise=(p1.avg_surprise + p2.avg_surprise) / 2.0,
        )

        # Update parent spawn tracking
        p1.last_spawn_cycle = self.global_cycle
        p1.children_spawned += 1
        p2.last_spawn_cycle = self.global_cycle
        p2.children_spawned += 1

        zone.crystals.append(child)
        return child

    def _maybe_prune(
        self, zone_name: str, approval_bonus: float = 0.0
    ) -> Optional[str]:
        """Prune weakest crystal if at population limit.

        High approval protects crystals from pruning - we don't want to
        discard directions that Ryan is currently approving of. This creates
        a "recognition shield" effect.

        Args:
            zone_name: Zone to check.
            approval_bonus: Recognition signal bonus [-1, 1]. Positive protects
                crystals, negative makes pruning more aggressive.

        Returns:
            Name of pruned crystal, or None.
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        cfg = self.crystal_config

        active = [c for c in zone.crystals if not c.retired]
        if len(active) <= cfg.max_crystals_per_zone:
            return None

        # High approval → skip pruning entirely (protect current directions)
        # This is the "recognition shield" - don't prune when user is happy
        if approval_bonus > cfg.approval_prune_protection:
            logger.debug(
                f"Pruning skipped due to high approval ({approval_bonus:.2f} > "
                f"{cfg.approval_prune_protection:.2f})"
            )
            return None

        # Sort by performance score (avg_surprise × (1 - staleness_penalty))
        def perf_score(c: CrystalEntry) -> float:
            staleness_penalty = c.staleness * cfg.staleness_penalty_scale
            return c.avg_surprise * (1.0 - staleness_penalty)

        sorted_crystals = sorted(active, key=perf_score, reverse=True)

        # Preserve top performers
        # Negative approval → less protection (prune more aggressively)
        protect_count = cfg.preserve_count
        if approval_bonus < cfg.approval_prune_aggression_threshold:
            protect_count = max(0, protect_count - 1)  # Reduce protection

        protected = set(c.name for c in sorted_crystals[:protect_count])

        # Find weakest that meets prune criteria
        for crystal in reversed(sorted_crystals):
            if crystal.name in protected:
                continue
            if crystal.selection_count < cfg.prune_min_selections:
                continue

            # SDFT: High-selection crystals resist pruning
            # Well-used crystals have proven their value through repeated selection
            selection_protection = min(1.0, crystal.selection_count / 50)
            if selection_protection > 0.5:
                logger.debug(
                    f"SDFT protection: {crystal.name} resists pruning "
                    f"(selection_count={crystal.selection_count}, protection={selection_protection:.2f})"
                )
                continue  # Skip pruning for well-used crystals

            if crystal.avg_surprise < cfg.prune_surprise_threshold:
                crystal.retired = True
                return crystal.name

        # If no crystal meets criteria, prune absolute weakest (except protected)
        for crystal in reversed(sorted_crystals):
            if crystal.name not in protected:
                crystal.retired = True
                return crystal.name

        return None

    def _get_crystal_by_name(
        self, zone_name: str, name: Optional[str]
    ) -> Optional[CrystalEntry]:
        """Get crystal by name from zone."""
        if name is None:
            return None
        zone = self.zones.get(zone_name)
        if zone is None:
            return None
        for crystal in zone.crystals:
            if crystal.name == name and not crystal.retired:
                return crystal
        return None

    def _get_zone_config(self, zone_name: str) -> Optional[SteeringZone]:
        """Get zone configuration by name."""
        for zone in self.config.zones:
            if zone.name == zone_name:
                return zone
        return None

    def get_effective_ema_alpha(self, zone_name: str, phase: str = "generation") -> float:
        """Get EMA alpha modulated by current cognitive phase.

        Different cognitive phases require different learning rates:
        - generation: Full speed exploration
        - curation: Slightly slower for analysis
        - simulation: Slower for rigorous hypothesis testing
        - integration: Slower for memory consolidation
        - reflexion: Much slower for meta-analysis
        - continuity: Much slower for synthesis

        This implements the "temporal hierarchy" concept from neuroscience,
        where different brain regions process at different timescales,
        and consolidation phases naturally involve slower dynamics.

        Args:
            zone_name: Name of the steering zone.
            phase: Current cognitive phase (generation, curation, simulation,
                   integration, reflexion, continuity).

        Returns:
            Effective EMA alpha (base_alpha * phase_modulation_factor).
            Returns 0.1 as default if zone not found.
        """
        zone = self._get_zone_config(zone_name)
        if zone is None:
            return 0.1  # Default fallback

        base_alpha = zone.ema_alpha
        modulation = self.phase_modulation_factors.get(phase, 1.0)
        return base_alpha * modulation

    @property
    def phase_modulation_factors(self) -> dict[str, float]:
        """Phase-based EMA modulation factors.

        Returns the module-level PHASE_MODULATION_FACTORS dict.
        This is a property to allow potential future customization
        while maintaining backward compatibility.
        """
        return PHASE_MODULATION_FACTORS

    def _get_baseline_for_zone(self, zone_name: str) -> Optional[np.ndarray]:
        """Get baseline vector for a zone (average of crystal vectors)."""
        zone = self.zones.get(zone_name)
        if zone is None:
            return None

        active_crystals = [c for c in zone.crystals if not c.retired]
        if not active_crystals:
            return None

        return np.mean([c.vector for c in active_crystals], axis=0)

    def get_zone_summary(self, zone_name: str) -> dict:
        """Get summary statistics for a zone.

        Args:
            zone_name: Zone to summarize.

        Returns:
            Dict with zone statistics.
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return {}

        active_crystals = [c for c in zone.crystals if not c.retired]
        retired_crystals = [c for c in zone.crystals if c.retired]

        return {
            "emergent_surprise_ema": zone.emergent.surprise_ema,
            "emergent_cumulative_surprise": zone.emergent.cumulative_surprise,
            "emergent_cycles": zone.emergent.cycles_since_crystallize,
            "crystal_count": len(active_crystals),
            "retired_count": len(retired_crystals),
            "current_selection": zone.current_selection,
            "current_is_emergent": zone.current_is_emergent,
            "crystals": [
                {
                    "name": c.name,
                    "avg_surprise": c.avg_surprise,
                    "selection_count": c.selection_count,
                    "staleness": c.staleness,
                    "children": c.children_spawned,
                }
                for c in active_crystals
            ],
        }

    def get_all_crystals(self) -> List[Tuple[str, CrystalEntry]]:
        """Get all active crystals across zones for persistence.

        Returns:
            List of (zone_name, crystal) tuples.
        """
        results = []
        for zone_name, zone in self.zones.items():
            for crystal in zone.crystals:
                if not crystal.retired:
                    results.append((zone_name, crystal))
        return results

    def load_crystal(self, zone_name: str, crystal: CrystalEntry) -> bool:
        """Load a crystal into a zone (for persistence restore).

        Args:
            zone_name: Zone to load into.
            crystal: Crystal to load.

        Returns:
            True if loaded successfully.
        """
        zone = self.zones.get(zone_name)
        if zone is None:
            return False

        # Check if already exists
        for existing in zone.crystals:
            if existing.name == crystal.name:
                return False

        zone.crystals.append(crystal)
        return True
