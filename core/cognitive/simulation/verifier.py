"""Prediction Verifier for simulation phase.

Checks pending predictions against current cognitive state and updates
hypothesis statistics based on verification outcomes. Also implements
the feedback loop for outcome-based steering, updating vector effectiveness
when predictions are verified or falsified.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from core.cognitive.simulation.schemas import (
    FailureReason,
    Hypothesis,
    HypothesisStatus,
    MetricsSnapshot,
    Prediction,
    PredictionConditionType,
    PredictionStatus,
)
from core.cognitive.simulation.lifecycle import (
    evaluate_lifecycle_transition,
    update_calibration,
)
from core.psyche.schema import PredictionPattern

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.cognitive.halt_collector import HALTTrainingCollector
    from core.cognitive.state import CognitiveState
    from core.embedding.service import TieredEmbeddingService
    from core.psyche.client import PsycheClient
    from core.self_model.goal_registry import GoalRegistry


class PredictionVerifier:
    """Verifies predictions against cognitive state.

    Checks pending predictions to see if their conditions have been met,
    scores their accuracy, and updates hypothesis statistics.

    Attributes:
        psyche: PsycheClient for reading/updating predictions
    """

    # Similarity threshold for concept matching
    CONCEPT_SIMILARITY_THRESHOLD = 0.7

    # Similarity threshold for claim content verification
    # Lower than concept matching since claims are more complex/verbose
    CLAIM_CONTENT_SIMILARITY_THRESHOLD = 0.6

    # Confidence thresholds for follow-up simulation triggers
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.4
    ACCURACY_VERIFIED_THRESHOLD = 0.7
    ACCURACY_FALSIFIED_THRESHOLD = 0.3

    # Overconfidence classification thresholds
    OVERCONFIDENT_THRESHOLD = 0.7  # Confidence above which falsification = overconfident
    OVERCONFIDENT_ACCURACY_THRESHOLD = 0.3  # Accuracy below which = clear falsification

    # Path reward scoring constants (used in _compute_path_reward)
    PATH_LENGTH_PENALTY = 0.25  # Score reduction per additional hop (1 hop=1.0, 2 hops=0.75, etc.)
    COMPOSITIONAL_BRIDGE_BONUS = 1.2  # 20% boost for short paths (1-3 hops) for compositionality

    # Pattern learning thresholds
    LOW_SUCCESS_RATE_WARNING = 0.35  # Warn when pattern success rate drops below this

    # Verification-rate-based retirement thresholds
    # If verification_rate < 0.15 after 5+ predictions, abandon hypothesis
    # Uses lineage counts to track across refined hypotheses
    VERIFICATION_RATE_ABANDON_THRESHOLD = 0.15
    MIN_PREDICTIONS_FOR_ABANDON = 5

    # Follow-up simulation cooldown (prevent same hypothesis from triggering consecutive follow-ups)
    # Increased from 3 to 10 to reduce repetitive simulation loops
    FOLLOW_UP_COOLDOWN_CYCLES = 10

    # Key metrics to include in TIME_BASED outcome descriptions
    KEY_METRICS_FOR_OUTCOME = [
        "semantic_entropy",
        "structural_entropy",
        "hub_concentration",
        "discovery_parameter",
    ]

    # Stopwords for keyword extraction (defined once at class level for efficiency)
    STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "our", "you", "your", "i", "my", "me", "to",
        "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "and", "or", "but", "if", "then", "than", "so", "what",
        "when", "where", "who", "which", "how", "why", "all", "each",
        "more", "most", "other", "some", "such", "no", "not", "only",
    })

    def __init__(
        self,
        psyche: "PsycheClient",
        goal_registry: "GoalRegistry | None" = None,
        embedder: "object | None" = None,
        similarity_threshold: float = 0.7,
        halt_collector: "HALTTrainingCollector | None" = None,
        embedding_service: "TieredEmbeddingService | None" = None,
    ) -> None:
        """Initialize the prediction verifier.

        Args:
            psyche: PsycheClient for persistence operations
            goal_registry: GoalRegistry for measuring goal alignment
            embedder: Optional embedder for semantic similarity matching.
                Must have an `encode(text)` method returning a numpy array.
            similarity_threshold: Cosine similarity threshold for concept matching.
                Defaults to 0.7.
            halt_collector: Optional HALTTrainingCollector for recording training
                examples when predictions are verified or falsified. This provides
                the strongest signal for HALT probe training.
            embedding_service: Optional TieredEmbeddingService for generating
                skill embeddings when hypotheses are verified.
        """
        self._psyche = psyche
        self._goal_registry = goal_registry
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._halt_collector = halt_collector
        self._embedding_service = embedding_service

    def _classify_failure(
        self,
        pred: Prediction,
        accuracy: float,
    ) -> FailureReason:
        """Classify why a prediction was falsified.

        Args:
            pred: The falsified prediction
            accuracy: Accuracy score at falsification

        Returns:
            FailureReason categorizing the failure mode
        """
        # Expired = condition never triggered
        if pred.status == PredictionStatus.EXPIRED:
            return FailureReason.EXPIRED

        # High confidence falsified = overconfident
        if (
            pred.confidence > self.OVERCONFIDENT_THRESHOLD
            and accuracy < self.OVERCONFIDENT_ACCURACY_THRESHOLD
        ):
            return FailureReason.OVERCONFIDENT

        # Everything else = wrong direction/claim
        return FailureReason.WRONG_DIRECTION

    def _check_concept_mentioned(
        self,
        target: str,
        text: str,
    ) -> bool:
        """Check if target concept is mentioned using semantic similarity.

        Uses cosine similarity between embeddings if an embedder is available,
        otherwise falls back to substring matching.

        Args:
            target: The target concept to look for (lowercased)
            text: The text to search in (lowercased)

        Returns:
            True if concept is semantically similar or substring match found
        """
        if not text:
            return False

        # If no embedder, fall back to substring matching
        if self._embedder is None:
            return target in text

        # Use semantic similarity
        try:
            import numpy as np

            # Get embeddings for target and text
            target_emb = self._embedder.encode(target)
            text_emb = self._embedder.encode(text)

            # Ensure we have numpy arrays
            if not isinstance(target_emb, np.ndarray):
                target_emb = np.array(target_emb)
            if not isinstance(text_emb, np.ndarray):
                text_emb = np.array(text_emb)

            # Compute cosine similarity
            norm_target = np.linalg.norm(target_emb)
            norm_text = np.linalg.norm(text_emb)

            if norm_target == 0 or norm_text == 0:
                # Fallback on zero vectors
                return target in text

            similarity = np.dot(target_emb, text_emb) / (norm_target * norm_text)

            return similarity >= self._similarity_threshold

        except Exception as e:
            # Fallback to substring matching on any error
            logger.debug(f"Embedding failed for concept check, using substring: {e}")
            return target in text

    def _compute_claim_similarity(
        self,
        claim: str,
        text: str,
    ) -> float | None:
        """Compute semantic similarity between claim content and text.

        Used to verify that the claim's content is actually supported by the
        thought/insight text, not just that a concept was mentioned.

        Args:
            claim: The prediction's claim text
            text: The text to compare against (thought or insight)

        Returns:
            Cosine similarity score (0.0-1.0) if embedder available, None otherwise
        """
        if not claim or not text:
            return None

        if self._embedder is None:
            return None

        try:
            import numpy as np

            # Get embeddings for claim and text
            claim_emb = self._embedder.encode(claim)
            text_emb = self._embedder.encode(text)

            # Ensure we have numpy arrays
            if not isinstance(claim_emb, np.ndarray):
                claim_emb = np.array(claim_emb)
            if not isinstance(text_emb, np.ndarray):
                text_emb = np.array(text_emb)

            # Compute cosine similarity
            norm_claim = np.linalg.norm(claim_emb)
            norm_text = np.linalg.norm(text_emb)

            if norm_claim == 0 or norm_text == 0:
                return None

            similarity = float(np.dot(claim_emb, text_emb) / (norm_claim * norm_text))
            return similarity

        except Exception as e:
            logger.debug(f"Embedding failed for claim similarity: {e}")
            return None

    async def check_pending_predictions(
        self,
        state: "CognitiveState",
    ) -> tuple[list[tuple[Prediction, bool]], list[str]]:
        """Check pending predictions against current state.

        Args:
            state: Current cognitive state

        Returns:
            Tuple of:
                - List of (prediction, verified) tuples for resolved predictions
                - List of hypothesis UIDs needing follow-up simulation
        """
        pending = await self._psyche.get_pending_predictions(state.cycle_count)
        results = []
        follow_up_uids: list[str] = []

        for pred in pending:
            # Check if expired
            if pred.expiry_cycle and state.cycle_count > pred.expiry_cycle:
                pred.status = PredictionStatus.EXPIRED
                await self._psyche.update_prediction(pred)
                logger.info(f"Prediction {pred.uid} expired at cycle {state.cycle_count}")
                continue

            # Check condition
            met, outcome = self._check_condition(pred, state)
            if not met:
                continue

            # Score the prediction based on condition type
            # METRIC_THRESHOLD: condition satisfaction = verified (no semantic overlap needed)
            # Other types: use semantic overlap heuristics via _score_prediction_async
            if pred.condition_type == PredictionConditionType.METRIC_THRESHOLD:
                # Metric conditions are binary: if met, accuracy = 1.0
                accuracy = 1.0
                logger.debug(
                    f"Prediction {pred.uid} verified via metric condition: {outcome[:100]}"
                )
            else:
                # Score using semantic overlap and path-derived rewards
                accuracy = await self._score_prediction_async(pred.claim, outcome, state)
            pred.outcome = outcome
            pred.accuracy_score = accuracy
            pred.verification_cycle = state.cycle_count

            # Measure goal delta if prediction targets a goal
            if pred.target_goal_uid and self._goal_registry:
                current_alignment = self._goal_registry.get_goal_alignment(
                    content=state.thought,
                    goal_uid=pred.target_goal_uid,
                )
                pred.goal_snapshot_after = current_alignment
                if pred.goal_snapshot_before is not None:
                    pred.actual_goal_delta = current_alignment - pred.goal_snapshot_before
                else:
                    logger.warning(
                        f"Prediction {pred.uid} has target_goal_uid but missing goal_snapshot_before"
                    )

            # Goal-delta based verification: override accuracy if goal delta is available
            # Positive delta = verified, negative delta = falsified, zero = falsified (no improvement)
            if (
                pred.target_goal_uid
                and pred.actual_goal_delta is not None  # Zero is valid!
            ):
                if pred.expected_goal_delta > 0:
                    if pred.actual_goal_delta > 0:
                        accuracy = 1.0
                        logger.debug(
                            f"Prediction {pred.uid} verified via positive goal delta "
                            f"({pred.actual_goal_delta:.3f})"
                        )
                    elif pred.actual_goal_delta < 0:
                        accuracy = 0.0
                        logger.debug(
                            f"Prediction {pred.uid} falsified via negative goal delta "
                            f"({pred.actual_goal_delta:.3f})"
                        )
                    else:  # actual_goal_delta == 0
                        accuracy = 0.3
                        logger.debug(
                            f"Prediction {pred.uid} falsified via zero goal delta "
                            f"(no improvement, expected positive)"
                        )
                elif pred.expected_goal_delta < 0:
                    # Expected negative delta (predicted decline)
                    if pred.actual_goal_delta < 0:
                        accuracy = 1.0
                        logger.debug(
                            f"Prediction {pred.uid} verified via negative goal delta "
                            f"({pred.actual_goal_delta:.3f}, expected decline)"
                        )
                    elif pred.actual_goal_delta > 0:
                        accuracy = 0.0
                        logger.debug(
                            f"Prediction {pred.uid} falsified via positive goal delta "
                            f"({pred.actual_goal_delta:.3f}, expected decline)"
                        )
                    else:  # actual_goal_delta == 0
                        accuracy = 0.3
                        logger.debug(
                            f"Prediction {pred.uid} falsified via zero goal delta "
                            f"(no change, expected decline)"
                        )
                else:  # expected_goal_delta == 0 (predicted no change)
                    if pred.actual_goal_delta == 0:
                        accuracy = 1.0
                        logger.debug(
                            f"Prediction {pred.uid} verified via zero goal delta "
                            f"(no change as expected)"
                        )
                    else:
                        accuracy = 0.0
                        logger.debug(
                            f"Prediction {pred.uid} falsified via non-zero goal delta "
                            f"({pred.actual_goal_delta:.3f}, expected zero)"
                        )
                # Update accuracy_score to reflect goal-delta decision
                pred.accuracy_score = accuracy

            if accuracy >= self.ACCURACY_VERIFIED_THRESHOLD:
                pred.status = PredictionStatus.VERIFIED
                verified = True
            elif accuracy <= self.ACCURACY_FALSIFIED_THRESHOLD:
                pred.status = PredictionStatus.FALSIFIED
                pred.failure_reason = self._classify_failure(pred, accuracy)
                verified = False
            else:
                # Inconclusive - keep as pending for future re-evaluation
                logger.debug(
                    f"Prediction {pred.uid} inconclusive (accuracy={accuracy:.2f}), "
                    f"keeping as pending"
                )
                continue

            # Update prediction in graph
            await self._psyche.update_prediction(pred)

            # Update prediction pattern for learning
            pattern = await self._psyche.get_prediction_pattern(
                pred.condition_type.value
            )
            if pattern is None:
                pattern = PredictionPattern(condition_type=pred.condition_type.value)

            pattern.record_outcome(
                success=verified,
                failure_reason=pred.failure_reason.value if pred.failure_reason else None,
            )
            await self._psyche.save_prediction_pattern(pattern)

            # Log warning for systematic failures
            if pattern.is_reliable and pattern.success_rate < self.LOW_SUCCESS_RATE_WARNING:
                logger.warning(
                    f"Low success rate for {pattern.condition_type}: "
                    f"{pattern.success_rate:.0%} (N={pattern.total})"
                )

            # Update hypothesis stats
            verified_delta = 1 if verified else 0
            falsified_delta = 0 if verified else 1
            await self._psyche.update_hypothesis_stats(
                pred.hypothesis_uid, verified_delta, falsified_delta
            )

            # Update hypothesis lifecycle and calibration
            hypothesis = await self._psyche.get_hypothesis(pred.hypothesis_uid)
            if hypothesis:
                # Update calibration (modifies hypothesis in place)
                update_calibration(hypothesis, verified)

                # Check for lifecycle transition
                new_status = evaluate_lifecycle_transition(
                    hypothesis, state.cycle_count
                )
                if new_status != hypothesis.status:
                    hypothesis.status = new_status
                    await self._emit_lifecycle_event(hypothesis, new_status)

                    # Generate skill from verified hypothesis (UPSKILL pattern)
                    if new_status == HypothesisStatus.VERIFIED:
                        await self._generate_skill_if_worthy(hypothesis)

                # Update last_evaluated_cycle
                hypothesis.last_evaluated_cycle = state.cycle_count

                # Persist updated hypothesis
                await self._psyche.update_hypothesis(hypothesis)

            # Update steering vector effectiveness (feedback loop)
            await self._update_vector_effectiveness(pred.hypothesis_uid, verified)

            logger.info(
                f"Prediction {pred.uid} {'verified' if verified else 'falsified'} "
                f"with accuracy {accuracy:.2f}"
            )

            results.append((pred, verified))

            # Record HALT training example (strongest signal for epistemic probe)
            # Prediction verification is ground truth: verified=1.0, falsified=0.0
            if self._halt_collector:
                label = 1.0 if verified else 0.0
                source = "prediction_verified" if verified else "prediction_failed"
                cycle_id = f"cycle_{state.cycle_count}"

                await self._halt_collector.record_example(
                    cycle_id=cycle_id,
                    label=label,
                    label_source=source,
                    thought_preview=state.thought[:200] if state.thought else "",
                    insight_preview=pred.claim[:200] if pred.claim else "",
                    prediction_id=pred.uid,
                )
                logger.debug(
                    f"Recorded HALT training example for prediction {pred.uid} "
                    f"(label={label}, source={source})"
                )

            # Check if follow-up simulation is warranted
            # Skip if this prediction already triggered a follow-up before
            if pred.triggered_follow_up:
                logger.debug(
                    f"Prediction {pred.uid} already triggered follow-up, skipping"
                )
            elif self._should_trigger_follow_up(pred, accuracy):
                if pred.hypothesis_uid not in follow_up_uids:
                    # Check cooldown before adding to follow-up list
                    hypothesis = await self._psyche.get_hypothesis(pred.hypothesis_uid)
                    if hypothesis and await self._is_in_follow_up_cooldown(
                        hypothesis, state.cycle_count
                    ):
                        logger.debug(
                            f"Skipping follow-up for hypothesis {pred.hypothesis_uid}: "
                            f"in cooldown (last follow-up cycle: {hypothesis.last_follow_up_cycle})"
                        )
                    else:
                        follow_up_uids.append(pred.hypothesis_uid)
                        # Mark this prediction as having triggered follow-up
                        pred.triggered_follow_up = True
                        await self._psyche.update_prediction(pred)
                        logger.debug(
                            f"Prediction {pred.uid} triggered follow-up for hypothesis {pred.hypothesis_uid}"
                        )

        # Update last_follow_up_cycle for hypotheses that will get follow-up simulation
        for hyp_uid in follow_up_uids:
            await self._psyche.update_hypothesis_last_follow_up_cycle(
                hyp_uid, state.cycle_count
            )
            logger.debug(
                f"Updated last_follow_up_cycle for hypothesis {hyp_uid} to {state.cycle_count}"
            )

        return results, follow_up_uids

    def _check_condition(
        self,
        pred: Prediction,
        state: "CognitiveState",
    ) -> tuple[bool, str]:
        """Check if prediction's condition is met.

        Args:
            pred: Prediction to check
            state: Current cognitive state

        Returns:
            Tuple of (condition_met, outcome_description)
        """
        if pred.condition_type == PredictionConditionType.TIME_BASED:
            # Check if enough cycles have passed
            try:
                cycles_needed = int(pred.condition_value) if pred.condition_value else 5
            except ValueError:
                cycles_needed = 5

            if pred.earliest_verify_cycle and state.cycle_count >= pred.earliest_verify_cycle:
                # Include current metric values for debugging
                outcome = f"After {cycles_needed} cycles"
                metrics = self._get_current_metrics_dict(state)
                if metrics:
                    # Include key metrics for debugging
                    key_metrics = []
                    for metric_name in self.KEY_METRICS_FOR_OUTCOME:
                        if metric_name in metrics:
                            value = metrics[metric_name]
                            if isinstance(value, (int, float)):
                                key_metrics.append(f"{metric_name}={value:.2f}")
                    if key_metrics:
                        outcome = f"{outcome}: {', '.join(key_metrics)}"
                return True, outcome
            return False, ""

        elif pred.condition_type == PredictionConditionType.CONCEPT_MENTIONED:
            # Check if target concept appears using semantic similarity
            # AND verify that the claim content semantically matches the text
            if not pred.condition_value:
                logger.warning(f"Prediction {pred.uid} has empty condition_value")
                return False, ""
            target = pred.condition_value.lower()
            thought_lower = state.thought.lower()
            insight_lower = state.current_insight.lower()

            # Track where concept was found and the claim similarity score
            concept_found_in: str | None = None
            claim_similarity: float | None = None

            # Check thought with semantic similarity (or substring fallback)
            if self._check_concept_mentioned(target, thought_lower):
                concept_found_in = "thought"
                # Verify claim content matches thought
                claim_similarity = self._compute_claim_similarity(pred.claim, state.thought)

            # Check insight with semantic similarity (or substring fallback)
            elif self._check_concept_mentioned(target, insight_lower):
                concept_found_in = "insight"
                # Verify claim content matches insight
                claim_similarity = self._compute_claim_similarity(pred.claim, state.current_insight)

            # Check in recent concepts (substring only for efficiency)
            else:
                for concept in state.recent_concepts:
                    if target in concept.lower():
                        concept_found_in = "recent_concepts"
                        # For recent_concepts, verify against thought (most relevant context)
                        claim_similarity = self._compute_claim_similarity(
                            pred.claim, state.thought
                        )
                        break

            # If concept not found anywhere, condition not met
            if concept_found_in is None:
                return False, ""

            # Concept found - now verify claim content
            # If embedder unavailable, fall back to traditional behavior (condition-only)
            if claim_similarity is None:
                # No embedder - fall back to condition-only verification
                logger.debug(
                    f"Prediction {pred.uid}: concept found in {concept_found_in}, "
                    f"claim similarity unavailable (no embedder), allowing condition-only"
                )
                return True, f"Concept '{pred.condition_value}' mentioned in {concept_found_in}"

            # Check if claim content semantically matches (above threshold)
            if claim_similarity >= self.CLAIM_CONTENT_SIMILARITY_THRESHOLD:
                return True, (
                    f"Concept '{pred.condition_value}' mentioned in {concept_found_in}; "
                    f"claim similarity: {claim_similarity:.3f}"
                )

            # Concept mentioned but claim content doesn't match
            # This means the concept was discussed but not in the context the claim predicted
            logger.debug(
                f"Prediction {pred.uid}: concept '{pred.condition_value}' found in "
                f"{concept_found_in}, but claim similarity ({claim_similarity:.3f}) "
                f"below threshold ({self.CLAIM_CONTENT_SIMILARITY_THRESHOLD})"
            )
            return False, ""

        elif pred.condition_type == PredictionConditionType.ENTITY_OBSERVED:
            # Check if entity appears in extraction
            if not pred.condition_value:
                logger.warning(f"Prediction {pred.uid} has empty condition_value")
                return False, ""
            target = pred.condition_value.lower()

            # Check active concepts
            for concept, _ in state.active_concepts:
                if target in concept.lower():
                    return True, f"Entity '{pred.condition_value}' observed in extraction"

            return False, ""

        elif pred.condition_type == PredictionConditionType.METRIC_THRESHOLD:
            # Evaluate metric expression against current and baseline metrics
            # Always evaluate at earliest_verify_cycle or later
            if pred.earliest_verify_cycle and state.cycle_count < pred.earliest_verify_cycle:
                return False, ""

            # Get baseline metrics from prediction
            baseline = pred.baseline_metrics or {}

            # Evaluate the expression
            met, value, explanation = self._evaluate_metric_expression(
                pred.condition_value,
                state,
                baseline,
            )

            if met:
                # Build detailed outcome with baseline vs current comparison
                outcome_parts = [f"Metric condition satisfied: {explanation}"]

                # Extract metric name from condition expression
                current_metrics = self._get_current_metrics_dict(state)
                if baseline and current_metrics:
                    # Find metrics that have both baseline and current values
                    comparisons = []
                    for metric_name in baseline:
                        if metric_name in current_metrics:
                            baseline_val = baseline[metric_name]
                            current_val = current_metrics[metric_name]
                            if isinstance(baseline_val, (int, float)) and isinstance(
                                current_val, (int, float)
                            ):
                                delta = current_val - baseline_val
                                sign = "+" if delta >= 0 else ""
                                comparisons.append(
                                    f"{metric_name}: {baseline_val:.2f} -> {current_val:.2f} ({sign}{delta:.2f})"
                                )
                    if comparisons:
                        outcome_parts.append("; ".join(comparisons))

                return True, " | ".join(outcome_parts)
            return False, ""

        elif pred.condition_type == PredictionConditionType.BELIEF_CHANGE:
            # Check if belief has changed as predicted
            return self._check_belief_change(pred, state)

        elif pred.condition_type == PredictionConditionType.SAE_FEATURE_PATTERN:
            # Check if SAE feature matches threshold condition
            return self._check_sae_feature_pattern(pred, state)

        elif pred.condition_type == PredictionConditionType.EMOTIONAL_SHIFT:
            # Check if emotional field has changed as predicted
            return self._check_emotional_shift(pred, state)

        elif pred.condition_type == PredictionConditionType.GOAL_PROGRESS:
            # Check if goal alignment has changed as predicted
            return self._check_goal_progress(pred, state)

        return False, ""

    def _check_sae_feature_pattern(
        self,
        pred: Prediction,
        state: "CognitiveState",
    ) -> tuple[bool, str]:
        """Check if SAE feature meets threshold condition.

        Condition value formats:
        - "feature_XXXX > 0.5" (above threshold)
        - "feature_XXXX >= 0.3" (at or above threshold)
        - "feature_XXXX < 0.2" (below threshold for decrease predictions)
        - "feature_XXXX <= 0.1" (at or below threshold)
        - "feature_XXXX == 0.5" (approximately equal, with 0.01 tolerance)

        Args:
            pred: Prediction with SAE_FEATURE_PATTERN condition
            state: Current cognitive state (must have sae_features attribute)

        Returns:
            Tuple of (condition_met, outcome_description)
        """
        condition_value = pred.condition_value or ""

        # Get SAE features from state
        sae_features = getattr(state, "sae_features", None)
        if sae_features is None:
            logger.debug(
                f"Prediction {pred.uid} SAE_FEATURE_PATTERN check: no sae_features in state"
            )
            return False, ""

        # Parse condition: "feature_XXXX operator threshold"
        match = re.match(r"(feature_\d+)\s*([<>=!]+)\s*([\d.]+)", condition_value)
        if not match:
            logger.warning(
                f"Prediction {pred.uid} has invalid SAE_FEATURE_PATTERN format: {condition_value}"
            )
            return False, ""

        feature_name = match.group(1)
        operator = match.group(2)
        threshold = float(match.group(3))

        # Get current feature value (default to 0.0 if not present)
        current_value = sae_features.get(feature_name, 0.0)

        # Compare using the common compare method
        if self._compare_values(current_value, operator, threshold):
            return True, f"SAE {feature_name} = {current_value:.4f} {operator} {threshold}"

        return False, ""

    def _check_emotional_shift(
        self,
        pred: Prediction,
        state: "CognitiveState",
    ) -> tuple[bool, str]:
        """Check if emotional field has changed as predicted.

        Condition value formats:
        - "valence_delta > 0.2" (mood improvement)
        - "arousal_delta < -0.1" (calming)
        - "valence > 0.5" (absolute valence threshold)
        - "arousal >= 0.6" (absolute arousal threshold)

        Note: This method uses sync helpers to handle both sync mocks
        in tests and async calls in production.

        Args:
            pred: Prediction with EMOTIONAL_SHIFT condition
            state: Current cognitive state (unused, but kept for consistency)

        Returns:
            Tuple of (condition_met, outcome_description)
        """
        condition_value = pred.condition_value or ""

        # Get baseline emotional state
        baseline_state = self._get_emotional_state_sync(at_cycle=pred.baseline_cycle)
        if baseline_state is None:
            logger.debug(
                f"Prediction {pred.uid} EMOTIONAL_SHIFT check: no baseline emotional state"
            )
            return False, ""

        # Get current emotional state
        current_state = self._get_emotional_state_sync()
        if current_state is None:
            logger.debug(
                f"Prediction {pred.uid} EMOTIONAL_SHIFT check: no current emotional state"
            )
            return False, ""

        # Parse condition: "metric_delta operator threshold" or "metric operator threshold"
        # Supports: valence_delta, arousal_delta, valence, arousal
        delta_match = re.match(r"(valence|arousal)_delta\s*([<>=!]+)\s*([-\d.]+)", condition_value)
        absolute_match = re.match(r"(valence|arousal)\s*([<>=!]+)\s*([-\d.]+)", condition_value)

        if delta_match:
            metric = delta_match.group(1)
            operator = delta_match.group(2)
            threshold = float(delta_match.group(3))

            baseline_val = baseline_state.get(metric, 0.0)
            current_val = current_state.get(metric, 0.0)
            delta = current_val - baseline_val

            if self._compare_values(delta, operator, threshold):
                return True, f"{metric}_delta = {delta:.4f} {operator} {threshold}"
            return False, ""

        elif absolute_match:
            metric = absolute_match.group(1)
            operator = absolute_match.group(2)
            threshold = float(absolute_match.group(3))

            current_val = current_state.get(metric, 0.0)

            if self._compare_values(current_val, operator, threshold):
                return True, f"{metric} = {current_val:.4f} {operator} {threshold}"
            return False, ""

        logger.warning(
            f"Prediction {pred.uid} has invalid EMOTIONAL_SHIFT format: {condition_value}"
        )
        return False, ""

    def _get_emotional_state_sync(
        self,
        at_cycle: int | None = None,
    ) -> dict | None:
        """Get emotional state, handling both sync mocks and async psyche clients.

        This helper bridges the sync _check_emotional_shift method with the
        potentially async get_emotional_state methods on PsycheClient.

        Args:
            at_cycle: If provided, get the emotional state as it was at that cycle

        Returns:
            The emotional state dict or None if not found
        """
        import asyncio
        import inspect

        # Choose the appropriate method based on whether we want historical or current state
        if at_cycle is not None:
            if not hasattr(self._psyche, "get_emotional_state_at_cycle"):
                logger.debug("PsycheClient does not have get_emotional_state_at_cycle method")
                return None
            method = self._psyche.get_emotional_state_at_cycle
            result = method(at_cycle)
        else:
            if not hasattr(self._psyche, "get_current_emotional_state"):
                logger.debug("PsycheClient does not have get_current_emotional_state method")
                return None
            method = self._psyche.get_current_emotional_state
            result = method()

        # If the result is a coroutine, we need to run it
        if inspect.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - can't use run_until_complete
                logger.warning(
                    "get_emotional_state returned coroutine in sync context"
                )
                result.close()  # Clean up the coroutine
                return None
            except RuntimeError:
                # No running loop - create one and run
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(result)
                finally:
                    loop.close()

        return result

    def _check_goal_progress(
        self,
        pred: Prediction,
        state: "CognitiveState",
    ) -> tuple[bool, str]:
        """Check if goal alignment has changed as predicted.

        Condition value formats:
        - "goal:goal_uid delta > 0.1" (improvement threshold)
        - "goal:goal_uid delta < -0.1" (decline threshold)
        - "goal:goal_uid alignment > 0.7" (absolute threshold)

        If pred.target_goal_uid is set, it takes precedence over parsing from
        condition_value. This allows simpler condition_value formats like:
        - "delta > 0.1"
        - "alignment > 0.7"

        Args:
            pred: Prediction with GOAL_PROGRESS condition
            state: Current cognitive state (unused, but kept for consistency)

        Returns:
            Tuple of (condition_met, outcome_description)
        """
        # Return False if no goal_registry is available
        if self._goal_registry is None:
            logger.debug(
                f"Prediction {pred.uid} GOAL_PROGRESS check: no goal_registry available"
            )
            return False, ""

        condition_value = pred.condition_value or ""

        # Determine goal_uid - prefer target_goal_uid if set
        goal_uid = pred.target_goal_uid

        if goal_uid is None:
            # Parse from condition_value: "goal:goal_uid <condition>"
            goal_match = re.match(r"goal:(\S+)\s+(.+)", condition_value)
            if goal_match:
                goal_uid = f"goal:{goal_match.group(1)}"
                condition_value = goal_match.group(2).strip()
            else:
                logger.warning(
                    f"Prediction {pred.uid} has invalid GOAL_PROGRESS format: {condition_value}"
                )
                return False, ""

        # Parse condition: "delta operator threshold" or "alignment operator threshold"
        delta_match = re.match(r"delta\s*([<>=!]+)\s*([-\d.]+)", condition_value)
        alignment_match = re.match(r"alignment\s*([<>=!]+)\s*([-\d.]+)", condition_value)

        if delta_match:
            operator = delta_match.group(1)
            threshold = float(delta_match.group(2))

            # Get baseline alignment
            baseline_alignment = self._goal_registry.get_alignment_at_cycle(
                goal_uid, pred.baseline_cycle
            )
            # Get current alignment
            current_alignment = self._goal_registry.get_current_alignment(goal_uid)

            delta = current_alignment - baseline_alignment

            if self._compare_values(delta, operator, threshold):
                return True, f"Goal {goal_uid} delta = {delta:.4f} {operator} {threshold}"
            return False, ""

        elif alignment_match:
            operator = alignment_match.group(1)
            threshold = float(alignment_match.group(2))

            # Get current alignment
            current_alignment = self._goal_registry.get_current_alignment(goal_uid)

            if self._compare_values(current_alignment, operator, threshold):
                return True, f"Goal {goal_uid} alignment = {current_alignment:.4f} {operator} {threshold}"
            return False, ""

        logger.warning(
            f"Prediction {pred.uid} has unrecognized GOAL_PROGRESS condition: {condition_value}"
        )
        return False, ""

    def _check_belief_change(
        self,
        pred: Prediction,
        state: "CognitiveState",
    ) -> tuple[bool, str]:
        """Check if belief has changed as predicted.

        Condition value formats:
        - "belief:<topic> confidence_delta > 0.1" (change threshold)
        - "belief:<topic> confidence_delta < -0.1" (decrease)
        - "belief:<topic> created" (new belief formation)
        - "belief:<topic> confidence > 0.8" (absolute threshold)

        Note: This method uses _get_belief_sync to handle both sync mocks
        in tests and async calls in production.

        Args:
            pred: Prediction with BELIEF_CHANGE condition
            state: Current cognitive state (unused, but kept for consistency)

        Returns:
            Tuple of (condition_met, outcome_description)
        """
        condition_value = pred.condition_value or ""

        # Parse condition value: "belief:<topic> <condition>"
        match = re.match(r"belief:(\S+)\s+(.+)", condition_value)
        if not match:
            logger.warning(
                f"Prediction {pred.uid} has invalid BELIEF_CHANGE format: {condition_value}"
            )
            return False, ""

        topic = match.group(1)
        condition = match.group(2).strip()

        # Get belief at baseline and current
        try:
            baseline_belief = self._get_belief_sync(topic, at_cycle=pred.baseline_cycle)
            current_belief = self._get_belief_sync(topic)
        except Exception as e:
            logger.warning(f"Error fetching beliefs for {pred.uid}: {e}")
            return False, ""

        # Check "created" condition
        if "created" in condition:
            if baseline_belief is None and current_belief is not None:
                return True, f"Belief about '{topic}' was created (confidence: {current_belief.get('confidence', 0.0):.2f})"
            return False, ""

        # Check "confidence_delta" condition
        if "confidence_delta" in condition:
            delta_match = re.search(
                r"confidence_delta\s*([<>=]+)\s*([-\d.]+)", condition
            )
            if not delta_match:
                logger.warning(
                    f"Prediction {pred.uid} has invalid confidence_delta format: {condition}"
                )
                return False, ""

            operator = delta_match.group(1)
            threshold = float(delta_match.group(2))

            baseline_conf = (
                baseline_belief.get("confidence", 0.0) if baseline_belief else 0.0
            )
            current_conf = (
                current_belief.get("confidence", 0.0) if current_belief else 0.0
            )
            delta = current_conf - baseline_conf

            if self._compare_values(delta, operator, threshold):
                return True, f"Belief '{topic}' confidence delta {delta:.2f} {operator} {threshold}"
            return False, ""

        # Check absolute "confidence" condition
        if "confidence" in condition:
            conf_match = re.search(r"confidence\s*([<>=]+)\s*([\d.]+)", condition)
            if not conf_match:
                logger.warning(
                    f"Prediction {pred.uid} has invalid confidence format: {condition}"
                )
                return False, ""

            if not current_belief:
                return False, ""

            operator = conf_match.group(1)
            threshold = float(conf_match.group(2))
            current_conf = current_belief.get("confidence", 0.0)

            if self._compare_values(current_conf, operator, threshold):
                return True, f"Belief '{topic}' confidence {current_conf:.2f} {operator} {threshold}"
            return False, ""

        logger.warning(
            f"Prediction {pred.uid} has unrecognized BELIEF_CHANGE condition: {condition}"
        )
        return False, ""

    def _get_belief_sync(
        self,
        topic: str,
        at_cycle: int | None = None,
    ) -> dict | None:
        """Get belief by topic, handling both sync mocks and async psyche clients.

        This helper bridges the sync _check_belief_change method with the
        potentially async get_belief_by_topic method on PsycheClient.

        Args:
            topic: The belief topic to search for
            at_cycle: If provided, get the belief as it was at that cycle

        Returns:
            The belief dict or None if not found
        """
        import asyncio
        import inspect

        # Check if get_belief_by_topic exists
        if not hasattr(self._psyche, "get_belief_by_topic"):
            logger.debug("PsycheClient does not have get_belief_by_topic method")
            return None

        method = self._psyche.get_belief_by_topic

        # Call the method - it may be sync (mock) or async (real)
        if at_cycle is not None:
            result = method(topic, at_cycle=at_cycle)
        else:
            result = method(topic)

        # If the result is a coroutine, we need to run it
        if inspect.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - can't use run_until_complete
                # This shouldn't happen since _check_condition is sync
                logger.warning(
                    "get_belief_by_topic returned coroutine in sync context"
                )
                result.close()  # Clean up the coroutine
                return None
            except RuntimeError:
                # No running loop - create one and run
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(result)
                finally:
                    loop.close()

        return result

    def _score_prediction(
        self,
        claim: str,
        outcome: str,
        state: "CognitiveState",
        path_reward: tuple[float, str] | None = None,
    ) -> float:
        """Score prediction accuracy based on claim vs outcome.

        Uses multiple heuristics:
        1. Semantic overlap between claim and current thought/insight
        2. Presence of claimed concepts in state
        3. Conditional logic evaluation
        4. Path-derived reward from knowledge graph (if provided)

        Args:
            claim: The prediction claim
            outcome: Description of what occurred
            state: Current cognitive state
            path_reward: Optional pre-computed (score, explanation) from _compute_path_reward

        Returns:
            Float 0.0-1.0 indicating prediction accuracy
        """
        scores = []

        # 1. Keyword overlap with thought and insight
        claim_keywords = self._extract_keywords(claim)
        thought_keywords = self._extract_keywords(state.thought)
        insight_keywords = self._extract_keywords(state.current_insight)

        if claim_keywords and thought_keywords:
            overlap = len(claim_keywords & thought_keywords) / len(claim_keywords)
            scores.append(overlap)

        if claim_keywords and insight_keywords:
            overlap = len(claim_keywords & insight_keywords) / len(claim_keywords)
            scores.append(overlap)

        # 2. Check for conditional predictions (if X then Y)
        conditional_match = re.match(
            r"if\s+(.+?),?\s+then\s+(.+)", claim, re.IGNORECASE
        )
        if conditional_match:
            antecedent = conditional_match.group(1).lower()
            consequent = conditional_match.group(2).lower()

            # Check if antecedent was observed
            antecedent_observed = (
                antecedent in state.thought.lower()
                or antecedent in state.current_insight.lower()
            )

            if antecedent_observed:
                # Antecedent true - check consequent
                consequent_observed = (
                    consequent in state.thought.lower()
                    or consequent in state.current_insight.lower()
                )
                scores.append(1.0 if consequent_observed else 0.0)
            else:
                # Antecedent false - prediction is neither verified nor falsified
                # Give moderate score
                scores.append(0.5)

        # 3. Check for recent concept alignment
        claim_lower = claim.lower()
        recent_match = sum(
            1 for c in state.recent_concepts if c.lower() in claim_lower
        )
        if state.recent_concepts:
            scores.append(min(recent_match / 3, 1.0))  # Cap at 3 matches

        # 4. Path-derived reward from knowledge graph
        # (Implements "Knowledge Graphs as Implicit Reward Models" - arXiv:2601.15160)
        # Path existence/length provides verifiable supervision for reasoning
        # Only include when we have actual path evidence (not neutral 0.5 fallbacks)
        if path_reward is not None:
            path_score, path_explanation = path_reward
            # Only include non-neutral path scores to avoid diluting heuristic signals
            # Neutral (0.5) indicates no path evidence was found - don't bias the average
            if abs(path_score - 0.5) > 0.01:  # Non-neutral path evidence
                scores.append(path_score)
                if path_explanation and "hops via" in path_explanation:
                    logger.debug(f"Path reward for '{claim[:50]}...': {path_explanation}")

        # Aggregate scores
        if not scores:
            return 0.5  # Neutral if no scoring possible

        return sum(scores) / len(scores)

    async def _score_prediction_async(
        self,
        claim: str,
        outcome: str,
        state: "CognitiveState",
    ) -> float:
        """Async version of _score_prediction that includes path-derived rewards.

        Computes path rewards from the knowledge graph and combines them with
        heuristic scoring for comprehensive prediction verification.

        Args:
            claim: The prediction claim
            outcome: Description of what occurred
            state: Current cognitive state

        Returns:
            Float 0.0-1.0 indicating prediction accuracy
        """
        # Compute path-derived reward asynchronously (with fallback on failure)
        path_reward = None
        try:
            path_reward = await self._compute_path_reward(claim)
        except Exception as e:
            # Graph path queries may fail in tests or when graph is unavailable
            # Fall back to heuristic-only scoring
            logger.debug(f"Path reward computation failed, using heuristics only: {e}")

        # Use synchronous scoring with pre-computed path reward
        return self._score_prediction(claim, outcome, state, path_reward=path_reward)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of lowercase keywords
        """
        if not text:
            return set()

        # Extract words and filter stopwords
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def _extract_entities_from_claim(self, claim: str) -> list[str]:
        """Extract potential entity names from a prediction claim.

        Uses capitalization patterns and common claim structures to identify
        entities that may exist in the knowledge graph.

        Args:
            claim: The prediction claim text

        Returns:
            List of potential entity names (capitalized words/phrases)
        """
        entities = []

        # Extract capitalized words/phrases (likely named entities)
        # Pattern: Capitalized word optionally followed by more capitalized words
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        entities.extend(re.findall(cap_pattern, claim))

        # Extract quoted phrases as potential entities
        quoted = re.findall(r'"([^"]+)"', claim)
        entities.extend(quoted)

        # Extract phrases after common relation indicators
        relation_patterns = [
            r"(?:relates? to|connects? to|leads? to|causes?|implies?)\s+([A-Za-z][A-Za-z\s]+)",
            r"(?:between|from)\s+([A-Za-z][A-Za-z\s]+?)\s+(?:and|to)\s+([A-Za-z][A-Za-z\s]+)",
        ]
        for pattern in relation_patterns:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.extend(m.strip() for m in match if m.strip())
                else:
                    entities.append(match.strip())

        # Deduplicate while preserving order
        # Only filter stopwords for single-word entities; keep multi-word phrases
        # even if they contain stopwords (e.g., "The Matrix", "War and Peace")
        seen = set()
        unique = []
        for e in entities:
            e_lower = e.lower().strip()
            if not e_lower or e_lower in seen:
                continue
            # Check if single-word: if so, filter stopwords; if multi-word, keep it
            is_single_word = " " not in e_lower
            if is_single_word and e_lower in self.STOPWORDS:
                continue
            seen.add(e_lower)
            unique.append(e.strip())

        return unique

    async def _compute_path_reward(
        self,
        claim: str,
        max_hops: int = 4,
    ) -> tuple[float, str]:
        """Compute path-derived reward signal from knowledge graph.

        Implements the key insight from "Knowledge Graphs as Implicit Reward Models":
        Path existence and length between entities provides verifiable supervision
        for reasoning verification.

        The compositional bridge principle: short-hop verification patterns (1-3)
        can generalize to multi-hop queries (4-5).

        Args:
            claim: The prediction claim to verify
            max_hops: Maximum path length to search (default 4)

        Returns:
            Tuple of (reward_score, explanation)
            - reward_score: 0.0-1.0, higher = stronger path evidence
            - explanation: Description of path findings
        """
        entities = self._extract_entities_from_claim(claim)

        if len(entities) < 2:
            return 0.5, "Insufficient entities for path analysis"

        # Query paths between entity pairs
        path_scores = []
        explanations = []

        for i, entity_a in enumerate(entities[:-1]):
            for entity_b in entities[i + 1 :]:
                try:
                    # Query shortest path between entities
                    cypher = """
                    MATCH (a:Entity)
                    WHERE toLower(a.name) CONTAINS toLower($entity_a)
                    WITH a LIMIT 1
                    MATCH (b:Entity)
                    WHERE toLower(b.name) CONTAINS toLower($entity_b)
                    WITH a, b LIMIT 1
                    MATCH path = shortestPath((a)-[*1..{max_hops}]-(b))
                    RETURN length(path) AS path_length,
                           [n IN nodes(path) | n.name] AS path_nodes
                    """.replace("{max_hops}", str(max_hops))

                    results = await self._psyche.query(
                        cypher,
                        {"entity_a": entity_a, "entity_b": entity_b}
                    )

                    if results and results[0].get("path_length") is not None:
                        path_len = results[0]["path_length"]
                        path_nodes = results[0].get("path_nodes", [])

                        # Score inversely proportional to path length
                        # 1 hop = 1.0, 2 hops = 0.75, 3 hops = 0.5, 4 hops = 0.25
                        score = max(0.0, 1.0 - (path_len - 1) * self.PATH_LENGTH_PENALTY)

                        # Compositional bridge: bonus for verifiable short paths
                        # Short paths (1-3 hops) get 20% boost for compositionality
                        if path_len <= 3:
                            score = min(1.0, score * self.COMPOSITIONAL_BRIDGE_BONUS)

                        path_scores.append(score)

                        explanations.append(
                            f"Path({entity_a}→{entity_b}): {path_len} hops via {path_nodes}"
                        )
                    else:
                        # No path found - slight negative signal
                        path_scores.append(0.2)
                        explanations.append(f"No path: {entity_a}↛{entity_b}")

                except Exception as e:
                    logger.debug(f"Path query failed for {entity_a}↔{entity_b}: {e}")
                    # Query failure - neutral (don't penalize)
                    continue

        if not path_scores:
            return 0.5, "No path queries completed"

        avg_score = sum(path_scores) / len(path_scores)
        return avg_score, "; ".join(explanations[:3])  # Limit explanation length

    def _evaluate_metric_expression(
        self,
        expression: str,
        state: "CognitiveState",
        baseline: dict,
    ) -> tuple[bool, float, str]:
        """Evaluate a metric threshold expression.

        Supports several expression formats:
        - "metric > value" / "metric < value" - absolute threshold
        - "metric > baseline + delta" - relative to baseline
        - "delta:metric > value" - change since baseline
        - "ratio:metric_a/metric_b > value" - ratio between metrics

        Args:
            expression: The metric expression (e.g., "semantic_entropy > 0.6")
            state: Current cognitive state (contains metrics_snapshot if available)
            baseline: Baseline metrics dict from prediction creation

        Returns:
            Tuple of (condition_met, actual_value, explanation)
        """
        # Get current metrics from state if available
        current = self._get_current_metrics_dict(state)

        if not current:
            logger.debug("No current metrics available for metric evaluation")
            return False, 0.0, "No metrics available"

        # Parse expression: "metric_name operator threshold_expr"
        # Examples: "semantic_entropy > 0.6", "orphan_rate < baseline + 0.05"
        match = re.match(r"([\w:\/]+)\s*([<>=!]+)\s*(.+)", expression.strip())
        if not match:
            logger.warning(f"Failed to parse metric expression: {expression}")
            return False, 0.0, f"Invalid expression: {expression}"

        metric_spec, operator, threshold_expr = match.groups()
        metric_spec = metric_spec.strip()
        threshold_expr = threshold_expr.strip()

        # Get the actual value based on metric specification
        try:
            if metric_spec.startswith("delta:"):
                # Delta: current - baseline
                actual_name = metric_spec[6:]
                current_val = current.get(actual_name, 0.0)
                baseline_val = baseline.get(actual_name, 0.0)
                value = current_val - baseline_val
                metric_desc = f"Δ{actual_name}"
            elif metric_spec.startswith("ratio:"):
                # Ratio: metric_a / metric_b
                parts = metric_spec[6:].split("/")
                if len(parts) != 2:
                    return False, 0.0, f"Invalid ratio format: {metric_spec}"
                a_val = current.get(parts[0], 0.0)
                b_val = current.get(parts[1], 1.0)
                value = a_val / b_val if b_val != 0 else 0.0
                metric_desc = f"{parts[0]}/{parts[1]}"
            else:
                # Simple metric lookup
                value = current.get(metric_spec, 0.0)
                metric_desc = metric_spec

            # Evaluate threshold expression
            # Replace "baseline" with actual baseline value if present
            if "baseline" in threshold_expr:
                baseline_val = baseline.get(metric_spec.replace("delta:", ""), 0.0)
                threshold_expr = threshold_expr.replace("baseline", str(baseline_val))

            # Safely evaluate the threshold expression
            threshold = self._safe_eval_threshold(threshold_expr)
            if threshold is None:
                return False, value, f"Invalid threshold: {threshold_expr}"

            # Compare value against threshold
            met = self._compare_values(value, operator, threshold)

            explanation = f"{metric_desc}={value:.4f} {operator} {threshold:.4f}"
            if met:
                logger.debug(f"Metric condition met: {explanation}")
            else:
                logger.debug(f"Metric condition not met: {explanation}")

            return met, value, explanation

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error evaluating metric expression '{expression}': {e}")
            return False, 0.0, f"Evaluation error: {e}"

    def _get_current_metrics_dict(self, state: "CognitiveState") -> dict:
        """Extract current metrics from cognitive state.

        Args:
            state: Current cognitive state

        Returns:
            Dict of metric name -> value, or empty dict if unavailable
        """
        # Check if state has a metrics_snapshot attribute
        if hasattr(state, "metrics_snapshot") and state.metrics_snapshot:
            if isinstance(state.metrics_snapshot, MetricsSnapshot):
                return state.metrics_snapshot.to_dict()
            elif isinstance(state.metrics_snapshot, dict):
                return state.metrics_snapshot

        # Fallback: no metrics available in state
        return {}

    def _safe_eval_threshold(self, expr: str) -> float | None:
        """Safely evaluate a threshold expression using a simple parser.

        Only allows numeric literals and basic arithmetic (+, -).
        Does not use eval() for security.

        Args:
            expr: Expression like "0.5 + 0.05" or "0.6"

        Returns:
            Evaluated float value, or None if invalid
        """
        expr = expr.strip()

        # Try direct float conversion first (most common case)
        try:
            return float(expr)
        except ValueError:
            pass

        # Parse simple arithmetic: number (+|-) number (+|-) number ...
        # Split on + and - while keeping the operators
        tokens = re.split(r"(\s*[\+\-]\s*)", expr)

        if not tokens:
            return None

        try:
            # First token should be a number
            result = float(tokens[0].strip())

            # Process remaining pairs of (operator, number)
            i = 1
            while i < len(tokens) - 1:
                op = tokens[i].strip()
                num = float(tokens[i + 1].strip())

                if op == "+":
                    result += num
                elif op == "-":
                    result -= num
                else:
                    return None  # Unknown operator

                i += 2

            return result
        except (ValueError, IndexError):
            return None

    def _compare_values(self, value: float, operator: str, threshold: float) -> bool:
        """Compare value against threshold using operator.

        Args:
            value: Actual metric value
            operator: Comparison operator (>, <, >=, <=, ==, !=)
            threshold: Threshold to compare against

        Returns:
            True if comparison is satisfied
        """
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==" or operator == "=":
            return abs(value - threshold) < 0.0001
        elif operator == "!=" or operator == "<>":
            return abs(value - threshold) >= 0.0001
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def _should_trigger_follow_up(
        self,
        pred: Prediction,
        accuracy: float,
    ) -> bool:
        """Determine if prediction outcome warrants follow-up simulation.

        Triggers follow-up for significant learning opportunities:
        - Falsified high-confidence prediction (surprising failure)
        - Verified low-confidence prediction (surprising success)

        Args:
            pred: The resolved prediction
            accuracy: Accuracy score

        Returns:
            True if follow-up simulation should be triggered
        """
        # Falsified high-confidence prediction = significant learning opportunity
        if (
            pred.confidence > self.HIGH_CONFIDENCE_THRESHOLD
            and accuracy < self.ACCURACY_FALSIFIED_THRESHOLD
        ):
            logger.info(
                f"Prediction {pred.uid} triggered follow-up: "
                f"high confidence ({pred.confidence:.2f}) falsified"
            )
            return True

        # Verified low-confidence prediction = surprising confirmation
        if (
            pred.confidence < self.LOW_CONFIDENCE_THRESHOLD
            and accuracy > self.ACCURACY_VERIFIED_THRESHOLD
        ):
            logger.info(
                f"Prediction {pred.uid} triggered follow-up: "
                f"low confidence ({pred.confidence:.2f}) verified"
            )
            return True

        return False

    async def _is_in_follow_up_cooldown(
        self,
        hypothesis: "Hypothesis",
        current_cycle: int,
    ) -> bool:
        """Check if hypothesis is in follow-up simulation cooldown.

        Prevents the same hypothesis from triggering follow-up simulation
        in consecutive cycles, which causes repetitive narration.

        Args:
            hypothesis: The hypothesis to check
            current_cycle: Current cognitive cycle

        Returns:
            True if hypothesis should skip follow-up (still in cooldown)
        """
        # Import here to avoid circular imports
        from core.cognitive.simulation.schemas import Hypothesis

        if not isinstance(hypothesis, Hypothesis):
            return False

        if hypothesis.last_follow_up_cycle is None:
            return False

        cycles_since_follow_up = current_cycle - hypothesis.last_follow_up_cycle
        return cycles_since_follow_up < self.FOLLOW_UP_COOLDOWN_CYCLES

    async def _emit_lifecycle_event(
        self,
        hypothesis: "Hypothesis",
        new_status: "HypothesisStatus",
    ) -> None:
        """Emit event and narrate hypothesis lifecycle transition.

        Args:
            hypothesis: The hypothesis that transitioned
            new_status: The new lifecycle status
        """
        total = hypothesis.verified_count + hypothesis.falsified_count
        rate = hypothesis.verified_count / total if total > 0 else 0.0

        if new_status == HypothesisStatus.VERIFIED:
            logger.info(
                f"Hypothesis VERIFIED: {hypothesis.uid} "
                f"({hypothesis.verified_count}/{total} = {rate:.0%})"
            )
        elif new_status == HypothesisStatus.FALSIFIED:
            logger.info(
                f"Hypothesis FALSIFIED: {hypothesis.uid} "
                f"({hypothesis.verified_count}/{total} = {rate:.0%})"
            )
        elif new_status == HypothesisStatus.ACTIVE:
            logger.info(
                f"Hypothesis ACTIVATED: {hypothesis.uid} "
                f"(first prediction evaluated)"
            )

    async def _generate_skill_if_worthy(self, hypothesis: "Hypothesis") -> None:
        """Generate and persist skill if hypothesis has sufficient content.

        Implements UPSKILL pattern: auto-generates portable skill packages
        from verified hypotheses to improve future cognitive quality.

        Args:
            hypothesis: A verified hypothesis to extract skill from
        """
        if not self._embedding_service:
            logger.debug(
                f"Skipping skill generation for {hypothesis.uid}: "
                "no embedding service configured"
            )
            return

        try:
            from core.cognitive.skill_generator import generate_skill_from_hypothesis

            skill = await generate_skill_from_hypothesis(
                hypothesis, self._embedding_service
            )
            if skill:
                success = await self._psyche.create_learned_skill(skill)
                if success:
                    logger.info(
                        f"Generated skill {skill.uid} ({skill.name}) "
                        f"from hypothesis {hypothesis.uid}"
                    )
                else:
                    logger.warning(
                        f"Failed to persist skill from hypothesis {hypothesis.uid}"
                    )
            else:
                logger.debug(
                    f"No skill generated from hypothesis {hypothesis.uid}: "
                    "insufficient content"
                )
        except Exception as e:
            logger.warning(f"Skill generation failed for {hypothesis.uid}: {e}")

    async def get_prediction_summary(
        self,
        hypothesis_uid: str,
    ) -> dict:
        """Get summary of prediction outcomes for a hypothesis.

        Args:
            hypothesis_uid: UID of the hypothesis

        Returns:
            Dict with verification statistics
        """
        hypothesis = await self._psyche.get_hypothesis(hypothesis_uid)
        if not hypothesis:
            return {
                "hypothesis_uid": hypothesis_uid,
                "total": 0,
                "verified": 0,
                "falsified": 0,
                "pending": 0,
                "verification_rate": 0.0,
            }

        return {
            "hypothesis_uid": hypothesis_uid,
            "statement": hypothesis.statement[:100],
            "total": hypothesis.predictions_count,
            "verified": hypothesis.verified_count,
            "falsified": hypothesis.falsified_count,
            "pending": hypothesis.predictions_count - hypothesis.verified_count - hypothesis.falsified_count,
            "verification_rate": hypothesis.verification_rate,
        }

    async def _update_vector_effectiveness(
        self,
        hypothesis_uid: str,
        verified: bool,
    ) -> None:
        """Update steering vector effectiveness based on prediction outcome.

        This implements the feedback loop for outcome-based self-steering:
        1. Get the hypothesis for the prediction
        2. Check if hypothesis should be abandoned based on verification rate
        3. If hypothesis has a steering_vector_uid, get the vector from psyche
        4. Call vector.update_effectiveness(verified=True/False)
        5. If vector.should_prune(), mark it and the hypothesis as abandoned
        6. Persist updated vector back to psyche

        Args:
            hypothesis_uid: UID of the hypothesis whose prediction was verified
            verified: True if prediction was verified, False if falsified
        """
        # Get the hypothesis to check for steering vector
        hypothesis = await self._psyche.get_hypothesis(hypothesis_uid)
        if not hypothesis:
            logger.debug(
                f"Hypothesis {hypothesis_uid} not found, skipping vector update"
            )
            return

        # Check verification-rate-based retirement (applies to ALL hypotheses)
        # This catches hypotheses without steering vectors that would otherwise persist forever
        # Use LINEAGE counts to track across refined hypotheses (prevents count resets)
        total_resolved = hypothesis.verified_count + hypothesis.falsified_count
        lineage_total = hypothesis.total_lineage_predictions + total_resolved

        # Check both current and lineage counts for abandonment
        should_abandon = False
        if lineage_total >= self.MIN_PREDICTIONS_FOR_ABANDON:
            # Calculate lineage rate including current hypothesis
            lineage_verified = hypothesis.lineage_verified_count + hypothesis.verified_count
            lineage_rate = lineage_verified / lineage_total if lineage_total > 0 else 0.0
            if lineage_rate < self.VERIFICATION_RATE_ABANDON_THRESHOLD:
                should_abandon = True
                logger.info(
                    f"Abandoning hypothesis {hypothesis_uid} due to poor LINEAGE verification rate: "
                    f"{lineage_rate:.2%} ({lineage_verified} verified, "
                    f"{lineage_total - lineage_verified} falsified across {lineage_total} predictions). "
                    f"Parent: {hypothesis.parent_hypothesis_uid or 'None'}"
                )
        elif total_resolved >= self.MIN_PREDICTIONS_FOR_ABANDON:
            # Fallback to current-only check for hypotheses without lineage
            if hypothesis.verification_rate < self.VERIFICATION_RATE_ABANDON_THRESHOLD:
                should_abandon = True
                logger.info(
                    f"Abandoning hypothesis {hypothesis_uid} due to poor verification rate: "
                    f"{hypothesis.verification_rate:.2%} ({hypothesis.verified_count} verified, "
                    f"{hypothesis.falsified_count} falsified after {total_resolved} predictions)"
                )

        if should_abandon:
            await self._psyche.update_hypothesis_status(
                hypothesis_uid, HypothesisStatus.ABANDONED.value
            )

            # Clean up steering vector if present
            if hypothesis.steering_vector_uid:
                await self._psyche.delete_hypothesis_steering_vector(
                    hypothesis.steering_vector_uid
                )
                logger.debug(
                    f"Also deleted steering vector {hypothesis.steering_vector_uid}"
                )
            return

        # Check if hypothesis has an associated steering vector
        if not hypothesis.steering_vector_uid:
            logger.debug(
                f"Hypothesis {hypothesis_uid} has no steering vector, skipping EMA update"
            )
            return

        # Get the steering vector
        vector = await self._psyche.get_hypothesis_steering_vector(
            hypothesis.steering_vector_uid
        )
        if not vector:
            logger.warning(
                f"Steering vector {hypothesis.steering_vector_uid} not found "
                f"for hypothesis {hypothesis_uid}"
            )
            return

        # Update vector effectiveness using EMA
        vector.update_effectiveness(verified=verified)

        logger.debug(
            f"Updated vector {vector.uid} effectiveness to {vector.effectiveness_score:.3f} "
            f"(verified={verified})"
        )

        # Check if vector should be pruned
        if vector.should_prune():
            logger.info(
                f"Pruning underperforming vector {vector.uid} "
                f"(effectiveness={vector.effectiveness_score:.3f}, "
                f"verified={vector.verified_count}, falsified={vector.falsified_count})"
            )

            # Mark hypothesis as abandoned
            await self._psyche.update_hypothesis_status(
                hypothesis_uid, HypothesisStatus.ABANDONED.value
            )

            # Delete the steering vector
            await self._psyche.delete_hypothesis_steering_vector(vector.uid)

            logger.info(
                f"Abandoned hypothesis {hypothesis_uid} and pruned vector {vector.uid}"
            )
        else:
            # Persist updated vector
            await self._psyche.save_hypothesis_steering_vector(vector)
