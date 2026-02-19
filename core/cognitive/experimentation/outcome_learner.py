"""Outcome-based learning for experiment decisions.

Tracks prediction outcomes related to parameters and uses them to
prioritize or deprioritize experiment proposals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from core.cognitive.experimentation.utils import extract_parameter_from_claim

if TYPE_CHECKING:
    from core.cognitive.experimentation.schemas import ExperimentProposal
    from core.cognitive.simulation.schemas import Prediction
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)

# --- Proposal adjustment thresholds ---
# Minimum experiments required before adjusting proposal parameters
MIN_EXPERIMENTS_FOR_ADJUSTMENT = 3
# Success rate below which min_effect_size is increased
LOW_SUCCESS_RATE_FOR_ADJUSTMENT = 0.3
# Floor value for min_effect_size when success rate is low
MIN_EFFECT_SIZE_FLOOR = 0.15
# Default rollback trigger when past experiments were rejected
DEFAULT_ROLLBACK_TRIGGER = -0.1

# --- Skip proposal thresholds ---
MIN_EXPERIMENTS_FOR_SKIP = 5  # Minimum experiments before skip logic applies
MIN_EXPERIMENT_SUCCESS_RATE = 0.2  # Skip if success rate below this
MIN_PREDICTIONS_FOR_SKIP = 10  # Minimum predictions before skip logic applies
MIN_PREDICTION_SUCCESS_RATE = 0.15  # Skip if prediction success below this


@dataclass
class ParameterOutcome:
    """Aggregated outcomes for a parameter."""

    parameter_path: str
    predictions_total: int = 0
    predictions_verified: int = 0
    predictions_falsified: int = 0
    experiments_total: int = 0
    experiments_adopted: int = 0
    experiments_rejected: int = 0
    experiments_inconclusive: int = 0

    @property
    def prediction_success_rate(self) -> float:
        """Success rate of predictions about this parameter."""
        if self.predictions_total == 0:
            return 0.5  # Neutral prior
        return self.predictions_verified / self.predictions_total

    @property
    def experiment_success_rate(self) -> float:
        """Success rate of experiments on this parameter."""
        if self.experiments_total == 0:
            return 0.5  # Neutral prior
        return self.experiments_adopted / self.experiments_total

    @property
    def confidence_modifier(self) -> float:
        """Modifier for proposal confidence based on history.

        Returns value between 0.5 (low confidence) and 1.5 (high confidence).
        """
        # Weight prediction and experiment success equally
        combined = (self.prediction_success_rate + self.experiment_success_rate) / 2
        # Map 0-1 to 0.5-1.5
        return 0.5 + combined


class ExperimentOutcomeLearner:
    """Learns from prediction and experiment outcomes.

    Maintains per-parameter statistics and adjusts proposal confidence
    based on historical success rates.
    """

    def __init__(self, psyche: Optional["PsycheClient"] = None):
        """Initialize learner.

        Args:
            psyche: PsycheClient for querying historical outcomes (optional)
        """
        self._psyche = psyche
        self._cache: dict[str, ParameterOutcome] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether historical outcomes have been loaded."""
        return self._loaded

    async def load_history(self) -> None:
        """Load historical outcomes from Psyche."""
        if not self._psyche:
            logger.debug("No PsycheClient provided, skipping history load")
            self._loaded = True
            return

        try:
            # Load prediction outcomes for parameters
            prediction_stats = await self._psyche.get_parameter_prediction_stats()

            # Load experiment outcomes
            experiment_stats = await self._psyche.get_parameter_experiment_stats()

            # Merge into cache
            all_params = set(prediction_stats.keys()) | set(experiment_stats.keys())
            for param in all_params:
                pred = prediction_stats.get(param, {})
                exp = experiment_stats.get(param, {})

                self._cache[param] = ParameterOutcome(
                    parameter_path=param,
                    predictions_total=pred.get("total", 0),
                    predictions_verified=pred.get("verified", 0),
                    predictions_falsified=pred.get("falsified", 0),
                    experiments_total=exp.get("total", 0),
                    experiments_adopted=exp.get("adopted", 0),
                    experiments_rejected=exp.get("rejected", 0),
                    experiments_inconclusive=exp.get("inconclusive", 0),
                )

            logger.info(f"Loaded outcome history for {len(self._cache)} parameters")
            self._loaded = True

        except Exception as e:
            logger.warning(f"Failed to load outcome history: {e}")
            self._loaded = True  # Mark as loaded to prevent retry loops

    def get_outcome(self, parameter_path: str) -> Optional[ParameterOutcome]:
        """Get outcome statistics for a parameter.

        Args:
            parameter_path: Full parameter path

        Returns:
            ParameterOutcome or None if no history
        """
        return self._cache.get(parameter_path)

    def adjust_proposal(
        self, proposal: "ExperimentProposal"
    ) -> "ExperimentProposal":
        """Adjust proposal based on historical outcomes.

        Modifies:
        - min_effect_size: Increase if past experiments showed small effects
        - rollback_trigger: Tighten if past experiments had negative effects

        Args:
            proposal: Original proposal

        Returns:
            Adjusted proposal (modified in place)
        """
        outcome = self._cache.get(proposal.parameter_path)
        if not outcome:
            return proposal

        # Adjust min_effect_size based on experiment history
        if outcome.experiments_total >= MIN_EXPERIMENTS_FOR_ADJUSTMENT:
            if outcome.experiment_success_rate < LOW_SUCCESS_RATE_FOR_ADJUSTMENT:
                # Low success rate: require larger effect to be worth it
                old_min = proposal.min_effect_size
                proposal.min_effect_size = max(MIN_EFFECT_SIZE_FLOOR, proposal.min_effect_size)
                if proposal.min_effect_size != old_min:
                    logger.debug(
                        f"Increased min_effect_size for {proposal.parameter_path} "
                        f"from {old_min} to {proposal.min_effect_size} "
                        f"(low historical success: {outcome.experiment_success_rate:.2%})"
                    )

        # Add rollback trigger if missing and history shows risk
        if proposal.rollback_trigger is None and outcome.experiments_rejected > 0:
            proposal.rollback_trigger = DEFAULT_ROLLBACK_TRIGGER
            logger.debug(
                f"Added rollback trigger for {proposal.parameter_path} "
                f"(past rejections: {outcome.experiments_rejected})"
            )

        return proposal

    def should_skip_proposal(
        self, proposal: "ExperimentProposal"
    ) -> tuple[bool, str]:
        """Determine if proposal should be skipped based on history.

        Args:
            proposal: Proposal to evaluate

        Returns:
            Tuple of (should_skip, reason)
        """
        outcome = self._cache.get(proposal.parameter_path)
        if not outcome:
            return False, ""

        # Skip if too many recent failures
        if (
            outcome.experiments_total >= MIN_EXPERIMENTS_FOR_SKIP
            and outcome.experiment_success_rate < MIN_EXPERIMENT_SUCCESS_RATE
        ):
            return True, (
                f"Parameter {proposal.parameter_path} has low experiment success rate "
                f"({outcome.experiment_success_rate:.0%} over "
                f"{outcome.experiments_total} experiments)"
            )

        # Skip if predictions about this parameter are consistently wrong
        if (
            outcome.predictions_total >= MIN_PREDICTIONS_FOR_SKIP
            and outcome.prediction_success_rate < MIN_PREDICTION_SUCCESS_RATE
        ):
            return True, (
                f"Predictions about {proposal.parameter_path} rarely verify "
                f"({outcome.prediction_success_rate:.0%} over "
                f"{outcome.predictions_total} predictions)"
            )

        return False, ""

    def record_prediction_outcome(
        self,
        prediction: "Prediction",
        verified: bool,
    ) -> None:
        """Record a prediction outcome for learning.

        Called when a prediction about a parameter is verified or falsified.

        Args:
            prediction: The prediction that resolved
            verified: Whether it was verified (True) or falsified (False)
        """
        # Extract parameter from prediction claim using shared utility
        parameter_path = extract_parameter_from_claim(prediction.claim)
        if not parameter_path:
            return

        # Update cache
        if parameter_path not in self._cache:
            self._cache[parameter_path] = ParameterOutcome(
                parameter_path=parameter_path
            )

        outcome = self._cache[parameter_path]
        outcome.predictions_total += 1
        if verified:
            outcome.predictions_verified += 1
        else:
            outcome.predictions_falsified += 1

        logger.debug(
            f"Recorded prediction outcome for {parameter_path}: "
            f"{'verified' if verified else 'falsified'} "
            f"(total: {outcome.predictions_total})"
        )

    def record_experiment_outcome(
        self,
        parameter_path: str,
        recommendation: str,
    ) -> None:
        """Record an experiment outcome for learning.

        Args:
            parameter_path: Parameter that was experimented on
            recommendation: "ADOPT", "REJECT", or "INCONCLUSIVE"
        """
        if parameter_path not in self._cache:
            self._cache[parameter_path] = ParameterOutcome(
                parameter_path=parameter_path
            )

        outcome = self._cache[parameter_path]
        outcome.experiments_total += 1

        if recommendation == "ADOPT":
            outcome.experiments_adopted += 1
        elif recommendation == "REJECT":
            outcome.experiments_rejected += 1
        else:
            outcome.experiments_inconclusive += 1

        logger.debug(
            f"Recorded experiment outcome for {parameter_path}: {recommendation} "
            f"(total: {outcome.experiments_total})"
        )

    def get_statistics(self) -> dict:
        """Get summary statistics for all tracked parameters.

        Returns:
            Dict with summary statistics
        """
        if not self._cache:
            return {"parameters_tracked": 0}

        total_predictions = sum(o.predictions_total for o in self._cache.values())
        total_experiments = sum(o.experiments_total for o in self._cache.values())
        total_verified = sum(o.predictions_verified for o in self._cache.values())
        total_adopted = sum(o.experiments_adopted for o in self._cache.values())

        return {
            "parameters_tracked": len(self._cache),
            "total_predictions": total_predictions,
            "total_experiments": total_experiments,
            "prediction_success_rate": (
                total_verified / total_predictions if total_predictions > 0 else 0
            ),
            "experiment_success_rate": (
                total_adopted / total_experiments if total_experiments > 0 else 0
            ),
        }
