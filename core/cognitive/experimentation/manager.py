"""ExperimentManager: Core logic for self-experimentation lifecycle.

This module orchestrates the experiment lifecycle including validation,
starting, phase transitions, rollback handling, and outcome analysis.

Experiments flow through phases:
    PENDING -> BASELINE -> TREATMENT -> WASHOUT -> COMPLETE
                                    |
                                    v (if rollback triggered)
                                  ABORTED
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from core.cognitive.experimentation.schemas import (
    DOMAIN_COOLDOWN,
    ExperimentMeasurement,
    ExperimentPhase,
    ExperimentProposal,
)
from core.cognitive.simulation.schemas import Hypothesis, HypothesisStatus, MetricsSnapshot

if TYPE_CHECKING:
    from core.cognitive.experimentation.outcome_learner import ExperimentOutcomeLearner
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


class ConfigStore(Protocol):
    """Protocol for configuration storage.

    Abstracts the config store to allow different implementations
    (in-memory for testing, persistent for production).
    """

    async def get(self, path: str) -> float:
        """Get a configuration value by dotted path."""
        ...

    async def set(self, path: str, value: float) -> None:
        """Set a configuration value by dotted path."""
        ...


class ExperimentManager:
    """Orchestrates the experiment lifecycle.

    The ExperimentManager handles:
    - Proposal validation (allowed params, active check, cooldown)
    - Experiment creation and starting
    - Phase transitions during each cognitive cycle
    - Rollback trigger detection and abort
    - Outcome analysis with ADOPT/REJECT/INCONCLUSIVE recommendations

    Attributes:
        _psyche: PsycheClient for graph persistence
        _config_store: ConfigStore for parameter manipulation
        _current_cycle: Current cognitive cycle number
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        config_store: ConfigStore,
        outcome_learner: "ExperimentOutcomeLearner | None" = None,
    ) -> None:
        """Initialize the ExperimentManager.

        Args:
            psyche: PsycheClient for persisting experiments
            config_store: ConfigStore for reading/writing parameters
            outcome_learner: Optional outcome learner for recording experiment results
        """
        self._psyche = psyche
        self._config_store = config_store
        self._outcome_learner = outcome_learner
        self._current_cycle: int = 0

    async def validate(self, proposal: ExperimentProposal) -> bool:
        """Validate an experiment proposal.

        Checks:
        1. Parameter path is in allowed list for domain
        2. No active experiment in the same domain
        3. Cooldown period has elapsed since last experiment in domain

        Args:
            proposal: The experiment proposal to validate

        Returns:
            True if proposal is valid, False otherwise
        """
        # Check 1: Parameter path is allowed
        if not proposal.is_valid_parameter():
            logger.info(
                f"Rejected proposal: {proposal.parameter_path} not in allowed list "
                f"for domain {proposal.domain.value}"
            )
            return False

        # Check 2: No active experiment in domain
        active_in_domain = await self._psyche.get_active_experiments(proposal.domain)
        if active_in_domain:
            logger.info(
                f"Rejected proposal: active experiment {active_in_domain[0].uid} "
                f"in domain {proposal.domain.value}"
            )
            return False

        # Check 3: Cooldown period elapsed
        last_result = await self._psyche.get_last_experiment_in_domain(proposal.domain)
        if last_result is not None:
            last_experiment, completed_cycle = last_result
            cycles_since = self._current_cycle - completed_cycle
            if cycles_since < DOMAIN_COOLDOWN:
                logger.info(
                    f"Rejected proposal: cooldown in effect for domain {proposal.domain.value}. "
                    f"{cycles_since} cycles since last experiment, need {DOMAIN_COOLDOWN}"
                )
                return False

        return True

    async def start(self, proposal: ExperimentProposal) -> Hypothesis:
        """Start an experiment from a validated proposal.

        Creates a Hypothesis with is_experiment=True, capturing:
        - Control value (current parameter value)
        - Treatment value (from proposal)
        - Phase configuration (baseline, treatment, washout cycles)
        - Success criteria (target metric, direction, min effect size)

        Args:
            proposal: Validated experiment proposal

        Returns:
            Created Hypothesis representing the experiment
        """
        # Get current value as control
        control_value = await self._config_store.get(proposal.parameter_path)

        # Build statement
        statement = (
            f"Experiment: {proposal.parameter_path} "
            f"{control_value} -> {proposal.treatment_value}. "
            f"Rationale: {proposal.rationale}"
        )

        # Create hypothesis with experiment fields
        hypothesis = Hypothesis(
            statement=statement,
            is_experiment=True,
            experiment_domain=proposal.domain,
            parameter_path=proposal.parameter_path,
            control_value=control_value,
            treatment_value=proposal.treatment_value,
            baseline_cycles=proposal.baseline_cycles,
            treatment_cycles=proposal.treatment_cycles,
            washout_cycles=proposal.washout_cycles,
            current_phase=ExperimentPhase.PENDING.value,
            phase_start_cycle=self._current_cycle,
            target_metric=proposal.target_metric,
            expected_direction=proposal.expected_direction,
            min_effect_size=proposal.min_effect_size,
            rollback_trigger=proposal.rollback_trigger,
            status=HypothesisStatus.ACTIVE,
            cycle_generated=self._current_cycle,
        )

        # Persist to graph
        await self._psyche.create_hypothesis(hypothesis)
        logger.info(
            f"Started experiment {hypothesis.uid}: {proposal.parameter_path} "
            f"{control_value} -> {proposal.treatment_value}"
        )

        return hypothesis

    async def has_active(self) -> bool:
        """Check if any experiments are currently active.

        Returns:
            True if there are active experiments, False otherwise
        """
        active = await self._psyche.get_active_experiments()
        return len(active) > 0

    async def tick(
        self,
        cycle: int,
        metrics: MetricsSnapshot,
    ) -> None:
        """Process one cognitive cycle for all active experiments.

        Called each cycle by the orchestrator. For each active experiment:
        1. Records current metrics as measurement
        2. Checks phase transition conditions
        3. Handles rollback triggers (if in treatment phase)
        4. Completes experiment if all phases done

        Args:
            cycle: Current cognitive cycle number
            metrics: Current cycle's MetricsSnapshot
        """
        self._current_cycle = cycle

        # Get all active experiments
        active_experiments = await self._psyche.get_active_experiments()
        if not active_experiments:
            return

        for experiment in active_experiments:
            await self._process_experiment(experiment, cycle, metrics)

    async def _process_experiment(
        self,
        experiment: Hypothesis,
        cycle: int,
        metrics: MetricsSnapshot,
    ) -> None:
        """Process a single experiment for the current cycle.

        Args:
            experiment: The experiment hypothesis
            cycle: Current cognitive cycle
            metrics: Current metrics snapshot
        """
        current_phase = ExperimentPhase(experiment.current_phase)
        phase_start = experiment.phase_start_cycle or cycle
        cycles_in_phase = cycle - phase_start

        # Record measurement for all phases except PENDING
        if current_phase != ExperimentPhase.PENDING:
            await self._record_measurement(experiment, cycle, current_phase, metrics)

        # Handle phase transitions
        if current_phase == ExperimentPhase.PENDING:
            await self._start_baseline(experiment, cycle)

        elif current_phase == ExperimentPhase.BASELINE:
            if cycles_in_phase >= experiment.baseline_cycles:
                await self._start_treatment(experiment, cycle)

        elif current_phase == ExperimentPhase.TREATMENT:
            # Check rollback trigger first
            if await self._should_abort(experiment, metrics):
                await self._abort_experiment(experiment, cycle)
            elif cycles_in_phase >= experiment.treatment_cycles:
                await self._start_washout(experiment, cycle)

        elif current_phase == ExperimentPhase.WASHOUT:
            if cycles_in_phase >= experiment.washout_cycles:
                await self._complete_experiment(experiment, cycle)

    async def _record_measurement(
        self,
        experiment: Hypothesis,
        cycle: int,
        phase: ExperimentPhase,
        metrics: MetricsSnapshot,
    ) -> None:
        """Record a measurement for the experiment.

        Args:
            experiment: The experiment hypothesis
            cycle: Current cycle
            phase: Current experiment phase
            metrics: Current metrics snapshot
        """
        measurement = ExperimentMeasurement(
            experiment_uid=experiment.uid,
            cycle=cycle,
            phase=phase,
            snapshot=metrics,
        )
        await self._psyche.record_experiment_measurement(measurement)

    async def _start_baseline(
        self,
        experiment: Hypothesis,
        cycle: int,
    ) -> None:
        """Transition from PENDING to BASELINE.

        Args:
            experiment: The experiment to transition
            cycle: Current cycle number
        """
        await self._psyche.update_experiment_phase(
            experiment.uid,
            ExperimentPhase.BASELINE,
            cycle,
        )
        logger.info(f"Experiment {experiment.uid}: PENDING -> BASELINE at cycle {cycle}")

    async def _start_treatment(
        self,
        experiment: Hypothesis,
        cycle: int,
    ) -> None:
        """Transition from BASELINE to TREATMENT.

        Computes baseline mean and applies treatment value.

        Args:
            experiment: The experiment to transition
            cycle: Current cycle number
        """
        # Compute baseline mean from measurements
        measurements = await self._psyche.get_experiment_measurements(
            experiment.uid,
            ExperimentPhase.BASELINE,
        )
        baseline_mean = self._compute_mean(
            measurements,
            experiment.target_metric,
        )

        # Update experiment with baseline mean
        # Note: We store baseline_mean on the hypothesis for rollback comparison
        experiment.baseline_mean = baseline_mean

        # Persist baseline_mean to graph for restart resilience
        await self._psyche.update_experiment_baseline_mean(
            experiment.uid,
            baseline_mean,
        )

        # Apply treatment value
        await self._config_store.set(
            experiment.parameter_path,
            experiment.treatment_value,
        )

        await self._psyche.update_experiment_phase(
            experiment.uid,
            ExperimentPhase.TREATMENT,
            cycle,
        )
        logger.info(
            f"Experiment {experiment.uid}: BASELINE -> TREATMENT at cycle {cycle}. "
            f"Baseline mean={baseline_mean:.3f}, applying treatment={experiment.treatment_value}"
        )

    async def _start_washout(
        self,
        experiment: Hypothesis,
        cycle: int,
    ) -> None:
        """Transition from TREATMENT to WASHOUT.

        Reverts parameter to control value.

        Args:
            experiment: The experiment to transition
            cycle: Current cycle number
        """
        # Revert to control value
        await self._config_store.set(
            experiment.parameter_path,
            experiment.control_value,
        )

        await self._psyche.update_experiment_phase(
            experiment.uid,
            ExperimentPhase.WASHOUT,
            cycle,
        )
        logger.info(
            f"Experiment {experiment.uid}: TREATMENT -> WASHOUT at cycle {cycle}. "
            f"Reverted to control={experiment.control_value}"
        )

    async def _complete_experiment(
        self,
        experiment: Hypothesis,
        cycle: int,
    ) -> None:
        """Complete the experiment with outcome analysis.

        Args:
            experiment: The experiment to complete
            cycle: Current cycle number
        """
        # Get measurements by phase (more efficient than fetching all and filtering)
        baseline_measurements = await self._psyche.get_experiment_measurements(
            experiment.uid, phase=ExperimentPhase.BASELINE
        )
        treatment_measurements = await self._psyche.get_experiment_measurements(
            experiment.uid, phase=ExperimentPhase.TREATMENT
        )

        baseline_mean = self._compute_mean(baseline_measurements, experiment.target_metric)
        treatment_mean = self._compute_mean(treatment_measurements, experiment.target_metric)

        # Compute effect size
        if baseline_mean != 0:
            effect_size = (treatment_mean - baseline_mean) / abs(baseline_mean)
        else:
            effect_size = treatment_mean - baseline_mean

        # Determine recommendation
        recommendation = self._determine_recommendation(
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            min_effect_size=experiment.min_effect_size,
            expected_direction=experiment.expected_direction,
        )

        # Update outcome in graph
        await self._psyche.update_experiment_outcome(
            experiment_uid=experiment.uid,
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            recommendation=recommendation,
        )

        # Update phase to COMPLETE
        await self._psyche.update_experiment_phase(
            experiment.uid,
            ExperimentPhase.COMPLETE,
            cycle,
        )

        # Record outcome for learning
        if self._outcome_learner and experiment.parameter_path:
            self._outcome_learner.record_experiment_outcome(
                experiment.parameter_path,
                recommendation,
            )

        logger.info(
            f"Experiment {experiment.uid}: COMPLETE. "
            f"Baseline={baseline_mean:.3f}, Treatment={treatment_mean:.3f}, "
            f"Effect={effect_size:.3f}, Recommendation={recommendation}"
        )

    async def _abort_experiment(
        self,
        experiment: Hypothesis,
        cycle: int,
    ) -> None:
        """Abort experiment due to rollback trigger.

        Args:
            experiment: The experiment to abort
            cycle: Current cycle number
        """
        # Revert to control value immediately
        await self._config_store.set(
            experiment.parameter_path,
            experiment.control_value,
        )

        await self._psyche.update_experiment_phase(
            experiment.uid,
            ExperimentPhase.ABORTED,
            cycle,
        )

        logger.warning(
            f"Experiment {experiment.uid}: ABORTED at cycle {cycle}. "
            f"Rollback triggered, reverted to control={experiment.control_value}"
        )

    async def _should_abort(
        self,
        experiment: Hypothesis,
        metrics: MetricsSnapshot,
    ) -> bool:
        """Check if rollback trigger condition is met.

        Args:
            experiment: The experiment to check
            metrics: Current metrics snapshot

        Returns:
            True if experiment should be aborted
        """
        if experiment.rollback_trigger is None:
            return False

        if experiment.baseline_mean is None:
            return False

        # Get current metric value
        current_value = metrics.get_metric(experiment.target_metric)
        baseline_mean = experiment.baseline_mean

        # Compute relative delta
        if baseline_mean != 0:
            delta = (current_value - baseline_mean) / abs(baseline_mean)
        else:
            delta = current_value - baseline_mean

        # Check against rollback trigger
        # For expected_direction="increase", rollback_trigger is negative (degradation)
        # For expected_direction="decrease", rollback_trigger is positive (improvement failure)
        if experiment.expected_direction == "increase":
            # Rollback if delta is more negative than trigger
            return delta < experiment.rollback_trigger
        else:
            # For decrease, rollback if delta is more positive than trigger
            return delta > experiment.rollback_trigger

    def _determine_recommendation(
        self,
        baseline_mean: float,
        treatment_mean: float,
        min_effect_size: float,
        expected_direction: str,
    ) -> str:
        """Determine experiment recommendation.

        Args:
            baseline_mean: Mean during baseline phase
            treatment_mean: Mean during treatment phase
            min_effect_size: Minimum effect size to consider significant
            expected_direction: "increase" or "decrease"

        Returns:
            "ADOPT", "REJECT", or "INCONCLUSIVE"
        """
        # Compute relative effect size
        if baseline_mean != 0:
            effect = (treatment_mean - baseline_mean) / abs(baseline_mean)
        else:
            effect = treatment_mean - baseline_mean

        # Check direction
        if expected_direction == "increase":
            if effect < 0:
                return "REJECT"  # Wrong direction
            elif effect >= min_effect_size:
                return "ADOPT"  # Significant improvement
            else:
                return "INCONCLUSIVE"  # Effect too small
        else:  # decrease
            if effect > 0:
                return "REJECT"  # Wrong direction
            elif abs(effect) >= min_effect_size:
                return "ADOPT"  # Significant reduction
            else:
                return "INCONCLUSIVE"  # Effect too small

    def _compute_mean(
        self,
        measurements: list[ExperimentMeasurement],
        metric_name: str,
    ) -> float:
        """Compute mean of a metric across measurements.

        Args:
            measurements: List of measurements
            metric_name: Name of metric to average

        Returns:
            Mean value, or 0.0 if no measurements
        """
        if not measurements:
            return 0.0

        values = [m.snapshot.get_metric(metric_name) for m in measurements]
        return sum(values) / len(values)
