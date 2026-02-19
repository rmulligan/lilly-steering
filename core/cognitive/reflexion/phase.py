"""ReflexionPhase coordinator for self-monitoring and autonomous modification.

This module provides the ReflexionPhase class that orchestrates the full
reflexion cycle: signal collection, health assessment, modification proposal
and application, and journal persistence.

The ReflexionPhase runs as Phase 5 of Lilly's cognitive loop, after Integration,
enabling autonomous self-monitoring and parameter adjustment based on health metrics.

Example usage:
    phase = ReflexionPhase(psyche=psyche, settings=settings)
    result = await phase.run(cognitive_state)
    if result.has_modifications:
        print(f"Applied {len(result.modifications)} modifications")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.cognitive.reflexion.assessment import HealthAssessor
from core.cognitive.reflexion.decision_tracker import DecisionTracker
from core.cognitive.reflexion.engine import ReflexionEngine
from core.cognitive.reflexion.journal import ReflexionJournal
from core.cognitive.reflexion.lesson_extractor import LessonExtractor
from core.cognitive.reflexion.modifications import ModificationEngine
from core.cognitive.reflexion.outcome_assessor import OutcomeAssessor
from core.cognitive.reflexion.schemas import (
    HealthAssessment,
    HealthCategory,
    HealthSignal,
    ReflexionResult,
)
from core.cognitive.reflexion.signals import HealthSignalCollector

if TYPE_CHECKING:
    from config.settings import Settings
    from core.cognitive.state import CognitiveState
    from core.cognitive.telemetry_evaluator import TelemetryEvaluator
    from core.model.curator_model import CuratorModel
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


class ReflexionPhase:
    """Orchestrates the full reflexion cycle for cognitive self-monitoring.

    The ReflexionPhase coordinates:
    1. Signal collection from Psyche and CognitiveState
    2. Health assessment with threshold-based categorization
    3. Modification proposal based on health signals
    4. Validation and application of approved modifications
    5. Narrative generation summarizing the reflexion
    6. Journal entry persistence to Psyche

    Safety features:
    - Skip condition for catastrophic failures
    - Failure streak tracking for escalation
    - Conservative mode for limited modifications
    - Error handling with graceful degradation

    Attributes:
        _psyche: Client for graph operations
        _settings: Runtime settings for modification targets
        _signal_collector: Collects health signals from various sources
        _assessor: Categorizes signals into health levels
        _modification_engine: Proposes and applies parameter changes
        _journal: Persists reflexion entries to Psyche
        _modification_failure_streak: Counter for consecutive modification failures
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        settings: "Settings" | None = None,
        window_size: int = 20,
        conservative_mode: bool = False,
        consequence_learning_enabled: bool = True,
        curator: "CuratorModel | None" = None,
        signal_collector: "HealthSignalCollector | None" = None,
        assessor: "HealthAssessor | None" = None,
        modification_engine: "ModificationEngine | None" = None,
        journal: "ReflexionJournal | None" = None,
        telemetry_evaluator: "TelemetryEvaluator | None" = None,
    ) -> None:
        """Initialize ReflexionPhase with dependencies.

        Args:
            psyche: Client for graph persistence operations
            settings: Runtime settings object for modification targets
            window_size: Number of cycles to include in rolling window (default: 20)
            conservative_mode: If True, only RUNTIME tier modifications are allowed
            consequence_learning_enabled: If True, assess decision outcomes (Phase 3)
            curator: Optional curator for lesson extraction (for testing)
            signal_collector: Optional signal collector (for testing)
            assessor: Optional health assessor (for testing)
            modification_engine: Optional modification engine (for testing)
            journal: Optional journal (for testing)
            telemetry_evaluator: Optional telemetry evaluator for biofeedback signals
        """
        self._psyche = psyche
        self._settings = settings
        self._window_size = window_size
        self._consequence_learning_enabled = consequence_learning_enabled

        # Initialize internal components (or use injected)
        self._signal_collector = signal_collector or HealthSignalCollector(
            psyche=psyche,
            window_size=window_size,
            telemetry_evaluator=telemetry_evaluator,
        )
        self._assessor = assessor or HealthAssessor()
        self._modification_engine = modification_engine or ModificationEngine(
            psyche=psyche,
            settings=settings,
            conservative_mode=conservative_mode,
        )
        self._journal = journal or ReflexionJournal(psyche=psyche)

        # Phase 3: Consequence learning components
        if self._consequence_learning_enabled:
            self._decision_tracker = DecisionTracker(psyche=psyche)
            self._outcome_assessor = OutcomeAssessor()
            self._lesson_extractor = LessonExtractor(curator=curator)

        # Failure tracking
        self._modification_failure_streak: int = 0

    async def _assess_decision_outcomes(self, state: "CognitiveState") -> None:
        """Assess outcomes of decisions from 10 cycles ago (Phase 3).

        Queries for pending decisions, compares health states, extracts
        lessons using curator, and creates zettels for significant learnings.

        Args:
            state: Current cognitive state with cycle_count
        """
        if not self._consequence_learning_enabled:
            return

        # Skip if system hasn't run long enough (cold start)
        if state.cycle_count < 10:
            logger.debug("Skipping consequence learning: cycle_count < 10")
            return

        # Get decisions ready for assessment (10 cycles old)
        pending_decisions = await self._decision_tracker.get_pending_decisions(
            current_cycle=state.cycle_count,
            offset=10,
            limit=5,  # Rate limiting
        )

        if not pending_decisions:
            logger.debug("No pending decisions to assess")
            return

        logger.info(f"Assessing {len(pending_decisions)} decision outcomes")

        # Get current health for comparison
        current_health = getattr(state, "current_health", HealthCategory.STABLE)

        for decision in pending_decisions:
            try:
                # Use health_created from decision (stored at creation time)
                health_before = decision.health_created

                # Assess outcome
                assessment = self._outcome_assessor.assess(
                    decision=decision,
                    health_before=health_before,
                    current_health=current_health,
                )

                # Extract lesson
                lesson = await self._lesson_extractor.extract(decision, assessment)

                # Create zettel if significant
                if lesson.should_create_zettel():
                    zettel_uid = await self._psyche.create_zettel_from_lesson(
                        lesson=lesson.text,
                        decision_id=decision.id,
                    )
                    logger.info(
                        f"Created zettel {zettel_uid} from decision {decision.id} "
                        f"(significance: {lesson.significance:.2f})"
                    )

                # Update decision node
                outcome_str = "success" if assessment.success else "failure"
                await self._decision_tracker.update_decision_outcome(
                    decision_id=decision.id,
                    outcome=outcome_str,
                    lesson=lesson.text,
                    success=assessment.success,
                    cycle_count_assessed=state.cycle_count,
                )

            except Exception as e:
                logger.error(f"Failed to assess decision {decision.id}: {e}")
                # Continue with other decisions (graceful degradation)
                continue

    async def run(self, state: "CognitiveState") -> ReflexionResult:
        """Execute the full reflexion cycle.

        Orchestrates:
        1. Skip check for catastrophic failure
        2. Signal collection from Psyche and state
        3. Health assessment
        4. Modification proposal and validation
        5. Modification application with failure tracking
        6. Narrative generation
        7. Journal entry persistence

        Args:
            state: Current cognitive state

        Returns:
            ReflexionResult containing assessment, modifications, and narrative
        """
        # Step 0: Skip condition for catastrophic failure
        if self._should_skip(state):
            logger.info("Skipping reflexion due to previous catastrophic failure")
            return self._skip_result()

        try:
            # Step 1: Collect signals
            snapshot = await self._signal_collector.collect_all(state)

            # Step 2: Assess health
            assessment = self._assessor.assess(snapshot)

            # Step 3: Propose modifications
            proposed = self._modification_engine.propose_modifications(assessment)

            # Step 4: Validate modifications
            validated = self._modification_engine.validate_modifications(proposed)
            approved = validated["approved"]
            skipped = validated["skipped"]

            # Step 5: Apply approved modifications
            cycle_id = f"cycle_{state.cycle_count}"
            applied, failed = await self._modification_engine.apply_all(
                approved,
                cycle_id=cycle_id,
                cycle_count=state.cycle_count,
                health_created=assessment.worst_category
            )

            # Track failure streak
            if failed:
                self._modification_failure_streak += len(failed)
                logger.warning(
                    f"Modification failures: {len(failed)}, "
                    f"streak: {self._modification_failure_streak}"
                )
            elif applied:
                # Reset streak on successful application
                self._modification_failure_streak = 0

            # Step 5a: Assess decision outcomes (Phase 3)
            await self._assess_decision_outcomes(state)

            # Step 6: Generate narrative
            narrative = self._generate_narrative(assessment, len(applied))

            # Step 7: Propose corrective experiment if health is STRESSED or CRITICAL
            experiment_proposal = None
            if assessment.worst_category in (
                HealthCategory.STRESSED,
                HealthCategory.CRITICAL,
            ):
                experiment_proposal = ReflexionEngine.propose_corrective_experiment(
                    assessment
                )
                if experiment_proposal:
                    logger.info(
                        f"Proposing corrective experiment: {experiment_proposal.domain.value} "
                        f"parameter={experiment_proposal.parameter_path}"
                    )

            # Step 8: Build result
            modifications_skipped = [
                (m.parameter_path, f"Confidence {m.confidence} below tier minimum")
                for m in skipped
            ]

            result = ReflexionResult(
                health_assessment=assessment,
                modifications=applied,
                modifications_skipped=modifications_skipped,
                narrative_summary=narrative,
                experiment_proposal=experiment_proposal,
            )

            # Step 8: Create journal entry
            cycle_number = getattr(state, "cycle_count", 0)
            metrics_snapshot = {
                "pred_accuracy": snapshot.get("prediction", {}).get("confirmation_rate", 0.0),
                "integration_rate": snapshot.get("integration", {}).get("success_rate", 1.0),
            }

            await self._journal.create_entry(
                result=result,
                cycle_number=cycle_number,
                metrics_snapshot=metrics_snapshot,
                baseline_comparison=snapshot.get("baselines"),
                phenomenological=snapshot.get("phenomenological"),
            )

            logger.info(
                f"Reflexion complete: {assessment.worst_category.value}, "
                f"{len(applied)} modifications applied"
            )

            return result

        except Exception as e:
            logger.error(f"Reflexion error: {e}", exc_info=True)
            return self._error_result(str(e))

    def _should_skip(self, state: "CognitiveState") -> bool:
        """Check if reflexion should be skipped.

        Args:
            state: Current cognitive state

        Returns:
            True if last_cycle_catastrophic is True, False otherwise
        """
        return getattr(state, "last_cycle_catastrophic", False)

    def _skip_result(self) -> ReflexionResult:
        """Create a result for skipped reflexion.

        Returns:
            ReflexionResult with empty assessment and skip narrative
        """
        return ReflexionResult(
            health_assessment=self._empty_assessment(),
            modifications=[],
            modifications_skipped=[],
            narrative_summary="Skipped due to previous catastrophic failure",
        )

    def _error_result(self, error: str) -> ReflexionResult:
        """Create a result for error condition.

        Args:
            error: Error message to include

        Returns:
            ReflexionResult with empty assessment and error narrative
        """
        return ReflexionResult(
            health_assessment=self._empty_assessment(),
            modifications=[],
            modifications_skipped=[],
            narrative_summary=f"Reflexion error: {error}",
        )

    def _empty_assessment(self) -> HealthAssessment:
        """Create an empty assessment with all STABLE signals.

        Returns:
            HealthAssessment with all signals at STABLE, value=0, baseline=0
        """
        stable_signal = HealthSignal(
            category=HealthCategory.STABLE,
            value=0.0,
            baseline=0.0,
            trend="unknown",
        )
        return HealthAssessment(
            prediction=stable_signal,
            integration=HealthSignal(
                category=HealthCategory.STABLE,
                value=0.0,
                baseline=0.0,
                trend="unknown",
            ),
            coherence=HealthSignal(
                category=HealthCategory.STABLE,
                value=0.0,
                baseline=0.0,
                trend="unknown",
            ),
        )

    def _generate_narrative(
        self,
        assessment: HealthAssessment,
        modifications_applied: int,
    ) -> str:
        """Generate human-readable narrative of reflexion outcome.

        Describes:
        - Overall system state based on worst_category
        - Specific signals that are stressed or critical
        - Actions taken (modifications applied or none needed)

        Args:
            assessment: The health assessment to describe
            modifications_applied: Number of modifications successfully applied

        Returns:
            Narrative string summarizing the reflexion
        """
        parts: list[str] = []

        # Overall state description
        worst = assessment.worst_category
        if worst == HealthCategory.THRIVING:
            parts.append("System is thriving with all metrics above baseline.")
        elif worst == HealthCategory.STABLE:
            parts.append("System is stable with metrics at baseline levels.")
        elif worst == HealthCategory.STRESSED:
            parts.append("System is experiencing stress in one or more areas.")
        elif worst == HealthCategory.CRITICAL:
            parts.append("System is in critical state requiring attention.")

        # Specific stressed/critical signals
        stressed_signals: list[str] = []
        for name, signal in [
            ("prediction", assessment.prediction),
            ("integration", assessment.integration),
            ("coherence", assessment.coherence),
        ]:
            if signal.category in (HealthCategory.STRESSED, HealthCategory.CRITICAL):
                stressed_signals.append(
                    f"{name} ({signal.category.value}: {signal.value:.2f} vs baseline {signal.baseline:.2f})"
                )

        if stressed_signals:
            parts.append(f"Affected areas: {', '.join(stressed_signals)}.")

        # Actions taken
        if modifications_applied > 0:
            parts.append(f"Applied {modifications_applied} modification(s) to address issues.")
        elif worst in (HealthCategory.STRESSED, HealthCategory.CRITICAL):
            parts.append("No modifications could be applied at this time.")
        else:
            parts.append("No modifications needed.")

        return " ".join(parts)

    @property
    def failure_streak(self) -> int:
        """Get the current modification failure streak count."""
        return self._modification_failure_streak

    def enter_conservative_mode(self) -> None:
        """Enable conservative mode on the modification engine.

        In conservative mode, only RUNTIME tier modifications are allowed.
        """
        self._modification_engine._conservative_mode = True
        logger.info("Entered conservative mode")

    def exit_conservative_mode(self) -> None:
        """Disable conservative mode on the modification engine."""
        self._modification_engine._conservative_mode = False
        logger.info("Exited conservative mode")
