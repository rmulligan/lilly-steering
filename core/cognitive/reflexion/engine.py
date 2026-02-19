"""ReflexionEngine for higher-level reflexion operations.

This module provides the ReflexionEngine class containing static methods
for reflexion-related functionality that doesn't fit into the specific
component classes (signals, assessment, modifications, journal).

The engine currently provides:
- propose_corrective_experiment: Maps health issues to targeted experiments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.experimentation.schemas import ExperimentProposal
    from core.cognitive.reflexion.schemas import HealthAssessment


class ReflexionEngine:
    """Static methods for reflexion operations.

    Provides high-level reflexion functionality including experiment proposals
    based on health assessments.
    """

    @staticmethod
    def propose_corrective_experiment(
        health: "HealthAssessment",
    ) -> Optional["ExperimentProposal"]:
        """Propose an experiment to address health issues.

        Maps health assessment issues to targeted experiments:
        - CRITICAL coherence -> reduce exploration magnitude (STEERING)
        - STRESSED coherence -> slightly reduce exploration (STEERING)
        - CRITICAL/STRESSED prediction -> lower simulation trigger (SIMULATION)
        - STRESSED integration -> extend episode duration (EPISODE)

        Priority order: coherence (CRITICAL) > coherence (STRESSED) >
                       prediction > integration

        Args:
            health: Current health assessment

        Returns:
            ExperimentProposal if corrective action warranted, None otherwise
        """
        from core.cognitive.experimentation.schemas import (
            ExperimentDomain,
            ExperimentProposal,
        )
        from core.cognitive.reflexion.schemas import HealthCategory

        # Priority: CRITICAL > STRESSED
        # Check coherence first (most impactful)
        if health.coherence.category == HealthCategory.CRITICAL:
            return ExperimentProposal(
                domain=ExperimentDomain.STEERING,
                parameter_path="steering.exploration.magnitude",
                treatment_value=0.7,
                rationale="Coherence critical; test reduced exploration to improve focus",
                target_metric="alignment_correlation",
                expected_direction="increase",
                rollback_trigger=-0.15,
            )

        if health.coherence.category == HealthCategory.STRESSED:
            return ExperimentProposal(
                domain=ExperimentDomain.STEERING,
                parameter_path="steering.exploration.magnitude",
                treatment_value=0.85,
                rationale="Coherence stressed; test slightly reduced exploration",
                target_metric="alignment_correlation",
                expected_direction="increase",
            )

        # Check prediction accuracy
        if health.prediction.category in (
            HealthCategory.CRITICAL,
            HealthCategory.STRESSED,
        ):
            return ExperimentProposal(
                domain=ExperimentDomain.SIMULATION,
                parameter_path="simulation.trigger_confidence",
                treatment_value=0.6,
                rationale="Prediction accuracy low; test earlier simulation trigger",
                target_metric="self_understanding",
                expected_direction="increase",
            )

        # Check integration
        if health.integration.category == HealthCategory.STRESSED:
            return ExperimentProposal(
                domain=ExperimentDomain.EPISODE,
                parameter_path="episode.max_segments",
                treatment_value=12,
                rationale="Integration stressed; test longer episodes for deeper processing",
                target_metric="alignment_correlation",
                expected_direction="increase",
            )

        return None
