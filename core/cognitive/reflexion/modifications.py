"""ModificationEngine for proposing and applying autonomous modifications.

This module implements the core modification logic for the Reflexion phase,
enabling Lilly to propose and apply parameter changes based on health assessments.

PATTERN CHANGE (Phase 1 Full Operational Autonomy):
Previous: propose → request approval → wait → apply if approved
Current: synthesize knowledge → judge → apply → observe → learn

Lilly now applies reflexion modifications directly based on her judgment
of appropriateness. No external validation gate. Outcomes are observed
and lessons learned are recorded as AutonomousDecision nodes.

The engine supports three tiers of modifications:
- RUNTIME: Transient parameter adjustments (lowest risk, min confidence 0.5)
- CONFIG: Session-persistent configuration changes (moderate risk, min confidence 0.6)
- PROMPT: Self-prompt modifications (highest impact, min confidence 0.8)

Example usage:
    engine = ModificationEngine(psyche=psyche, settings=settings)
    assessment = assessor.assess(snapshot)
    modifications = engine.propose_modifications(assessment)
    validated = engine.validate_modifications(modifications)
    applied, failed = await engine.apply_all(validated["approved"], cycle_id=cycle_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.cognitive.reflexion.schemas import (
    HealthAssessment,
    HealthCategory,
    Modification,
    ModificationTier,
)

if TYPE_CHECKING:
    from config.settings import Settings
    from core.psyche.client import PsycheClient

logger = logging.getLogger(__name__)


# Minimum values for parameter adjustments
MIN_CONFIDENCE_THRESHOLD = 0.4  # Absolute minimum for simulation confidence
MIN_CONFIDENCE_THRESHOLD_STRESSED = 0.5  # Minimum when STRESSED (not CRITICAL)


class ModificationEngine:
    """Proposes and applies tiered modifications based on health assessments.

    The engine maintains a history of modifications and enforces confidence
    thresholds based on modification tier. In conservative mode, only RUNTIME
    tier modifications are allowed.

    Attributes:
        psyche: PsycheClient for persisting config/prompt changes
        conservative_mode: If True, only RUNTIME tier modifications allowed
    """

    def __init__(
        self,
        psyche: "PsycheClient",
        settings: "Settings",
        conservative_mode: bool = False,
    ) -> None:
        """Initialize the ModificationEngine.

        Args:
            psyche: PsycheClient for graph persistence
            settings: Runtime settings object for Tier 1 modifications
            conservative_mode: If True, only RUNTIME tier allowed
        """
        self._psyche = psyche
        self._settings = settings
        self._conservative_mode = conservative_mode
        self._modification_history: list[Modification] = []

    def propose_modifications(
        self, assessment: HealthAssessment
    ) -> list[Modification]:
        """Propose modifications based on health assessment.

        Examines prediction, integration, and coherence signals and proposes
        appropriate fixes for STRESSED or CRITICAL categories.

        Args:
            assessment: The health assessment to analyze

        Returns:
            List of proposed modifications (may be empty if all healthy)
        """
        modifications: list[Modification] = []

        # Check prediction health
        if assessment.prediction.category in (
            HealthCategory.STRESSED,
            HealthCategory.CRITICAL,
        ):
            modifications.extend(self._propose_prediction_fixes(assessment))

        # Check integration health
        if assessment.integration.category in (
            HealthCategory.STRESSED,
            HealthCategory.CRITICAL,
        ):
            modifications.extend(self._propose_integration_fixes(assessment))

        # Check coherence health (only for CRITICAL)
        if assessment.coherence.category == HealthCategory.CRITICAL:
            modifications.extend(self._propose_coherence_fixes(assessment))

        return modifications

    def _propose_prediction_fixes(
        self, assessment: HealthAssessment
    ) -> list[Modification]:
        """Propose fixes for prediction health issues.

        For STRESSED: Lower confidence threshold by 0.1 (min 0.5)
        For CRITICAL: Lower by 0.2 (min 0.4) AND extend verification window by 5

        Args:
            assessment: The health assessment with prediction issues

        Returns:
            List of modifications to improve prediction health
        """
        modifications: list[Modification] = []
        is_critical = assessment.prediction.category == HealthCategory.CRITICAL

        # Get current threshold
        current_threshold = getattr(
            self._settings, "simulation_confidence_threshold", 0.7
        )

        # Determine reduction and minimum
        if is_critical:
            reduction = 0.2
            min_threshold = MIN_CONFIDENCE_THRESHOLD
        else:
            reduction = 0.1
            min_threshold = MIN_CONFIDENCE_THRESHOLD_STRESSED

        new_threshold = max(current_threshold - reduction, min_threshold)

        # Only propose if there's a change to make
        if new_threshold < current_threshold:
            modifications.append(
                Modification(
                    tier=ModificationTier.RUNTIME,
                    parameter_path="simulation.confidence_threshold",
                    old_value=current_threshold,
                    new_value=new_threshold,
                    rationale=(
                        f"Prediction health is {assessment.prediction.category.value}. "
                        f"Lowering confidence threshold from {current_threshold} to "
                        f"{new_threshold} to improve hypothesis acceptance rate."
                    ),
                    confidence=0.7,
                    revert_condition=(
                        "Revert when prediction health returns to STABLE or better "
                        "for 3 consecutive cycles."
                    ),
                )
            )

        # For CRITICAL, also extend verification window
        if is_critical:
            current_window = getattr(
                self._settings, "prediction_default_window_cycles", 20
            )
            new_window = current_window + 5

            modifications.append(
                Modification(
                    tier=ModificationTier.RUNTIME,
                    parameter_path="prediction.verification_window",
                    old_value=current_window,
                    new_value=new_window,
                    rationale=(
                        f"Prediction health is CRITICAL. Extending verification "
                        f"window from {current_window} to {new_window} cycles to "
                        "allow more time for predictions to be confirmed."
                    ),
                    confidence=0.7,
                    revert_condition=(
                        "Revert when prediction health returns to STABLE or better "
                        "for 3 consecutive cycles."
                    ),
                )
            )

        return modifications

    def _propose_integration_fixes(
        self, assessment: HealthAssessment
    ) -> list[Modification]:
        """Propose fixes for integration health issues.

        Currently returns empty list as integration issues need investigation.
        Future implementations may propose zettel retention threshold changes
        or knowledge graph pruning strategies.

        Args:
            assessment: The health assessment with integration issues

        Returns:
            Empty list (integration fixes require deeper investigation)
        """
        # Integration issues typically require human investigation
        # Future: Could propose zettel threshold changes, graph pruning, etc.
        return []

    def _propose_coherence_fixes(
        self, assessment: HealthAssessment
    ) -> list[Modification]:
        """Propose fixes for coherence health issues.

        For CRITICAL coherence (value < 0.5), increase exploration_magnitude_cap
        from 0.5 to 0.7 to encourage more diverse thinking patterns.

        Args:
            assessment: The health assessment with coherence issues

        Returns:
            List of modifications to improve coherence
        """
        modifications: list[Modification] = []

        # Only fix if coherence value is significantly low
        if assessment.coherence.value < 0.5:
            # Try to get current value from settings
            current_cap = getattr(self._settings, "exploration_magnitude_cap", None)
            if current_cap is None:
                current_cap = getattr(
                    getattr(self._settings, "steering", None),
                    "exploration_magnitude_cap",
                    0.5,
                )

            new_cap = 0.7

            if new_cap > current_cap:
                modifications.append(
                    Modification(
                        tier=ModificationTier.RUNTIME,
                        parameter_path="steering.exploration_magnitude_cap",
                        old_value=current_cap,
                        new_value=new_cap,
                        rationale=(
                            f"Coherence health is CRITICAL (value={assessment.coherence.value:.2f}). "
                            f"Increasing exploration magnitude cap from {current_cap} to "
                            f"{new_cap} to encourage more diverse thinking patterns."
                        ),
                        confidence=0.65,
                        revert_condition=(
                            "Revert when coherence health returns to STABLE or better "
                            "for 3 consecutive cycles."
                        ),
                    )
                )

        return modifications

    def validate_modifications(
        self, modifications: list[Modification]
    ) -> dict[str, list[Modification]]:
        """Validate modifications against confidence thresholds and mode.

        Checks each modification against:
        1. Confidence meets tier requirement via mod.validate_confidence()
        2. Conservative mode (rejects non-RUNTIME if conservative)

        Args:
            modifications: List of proposed modifications

        Returns:
            Dict with "approved" and "skipped" lists
        """
        approved: list[Modification] = []
        skipped: list[Modification] = []

        for mod in modifications:
            # Check confidence meets tier requirement
            if not mod.validate_confidence():
                logger.info(
                    f"Skipping modification {mod.parameter_path}: "
                    f"confidence {mod.confidence} below tier minimum "
                    f"{mod.tier.min_confidence}"
                )
                skipped.append(mod)
                continue

            # Check conservative mode
            if self._conservative_mode and mod.tier != ModificationTier.RUNTIME:
                logger.info(
                    f"Skipping modification {mod.parameter_path}: "
                    f"tier {mod.tier.value} not allowed in conservative mode"
                )
                skipped.append(mod)
                continue

            approved.append(mod)

        return {"approved": approved, "skipped": skipped}

    async def apply_modification(
        self, mod: Modification, cycle_id: str | None = None, cycle_count: int | None = None, health_created: "HealthCategory | None" = None
    ) -> bool:
        """Apply a single modification and log autonomous decision.

        Phase 1 Full Operational Autonomy: Modifications are applied immediately
        based on autonomous judgment. Each modification is logged as an
        AutonomousDecision for observability and learning.

        Phase 2 Knowledge Synthesis: Before applying, queries relevant past
        decisions, zettels, and experiments to inform the decision log.

        Phase 3 Consequence Learning: Captures cycle_count and health at decision
        time for robust outcome assessment.

        Args:
            mod: The modification to apply
            cycle_id: Optional cycle ID for decision logging
            cycle_count: Optional cycle number for decision logging
            health_created: Optional health category when decision was made

        Returns:
            True if successful, False otherwise
        """
        # Set applied timestamp
        mod.applied_at = datetime.now(timezone.utc)

        # Synthesize knowledge before applying (Phase 2)
        knowledge: list[str] = []
        if cycle_id:
            try:
                knowledge = await self._psyche.synthesize_knowledge_for_decision(
                    question=f"Should I modify {mod.parameter_path}?",
                    domain="reflexion",
                )
            except Exception as e:
                logger.warning(f"Knowledge synthesis failed, continuing with modification: {e}")
                knowledge = []

        # Route by tier
        try:
            if mod.tier == ModificationTier.RUNTIME:
                result = self._apply_runtime(mod)
            elif mod.tier == ModificationTier.CONFIG:
                result = await self._apply_config(mod)
            elif mod.tier == ModificationTier.PROMPT:
                result = await self._apply_prompt(mod)
            else:
                logger.error(f"Unknown modification tier: {mod.tier}")
                return False

            if result:
                self._modification_history.append(mod)
                logger.info(
                    f"Applied {mod.tier.value} modification to {mod.parameter_path}"
                )

                # Log autonomous decision to Psyche (Phase 3)
                if cycle_id and cycle_count is not None and health_created is not None:
                    await self._log_autonomous_decision(mod, cycle_id, cycle_count, health_created, knowledge)
            else:
                mod.applied_at = None  # Clear timestamp on failure

            return result

        except Exception as e:
            logger.error(f"Failed to apply modification {mod.parameter_path}: {e}")
            mod.applied_at = None
            return False

    def _apply_runtime(self, mod: Modification) -> bool:
        """Apply a RUNTIME tier modification to settings.

        Handles known parameters:
        - simulation.confidence_threshold
        - prediction.verification_window
        - steering.exploration_magnitude_cap

        Args:
            mod: The RUNTIME modification to apply

        Returns:
            True if successful, False if parameter unknown
        """
        param_path = mod.parameter_path

        if param_path == "simulation.confidence_threshold":
            self._settings.simulation_confidence_threshold = mod.new_value
            return True

        elif param_path == "prediction.verification_window":
            self._settings.prediction_default_window_cycles = mod.new_value
            return True

        elif param_path == "steering.exploration_magnitude_cap":
            # Try direct attribute first
            if hasattr(self._settings, "exploration_magnitude_cap"):
                self._settings.exploration_magnitude_cap = mod.new_value
                return True
            # Try nested steering object
            elif hasattr(self._settings, "steering") and hasattr(
                self._settings.steering, "exploration_magnitude_cap"
            ):
                self._settings.steering.exploration_magnitude_cap = mod.new_value
                return True
            else:
                logger.warning(
                    "exploration_magnitude_cap not found in settings, "
                    "creating direct attribute"
                )
                self._settings.exploration_magnitude_cap = mod.new_value
                return True

        else:
            logger.warning(f"Unknown RUNTIME parameter path: {param_path}")
            return False

    async def _apply_config(self, mod: Modification) -> bool:
        """Apply a CONFIG tier modification to Psyche.

        Persists the modification as a CognitiveParameter node using MERGE.

        Args:
            mod: The CONFIG modification to apply

        Returns:
            True if persisted successfully (execute returned > 0)
        """
        cypher = """
        MERGE (p:CognitiveParameter {path: $path})
        SET p.value = $value,
            p.old_value = $old_value,
            p.modified_at = $modified_at,
            p.rationale = $rationale,
            p.revert_condition = $revert_condition
        RETURN p
        """
        params = {
            "path": mod.parameter_path,
            "value": str(mod.new_value),  # Store as string for flexibility
            "old_value": str(mod.old_value),
            "modified_at": mod.applied_at.isoformat() if mod.applied_at else None,
            "rationale": mod.rationale,
            "revert_condition": mod.revert_condition,
        }

        result = await self._psyche.execute(cypher, params)
        return result > 0

    async def _apply_prompt(self, mod: Modification) -> bool:
        """Apply a PROMPT tier modification to Psyche.

        Persists the modification as a PromptComponent node using MERGE.

        Args:
            mod: The PROMPT modification to apply

        Returns:
            True if persisted successfully (execute returned > 0)
        """
        cypher = """
        MERGE (p:PromptComponent {path: $path})
        SET p.content = $content,
            p.old_content = $old_content,
            p.modified_at = $modified_at,
            p.rationale = $rationale,
            p.revert_condition = $revert_condition
        RETURN p
        """
        params = {
            "path": mod.parameter_path,
            "content": str(mod.new_value),
            "old_content": str(mod.old_value),
            "modified_at": mod.applied_at.isoformat() if mod.applied_at else None,
            "rationale": mod.rationale,
            "revert_condition": mod.revert_condition,
        }

        result = await self._psyche.execute(cypher, params)
        return result > 0

    async def _log_autonomous_decision(
        self, mod: Modification, cycle_id: str, cycle_count: int, health_created: "HealthCategory", knowledge: list[str]
    ) -> None:
        """Log an autonomous decision to Psyche.

        Records the judgment, action, and expectation for this modification
        as an AutonomousDecision node for observability and learning.

        Phase 2: Includes synthesized knowledge from past decisions, zettels,
        and experiments to document the decision context.

        Phase 3: Captures cycle_count_created and health_created for robust
        outcome assessment.

        Args:
            mod: The applied modification
            cycle_id: The current cognitive cycle ID
            cycle_count: The current cycle number
            health_created: Health category when decision was made
            knowledge: Synthesized knowledge items informing this decision
        """
        from core.psyche.schema import AutonomousDecision

        # Rebuild model to resolve forward reference to HealthCategory, but only once.
        if not getattr(AutonomousDecision, '_health_category_rebuilt', False):
            AutonomousDecision.model_rebuild(_types_namespace={"HealthCategory": HealthCategory})
            AutonomousDecision._health_category_rebuilt = True

        decision = AutonomousDecision(
            timestamp=mod.applied_at or datetime.now(timezone.utc),
            cycle_id=cycle_id,
            cycle_count_created=cycle_count,
            health_created=health_created,
            question=f"Should I modify {mod.parameter_path}?",
            knowledge_synthesized=knowledge,
            judgment=mod.rationale,
            action={
                "type": "apply_modification",
                "tier": mod.tier.value,
                "parameter_path": mod.parameter_path,
                "old_value": mod.old_value,
                "value": mod.new_value,
            },
            expectation=mod.revert_condition,
        )

        try:
            await self._psyche.record_autonomous_decision(decision)
            logger.info(
                f"Logged autonomous decision for {mod.parameter_path} "
                f"with {len(knowledge)} knowledge items",
                extra={"decision_id": decision.id}
            )
        except Exception as e:
            logger.error(f"Failed to log autonomous decision: {e}")
            # Don't fail the modification if logging fails

    async def apply_all(
        self, modifications: list[Modification], cycle_id: str | None = None, cycle_count: int | None = None, health_created: "HealthCategory | None" = None
    ) -> tuple[list[Modification], list[Modification]]:
        """Apply all modifications and return results.

        Args:
            modifications: List of modifications to apply
            cycle_id: Optional cycle ID for decision logging
            cycle_count: Optional cycle number for decision logging (Phase 3)
            health_created: Optional health category when decisions were made (Phase 3)

        Returns:
            Tuple of (applied, failed) modification lists
        """
        applied: list[Modification] = []
        failed: list[Modification] = []

        for mod in modifications:
            success = await self.apply_modification(mod, cycle_id=cycle_id, cycle_count=cycle_count, health_created=health_created)
            if success:
                applied.append(mod)
            else:
                failed.append(mod)

        return applied, failed

    @property
    def modification_history(self) -> list[Modification]:
        """Get the history of applied modifications."""
        return self._modification_history.copy()
