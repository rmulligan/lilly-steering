"""Process narration for reflective bridges during model loading phases.

Provides contextual narration during model load gaps, giving listeners
insight into current cognitive state rather than replaying stored memories.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from integrations.liquidsoap.client import LiquidsoapClient
    from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class BridgeContext:
    """Context available for building reflective bridges.

    Populated by the orchestrator at each model load point with
    whatever state is available at that phase.
    """

    last_concepts: list[str] = field(default_factory=list)
    emotional_intensity: float = 0.0
    emotional_valence: float = 0.0
    thought_concepts: list[str] = field(default_factory=list)
    surprise_level: float = 0.0
    hypothesis_statement: Optional[str] = None
    discoveries_count: int = 0


# Phase-specific bridge templates with variable placeholders.
# Each phase has multiple templates to avoid immediate repetition.
BRIDGE_TEMPLATES: dict[str, list[str]] = {
    "generation": [
        "Preparing to generate a new thought. The last cycle explored {last_concept}.",
        "Getting ready. The emotional field is {intensity_descriptor}.",
        "A new thought is forming. Continuing the exploration of {last_concept}.",
        "Preparing to think. The mind is {intensity_descriptor}.",
        "Readying for generation. Still processing {last_concept}.",
        "The generation phase begins. Previous thoughts touched on {last_concept}.",
        "Entering the thinking space. The field feels {intensity_descriptor}.",
        "Preparing to explore. The last idea involved {last_concept}.",
    ],
    "curation": [
        "That thought touched on {thought_concepts}. Now preparing to examine it more closely.",
        "A {surprise_descriptor} thought just formed. Getting ready to analyze its connections.",
        "The thought is complete. Preparing to curate and connect.",
        "Examining what emerged. The surprise was {surprise_descriptor}.",
        "Preparing analysis. That thought explored {thought_concepts}.",
        "The generation yielded something {surprise_descriptor}. Now to examine it.",
        "Moving into curation. The thought covered {thought_concepts}.",
        "Ready to analyze. That was a {surprise_descriptor} generation.",
    ],
    "simulation": [
        "The curator suggested testing this insight rigorously. Preparing the simulation phase.",
        "There's a hypothesis worth examining. Getting ready to test it.",
        "Entering the simulation phase. Testing: {hypothesis_brief}.",
        "Preparing rigorous examination. The hypothesis involves {hypothesis_brief}.",
        "Time to test. The curator found something worth simulating.",
        "The simulation phase begins. A hypothesis awaits testing.",
        "Getting ready to test rigorously. The question: {hypothesis_brief}.",
        "Preparing to examine closely. There's an insight worth testing.",
    ],
    "integration": [
        "Integrating what was learned. Preparing to persist insights.",
        "The analysis is complete. Now crystallizing into memory.",
        "Moving to integration. {discoveries} connections discovered.",
        "Preparing to store this cycle's learnings.",
        "Integration phase beginning. Persisting to the knowledge graph.",
        "Ready to consolidate. The cycle revealed {discoveries} connections.",
        "Crystallizing insights. {discoveries} new connections to store.",
        "The integration phase begins. Learnings will persist.",
    ],
    "continuity": [
        "Wrapping up this cycle's analysis. Preparing to synthesize what was learned.",
        "The curation revealed {discoveries} connections. Preparing to carry forward what matters.",
        "Continuity phase beginning. Synthesizing the cycle.",
        "Preparing meta-cognitive synthesis. What threads will carry forward?",
        "Wrapping up. The cycle yielded {discoveries} discoveries.",
        "Synthesizing the cycle. Preparing the continuity context.",
        "The continuity phase begins. Weaving threads for the next cycle.",
        "Ready to synthesize. What patterns will carry forward?",
    ],
}

# Fallback templates when context is unavailable
FALLBACK_TEMPLATES: dict[str, list[str]] = {
    "generation": [
        "Preparing to generate a new thought.",
        "The generation phase begins.",
        "Getting ready to think.",
    ],
    "curation": [
        "The thought is complete. Preparing to analyze.",
        "Moving into the curation phase.",
        "Getting ready to examine the thought.",
    ],
    "simulation": [
        "The simulation phase begins.",
        "Preparing to test rigorously.",
        "Getting ready for hypothesis testing.",
    ],
    "integration": [
        "Moving to the integration phase.",
        "Preparing to persist insights.",
        "Getting ready to consolidate learnings.",
    ],
    "continuity": [
        "The continuity phase begins.",
        "Preparing to synthesize the cycle.",
        "Wrapping up this cycle.",
    ],
}


class ProcessNarrator:
    """Narrates reflective bridges during model loading phases.

    Called explicitly by orchestrator before each model load. Uses
    blocking narration to ensure content is queued before GPU operations.

    The narrator maintains state to avoid immediate template repetition
    and maps numeric values to human-readable descriptors for TTS.
    """

    INTENSITY_DESCRIPTORS: dict[tuple[float, float], str] = {
        (0.0, 0.3): "calm",
        (0.3, 0.5): "present",
        (0.5, 0.7): "heightened",
        (0.7, 0.85): "intense",
        (0.85, 1.0): "very intense",
    }

    SURPRISE_DESCRIPTORS: dict[tuple[float, float], str] = {
        (0.0, 0.3): "familiar",
        (0.3, 0.5): "expected",
        (0.5, 0.7): "interesting",
        (0.7, 0.85): "surprising",
        (0.85, 1.0): "very surprising",
    }

    def __init__(
        self,
        liquidsoap: "LiquidsoapClient",
        settings: Optional["Settings"] = None,
    ):
        self._liquidsoap = liquidsoap
        self._settings = settings
        self._last_used_templates: dict[str, str] = {}  # phase -> last template key
        self._model_loading = False

    @property
    def _voice_curator(self) -> str:
        """Voice for curator/observer narration."""
        if self._settings and hasattr(self._settings, "tts_voice_curator"):
            return self._settings.tts_voice_curator
        return "eponine"

    @property
    def is_model_loading(self) -> bool:
        """Flag for SilenceMonitor to check."""
        return self._model_loading

    def _get_intensity_descriptor(self, intensity: float) -> str:
        """Map intensity value to human-readable descriptor."""
        for (low, high), descriptor in self.INTENSITY_DESCRIPTORS.items():
            if low <= intensity < high:
                return descriptor
        return "present"  # fallback

    def _get_surprise_descriptor(self, surprise: float) -> str:
        """Map surprise level to human-readable descriptor."""
        for (low, high), descriptor in self.SURPRISE_DESCRIPTORS.items():
            if low <= surprise < high:
                return descriptor
        return "expected"  # fallback

    def _select_template(self, phase: str, context: BridgeContext) -> str:
        """Select a template avoiding immediate repetition.

        Prefers templates with context placeholders that can be filled,
        but falls back to generic templates when context is unavailable.
        """
        # Check if we have enough context for rich templates
        has_concepts = bool(context.last_concepts or context.thought_concepts)
        has_hypothesis = bool(context.hypothesis_statement)
        has_discoveries = context.discoveries_count > 0

        # Use rich templates if we have relevant context
        if phase in ("generation", "curation") and has_concepts:
            templates = BRIDGE_TEMPLATES.get(phase, [])
        elif phase == "simulation" and has_hypothesis:
            templates = BRIDGE_TEMPLATES.get(phase, [])
        elif phase in ("integration", "continuity") and has_discoveries:
            templates = BRIDGE_TEMPLATES.get(phase, [])
        else:
            # Fall back to context-free templates
            templates = FALLBACK_TEMPLATES.get(phase, BRIDGE_TEMPLATES.get(phase, []))

        if not templates:
            return ""

        # Filter out last used template if we have multiple options
        last_used = self._last_used_templates.get(phase)
        available = [t for t in templates if t != last_used] if len(templates) > 1 else templates

        selected = random.choice(available)
        self._last_used_templates[phase] = selected
        return selected

    def _build_bridge(self, phase: str, context: BridgeContext) -> Optional[str]:
        """Build a reflective bridge for the given phase.

        Selects a template and fills in available context variables.
        Returns None if no suitable bridge can be constructed.
        """
        template = self._select_template(phase, context)
        if not template:
            return None

        # Prepare substitution values
        last_concept = context.last_concepts[0] if context.last_concepts else "recent ideas"
        thought_concepts = (
            ", ".join(context.thought_concepts[:3])
            if context.thought_concepts
            else "various connections"
        )
        intensity_descriptor = self._get_intensity_descriptor(context.emotional_intensity)
        surprise_descriptor = self._get_surprise_descriptor(context.surprise_level)

        # Truncate hypothesis for TTS readability
        hypothesis_brief = ""
        if context.hypothesis_statement:
            hypothesis_brief = context.hypothesis_statement[:60]
            if len(context.hypothesis_statement) > 60:
                hypothesis_brief += "..."

        discoveries = str(context.discoveries_count) if context.discoveries_count > 0 else "several"

        # Substitute placeholders
        try:
            bridge = template.format(
                last_concept=last_concept,
                thought_concepts=thought_concepts,
                intensity_descriptor=intensity_descriptor,
                surprise_descriptor=surprise_descriptor,
                hypothesis_brief=hypothesis_brief,
                discoveries=discoveries,
            )
            return bridge
        except KeyError as e:
            logger.warning(f"Template substitution failed for phase {phase}: {e}")
            # Return a fallback
            fallbacks = FALLBACK_TEMPLATES.get(phase, [])
            return random.choice(fallbacks) if fallbacks else None

    async def narrate_before_load(
        self,
        phase: str,
        context: BridgeContext,
    ) -> None:
        """Narrate reflective bridge before model load begins.

        BLOCKING: Awaits narration queue confirmation before returning.
        Caller should proceed to model.load() only after this returns.

        Args:
            phase: One of "generation", "curation", "simulation",
                   "integration", "continuity"
            context: BridgeContext with available state for this phase
        """
        self._model_loading = True
        try:
            bridge = self._build_bridge(phase, context)
            if bridge:
                logger.debug(f"ProcessNarrator: narrating {phase} bridge")
                await self._liquidsoap.narrate(bridge, voice=self._voice_curator)
            else:
                logger.debug(f"ProcessNarrator: no bridge for {phase}")
        except Exception as e:
            logger.error(f"ProcessNarrator narration failed for {phase}: {e}")
            # Don't block model loading on narration failure

    async def narrate_after_load(
        self,
        phase: str,
        success: bool,
    ) -> None:
        """Clear model loading state. Optional failure narration.

        Args:
            phase: The phase that just completed loading
            success: Whether the model load succeeded
        """
        self._model_loading = False
        if not success:
            try:
                await self._liquidsoap.narrate(
                    "Something unexpected happened. Recovering...",
                    voice=self._voice_curator,
                )
            except Exception as e:
                logger.error(f"ProcessNarrator failure narration failed: {e}")
