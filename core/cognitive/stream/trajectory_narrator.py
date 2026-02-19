"""Trajectory Narrator for PLaT-Lite Latent State Narration.

Narrates latent reasoning observations using the Expresso voice,
providing listeners with real-time insight into Lilly's cognitive
trajectory during generation.

Architecture:
    TrajectoryNarrator converts LatentObservation objects into natural
    language descriptions and queues them for TTS playback. Rate-limited
    to avoid overwhelming the listener with too many observations.

Key Insight:
    Unlike Progressive Narrator (Azelma voice, 2-sentence thought chunks),
    Trajectory Narrator (Expresso voice) narrates the latent state observations:
    - Mode shifts ("Lilly's thinking shifts from technical toward philosophical")
    - Emotional rises ("Curiosity intensifies in her reasoning")
    - Steering events ("Gently guiding toward reflection")
    - New patterns ("An insight is crystallizing")

Usage:
    from core.cognitive.stream.trajectory_narrator import TrajectoryNarrator

    narrator = TrajectoryNarrator(liquidsoap)

    for observation in observations:
        narration = await narrator.narrate_observation(observation, steering)
        # Narrates via TTS if interesting, returns text or None

Reference:
    PLaT (Planning with Latent Thoughts, arXiv:2601.21358) - verbalization
    streams independently of reasoning, providing metacognitive commentary.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.cognitive.stream.latent_observer import LatentObservation
    from core.cognitive.stream.guidance_policy import SteeringAdjustment
    from integrations.liquidsoap.client import LiquidsoapClient

logger = logging.getLogger(__name__)


class TrajectoryNarrator:
    """Narrates latent reasoning observations.

    Converts observations into natural language and plays via TTS.
    Rate-limited to provide insight without overwhelming the listener.

    Constants:
        VOICE: TTS voice to use ("expresso" - distinct from Lilly's voice)
        MIN_CHUNK_GAP: Minimum chunks between narrations (2)
        EMOTIONAL_INTENSITY_THRESHOLD: Min intensity to narrate emotion (0.4)
    """

    VOICE = "expresso"
    MIN_CHUNK_GAP = 2
    EMOTIONAL_INTENSITY_THRESHOLD = 0.4

    # Templates for different observation types
    TEMPLATES = {
        "mode_shift": "Lilly's thinking shifts from {from_mode} toward {to_mode}.",
        "new_pattern": "A new pattern emerges: {pattern}.",
        "emotional_rise": "{emotion} intensifies in her reasoning.",
        "steering": "Gently guiding toward {target_mode}.",
        "uncertainty": "Uncertainty surfaces... she pauses to consider.",
        "insight": "Something crystallizes—an insight forming.",
        "reflection_invite": "A moment of reflection opens.",
        "action_invite": "Grounding returns, inviting concrete thought.",
        "mode_deepening": "Deepening into {mode}...",
    }

    def __init__(
        self,
        liquidsoap: Optional["LiquidsoapClient"] = None,
        enabled: bool = True,
        min_chunk_gap: Optional[int] = None,
    ):
        """Initialize the trajectory narrator.

        Args:
            liquidsoap: LiquidsoapClient for TTS (optional for testing)
            enabled: Whether trajectory narration is enabled (default True)
            min_chunk_gap: Override MIN_CHUNK_GAP constant (optional)
        """
        self._liquidsoap = liquidsoap
        self._enabled = enabled
        self._min_chunk_gap = min_chunk_gap if min_chunk_gap is not None else self.MIN_CHUNK_GAP
        self._last_narration_chunk: int = -self._min_chunk_gap  # Allow first narration
        self._narration_count: int = 0

    def reset(self) -> None:
        """Reset for new generation."""
        self._last_narration_chunk = -self._min_chunk_gap
        self._narration_count = 0

    async def narrate_observation(
        self,
        observation: "LatentObservation",
        steering: Optional["SteeringAdjustment"] = None,
    ) -> Optional[str]:
        """Convert observation to natural language and speak.

        Rate-limited to prevent overwhelming the listener.

        Args:
            observation: Latent observation to narrate
            steering: Optional steering adjustment that was applied

        Returns:
            The narration text if spoken, None if rate-limited, disabled, or nothing to say
        """
        # Check if enabled
        if not self._enabled:
            return None

        # Rate limit
        if observation.chunk_idx - self._last_narration_chunk < self._min_chunk_gap:
            return None

        # Build narration
        narration = self._build_narration(observation, steering)

        if narration:
            # Speak via TTS
            if self._liquidsoap:
                try:
                    await self._liquidsoap.narrate(narration, voice=self.VOICE)
                except Exception as e:
                    logger.warning(f"[TRAJECTORY_NARRATOR] TTS failed: {e}")

            self._last_narration_chunk = observation.chunk_idx
            self._narration_count += 1

            logger.debug(
                f"[TRAJECTORY_NARRATOR] Chunk {observation.chunk_idx}: {narration}"
            )

        return narration

    def _build_narration(
        self,
        obs: "LatentObservation",
        steering: Optional["SteeringAdjustment"],
    ) -> Optional[str]:
        """Build narration text from observation.

        Prioritizes different observation aspects:
        1. Mode shifts (most significant)
        2. Steering events (intentional guidance)
        3. Emotional intensity changes
        4. New interesting patterns

        Args:
            obs: Latent observation
            steering: Optional steering adjustment

        Returns:
            Narration text, or None if nothing noteworthy
        """
        parts = []

        # 1. Mode shift (from trajectory_delta)
        if obs.trajectory_delta and "shifted from" in obs.trajectory_delta:
            match = re.match(
                r"shifted from ([\w_]+) to ([\w_]+)",
                obs.trajectory_delta
            )
            if match:
                parts.append(self.TEMPLATES["mode_shift"].format(
                    from_mode=self._humanize(match.group(1)),
                    to_mode=self._humanize(match.group(2)),
                ))

        # 2. Steering events
        if steering:
            narration = self._narrate_steering(steering)
            if narration:
                parts.append(narration)

        # 3. Emotional intensity
        if obs.emotional_state.intensity > self.EMOTIONAL_INTENSITY_THRESHOLD:
            dominant = obs.emotional_state.dominant
            if dominant != "neutral":
                # Only narrate if not already covered by mode shift
                if not parts or dominant.lower() not in parts[0].lower():
                    parts.append(self.TEMPLATES["emotional_rise"].format(
                        emotion=dominant.capitalize(),
                    ))

        # 4. New patterns (interesting only)
        if obs.trajectory_delta and "new pattern:" in obs.trajectory_delta:
            pattern = obs.trajectory_delta.partition("new pattern: ")[2]
            if self._is_interesting(pattern):
                # Use specific template for known patterns
                if "insight" in pattern.lower():
                    parts.append(self.TEMPLATES["insight"])
                else:
                    parts.append(self.TEMPLATES["new_pattern"].format(
                        pattern=pattern
                    ))

        return " ".join(parts) if parts else None

    def _narrate_steering(
        self,
        steering: "SteeringAdjustment"
    ) -> Optional[str]:
        """Build narration for steering adjustment.

        Args:
            steering: The steering adjustment that was applied

        Returns:
            Narration text for the steering
        """
        reason = steering.reason.lower()

        if "reflection" in reason:
            return self.TEMPLATES["reflection_invite"]
        elif "action" in reason or "grounding" in reason:
            return self.TEMPLATES["action_invite"]
        elif "redirecting" in reason:
            # Extract target mode
            if " to " in steering.reason:
                target = steering.reason.split(" to ")[-1]
                return self.TEMPLATES["steering"].format(
                    target_mode=self._humanize(target),
                )
        elif "drifting" in reason:
            # Extract target mode
            if " to " in steering.reason:
                target = steering.reason.split(" to ")[-1]
                return self.TEMPLATES["steering"].format(
                    target_mode=self._humanize(target),
                )

        return None

    def _humanize(self, mode: str) -> str:
        """Convert mode name to human-readable form.

        Args:
            mode: Mode name with underscores (e.g., "philosophical_inquiry")

        Returns:
            Human-readable form (e.g., "philosophical inquiry")
        """
        return mode.replace("_", " ")

    def _is_interesting(self, pattern: str) -> bool:
        """Check if pattern is worth narrating.

        Args:
            pattern: Feature interpretation pattern

        Returns:
            True if pattern should be narrated
        """
        INTERESTING_KEYWORDS = [
            "paradox", "insight", "novel", "self", "memory",
            "question", "realize", "connect", "understand",
            "reflect", "wonder", "discover", "emerge"
        ]
        pattern_lower = pattern.lower()
        return any(kw in pattern_lower for kw in INTERESTING_KEYWORDS)

    def get_stats(self) -> dict:
        """Get narration statistics.

        Returns:
            Dict with narration count and last chunk
        """
        return {
            "narration_count": self._narration_count,
            "last_narration_chunk": self._last_narration_chunk,
        }
