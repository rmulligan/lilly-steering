"""Configuration adapter for self-experimentation.

This module provides a SettingsConfigStore that implements the ConfigStore
protocol, allowing the ExperimentManager to read and write configuration
values via dotted paths (e.g., "steering.exploration.magnitude").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


class SettingsConfigStore:
    """ConfigStore adapter for Pydantic Settings.

    Maps dotted parameter paths to Settings attributes for experiment
    parameter manipulation.

    Path Format:
        "domain.subsystem.parameter" -> settings.domain_subsystem_parameter

    Example:
        "steering.exploration.magnitude" -> settings.steering_exploration_magnitude
    """

    # Mapping from experiment parameter paths to settings attributes
    # This provides explicit, safe mapping rather than dynamic attribute access
    _PATH_TO_ATTR: dict[str, str] = {
        # Steering domain
        "steering.exploration.magnitude": "steering_exploration_magnitude",
        "steering.concept.magnitude": "steering_concept_magnitude",
        "steering.identity.magnitude": "steering_identity_magnitude",
        "steering.exploration.ema_alpha": "steering_exploration_ema_alpha",
        # Episode domain
        "episode.max_segments": "episode_max_segments",
        "episode.min_segments": "episode_min_segments",
        "episode.deep_dive_probability": "episode_deep_dive_probability",
        # Emotional field domain
        "emotional_field.decay_rate": "emotional_field_decay_rate",
        "emotional_field.diffusion_rate": "emotional_field_diffusion_rate",
        "emotional_field.blend_weight": "emotional_field_blend_weight",
        # Simulation domain
        "simulation.trigger_confidence": "simulation_trigger_confidence",
        "simulation.max_hypotheses": "simulation_max_hypotheses",
        "simulation.max_predictions_per_hypothesis": "simulation_max_predictions_per_hypothesis",
        # Tool pattern domain
        "tool_pattern.graph_exploration_weight": "tool_pattern_graph_exploration_weight",
        "tool_pattern.zettel_retrieval_weight": "tool_pattern_zettel_retrieval_weight",
        "tool_pattern.belief_query_weight": "tool_pattern_belief_query_weight",
    }

    def __init__(self, settings: "Settings") -> None:
        """Initialize the config store adapter.

        Args:
            settings: Pydantic Settings instance to wrap
        """
        self._settings = settings
        # Runtime overrides for experiment treatments
        # These take precedence over settings values during experiments
        self._overrides: dict[str, float] = {}

    async def get(self, path: str) -> float:
        """Get a configuration value by dotted path.

        Checks runtime overrides first, then falls back to settings.

        Args:
            path: Dotted parameter path (e.g., "steering.exploration.magnitude")

        Returns:
            Current value of the parameter

        Raises:
            KeyError: If path is not a valid parameter
        """
        # Check overrides first
        if path in self._overrides:
            return self._overrides[path]

        # Map to settings attribute
        if path not in self._PATH_TO_ATTR:
            raise KeyError(f"Unknown parameter path: {path}")

        attr_name = self._PATH_TO_ATTR[path]
        if not hasattr(self._settings, attr_name):
            # Return sensible default if attribute doesn't exist yet
            logger.warning(
                f"Settings attribute {attr_name} not found, returning 0.5 default"
            )
            return 0.5

        return float(getattr(self._settings, attr_name))

    async def set(self, path: str, value: float) -> None:
        """Set a configuration value by dotted path.

        Stores as runtime override rather than mutating settings directly.
        This allows safe rollback without affecting the base configuration.

        Args:
            path: Dotted parameter path
            value: New value to set

        Raises:
            KeyError: If path is not a valid parameter
        """
        if path not in self._PATH_TO_ATTR:
            raise KeyError(f"Unknown parameter path: {path}")

        self._overrides[path] = value
        logger.info(f"Set {path} = {value} (runtime override)")

    def clear_overrides(self) -> None:
        """Clear all runtime overrides.

        Useful for resetting state after experiment completion.
        """
        self._overrides.clear()
        logger.info("Cleared all experiment runtime overrides")

    def get_override(self, path: str) -> float | None:
        """Get a runtime override value if set.

        Args:
            path: Dotted parameter path

        Returns:
            Override value or None if not overridden
        """
        return self._overrides.get(path)
