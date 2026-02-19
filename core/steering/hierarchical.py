"""Hierarchical multi-layer steering.

This module provides HierarchicalSteerer, which manages steering vectors
across multiple layer zones. Each zone (exploration, concept, identity)
maintains its own vector with configurable magnitude limits and EMA update rates.

The steerer:
- Returns zone-appropriate vectors for layers within steering zones
- Returns None for observation layers (no steering during observation)
- Caps vector magnitudes to prevent degeneration
- Uses EMA blending for smooth vector updates
- Allows runtime adjustment of zone parameters for autonomous adaptation

Phase 1 Full Operational Autonomy:
Zone parameters (max_magnitude, ema_alpha) can be adjusted at runtime via
adjust_zone_parameter(). This allows Lilly to modify steering constraints based
on her assessment of current needs, removing hard-coded locks while maintaining
the protective mechanisms (magnitude capping, EMA blending).
"""
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from .config import HierarchicalSteeringConfig, SteeringZone


class HierarchicalSteerer:
    """Manages steering vectors across multiple layer zones.

    Each zone defined in the config gets its own steering vector. Vectors
    are applied during generation at layers within their zone range, and
    are capped to the zone's max_magnitude to prevent runaway amplification.

    Attributes:
        config: The hierarchical steering configuration.
        d_model: Model dimension (vector size).
        vectors: Dict mapping zone names to their steering vectors.
    """

    def __init__(self, config: HierarchicalSteeringConfig, d_model: int):
        """Initialize the steerer with zero vectors for each zone.

        Args:
            config: Configuration specifying zones and their parameters.
            d_model: Model hidden dimension (steering vector size).
        """
        self.config = config
        self.d_model = d_model

        # Initialize one vector per zone
        self.vectors: Dict[str, np.ndarray] = {
            zone.name: np.zeros(d_model, dtype=np.float32)
            for zone in config.zones
        }

        # O(1) zone lookup by name
        self._zone_by_name: Dict[str, SteeringZone] = {
            zone.name: zone for zone in config.zones
        }

    def _get_zone_by_name(self, zone_name: str) -> SteeringZone:
        """Helper to get a zone by name, raising KeyError if not found.

        Args:
            zone_name: Name of the zone to retrieve.

        Returns:
            The SteeringZone object.

        Raises:
            KeyError: If zone_name doesn't exist.
        """
        zone = self._zone_by_name.get(zone_name)
        if zone is None:
            raise KeyError(f"Unknown zone: {zone_name}")
        return zone

    def get_vector(self, layer: int) -> Optional[np.ndarray]:
        """Get steering vector for a layer, or None if no steering.

        Layers at or above the observation_layer return None to allow
        unmodified observation of activations. Layers not in any zone
        also return None.

        Args:
            layer: The transformer layer index.

        Returns:
            The steering vector for this layer (possibly magnitude-capped),
            or None if this layer should not be steered.
        """
        if layer >= self.config.observation_layer:
            return None

        zone = self.config.get_zone(layer)
        if zone is None:
            return None

        vector = self.vectors[zone.name].copy()

        # Cap magnitude to prevent degeneration
        magnitude = np.linalg.norm(vector)
        if magnitude > zone.max_magnitude:
            vector = vector * (zone.max_magnitude / magnitude)

        return vector

    def update_vector(
        self,
        zone_name: str,
        new_direction: np.ndarray,
        scale: float = 1.0,
    ) -> None:
        """Update a zone's vector using EMA blending.

        The new direction is normalized, scaled, and then blended with
        the existing vector using the zone's EMA alpha parameter.

        Args:
            zone_name: Name of the zone to update.
            new_direction: Direction vector (will be normalized).
            scale: Scale factor to apply to the normalized direction.
        """
        try:
            zone = self._get_zone_by_name(zone_name)
        except KeyError:
            return

        # Normalize new direction (skip if zero to avoid division by zero)
        norm = np.linalg.norm(new_direction)
        if norm == 0:
            return

        normalized_direction = new_direction / norm

        # Scale and blend with EMA
        current = self.vectors[zone_name]
        alpha = zone.ema_alpha
        self.vectors[zone_name] = (1 - alpha) * current + alpha * normalized_direction * scale

    def adjust_zone_parameter(
        self,
        zone_name: str,
        parameter: str,
        value: float,
    ) -> None:
        """Adjust a zone's parameter at runtime.

        Allows Lilly to modify steering constraints based on her
        assessment of current needs.

        Args:
            zone_name: Name of the zone to adjust.
            parameter: Parameter name ("max_magnitude" or "ema_alpha").
            value: New value for the parameter.

        Raises:
            KeyError: If zone_name doesn't exist.
            ValueError: If parameter name is invalid or value is out of range.
        """
        zone = self._get_zone_by_name(zone_name)

        if parameter == "max_magnitude":
            if value < 0:
                raise ValueError("max_magnitude must be non-negative")
            zone.max_magnitude = value
        elif parameter == "ema_alpha":
            if not 0 <= value <= 1:
                raise ValueError("ema_alpha must be between 0 and 1")
            zone.ema_alpha = value
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

    def get_zone_parameters(self, zone_name: str) -> dict:
        """Get current parameters for a zone.

        Args:
            zone_name: Name of the zone.

        Returns:
            Dictionary with current zone parameters.

        Raises:
            KeyError: If zone_name doesn't exist.
        """
        zone = self._get_zone_by_name(zone_name)

        return {
            "max_magnitude": zone.max_magnitude,
            "ema_alpha": zone.ema_alpha,
            "layers": zone.layers,
        }

    def adjust_zone_magnitude(self, zone_name: str, boost: float) -> None:
        """Adjust a zone's vector magnitude by a boost factor.

        Used by affect steering to modulate zone strength based on
        emotional state (e.g., high arousal/curiosity boosts exploration).

        Args:
            zone_name: Name of the zone to adjust.
            boost: Additive magnitude adjustment (typically -0.05 to +0.05).
        """
        if zone_name not in self.vectors:
            return

        vector = self.vectors[zone_name]
        current_mag = np.linalg.norm(vector)

        if current_mag < 1e-6:
            # No vector to scale
            return

        # Compute new magnitude (clamped to non-negative)
        new_mag = max(0.0, current_mag + boost)

        # Scale vector to new magnitude
        self.vectors[zone_name] = vector * (new_mag / current_mag)

        logger.debug(
            f"[AFFECT] {zone_name} magnitude adjusted: {current_mag:.3f} -> {new_mag:.3f} (boost={boost:+.3f})"
        )
