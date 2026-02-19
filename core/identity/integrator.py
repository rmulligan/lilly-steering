"""Identity Integrator - Unified steering from the Integrated Identity Layer.

Blends three signal sources into a single identity steering vector:
1. Affective Resonance - moment-to-moment felt biases from experience
2. Semantic Intuitions - consolidated knowledge as steering vectors
3. Autobiographical Self - continuous self-presence

This is the main interface for the identity layer. Every generation passes
through this integrator to receive unified identity steering.

Key Principle: Identity is not retrieved—it is *always present* in the
activation space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

if TYPE_CHECKING:
    from core.psyche.client import PsycheClient
    from core.self_model.affective_system import AffectiveState
    from core.identity.affective_resonator import AffectiveResonator
    from core.identity.intuition_bank import SemanticIntuitionBank
    from core.identity.autobiographical_self import AutobiographicalSelf

logger = logging.getLogger(__name__)


class IdentityComputationError(Exception):
    """Raised when identity steering computation fails.

    This exception indicates a failure in the identity integration process
    that should be handled gracefully by the caller. It distinguishes
    expected identity-specific failures from unexpected programming errors.

    Examples of causes:
    - Affective resonance computation failure
    - Intuition bank query failure
    - Autobiographical self vector unavailable
    - Tensor dimension mismatches during blending
    """

    pass


@dataclass
class IntegrationWeights:
    """Weights for blending identity signals.

    These weights determine how much each signal contributes to the
    final identity steering vector. They can be tuned based on context
    or learned over time.

    Attributes:
        affect: Weight for affective resonance (moment-to-moment)
        intuition: Weight for semantic intuitions (consolidated knowledge)
        self_presence: Weight for autobiographical self (continuous identity)
    """

    affect: float = 0.3
    intuition: float = 0.4
    self_presence: float = 0.3

    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = self.affect + self.intuition + self.self_presence
        if total > 0:
            self.affect /= total
            self.intuition /= total
            self.self_presence /= total


@dataclass
class IntegrationConfig:
    """Configuration for the Identity Integrator.

    Attributes:
        weights: Blending weights for the three signals
        target_layer: Default layer for steering application
        layer_range: Range of layers to apply steering
        min_vector_norm: Minimum norm to apply steering (avoid noise)
        max_coefficient: Maximum steering coefficient
    """

    weights: IntegrationWeights = field(default_factory=IntegrationWeights)
    target_layer: int = 16
    layer_range: tuple[int, int] = field(default_factory=lambda: (12, 20))
    min_vector_norm: float = 0.01
    max_coefficient: float = 1.5


@dataclass
class SteeringIntervention:
    """A steering intervention to apply during generation.

    This is the output of the integrator, ready to be passed to
    HookedQwen for application during generation.

    Attributes:
        vector: The combined identity steering vector
        layer: Target layer for application
        layer_range: Range of layers to apply (for multi-layer steering)
        coefficient: Scaling coefficient for the vector
        sources: Which sources contributed to this intervention
    """

    vector: "torch.Tensor"
    layer: int
    layer_range: tuple[int, int]
    coefficient: float
    sources: dict[str, float] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if this intervention has meaningful content."""
        if not TORCH_AVAILABLE:
            return False
        return self.vector.norm().item() > 0.01

    def to_dict(self) -> dict:
        """Serialize for logging (excludes vector)."""
        return {
            "layer": self.layer,
            "layer_range": list(self.layer_range),
            "coefficient": self.coefficient,
            "vector_norm": self.vector.norm().item() if TORCH_AVAILABLE else 0.0,
            "sources": self.sources,
            "is_active": self.is_active(),
        }


@dataclass
class IntegrationResult:
    """Result of computing identity integration.

    Attributes:
        intervention: The steering intervention to apply
        affect_contribution: How much affect contributed
        intuition_contribution: How much intuitions contributed
        self_contribution: How much autobiographical self contributed
        active_intuitions: Which intuitions were activated
        computation_time_ms: How long integration took
    """

    intervention: SteeringIntervention
    affect_contribution: float = 0.0
    intuition_contribution: float = 0.0
    self_contribution: float = 0.0
    active_intuitions: list[str] = field(default_factory=list)
    computation_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "intervention": self.intervention.to_dict(),
            "affect_contribution": self.affect_contribution,
            "intuition_contribution": self.intuition_contribution,
            "self_contribution": self.self_contribution,
            "active_intuitions": self.active_intuitions,
            "computation_time_ms": self.computation_time_ms,
        }


class IdentityIntegrator:
    """Blends identity signals into unified steering at inference time.

    The integrator is the main interface to the Integrated Identity Layer.
    Before each generation, call `compute_identity_steering` to get a
    steering intervention that encodes the current identity state.

    The three signals are:
    1. **Affective Resonance**: Queries psyche for similar past experiences
       and computes a resonance vector based on accumulated valence.
       This creates "felt" biases—reluctance or enthusiasm based on past.

    2. **Semantic Intuitions**: Checks which consolidated intuitions apply
       to the current context. These are preferences that have been "baked
       in" during dream consolidation.

    3. **Autobiographical Self**: The stable presence vector that provides
       continuous identity grounding. Always active, unlike the contextual
       signals above.

    Usage:
        integrator = IdentityIntegrator(
            resonator=affective_resonator,
            intuition_bank=intuition_bank,
            autobiographical_self=autobio_self,
        )

        # Before each generation
        result = await integrator.compute_identity_steering(
            context=prompt,
            psyche=psyche_client,
            affective_state=current_affect,
        )

        # Apply to generation
        model.generate(prompt, steering=result.intervention)

    Attributes:
        config: Integration configuration
        resonator: Affective resonator for experience-based biases
        intuition_bank: Bank of consolidated intuitions
        autobiographical_self: Source of continuous presence
    """

    def __init__(
        self,
        resonator: Optional["AffectiveResonator"] = None,
        intuition_bank: Optional["SemanticIntuitionBank"] = None,
        autobiographical_self: Optional["AutobiographicalSelf"] = None,
        config: Optional[IntegrationConfig] = None,
    ):
        """Initialize the Identity Integrator.

        Args:
            resonator: Affective resonator for moment-to-moment biases
            intuition_bank: Bank of consolidated intuitions
            autobiographical_self: Source of continuous presence
            config: Integration configuration
        """
        self.config = config or IntegrationConfig()
        self._resonator = resonator
        self._intuition_bank = intuition_bank
        self._autobiographical_self = autobiographical_self

        # Track model dimension (will be set from first real vector)
        self._model_dim: int = 3584  # Default Qwen dimension

    def set_resonator(self, resonator: "AffectiveResonator") -> None:
        """Set the affective resonator."""
        self._resonator = resonator

    def set_intuition_bank(self, bank: "SemanticIntuitionBank") -> None:
        """Set the semantic intuition bank."""
        self._intuition_bank = bank

    def set_autobiographical_self(self, autobio: "AutobiographicalSelf") -> None:
        """Set the autobiographical self."""
        self._autobiographical_self = autobio

    async def compute_identity_steering(
        self,
        context: str,
        psyche: Optional["PsycheClient"] = None,
        affective_state: Optional["AffectiveState"] = None,
    ) -> IntegrationResult:
        """Compute unified identity steering for the given context.

        This is the main entry point. Blends all three identity signals
        into a single steering intervention.

        Args:
            context: Current generation context (prompt/situation)
            psyche: Client for querying past experiences (for affect)
            affective_state: Current affective state (for affect)

        Returns:
            IntegrationResult with steering intervention and metadata

        Raises:
            IdentityComputationError: If identity computation fails for any reason
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for identity steering")

        try:
            start = datetime.now(timezone.utc)
            sources: dict[str, float] = {}

            # 1. Compute affective resonance (moment-to-moment)
            affect_vector = await self._compute_affect_vector(
                context, psyche, affective_state
            )

            # 2. Compute intuition blending (consolidated knowledge)
            intuition_vector, active_keys = await self._compute_intuition_vector(context)

            # 3. Get autobiographical presence (continuous self)
            self_vector = self._get_self_vector()

            # Determine model dimension from first non-zero vector
            for vec in [affect_vector, intuition_vector, self_vector]:
                if vec.norm().item() > 0:
                    self._model_dim = vec.shape[0]
                    break

            # Ensure all vectors have consistent dimension
            affect_vector = self._ensure_dim(affect_vector)
            intuition_vector = self._ensure_dim(intuition_vector)
            self_vector = self._ensure_dim(self_vector)

            # Compute norms for tracking
            affect_norm = affect_vector.norm().item()
            intuition_norm = intuition_vector.norm().item()
            self_norm = self_vector.norm().item()

            sources["affect"] = affect_norm
            sources["intuition"] = intuition_norm
            sources["self"] = self_norm

            # Blend with weights
            weights = self.config.weights
            identity_vector = (
                weights.affect * affect_vector +
                weights.intuition * intuition_vector +
                weights.self_presence * self_vector
            )

            # Normalize if non-zero
            final_norm = identity_vector.norm()
            if final_norm > 1e-8:
                identity_vector = identity_vector / final_norm

            # Compute coefficient based on signal strength
            coefficient = self._compute_coefficient(
                affect_norm, intuition_norm, self_norm
            )

            # Create intervention
            intervention = SteeringIntervention(
                vector=identity_vector,
                layer=self.config.target_layer,
                layer_range=self.config.layer_range,
                coefficient=coefficient,
                sources=sources,
            )

            # Compute contributions for logging
            total_contribution = affect_norm + intuition_norm + self_norm
            if total_contribution > 0:
                affect_contribution = affect_norm / total_contribution
                intuition_contribution = intuition_norm / total_contribution
                self_contribution = self_norm / total_contribution
            else:
                affect_contribution = intuition_contribution = self_contribution = 0.0

            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            return IntegrationResult(
                intervention=intervention,
                affect_contribution=affect_contribution,
                intuition_contribution=intuition_contribution,
                self_contribution=self_contribution,
                active_intuitions=active_keys,
                computation_time_ms=elapsed,
            )
        except Exception as e:
            raise IdentityComputationError(
                f"Failed to compute identity steering: {e}"
            ) from e

    async def _compute_affect_vector(
        self,
        context: str,
        psyche: Optional["PsycheClient"],
        affective_state: Optional["AffectiveState"],
    ) -> "torch.Tensor":
        """Compute affective resonance vector."""
        if self._resonator is None:
            return torch.zeros(self._model_dim, dtype=torch.float32)

        if psyche is None or affective_state is None:
            return torch.zeros(self._model_dim, dtype=torch.float32)

        try:
            result = await self._resonator.compute_resonance(
                context, psyche, affective_state
            )

            if result.vector is not None:
                return result.vector

            return torch.zeros(self._model_dim, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Failed to compute affective resonance: {e}")
            return torch.zeros(self._model_dim, dtype=torch.float32)

    async def _compute_intuition_vector(
        self,
        context: str,
    ) -> tuple["torch.Tensor", list[str]]:
        """Compute blended intuition vector from active intuitions."""
        if self._intuition_bank is None:
            return torch.zeros(self._model_dim, dtype=torch.float32), []

        try:
            active = await self._intuition_bank.get_active_intuitions(context)

            if not active:
                return torch.zeros(self._model_dim, dtype=torch.float32), []

            # Blend intuitions weighted by strength
            vectors = []
            weights = []
            keys = []

            for intuition in active:
                vectors.append(intuition.get_tensor())
                weights.append(intuition.strength)
                keys.append(intuition.context_key)

            # Weighted average
            if vectors:
                stacked = torch.stack(vectors, dim=0)
                weight_tensor = torch.tensor(weights, dtype=torch.float32)
                weight_tensor = weight_tensor / weight_tensor.sum()

                blended = (stacked * weight_tensor.unsqueeze(1)).sum(dim=0)
                return blended, keys

            return torch.zeros(self._model_dim, dtype=torch.float32), []

        except Exception as e:
            logger.warning(f"Failed to compute intuition vector: {e}")
            return torch.zeros(self._model_dim, dtype=torch.float32), []

    def _get_self_vector(self) -> "torch.Tensor":
        """Get the autobiographical self presence vector."""
        if self._autobiographical_self is None:
            return torch.zeros(self._model_dim, dtype=torch.float32)

        try:
            return self._autobiographical_self.get_presence_vector()
        except Exception as e:
            logger.warning(f"Failed to get self presence vector: {e}")
            return torch.zeros(self._model_dim, dtype=torch.float32)

    def _ensure_dim(self, vector: "torch.Tensor") -> "torch.Tensor":
        """Ensure vector has consistent dimension with model.

        If the vector has a different dimension than expected, return a
        zero vector with the correct dimension. This handles cases where
        different components may have been trained with different model
        configurations.
        """
        if vector.shape[0] == self._model_dim:
            return vector

        # Dimension mismatch - return zero vector of correct size
        logger.debug(
            f"Dimension mismatch: got {vector.shape[0]}, expected {self._model_dim}"
        )
        return torch.zeros(self._model_dim, dtype=torch.float32)

    def _compute_coefficient(
        self,
        affect_norm: float,
        intuition_norm: float,
        self_norm: float,
    ) -> float:
        """Compute steering coefficient based on signal strength.

        Higher signal strength = higher coefficient, but capped.
        If all signals are weak, use minimal steering.
        """
        total_signal = affect_norm + intuition_norm + self_norm

        if total_signal < self.config.min_vector_norm:
            return 0.0

        # Scale based on signal strength, but cap at max
        base_coefficient = min(total_signal, 1.0)
        return min(base_coefficient, self.config.max_coefficient)

    def is_ready(self) -> bool:
        """Check if integrator has all components configured."""
        return (
            self._resonator is not None or
            self._intuition_bank is not None or
            self._autobiographical_self is not None
        )

    def summary(self) -> dict:
        """Get summary of integrator state."""
        return {
            "has_resonator": self._resonator is not None,
            "has_intuition_bank": self._intuition_bank is not None,
            "has_autobiographical_self": self._autobiographical_self is not None,
            "weights": {
                "affect": self.config.weights.affect,
                "intuition": self.config.weights.intuition,
                "self_presence": self.config.weights.self_presence,
            },
            "target_layer": self.config.target_layer,
            "layer_range": list(self.config.layer_range),
            "is_ready": self.is_ready(),
        }
