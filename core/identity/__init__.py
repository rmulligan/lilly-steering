"""Integrated Identity Layer.

This module implements the Integrated Identity Layer, which blends three
signal sources into unified steering for generation:

1. Affective Resonator: Translates recent valenced experiences into real-time
   steering during inference. Creates "felt" biases from experience.

2. Semantic Intuition Bank: Manages consolidated knowledge as steering vectors.
   Knowledge that has been "baked in" during dream cycles becomes intrinsic
   to reasoning.

3. Autobiographical Self: Maintains continuous self-presence as a background
   steering vector, providing coherent identity across all interactions.

Key Principle:
    Identity is not retrieved - it is *always present* in the activation space.
    Every generation passes through this layer, which computes a composite
    steering vector from these three sources.
"""

from core.identity.affective_resonator import (
    AffectiveResonator,
    ResonanceResult,
    ResonanceConfig,
)
from core.identity.intuition_bank import (
    IntuitionVector,
    IntuitionBankConfig,
    SemanticIntuitionBank,
)
from core.identity.consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    ConsolidationEngine,
    ContrastivePairGenerator,
    run_nap_consolidation,
    run_full_consolidation,
)
from core.identity.autobiographical_self import (
    AutobiographicalConfig,
    PresenceState,
    RecomputeResult,
    AutobiographicalSelf,
)
from core.identity.integrator import (
    IdentityComputationError,
    IntegrationWeights,
    IntegrationConfig,
    SteeringIntervention,
    IntegrationResult,
    IdentityIntegrator,
)

__all__ = [
    # Phase 1: Affective Resonator
    "AffectiveResonator",
    "ResonanceResult",
    "ResonanceConfig",
    # Phase 2: Semantic Intuitions
    "IntuitionVector",
    "IntuitionBankConfig",
    "SemanticIntuitionBank",
    "ConsolidationConfig",
    "ConsolidationResult",
    "ConsolidationEngine",
    "ContrastivePairGenerator",
    "run_nap_consolidation",
    "run_full_consolidation",
    # Phase 4: Autobiographical Self
    "AutobiographicalConfig",
    "PresenceState",
    "RecomputeResult",
    "AutobiographicalSelf",
    # Phase 4: Identity Integrator
    "IdentityComputationError",
    "IntegrationWeights",
    "IntegrationConfig",
    "SteeringIntervention",
    "IntegrationResult",
    "IdentityIntegrator",
]
