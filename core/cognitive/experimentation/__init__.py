"""Self-experimentation framework for bounded cognitive self-modification."""

from core.cognitive.experimentation.converter import HypothesisToExperimentConverter
from core.cognitive.experimentation.manager import ExperimentManager
from core.cognitive.experimentation.outcome_learner import (
    ExperimentOutcomeLearner,
    ParameterOutcome,
)
from core.cognitive.experimentation.schemas import (
    DOMAIN_COOLDOWN,
    MAX_EXPERIMENT_DURATION,
    ExperimentDomain,
    ExperimentMeasurement,
    ExperimentPhase,
    ExperimentProposal,
)
from core.cognitive.experimentation.utils import (
    PARAMETER_KEYWORD_MAP,
    extract_parameter_from_claim,
)

__all__ = [
    "ExperimentDomain",
    "ExperimentPhase",
    "ExperimentProposal",
    "ExperimentMeasurement",
    "ExperimentManager",
    "HypothesisToExperimentConverter",
    "ExperimentOutcomeLearner",
    "ParameterOutcome",
    "MAX_EXPERIMENT_DURATION",
    "DOMAIN_COOLDOWN",
    "PARAMETER_KEYWORD_MAP",
    "extract_parameter_from_claim",
]
