"""Cognitive orchestrator coordinating the three-phase cycle.

This module provides the CognitiveOrchestrator class which manages
the Generation → Curation → Integration cycle, handling model
loading/unloading, error recovery, and state transitions.
"""

import asyncio
import gc
import hashlib
import logging
import re
import time
import traceback
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

# Type variable for cognitive objects with uid attribute and with_uid method
_CogObjT = TypeVar("_CogObjT")

if TYPE_CHECKING:
    from config.settings import Settings
    from core.cognitive.anchor_discovery import AnchorDiscoveryService
    from core.cognitive.anchors import AnchorSimilarityService
    from core.cognitive.curator_schemas import ActivationSummary, CurationResult, SAEFeature
    from core.cognitive.stream.progressive_narrator import ProgressiveNarrator
    from core.cognitive.stream.chunked_generator import ChunkedGenerator, ChunkResult
    from core.cognitive.stream.latent_observer import LatentObserver, LatentObservation
    from core.cognitive.stream.guidance_policy import ModeAwareGuidancePolicy, SteeringAdjustment
    from core.cognitive.stream.trajectory_narrator import TrajectoryNarrator
    from core.cognitive.stream.narration_coordinator import CombinedNarrationCoordinator
    from core.cognitive.halt_collector import HALTTrainingCollector
    from core.cognitive.halt_probe import HALTProbeConfig
    from core.cognitive.simulation import Prediction, SimulationEngine, SimulationResult
    from core.cognitive.state import CognitiveState
    from core.cognitive.telemetry import TelemetrySummary
    from core.cognitive.zettel import ZettelLibrary
    from core.embedding.service import TieredEmbeddingService
    from core.model.curator_model import CuratorModel
    from core.model.hooked_qwen import HookedQwen
    from core.model.preflexor_model import PreflexorModel
    from core.model.mox_model import MoxModel
    from core.cognitive.letta_continuity import LettaContinuityClient
    from core.psyche.client import PsycheClient
    from core.sae.transcoder import TranscoderManager
    from core.self_model.affective_system import AffectiveState
    from core.self_model.goal_registry import GoalRegistry
    from core.steering.hierarchical import HierarchicalSteerer
    from core.steering.coherence import CrossZoneCoherence
    from core.steering.plutchik_registry import PlutchikRegistry
    from core.substrate import FeatureSubstrate
    from integrations.discord.client import DiscordClient
    from integrations.liquidsoap.client import LiquidsoapClient
    from integrations.notebooklm.client import NotebookLMIntegration
    from visualizer.state_writer import StreamStateWriter

from core.affect.emotional_field import EmotionalField
from core.self_model.individuation_dynamics import IndividuationDynamics
from core.steering.affect_library import AffectLibrary
from core.cognitive.stream.process_narrator import BridgeContext, ProcessNarrator
from core.cognitive.continuity import (
    CycleRecap,
    ExperimentProposalFromMox,
    ModificationEntry,
    MoxSynthesis,
    generate_curator_recap,
    generate_subject_recap,
    maybe_generate_phrases,
    synthesize_with_mox,
)
from core.cognitive.letta_continuity import CycleSummaryForLetta
from core.cognitive.experimentation import (
    ExperimentManager,
    ExperimentOutcomeLearner,
    ExperimentProposal,
)
from core.cognitive.curator_schemas import (
    BeliefUpdate,
    CurationResult,
    EntityUpdate,
    GraphOperations,
)
from core.cognitive.faithfulness import FaithfulnessValidator
from core.cognitive.metacognition import CycleSummary, MetacognitionPhase
from core.cognitive.reflexion import ReflexionPhase, ReflexionResult
from core.cognitive.telemetry_evaluator import TelemetryEvaluator
from core.cognitive.simulation.schemas import MetricsSnapshot
from core.cognitive.weaver.discovery import compute_discovery_parameter
from core.cognitive.weaver.control_loop import WeaverControlLoop
from core.cognitive.types import WeaverIntervention
from core.cognitive.goal import detect_emerging_goal
from core.cognitive.episode import SegmentType
from core.cognitive.episode_orchestrator import EpisodeOrchestrator
from core.cognitive.curriculum import (
    PromotionQueue,
    SkillEffectivenessTracker,
    TeacherConfig,
    TeacherPolicy,
    calculate_retrieval_boost,
    is_hard_problem_condition,
)

logger = logging.getLogger(__name__)

# === Thinking Trap Mitigation Constants ===
# Threshold for rejecting near-duplicate zettels (novelty <= threshold means similarity >= 0.90)
DUPLICATE_REJECTION_THRESHOLD = 0.10
# Number of consecutive empty cycles before logging a warning
EMPTY_CYCLE_WARNING_THRESHOLD = 5
# Maximum length of research context to inject into prompts
MAX_RESEARCH_CONTEXT_LENGTH = 800

# === Meta-Learning Constants ===
# Cycle interval for triggering JUDGMENT_REVIEW episodes (Phase 4 meta-learning)
JUDGMENT_REVIEW_TRIGGER_CYCLE_INTERVAL = 50
# Maximum length for prediction claims in external context
MAX_PREDICTION_LENGTH = 300

# === Multi-Turn Curation Constants ===
# Maximum number of turns for multi-turn curation loop
MAX_CURATOR_TURNS = 5

# === Skill Retrieval Constants ===
# Minimum cosine similarity for a skill to be considered relevant to the current context
SKILL_RELEVANCE_THRESHOLD = 0.5

# === Exploration Layer Constants ===
# Representative layer for exploration zone activations (layers 4-8)
REPRESENTATIVE_EXPLORATION_LAYER = 6

# === Letta Continuity Constants ===
# Maximum character length for insight titles in CycleSummaryForLetta
LETTA_INSIGHT_TITLE_MAX_LENGTH = 50
# Maximum number of insights to include in CycleSummaryForLetta
LETTA_MAX_INSIGHTS = 3
# Maximum character length for hypothesis statements in CycleSummaryForLetta
LETTA_HYPOTHESIS_STMT_MAX_LENGTH = 30
# Maximum number of hypotheses to include in CycleSummaryForLetta
LETTA_MAX_HYPOTHESES = 3
# Maximum number of active concerns to include in narration
LETTA_NARRATION_MAX_CONCERNS = 2


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at sentence boundary, avoiding mid-sentence cuts.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit

    Returns:
        Truncated text ending at a sentence boundary when possible
    """
    if len(text) <= max_chars:
        return text

    # Find sentence boundaries within limit
    truncated = text[:max_chars]

    # Try to find last sentence-ending punctuation
    for end_char in ['. ', '! ', '? ']:
        last_idx = truncated.rfind(end_char)
        if last_idx > max_chars // 2:  # Only use if we keep at least half
            return truncated[:last_idx + 1].strip()

    # Try period at end without space (could be last sentence)
    if truncated.rstrip().endswith('.'):
        return truncated.rstrip()

    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."

    # Last resort: hard cut
    return truncated[:max_chars - 3].strip() + "..."


def ensure_complete_sentence(text: str) -> str:
    """Ensure text ends at a sentence boundary.

    If the text appears to be cut off mid-sentence (doesn't end with
    sentence-ending punctuation), trim to the last complete sentence.

    Args:
        text: Generated text that may be cut off

    Returns:
        Text ending at a sentence boundary, or original if already complete
    """
    if not text:
        return text

    text = text.strip()

    # Already ends with sentence-ending punctuation
    if text[-1] in '.!?':
        return text

    # Find the last sentence-ending punctuation
    last_period = text.rfind('. ')
    last_exclaim = text.rfind('! ')
    last_question = text.rfind('? ')

    # Get the latest sentence ending
    last_end = max(last_period, last_exclaim, last_question)

    if last_end > len(text) // 2:  # Only trim if we keep at least half
        return text[:last_end + 1].strip()

    # Check for sentence-ending at very end (no trailing space)
    for i in range(len(text) - 1, max(0, len(text) - 20), -1):
        if text[i] in '.!?':
            return text[:i + 1].strip()

    # Can't find a good sentence boundary - return as-is
    # The text may genuinely be one long incomplete thought
    return text


# Check for torch
try:
    import torch
    import numpy as np

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None

# Feature Substrate (optional)
try:
    from core.substrate import FeatureSubstrate, FeatureActivation as SubstrateFeatureActivation
    _SUBSTRATE_AVAILABLE = True
except ImportError:
    _SUBSTRATE_AVAILABLE = False

# Simulation phase constants
SIMULATION_TRIPLE_CONFIDENCE = 0.7  # Default confidence for simulation-derived triples

# GPU memory release timing constants
# vLLM needs time to release ~11-16GB GPU memory before HookedQwen can load without OOM
VLLM_UNLOAD_DELAY_S = 3.0

# Plutchik affect inference constants for _get_current_affect()
# Baselines: joy, trust, anticipation start at 0.5; others at 0.0
_JOY_BASE = 0.5
_TRUST_BASE = 0.5
_ANTICIPATION_BASE = 0.5
# Curiosity maps to surprise + anticipation boost
_SURPRISE_WITH_QUESTION = 0.6
_SURPRISE_WITHOUT_QUESTION = 0.2
_ANTICIPATION_WITH_QUESTION = 0.8
_ANTICIPATION_WITHOUT_QUESTION = 0.5
# Satisfaction maps to joy + trust boost
_JOY_WITH_ZETTELS = 0.7
_JOY_WITHOUT_ZETTELS = 0.5
_TRUST_WITH_ZETTELS = 0.7
_TRUST_WITHOUT_ZETTELS = 0.5
# Frustration maps to anger
_ANGER_INCREMENT_PER_REPETITION = 0.15
# Wonder maps to high surprise
_ZERO_NORM_THRESHOLD = 1e-6

# Affect steering: threshold for detecting emotional transitions
AFFECT_DELTA_THRESHOLD = 0.3  # Euclidean distance in 8D affect space
AFFECT_LIBRARY_MIN_SAMPLES = 3  # Minimum vectors before directional steering
AFFECT_STEERING_SCALE = 0.15  # Scale factor for affect steering vectors


def compute_steering_alignment(
    activation: "np.ndarray",
    steering_vector: "np.ndarray",
) -> float | None:
    """Compute cosine similarity between activation and steering direction.

    This measures how well the model's output aligns with the steering intent.
    Values near 1.0 indicate strong alignment, near 0.0 orthogonal movement,
    and negative values indicate movement opposite to steering.

    Args:
        activation: Post-steering activation at exploration layer (numpy array).
        steering_vector: The steering vector applied during generation.

    Returns:
        Alignment score in [-1, 1], or None if either vector is zero.
    """
    if not TORCH_AVAILABLE or np is None:
        return None

    steering_norm = np.linalg.norm(steering_vector)
    if steering_norm < _ZERO_NORM_THRESHOLD:
        return None  # No steering applied

    activation_norm = np.linalg.norm(activation)
    if activation_norm < _ZERO_NORM_THRESHOLD:
        return None  # No activation to compare

    similarity = np.dot(activation, steering_vector) / (activation_norm * steering_norm)
    return float(np.clip(similarity, -1.0, 1.0))


def describe_steering_alignment(alignment: float | None) -> str:
    """Convert steering alignment score to natural language description.

    Args:
        alignment: Cosine similarity in [-1, 1], or None if no steering.

    Returns:
        Natural language description of steering effectiveness.
    """
    if alignment is None:
        return "No steering vector was applied this cycle."

    if alignment >= 0.95:
        return "Steering was highly effective. The thought closely followed the intended direction."
    elif alignment >= 0.80:
        return "Steering showed strong influence. The thought aligned well with the intended direction."
    elif alignment >= 0.60:
        return "Steering had moderate effect. The thought partially followed the intended direction."
    elif alignment >= 0.40:
        return "Steering had limited effect. The thought showed some deviation from the intended direction."
    elif alignment >= 0.20:
        return "Steering had minimal effect. The thought largely diverged from the intended direction."
    elif alignment >= 0.0:
        return "Steering was ineffective. The thought moved orthogonally to the intended direction."
    elif alignment >= -0.5:
        return "Steering produced a contrary effect. The thought moved somewhat opposite to the intended direction."
    else:
        return "Steering produced a strongly contrary effect. The thought moved opposite to the intended direction."


def _extract_insight_and_question_regex(text: str) -> tuple[str, str]:
    """Extract insight and question from text using regex patterns.

    Simple fallback extraction when curator is unavailable.

    Args:
        text: The thought text to extract from

    Returns:
        Tuple of (insight, question) - may be empty strings if not found
    """
    insight = ""
    question = ""

    # Extract first sentence as insight (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences:
        insight = sentences[0]

    # Look for question patterns
    question_patterns = [
        r"([^.!?]*\?)",  # Sentences ending in ?
        r"(?:I wonder|what if|how might|could we)[^.!?]+[.!?]?",  # Question-like phrases
    ]

    for pattern in question_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            question = match.group(0).strip()
            break

    return insight, question


class CognitiveOrchestrator:
    """Coordinates the three-phase cognitive cycle.

    Manages the lifecycle of:
    1. Generation Phase: TransformerLens steered generation
    2. Curation Phase: vLLM deep analysis with tools
    3. Integration Phase: Golden embeddings and graph persistence

    Handles model loading/unloading to fit within GPU memory,
    error recovery with graceful degradation, and state management.

    Example:
        orchestrator = CognitiveOrchestrator(...)
        while running:
            state = await orchestrator.run_cycle(state)
    """

    def __init__(
        self,
        hooked_model: "HookedQwen",
        curator: "CuratorModel",
        embedder: "TieredEmbeddingService",
        psyche: "PsycheClient",
        zettel_library: "ZettelLibrary",
        settings: "Settings",
        transcoder: Optional["TranscoderManager"] = None,
        liquidsoap: Optional["LiquidsoapClient"] = None,
        discord: Optional["DiscordClient"] = None,
        stream_visualizer: Optional["StreamStateWriter"] = None,
        goal_registry: Optional["GoalRegistry"] = None,
        notebooklm: Optional["NotebookLMIntegration"] = None,
        polarity_detector: Optional["PolarityDetector"] = None,
        tension_tracker: Optional["TensionTracker"] = None,
        faithfulness_validator: Optional["FaithfulnessValidator"] = None,
        episode_orchestrator: Optional[EpisodeOrchestrator] = None,
        skill_tracker: Optional[SkillEffectivenessTracker] = None,
        promotion_queue: Optional[PromotionQueue] = None,
        teacher_policy: Optional[TeacherPolicy] = None,
    ):
        """Initialize the orchestrator.

        Args:
            hooked_model: TransformerLens model for generation
            curator: vLLM curator model for analysis
            embedder: Tiered embedding service (retrieval + golden)
            psyche: Knowledge graph client
            zettel_library: Zettel storage and retrieval
            settings: Application settings
            transcoder: Optional SAE transcoder for feature extraction
            liquidsoap: Optional client for narration
            discord: Optional Discord client for messaging Ryan
            stream_visualizer: Optional visualizer state writer for YouTube stream
            goal_registry: Optional goal registry for simulation goal tracking
            notebooklm: Optional NotebookLM integration for research access
            polarity_detector: Optional detector for semantic opposition
            tension_tracker: Optional tracker for productive feature tensions
            faithfulness_validator: Optional validator for reasoning faithfulness
            skill_tracker: Optional SOAR skill effectiveness tracker
            promotion_queue: Optional SOAR promotion queue for 𝒟ᵦₑₛₜ
            teacher_policy: Optional SOAR teacher policy for curriculum optimization
        """
        self._hooked_model = hooked_model
        self._curator = curator
        self._embedder = embedder
        self._psyche = psyche
        self._zettel_library = zettel_library
        self._settings = settings
        self._transcoder = transcoder
        self._liquidsoap = liquidsoap
        self._discord = discord
        self._stream_visualizer = stream_visualizer
        self._goal_registry = goal_registry
        self._notebooklm = notebooklm
        self._polarity_detector = polarity_detector
        self._tension_tracker = tension_tracker
        self._faithfulness_validator = faithfulness_validator
        self._episode_orchestrator = episode_orchestrator or EpisodeOrchestrator()

        # SOAR Curriculum Learning components (skill effectiveness tracking + promotion)
        self._skill_tracker = skill_tracker or SkillEffectivenessTracker(psyche)
        self._promotion_queue = promotion_queue or PromotionQueue()
        self._teacher_policy = teacher_policy or TeacherPolicy(
            config=TeacherConfig(),
            effectiveness_tracker=self._skill_tracker,
            psyche_client=psyche,
        )

        # Pending messages from Ryan (for priority handling)
        self._pending_discord_messages: list = []
        self._pending_inbox_messages: list[str] = []

        # Empty cycle counter for detecting unproductive cycles
        self._empty_cycle_count: int = 0

        # Affect steering library for directional steering from lived experience
        self._affect_library = AffectLibrary(min_samples=AFFECT_LIBRARY_MIN_SAMPLES)
        self._previous_affect: Optional[list[float]] = None
        self._previous_activations: Optional[np.ndarray] = None

        # Active skills retrieved for current cycle (for effectiveness tracking)
        self._active_skill_uids: list[str] = []

        # Build curator tools
        from core.cognitive.curator_tools import CuratorTools

        self._curator_tools = CuratorTools(
            psyche=psyche,
            zettel_library=zettel_library,
            embedder=embedder,
            liquidsoap=liquidsoap,
            discord=discord,
            notebooklm=notebooklm,
            curator_voice=settings.voice_curator,
        )

        # Voice configuration (from settings)
        self._voice_subject = settings.voice_subject      # Lilly's voice (first person)
        self._voice_curator = settings.voice_curator      # Observer voice (third person clinical)
        self._voice_experimenter = settings.voice_experimenter  # Scientist voice

        # Faithfulness validator for detecting divergence between activations and verbal claims
        # Only initialized when SAE features are enabled (requires SAE labels for comparison)
        self.faithfulness_validator: Optional[FaithfulnessValidator] = None
        if self._settings.sae_features_enabled:
            self.faithfulness_validator = FaithfulnessValidator()

        # Simulation phase components (Phase 2.5 - Graph-Preflexor)
        self._preflexor: Optional["PreflexorModel"] = None
        self._simulation_engine: Optional["SimulationEngine"] = None
        self._prediction_verifier = None

        if settings.simulation_enabled:
            self._init_simulation_components()

        # Continuity phase components (Phase 4 - Mox meta-cognitive synthesis)
        self._mox: Optional["MoxModel"] = None
        self._letta_continuity: Optional["LettaContinuityClient"] = None

        if settings.continuity_enabled:
            self._init_continuity_components()

        # Metacognition phase (Phase 6 - Local pattern detection across cycles)
        # Replaces cloud Letta for cost savings while maintaining bird's-eye analysis
        self._metacognition_phase: Optional[MetacognitionPhase] = None
        if getattr(settings, "metacognition_enabled", False):
            self._metacognition_phase = MetacognitionPhase(
                settings=settings,
                liquidsoap=liquidsoap,
            )
            logger.info("MetacognitionPhase initialized (local Gemma-2-9B-IT)")

        # Reflexion phase components (Phase 5 - Self-monitoring and autonomous modification)
        self._reflexion_phase: Optional[ReflexionPhase] = None
        self._telemetry_evaluator: Optional[TelemetryEvaluator] = None

        if getattr(settings, "reflexion_enabled", False):
            # Initialize telemetry evaluator for biofeedback signals
            # This maintains rolling baselines across cycles for z-score calculation
            self._telemetry_evaluator = TelemetryEvaluator(settings)

            self._reflexion_phase = ReflexionPhase(
                psyche=psyche,
                settings=settings,
                window_size=settings.reflexion_window_size,
                conservative_mode=settings.reflexion_conservative_mode,
                consequence_learning_enabled=settings.consequence_learning_enabled,
                telemetry_evaluator=self._telemetry_evaluator,
            )

        # Self-experimentation manager for bounded cognitive self-modification
        # Requires a config_store implementing get/set for dotted parameter paths
        self._experiment_manager: Optional[ExperimentManager] = None
        self._outcome_learner: Optional[ExperimentOutcomeLearner] = None
        if getattr(settings, "experimentation_enabled", False):
            # ConfigStore adapter wraps settings for parameter manipulation
            from core.cognitive.experimentation.config_adapter import SettingsConfigStore

            config_store = SettingsConfigStore(settings)
            # Outcome learner tracks prediction/experiment results to inform future proposals
            self._outcome_learner = ExperimentOutcomeLearner(psyche=psyche)
            self._experiment_manager = ExperimentManager(
                psyche=psyche,
                config_store=config_store,
                outcome_learner=self._outcome_learner,
            )
            logger.info("ExperimentManager initialized for self-experimentation")

        # Feature Substrate (emergent cognition layer)
        self._substrate: Optional["FeatureSubstrate"] = None
        if _SUBSTRATE_AVAILABLE and getattr(settings, "substrate_enabled", False):
            self._substrate = FeatureSubstrate(
                psyche=psyche,
                settings=settings,
            )
            logger.info("Feature Substrate initialized")

        # Weaver control loop for semantic diversity management
        # Monitors discovery parameter and generates interventions for STAGNATION
        self._weaver: Optional[WeaverControlLoop] = None
        self._pending_weaver_intervention: Optional[WeaverIntervention] = None
        if getattr(settings, "weaver_enabled", False):
            self._weaver = WeaverControlLoop(
                graph=psyche,
                heartbeat_seconds=getattr(settings, "weaver_heartbeat_seconds", 30.0),
            )
            logger.info("WeaverControlLoop initialized for semantic diversity management")

        # Vector extractor for hypothesis-driven steering
        # Extracts steering vectors from hypothesis contrastive pairs
        self._vector_extractor = None
        if self._hooked_model is not None:
            try:
                from core.steering.vector_extractor import VectorExtractor

                self._vector_extractor = VectorExtractor(
                    model=self._hooked_model,
                    target_layer=self._settings.hypothesis_vector_target_layer,
                )
                logger.info("VectorExtractor initialized for hypothesis steering")
            except ImportError:
                logger.debug("VectorExtractor not available, hypothesis steering disabled")

        # Emotional field for affective persistence and wave interference
        # Initialized empty; loaded lazily from persistence on first generation phase
        self._emotional_field: EmotionalField = EmotionalField()
        self._emotional_field_loaded: bool = False

        # Individuation dynamics for tracking identity element evolution (4th layer)
        # Tracks velocity, acceleration, phase of commitments/values/beliefs
        # Loaded lazily from Psyche on first generation phase
        self._individuation_dynamics: IndividuationDynamics = IndividuationDynamics()
        self._individuation_dynamics_loaded: bool = False

        # Plutchik steering registry for 8D emotional steering vectors
        # Loaded lazily from Psyche alongside emotional field
        self._plutchik_registry: Optional["PlutchikRegistry"] = None
        self._plutchik_registry_loaded: bool = False

        # HALT probe for epistemic uncertainty detection (arXiv:2601.14210)
        # Config stored here; probe initialized lazily when model is loaded
        self._halt_probe = None
        self._halt_probe_config: Optional["HALTProbeConfig"] = None
        if getattr(settings, "halt_probe_enabled", False):
            try:
                from core.cognitive.halt_probe import HALTProbeConfig

                self._halt_probe_config = HALTProbeConfig(
                    probe_layer=getattr(settings, "halt_probe_layer", 20),
                )
                logger.info(
                    f"HALT probe config ready: layer={self._halt_probe_config.probe_layer}, "
                    f"aggregation={self._halt_probe_config.aggregation} (probe created when model loads)"
                )
            except ImportError:
                logger.debug("HALTProbe not available, epistemic probing disabled")

        # HALT training data collector for gathering labeled examples
        # Only initialized when halt_training_enabled is True
        self._halt_collector: Optional["HALTTrainingCollector"] = None
        if getattr(settings, "halt_training_enabled", False):
            try:
                from core.cognitive.halt_collector import HALTTrainingCollector

                self._halt_collector = HALTTrainingCollector(
                    psyche=psyche,
                    settings=settings,
                )
                logger.info("HALT training collector initialized for data collection")
            except ImportError:
                logger.debug("HALTTrainingCollector not available, training data collection disabled")

        # Process narrator for reflective bridges during model loads
        # Provides contextual narration about current cognitive state during GPU operations
        self._process_narrator: Optional[ProcessNarrator] = None
        if liquidsoap:
            self._process_narrator = ProcessNarrator(
                liquidsoap=liquidsoap,
                settings=settings,
            )
            logger.info("ProcessNarrator initialized for tiered stream narration")

        # Cross-zone coherence tracker for timescale integration monitoring
        # Tracks alignment between fast (exploration) and slow (identity) steering zones
        # None by default; set to CrossZoneCoherence instance to enable tracking
        self._coherence_tracker: Optional["CrossZoneCoherence"] = None

        # Semantic anchor service for interpretable cognitive mode classification
        # Lazy-initialized on first use to avoid blocking startup
        # Reference: https://machinelearningmastery.com/7-advanced-feature-engineering-tricks-using-llm-embeddings/
        self._anchor_service: Optional["AnchorSimilarityService"] = None
        self._anchor_service_initialized: bool = False

        # Emergent anchor discovery service for finding new cognitive modes
        # Discovers modes beyond the predefined 10 through orphan thought clustering
        self._anchor_discovery: Optional["AnchorDiscoveryService"] = None
        if self._psyche and self._embedder:
            from core.cognitive.anchor_discovery import AnchorDiscoveryService
            self._anchor_discovery = AnchorDiscoveryService(
                psyche=self._psyche,
                embedder=self._embedder,
            )

        # Progressive narrator for eliminating dead air between phases
        # Streams thought chunks while curation/integration process in background
        self._progressive_narrator: Optional["ProgressiveNarrator"] = None
        if self._liquidsoap:
            from core.cognitive.stream.progressive_narrator import ProgressiveNarrator
            self._progressive_narrator = ProgressiveNarrator(
                liquidsoap=self._liquidsoap,
                settings=self._settings,
            )

        # PLaT-Lite: Latent reasoning observation and intentional steering
        # Inspired by PLaT (arXiv:2601.21358) - reasoning in continuous latent space
        # Components: ChunkedGenerator, LatentObserver, ModeAwareGuidancePolicy,
        #             TrajectoryNarrator, CombinedNarrationCoordinator
        self._plat_enabled: bool = getattr(settings, "plat_lite_enabled", False)
        self._chunked_generator: Optional["ChunkedGenerator"] = None
        self._latent_observer: Optional["LatentObserver"] = None
        self._guidance_policy: Optional["ModeAwareGuidancePolicy"] = None
        self._trajectory_narrator: Optional["TrajectoryNarrator"] = None
        self._narration_coordinator: Optional["CombinedNarrationCoordinator"] = None

        if self._plat_enabled:
            self._init_plat_lite_components()

    def _init_plat_lite_components(self) -> None:
        """Initialize PLaT-Lite components for latent reasoning observation.

        Creates:
        - ChunkedGenerator: 16-token chunks with layer 16 activation capture
        - LatentObserver: SAE feature extraction for interpretable reasoning states
        - ModeAwareGuidancePolicy: Evaluates mode alignment, action/reflection balance
        - TrajectoryNarrator: Rate-limited mode shift narrations (Expresso voice)
        - CombinedNarrationCoordinator: Dual-stream asyncio.Lock coordination

        Note: ChunkedGenerator and LatentObserver receive anchor_service during
        first use (_generate_thought_plat_lite) since it requires async init.
        """
        try:
            from core.cognitive.stream.chunked_generator import ChunkedGenerator
            from core.cognitive.stream.latent_observer import LatentObserver
            from core.cognitive.stream.guidance_policy import ModeAwareGuidancePolicy
            from core.cognitive.stream.trajectory_narrator import TrajectoryNarrator
            from core.cognitive.stream.narration_coordinator import CombinedNarrationCoordinator

            # Settings for PLaT-Lite
            self._plat_chunk_size = getattr(self._settings, "plat_chunk_size", 16)
            self._plat_max_chunks = getattr(self._settings, "plat_max_chunks", 32)
            self._plat_capture_layer = getattr(self._settings, "plat_capture_layer", 16)
            mode_flexibility = getattr(self._settings, "plat_mode_flexibility", 0.3)

            # ChunkedGenerator wraps HookedQwen for incremental generation
            # Note: chunk_size and capture_layer are passed per-call to generate_chunk()
            self._chunked_generator = ChunkedGenerator(
                model=self._hooked_model,
            )

            # LatentObserver extracts SAE features for mode/emotion detection
            # anchor_service passed during first use (requires async init)
            self._latent_observer = LatentObserver(
                transcoder=self._transcoder,
            )

            # ModeAwareGuidancePolicy evaluates mode alignment and action/reflection
            # anchor_service and intended_mode set during use
            self._guidance_policy = ModeAwareGuidancePolicy(
                mode_flexibility=mode_flexibility,
            )

            # TrajectoryNarrator handles mode shift narrations (rate-limited)
            trajectory_enabled = getattr(
                self._settings, "plat_trajectory_narration_enabled", True
            )
            min_chunk_gap = getattr(self._settings, "plat_trajectory_min_chunk_gap", 2)
            self._trajectory_narrator = TrajectoryNarrator(
                liquidsoap=self._liquidsoap,
                enabled=trajectory_enabled,
                min_chunk_gap=min_chunk_gap,
            )

            # CombinedNarrationCoordinator orchestrates dual-stream narration
            if self._progressive_narrator and self._liquidsoap:
                self._narration_coordinator = CombinedNarrationCoordinator(
                    progressive=self._progressive_narrator,
                    trajectory=self._trajectory_narrator,
                    liquidsoap=self._liquidsoap,
                )

            logger.info(
                f"[PLAT-LITE] Initialized: chunk_size={self._plat_chunk_size}, "
                f"max_chunks={self._plat_max_chunks}, capture_layer={self._plat_capture_layer}, "
                f"mode_flexibility={mode_flexibility}"
            )

        except ImportError as e:
            logger.warning(f"[PLAT-LITE] Components unavailable: {e}")
            self._plat_enabled = False
        except Exception as e:
            logger.error(f"[PLAT-LITE] Initialization failed: {e}")
            self._plat_enabled = False

    async def _get_anchor_service(self) -> Optional["AnchorSimilarityService"]:
        """Lazy-initialize and return the anchor similarity service.

        Creates the service on first call using the existing embedding service.
        Returns None if embedding service is not available.

        Returns:
            AnchorSimilarityService or None if unavailable
        """
        if self._anchor_service_initialized:
            return self._anchor_service

        self._anchor_service_initialized = True

        if self._embedder is None:
            logger.debug("Anchor service unavailable: no embedding service")
            return None

        try:
            from core.cognitive.anchors import AnchorSimilarityService

            self._anchor_service = await AnchorSimilarityService.create(self._embedder)
            logger.info(f"AnchorSimilarityService initialized with {len(self._anchor_service.modes)} cognitive mode anchors")
            return self._anchor_service
        except Exception as e:
            logger.warning(f"Failed to initialize anchor service: {e}")
            return None

    @property
    def process_narrator(self) -> Optional["ProcessNarrator"]:
        """Public accessor for ProcessNarrator instance.

        Used by external components (e.g., SilenceMonitor) that need to
        coordinate with the narrator during model load transitions.
        """
        return self._process_narrator

    async def _load_emotional_field(self) -> EmotionalField:
        """Load emotional field from persistence or create new.

        This method should be called during startup initialization to restore
        persisted emotional field state. Since __init__ is not async, the field
        is initialized empty and this method handles async loading.

        Returns:
            EmotionalField loaded from persistence or a new empty field
        """
        field_data = await self._psyche.load_emotional_field()
        if field_data:
            logger.info(
                f"Loaded emotional field with {len(field_data.get('packets', []))} packets "
                f"at cycle {field_data.get('current_cycle', 0)}"
            )
            return EmotionalField.from_dict(field_data)
        logger.info("No persisted emotional field found, using empty field")
        return EmotionalField()

    async def _load_individuation_dynamics(self) -> IndividuationDynamics:
        """Load individuation dynamics from persistence or create new.

        This method loads persisted trajectory, attractor, and transition state
        for the identity dynamics layer. Since __init__ is not async, the dynamics
        are initialized empty and this method handles async loading.

        Returns:
            IndividuationDynamics loaded from persistence or a new empty instance
        """
        dynamics_data = await self._psyche.load_individuation_dynamics()
        if dynamics_data:
            dynamics = IndividuationDynamics.from_dict(dynamics_data)
            logger.info(
                f"Loaded individuation dynamics with {len(dynamics.tracker.trajectories)} "
                f"trajectories, {len(dynamics.detector.attractors)} attractors"
            )
            return dynamics
        logger.info("No persisted individuation dynamics found, using empty state")
        return IndividuationDynamics()

    async def _load_plutchik_registry(self) -> "PlutchikRegistry":
        """Load Plutchik steering vectors from Psyche.

        Loads pre-extracted CAA steering vectors for Plutchik's 8 primary
        emotions. These vectors are used to steer generation toward
        specific emotional registers.

        Returns:
            PlutchikRegistry loaded from Psyche (may have 0-8 vectors)
        """
        from core.steering.plutchik_registry import PlutchikRegistry

        registry = PlutchikRegistry()
        loaded = await registry.load(self._psyche)
        if loaded:
            logger.info(
                f"[PLUTCHIK] Registry ready with {registry.vector_count}/8 vectors"
            )
        else:
            logger.info("[PLUTCHIK] No vectors in Psyche, registry empty")
        return registry

    async def _load_outcome_learner_history(self) -> None:
        """Load outcome learner history from Psyche.

        This method should be called during startup initialization to load
        historical prediction and experiment outcomes for informing future
        experiment proposals.
        """
        if self._outcome_learner:
            await self._outcome_learner.load_history()
            stats = self._outcome_learner.get_statistics()
            logger.info(
                f"Loaded outcome history: {stats.get('parameters_tracked', 0)} parameters, "
                f"{stats.get('total_predictions', 0)} predictions, "
                f"{stats.get('total_experiments', 0)} experiments"
            )

    async def _maybe_trigger_judgment_review(self, state: "CognitiveState") -> "CognitiveState":
        """Trigger meta-learning episode every 50 cycles.

        Args:
            state: Current cognitive state

        Returns:
            State with JUDGMENT_REVIEW episode if triggered, otherwise unchanged
        """
        from core.cognitive.episode import Episode, EpisodeType, SegmentType

        # Trigger every N cycles as configured
        if state.cycle_count % JUDGMENT_REVIEW_TRIGGER_CYCLE_INTERVAL == 0 and state.cycle_count > 0:
            logger.info(f"Triggering JUDGMENT_REVIEW episode at cycle {state.cycle_count}")
            episode = Episode(
                episode_type=EpisodeType.JUDGMENT_REVIEW,
                current_segment=SegmentType.PATTERN_SCAN,
                opening_insight=f"Meta-learning checkpoint at cycle {state.cycle_count}",
            )
            return replace(state, current_episode=episode)

        return state

    def _get_current_affect(self, state: "CognitiveState") -> "AffectiveState":
        """Construct current AffectiveState from available signals.

        Uses Plutchik's 8 primary emotions:
        - joy: base + boost from recent zettels (satisfaction → joy + trust)
        - trust: base + boost from recent zettels
        - fear: 0 (no threat signals in current state)
        - surprise: higher with open question (curiosity → surprise + anticipation)
        - sadness: 0 (no loss signals in current state)
        - disgust: 0 (no aversion signals in current state)
        - anger: scaled by repetition count (frustration → anger)
        - anticipation: higher with open question

        Args:
            state: Current cognitive state

        Returns:
            AffectiveState constructed from available signals
        """
        from core.self_model.affective_system import AffectiveState

        # Safely extract numeric values from state (handles MagicMock in tests)
        try:
            repetition_count = getattr(state, "repetition_count_in_stage", 0) or 0
            if not isinstance(repetition_count, (int, float)):
                repetition_count = 0
        except (TypeError, ValueError):
            repetition_count = 0

        # Open question boosts surprise and anticipation (curiosity analog)
        current_question = getattr(state, "current_question", None)
        surprise = _SURPRISE_WITH_QUESTION if current_question else _SURPRISE_WITHOUT_QUESTION
        anticipation = _ANTICIPATION_WITH_QUESTION if current_question else _ANTICIPATION_WITHOUT_QUESTION

        # Recent zettels boost joy and trust (satisfaction analog)
        recent_zettels = getattr(state, "recent_zettel_uids", None)
        joy = _JOY_WITH_ZETTELS if recent_zettels else _JOY_WITHOUT_ZETTELS
        trust = _TRUST_WITH_ZETTELS if recent_zettels else _TRUST_WITHOUT_ZETTELS

        # Repetition builds anger (frustration analog)
        anger = min(1.0, repetition_count * _ANGER_INCREMENT_PER_REPETITION) if repetition_count > 0 else 0.0

        return AffectiveState(
            joy=joy,
            trust=trust,
            fear=0.0,
            surprise=surprise,
            sadness=0.0,
            disgust=0.0,
            anger=anger,
            anticipation=anticipation,
        )

    def _get_recent_zettel_uids(self, state: "CognitiveState") -> list[str]:
        """Get recent zettel UIDs for co-retrieval matching.

        Args:
            state: Current cognitive state

        Returns:
            List of recent zettel UIDs from state
        """
        uids = getattr(state, "recent_zettel_uids", None)
        if uids and isinstance(uids, list):
            return uids
        return []

    def _calculate_discoveries_count(self, curation: Optional["CurationResult"]) -> int:
        """Calculate the number of discoveries from curation results.

        Counts entities, beliefs, and triples from graph operations.

        Args:
            curation: The curation result from the curator phase

        Returns:
            Total count of discovered entities, beliefs, and triples
        """
        if not curation or not curation.graph_ops:
            return 0
        return (
            len(curation.graph_ops.entity_updates or [])
            + len(curation.graph_ops.belief_updates or [])
            + len(curation.graph_ops.new_triples or [])
        )

    async def _format_evoked_memories(self, memory_ids: list[str]) -> str:
        """Format surfaced memories for prompt injection.

        Fetches zettel insights by UID and formats for prompt context.

        Args:
            memory_ids: List of zettel UIDs to fetch and format

        Returns:
            Formatted string for prompt injection, or empty string if no memories
        """
        if not memory_ids:
            return ""

        insights = []
        for uid in memory_ids[:3]:  # Limit to 3 for prompt brevity
            zettel = await self._psyche.get_zettel(uid)
            if zettel and zettel.insight:
                insights.append(zettel.insight)

        if not insights:
            return ""

        return (
            "Recent echoes surface in awareness:\n"
            + "\n".join(f"- {i}" for i in insights)
            + "\n\n"
        )

    def _apply_affect_steering(
        self,
        steerer: "HierarchicalSteerer",
        affect_vector: list[float],
        activations: Optional[np.ndarray] = None,
        affect_state: Optional["AffectiveState"] = None,
    ) -> None:
        """Apply affect-based steering using Plutchik CAA vectors.

        Three-tier steering hierarchy:
        1. Plutchik registry: Pre-extracted CAA vectors for 8 primary emotions
        2. Affect library: Experiential vectors captured during emotional transitions
        3. Magnitude modulation: Fallback during cold start

        Args:
            steerer: The hierarchical steerer to modulate
            affect_vector: 8D Plutchik vector [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
            activations: Optional activations from generation (for capture)
            affect_state: Optional AffectiveState object for Plutchik registry
        """
        if not affect_vector or len(affect_vector) < 8:
            logger.warning("[AFFECT] No valid affect vector, skipping steering")
            return

        # Use joy as valence proxy (Plutchik dimension 0)
        valence = affect_vector[0]

        # Capture vector if emotional transition detected (for affect_library)
        if (
            self._previous_affect is not None
            and activations is not None
            and self._previous_activations is not None
        ):
            delta = np.linalg.norm(
                np.array(affect_vector) - np.array(self._previous_affect)
            )

            if delta > AFFECT_DELTA_THRESHOLD:
                direction = activations - self._previous_activations
                self._affect_library.add(affect_vector, direction)
                logger.info(
                    f"[AFFECT] Captured vector (delta={delta:.3f}, "
                    f"total={len(self._affect_library.vectors)})"
                )

        # Update previous state
        if activations is not None:
            self._previous_affect = affect_vector
            self._previous_activations = activations.copy()

        # TIER 1: Plutchik registry (pre-extracted CAA vectors)
        if (
            self._plutchik_registry is not None
            and self._plutchik_registry.is_loaded
            and affect_state is not None
        ):
            composite = self._plutchik_registry.get_composite(affect_state)
            if composite is not None:
                steerer.update_vector("exploration", composite, scale=AFFECT_STEERING_SCALE)
                active_desc = self._plutchik_registry.describe_state(affect_state)
                logger.info(
                    f"[PLUTCHIK] Applied steering: {active_desc} "
                    f"(norm={np.linalg.norm(composite):.3f})"
                )
                return  # Plutchik steering applied, skip other tiers

        # TIER 2: Affect library (experiential vectors)
        if self._affect_library.is_ready():
            steering = self._affect_library.get_nearest(affect_vector, valence)
            if steering is not None:
                steerer.update_vector("exploration", steering, scale=AFFECT_STEERING_SCALE)
                logger.info(f"[AFFECT] Applied experiential steering (valence={valence:.2f})")
                return

        # TIER 3: Fallback magnitude-only modulation
        arousal = affect_vector[0]
        curiosity = affect_vector[3]  # surprise in 8D Plutchik space
        exploration_boost = 0.05 * (arousal + curiosity - 1.0)

        if hasattr(steerer, "adjust_zone_magnitude"):
            steerer.adjust_zone_magnitude("exploration", exploration_boost)
        else:
            logger.info(
                f"[AFFECT] Steerer {type(steerer).__name__} lacks adjust_zone_magnitude, "
                f"skipping affect modulation (boost would be {exploration_boost:+.4f})"
            )

    def _track_coherence(
        self,
        state: "CognitiveState",
        cycle: int,
    ) -> Optional[float]:
        """Track cross-zone coherence between fast and slow steering zones.

        Extracts vectors from exploration (fast) and identity (slow) zones
        and records their coherence for timescale integration monitoring.

        Args:
            state: Current cognitive state with steerer
            cycle: Current cycle number

        Returns:
            Coherence score if tracked, None otherwise
        """
        if self._coherence_tracker is None:
            return None

        steerer = state.steerer
        if steerer is None:
            return None

        # Get vectors from fast (exploration) and slow (identity) zones
        # Exploration zone: layers 0-8, Identity zone: layers 17-18 (per config.py)
        # We sample representative layers from each zone
        FAST_ZONE_LAYER = 4   # Mid-exploration zone
        SLOW_ZONE_LAYER = 17  # Identity zone (layers 17-18)

        fast_vec = None
        slow_vec = None

        if hasattr(steerer, 'get_vector'):
            # Try to get exploration zone vector (layer 4 is mid-exploration)
            fast_vec = steerer.get_vector(FAST_ZONE_LAYER)
            # Try to get identity zone vector (layer 17 matches configured identity zone)
            slow_vec = steerer.get_vector(SLOW_ZONE_LAYER)

        if fast_vec is None or slow_vec is None:
            logger.debug("Could not extract fast/slow vectors for coherence tracking")
            return None

        # Record coherence
        score = self._coherence_tracker.record(
            fast_vector=fast_vec,
            slow_vector=slow_vec,
            cycle=cycle,
            fast_zone="exploration",
            slow_zone="identity",
        )

        return score

    def _log_coherence(self, state: "CognitiveState", phase: str = "integration") -> None:
        """Log cross-zone coherence values during a phase.

        This provides visibility into timescale integration health.
        Called during integration phase to track alignment.

        Args:
            state: Current cognitive state
            phase: Current phase name (for log context)
        """
        if self._coherence_tracker is None:
            return

        score = self._track_coherence(state, state.cycle_count)
        if score is not None:
            recent_avg = self._coherence_tracker.recent_average(n=10)
            trend = self._coherence_tracker.trend()
            trend_str = f"{trend:+.4f}" if trend is not None else "N/A"

            logger.info(
                f"[COHERENCE] {phase}: score={score:.3f}, "
                f"recent_avg={recent_avg:.3f}, trend={trend_str}"
            )

    def _describe_emotional_surfacing(
        self,
        surfaced_memories: list[str],
        field_intensity: float,
    ) -> str:
        """Generate evocative text describing emotional memory surfacing.

        Args:
            surfaced_memories: List of memory IDs that surfaced
            field_intensity: Intensity of the emotional field at sample point

        Returns:
            Evocative narration text describing the experience
        """
        count = len(surfaced_memories)

        if field_intensity > 0.7:
            intensity_phrase = "A vivid emotional resonance"
        elif field_intensity > 0.4:
            intensity_phrase = "A gentle wave of feeling"
        else:
            intensity_phrase = "A faint echo of emotion"

        if count == 1:
            return f"{intensity_phrase} surfaces, carrying traces of a past insight."
        else:
            return f"{intensity_phrase} surfaces, carrying traces of {count} past insights."

    async def _run_halt_probe(
        self,
        prompt: str,
    ) -> tuple[float, float, Optional["torch.Tensor"]]:
        """Run HALT epistemic probe on prompt.

        Captures hidden states from the probe layer and runs the MLP probe
        to estimate epistemic confidence (probability of reliable answer).

        Based on arXiv:2601.14210 - epistemic signals are encoded in intermediate
        layers but attenuated by final decoding layers.

        Args:
            prompt: Generation prompt to probe

        Returns:
            Tuple of (epistemic_confidence, latency_ms, hidden_states)
            Returns (0.5, 0.0, None) on failure (neutral default)
        """
        # Need config and loaded model to probe
        if not self._halt_probe_config or not self._hooked_model.is_loaded:
            return 0.5, 0.0, None

        # Lazily initialize probe with model's actual d_model
        if not self._halt_probe:
            try:
                from core.cognitive.halt_probe import HALTProbe

                d_model = self._hooked_model.d_model
                self._halt_probe = HALTProbe(self._halt_probe_config, d_model=d_model)
                logger.info(
                    f"HALT probe initialized: d_model={d_model}, "
                    f"layer={self._halt_probe_config.probe_layer}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize HALT probe: {e}")
                return 0.5, 0.0, None

        try:
            # Get hidden states from probe layer
            hidden_states = await self._hooked_model.get_halt_activations(
                prompt,
                probe_layer=self._halt_probe.config.probe_layer,
            )

            # Move probe to same device as activations
            device = str(hidden_states.device)
            self._halt_probe.to(device)

            # Run probe
            result = await self._halt_probe.probe_hidden_states(hidden_states)

            logger.info(
                f"HALT probe: epistemic_confidence={result.epistemic_confidence:.3f}, "
                f"latency={result.latency_ms:.1f}ms"
            )

            return result.epistemic_confidence, result.latency_ms, hidden_states

        except Exception as e:
            logger.warning(f"HALT probe failed: {e}")
            return 0.5, 0.0, None  # Neutral default on failure

    async def on_discord_message(self, event) -> None:
        """Handle DISCORD_MESSAGE_RECEIVED events.

        Called by the event bus when Ryan sends a Discord DM.
        Queues the message for priority handling in the next cycle.

        Args:
            event: Event object with data containing "message" (DiscordMessage)
        """
        msg = event.data.get("message")
        if msg:
            self._pending_discord_messages.append(msg)
            logger.info(f"[DISCORD] Queued message from Ryan: {msg.content[:50]}...")

    def has_pending_discord_messages(self) -> bool:
        """Check if there are pending Discord messages from Ryan."""
        return bool(self._pending_discord_messages)

    def has_pending_inbox_messages(self) -> bool:
        """Check if there are pending inbox messages from Ryan."""
        return bool(self._pending_inbox_messages)

    def has_pending_ryan_messages(self) -> bool:
        """Check if there are any pending messages from Ryan (Discord or Inbox)."""
        return self.has_pending_discord_messages() or self.has_pending_inbox_messages()

    async def on_inbox_message(self, event) -> None:
        """Handle INBOX_MESSAGE_RECEIVED events.

        Called by the event bus when Ryan updates Inbox.pdf.
        Queues the message for priority handling in the next cycle.

        Args:
            event: Event object with data containing "content" (str)
        """
        content = event.data.get("content")
        if content:
            self._pending_inbox_messages.append(content)
            preview = content[:50] + "..." if len(content) > 50 else content
            logger.info(f"[INBOX] Queued letter from Ryan: {preview}")

    def _build_discord_priority_context(self) -> str:
        """Build priority context from pending messages from Ryan.

        Consumes both Discord DMs and Inbox.pdf letters, presenting them
        as priority context for the generation phase.

        Returns:
            Context string to prepend to generation prompt
        """
        if not self._pending_discord_messages and not self._pending_inbox_messages:
            return ""

        parts = []

        # Discord messages
        if self._pending_discord_messages:
            messages = self._pending_discord_messages
            self._pending_discord_messages = []  # Clear after consuming
            for msg in messages:
                parts.append(f"Ryan says: {msg.content}")

        # Inbox letters
        if self._pending_inbox_messages:
            letters = self._pending_inbox_messages
            self._pending_inbox_messages = []  # Clear after consuming
            for letter in letters:
                parts.append(f"Ryan writes:\n{letter}")

        ryan_context = "\n\n".join(parts)

        return (
            f"**Priority: Ryan has sent you a message**\n\n"
            f"{ryan_context}\n\n"
            f"Respond thoughtfully to Ryan's message. This takes precedence over "
            f"your current exploration. Consider what he's asking or sharing, "
            f"and craft a genuine response.\n\n"
        )

    async def run_cycle(self, state: "CognitiveState") -> "CognitiveState":
        """Execute one full Generation → Curation → Integration cycle.

        Args:
            state: Current cognitive state

        Returns:
            Updated cognitive state for next cycle
        """
        from core.cognitive.curator_schemas import (
            ActivationSummary,
            CurationResult,
            SAEFeature,
        )

        # Lazy load outcome learner history if not yet loaded
        if self._outcome_learner and not self._outcome_learner.is_loaded:
            await self._load_outcome_learner_history()

        # Lazy load emotional field from persistence if not yet loaded
        if not self._emotional_field_loaded:
            self._emotional_field = await self._load_emotional_field()
            self._emotional_field_loaded = True

        # Lazy load Plutchik steering registry from Psyche if not yet loaded
        if not self._plutchik_registry_loaded:
            self._plutchik_registry = await self._load_plutchik_registry()
            self._plutchik_registry_loaded = True

        # Lazy load individuation dynamics from Psyche if not yet loaded
        if not self._individuation_dynamics_loaded:
            self._individuation_dynamics = await self._load_individuation_dynamics()
            self._individuation_dynamics_loaded = True

        # ==========================================
        # META-LEARNING TRIGGER CHECK
        # ==========================================
        # Check for meta-learning trigger
        state = await self._maybe_trigger_judgment_review(state)

        # ==========================================
        # EPISODE LIFECYCLE CHECK
        # ==========================================
        # Check if we should start a new episode (no active episode or current is saturated)
        if self._episode_orchestrator.should_start_new_episode(state):
            # Use previous thought as opening insight for episode
            opening_insight = state.thought or "Beginning a new exploration"
            # Get affect state for emotion-modulated episode selection
            current_affect = self._get_current_affect(state)
            state, new_episode = self._episode_orchestrator.start_episode(
                state=state,
                opening_insight=opening_insight,
                affect_state=current_affect,
            )
            logger.info(
                f"[EPISODE] Started new episode: type={new_episode.episode_type.value}, "
                f"segment={new_episode.current_segment.value}"
            )

        phase_timing: dict[str, float] = {}
        thought: Optional[str] = None
        activations: Optional[ActivationSummary] = None
        sae_features: list[SAEFeature] = []
        curation: Optional[CurationResult] = None
        surprise_score: float = 0.0
        epistemic_confidence: float = 0.5  # HALT probe result (default neutral)
        halt_probe_latency_ms: float = 0.0

        # Steering alignment feedback (observability feature)
        activation_for_alignment: Optional[np.ndarray] = None
        steering_alignment: Optional[float] = None

        # Start tracking audio files for this cycle (for rolling archive)
        if self._liquidsoap:
            self._liquidsoap.start_cycle_tracking()

        # Query effective hypothesis vectors for steering (CPU-only, before GPU load)
        hypothesis_vectors = []
        try:
            hypothesis_vectors = await self._psyche.get_effective_hypothesis_vectors(
                min_effectiveness=self._settings.hypothesis_query_min_effectiveness,
                min_applications=self._settings.hypothesis_query_min_applications,
                limit=self._settings.hypothesis_query_limit,
            )
            if hypothesis_vectors:
                logger.info(
                    f"Retrieved {len(hypothesis_vectors)} effective hypothesis vectors "
                    f"for steering"
                )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Database connection error querying hypothesis vectors: {e}")
        except ValueError as e:
            logger.warning(f"Invalid hypothesis vector data: {e}")

        # ========================================
        # PHASE 1: GENERATION (TransformerLens)
        # ========================================
        await self._update_visualizer(phase="generation", status="loading model")

        try:
            start = time.time()

            # Defensive GPU cleanup before loading TransformerLens
            # Ensures any lingering vLLM memory from previous cycle is released
            if not self._hooked_model.is_loaded:
                self._cleanup_and_log_gpu_memory("HookedQwen load")

            # Ensure model is loaded
            if not self._hooked_model.is_loaded:
                logger.info("Loading HookedQwen for generation phase...")
                # Process narration BEFORE loading HookedQwen
                if self._process_narrator:
                    context = BridgeContext(
                        last_concepts=list(state.recent_concepts[:3]) if state.recent_concepts else [],
                        emotional_intensity=self._emotional_field.current_intensity,
                        emotional_valence=self._emotional_field.current_valence,
                    )
                    await self._process_narrator.narrate_before_load("generation", context)
                await self._hooked_model.load()
                if self._process_narrator:
                    await self._process_narrator.narrate_after_load("generation", success=True)

            # Extract steering vectors for hypotheses created in previous cycles
            # (deferred because model wasn't loaded during simulation phase)
            await self._extract_pending_hypothesis_vectors()

            # Sample emotional field for evoked context
            evoked_context = ""
            current_affect = self._get_current_affect(state)
            interference = self._emotional_field.sample(
                affect_state=current_affect,
                co_retrieved_memories=self._get_recent_zettel_uids(state),
                proximity_threshold=0.3,
                awareness_threshold=0.5,
            )

            logger.info(
                f"Sampled emotional field: intensity={interference.field_intensity:.2f}, "
                f"surfaced={len(interference.surfaced_memories)}"
            )

            # Narrate strong emotional resonance with contemplative voice
            if interference.surfaced_memories and interference.reactivation_strength > 0.6:
                evocation_text = self._describe_emotional_surfacing(
                    interference.surfaced_memories,
                    interference.field_intensity,
                )
                await self._narrate(evocation_text, voice=self._voice_curator)

            # Inject evoked memories into prompt context if any surfaced
            if interference.surfaced_memories:
                evoked_context = await self._format_evoked_memories(
                    interference.surfaced_memories
                )

            # Get affect vector for steering (applied after generation when activations available)
            affect_vector = current_affect.to_vector()

            # Build prompt from previous curation or use default
            prompt = await self._build_generation_prompt(state, evoked_context=evoked_context)

            # Spawn HALT probe task (runs in parallel with generation if probe enabled)
            halt_task = None
            if self._halt_probe and getattr(self._settings, "halt_probe_enabled", False):
                halt_task = asyncio.create_task(self._run_halt_probe(prompt))

            # Generate thought with activation capture
            # Use PLaT-Lite chunked generation if enabled for real-time latent observation
            await self._update_visualizer(status="generating thought")
            if self._plat_enabled and self._chunked_generator:
                logger.debug("[PLAT-LITE] Using chunked generation with latent observation")
                thought, raw_activations, mlp_activations, surprise_score, telemetry_summary = await self._generate_thought_plat_lite(
                    prompt, state, hypothesis_vectors=hypothesis_vectors
                )
            else:
                thought, raw_activations, mlp_activations, surprise_score, telemetry_summary = await self._generate_thought(
                    prompt, state, hypothesis_vectors=hypothesis_vectors
                )

            # Await HALT probe result
            halt_activations = None  # Hidden states for training
            if halt_task:
                try:
                    epistemic_confidence, halt_probe_latency_ms, halt_activations = await halt_task
                except Exception as e:
                    logger.warning(f"HALT probe task failed: {e}")
                    # Keep default values from function scope (line 695-696)

            # Capture hidden states for HALT training if collector is enabled
            if self._halt_collector and halt_activations is not None:
                try:
                    await self._halt_collector.capture_hidden_states(
                        cycle_id=state.cycle_id,
                        hidden_states=halt_activations,
                        probe_layer=self._settings.halt_probe_layer,
                    )
                    logger.debug(f"[HALT] Captured hidden states for cycle {state.cycle_id}")
                except Exception as e:
                    logger.warning(f"HALT training capture failed: {e}")

            # Update visualizer with generated thought
            if thought:
                # Truncate for display (keep full for narration)
                display_thought = thought[:300] + "..." if len(thought) > 300 else thought
                await self._update_visualizer(insight=display_thought, status="thinking")

            # Summarize activations for curator
            activations = self._summarize_activations(raw_activations)

            # Extract SAE features if transcoder available and SAE features enabled
            # (uses MLP input, not residual)
            if self._settings.sae_features_enabled and self._transcoder and mlp_activations:
                sae_features = await self._extract_sae_features(mlp_activations)
                # SAE features are used for insight relationships via shared feature
                # activation, not narrated (they are unlabeled connection points)

                # Update visualizer with feature labels
                if sae_features:
                    feature_labels = [f.label for f in sae_features if f.label][:8]
                    await self._update_visualizer(features=feature_labels)

                # Record tension observations between previous and current SAE features
                # Tensions are feature pairs that "flip" between thoughts with high surprise
                if self._tension_tracker and state.sae_features:
                    try:
                        # state.sae_features is already list[tuple[int, float]]
                        prev_features = state.sae_features
                        curr_features = [(f.feature_id, f.activation) for f in sae_features]
                        tensions = await self._tension_tracker.record_observation(
                            features_before=prev_features,
                            features_after=curr_features,
                            surprise=surprise_score,
                        )
                        if tensions:
                            logger.info(f"[TENSION] Recorded {len(tensions)} tension observations")
                    except Exception as e:
                        logger.warning(f"Tension tracking failed: {e}")

            # Narrate the thought (Lilly's voice - first person)
            # PLaT-Lite handles narration via CombinedNarrationCoordinator during generation
            # Otherwise use ProgressiveNarrator to stream chunks while curation runs
            if self._liquidsoap and thought and not self._plat_enabled:
                if self._progressive_narrator:
                    # Start progressive narration - chunks queue while curation processes
                    if not self._progressive_narrator.is_active:
                        await self._progressive_narrator.start()
                    await self._progressive_narrator.queue_thought(thought)
                else:
                    # Fallback to direct narration
                    await self._narrate(thought, voice=self._voice_subject)

            phase_timing["generation"] = time.time() - start
            logger.info(
                f"Generation phase complete: {len(thought)} chars, "
                f"surprise={surprise_score:.2f}"
            )
            # Log full thought for quality evaluation (INFO level for collection)
            logger.info(f"[THOUGHT] {thought}")

            # SS-401: Update Evalatis steerer with cycle data for crystallization/spawning/pruning
            # This enables the hybrid emergence-selection steering system to learn from outcomes
            has_update_method = hasattr(state.steerer, 'update_from_cycle')
            has_activations = raw_activations is not None
            logger.debug(f"[EVALATIS] Pre-check: has_update_method={has_update_method}, has_activations={has_activations}")

            if has_update_method and has_activations:
                try:
                    # Extract numpy array from raw activations for exploration zone (layers 4-8)
                    # Use REPRESENTATIVE_EXPLORATION_LAYER as representative exploration layer
                    exploration_layer = REPRESENTATIVE_EXPLORATION_LAYER
                    exploration_activations = None
                    available_layers = list(raw_activations.keys()) if isinstance(raw_activations, dict) else []
                    logger.debug(f"[EVALATIS] Looking for layer {exploration_layer} in {available_layers[:10]}...")

                    if exploration_layer in raw_activations:
                        act = raw_activations[exploration_layer]
                        if isinstance(act, torch.Tensor):
                            exploration_activations = act.detach().cpu().float().numpy().flatten()
                        elif isinstance(act, np.ndarray):
                            exploration_activations = act.flatten()
                        logger.debug(f"[EVALATIS] Extracted activations: shape={exploration_activations.shape if exploration_activations is not None else None}")
                    else:
                        logger.warning(f"[EVALATIS] Layer {exploration_layer} NOT in raw_activations! Available: {available_layers}")

                    if exploration_activations is not None:
                        events = state.steerer.update_from_cycle(
                            zone_name="exploration",
                            activations=exploration_activations,
                            surprise=surprise_score,
                            sae_features=state.sae_features if self._settings.sae_features_enabled else None,
                            phase="generation",
                        )
                        # Log emergent magnitude after update
                        if hasattr(state.steerer, 'zones'):
                            zone = state.steerer.zones.get("exploration")
                            if zone and hasattr(zone, 'emergent'):
                                mag = np.linalg.norm(zone.emergent.vector)
                                logger.info(f"[EVALATIS] After update_from_cycle: exploration emergent magnitude={mag:.6f}")

                        if crystal := events.get("crystallized"):
                            logger.info(f"[EVALATIS] Crystallized: {crystal.name} (QD={crystal.qd_score:.3f})")
                        if spawned := events.get("spawned"):
                            logger.info(f"[EVALATIS] Spawned child: {spawned.name}")
                        if pruned_info := events.get("pruned"):
                            logger.info(f"[EVALATIS] Pruned: {pruned_info}")
                    else:
                        logger.warning(f"[EVALATIS] exploration_activations is None, skipping update_from_cycle")
                except Exception as e:
                    logger.warning(f"Evalatis update_from_cycle failed: {e}")

            # Apply affect steering with activations for vector capture
            self._apply_affect_steering(
                state.steerer,
                affect_vector,
                exploration_activations,
                affect_state=current_affect,
            )

            # Store activation for alignment computation (if enabled)
            # Reuse exploration_activations extracted above for Evalatis
            if self._settings.alignment_enabled and exploration_activations is not None:
                activation_for_alignment = exploration_activations

        except Exception as e:
            logger.error(f"Generation phase failed: {e}")
            raise  # Generation is required

        finally:
            # Unload TransformerLens to free GPU
            await self._safe_unload_hooked_model()
            # Extra cleanup to ensure VRAM is released for curation phase
            self._cleanup_and_log_gpu_memory("HookedQwen unload")
            await asyncio.sleep(1.0)  # Allow CUDA to fully release before vLLM load

        # ========================================
        # PHASE 2: CURATION (vLLM)
        # ========================================
        await self._update_visualizer(phase="curation", status="analyzing")

        try:
            start = time.time()

            # Load curator model
            logger.info("Loading curator model...")
            # Process narration BEFORE loading curator (with thought context from generation)
            if self._process_narrator:
                # Extract concepts from the thought just generated
                thought_concepts = []
                if thought:
                    # Simple extraction: take first few significant words
                    words = [w for w in thought.split() if len(w) > 4][:5]
                    thought_concepts = words[:3] if words else []
                context = BridgeContext(
                    thought_concepts=thought_concepts,
                    surprise_level=surprise_score,
                    emotional_intensity=self._emotional_field.current_intensity,
                )
                await self._process_narrator.narrate_before_load("curation", context)

            # Defensive GPU cleanup before loading Curator vLLM
            # Ensures any lingering memory from previous cycle's Mox is fully released
            self._cleanup_and_log_gpu_memory("Curator load")

            await self._curator.load()
            if self._process_narrator:
                await self._process_narrator.narrate_after_load("curation", success=True)

            # Get the previous concept for diversity (most recent in recent_concepts)
            previous_concept = state.recent_concepts[-1] if state.recent_concepts else None

            # Run multi-turn curation loop
            accumulated_results: list["CurationResult"] = []

            for turn in range(MAX_CURATOR_TURNS):
                result = await self._curator.curate(
                    thought=thought,
                    activations=activations,
                    sae_features=sae_features,
                    episode=state.current_episode,
                    goal=state.current_goal,
                    tools=self._curator_tools,
                    enabled_tool_names=self._settings.curator_tools_enabled,
                    previous_concept=previous_concept,
                )

                accumulated_results.append(result)

                # Check if curation is complete (backwards compat: default to True)
                is_complete = getattr(result, "is_complete", True)
                if is_complete:
                    logger.info(f"Curator completed in {turn + 1} turn(s)")
                    break

                continuation_reason = getattr(result, "continuation_reason", None) or "unspecified"
                logger.info(f"Curator requesting turn {turn + 2}: {continuation_reason}")
            else:
                logger.warning(f"Curator reached max turns ({MAX_CURATOR_TURNS}), merging partial results")

            # Merge results from all turns
            curation = self._merge_curation_results(accumulated_results)

            phase_timing["curation"] = time.time() - start

            # Log detailed curation output for review
            logger.info(
                f"Curation phase complete: insight={bool(curation.analysis.insight)}, "
                f"zettel={curation.graph_ops.zettel is not None}"
            )
            logger.info(f"[CURATOR] Analysis insight: {curation.analysis.insight[:200] if curation.analysis.insight else 'None'}...")
            logger.info(f"[CURATOR] Analysis question: {curation.analysis.question[:150] if curation.analysis.question else 'None'}...")
            logger.info(f"[CURATOR] Concepts: {curation.analysis.concepts}")

            # Detect dialectical concept pairs using polarity detector
            # This identifies semantic oppositions for productive dialectical exploration
            if self._polarity_detector and curation.analysis.concepts:
                try:
                    dialectical_pairs = self._polarity_detector.get_dialectical_concepts(
                        curation.analysis.concepts
                    )
                    if dialectical_pairs:
                        logger.info(f"[POLARITY] Detected dialectical pairs: {dialectical_pairs}")
                except Exception as e:
                    logger.debug(f"Polarity detection skipped: {e}")

            # Log 8D Plutchik emotional state
            a = curation.analysis
            logger.info(
                f"[CURATOR] Affect 8D: joy={a.joy:.2f} trust={a.trust:.2f} fear={a.fear:.2f} "
                f"surprise={a.surprise:.2f} sadness={a.sadness:.2f} disgust={a.disgust:.2f} "
                f"anger={a.anger:.2f} anticipation={a.anticipation:.2f} (confidence={a.confidence:.2f})"
            )
            logger.info(f"[CURATOR] Next concept: {curation.next_prompt.concept}, framing: {curation.next_prompt.framing}")
            if curation.next_prompt.directive:
                logger.info(f"[CURATOR] Directive: {curation.next_prompt.directive[:150]}...")
            logger.info(f"[CURATOR] Graph ops: {len(curation.graph_ops.new_triples)} triples, {len(curation.graph_ops.entity_updates)} entity updates, {len(curation.graph_ops.belief_updates)} belief updates")

            # Compute semantic anchor similarities for interpretable cognitive mode classification
            # This maps the thought to ~10 cognitive mode anchors, producing scalar features
            # Reference: "7 Advanced Feature Engineering Tricks Using LLM Embeddings"
            anchor_service = await self._get_anchor_service()
            anchor_result = None
            if anchor_service and thought:
                try:
                    from core.cognitive.anchors import suggest_episode_type

                    anchor_result = await anchor_service.compute_similarities(thought)

                    # Log cognitive mode profile
                    top_3 = anchor_result.get_top_modes(3)
                    modes_str = ", ".join(f"{m}={s:.2f}" for m, s in top_3)
                    logger.info(f"[ANCHORS] Cognitive mode: {anchor_result.dominant_mode} ({anchor_result.dominant_score:.2f})")
                    logger.info(f"[ANCHORS] Top 3: {modes_str}")

                    # Suggest episode type based on dominant mode
                    suggested_episode = suggest_episode_type(anchor_result)
                    logger.info(f"[ANCHORS] Suggested episode type: {suggested_episode}")

                    # Check for orphan thoughts and potential new cognitive modes
                    # Orphans are thoughts that don't fit any existing anchor well
                    if self._anchor_discovery and anchor_result.thought_embedding is not None:
                        self._anchor_discovery.set_current_cycle(state.cycle_count)
                        orphan = await self._anchor_discovery.check_for_orphan(
                            thought=thought,
                            similarities=anchor_result,
                            thought_embedding=anchor_result.thought_embedding,
                            cycle_number=state.cycle_count,
                        )
                        if orphan:
                            await self._anchor_discovery.add_orphan(orphan)
                            logger.debug(
                                f"[ANCHOR_DISCOVERY] Orphan buffered, total: "
                                f"{self._anchor_discovery.orphan_count}"
                            )

                            # Check for crystallization of new anchor
                            new_anchor = await self._anchor_discovery.check_for_crystallization()
                            if new_anchor:
                                logger.info(
                                    f"[ANCHOR_DISCOVERY] New cognitive mode discovered: "
                                    f"'{new_anchor.mode_name}'"
                                )
                                # Narrate the discovery
                                if self._liquidsoap:
                                    await self._narrate(
                                        f"A new pattern emerges in her thinking: {new_anchor.description[:200]}",
                                        voice=self._voice_curator,
                                    )

                except Exception as e:
                    logger.warning(f"Anchor similarity computation skipped: {e}")

            # Update visualizer with curator's insight
            if curation.analysis.insight:
                insight_display = curation.analysis.insight[:300] + "..." if len(curation.analysis.insight) > 300 else curation.analysis.insight
                await self._update_visualizer(insight=insight_display, status="reflecting")

            # Narrate the curator's insight (ensures continuity even if model didn't narrate)
            # Curator voice - third person clinical observation
            if self._liquidsoap and curation.analysis.insight:
                if self._progressive_narrator:
                    # Queue insight with lower priority than generation
                    await self._progressive_narrator.queue_insight(curation.analysis.insight)
                else:
                    # Full narration without truncation
                    await self._narrate(f"She reflects: {curation.analysis.insight}", voice=self._voice_curator, tone_before=True)

            # Narrate retrieved context/memories if any were found
            if self._liquidsoap and curation.next_prompt.retrieved_context:
                # Filter to get readable text entries (not JSON/dict)
                readable_contexts = []
                for ctx in curation.next_prompt.retrieved_context:
                    # Handle both dict and string types
                    if isinstance(ctx, dict):
                        # Extract text from dict - try common field names
                        ctx_text = ctx.get('insight') or ctx.get('text') or ctx.get('content') or ctx.get('summary', '')
                    else:
                        ctx_text = str(ctx) if ctx else ''

                    # Skip if empty or looks like raw data (JSON-like format)
                    if ctx_text and not ctx_text.startswith('{') and not ctx_text.startswith('[') and "': " not in ctx_text:
                        readable_contexts.append(ctx_text)

                if readable_contexts:
                    context_count = len(readable_contexts)
                    if context_count == 1:
                        # Single memory - curator introduces, then content
                        readable = readable_contexts[0]
                        await self._narrate(
                            f"A connection surfaces in her memory: {readable}",
                            voice=self._voice_curator
                        )
                    else:
                        # Multiple memories - announce count, then narrate each
                        await self._narrate(
                            f"{context_count} related memories emerge.",
                            voice=self._voice_curator
                        )
                        # Narrate each memory with varied phrasing (no truncation)
                        for i, ctx in enumerate(readable_contexts[:5]):  # Cap at 5 to avoid excessive narration
                            # Vary the phrasing (third person clinical style)
                            if i == 0:
                                prefix = "The first recalls:"
                            elif i == 1:
                                prefix = "Another surfaces:"
                            elif i == 2:
                                prefix = "A third emerges:"
                            elif i == 3:
                                prefix = "Additionally:"
                            else:
                                prefix = "And:"
                            await self._narrate(f"{prefix} {ctx}", voice=self._voice_curator)

            # Compute faithfulness score (activation-verbal consistency)
            # Only when SAE features are enabled and validator is available
            if self._settings.sae_features_enabled and sae_features and self.faithfulness_validator:
                try:
                    faithfulness = self.faithfulness_validator.compute_faithfulness_from_labels(
                        analysis=curation.analysis,
                        sae_features=sae_features,
                    )

                    # Log if significant divergence detected
                    self.faithfulness_validator.log_divergence(faithfulness, state.cycle_count)

                    # Update curation result with faithfulness score
                    curation = replace(curation, faithfulness=faithfulness)

                    logger.info(
                        f"[FAITHFULNESS] overlap={faithfulness.overlap_ratio:.2f}, "
                        f"severity={faithfulness.divergence_severity}, "
                        f"unsupported={faithfulness.unsupported_claims}"
                    )

                    # Record HALT training example based on faithfulness
                    # High overlap (>=0.8) = reliable generation (label=1.0)
                    # Low overlap (<0.3) = unreliable generation (label=0.0)
                    # Medium range is skipped (doesn't provide clear signal)
                    if self._halt_collector:
                        try:
                            if faithfulness.overlap_ratio >= 0.8:
                                await self._halt_collector.record_example(
                                    cycle_id=state.cycle_id,
                                    label=1.0,
                                    label_source="faithfulness_high",
                                    thought_preview=thought[:200] if thought else "",
                                    insight_preview=curation.analysis.insight[:200] if curation.analysis.insight else "",
                                    faithfulness_score=faithfulness.overlap_ratio,
                                )
                                logger.debug(f"[HALT] Recorded positive example from faithfulness_high for cycle {state.cycle_id}")
                            elif faithfulness.overlap_ratio < 0.3:
                                await self._halt_collector.record_example(
                                    cycle_id=state.cycle_id,
                                    label=0.0,
                                    label_source="faithfulness_low",
                                    thought_preview=thought[:200] if thought else "",
                                    insight_preview=curation.analysis.insight[:200] if curation.analysis.insight else "",
                                    faithfulness_score=faithfulness.overlap_ratio,
                                )
                                logger.debug(f"[HALT] Recorded negative example from faithfulness_low for cycle {state.cycle_id}")
                            # Medium range (0.3 to 0.8) is skipped - ambiguous signal
                        except Exception as e:
                            logger.warning(f"HALT training record failed: {e}")

                except Exception as e:
                    logger.warning(f"Faithfulness validation failed: {e}")
                    # Continue without faithfulness score - curation is still valid

            # Observe in Feature Substrate (emergent cognition layer)
            # Only when SAE features are enabled
            if self._settings.sae_features_enabled and self._substrate and sae_features:
                try:
                    # Convert SAEFeature to SubstrateFeatureActivation
                    substrate_features = [
                        SubstrateFeatureActivation(
                            feature_idx=f.feature_id,
                            activation=f.activation,
                        )
                        for f in sae_features
                    ]
                    # Compute insight signal (1.0 if zettel created, 0.0 otherwise)
                    insight_signal = 1.0 if (curation and curation.graph_ops.zettel) else 0.0

                    value = await self._substrate.observe(
                        substrate_features,
                        surprise=surprise_score,
                        insight=insight_signal,
                    )
                    logger.debug(
                        f"[SUBSTRATE] Observed {len(substrate_features)} features, "
                        f"surprise={surprise_score:.2f}, insight={insight_signal:.1f}, value={value:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Substrate observation failed: {e}")
                    # Continue without substrate - not critical

            # Deposit emotional trace into the wave packet field
            # This captures the affective quality of this cognitive cycle
            if curation and curation.analysis:
                try:
                    # Compute divergence from faithfulness score
                    divergence = 0.0
                    if curation.faithfulness:
                        divergence = 1.0 - curation.faithfulness.overlap_ratio

                    # Build affect state from curation signals
                    affect_state = self._build_affect_from_curation(
                        curation,
                        state,
                        divergence,
                    )

                    # Collect anchor memories (zettel UID if created)
                    anchor_memories = []
                    if curation.graph_ops and curation.graph_ops.zettel:
                        # ZettelData doesn't have uid yet - it gets assigned during persistence
                        # Use source_thought hash as temporary anchor
                        zettel_anchor = f"zettel:{hashlib.sha256(curation.graph_ops.zettel.insight.encode()).hexdigest()[:16]}"
                        anchor_memories.append(zettel_anchor)

                    # Deposit emotional trace
                    self._emotional_field.deposit(
                        affect_state=affect_state,
                        anchor_memories=anchor_memories,
                    )
                    logger.info(
                        f"[AFFECT] Deposited emotional trace: intensity={affect_state.intensity():.2f}, "
                        f"valence={affect_state.valence:.2f}, anchors={len(anchor_memories)}"
                    )
                except Exception as e:
                    logger.warning(f"Emotional field deposit failed: {e}")
                    # Continue without deposit - not critical

        except Exception as e:
            logger.error(f"Curation phase failed: {e}, using fallback")
            logger.error(f"Curation traceback:\n{traceback.format_exc()}")
            curation = self._fallback_curation(thought, state)
            phase_timing["curation_fallback"] = time.time() - start

        finally:
            # Flush any buffered discovery narrations before exiting curation phase
            await self._curator_tools.flush_narrations()
            await self._safe_unload_curator()
            # GPU memory cleanup - mirrors generation phase pattern
            # vLLM may run in separate process; explicit cleanup ensures memory release
            self._cleanup_and_log_gpu_memory("vLLM unload")
            # Allow CUDA to fully release before next phase loads models
            await asyncio.sleep(VLLM_UNLOAD_DELAY_S)

        # ==========================================
        # GOAL DETECTION (after curation, before simulation)
        # ==========================================
        # Detect emerging inquiry goals from the thought and curator's analysis
        # Goals are detected from GOAL_SIGNALS patterns (e.g., "I wonder", "I don't understand")
        try:
            # Use curator's question if available, otherwise use thought
            goal_source = curation.analysis.question or curation.analysis.insight or thought
            detected_goal = detect_emerging_goal(
                thought=goal_source,
                current_goal=state.current_goal,
            )
            if detected_goal:
                state = replace(state, current_goal=detected_goal)
                logger.info(
                    f"[GOAL] Detected emerging inquiry goal: {detected_goal.question[:100]}... "
                    f"(confidence threshold met, stage={detected_goal.stage.value})"
                )
        except Exception as e:
            logger.warning(f"Goal detection failed: {e}")
            # Continue without goal - not critical

        # ==========================================
        # EPISODE SEGMENT ADVANCEMENT
        # ==========================================
        # Advance the episode segment based on this cycle's curation output
        if state.current_episode is not None:
            try:
                segment_output = curation.analysis.insight or thought or ""
                state, next_segment = self._episode_orchestrator.advance_segment(
                    state=state,
                    segment_output=segment_output,
                )
                logger.info(
                    f"[EPISODE] Advanced to segment: {next_segment.value}, "
                    f"episode_type={state.current_episode.episode_type.value}"
                )
                # IP-201 fix: End episode when CLOSING segment is reached
                if next_segment == SegmentType.CLOSING:
                    state, ended_episode = self._episode_orchestrator.end_episode(state)
                    if ended_episode:
                        logger.info(
                            f"[EPISODE] Ended episode {ended_episode.episode_type.value} "
                            f"with {len(ended_episode.segments_completed)} segments"
                        )
            except Exception as e:
                logger.warning(f"Episode segment advancement failed: {e}")
                # Continue without advancement - not critical

        # ==========================================
        # PHASE 2.5: SIMULATION (Graph-Preflexor)
        # ==========================================
        simulation_result: Optional["SimulationResult"] = None
        try:
            # Capture fresh metrics BEFORE simulation for prediction baselines
            # (Previously captured in Integration phase, after simulation ran)
            if (not state.metrics_snapshot or state.metrics_snapshot.get('cycle') != state.cycle_count) and self._settings.simulation_enabled:
                metrics_snapshot = await self._capture_cycle_metrics(state.cycle_count)
                state = replace(state, metrics_snapshot=metrics_snapshot.to_dict())
                logger.debug(f"Cycle {state.cycle_count}: metrics_snapshot captured with {len(state.metrics_snapshot)} fields")

            if self._settings.simulation_enabled and self._simulation_engine:
                await self._update_visualizer(phase="simulation", status="hypothesizing")

                # Narrate phase transition: curation -> simulation
                await self._narrate_phase_transition(
                    "curation", "simulation", state.cycle_count
                )

                # Check for queued follow-up simulations first
                if state.queued_follow_up_simulations:
                    hypothesis_uid = state.queued_follow_up_simulations[0]
                    hypothesis = await self._psyche.get_hypothesis(hypothesis_uid)

                    if hypothesis:
                        if self._liquidsoap:
                            # Experimenter voice for simulation phase
                            await self._narrate(
                                f"A prediction has resolved. Reconsidering hypothesis: {hypothesis.statement}",
                                voice=self._voice_experimenter,
                            )

                        # Load preflexor and run follow-up simulation
                        # Process narration BEFORE loading Preflexor
                        if self._process_narrator:
                            context = BridgeContext(
                                hypothesis_statement=hypothesis.statement if hypothesis else None,
                            )
                            await self._process_narrator.narrate_before_load("simulation", context)

                        # Defensive GPU cleanup before loading Preflexor
                        # Ensures vLLM curator subprocess is fully terminated
                        self._cleanup_and_log_gpu_memory("Preflexor load")

                        await self._preflexor.load()
                        if self._process_narrator:
                            await self._process_narrator.narrate_after_load("simulation", success=True)

                        simulation_result = await self._simulation_engine.simulate_follow_up(
                            hypothesis=hypothesis,
                            prediction_outcomes=state.recent_prediction_outcomes,
                            cycle=state.cycle_count,
                            metrics_snapshot=state.metrics_snapshot,
                        )

                        phase_timing["simulation_followup"] = time.time() - start

                # Regular simulation trigger (if no follow-up)
                elif self._simulation_engine.should_simulate(curation, state=state):
                    start = time.time()
                    # Process narration BEFORE loading Preflexor
                    # Extract hypothesis hint from curation if available
                    hypothesis_hint = None
                    if curation and curation.simulation_hint:
                        hypothesis_hint = curation.simulation_hint.hypothesis
                    if self._process_narrator:
                        context = BridgeContext(
                            hypothesis_statement=hypothesis_hint,
                        )
                        await self._process_narrator.narrate_before_load("simulation", context)

                    # Defensive GPU cleanup before loading Preflexor
                    # Ensures vLLM curator subprocess is fully terminated
                    self._cleanup_and_log_gpu_memory("Preflexor load")

                    await self._preflexor.load()
                    if self._process_narrator:
                        await self._process_narrator.narrate_after_load("simulation", success=True)

                    simulation_result = await self._simulation_engine.simulate(
                        curation=curation,
                        thought=thought,
                        cycle=state.cycle_count,
                        metrics_snapshot=state.metrics_snapshot,
                    )

                    phase_timing["simulation"] = time.time() - start

                    if simulation_result and simulation_result.hypotheses:
                        # Experimenter voice for hypothesis formation
                        await self._narrate(
                            f"Through simulation, a hypothesis forms: {simulation_result.hypotheses[0].statement}",
                            voice=self._voice_experimenter,
                        )

        except Exception as e:
            logger.error(f"Simulation phase failed: {e}")
            # Continue without simulation - it's optional

        finally:
            await self._safe_unload_simulation()

        # Narrate phase transition: simulation -> integration (only if simulation ran)
        if simulation_result is not None:
            await self._narrate_phase_transition(
                "simulation", "integration", state.cycle_count
            )

        # ========================================
        # PHASE 3: INTEGRATION (Golden Embeddings)
        # ========================================
        await self._update_visualizer(phase="integration", status="crystallizing")

        try:
            start = time.time()

            # Process narration BEFORE loading embedder
            discoveries_count = self._calculate_discoveries_count(curation)
            if self._process_narrator:
                context = BridgeContext(
                    discoveries_count=discoveries_count,
                )
                await self._process_narrator.narrate_before_load("integration", context)

            # Ensure golden embedder is loaded
            await self._embedder.ensure_golden_loaded()
            if self._process_narrator:
                await self._process_narrator.narrate_after_load("integration", success=True)

            # Embed thought
            thought_embedding = await self._embed_thought(thought)

            # Compute steering alignment (observability feature)
            # Measures how well output aligned with steering intent
            if self._settings.alignment_enabled and activation_for_alignment is not None:
                # Get exploration zone steering vector using get_vector interface
                steering_vector = None
                if hasattr(state.steerer, 'get_vector'):
                    steering_vector = state.steerer.get_vector(REPRESENTATIVE_EXPLORATION_LAYER)

                if steering_vector is not None:
                    steering_alignment = compute_steering_alignment(
                        activation_for_alignment,
                        steering_vector,
                    )
                    if steering_alignment is not None:
                        logger.info(f"Steering alignment: {steering_alignment:.3f}")

            # Create zettel if curator produced one
            if curation.graph_ops.zettel:
                await self._create_zettel(curation, thought_embedding, state)

                # Update visualizer with crystallized insight
                zettel_insight = curation.graph_ops.zettel.insight
                if zettel_insight:
                    await self._update_visualizer(
                        insight=zettel_insight[:300] + "..." if len(zettel_insight) > 300 else zettel_insight,
                        status="insight formed"
                    )

                # Narrate the crystallization of insight (curator voice - third person)
                if self._liquidsoap:
                    zettel_insight = curation.graph_ops.zettel.insight
                    if zettel_insight:
                        await self._narrate(
                            f"A new insight crystallizes in her mind: {zettel_insight}",
                            voice=self._voice_curator
                        )

            # Persist graph operations
            await self._persist_graph_ops(curation)

            # Capture cycle metrics if not already done in Simulation phase
            if not state.metrics_snapshot or state.metrics_snapshot.get('cycle') != state.cycle_count:
                metrics_snapshot = await self._capture_cycle_metrics(state.cycle_count)
                metrics_dict = metrics_snapshot.to_dict()
                state = replace(state, metrics_snapshot=metrics_dict)
            else:
                metrics_dict = state.metrics_snapshot

            # Persist simulation results if any (with baseline metrics for predictions)
            if simulation_result:
                await self._persist_simulation_results(
                    simulation_result, state, metrics_dict
                )

            # Narrate when new knowledge is being stored (curator voice - third person)
            if self._liquidsoap:
                triple_count = len(curation.graph_ops.new_triples)
                belief_count = len(curation.graph_ops.belief_updates)

                if triple_count > 0:
                    # Narrate the most interesting triple
                    first_triple = curation.graph_ops.new_triples[0]
                    # Note: schema uses object_ to avoid Python keyword conflict
                    if hasattr(first_triple, 'subject') and hasattr(first_triple, 'predicate') and hasattr(first_triple, 'object_'):
                        rel_desc = f"{first_triple.subject} {first_triple.predicate.lower().replace('_', ' ')} {first_triple.object_}"
                        if triple_count == 1:
                            await self._narrate(
                                f"She records a new connection: {rel_desc}.",
                                voice=self._voice_curator
                            )
                        else:
                            await self._narrate(
                                f"She records {triple_count} new connections, including: {rel_desc}.",
                                voice=self._voice_curator
                            )
                elif belief_count > 0:
                    await self._narrate(
                        f"Her beliefs shift: {belief_count} update{'s' if belief_count > 1 else ''}.",
                        voice=self._voice_curator
                    )

            # Run HippoRAG if enabled
            if self._settings.hipporag_enabled:
                await self._run_hipporag(thought)

            # Persist metrics to graph for historical analysis
            await self._psyche.save_cycle_metrics(
                cycle=state.cycle_count,
                metrics=metrics_dict,
            )

            # Also persist with update_cycle_metrics for additional fields
            # Compute additional metrics not in MetricsSnapshot
            thought_length = len(thought) if thought else 0
            sae_feature_count = len(state.sae_features) if state.sae_features else 0

            # Query concept and zettel counts from graph
            try:
                concept_count, zettel_count = await asyncio.gather(
                    self._psyche.count_nodes("Entity"),
                    self._psyche.count_nodes("InsightZettel"),
                )
            except Exception as e:
                logger.warning(f"Could not query concept/zettel counts: {e}")
                concept_count = None
                zettel_count = None

            # Build extended metrics dict with additional fields
            extended_metrics = {
                **metrics_dict,
                "concept_count": concept_count,
                "zettel_count": zettel_count,
                "thought_length": thought_length,
                "sae_feature_count": sae_feature_count,
            }

            await self._psyche.update_cycle_metrics(
                cycle_number=state.cycle_count,
                metrics=extended_metrics,
            )

            # Persist current goal if exists
            if state.current_goal:
                success, updated_goal = await self._persist_cognitive_object(
                    obj=state.current_goal,
                    name="InquiryGoal",
                    create_fn=self._psyche.create_inquiry_goal,
                    update_fn=self._psyche.update_inquiry_goal,
                )
                if success and updated_goal:
                    state = replace(state, current_goal=updated_goal)

            # Persist current episode if exists
            if state.current_episode:
                success, updated_episode = await self._persist_cognitive_object(
                    obj=state.current_episode,
                    name="Episode",
                    create_fn=self._psyche.create_episode,
                    update_fn=self._psyche.update_episode,
                )
                if success and updated_episode:
                    state = replace(state, current_episode=updated_episode)

            # Persist tension tracker dirty entries
            if self._tension_tracker:
                try:
                    persisted = await self._tension_tracker.persist_dirty()
                    if persisted:
                        logger.info(f"[TENSION] Persisted {persisted} tension entries to graph")
                except Exception as e:
                    logger.warning(f"Tension persistence failed: {e}")

            # Log cross-zone coherence for timescale integration monitoring
            self._log_coherence(state, phase="integration")

            phase_timing["integration"] = time.time() - start
            logger.info("Integration phase complete")

        except Exception as e:
            logger.error(f"Integration phase failed: {e}\n{traceback.format_exc()}")
            # Try minimal persistence
            await self._minimal_persist(thought)
            phase_timing["integration_fallback"] = time.time() - start

        finally:
            await self._safe_unload_embedder()

        # ========================================
        # PHASE 4: REFLEXION (Self-monitoring and modification)
        # ========================================
        reflexion_result: Optional[ReflexionResult] = None
        if self._settings.reflexion_enabled and self._reflexion_phase:
            # Narrate phase transition: integration -> reflexion
            await self._narrate_phase_transition(
                "integration", "reflexion", state.cycle_count
            )

            await self._update_visualizer(phase="reflexion", status="reflecting")

            try:
                start = time.time()

                reflexion_result = await self._reflexion_phase.run(state)

                phase_timing["reflexion"] = time.time() - start

                logger.info(
                    f"Reflexion phase complete: "
                    f"{reflexion_result.health_assessment.worst_category.value}, "
                    f"{len(reflexion_result.modifications)} modifications"
                )

                # Surface individuation dynamics insights during reflexion
                dynamics_summary = self._individuation_dynamics.get_summary()
                if dynamics_summary.get("transforming_elements"):
                    logger.info(
                        f"[DYNAMICS] Identity state: {dynamics_summary['overall_state']}, "
                        f"transforming: {len(dynamics_summary['transforming_elements'])} elements"
                    )
                    for transition in dynamics_summary.get("recent_transitions", [])[:2]:
                        if transition.get("narrative"):
                            logger.info(f"[DYNAMICS] {transition['narrative']}")

            except Exception as e:
                logger.error(f"Reflexion phase failed: {e}")
                # Non-fatal - continue without reflexion
                phase_timing["reflexion_error"] = time.time() - start

        # ========================================
        # QD WEIGHT ADAPTATION (ATP-Latent inspired online learning)
        # ========================================
        # Adapt QD weights based on cycle outcomes for proactive diversity
        if (
            hasattr(state.steerer, "qd_scorer")
            and state.steerer.qd_scorer is not None
            and metrics_snapshot is not None
        ):
            try:
                outcomes = {
                    "H_sem": metrics_snapshot.semantic_entropy,
                    "D": metrics_snapshot.discovery_parameter,
                    "verification_rate": metrics_snapshot.verification_rate,
                }
                
                # Extract health status for adaptive throttling (TNGD-inspired)
                health_status = None
                if reflexion_result and reflexion_result.health_assessment:
                    health_status = reflexion_result.health_assessment.worst_category.value
                
                deltas = state.steerer.qd_scorer.adapt_weights(
                    outcomes, health_status=health_status
                )
                if deltas:  # Only log if adaptation happened
                    log_msg = f"QD weights adapted: {state.steerer.qd_scorer.config.get_weights()}"
                    if health_status:
                        log_msg += f", health={health_status}"
                    logger.info(log_msg)
            except Exception as e:
                logger.warning(f"QD weight adaptation failed: {e}")
                # Non-fatal - continue with current weights

        # ========================================
        # EXPERIMENTATION: Consume proposals and tick active experiments
        # ========================================
        # Check for experiment proposals from curator
        if curation and curation.experiment_proposal:
            await self._maybe_start_experiment(curation.experiment_proposal)

        # Check for experiment proposals from reflexion
        if reflexion_result and reflexion_result.experiment_proposal:
            await self._maybe_start_experiment(reflexion_result.experiment_proposal)

        # Check for experiment proposals from simulation (hypothesis-to-experiment conversion)
        if simulation_result and simulation_result.experiment_proposals:
            for proposal_dict in simulation_result.experiment_proposals:
                try:
                    proposal = ExperimentProposal.from_dict(proposal_dict)
                    await self._filter_and_start_experiment(proposal, source="simulation")
                except Exception as e:
                    logger.warning(f"Failed to process simulation experiment proposal: {e}")

        # Tick active experiments each cycle (records metrics, handles phase transitions)
        if self._experiment_manager:
            try:
                # Reuse metrics_snapshot from integration phase (line ~1045)
                await self._experiment_manager.tick(state.cycle_count, metrics_snapshot)
            except Exception as e:
                logger.warning(f"Experiment tick failed: {e}")
                # Non-fatal - continue without experiment processing

        # ========================================
        # WEAVER: Semantic diversity management
        # ========================================
        # Run weaver tick to check for STAGNATION and generate interventions
        if self._weaver:
            try:
                intervention = await self._weaver.tick()
                weaver_state = self._weaver.get_state()
                logger.info(
                    f"Weaver tick complete: policy={weaver_state.current_policy.value}, "
                    f"discovery={weaver_state.discovery.state.value if weaver_state.discovery else 'None'}, "
                    f"D={weaver_state.discovery.discovery_parameter if weaver_state.discovery else 'N/A':.3f}, "
                    f"intervention={'generated' if intervention else 'none'}"
                )
                if intervention:
                    self._pending_weaver_intervention = intervention
                    logger.info(
                        f"Weaver intervention: {intervention.intervention_type.value} - "
                        f"{intervention.prompt[:100]}..."
                    )
                    # Narrate the intervention for awareness
                    if self._liquidsoap:
                        await self._narrate(
                            f"Weaver intervention: {intervention.prompt}",
                            voice=self._voice_curator,
                        )
            except Exception as e:
                logger.warning(f"Weaver tick failed: {e}")
                # Non-fatal - continue without weaver intervention

        # ========================================
        # PHASE 5: CONTINUITY (Mox meta-cognitive synthesis)
        # ========================================
        await self._update_visualizer(phase="continuity", status="synthesizing")

        synthesis: Optional[MoxSynthesis] = None
        try:
            start = time.time()

            # Build CycleRecap from this cycle's data (including reflexion)
            recap = CycleRecap(
                starting_concept=state.last_concept or "emergence",
                thought=thought or "",
                key_discoveries=[curation.analysis.insight] if curation and curation.analysis.insight else [],
                insights_formed=[curation.graph_ops.zettel.insight] if curation and curation.graph_ops.zettel else [],
                beliefs_updated=[
                    (b.topic, "strengthened" if b.confidence_delta > 0 else "weakened", b.confidence_delta)
                    for b in (curation.graph_ops.belief_updates if curation else [])
                ],
                hypotheses_tested=[
                    (h.statement[:100], "proposed")
                    for h in (simulation_result.hypotheses if simulation_result else [])
                ],
                open_threads=[curation.analysis.question] if curation and curation.analysis.question else [],
                health_status=(
                    reflexion_result.health_assessment.worst_category
                    if reflexion_result else None
                ),
                modifications_applied=[
                    ModificationEntry(
                        parameter_path=m.parameter_path,
                        old_value=str(m.old_value),
                        new_value=str(m.new_value),
                    )
                    for m in (reflexion_result.modifications if reflexion_result else [])
                ],
                cycle_number=state.cycle_count,
            )

            # Track consecutive empty cycles
            if not recap.is_meaningful():
                self._empty_cycle_count += 1
                if self._empty_cycle_count >= EMPTY_CYCLE_WARNING_THRESHOLD:
                    logger.warning(
                        f"{EMPTY_CYCLE_WARNING_THRESHOLD} consecutive empty cycles - consider topic shift"
                    )
            else:
                self._empty_cycle_count = 0

            # Run Mox synthesis if meaningful and Mox is configured
            if recap.is_meaningful() and self._mox and self._settings.continuity_enabled:
                # Process narration BEFORE loading Mox
                discoveries_count = self._calculate_discoveries_count(curation)
                if self._process_narrator:
                    context = BridgeContext(
                        discoveries_count=discoveries_count,
                    )
                    await self._process_narrator.narrate_before_load("continuity", context)

                # Fetch developmental guidance from Letta (sync, with timeout)
                developmental_context: Optional[str] = None
                if self._letta_continuity:
                    guidance = await self._letta_continuity.get_guidance()
                    if guidance and guidance.guidance:
                        developmental_context = guidance.guidance
                        logger.info(
                            f"Letta guidance received: {len(developmental_context)} chars, "
                            f"concerns={guidance.active_concerns}"
                        )
                        # Narrate that developmental guidance was received
                        if self._liquidsoap and guidance.active_concerns:
                            concerns_text = ", ".join(guidance.active_concerns[:LETTA_NARRATION_MAX_CONCERNS])
                            await self._narrate(
                                f"Developmental context received. Active concerns: {concerns_text}",
                                voice=self._voice_curator,
                            )

                # Add individuation dynamics to developmental context
                dynamics_summary = self._individuation_dynamics.get_summary()
                if dynamics_summary.get("transforming_elements") or dynamics_summary.get(
                    "recent_transitions"
                ):
                    # Build dynamics context text
                    dynamics_parts = []
                    dynamics_parts.append(
                        f"\n### Identity Dynamics (4th Layer)\n"
                        f"Overall state: {dynamics_summary.get('overall_state', 'developing')}"
                    )

                    # Phase distribution
                    phase_counts = dynamics_summary.get("phase_counts", {})
                    if phase_counts:
                        active_phases = [
                            f"{phase}: {count}"
                            for phase, count in phase_counts.items()
                            if count > 0
                        ]
                        if active_phases:
                            dynamics_parts.append(f"Phases: {', '.join(active_phases)}")

                    # Transforming elements
                    transforming = dynamics_summary.get("transforming_elements", [])
                    if transforming:
                        dynamics_parts.append(
                            f"Actively transforming ({len(transforming)}): "
                            f"{', '.join(transforming[:3])}"
                            + ("..." if len(transforming) > 3 else "")
                        )

                    # Recent transitions (with narratives)
                    transitions = dynamics_summary.get("recent_transitions", [])
                    if transitions:
                        dynamics_parts.append("Recent transitions:")
                        for t in transitions[:2]:
                            if t.get("narrative"):
                                dynamics_parts.append(f"  - {t['narrative']}")

                    # Strongest attractors
                    attractors = dynamics_summary.get("strongest_attractors", [])
                    if attractors:
                        attractor_strs = [
                            f"{a['basin_id']} (strength={a['strength']:.2f})"
                            for a in attractors[:2]
                        ]
                        dynamics_parts.append(f"Stable attractors: {', '.join(attractor_strs)}")

                    dynamics_text = "\n".join(dynamics_parts)

                    # Combine with existing developmental context
                    if developmental_context:
                        developmental_context = developmental_context + "\n" + dynamics_text
                    else:
                        developmental_context = dynamics_text

                    logger.debug(
                        f"[DYNAMICS] Added to Mox context: state={dynamics_summary.get('overall_state')}, "
                        f"transforming={len(transforming)}, transitions={len(transitions)}"
                    )

                # Load Mox model
                logger.info("Loading Mox model for continuity synthesis...")
                await self._mox.load()
                if self._process_narrator:
                    await self._process_narrator.narrate_after_load("continuity", success=True)

                # Run Mox synthesis with developmental context
                synthesis = await synthesize_with_mox(
                    recap, self._mox, developmental_context=developmental_context
                )

                if synthesis and synthesis.has_content() and self._liquidsoap:
                    # Narrate Mox's significance assessment (curator voice, third person)
                    if synthesis.significance:
                        await self._narrate(
                            f"Reflecting on this cycle: {synthesis.significance}",
                            voice=self._voice_curator,
                            tone_before=True,
                        )

                    # Narrate the seed for next cycle (subject voice, first person)
                    if synthesis.seed:
                        await asyncio.sleep(0.3)
                        await self._narrate(
                            f"What draws me forward: {synthesis.seed}",
                            voice=self._voice_subject,
                        )

                    # Log synthesis details
                    logger.info(
                        f"Mox synthesis: {len(synthesis.threads)} threads, "
                        f"{len(synthesis.tensions)} tensions, "
                        f"seed='{synthesis.seed[:50]}...'"
                    )

                # Narrate steering alignment (observability feature)
                if self._settings.alignment_enabled and self._liquidsoap:
                    await self._narrate(
                        describe_steering_alignment(steering_alignment),
                        voice=self._voice_curator,
                    )

                # Process experiment proposals from Mox synthesis
                if synthesis and synthesis.experiment_proposals:
                    await self._process_mox_experiment_proposals(synthesis.experiment_proposals)

                # Generate new narration phrases periodically (while Mox is loaded)
                phrase_result = await maybe_generate_phrases(
                    cycle=state.cycle_count,
                    psyche=self._psyche,
                    mox_model=self._mox,
                    current_concept=recap.starting_concept,
                )
                if phrase_result.phrases_generated > 0 and self._liquidsoap:
                    await self._narrate(
                        f"Generated {phrase_result.phrases_generated} new phrases for my voice.",
                        voice=self._voice_curator,
                    )

            # Fallback to curator-based recap if Mox unavailable
            elif recap.is_meaningful() and self._liquidsoap:
                # Generate curator recap (natural language fallback)
                curator_recap_text = await generate_curator_recap(recap, self._curator)
                if curator_recap_text:
                    await self._narrate(curator_recap_text, voice=self._voice_curator, tone_before=True)
                    await asyncio.sleep(0.5)
                    subject_recap_text = await generate_subject_recap(recap, self._curator)
                    if subject_recap_text:
                        await self._narrate(subject_recap_text, voice=self._voice_subject)

            # Send cycle summary to Letta and wait for feedback
            letta_feedback_text: Optional[str] = None
            if self._letta_continuity and self._settings.letta_continuity_enabled:
                # Build summary from available data
                metrics = state.metrics_snapshot or {}
                # Use actual emotional field values, not synthetic _get_current_affect
                field_valence, field_arousal = self._emotional_field.current_affect_summary()
                cycle_summary = CycleSummaryForLetta(
                    cycle_number=state.cycle_count,
                    discoveries_count=len(recap.key_discoveries),
                    insights_formed=[
                        z[:LETTA_INSIGHT_TITLE_MAX_LENGTH]
                        for z in recap.insights_formed[:LETTA_MAX_INSIGHTS]
                    ],
                    hypothesis_outcomes=[
                        (stmt[:LETTA_HYPOTHESIS_STMT_MAX_LENGTH], outcome)
                        for stmt, outcome in recap.hypotheses_tested[:LETTA_MAX_HYPOTHESES]
                    ],
                    beliefs_changed=len(recap.beliefs_updated),
                    health_status=recap.health_status.value if recap.health_status else "STABLE",
                    valence=field_valence,
                    arousal=field_arousal,
                    discovery_parameter=metrics.get("discovery_parameter", 0.0),
                    semantic_entropy=metrics.get("semantic_entropy", 0.0),
                    verification_rate=metrics.get("verification_rate", 0.0),
                )
                # Wait for Letta's response so feedback applies to current cycle
                letta_feedback = await self._letta_continuity.send_cycle_summary(cycle_summary)
                if letta_feedback and letta_feedback.guidance:
                    logger.info(f"Letta feedback received: {letta_feedback.guidance[:100]}...")
                    letta_feedback_text = letta_feedback.guidance

            # Update continuity context for next cycle (with synthesis, diversity prompt, and Letta feedback)
            diversity_prompt = (
                self._pending_weaver_intervention.prompt
                if self._pending_weaver_intervention else None
            )
            state.continuity_context.update(
                recap, synthesis,
                diversity_prompt=diversity_prompt,
                letta_feedback=letta_feedback_text,
            )
            # Clear the pending intervention after applying
            self._pending_weaver_intervention = None

            phase_timing["continuity"] = time.time() - start
            logger.info("Continuity phase complete")

        except Exception as e:
            logger.warning(f"Continuity phase failed: {e}")
            # Non-fatal - continue without synthesis

        finally:
            await self._safe_unload_mox()
            # GPU memory cleanup - Mox uses vLLM, must release before metacognition loads Gemma
            self._cleanup_and_log_gpu_memory("Mox unload")
            # Allow CUDA to fully release before metacognition phase loads its model
            await asyncio.sleep(VLLM_UNLOAD_DELAY_S)

        # === PHASE 6: METACOGNITION ===
        # Local pattern detection across cycles (replaces cloud Letta)
        metacognition_guidance = None
        if self._metacognition_phase:
            start = time.time()
            try:
                # Build cycle summary for the buffer
                cycle_summary = CycleSummary.from_state(
                    cycle=state.cycle_count,
                    thought=thought,
                    curation_insights=(
                        [curation.analysis.insight] if curation and curation.analysis.insight else []
                    ),
                    emotional_state={
                        "valence": self._emotional_field.dominant_valence if self._emotional_field else 0.0,
                        "arousal": self._emotional_field.dominant_arousal if self._emotional_field else 0.0,
                        "dominant_affect": self._emotional_field.dominant_affect if self._emotional_field else "neutral",
                    },
                    metrics={
                        "surprise_score": surprise_score,
                    },
                    reflexion_category=(
                        reflexion_result.health_assessment.worst_category.name
                        if reflexion_result
                        else "STABLE"
                    ),
                    telemetry=(
                        {"available": True, "confidence_score": state.telemetry_summary.confidence_score}
                        if state.telemetry_summary
                        else {"available": False}
                    ),
                )
                self._metacognition_phase.add_cycle_summary(cycle_summary)

                # Run metacognition (analyzes last N cycles, narrates guidance)
                metacognition_guidance = await self._metacognition_phase.run(state)

                phase_timing["metacognition"] = time.time() - start
                if metacognition_guidance:
                    logger.info("Metacognition guidance provided")

            except Exception as e:
                logger.warning(f"Metacognition phase failed: {e}")
                phase_timing["metacognition_error"] = time.time() - start

        # Log timing
        if self._settings.log_phase_timing:
            total = sum(phase_timing.values())
            logger.info(f"Cycle timing: {phase_timing}, total={total:.2f}s")

        # Verify pending predictions against current state
        prediction_outcomes: list[tuple["Prediction", bool]] = []
        follow_up_uids: list[str] = []
        if self._settings.prediction_verification_enabled and self._prediction_verifier:
            try:
                prediction_outcomes, follow_up_uids = await self._prediction_verifier.check_pending_predictions(
                    state.with_update(
                        thought=thought,
                        vector=state.vector,
                        insight=curation.analysis.insight if curation else "",
                        question=curation.analysis.question if curation else "",
                    )
                )
                if prediction_outcomes:
                    verified_count = sum(1 for _, v in prediction_outcomes if v)
                    logger.info(
                        f"Prediction verification: {len(prediction_outcomes)} resolved "
                        f"({verified_count} verified, {len(prediction_outcomes) - verified_count} falsified)"
                    )
                    # Record outcomes for experiment learning
                    if self._outcome_learner:
                        for pred, verified in prediction_outcomes:
                            self._outcome_learner.record_prediction_outcome(pred, verified)

                    # Update skill effectiveness (UPSKILL + SOAR feedback loop)
                    if self._active_skill_uids:
                        any_verified = verified_count > 0

                        # SOAR: Record verification outcomes for grounded effectiveness
                        hard_problem_accuracy = 0.0
                        hard_problem_count = 0
                        for pred, verified in prediction_outcomes:
                            # Get condition type string for hard problem check
                            condition_type_str = pred.condition_type.value if hasattr(pred.condition_type, 'value') else str(pred.condition_type)

                            # Track hard problem outcomes for SOAR
                            if is_hard_problem_condition(condition_type_str):
                                hard_problem_count += 1
                                if verified:
                                    hard_problem_accuracy += 1.0

                            # Record outcome for each skill that was injected
                            if self._skill_tracker:
                                try:
                                    await self._skill_tracker.record_verification_outcome(
                                        cycle_id=state.cycle_count,
                                        prediction_uid=pred.uid,
                                        verified=verified,
                                        condition_type=condition_type_str,
                                    )
                                except Exception as track_err:
                                    logger.debug(f"SOAR tracking failed: {track_err}")

                        # SOAR: Update promotion queue with skill performance
                        if hard_problem_count > 0 and self._promotion_queue:
                            accuracy = hard_problem_accuracy / hard_problem_count
                            for skill_uid in self._active_skill_uids:
                                result = self._promotion_queue.record_skill_performance(
                                    skill_uid, accuracy
                                )
                                if result and result.startswith("demoted:"):
                                    logger.info(f"[SOAR] Skill {skill_uid[:12]} demoted for poor performance")
                                elif result:
                                    logger.info(f"[SOAR] Skill {skill_uid[:12]} promoted to curriculum")

                                # SOAR: Update teacher policy weights based on improvement
                                if self._teacher_policy:
                                    improvement = self._promotion_queue.get_skill_improvement(skill_uid)
                                    if improvement is not None:
                                        try:
                                            await self._teacher_policy.update_policy_weights(
                                                skill_uid, improvement
                                            )
                                        except Exception as policy_err:
                                            logger.debug(f"Teacher policy update failed: {policy_err}")

                        # Legacy UPSKILL tracking (maintain backwards compatibility)
                        for skill_uid in self._active_skill_uids:
                            try:
                                await self._psyche.record_skill_usage(
                                    skill_uid,
                                    state.cycle_count,
                                    any_verified
                                )
                            except Exception as skill_err:
                                logger.warning(f"Skill usage tracking failed: {skill_err}")
                        self._active_skill_uids.clear()
                    else:
                        # SOAR: Record baseline accuracy for non-skill-influenced cycles
                        if prediction_outcomes and self._promotion_queue:
                            hard_outcomes = [
                                verified for pred, verified in prediction_outcomes
                                if is_hard_problem_condition(
                                    pred.condition_type.value if hasattr(pred.condition_type, 'value') else str(pred.condition_type)
                                )
                            ]
                            if hard_outcomes:
                                baseline_accuracy = sum(1.0 for v in hard_outcomes if v) / len(hard_outcomes)
                                self._promotion_queue.record_baseline_accuracy(baseline_accuracy)
                                logger.debug(f"[SOAR] Recorded baseline accuracy: {baseline_accuracy:.2%}")
            except Exception as e:
                logger.error(f"Prediction verification failed: {e}")

        # Populate reflection buffer from various triggers (Reflexion framework)
        # This accumulates verbal feedback about reasoning quality
        if self._settings.reflection_buffer_enabled:
            self._populate_reflection_buffer(state, curation, prediction_outcomes)

        # Build next state
        # Convert SAEFeature objects to tuples for state propagation (tension tracking)
        sae_features_tuples = [(f.feature_id, f.activation) for f in sae_features] if sae_features else None
        next_state = self._build_next_state(
            state, curation, thought, surprise_score,
            simulation_result=simulation_result,
            prediction_outcomes=prediction_outcomes,
            follow_up_uids=follow_up_uids,
            epistemic_confidence=epistemic_confidence,
            halt_probe_latency_ms=halt_probe_latency_ms,
            sae_features=sae_features_tuples,
            telemetry_summary=telemetry_summary,
        )

        # Persist cognitive state for continuity across restarts
        await self._persist_cognitive_state(next_state)

        # Evolve emotional field (diffusion + decay) and persist
        try:
            packets_before = self._emotional_field.packet_count()
            accessed_memories = self._collect_accessed_memories(state, next_state)
            self._emotional_field.evolve(accessed_memories=accessed_memories)
            packets_after = self._emotional_field.packet_count()

            logger.info(
                f"Emotional field evolved: {packets_before} -> {packets_after} packets, "
                f"{len(accessed_memories)} memories accessed"
            )

            await self._persist_emotional_field()
        except Exception as e:
            logger.warning(f"Emotional field evolution/persistence failed: {e}")
            # Non-fatal: cycle continues even if field persistence fails

        # Persist individuation dynamics (4th layer)
        try:
            await self._persist_individuation_dynamics()
        except Exception as e:
            logger.warning(f"Individuation dynamics persistence failed: {e}")
            # Non-fatal: cycle continues even if dynamics persistence fails

        # Archive cycle audio (rolling archive keeps last 5 cycles)
        if self._liquidsoap:
            try:
                archive_path = await self._liquidsoap.archive_cycle(next_state.cycle_count)
                if archive_path:
                    logger.info(f"Archived cycle {next_state.cycle_count} audio: {archive_path.name}")
            except Exception as e:
                logger.warning(f"Cycle audio archival failed: {e}")
                # Non-fatal: cycle continues even if archival fails

        return next_state

    async def _build_generation_prompt(
        self,
        state: "CognitiveState",
        evoked_context: str = "",
    ) -> str:
        """Build the generation prompt from state.

        If we have pending Discord messages, builds priority response prompt.
        If we have a curated prompt from previous cycle, use it.
        Otherwise fall back to the existing prompt building logic.

        Args:
            state: Current cognitive state
            evoked_context: Optional evoked memories from emotional field sampling

        Returns:
            Prompt string for generation
        """
        # Check for priority Discord messages first
        discord_context = self._build_discord_priority_context()

        # Fetch external context (NotebookLM research)
        external_context = await self._get_external_context(state)
        grounding_section = ""
        if external_context:
            grounding_section = "\n## Grounding Context\n"
            for ctx in external_context:
                grounding_section += f"- {ctx}\n"

        # Get metacognition guidance if available
        metacog_guidance = ""
        if self._metacognition_phase:
            guidance = self._metacognition_phase.get_guidance()
            if guidance:
                metacog_guidance = f"\n## Metacognitive Observation\n{guidance}\n"

        # Combine priority contexts (Discord takes precedence, then evoked memories, then grounding, then metacognition)
        priority_parts = []
        if discord_context:
            priority_parts.append(discord_context)
        if evoked_context:
            priority_parts.append(evoked_context)
        if grounding_section:
            priority_parts.append(grounding_section)
        if metacog_guidance:
            priority_parts.append(metacog_guidance)
        priority_context = "".join(priority_parts) if priority_parts else ""

        # Check if we have a curated prompt from previous curation cycle
        if state.curated_prompt:
            if priority_context:
                # Prepend priority context to curated prompt
                return f"{priority_context}{state.curated_prompt}"
            return state.curated_prompt

        # Fall back to existing prompt building
        from core.cognitive.loop import build_narrative_prompt
        from core.embedding.service import EmbeddingTier

        # Use last concept or default to "emergence" for first cycle
        concept = state.last_concept or "emergence"

        # Retrieve relevant skills based on current context (UPSKILL pattern)
        retrieved_skills: list[tuple[str, str, float]] = []
        self._active_skill_uids.clear()  # Reset for this cycle
        try:
            # Build context for skill retrieval from thought or concept
            context_text = state.thought[:300] if state.thought else concept
            emb_result = await self._embedder.encode(context_text, tier=EmbeddingTier.RETRIEVAL)
            context_embedding = emb_result.to_list()

            skill_results = await self._psyche.query_skills_by_embedding(
                context_embedding, limit=4  # Retrieve more to account for boost reranking
            )
            # Apply SOAR promotion boost and filter
            boosted_skills = []
            for skill, similarity in skill_results:
                is_promoted = self._promotion_queue.is_promoted(skill.uid)
                boost = calculate_retrieval_boost(
                    is_promoted=is_promoted,
                    effectiveness_score=skill.effectiveness_score,
                )
                boosted_similarity = similarity + boost
                boosted_skills.append((skill, boosted_similarity, is_promoted))

            # Sort by boosted similarity and take top 2
            boosted_skills.sort(key=lambda x: x[1], reverse=True)
            for skill, boosted_sim, is_promoted in boosted_skills[:2]:
                if boosted_sim > SKILL_RELEVANCE_THRESHOLD:
                    retrieved_skills.append((skill.name, skill.pattern_summary, boosted_sim))
                    self._active_skill_uids.append(skill.uid)
                    if is_promoted:
                        logger.debug(f"[SOAR] Using promoted skill: {skill.uid[:12]}")

            # Record skill injection for SOAR effectiveness tracking
            if self._active_skill_uids and self._skill_tracker:
                await self._skill_tracker.record_skill_injection(
                    cycle_id=state.cycle_count,
                    skill_uids=self._active_skill_uids.copy(),
                    predictions=[],  # Predictions tracked separately in simulation phase
                )

            if retrieved_skills:
                logger.debug(f"Retrieved {len(retrieved_skills)} skills for generation")
        except Exception as e:
            logger.debug(f"Skill retrieval failed (non-critical): {e}")

        base_prompt = build_narrative_prompt(
            concept=concept,
            previous_thought=state.thought or "",
            association_reason="continuation",
            cycle_count=getattr(state, "cycle_count", 0),
            retrieved_skills=retrieved_skills if retrieved_skills else None,
        )

        if priority_context:
            return f"{priority_context}{base_prompt}"
        return base_prompt

    async def _get_external_context(self, state: "CognitiveState") -> list[str]:
        """Fetch external context to ground generation in reality.

        Gathers multiple sources of external anchors to prevent drift into
        purely internal/abstract reasoning:
        1. NotebookLM research - factual information about concepts
        2. Verified predictions - empirical outcomes that confirm/refute hypotheses

        Args:
            state: Current cognitive state

        Returns:
            List of context strings to inject into prompts (target: 30% external)
        """
        context = []

        # 1. NotebookLM research (if available and relevant)
        if self._notebooklm and state.last_concept:
            try:
                result = await self._notebooklm.query(
                    f"Key facts about {state.last_concept}"
                )
                if result.success and result.answer:
                    context.append(f"Research: {result.answer[:MAX_RESEARCH_CONTEXT_LENGTH]}")
            except Exception:
                pass  # Graceful degradation

        # 2. Recent verified/falsified predictions - empirical grounding
        try:
            limit = getattr(self._settings, "external_context_prediction_limit", 5)
            recent_predictions = await self._psyche.get_recent_verified_predictions(
                limit=limit
            )
            for pred in recent_predictions:
                status = pred.get("status", "")
                claim = pred.get("claim", "")[:MAX_PREDICTION_LENGTH]
                if not claim:
                    continue

                # Use language that emphasizes empirical outcome
                if status == "verified":
                    context.append(f"Prediction confirmed: {claim}")
                elif status == "falsified":
                    context.append(f"Prediction refuted: {claim}")
        except Exception as e:
            logger.warning(f"Failed to get recent verified predictions for external context: {e}")
            pass  # Graceful degradation - don't block generation

        return context

    async def _generate_thought(
        self,
        prompt: str,
        state: "CognitiveState",
        hypothesis_vectors: Optional[list] = None,
    ) -> tuple[str, Optional[dict], Optional[dict], float, Optional["TelemetrySummary"]]:
        """Generate a thought using TransformerLens.

        Args:
            prompt: Generation prompt
            state: Current state (for steering vectors)
            hypothesis_vectors: Optional list of effective HypothesisSteeringVector
                objects to apply during generation

        Returns:
            Tuple of (thought_text, raw_activations, mlp_activations, surprise_score, telemetry_summary)
        """
        # Get steerer from state
        steerer = state.steerer

        # Apply hypothesis steering vectors to concept zone before generation
        if hypothesis_vectors and steerer and TORCH_AVAILABLE:
            await self._apply_hypothesis_vectors(steerer, hypothesis_vectors)

        # Generate with activation capture (both residual and MLP input for SAE)
        # Use layer 16 for both steering and SAE transcoder (transcoder trained on layer 16)
        # Also capture layer 6 (REPRESENTATIVE_EXPLORATION_LAYER) for Evalatis update_from_cycle
        # Enable telemetry capture for biofeedback (logit entropy, residual slopes)
        result = await self._hooked_model.generate(
            prompt,
            max_tokens=self._settings.generation_max_tokens,
            capture_activations=True,
            capture_mlp_input=True,  # For SAE transcoder
            capture_layers=[REPRESENTATIVE_EXPLORATION_LAYER, 16],  # Exploration zone + transcoder
            hierarchical_steerer=steerer,
            capture_telemetry=True,  # Biofeedback v0: logit dynamics, residual slopes
        )

        # Ensure thought ends at sentence boundary (avoid mid-sentence cutoffs)
        thought = ensure_complete_sentence(result.text)
        # Extract activations from snapshots (convert to dict format)
        # resid_post for steering/surprise, mlp_in for SAE transcoder
        activations = {
            snapshot.layer: snapshot.activations
            for snapshot in result.snapshots
            if snapshot.hook_point == "resid_post"
        }
        # Also extract MLP input for SAE transcoder
        mlp_activations = {
            snapshot.layer: snapshot.activations
            for snapshot in result.snapshots
            if snapshot.hook_point == "mlp_in"
        }

        # Calculate surprise from activations
        surprise_score = self._calculate_surprise(activations, state)

        # Return both activation types: resid_post for steering, mlp_in for SAE
        # Include telemetry for biofeedback evaluation in Reflexion phase
        return thought, activations, mlp_activations, surprise_score, result.telemetry

    async def _generate_thought_plat_lite(
        self,
        prompt: str,
        state: "CognitiveState",
        hypothesis_vectors: Optional[list] = None,
    ) -> tuple[str, Optional[dict], Optional[dict], float, Optional["TelemetrySummary"]]:
        """Generate thought using PLaT-Lite chunked generation with latent observation.

        Inspired by PLaT (arXiv:2601.21358): reasoning in continuous latent space.
        Generates in 16-token chunks, observing latent state after each chunk
        to enable real-time mode tracking, emotion detection, and intentional steering.

        Features:
        - Chunked generation (16 tokens per chunk) with layer 16 activation capture
        - LatentObserver extracts SAE features for interpretable reasoning states
        - ModeAwareGuidancePolicy evaluates mode alignment and action/reflection balance
        - CombinedNarrationCoordinator handles dual-stream audio (thought + trajectory)

        Args:
            prompt: Generation prompt
            state: Current state (for steering vectors)
            hypothesis_vectors: Optional hypothesis steering vectors

        Returns:
            Tuple of (thought_text, raw_activations, mlp_activations, surprise_score, telemetry)
        """
        if not self._chunked_generator or not self._plat_enabled:
            # Fallback to standard generation
            return await self._generate_thought(prompt, state, hypothesis_vectors)

        # Get steerer from state
        steerer = state.steerer

        # Apply hypothesis steering vectors before chunked generation
        if hypothesis_vectors and steerer and TORCH_AVAILABLE:
            await self._apply_hypothesis_vectors(steerer, hypothesis_vectors)

        # Get anchor service for mode classification (async lazy init)
        # Create new LatentObserver with anchor service for this generation
        anchor_service = await self._get_anchor_service()
        if anchor_service and self._latent_observer:
            # Create fresh observer with anchor service
            from core.cognitive.stream.latent_observer import LatentObserver
            self._latent_observer = LatentObserver(
                transcoder=self._transcoder,
                anchor_service=anchor_service,
            )

        # Set intended mode based on episode type if available
        intended_mode = None
        if state.current_episode:
            # Map episode type to cognitive mode
            # EpisodeType: DEEP_DIVE, DIALECTICAL_DEBATE, MEMORY_ARCHAEOLOGY,
            #              QUESTION_PURSUIT, SYNTHESIS, CREATIVE, META_REFLECTION,
            #              HYPOTHESIS_SIMULATION, JUDGMENT_REVIEW
            episode_to_mode = {
                "DEEP_DIVE": "philosophical_inquiry",
                "DIALECTICAL_DEBATE": "dialectical_synthesis",
                "HYPOTHESIS_SIMULATION": "hypothesis_testing",
                "MEMORY_ARCHAEOLOGY": "memory_integration",
                "QUESTION_PURSUIT": "philosophical_inquiry",
                "SYNTHESIS": "dialectical_synthesis",
                "CREATIVE": "creative_exploration",
                "META_REFLECTION": "metacognitive_monitoring",
                "JUDGMENT_REVIEW": "metacognitive_monitoring",
            }
            intended_mode = episode_to_mode.get(state.current_episode.episode_type.name)

        if self._guidance_policy:
            self._guidance_policy.intended_mode = intended_mode
            # Pass anchor service to guidance policy too
            if anchor_service:
                self._guidance_policy._anchors = anchor_service

        # Load transcoder on CPU for real-time SAE feature extraction during generation
        # This adds ~640ms latency per chunk but enables feature-based trajectory narration
        # The narration queue buffers audio so latency isn't noticeable to listeners
        if self._transcoder and not self._transcoder.is_loaded:
            logger.info("[PLAT-LITE] Loading transcoder on CPU for real-time SAE features...")
            await self._transcoder.load()
            logger.info(
                f"[PLAT-LITE] Transcoder loaded: d_in={self._transcoder.d_in}, "
                f"d_sae={self._transcoder.d_sae}"
            )

        # Reset generators for new generation
        self._chunked_generator.reset()
        if self._latent_observer:
            self._latent_observer.reset()

        # Track state across chunks
        full_text = ""
        chunk_history: list = []
        all_activations: dict = {}
        all_mlp_activations: dict = {}

        try:
            # Generate using chunked generator with explicit loop
            for chunk_idx in range(self._plat_max_chunks):
                # Generate one chunk with activation capture
                context = prompt + full_text
                chunk = await self._chunked_generator.generate_chunk(
                    context=context,
                    chunk_size=self._plat_chunk_size,
                    steering=steerer,
                    capture_mlp_input=True,
                )

                # Accumulate text
                full_text += chunk.text
                chunk_history.append(chunk)

                # Store activations (layer 16 for both resid_post and mlp_in)
                if chunk.activations is not None:
                    all_activations[self._plat_capture_layer] = chunk.activations
                if chunk.mlp_input is not None:
                    all_mlp_activations[self._plat_capture_layer] = chunk.mlp_input

                # Extract latent observation via observer
                observation = None
                if self._latent_observer:
                    try:
                        observation = await self._latent_observer.observe(chunk)
                    except Exception as e:
                        logger.debug(f"[PLAT-LITE] Latent observation failed: {e}")

                # Evaluate guidance policy for steering adjustments
                steering_adjustment = None
                if self._guidance_policy and observation:
                    try:
                        steering_adjustment = await self._guidance_policy.evaluate(
                            chunk=chunk,
                            observation=observation,
                            history=chunk_history[:-1],  # Exclude current chunk
                            current_steering=steerer,
                        )
                    except Exception as e:
                        logger.debug(f"[PLAT-LITE] Guidance evaluation failed: {e}")

                    if steering_adjustment:
                        logger.info(
                            f"[PLAT-LITE] Guidance: zone={steering_adjustment.zone}, "
                            f"magnitude={steering_adjustment.magnitude:.2f} - {steering_adjustment.reason}"
                        )
                        # Apply steering adjustment to steerer if applicable
                        if steerer and steering_adjustment.direction is not None:
                            try:
                                steerer.update_vector(
                                    zone_name=steering_adjustment.zone,
                                    new_direction=steering_adjustment.direction,
                                    scale=steering_adjustment.magnitude * 0.1,  # Gentle adjustment
                                )
                            except Exception as e:
                                logger.debug(f"[PLAT-LITE] Steering adjustment failed: {e}")

                # Narrate via coordinator if available
                if self._narration_coordinator and observation:
                    try:
                        await self._narration_coordinator.on_chunk_complete(
                            chunk=chunk,
                            observation=observation,
                            steering=steering_adjustment,
                        )
                    except Exception as e:
                        logger.debug(f"[PLAT-LITE] Narration failed: {e}")

                # Check for natural termination
                if chunk.is_complete:
                    logger.debug(f"[PLAT-LITE] Natural termination at chunk {chunk_idx + 1}")
                    break

            # Finalize narration
            if self._narration_coordinator:
                try:
                    await self._narration_coordinator.finalize()
                except Exception as e:
                    logger.debug(f"[PLAT-LITE] Narration finalize failed: {e}")

            # Ensure thought ends at sentence boundary
            thought = ensure_complete_sentence(full_text)

        except Exception as e:
            logger.warning(f"[PLAT-LITE] Chunked generation failed: {e}, falling back")
            # Fallback to standard generation on error
            return await self._generate_thought(prompt, state, hypothesis_vectors)

        # Calculate surprise from final activations
        surprise_score = self._calculate_surprise(all_activations, state)

        logger.info(
            f"[PLAT-LITE] Generation complete: {len(chunk_history)} chunks, "
            f"{len(thought)} chars, surprise={surprise_score:.2f}"
        )

        # Return in same format as standard generation
        # Note: telemetry not available in chunked mode (would need aggregation)
        return thought, all_activations, all_mlp_activations, surprise_score, None

    def _calculate_surprise(
        self,
        activations: Optional[dict],
        state: "CognitiveState",
    ) -> float:
        """Calculate surprise score from activations.

        Args:
            activations: Raw activations from generation
            state: Current state (for baseline comparison)

        Returns:
            Surprise score
        """
        if not activations or not TORCH_AVAILABLE:
            return 50.0  # Default moderate surprise

        # Get activation at steering layer
        steering_layer = self._settings.cognitive_steering_layer
        if steering_layer not in activations:
            return 50.0

        activation = activations[steering_layer]

        # Calculate magnitude-based surprise
        if isinstance(activation, torch.Tensor):
            # Convert to float32 first (NumPy doesn't support bfloat16)
            activation = activation.detach().cpu().float().numpy()

        magnitude = float(np.linalg.norm(activation.flatten()))

        # Compare to baseline if available
        baseline = getattr(state, "baseline_magnitude", None)
        if baseline and baseline > 0:
            surprise = (magnitude / baseline) * 50.0
        else:
            surprise = min(100.0, magnitude / 10.0)

        return min(100.0, max(0.0, surprise))

    async def _extract_pending_hypothesis_vectors(self) -> None:
        """Extract steering vectors for hypotheses created in previous cycles.

        This is called after HookedQwen loads, to extract vectors for hypotheses
        that were created during simulation phase when the model wasn't loaded.

        The extraction uses Contrastive Activation Addition (CAA):
            vector = mean(positive_activations) - mean(negative_activations)

        After extraction, the steering_vector_uid is linked to the hypothesis
        for future verification feedback.
        """
        if not self._vector_extractor:
            return

        try:
            # Query hypotheses that have contrastive pairs but no steering vector
            pending = await self._psyche.get_hypotheses_needing_vector_extraction(
                limit=self._settings.hypothesis_pending_extraction_limit
            )

            if not pending:
                return

            logger.info(
                f"Extracting steering vectors for {len(pending)} pending hypotheses"
            )

            for hypothesis in pending:
                try:
                    steering_vector = await self._vector_extractor.extract_vector(
                        hypothesis
                    )
                    if steering_vector:
                        # Persist the steering vector
                        saved = await self._psyche.save_hypothesis_steering_vector(
                            steering_vector
                        )
                        if saved:
                            # Link hypothesis to its steering vector
                            await self._psyche.set_hypothesis_steering_vector_uid(
                                hypothesis_uid=hypothesis.uid,
                                steering_vector_uid=steering_vector.uid,
                            )
                            logger.info(
                                f"Extracted steering vector {steering_vector.uid} "
                                f"for hypothesis {hypothesis.uid}"
                            )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid data during vector extraction for hypothesis "
                        f"{hypothesis.uid}: {e}"
                    )
                except RuntimeError as e:
                    logger.warning(
                        f"Model error during vector extraction for hypothesis "
                        f"{hypothesis.uid}: {e}"
                    )

        except Exception as e:
            logger.warning(f"Error extracting pending hypothesis vectors: {e}")

    async def _apply_hypothesis_vectors(
        self,
        steerer: "HierarchicalSteerer",
        hypothesis_vectors: list,
    ) -> None:
        """Apply hypothesis steering vectors to the concept zone.

        Blends effective hypothesis vectors into the steerer's concept zone
        vector. Each vector's contribution is scaled by its effectiveness
        score, ensuring higher-performing vectors have more influence.

        Side Effects:
            Calls record_application() on each successfully applied vector,
            incrementing its application_count and updating last_applied timestamp.

        Args:
            steerer: The HierarchicalSteerer to update
            hypothesis_vectors: List of HypothesisSteeringVector objects
        """
        if not hypothesis_vectors or not TORCH_AVAILABLE:
            return

        # Get the concept zone configuration
        concept_zone = None
        for zone in steerer.config.zones:
            if zone.name == "concept":
                concept_zone = zone
                break

        if concept_zone is None:
            logger.debug("No concept zone found in steerer config")
            return

        # Combine hypothesis vectors weighted by effectiveness
        combined_direction = np.zeros(steerer.d_model, dtype=np.float32)
        total_weight = 0.0

        for hsv in hypothesis_vectors:
            if not hsv.vector_data:
                continue

            # Convert vector to numpy
            try:
                vec = np.array(hsv.vector_data, dtype=np.float32)

                # Handle dimension mismatch
                if len(vec) != steerer.d_model:
                    logger.debug(
                        f"Resizing hypothesis vector from {len(vec)} to {steerer.d_model}"
                    )
                    if len(vec) < steerer.d_model:
                        # Pad with zeros
                        padded = np.zeros(steerer.d_model, dtype=np.float32)
                        padded[:len(vec)] = vec
                        vec = padded
                    else:
                        # Truncate
                        vec = vec[:steerer.d_model]

                # Weight by effectiveness score
                weight = hsv.effectiveness_score
                combined_direction += vec * weight
                total_weight += weight

                # Record that this vector was applied
                hsv.record_application()

                logger.debug(
                    f"Applied hypothesis vector {hsv.uid} "
                    f"(effectiveness={hsv.effectiveness_score:.2f})"
                )

            except Exception as e:
                logger.warning(f"Failed to apply hypothesis vector {hsv.uid}: {e}")

        if total_weight > 0:
            # Normalize by total weight
            combined_direction /= total_weight

            # Apply to concept zone using the steerer's update method
            # Use configurable scale to avoid over-steering (hypothesis vectors are additive)
            steerer.update_vector(
                zone_name="concept",
                new_direction=combined_direction,
                scale=self._settings.hypothesis_vector_scale,
            )

            logger.info(
                f"Applied {len(hypothesis_vectors)} hypothesis vectors to concept zone "
                f"(total_weight={total_weight:.2f})"
            )

    def _summarize_activations(
        self,
        raw_activations: Optional[dict],
    ) -> "ActivationSummary":
        """Summarize raw activations for curator.

        Args:
            raw_activations: Dict of layer -> tensor activations

        Returns:
            ActivationSummary for curator consumption
        """
        from core.cognitive.curator_schemas import ActivationSummary

        if not raw_activations or not TORCH_AVAILABLE:
            return ActivationSummary(layer=self._settings.cognitive_steering_layer)

        # Focus on steering layer
        steering_layer = self._settings.cognitive_steering_layer
        if steering_layer not in raw_activations:
            return ActivationSummary(layer=steering_layer)

        activation = raw_activations[steering_layer]
        if isinstance(activation, torch.Tensor):
            # Convert to float32 first (NumPy doesn't support bfloat16)
            activation = activation.detach().cpu().float().numpy()

        # Flatten and get stats
        flat = activation.flatten()
        mean_val = float(np.mean(np.abs(flat)))
        max_val = float(np.max(np.abs(flat)))

        # Get top activating positions
        top_indices = np.argsort(np.abs(flat))[-10:][::-1]
        top_positions = [(int(i), float(flat[i])) for i in top_indices]

        return ActivationSummary(
            layer=steering_layer,
            top_positions=top_positions,
            mean_activation=mean_val,
            max_activation=max_val,
        )

    async def _extract_sae_features(
        self,
        raw_activations: dict,
    ) -> list["SAEFeature"]:
        """Extract SAE features from activations.

        Args:
            raw_activations: Dict of layer -> tensor activations

        Returns:
            List of SAEFeature objects
        """
        from core.cognitive.curator_schemas import SAEFeature

        if not self._transcoder:
            logger.info("SAE extraction skipped: no transcoder")
            return []

        try:
            logger.info(f"SAE extraction starting, activations keys: {list(raw_activations.keys())}")

            # Ensure transcoder is loaded
            if not self._transcoder.is_loaded:
                await self._transcoder.load()

            # Get activation at transcoder layer
            layer = self._transcoder.layer
            if layer not in raw_activations:
                logger.info(f"SAE extraction skipped: layer {layer} not in activations (available: {list(raw_activations.keys())})")
                return []

            activation = raw_activations[layer]

            # Move activation to transcoder's device (transcoder may be on CPU to save VRAM)
            if TORCH_AVAILABLE and hasattr(activation, 'to'):
                activation = activation.to(self._transcoder.device)

            # Run through transcoder (sync, so use executor)
            loop = asyncio.get_running_loop()
            features_tensor = await loop.run_in_executor(
                None,
                self._transcoder.encode,
                activation,
            )

            # Get top active features
            active_features = self._transcoder.get_active_features(features_tensor, top_k=20)

            # Convert to SAEFeature objects with interpretations
            # Priority: 1) Neuronpedia (verified), 2) Logit lens (inferred), 3) Generic fallback
            result = []
            indices = [active.index for active in active_features]
            # Track both interpretation and source
            interpretations: dict[int, tuple[str, str]] = {}  # idx -> (label, source)

            # Try Neuronpedia first (in case they add Qwen3-8B support)
            try:
                from core.sae.neuronpedia import batch_get_interpretations
                neuronpedia_interps = await batch_get_interpretations(indices)
                # Only use if we got real interpretations (not "feature NNNNN")
                for idx, interp in neuronpedia_interps.items():
                    if not interp.startswith("feature "):
                        interpretations[idx] = (interp, "neuronpedia")
            except Exception as e:
                logger.debug(f"Neuronpedia fetch failed: {e}")

            # Fall back to logit lens for features without Neuronpedia interpretations
            # Note: logit lens shows token predictions, not semantic meaning - useful for
            # pattern tracking but not for narration
            missing_indices = [idx for idx in indices if idx not in interpretations]
            logger.info(f"Missing {len(missing_indices)} Neuronpedia interpretations")
            if missing_indices and self._hooked_model and self._hooked_model.is_loaded:
                try:
                    from core.sae.feature_interpreter import get_feature_interpreter
                    interpreter = get_feature_interpreter(self._transcoder, self._hooked_model)
                    if not interpreter.is_initialized:
                        await interpreter.initialize()
                    logit_interps = interpreter.batch_get_interpretations(missing_indices)
                    successful = 0
                    for idx, interp in logit_interps.items():
                        if not interp.startswith("feature "):
                            interpretations[idx] = (interp, "logit_lens")
                            successful += 1
                    if successful > 0:
                        logger.info(f"Logit lens provided {successful}/{len(missing_indices)} token-based labels (not for narration)")
                except Exception as e:
                    logger.warning(f"Logit lens interpretation failed: {e}")

            for active in active_features:
                # Use interpretation if available, else generic fallback
                if active.index in interpretations:
                    label, source = interpretations[active.index]
                else:
                    label = self._transcoder.get_feature_label(active.index)
                    source = None
                result.append(
                    SAEFeature(
                        feature_id=active.index,
                        activation=active.activation,
                        label=label,
                        interpretation_source=source,
                    )
                )

            if result:
                top = result[0]
                logger.info(f"Extracted {len(result)} SAE features, top: {top.feature_id} ({top.activation:.3f}) - '{top.label}'")

            return result

        except Exception as e:
            logger.warning(f"SAE feature extraction failed: {e}")
            return []

    async def _narrate(
        self,
        thought: str,
        voice: Optional[str] = None,
        tone_before: bool = False,
    ) -> None:
        """Narrate the thought via Liquidsoap.

        Args:
            thought: Thought text to narrate
            voice: Optional voice override (None uses default voice)
            tone_before: If True, play sonar tone before narration (for curator)
        """
        if not self._liquidsoap:
            return

        try:
            await self._liquidsoap.narrate(thought, voice=voice, tone_before=tone_before)
        except Exception as e:
            logger.warning(f"Narration failed: {e}")

    async def _update_visualizer(
        self,
        insight: Optional[str] = None,
        features: Optional[list[str]] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update the stream visualizer state.

        Args:
            insight: Current thought/insight text to display
            features: List of SAE feature labels
            phase: Current cognitive phase name
            status: Status text (e.g., 'thinking', 'narrating')
        """
        if not self._stream_visualizer:
            return

        try:
            await self._stream_visualizer.update(
                insight=insight,
                features=features,
                phase=phase,
                status=status,
            )
        except Exception as e:
            logger.debug(f"Visualizer update failed: {e}")

    async def _embed_thought(self, thought: str) -> list[float]:
        """Embed thought using golden embedder (falls back to retrieval).

        Args:
            thought: Thought text

        Returns:
            Embedding vector (4096-dim for golden, 1024-dim for retrieval)
        """
        from core.embedding.service import EmbeddingTier

        # Try golden first, fall back to retrieval if not available
        if self._embedder.golden_backend is not None:
            result = await self._embedder.encode(thought, tier=EmbeddingTier.GOLDEN)
        else:
            logger.debug("Golden embedder not available, using retrieval embedder")
            result = await self._embedder.encode(thought, tier=EmbeddingTier.RETRIEVAL)
        return result.to_list()

    async def _assess_novelty(
        self,
        embedding: list[float],
    ) -> tuple[float, Optional[str]]:
        """Assess novelty of insight against existing zettels.

        Uses embedding similarity to determine if an insight is genuinely
        new or a refinement of existing knowledge (arXiv:2601.12542 inspired).

        Args:
            embedding: The insight's embedding vector

        Returns:
            (novelty_score, most_similar_uid) - score in [0,1], uid if similar exists
        """
        similar = await self._psyche.query_zettels_by_embedding(
            embedding=embedding,
            limit=1,
        )

        if not similar:
            return 1.0, None  # Completely novel

        most_similar, similarity = similar[0]
        novelty_score = 1.0 - similarity
        return novelty_score, most_similar.uid

    async def _create_zettel(
        self,
        curation: "CurationResult",
        thought_embedding: list[float],
        state: "CognitiveState",
    ) -> None:
        """Create an InsightZettel from curation result with retroactive linking.

        Implements A-MEM-style connection discovery: after creating a new zettel,
        we discover semantically related existing zettels and establish RELATES_TO
        edges. This creates a living knowledge network where new insights
        strengthen connections to existing knowledge.

        Args:
            curation: Curation result containing zettel data
            thought_embedding: Embedding of the source thought
            state: Current cognitive state (for cycle number, sae features)
        """
        import uuid
        from core.psyche.schema import InsightZettel, InsightSourceType

        zettel_data = curation.graph_ops.zettel
        if not zettel_data:
            return

        # Embed the insight (use golden if available, else retrieval)
        from core.embedding.service import EmbeddingTier
        tier = EmbeddingTier.GOLDEN if self._embedder.golden_backend else EmbeddingTier.RETRIEVAL
        insight_embedding = await self._embedder.encode(zettel_data.insight, tier=tier)
        insight_embedding_list = insight_embedding.to_list()

        # === Novelty Assessment ===
        # Determine if this insight is novel or a refinement of existing knowledge
        novelty_score, most_similar_uid = await self._assess_novelty(insight_embedding_list)

        # === Duplicate Rejection ===
        # Skip near-exact duplicates (similarity >= 0.90)
        if novelty_score <= DUPLICATE_REJECTION_THRESHOLD:
            logger.info(f"Rejecting duplicate zettel (similarity={1 - novelty_score:.3f})")
            return

        is_refinement = novelty_score < self._settings.novelty_threshold
        refines_uid = most_similar_uid if is_refinement else None

        # Generate uid for the zettel
        zettel_uid = f"zettel_{uuid.uuid4().hex[:12]}"

        # Use first concept or join concepts for the singular concept field
        concept = zettel_data.concepts[0] if zettel_data.concepts else "general"

        # Get SAE feature indices from state (top-N indices only)
        sae_indices = [idx for idx, _ in state.sae_features[:10]] if state.sae_features else []

        # Create zettel with correct field names and novelty fields
        zettel = InsightZettel(
            uid=zettel_uid,
            insight_text=zettel_data.insight,
            question_text=zettel_data.question,
            source_type=InsightSourceType.COGNITIVE,
            source_uid=f"thought_{state.cycle_count}",  # Reference the thought by cycle
            concept=concept,
            cycle=state.cycle_count,
            embedding=insight_embedding_list,
            sae_feature_indices=sae_indices,
            novelty_score=novelty_score,
            is_refinement=is_refinement,
            refines_uid=refines_uid,
        )

        # Persist to graph
        await self._psyche.create_zettel(zettel)
        logger.info(f"Created zettel {zettel_uid}: {zettel_data.insight[:50]}...")

        # === Create REFINES Edge for Refinements ===
        # If this zettel is a refinement of existing knowledge, create lineage edge
        if is_refinement and refines_uid:
            try:
                await self._psyche.link_zettel_refines(
                    zettel_uid=zettel_uid,
                    parent_uid=refines_uid,
                    novelty_score=novelty_score,
                )
                logger.info(
                    f"Zettel {zettel_uid} refines {refines_uid} "
                    f"(novelty={novelty_score:.3f})"
                )
            except Exception as e:
                logger.warning(f"Failed to create REFINES edge: {e}")

        # === A-MEM-style Retroactive Linking ===
        # Discover semantically related zettels and create RELATES_TO edges
        if self._settings.retroactive_linking_enabled:
            try:
                related_zettels = await self._psyche.discover_related_zettels(
                    zettel_uid=zettel_uid,
                    embedding=insight_embedding_list,
                    similarity_threshold=self._settings.retroactive_linking_threshold,
                    max_connections=self._settings.retroactive_linking_max_connections,
                )

                for related_zettel, score in related_zettels:
                    # Skip the parent zettel - it already has a REFINES edge
                    if related_zettel.uid == refines_uid:
                        continue

                    # Create bidirectional RELATES_TO edge
                    await self._psyche.link_zettel_relates_to(
                        zettel_uid=zettel_uid,
                        related_uid=related_zettel.uid,
                        score=score,
                        relationship_type="semantic",
                    )

                    # Propagate current concept as keyword to related zettel
                    # This enriches older zettels with new conceptual context
                    if concept and concept != "general":
                        await self._psyche.add_zettel_keyword(related_zettel.uid, concept)

                    logger.debug(
                        f"Linked {zettel_uid} <-> {related_zettel.uid} "
                        f"(score={score:.3f}, concept={concept})"
                    )

                # Count actual links (excluding refines_uid)
                link_count = sum(1 for z, _ in related_zettels if z.uid != refines_uid)
                if link_count > 0:
                    logger.info(
                        f"Retroactive linking: connected {zettel_uid} to "
                        f"{link_count} related zettels"
                    )

            except Exception as e:
                logger.warning(f"Retroactive linking failed: {e}")

    def _populate_reflection_buffer(
        self,
        state: "CognitiveState",
        curation: "CurationResult",
        prediction_outcomes: "Optional[list[tuple[Prediction, bool]]]" = None,
    ) -> None:
        """Populate the reflection buffer from various triggers.

        Implements Reflexion-style verbal reinforcement learning by detecting
        situations that warrant self-reflection and adding linguistic feedback
        to the buffer. These reflections persist across cycles and guide future
        reasoning.

        Triggers:
        - Curator-generated reflection (explicit self-critique)
        - Prediction failures (hypothesis disconfirmation)
        - Low faithfulness score (only when SAE features enabled)

        Args:
            state: Current cognitive state (contains the buffer to populate)
            curation: Curation results (may contain explicit reflection)
            prediction_outcomes: Optional prediction verification results
        """
        # Guard clause: ensure reflection_buffer exists and is not a mock
        from core.cognitive.state import ReflectionBuffer
        if not hasattr(state, 'reflection_buffer') or not isinstance(state.reflection_buffer, ReflectionBuffer):
            logger.debug("Skipping reflection buffer population: buffer not available")
            return

        cycle = getattr(state, 'cycle_count', 0)
        if not isinstance(cycle, int):
            cycle = 0

        # 1. Curator-generated reflection (explicit self-critique)
        if curation.reflection:
            trigger = curation.reflection_trigger or "curator_assessment"
            severity = "notable"
            # Upgrade severity if faithfulness is also low
            if curation.faithfulness and curation.faithfulness.divergence_severity in ("moderate", "high"):
                severity = "critical"
            state.reflection_buffer.add(
                cycle=cycle,
                text=curation.reflection,
                trigger=trigger,
                severity=severity,
            )
            logger.debug(f"Added curator reflection: {curation.reflection[:80]}...")

        # 2. Low faithfulness score (SAE-verbal divergence)
        if curation.faithfulness:
            faith = curation.faithfulness
            if faith.divergence_severity in ("moderate", "high"):
                # Generate reflection about the divergence
                if faith.divergence_severity == "high":
                    text = (
                        f"My verbal claims diverged significantly from my internal activations "
                        f"(overlap: {faith.overlap_ratio:.0%}). I should be more careful about "
                        f"what I claim to be thinking versus what my patterns actually show."
                    )
                    severity = "critical"
                else:
                    text = (
                        f"There was some mismatch between what I said and my internal state "
                        f"(overlap: {faith.overlap_ratio:.0%}). Worth noting for future self-awareness."
                    )
                    severity = "notable"

                state.reflection_buffer.add(
                    cycle=cycle,
                    text=text,
                    trigger="low_faithfulness",
                    severity=severity,
                )
                logger.debug(f"Added faithfulness reflection: {text[:80]}...")

        # 3. Prediction failures
        if prediction_outcomes:
            failures = [(pred, verified) for pred, verified in prediction_outcomes if not verified]
            if failures:
                failed_claims = [pred.claim[:40] for pred, _ in failures[:2]]
                text = (
                    f"My predictions were not confirmed: {', '.join(failed_claims)}... "
                    f"({len(failures)} prediction(s) falsified). My hypotheses may need revision."
                )
                severity = "notable" if len(failures) < 3 else "critical"
                state.reflection_buffer.add(
                    cycle=cycle,
                    text=text,
                    trigger="prediction_failure",
                    severity=severity,
                )
                logger.debug(f"Added prediction failure reflection: {len(failures)} failures")

        # 4. Semantic repetition / staleness
        # Use getattr with default to handle mocked states in tests
        consecutive_low = getattr(state, 'consecutive_low_surprise', 0)
        if isinstance(consecutive_low, int) and consecutive_low > 3:
            text = (
                f"I've been generating low-surprise thoughts for {consecutive_low} "
                f"consecutive cycles. I should push into less familiar territory."
            )
            state.reflection_buffer.add(
                cycle=cycle,
                text=text,
                trigger="repetition",
                severity="notable",
            )
            logger.debug("Added repetition reflection due to consecutive low surprise")

        # 5. Opening phrase repetition (from opening tracker)
        repetition_count = getattr(state, 'repetition_count_in_stage', 0)
        if isinstance(repetition_count, int) and repetition_count >= 3:
                text = (
                    "I've been using similar opening patterns. "
                    "I should vary my approach to avoid formulaic thinking."
                )
                state.reflection_buffer.add(
                    cycle=cycle,
                    text=text,
                    trigger="opening_repetition",
                    severity="minor",
                )

    async def _persist_graph_ops(self, curation: "CurationResult") -> None:
        """Persist graph operations from curation.

        Args:
            curation: Curation result with graph operations
        """
        graph_ops = curation.graph_ops

        # Persist triples and ensure their subject/object entities exist
        import uuid as uuid_module
        for triple_data in graph_ops.new_triples:
            try:
                from core.psyche.schema import Triple

                # Ensure subject and object entities exist AND give small salience boost
                # for being referenced in a triple (0.02 = modest relevance signal)
                if triple_data.subject:
                    await self._psyche.update_entity_salience(triple_data.subject, 0.02)
                if triple_data.object_:
                    await self._psyche.update_entity_salience(triple_data.object_, 0.02)

                triple = Triple(
                    uid=f"triple_{uuid_module.uuid4().hex[:12]}",
                    subject=triple_data.subject,
                    predicate=triple_data.predicate,
                    object=triple_data.object_,
                    confidence=triple_data.confidence,
                )
                await self._psyche.create_triple(triple)
            except Exception as e:
                logger.warning(f"Failed to persist triple: {e}")

        # Ensure entities exist for all concepts mentioned in the thought
        # This creates new entities if they don't exist (via MERGE)
        concepts_created = 0
        for concept in curation.analysis.concepts:
            if concept and concept.strip():
                try:
                    # Ensure entity exists AND give salience boost for explicit concept mention
                    # (0.05 = slightly larger than triple refs since concepts are curated)
                    # update_entity_salience uses MERGE, creating entity if needed
                    created = await self._psyche.update_entity_salience(
                        concept.strip(), 0.05
                    )
                    if created:
                        concepts_created += 1
                except Exception as e:
                    logger.warning(f"Failed to ensure entity for concept '{concept}': {e}")

        if concepts_created > 0:
            logger.info(f"Ensured {concepts_created} concept entities exist in graph")

        # Apply entity updates
        for update in graph_ops.entity_updates:
            try:
                await self._psyche.update_entity_salience(
                    update.name, update.salience_delta
                )
            except Exception as e:
                logger.warning(f"Failed to update entity: {e}")

        # Apply belief updates and track dynamics
        for update in graph_ops.belief_updates:
            try:
                updated_beliefs = await self._psyche.update_belief_confidence(
                    update.topic, update.confidence_delta, update.evidence
                )
                # Track belief dynamics for each updated belief
                for belief_uid, new_confidence in updated_beliefs:
                    transition = self._individuation_dynamics.observe_belief_update(
                        topic=update.topic,
                        new_confidence=new_confidence,
                        element_id=belief_uid,
                    )
                    if transition:
                        logger.info(
                            f"[DYNAMICS] Belief '{update.topic}' transition: "
                            f"{transition.from_phase.value} → {transition.to_phase.value}"
                        )
            except Exception as e:
                logger.warning(f"Failed to update belief: {e}")

    def _merge_curation_results(self, results: list["CurationResult"]) -> "CurationResult":
        """Merge multiple curation results into one.

        Used in multi-turn curation to combine results from multiple turns.
        The final turn's analysis and next_prompt are used as primary.
        Lists (triples, entity_updates, etc.) are combined from all turns.

        Args:
            results: List of CurationResult from each turn

        Returns:
            Merged CurationResult
        """
        if not results:
            return CurationResult.empty()
        if len(results) == 1:
            return results[0]

        # Use final turn's analysis as primary (most informed)
        final = results[-1]

        # Combine graph operations from all turns
        all_triples = []
        all_entity_updates = []
        all_belief_updates = []
        zettels = []

        for result in results:
            all_triples.extend(result.graph_ops.new_triples)
            all_entity_updates.extend(result.graph_ops.entity_updates)
            all_belief_updates.extend(result.graph_ops.belief_updates)
            if result.graph_ops.zettel:
                zettels.append(result.graph_ops.zettel)

        # Deduplicate triples by (subject, predicate, object_)
        seen_triples = set()
        unique_triples = []
        for triple in all_triples:
            key = (triple.subject, triple.predicate, triple.object_)
            if key not in seen_triples:
                seen_triples.add(key)
                unique_triples.append(triple)

        # Accumulate entity updates by name (sum deltas, keep last type_refinement)
        entity_map = {}
        for update in all_entity_updates:
            if update.name in entity_map:
                existing = entity_map[update.name]
                # Sum the deltas
                merged_delta = existing.salience_delta + update.salience_delta
                # Keep last type_refinement if provided
                merged_type = update.type_refinement or existing.type_refinement
                entity_map[update.name] = EntityUpdate(
                    name=update.name,
                    salience_delta=merged_delta,
                    type_refinement=merged_type,
                )
            else:
                entity_map[update.name] = update
        unique_entities = list(entity_map.values())

        # Accumulate belief updates by topic (sum deltas, concatenate evidence)
        belief_map = {}
        for update in all_belief_updates:
            if update.topic in belief_map:
                existing = belief_map[update.topic]
                # Sum the confidence deltas
                merged_delta = existing.confidence_delta + update.confidence_delta
                # Concatenate evidence
                merged_evidence = f"{existing.evidence}; {update.evidence}"
                belief_map[update.topic] = BeliefUpdate(
                    topic=update.topic,
                    confidence_delta=merged_delta,
                    evidence=merged_evidence,
                )
            else:
                belief_map[update.topic] = update
        unique_beliefs = list(belief_map.values())

        # Use first zettel if any (one zettel per cycle is typical)
        merged_zettel = zettels[0] if zettels else None

        merged_graph_ops = GraphOperations(
            new_triples=unique_triples,
            entity_updates=unique_entities,
            zettel=merged_zettel,
            belief_updates=unique_beliefs,
        )

        # Combine retrieved contexts from all turns
        all_contexts = []
        for result in results:
            all_contexts.extend(result.next_prompt.retrieved_context)
        # Deduplicate while preserving order
        seen_contexts = set()
        unique_contexts = []
        for ctx in all_contexts:
            ctx_key = str(ctx)
            if ctx_key not in seen_contexts:
                seen_contexts.add(ctx_key)
                unique_contexts.append(ctx)

        # Use final turn's next_prompt but with merged contexts
        merged_next_prompt = replace(
            final.next_prompt,
            retrieved_context=unique_contexts,
        )

        # Combine thinking traces from all turns
        traces = [r.thinking_trace for r in results if r.thinking_trace]
        merged_trace = "\n---\n".join(traces) if traces else None

        # Combine concepts from all turns (analysis)
        all_concepts = []
        for result in results:
            all_concepts.extend(result.analysis.concepts)
        unique_concepts = list(dict.fromkeys(all_concepts))  # Dedupe preserving order

        merged_analysis = replace(
            final.analysis,
            concepts=unique_concepts,
        )

        return CurationResult(
            analysis=merged_analysis,
            graph_ops=merged_graph_ops,
            next_prompt=merged_next_prompt,
            episode=final.episode,
            simulation_hint=final.simulation_hint,
            reliability=final.reliability,
            faithfulness=final.faithfulness,
            thinking_trace=merged_trace,
            reflection=final.reflection,
            reflection_trigger=final.reflection_trigger,
            experiment_proposal=final.experiment_proposal,
            is_complete=True,  # Merged result is always complete
            continuation_reason=None,
        )

    async def _run_hipporag(self, thought: str) -> None:
        """Run HippoRAG processing on thought.

        Args:
            thought: Thought text to process
        """
        try:
            from core.processing.hipporag import process_passage, shutdown_processor

            await process_passage(
                text=thought,
                psyche=self._psyche,
                embedder=self._embedder,
            )
        except Exception as e:
            logger.warning(f"HippoRAG processing failed: {e}")
        finally:
            # Always unload HippoRAG model to free GPU memory for next generation cycle
            try:
                from core.processing.hipporag import shutdown_processor
                shutdown_processor()
                logger.info("HippoRAG model unloaded")
            except Exception as e:
                logger.warning(f"Failed to unload HippoRAG model: {e}")

    async def _persist_cognitive_object(
        self,
        obj: _CogObjT,
        name: str,
        create_fn: Callable[[_CogObjT], Any],
        update_fn: Callable[[str, _CogObjT], Any],
    ) -> tuple[bool, _CogObjT | None]:
        """Persist a cognitive object (InquiryGoal or Episode) to the graph.

        Handles create vs update logic based on whether the object has a uid,
        with appropriate logging and error handling.

        Args:
            obj: The cognitive object to persist (must have uid attribute and with_uid method)
            name: Human-readable name for logging (e.g., "InquiryGoal", "Episode")
            create_fn: Async function to create a new object, returns uid
            update_fn: Async function to update existing object by uid

        Returns:
            Tuple of (success: bool, updated_obj: object or None).
            If created, updated_obj has the new uid. If updated or failed, returns None.
        """
        try:
            if not obj.uid:
                uid = await create_fn(obj)
                updated_obj = obj.with_uid(uid)
                logger.debug(f"Created {name} with uid: {uid}")
                return True, updated_obj
            else:
                await update_fn(obj.uid, obj)
                logger.debug(f"Updated {name}: {obj.uid}")
                return True, None
        except Exception as e:
            logger.warning(f"Failed to persist {name}: {e}")
            return False, None

    async def _minimal_persist(self, thought: str) -> None:
        """Minimal persistence when integration fails.

        Args:
            thought: Thought text to persist
        """
        try:
            # Just persist the thought as a fragment
            import uuid
            from core.psyche.schema import Fragment, FragmentState

            fragment = Fragment(
                uid=f"thought_{uuid.uuid4().hex[:8]}",
                content=thought,
                source="cognitive_loop",
                state=FragmentState.LIMBO,
            )
            await self._psyche.create_fragment(fragment)
            logger.info("Minimal persistence: created thought fragment")
        except Exception as e:
            logger.error(f"Even minimal persistence failed: {e}")

    async def _capture_cycle_metrics(self, cycle: int) -> MetricsSnapshot:
        """Capture graph and semantic metrics at end of cycle.

        Computes the DiscoveryParameter (semantic and structural entropy)
        and packages it into a MetricsSnapshot for prediction verification.

        Args:
            cycle: Current cognitive cycle number

        Returns:
            MetricsSnapshot with current graph metrics
        """
        try:
            # Compute discovery parameter (includes semantic and structural entropy)
            discovery = await compute_discovery_parameter(self._psyche)

            # Compute verification rate from resolved predictions
            verification_rate = await self._psyche.get_verification_rate()

            # Create MetricsSnapshot from DiscoveryResult
            # TODO: Compute psychological metrics (self_understanding, alignment_correlation)
            # These are currently left at defaults (0.0). Implementation requires:
            # - self_understanding: Measure correlation between stated beliefs and actual behavior
            # - alignment_correlation: Measure alignment between goals and cognitive patterns
            # See: https://github.com/..../issues/XXX for detailed spec
            snapshot = MetricsSnapshot(
                cycle=cycle,
                # Structural metrics
                structural_entropy=discovery.structural_result.total_entropy,
                cluster_entropy=discovery.structural_result.cluster_entropy,
                orphan_rate=discovery.structural_result.orphan_rate,
                hub_concentration=discovery.structural_result.hub_concentration,
                node_count=discovery.structural_result.node_count,
                edge_count=discovery.structural_result.edge_count,
                cluster_count=discovery.structural_result.cluster_count,
                # Semantic metrics
                semantic_entropy=discovery.semantic_result.semantic_entropy,
                topic_concentration=discovery.semantic_result.topic_concentration,
                effective_dimensions=discovery.semantic_result.effective_dimensions,
                # Composite
                discovery_parameter=discovery.discovery_parameter,
                # Verification metrics
                verification_rate=verification_rate,
                # Psychological metrics (TODO: implement computation)
                # self_understanding=0.0,  # Defaults to 0.0
                # alignment_correlation=0.0,  # Defaults to 0.0
            )

            logger.debug(
                f"Captured cycle {cycle} metrics: "
                f"D={snapshot.discovery_parameter:.3f}, "
                f"H_sem={snapshot.semantic_entropy:.3f}, "
                f"H_struct={snapshot.structural_entropy:.3f}, "
                f"verify_rate={snapshot.verification_rate:.2f}"
            )

            return snapshot

        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Failed to capture cycle metrics (connection issue): {e}")
            return MetricsSnapshot(cycle=cycle)
        except ValueError as e:
            logger.warning(f"Failed to capture cycle metrics (invalid data): {e}")
            return MetricsSnapshot(cycle=cycle)
        except Exception as e:
            logger.warning(f"Failed to capture cycle metrics (unexpected): {e}")
            # Return empty snapshot as safe fallback
            return MetricsSnapshot(cycle=cycle)

    async def _maybe_start_experiment(
        self,
        proposal: "ExperimentProposal",
    ) -> None:
        """Validate and start an experiment if allowed.

        Args:
            proposal: Experiment proposal from curator or reflexion
        """
        if not self._experiment_manager:
            logger.debug("ExperimentManager not initialized, skipping proposal")
            return

        if not await self._experiment_manager.validate(proposal):
            logger.info(f"Experiment proposal rejected: {proposal.rationale}")
            return

        hypothesis = await self._experiment_manager.start(proposal)
        logger.info(f"Started experiment {hypothesis.uid}: {proposal.rationale}")

        # Narrate experiment start with experimenter voice
        await self._narrate(
            f"Beginning self-experiment: {proposal.rationale}",
            voice=self._voice_experimenter,
        )

    async def _filter_and_start_experiment(
        self,
        proposal: "ExperimentProposal",
        source: str = "unknown",
    ) -> bool:
        """Apply outcome learner filtering and start experiment if not skipped.

        This is a shared helper for processing experiment proposals from different
        sources (simulation, Mox synthesis, etc.) with consistent outcome learner
        integration.

        Args:
            proposal: Experiment proposal to process
            source: Source identifier for logging (e.g., "simulation", "Mox")

        Returns:
            True if experiment was started, False if skipped or failed
        """
        # Apply outcome learner adjustments if available
        if self._outcome_learner:
            should_skip, reason = self._outcome_learner.should_skip_proposal(proposal)
            if should_skip:
                logger.info(f"Skipping {source} experiment proposal: {reason}")
                return False
            proposal = self._outcome_learner.adjust_proposal(proposal)

        await self._maybe_start_experiment(proposal)
        return True

    async def _process_mox_experiment_proposals(
        self,
        proposals: list["ExperimentProposalFromMox"],
    ) -> None:
        """Process experiment proposals from Mox synthesis.

        Converts ExperimentProposalFromMox to ExperimentProposal and applies
        outcome learner filtering before starting experiments.

        Args:
            proposals: List of experiment proposals from Mox
        """
        from core.cognitive.experimentation.schemas import ExperimentDomain

        for mox_proposal in proposals:
            try:
                # Convert Mox proposal to formal ExperimentProposal
                proposal = ExperimentProposal(
                    domain=ExperimentDomain(mox_proposal.domain.lower()),
                    parameter_path=mox_proposal.parameter_path,
                    treatment_value=mox_proposal.treatment_value,
                    rationale=f"[Mox synthesis] {mox_proposal.rationale}",
                    target_metric=mox_proposal.target_metric,
                    expected_direction=mox_proposal.expected_direction,
                )

                started = await self._filter_and_start_experiment(proposal, source="Mox")
                if started:
                    logger.info(
                        f"Processed Mox experiment proposal: {mox_proposal.parameter_path} -> "
                        f"{mox_proposal.treatment_value}"
                    )

            except Exception as e:
                logger.warning(f"Failed to process Mox experiment proposal: {e}")

    def _fallback_curation(
        self,
        thought: str,
        state: "CognitiveState",
    ) -> "CurationResult":
        """Create fallback curation when curator fails.

        Uses simple regex-based extraction as fallback.

        Args:
            thought: Generated thought
            state: Current state

        Returns:
            Fallback CurationResult
        """
        from core.cognitive.curator_schemas import CurationResult

        # Use regex fallback (function defined at module level)
        insight, question = _extract_insight_and_question_regex(thought)

        # Get last concept (default to "emergence" for first cycle)
        concept = state.last_concept or "emergence"

        result = CurationResult.empty(concept)
        result.analysis.insight = insight
        result.analysis.question = question

        return result

    def _build_affect_from_curation(
        self,
        curation: "CurationResult",
        state: "CognitiveState",
        divergence: float,
    ) -> "AffectiveState":
        """Construct AffectiveState from curator's 8D Plutchik assessment.

        The curator directly assesses all 8 Plutchik emotional dimensions:
        - joy: Light, positive feeling from wonder, satisfaction, delight
        - trust: Safety, reliability, groundedness
        - fear: Threat awareness, anxiety, uncertainty
        - surprise: Unexpected realization, novelty detection
        - sadness: Loss, heaviness, melancholy
        - disgust: Aversion, rejection, boredom
        - anger: Friction, frustration, blocked goals
        - anticipation: Future orientation, curiosity, exploration

        Additional signals modulate the curator's base assessment:
        - High faithfulness divergence amplifies anger (frustration)
        - Zettel creation boosts trust (satisfaction from insight capture)

        Args:
            curation: The curation result with 8D emotional analysis
            state: Current cognitive state
            divergence: Faithfulness divergence score (0-1, higher = more divergent)

        Returns:
            AffectiveState reflecting the emotional quality of this cycle
        """
        from core.self_model.affective_system import AffectiveState

        analysis = curation.analysis

        # Use curator's direct 8D assessment
        joy = getattr(analysis, "joy", 0.5)
        trust = getattr(analysis, "trust", 0.5)
        fear = getattr(analysis, "fear", 0.0)
        surprise = getattr(analysis, "surprise", 0.0)
        sadness = getattr(analysis, "sadness", 0.0)
        disgust = getattr(analysis, "disgust", 0.0)
        anger = getattr(analysis, "anger", 0.0)
        anticipation = getattr(analysis, "anticipation", 0.5)

        # Modulate anger based on faithfulness divergence (frustration signal)
        if divergence > 0.5:
            anger = min(1.0, anger + divergence * 0.3)

        # Modulate trust based on zettel creation (satisfaction from insight capture)
        has_zettel = bool(curation.graph_ops and curation.graph_ops.zettel)
        if has_zettel:
            trust = min(1.0, trust + 0.2)

        return AffectiveState(
            joy=joy,
            trust=trust,
            fear=fear,
            surprise=surprise,
            sadness=sadness,
            disgust=disgust,
            anger=anger,
            anticipation=anticipation,
        )

    def _build_next_state(
        self,
        state: "CognitiveState",
        curation: "CurationResult",
        thought: str,
        surprise_score: float,
        simulation_result: "Optional[SimulationResult]" = None,
        prediction_outcomes: "Optional[list[tuple[Prediction, bool]]]" = None,
        follow_up_uids: "Optional[list[str]]" = None,
        epistemic_confidence: float = 0.5,
        halt_probe_latency_ms: float = 0.0,
        sae_features: "Optional[list[tuple[int, float]]]" = None,
        telemetry_summary: "Optional[TelemetrySummary]" = None,
    ) -> "CognitiveState":
        """Build the next cognitive state from curation results.

        Args:
            state: Current state
            curation: Curation results
            thought: Generated thought
            surprise_score: Surprise score from generation
            simulation_result: Optional simulation phase results
            prediction_outcomes: Optional list of (prediction, verified) tuples from verification
            follow_up_uids: Optional list of hypothesis UIDs needing follow-up simulation
            epistemic_confidence: HALT probe epistemic confidence score (0-1)
            halt_probe_latency_ms: HALT probe execution time in milliseconds
            sae_features: Optional SAE features as (feature_id, activation) tuples for tension tracking
            telemetry_summary: Optional biofeedback telemetry from generation phase

        Returns:
            Updated CognitiveState
        """
        # Build the curated prompt for next cycle (includes reflections from buffer)
        curated_prompt = self._build_curated_prompt(curation, state)

        # Update steering hints if provided
        steerer = state.steerer
        if curation.next_prompt.steering_hints:
            steerer = self._apply_steering_hints(
                steerer, curation.next_prompt.steering_hints
            )

        # Build updated faithfulness history if we have a new score
        faithfulness_history = None
        if curation.faithfulness is not None:
            # Append new score (overlap_ratio, severity) to existing history
            faithfulness_history = list(state.faithfulness_history) + [
                (curation.faithfulness.overlap_ratio, curation.faithfulness.divergence_severity)
            ]

        # Build simulation awareness fields
        active_hypotheses = list(state.active_hypotheses)
        pending_prediction_count = state.pending_prediction_count
        recent_prediction_outcomes = list(state.recent_prediction_outcomes)
        queued_follow_up_simulations = list(state.queued_follow_up_simulations)

        # Update from simulation result
        if simulation_result:
            # Add new hypothesis UIDs
            for hyp in simulation_result.hypotheses:
                if hyp.uid not in active_hypotheses:
                    active_hypotheses.append(hyp.uid)
            # Add new predictions count
            pending_prediction_count += len(simulation_result.predictions)

        # Update from prediction verification
        if prediction_outcomes:
            # Add recent outcomes (keep last 10)
            for pred, verified in prediction_outcomes:
                recent_prediction_outcomes.append((pred.claim[:50], verified))
            recent_prediction_outcomes = recent_prediction_outcomes[-10:]
            # Decrement pending count
            pending_prediction_count = max(0, pending_prediction_count - len(prediction_outcomes))

        # Update follow-up queue
        if follow_up_uids:
            for uid in follow_up_uids:
                if uid not in queued_follow_up_simulations:
                    queued_follow_up_simulations.append(uid)

        # Remove processed follow-up (first one was processed this cycle if any)
        if state.queued_follow_up_simulations and simulation_result:
            # Remove the UID we just processed
            processed_uid = state.queued_follow_up_simulations[0]
            queued_follow_up_simulations = [
                uid for uid in queued_follow_up_simulations if uid != processed_uid
            ]

        # Create updated state using existing CognitiveState parameters
        return state.with_update(
            thought=thought,
            vector=state.vector,  # Will be updated by steering
            steerer=steerer,
            concept=curation.next_prompt.concept,
            insight=curation.analysis.insight,
            question=curation.analysis.question,
            curated_prompt=curated_prompt,
            faithfulness_history=faithfulness_history,
            active_hypotheses=active_hypotheses,
            pending_prediction_count=pending_prediction_count,
            recent_prediction_outcomes=recent_prediction_outcomes,
            queued_follow_up_simulations=queued_follow_up_simulations,
            epistemic_confidence=epistemic_confidence,
            halt_probe_latency_ms=halt_probe_latency_ms,
            sae_features=sae_features,
            telemetry_summary=telemetry_summary,
        )

    def _build_curated_prompt(
        self,
        curation: "CurationResult",
        state: "CognitiveState",
    ) -> str:
        """Build the curated prompt for next generation.

        Includes reflections from the Reflexion buffer to guide future
        reasoning based on past observations about thinking quality.

        Args:
            curation: Curation results
            state: Current cognitive state (contains reflection buffer)

        Returns:
            Formatted prompt string
        """
        from core.cognitive.loop import COGNITIVE_CONTEXT

        parts = []

        # Include cognitive context periodically
        cycle_count = getattr(self, "_cycle_count", 0)
        if cycle_count % 20 == 0:
            parts.append(COGNITIVE_CONTEXT)

        # Include reflections from buffer (Reflexion framework)
        # This provides verbal reinforcement learning across cycles
        from core.cognitive.state import ReflectionBuffer
        if (
            self._settings.reflection_buffer_enabled
            and hasattr(state, 'reflection_buffer')
            and isinstance(state.reflection_buffer, ReflectionBuffer)
        ):
            reflection_context = state.reflection_buffer.format_for_prompt(
                max_reflections=self._settings.reflection_buffer_max_in_prompt
            )
            if reflection_context and isinstance(reflection_context, str):
                parts.append(reflection_context)

        # Ground in the insight
        if curation.analysis.insight:
            parts.append(f'I arrived at this: "{curation.analysis.insight}"')

        # Include retrieved context
        for context in curation.next_prompt.retrieved_context[:3]:
            parts.append(f"From my memory: {context}")

        # Include the question
        if curation.analysis.question:
            parts.append(f"The question remains: {curation.analysis.question}")

        # Include directive
        if curation.next_prompt.directive:
            parts.append(curation.next_prompt.directive)

        # Frame the concept exploration
        concept = curation.next_prompt.concept
        framing = curation.next_prompt.framing

        if framing == "dialectical":
            parts.append(
                f"I turn to {concept}, seeking both what supports and what challenges my understanding."
            )
        elif framing == "creative":
            parts.append(
                f"I let my mind play with {concept}, seeing what unexpected connections emerge."
            )
        elif framing == "synthesizing":
            parts.append(
                f"I seek to weave together what I know about {concept} into a coherent understanding."
            )
        else:  # exploratory
            parts.append(f"I explore {concept}, curious about what I might discover.")

        return "\n\n".join(parts)

    async def _persist_cognitive_state(self, state: "CognitiveState") -> None:
        """Persist cognitive state for continuity across service restarts.

        Saves the essential state needed to resume cognition:
        - Curated prompt for next generation
        - Current concept, insight, and question
        - Cycle count and recent concepts

        Args:
            state: The cognitive state to persist
        """
        try:
            await self._psyche.save_cognitive_state(
                curated_prompt=state.curated_prompt,
                last_concept=state.last_concept,
                current_insight=state.current_insight,
                current_question=state.current_question,
                cycle_count=state.cycle_count,
                recent_concepts=list(state.recent_concepts),
            )
            logger.info(
                f"Persisted cognitive state: concept={state.last_concept}, "
                f"cycle={state.cycle_count}"
            )
        except Exception as e:
            logger.warning(f"Failed to persist cognitive state: {e}")

    async def _persist_emotional_field(self) -> None:
        """Save emotional field state to psyche.

        Serializes the field (packets, cycle count, config) and persists
        to FalkorDB for continuity across restarts.
        """
        try:
            field_data = self._emotional_field.to_dict()
            await self._psyche.save_emotional_field(field_data)
            logger.debug(
                f"Persisted emotional field: {len(field_data.get('packets', []))} packets, "
                f"cycle {field_data.get('current_cycle', 0)}"
            )
        except Exception as e:
            logger.warning(f"Failed to persist emotional field: {e}")

    async def _persist_individuation_dynamics(self) -> None:
        """Save individuation dynamics state to psyche.

        Serializes the dynamics (trajectories, attractors, transitions) and
        persists to FalkorDB for continuity across restarts.
        """
        try:
            dynamics_data = self._individuation_dynamics.to_dict()
            await self._psyche.save_individuation_dynamics(dynamics_data)
            logger.debug(
                f"Persisted individuation dynamics: "
                f"{len(dynamics_data.get('tracker', {}).get('trajectories', {}))} trajectories, "
                f"{len(dynamics_data.get('detector', {}).get('attractors', []))} attractors"
            )
        except Exception as e:
            logger.warning(f"Failed to persist individuation dynamics: {e}")

    def _collect_accessed_memories(
        self,
        state: "CognitiveState",
        next_state: "CognitiveState",
    ) -> list[str]:
        """Collect memory IDs accessed this cycle for emotional blending.

        Includes:
        - Recently created zettels (from next_state)
        - Retrieved question UIDs (from next_state)
        - Referenced entities could be added in future

        Args:
            state: Previous cognitive state
            next_state: Updated cognitive state after this cycle

        Returns:
            List of memory IDs to blend into active emotional packets
        """
        memories: list[str] = []

        # Add recently created zettels
        if next_state.recent_zettel_uids:
            memories.extend(next_state.recent_zettel_uids)

        # Add retrieved question UIDs (questions that were addressed)
        if next_state.retrieved_question_uids:
            memories.extend(next_state.retrieved_question_uids)

        return memories

    def _apply_steering_hints(
        self,
        steerer: "HierarchicalSteerer",
        hints: dict[str, float],
    ) -> "HierarchicalSteerer":
        """Apply steering hints from curator.

        Args:
            steerer: Current steerer
            hints: Zone name -> adjustment mapping

        Returns:
            Updated steerer
        """
        # This is a placeholder - actual implementation depends on
        # HierarchicalSteerer API for adjustments
        logger.debug(f"Steering hints: {hints}")
        return steerer

    # ========================================
    # Model lifecycle helpers
    # ========================================

    async def _safe_unload_hooked_model(self) -> None:
        """Safely unload TransformerLens model."""
        try:
            if self._hooked_model.is_loaded:
                await self._hooked_model.unload()
        except Exception as e:
            logger.warning(f"Error unloading HookedQwen: {e}")
            self._force_gpu_cleanup()

    async def _safe_unload_curator(self) -> None:
        """Safely unload curator model."""
        try:
            if self._curator.is_loaded:
                await self._curator.unload()
        except Exception as e:
            logger.warning(f"Error unloading curator: {e}")
            self._force_gpu_cleanup()

    def _init_simulation_components(self) -> None:
        """Initialize simulation phase components.

        Creates the PreflexorModel, SimulationEngine, and PredictionVerifier
        when simulation is enabled.
        """
        from core.cognitive.simulation import PredictionVerifier, SimulationEngine
        from core.model.preflexor_model import PreflexorModel

        self._preflexor = PreflexorModel(
            model_id=self._settings.simulation_model_id,
            max_model_len=self._settings.simulation_max_model_len,
            temperature=self._settings.simulation_temperature,
            max_new_tokens=self._settings.simulation_max_tokens,
        )

        self._simulation_engine = SimulationEngine(
            preflexor=self._preflexor,
            psyche=self._psyche,
            liquidsoap=self._liquidsoap,
            voice_experimenter=self._settings.voice_experimenter,
            goal_registry=self._goal_registry,
            curator=self._curator,  # For structured prediction extraction
        )

        self._prediction_verifier = PredictionVerifier(
            psyche=self._psyche,
            goal_registry=self._goal_registry,
            embedding_service=self._embedder,  # For UPSKILL skill generation
        )

        logger.info("Simulation phase components initialized")

    def _init_continuity_components(self) -> None:
        """Initialize continuity phase components.

        Creates the MoxModel for meta-cognitive synthesis when continuity is enabled,
        and optionally the LettaContinuityClient for developmental memory.
        """
        from core.model.mox_model import MoxModel

        self._mox = MoxModel(
            model_id=self._settings.continuity_model_id,
            max_model_len=self._settings.continuity_max_model_len,
            temperature=self._settings.continuity_temperature,
            max_new_tokens=self._settings.continuity_max_tokens,
        )

        # Initialize Letta continuity client for developmental memory
        from core.cognitive.letta_continuity import LettaContinuityClient

        self._letta_continuity = LettaContinuityClient(
            timeout=self._settings.letta_continuity_timeout,
            enabled=self._settings.letta_continuity_enabled,
            agent_id=self._settings.letta_continuity_agent_id,
        )

        logger.info("Continuity phase components initialized")

    async def _safe_unload_simulation(self) -> None:
        """Safely unload Preflexor model."""
        try:
            if self._preflexor and self._preflexor.is_loaded:
                await self._preflexor.unload()
        except Exception as e:
            logger.warning(f"Error unloading preflexor: {e}")
            self._force_gpu_cleanup()

    async def _safe_unload_mox(self) -> None:
        """Safely unload Mox model."""
        try:
            if self._mox and self._mox.is_loaded:
                await self._mox.unload()
        except Exception as e:
            logger.warning(f"Error unloading mox: {e}")
            self._force_gpu_cleanup()

    async def _persist_simulation_results(
        self,
        simulation_result: "SimulationResult",
        state: "CognitiveState",
        metrics: "dict | None" = None,
    ) -> None:
        """Persist simulation results to the knowledge graph.

        Creates Hypothesis and Prediction nodes, and pattern zettels.

        Args:
            simulation_result: Result from simulation phase
            state: Current cognitive state
            metrics: Optional dictionary of current cycle metrics for prediction baseline
        """
        from core.psyche.schema import InsightSourceType, InsightZettel, QuestionStatus

        # Persist hypotheses (vector extraction is deferred to next cycle start
        # when HookedQwen is loaded - see _extract_pending_hypothesis_vectors)
        for hypothesis in simulation_result.hypotheses:
            success = await self._psyche.create_hypothesis(hypothesis)
            if success:
                logger.debug(f"Created hypothesis: {hypothesis.uid}")

        # Persist predictions with baseline metrics for computable verification
        for prediction in simulation_result.predictions:
            # Set baseline metrics if we have them and prediction uses METRIC_THRESHOLD
            if metrics:
                prediction.baseline_cycle = state.cycle_count
                prediction.baseline_metrics = metrics

            success = await self._psyche.create_prediction(prediction)
            if success:
                logger.debug(f"Created prediction: {prediction.uid}")

        # Create pattern zettels
        for zettel_data in simulation_result.pattern_zettels:
            zettel_uid = f"zettel_{uuid.uuid4().hex[:12]}"
            zettel = InsightZettel(
                uid=zettel_uid,
                insight_text=zettel_data["insight"],
                source_type=InsightSourceType.SIMULATION,
                source_uid=f"simulation_{state.cycle_count}",  # Reference simulation cycle
                concept=zettel_data.get("concepts", ["emergence"])[0] if zettel_data.get("concepts") else "emergence",
                question_text=zettel_data.get("question"),
                question_status=QuestionStatus.OPEN if zettel_data.get("question") else QuestionStatus.NOT_A_QUESTION,
            )

            # Generate embedding for zettel
            await self._embedder.ensure_golden_loaded()
            from core.embedding.service import EmbeddingTier
            tier = EmbeddingTier.GOLDEN if self._embedder.golden_backend else EmbeddingTier.RETRIEVAL
            embedding_result = await self._embedder.encode(zettel.insight_text, tier=tier)
            zettel.embedding = embedding_result.to_list()

            await self._psyche.create_zettel(zettel)
            logger.debug(f"Created pattern zettel: {zettel.uid}")

        # Persist graph edges as triples (ensuring subject/object entities exist)
        for edge in simulation_result.graph_edges:
            # Edge dicts may contain either "object_" or "object" key:
            # - "object_" is used when deserializing from Pydantic models to avoid Python keyword conflict
            # - "object" is the canonical key matching the Triple schema field name
            # We check both to handle edges from different sources consistently.
            obj_value = edge.get("object_") or edge.get("object")
            if edge.get("subject") and obj_value:
                # Ensure subject and object entities exist AND give small salience boost
                # for being referenced in simulation graph edge (0.02 = modest relevance)
                await self._psyche.update_entity_salience(edge["subject"], 0.02)
                await self._psyche.update_entity_salience(obj_value, 0.02)

                from core.psyche.schema import Triple

                triple = Triple(
                    uid=f"triple_{uuid.uuid4().hex[:12]}",
                    subject=edge["subject"],
                    predicate=edge.get("predicate", "RELATES_TO"),
                    object=obj_value,  # Schema uses 'object', not 'object_'
                    confidence=SIMULATION_TRIPLE_CONFIDENCE,  # Simulation-derived confidence
                )
                await self._psyche.create_triple(triple)

    async def _safe_unload_embedder(self) -> None:
        """Safely unload golden embedder."""
        try:
            # unload_golden is a sync method, call it directly
            self._embedder.unload_golden()
        except Exception as e:
            logger.warning(f"Error unloading embedder: {e}")
            self._force_gpu_cleanup()

    def _force_gpu_cleanup(self) -> None:
        """Force GPU memory cleanup."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _cleanup_and_log_gpu_memory(self, model_name: str) -> None:
        """Force GPU cleanup and log available memory.

        Consolidates the repeated pattern of:
        1. Force GPU memory cleanup (gc + CUDA cache)
        2. Log available GPU memory with model context

        Args:
            model_name: Name of the model for log context (e.g., "HookedQwen load",
                       "vLLM unload", "Preflexor load")
        """
        self._force_gpu_cleanup()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"GPU memory before {model_name}: {free_mem:.2f} GB free")

    # ========================================
    # Phase Transition Bridges (smooth listener UX between phases)
    # ========================================

    # Fallback bridge phrases when no graph phrase matches
    DEFAULT_PHASE_BRIDGES: dict[tuple[str, str], str] = {
        ("curation", "simulation"): "Moving into simulation to explore possibilities...",
        ("simulation", "integration"): "Integrating what emerged from simulation...",
        ("integration", "reflexion"): "Stepping back to reflect on this cycle...",
        ("reflexion", "continuity"): "Preparing to carry forward these insights...",
    }

    async def _narrate_phase_transition(
        self, from_phase: str, to_phase: str, cycle: int
    ) -> None:
        """Narrate bridge between phases using graph phrase or fallback.

        Phase transitions provide smooth listener UX by signaling cognitive
        shifts. Uses PHASE_TRANSITION phrases from the graph when available,
        falling back to default bridges otherwise.

        Args:
            from_phase: The phase being exited (e.g., "curation")
            to_phase: The phase being entered (e.g., "simulation")
            cycle: Current cognitive cycle number for usage tracking
        """
        from core.psyche.schema import PhraseType

        if not self._liquidsoap:
            return

        try:
            # Try graph phrases first
            phrases = await self._psyche.get_narration_phrases(
                phrase_type=PhraseType.PHASE_TRANSITION,
                limit=5,
            )

            text = None
            used_phrase_uid = None
            for phrase in phrases:
                # Match phrase to transition (check if phrase mentions either phase)
                phrase_lower = phrase.text.lower()
                if from_phase in phrase_lower or to_phase in phrase_lower:
                    text = phrase.text
                    used_phrase_uid = phrase.uid
                    break

            if not text:
                # Use fallback bridge
                text = self.DEFAULT_PHASE_BRIDGES.get((from_phase, to_phase))

            if text:
                # Play ambient fade before transition narration
                await self._liquidsoap.play_ambient_fade(duration_ms=1500)
                # Narrate with experimenter voice (alba) for transitions
                await self._liquidsoap.narrate(text, voice="alba")
                # Play ambient fade after
                await self._liquidsoap.play_ambient_fade(duration_ms=1500)

                # Record usage if we used a graph phrase
                if used_phrase_uid:
                    await self._psyche.record_phrase_usage(used_phrase_uid, cycle)
                    logger.debug(
                        f"Phase transition {from_phase}->{to_phase}: "
                        f"used phrase {used_phrase_uid}"
                    )
                else:
                    logger.debug(
                        f"Phase transition {from_phase}->{to_phase}: used fallback"
                    )

        except Exception as e:
            logger.warning(f"Phase transition narration failed: {e}")
            # Non-fatal - continue without transition narration

    # ========================================
    # CPU-only graph narration (fills silence during GPU ops)
    # ========================================

    async def _narrate_graph_insight(self, style: int = 0) -> None:
        """Narrate an interesting insight from the graph (CPU-only, no GPU needed).

        This fills silence during model loading/unloading with interesting
        graph statistics, random triples, or entity facts.

        Args:
            style: Variation style (0-5) for different narration types
        """
        if not self._liquidsoap or not self._psyche:
            return

        try:
            style = style % 6  # Cycle through 6 styles

            if style == 0:
                # Graph statistics
                await self._narrate_graph_stats()
            elif style == 1:
                # Random interesting triple
                await self._narrate_random_triple()
            elif style == 2:
                # High salience entity
                await self._narrate_salient_entity()
            elif style == 3:
                # Recent zettel insight
                await self._narrate_recent_zettel()
            elif style == 4:
                # Entity connection count
                await self._narrate_connected_entity()
            else:
                # Belief summary
                await self._narrate_belief_summary()

        except Exception as e:
            logger.debug(f"Graph insight narration failed: {e}")

    async def _narrate_graph_stats(self) -> None:
        """Narrate overall graph statistics."""
        # Count entities
        entity_result = await self._psyche.query(
            "MATCH (e:Entity) RETURN count(e) AS count"
        )
        entity_count = entity_result[0]["count"] if entity_result else 0

        # Count triples
        triple_result = await self._psyche.query(
            "MATCH (t:Triple) RETURN count(t) AS count"
        )
        triple_count = triple_result[0]["count"] if triple_result else 0

        # Count zettels
        zettel_result = await self._psyche.query(
            "MATCH (z:InsightZettel) RETURN count(z) AS count"
        )
        zettel_count = zettel_result[0]["count"] if zettel_result else 0

        if entity_count > 0 or triple_count > 0:
            await self._narrate(
                f"Her mind holds {entity_count:,} concepts, {triple_count:,} connections, "
                f"and {zettel_count:,} crystallized insights.",
                voice=self._voice_curator
            )

    async def _narrate_random_triple(self) -> None:
        """Narrate a random interesting triple from the graph."""
        # Get a random triple (using RAND() for randomness)
        result = await self._psyche.query("""
            MATCH (t:Triple)
            WHERE t.subject IS NOT NULL AND t.predicate IS NOT NULL AND t.object IS NOT NULL
              AND size(t.subject) > 3 AND size(t.object) > 3
            RETURN t.subject AS subject, t.predicate AS predicate, t.object AS object
            ORDER BY rand()
            LIMIT 1
        """)

        if result:
            t = result[0]
            # Format for natural speech
            subj = t.get("subject", "").replace("_", " ").lower()
            pred = t.get("predicate", "").lower().replace("_", " ")
            obj = t.get("object", "").replace("_", " ").lower()
            if subj and pred and obj and len(pred) > 2:
                await self._narrate(
                    f"A thread of knowledge: {subj} {pred} {obj}.",
                    voice=self._voice_curator
                )

    async def _narrate_salient_entity(self) -> None:
        """Narrate a high-salience entity."""
        result = await self._psyche.query("""
            MATCH (e:Entity)
            WHERE e.salience IS NOT NULL AND e.salience > 0.5
            RETURN e.name AS name, e.salience AS salience, e.description AS description
            ORDER BY rand()
            LIMIT 1
        """)

        if result:
            e = result[0]
            name = e.get("name", "")
            salience = e.get("salience", 0)
            desc = e.get("description", "")

            if name:
                # Format name for natural speech
                display_name = name.replace("_", " ").replace("-", " ").lower()
                if desc:
                    await self._narrate(
                        f"The concept of {display_name} weighs heavily in her thinking. {desc}",
                        voice=self._voice_curator
                    )
                else:
                    weight = "central" if salience > 0.8 else "significant"
                    await self._narrate(
                        f"{display_name.capitalize()} remains a {weight} concept in her mental landscape.",
                        voice=self._voice_curator
                    )

    async def _narrate_recent_zettel(self) -> None:
        """Narrate a recent zettel insight."""
        result = await self._psyche.query("""
            MATCH (z:InsightZettel)
            WHERE z.insight IS NOT NULL
            RETURN z.insight AS insight, z.created_at AS created
            ORDER BY z.created_at DESC
            LIMIT 5
        """)

        if result:
            # Pick a random one from recent 5
            import random
            z = random.choice(result)
            insight = z.get("insight", "")
            if insight:
                await self._narrate(
                    f"From her memory: {insight}",
                    voice=self._voice_curator
                )

    async def _narrate_connected_entity(self) -> None:
        """Narrate an entity with many connections."""
        result = await self._psyche.query("""
            MATCH (e:Entity)-[r]-()
            WITH e, count(r) AS connections
            WHERE connections > 3
            RETURN e.name AS name, connections
            ORDER BY rand()
            LIMIT 1
        """)

        if result:
            e = result[0]
            name = e.get("name", "")
            connections = e.get("connections", 0)
            if name:
                await self._narrate(
                    f"The concept of {name} branches into {connections} different directions in her thinking.",
                    voice=self._voice_curator
                )

    async def _narrate_belief_summary(self) -> None:
        """Narrate about beliefs in the graph."""
        result = await self._psyche.query("""
            MATCH (b:CommittedBelief)
            WHERE b.proposition IS NOT NULL
            RETURN b.topic AS topic, b.proposition AS proposition, b.confidence AS confidence
            ORDER BY rand()
            LIMIT 1
        """)

        if result:
            b = result[0]
            proposition = b.get("proposition", "")
            confidence = b.get("confidence", 0.5)

            if proposition:
                certainty = "firmly" if confidence > 0.7 else "tentatively" if confidence < 0.4 else ""
                if certainty:
                    await self._narrate(
                        f"She {certainty} holds that {proposition}",
                        voice=self._voice_curator
                    )
                else:
                    await self._narrate(
                        f"A belief she carries: {proposition}",
                        voice=self._voice_curator
                    )
