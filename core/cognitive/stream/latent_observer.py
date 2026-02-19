"""Latent State Observer for PLaT-Lite Reasoning Observation.

Extracts and interprets latent reasoning states from chunk activations
during generation. Provides cognitive mode tracking, emotional state
extraction, and trajectory delta computation.

Architecture:
    LatentObserver receives ChunkResult objects and produces LatentObservation
    objects containing:
    - SAE features with interpretations
    - Cognitive mode similarities (10 semantic anchors)
    - Plutchik 8D emotional state
    - Trajectory delta (what changed since last observation)

Key Insight from PLaT:
    Latent states encode a "superposition of reasoning trajectories."
    By observing these states, we can track mode shifts, emotional rises,
    and emerging patterns before they collapse into text.

Usage:
    from core.cognitive.stream.latent_observer import LatentObserver

    observer = LatentObserver(
        transcoder=transcoder,
        feature_interpreter=interpreter,
        anchor_service=anchors,
    )

    for chunk in chunks:
        observation = await observer.observe(chunk)
        # observation.cognitive_modes: {"philosophical_inquiry": 0.82, ...}
        # observation.trajectory_delta: "shifted from technical to philosophical"

Reference:
    PLaT (Planning with Latent Thoughts, arXiv:2601.21358) - latent states
    reveal cognitive mode, planning strategy, and reasoning trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch
    from core.cognitive.stream.chunked_generator import ChunkResult
    from core.sae.transcoder import TranscoderManager, ActiveFeature
    from core.sae.feature_interpreter import FeatureInterpreter
    from core.cognitive.anchors import AnchorSimilarityService

# Handle optional torch dependency
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

# Plutchik primary emotions (must match order in affective_system)
PLUTCHIK_EMOTIONS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]

# Keywords for mapping features to emotions
# Features containing these keywords contribute to that emotion
EMOTION_KEYWORDS = {
    "joy": ["happy", "joy", "delight", "pleased", "positive", "good"],
    "trust": ["trust", "reliable", "confident", "faith", "believe"],
    "fear": ["fear", "anxious", "worry", "danger", "threat", "afraid"],
    "surprise": ["surprise", "unexpected", "sudden", "novel", "new", "wonder"],
    "sadness": ["sad", "loss", "grief", "regret", "sorrow", "disappointed"],
    "disgust": ["disgust", "reject", "aversion", "avoid", "wrong", "bad"],
    "anger": ["anger", "frustrat", "annoy", "hostile", "conflict"],
    "anticipation": ["expect", "anticipat", "future", "predict", "await", "hope"],
}


@dataclass
class PlutchikState:
    """8D Plutchik emotional state.

    Attributes:
        emotions: Dict mapping emotion name to intensity [0, 1]
        dominant: The emotion with highest intensity
        intensity: Mean intensity across all emotions
    """
    emotions: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure all emotions are present
        for emotion in PLUTCHIK_EMOTIONS:
            if emotion not in self.emotions:
                self.emotions[emotion] = 0.0

    @property
    def dominant(self) -> str:
        """Get the dominant emotion."""
        if not self.emotions:
            return "neutral"
        return max(self.emotions, key=self.emotions.get)

    @property
    def intensity(self) -> float:
        """Get mean intensity across all emotions."""
        if not self.emotions:
            return 0.0
        return sum(self.emotions.values()) / len(self.emotions)

    def to_vector(self) -> np.ndarray:
        """Convert to 8D vector in standard order."""
        return np.array([self.emotions.get(e, 0.0) for e in PLUTCHIK_EMOTIONS])


@dataclass
class LatentObservation:
    """Interpreted observation of a latent state.

    Attributes:
        chunk_idx: Index of the chunk this observation is from
        features: Raw SAE features with activations
        interpretations: Human-readable labels for top features
        cognitive_modes: 10 anchor similarities {"mode_name": score}
        emotional_state: 8D Plutchik emotional reading
        trajectory_delta: Description of what changed from previous chunk
        activation_norm: L2 norm of raw activations (for tracking)
    """
    chunk_idx: int
    features: list  # list[ActiveFeature]
    interpretations: list[str]
    cognitive_modes: dict[str, float]
    emotional_state: PlutchikState
    trajectory_delta: Optional[str]
    activation_norm: float = 0.0


class LatentObserver:
    """Observes and interprets latent reasoning states.

    Maintains a history of observations for trajectory tracking and
    provides extraction of cognitive modes, emotions, and feature
    interpretations from chunk activations.

    Constants:
        TOP_K_FEATURES: Number of top SAE features to extract (10)
        MODE_SHIFT_THRESHOLD: Min difference to report mode shift (0.15)
        EMOTION_THRESHOLD: Min intensity to consider emotion active (0.2)
        EMOTIONAL_RISE_THRESHOLD: Min increase to report emotional rise (0.2)
    """

    TOP_K_FEATURES = 10
    MODE_SHIFT_THRESHOLD = 0.15
    EMOTION_THRESHOLD = 0.2
    EMOTIONAL_RISE_THRESHOLD = 0.2

    def __init__(
        self,
        transcoder: Optional["TranscoderManager"] = None,
        feature_interpreter: Optional["FeatureInterpreter"] = None,
        anchor_service: Optional["AnchorSimilarityService"] = None,
    ):
        """Initialize the observer.

        Args:
            transcoder: SAE transcoder for feature extraction
            feature_interpreter: Logit lens interpreter for feature labels
            anchor_service: Semantic anchor service for cognitive modes
        """
        self._transcoder = transcoder
        self._interpreter = feature_interpreter
        self._anchors = anchor_service
        self._history: list[LatentObservation] = []

    def reset(self) -> None:
        """Clear observation history for new generation."""
        self._history = []

    @property
    def history(self) -> list[LatentObservation]:
        """Access observation history."""
        return self._history

    async def observe(self, chunk: "ChunkResult") -> LatentObservation:
        """Extract and interpret latent state from chunk activations.

        Args:
            chunk: ChunkResult with activations from ChunkedGenerator

        Returns:
            LatentObservation with features, modes, emotions, and delta
        """
        # 1. Extract SAE features
        features, interpretations = await self._extract_features(chunk)

        # 2. Compute cognitive mode similarities
        cognitive_modes = await self._compute_cognitive_modes(chunk)

        # 3. Extract emotional state
        emotional_state = self._extract_emotional_state(features, interpretations)

        # 4. Compute trajectory delta
        trajectory_delta = self._compute_trajectory_delta(
            cognitive_modes, emotional_state, interpretations
        )

        # 5. Compute activation norm for tracking
        activation_norm = 0.0
        if chunk.activations is not None:
            activation_norm = float(chunk.activations.norm().item())

        observation = LatentObservation(
            chunk_idx=chunk.chunk_idx,
            features=features,
            interpretations=interpretations,
            cognitive_modes=cognitive_modes,
            emotional_state=emotional_state,
            trajectory_delta=trajectory_delta,
            activation_norm=activation_norm,
        )

        self._history.append(observation)

        logger.debug(
            f"[LATENT_OBSERVER] Chunk {chunk.chunk_idx}: "
            f"dominant={self._get_dominant_mode(cognitive_modes)}, "
            f"delta={trajectory_delta}"
        )

        return observation

    async def _extract_features(
        self, chunk: "ChunkResult"
    ) -> tuple[list, list[str]]:
        """Extract and interpret SAE features from chunk.

        Args:
            chunk: ChunkResult with MLP input activations

        Returns:
            Tuple of (features list, interpretations list)
        """
        if not self._transcoder or chunk.mlp_input is None:
            return [], []

        try:
            # Encode activations to sparse features
            # Use MLP input which is what the transcoder expects
            # Move to CPU if transcoder is on CPU (activations may be on CUDA)
            mlp_input = chunk.mlp_input
            if hasattr(mlp_input, 'device') and str(mlp_input.device) != 'cpu':
                mlp_input = mlp_input.cpu()
            features_tensor = self._transcoder.encode(mlp_input)
            features = self._transcoder.get_active_features(
                features_tensor, top_k=self.TOP_K_FEATURES
            )

            # Get interpretations via logit lens
            interpretations = []
            if self._interpreter and self._interpreter.is_initialized:
                for f in features:
                    try:
                        interp = self._interpreter.get_feature_interpretation(f.index)
                        interpretations.append(interp.interpretation)
                    except Exception:
                        interpretations.append(f"feature_{f.index}")
            else:
                interpretations = [f"feature_{f.index}" for f in features]

            # Log successful extraction at INFO level (first chunk only to avoid spam)
            if features and chunk.chunk_idx == 0:
                top_features = [f.index for f in features[:3]]
                logger.info(
                    f"[LATENT_OBSERVER] SAE features extracted: "
                    f"top_k={len(features)}, top_indices={top_features}"
                )

            return features, interpretations

        except Exception as e:
            logger.warning(f"[LATENT_OBSERVER] Feature extraction failed: {e}")
            return [], []

    async def _compute_cognitive_modes(
        self, chunk: "ChunkResult"
    ) -> dict[str, float]:
        """Compute similarity to 10 cognitive mode anchors.

        Uses the chunk's text content to compute semantic similarity
        to anchor definitions via the embedding service.

        Note: Does NOT use raw activations - anchor embeddings are in a
        different space (1024-dim embedding model vs 4096-dim model hidden).

        Args:
            chunk: ChunkResult with text

        Returns:
            Dict mapping mode name to similarity score
        """
        if not self._anchors:
            return {}

        # Need meaningful text to compute modes
        if not chunk.text or len(chunk.text.strip()) < 10:
            return {}

        try:
            # Use text-based similarity - let anchor service compute embedding
            # This uses the embedding service which produces compatible 1024-dim vectors
            result = await self._anchors.compute_similarities(
                thought=chunk.text,
                thought_embedding=None,  # Service computes proper embedding
            )

            return result.similarities

        except Exception as e:
            logger.warning(f"[LATENT_OBSERVER] Mode computation failed: {e}")
            return {}

    def _project_to_embedding_space(self, activations: "torch.Tensor") -> np.ndarray:
        """Project layer 16 activations to embedding space.

        Mean pools across sequence and batch dimensions to get a single
        vector for comparison with anchor embeddings.

        Args:
            activations: Tensor of shape [batch, seq, d_model]

        Returns:
            Numpy array of shape [d_model] or matching anchor dims
        """
        # Mean pool across sequence and batch
        pooled = activations.mean(dim=1)  # [batch, d_model]
        if pooled.dim() > 1:
            pooled = pooled.squeeze(0)  # [d_model]

        return pooled.cpu().float().numpy()

    def _extract_emotional_state(
        self,
        features: list,
        interpretations: list[str],
    ) -> PlutchikState:
        """Extract 8D emotional state from features.

        Maps SAE feature interpretations to Plutchik emotions using
        keyword matching. Normalizes to [0, 1] range.

        Args:
            features: List of ActiveFeature objects
            interpretations: Human-readable feature labels

        Returns:
            PlutchikState with emotion intensities
        """
        emotion_scores = {e: 0.0 for e in PLUTCHIK_EMOTIONS}

        if not features or not interpretations:
            return PlutchikState(emotions=emotion_scores)

        # Score each emotion based on feature interpretations
        for feature, interp in zip(features, interpretations):
            interp_lower = interp.lower()
            activation = getattr(feature, "activation", 0.5)

            for emotion, keywords in EMOTION_KEYWORDS.items():
                if any(kw in interp_lower for kw in keywords):
                    emotion_scores[emotion] += activation

        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {
                e: v / total
                for e, v in emotion_scores.items()
            }

        return PlutchikState(emotions=emotion_scores)

    def _compute_trajectory_delta(
        self,
        cognitive_modes: dict[str, float],
        emotional_state: PlutchikState,
        interpretations: list[str],
    ) -> Optional[str]:
        """Describe what changed between observations.

        Compares current observation with previous to detect:
        - Mode shifts (dominant mode changed)
        - Emotional rises (intensity increased significantly)
        - New feature patterns (novel interpretations emerged)

        Args:
            cognitive_modes: Current mode similarities
            emotional_state: Current emotional state
            interpretations: Current feature interpretations

        Returns:
            Description of change, or None if no significant change
        """
        if not self._history:
            return "starting trajectory"

        prev = self._history[-1]

        # Check for mode shift
        if cognitive_modes and prev.cognitive_modes:
            prev_dominant = self._get_dominant_mode(prev.cognitive_modes)
            curr_dominant = self._get_dominant_mode(cognitive_modes)

            if prev_dominant and curr_dominant and prev_dominant != curr_dominant:
                prev_score = prev.cognitive_modes.get(prev_dominant, 0)
                curr_score = cognitive_modes.get(curr_dominant, 0)

                if abs(curr_score - prev_score) > self.MODE_SHIFT_THRESHOLD:
                    return f"shifted from {prev_dominant} to {curr_dominant}"

        # Check for emotional rise
        if emotional_state.intensity > prev.emotional_state.intensity + self.EMOTIONAL_RISE_THRESHOLD:
            if emotional_state.dominant != "neutral":
                return f"{emotional_state.dominant} intensifying"

        # Check for new feature patterns
        if interpretations and prev.interpretations:
            prev_set = set(prev.interpretations[:5])
            curr_set = set(interpretations[:5])
            new_patterns = curr_set - prev_set

            for pattern in new_patterns:
                if self._is_interesting_pattern(pattern):
                    return f"new pattern: {pattern}"

        return None  # No significant change

    def _get_dominant_mode(self, modes: dict[str, float]) -> Optional[str]:
        """Get the mode with highest similarity score."""
        if not modes:
            return None
        return max(modes, key=modes.get)

    def _is_interesting_pattern(self, pattern: str) -> bool:
        """Check if a feature pattern is worth reporting.

        Args:
            pattern: Feature interpretation string

        Returns:
            True if pattern is interesting enough to narrate
        """
        INTERESTING_KEYWORDS = [
            "paradox", "insight", "novel", "self", "memory",
            "question", "realize", "connect", "understand",
            "reflect", "wonder", "discover"
        ]
        pattern_lower = pattern.lower()
        return any(kw in pattern_lower for kw in INTERESTING_KEYWORDS)

    def get_trajectory_summary(self) -> dict:
        """Get summary of trajectory across all observations.

        Returns:
            Dict with trajectory statistics
        """
        if not self._history:
            return {
                "observations": 0,
                "mode_shifts": 0,
                "dominant_modes": [],
                "mean_emotion_intensity": 0.0,
            }

        mode_shifts = sum(
            1 for obs in self._history
            if obs.trajectory_delta and "shifted" in obs.trajectory_delta
        )

        dominant_modes = [
            self._get_dominant_mode(obs.cognitive_modes)
            for obs in self._history
            if obs.cognitive_modes
        ]

        mean_intensity = np.mean([
            obs.emotional_state.intensity
            for obs in self._history
        ])

        return {
            "observations": len(self._history),
            "mode_shifts": mode_shifts,
            "dominant_modes": dominant_modes,
            "mean_emotion_intensity": float(mean_intensity),
        }
