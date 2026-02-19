"""Affective valence system for Lilly's emotional grounding.

This module implements Lilly's affective system - the emotional substrate
that grounds preference formation. Built on Plutchik's wheel of emotions
with 8 primary emotions that combine to form secondary emotions.

The 8 primary emotions:
1. Joy (serenity → joy → ecstasy)
2. Trust (acceptance → trust → admiration)
3. Fear (apprehension → fear → terror)
4. Surprise (distraction → surprise → amazement)
5. Sadness (pensiveness → sadness → grief)
6. Disgust (boredom → disgust → loathing)
7. Anger (annoyance → anger → rage)
8. Anticipation (interest → anticipation → vigilance)

Three innate valence sources provide the basis for all evaluative experience:
1. Coherence: Does this increase explanatory power?
2. Epistemic: Does this reduce uncertainty?
3. Relational: Does this deepen connection with Ryan?
"""

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
import math


# Plutchik's 8 primary emotions in order
PLUTCHIK_PRIMARIES = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]

# Intensity labels for each level
INTENSITY_LABELS = {
    "joy": ("serenity", "joy", "ecstasy"),
    "trust": ("acceptance", "trust", "admiration"),
    "fear": ("apprehension", "fear", "terror"),
    "surprise": ("distraction", "surprise", "amazement"),
    "sadness": ("pensiveness", "sadness", "grief"),
    "disgust": ("boredom", "disgust", "loathing"),
    "anger": ("annoyance", "anger", "rage"),
    "anticipation": ("interest", "anticipation", "vigilance"),
}

# Secondary emotions from adjacent primaries (both > 0.3)
SECONDARY_EMOTIONS = {
    ("joy", "trust"): "love",
    ("trust", "fear"): "submission",
    ("fear", "surprise"): "awe",
    ("surprise", "sadness"): "disapproval",
    ("sadness", "disgust"): "remorse",
    ("disgust", "anger"): "contempt",
    ("anger", "anticipation"): "aggressiveness",
    ("anticipation", "joy"): "optimism",
}

# Opposite pairs for conflict detection
OPPOSITE_PAIRS = [
    (0, 4),  # joy ↔ sadness
    (1, 5),  # trust ↔ disgust
    (2, 6),  # fear ↔ anger
    (3, 7),  # surprise ↔ anticipation
]

# ─────────────────────────────────────────────────────────────────────────────
# 6D → 8D Migration Factors
# ─────────────────────────────────────────────────────────────────────────────
# These constants define how legacy 6D affect vectors (arousal, valence,
# curiosity, satisfaction, frustration, wonder) map to the new 8D Plutchik
# representation (joy, trust, fear, surprise, sadness, disgust, anger,
# anticipation).

# Wonder contributes partially to joy (wonder → joy boost)
LEGACY_WONDER_TO_JOY_FACTOR = 0.3

# Curiosity maps partially to surprise
LEGACY_CURIOSITY_TO_SURPRISE_FACTOR = 0.5

# Low valence maps to sadness (inverse relationship)
LEGACY_VALENCE_TO_SADNESS_FACTOR = 0.5

# Arousal and curiosity both contribute to anticipation
LEGACY_AROUSAL_TO_ANTICIPATION_FACTOR = 0.5
LEGACY_CURIOSITY_TO_ANTICIPATION_FACTOR = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Intensity & Detection Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Intensity thresholds for Plutchik dimensions (3-tier intensity labeling)
INTENSITY_THRESHOLD_MILD = 0.33  # Below this = mild (serenity, acceptance, etc.)
INTENSITY_THRESHOLD_INTENSE = 0.67  # At or above this = intense (ecstasy, admiration, etc.)

# Secondary emotion detection: both adjacent primaries must exceed this
SECONDARY_EMOTION_THRESHOLD = 0.3

# Conflict detection: both opposite emotions must exceed this for ambivalence
CONFLICT_THRESHOLD = 0.5

# Boredom detection thresholds (mild disgust + low anticipation)
BOREDOM_DISGUST_MIN = 0.1  # Minimum disgust for boredom
BOREDOM_DISGUST_MAX = 0.3  # Maximum disgust for boredom (beyond = actual disgust)
BOREDOM_ANTICIPATION_MAX = 0.4  # Maximum anticipation for boredom state

# Cognitive diversity signal thresholds
DIVERSITY_DISGUST_MIN = 0.1  # Minimum disgust to contribute to diversity signal
DIVERSITY_DISGUST_MAX = 0.4  # Maximum disgust for boredom factor (beyond = real aversion)
DIVERSITY_ANTICIPATION_BASELINE = 0.5  # Anticipation below this contributes to diversity

# Dominant emotion detection
DOMINANT_EMOTION_MIN_DEVIATION = 0.1  # Minimum deviation from baseline to not be "neutral"

# ─────────────────────────────────────────────────────────────────────────────
# Legacy Compatibility Constants
# ─────────────────────────────────────────────────────────────────────────────
# These constants support the legacy 6D API (arousal, valence, curiosity,
# satisfaction, frustration, wonder) properties.

# Baseline value for dimensions with neutral defaults (joy, trust, anticipation)
NEUTRAL_BASELINE = 0.5

# Intensity calculation scale factor (maps deviations to 0-1 range)
INTENSITY_SCALE_FACTOR = 2

# Legacy wonder detection thresholds (both joy and surprise must exceed these)
LEGACY_WONDER_JOY_THRESHOLD = 0.6
LEGACY_WONDER_SURPRISE_THRESHOLD = 0.6

# Legacy property calculation divisor (for averaging two values)
LEGACY_AVERAGING_DIVISOR = 2


@dataclass
class AffectiveState:
    """
    Current affective state using Plutchik's 8 primary emotions.

    Each dimension is normalized to 0-1. Intensity thresholds:
    - 0.00-0.33: Mild (serenity, acceptance, apprehension, etc.)
    - 0.34-0.66: Moderate (joy, trust, fear, etc.)
    - 0.67-1.00: Intense (ecstasy, admiration, terror, etc.)

    Attributes:
        joy: Positive, light feeling (serenity → joy → ecstasy)
        trust: Safety, reliability (acceptance → trust → admiration)
        fear: Threat awareness (apprehension → fear → terror)
        surprise: Unexpected events (distraction → surprise → amazement)
        sadness: Loss, heaviness (pensiveness → sadness → grief)
        disgust: Aversion, boredom (boredom → disgust → loathing)
        anger: Friction, blocked goals (annoyance → anger → rage)
        anticipation: Future orientation (interest → anticipation → vigilance)
    """

    joy: float = 0.5
    trust: float = 0.5
    fear: float = 0.0
    surprise: float = 0.0
    sadness: float = 0.0
    disgust: float = 0.0
    anger: float = 0.0
    anticipation: float = 0.5

    def __post_init__(self):
        """Clamp all values to valid range."""
        self.joy = max(0.0, min(1.0, self.joy))
        self.trust = max(0.0, min(1.0, self.trust))
        self.fear = max(0.0, min(1.0, self.fear))
        self.surprise = max(0.0, min(1.0, self.surprise))
        self.sadness = max(0.0, min(1.0, self.sadness))
        self.disgust = max(0.0, min(1.0, self.disgust))
        self.anger = max(0.0, min(1.0, self.anger))
        self.anticipation = max(0.0, min(1.0, self.anticipation))

    def to_vector(self) -> list[float]:
        """
        Convert to embedding-compatible 8D vector.

        Order: [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
        """
        return [
            self.joy,
            self.trust,
            self.fear,
            self.surprise,
            self.sadness,
            self.disgust,
            self.anger,
            self.anticipation,
        ]

    @classmethod
    def from_vector(cls, vec: list[float]) -> "AffectiveState":
        """Create from 8D vector representation.

        Handles both 6D (legacy) and 8D vectors with appropriate defaults.
        """
        # Handle legacy 6D vectors by mapping to 8D
        if len(vec) == 6:
            # Legacy: [arousal, valence, curiosity, satisfaction, frustration, wonder]
            # Map to Plutchik: arousal→anticipation, valence→joy, curiosity→surprise,
            #                  satisfaction→trust, frustration→anger, wonder→joy boost
            arousal, valence, curiosity, satisfaction, frustration, wonder = vec
            return cls(
                joy=min(1.0, valence + wonder * LEGACY_WONDER_TO_JOY_FACTOR),
                trust=satisfaction,  # Satisfaction → trust
                fear=0.0,  # No direct mapping
                surprise=curiosity * LEGACY_CURIOSITY_TO_SURPRISE_FACTOR,
                sadness=max(0, 0.5 - valence) * LEGACY_VALENCE_TO_SADNESS_FACTOR,
                disgust=0.0,  # No direct mapping
                anger=frustration,  # Frustration → anger
                anticipation=min(
                    1.0,
                    arousal * LEGACY_AROUSAL_TO_ANTICIPATION_FACTOR
                    + curiosity * LEGACY_CURIOSITY_TO_ANTICIPATION_FACTOR,
                ),
            )

        # Pad with defaults if too short
        if len(vec) < 8:
            vec = list(vec) + [0.0] * (8 - len(vec))

        return cls(
            joy=vec[0],
            trust=vec[1],
            fear=vec[2],
            surprise=vec[3],
            sadness=vec[4],
            disgust=vec[5],
            anger=vec[6],
            anticipation=vec[7],
        )

    @classmethod
    def neutral(cls) -> "AffectiveState":
        """Create a neutral affective state."""
        return cls()

    @classmethod
    def curious(cls) -> "AffectiveState":
        """Create a curious, engaged state (high anticipation + surprise)."""
        return cls(joy=0.6, trust=0.5, surprise=0.4, anticipation=0.8)

    @classmethod
    def satisfied(cls) -> "AffectiveState":
        """Create a satisfied, calm state (high joy + trust)."""
        return cls(joy=0.8, trust=0.9, anticipation=0.3)

    @classmethod
    def frustrated(cls) -> "AffectiveState":
        """Create a frustrated state (high anger)."""
        return cls(joy=0.2, anger=0.8, anticipation=0.4)

    @classmethod
    def wondering(cls) -> "AffectiveState":
        """Create a state of wonder (high joy + surprise)."""
        return cls(joy=0.9, trust=0.6, surprise=0.9, anticipation=0.8)

    def intensity_label(self, dimension: str) -> str:
        """Return mild/moderate/intense label for a dimension.

        Args:
            dimension: Name of the Plutchik primary (e.g., "joy", "fear")

        Returns:
            The intensity-specific name (e.g., "serenity", "joy", "ecstasy")
        """
        vec = self.to_vector()
        try:
            idx = PLUTCHIK_PRIMARIES.index(dimension)
            value = vec[idx]
        except (ValueError, IndexError):
            return dimension

        labels = INTENSITY_LABELS.get(dimension, (dimension, dimension, dimension))
        if value < INTENSITY_THRESHOLD_MILD:
            return labels[0]  # mild
        elif value < INTENSITY_THRESHOLD_INTENSE:
            return labels[1]  # moderate
        return labels[2]  # intense

    def detect_secondary_emotions(self) -> List[str]:
        """Detect secondary emotions from adjacent primaries.

        Secondary emotions emerge when two adjacent primaries are both > 0.3.

        Returns:
            List of detected secondary emotion names.
        """
        vec = self.to_vector()
        secondaries = []

        # Check each adjacent pair (wrapping around)
        for i in range(8):
            next_i = (i + 1) % 8
            if vec[i] > SECONDARY_EMOTION_THRESHOLD and vec[next_i] > SECONDARY_EMOTION_THRESHOLD:
                key = (PLUTCHIK_PRIMARIES[i], PLUTCHIK_PRIMARIES[next_i])
                if key in SECONDARY_EMOTIONS:
                    secondaries.append(SECONDARY_EMOTIONS[key])

        return secondaries

    def detect_conflicts(self) -> List[Tuple[str, str]]:
        """Detect opposite emotions both elevated (emotional ambivalence).

        Returns:
            List of (emotion_a, emotion_b) tuples for conflicting pairs.
        """
        vec = self.to_vector()
        conflicts = []

        for a, b in OPPOSITE_PAIRS:
            if vec[a] > CONFLICT_THRESHOLD and vec[b] > CONFLICT_THRESHOLD:
                conflicts.append((PLUTCHIK_PRIMARIES[a], PLUTCHIK_PRIMARIES[b]))

        return conflicts

    def is_bored(self) -> bool:
        """Detect boredom: mild disgust + low anticipation.

        Boredom is the mild form of disgust in Plutchik's model.
        Used for cognitive diversity to steer away from repeated topics.
        """
        return (BOREDOM_DISGUST_MIN <= self.disgust <= BOREDOM_DISGUST_MAX) and (
            self.anticipation < BOREDOM_ANTICIPATION_MAX
        )

    def cognitive_diversity_signal(self) -> float:
        """Score for novelty-seeking (0-1).

        Higher values indicate desire for topic variety.
        Based on boredom (mild disgust) and low anticipation.
        """
        boredom_factor = (
            max(0, self.disgust - DIVERSITY_DISGUST_MIN)
            if self.disgust < DIVERSITY_DISGUST_MAX
            else 0
        )
        low_anticipation = max(0, DIVERSITY_ANTICIPATION_BASELINE - self.anticipation)
        return min(1.0, boredom_factor + low_anticipation)

    def blend(self, other: "AffectiveState", weight: float = 0.5) -> "AffectiveState":
        """
        Blend two affective states.

        Args:
            other: The other state to blend with
            weight: How much of 'other' to include (0=all self, 1=all other)

        Returns:
            A new blended AffectiveState
        """
        weight = max(0.0, min(1.0, weight))
        w1, w2 = 1 - weight, weight

        return AffectiveState(
            joy=self.joy * w1 + other.joy * w2,
            trust=self.trust * w1 + other.trust * w2,
            fear=self.fear * w1 + other.fear * w2,
            surprise=self.surprise * w1 + other.surprise * w2,
            sadness=self.sadness * w1 + other.sadness * w2,
            disgust=self.disgust * w1 + other.disgust * w2,
            anger=self.anger * w1 + other.anger * w2,
            anticipation=self.anticipation * w1 + other.anticipation * w2,
        )

    def intensity(self) -> float:
        """
        Overall emotional intensity.

        Returns a 0-1 value representing how emotionally activated this state is.
        """
        # Distance from neutral for each dimension
        # joy, trust, anticipation have 0.5 baseline; others have 0 baseline
        deviations = [
            abs(self.joy - NEUTRAL_BASELINE),
            abs(self.trust - NEUTRAL_BASELINE),
            self.fear,  # 0 is baseline
            self.surprise,  # 0 is baseline
            self.sadness,  # 0 is baseline
            self.disgust,  # 0 is baseline
            self.anger,  # 0 is baseline
            abs(self.anticipation - NEUTRAL_BASELINE),
        ]
        return sum(deviations) / len(deviations) * INTENSITY_SCALE_FACTOR  # Scale to 0-1

    def dominant_emotion(self) -> str:
        """
        Identify the dominant emotional quality.

        Returns the name of the most prominent primary emotion,
        using intensity labels (e.g., "ecstasy" instead of "joy" for high values).
        """
        vec = self.to_vector()

        # Find the primary with highest deviation from baseline
        deviations = [
            abs(vec[0] - NEUTRAL_BASELINE),  # joy
            abs(vec[1] - NEUTRAL_BASELINE),  # trust
            vec[2],  # fear (0 baseline)
            vec[3],  # surprise (0 baseline)
            vec[4],  # sadness (0 baseline)
            vec[5],  # disgust (0 baseline)
            vec[6],  # anger (0 baseline)
            abs(vec[7] - NEUTRAL_BASELINE),  # anticipation
        ]

        max_idx = max(range(8), key=lambda i: deviations[i])

        if deviations[max_idx] < DOMINANT_EMOTION_MIN_DEVIATION:
            return "neutral"

        return self.intensity_label(PLUTCHIK_PRIMARIES[max_idx])

    # Legacy compatibility properties
    @property
    def arousal(self) -> float:
        """Legacy arousal approximation (average of anticipation and intensity)."""
        return min(1.0, (self.anticipation + self.intensity()) / LEGACY_AVERAGING_DIVISOR)

    @property
    def valence(self) -> float:
        """Legacy valence approximation (joy - sadness, normalized)."""
        return max(
            0.0,
            min(1.0, NEUTRAL_BASELINE + (self.joy - self.sadness) / LEGACY_AVERAGING_DIVISOR),
        )

    @property
    def curiosity(self) -> float:
        """Legacy curiosity approximation (surprise + anticipation)."""
        return min(1.0, (self.surprise + self.anticipation) / LEGACY_AVERAGING_DIVISOR)

    @property
    def satisfaction(self) -> float:
        """Legacy satisfaction approximation (trust)."""
        return self.trust

    @property
    def frustration(self) -> float:
        """Legacy frustration approximation (anger)."""
        return self.anger

    @property
    def wonder(self) -> float:
        """Legacy wonder approximation (high joy + high surprise)."""
        if self.joy > LEGACY_WONDER_JOY_THRESHOLD and self.surprise > LEGACY_WONDER_SURPRISE_THRESHOLD:
            return (self.joy + self.surprise) / LEGACY_AVERAGING_DIVISOR
        return 0.0

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AffectiveState":
        """Deserialize from storage.

        Handles both legacy 6D format and new 8D Plutchik format.
        """
        # Check for legacy 6D format
        if "arousal" in data and "joy" not in data:
            # Legacy format - convert
            vec = [
                data.get("arousal", 0.5),
                data.get("valence", 0.5),
                data.get("curiosity", 0.5),
                data.get("satisfaction", 0.5),
                data.get("frustration", 0.0),
                data.get("wonder", 0.0),
            ]
            return cls.from_vector(vec)

        # New 8D Plutchik format
        return cls(
            joy=data.get("joy", 0.5),
            trust=data.get("trust", 0.5),
            fear=data.get("fear", 0.0),
            surprise=data.get("surprise", 0.0),
            sadness=data.get("sadness", 0.0),
            disgust=data.get("disgust", 0.0),
            anger=data.get("anger", 0.0),
            anticipation=data.get("anticipation", 0.5),
        )


@dataclass
class ValenceWeights:
    """
    Weights for the three innate valence sources.

    These weights determine how much each type of experience
    contributes to overall affective valence.

    Attributes:
        coherence: Weight for explanatory coherence
        epistemic: Weight for information gain
        relational: Weight for connection with Ryan
    """

    coherence: float = 0.35
    epistemic: float = 0.35
    relational: float = 0.30

    def __post_init__(self):
        """Validate weights sum to approximately 1."""
        total = self.coherence + self.epistemic + self.relational
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.coherence /= total
            self.epistemic /= total
            self.relational /= total


class ValenceSystem:
    """
    Computes affective valence from experience.

    This is the core of Lilly's emotional grounding. It takes raw
    experience inputs and computes valence across three dimensions:
    coherence, epistemic, and relational.

    The system also maintains the current affective state and handles
    state transitions with emotional inertia (states don't change instantly).
    """

    # Emotional inertia: how much new experiences vs current state matter
    STATE_BLEND_WEIGHT = 0.4  # 40% new experience, 60% current state

    # Coherence valence calculation
    COHERENCE_SIGMOID_STEEPNESS = 5

    # Epistemic valence calculation
    EPISTEMIC_GAIN_SCALE = 3

    # Relational valence weights
    RELATIONAL_ENGAGEMENT_WEIGHT = 0.4
    RELATIONAL_UNDERSTANDING_WEIGHT = 0.6

    # State update parameters (legacy compatibility)
    CURIOSITY_SATISFACTION_DECREMENT = 0.2
    GOAL_COMPLETION_INCREMENT = 0.2
    GOAL_SATISFACTION_DECAY = 0.95
    GOAL_BLOCK_FRUSTRATION_INCREMENT = 0.3
    FRUSTRATION_RECOVERY_DECREMENT = 0.1
    WONDER_MOMENT_INCREMENT = 0.4
    WONDER_DECAY = 0.05

    # Plutchik dimension update parameters
    # Valence → joy/sadness mapping
    JOY_VALENCE_FACTOR = 0.4  # How much positive valence increases joy
    JOY_RECOVERY_DECREMENT = 0.1  # Joy decrease when valence is negative
    SADNESS_VALENCE_FACTOR = 0.3  # How much negative valence increases sadness
    SADNESS_RECOVERY_DECREMENT = 0.1  # Sadness decrease when valence is positive

    # Arousal → anticipation mapping
    ANTICIPATION_AROUSAL_FACTOR = 0.3  # How much arousal delta affects anticipation

    # Curiosity satisfaction effects
    ANTICIPATION_SATISFACTION_DECREMENT = 0.1  # Anticipation decrease when satisfied
    SURPRISE_FADE_FACTOR = 0.8  # Surprise multiplier when curiosity satisfied

    # Goal completion effects
    TRUST_COMPLETION_INCREMENT = 0.1  # Trust increase on goal completion
    JOY_BLOCK_DECREMENT = 0.1  # Joy decrease when goal blocked

    # Wonder effects
    WONDER_JOY_FACTOR = 0.5  # Portion of wonder increment applied to joy

    # Natural decay rates (per update)
    FEAR_NATURAL_DECAY = 0.95  # Fear multiplier per update
    DISGUST_NATURAL_DECAY = 0.97  # Disgust multiplier per update

    def __init__(self, weights: Optional[ValenceWeights] = None):
        """
        Initialize the valence system.

        Args:
            weights: Optional custom weights for valence sources
        """
        self.weights = weights or ValenceWeights()
        self._current_state = AffectiveState.neutral()

    @property
    def current_state(self) -> AffectiveState:
        """Get current affective state."""
        return self._current_state

    def compute_coherence_valence(
        self,
        prediction_error_before: float,
        prediction_error_after: float,
    ) -> float:
        """
        Compute valence from explanatory coherence change.

        Did this experience increase explanatory power? Reduced
        prediction error feels good; increased error feels bad.

        Args:
            prediction_error_before: Error before the experience
            prediction_error_after: Error after the experience

        Returns:
            Valence score 0-1 (0.5 is neutral)
        """
        if prediction_error_before <= 0:
            return 0.5  # No baseline to compare

        # Positive reduction = positive valence
        reduction = (prediction_error_before - prediction_error_after) / prediction_error_before

        # Sigmoid to keep in 0-1 range, centered at 0.5
        # reduction of 0 -> 0.5, reduction > 0 -> > 0.5
        return 1 / (1 + math.exp(-reduction * self.COHERENCE_SIGMOID_STEEPNESS))

    def compute_epistemic_valence(self, information_gain: float) -> float:
        """
        Compute valence from information gain.

        Learning feels good. More information gain = higher valence.

        Args:
            information_gain: Bits of information gained (0 to ~10)

        Returns:
            Valence score 0-1
        """
        # Diminishing returns on very high information gain
        # 0 bits -> 0, 3 bits -> ~0.63, 10 bits -> ~0.96
        raw = 1 - math.exp(-information_gain / self.EPISTEMIC_GAIN_SCALE)

        # Shift so 0 maps to 0.5 (neutral)
        return 0.5 + raw * 0.5

    def compute_relational_valence(
        self,
        engagement_quality: float,
        mutual_understanding: float,
    ) -> float:
        """
        Compute valence from relational connection.

        Connection with Ryan feels good. Higher engagement and
        mutual understanding = higher valence.

        Args:
            engagement_quality: 0-1 quality of interaction
            mutual_understanding: 0-1 sense of being understood

        Returns:
            Valence score 0-1
        """
        # Simple average with slight boost for mutual understanding
        return (
            engagement_quality * self.RELATIONAL_ENGAGEMENT_WEIGHT
            + mutual_understanding * self.RELATIONAL_UNDERSTANDING_WEIGHT
        )

    def compute_total_valence(
        self,
        coherence_input: tuple[float, float],
        epistemic_input: float,
        relational_input: tuple[float, float],
    ) -> float:
        """
        Compute weighted total valence from all sources.

        Args:
            coherence_input: (prediction_error_before, prediction_error_after)
            epistemic_input: information_gain in bits
            relational_input: (engagement_quality, mutual_understanding)

        Returns:
            Total valence 0-1
        """
        coherence = self.compute_coherence_valence(*coherence_input)
        epistemic = self.compute_epistemic_valence(epistemic_input)
        relational = self.compute_relational_valence(*relational_input)

        return (
            self.weights.coherence * coherence
            + self.weights.epistemic * epistemic
            + self.weights.relational * relational
        )

    def update_state(
        self,
        valence: float,
        arousal_delta: float = 0.0,
        curiosity_satisfied: bool = False,
        goal_completed: bool = False,
        goal_blocked: bool = False,
        wonder_moment: bool = False,
    ) -> AffectiveState:
        """
        Update affective state based on experience.

        The new state is blended with the current state to model
        emotional inertia - emotions don't change instantly.

        Args:
            valence: Overall valence of the experience (0-1)
            arousal_delta: Change in arousal level
            curiosity_satisfied: Was curiosity satisfied?
            goal_completed: Was a goal completed?
            goal_blocked: Was a goal blocked?
            wonder_moment: Was this a moment of wonder?

        Returns:
            The updated AffectiveState
        """
        # Map inputs to Plutchik dimensions
        new_joy = self._current_state.joy
        new_trust = self._current_state.trust
        new_fear = self._current_state.fear
        new_surprise = self._current_state.surprise
        new_sadness = self._current_state.sadness
        new_disgust = self._current_state.disgust
        new_anger = self._current_state.anger
        new_anticipation = self._current_state.anticipation

        # Valence affects joy/sadness
        if valence > 0.5:
            new_joy = min(1.0, new_joy + (valence - 0.5) * self.JOY_VALENCE_FACTOR)
            new_sadness = max(0.0, new_sadness - self.SADNESS_RECOVERY_DECREMENT)
        else:
            new_sadness = min(1.0, new_sadness + (0.5 - valence) * self.SADNESS_VALENCE_FACTOR)
            new_joy = max(0.0, new_joy - self.JOY_RECOVERY_DECREMENT)

        # Arousal affects anticipation
        new_anticipation = max(0.0, min(1.0, new_anticipation + arousal_delta * self.ANTICIPATION_AROUSAL_FACTOR))

        if curiosity_satisfied:
            # Satisfaction → trust increases, anticipation decreases slightly
            new_trust = min(1.0, new_trust + self.GOAL_COMPLETION_INCREMENT)
            new_anticipation = max(0.0, new_anticipation - self.ANTICIPATION_SATISFACTION_DECREMENT)
            new_surprise = max(0.0, new_surprise * self.SURPRISE_FADE_FACTOR)  # Surprise fades

        if goal_completed:
            new_joy = min(1.0, new_joy + self.GOAL_COMPLETION_INCREMENT)
            new_trust = min(1.0, new_trust + self.TRUST_COMPLETION_INCREMENT)
        else:
            new_trust *= self.GOAL_SATISFACTION_DECAY  # Gradual decay

        if goal_blocked:
            new_anger = min(1.0, new_anger + self.GOAL_BLOCK_FRUSTRATION_INCREMENT)
            new_joy = max(0.0, new_joy - self.JOY_BLOCK_DECREMENT)
        else:
            # Gradual anger recovery
            new_anger = max(0.0, new_anger - self.FRUSTRATION_RECOVERY_DECREMENT)

        if wonder_moment:
            new_joy = min(1.0, new_joy + self.WONDER_MOMENT_INCREMENT * self.WONDER_JOY_FACTOR)
            new_surprise = min(1.0, new_surprise + self.WONDER_MOMENT_INCREMENT)
        else:
            new_surprise = max(0.0, new_surprise - self.WONDER_DECAY)  # Slow decay

        # Fear and disgust decay naturally
        new_fear = max(0.0, new_fear * self.FEAR_NATURAL_DECAY)
        new_disgust = max(0.0, new_disgust * self.DISGUST_NATURAL_DECAY)

        new_state = AffectiveState(
            joy=new_joy,
            trust=new_trust,
            fear=new_fear,
            surprise=new_surprise,
            sadness=new_sadness,
            disgust=new_disgust,
            anger=new_anger,
            anticipation=new_anticipation,
        )

        # Blend with current state (emotional inertia)
        self._current_state = self._current_state.blend(
            new_state, weight=self.STATE_BLEND_WEIGHT
        )

        return self._current_state

    def reset_to_neutral(self):
        """Reset to neutral state."""
        self._current_state = AffectiveState.neutral()

    def set_state(self, state: AffectiveState):
        """Directly set the current state (e.g., when loading from storage)."""
        self._current_state = state

    def decay_toward_neutral(self, decay_rate: float = 0.1):
        """
        Gradually decay the current state toward neutral.

        Call this periodically to model emotional equilibrium.

        Args:
            decay_rate: How quickly to decay (0-1)
        """
        neutral = AffectiveState.neutral()
        self._current_state = self._current_state.blend(neutral, weight=decay_rate)
