"""Affect coloring for speech synthesis.

This module translates affective states into speech delivery parameters,
allowing Lilly's emotional state to influence how she speaks - pace, warmth,
certainty, and other prosodic features.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.self_model.affective_system import AffectiveState


class SpeechEmotion(Enum):
    """High-level emotional categories for speech."""

    NEUTRAL = "neutral"
    WARM = "warm"
    EXCITED = "excited"
    CONTEMPLATIVE = "contemplative"
    CURIOUS = "curious"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    TENTATIVE = "tentative"


@dataclass
class AffectColoring:
    """Speech delivery parameters derived from affect.

    These parameters influence how text is spoken - pace, warmth,
    certainty, and emphasis patterns.

    Attributes:
        pace: Speech rate (0.8=slow, 1.0=normal, 1.2=fast)
        warmth: Emotional warmth (0.0=neutral, 1.0=very warm)
        certainty: Confidence in delivery (0.0=tentative, 1.0=confident)
        emphasis: Overall expressiveness (0.0=flat, 1.0=animated)
        pause_tendency: Likelihood of thoughtful pauses (0.0=flowing, 1.0=frequent)
        emotion: Primary emotional category
    """

    pace: float = 1.0
    warmth: float = 0.5
    certainty: float = 0.5
    emphasis: float = 0.5
    pause_tendency: float = 0.3
    emotion: SpeechEmotion = SpeechEmotion.NEUTRAL

    def __post_init__(self):
        self.pace = max(0.6, min(1.4, self.pace))
        self.warmth = max(0.0, min(1.0, self.warmth))
        self.certainty = max(0.0, min(1.0, self.certainty))
        self.emphasis = max(0.0, min(1.0, self.emphasis))
        self.pause_tendency = max(0.0, min(1.0, self.pause_tendency))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "pace": self.pace,
            "warmth": self.warmth,
            "certainty": self.certainty,
            "emphasis": self.emphasis,
            "pause_tendency": self.pause_tendency,
            "emotion": self.emotion.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AffectColoring":
        """Deserialize from storage."""
        emotion = SpeechEmotion.NEUTRAL
        if data.get("emotion"):
            try:
                emotion = SpeechEmotion(data["emotion"])
            except ValueError:
                pass

        return cls(
            pace=data.get("pace", 1.0),
            warmth=data.get("warmth", 0.5),
            certainty=data.get("certainty", 0.5),
            emphasis=data.get("emphasis", 0.5),
            pause_tendency=data.get("pause_tendency", 0.3),
            emotion=emotion,
        )


class AffectToSpeech:
    """Translates affective state to speech delivery parameters.

    Maps the affect dimensions to speech parameters:
    - arousal -> pace, emphasis
    - valence -> warmth
    - wonder -> pace (slower), pause tendency
    """

    def color_from_affect(self, affect: "AffectiveState") -> AffectColoring:
        """Generate speech coloring from affective state."""
        # Base pace: higher arousal = faster
        pace = 1.0 + (affect.arousal - 0.5) * 0.3

        # Wonder slows pace
        wonder = getattr(affect, "wonder", 0.0)
        if wonder > 0.5:
            pace -= (wonder - 0.5) * 0.2

        # Warmth from valence
        warmth = affect.valence * 0.6 + getattr(affect, "satisfaction", 0.5) * 0.3

        # Certainty from satisfaction, reduced by frustration
        frustration = getattr(affect, "frustration", 0.0)
        certainty = 0.5 + getattr(affect, "satisfaction", 0.5) * 0.4 - frustration * 0.3

        # Emphasis from arousal and wonder
        curiosity = getattr(affect, "curiosity", 0.0)
        emphasis = affect.arousal * 0.4 + wonder * 0.3 + curiosity * 0.2

        # Pause tendency from wonder and low arousal
        pause_tendency = wonder * 0.5 + (1 - affect.arousal) * 0.3

        # Determine primary emotion
        emotion = self._classify_emotion(affect)

        return AffectColoring(
            pace=pace,
            warmth=warmth,
            certainty=certainty,
            emphasis=emphasis,
            pause_tendency=pause_tendency,
            emotion=emotion,
        )

    def _classify_emotion(self, affect: "AffectiveState") -> SpeechEmotion:
        """Classify the primary emotion for speech."""
        wonder = getattr(affect, "wonder", 0.0)
        curiosity = getattr(affect, "curiosity", 0.0)
        frustration = getattr(affect, "frustration", 0.0)
        satisfaction = getattr(affect, "satisfaction", 0.5)

        if wonder > 0.6:
            return SpeechEmotion.CONTEMPLATIVE

        if curiosity > 0.7:
            return SpeechEmotion.CURIOUS

        if frustration > 0.5:
            return SpeechEmotion.CONCERNED

        if satisfaction > 0.7 and affect.valence > 0.7:
            return SpeechEmotion.WARM

        if affect.arousal > 0.7 and affect.valence > 0.6:
            return SpeechEmotion.EXCITED

        if satisfaction > 0.6:
            return SpeechEmotion.CONFIDENT

        if frustration > 0.3 or curiosity > 0.5:
            return SpeechEmotion.TENTATIVE

        return SpeechEmotion.NEUTRAL

    @classmethod
    def default_coloring(cls) -> AffectColoring:
        """Get default/neutral coloring."""
        return AffectColoring()


class SSMLGenerator:
    """Generates SSML markup for speech synthesis.

    Translates AffectColoring into SSML hints for TTS engines.
    """

    PAUSE_SHORT: str = "200ms"
    PAUSE_MEDIUM: str = "500ms"
    PAUSE_LONG: str = "1s"

    def wrap_with_coloring(
        self,
        text: str,
        coloring: AffectColoring,
        add_pauses: bool = True,
    ) -> str:
        """Wrap text with SSML based on affect coloring."""
        ssml_parts = []

        rate_percent = int(coloring.pace * 100)
        ssml_parts.append(f'<prosody rate="{rate_percent}%">')

        if add_pauses and coloring.pause_tendency > 0.6:
            ssml_parts.append(f'<break time="{self.PAUSE_SHORT}"/>')

        if add_pauses and coloring.pause_tendency > 0.7:
            text = self._insert_pauses(text, coloring.pause_tendency)

        ssml_parts.append(text)
        ssml_parts.append("</prosody>")

        return "".join(ssml_parts)

    def _insert_pauses(self, text: str, pause_tendency: float) -> str:
        """Insert pause markers into text at natural breakpoints.

        Uses regex to properly split on sentence-ending punctuation (.?!)
        followed by whitespace, which is more robust than simple split.
        """
        # Split on sentence-ending punctuation followed by whitespace
        # The lookbehind ensures we keep the punctuation with the sentence
        sentences = re.split(r"(?<=[.?!])\s+", text)
        if len(sentences) <= 1:
            return text

        if pause_tendency > 0.8:
            pause = f'<break time="{self.PAUSE_MEDIUM}"/>'
        else:
            pause = f'<break time="{self.PAUSE_SHORT}"/>'

        return f" {pause}".join(sentences)

    def wrap_for_emotion(self, text: str, emotion: SpeechEmotion) -> str:
        """Wrap text with emotion-specific markup."""
        emotion_settings = {
            SpeechEmotion.WARM: 'rate="95%" pitch="-5%"',
            SpeechEmotion.EXCITED: 'rate="110%" pitch="+5%"',
            SpeechEmotion.CONTEMPLATIVE: 'rate="85%"',
            SpeechEmotion.CURIOUS: 'rate="100%" pitch="+3%"',
            SpeechEmotion.CONCERNED: 'rate="90%" pitch="-3%"',
            SpeechEmotion.CONFIDENT: 'rate="100%"',
            SpeechEmotion.TENTATIVE: 'rate="90%"',
            SpeechEmotion.NEUTRAL: 'rate="100%"',
        }

        settings = emotion_settings.get(emotion, 'rate="100%"')
        return f"<prosody {settings}>{text}</prosody>"


def color_text_for_affect(
    text: str,
    affect: "AffectiveState",
    add_ssml: bool = True,
    translator: Optional[AffectToSpeech] = None,
    generator: Optional[SSMLGenerator] = None,
) -> tuple[str, AffectColoring]:
    """Color text based on affective state.

    Convenience function that translates affect to coloring
    and optionally wraps text with SSML.
    """
    if translator is None:
        translator = AffectToSpeech()
    coloring = translator.color_from_affect(affect)

    if add_ssml:
        if generator is None:
            generator = SSMLGenerator()
        text = generator.wrap_with_coloring(text, coloring)

    return text, coloring
