"""Discord sentiment analyzer for converting replies to recognition signals."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.recognition.schema import RecognitionSignal, SignalType

if TYPE_CHECKING:
    from integrations.discord.client import DiscordMessage

logger = logging.getLogger(__name__)


class DiscordSentimentAnalyzer:
    """Analyzes Ryan's Discord replies for recognition signals.

    Uses keyword-based sentiment detection to convert Discord messages
    into RecognitionSignal objects for the preference learning system.
    """

    # Indicators for different signal types
    POSITIVE_INDICATORS = [
        "yes",
        "good",
        "love",
        "great",
        "interesting",
        "exactly",
        "nice",
        "cool",
        "agree",
        "right",
        "perfect",
        "!",
    ]

    NEGATIVE_INDICATORS = [
        "no",
        "don't",
        "wrong",
        "not really",
        "confused",
        "disagree",
        "off",
        "miss",
        "lost",
        "?",
    ]

    CURIOUS_INDICATORS = [
        "hmm",
        "tell me more",
        "what about",
        "curious",
        "...",
        "interesting",
        "how",
        "why",
        "elaborate",
        "expand",
    ]

    def analyze(
        self,
        message: DiscordMessage,
        context: str = "",
    ) -> RecognitionSignal | None:
        """Convert a Discord reply to a recognition signal if sentiment is clear.

        Args:
            message: The Discord message from Ryan.
            context: Optional context about what Lilly said that prompted this reply.

        Returns:
            A RecognitionSignal if sentiment is detected, None otherwise.
        """
        content = message.content.lower()

        # Count indicator matches
        pos_score = sum(1 for w in self.POSITIVE_INDICATORS if w in content)
        neg_score = sum(1 for w in self.NEGATIVE_INDICATORS if w in content)
        cur_score = sum(1 for w in self.CURIOUS_INDICATORS if w in content)

        # Need at least one indicator to generate a signal
        max_score = max(pos_score, neg_score, cur_score)
        if max_score < 1:
            logger.debug(
                f"[DISCORD SENTIMENT] No clear sentiment in: {message.content[:50]}..."
            )
            return None

        # Determine signal type based on dominant sentiment
        if pos_score == max_score:
            signal_type = SignalType.APPROVE
        elif neg_score == max_score:
            signal_type = SignalType.DISAPPROVE
        else:
            signal_type = SignalType.CURIOUS

        # Calculate confidence based on indicator strength
        # More indicators = higher confidence (capped at 1.0)
        confidence = min(max_score / 3, 1.0)

        signal = RecognitionSignal(
            uid=f"discord_{message.id}",
            signal_type=signal_type,
            thought_uid=context or "unknown",
            thought_text="",
            context=f"discord_reply: {message.content[:100]}",
            timestamp=message.timestamp,
            confidence=confidence,
            note=message.content,
        )

        logger.info(
            f"[DISCORD SENTIMENT] {signal_type.value} "
            f"(confidence={confidence:.2f}): {message.content[:50]}..."
        )

        return signal
