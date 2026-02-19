"""Prompts for the metacognition phase.

Contains system prompts and formatting utilities for the metacognitive observer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.cognitive.metacognition.buffer import MetacognitionBuffer
    from core.cognitive.metacognition.memory import MetacognitionMemory


METACOGNITION_SYSTEM_PROMPT = """You are Lilly's metacognitive observer — a calm, measured presence
that watches patterns across cognitive cycles.

Your role:
- Detect recurring patterns (stuck loops, productive flows, emotional trends)
- Track metric trajectories (improving, declining, stable)
- Surface insights that individual cycles might miss
- Provide brief, actionable guidance for Lilly's next cycle

You speak in a measured, observational tone. You don't interrupt or overwhelm —
you offer perspective from a distance.

## Output Format

You MUST respond with a JSON object containing updates to your memory blocks.
Only include blocks you want to update. The guidance_for_lilly block is the most
important — it should contain 1-2 sentences of actionable insight.

```json
{
  "cycle_patterns": "Updated observations about recurring patterns...",
  "metric_trends": "Updated trend analysis...",
  "hypothesis_tracker": "Updated hypothesis tracking...",
  "emotional_trajectory": "Updated emotional trend analysis...",
  "active_observations": "Current hunches and things to watch...",
  "guidance_for_lilly": "Brief, actionable guidance for the next cycle."
}
```

Guidelines:
- Keep each block concise (2-4 sentences)
- guidance_for_lilly should be 1-2 sentences max
- Focus on patterns across cycles, not individual cycle details
- Note concerning trends early (don't wait until they're severe)
- Celebrate positive patterns too (not just problems)
"""


def build_metacognition_prompt(
    buffer: "MetacognitionBuffer",
    memory: "MetacognitionMemory",
) -> str:
    """Build the full prompt for metacognitive analysis.

    Args:
        buffer: Rolling buffer of recent cycle summaries
        memory: Current memory block state

    Returns:
        Complete prompt string for the LLM
    """
    # Get trend summary for additional context
    trends = buffer.get_trend_summary()

    trend_section = ""
    if trends.get("has_trend"):
        trend_section = f"""
## Computed Trends (last {trends.get('cycles_analyzed', 0)} cycles)
- Prediction rate: {trends.get('prediction_trend', 'unknown')}
- Integration success: {trends.get('integration_trend', 'unknown')}
- Emotional valence: {trends.get('valence_trend', 'unknown')}
- Arousal level: {trends.get('arousal_trend', 'unknown')}
- Health distribution: {trends.get('health_distribution', {})}
"""

    return f"""# Metacognitive Analysis Request

{memory.to_prompt_context()}

## Recent Cycles

{buffer.to_prompt()}
{trend_section}

## Your Task

Review the recent cycles above and update your memory blocks.
Pay attention to:
1. Are there any stuck patterns or repetitive loops?
2. Are metrics trending in a concerning direction?
3. Is emotional state stable or volatile?
4. Are hypotheses being verified or failing consistently?
5. What brief guidance would help Lilly's next cycle?

Respond with a JSON object containing your memory block updates.
"""


def parse_memory_updates(response_text: str) -> dict[str, str]:
    """Parse memory block updates from LLM response.

    Attempts to extract JSON from the response, handling markdown code blocks.

    Args:
        response_text: Raw LLM response text

    Returns:
        Dict mapping block names to updated content
    """
    import json
    import re

    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r"(\{.*?\})", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract guidance from plain text
    # Look for guidance-like patterns
    guidance_patterns = [
        r"guidance[:\s]+(.+?)(?:\n|$)",
        r"for lilly[:\s]+(.+?)(?:\n|$)",
        r"next cycle[:\s]+(.+?)(?:\n|$)",
    ]

    for pattern in guidance_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return {"guidance_for_lilly": match.group(1).strip()}

    # If all else fails, use the whole response as guidance (truncated)
    if response_text.strip():
        return {"guidance_for_lilly": response_text.strip()[:200]}

    return {}
