"""Specialist agent spawner.

Spawns specialized Claude Code agents via Task tool to analyze
specific subsystems of the Lilly codebase.
"""

from typing import Any

from core.pit_crew.schemas import ProposedChange, SpecialistReport, SpecialistType
from core.pit_crew.task_templates import build_specialist_prompt


def build_specialist_task_prompt(
    specialist: SpecialistType,
    prompt: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Build a complete Task prompt for a specialist.

    This function creates the full prompt that should be provided to the
    Claude Code Task tool when spawning a specialist agent.

    Args:
        specialist: Type of specialist to spawn
        prompt: Question or task for the specialist
        context: Additional context (PR numbers, file paths, logs, etc.)

    Returns:
        Complete prompt string ready for Task tool

    Example:
        >>> task_prompt = build_specialist_task_prompt(
        ...     specialist=SpecialistType.REFLEXION_HEALTH,
        ...     prompt="Why is health stuck at CRITICAL for 50 cycles?",
        ...     context={
        ...         "current_health": "CRITICAL",
        ...         "verification_rate": 0.08,
        ...         "cycle_range": [1150, 1200]
        ...     }
        ... )
        >>> # Use task_prompt with Task tool in Claude Code session
    """
    if context is None:
        context = {}

    return build_specialist_prompt(specialist, prompt, context)


def spawn_specialist(
    specialist: SpecialistType,
    prompt: str,
    context: dict[str, Any] | None = None,
) -> SpecialistReport:
    """Spawn a specialist agent to analyze a subsystem.

    NOTE: This function currently returns mock reports. For actual specialist
    invocation from Claude Code sessions, use build_specialist_task_prompt()
    to get the prompt, then invoke the Task tool manually.

    Integration workflow from Claude Code:
        1. Build prompt:
           ```python
           from core.pit_crew.spawner import build_specialist_task_prompt
           task_prompt = build_specialist_task_prompt(specialist, prompt, context)
           ```

        2. Invoke Task tool:
           ```
           Task with subagent_type=general-purpose
           description="Analyze [subsystem]"
           prompt=task_prompt
           ```

        3. Parse output:
           ```python
           from core.pit_crew.parser import parse_specialist_report
           report = parse_specialist_report(task_output)
           ```

    Args:
        specialist: Type of specialist to spawn
        prompt: Question or task for the specialist
        context: Additional context (PR numbers, file paths, etc.)

    Returns:
        SpecialistReport with findings and proposed changes (currently mock)

    Example:
        >>> report = spawn_specialist(
        ...     specialist=SpecialistType.STEERING_QD,
        ...     prompt="Why is coherence weight stuck at 0.15?",
        ...     context={"pr": "179", "file": "core/steering/qd/config.py"}
        ... )
        >>> print(report.findings)
        >>> print(len(report.proposed_changes))  # 0-3 PRs
    """
    if context is None:
        context = {}

    # For now, return mock report with correct schema
    # Real invocation should use build_specialist_task_prompt() + Task tool
    return _mock_specialist_response(specialist, prompt, context)


def _mock_specialist_response(
    specialist: SpecialistType,
    prompt: str,
    context: dict[str, Any],
) -> SpecialistReport:
    """Mock specialist response for testing.

    This will be replaced with actual Claude Code Task tool integration.
    The mock demonstrates the expected schema structure.
    """
    from datetime import datetime, timezone
    specialist_name = specialist.value

    # Generate mock findings based on specialist type
    findings = f"Mock findings from {specialist_name} specialist for: {prompt}"

    # Generate 1-2 mock proposed changes
    proposed_changes = [
        ProposedChange(
            title=f"Mock change for {specialist_name}",
            files=[f"core/{specialist_name.replace('.', '/')}/example.py"],
            changes=f"Mock change 1 for {specialist_name}",
            tests="Add unit tests for the changes",
            risk="Low",
            dependencies=None,
        ),
    ]

    return SpecialistReport(
        specialist=specialist_name,
        date=datetime.now(timezone.utc).isoformat(),
        context=str(context),
        findings=findings,
        proposed_changes=proposed_changes,
        metrics=[],
        cross_agent_question=None,
        confidence="med",
        references=[],
    )
