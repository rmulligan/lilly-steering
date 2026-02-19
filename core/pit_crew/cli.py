"""Click-based CLI for pit crew specialist system."""

import concurrent.futures
import logging
from pathlib import Path
from typing import Any

import click

from core.pit_crew.journal import append_journal_entry
from core.pit_crew.parser import parse_specialist_report
from core.pit_crew.schemas import SpecialistType
from core.pit_crew.spawner import spawn_specialist

logger = logging.getLogger(__name__)


def parse_context_args(context_args: list[str]) -> dict[str, Any]:
    """Parse key=value context arguments.

    Args:
        context_args: List of "key=value" strings

    Returns:
        Dictionary of parsed context

    Raises:
        ValueError: If any arg is not in key=value format

    Example:
        >>> parse_context_args(["pr=179", "file=config.py"])
        {"pr": "179", "file": "config.py"}
    """
    context: dict[str, Any] = {}
    for arg in context_args:
        if "=" not in arg:
            raise ValueError(f"Invalid context format: {arg}. Expected key=value")
        key, value = arg.split("=", 1)
        context[key.strip()] = value.strip()
    return context


@click.group()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def cli(verbose: bool) -> None:
    """Pit crew CLI for specialist agent coordination.

    Command-line interface for spawning specialist agents, parsing reports,
    and managing the development hub.
    """
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug("CLI initialized")


@cli.command()
@click.option(
    "--specialist",
    "-s",
    type=click.Choice(
        [s.value for s in SpecialistType],
        case_sensitive=False,
    ),
    required=True,
    help="Type of specialist to spawn",
)
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Question or task for the specialist",
)
@click.option(
    "--context",
    "-c",
    multiple=True,
    help="Context in key=value format (can be used multiple times)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for JSON report (default: stdout)",
)
def spawn(
    specialist: str,
    prompt: str,
    context: tuple[str, ...],
    output: Path | None,
) -> None:
    """Spawn a specialist agent to analyze a subsystem.

    Examples:

        # Basic spawn
        pit-crew spawn --specialist steering.qd --prompt "Why is coherence stuck?"

        # With context
        pit-crew spawn -s steering.qd -p "Analyze PR" -c pr=179 -c file=config.py

        # Save to file
        pit-crew spawn -s steering.qd -p "Check metrics" -o report.json
    """
    logger.info(f"Spawning specialist: {specialist}")

    # Parse context arguments
    try:
        context_dict = parse_context_args(list(context))
    except ValueError as e:
        raise click.BadParameter(str(e))

    logger.debug(f"Context: {context_dict}")

    # Map specialist string to enum
    specialist_type = SpecialistType(specialist)

    # Spawn specialist
    try:
        report = spawn_specialist(
            specialist=specialist_type,
            prompt=prompt,
            context=context_dict,
        )
    except RuntimeError as e:
        logger.error(f"Failed to spawn specialist: {e}")
        raise click.ClickException(f"Spawn failed: {e}")
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise click.ClickException(f"Spawn failed: {e}")
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise click.ClickException(f"Spawn failed: {e}")

    # Convert to JSON
    report_json = report.model_dump_json(indent=2)

    # Output to file or stdout
    if output is not None:
        logger.info(f"Writing report to {output}")
        # Create parent directory if it doesn't exist
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_json)
        click.echo(f"Report written to {output}")
    else:
        click.echo(report_json)

    logger.info("Specialist completed successfully")


@cli.command(name="update-journal")
@click.option(
    "--report",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to specialist report markdown file",
)
@click.option(
    "--journal",
    "-j",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to journal markdown file",
)
@click.option(
    "--ticket-id",
    "-t",
    required=True,
    help="Letta hub ticket identifier (e.g., ticket-123)",
)
@click.option(
    "--pr",
    help="Optional GitHub PR number",
)
def update_journal(
    report: Path,
    journal: Path,
    ticket_id: str,
    pr: str | None,
) -> None:
    """Parse report markdown and update journal file.

    Reads a specialist report in markdown format, parses it to structured
    data, and appends an entry to the specialist's journal file.

    Examples:

        # Basic journal update
        pit-crew update-journal -r report.md -j journal.md -t ticket-123

        # With PR number
        pit-crew update-journal -r report.md -j journal.md -t ticket-123 --pr 179
    """
    logger.info(f"Updating journal: {journal}")
    logger.debug(f"Report: {report}, Ticket: {ticket_id}, PR: {pr}")

    # Parse report markdown
    try:
        report_text = report.read_text()
        parsed_report = parse_specialist_report(report_text)
    except FileNotFoundError as e:
        logger.error(f"Report file not found: {e}")
        raise click.ClickException(f"Parse failed: {e}")
    except ValueError as e:
        logger.error(f"Invalid report format: {e}")
        raise click.ClickException(f"Parse failed: {e}")
    except KeyError as e:
        logger.error(f"Missing required field in report: {e}")
        raise click.ClickException(f"Parse failed: {e}")

    logger.debug(f"Parsed report for specialist: {parsed_report.specialist}")

    # Append to journal
    try:
        append_journal_entry(
            journal_path=journal,
            report=parsed_report,
            hub_ticket_id=ticket_id,
            pr_number=pr,
        )
    except FileNotFoundError as e:
        logger.error(f"Journal file not found: {e}")
        raise click.ClickException(f"Journal update failed: {e}")
    except IOError as e:
        logger.error(f"Failed to write to journal: {e}")
        raise click.ClickException(f"Journal update failed: {e}")
    except ValueError as e:
        logger.error(f"Invalid journal format: {e}")
        raise click.ClickException(f"Journal update failed: {e}")

    click.echo(f"Updated journal: {journal}")
    click.echo(f"Entry added for ticket: {ticket_id}")
    logger.info("Journal updated successfully")


@cli.command()
@click.option(
    "--health",
    type=click.Choice(["CRITICAL", "STRESSED", "STABLE", "THRIVING"], case_sensitive=False),
    help="Current Lilly health status (if known)",
)
@click.option(
    "--context",
    multiple=True,
    help="Additional context (format: key=value). Can be repeated.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Save evaluation report to file (JSON format)",
)
@click.option(
    "--format",
    type=click.Choice(["json", "markdown"], case_sensitive=False),
    default="markdown",
    help="Output format (default: markdown)",
)
def evaluate(
    health: str | None,
    context: tuple[str, ...],
    output: Path | None,
    format: str,
) -> None:
    """Evaluate Lilly's health and generate recommendations.

    Spawns all specialist agents in parallel to analyze their domains,
    then synthesizes findings into actionable recommendations.

    Example:

        bin/pit-crew evaluate --health CRITICAL --context verification=0.11
    """
    import concurrent.futures
    import json
    from datetime import datetime, timezone

    logger.info("Starting pit crew evaluation")

    # Parse context
    context_dict = parse_context_args(list(context))
    if health:
        context_dict["health"] = health

    # Build evaluation prompt
    eval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_prompt = f"Evaluate Lilly's health and performance as of {eval_date}"

    if health:
        base_prompt += f" (Health: {health})"

    # Spawn all specialists in parallel
    if format == "markdown":
        click.echo("Spawning specialists in parallel...")
    specialists = list(SpecialistType)

    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(specialists)) as executor:
        future_to_specialist = {
            executor.submit(
                spawn_specialist,
                specialist=s,
                prompt=base_prompt,
                context=context_dict,
            ): s
            for s in specialists
        }
        for future in concurrent.futures.as_completed(future_to_specialist):
            specialist = future_to_specialist[future]
            try:
                report = future.result()
                reports.append(report)
                if format == "markdown":
                    click.echo(f"✓ {specialist.value}: {len(report.proposed_changes)} PRs proposed")
            except Exception as e:
                if format == "markdown":
                    click.echo(f"✗ {specialist.value}: {e}", err=True)

    if not reports:
        raise click.ClickException("No specialist reports generated")

    # Synthesize findings
    if format == "markdown":
        click.echo("\nSynthesizing findings...")

    # Collect all proposed changes by priority
    high_risk = []
    medium_risk = []
    low_risk = []

    for report in reports:
        for change in report.proposed_changes:
            if change.risk == "High":
                high_risk.append((report.specialist, change))
            elif change.risk == "Medium":
                medium_risk.append((report.specialist, change))
            else:
                low_risk.append((report.specialist, change))

    # Collect cross-agent questions
    cross_agent_questions = [
        (r.specialist, r.cross_agent_question)
        for r in reports
        if r.cross_agent_question
    ]

    # Collect metrics to monitor
    all_metrics = {metric for report in reports for metric in report.metrics}

    # Build evaluation summary
    evaluation = {
        "date": eval_date,
        "health_status": health or "UNKNOWN",
        "specialists_consulted": [r.specialist for r in reports],
        "total_findings": len(reports),
        "proposed_changes": {
            "high_risk": len(high_risk),
            "medium_risk": len(medium_risk),
            "low_risk": len(low_risk),
            "total": len(high_risk) + len(medium_risk) + len(low_risk),
        },
        "cross_agent_questions": len(cross_agent_questions),
        "metrics_to_monitor": len(set(all_metrics)),
        "reports": [
            {
                "specialist": r.specialist,
                "confidence": r.confidence,
                "proposed_changes": len(r.proposed_changes),
                "cross_agent_question": r.cross_agent_question,
            }
            for r in reports
        ],
        "recommendations": {
            "high_priority": [
                {
                    "specialist": spec,
                    "title": change.title,
                    "files": change.files,
                    "risk": change.risk,
                    "dependencies": change.dependencies,
                }
                for spec, change in high_risk
            ],
            "medium_priority": [
                {
                    "specialist": spec,
                    "title": change.title,
                    "files": change.files,
                    "risk": change.risk,
                }
                for spec, change in medium_risk
            ],
            "low_priority": [
                {
                    "specialist": spec,
                    "title": change.title,
                }
                for spec, change in low_risk
            ],
        },
        "cross_agent_coordination": [
            {"from": spec, "question": q} for spec, q in cross_agent_questions
        ],
        "metrics": sorted(set(all_metrics)),
    }

    # Output results
    if format == "json":
        output_text = json.dumps(evaluation, indent=2)
    else:
        # Markdown format
        lines = [
            "# Pit Crew Evaluation Report",
            f"\n**Date**: {eval_date}",
            f"**Health Status**: {health or 'UNKNOWN'}",
            f"\n## Summary",
            f"- **Specialists Consulted**: {len(reports)}",
            f"- **Total Proposed Changes**: {evaluation['proposed_changes']['total']}",
            f"  - High Risk: {evaluation['proposed_changes']['high_risk']}",
            f"  - Medium Risk: {evaluation['proposed_changes']['medium_risk']}",
            f"  - Low Risk: {evaluation['proposed_changes']['low_risk']}",
            f"- **Cross-Agent Questions**: {len(cross_agent_questions)}",
            f"- **Metrics to Monitor**: {len(set(all_metrics))}",
        ]

        # High priority recommendations
        if high_risk:
            lines.append("\n## High Priority Recommendations")
            for i, (spec, change) in enumerate(high_risk, 1):
                lines.extend([
                    f"\n### {i}. {change.title}",
                    f"**Specialist**: {spec}",
                    f"**Risk**: {change.risk}",
                    f"**Files**: {', '.join(change.files)}",
                    f"**Dependencies**: {change.dependencies or 'None'}",
                ])

        # Medium priority
        if medium_risk:
            lines.append("\n## Medium Priority Recommendations")
            for i, (spec, change) in enumerate(medium_risk, 1):
                lines.extend([
                    f"\n### {i}. {change.title}",
                    f"**Specialist**: {spec}",
                    f"**Files**: {', '.join(change.files)}",
                ])

        # Low priority
        if low_risk:
            lines.append("\n## Low Priority Recommendations")
            for i, (spec, change) in enumerate(low_risk, 1):
                lines.extend([
                    f"\n### {i}. {change.title}",
                    f"**Specialist**: {spec}",
                ])

        # Cross-agent questions
        if cross_agent_questions:
            lines.append("\n## Cross-Agent Coordination Needed")
            for spec, q in cross_agent_questions:
                lines.append(f"- **{spec}**: {q}")

        # Metrics
        if all_metrics:
            lines.append("\n## Metrics to Monitor")
            for metric in sorted(set(all_metrics)):
                lines.append(f"- {metric}")

        output_text = "\n".join(lines)

    # Save or print
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(output_text)
        click.echo(f"\nEvaluation saved to: {output}")
    else:
        click.echo("\n" + output_text)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    cli()
