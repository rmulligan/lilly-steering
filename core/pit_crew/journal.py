"""Journal updater for specialist system."""

import os
import tempfile
from pathlib import Path
from core.pit_crew.schemas import SpecialistReport


def append_journal_entry(
    journal_path: Path,
    report: SpecialistReport,
    hub_ticket_id: str,
    pr_number: str | None = None
) -> None:
    """Append entry to specialist journal markdown file.

    Inserts entry after "## Journal Entries" marker, before existing entries
    (reverse chronological order). Atomic write ensures no partial updates.

    Args:
        journal_path: Path to journal markdown file
        report: Parsed specialist report
        hub_ticket_id: Letta hub ticket identifier (e.g. "ticket-123")
        pr_number: Optional GitHub PR number as string (e.g. "180")

    Raises:
        FileNotFoundError: Journal file does not exist
        ValueError: Journal marker "## Journal Entries" not found
    """
    if not journal_path.exists():
        raise FileNotFoundError(f"Journal file not found: {journal_path}")

    # Read existing content
    content = journal_path.read_text()

    # Find insertion point after marker
    marker = "## Journal Entries"
    if marker not in content:
        raise ValueError(f"Journal marker '{marker}' not found in {journal_path}")

    marker_idx = content.find(marker)
    marker_end = content.find("\n", marker_idx) + 1

    # Build new entry
    entry_lines = [
        f"### {report.date}",
        f"**Hub ticket**: #{hub_ticket_id}",
    ]

    if pr_number is not None:
        entry_lines.append(f"**PR**: #{pr_number}")

    entry_lines.append("**Status**: open")

    # Summary is first sentence/paragraph of findings
    summary = report.findings.split("\n")[0].strip()
    if summary.startswith("1. ") or summary.startswith("- "):
        # If findings start with a list, use the first substantive content
        for line in report.findings.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove list marker if present
                if line.startswith("1. ") or line.startswith("- "):
                    summary = line[3:]
                else:
                    summary = line
                break

    entry_lines.append(f"**Summary:** {summary}")

    # Proposed changes
    if report.proposed_changes:
        entry_lines.append("**Proposed Changes:**")
        for i, change in enumerate(report.proposed_changes, 1):
            entry_lines.append(f"- PR {i}: {change.title}")

    # Metrics
    if report.metrics:
        metrics_str = ", ".join(report.metrics)
        entry_lines.append(f"**Metrics:** {metrics_str}")

    # Cross-agent question
    if report.cross_agent_question:
        entry_lines.append(f"**Cross-Agent Question:** {report.cross_agent_question}")

    # Confidence
    entry_lines.append(f"**Confidence:** {report.confidence}")

    # References
    if report.references:
        refs_str = ", ".join(report.references)
        entry_lines.append(f"**References:** {refs_str}")

    # Build complete entry with blank line after
    entry = "\n".join(entry_lines) + "\n\n"

    # Insert entry after marker
    new_content = content[:marker_end] + "\n" + entry + content[marker_end:]

    # Atomic write - use temp file and rename to prevent corruption
    temp_fd, temp_path_str = tempfile.mkstemp(dir=journal_path.parent, text=True)
    try:
        with os.fdopen(temp_fd, 'w') as temp_file:
            temp_file.write(new_content)
        os.rename(temp_path_str, journal_path)
    except Exception:
        os.unlink(temp_path_str)
        raise
