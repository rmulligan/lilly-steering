"""Parser for specialist markdown reports to SpecialistReport schema."""

import re
from core.pit_crew.schemas import SpecialistReport, ProposedChange


def _is_new_format(markdown: str) -> bool:
    """Detect if markdown uses new BASE_TEMPLATE format.

    New format has "## Executive Summary" section.
    Old format has "# Specialist Report: [name]" title.

    Args:
        markdown: Markdown-formatted specialist report

    Returns:
        True if new format, False if old format
    """
    return bool(re.search(r"^## Executive Summary", markdown, re.MULTILINE))


def parse_specialist_report(markdown: str) -> SpecialistReport:
    """Parse markdown specialist report to SpecialistReport schema.

    Supports both formats:
    - New format (BASE_TEMPLATE): Executive Summary, Findings, Root Cause, etc.
    - Old format: Specialist Report title, Context, Findings, Proposed Changes

    Tries new format first, falls back to old format if not detected.

    Args:
        markdown: Markdown-formatted specialist report

    Returns:
        Parsed SpecialistReport instance

    Raises:
        ValueError: If required sections are missing or malformed
    """
    if _is_new_format(markdown):
        return _parse_new_format(markdown)
    else:
        return _parse_old_format(markdown)
def _parse_new_format(markdown: str) -> SpecialistReport:
    """Parse new BASE_TEMPLATE format.

    New format sections:
    - ## Executive Summary
    - ## Findings
    - ## Root Cause Analysis
    - ## Recommended Actions
    - ## Cross-Agent Questions
    - ## Evidence
    - ## Confidence

    Maps to SpecialistReport schema:
    - specialist: extracted from context or "unknown"
    - date: current date or extracted from context
    - context: Executive Summary
    - findings: Findings + Root Cause Analysis combined
    - proposed_changes: parsed from Recommended Actions
    - cross_agent_question: from Cross-Agent Questions
    - confidence: from Confidence section
    - references: from Evidence section
    """
    # Extract Executive Summary -> context
    summary_match = re.search(
        r"## Executive Summary\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if not summary_match:
        raise ValueError("Missing required section: Executive Summary")
    context = summary_match.group(1).strip()

    # Extract Findings
    findings_match = re.search(
        r"## Findings\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    findings_text = findings_match.group(1).strip() if findings_match else ""

    # Extract Root Cause Analysis
    root_cause_match = re.search(
        r"## Root Cause Analysis\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    root_cause_text = root_cause_match.group(1).strip() if root_cause_match else ""

    # Combine Findings + Root Cause
    findings_parts = []
    if findings_text:
        findings_parts.append(findings_text)
    if root_cause_text:
        findings_parts.append(f"\n\n**Root Cause:**\n{root_cause_text}")
    findings = "\n".join(findings_parts) if findings_parts else "No findings."

    # Extract Recommended Actions -> proposed_changes
    proposed_changes = []
    actions_match = re.search(
        r"## Recommended Actions\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )

    if actions_match:
        actions_section = actions_match.group(1).strip()

        # Check if "None at this time" or similar
        if not re.search(r"^\d+\.\s+\*\*\[", actions_section, re.MULTILINE):
            # No numbered actions found
            pass
        else:
            # Split by numbered action items
            # Pattern: 1. **[Type]**: Description
            action_items = re.split(r'\n(?=\d+\.\s+\*\*\[)', actions_section.strip())

            for item in action_items:
                if not item.strip():
                    continue

                # Parse action header: 1. **[PR]**: Description
                # Note: Colon is OUTSIDE the bold markers
                header_match = re.match(
                    r"^\d+\.\s+\*\*\[([^\]]+)\]\*\*:\s+(.+)",
                    item
                )
                if not header_match:
                    continue

                action_type = header_match.group(1).strip()
                description = header_match.group(2).strip()

                # Extract Files field
                files = []
                files_match = re.search(
                    r"^\s*-\s+Files:\s+(.+?)(?=\n\s*-|\n\d+\.|\Z)",
                    item,
                    re.MULTILINE | re.DOTALL
                )
                if files_match:
                    files_text = files_match.group(1).strip()
                    # Handle comma or space separated files
                    files = [f.strip() for f in re.split(r'[,\s]+', files_text) if f.strip()]

                # Extract Risk field
                risk = "Low"  # Default
                risk_match = re.search(
                    r"^\s*-\s+Risk:\s+(\w+)",
                    item,
                    re.MULTILINE | re.IGNORECASE
                )
                if risk_match:
                    risk = risk_match.group(1).strip().title()

                # Extract Change field
                change_detail = ""
                change_match = re.search(
                    r"^\s*-\s+Change:\s+(.+?)(?=\n\s*-|\n\d+\.|\Z)",
                    item,
                    re.MULTILINE | re.DOTALL
                )
                if change_match:
                    change_detail = change_match.group(1).strip()

                # Combine description and change detail
                changes = description
                if change_detail:
                    changes = f"{description}\n\n{change_detail}"

                proposed_changes.append(ProposedChange(
                    title=f"{action_type}: {description[:50]}",  # Truncate for title
                    files=files,
                    changes=changes,
                    tests="See specialist report for testing strategy.",
                    risk=risk,
                    dependencies=None
                ))

    # Extract Cross-Agent Questions
    cross_agent_question = None
    questions_match = re.search(
        r"## Cross-Agent Questions\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if questions_match:
        questions_text = questions_match.group(1).strip()
        # Extract first question as cross-agent question
        question_line = next(
            (line.strip("- ").strip() for line in questions_text.split("\n")
             if line.strip().startswith("-")),
            None
        )
        if question_line:
            cross_agent_question = question_line

    # Extract Evidence -> references
    references = []
    evidence_match = re.search(
        r"## Evidence\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if evidence_match:
        evidence_text = evidence_match.group(1).strip()
        # Store code blocks and snippets as references
        references = [f"Evidence: {evidence_text[:200]}..."]

    # Extract Confidence
    confidence_match = re.search(
        r"## Confidence\n\s*\[?(Low|Medium|High)\]?",
        markdown,
        re.IGNORECASE
    )
    if not confidence_match:
        raise ValueError("Missing required section: Confidence")
    confidence_value = confidence_match.group(1).lower()
    # Normalize to low/med/high
    if confidence_value == "medium":
        confidence_value = "med"
    confidence = confidence_value

    # Try to extract specialist from context or default to "unknown"
    specialist = "unknown"
    specialist_hint = re.search(r"(?:steering|simulation|substrate|reflexion)[\w.]*", context, re.IGNORECASE)
    if specialist_hint:
        specialist = specialist_hint.group(0).lower()

    # Date defaults to placeholder
    date = "generated"

    return SpecialistReport(
        specialist=specialist,
        date=date,
        context=context,
        findings=findings,
        proposed_changes=proposed_changes,
        metrics=[],  # New format doesn't have explicit metrics section
        cross_agent_question=cross_agent_question,
        confidence=confidence,
        references=references
    )


def _parse_old_format(markdown: str) -> SpecialistReport:
    """Parse old format (pre-BASE_TEMPLATE).

    Original format sections:
    - # Specialist Report: [name]
    - **Date:** [date]
    - ## Context
    - ## Findings
    - ## Proposed Changes
    - ## Metrics to Monitor
    - ## References
    - **Confidence:** [low/med/high]
    """
    # Extract specialist name from title
    specialist_match = re.search(
        r"# Specialist Report: (.+?)(?:\n|$)",
        markdown
    )
    if not specialist_match:
        raise ValueError("Missing required section: Specialist Report title")
    specialist = specialist_match.group(1).strip()

    # Extract date
    date_match = re.search(
        r"\*\*Date:\*\* (.+?)(?:\n|$)",
        markdown
    )
    if not date_match:
        raise ValueError("Missing required section: Date")
    date = date_match.group(1).strip()

    # Extract context
    context_match = re.search(
        r"## Context\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if not context_match:
        raise ValueError("Missing required section: Context")
    context = context_match.group(1).strip()

    # Extract findings
    findings_match = re.search(
        r"## Findings\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if not findings_match:
        raise ValueError("Missing required section: Findings")
    findings = findings_match.group(1).strip()

    # Extract cross-agent question (optional)
    cross_agent_question = None
    question_match = re.search(
        r"\*\*For (.+?):\*\* (.+?)(?=\n\n|\n##|\Z)",
        findings,
        re.DOTALL
    )
    if question_match:
        agent_name = question_match.group(1).strip()
        question_text = question_match.group(2).strip()
        cross_agent_question = f"For {agent_name}: {question_text}"

    # Extract proposed changes
    proposed_changes = []
    changes_section_match = re.search(
        r"## Proposed Changes\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )

    if changes_section_match:
        changes_section = changes_section_match.group(1)

        # Check if "None at this time" or similar
        if not re.search(r"####\s+PR\s+\d+:", changes_section, re.IGNORECASE):
            # No PR sections found - either "None" or empty
            pass
        else:
            # Try primary pattern first (with separate Title field)
            pr_sections = list(re.finditer(
                r"####\s+PR\s+\d+:\s+(.+?)\n"  # PR heading with title
                r"\s*\*\*Title:\*\*\s+(.+?)\n"  # Title field
                r"\s*\*\*Files:\*\*\s*\n(.*?)\n"  # Files section
                r"\s*\*\*Changes:\*\*\s*\n(.*?)\n"  # Changes section
                r"\s*\*\*Tests:\*\*\s*\n(.*?)\n"  # Tests section
                r"\s*\*\*Risk:\*\*\s+([^\n]+)\n"  # Risk on same line
                r"(?:\s*\*\*Dependencies:\*\*\s+([^\n]+)\n)?",  # Optional dependencies
                changes_section,
                re.DOTALL
            ))

            # Fallback: if primary pattern finds nothing, try without Title field
            if not pr_sections:
                pr_sections = list(re.finditer(
                    r"####\s+PR\s+\d+:\s+(.+?)\n"  # PR heading with title
                    r"\s*\*\*Files:\*\*\s*\n(.*?)\n"  # Files section
                    r"\s*\*\*Changes:\*\*\s*\n(.*?)\n"  # Changes section
                    r"\s*\*\*Tests:\*\*\s*\n(.*?)\n"  # Tests section
                    r"\s*\*\*Risk:\*\*\s+([^\n]+)\n"  # Risk on same line
                    r"(?:\s*\*\*Dependencies:\*\*\s+([^\n]+)\n)?",  # Optional dependencies
                    changes_section,
                    re.DOTALL
                ))
                # Use heading title for fallback format
                for pr_match in pr_sections:
                    title = pr_match.group(1).strip()  # From heading
                    files_text = pr_match.group(2).strip()
                    changes = pr_match.group(3).strip()
                    tests = pr_match.group(4).strip()
                    # Normalize risk capitalization (Low/Medium/High)
                    risk = pr_match.group(5).strip().title()
                    dependencies = pr_match.group(6).strip() if pr_match.group(6) else None

                    # Parse files list
                    files = [
                        line.strip("- ").strip()
                        for line in files_text.split("\n")
                        if line.strip().startswith("-")
                    ]

                    proposed_changes.append(ProposedChange(
                        title=title,
                        files=files,
                        changes=changes,
                        tests=tests,
                        risk=risk,
                        dependencies=dependencies
                    ))
            else:
                # Primary format matched
                for pr_match in pr_sections:
                    title = pr_match.group(2).strip()  # From Title field
                    files_text = pr_match.group(3).strip()
                    changes = pr_match.group(4).strip()
                    tests = pr_match.group(5).strip()
                    # Normalize risk capitalization (Low/Medium/High)
                    risk = pr_match.group(6).strip().title()
                    dependencies = pr_match.group(7).strip() if pr_match.group(7) else None

                    # Parse files list
                    files = [
                        line.strip("- ").strip()
                        for line in files_text.split("\n")
                        if line.strip().startswith("-")
                    ]

                    proposed_changes.append(ProposedChange(
                        title=title,
                        files=files,
                        changes=changes,
                        tests=tests,
                        risk=risk,
                        dependencies=dependencies
                    ))

    # Extract metrics to monitor
    metrics = []
    metrics_match = re.search(
        r"## Metrics to Monitor\n(.*?)(?=\n## |\Z)",
        markdown,
        re.DOTALL
    )
    if metrics_match:
        metrics_text = metrics_match.group(1).strip()
        metrics = [
            line.strip("- ").strip()
            for line in metrics_text.split("\n")
            if line.strip().startswith("-")
        ]

    # Extract references
    references = []
    refs_match = re.search(
        r"## References\n(.*?)(?=\n## |\*\*Confidence:\*\*|\Z)",
        markdown,
        re.DOTALL
    )
    if refs_match:
        refs_text = refs_match.group(1).strip()
        references = [
            line.strip("- ").strip()
            for line in refs_text.split("\n")
            if line.strip().startswith("-")
        ]

    # Extract confidence
    confidence_match = re.search(
        r"\*\*Confidence:\*\*\s+(low|med|high)(?:\n|\Z)",
        markdown,
        re.IGNORECASE
    )
    if not confidence_match:
        raise ValueError("Missing required section: Confidence")
    confidence = confidence_match.group(1).lower()

    return SpecialistReport(
        specialist=specialist,
        date=date,
        context=context,
        findings=findings,
        proposed_changes=proposed_changes,
        metrics=metrics,
        cross_agent_question=cross_agent_question,
        confidence=confidence,
        references=references
    )
