"""Human Jenkins report and the compact text sent to the model."""

import re

from ..logs.jenkins.analysis import JenkinsAnalysis
from ..models import IncidentReport, JenkinsFacts


def format_jenkins_model_input(analysis: JenkinsAnalysis) -> str:
    """Structured incident text. The model does not receive the raw console log."""
    hit = analysis.primary
    if hit is None:
        return "Log type: Jenkins\nPrimary failure:\nNone"
    stage = hit.stage if hit.stage_association == "confirmed" else "uncertain"
    command = hit.command if hit.stage_association != "uncertain" else "unknown"
    lines = [
        "Log type: Jenkins",
        f"Build result: {analysis.build_result or 'unknown'}",
        f"Stage: {stage}",
        f"Stage association: {hit.stage_association}",
        f"Category: {hit.category}",
        f"Command: {command or 'unknown'}",
        f"Exit code: {hit.exit_code if hit.exit_code is not None else 'unknown'}",
        f"Symptom: {'yes' if hit.symptom else 'no'}",
        "Failure:",
    ]
    event = analysis.primary_event
    if event is not None:
        for line in event.lines():
            stamp = f"{line.timestamp} " if line.timestamp else ""
            label = "possible" if line.number in event.possible else "confirmed"
            lines.append(f"{label} {line.number} {stamp}{line.message or line.raw}".rstrip())
    else:
        lines.append(f"confirmed {hit.log_line} {hit.message}")
    if analysis.others:
        lines.append("Other independent failures:")
        for item in analysis.others:
            where = item.stage if item.stage_association == "confirmed" else "uncertain stage"
            lines.append(f"{item.log_line} {item.category} {where}: {item.message}")
    return "\n".join(lines)


def _section(narrative: str, heading: str) -> str | None:
    """The body under one heading, when the model followed the requested shape."""
    match = re.search(
        rf"(?:^|\n)#*\s*{heading}\s*\n(?P<body>.*?)(?=\n#*\s*(?:Root cause|Related symptoms)\b|\Z)",
        narrative,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None
    body = match.group("body").strip()
    return body or None


def _root_cause(facts: JenkinsFacts | None) -> str:
    """Deterministic wording used when the model is skipped or returns nothing."""
    if facts is None or not facts.primary_message:
        return "No actionable failure was identified in the extracted lines."
    if facts.symptom:
        return (
            f"The log records `{facts.primary_message}`. "
            "No earlier diagnostic identifies the underlying cause. "
            "This line is a Jenkins status, not a root cause."
        )
    sentence = f"The log shows: {facts.primary_message}"
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    if facts.source_file and facts.source_line:
        sentence += f" The diagnostic names {facts.source_file} around line {facts.source_line}."
    elif facts.source_file:
        sentence += f" The diagnostic names {facts.source_file}."
    return sentence


def _evidence_bullets(report: IncidentReport) -> list[str]:
    if report.jenkins is None:
        return []
    bullets = []
    seen: set[int] = set()
    for line in report.lines:
        if line.number in seen:
            continue
        seen.add(line.number)
        text = " ".join(line.text.split())
        if len(text) > 240:
            text = text[:237].rstrip() + "..."
        bullets.append(f"- Line {line.number}: {text}")
    return bullets


def _header(facts: JenkinsFacts) -> list[str]:
    rows = ["# Main failure", ""]
    if facts.build_result:
        rows.append(f"Build result: {facts.build_result}")
    if facts.stage_association == "uncertain":
        names = ", ".join(facts.possible_stages) if facts.possible_stages else "unknown"
        rows.append(f"Failed stage: uncertain (open stages: {names})")
    elif facts.failed_stage:
        rows.append(f"Failed stage: {facts.failed_stage}")
    if facts.failure_category:
        rows.append(f"Failure category: {facts.failure_category}")
    if facts.failed_command:
        rows.append(f"Failed command: {facts.failed_command}")
    if facts.exit_code is not None:
        rows.append(f"Exit code: {facts.exit_code}")
    return rows


def render_jenkins_human(report: IncidentReport, narrative: str | None = None) -> str:
    """The human report. Facts and line numbers come from the parser."""
    facts = report.jenkins
    if facts is None or facts.primary_message is None:
        result = facts.build_result if facts and facts.build_result else "unknown"
        if facts and facts.primary_message is None and report.other_warnings:
            rows = [
                f"Build result: {result}",
                "",
                "No primary failure was identified.",
                "",
                "## Other failures",
                "",
            ]
            rows.extend(
                f"- {item['id']} (log line {item['log_line']}). {item['message']}"
                for item in report.other_warnings
            )
            return "\n".join(rows).rstrip() + "\n"
        if facts and facts.build_result == "SUCCESS":
            return f"Build result: {result}\n\nNo relevant errors or warnings were found.\n"
        if facts and facts.primary_message is None:
            return (
                f"Build result: {result}\n\n"
                "No actionable diagnostic was identified in the console log.\n"
            )
        return "No relevant errors or warnings were found.\n"
    root = _root_cause(facts)
    related = "None"
    if narrative:
        parsed_root = _section(narrative, "Root cause")
        parsed_related = _section(narrative, "Related symptoms")
        if parsed_root:
            root = parsed_root
        elif narrative.strip():
            root = narrative.strip()
        if parsed_related:
            related = parsed_related
    if related == "None" and facts.symptom:
        related = "No separate underlying diagnostic was extracted."
    elif related == "None" and facts.exit_code is not None:
        related = (
            f"Jenkins reported exit code {facts.exit_code} after the diagnostic above."
        )
        if facts.build_result:
            related += f" The build result is {facts.build_result}."
    parts = [
        *_header(facts),
        "",
        "## Root cause",
        "",
        root,
        "",
        "## Evidence",
        "",
    ]
    bullets = _evidence_bullets(report)
    parts.extend(bullets or ["None"])
    parts.extend(["", "## Suggested investigation", ""])
    if facts.suggested_actions:
        parts.extend(facts.suggested_actions)
    else:
        parts.append("None")
    parts.extend(["", "## Related symptoms", "", related])
    if report.other_warnings:
        parts.extend(["", "## Other failures", ""])
        for item in report.other_warnings:
            parts.append(f"- {item['id']} (log line {item['log_line']}). {item['message']}")
    return "\n".join(parts).rstrip() + "\n"


def dump_jenkins(analysis: JenkinsAnalysis) -> str:
    """Show the Jenkins incident that would be sent to the model."""
    primary = format_jenkins_model_input(analysis)
    if not analysis.recovered and not analysis.others:
        reported = "Other warnings\nNone"
    else:
        rows = []
        for hit in [*analysis.others, *analysis.recovered]:
            label = "recovered" if hit.recovered else "independent"
            rows.append(f"{event_id_of(hit.log_line)} {label} {hit.category}: {hit.message}")
        reported = "Other warnings\n" + "\n".join(rows)
    return f"# Sent to the model\n{primary}\n\n# Reported by Python\n{reported}\n"


def event_id_of(number: int) -> str:
    return f"E{number}"
