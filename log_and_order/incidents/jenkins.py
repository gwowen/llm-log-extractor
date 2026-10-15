"""Build the shared incident report from a parsed Jenkins console log."""

from pathlib import Path

from ..logs.jenkins.analysis import JenkinsAnalysis
from ..logs.jenkins.failures import FailureHit
from ..logs.selection import event_id
from ..models import (
    EvidenceRecord,
    IncidentReport,
    JenkinsCandidate,
    JenkinsFacts,
    ReportLocation,
    ReportMoment,
    ReportStatement,
)

_LANGUAGE = {
    ".java": "Java",
    ".py": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".cs": "C#",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".h": "C",
    ".hpp": "C++",
}


def _language(path: str | None) -> str | None:
    if not path:
        return None
    suffix = Path(path).suffix.lower()
    return _LANGUAGE.get(suffix)


def _suggestions(hit: FailureHit | None, command: str | None) -> tuple[list[str], list[str]]:
    """Investigation targets and actions taken from the extracted evidence."""
    if hit is None:
        return [], []
    targets: list[str] = []
    actions: list[str] = []
    if hit.source_file:
        targets.append(hit.source_file)
        if hit.source_line:
            actions.append(f"Inspect {hit.source_file} around line {hit.source_line}.")
        else:
            actions.append(f"Inspect {hit.source_file}.")
    if hit.test_name and hit.test_name not in targets:
        targets.append(hit.test_name)
        actions.append(f"Re-run the failing test {hit.test_name} and compare it with this log.")
    if command:
        actions.append(f"Run `{command}` locally and compare its output with this log.")
    if hit.symptom:
        actions.append(
            "The log shows a Jenkins status line without a tool diagnostic. "
            "Inspect the command output above that status for the underlying error."
        )
    elif hit.category == "BUILD":
        actions.append("Correct the diagnostic and re-run the build command.")
    elif hit.category == "TEST":
        actions.append("Compare the expected and actual values recorded in the test output.")
    elif hit.category == "DEPLOYMENT":
        actions.append("Inspect the deployment command, its target, and the error returned by that tool.")
    elif hit.category == "INFRASTRUCTURE":
        actions.append("Check the agent, workspace, network path, or missing executable named in the log.")
    elif hit.category == "CONFIGURATION":
        actions.append("Check the Jenkinsfile, credentials, or configuration named in the log.")
    elif hit.category == "APPLICATION":
        actions.append("Inspect the application frame named in the log.")
    return targets[:5], actions[:5]


def _facts(analysis: JenkinsAnalysis) -> JenkinsFacts:
    hit = analysis.primary
    command = hit.command if hit and hit.stage_association != "uncertain" else None
    stage = hit.stage if hit and hit.stage_association == "confirmed" else None
    targets, actions = _suggestions(hit, command)
    numbers: list[int] = []
    if hit is not None:
        numbers.append(hit.log_line)
        numbers.extend(number for number in hit.detail_lines if number not in numbers)
        if hit.command_line and hit.command_line not in numbers:
            numbers.append(hit.command_line)
    candidates = [
        JenkinsCandidate(
            message=item.message,
            category=item.category,
            stage=item.stage if item.stage_association == "confirmed" else None,
            stage_association=item.stage_association,
            log_line=item.log_line,
            source_file=item.source_file,
            source_line=item.source_line,
        )
        for item in analysis.others
    ]
    return JenkinsFacts(
        job_style=analysis.job_style,
        build_result=analysis.build_result,
        failed_stage=stage,
        stage_association=hit.stage_association if hit else None,
        failure_category=hit.category if hit else None,
        failed_command=command,
        exit_code=hit.exit_code if hit else None,
        primary_message=hit.message if hit else None,
        source_file=hit.source_file if hit else None,
        source_line=hit.source_line if hit else None,
        source_column=hit.source_column if hit else None,
        symptom=bool(hit and hit.symptom),
        log_lines=sorted(numbers),
        possible_stages=list(hit.possible_stages) if hit else [],
        candidates=candidates,
        investigation_targets=targets,
        suggested_actions=actions,
    )


def _locations(hit: FailureHit | None) -> list[ReportLocation]:
    if hit is None or not hit.source_file or hit.source_line is None:
        return []
    return [
        ReportLocation(
            role="throw_site",
            order=0,
            path=hit.source_file,
            symbol=hit.test_name or "",
            source_line=hit.source_line,
            source_column=hit.source_column,
            log_line=hit.log_line,
        )
    ]


def _outcome(result: str | None, recovered: bool) -> str:
    if result == "SUCCESS":
        return "recovered" if recovered else "none"
    if result in {"FAILURE", "ABORTED"}:
        return "terminated"
    return "no_termination_observed"


def _summary(analysis: JenkinsAnalysis, facts: JenkinsFacts) -> str:
    result = analysis.build_result or "unknown"
    if analysis.primary is None:
        if analysis.build_result == "SUCCESS" and analysis.recovered:
            return (
                f"The Jenkins build finished with result {result}. "
                "An earlier error was followed by a recovery line, so it is not the primary failure."
            )
        if analysis.build_result == "SUCCESS":
            return f"The Jenkins build finished with result {result}. No primary failure was identified."
        return (
            f"The Jenkins build result is {result}. "
            "No actionable diagnostic was identified in the console log."
        )
    where = "The failing stage could not be determined from the interleaved output."
    if facts.failed_stage:
        where = f"It was recorded in stage {facts.failed_stage}."
    cause = facts.primary_message or "an error"
    kind = facts.failure_category or "UNKNOWN"
    opening = f"The Jenkins build result is {result}. The primary failure is a {kind} error: {cause}"
    if not opening.endswith("."):
        opening += "."
    if facts.symptom:
        opening += " This line is a Jenkins status, and the log does not show a more specific cause."
    return opening + " " + where


def _observed(analysis: JenkinsAnalysis) -> list[ReportStatement]:
    hit = analysis.primary
    if hit is None:
        return []
    statements = [
        ReportStatement(f"Log line {hit.log_line} records `{hit.message}`.", [hit.log_line])
    ]
    if hit.command and hit.command_line and hit.stage_association != "uncertain":
        statements.append(
            ReportStatement(
                f"The command on log line {hit.command_line} is `{hit.command}`.",
                [hit.command_line],
            )
        )
    if hit.exit_code is not None:
        exit_lines = [number for number in hit.detail_lines if number != hit.log_line]
        cited = exit_lines[:1] or [hit.log_line]
        statements.append(
            ReportStatement(
                f"A Jenkins status line records exit code {hit.exit_code}.",
                cited,
            )
        )
    if analysis.build_result and analysis.build_result_line:
        statements.append(
            ReportStatement(
                f"The build result is {analysis.build_result}.",
                [analysis.build_result_line],
            )
        )
    return statements


def _timeline(analysis: JenkinsAnalysis) -> list[ReportMoment]:
    event = analysis.primary_event
    hit = analysis.primary
    if event is None or hit is None:
        if analysis.build_result_line:
            return [
                ReportMoment(
                    timestamp="",
                    log_line=analysis.build_result_line,
                    kind="finished",
                    text=f"The build result is {analysis.build_result}.",
                )
            ]
        return []
    moments: list[ReportMoment] = []
    if hit.command_line and hit.stage_association != "uncertain":
        stamp = next((line.timestamp for line in event.lines() if line.number == hit.command_line), "")
        moments.append(
            ReportMoment(
                timestamp=stamp,
                log_line=hit.command_line,
                kind="command",
                text=f"The stage ran `{hit.command}`.",
            )
        )
    moments.append(
        ReportMoment(
            timestamp=event.seed.timestamp,
            log_line=hit.log_line,
            kind="failure",
            text="The failure was recorded on this line.",
        )
    )
    if analysis.build_result_line:
        moments.append(
            ReportMoment(
                timestamp="",
                log_line=analysis.build_result_line,
                kind="finished",
                text=f"The build result is {analysis.build_result}.",
            )
        )
    return moments


def _records(event) -> list[EvidenceRecord]:
    if event is None:
        return []
    stack_numbers = {line.number for line in event.stack}
    context_numbers = {line.number for line in event.context}
    records: list[EvidenceRecord] = []
    for line in event.lines():
        if line.number == event.seed.number or line.number in stack_numbers:
            correlation = "failure"
            role = "failure" if line.number == event.seed.number else "stack"
        elif line.number in event.possible and line.number not in event.confirmed:
            correlation = "possible"
            role = "context" if line.number in context_numbers else "later"
        else:
            correlation = "confirmed"
            role = "context" if line.number in context_numbers and line.number < event.seed.number else "later"
        records.append(
            EvidenceRecord(
                number=line.number,
                text=line.raw,
                correlation=correlation,
                role=role,
            )
        )
    return records


def _warnings(analysis: JenkinsAnalysis) -> list[dict]:
    rows = []
    for hit in [*analysis.others, *analysis.recovered]:
        prefix = "Recovered: " if hit.recovered else ""
        stage = f" stage {hit.stage}." if hit.stage and hit.stage_association == "confirmed" else ""
        rows.append(
            {
                "id": f"E{hit.log_line}",
                "log_line": hit.log_line,
                "message": f"{prefix}{hit.message}{stage}",
            }
        )
    return rows


def _numbered(lines) -> str:
    return "\n".join(f"{line.number}: {line.raw}" for line in lines)


def _unknowns(analysis: JenkinsAnalysis, facts: JenkinsFacts) -> list[str]:
    unknowns = [
        "The repository was not opened, so the current source may differ from these locations.",
        "A Jenkins stage and a command show where the line was printed. They do not by themselves prove causation.",
    ]
    if facts.symptom:
        unknowns.append("The primary line is a Jenkins status. The underlying tool error was not in the extracted lines.")
    if facts.stage_association == "uncertain":
        unknowns.append("Parallel output was interleaved, so the failing stage was not identified.")
    if analysis.others:
        unknowns.append("Another stage recorded its own failure. The log does not show that one caused the other.")
    if analysis.build_result is None:
        unknowns.append("The log does not contain a Finished line.")
    return unknowns


def build_jenkins_report(path: Path, analysis: JenkinsAnalysis) -> IncidentReport:
    """The same incident document the application-log builder produces, plus Jenkins facts."""
    facts = _facts(analysis)
    hit = analysis.primary
    event = analysis.primary_event
    outcome = _outcome(analysis.build_result, bool(analysis.recovered))
    locations = _locations(hit)
    observed = _observed(analysis)
    return IncidentReport(
        source=path.name,
        incident_id=event_id(event) if event is not None else None,
        service=analysis.job_name,
        language=_language(facts.source_file),
        framework=None,
        timestamp=event.seed.timestamp or None if event is not None else None,
        exception=None,
        message=facts.primary_message,
        summary=_summary(analysis, facts),
        outcome=outcome if hit is not None or analysis.build_result else "none",
        affected=[],
        locations=locations,
        timeline=_timeline(analysis),
        observed=observed,
        explanations=[],
        unknowns=_unknowns(analysis, facts),
        lines=_records(event),
        stack=_numbered([event.seed, *event.stack]) if event is not None else "",
        excerpts=_numbered(event.lines()) if event is not None else "",
        other_warnings=_warnings(analysis),
        jenkins=facts,
    )
