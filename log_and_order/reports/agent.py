"""Markdown incident brief for a developer or a coding agent."""

from ..models import (
    IncidentEnrichment,
    IncidentReport,
    JenkinsFacts,
    ReportLocation,
)
from ..incidents.builder import (
    _impact_sentence,
)


def _role_phrase(location: ReportLocation) -> str:
    """How this frame relates to the failure. Source line and log line stay distinct."""
    if location.role == "throw_site":
        return "This is where the exception or panic was thrown."
    if location.role == "innermost":
        return (
            "This is the innermost application frame the log kept. "
            "Runtime frames were left out, so the raising line may be further inside the runtime."
        )
    return "This frame called toward the failure."


def _location_bullet(location: ReportLocation) -> str:
    """One application frame, with its source line and the log line that named it."""
    column = f", source column {location.source_column}" if location.source_column else ""
    symbol = f"`{location.symbol}`" if location.symbol else "the logged panic location"
    return (
        f"- {symbol} in `{location.path}` at source line {location.source_line}{column}. "
        f"Log line {location.log_line}. {_role_phrase(location)}"
    )


def _task_lines(report: IncidentReport, enrichment: IncidentEnrichment) -> list[str]:
    """A handoff built from the extracted incident, not a generic debugging prompt."""
    lines = ["## Task for coding agent", ""]
    if report.incident_id:
        lines.append(f"Investigate incident {report.incident_id} in `{report.source}`.")
    else:
        lines.append(f"No primary failure was identified in `{report.source}`.")
    if report.exception:
        sentence = f"The logged failure is `{report.exception}`."
        if report.message:
            sentence += f" The logged description is: {report.message}"
        lines.append(sentence)
    named = [
        f"`{location.symbol or location.path}` in `{location.path}` "
        f"at source line {location.source_line} (log line {location.log_line})"
        for location in report.locations
    ]
    if named:
        lines.append("Logged application locations: " + "; ".join(named) + ".")
    if report.jenkins:
        lines.append(_jenkins_task_sentence(report.jenkins))
    else:
        lines.append(_impact_sentence(report.outcome))
    lines.append("")
    lines.append(
        "Hypotheses in this report are model interpretations, not facts. "
        "Do not treat them as a confirmed cause or a fix. "
        "Logged source line numbers may not match the current checkout."
    )
    lines.append("")
    if report.locations:
        focus = next(
            (location for location in report.locations if location.role in {"throw_site", "innermost"}),
            report.locations[0],
        )
        shown = f"`{focus.symbol}`" if focus.symbol else f"`{focus.path}`"
        lines.append(
            f"1. Locate {shown} in `{focus.path}` and inspect its implementation, "
            "then inspect the other logged application functions."
        )
    else:
        lines.append(
            "1. Locate the logged application functions from the excerpts and inspect their implementations."
        )
    if enrichment.failure_mechanism:
        lines.append(
            "2. Check the reported failure mechanism against the actual source code: "
            f"{enrichment.failure_mechanism}"
        )
    else:
        lines.append(
            "2. Check the logged failure against the actual source code. "
            "No model failure mechanism was produced."
        )
    lines.append("3. Trace the relevant inputs, state transitions, and callers that can reach this failure.")
    if enrichment.hypotheses:
        rendered = "; ".join(
            f"{item.confidence} confidence: {item.description}" for item in enrichment.hypotheses
        )
        lines.append(f"4. Validate or reject these hypotheses using the repository: {rendered}")
    else:
        lines.append("4. There are no model hypotheses to accept. Use the logged facts and the repository.")
    lines.append("5. Identify the smallest appropriate fix supported by the repository.")
    if enrichment.regression_tests:
        scenarios = "; ".join(item.scenario for item in enrichment.regression_tests)
        lines.append(
            "6. After inspecting how this repository tests, add a regression test where feasible "
            f"for: {scenarios}"
        )
    else:
        lines.append(
            "6. After inspecting how this repository tests, add a regression test where feasible "
            "for the logged failure."
        )
    lines.append("7. Run relevant existing tests when the environment permits.")
    lines.append("8. Explain the verified root cause, the modifications, and the validation results.")
    return lines


def _jenkins_task_sentence(facts: JenkinsFacts) -> str:
    """Where the console log places the failure. This is not a confirmed cause."""
    result = facts.build_result or "unknown"
    if facts.primary_message is None:
        return f"The Jenkins build result is {result}. No primary failure was identified."
    stage = facts.failed_stage or "an unidentified stage"
    if facts.stage_association == "uncertain":
        stage = "a stage the interleaved output did not identify"
    command = f" The failed command is `{facts.failed_command}`." if facts.failed_command else ""
    category = facts.failure_category or "UNKNOWN"
    return (
        f"The Jenkins build result is {result}. "
        f"The extracted failure is category {category} in {stage}.{command}"
    )


def _jenkins_lines(facts: JenkinsFacts) -> list[str]:
    """Structured Jenkins fields. Missing values are left out."""
    lines = ["## Jenkins build", ""]
    if facts.build_result:
        lines.append(f"- Build result: {facts.build_result}")
    if facts.stage_association == "uncertain":
        names = ", ".join(facts.possible_stages) if facts.possible_stages else "unknown"
        lines.append(f"- Failed stage: uncertain (open stages: {names})")
    elif facts.failed_stage:
        lines.append(f"- Failed stage: {facts.failed_stage}")
    if facts.failure_category:
        lines.append(f"- Failure category: {facts.failure_category}")
    if facts.failed_command:
        lines.append(f"- Failed command: `{facts.failed_command}`")
    if facts.exit_code is not None:
        lines.append(f"- Exit code: {facts.exit_code}")
    if facts.source_file:
        place = facts.source_file
        if facts.source_line:
            place = f"{place} source line {facts.source_line}"
        log_line = facts.log_lines[0] if facts.log_lines else None
        cited = f" Log line {log_line}." if log_line else ""
        lines.append(f"- Source: `{place}`.{cited}")
    if facts.symptom:
        lines.append("- The primary line is a Jenkins status. A more specific cause was not extracted.")
    if facts.suggested_actions:
        lines.append("- Investigation suggested from the extracted evidence:")
        lines.extend(f"  - {action}" for action in facts.suggested_actions)
    lines.append("")
    lines.append(
        "These fields are extracted from the console log. "
        "A stage or a command shows where a line was printed. It does not prove causation."
    )
    return lines


def _mechanism_lines(enrichment: IncidentEnrichment) -> list[str]:
    """The failure mechanism, labeled as model analysis."""
    lines = ["## Failure mechanism", ""]
    if enrichment.enriched and enrichment.failure_mechanism:
        lines.append("This is model analysis, not a confirmed root cause.")
        lines.append("")
        lines.append(enrichment.failure_mechanism)
    elif enrichment.enriched:
        lines.append("The model did not state a failure mechanism.")
    else:
        lines.append("No model analysis was produced.")
    return lines


def _hypothesis_lines(enrichment: IncidentEnrichment) -> list[str]:
    """Hypotheses in the model's order, each marked as an interpretation."""
    lines = ["## Ranked hypotheses", ""]
    if not enrichment.hypotheses:
        if enrichment.enriched:
            lines.append("The model returned no hypothesis.")
        else:
            lines.append("No model hypotheses were produced.")
        return lines
    for index, item in enumerate(enrichment.hypotheses, start=1):
        cited = ", ".join(str(number) for number in item.supporting_log_lines)
        lines.extend(
            [
                f"### {index}. {item.description}",
                "",
                f"- Confidence: {item.confidence}. This is a model interpretation, not a confirmed cause or a fix.",
                f"- Supporting log lines: {cited}.",
                "- Assumptions: " + ("; ".join(item.assumptions) if item.assumptions else "none stated.") + ".",
                "- Missing evidence: "
                + ("; ".join(item.missing_evidence) if item.missing_evidence else "none stated.")
                + ".",
                f"- Verification: {item.verification}",
                "",
            ]
        )
    return lines


def _investigation_lines(enrichment: IncidentEnrichment) -> list[str]:
    """Investigation steps in priority order, after validation has capped the list."""
    lines = ["## Recommended investigation", ""]
    if not enrichment.investigation_steps:
        lines.append("No investigation steps were produced.")
        return lines
    for index, step in enumerate(enrichment.investigation_steps, start=1):
        cited = ", ".join(str(number) for number in step.supporting_log_lines)
        lines.append(
            f"{index}. Inspect `{step.target}`. {step.action} "
            f"Why: {step.rationale} "
            f"A confirming or disproving finding would be: {step.expected_finding} "
            f"Log lines {cited}."
        )
    return lines


def _regression_lines(enrichment: IncidentEnrichment) -> list[str]:
    """Scenarios to verify. They are not test filenames or commands."""
    lines = [
        "## Regression tests to consider",
        "",
        "These are failure scenarios and the behaviour to verify after the repository's tests "
        "have been inspected. They are not filenames, frameworks, or commands.",
        "",
    ]
    if not enrichment.regression_tests:
        lines.append("None.")
        return lines
    for item in enrichment.regression_tests:
        cited = ", ".join(str(number) for number in item.supporting_log_lines)
        lines.append(
            f"- Scenario: {item.scenario} Expected behaviour: {item.expected_behaviour} Log lines {cited}."
        )
    return lines


def _unknown_lines(report: IncidentReport, enrichment: IncidentEnrichment) -> list[str]:
    """Limits Python recorded, plus any extra limits the model stated."""
    lines = ["## Unknowns and limitations", ""]
    lines.extend(f"- {item}" for item in report.unknowns)
    if enrichment.reason == "unavailable":
        lines.append("- Model enrichment was not available, so this report has no model-generated hypotheses.")
    elif enrichment.reason == "invalid":
        lines.append("- The model response failed validation, so this report has no model-generated hypotheses.")
    elif enrichment.reason == "disabled":
        lines.append("- Model enrichment was not requested.")
    elif enrichment.reason == "no_incident":
        lines.append("- There is no primary incident to analyse.")
    if enrichment.additional_unknowns:
        lines.append("- The model also reported these limits, which are not logged facts:")
        lines.extend(f"  - {item}" for item in enrichment.additional_unknowns)
    return lines

def render_agent_report(report: IncidentReport, enrichment: IncidentEnrichment) -> str:
    """Markdown brief. Facts stay in Python's sections. Analysis stays labeled as analysis."""
    thrown = [location for location in report.locations if location.role in {"throw_site", "innermost"}]
    callers = [location for location in report.locations if location.role == "caller"]
    failure_lines = ["## Primary failure", ""]
    if report.exception:
        failure_lines.append(f"The failure is `{report.exception}`.")
        failure_lines.append("")
    if report.message:
        failure_lines.append(f"The logged description is: {report.message}")
        failure_lines.append("")
    failure_lines.append(
        "Source line is a line in the application file. Log line is a line in this log. "
        "Those numbers are not interchangeable."
    )
    if thrown:
        failure_lines.append("")
        failure_lines.extend(_location_bullet(location) for location in thrown)
    elif report.incident_id:
        failure_lines.append("")
        failure_lines.append("No application source location was identified in the log.")
    else:
        failure_lines.append("")
        failure_lines.append("None.")
    if callers:
        failure_lines.extend(["", "Callers, from the failure outward:", ""])
        failure_lines.extend(_location_bullet(location) for location in callers)

    observed = ["## Evidence and observations", ""]
    if report.observed:
        observed.extend(f"- {fact.statement}" for fact in report.observed)
    else:
        observed.append("None.")

    if report.timeline:
        timeline = ["## Chronological timeline", ""]
        for index, moment in enumerate(report.timeline, start=1):
            when = moment.timestamp or "no timestamp"
            timeline.append(f"{index}. {when} (log line {moment.log_line}). {moment.text}")
    else:
        timeline = ["## Chronological timeline", "", "None."]

    detail = ["## Supporting details", "", "### Application stack", ""]
    detail.append(f"```\n{report.stack}\n```" if report.stack else "None.")
    detail.extend(["", "### Log excerpts", ""])
    detail.append(f"```\n{report.excerpts}\n```" if report.excerpts else "None.")
    detail.extend(["", "### Other warnings", ""])
    if report.other_warnings:
        detail.extend(
            f"- {item['id']} (log line {item['log_line']}). {item['message']}"
            for item in report.other_warnings
        )
    else:
        detail.append("None.")

    parts = [
        "# Incident report",
        "",
        "## Executive summary",
        "",
        report.summary,
        "",
        *([*_jenkins_lines(report.jenkins), ""] if report.jenkins else []),
        *_task_lines(report, enrichment),
        "",
        *failure_lines,
        "",
        *_mechanism_lines(enrichment),
        "",
        *observed,
        "",
        *_hypothesis_lines(enrichment),
        "",
        *_investigation_lines(enrichment),
        "",
        *_regression_lines(enrichment),
        "",
        *_unknown_lines(report, enrichment),
        "",
        *timeline,
        "",
        *detail,
    ]
    return "\n".join(parts).rstrip() + "\n"
