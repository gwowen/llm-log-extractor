"""JSON incident document. Schema version stays 1."""
import json
from dataclasses import asdict

from ..config import (
    SCHEMA_VERSION,
)
from ..models import (
    IncidentEnrichment,
    IncidentReport,
    JenkinsFacts,
)


def _analysis_payload(enrichment: IncidentEnrichment) -> dict:
    """The analysis object shared with the Markdown brief. An empty fallback is still present."""
    return {
        "model": enrichment.model,
        "enriched": enrichment.enriched,
        "fallback": enrichment.fallback,
        "reason": enrichment.reason,
        "failure_mechanism": enrichment.failure_mechanism,
        "hypotheses": [asdict(item) for item in enrichment.hypotheses],
        "investigation_steps": [asdict(item) for item in enrichment.investigation_steps],
        "regression_tests": [asdict(item) for item in enrichment.regression_tests],
        "additional_unknowns": enrichment.additional_unknowns,
    }

def _jenkins_payload(facts: JenkinsFacts) -> dict:
    """Jenkins fields. Source line and log line stay in different places."""
    primary = None
    if facts.primary_message is not None:
        primary = {
            "message": facts.primary_message,
            "category": facts.failure_category,
            "source_file": facts.source_file,
            "source_line": facts.source_line,
            "source_column": facts.source_column,
            "log_lines": facts.log_lines,
            "symptom": facts.symptom,
        }
    return {
        "log_type": "jenkins",
        "job_style": facts.job_style,
        "build_result": facts.build_result,
        "failed_stage": facts.failed_stage,
        "stage_association": facts.stage_association,
        "possible_stages": facts.possible_stages,
        "failure_category": facts.failure_category,
        "failed_command": facts.failed_command,
        "exit_code": facts.exit_code,
        "primary_failure": primary,
        "candidates": [asdict(item) for item in facts.candidates],
        "investigation_targets": facts.investigation_targets,
        "suggested_actions": facts.suggested_actions,
    }


def render_json_report(report: IncidentReport, enrichment: IncidentEnrichment) -> str:
    """The same incident and the same analysis object as the Markdown brief."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": report.source,
        "incident": {
            "id": report.incident_id,
            "service": report.service,
            "language": report.language,
            "framework": report.framework,
            "timestamp": report.timestamp,
            "exception": report.exception,
            "message": report.message,
            "summary": report.summary,
            "outcome": report.outcome,
            "affected": report.affected,
        },
        "locations": [asdict(location) for location in report.locations],
        "timeline": [asdict(moment) for moment in report.timeline],
        "evidence": {
            "observed": [asdict(fact) for fact in report.observed],
            # Always empty. Explanations are not filled from the log or the model.
            "possible_explanations": [asdict(item) for item in report.explanations],
            "unknowns": report.unknowns,
            "lines": [asdict(line) for line in report.lines],
        },
        "analysis": _analysis_payload(enrichment),
        "other_warnings": report.other_warnings,
    }
    if report.jenkins is not None:
        payload["jenkins"] = _jenkins_payload(report.jenkins)
    return json.dumps(payload, indent=2) + "\n"
