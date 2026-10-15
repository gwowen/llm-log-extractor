"""JSON incident document. Schema version stays 1."""
import json
from dataclasses import asdict

from ..config import (
    SCHEMA_VERSION,
)
from ..models import (
    IncidentEnrichment,
    IncidentReport,
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
    return json.dumps(payload, indent=2) + "\n"
