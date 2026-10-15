"""Shared incident data. Independent of parsing, the model, and report text."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LogLine:
    """One source line. `number` is 1-based and matches the file."""

    number: int
    raw: str
    timestamp: str = ""
    level: str = ""
    trace_id: str | None = None
    logger: str = ""
    message: str = ""
    is_record: bool = False
    moment: datetime | None = None
    request_id: str | None = None


@dataclass
class Event:
    """One incident built from a seed line plus the lines that belong to it.

    `stack` is the traceback. `context` is nearby evidence: the same request,
    or setup lines that name this failure. `members` are later warnings on
    the same trace. `consequences` are the process exit and proxy failures
    that follow. `repeat_lines` are later copies of a bare warning, kept so
    the report can show the first and last time.
    """

    seed: LogLine
    stack: list[LogLine] = field(default_factory=list)
    context: list[LogLine] = field(default_factory=list)
    members: list[LogLine] = field(default_factory=list)
    consequences: list[LogLine] = field(default_factory=list)
    repeat_numbers: list[int] = field(default_factory=list)
    repeat_lines: list[LogLine] = field(default_factory=list)
    # Line numbers whose link to this incident is an id, a unit, or a pid.
    confirmed: set[int] = field(default_factory=set)
    # Line numbers kept only because they name the component and sit immediately before it.
    possible: set[int] = field(default_factory=set)

    def lines(self) -> list[LogLine]:
        """Every attached line, in file order. Repeat copies are separate."""
        return sorted(
            [self.seed, *self.stack, *self.context, *self.members, *self.consequences],
            key=lambda line: line.number,
        )


@dataclass
class ReportLocation:
    """One application frame. Source line and log line are different numbers."""

    role: str  # throw_site, innermost, or caller
    order: int
    path: str
    symbol: str
    source_line: int | None
    source_column: int | None
    log_line: int  # the log line that named this source location


@dataclass
class ReportMoment:
    """One timeline step, in file order."""

    timestamp: str
    log_line: int
    kind: str
    text: str


@dataclass
class ReportStatement:
    """A fact taken from the log, with the lines it quotes."""

    statement: str
    log_lines: list[int]


@dataclass
class EvidenceRecord:
    """One incident line a report or a coding agent can resolve by number."""

    number: int
    text: str
    correlation: str  # failure, confirmed, or possible
    role: str  # failure, stack, context, or later


@dataclass
class JenkinsCandidate:
    """Another failure the log shows. It is not claimed as the cause of the primary."""

    message: str
    category: str
    stage: str | None
    stage_association: str
    log_line: int
    source_file: str | None = None
    source_line: int | None = None


@dataclass
class JenkinsFacts:
    """Deterministic Jenkins fields. Absent on an application-log report."""

    job_style: str
    build_result: str | None
    failed_stage: str | None
    stage_association: str | None
    failure_category: str | None
    failed_command: str | None
    exit_code: int | None
    primary_message: str | None
    source_file: str | None
    source_line: int | None
    source_column: int | None
    symptom: bool
    log_lines: list[int]
    possible_stages: list[str] = field(default_factory=list)
    candidates: list[JenkinsCandidate] = field(default_factory=list)
    investigation_targets: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class IncidentReport:
    """The one incident both the Markdown brief and the JSON document describe."""

    source: str
    incident_id: str | None
    service: str | None
    language: str | None
    framework: str | None
    timestamp: str | None
    exception: str | None
    message: str | None
    summary: str
    # restarted, restart_attempted, terminated, recovered, no_termination_observed, or none.
    outcome: str
    affected: list[dict]
    locations: list[ReportLocation]
    timeline: list[ReportMoment]
    observed: list[ReportStatement]
    # Left empty. The report quotes the log and does not invent a cause.
    explanations: list[ReportStatement]
    unknowns: list[str]
    lines: list[EvidenceRecord]
    stack: str
    excerpts: str
    other_warnings: list[dict]
    jenkins: JenkinsFacts | None = None


@dataclass
class DiagnosticHypothesis:
    """One model interpretation. It is not a logged fact."""

    description: str
    confidence: str
    supporting_log_lines: list[int]
    assumptions: list[str]
    missing_evidence: list[str]
    verification: str


@dataclass
class InvestigationStep:
    """One prioritised check for a coding agent."""

    priority: int
    target: str
    action: str
    rationale: str
    supporting_log_lines: list[int]
    expected_finding: str


@dataclass
class RegressionTestSuggestion:
    """A scenario to verify after the repository's tests have been inspected."""

    scenario: str
    expected_behaviour: str
    supporting_log_lines: list[int]


@dataclass
class IncidentEnrichment:
    """Validated model analysis, or an explicit fallback with empty claims."""

    failure_mechanism: str | None
    hypotheses: list[DiagnosticHypothesis]
    investigation_steps: list[InvestigationStep]
    regression_tests: list[RegressionTestSuggestion]
    additional_unknowns: list[str]
    model: str | None = None
    enriched: bool = False
    fallback: bool = True
    reason: str | None = "disabled"


@dataclass
class EvidencePackage:
    """The only incident text the enrichment model is allowed to see."""

    text: str
    allowed_lines: set[int]
    correlations: dict[int, str]
    known_text: str
    omitted: int
