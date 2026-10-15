"""Deterministic incident reports. This module does not call a model."""
import re
from pathlib import Path
from ..models import (
    Event,
    LogLine,
    ReportLocation,
    ReportMoment,
    ReportStatement,
    EvidenceRecord,
    IncidentReport,
)
from ..logs.patterns import (
    AUTH_SUCCESS,
    CALLBACK_ID,
    JOB_ID,
    PROCESS_DEATH,
    REQUEST_ID,
    SERVICE_UNIT,
)
from ..logs.correlation import (
    exception_signature,
    lifecycle_kind,
)
from ..logs.selection import (
    event_id,
    independent_events,
    select_primary,
)
from .stacktraces import (
    TYPED_EXCEPTION,
    application_locations,
    stack_region,
)


SPRING_APP = re.compile(r"---\s+\[(?P<app>[^\]]+)\]")
NAMED_SERVICE = re.compile(r"\b(?:app|service)=([A-Za-z0-9_.@+-]+)")
HTTP_STATUS = re.compile(r"\bstatus=(\d{3})\b")
PATH_VALUE = re.compile(r"\bpath=(\S+)")
WORKER_VALUE = re.compile(r"\bworker=(\S+)")
NULL_DEREFERENCE = re.compile(
    r"(NullReferenceException|nil pointer|invalid memory address|Cannot read properties of null)",
    re.IGNORECASE,
)
MISSING_VALUE = re.compile(
    r"(<nil>|\b=null\b|_present=false|client_present=false|record=null|"
    r"tenant_id(?:_present)?=(?:false|None)|\bnot found\b|matching_keys=0)",
    re.IGNORECASE,
)
CONFIG_CHANGE = re.compile(
    r"(change detected|\bactivated\b|\bdiff\b|swap=scheduled|soft_fail=true)",
    re.IGNORECASE,
)
FAILED_INIT = re.compile(r"(initialization failed|invalid endpoint)", re.IGNORECASE)
LANGUAGE_BY_SUFFIX = {
    ".java": "Java",
    ".py": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".js": "JavaScript",
    ".cs": "C#",
}

def _language_of(locations: list[ReportLocation]) -> str | None:
    """Language of the first application frame whose suffix we recognise."""
    for location in locations:
        suffix = Path(location.path).suffix
        language = LANGUAGE_BY_SUFFIX.get(suffix)
        if language:
            return language
    return None


def _framework_of(event: Event, region: str) -> str | None:
    """Name a framework only when the incident text, or a dropped runtime frame, names it."""
    blob = "\n".join(line.raw for line in event.lines()) + "\n" + region
    if "org.springframework" in blob or "springframework" in blob or SPRING_APP.search(event.seed.raw):
        return "Spring Boot"
    if "gin-gonic" in blob or re.search(r"\bframework=gin\b", blob):
        return "Gin"
    if re.search(r"\bactix\b", blob, re.IGNORECASE):
        return "Actix"
    if re.search(r"django(?:/|\\|\.)", blob, re.IGNORECASE):
        return "Django"
    if re.search(r"\bexpress\b", blob, re.IGNORECASE):
        return "Express"
    if "AspNetCore" in blob:
        return "ASP.NET Core"
    return None


def _service_of(event: Event) -> str | None:
    """Name the application, preferring app= over a worker or plan label.

    `service=standard`, `service=express`, and `service=premium` are labels on
    the process, not the application name. A Spring bracket, a systemd unit,
    or the logger package is the fallback.
    """
    app_names: list[str] = []
    service_names: list[str] = []
    for line in event.lines():
        for match in NAMED_SERVICE.finditer(line.raw):
            if match.group(0).startswith("app="):
                app_names.append(match.group(1))
            else:
                service_names.append(match.group(1))
    if app_names:
        return app_names[0]
    for name in service_names:
        if name not in {"standard", "express", "premium"}:
            return name
    bracket = SPRING_APP.search(event.seed.raw)
    if bracket:
        return bracket.group(1)
    units = []
    for line in event.lines():
        units.extend(SERVICE_UNIT.findall(line.raw))
    if units:
        return units[0].removesuffix(".service")
    logger = event.seed.logger
    if not logger or logger in {"runtime.panic", "gunicorn.error", "process.crash"}:
        return None
    if "::" in logger:
        return logger.split("::", 1)[0]
    return logger.split(".", 1)[0]


def _exception_of(event: Event, region: str) -> tuple[str | None, str | None]:
    """Innermost exception type and its message. Panic text is read from the stack region."""
    texts = [event.seed.raw, *[line.raw for line in event.stack]]
    for text in reversed(texts):
        match = TYPED_EXCEPTION.search(text.strip())
        if match:
            message = match.group("message").strip() or None
            return match.group("type"), message
    panic = re.search(r"(?m)^panic:\s*(.+)$", region)
    if panic:
        return "panic", panic.group(1).strip()
    panicked = re.search(r"(?m)^.*panicked at .+\n(?P<message>.+)$", region)
    if panicked:
        message = panicked.group("message").strip()
        if message and not message.lower().startswith("stack backtrace"):
            return "panic", message
    if exception_signature(event) == "panic":
        return "panic", None
    named = re.search(r"\bexception=([A-Za-z0-9_.]+)", event.seed.raw)
    if named:
        return named.group(1), None
    quoted = re.search(r'\berror="([^"]+)"', event.seed.raw)
    if quoted:
        kind, separator, message = quoted.group(1).partition(":")
        return kind.strip(), (message.strip() or None) if separator else None
    return None, None


def _outcome_of(event: Event) -> str:
    """How the process ended, from the lifecycle lines already attached."""
    kinds = {lifecycle_kind(line) for line in event.consequences}
    if PROCESS_DEATH.search(event.seed.raw):
        kinds.add("death")
    if "death" in kinds and "started" in kinds:
        return "restarted"
    if "death" in kinds and "restart" in kinds:
        return "restart_attempted"
    if "death" in kinds:
        return "terminated"
    if "recovery" in kinds:
        return "recovered"
    return "no_termination_observed"


def _impact_sentence(outcome: str) -> str:
    """One sentence for the outcome. Every outcome the builder emits is a key."""
    return {
        "restarted": "The process terminated and the service came back.",
        "restart_attempted": "The process terminated and a restart was attempted.",
        "terminated": "The process terminated.",
        "recovered": "The service recovered.",
        "no_termination_observed": (
            "No process termination was identified in the extracted evidence."
        ),
        "none": "No primary failure was identified.",
    }[outcome]


def _affected_of(event: Event) -> list[dict]:
    """Ids named on the failure itself, not ids from unrelated nearby requests."""
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    text = event.seed.raw
    path = PATH_VALUE.search(text)
    detail = path.group(1) if path else None
    for kind, pattern in (
        ("request", REQUEST_ID),
        ("job", JOB_ID),
        ("callback", CALLBACK_ID),
        ("worker", WORKER_VALUE),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        key = (kind, match.group(1))
        if key in seen:
            continue
        seen.add(key)
        found.append({"kind": kind, "id": match.group(1), "detail": detail})
    return found


def _status_of(event: Event) -> tuple[str, int] | None:
    """The first 5xx status on or after the failure, with its log line."""
    for line in event.lines():
        if line.number < event.seed.number:
            continue
        match = HTTP_STATUS.search(line.raw)
        if match and match.group(1).startswith("5"):
            return match.group(1), line.number
    return None


def _timeline_kind(line: LogLine, event: Event) -> str | None:
    """A timeline step, or None for a line that only repeats the stack."""
    if not line.is_record:
        return None
    if line.number == event.seed.number:
        return "failure"
    if MISSING_VALUE.search(line.raw):
        return "invalid_state"
    if FAILED_INIT.search(line.raw):
        return "init"
    if CONFIG_CHANGE.search(line.raw):
        return "configuration"
    kind = lifecycle_kind(line)
    if kind in {"death", "restart", "started", "recovery"}:
        return kind
    return None


def _timeline_text(line: LogLine, kind: str, thrown: ReportLocation | None) -> str:
    """Wording for one timeline step. The failure step names the source location."""
    message = re.sub(r"\s+", " ", (line.message or line.raw).strip())
    if kind == "failure":
        if thrown and thrown.path and thrown.source_line:
            place = f"{thrown.path}:{thrown.source_line}"
            if thrown.source_column:
                place = f"{place}:{thrown.source_column}"
            if thrown.role == "innermost":
                return f"The innermost application frame kept in the log is {place}."
            return f"The failure was raised at {place}."
        return "The failure was recorded on this line."
    if kind == "death":
        return "The process exited."
    if kind == "restart":
        return "A restart was attempted."
    if kind == "started":
        return "The service started again."
    if kind == "recovery":
        return "The service reported that it was listening or healthy."
    if kind == "configuration":
        if re.search(r"change detected|\bdiff\b|\bactivated\b|swap=scheduled", line.raw, re.IGNORECASE):
            return f"Configuration changed. The line says: {message}"
        return f"A configuration check was recorded. The line says: {message}"
    if kind == "init":
        return f"Initialization failed. The line says: {message}"
    return f"A missing or invalid value was recorded. The line says: {message}"


def _timeline_of(event: Event, locations: list[ReportLocation]) -> list[ReportMoment]:
    """File order. Consecutive lifecycle lines of the same kind collapse to the first."""
    thrown = next((location for location in locations if location.role in {"throw_site", "innermost"}), None)
    moments: list[ReportMoment] = []
    previous: str | None = None
    for line in event.lines():
        kind = _timeline_kind(line, event)
        if kind is None:
            continue
        if kind == previous and kind in {"death", "restart", "started", "recovery"}:
            continue
        previous = kind
        moments.append(
            ReportMoment(
                timestamp=line.timestamp,
                log_line=line.number,
                kind=kind,
                text=_timeline_text(line, kind, thrown),
            )
        )
    return moments


def _quote_fact(line: LogLine) -> ReportStatement:
    """One observed fact that quotes this log line."""
    message = re.sub(r"\s+", " ", (line.message or line.raw).strip())
    return ReportStatement(f"Log line {line.number} records `{message}`.", [line.number])


def _line_with(event: Event, pattern: re.Pattern[str]) -> LogLine | None:
    """The earliest incident line that matches."""
    return next((line for line in event.lines() if pattern.search(line.raw)), None)


def _last_line_with(event: Event, pattern: re.Pattern[str]) -> LogLine | None:
    """The latest incident line that matches. Used for the innermost exception text."""
    found = None
    for line in event.lines():
        if pattern.search(line.raw):
            found = line
    return found


def _evidence_of(
    event: Event,
    exception: str | None,
    message: str | None,
    outcome: str,
    moments: list[ReportMoment],
) -> tuple[list[ReportStatement], list[ReportStatement], list[str]]:
    """Quoted facts, an empty explanations list, and limits the log cannot settle.

    Explanations stay empty so a later renderer cannot present a guessed cause
    as evidence.
    """
    observed: list[ReportStatement] = [_quote_fact(event.seed)]
    explanations: list[ReportStatement] = []
    unknowns = [
        "The repository was not opened, so the current source may differ from these frames.",
        "The log does not establish a code change.",
    ]
    described = " ".join(part for part in (exception, message) if part)
    missing = _line_with(event, MISSING_VALUE)
    if "NoSuchMethodError" in described:
        site = _last_line_with(event, re.compile(r"NoSuchMethodError")) or event.seed
        if site.number != event.seed.number:
            observed.append(_quote_fact(site))
        unknowns.append("The log does not say which copy of the class was loaded.")
    elif "ModuleNotFoundError" in described:
        site = _last_line_with(event, re.compile(r"ModuleNotFoundError")) or event.seed
        if site.number != event.seed.number:
            observed.append(_quote_fact(site))
        unknowns.append("The log does not say whether that module is declared as a dependency.")
    elif NULL_DEREFERENCE.search(described) and missing is not None and missing.number != event.seed.number:
        observed.append(_quote_fact(missing))
        unknowns.append("The log does not say why that value was missing.")
    auth_ok = _line_with(event, AUTH_SUCCESS)
    tenant_absent = _line_with(
        event,
        re.compile(r"tenant_id(?:_present)?=(?:false|None)", re.IGNORECASE),
    )
    if auth_ok is not None and tenant_absent is not None and auth_ok.number != tenant_absent.number:
        observed.append(
            ReportStatement(
                "Token validation succeeded on this request, and a same-request line "
                "shows the tenant claim was absent.",
                sorted([auth_ok.number, tenant_absent.number]),
            )
        )
        unknowns.append(
            "A valid token does not show that every claim the operation requires was present."
        )
    wanted = {
        "restarted": {"death", "started"},
        "restart_attempted": {"death", "restart"},
        "terminated": {"death"},
        "recovered": {"recovery"},
    }.get(outcome, set())
    quoted = {number for fact in observed for number in fact.log_lines}
    for moment in moments:
        if moment.kind not in wanted or moment.log_line in quoted:
            continue
        line = next(item for item in event.lines() if item.number == moment.log_line)
        observed.append(_quote_fact(line))
        quoted.add(moment.log_line)
    status = _status_of(event)
    if status is not None and outcome == "no_termination_observed":
        number = status[1]
        if number not in quoted:
            line = next(item for item in event.lines() if item.number == number)
            observed.append(_quote_fact(line))
    return observed, explanations, unknowns


def _summary_of(
    event: Event,
    *,
    service: str | None,
    language: str | None,
    framework: str | None,
    exception: str | None,
    message: str | None,
    outcome: str,
    affected: list[dict],
) -> str:
    """Two paragraphs: the failure, then the process outcome and named ids."""
    who = service or "The application"
    traits = " ".join(part for part in (language, framework) if part)
    subject = f"{who} ({traits})" if traits else who
    when = event.seed.timestamp or "an unknown time"
    failure = exception or "an error"
    opening = f"{subject} failed at {when} with {failure}."
    if message:
        sentence = message if message.endswith((".", "!", "?")) else f"{message}."
        opening += f" The logged description is: {sentence}"
    impact = _impact_sentence(outcome)
    status = _status_of(event)
    if outcome == "no_termination_observed" and status is not None:
        impact += f" Status {status[0]} was recorded for this incident."
    if affected:
        named = ", ".join(f"{item['kind']} {item['id']}" for item in affected)
        extra = ""
        detail = next((item["detail"] for item in affected if item.get("detail")), None)
        if detail:
            extra = f" Path {detail} is named on the failure."
        impact += f" The failure names {named}.{extra}"
    return opening + "\n\n" + impact


def _evidence_records(event: Event) -> list[EvidenceRecord]:
    """Incident lines in file order, with correlation kept apart from role."""
    stack_numbers = {line.number for line in event.stack}
    context_numbers = {line.number for line in event.context}
    records: list[EvidenceRecord] = []
    for line in event.lines():
        if line.number == event.seed.number or line.number in stack_numbers:
            correlation = "failure"
            role = "failure" if line.number == event.seed.number else "stack"
        elif line.number in event.possible and line.number not in event.confirmed:
            correlation = "possible"
            role = "context" if line.number in context_numbers and line.number < event.seed.number else "later"
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


def _warning_entry(event: Event) -> dict:
    """One independent event for the other-warnings list."""
    message = event.seed.message or event.seed.raw.strip()
    signature = exception_signature(event)
    if signature and signature not in message:
        message = f"{message} | {signature}"
    return {"id": event_id(event), "log_line": event.seed.number, "message": message}


def _numbered(lines: list[LogLine]) -> str:
    """Raw lines with their source numbers, for the stack and excerpt blocks."""
    return "\n".join(f"{line.number}: {line.raw}" for line in lines)


def build_incident_report(path: Path, text: str, events: list[Event]) -> IncidentReport:
    """Assemble one incident from the lines already grouped for it."""
    primary = select_primary(events)
    independent = independent_events(events, primary)
    warnings = [_warning_entry(event) for event in independent]
    if primary is None:
        summary = "No primary failure was identified.\n\n" + _impact_sentence("none")
        return IncidentReport(
            source=path.name,
            incident_id=None,
            service=None,
            language=None,
            framework=None,
            timestamp=None,
            exception=None,
            message=None,
            summary=summary,
            outcome="none",
            affected=[],
            locations=[],
            timeline=[],
            observed=[],
            explanations=[],
            unknowns=[
                "The repository was not opened, so the current source may differ from these frames.",
                "The log does not establish a code change.",
            ],
            lines=[],
            stack="",
            excerpts="",
            other_warnings=warnings,
        )
    region = "\n".join(raw for _, raw in stack_region(text, primary))
    locations = application_locations(text, primary)
    exception, message = _exception_of(primary, region)
    outcome = _outcome_of(primary)
    affected = _affected_of(primary)
    service = _service_of(primary)
    language = _language_of(locations)
    framework = _framework_of(primary, region)
    moments = _timeline_of(primary, locations)
    observed, explanations, unknowns = _evidence_of(primary, exception, message, outcome, moments)
    summary = _summary_of(
        primary,
        service=service,
        language=language,
        framework=framework,
        exception=exception,
        message=message,
        outcome=outcome,
        affected=affected,
    )
    return IncidentReport(
        source=path.name,
        incident_id=event_id(primary),
        service=service,
        language=language,
        framework=framework,
        timestamp=primary.seed.timestamp or None,
        exception=exception,
        message=message,
        summary=summary,
        outcome=outcome,
        affected=affected,
        locations=locations,
        timeline=moments,
        observed=observed,
        explanations=explanations,
        unknowns=unknowns,
        lines=_evidence_records(primary),
        stack=_numbered([primary.seed, *primary.stack]),
        excerpts=_numbered(primary.lines()),
        other_warnings=warnings,
    )
