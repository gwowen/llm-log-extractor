"""Turn a Jenkins console log into events the rest of the program can report."""

import re
from dataclasses import dataclass, field

from ...models import Event, LogLine
from .detect import looks_like_jenkins
from .failures import FailureHit, collect_failures
from .stages import StageSpan, reconstruct
from .text import PreparedLine, prepare_lines

FINISHED = re.compile(r"^Finished:\s+(SUCCESS|FAILURE|UNSTABLE|ABORTED|NOT_BUILT)\s*$")
WORKSPACE = re.compile(r"workspace/([^/\s]+)")
STATUS_LINE = re.compile(
    r"^(?:(?:ERROR:\s*)?script returned exit code \d+|Build step '.+' marked build as failure|Finished: )"
)
MAX_STACK = 16


@dataclass
class JenkinsAnalysis:
    """What the Jenkins parser established. The model is not involved."""

    job_style: str
    job_name: str | None
    build_result: str | None
    build_result_line: int | None
    stages: list[StageSpan] = field(default_factory=list)
    primary: FailureHit | None = None
    others: list[FailureHit] = field(default_factory=list)
    recovered: list[FailureHit] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    primary_event: Event | None = None
    lines: list[PreparedLine] = field(default_factory=list)


def _job_name(lines: list[PreparedLine]) -> str | None:
    """The workspace directory is the closest name the console gives the job."""
    for line in lines:
        match = WORKSPACE.search(line.text)
        if match:
            return match.group(1)
    return None


def _build_result(lines: list[PreparedLine]) -> tuple[str | None, int | None]:
    """The last Finished line wins. A truncated log has no result."""
    found: tuple[str | None, int | None] = (None, None)
    for line in lines:
        match = FINISHED.match(line.text.strip())
        if match:
            found = (match.group(1), line.number)
    return found


def _by_number(lines: list[PreparedLine]) -> dict[int, PreparedLine]:
    return {line.number: line for line in lines}


def _stage_line(stages: list[StageSpan], name: str | None) -> int | None:
    if not name:
        return None
    for stage in stages:
        if stage.name == name:
            return stage.start
    return None


def _event_from_hit(
    hit: FailureHit,
    lines: list[PreparedLine],
    stages: list[StageSpan],
    build_result_line: int | None,
    *,
    include_result: bool,
) -> Event:
    """One incident event. Context is the command and stage, not the whole log."""
    index = _by_number(lines)
    seed = index[hit.log_line].as_log_line(level="ERROR", message=hit.message)
    context: list[LogLine] = []
    stack: list[LogLine] = []
    consequences: list[LogLine] = []
    confirmed: set[int] = set()
    possible: set[int] = set()

    def add_context(number: int | None) -> None:
        if number is None or number == hit.log_line or number not in index:
            return
        line = index[number].as_log_line()
        if number < hit.log_line:
            context.append(line)
        if hit.stage_association == "uncertain":
            possible.add(number)
        else:
            confirmed.add(number)

    add_context(_stage_line(stages, hit.stage))
    add_context(hit.command_line)
    for number in hit.detail_lines:
        if number == hit.log_line or number not in index or len(stack) + len(consequences) >= MAX_STACK:
            continue
        prepared = index[number]
        record = prepared.as_log_line()
        if STATUS_LINE.match(prepared.text.strip()) or prepared.text.strip().startswith("Finished:"):
            consequences.append(record)
        elif number > hit.log_line:
            stack.append(record)
        else:
            context.append(record)
        confirmed.add(number)
    if include_result and build_result_line and build_result_line in index:
        consequences.append(index[build_result_line].as_log_line())
        confirmed.add(build_result_line)
    return Event(
        seed=seed,
        stack=stack,
        context=context,
        consequences=consequences,
        confirmed=confirmed,
        possible=possible,
    )


def _analyse(lines: list[PreparedLine]) -> JenkinsAnalysis:
    stages, places, style = reconstruct(lines)
    result, result_line = _build_result(lines)
    primary, others, recovered = collect_failures(lines, places)
    events: list[Event] = []
    primary_event = None
    if primary is not None:
        primary_event = _event_from_hit(
            primary, lines, stages, result_line, include_result=True
        )
        events.append(primary_event)
    for hit in others:
        events.append(_event_from_hit(hit, lines, stages, None, include_result=False))
    for hit in recovered:
        event = _event_from_hit(hit, lines, stages, None, include_result=False)
        event.seed.level = "WARN"
        events.append(event)
    return JenkinsAnalysis(
        job_style=style,
        job_name=_job_name(lines),
        build_result=result,
        build_result_line=result_line,
        stages=stages,
        primary=primary,
        others=others,
        recovered=recovered,
        events=events,
        primary_event=primary_event,
        lines=lines,
    )


def analyse_jenkins(text: str) -> JenkinsAnalysis | None:
    """Parse a Jenkins console log. Returns None when the file is not Jenkins.

    A detected log that is truncated or unbalanced still returns an analysis.
    Parser failures become an empty analysis rather than an exception.
    """
    lines = prepare_lines(text)
    if not looks_like_jenkins(lines):
        return None
    try:
        return _analyse(lines)
    except Exception:
        return JenkinsAnalysis(job_style="pipeline", job_name=_job_name(lines), lines=lines)
