"""Summarize a text file with a local Ollama model via LangChain.

A timestamped log is reduced to events in this module. Python chooses the
primary incident, its related lines, and the independent warnings. The model
narrates only the incident. Any other text file is summarized directly.
Progress goes to stderr and the report goes to stdout.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError

MODEL_NAME = "LiquidAI/lfm2.5-2.6b"
# Leave room in the context window for the prompt and the summary itself.
CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 400

# Spring-style records: timestamp, then a level. Stack frames have neither.
LOG_RECORD = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\s+"
    r"(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
    re.IGNORECASE,
)
# Runtime frames bury the application frame that names the failure.
FRAMEWORK_FRAME = re.compile(
    r"^\s+at (?:[\w.]+/)?"
    r"(?:java\.|javax\.|jdk\.|sun\.|com\.sun\.|"
    r"org\.springframework\.|org\.apache\.|jakarta\.|org\.hibernate\.)"
)
OMITTED_FRAMES = re.compile(r"^\s+\.\.\. \d+ (?:more|common frames)")
RUST_RUNTIME_FRAME = re.compile(
    r"^\s+\d+:\s+(?:rust_begin_unwind|core::|std::|alloc::|tokio::|"
    r"actix(?:_web|_http|_rt)?::)"
)
RUST_RUNTIME_AT = re.compile(r"^\s+at\s+/(?:rustc|cargo)/")
PYTHON_RUNTIME_FILE = re.compile(
    r'^\s*File "(?:[^"]*(?:site-packages|/usr/lib/python|/usr/local/lib/python)[^"]*)"'
)
# Warnings that do not describe a failure worth reading.
NOISY_WARNING = re.compile(
    r"(command=PING\b|\bheartbeat\b|\breadiness\b|status=404\b|"
    r"\bskip(?:ped|ping)?\b|\bunchanged\b|\bsuccessfully\b)",
    re.IGNORECASE,
)
# A later line on the same trace shows that the warning's operation recovered.
RECOVERED = re.compile(r"(?:\brecovered=true\b|\bretry succeeded\b)", re.IGNORECASE)
# Spring uses "Logger : message". Tracing, gunicorn, nginx, and systemd use "target: message".
RECORD_BODY = re.compile(
    r"^(?P<ts>\S+)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"
    r"(?:\s+\d+\s+---\s+\[[^\]]*\]\s+\[[^\]]*\]\s+\[(?P<mdc>[^\]]*)\])?"
    r"\s+(?P<logger>.+?)\s*:\s+(?P<message>.*)$",
    re.IGNORECASE,
)
# Spring records use traceId=. Service logs use trace_id=.
TRACE_ID = re.compile(r"trace_?[Ii]d=([0-9a-fA-F]+)")
ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}
WARN_LEVELS = {"WARN", "WARNING"}
# Request and response lines are the useful context around an error.
CONTEXT_SIGNAL = re.compile(r"(\bIncoming\b|\bOutgoing\b|\bstatus=\d{3}\b)", re.IGNORECASE)
# SQL and cache chatter shares a trace id without describing the request.
ROUTINE_CONTEXT = re.compile(
    r"(JdbcTemplate|hibernate\.SQL|\bselect\b.+\bfrom\b|Cache lookup|"
    r"HikariPool|Executing prepared SQL|lettuce|Redis PONG)",
    re.IGNORECASE,
)
# Cap on same-request and setup lines attached to one error.
MAX_CONTEXT_LINES = 8
# How far a process death, restart, or proxy failure can follow its exception.
CASCADE_WINDOW = timedelta(seconds=30)
# Identical exceptions this close can be one incident. Hours apart cannot.
SIGNATURE_WINDOW = timedelta(seconds=90)
# Diagnostic lines older than this are a different episode.
# Pass another timedelta to extract_events(lookback=...) to change the bound.
LOOKBACK = timedelta(seconds=30)
# Same-component lines in this gap are the step that entered the failure.
IMMEDIATE = timedelta(seconds=3)
# Recovery lines kept after a restart that belongs to the incident.
MAX_RECOVERY_LINES = 2
# Process death, a restart attempt, or the proxy in front of a dead process.
PROCESS_DEATH = re.compile(
    r"(exited with code|failed to boot|Shutting down: Master|"
    r"Main process exited|Failed with result|"
    r"code=killed|status=6/ABRT|Consumed .+ CPU time)",
    re.IGNORECASE,
)
PROCESS_RESTART = re.compile(
    r"(Scheduled restart|Start request repeated too quickly|Failed to start )",
    re.IGNORECASE,
)
UPSTREAM_FAILURE = re.compile(
    r"(connect\(\) failed \(111: Connection refused\)|"
    r"upstream prematurely closed|upstream_status=502)",
    re.IGNORECASE,
)
# A completed start. Warnings after this belong to the new process.
STARTED_SERVICE = re.compile(
    r"(?:\bStarted [\w.@+-]+\.service\b|\bApplication started\b|\bNow listening\b)",
    re.IGNORECASE,
)
SERVICE_UNIT = re.compile(r"\b([\w.@+-]+\.service)\b")
REQUEST_ID = re.compile(r"\brequest[_-]?[Ii]d=([A-Za-z0-9._:-]+)")
PID_VALUE = re.compile(r"\bpid[=:\s]+(\d+)\b", re.IGNORECASE)
JOB_ID = re.compile(r"\bjob[_-]?id=([A-Za-z0-9._:-]+)", re.IGNORECASE)
KEY_ID = re.compile(
    r"\b(?:key_id|selected_key_id|required_key_id|current_key_id)=(\S+)",
    re.IGNORECASE,
)
FIELD_ID = re.compile(r"\bfield=(\S+)")
COMPONENT_ID = re.compile(r"\bcomponent=(\S+)")
CALLBACK_ID = re.compile(r"\bcallback=([A-Za-z0-9_.]+)")
REVISION_ID = re.compile(r"\brevision=(\d+)")
DIFF_ARROW = re.compile(r"\b([A-Za-z0-9][\w./:-]{5,})\s*->\s*([A-Za-z0-9][\w./:-]{5,})")
NODE_RUNTIME = re.compile(r"node:internal|triggerUncaughtException|^\s*\^$|^Node\.js v")
DOTNET_RUNTIME = re.compile(
    r"^\s+at (?:System\.|Microsoft\.(?:Extensions|AspNetCore|EntityFrameworkCore)\.)"
)
RECOVERY_LINE = re.compile(
    r"(Application started|Now listening|\blistening (?:addr|host)=|"
    r"\breadiness=healthy\b|\bliveness=healthy\b)",
    re.IGNORECASE,
)
# Configuration changes, failed setup, and nil or missing state.
DIAGNOSTIC = re.compile(
    r"(change detected|initialization failed|invalid endpoint|"
    r"<nil>|client_present=false|tenant_id(?:_present)?=(?:false|None)|"
    r"soft_fail=true|swap=scheduled|"
    r"marking existing transaction as rollback-only|"
    r"\bnot found\b|=null\b|_present=false|matching_keys=0|"
    r"\bactivated\b|\bdiff\b)",
    re.IGNORECASE,
)
# Words too common to tie two lines to the same component.
GENERIC_TOKENS = {
    "api",
    "app",
    "client",
    "code",
    "com",
    "cmd",
    "debug",
    "error",
    "exception",
    "failed",
    "failure",
    "false",
    "github",
    "goroutine",
    "http",
    "https",
    "immediate",
    "info",
    "internal",
    "java",
    "level",
    "main",
    "none",
    "null",
    "order",
    "orders",
    "org",
    "panic",
    "request",
    "response",
    "runtime",
    "server",
    "service",
    "signal",
    "span",
    "start",
    "started",
    "status",
    "trace",
    "true",
    "uncaught",
    "warn",
    "warning",
    "active",
    "build",
    "present",
    "record",
    "result",
    "running",
    "update",
    "value",
}
EXCEPTION_SIGNATURE = re.compile(r"((?:\w+\.)*(?:Error|Exception):.+)$")
REPEAT_REQUEST = re.compile(r'request="[^"]*"')
REPEAT_PATH = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD)\s+\S+")
THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN = re.compile(r"<think\b", re.IGNORECASE)

# Plain-text summary. A part with nothing to say answers NONE so it can be dropped.
SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You summarize files for someone who has not read them. "
            "Write a clear summary in plain prose. "
            "Cover the purpose of the file and the important details. "
            "If this part has nothing worth reporting, reply with exactly NONE. "
            "Do not invent anything that is not in the file.",
        ),
        ("human", "File: {source}\n\n{text}"),
    ]
)

# Second pass over the partial summaries of one non-log file.
COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You combine partial summaries of one file into a single summary. "
            "Ignore any part that says NONE. "
            "Keep it concise and do not invent details.",
        ),
        ("human", "File: {source}\n\nPartial summaries:\n\n{text}"),
    ]
)

# Narrates the one incident Python already grouped. Other warnings are added later.
REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You explain one incident from lines that were already extracted. "
            "Write exactly these headings, in this order:\n"
            "Main failure\n"
            "Related events\n"
            "Under Main failure, explain the Failure lines and the Context lines. "
            "Include the exception name and the application frames that are shown. "
            "Under Related events, describe the Related lines in the order given. "
            "Those lines are later lines from the same request, process exit, "
            "proxy failures, restarts, and recovery. "
            "A context line marked possible is only nearby; do not say it caused the failure. "
            "Use the timestamp on the Failure line for when the failure happened. "
            "After each statement, cite the source line numbers shown, "
            "like (line 616) or (lines 649-651). "
            "Cite only line numbers that appear below. "
            "Do not invent line numbers, events, or a cause the lines do not state. "
            "Do not suggest a fix. "
            "Write each section once. Do not add notes, drafts, or a second copy. "
            "If a section has no lines, write None under that heading.",
        ),
        ("human", "File: {source}\n\nIncident:\n\n{text}"),
    ]
)


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

class GenerationError(Exception):
    """The model finished without producing an answer."""


def message_text(message: BaseMessage) -> str:
    """Flatten a LangChain message that is either a string or content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def strip_think(text: str) -> str | None:
    """Return the answer outside any thinking trace.

    A reply that never leaves an unclosed think block is not an answer.
    """
    if THINK_OPEN.search(text) and "</think>" not in text.lower():
        before = THINK_OPEN.split(text, maxsplit=1)[0].strip()
        return before or None
    cleaned = THINK_BLOCK.sub("", text)
    cleaned = re.sub(r"</?think\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or None


def is_none(text: str) -> bool:
    """True when a chunk summary has nothing worth combining."""
    return text.strip().upper() in {"NONE", "NONE."}


def read_file(path: Path) -> str:
    """Read text. A null byte means the file is binary."""
    data = path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"{path} looks like a binary file")
    return data.decode("utf-8", errors="replace")


def parse_moment(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp. A trailing Z is UTC."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def keep_stack_line(line: str) -> bool:
    """Keep the exception and the application frame. Drop runtime frames."""
    if not line.strip():
        return False
    if FRAMEWORK_FRAME.match(line) or OMITTED_FRAMES.match(line):
        return False
    if RUST_RUNTIME_FRAME.match(line) or RUST_RUNTIME_AT.match(line):
        return False
    if PYTHON_RUNTIME_FILE.match(line):
        return False
    if NODE_RUNTIME.search(line) or DOTNET_RUNTIME.match(line):
        return False
    return True


def parse_line(number: int, raw: str) -> LogLine:
    """Parse a Spring or service record.

    A timestamped level with no logger body is still a record, so the stack
    above it closes. Other lines stay attached to the event they follow.
    """
    match = RECORD_BODY.match(raw)
    if match is None:
        level_match = LOG_RECORD.match(raw)
        if level_match is None:
            return LogLine(number=number, raw=raw)
        return LogLine(
            number=number,
            raw=raw,
            level=level_match.group(1).upper(),
            is_record=True,
        )
    message = match.group("message")
    trace_match = TRACE_ID.search(match.group("mdc") or "") or TRACE_ID.search(message)
    request_match = REQUEST_ID.search(message)
    timestamp = match.group("ts")
    return LogLine(
        number=number,
        raw=raw,
        timestamp=timestamp,
        level=match.group("level").upper(),
        trace_id=trace_match.group(1) if trace_match else None,
        logger=match.group("logger"),
        message=message,
        is_record=True,
        moment=parse_moment(timestamp),
        request_id=request_match.group(1) if request_match else None,
    )


def short_logger(logger: str) -> str:
    """Use the class name for a Java logger and keep shorter names intact."""
    if "::" in logger or logger.count(".") < 3:
        return logger
    return logger.rsplit(".", 1)[-1]


def format_line(line: LogLine) -> str:
    """One source line as the model should see it, still carrying its number."""
    if line.is_record and line.timestamp and line.logger:
        return f"{line.number} {line.timestamp} {line.level} {short_logger(line.logger)}: {line.message}"
    return f"{line.number} {line.raw.strip()}"


def format_span(numbers: list[int]) -> str:
    """Compress line numbers into ranges. A dense cluster becomes one span."""
    numbers = sorted(set(numbers))
    if not numbers:
        return ""
    if len(numbers) > 4 and numbers[-1] - numbers[0] + 1 <= len(numbers) * 3:
        return f"{numbers[0]}-{numbers[-1]}"
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = number
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def collapse_key(line: LogLine) -> str:
    """Identity of a repeated message. The request path and id are ignored."""
    message = line.message or line.raw.strip()
    message = REPEAT_REQUEST.sub("", message)
    message = REPEAT_PATH.sub("", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{line.level}|{line.logger}|{message}"


def collapse_lines(lines: list[LogLine]) -> list[list[LogLine]]:
    """Group identical messages, preserving first-seen order."""
    groups: dict[str, list[LogLine]] = {}
    order: list[str] = []
    for line in lines:
        key = collapse_key(line)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(line)
    return [groups[key] for key in order]


def format_collapsed(lines: list[LogLine]) -> str:
    """Show one repeated message with its count, lines, and time span."""
    text = format_line(lines[0])
    if len(lines) == 1:
        return text
    numbers = [line.number for line in lines]
    first = lines[0].timestamp
    last = lines[-1].timestamp
    span = format_span(numbers)
    if first and last and first != last:
        return f"{text} ({len(lines)} times, lines {span}, {first} to {last})"
    return f"{text} ({len(lines)} times, lines {span})"


def recovery_lines(records: list[LogLine]) -> dict[str, int]:
    """Map each trace id to the last line that says its operation recovered."""
    last: dict[str, int] = {}
    for record in records:
        if record.trace_id and RECOVERED.search(record.raw):
            last[record.trace_id] = record.number
    return last


def recovered_warning(line: LogLine, recovered: dict[str, int]) -> bool:
    """A warning is noise when this trace recovers on a later line."""
    if line.level not in WARN_LEVELS:
        return False
    if RECOVERED.search(line.raw):
        return True
    if not line.trace_id:
        return False
    return line.number < recovered.get(line.trace_id, 0)


def words(value: str) -> set[str]:
    """Split a logger, path, or message into correlation tokens."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    found: set[str] = set()
    for part in re.split(r"[^A-Za-z0-9]+", spaced):
        word = part.lower()
        if len(word) >= 5 and word not in GENERIC_TOKENS:
            found.add(word)
    return found


def logger_package(logger: str) -> str:
    """Drop the class or function, leaving the package or module path."""
    if "::" in logger:
        return logger.rsplit("::", 1)[0]
    if "." in logger:
        return logger.rsplit(".", 1)[0]
    return ""


def line_tokens(line: LogLine) -> set[str]:
    return words(line.logger) | words(line.message or line.raw)


def event_tokens(event: Event) -> set[str]:
    """Names that identify this failure, without the shared package prefix."""
    found = line_tokens(event.seed)
    for line in event.stack:
        found |= words(line.raw)
    return found - words(logger_package(event.seed.logger))


def frame_words(raw: str) -> set[str]:
    """File and function names from a stack frame, not its package path."""
    found: set[str] = set()
    for match in re.finditer(r"([\w$]+)\.(?:java|go|py|rs|js|ts|cs)(?::\d+|\b)", raw):
        found |= words(match.group(1))
    for match in re.finditer(r"\(\*([\w]+)\)\.([\w]+)", raw):
        found |= words(match.group(1)) | words(match.group(2))
    for match in re.finditer(r"\bat (?:async )?([\w$.]+)", raw):
        parts = [part for part in match.group(1).split(".") if part]
        tail = parts[-2:] if len(parts) > 1 else parts
        found |= words(" ".join(tail))
    return found


def strong_tokens(event: Event) -> set[str]:
    """Component, config key, and exception names. Not the service or package."""
    found: set[str] = set()
    for match in re.finditer(r"\b(?:component|field|key)=(\S+)", event.seed.raw):
        found |= words(match.group(1))
    signature = exception_signature(event)
    if signature and signature != "panic":
        found |= words(signature.split(":", 1)[0].rsplit(".", 1)[-1])
    for line in event.stack:
        found |= frame_words(line.raw)
    return found - words(logger_package(event.seed.logger))


def is_routine(line: LogLine) -> bool:
    """SQL, cache, export, and successful checks are not crash evidence.

    A line that also reports a missing value or a failed setup is kept.
    """
    if DIAGNOSTIC.search(line.raw):
        return False
    if ROUTINE_CONTEXT.search(line.raw):
        return True
    return (
        re.search(
            r"(state=healthy|validation[ =]ok|exported_spans=|query=SELECT|"
            r"\back=true\b|\bstatus=200\b)",
            line.raw,
            re.IGNORECASE,
        )
        is not None
    )


def gap(earlier: datetime | None, later: datetime | None) -> timedelta | None:
    if earlier is None or later is None:
        return None
    return later - earlier


def near(earlier: LogLine, later: LogLine, window: timedelta) -> bool:
    """True when later follows earlier inside the time window.

    Lines whose timestamps do not parse fall back to a 40-line neighborhood.
    """
    delta = gap(earlier.moment, later.moment)
    if delta is None:
        return 0 < later.number - earlier.number <= 40
    return timedelta(0) <= delta <= window


def shares_call(record: LogLine, trace_id: str | None, request_id: str | None) -> bool:
    """True when the record is on this trace or this request."""
    if trace_id and record.trace_id == trace_id:
        return True
    return bool(request_id and record.request_id == request_id)


def attach_context(event: Event, records: list[LogLine]) -> None:
    """Keep non-routine lines from the same trace or request.

    Diagnostic and request or response lines outrank other same-call lines.
    At most MAX_CONTEXT_LINES are kept, preferring lines closest to the error.
    """
    if event.seed.level not in ERROR_LEVELS:
        return
    trace_id = event.seed.trace_id
    request_id = event.seed.request_id
    if not trace_id and not request_id:
        return
    occupied = {line.number for line in event.lines()}
    ranked: list[tuple[int, int, LogLine]] = []
    for record in records:
        if record.number in occupied or not shares_call(record, trace_id, request_id):
            continue
        if not is_routine(record) and (
            near(record, event.seed, LOOKBACK) or near(event.seed, record, LOOKBACK)
        ):
            # Prefer a diagnostic or request line, then the line closest to the error.
            rank = 2 if DIAGNOSTIC.search(record.raw) or CONTEXT_SIGNAL.search(record.raw) else 1
            ranked.append((rank, -abs(record.number - event.seed.number), record))
    ranked.sort(reverse=True)
    chosen = [item[2] for item in ranked[:MAX_CONTEXT_LINES]]
    chosen.sort(key=lambda line: line.number)
    event.context.extend(chosen)
    event.confirmed.update(line.number for line in chosen)


def line_pid(line: LogLine) -> str | None:
    match = PID_VALUE.search(line.raw)
    return match.group(1) if match else None


def event_pids(event: Event) -> set[str]:
    found = {pid for pid in (line_pid(event.seed),) if pid}
    for line in event.stack:
        pid = line_pid(line)
        if pid:
            found.add(pid)
    return found


def _specific_ids(line: LogLine) -> set[str]:
    """Ids that can tie a background failure to earlier setup lines."""
    found: set[str] = set()
    for pattern in (JOB_ID, KEY_ID, FIELD_ID, COMPONENT_ID, CALLBACK_ID):
        found.update(pattern.findall(line.raw))
    for left, right in DIFF_ARROW.findall(line.raw):
        found.add(left)
        found.add(right)
    return {value for value in found if len(value) >= 4}


def identifier_values(event: Event) -> set[str]:
    found = _specific_ids(event.seed)
    for line in [*event.stack, *event.context]:
        found |= _specific_ids(line)
    return found


def revision_values(lines: list[LogLine]) -> set[str]:
    found: set[str] = set()
    for line in lines:
        found.update(REVISION_ID.findall(line.raw))
    return found


def _prelude_noise(record: LogLine) -> bool:
    if record.level in ERROR_LEVELS or is_routine(record):
        return True
    return record.level in WARN_LEVELS and NOISY_WARNING.search(record.raw) is not None


def _take_context(
    ranked: list[tuple[int, int, LogLine, str]],
    room: int,
) -> list[tuple[LogLine, str]]:
    """Keep the highest tier, then the lines closest to the failure."""
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked: list[tuple[LogLine, str]] = []
    seen: set[int] = set()
    for _tier, _closeness, line, relation in ranked:
        if line.number in seen:
            continue
        seen.add(line.number)
        picked.append((line, relation))
        if len(picked) >= room:
            break
    return picked


def _remember(event: Event, picked: list[tuple[LogLine, str]]) -> None:
    for line, relation in picked:
        event.context.append(line)
        if relation == "possible":
            event.possible.add(line.number)
        else:
            event.confirmed.add(line.number)


def attach_prelude(
    event: Event,
    records: list[LogLine],
    *,
    lookback: timedelta = LOOKBACK,
    max_context: int = MAX_CONTEXT_LINES,
) -> None:
    """Keep config changes, failed setup, and nil state before a crash.

    Same-trace lines are handled by attach_context. A line is kept when it
    shares a component token and is a state change or the immediate step
    into the failure, or when it shares a job, key, field, or callback id
    learned from that step. Ordinary debug stays out.
    """
    if not is_serious(event):
        return
    tokens = strong_tokens(event)
    anchors = _specific_ids(event.seed)
    for line in event.stack:
        anchors |= _specific_ids(line)
    occupied = {line.number for line in event.lines()}
    ranked: list[tuple[int, int, LogLine, str]] = []
    for record in records:
        if record.number >= event.seed.number or record.number in occupied:
            continue
        if not near(record, event.seed, lookback) or _prelude_noise(record):
            continue
        shared = bool(tokens & line_tokens(record))
        signal = DIAGNOSTIC.search(record.raw) is not None
        delta = gap(record.moment, event.seed.moment)
        immediate = delta is not None and delta <= IMMEDIATE
        closeness = -abs(record.number - event.seed.number)
        if shared and (signal or immediate):
            relation = "confirmed" if signal else "possible"
            ranked.append((4 if signal else 3, closeness, record, relation))
            continue
        if anchors and (_specific_ids(record) & anchors):
            ranked.append((3, closeness, record, "confirmed"))
    room = max_context - len(event.context)
    if room > 0:
        _remember(event, _take_context(ranked, room))
    _attach_linked_setup(event, records, lookback=lookback, max_context=max_context)
    event.context.sort(key=lambda line: line.number)


def _attach_linked_setup(
    event: Event,
    records: list[LogLine],
    *,
    lookback: timedelta,
    max_context: int,
) -> None:
    """Pull earlier setup lines that share an id found on the lines already kept."""
    room = max_context - len(event.context)
    if room <= 0:
        return
    anchors = identifier_values(event)
    revisions = revision_values([event.seed, *event.stack, *event.context])
    if not anchors and not revisions:
        return
    occupied = {line.number for line in event.lines()}
    ranked: list[tuple[int, int, LogLine, str]] = []
    for record in records:
        if record.number >= event.seed.number or record.number in occupied:
            continue
        if not near(record, event.seed, lookback) or _prelude_noise(record):
            continue
        if not DIAGNOSTIC.search(record.raw):
            continue
        closeness = -abs(record.number - event.seed.number)
        if anchors and (_specific_ids(record) & anchors):
            ranked.append((3, closeness, record, "confirmed"))
            continue
        shared_revision = revisions and (set(REVISION_ID.findall(record.raw)) & revisions)
        if shared_revision:
            ranked.append((2, closeness, record, "confirmed"))
    _remember(event, _take_context(ranked, room))


def is_serious(event: Event) -> bool:
    """An exception or panic, not the process-exit line that follows one."""
    if consequence_kind(event) is not None:
        return False
    if exception_signature(event):
        return True
    return event.seed.level in {"FATAL", "CRITICAL"}


def absorb_prelude_warnings(events: list[Event]) -> list[Event]:
    """Drop a warning once its line is evidence on a following crash."""
    claimed = {
        line.number
        for event in events
        if event.seed.level in ERROR_LEVELS
        for line in event.context
        if line.level in WARN_LEVELS and line.number < event.seed.number
    }
    return [event for event in events if event.seed.number not in claimed]


def extract_events(
    text: str,
    *,
    lookback: timedelta = LOOKBACK,
    max_context: int = MAX_CONTEXT_LINES,
) -> list[Event] | None:
    """Group a timestamped log into numbered events.

    Returns None when the file is not a timestamped log.
    `lookback` bounds how far setup lines may precede a failure.
    `max_context` bounds how many of those lines stay with it.
    """
    raw_lines = text.splitlines()
    if sum(1 for line in raw_lines if LOG_RECORD.match(line)) < 5:
        return None

    parsed = [parse_line(number, raw) for number, raw in enumerate(raw_lines, start=1)]
    recovered = recovery_lines(parsed)
    events: list[Event] = []
    current: Event | None = None
    skip_python_body = False
    for line in parsed:
        if not line.is_record:
            # Continuation of the open event: exception text and frames.
            # A dropped Python runtime frame takes the source line under it too.
            if current is None or not line.raw.strip():
                skip_python_body = False
                continue
            if skip_python_body:
                skip_python_body = False
                continue
            if PYTHON_RUNTIME_FILE.match(line.raw):
                skip_python_body = True
                continue
            if keep_stack_line(line.raw):
                current.stack.append(line)
            continue

        # A new record closes the stack. Only errors and kept warnings open an event.
        current = None
        if line.level in WARN_LEVELS and (
            NOISY_WARNING.search(line.raw) or recovered_warning(line, recovered)
        ):
            continue
        if line.level in ERROR_LEVELS:
            current = Event(seed=line)
            events.append(current)
            continue
        if line.level not in WARN_LEVELS:
            continue

        # A warning on an error's trace belongs to that error, so its stack follows too.
        owner = next(
            (
                event
                for event in reversed(events)
                if event.seed.level in ERROR_LEVELS and event.seed.trace_id == line.trace_id
            ),
            None,
        )
        if owner is not None and line.trace_id:
            owner.members.append(line)
            owner.confirmed.add(line.number)
            current = owner
            continue
        current = Event(seed=line)
        events.append(current)

    records = [line for line in parsed if line.is_record]
    for event in events:
        attach_context(event, records)
        attach_prelude(event, records, lookback=lookback, max_context=max_context)
    events = absorb_prelude_warnings(events)
    grouped = group_cascades(events, records)
    attach_lifecycle(grouped, records)
    return collapse_repeat_events(grouped, records)


def exception_signature(event: Event) -> str | None:
    """The innermost exception text, or panic. Used to tell incidents apart."""
    texts = [event.seed.raw, *[line.raw for line in event.stack]]
    for text in reversed(texts):
        match = EXCEPTION_SIGNATURE.search(text.strip())
        if match:
            return re.sub(r"\s+", " ", match.group(1))
    if re.search(r"\bpanic(?:ked)?\b", event.seed.raw, re.IGNORECASE):
        return "panic"
    return None


def consequence_kind(event: Event) -> str | None:
    """Classify process death, a restart attempt, or a proxy failure.

    An application line that merely contains status=502 is not a cascade.
    A stack means this event is its own exception.
    """
    if event.stack:
        return None
    text = event.seed.message or event.seed.raw
    logger = event.seed.logger.lower()
    if PROCESS_DEATH.search(text):
        return "death"
    if PROCESS_RESTART.search(text) and ("systemd" in logger or ".service" in text):
        return "restart"
    proxy = "nginx" in logger or "upstream" in logger or "connect() failed" in text
    if proxy and UPSTREAM_FAILURE.search(text):
        return "upstream"
    return None


def service_started(line: LogLine) -> bool:
    """True when this line is a completed service start, not a start attempt."""
    if STARTED_SERVICE.search(line.raw):
        return True
    return "systemd" in line.logger.lower() and re.search(r"\bStarted\b", line.message) is not None


def started_between(earlier: LogLine, later: LogLine, records: list[LogLine]) -> bool:
    """True when a service finished starting between the two lines."""
    for record in records:
        if earlier.number < record.number < later.number and service_started(record):
            return True
    return False


def same_incident(owner: Event, event: Event, records: list[LogLine]) -> bool:
    """True when two exceptions are one failure, not the same text hours apart.

    Matching signatures are not enough. The later one must share the logger,
    agree on any trace id, sit inside the time window, and not follow a restart.
    """
    left = exception_signature(owner)
    right = exception_signature(event)
    if not left or left != right:
        return False
    if owner.seed.logger and event.seed.logger and owner.seed.logger != event.seed.logger:
        return False
    if (
        owner.seed.trace_id
        and event.seed.trace_id
        and owner.seed.trace_id != event.seed.trace_id
    ):
        return False
    if not near(owner.seed, event.seed, SIGNATURE_WINDOW):
        return False
    return not started_between(owner.seed, event.seed, records)


def unit_parts(unit: str) -> set[str]:
    stem = unit.removesuffix(".service").lower()
    return {part for part in re.split(r"[^a-z0-9]+", stem) if len(part) >= 5 and part not in GENERIC_TOKENS}


def mentions_unit(event: Event, unit: str) -> bool:
    """True when the exception names this systemd unit or its service."""
    stem = unit.removesuffix(".service").lower()
    if stem and stem in event.seed.logger.lower():
        return True
    parts = unit_parts(unit)
    if not parts:
        return False
    return bool(parts & (words(event.seed.logger) | event_tokens(event)))


def serious_in_window(event: Event, grouped: list[Event]) -> list[Event]:
    return [
        owner
        for owner in grouped
        if is_serious(owner) and near(owner.seed, event.seed, CASCADE_WINDOW)
    ]


def pids_agree(owner: Event, line: LogLine) -> bool:
    """A pid on both sides must be the same pid. A missing pid does not rule the line out."""
    pid = line_pid(line)
    known = event_pids(owner)
    if pid and known:
        return pid in known
    return True


def has_death(event: Event) -> bool:
    if consequence_kind(event) == "death":
        return True
    return any(lifecycle_kind(line) == "death" for line in event.consequences)


def cascade_owner(event: Event, grouped: list[Event]) -> Event | None:
    """Find the exception this process death or proxy failure belongs to.

    A unit name or a shared pid is a confirmed link. A proxy line joins the
    exception that already has that process death. Time by itself is not enough
    when more than one exception is in the window.
    """
    kind = consequence_kind(event)
    if kind is None:
        return None
    in_window = serious_in_window(event, grouped)
    if not in_window:
        return None
    if kind == "upstream":
        owners = [owner for owner in in_window if has_death(owner)]
        return owners[-1] if len(owners) == 1 else (in_window[0] if len(in_window) == 1 else None)
    pid = line_pid(event.seed)
    if pid:
        named = [owner for owner in in_window if pid in event_pids(owner)]
        if named:
            return named[-1]
    units = SERVICE_UNIT.findall(event.seed.raw)
    if units:
        named = [
            owner
            for owner in in_window
            if pids_agree(owner, event.seed) and any(mentions_unit(owner, unit) for unit in units)
        ]
        return named[-1] if named else None
    same_logger = [
        owner
        for owner in in_window
        if owner.seed.logger and owner.seed.logger == event.seed.logger and pids_agree(owner, event.seed)
    ]
    if len(same_logger) == 1:
        return same_logger[0]
    if len(in_window) == 1 and pids_agree(in_window[0], event.seed):
        return in_window[0]
    return None


def take_in(owner: Event, event: Event) -> None:
    """Fold a later copy of the same exception into the open incident."""
    owner.stack.append(event.seed)
    owner.stack.extend(event.stack)
    owner.members.extend(event.context)
    owner.members.extend(event.members)
    owner.consequences.extend(event.consequences)
    owner.confirmed.update(event.confirmed)
    owner.possible.update(event.possible)


def _claim(owner: Event, line: LogLine) -> None:
    owner.consequences.append(line)
    owner.confirmed.add(line.number)


def group_cascades(events: list[Event], records: list[LogLine]) -> list[Event]:
    """Attach process death and proxy failures to the exception that caused them."""
    grouped: list[Event] = []
    for event in events:
        if not grouped:
            grouped.append(event)
            continue
        # A second copy of this exception joins the incident. Process death and
        # proxy failures attach as consequences. Anything else starts its own event.
        if any(same_incident(owner, event, records) for owner in grouped):
            owner = next(item for item in reversed(grouped) if same_incident(item, event, records))
            take_in(owner, event)
            continue
        owner = cascade_owner(event, grouped)
        if (
            owner is not None
            and consequence_kind(event) == "upstream"
            and started_between(owner.seed, event.seed, records)
        ):
            owner = None
        if owner is not None:
            for line in event.lines():
                _claim(owner, line)
            owner.repeat_numbers.extend(event.repeat_numbers)
            continue
        grouped.append(event)
    return grouped


def lifecycle_kind(line: LogLine) -> str | None:
    """Classify a record as process death, restart, start, proxy failure, or recovery."""
    text = line.message or line.raw
    logger = line.logger.lower()
    if PROCESS_DEATH.search(text):
        return "death"
    if service_started(line):
        return "started"
    if PROCESS_RESTART.search(text) and ("systemd" in logger or ".service" in text):
        return "restart"
    proxy = "nginx" in logger or logger.startswith("upstream") or "upstream" in logger
    if proxy and UPSTREAM_FAILURE.search(text):
        return "upstream"
    if RECOVERY_LINE.search(text):
        return "recovery"
    return None


def _units_of(event: Event) -> set[str]:
    units = set(SERVICE_UNIT.findall(event.seed.raw))
    for line in event.lines():
        units.update(SERVICE_UNIT.findall(line.raw))
    return units


def _service_overlap(event: Event, line: LogLine) -> bool:
    units = SERVICE_UNIT.findall(line.raw)
    if units and any(mentions_unit(event, unit) for unit in units):
        return pids_agree(event, line)
    known = _units_of(event)
    if units and known & set(units):
        return pids_agree(event, line)
    blob = words(line.logger) | words(line.message or line.raw)
    return any(bool(unit_parts(unit) & blob) for unit in known)


def attach_lifecycle(events: list[Event], records: list[LogLine]) -> None:
    """Add supervisor, proxy, restart, and recovery lines that are not their own events.

    Upstream failures attach only after a confirmed process death and before
    the service is back. A later 502 from the application stays independent.
    """
    occupied = {line.number for event in events for line in event.lines()}
    for event in events:
        if not is_serious(event):
            continue
        started = False
        recovery_left = MAX_RECOVERY_LINES
        for record in records:
            if record.number <= event.seed.number:
                continue
            if not near(event.seed, record, CASCADE_WINDOW):
                if (
                    event.seed.moment
                    and record.moment
                    and record.moment > event.seed.moment + CASCADE_WINDOW
                ):
                    break
                continue
            kind = lifecycle_kind(record)
            if record.number in occupied:
                if kind == "started":
                    started = True
                continue
            if kind is None or not pids_agree(event, record):
                continue
            if kind == "upstream":
                if started or not has_death(event):
                    continue
            elif kind == "recovery":
                if not started or recovery_left <= 0 or not _service_overlap(event, record):
                    continue
                recovery_left -= 1
            elif not _service_overlap(event, record):
                continue
            if kind == "started":
                started = True
            _claim(event, record)
            occupied.add(record.number)
        event.consequences.sort(key=lambda line: line.number)


def collapse_repeat_events(events: list[Event], records: list[LogLine]) -> list[Event]:
    """Fold repeats inside one incident, and keep the first and last times.

    The same warning after a completed service start is a new incident.
    """
    result: list[Event] = []
    index_by_key: dict[str, int] = {}
    for event in events:
        # An event that already has a stack or other evidence stays intact.
        if event.stack or event.context or event.members or event.consequences:
            result.append(event)
            continue
        key = collapse_key(event.seed)
        existing = index_by_key.get(key)
        # A completed service start means this copy belongs to the new process.
        if existing is not None and started_between(result[existing].seed, event.seed, records):
            existing = None
        if existing is None:
            event.repeat_numbers = [event.seed.number]
            event.repeat_lines = [event.seed]
            index_by_key[key] = len(result)
            result.append(event)
            continue
        result[existing].repeat_numbers.append(event.seed.number)
        result[existing].repeat_numbers.extend(event.repeat_numbers)
        result[existing].repeat_lines.append(event.seed)
        result[existing].repeat_lines.extend(event.repeat_lines)
    return result


def event_kind(event: Event) -> str:
    """Label the report prompt uses to place an event in a section.

    An error with an exception and a stack, same-trace warnings, or fallout
    is the primary error. A bare error is a separate error. A warning stays
    a warning. Prelude lines alone do not make an error primary.
    """
    if event.seed.level not in ERROR_LEVELS:
        return "warning"
    if exception_signature(event) and (event.stack or event.members or event.consequences):
        return "primary error"
    return "separate error"


def format_events(events: list[Event]) -> str:
    """Render every extracted event. The model receives format_model_input instead.

    A primary error is labeled Failure and Same request. Fallout stays under
    Failure. Prelude and same-request lines stay under Same request. Other
    events are labeled Detail, or emitted as a collapsed warning.
    """
    blocks: list[str] = []
    for event in events:
        kind = event_kind(event)
        numbers = [line.number for line in event.lines()]
        numbers.extend(event.repeat_numbers)
        header = f"Event E{event.seed.number} ({kind}) lines {format_span(numbers)}"
        if kind == "warning":
            body = "\n".join(format_collapsed(group) for group in collapse_lines(event.lines()))
            if (
                len(event.repeat_lines) > 1
                and not event.members
                and not event.stack
                and not event.consequences
            ):
                body = format_collapsed(event.repeat_lines)
            blocks.append(f"{header}\n{body}")
            continue
        failure_lines = collapse_lines([event.seed, *event.stack, *event.consequences])
        related_lines = collapse_lines(
            sorted([*event.context, *event.members], key=lambda line: line.number)
        )
        if kind == "primary error":
            failure_heading = "Failure (include the exception and these stack frames in Main failure):"
            related_heading = "Same request:"
        else:
            failure_heading = "Detail:"
            related_heading = "Detail:"
        parts = [
            header,
            failure_heading,
            "\n".join(format_collapsed(group) for group in failure_lines),
        ]
        if related_lines:
            parts.extend(
                [
                    related_heading,
                    "\n".join(format_collapsed(group) for group in related_lines),
                ]
            )
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def event_id(event: Event) -> str:
    """Stable id taken from the source line that opened the event."""
    return f"E{event.seed.number}"


def select_primary(events: list[Event]) -> Event | None:
    """The incident to explain. A terminating failure outranks an earlier error."""
    serious = [event for event in events if is_serious(event)]
    if not serious:
        return None

    def rank(event: Event) -> tuple[int, int, int, int]:
        return (
            1 if event.consequences else 0,
            1 if event.seed.level in {"FATAL", "CRITICAL"} else 0,
            len(event.stack),
            event.seed.number,
        )

    return max(serious, key=rank)


def independent_events(events: list[Event], primary: Event | None) -> list[Event]:
    if primary is None:
        return list(events)
    return [event for event in events if event is not primary]


def _context_label(event: Event, line: LogLine) -> str:
    if line.number in event.possible and line.number not in event.confirmed:
        return "possible"
    return "confirmed"


def split_incident(event: Event) -> tuple[list[LogLine], list[LogLine]]:
    """Preceding context, then associated lines that follow the failure."""
    surrounding = sorted([*event.context, *event.members], key=lambda line: line.number)
    before = [line for line in surrounding if line.number < event.seed.number]
    after = [line for line in surrounding if line.number > event.seed.number]
    related = sorted([*after, *event.consequences], key=lambda line: line.number)
    return before, related


def format_model_input(event: Event) -> str:
    """Incident text for the model. Independent events are left out."""
    before, related = split_incident(event)
    parts = [
        f"Event {event_id(event)}",
        "Failure:",
        "\n".join(format_line(line) for line in [event.seed, *event.stack]),
        "Context:",
        "\n".join(f"{_context_label(event, line)} {format_line(line)}" for line in before) or "None",
        "Related:",
        "\n".join(format_collapsed(group) for group in collapse_lines(related)) or "None",
    ]
    return "\n".join(parts)


def format_warning_row(event: Event) -> str:
    """One independent event, identified by its source line."""
    if (
        len(event.repeat_lines) > 1
        and not event.stack
        and not event.members
        and not event.consequences
    ):
        return f"{event_id(event)} {format_collapsed(event.repeat_lines)}"
    parts = [format_line(event.seed)]
    signature = exception_signature(event)
    if signature and signature not in parts[0]:
        parts.append(signature)
    frame = next(
        (
            line
            for line in event.stack
            if re.search(r"\.(?:java|py|go|rs|js|cs)(?::\d+|:line \d+)", line.raw)
        ),
        None,
    )
    if frame is not None:
        parts.append(format_line(frame))
    return f"{event_id(event)} " + " | ".join(parts)


def format_other_warnings(events: list[Event]) -> str:
    if not events:
        return "Other warnings\nNone"
    return "Other warnings\n" + "\n".join(format_warning_row(event) for event in events)


def fallback_narrative(event: Event | None) -> str:
    """Main failure and Related events taken straight from the extracted lines."""
    if event is None:
        return "Main failure\nNone\n\nRelated events\nNone"
    before, related = split_incident(event)
    main = ["Main failure", "\n".join(format_line(line) for line in [event.seed, *event.stack])]
    if before:
        main.append("Preceding context:")
        main.append("\n".join(format_line(line) for line in before))
    related_body = (
        "\n".join(format_collapsed(group) for group in collapse_lines(related)) if related else "None"
    )
    return "\n".join(main) + "\n\nRelated events\n" + related_body


def compose_report(narrative: str, independent: list[Event]) -> str:
    """Keep the model's two sections and append warnings Python already listed."""
    text = narrative.strip()
    text = re.split(r"\n(?=\**\s*Other warnings\b)", text, maxsplit=1)[0].rstrip()
    return f"{text}\n\n{format_other_warnings(independent)}"


def fallback_report(primary: Event | None, independent: list[Event]) -> str:
    return compose_report(fallback_narrative(primary), independent)


def dump_events(events: list[Event]) -> str:
    """Show the model input and the independent events reported by Python."""
    primary = select_primary(events)
    independent = independent_events(events, primary)
    model_input = format_model_input(primary) if primary is not None else "Primary incident\nNone"
    return f"# Sent to the model\n{model_input}\n\n# Reported by Python\n{format_other_warnings(independent)}"


def collect_summary(chain, payload: dict[str, str]) -> str | None:
    """Stream one reply and drop a thinking trace. None means there was no answer."""
    pieces: list[str] = []
    for chunk in chain.stream(payload):
        text = message_text(chunk)
        if text:
            pieces.append(text)
    return strip_think("".join(pieces))


def complete(chain, payload: dict[str, str]) -> str:
    """Run the chain, retrying once when the model only emits a think block."""
    for attempt in range(2):
        answer = collect_summary(chain, payload)
        if answer:
            return answer
        if attempt == 0:
            print(
                "The model returned only a thinking trace. Retrying...",
                file=sys.stderr,
            )
    raise GenerationError("The model did not produce a summary.")


def summarize_chunks(
    path: Path,
    chunks: list[str],
    *,
    piece_prompt: ChatPromptTemplate,
    combine_prompt: ChatPromptTemplate,
    llm: ChatOllama,
) -> str:
    """Summarize each chunk, drop NONE parts, and combine what remains."""
    piece_chain = piece_prompt | llm
    if len(chunks) == 1:
        return complete(piece_chain, {"source": path.name, "text": chunks[0]})

    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Summarizing part {index} of {len(chunks)}...",
            file=sys.stderr,
        )
        partial = complete(piece_chain, {"source": path.name, "text": chunk})
        if not is_none(partial):
            partials.append(partial)

    if not partials:
        return "NONE"
    if len(partials) == 1:
        return partials[0]

    print("Combining partial summaries...", file=sys.stderr)
    combine_chain = combine_prompt | llm
    return complete(
        combine_chain,
        {"source": path.name, "text": "\n\n".join(partials)},
    )


def build_llm(*, reasoning: bool = False, num_predict: int = 1024) -> ChatOllama:
    """Build the local chat model.

    The log report turns reasoning on so a thinking trace stays out of the
    printed answer. The plain summary leaves it off.
    """
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.1,
        reasoning=reasoning,
        num_ctx=8192,
        num_predict=num_predict,
    )


def interpret_events(path: Path, event: Event) -> str:
    print(
        f"Interpreting event {event_id(event)} with {MODEL_NAME}...",
        file=sys.stderr,
    )
    payload = {"source": path.name, "text": format_model_input(event)}
    budgets = (2048, 4096)
    for attempt, budget in enumerate(budgets):
        if attempt:
            print(
                "The model returned only a thinking trace. "
                "Retrying with a larger token budget...",
                file=sys.stderr,
            )
        chain = REPORT_PROMPT | build_llm(reasoning=True, num_predict=budget)
        answer = collect_summary(chain, payload)
        if answer:
            return answer
    raise GenerationError("The model did not produce a summary.")


def summarize(path: Path, text: str) -> str:
    """Report a timestamped log, or summarize other text.

    The log path extracts events here and asks the model only to narrate them.
    Fewer than five timestamped records is treated as ordinary text.
    """
    events = extract_events(text)
    if events is not None:
        if not events:
            return "No relevant errors or warnings were found."
        primary = select_primary(events)
        independent = independent_events(events, primary)
        print(
            f"Extracted {len(events)} events from {len(text.splitlines())} lines.",
            file=sys.stderr,
        )
        if primary is None:
            return fallback_report(None, independent)
        try:
            narrative = interpret_events(path, primary)
        except GenerationError:
            print(
                "The model did not produce a summary. Using the extracted events.",
                file=sys.stderr,
            )
            narrative = fallback_narrative(primary)
        return compose_report(narrative, independent)

    print(f"Summarizing {path} with {MODEL_NAME}...", file=sys.stderr)
    llm = build_llm()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return summarize_chunks(
        path,
        splitter.split_text(text),
        piece_prompt=SUMMARIZE_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        llm=llm,
    )


def main(argv: list[str] | None = None) -> int:
    """Read one file and print its summary. Returns a process status code."""
    parser = argparse.ArgumentParser(
        description="Digest a log, or summarize another text file with a local Ollama model.",
    )
    parser.add_argument("file", type=Path, help="Path to the file to summarize")
    parser.add_argument(
        "--dump-events",
        action="store_true",
        help="Print the extracted incident groups and exit without calling the model.",
    )
    args = parser.parse_args(argv)

    path: Path = args.file
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    try:
        text = read_file(path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not read {path}: {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print("The file is empty.")
        return 0

    if args.dump_events:
        events = extract_events(text)
        if events is None:
            print("Not a timestamped log.")
            return 0
        print(dump_events(events))
        return 0

    try:
        summary = summarize(path, text)
    except httpx.ConnectError:
        print(
            "Could not connect to Ollama. Start it and try again.",
            file=sys.stderr,
        )
        return 1
    except ResponseError as exc:
        print(f"Ollama request failed: {exc}", file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
