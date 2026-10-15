"""Summarize a text file with a local Ollama model via LangChain.

A timestamped log is reduced to events in this module, then narrated by the
model. Any other text file is summarized directly. Progress goes to stderr
and the report goes to stdout.
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
LOOKBACK = timedelta(seconds=30)
# Same-component lines in this gap are the step that entered the failure.
IMMEDIATE = timedelta(seconds=3)
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
STARTED_SERVICE = re.compile(r"\bStarted [\w.@+-]+\.service\b")
SERVICE_UNIT = re.compile(r"\b([\w.@+-]+\.service)\b")
REQUEST_ID = re.compile(r"\brequest[_-]?[Ii]d=([A-Za-z0-9._:-]+)")
# Configuration changes, failed setup, and nil or missing state.
DIAGNOSTIC = re.compile(
    r"(change detected|initialization failed|invalid endpoint|"
    r"<nil>|client_present=false|tenant_id(?:_present)?=(?:false|None)|"
    r"soft_fail=true|swap=scheduled|"
    r"marking existing transaction as rollback-only)",
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
    "status",
    "trace",
    "true",
    "uncaught",
    "warn",
    "warning",
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

# Narrates events extract_events already chose. The three headings are fixed.
REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write a short incident report from extracted log events. "
            "The event labeled primary error is the main failure. "
            "Describe its Failure lines, including the exception and the application frames. "
            "Its Same request lines are the related events. "
            "Events labeled separate error or warning are other warnings. "
            "Do not describe those as related to the main failure, and do not suggest that they caused it. "
            "Write exactly these headings, in this order:\n"
            "Main failure\n"
            "Related events\n"
            "Other warnings\n"
            "Use plain language. After each statement, cite the source line numbers shown, "
            "like (line 616) or (lines 649-651). "
            "Cite only line numbers that appear in the events. "
            "Use the timestamp on the ERROR line for when the failure happened. "
            "Do not invent line numbers. Do not suggest a fix. "
            "Write the three sections once. Do not add notes, drafts, or a second copy. "
            "If a section has nothing to say, write None under that heading.",
        ),
        ("human", "File: {source}\n\nEvents:\n\n{text}"),
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
    for match in re.finditer(r"([\w$]+)\.(?:java|go|py|rs)(?::\d+|\b)", raw):
        found |= words(match.group(1))
    for match in re.finditer(r"\(\*([\w]+)\)\.([\w]+)", raw):
        found |= words(match.group(1)) | words(match.group(2))
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
    return found


def is_routine(line: LogLine) -> bool:
    """SQL, cache, export, and successful checks are not crash evidence."""
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


def attach_prelude(event: Event, records: list[LogLine]) -> None:
    """Keep config changes, failed setup, and nil state before a crash.

    Same-trace lines are handled by attach_context. This pass keeps other
    records that name the same component, and skips ordinary debug chatter.
    """
    if not is_serious(event):
        return
    tokens = strong_tokens(event)
    if not tokens:
        return
    occupied = {line.number for line in event.lines()}
    chosen: list[tuple[int, int, LogLine]] = []
    for record in records:
        if record.number >= event.seed.number or record.number in occupied:
            continue
        if record.level in ERROR_LEVELS:
            continue
        if not near(record, event.seed, LOOKBACK):
            continue
        if is_routine(record):
            continue
        if record.level in WARN_LEVELS and NOISY_WARNING.search(record.raw):
            continue
        if not (tokens & line_tokens(record)):
            continue
        diagnostic = DIAGNOSTIC.search(record.raw) is not None
        delta = gap(record.moment, event.seed.moment)
        immediate = delta is not None and delta <= IMMEDIATE
        if not diagnostic and not immediate:
            continue
        chosen.append((1 if diagnostic else 0, record.number, record))
    room = MAX_CONTEXT_LINES - len(event.context)
    if room <= 0:
        return
    chosen.sort(reverse=True)
    event.context.extend(item[2] for item in chosen[:room])
    event.context.sort(key=lambda line: line.number)


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


def extract_events(text: str) -> list[Event] | None:
    """Group a timestamped log into numbered events.

    Returns None when the file is not a timestamped log.
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
            current = owner
            continue
        current = Event(seed=line)
        events.append(current)

    records = [line for line in parsed if line.is_record]
    for event in events:
        attach_context(event, records)
        attach_prelude(event, records)
    events = absorb_prelude_warnings(events)
    return collapse_repeat_events(group_cascades(events, records), records)


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


def started_between(earlier: LogLine, later: LogLine, records: list[LogLine]) -> bool:
    """True when a service finished starting between the two lines."""
    for record in records:
        if earlier.number < record.number < later.number and STARTED_SERVICE.search(record.raw):
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


def mentions_unit(event: Event, unit: str) -> bool:
    """True when the exception names this systemd unit or its service."""
    stem = unit.removesuffix(".service").lower()
    if stem in event.seed.logger.lower():
        return True
    return bool(words(stem) & event_tokens(event))


def cascade_owner(event: Event, grouped: list[Event]) -> Event | None:
    """Find the exception this process death or proxy failure belongs to."""
    if consequence_kind(event) is None:
        return None
    in_window = [
        owner
        for owner in grouped
        if exception_signature(owner) and near(owner.seed, event.seed, CASCADE_WINDOW)
    ]
    if not in_window:
        return None
    units = SERVICE_UNIT.findall(event.seed.raw)
    if units:
        named = [owner for owner in in_window if any(mentions_unit(owner, unit) for unit in units)]
        return named[-1] if named else None
    return in_window[-1]


def take_in(owner: Event, event: Event) -> None:
    """Fold a later copy of the same exception into the open incident."""
    owner.stack.append(event.seed)
    owner.stack.extend(event.stack)
    owner.members.extend(event.context)
    owner.members.extend(event.members)
    owner.consequences.extend(event.consequences)


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
        if owner is not None:
            owner.consequences.extend(event.lines())
            owner.repeat_numbers.extend(event.repeat_numbers)
            continue
        grouped.append(event)
    return grouped


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
    """Render the blocks the report prompt is allowed to narrate.

    A primary error is labeled Failure and Same request. Those headings are
    what the prompt maps onto Main failure and Related events. Fallout stays
    under Failure. Prelude and same-request lines stay under Same request.
    Other events are labeled Detail, or emitted as a collapsed warning.
    """
    blocks: list[str] = []
    for index, event in enumerate(events, start=1):
        kind = event_kind(event)
        numbers = [line.number for line in event.lines()]
        numbers.extend(event.repeat_numbers)
        header = f"Event {index} ({kind}) lines {format_span(numbers)}"
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


def interpret_events(path: Path, events: list[Event]) -> str:
    print(
        f"Interpreting {len(events)} events with {MODEL_NAME}...",
        file=sys.stderr,
    )
    payload = {"source": path.name, "text": format_events(events)}
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
        print(
            f"Extracted {len(events)} events from {len(text.splitlines())} lines.",
            file=sys.stderr,
        )
        return interpret_events(path, events)

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
