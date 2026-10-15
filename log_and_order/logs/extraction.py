"""Turn a timestamped log into events."""
import re
from datetime import timedelta
from ..models import Event, LogLine
from ..config import (
    LOOKBACK,
    MAX_CONTEXT_LINES,
)
from .patterns import (
    ERROR_LEVELS,
    LOG_RECORD,
    NOISY_WARNING,
    PYTHON_RUNTIME_FILE,
    RECOVERED,
    REPEAT_PATH,
    REPEAT_REQUEST,
    WARN_LEVELS,
)
from .parser import (
    keep_stack_line,
    parse_line,
)
from .correlation import (
    attach_context,
    attach_lifecycle,
    attach_prelude,
    group_cascades,
    started_between,
)


def collapse_key(line: LogLine) -> str:
    """Identity of a repeated message. The request path and id are ignored."""
    message = line.message or line.raw.strip()
    message = REPEAT_REQUEST.sub("", message)
    message = REPEAT_PATH.sub("", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{line.level}|{line.logger}|{message}"


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
    # Five timestamped records is the line between a log and ordinary text.
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
