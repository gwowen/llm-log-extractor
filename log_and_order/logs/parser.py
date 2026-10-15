"""Parse raw lines into LogLine records. Line numbers stay 1-based."""
from datetime import datetime
from ..models import LogLine
from .patterns import (
    DOTNET_RUNTIME,
    FRAMEWORK_FRAME,
    LOG_RECORD,
    NODE_RUNTIME,
    OMITTED_FRAMES,
    PYTHON_RUNTIME_FILE,
    RECORD_BODY,
    REQUEST_ID,
    RUST_RUNTIME_AT,
    RUST_RUNTIME_FRAME,
    TRACE_ID,
)

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
        # The source line under this frame is dropped by extract_events, not here.
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
