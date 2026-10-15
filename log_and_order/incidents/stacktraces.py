"""Application source locations from stack traces."""
import re
from ..models import (
    Event,
    ReportLocation,
)
from ..logs.patterns import (
    DOTNET_RUNTIME,
    FRAMEWORK_FRAME,
    LOG_RECORD,
    NODE_RUNTIME,
    OMITTED_FRAMES,
    PYTHON_RUNTIME_FILE,
    RUST_RUNTIME_AT,
    RUST_RUNTIME_FRAME,
)
from ..logs.parser import (
    keep_stack_line,
)

# Application frames. Runtime frames stay out of the source-location list.
JAVA_FRAME = re.compile(
    r"^\s*at\s+(?P<symbol>[\w.$]+)\((?P<file>[\w$]+\.java):(?P<line>\d+)\)"
)
CSHARP_FRAME = re.compile(
    r"^\s*at\s+(?P<symbol>[\w.]+)\([^)]*\)\s+in\s+(?P<path>.+?):line\s+(?P<line>\d+)"
)
NODE_FRAME = re.compile(
    r"^\s*at\s+(?:async\s+)?(?P<symbol>.+?)\s+\((?P<path>[^()]+?):(?P<line>\d+):(?P<col>\d+)\)"
)
PYTHON_FRAME = re.compile(
    r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<symbol>.+?)\s*$'
)
GO_RECEIVER = re.compile(r"^(?:created by\s+)?(?P<symbol>\S+\(\*?\w+\)\.\w+)")
GO_PLAIN = re.compile(r"^(?:created by\s+)?(?P<symbol>[A-Za-z0-9_./]+\.[A-Za-z0-9_.]+)")
GO_PATH = re.compile(r"^\s+(?P<path>\S+\.go):(?P<line>\d+)\b")
TYPED_EXCEPTION = re.compile(
    r"(?P<type>(?:[\w$]+\.)*[\w$]+(?:Error|Exception))\s*:\s*(?P<message>.*)$"
)
RUST_SYMBOL = re.compile(r"^\s*\d+:\s+(?P<symbol>\S+)\s*$")
RUST_AT = re.compile(r"^\s*at\s+(?P<path>\S+\.rs):(?P<line>\d+)(?::(?P<col>\d+))?")
PANIC_AT = re.compile(
    r"(?:panicked at|called on None at)\s+(?P<path>\S+?):(?P<line>\d+)(?::(?P<col>\d+))?"
)

def stack_region(text: str, event: Event) -> list[tuple[int, str]]:
    """Failure line, the following stack, and any later stack folded into the incident."""
    raw_lines = text.splitlines()
    found: dict[int, str] = {event.seed.number: event.seed.raw}
    for number in range(event.seed.number + 1, len(raw_lines) + 1):
        raw = raw_lines[number - 1]
        if LOG_RECORD.match(raw):
            break
        found[number] = raw
    for line in event.stack:
        found.setdefault(line.number, line.raw)
    return sorted(found.items())


def _go_symbol(raw: str) -> str | None:
    """Function name from a Go frame. A path line is not a symbol."""
    text = raw.strip()
    if not text or ".go:" in text:
        return None
    if text.startswith("created by "):
        text = text.removeprefix("created by ").split(" in goroutine", 1)[0].strip()
        if "/" not in text or "." not in text:
            return None
        return text
    receiver = GO_RECEIVER.match(text)
    if receiver and "/" in receiver.group("symbol"):
        return receiver.group("symbol")
    plain = GO_PLAIN.match(text)
    if plain and "/" in plain.group("symbol") and not plain.group("symbol").endswith(".go"):
        return plain.group("symbol")
    return None


def _norm_path(path: str) -> str:
    """Drop a leading ./ so two writings of the same file compare equal."""
    return path.removeprefix("./")


def _same_place(location: ReportLocation, path: str, source_line: int) -> bool:
    """True when a frame and a panic location name the same file and source line."""
    return _norm_path(location.path) == _norm_path(path) and location.source_line == source_line


def _exception_boundary(raw: str) -> bool:
    """True when this stack line starts a new exception, including Caused by."""
    text = raw.strip()
    folded = text.lower()
    if folded.startswith("caused by"):
        return True
    if folded.startswith("panic:"):
        return True
    if "panicked at" in folded:
        return True
    return TYPED_EXCEPTION.search(text) is not None


def _dropped_frame(raw: str) -> bool:
    """True when a frame was removed above the application frames we keep."""
    if not raw.strip():
        return False
    if FRAMEWORK_FRAME.match(raw) or OMITTED_FRAMES.match(raw):
        return True
    if RUST_RUNTIME_FRAME.match(raw) or RUST_RUNTIME_AT.match(raw):
        return True
    if PYTHON_RUNTIME_FILE.match(raw):
        return True
    return NODE_RUNTIME.search(raw) is not None or DOTNET_RUNTIME.match(raw) is not None


def application_locations(text: str, event: Event) -> list[ReportLocation]:
    """Application frames in stack order. Runtime frames are skipped.

    Java, C#, JavaScript, and Go list the throwing frame first. Python lists
    the raising frame last, so that frame is the innermost one kept. Rust uses
    the panic location when the log states one. A first kept frame is called
    the throw site only when no framework or runtime frame was dropped above
    it, or when the log names the panic location.
    """
    frames: list[ReportLocation] = []
    seen: set[tuple[str, int | None, str]] = set()
    pending_go: tuple[int, str] | None = None
    pending_rust: tuple[int, str] | None = None
    skip_python_body = False
    dropped_above = False
    first_covered = False
    panic_at: tuple[str, int, int | None, int] | None = None

    def add(role: str, path: str, symbol: str, source_line: int, column: int | None, log_line: int) -> None:
        """Record one application frame. The same file, line, and symbol is kept once."""
        nonlocal first_covered
        key = (_norm_path(path), source_line, symbol)
        if key in seen:
            return
        seen.add(key)
        if not frames:
            first_covered = dropped_above
        frames.append(
            ReportLocation(
                role=role,
                order=len(frames),
                path=path,
                symbol=symbol,
                source_line=source_line,
                source_column=column,
                log_line=log_line,
            )
        )

    for number, raw in stack_region(text, event):
        panic = PANIC_AT.search(raw)
        if panic and panic_at is None:
            column = int(panic.group("col")) if panic.group("col") else None
            panic_at = (panic.group("path"), int(panic.group("line")), column, number)
        if _exception_boundary(raw):
            dropped_above = False
        if skip_python_body:
            skip_python_body = False
            dropped_above = True
            pending_go = None
            pending_rust = None
            continue
        if PYTHON_RUNTIME_FILE.match(raw):
            skip_python_body = True
            dropped_above = True
            pending_go = None
            pending_rust = None
            continue
        if not keep_stack_line(raw):
            if _dropped_frame(raw):
                dropped_above = True
            pending_go = None
            pending_rust = None
            continue
        java = JAVA_FRAME.match(raw)
        if java:
            add("caller", java.group("file"), java.group("symbol"), int(java.group("line")), None, number)
            pending_go = pending_rust = None
            continue
        csharp = CSHARP_FRAME.match(raw)
        if csharp:
            add("caller", csharp.group("path"), csharp.group("symbol"), int(csharp.group("line")), None, number)
            pending_go = pending_rust = None
            continue
        node = NODE_FRAME.match(raw)
        if node:
            add(
                "caller",
                node.group("path"),
                node.group("symbol"),
                int(node.group("line")),
                int(node.group("col")),
                number,
            )
            pending_go = pending_rust = None
            continue
        python = PYTHON_FRAME.match(raw)
        if python:
            add("caller", python.group("path"), python.group("symbol"), int(python.group("line")), None, number)
            pending_go = pending_rust = None
            continue
        rust_at = RUST_AT.match(raw)
        if rust_at and pending_rust is not None:
            column = int(rust_at.group("col")) if rust_at.group("col") else None
            add("caller", rust_at.group("path"), pending_rust[1], int(rust_at.group("line")), column, number)
            pending_go = pending_rust = None
            continue
        go_path = GO_PATH.match(raw)
        if go_path and pending_go is not None:
            add("caller", go_path.group("path"), pending_go[1], int(go_path.group("line")), None, number)
            pending_go = pending_rust = None
            continue
        # Go and Rust split one frame across two lines: the symbol, then the path.
        rust_symbol = RUST_SYMBOL.match(raw)
        if rust_symbol:
            pending_rust = (number, rust_symbol.group("symbol"))
            pending_go = None
            continue
        go_symbol = _go_symbol(raw)
        if go_symbol:
            pending_go = (number, go_symbol)
            pending_rust = None
            continue
        pending_go = pending_rust = None

    if not frames:
        return []
    # Python prints the raising frame last. Other languages print it first.
    python_stack = all(location.path.endswith(".py") for location in frames)
    if python_stack:
        for index, location in enumerate(frames):
            location.order = index
            location.role = "innermost" if index == len(frames) - 1 else "caller"
        return frames
    throw_index = 0
    if panic_at is not None:
        path, source_line, column, log_line = panic_at
        matched = next(
            (index for index, location in enumerate(frames) if _same_place(location, path, source_line)),
            None,
        )
        if matched is None:
            add("throw_site", path, "", source_line, column, log_line)
            throw_index = len(frames) - 1
        else:
            throw_index = matched
            if column is not None:
                frames[matched].source_column = column
    for index, location in enumerate(frames):
        location.order = index
        location.role = "throw_site" if index == throw_index else "caller"
    if panic_at is None and first_covered:
        # Runtime frames sat above the first kept frame, so it is not the throw site.
        frames[throw_index].role = "innermost"
    return frames
