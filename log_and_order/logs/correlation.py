"""Relate log lines that belong to the same incident."""
import re
from datetime import datetime, timedelta
from ..models import Event, LogLine
from ..config import (
    CASCADE_WINDOW,
    IMMEDIATE,
    LOOKBACK,
    MAX_CONTEXT_LINES,
    MAX_RECOVERY_LINES,
    SIGNATURE_WINDOW,
)
from .patterns import (
    AUTHZ_GAP,
    AUTH_SUCCESS,
    CALLBACK_ID,
    COMPONENT_ID,
    CONTEXT_SIGNAL,
    DIAGNOSTIC,
    DIFF_ARROW,
    ERROR_LEVELS,
    EXCEPTION_SIGNATURE,
    FIELD_ID,
    GENERIC_TOKENS,
    JOB_ID,
    KEY_ID,
    NOISY_WARNING,
    PID_VALUE,
    PROCESS_DEATH,
    PROCESS_RESTART,
    RECOVERY_LINE,
    REVISION_ID,
    ROUTINE_CONTEXT,
    SERVICE_UNIT,
    STARTED_SERVICE,
    UPSTREAM_FAILURE,
    WARN_LEVELS,
)

def words(value: str) -> set[str]:
    """Split a logger, path, or message into correlation tokens.

    CamelCase is split so a class name can match the same word in a message.
    Tokens shorter than five characters, and the generic list, are left out.
    """
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
    """Words from the logger and the message that can name a component."""
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
    """Elapsed time from earlier to later. None when either timestamp did not parse."""
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


def _call_shows_authz_gap(event: Event, same_call: list[LogLine]) -> bool:
    """True when this call authenticated but a tenant or required claim was absent."""
    return any(AUTHZ_GAP.search(line.raw) for line in [event.seed, *event.stack, *same_call])


def attach_context(event: Event, records: list[LogLine]) -> None:
    """Keep non-routine lines from the same trace or request.

    Diagnostic and request or response lines outrank other same-call lines.
    A token or JWT success stays when that same call shows a missing tenant
    or claim. Other successful checks stay out. At most MAX_CONTEXT_LINES
    are kept, preferring lines closest to the error.
    """
    if event.seed.level not in ERROR_LEVELS:
        return
    trace_id = event.seed.trace_id
    request_id = event.seed.request_id
    if not trace_id and not request_id:
        return
    occupied = {line.number for line in event.lines()}
    same_call: list[LogLine] = []
    for record in records:
        if record.number in occupied or not shares_call(record, trace_id, request_id):
            continue
        if near(record, event.seed, LOOKBACK) or near(event.seed, record, LOOKBACK):
            same_call.append(record)
    explains_gap = _call_shows_authz_gap(event, same_call)
    ranked: list[tuple[int, int, LogLine]] = []
    for record in same_call:
        explanatory = explains_gap and AUTH_SUCCESS.search(record.raw) is not None
        if is_routine(record) and not explanatory:
            continue
        # Prefer a diagnostic, request, or explanatory auth line, then the closest line.
        useful = explanatory or DIAGNOSTIC.search(record.raw) or CONTEXT_SIGNAL.search(record.raw)
        ranked.append((2 if useful else 1, -abs(record.number - event.seed.number), record))
    ranked.sort(reverse=True)
    chosen = [item[2] for item in ranked[:MAX_CONTEXT_LINES]]
    chosen.sort(key=lambda line: line.number)
    event.context.extend(chosen)
    event.confirmed.update(line.number for line in chosen)


def line_pid(line: LogLine) -> str | None:
    """The pid named on this line, when the text has one."""
    match = PID_VALUE.search(line.raw)
    return match.group(1) if match else None


def event_pids(event: Event) -> set[str]:
    """Pids named on the failure line or its stack."""
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
    """Job, key, field, component, callback, and diff ids already attached to the failure."""
    found = _specific_ids(event.seed)
    for line in [*event.stack, *event.context]:
        found |= _specific_ids(line)
    return found


def revision_values(lines: list[LogLine]) -> set[str]:
    """Config revision numbers named on these lines."""
    found: set[str] = set()
    for line in lines:
        found.update(REVISION_ID.findall(line.raw))
    return found


def _prelude_noise(record: LogLine) -> bool:
    """Errors, routine chatter, and noisy warnings are not setup evidence."""
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
        # Tier 4 is a diagnostic that names the component. Tier 3 is the immediate
        # step into the failure, or a line that shares a job, key, field, or callback.
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
        # A shared revision is weaker than a shared job, key, or callback.
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
    """Distinctive words in a systemd unit name."""
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
    """Exceptions close enough in time for this line to be their fallout."""
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
    """Systemd units named on the incident."""
    units = set(SERVICE_UNIT.findall(event.seed.raw))
    for line in event.lines():
        units.update(SERVICE_UNIT.findall(line.raw))
    return units


def _service_overlap(event: Event, line: LogLine) -> bool:
    """True when the line names this incident's unit or service and the pids do not conflict."""
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
