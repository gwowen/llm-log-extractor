"""Human incident narrative and the text sent to that narrative."""
import re

from ..models import (
    Event,
    LogLine,
)
from .formatting import (
    collapse_lines,
    format_collapsed,
    format_line,
)
from ..logs.correlation import (
    exception_signature,
)
from ..logs.selection import (
    event_id,
    independent_events,
    select_primary,
)


def _context_label(event: Event, line: LogLine) -> str:
    """confirmed when an id, unit, or pid ties the line in. possible when only proximity does."""
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
    """The Other warnings section. Python writes this after the model returns."""
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
    """Keep the model's two sections and append warnings Python already listed.

    A model that adds its own Other warnings heading is cut there, so that
    section is written once.
    """
    text = narrative.strip()
    text = re.split(r"\n(?=\**\s*Other warnings\b)", text, maxsplit=1)[0].rstrip()
    return f"{text}\n\n{format_other_warnings(independent)}"


def fallback_report(primary: Event | None, independent: list[Event]) -> str:
    """The human report when the model is skipped or returns no answer."""
    return compose_report(fallback_narrative(primary), independent)


def dump_events(events: list[Event]) -> str:
    """Show the model input and the independent events reported by Python."""
    primary = select_primary(events)
    independent = independent_events(events, primary)
    model_input = format_model_input(primary) if primary is not None else "Primary incident\nNone"
    return f"# Sent to the model\n{model_input}\n\n# Reported by Python\n{format_other_warnings(independent)}"
