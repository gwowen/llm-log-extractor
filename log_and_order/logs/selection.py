"""Choose the primary incident and label the remaining events."""
from ..models import Event
from .patterns import (
    ERROR_LEVELS,
)
from .correlation import (
    exception_signature,
    is_serious,
)


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


def event_id(event: Event) -> str:
    """Stable id taken from the source line that opened the event."""
    return f"E{event.seed.number}"


def select_primary(events: list[Event]) -> Event | None:
    """The incident to explain. A terminating failure outranks an earlier error."""
    serious = [event for event in events if is_serious(event)]
    if not serious:
        return None

    def rank(event: Event) -> tuple[int, int, int, int]:
        # Fallout, then a fatal level, then a longer stack, then the later line.
        return (
            1 if event.consequences else 0,
            1 if event.seed.level in {"FATAL", "CRITICAL"} else 0,
            len(event.stack),
            event.seed.number,
        )

    return max(serious, key=rank)


def independent_events(events: list[Event], primary: Event | None) -> list[Event]:
    """Events other than the primary incident. The model does not see these."""
    if primary is None:
        return list(events)
    return [event for event in events if event is not primary]
