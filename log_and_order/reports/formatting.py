"""Shared line formatting for reports."""

from ..models import (
    Event,
    LogLine,
)
from ..logs.selection import (
    event_kind,
)
from ..logs.extraction import (
    collapse_key,
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
    # More than four lines, packed into a span no wider than three times the count.
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
