"""Reconstruct Jenkins stages from console block markers.

Nested `[Pipeline] { ... }` blocks use a stack. Parallel branches that are
open at the same time do not claim a line unless the line names the branch.
"""

import re
from dataclasses import dataclass, field

from .text import PreparedLine

STAGE_OPEN = re.compile(r"^\[Pipeline\] \{ \((.+)\)\s*$")
SHELL = re.compile(r"^\+ (.+)$")
BRANCH = re.compile(r"^\[([^\]]+)\]\s+(.*)$")


@dataclass
class StageSpan:
    """One stage body, from its opening marker through the matching close."""

    name: str
    start: int
    end: int | None = None
    depth: int = 0
    parent: str | None = None
    parallel: bool = False
    commands: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class LinePlace:
    """Where one content line sits. `uncertain` means the branch was not named."""

    stage: str | None
    association: str  # confirmed, uncertain, or none
    possible: list[str]
    command: str | None
    command_line: int | None
    body: str


@dataclass
class _Frame:
    kind: str  # stage, parallel, or block
    name: str | None
    start: int
    parallel: bool = False
    command: str | None = None
    command_line: int | None = None


def _pop_stage(stack: list[_Frame], stages: list[StageSpan], line_number: int) -> None:
    """Close the innermost open stage, if the marker still has one to close."""
    if not stack or stack[-1].kind != "stage":
        return
    stack.pop()
    for stage in reversed(stages):
        if stage.end is None:
            stage.end = line_number
            return


def reconstruct(lines: list[PreparedLine]) -> tuple[list[StageSpan], dict[int, LinePlace], str]:
    """Walk the console once. Unbalanced markers leave stages open instead of raising.

    The job style is pipeline when a `[Pipeline]` marker appears, and freestyle
    when the log is Jenkins but has none.
    """
    stack: list[_Frame] = []
    stages: list[StageSpan] = []
    places: dict[int, LinePlace] = {}
    next_block: str | None = None
    loose_command: str | None = None
    loose_command_line: int | None = None
    saw_pipeline = False
    # `}` already closed the block. The following `// stage` names that close.
    just_closed = False

    for line in lines:
        text = line.text
        if text.startswith("[Pipeline]"):
            saw_pipeline = True
        if text == "[Pipeline] parallel":
            next_block = "parallel"
            just_closed = False
            continue
        if text == "[Pipeline] stage":
            next_block = "stage"
            just_closed = False
            continue
        if text in {"[Pipeline] sh", "[Pipeline] bat", "[Pipeline] powershell"}:
            just_closed = False
            continue

        opened = STAGE_OPEN.match(text)
        if opened:
            parent = next((frame.name for frame in reversed(stack) if frame.kind == "stage"), None)
            parallel = any(frame.kind == "parallel" for frame in stack)
            depth = sum(1 for frame in stack if frame.kind == "stage")
            stages.append(
                StageSpan(
                    name=opened.group(1).strip(),
                    start=line.number,
                    depth=depth,
                    parent=parent,
                    parallel=parallel,
                )
            )
            stack.append(
                _Frame(
                    "stage",
                    opened.group(1).strip(),
                    line.number,
                    parallel=parallel,
                )
            )
            next_block = None
            just_closed = False
            _remember(places, line, stack, loose_command, loose_command_line, text)
            continue

        if text == "[Pipeline] {":
            kind = "parallel" if next_block == "parallel" else "block"
            stack.append(_Frame(kind, None, line.number))
            next_block = None
            just_closed = False
            continue

        if text == "[Pipeline] }":
            if stack and stack[-1].kind == "stage":
                _pop_stage(stack, stages, line.number)
            elif stack:
                stack.pop()
            just_closed = True
            continue

        if text.startswith("[Pipeline] // "):
            label = text[len("[Pipeline] // ") :]
            if not just_closed and label == "stage":
                _pop_stage(stack, stages, line.number)
            elif not just_closed and label == "parallel" and stack and stack[-1].kind == "parallel":
                stack.pop()
            just_closed = False
            continue
        just_closed = False

        body, place = _place(text, stack, loose_command, loose_command_line)
        shell = SHELL.match(body)
        if shell:
            command = shell.group(1).strip()
            # An uncertain line is interleaved parallel output, so it names no command.
            if place.association != "uncertain" and place.stage:
                for frame in reversed(stack):
                    if frame.kind == "stage" and frame.name == place.stage:
                        frame.command = command
                        frame.command_line = line.number
                        break
                for stage in reversed(stages):
                    if stage.name == place.stage and stage.end is None:
                        stage.commands.append((line.number, command))
                        break
            elif place.association != "uncertain":
                loose_command = command
                loose_command_line = line.number
            place = LinePlace(
                stage=place.stage,
                association=place.association,
                possible=place.possible,
                command=command if place.association != "uncertain" else None,
                command_line=line.number if place.association != "uncertain" else None,
                body=body,
            )
        places[line.number] = place

    style = "pipeline" if saw_pipeline or stages else "freestyle"
    return stages, places, style


def _place(
    text: str,
    stack: list[_Frame],
    loose_command: str | None,
    loose_command_line: int | None,
) -> tuple[str, LinePlace]:
    """Assign a content line to the innermost stage, or to no stage when branches interleave."""
    open_stages = [frame for frame in stack if frame.kind == "stage"]
    parallel = any(frame.kind == "parallel" for frame in stack)
    if parallel and len(open_stages) >= 2:
        names = [frame.name for frame in open_stages if frame.name]
        branch_names = [frame.name for frame in open_stages if frame.parallel and frame.name]
        prefixed = BRANCH.match(text)
        if prefixed and prefixed.group(1) in names:
            name = prefixed.group(1)
            frame = next(frame for frame in open_stages if frame.name == name)
            return prefixed.group(2), LinePlace(
                stage=name,
                association="confirmed",
                possible=[],
                command=frame.command,
                command_line=frame.command_line,
                body=prefixed.group(2),
            )
        return text, LinePlace(
            stage=None,
            association="uncertain",
            possible=branch_names or names,
            command=None,
            command_line=None,
            body=text,
        )
    if open_stages:
        frame = open_stages[-1]
        return text, LinePlace(
            stage=frame.name,
            association="confirmed",
            possible=[],
            command=frame.command,
            command_line=frame.command_line,
            body=text,
        )
    return text, LinePlace(
        stage=None,
        association="none",
        possible=[],
        command=loose_command,
        command_line=loose_command_line,
        body=text,
    )


def _remember(
    places: dict[int, LinePlace],
    line: PreparedLine,
    stack: list[_Frame],
    loose_command: str | None,
    loose_command_line: int | None,
    text: str,
) -> None:
    """Record the stage marker itself so a report can cite the stage boundary."""
    _, place = _place(text, stack, loose_command, loose_command_line)
    places[line.number] = place
