"""Command line. The report goes to stdout and progress goes to stderr."""

import argparse
import sys
from pathlib import Path

import httpx
from ollama import ResponseError

from .config import MODEL_NAME
from .llm.client import GenerationError
from .logs.extraction import extract_events
from .pipeline import FormatError, summarize
from .reports.human import dump_events


def read_file(path: Path) -> str:
    """Read text. A null byte means the file is binary."""
    data = path.read_bytes()
    if b"\0" in data:
        raise ValueError(f"{path} looks like a binary file")
    return data.decode("utf-8", errors="replace")


def publish(text: str, output: Path | None) -> None:
    """Print the report, or write it when --output is set. Stdout stays empty then."""
    if not text.endswith("\n"):
        text += "\n"
    if output is None:
        print(text, end="")
        return
    output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Command line. The human report stays the default."""
    parser = argparse.ArgumentParser(
        description="Order a log into an incident report, or summarize another text file.",
    )
    parser.add_argument("file", type=Path, help="Path to the file to summarize")
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=("human", "agent", "json"),
        default="human",
        help="human is the narrative report. agent is a Markdown brief. json is the same incident as data.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Do not call the model. Agent and JSON reports stay deterministic. The human log report uses the extracted lines.",
    )
    parser.add_argument(
        "--model",
        help=f"Ollama model name. Defaults to {MODEL_NAME}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to this file instead of stdout.",
    )
    parser.add_argument(
        "--dump-events",
        action="store_true",
        help="Print the extracted incident groups and exit without calling the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read one file and print its summary. Returns a process status code."""
    parser = build_parser()
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
        # Dump ignores --format. It shows the human-report model input.
        events = extract_events(text)
        if events is None:
            publish("Not a timestamped log.", args.output)
            return 0
        try:
            publish(dump_events(events), args.output)
        except OSError as exc:
            print(f"Could not write {args.output}: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        summary = summarize(
            path,
            text,
            args.report_format,
            use_llm=not args.no_llm,
            model=args.model,
        )
    except FormatError as exc:
        print(exc, file=sys.stderr)
        return 1
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

    try:
        publish(summary, args.output)
    except OSError as exc:
        print(f"Could not write {args.output}: {exc}", file=sys.stderr)
        return 1
    return 0
