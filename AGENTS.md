# AGENTS.md

Guidance for working on log-reader. The program is one module, [`main.py`](main.py), managed with [uv](https://docs.astral.sh/uv/).

## Commands

```bash
uv sync
uv run main.py path/to/file
uv run python -m unittest
```

Ollama must be running locally with `LiquidAI/lfm2.5-2.6b` for a full run. The report goes to stdout. Progress goes to stderr.

`tests/test_extraction.py` checks extraction without the model. It covers cascade grouping, prelude lines, repeat collapsing, and the Spring and Django fixtures. A full run of `spring_boot_incident_noise.log` is the regression check for the report.

## What the program does

`summarize` branches on the file:

- A timestamped log (at least five `timestamp LEVEL` records) becomes events, then one ChatOllama call, then a report with the headings **Main failure**, **Related events**, and **Other warnings**.
- Any other text file is summarized with LangChain. Long files are split, partial summaries of `NONE` are dropped, and the rest are combined.
- An empty file prints `The file is empty.` A file containing a null byte is rejected.

Do not strip the file before `extract_events`. Line numbers are 1-based and must match the source file.

## Log extraction

Keep this work in code. Do not ask the model to filter, group, or recover line numbers.

`extract_events` walks the file once, then attaches context and groups incidents:

- Keep `ERROR`, `FATAL`, and `CRITICAL`. Keep `WARN` unless it is a health check, skipped work, a not-found response, or a successful recovery. A warning is a recovery when a later line on its trace says the operation recovered.
- Attach following stack lines to that event. Keep the exception line, `Caused by`, and the application frame. Drop Java, Python, and Rust runtime frames. When a Python runtime `File` line is dropped, drop the source line under it too.
- Parse Spring `Logger : message` lines and tracing or service `target: message` lines. Read `traceId=` or `trace_id=` from the message, and `request_id=` when it is present.
- For an error with a trace id or request id, keep nearby lines that share it. Prefer incoming-request, outgoing-status, and diagnostic lines. Skip SQL, cache, and other routine lines on that trace.
- Also keep setup lines from the previous 30 seconds that name the same component: a config change, a failed init, a nil or missing value, or a line within 3 seconds that shares a component, config key, or exception name. Drop a standalone warning once its line has been attached this way.
- A later warning with the same trace id belongs to that error, and its stack follows the error.
- The process exit, restart attempt, and connection refusals within 30 seconds of an exception belong to that exception. If the fallout line names a systemd unit, only an exception that names that unit may take it. An application line that merely contains `status=502` is its own event. A stacked line is not fallout: it stays an exception, and joins another event only when `same_incident` matches.
- Two errors with the same exception text, the same logger, and compatible trace ids are one incident when they fall within 90 seconds and no service start sits between them. The same text hours apart stays two incidents.
- Collapse repeated bare warnings into one event with a count, a line span, and the first and last timestamp. A repeated warning after a completed service start is a new incident.

`event_kind` labels an error `primary error` when it has an exception and a stack, same-trace warnings, or fallout. A bare error is a `separate error`. Anything else is a `warning`. Prelude lines alone do not make an error primary.

`format_events` sends a short block per event: `line_number timestamp LEVEL logger: message`. On a primary error, label the exception, frames, and fallout `Failure`, and label the same-request and prelude lines `Same request`. Those two headings are what the report prompt maps to Main failure and Related events. Leave the heading text alone. Other events use `Detail` or a collapsed warning line.

## The model

`interpret_events` uses `reasoning=True` so a thinking trace stays out of the printed answer. If that reply is empty, retry once with a larger token budget, then raise `GenerationError`. Do not retry the same budget.

The report prompt may only ask the model to narrate the extracted events. It must cite line numbers that appear in the event list, use the timestamp on the `ERROR` line, and not suggest a fix. Extra instructions that ask it to reclassify warning events or invent missing lines make this model write a thinking trace, repeat the report, or cite lines that were filtered out.

The non-log summary keeps `reasoning=False`. Do not point that path at the log report prompt.
