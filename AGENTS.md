# AGENTS.md

Guidance for working on Log and Order. The program is the [`log_and_order`](log_and_order) package, run through the [`main.py`](main.py) wrapper, and managed with [uv](https://docs.astral.sh/uv/).

## Commands

```bash
uv sync
uv run main.py path/to/file
uv run main.py path/to/file --format agent
uv run main.py path/to/file --format json --output incident.json
uv run main.py path/to/file --format agent --no-llm
uv run python -m unittest
```

Ollama must be running locally with `LiquidAI/lfm2.5-2.6b` for a human report, a non-log summary, or the default agent and JSON enrichment. `--no-llm` skips the model. The report goes to stdout unless `--output` is set. Progress goes to stderr.

`tests/test_parser.py` checks record parsing and runtime-frame filtering. `tests/test_extraction.py` checks repeat collapsing and the fixture events. `tests/test_correlation.py` checks cascade grouping, prelude lines, and lifecycle. `tests/test_stacktraces.py` and `tests/test_incidents.py` check source locations and the deterministic incident. `tests/test_reports.py` checks the human, agent, and JSON reports. `tests/test_enrichment.py` checks enrichment with a fake model. `tests/test_cli.py` checks the command line. A full run of `example-logs/spring_boot_incident_noise.log` is the regression check for the human report.

## Layout

`log_and_order/cli.py` reads the file and writes the report. `pipeline.py` chooses the path. `config.py` holds the limits and the default model. `models.py` holds the shared dataclasses. `logs/` parses and correlates events. `is_serious` stays in `logs/correlation.py` because prelude and lifecycle attachment call it while grouping. `incidents/` builds the deterministic report and reads stack frames. `llm/` talks to Ollama. `reports/` only renders. Report modules do not call the model.

## What the program does

`summarize` branches on the file:

- A timestamped log (at least five `timestamp LEVEL` records) becomes events. Python selects one primary incident, the lines related to it, and the independent events. With `--format human` (the default), one ChatOllama call narrates only the incident, under **Main failure** and **Related events**. Python appends **Other warnings** from the independent events. If the model returns no answer after the retry, those two narrative sections are filled from the extracted lines. A log that yields no events prints `No relevant errors or warnings were found.`
- `--format agent` writes one Markdown incident brief from those same events. `--format json` writes that same incident as data, including `schema_version` and `analysis`. Unless `--no-llm` is set, `enrich_incident` asks the model for a failure mechanism, hypotheses, investigation steps, and regression scenarios. Python validates that reply and drops claims that cite unseen log lines or unseen paths and symbols. A confirmed correlation is not causation. If Ollama cannot be reached, or the reply is empty or invalid, the factual report is still written and stderr says enrichment failed. The Markdown brief leads with the executive summary and a Python-built task for a coding agent, then the failure, the labeled analysis, the observed facts, and the timeline. Stack text, raw excerpts, and other warnings come after that. Source lines and log lines are labeled differently. `evidence.possible_explanations` stays an empty list: the builder quotes logged facts and does not invent a cause. A log that yields no events still writes that document, with no primary failure. A non-log file with either format exits 1.
- Any other text file is summarized with LangChain when the format is human. Long files are split, partial summaries of `NONE` are dropped, and the rest are combined. `--no-llm` on that path exits 1, because the summary requires the model.
- An empty file prints `The file is empty.` A file containing a null byte is rejected.

Do not strip the file before `extract_events`. Line numbers are 1-based and must match the source file.

## Log extraction

Keep this work in code. Do not ask the model to filter, group, or recover line numbers.

`extract_events` walks the file once, then attaches context and groups incidents. Fewer than five `timestamp LEVEL` records returns None, and the file is summarized as ordinary text:

- Keep `ERROR`, `FATAL`, and `CRITICAL`. Keep `WARN` unless it is a health check, skipped work, a not-found response, or a successful recovery. A warning is a recovery when a later line on its trace says the operation recovered.
- Attach following stack lines to that event. Keep the exception line, `Caused by`, and the application frame. Drop Java, Python, Rust, Node.js, and .NET runtime frames. When a Python runtime `File` line is dropped, drop the source line under it too.
- Parse Spring `Logger : message` lines and tracing or service `target: message` lines. Read `traceId=` or `trace_id=` from the message, and `request_id=` when it is present.
- For an error with a trace id or request id, keep nearby lines that share it. Prefer incoming-request, outgoing-status, and diagnostic lines. Skip SQL, cache, and other routine lines on that trace. A token or JWT validation success on that same call is kept when the call also shows a missing tenant or claim. Other successful checks stay out.
- Also keep setup lines from the previous 30 seconds (`LOOKBACK`, overridable on `extract_events`) that name the same component: a config change, a failed init, a nil or missing value, or a line within 3 seconds that shares a component, config key, or exception name. A job id, key id, field, component, or callback shared with those lines can pull in an earlier setup line inside the same window. At most `MAX_CONTEXT_LINES` are kept. A line kept only because it is immediate is marked possible. Drop a standalone warning once its line has been attached this way.
- A later warning with the same trace id belongs to that error, and its stack follows the error.
- The process exit, restart attempt, and connection refusals within 30 seconds of an exception belong to that exception when a pid, systemd unit, or the same logger ties them together. A proxy failure joins the exception that already has that process death, and only before the service has started again. A shared pid is enough on its own. If the fallout line names a systemd unit and no pid matches, only an exception that names that unit may take it. An application line that merely contains `status=502` is its own event. A stacked line is not fallout: it stays an exception, and joins another event only when `same_incident` matches. Info lines that record the scheduled restart, the completed start, and a healthy or listening recovery join the same incident inside that window.
- Two errors with the same exception text, the same logger, and compatible trace ids are one incident when they fall within 90 seconds and no service start sits between them. The same text hours apart stays two incidents.
- Collapse repeated bare warnings into one event with a count, a line span, and the first and last timestamp. A repeated warning after a completed service start is a new incident.

`event_kind` labels an error `primary error` when it has an exception and a stack, same-trace warnings, or fallout. A bare error is a `separate error`. Anything else is a `warning`. Prelude lines alone do not make an error primary.

`format_model_input` is what the report prompt sees for the one primary incident. It has three labels: `Failure` (the error and its stack), `Context` (preceding lines, each marked confirmed or possible), and `Related` (later same-request lines, process exit, proxy failures, restart, and recovery). Event ids are `E` plus the seed line number. Leave those labels alone. `format_events` still renders every event for inspection. Independent events are not sent to the model.

## The model

`interpret_events` uses `reasoning=True` so a thinking trace stays out of the printed answer. If that reply is empty, retry once with a larger token budget, then raise `GenerationError`. Do not retry the same budget. `summarize` catches that error and fills Main failure and Related events from the extracted lines.

The human report prompt may only ask the model to narrate the primary incident. It must cite line numbers that appear in that incident, use the timestamp on the failure line, and not suggest a fix. Do not ask it to classify independent warnings or invent missing lines. Python writes **Other warnings** after the model returns. Do not point the human report at the enrichment prompt, or the enrichment prompt at `REPORT_PROMPT`.

`enrich_incident` is only for agent and JSON reports. It sends a compact evidence package, with log text fenced as untrusted data, and asks for JSON with reasoning off. Retry once when the reply is only a thinking trace, is not JSON, or fails validation. Do not retry a connection failure. Keep at most three hypotheses and five investigation steps. High confidence requires the failure line or a confirmed line. `--dump-events` prints the human-report model input and the independent events without calling Ollama. It ignores `--format`. `--output` writes that dump, or the report, to a file and leaves stdout empty.

The non-log summary keeps `reasoning=False`. Do not point that path at the log report prompt.

The project is under the MIT license. The notice is in [`LICENSE`](LICENSE).
