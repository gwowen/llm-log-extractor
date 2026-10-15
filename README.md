# Log and Order
In the software engineering system, production failures are investigated by two separate but equally important groups: the parsers, who uncover the evidence, and the local language models, who reconstruct the incidents. These are their reports.

DUN DUN. 🔨

===========================================================================================

Order a noisy log into one incident report, or summarize another text file.

A timestamped log becomes a short incident report. The script extracts the events in code and decides which lines belong to the main failure, which lines are related, and which warnings stand alone. `LiquidAI/lfm2.5-2.6b` writes **Main failure** and **Related events** from that incident only. **Other warnings** is written in code from the remaining lines, so the same line is not listed twice. Line numbers in the report point back at the source file. The model does not suggest a fix. If a reply never leaves a thinking trace, it is retried once with a larger token budget. If that retry is still empty, the two narrative sections are filled from the extracted lines.

`--format agent` and `--format json` use the same extracted incident. They ask the local model for hypotheses, investigation steps, and regression scenarios, then drop any claim that cites a log line or a source name the extraction did not supply. `--no-llm` skips that call and still writes the full factual report. If Ollama cannot be reached or the reply fails validation, the factual report is written anyway.

Extraction keeps errors and the warnings that describe a real problem. Health checks, skipped work, not-found responses, and warnings that later recover are left out. A stack trace keeps the exception and the application frame. Request and response lines that share the error's trace stay with it, as do the config change, missing value, and failed setup lines just before the crash when they share a component, job, key, or other id. The worker exit, service failure, connection refusals, restart, and recovery that follow an exception stay with that exception. The same exception a few seconds apart is one incident. The same text hours later, or after the service has started again, is another. Repeated lines are counted once, with the first and last time.

Any other file is summarized with `LiquidAI/lfm2.5-2.6b` through [LangChain](https://python.langchain.com). Longer files are split into chunks, summarized one part at a time, and then combined. A chunk with nothing to report is skipped. A reply that never leaves a thinking trace is discarded and retried once.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com), running locally
- The `LiquidAI/lfm2.5-2.6b` model

## Setup

```bash
uv sync
ollama pull LiquidAI/lfm2.5-2.6b
```

## Usage

```bash
uv run main.py path/to/file
uv run main.py path/to/file --format agent
uv run main.py path/to/file --format json --output incident.json
uv run main.py path/to/file --format agent --no-llm
uv run main.py path/to/file --format json --model LiquidAI/lfm2.5-2.6b
uv run main.py --dump-events path/to/file
```

`--format human` is the default. It asks the model for **Main failure** and **Related events**. `--format agent` writes a Markdown incident brief, including a task for a coding agent. `--format json` writes that same incident as JSON, with an `analysis` object. Both call Ollama unless `--no-llm` is set. `--model` selects a different local model. `--output` writes the report to a file instead of stdout.

`--format agent` and `--format json` need a timestamped log. Another text file exits 1. `--no-llm` on a non-log file also exits 1, because that summary is a model call.

A timestamped log with no kept errors or warnings prints `No relevant errors or warnings were found.` for the human report. Agent and JSON still write a report that says no primary failure was identified.

`--dump-events` prints the incident grouping and the lines that would be sent to the model, and does not call Ollama. It ignores `--format`. A file that is not a timestamped log prints `Not a timestamped log.`

The summary is written to stdout. Progress messages go to stderr, so you can pipe the summary:

```bash
uv run main.py app.log > summary.txt
```

The file must be text. An empty file prints `The file is empty.` A file containing a null byte is rejected.

These logs in `example-logs/` are noisy incidents the report is meant to digest:

- `example-logs/spring_boot_incident_noise.log`
- `example-logs/django_rest_crash_noisy.log`
- `example-logs/gin_web_crash_noisy.log`
- `example-logs/actix_web_crash_noisy.log`
- `example-logs/nodejs_express_crash_noisy.log`
- `example-logs/aspnet_core_crash_noisy.log`

## Tests

Extraction and report enrichment are checked without the model:

```bash
uv run python -m unittest
```

## License

[MIT](LICENSE)
