# log-reader

Read a file and print a short version of it.

A timestamped log becomes a short incident report. The script extracts the events in code, then `LiquidAI/lfm2.5-2.6b` writes three sections: **Main failure**, **Related events**, and **Other warnings**. Line numbers in the report point back at the source file. The model does not suggest a fix. If a reply never leaves a thinking trace, it is retried once with a larger token budget.

Extraction keeps errors and the warnings that describe a real problem. Health checks, skipped work, not-found responses, and warnings that later recover are left out. A stack trace keeps the exception and the application frame. Request and response lines that share the error's trace stay with it, as do the config change, missing value, and failed setup lines just before the crash. The worker exit, service failure, and connection refusals that follow an exception stay with that exception. The same exception a few seconds apart is one incident. The same text hours later, or after the service has started again, is another. Repeated lines are counted once, with the first and last time.

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
```

The summary is written to stdout. Progress messages go to stderr, so you can pipe the summary:

```bash
uv run main.py app.log > summary.txt
```

The file must be text. An empty file prints `The file is empty.` A file containing a null byte is rejected.

These logs in the repository are noisy incidents the report is meant to digest:

- `spring_boot_incident_noise.log`
- `django_rest_crash_noisy.log`
- `gin_web_crash_noisy.log`
- `actix_web_crash_noisy.log`

## Tests

Extraction is checked without the model:

```bash
uv run python -m unittest
```

## License

[MIT](LICENSE)
