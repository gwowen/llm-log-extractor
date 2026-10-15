"""Coordinate extraction, the optional model, and the chosen report."""

import sys
from pathlib import Path

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import ResponseError

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MODEL_NAME
from .incidents.builder import build_incident_report
from .llm.client import GenerationError, build_llm
from .llm.enrichment import disabled_enrichment, enrich_incident, fallback_enrichment
from .llm.prompts import COMBINE_PROMPT, SUMMARIZE_PROMPT
from .llm.summarization import interpret_events, summarize_chunks
from .logs.extraction import extract_events
from .logs.selection import independent_events, select_primary
from .models import Event, IncidentEnrichment
from .reports.agent import render_agent_report
from .reports.human import compose_report, fallback_narrative, fallback_report
from .reports.json import render_json_report


class FormatError(Exception):
    """An agent or JSON report was requested for a file that is not a log."""


def format_agent_report(
    path: Path,
    text: str,
    events: list[Event],
    enrichment: IncidentEnrichment | None = None,
) -> str:
    """Markdown brief for a developer and for a coding agent. One incident, told once."""
    report = build_incident_report(path, text, events)
    return render_agent_report(report, enrichment or disabled_enrichment())


def format_json_report(
    path: Path,
    text: str,
    events: list[Event],
    enrichment: IncidentEnrichment | None = None,
) -> str:
    """The same incident as the Markdown brief, as JSON."""
    report = build_incident_report(path, text, events)
    return render_json_report(report, enrichment or disabled_enrichment())


def _agent_or_json(
    path: Path,
    text: str,
    events: list[Event],
    report_format: str,
    *,
    use_llm: bool,
    model: str | None,
) -> str:
    """One incident document. The model is optional and cannot replace the facts."""
    report = build_incident_report(path, text, events)
    primary = select_primary(events)
    if not use_llm:
        enrichment = disabled_enrichment()
    elif primary is None:
        enrichment = fallback_enrichment("no_incident")
    else:
        try:
            llm = build_llm(reasoning=False, num_predict=1024, model=model, json_mode=True)
            enrichment = enrich_incident(report, primary, llm)
        except (httpx.ConnectError, httpx.TimeoutException, ResponseError, GenerationError) as exc:
            print(
                f"Enrichment failed ({exc}). Writing the deterministic incident report.",
                file=sys.stderr,
            )
            enrichment = fallback_enrichment("unavailable", model or MODEL_NAME)
    if report_format == "json":
        return render_json_report(report, enrichment)
    return render_agent_report(report, enrichment)


def summarize(
    path: Path,
    text: str,
    report_format: str = "human",
    *,
    use_llm: bool = True,
    model: str | None = None,
) -> str:
    """Report a timestamped log, or summarize other text.

    The human log path extracts events here and asks the model only to narrate
    them. Agent and JSON reports use the same events and, unless use_llm is
    false, add a validated diagnostic analysis. Fewer than five timestamped
    records is treated as ordinary text.
    """
    chosen = model or MODEL_NAME
    events = extract_events(text)
    if events is not None:
        # A timestamped log with nothing kept still has a path: human says so in one line,
        # and agent or JSON still emit the empty incident document.
        if not events:
            if report_format == "human":
                return "No relevant errors or warnings were found."
            return _agent_or_json(path, text, events, report_format, use_llm=use_llm, model=model)
        print(
            f"Extracted {len(events)} events from {len(text.splitlines())} lines.",
            file=sys.stderr,
        )
        if report_format in {"agent", "json"}:
            return _agent_or_json(path, text, events, report_format, use_llm=use_llm, model=model)
        primary = select_primary(events)
        independent = independent_events(events, primary)
        if not use_llm or primary is None:
            return fallback_report(primary, independent)
        try:
            narrative = interpret_events(path, primary, model)
        except GenerationError:
            print(
                "The model did not produce a summary. Using the extracted events.",
                file=sys.stderr,
            )
            narrative = fallback_narrative(primary)
        return compose_report(narrative, independent)
    # Ordinary text. Agent and JSON have no incident to describe, and the
    # human summary is a model call.
    if report_format != "human":
        raise FormatError("Agent and JSON reports require a timestamped log.")
    if not use_llm:
        raise FormatError("A non-log summary requires the model.")

    print(f"Summarizing {path} with {chosen}...", file=sys.stderr)
    llm = build_llm(model=model)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return summarize_chunks(
        path,
        splitter.split_text(text),
        piece_prompt=SUMMARIZE_PROMPT,
        combine_prompt=COMBINE_PROMPT,
        llm=llm,
    )
