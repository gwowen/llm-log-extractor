"""Summarize an ordinary text file, or narrate one extracted incident."""

import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from ..config import MODEL_NAME
from ..logs.selection import event_id
from ..models import Event
from ..reports.human import format_model_input
from .client import (
    GenerationError,
    build_llm,
    collect_summary,
    complete,
)
from .prompts import REPORT_PROMPT


def is_none(text: str) -> bool:
    """True when a chunk summary has nothing worth combining."""
    return text.strip().upper() in {"NONE", "NONE."}


def summarize_chunks(
    path: Path,
    chunks: list[str],
    *,
    piece_prompt: ChatPromptTemplate,
    combine_prompt: ChatPromptTemplate,
    llm: ChatOllama,
) -> str:
    """Summarize each chunk, drop NONE parts, and combine what remains."""
    piece_chain = piece_prompt | llm
    if len(chunks) == 1:
        return complete(piece_chain, {"source": path.name, "text": chunks[0]})

    partials: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Summarizing part {index} of {len(chunks)}...",
            file=sys.stderr,
        )
        partial = complete(piece_chain, {"source": path.name, "text": chunk})
        if not is_none(partial):
            partials.append(partial)

    if not partials:
        return "NONE"
    if len(partials) == 1:
        return partials[0]

    print("Combining partial summaries...", file=sys.stderr)
    combine_chain = combine_prompt | llm
    return complete(
        combine_chain,
        {"source": path.name, "text": "\n\n".join(partials)},
    )


def interpret_events(path: Path, event: Event, model: str | None = None) -> str:
    """Narrate one incident. An empty thinking trace is retried once with a larger budget.

    The retry builds a new client with `num_predict` raised from 2048 to 4096.
    The same budget is not tried twice. A second empty reply raises GenerationError.
    """
    chosen = model or MODEL_NAME
    print(
        f"Interpreting event {event_id(event)} with {chosen}...",
        file=sys.stderr,
    )
    payload = {"source": path.name, "text": format_model_input(event)}
    budgets = (2048, 4096)
    for attempt, budget in enumerate(budgets):
        if attempt:
            print(
                "The model returned only a thinking trace. "
                "Retrying with a larger token budget...",
                file=sys.stderr,
            )
        chain = REPORT_PROMPT | build_llm(reasoning=True, num_predict=budget, model=model)
        answer = collect_summary(chain, payload)
        if answer:
            return answer
    raise GenerationError("The model did not produce a summary.")
