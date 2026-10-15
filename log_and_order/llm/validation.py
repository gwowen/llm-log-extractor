"""Check structured model output against the evidence that was sent."""
import re
import json

from ..config import (
    MAX_ANALYSIS_TEXT,
    MAX_HYPOTHESES,
    MAX_MODEL_UNKNOWNS,
    MAX_REGRESSION_TESTS,
    MAX_STEPS,
)
from ..models import (
    DiagnosticHypothesis,
    EvidencePackage,
    IncidentEnrichment,
    IncidentReport,
    InvestigationStep,
    RegressionTestSuggestion,
)

_FILE_TOKEN = re.compile(
    r"(?:[A-Za-z0-9_./\\-]+\.(?:java|py|go|rs|js|jsx|ts|tsx|cs|c|cc|cpp|h|hpp)|"
    r"(?:[/\\][A-Za-z0-9_.-]+){2,})"
)
_SYMBOL_TOKEN = re.compile(r"\b[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)+\b")


def _clip_text(value: object, limit: int = MAX_ANALYSIS_TEXT) -> str | None:
    """Collapse whitespace and cap length. A non-string is not analysis text."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _string_items(value: object, *, limit: int = 4) -> list[str]:
    """Unique strings from a list, clipped and capped."""
    if not isinstance(value, list):
        return []
    found: list[str] = []
    for item in value:
        text = _clip_text(item, 240)
        if text and text not in found:
            found.append(text)
        if len(found) == limit:
            break
    return found


def _as_line_number(value: object) -> int | None:
    """An int, or a digit string. Booleans are not line numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _log_lines(value: object, allowed: set[int]) -> list[int]:
    """Line numbers the model cited that were actually in the evidence package."""
    if not isinstance(value, list):
        return []
    found: list[int] = []
    for item in value:
        number = _as_line_number(item)
        if number is not None and number in allowed and number not in found:
            found.append(number)
    return found


def _known_text(report: IncidentReport) -> str:
    """Paths, symbols, and log text a claim is allowed to name."""
    parts = [location.path for location in report.locations]
    parts.extend(location.symbol for location in report.locations if location.symbol)
    parts.extend(record.text for record in report.lines)
    if report.jenkins:
        facts = report.jenkins
        parts.extend(
            part
            for part in (
                facts.failed_command,
                facts.failed_stage,
                facts.source_file,
                facts.primary_message,
            )
            if part
        )
    return "\n".join(parts)


def _file_tokens(text: str) -> list[str]:
    """Paths and filenames a claim mentions. `node.js` is a runtime label, not a path."""
    found: list[str] = []
    for match in _FILE_TOKEN.finditer(text):
        token = match.group(0).strip(".,:;)")
        if token.lower() == "node.js":
            continue
        if token not in found:
            found.append(token)
    return found


def _symbol_tokens(text: str) -> list[str]:
    """Dotted names a claim mentions. File suffixes are paths, not symbols."""
    found: list[str] = []
    for match in _SYMBOL_TOKEN.finditer(text):
        token = match.group(0)
        if token.lower() in {"node.js", "e.g", "i.e"}:
            continue
        if any(
            token.endswith(suffix)
            for suffix in (".java", ".py", ".go", ".rs", ".js", ".jsx", ".ts", ".tsx", ".cs", ".cpp", ".cc")
        ):
            continue
        if token not in found:
            found.append(token)
    return found


def _unsupported_name(text: str, known: str) -> bool:
    """True when the model names a path or symbol the evidence does not contain."""
    for token in [*_file_tokens(text), *_symbol_tokens(text)]:
        if token not in known:
            return True
    return False

def _json_object(text: str) -> dict | None:
    """Take the first JSON object out of a reply. None when it is not an object."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _cap_confidence(
    confidence: str,
    numbers: list[int],
    package: EvidencePackage,
    failure_line: int,
) -> str:
    """High confidence needs the failure line or a confirmed line, not proximity."""
    if confidence != "high":
        return confidence
    kinds = [package.correlations.get(number) for number in numbers]
    if kinds and all(kind == "possible" for kind in kinds):
        return "low"
    strong = any(
        number == failure_line or package.correlations.get(number) in {"failure", "confirmed"}
        for number in numbers
    )
    if not strong:
        return "medium"
    return confidence


def _validate_enrichment(
    payload: dict,
    package: EvidencePackage,
    model: str,
    failure_line: int,
) -> IncidentEnrichment | None:
    """Drop claims that cite unseen lines or unseen names. None means the shape is wrong."""
    # Local import: enrichment calls this function, and the invalid reply calls back.
    from .enrichment import fallback_enrichment

    raw_mechanism = payload.get("failure_mechanism")
    raw_hypotheses = payload.get("hypotheses", [])
    raw_steps = payload.get("investigation_steps", [])
    raw_tests = payload.get("regression_tests", [])
    raw_unknowns = payload.get("additional_unknowns", [])
    if raw_mechanism is not None and not isinstance(raw_mechanism, str):
        return None
    for name, raw in (
        ("hypotheses", raw_hypotheses),
        ("investigation_steps", raw_steps),
        ("regression_tests", raw_tests),
        ("additional_unknowns", raw_unknowns),
    ):
        if name in payload and not isinstance(payload[name], list):
            return None
    if not isinstance(raw_hypotheses, list) or not isinstance(raw_steps, list):
        return None
    if not isinstance(raw_tests, list) or not isinstance(raw_unknowns, list):
        return None

    mechanism = _clip_text(raw_mechanism) if isinstance(raw_mechanism, str) else None
    mechanism_rejected = False
    if mechanism and _unsupported_name(mechanism, package.known_text):
        mechanism = None
        mechanism_rejected = True

    hypotheses: list[DiagnosticHypothesis] = []
    seen_hypotheses: set[str] = set()
    for item in raw_hypotheses:
        if len(hypotheses) == MAX_HYPOTHESES:
            break
        if not isinstance(item, dict):
            continue
        description = _clip_text(item.get("description"))
        verification = _clip_text(item.get("verification"))
        confidence = item.get("confidence")
        if not description or not verification or not isinstance(confidence, str):
            continue
        confidence = confidence.strip().lower()
        if confidence not in {"low", "medium", "high"}:
            continue
        numbers = _log_lines(item.get("supporting_log_lines"), package.allowed_lines)
        if not numbers:
            continue
        assumptions = _string_items(item.get("assumptions", []))
        missing = _string_items(item.get("missing_evidence", []))
        blob = "\n".join([description, verification, *assumptions, *missing])
        if _unsupported_name(blob, package.known_text):
            continue
        key = description.casefold()
        if key in seen_hypotheses:
            continue
        seen_hypotheses.add(key)
        hypotheses.append(
            DiagnosticHypothesis(
                description=description,
                confidence=_cap_confidence(confidence, numbers, package, failure_line),
                supporting_log_lines=numbers,
                assumptions=assumptions,
                missing_evidence=missing,
                verification=verification,
            )
        )

    steps: list[tuple[int, int, InvestigationStep]] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            continue
        priority = item.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 1:
            continue
        target = _clip_text(item.get("target"))
        action = _clip_text(item.get("action"))
        rationale = _clip_text(item.get("rationale"))
        expected = _clip_text(item.get("expected_finding"))
        if not target or not action or not rationale or not expected:
            continue
        numbers = _log_lines(item.get("supporting_log_lines"), package.allowed_lines)
        if not numbers:
            continue
        blob = "\n".join([target, action, rationale, expected])
        if _unsupported_name(blob, package.known_text):
            continue
        steps.append(
            (
                priority,
                index,
                InvestigationStep(
                    priority=priority,
                    target=target,
                    action=action,
                    rationale=rationale,
                    supporting_log_lines=numbers,
                    expected_finding=expected,
                ),
            )
        )
    steps.sort(key=lambda item: (item[0], item[1]))
    kept_steps: list[InvestigationStep] = []
    seen_steps: set[tuple[str, str]] = set()
    for _, _, step in steps:
        key = (" ".join(step.target.casefold().split()), " ".join(step.action.casefold().split()))
        if key in seen_steps:
            continue
        seen_steps.add(key)
        kept_steps.append(step)
        if len(kept_steps) == MAX_STEPS:
            break

    tests: list[RegressionTestSuggestion] = []
    for item in raw_tests:
        if len(tests) == MAX_REGRESSION_TESTS:
            break
        if not isinstance(item, dict):
            continue
        scenario = _clip_text(item.get("scenario"))
        expected = _clip_text(item.get("expected_behaviour"))
        if not scenario or not expected:
            continue
        numbers = _log_lines(item.get("supporting_log_lines"), package.allowed_lines)
        if not numbers:
            continue
        if _unsupported_name(f"{scenario}\n{expected}", package.known_text):
            continue
        tests.append(
            RegressionTestSuggestion(
                scenario=scenario,
                expected_behaviour=expected,
                supporting_log_lines=numbers,
            )
        )

    unknowns = [
        item
        for item in _string_items(raw_unknowns, limit=MAX_MODEL_UNKNOWNS)
        if not _unsupported_name(item, package.known_text)
    ]
    had_claims = bool(
        (isinstance(raw_mechanism, str) and raw_mechanism.strip())
        or raw_hypotheses
        or raw_steps
        or raw_tests
    )
    kept_any = bool(mechanism or hypotheses or kept_steps or tests or unknowns)
    # The model offered analysis and every claim was rejected.
    if had_claims and not kept_any:
        return fallback_enrichment("invalid", model)
    if mechanism_rejected and not kept_any:
        return fallback_enrichment("invalid", model)
    return IncidentEnrichment(
        failure_mechanism=mechanism,
        hypotheses=hypotheses,
        investigation_steps=kept_steps,
        regression_tests=tests,
        additional_unknowns=unknowns,
        model=model,
        enriched=True,
        fallback=False,
        reason=None,
    )
