"""Defaults and limits. Values match the original single-module program."""

from datetime import timedelta

MODEL_NAME = "LiquidAI/lfm2.5-2.6b"
# Leave room in the context window for the prompt and the summary itself.
CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 400

# Cap on same-request and setup lines attached to one error.
MAX_CONTEXT_LINES = 8
# How far a process death, restart, or proxy failure can follow its exception.
CASCADE_WINDOW = timedelta(seconds=30)
# Identical exceptions this close can be one incident. Hours apart cannot.
SIGNATURE_WINDOW = timedelta(seconds=90)
# Diagnostic lines older than this are a different episode.
# Pass another timedelta to extract_events(lookback=...) to change the bound.
LOOKBACK = timedelta(seconds=30)
# Same-component lines in this gap are the step that entered the failure.
IMMEDIATE = timedelta(seconds=3)
# Recovery lines kept after a restart that belongs to the incident.
MAX_RECOVERY_LINES = 2

SCHEMA_VERSION = 1
EVIDENCE_BUDGET = 6_000
MAX_HYPOTHESES = 3
MAX_STEPS = 5
MAX_REGRESSION_TESTS = 3
MAX_MODEL_UNKNOWNS = 5
MAX_ANALYSIS_TEXT = 400
LINE_TEXT_LIMIT = 500
# Room left in the evidence package for the instruction note.
_NOTE_RESERVE = 180
