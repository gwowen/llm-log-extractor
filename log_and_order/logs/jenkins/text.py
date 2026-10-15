"""Normalize Jenkins console lines without changing their numbers.

ANSI codes and hidden console annotations are removed. A Timestamper prefix
is stored separately. Secrets are replaced in place so a later report cannot
repeat them. The line list stays aligned with the source file.
"""

import re

from ...logs.parser import parse_moment
from ...models import LogLine

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
HIDDEN_SPAN = re.compile(r"\x1b\[8m.*?\x1b\[0m", re.DOTALL)
# Jenkins console notes of the form ha:////<opaque>.
ANNOTATION = re.compile(r"ha:/{2,}\S*")
TIMESTAMP = re.compile(
    r"^(?P<ts>"
    r"(?:\[\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\])"
    r"|(?:\[)?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?(?:\])?"
    r")\s+(?P<rest>.*)$"
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(aws_secret_access_key|aws_session_token|aws_access_key_id|"
    r"password|passwd|secret|token|api[_-]?key|authorization)\b"
    r"(\s*[=:]\s*)(\"?[^\"\s]+\"?)"
)
BEARER = re.compile(r"(?i)(bearer\s+)\S+")
AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
GITHUB_TOKEN = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
SLACK_TOKEN = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
PRIVATE_KEY = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----)"
)


def redact(text: str) -> str:
    """Replace obvious credentials. The surrounding words stay so the line is still readable."""
    text = PRIVATE_KEY.sub(r"\1 [REDACTED] \2", text)
    text = AWS_KEY.sub("[REDACTED]", text)
    text = GITHUB_TOKEN.sub("[REDACTED]", text)
    text = SLACK_TOKEN.sub("[REDACTED]", text)
    text = BEARER.sub(r"\1[REDACTED]", text)
    text = SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    return text


def clean_console(raw: str) -> str:
    """Drop ANSI, carriage returns, and Jenkins console annotations."""
    text = raw.replace("\r", "")
    text = HIDDEN_SPAN.sub("", text)
    text = ANSI.sub("", text)
    text = ANNOTATION.sub("", text)
    return text


def split_timestamp(text: str) -> tuple[str, str]:
    """Return (timestamp, remainder). A line with no Timestamper prefix keeps an empty timestamp.

    Leading spaces stay on an unstamped line so a stack frame is still indented.
    """
    if not text.strip():
        return "", ""
    left = text.lstrip()
    match = TIMESTAMP.match(left)
    if match is None:
        return "", text.rstrip()
    return match.group("ts").strip("[]"), match.group("rest").rstrip()


class PreparedLine:
    """One source line after console cleanup. `number` is still the file line."""

    __slots__ = ("number", "raw", "timestamp", "text", "moment")

    def __init__(self, number: int, raw: str, timestamp: str, text: str) -> None:
        self.number = number
        self.raw = raw
        self.timestamp = timestamp
        self.text = text
        self.moment = parse_moment(timestamp.replace(",", ".")) if timestamp else None

    def as_log_line(self, *, level: str = "", message: str | None = None) -> LogLine:
        """A LogLine whose number matches the file. The text is already redacted."""
        return LogLine(
            number=self.number,
            raw=self.raw,
            timestamp=self.timestamp,
            level=level,
            message=message if message is not None else self.text,
            is_record=bool(level),
            moment=self.moment,
        )


def prepare_lines(text: str) -> list[PreparedLine]:
    """Clean every line. The result has one entry per source line."""
    prepared: list[PreparedLine] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        cleaned = redact(clean_console(raw))
        timestamp, body = split_timestamp(cleaned)
        prepared.append(PreparedLine(number, cleaned, timestamp, body))
    return prepared
