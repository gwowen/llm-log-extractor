"""Decide whether console text is a Jenkins log.

A single generic word such as Pipeline, Build, or Deploy is not enough.
Detection counts distinct structural signals.
"""

import re

from .text import PreparedLine

# Each pattern is one signal, however many times it occurs.
STRONG = (
    re.compile(r"^\[Pipeline\] Start of Pipeline$"),
    re.compile(r"^\[Pipeline\] End of Pipeline$"),
    re.compile(r"^Finished: (?:SUCCESS|FAILURE|UNSTABLE|ABORTED|NOT_BUILT)$"),
    re.compile(
        r"^Started by (?:user\b|an SCM change|upstream project|timer\b|GitHub\b|upstream cause)"
    ),
    re.compile(r"^(?:ERROR:\s*)?script returned exit code \d+\s*$"),
    re.compile(r"\bhudson\.AbortException\b"),
    re.compile(r"\borg\.jenkinsci\.plugins\.workflow\b"),
    re.compile(r"^Building (?:in workspace|remotely)\b"),
    re.compile(r"^Build step '.+' marked build as failure$"),
)
MEDIUM = (
    re.compile(r"^\[Pipeline\] stage$"),
    re.compile(r"^\[Pipeline\] \{ \(.+\)$"),
    re.compile(r"^\[Pipeline\] sh$"),
    re.compile(r"^\[Pipeline\] // stage$"),
    re.compile(r"^\[Pipeline\] node$"),
    re.compile(r"^Obtained Jenkinsfile\b"),
)


def looks_like_jenkins(lines: list[PreparedLine]) -> bool:
    """True when several Jenkins console signals are present.

    Two strong signals are enough. One strong signal also qualifies when two
    different pipeline markers back it up. That covers a truncated pipeline
    that never reached Finished, and rejects a note that merely says Pipeline.
    """
    strong: set[int] = set()
    medium: set[int] = set()
    for line in lines:
        if not line.text:
            continue
        for index, pattern in enumerate(STRONG):
            if index not in strong and pattern.search(line.text):
                strong.add(index)
        for index, pattern in enumerate(MEDIUM):
            if index not in medium and pattern.search(line.text):
                medium.add(index)
        if len(strong) >= 2 or (len(strong) >= 1 and len(medium) >= 2):
            return True
    return len(strong) >= 2 or (len(strong) >= 1 and len(medium) >= 2)
