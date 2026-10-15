"""Classify Jenkins console lines and choose the actionable failure.

Generic Jenkins status lines stay symptoms. A compiler, test, or tool
diagnostic outranks them. The same text repeated in one stage collapses to
one hit so a noisy log does not multiply the incident.
"""

import re
from dataclasses import dataclass, field

from .stages import LinePlace
from .text import PreparedLine

CATEGORIES = (
    "APPLICATION",
    "BUILD",
    "TEST",
    "DEPLOYMENT",
    "INFRASTRUCTURE",
    "CONFIGURATION",
    "UNKNOWN",
)

NOISE = re.compile(
    r"(recovered from expected|intentionally simulated|skipped due to earlier failure|"
    r"\bnon-fatal\b|npm verbose|npm http fetch|^\[build\] compiling|"
    r"^Test Suites:|^Tests:\s+\d+ passed|^Snapshots:|^✓ |^PASS\b|"
    r"^\[WS-CLEANUP\]|^Archiving artifacts|^Recording test results)",
    re.IGNORECASE,
)
RECOVERY = re.compile(
    r"(\bretry succeeded\b|\brecovered\b|\bbuild succeeded\b|\bcompleted successfully\b)",
    re.IGNORECASE,
)
EXIT_CODE = re.compile(r"^(?:ERROR:\s*)?script returned exit code (?P<code>\d+)\s*$")
ABORT_EXIT = re.compile(
    r"hudson\.AbortException:\s*script returned exit code (?P<code>\d+)\s*$"
)
BUILD_STEP = re.compile(r"^Build step '.+' marked build as failure$")
TYPESCRIPT = re.compile(
    r"^(?P<file>[\w./\\-]+\.(?:ts|tsx|js|jsx|mts|cts))"
    r"(?:\((?P<line>\d+),(?P<col>\d+)\)|:(?P<line2>\d+):(?P<col2>\d+))\s*"
    r"(?::|-)\s+error TS\d+:\s*(?P<msg>.+)$"
)
GCC = re.compile(
    r"^(?P<file>[\w./\\-]+\.(?:c|cc|cpp|cxx|h|hpp)):"
    r"(?P<line>\d+):(?P<col>\d+):\s+(?:fatal\s+)?error:\s+(?P<msg>.+)$"
)
RUST = re.compile(r"^error(?:\[E\d+\])?:\s+(?P<msg>.+)$")
RUST_AT = re.compile(r"^-->\s+(?P<file>\S+?):(?P<line>\d+):(?P<col>\d+)\s*$")
MAVEN_FILE = re.compile(
    r"^\[ERROR\]\s+(?P<file>\S+\.java):\[(?P<line>\d+),(?P<col>\d+)\]\s+(?P<msg>.+)$"
)
MAVEN_GOAL = re.compile(r"^\[ERROR\]\s+Failed to execute goal\b")
MSBUILD = re.compile(
    r"^(?P<file>\S+\.(?:cs|vb|fs))(?:\((?P<line>\d+),(?P<col>\d+)\))?:\s+error\s+\w+\d+:\s*(?P<msg>.+)$"
)
GRADLE_TASK = re.compile(r"^Execution failed for task '(?P<task>[^']+)'\.?")
GRADLE_BANNER = re.compile(r"^FAILURE:\s+Build failed\b")
NPM = re.compile(r"^npm ERR!\s+(?P<msg>.+)$")
PY_PACKAGING = re.compile(
    r"^(?:ERROR:\s+Could not find a version that satisfies|error:\s+subprocess-exited-with-error)\b"
)
PYTEST_FAILED = re.compile(r"^FAILED\s+(?P<name>\S+?)(?:\s+-\s+(?P<msg>.+))?$")
PYTEST_LOC = re.compile(
    r"^(?P<file>[\w./\\-]+\.py):(?P<line>\d+):\s+(?P<msg>(?:AssertionError|Failed)\b.*)$"
)
PYTEST_E = re.compile(r"^E\s+(?P<msg>AssertionError:.+|E\s+assert .+)$")
JEST_FAIL = re.compile(r"^FAIL\s+(?P<file>\S+\.(?:js|jsx|ts|tsx))\b")
JEST_TEST = re.compile(r"^●\s+(?P<name>.+)$")
SUREFIRE = re.compile(r"^(?P<name>[\w.$]+)\s+<{3,}\s+FAILURE!")
SUREFIRE_SUM = re.compile(r"^Tests run:\s*\d+,\s*Failures:\s*(?P<count>[1-9]\d*)")
DOCKER = re.compile(
    r"^(?:ERROR:\s+failed to solve:\s*(?P<msg>.+)|docker:\s+Error response from daemon:\s*(?P<msg2>.+))$",
    re.IGNORECASE,
)
DOCKER_DAEMON = re.compile(r"Cannot connect to the Docker daemon")
AWS = re.compile(
    r"^An error occurred \((?P<code>[^)]+)\) when calling the (?P<op>\w+) operation:\s*(?P<msg>.*)$"
)
WAITER = re.compile(r"\bWaiter \w+ failed:")
CFN_STATUS = re.compile(r"\b(?:CREATE_FAILED|UPDATE_FAILED|ROLLBACK_COMPLETE)\b")
KUBECTL = re.compile(r"^error:\s+(?P<msg>.+)$")
SSH = re.compile(r"(Permission denied \(publickey\)|ssh: connect to host \S+|Host key verification failed)")
GIT = re.compile(r"^fatal:\s+(?P<msg>.+)$")
CHECKOUT = re.compile(r"^ERROR:\s+Checkout failed\b")
MISSING = re.compile(
    r"(?:(?P<file>[\w./\\-]+):\s+line\s+\d+:\s+)?(?P<cmd>[\w./+-]+):\s+command not found"
)
NETWORK = re.compile(
    r"(Could not resolve host|Temporary failure in name resolution|"
    r"Connection timed out|Network is unreachable|Connection refused)"
)
AUTH = re.compile(
    r"(AccessDenied|is not authorized to perform|401 Unauthorized|403 Forbidden|"
    r"invalid credentials|HTTP Basic: Access denied)",
    re.IGNORECASE,
)
TIMEOUT = re.compile(
    r"(Timeout has been exceeded|Cancelling nested steps due to timeout|ExceededTimeout)"
)
AGENT = re.compile(
    r"(There are no nodes with the label|Agent .+ was removed|Could not connect to .+ agent)",
    re.IGNORECASE,
)
AGENT_OFFLINE = re.compile(r".+\bis offline\b", re.IGNORECASE)
MAKE_ERROR = re.compile(r"^make: \*\*\* .+ Error (?P<code>\d+)\s*$")
WORKSPACE = re.compile(r"(Unable to create live FilePath|java\.nio\.file\.AccessDeniedException)")
CHANNEL = re.compile(r"hudson\.remoting\.(?:ChannelClosedException|RequestAbortedException)")
PIPELINE_SYNTAX = re.compile(
    r"(CpsCompilationErrorsException|No such DSL method|script not yet approved|"
    r"ERROR:\s+Invalid Jenkinsfile|WorkflowScript: \d+:)"
)
CREDENTIALS = re.compile(
    r"(Could not find credentials|No credentials found|Credentials \S+ not found)",
    re.IGNORECASE,
)
JAVA = re.compile(
    r"^(?:Exception in thread \"[^\"]+\" )?"
    r"(?P<type>(?:[\w$]+\.)*(?:Exception|Error)):\s*(?P<msg>.*)$"
)
JAVA_AT = re.compile(r"^\s*at\s+(?P<symbol>[\w.$]+)\((?P<file>[^:()]+):(?P<line>\d+)\)")
PY_FILE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')
PY_ERR = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception)):\s*(?P<msg>.*)$")
ABORT = re.compile(r"hudson\.AbortException:\s*(?P<msg>.*)$")
GENERIC_ERROR = re.compile(r"^ERROR:\s+(?P<msg>.+)$")
CARET = re.compile(r"^[\s^~]+$")
SOURCE_SNIPPET = re.compile(r"^\d+\s+\S")


@dataclass
class FailureHit:
    """One extracted failure. A symptom is a Jenkins wrapper, not a tool diagnostic."""

    category: str
    message: str
    log_line: int
    specificity: int
    symptom: bool
    source_file: str | None = None
    source_line: int | None = None
    source_column: int | None = None
    exit_code: int | None = None
    test_name: str | None = None
    detail_lines: list[int] = field(default_factory=list)
    stage: str | None = None
    stage_association: str = "none"
    possible_stages: list[str] = field(default_factory=list)
    command: str | None = None
    command_line: int | None = None
    recovered: bool = False


def _hit(category: str, message: str, specificity: int, *, symptom: bool = False, **extra) -> FailureHit:
    """A hit whose line number is filled in by the caller."""
    return FailureHit(
        category=category,
        message=message.strip(),
        log_line=0,
        specificity=specificity,
        symptom=symptom,
        **extra,
    )


def _deploy_or_infra(command: str | None) -> str:
    """SSH and network errors during a deploy command stay with the deployment."""
    text = command or ""
    if any(word in text for word in ("ssh", "scp", "deploy", "kubectl", "docker", "aws")):
        return "DEPLOYMENT"
    return "INFRASTRUCTURE"


def match_line(body: str, command: str | None) -> FailureHit | None:
    """Classify one console line. None means the line is not a failure seed."""
    text = body.strip()
    if not text or text.startswith("[Pipeline]") or NOISE.search(text):
        return None
    exit_match = EXIT_CODE.match(text) or ABORT_EXIT.match(text)
    if exit_match:
        code = int(exit_match.group("code"))
        return _hit(
            "UNKNOWN",
            f"script returned exit code {code}",
            20,
            symptom=True,
            exit_code=code,
        )
    if BUILD_STEP.match(text):
        return _hit("UNKNOWN", text, 15, symptom=True)
    if TIMEOUT.search(text):
        return _hit("INFRASTRUCTURE", text, 90)
    if AGENT.search(text):
        return _hit("INFRASTRUCTURE", text, 95)
    if AGENT_OFFLINE.search(text):
        return _hit("INFRASTRUCTURE", text, 80)
    make = MAKE_ERROR.match(text)
    if make:
        return _hit("BUILD", text, 25, symptom=True, exit_code=int(make.group("code")))
    if WORKSPACE.search(text):
        return _hit("INFRASTRUCTURE", text, 80)
    if CHANNEL.search(text):
        return _hit("INFRASTRUCTURE", text, 85)
    if PIPELINE_SYNTAX.search(text):
        return _hit("CONFIGURATION", text, 85)
    if CREDENTIALS.search(text):
        return _hit("CONFIGURATION", text, 85)
    typescript = TYPESCRIPT.match(text)
    if typescript:
        return _hit(
            "BUILD",
            typescript.group("msg"),
            100,
            source_file=typescript.group("file"),
            source_line=int(typescript.group("line") or typescript.group("line2")),
            source_column=int(typescript.group("col") or typescript.group("col2")),
        )
    gcc = GCC.match(text)
    if gcc:
        return _hit(
            "BUILD",
            gcc.group("msg"),
            100,
            source_file=gcc.group("file"),
            source_line=int(gcc.group("line")),
            source_column=int(gcc.group("col")),
        )
    if RUST.match(text):
        return _hit("BUILD", RUST.match(text).group("msg"), 95)
    maven = MAVEN_FILE.match(text)
    if maven:
        return _hit(
            "BUILD",
            maven.group("msg"),
            100,
            source_file=maven.group("file"),
            source_line=int(maven.group("line")),
            source_column=int(maven.group("col")),
        )
    if MAVEN_GOAL.match(text):
        return _hit("BUILD", text, 30, symptom=True)
    msbuild = MSBUILD.match(text)
    if msbuild:
        return _hit(
            "BUILD",
            msbuild.group("msg"),
            100,
            source_file=msbuild.group("file"),
            source_line=int(msbuild.group("line")) if msbuild.group("line") else None,
            source_column=int(msbuild.group("col")) if msbuild.group("col") else None,
        )
    gradle = GRADLE_TASK.match(text)
    if gradle:
        return _hit("BUILD", text, 75)
    if GRADLE_BANNER.match(text):
        return _hit("BUILD", text, 35, symptom=True)
    npm = NPM.match(text)
    if npm:
        message = npm.group("msg")
        wrapper = message.startswith(("code ", "path ", "command failed", "errno "))
        return _hit("BUILD", message, 25 if wrapper else 80, symptom=wrapper)
    if PY_PACKAGING.match(text):
        return _hit("BUILD", text, 80)
    pytest_loc = PYTEST_LOC.match(text)
    if pytest_loc:
        return _hit(
            "TEST",
            pytest_loc.group("msg"),
            92,
            source_file=pytest_loc.group("file"),
            source_line=int(pytest_loc.group("line")),
        )
    pytest_failed = PYTEST_FAILED.match(text)
    if pytest_failed:
        message = pytest_failed.group("msg") or pytest_failed.group("name")
        return _hit("TEST", message, 88, test_name=pytest_failed.group("name"))
    if PYTEST_E.match(text):
        return _hit("TEST", PYTEST_E.match(text).group("msg"), 90)
    jest = JEST_FAIL.match(text)
    if jest:
        return _hit("TEST", text, 88, source_file=jest.group("file"))
    jest_test = JEST_TEST.match(text)
    if jest_test:
        return _hit("TEST", jest_test.group("name"), 86, test_name=jest_test.group("name"))
    surefire = SUREFIRE.match(text)
    if surefire:
        return _hit("TEST", text, 90, test_name=surefire.group("name"))
    if SUREFIRE_SUM.match(text):
        return _hit("TEST", text, 55)
    docker = DOCKER.match(text)
    if docker:
        return _hit("DEPLOYMENT", docker.group("msg") or docker.group("msg2") or text, 85)
    if DOCKER_DAEMON.search(text):
        return _hit("DEPLOYMENT", text, 85)
    aws = AWS.match(text)
    if aws:
        message = aws.group("msg") or aws.group("code")
        return _hit("DEPLOYMENT", f"{aws.group('code')}: {message}".strip(), 90)
    if WAITER.search(text) or CFN_STATUS.search(text):
        return _hit("DEPLOYMENT", text, 80)
    if KUBECTL.match(text) and command and "kubectl" in command:
        return _hit("DEPLOYMENT", KUBECTL.match(text).group("msg"), 80)
    if SSH.search(text):
        return _hit(_deploy_or_infra(command), text, 85)
    git = GIT.match(text)
    if git:
        return _hit("INFRASTRUCTURE", git.group("msg"), 80)
    if CHECKOUT.match(text):
        return _hit("INFRASTRUCTURE", text, 80)
    missing = MISSING.search(text)
    if missing and "command not found" in text:
        return _hit("INFRASTRUCTURE", text, 85)
    if NETWORK.search(text):
        return _hit("INFRASTRUCTURE", text, 80)
    if AUTH.search(text):
        category = "DEPLOYMENT" if _deploy_or_infra(command) == "DEPLOYMENT" else "CONFIGURATION"
        return _hit(category, text, 80)
    abort = ABORT.match(text)
    if abort:
        return _hit("UNKNOWN", abort.group("msg") or text, 45, symptom=not abort.group("msg"))
    java = JAVA.match(text)
    if java and java.group("type") not in {"Error", "Exception"}:
        kind = java.group("type")
        if kind.startswith(("hudson.", "org.jenkinsci.", "jenkins.")):
            category = "INFRASTRUCTURE"
        elif command and any(word in command for word in ("pytest", "mvn test", "gradle test")):
            category = "TEST"
        else:
            category = "APPLICATION"
        return _hit(category, text, 70)
    python_error = PY_ERR.match(text)
    if python_error and not text.startswith("[ERROR]"):
        kind = python_error.group("type")
        if command and any(word in command for word in ("pip", "poetry", "uv ", "python -m build")):
            category = "BUILD"
        elif command and "pytest" in command or kind == "AssertionError":
            category = "TEST"
        else:
            category = "APPLICATION"
        return _hit(category, text, 70)
    generic = GENERIC_ERROR.match(text)
    if generic:
        message = generic.group("msg")
        if message.startswith("script returned"):
            return None
        return _hit("UNKNOWN", message, 40, symptom=len(message) < 24)
    return None


def _continues(body: str) -> bool:
    """True when this line still belongs to the diagnostic above it."""
    text = body.rstrip()
    if not text.strip():
        return True
    stripped = text.strip()
    if stripped.startswith("[Pipeline]") or stripped.startswith("Finished:"):
        return False
    if SHELL_LINE.match(stripped) or EXIT_CODE.match(stripped):
        return False
    if stripped.startswith(
        ("Type ", "Found ", "Expected", "Received", "assert ", "at ", "Caused by:", "+ ", "- ")
    ):
        return True
    if text[:1].isspace() or CARET.match(stripped) or SOURCE_SNIPPET.match(stripped):
        return True
    if stripped.startswith(("E   ", "E\t")):
        return True
    return False


SHELL_LINE = re.compile(r"^\+ ")


def _absorb(lines: list[PreparedLine], start: int) -> list[int]:
    """Line numbers that continue the diagnostic. At most twelve, then the block ends."""
    found: list[int] = []
    blanks = 0
    for line in lines[start + 1 : start + 13]:
        if not _continues(line.text):
            break
        if not line.text.strip():
            blanks += 1
            if blanks > 2:
                break
            continue
        if match_line(line.text, None) is not None and found:
            break
        found.append(line.number)
    return found


def _apply_place(hit: FailureHit, place: LinePlace | None) -> None:
    """Copy stage and command from the parser. Uncertain lines do not gain a command."""
    if place is None:
        return
    hit.stage = place.stage
    hit.stage_association = place.association
    hit.possible_stages = list(place.possible)
    if place.association != "uncertain":
        hit.command = place.command
        hit.command_line = place.command_line


def _last_command(places: dict[int, LinePlace], before: int, stage: str | None) -> tuple[str | None, int | None]:
    """The latest confirmed command before this line, in the same stage when the stage is known."""
    found: tuple[str | None, int | None] = (None, None)
    for number in sorted(places):
        if number >= before:
            break
        place = places[number]
        if not place.command or place.association == "uncertain":
            continue
        if stage and place.stage not in {stage, None}:
            continue
        found = (place.command, place.command_line)
    return found


def _same_failure(left: FailureHit, right: FailureHit) -> bool:
    """True when two hits are the same diagnostic printed more than once."""
    if left.stage != right.stage or left.category != right.category:
        return False
    if left.message and right.message and (left.message in right.message or right.message in left.message):
        return True
    if (
        left.source_file
        and left.source_file == right.source_file
        and left.source_line
        and left.source_line == right.source_line
    ):
        return True
    return bool(left.test_name and left.test_name == right.test_name)


def _collapse_duplicates(hits: list[FailureHit]) -> list[FailureHit]:
    """Keep the richer hit when one stage repeats the same failure."""
    kept: list[FailureHit] = []
    for hit in hits:
        match = next((item for item in kept if _same_failure(item, hit)), None)
        if match is None:
            kept.append(hit)
            continue
        if hit.specificity > match.specificity:
            _fold(hit, match)
            kept[kept.index(match)] = hit
        else:
            _fold(match, hit)
    return kept


def _dedupe(hits: list[FailureHit]) -> list[FailureHit]:
    """Collapse the same diagnostic reported on several lines."""
    kept: list[FailureHit] = []
    index: dict[tuple, int] = {}
    for hit in hits:
        key = (hit.category, hit.message, hit.source_file, hit.source_line, hit.stage)
        existing = index.get(key)
        if existing is None:
            index[key] = len(kept)
            kept.append(hit)
            continue
        prior = kept[existing]
        if hit.log_line not in prior.detail_lines:
            prior.detail_lines.append(hit.log_line)
        prior.detail_lines.extend(number for number in hit.detail_lines if number not in prior.detail_lines)
        if prior.test_name is None and hit.test_name:
            prior.test_name = hit.test_name
        if prior.source_file is None and hit.source_file:
            prior.source_file = hit.source_file
            prior.source_line = hit.source_line
            prior.source_column = hit.source_column
        if prior.exit_code is None and hit.exit_code is not None:
            prior.exit_code = hit.exit_code
    return kept


def _mark_recovered(hits: list[FailureHit], lines: list[PreparedLine], places: dict[int, LinePlace]) -> None:
    """A later recovery line in the same stage retires that failure."""
    by_number = {line.number: line for line in lines}
    ordered = sorted(by_number)
    for hit in hits:
        if hit.symptom:
            continue
        for number in ordered:
            if number <= hit.log_line:
                continue
            place = places.get(number)
            stage = place.stage if place else None
            if hit.stage and stage and stage != hit.stage:
                break
            text = by_number[number].text
            if text.startswith("[Pipeline] stage"):
                break
            if RECOVERY.search(text) and "expected Error" not in text:
                hit.recovered = True
                break


def _fold(primary: FailureHit, hit: FailureHit) -> None:
    """Keep a same-stage line as evidence on the chosen failure."""
    if (
        hit.message
        and primary.message
        and primary.message in hit.message
        and len(hit.message) > len(primary.message)
    ):
        primary.message = hit.message
    if hit.log_line not in primary.detail_lines:
        primary.detail_lines.append(hit.log_line)
    if hit.exit_code is not None and primary.exit_code is None:
        primary.exit_code = hit.exit_code
    if primary.source_file is None and hit.source_file:
        primary.source_file = hit.source_file
        primary.source_line = hit.source_line
        primary.source_column = hit.source_column
    if primary.test_name is None and hit.test_name:
        primary.test_name = hit.test_name


def _choose(hits: list[FailureHit]) -> tuple[FailureHit | None, list[FailureHit]]:
    """Pick the actionable failure. Close scores keep the earlier line and list the rest.

    Symptoms are used only when no diagnostic exists. Same-stage leftovers fold
    into the chosen hit. A later stage with its own diagnostic stays independent.
    """
    live = [hit for hit in hits if not hit.recovered]
    diagnostics = [hit for hit in live if not hit.symptom]
    strong = [hit for hit in diagnostics if hit.specificity >= 50]
    pool = strong or diagnostics or [hit for hit in live if hit.symptom]
    if not pool:
        return None, []
    best = max(hit.specificity for hit in pool)
    # A clearly more specific diagnostic wins. Near ties keep the earlier line.
    contenders = [hit for hit in pool if best - hit.specificity <= 5]
    primary = min(contenders, key=lambda hit: hit.log_line)
    others: list[FailureHit] = []
    for hit in live:
        if hit is primary:
            continue
        same_stage = hit.stage == primary.stage and hit.stage_association == primary.stage_association
        if same_stage:
            _fold(primary, hit)
            continue
        if hit.symptom or hit.specificity < 50:
            continue
        others.append(hit)
    return primary, others


def _attach_exit(primary: FailureHit | None, others: list[FailureHit], hits: list[FailureHit]) -> None:
    """Give the trailing script status to the last unrecovered diagnostic that lacks one."""
    if primary is None:
        return
    exits = [hit for hit in hits if hit.exit_code is not None]
    if not exits:
        return
    anchors = [primary, *others]
    anchor = max(anchors, key=lambda hit: hit.log_line)
    if anchor.exit_code is not None:
        return
    later = [hit for hit in exits if hit.log_line > anchor.log_line]
    chosen = later[-1] if later else exits[-1]
    anchor.exit_code = chosen.exit_code
    if chosen.log_line not in anchor.detail_lines and chosen.log_line != anchor.log_line:
        anchor.detail_lines.append(chosen.log_line)


def _copy_rust_location(hit: FailureHit, lines: list[PreparedLine]) -> None:
    """Rust prints the file on the following `-->` line."""
    if hit.source_file or not hit.message:
        return
    by_number = {line.number: line.text.strip() for line in lines}
    for number in hit.detail_lines:
        match = RUST_AT.match(by_number.get(number, ""))
        if match:
            hit.source_file = match.group("file")
            hit.source_line = int(match.group("line"))
            hit.source_column = int(match.group("col"))
            return


def _copy_frame(hit: FailureHit, lines: list[PreparedLine]) -> None:
    """Use the first application frame under a Java exception, or the Python file above it."""
    if hit.source_file or hit.category not in {"APPLICATION", "TEST", "BUILD", "INFRASTRUCTURE"}:
        return
    by_number = {line.number: line.text for line in lines}
    for number in hit.detail_lines:
        match = JAVA_AT.match(by_number.get(number, ""))
        if match is None:
            continue
        path = match.group("file")
        if "jenkinsci" in path or path.startswith(("Native Method", "Unknown Source")):
            continue
        hit.source_file = path
        hit.source_line = int(match.group("line"))
        return
    for number in range(hit.log_line - 1, max(0, hit.log_line - 8), -1):
        match = PY_FILE.search(by_number.get(number, ""))
        if match is None:
            continue
        path = match.group("file")
        if "site-packages" in path or "/usr/lib/python" in path:
            continue
        hit.source_file = path
        hit.source_line = int(match.group("line"))
        return


def collect_failures(
    lines: list[PreparedLine],
    places: dict[int, LinePlace],
) -> tuple[FailureHit | None, list[FailureHit], list[FailureHit]]:
    """Return the primary failure, independent failures, and recovered failures."""
    hits: list[FailureHit] = []
    consumed: set[int] = set()
    for index, line in enumerate(lines):
        if line.number in consumed or not line.text.strip():
            continue
        place = places.get(line.number)
        command = None if place is None or place.association == "uncertain" else place.command
        hit = match_line(place.body if place else line.text, command)
        if hit is None:
            continue
        hit.log_line = line.number
        hit.detail_lines = _absorb(lines, index)
        consumed.update(hit.detail_lines)
        _apply_place(hit, place)
        if hit.command is None and hit.stage_association != "uncertain":
            hit.command, hit.command_line = _last_command(places, line.number, hit.stage)
        _copy_rust_location(hit, lines)
        _copy_frame(hit, lines)
        hits.append(hit)
    hits = _collapse_duplicates(_dedupe(hits))
    _mark_recovered(hits, lines, places)
    primary, others = _choose(hits)
    _attach_exit(primary, others, hits)
    recovered = [hit for hit in hits if hit.recovered]
    return primary, others, recovered
