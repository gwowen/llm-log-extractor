"""Patterns that recognise log records, stacks, and lifecycle lines."""
import re

# Spring-style records: timestamp, then a level. Stack frames have neither.
LOG_RECORD = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\s+"
    r"(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b",
    re.IGNORECASE,
)
# Runtime frames bury the application frame that names the failure.
FRAMEWORK_FRAME = re.compile(
    r"^\s+at (?:[\w.]+/)?"
    r"(?:java\.|javax\.|jdk\.|sun\.|com\.sun\.|"
    r"org\.springframework\.|org\.apache\.|jakarta\.|org\.hibernate\.)"
)
OMITTED_FRAMES = re.compile(r"^\s+\.\.\. \d+ (?:more|common frames)")
RUST_RUNTIME_FRAME = re.compile(
    r"^\s+\d+:\s+(?:rust_begin_unwind|core::|std::|alloc::|tokio::|"
    r"actix(?:_web|_http|_rt)?::)"
)
RUST_RUNTIME_AT = re.compile(r"^\s+at\s+/(?:rustc|cargo)/")
PYTHON_RUNTIME_FILE = re.compile(
    r'^\s*File "(?:[^"]*(?:site-packages|/usr/lib/python|/usr/local/lib/python)[^"]*)"'
)
# Warnings that do not describe a failure worth reading.
NOISY_WARNING = re.compile(
    r"(command=PING\b|\bheartbeat\b|\breadiness\b|status=404\b|"
    r"\bskip(?:ped|ping)?\b|\bunchanged\b|\bsuccessfully\b)",
    re.IGNORECASE,
)
# A later line on the same trace shows that the warning's operation recovered.
RECOVERED = re.compile(r"(?:\brecovered=true\b|\bretry succeeded\b)", re.IGNORECASE)
# Spring uses "Logger : message". Tracing, gunicorn, nginx, and systemd use "target: message".
RECORD_BODY = re.compile(
    r"^(?P<ts>\S+)\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"
    r"(?:\s+\d+\s+---\s+\[[^\]]*\]\s+\[[^\]]*\]\s+\[(?P<mdc>[^\]]*)\])?"
    r"\s+(?P<logger>.+?)\s*:\s+(?P<message>.*)$",
    re.IGNORECASE,
)
# Spring records use traceId=. Service logs use trace_id=.
TRACE_ID = re.compile(r"trace_?[Ii]d=([0-9a-fA-F]+)")
ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}
WARN_LEVELS = {"WARN", "WARNING"}
# Request and response lines are the useful context around an error.
CONTEXT_SIGNAL = re.compile(r"(\bIncoming\b|\bOutgoing\b|\bstatus=\d{3}\b)", re.IGNORECASE)
# SQL and cache chatter shares a trace id without describing the request.
ROUTINE_CONTEXT = re.compile(
    r"(JdbcTemplate|hibernate\.SQL|\bselect\b.+\bfrom\b|Cache lookup|"
    r"HikariPool|Executing prepared SQL|lettuce|Redis PONG)",
    re.IGNORECASE,
)
# Process death, a restart attempt, or the proxy in front of a dead process.
PROCESS_DEATH = re.compile(
    r"(exited with code|failed to boot|Shutting down: Master|"
    r"Main process exited|Failed with result|"
    r"code=killed|status=6/ABRT|Consumed .+ CPU time)",
    re.IGNORECASE,
)
PROCESS_RESTART = re.compile(
    r"(Scheduled restart|Start request repeated too quickly|Failed to start )",
    re.IGNORECASE,
)
UPSTREAM_FAILURE = re.compile(
    r"(connect\(\) failed \(111: Connection refused\)|"
    r"upstream prematurely closed|upstream_status=502)",
    re.IGNORECASE,
)
# A completed start. Warnings after this belong to the new process.
STARTED_SERVICE = re.compile(
    r"(?:\bStarted [\w.@+-]+\.service\b|\bApplication started\b|\bNow listening\b)",
    re.IGNORECASE,
)
SERVICE_UNIT = re.compile(r"\b([\w.@+-]+\.service)\b")
REQUEST_ID = re.compile(r"\brequest[_-]?[Ii]d=([A-Za-z0-9._:-]+)")
PID_VALUE = re.compile(r"\bpid[=:\s]+(\d+)\b", re.IGNORECASE)
JOB_ID = re.compile(r"\bjob[_-]?id=([A-Za-z0-9._:-]+)", re.IGNORECASE)
KEY_ID = re.compile(
    r"\b(?:key_id|selected_key_id|required_key_id|current_key_id)=(\S+)",
    re.IGNORECASE,
)
FIELD_ID = re.compile(r"\bfield=(\S+)")
COMPONENT_ID = re.compile(r"\bcomponent=(\S+)")
CALLBACK_ID = re.compile(r"\bcallback=([A-Za-z0-9_.]+)")
REVISION_ID = re.compile(r"\brevision=(\d+)")
DIFF_ARROW = re.compile(r"\b([A-Za-z0-9][\w./:-]{5,})\s*->\s*([A-Za-z0-9][\w./:-]{5,})")
NODE_RUNTIME = re.compile(r"node:internal|triggerUncaughtException|^\s*\^$|^Node\.js v")
DOTNET_RUNTIME = re.compile(
    r"^\s+at (?:System\.|Microsoft\.(?:Extensions|AspNetCore|EntityFrameworkCore)\.)"
)
RECOVERY_LINE = re.compile(
    r"(Application started|Now listening|\blistening (?:addr|host)=|"
    r"\breadiness=healthy\b|\bliveness=healthy\b)",
    re.IGNORECASE,
)
# Configuration changes, failed setup, and nil or missing state.
DIAGNOSTIC = re.compile(
    r"(change detected|initialization failed|invalid endpoint|"
    r"<nil>|client_present=false|tenant_id(?:_present)?=(?:false|None)|"
    r"soft_fail=true|swap=scheduled|"
    r"marking existing transaction as rollback-only|"
    r"\bnot found\b|=null\b|_present=false|matching_keys=0|"
    r"\bactivated\b|\bdiff\b)",
    re.IGNORECASE,
)
# A token or JWT check that succeeded. Kept only when the same call shows a claim gap.
AUTH_SUCCESS = re.compile(
    r"(\btoken validation ok\b|"
    r"\bsignature=valid\b|"
    r"(?:auth\.jwt|::auth::jwt).{0,80}\bvalidation[ =]ok\b|"
    r"\bvalidation[ =]ok\b.{0,80}\b(?:kid|subject|signature|scopes)=)",
    re.IGNORECASE,
)
# The call authenticated, but a tenant or required claim was not present.
AUTHZ_GAP = re.compile(
    r"(tenant_id(?:_present)?=(?:false|None)|"
    r"\bmust contain\b.{0,40}\bclaim\b)",
    re.IGNORECASE,
)
# Words too common to tie two lines to the same component.
GENERIC_TOKENS = {
    "api",
    "app",
    "client",
    "code",
    "com",
    "cmd",
    "debug",
    "error",
    "exception",
    "failed",
    "failure",
    "false",
    "github",
    "goroutine",
    "http",
    "https",
    "immediate",
    "info",
    "internal",
    "java",
    "level",
    "main",
    "none",
    "null",
    "order",
    "orders",
    "org",
    "panic",
    "request",
    "response",
    "runtime",
    "server",
    "service",
    "signal",
    "span",
    "start",
    "started",
    "status",
    "trace",
    "true",
    "uncaught",
    "warn",
    "warning",
    "active",
    "build",
    "present",
    "record",
    "result",
    "running",
    "update",
    "value",
}
EXCEPTION_SIGNATURE = re.compile(r"((?:\w+\.)*(?:Error|Exception):.+)$")
REPEAT_REQUEST = re.compile(r'request="[^"]*"')
REPEAT_PATH = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD)\s+\S+")
