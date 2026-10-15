"""Deterministic incident reports built without a model."""

import unittest
from pathlib import Path

from log_and_order.incidents.builder import build_incident_report
from log_and_order.logs.extraction import extract_events
from log_and_order.logs.selection import select_primary

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"

EXPECTED = {
    "spring_boot_incident_noise.log": ("java.lang.NoSuchMethodError", "no_termination_observed"),
    "django_rest_crash_noisy.log": ("ModuleNotFoundError", "restart_attempted"),
    "gin_web_crash_noisy.log": ("panic", "restarted"),
    "actix_web_crash_noisy.log": ("panic", "restarted"),
    "nodejs_express_crash_noisy.log": ("TypeError", "restarted"),
    "aspnet_core_crash_noisy.log": ("System.NullReferenceException", "restarted"),
}


class IncidentBuildTest(unittest.TestCase):
    def test_fixtures_keep_exception_outcome_and_ordered_evidence(self) -> None:
        for name, (exception, outcome) in EXPECTED.items():
            with self.subTest(name=name):
                text = (LOGS / name).read_text()
                events = extract_events(text)
                assert events is not None
                primary = select_primary(events)
                assert primary is not None
                report = build_incident_report(Path(name), text, events)
                self.assertEqual(report.incident_id, f"E{primary.seed.number}")
                self.assertEqual(report.exception, exception)
                self.assertEqual(report.outcome, outcome)
                self.assertEqual(report.source, name)
                timeline = [moment.log_line for moment in report.timeline]
                self.assertEqual(timeline, sorted(timeline))
                failure = next(line for line in report.lines if line.role == "failure")
                self.assertEqual(failure.number, primary.seed.number)
                self.assertEqual(failure.correlation, "failure")
                self.assertNotEqual(failure.number, report.locations[0].source_line)

    def test_a_terminating_failure_outranks_an_earlier_error(self) -> None:
        text = "\n".join(
            [
                "2026-10-15T10:00:00.000Z INFO app: starting",
                "2026-10-15T10:00:00.100Z INFO app: listening",
                "2026-10-15T10:00:00.200Z INFO app: ready",
                "2026-10-15T10:00:01.000Z ERROR billing.api: PaymentException: gateway timeout trace_id=aaa111",
                "2026-10-15T10:00:01.500Z ERROR orders.api: IllegalStateException: invalid state trace_id=bbb222",
                "2026-10-15T10:00:01.800Z ERROR systemd[1]: billing.service: Main process exited, code=exited, status=1",
                "2026-10-15T10:00:02.000Z WARN nginx.upstream: connect() failed (111: Connection refused) while connecting to upstream=127.0.0.1:8080",
            ]
        )
        events = extract_events(text)
        assert events is not None
        report = build_incident_report(Path("billing.log"), text, events)
        self.assertIn("PaymentException", report.summary)
        self.assertEqual(report.outcome, "terminated")


if __name__ == "__main__":
    unittest.main()
