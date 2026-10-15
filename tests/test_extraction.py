"""Event extraction from timestamped logs."""

import unittest
from pathlib import Path

from log_and_order.logs.extraction import extract_events
from log_and_order.models import Event
from log_and_order.reports.formatting import format_events

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"


def line_numbers(event: Event) -> set[int]:
    numbers = {line.number for line in event.lines()}
    numbers.update(event.repeat_numbers)
    return numbers


def event_with(events: list[Event], number: int) -> Event:
    return next(event for event in events if number in line_numbers(event))



class RepeatCollapseTest(unittest.TestCase):
    def test_repeats_in_one_incident_keep_both_timestamps(self) -> None:
        events = extract_events(
            "\n".join(
                [
                    "2026-10-15T10:00:00.000Z INFO app: starting",
                    "2026-10-15T10:00:00.100Z INFO app: listening",
                    "2026-10-15T10:00:00.200Z INFO app: ready",
                    "2026-10-15T10:00:01.000Z WARN otel.exporter: collector export rejected status=429 batch=1142",
                    "2026-10-15T10:00:03.000Z WARN otel.exporter: collector export rejected status=429 batch=1142",
                    "2026-10-15T10:00:04.000Z INFO app: still up",
                ]
            )
        )
        self.assertIsNotNone(events)
        assert events is not None
        warning = next(event for event in events if event.seed.level == "WARN")
        rendered = format_events([warning])
        self.assertIn("2 times", rendered)
        self.assertIn("2026-10-15T10:00:01.000Z", rendered)
        self.assertIn("2026-10-15T10:00:03.000Z", rendered)
        self.assertIn("lines 4-5", rendered)

    def test_repeated_warning_after_restart_is_a_new_incident(self) -> None:
        events = extract_events(
            "\n".join(
                [
                    "2026-10-15T10:00:00.000Z INFO app: starting",
                    "2026-10-15T10:00:00.100Z INFO app: listening",
                    "2026-10-15T10:00:01.000Z WARN rules.client: client initialization failed: invalid endpoint value=\"\" field=pricing.rules_endpoint",
                    "2026-10-15T10:00:02.000Z WARN rules.client: client initialization failed: invalid endpoint value=\"\" field=pricing.rules_endpoint",
                    "2026-10-15T10:00:03.000Z INFO systemd[1]: Started harborline-orders-api.service",
                    "2026-10-15T10:00:04.000Z WARN rules.client: client initialization failed: invalid endpoint value=\"\" field=pricing.rules_endpoint",
                ]
            )
        )
        self.assertIsNotNone(events)
        assert events is not None
        warnings = [event for event in events if "initialization failed" in event.seed.raw]
        self.assertEqual(len(warnings), 2)
        rendered = format_events(warnings)
        self.assertIn("2 times", rendered)
        self.assertIn("2026-10-15T10:00:01.000Z to 2026-10-15T10:00:02.000Z", rendered)
        self.assertEqual(warnings[1].repeat_lines[0].timestamp, "2026-10-15T10:00:04.000Z")
        self.assertNotIn("2026-10-15T10:00:04.000Z", format_events([warnings[0]]))



class FixtureRegressionTest(unittest.TestCase):
    def test_spring_request_lines_stay_with_the_exception(self) -> None:
        events = extract_events((LOGS / "spring_boot_incident_noise.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 616)
        numbers = line_numbers(failure)
        self.assertIn(611, numbers)
        self.assertIn(649, numbers)
        self.assertIn(663, numbers)
        self.assertNotIn(610, numbers)

    def test_django_boot_loop_stays_one_incident(self) -> None:
        events = extract_events((LOGS / "django_rest_crash_noisy.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 3252)
        numbers = line_numbers(failure)
        self.assertIn(3295, numbers)
        self.assertIn(3288, numbers)
        self.assertIn(3314, numbers)



if __name__ == '__main__':
    unittest.main()
