"""Unit tests for deterministic log extraction."""

import unittest
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parents[1]


def line_numbers(event: main.Event) -> set[int]:
    numbers = {line.number for line in event.lines()}
    numbers.update(event.repeat_numbers)
    return numbers


def event_with(events: list[main.Event], number: int) -> main.Event:
    return next(event for event in events if number in line_numbers(event))


class CascadeGroupingTest(unittest.TestCase):
    def test_unrelated_errors_close_together_stay_separate(self) -> None:
        events = main.extract_events(
            "\n".join(
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
        )
        self.assertIsNotNone(events)
        assert events is not None
        billing = next(event for event in events if "PaymentException" in event.seed.raw)
        orders = next(event for event in events if "IllegalStateException" in event.seed.raw)
        self.assertNotIn(orders.seed.number, line_numbers(billing))
        self.assertNotIn(billing.seed.number, line_numbers(orders))
        self.assertTrue(any("Main process exited" in line.raw for line in billing.lines()))
        self.assertFalse(any("Main process exited" in line.raw for line in orders.lines()))

    def test_identical_exceptions_hours_apart_stay_separate(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T08:00:00.000Z INFO app: starting",
                    "2026-10-15T08:00:00.100Z INFO app: listening",
                    "2026-10-15T08:00:00.200Z INFO app: ready",
                    "2026-10-15T08:00:01.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                    "2026-10-15T10:00:01.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                    "2026-10-15T10:00:01.200Z WARN nginx.upstream: connect() failed (111: Connection refused) while connecting to upstream=127.0.0.1:9000",
                ]
            )
        )
        self.assertIsNotNone(events)
        assert events is not None
        failures = [event for event in events if "Job failed" in event.seed.raw]
        self.assertEqual(len(failures), 2)
        self.assertLess(failures[0].seed.number, failures[1].seed.number)
        self.assertFalse(any("Connection refused" in line.raw for line in failures[0].lines()))
        self.assertTrue(any("Connection refused" in line.raw for line in failures[1].lines()))

    def test_identical_exceptions_seconds_apart_stay_one_incident(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T08:00:00.000Z INFO app: starting",
                    "2026-10-15T08:00:00.100Z INFO app: listening",
                    "2026-10-15T08:00:00.200Z INFO app: ready",
                    "2026-10-15T08:00:01.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                    "2026-10-15T08:00:03.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                ]
            )
        )
        self.assertIsNotNone(events)
        assert events is not None
        failures = [event for event in events if "disk full" in event.seed.raw or any("disk full" in line.raw for line in event.stack)]
        self.assertEqual(len(failures), 1)

    def test_panic_keeps_systemd_and_proxy_lines(self) -> None:
        text = (ROOT / "gin_web_crash_noisy.log").read_text()
        events = main.extract_events(text)
        self.assertIsNotNone(events)
        assert events is not None
        panic = event_with(events, 3465)
        numbers = line_numbers(panic)
        self.assertIn(3478, numbers)
        self.assertIn(3480, numbers)
        self.assertIn(3481, numbers)
        later = event_with(events, 3487)
        self.assertNotEqual(panic.seed.number, later.seed.number)


class PreludeContextTest(unittest.TestCase):
    def test_debug_context_before_a_crash_is_kept(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T10:34:00.000Z INFO app: starting",
                    "2026-10-15T10:34:00.100Z INFO app: listening",
                    "2026-10-15T10:34:00.200Z INFO app: ready",
                    "2026-10-15T10:34:00.300Z INFO app: serving",
                    "2026-10-15T10:34:01.000Z INFO config.watcher: change detected source=consul key=orders/pricing-rules revision=174",
                    "2026-10-15T10:34:01.100Z DEBUG storage.postgres: query=SELECT pool=primary elapsed_ms=2",
                    "2026-10-15T10:34:01.200Z DEBUG rules.registry: revision=174 active_client=<nil> watcher=subscribed",
                    "2026-10-15T10:34:01.400Z DEBUG rules.refresh: tick=periodic client_present=false",
                    "2026-10-15T10:34:01.500Z ERROR runtime.panic: uncaught panic component=rules.refresh_loop signal=SIGSEGV",
                    "panic: runtime error: invalid memory address or nil pointer dereference",
                ]
            )
        )
        self.assertIsNotNone(events)
        assert events is not None
        panic = next(event for event in events if "rules.refresh_loop" in event.seed.raw)
        text = "\n".join(line.raw for line in panic.lines())
        self.assertIn("change detected", text)
        self.assertIn("active_client=<nil>", text)
        self.assertIn("client_present=false", text)
        self.assertNotIn("query=SELECT", text)
        self.assertEqual(panic.context[0].number, 5)

    def test_gin_panic_keeps_config_and_nil_client_lines(self) -> None:
        events = main.extract_events((ROOT / "gin_web_crash_noisy.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        panic = event_with(events, 3465)
        numbers = line_numbers(panic)
        for number in (3430, 3432, 3434, 3436, 3463, 3464):
            self.assertIn(number, numbers)
        self.assertNotIn(3462, numbers)
        rendered = main.format_events(events)
        self.assertNotIn("2 times, lines 3432,3487", rendered)
        self.assertIn("3432", rendered)
        self.assertIn("3487", rendered)


class RepeatCollapseTest(unittest.TestCase):
    def test_repeats_in_one_incident_keep_both_timestamps(self) -> None:
        events = main.extract_events(
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
        rendered = main.format_events([warning])
        self.assertIn("2 times", rendered)
        self.assertIn("2026-10-15T10:00:01.000Z", rendered)
        self.assertIn("2026-10-15T10:00:03.000Z", rendered)
        self.assertIn("lines 4-5", rendered)

    def test_repeated_warning_after_restart_is_a_new_incident(self) -> None:
        events = main.extract_events(
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
        rendered = main.format_events(warnings)
        self.assertIn("2 times", rendered)
        self.assertIn("2026-10-15T10:00:01.000Z to 2026-10-15T10:00:02.000Z", rendered)
        self.assertEqual(warnings[1].repeat_lines[0].timestamp, "2026-10-15T10:00:04.000Z")
        self.assertNotIn("2026-10-15T10:00:04.000Z", main.format_events([warnings[0]]))


class FixtureRegressionTest(unittest.TestCase):
    def test_spring_request_lines_stay_with_the_exception(self) -> None:
        events = main.extract_events((ROOT / "spring_boot_incident_noise.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 616)
        numbers = line_numbers(failure)
        self.assertIn(611, numbers)
        self.assertIn(649, numbers)
        self.assertIn(663, numbers)
        self.assertNotIn(610, numbers)

    def test_django_boot_loop_stays_one_incident(self) -> None:
        events = main.extract_events((ROOT / "django_rest_crash_noisy.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 3252)
        numbers = line_numbers(failure)
        self.assertIn(3295, numbers)
        self.assertIn(3288, numbers)
        self.assertIn(3314, numbers)


if __name__ == "__main__":
    unittest.main()
