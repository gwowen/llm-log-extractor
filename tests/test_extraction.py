"""Unit tests for deterministic log extraction."""

import io
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path

import main

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"


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
        text = (LOGS / "gin_web_crash_noisy.log").read_text()
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
        events = main.extract_events((LOGS / "gin_web_crash_noisy.log").read_text())
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
        events = main.extract_events((LOGS / "spring_boot_incident_noise.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 616)
        numbers = line_numbers(failure)
        self.assertIn(611, numbers)
        self.assertIn(649, numbers)
        self.assertIn(663, numbers)
        self.assertNotIn(610, numbers)

    def test_django_boot_loop_stays_one_incident(self) -> None:
        events = main.extract_events((LOGS / "django_rest_crash_noisy.log").read_text())
        self.assertIsNotNone(events)
        assert events is not None
        failure = event_with(events, 3252)
        numbers = line_numbers(failure)
        self.assertIn(3295, numbers)
        self.assertIn(3288, numbers)
        self.assertIn(3314, numbers)


class StackAndLifecycleTest(unittest.TestCase):
    def test_spring_keeps_the_application_frame(self) -> None:
        events = main.extract_events((LOGS / "spring_boot_incident_noise.log").read_text())
        assert events is not None
        failure = event_with(events, 616)
        stack = "\n".join(line.raw for line in failure.stack)
        self.assertIn("NoSuchMethodError", stack)
        self.assertIn("OrderEventFactory.java:87", stack)
        self.assertNotIn("DispatcherServlet.java", stack)
        self.assertEqual(main.select_primary(events), failure)

    def test_django_boot_failure_includes_worker_exit_and_start_failure(self) -> None:
        events = main.extract_events((LOGS / "django_rest_crash_noisy.log").read_text())
        assert events is not None
        failure = event_with(events, 3252)
        text = "\n".join(line.raw for line in failure.lines())
        self.assertIn("ModuleNotFoundError: No module named 'django_filters'", text)
        self.assertIn("Worker (pid:21601) exited with code 3", text)
        self.assertIn("Failed to start Meridian REST API.", text)
        self.assertIn("wsgi.py", text)
        self.assertNotIn("site-packages/gunicorn/arbiter.py", text)

    def test_actix_panic_keeps_the_abort_and_the_tenant_frame(self) -> None:
        events = main.extract_events((LOGS / "actix_web_crash_noisy.log").read_text())
        assert events is not None
        failure = event_with(events, 3599)
        text = "\n".join(line.raw for line in failure.lines())
        self.assertIn("tenant_id_present=false", text)
        self.assertIn("tenant.rs:126", text)
        self.assertIn("status=6/ABRT", text)
        self.assertIn("Started northstar-api.service", text)
        self.assertNotIn("rust_begin_unwind", text)

    def test_nodejs_rejection_keeps_the_job_and_the_process_exit(self) -> None:
        events = main.extract_events((LOGS / "nodejs_express_crash_noisy.log").read_text())
        assert events is not None
        failure = event_with(events, 3525)
        numbers = line_numbers(failure)
        self.assertIn(3512, numbers)
        self.assertIn(3515, numbers)
        self.assertIn(3538, numbers)
        self.assertNotIn(2105, numbers)
        stack = "\n".join(line.raw for line in failure.stack)
        self.assertIn("address.js:76", stack)
        self.assertNotIn("node:internal", stack)
        self.assertIs(main.select_primary(events), failure)

    def test_aspnet_background_failure_keeps_key_setup_and_systemd(self) -> None:
        events = main.extract_events((LOGS / "aspnet_core_crash_noisy.log").read_text())
        assert events is not None
        failure = event_with(events, 3524)
        numbers = line_numbers(failure)
        for number in (3460, 3461, 3464, 3465, 3466, 3521, 3522, 3523, 3532, 3534, 3538):
            self.assertIn(number, numbers)
        self.assertNotIn(2085, numbers)
        self.assertNotIn(3543, numbers)
        self.assertIsNone(failure.seed.trace_id)
        self.assertIsNone(failure.seed.request_id)
        stack = "\n".join(line.raw for line in failure.stack)
        self.assertIn("KeyRingBuilder.cs:line 118", stack)
        self.assertNotIn("System.Threading", stack)
        self.assertIs(main.select_primary(events), failure)
        independent = main.independent_events(events, failure)
        self.assertTrue(any(event.seed.number == 3543 for event in independent))
        self.assertFalse(any(event.seed.number == 3465 for event in independent))


class GroupingRulesTest(unittest.TestCase):
    def test_primary_is_the_terminating_failure_not_an_earlier_error(self) -> None:
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
        assert events is not None
        primary = main.select_primary(events)
        assert primary is not None
        self.assertIn("PaymentException", primary.seed.raw)
        orders = next(event for event in events if "IllegalStateException" in event.seed.raw)
        self.assertIn(orders, main.independent_events(events, primary))

    def test_same_exception_after_a_start_is_a_new_incident(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T08:00:00.000Z INFO app: starting",
                    "2026-10-15T08:00:00.100Z INFO app: listening",
                    "2026-10-15T08:00:00.200Z INFO app: ready",
                    "2026-10-15T08:00:01.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                    "2026-10-15T08:00:02.000Z INFO systemd[1]: Started app.service",
                    "2026-10-15T08:00:10.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                ]
            )
        )
        assert events is not None
        failures = [event for event in events if "Job failed" in event.seed.raw]
        self.assertEqual(len(failures), 2)

    def test_application_502_after_recovery_stays_independent(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T10:00:00.000Z INFO app: starting",
                    "2026-10-15T10:00:00.100Z INFO app: listening",
                    "2026-10-15T10:00:00.200Z INFO app: ready",
                    "2026-10-15T10:00:01.000Z FATAL app.worker: pid=10 uncaught panic component=worker.loop",
                    "panic: nil pointer",
                    "2026-10-15T10:00:01.100Z ERROR systemd[1]: app.service: Main process exited, code=exited, status=2 (pid=10)",
                    "2026-10-15T10:00:01.200Z ERROR nginx.error: connect() failed (111: Connection refused) while connecting to upstream status=502",
                    "2026-10-15T10:00:02.000Z INFO systemd[1]: Started app.service",
                    "2026-10-15T10:00:20.000Z ERROR app.http: method=GET path=/api/v1/payments status=502 reason=provider_unavailable requestId=req-later",
                ]
            )
        )
        assert events is not None
        failure = next(event for event in events if "worker.loop" in event.seed.raw)
        self.assertTrue(any("Connection refused" in line.raw for line in failure.consequences))
        later = next(event for event in events if "req-later" in event.seed.raw)
        self.assertNotIn(later.seed.number, line_numbers(failure))
        self.assertIn(later, main.independent_events(events, failure))

    def test_proxy_failure_outside_the_window_stays_independent(self) -> None:
        events = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T10:00:00.000Z INFO app: starting",
                    "2026-10-15T10:00:00.100Z INFO app: listening",
                    "2026-10-15T10:00:00.200Z INFO app: ready",
                    "2026-10-15T10:00:01.000Z ERROR app.worker: Job failed",
                    "java.lang.IllegalStateException: disk full",
                    "\tat com.example.app.Worker.run(Worker.java:10)",
                    "2026-10-15T10:00:01.100Z ERROR systemd[1]: app.service: Main process exited, code=exited, status=1",
                    "2026-10-15T10:02:00.000Z ERROR nginx.error: connect() failed (111: Connection refused) while connecting to upstream status=502",
                ]
            )
        )
        assert events is not None
        failure = next(event for event in events if "disk full" in "".join(line.raw for line in failure_stack(event)))
        self.assertFalse(any("Connection refused" in line.raw for line in failure.lines()))

    def test_lookback_drops_setup_outside_the_bound(self) -> None:
        text = "\n".join(
            [
                "2026-10-15T10:00:00.000Z INFO app: starting",
                "2026-10-15T10:00:00.100Z INFO app: listening",
                "2026-10-15T10:00:00.200Z INFO app: ready",
                "2026-10-15T10:00:00.300Z INFO config.watcher: change detected key=orders/pricing-rules revision=1",
                "2026-10-15T10:00:40.000Z INFO config.watcher: change detected key=orders/pricing-rules revision=2",
                "2026-10-15T10:00:50.000Z ERROR runtime.panic: uncaught panic component=rules.refresh_loop",
                "panic: nil pointer",
            ]
        )
        events = main.extract_events(text)
        assert events is not None
        failure = events[0]
        context = "\n".join(line.raw for line in failure.context)
        self.assertIn("revision=2", context)
        self.assertNotIn("revision=1", context)
        bounded = main.extract_events(
            "\n".join(
                [
                    "2026-10-15T10:00:00.000Z INFO app: starting",
                    "2026-10-15T10:00:00.100Z INFO app: listening",
                    "2026-10-15T10:00:00.200Z INFO app: ready",
                    "2026-10-15T10:00:40.000Z INFO config.watcher: change detected key=orders/pricing-rules revision=2",
                    "2026-10-15T10:00:50.000Z ERROR runtime.panic: uncaught panic component=rules.refresh_loop",
                    "panic: nil pointer",
                ]
            ),
            lookback=timedelta(seconds=5),
        )
        assert bounded is not None
        self.assertFalse(any("revision=2" in line.raw for line in bounded[0].context))

    def test_no_serious_failure_has_no_primary_incident(self) -> None:
        quiet = "\n".join(
            f"2026-10-15T10:00:0{index}.000Z INFO app: note {index}" for index in range(5)
        )
        self.assertEqual(main.extract_events(quiet + "\n2026-10-15T10:00:06.000Z WARN app: command=PING ok"), [])
        warned = main.extract_events(
            quiet + "\n2026-10-15T10:00:06.000Z WARN otel.exporter: collector export rejected status=429 batch=1"
        )
        assert warned is not None
        self.assertIsNone(main.select_primary(warned))
        report = main.fallback_report(None, warned)
        self.assertIn("Main failure\nNone", report)
        self.assertIn("status=429", report)


class ReportAssemblyTest(unittest.TestCase):
    def test_model_input_omits_independent_events(self) -> None:
        events = main.extract_events((LOGS / "aspnet_core_crash_noisy.log").read_text())
        assert events is not None
        primary = main.select_primary(events)
        assert primary is not None
        model_input = main.format_model_input(primary)
        self.assertIn("Event E3524", model_input)
        self.assertIn("NullReferenceException", model_input)
        self.assertIn("confirmed 3460", model_input)
        self.assertIn("Main process exited", model_input)
        self.assertNotIn("2085", model_input)
        self.assertNotIn("3543", model_input)

    def test_compose_report_replaces_a_model_written_warning_section(self) -> None:
        narrative = "Main failure\nThe worker stopped (line 4).\n\nRelated events\nNone\n\nOther warnings\nThe model invented this."
        report = main.compose_report(narrative, [])
        self.assertIn("Main failure", report)
        self.assertNotIn("invented", report)
        self.assertTrue(report.strip().endswith("Other warnings\nNone"))

    def test_fallback_cites_extracted_lines_only(self) -> None:
        events = main.extract_events((LOGS / "aspnet_core_crash_noisy.log").read_text())
        assert events is not None
        primary = main.select_primary(events)
        assert primary is not None
        independent = main.independent_events(events, primary)
        report = main.fallback_report(primary, independent)
        self.assertIn("Main failure", report)
        self.assertIn("Related events", report)
        self.assertIn("Other warnings", report)
        self.assertIn("3524", report)
        self.assertIn("3532", report)
        main_section, _, warnings = report.partition("Other warnings")
        self.assertNotIn("2085", main_section)
        self.assertIn("2085", warnings)

    def test_dump_events_prints_groups_without_the_model(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = main.main(["--dump-events", str(LOGS / "aspnet_core_crash_noisy.log")])
        self.assertEqual(status, 0)
        text = buffer.getvalue()
        model, _, reported = text.partition("# Reported by Python")
        self.assertIn("# Sent to the model", model)
        self.assertIn("E3524", model)
        self.assertIn("3460", model)
        self.assertNotIn("2085", model)
        self.assertIn("2085", reported)
        self.assertIn("3543", reported)


def failure_stack(event: main.Event) -> list[main.LogLine]:
    return [event.seed, *event.stack]


if __name__ == "__main__":
    unittest.main()
