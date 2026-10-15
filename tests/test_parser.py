"""Parse timestamped records and decide which stack lines to keep."""

import unittest
from datetime import datetime, timezone

from log_and_order.logs.parser import keep_stack_line, parse_line, parse_moment


class ParserTest(unittest.TestCase):
    def test_spring_record_keeps_level_trace_and_request(self) -> None:
        raw = (
            "2026-10-15T10:00:01.010Z ERROR com.example.App : "
            "boom traceId=abc123 requestId=req-9"
        )
        line = parse_line(4, raw)
        self.assertEqual(line.number, 4)
        self.assertTrue(line.is_record)
        self.assertEqual(line.level, "ERROR")
        self.assertEqual(line.timestamp, "2026-10-15T10:00:01.010Z")
        self.assertEqual(line.trace_id, "abc123")
        self.assertEqual(line.request_id, "req-9")
        self.assertEqual(line.logger, "com.example.App")
        self.assertEqual(
            line.moment,
            datetime(2026, 10, 15, 10, 0, 1, 10000, tzinfo=timezone.utc),
        )

    def test_service_record_reads_trace_id_and_request_id(self) -> None:
        raw = (
            "2026-10-15T10:00:01.000Z INFO app.http: "
            "Incoming GET /usage trace_id=def456 request_id=req-2"
        )
        line = parse_line(1, raw)
        self.assertEqual(line.level, "INFO")
        self.assertEqual(line.logger, "app.http")
        self.assertEqual(line.trace_id, "def456")
        self.assertEqual(line.request_id, "req-2")

    def test_stack_continuation_is_not_a_record(self) -> None:
        line = parse_line(2, "\tat com.example.App.handle(App.java:10)")
        self.assertEqual(line.number, 2)
        self.assertFalse(line.is_record)
        self.assertEqual(line.level, "")
        self.assertIsNone(line.moment)

    def test_unparseable_timestamp_stays_empty(self) -> None:
        self.assertIsNone(parse_moment(""))
        self.assertIsNone(parse_moment("not-a-time"))

    def test_runtime_frames_are_dropped_and_application_frames_stay(self) -> None:
        self.assertFalse(keep_stack_line("\tat java.lang.Thread.run(Thread.java:1)"))
        self.assertFalse(
            keep_stack_line(
                "\tat org.springframework.web.servlet.FrameworkServlet.doPost(FrameworkServlet.java:1)"
            )
        )
        self.assertFalse(
            keep_stack_line(
                '  File "/usr/lib/python3.12/site-packages/gunicorn/arbiter.py", line 1, in run'
            )
        )
        self.assertFalse(keep_stack_line("  12: rust_begin_unwind"))
        self.assertFalse(keep_stack_line("    at node:internal/process/execution:1:1"))
        self.assertFalse(keep_stack_line("   at System.Threading.Tasks.Task.Execute()"))
        self.assertTrue(keep_stack_line("\tat com.example.App.handle(App.java:10)"))
        self.assertTrue(keep_stack_line("java.lang.NullPointerException: x"))


if __name__ == "__main__":
    unittest.main()
