"""Application source locations stay distinct from log line numbers."""

import unittest
from pathlib import Path

from log_and_order.incidents.stacktraces import application_locations
from log_and_order.logs.extraction import extract_events
from log_and_order.logs.selection import select_primary
from log_and_order.models import Event, LogLine

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"


class StacktraceTest(unittest.TestCase):
    def test_a_dropped_framework_frame_is_not_the_throw_site(self) -> None:
        covered = "\n".join(
            [
                "2026-10-15T00:00:00Z ERROR app: boom",
                "java.lang.NullPointerException: x",
                "\tat org.springframework.web.servlet.FrameworkServlet.doPost(FrameworkServlet.java:1)",
                "\tat com.example.App.handle(App.java:10)",
            ]
        )
        bare = "\n".join(
            [
                "2026-10-15T00:00:00Z ERROR app: boom",
                "java.lang.NullPointerException: x",
                "\tat com.example.App.handle(App.java:10)",
            ]
        )
        seed = LogLine(number=1, raw=covered.splitlines()[0], level="ERROR", is_record=True)
        covered_locations = application_locations(covered, Event(seed=seed))
        bare_locations = application_locations(bare, Event(seed=seed))
        self.assertEqual(covered_locations[0].role, "innermost")
        self.assertEqual(covered_locations[0].path, "App.java")
        self.assertEqual(covered_locations[0].source_line, 10)
        self.assertEqual(covered_locations[0].log_line, 4)
        self.assertNotEqual(covered_locations[0].source_line, covered_locations[0].log_line)
        self.assertEqual(bare_locations[0].role, "throw_site")

    def test_fixture_locations_use_source_lines(self) -> None:
        expected = {
            "spring_boot_incident_noise.log": (87, "OrderEventFactory.java"),
            "gin_web_crash_noisy.log": (94, "client.go"),
            "nodejs_express_crash_noisy.log": (76, "address.js"),
            "actix_web_crash_noisy.log": (126, "tenant.rs"),
            "aspnet_core_crash_noisy.log": (118, "KeyRingBuilder.cs"),
            "django_rest_crash_noisy.log": (16, "wsgi.py"),
        }
        for name, (source_line, path_tail) in expected.items():
            with self.subTest(name=name):
                text = (LOGS / name).read_text()
                events = extract_events(text)
                assert events is not None
                primary = select_primary(events)
                assert primary is not None
                locations = application_locations(text, primary)
                self.assertTrue(locations)
                match = next(item for item in locations if item.source_line == source_line)
                self.assertTrue(match.path.endswith(path_tail))
                self.assertNotEqual(match.log_line, match.source_line)


if __name__ == "__main__":
    unittest.main()
