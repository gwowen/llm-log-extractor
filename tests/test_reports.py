"""Agent and JSON reports, built without the model."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from log_and_order.cli import build_parser, main
from log_and_order.incidents.stacktraces import application_locations
from log_and_order.llm.prompts import REPORT_PROMPT
from log_and_order.logs.extraction import extract_events
from log_and_order.logs.selection import independent_events, select_primary
from log_and_order.models import Event, LogLine
from log_and_order.pipeline import format_agent_report, format_json_report
from log_and_order.reports.human import compose_report, fallback_report, format_model_input

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"

HEADINGS = (
    "## Executive summary",
    "## Task for coding agent",
    "## Primary failure",
    "## Failure mechanism",
    "## Evidence and observations",
    "## Ranked hypotheses",
    "## Recommended investigation",
    "## Regression tests to consider",
    "## Unknowns and limitations",
    "## Chronological timeline",
    "## Supporting details",
    "### Application stack",
    "### Log excerpts",
    "### Other warnings",
)


def section(report: str, heading: str, following: str) -> str:
    return report.split(heading, 1)[1].split(following, 1)[0]


def load(name: str) -> tuple[str, list[Event]]:
    text = (LOGS / name).read_text()
    events = extract_events(text)
    assert events is not None
    return text, events


class ReportShapeTest(unittest.TestCase):
    def test_default_format_is_human(self) -> None:
        args = build_parser().parse_args(["application.log"])
        self.assertEqual(args.report_format, "human")
        self.assertIsNone(args.output)
        self.assertFalse(args.no_llm)
        self.assertIsNone(args.model)

    def test_sections_lead_with_the_summary_and_do_not_repeat_it(self) -> None:
        text, events = load("spring_boot_incident_noise.log")
        report = format_agent_report(Path("spring.log"), text, events)
        indexes = [report.index(heading) for heading in HEADINGS]
        self.assertEqual(indexes, sorted(indexes))
        summary = section(report, "## Executive summary", "## Task for coding agent")
        timeline = section(report, "## Chronological timeline", "## Supporting details")
        for sentence in summary.split("."):
            sentence = sentence.strip()
            if len(sentence) > 40:
                self.assertNotIn(sentence, timeline)
        detail = report.split("## Supporting details", 1)[1]
        self.assertIn("### Application stack", detail)
        self.assertIn("### Other warnings", detail)
        self.assertLess(report.index("## Evidence and observations"), report.index("## Supporting details"))
        self.assertIn("## Task for coding agent", report)
        self.assertIn("not facts", report)

    def test_spring_keeps_source_lines_apart_from_log_lines(self) -> None:
        text, events = load("spring_boot_incident_noise.log")
        report = format_agent_report(Path("spring.log"), text, events)
        payload = json.loads(format_json_report(Path("spring.log"), text, events))
        self.assertNotIn("orders-domain-4.12.1", report)
        self.assertNotIn("orders-domain-4.12.1", json.dumps(payload))
        failure = section(report, "## Primary failure", "## Failure mechanism")
        self.assertIn("source line 87", failure)
        self.assertIn("Log line 649", failure)
        self.assertNotIn("source line 649", failure)
        self.assertIn("OrderEventFactory.toConfirmedEvent", failure)
        location = payload["locations"][0]
        self.assertEqual(location["role"], "throw_site")
        self.assertEqual(location["source_line"], 87)
        self.assertEqual(location["log_line"], 649)
        self.assertEqual(payload["incident"]["exception"], "java.lang.NoSuchMethodError")
        self.assertEqual(payload["incident"]["outcome"], "no_termination_observed")
        self.assertIn("No process termination was identified", payload["incident"]["summary"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertFalse(payload["analysis"]["enriched"])
        self.assertTrue(payload["analysis"]["fallback"])
        self.assertEqual(payload["analysis"]["hypotheses"], [])
        self.assertEqual(payload["evidence"]["possible_explanations"], [])
        failure_line = next(
            line for line in payload["evidence"]["lines"] if line["role"] == "failure"
        )
        self.assertNotEqual(failure_line["number"], location["source_line"])
        self.assertEqual(failure_line["correlation"], "failure")
        cited = next(line for line in payload["evidence"]["lines"] if "NoSuchMethodError" in line["text"])
        self.assertEqual(cited["correlation"], "failure")
        self.assertEqual(payload["incident"]["service"], "orders-api")
        self.assertEqual(payload["incident"]["language"], "Java")
        self.assertEqual(payload["incident"]["framework"], "Spring Boot")
        self.assertIn("which copy of the class", section(report, "## Unknowns and limitations", "## Chronological timeline"))
        self.assertIn(payload["incident"]["summary"], report)
        callers = [item["symbol"] for item in payload["locations"] if item["role"] == "caller"]
        self.assertIn("com.riverton.orders.application.OrderConfirmationService.confirm", callers)
        self.assertIn("com.riverton.orders.web.OrderController.confirm", callers)

    def test_dropped_framework_frame_is_not_called_the_throw_site(self) -> None:
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
        self.assertEqual(bare_locations[0].role, "throw_site")

    def test_gin_timeline_orders_the_nil_client_before_the_restart(self) -> None:
        text, events = load("gin_web_crash_noisy.log")
        report = format_agent_report(Path("gin.log"), text, events)
        timeline = section(report, "## Chronological timeline", "## Supporting details")
        nil = timeline.index("client_present=false")
        panic = timeline.index("/app/internal/rules/client.go:94")
        exited = timeline.index("The process exited.")
        restart = timeline.index("A restart was attempted.")
        self.assertLess(nil, panic)
        self.assertLess(panic, exited)
        self.assertLess(exited, restart)
        payload = json.loads(format_json_report(Path("gin.log"), text, events))
        self.assertEqual(payload["incident"]["exception"], "panic")
        self.assertEqual(payload["incident"]["outcome"], "restarted")
        self.assertEqual(payload["incident"]["language"], "Go")
        thrown = payload["locations"][0]
        self.assertEqual(thrown["role"], "throw_site")
        self.assertEqual(thrown["source_line"], 94)
        self.assertIn("(*Client).Refresh", thrown["symbol"])

    def test_node_keeps_the_column(self) -> None:
        text, events = load("nodejs_express_crash_noisy.log")
        payload = json.loads(format_json_report(Path("node.log"), text, events))
        thrown = payload["locations"][0]
        self.assertEqual(thrown["source_line"], 76)
        self.assertEqual(thrown["source_column"], 29)
        self.assertTrue(thrown["path"].endswith("address.js"))
        self.assertNotEqual(thrown["log_line"], 76)
        self.assertEqual(payload["incident"]["exception"], "TypeError")
        self.assertEqual(payload["evidence"]["possible_explanations"], [])
        self.assertTrue(payload["evidence"]["lines"])

    def test_actix_keeps_the_panic_column(self) -> None:
        text, events = load("actix_web_crash_noisy.log")
        payload = json.loads(format_json_report(Path("actix.log"), text, events))
        thrown = payload["locations"][0]
        self.assertEqual(thrown["role"], "throw_site")
        self.assertEqual(thrown["source_line"], 126)
        self.assertEqual(thrown["source_column"], 18)
        self.assertTrue(thrown["path"].endswith("tenant.rs"))
        self.assertEqual(payload["incident"]["framework"], "Actix")
        self.assertEqual(payload["incident"]["outcome"], "restarted")
        self.assertIn("The process terminated and the service came back.", payload["incident"]["summary"])
        validation = next(line for line in payload["evidence"]["lines"] if line["number"] == 3596)
        self.assertIn("token validation ok", validation["text"])
        self.assertEqual(validation["correlation"], "confirmed")
        report = format_agent_report(Path("actix.log"), text, events)
        self.assertIn("src/http/middleware/tenant.rs:126", report)
        self.assertIn("The process terminated and the service came back.", report)
        evidence = section(report, "## Evidence and observations", "## Ranked hypotheses")
        self.assertIn("Token validation succeeded on this request", evidence)
        self.assertIn("tenant claim was absent", evidence)
        unknowns = section(report, "## Unknowns and limitations", "## Chronological timeline")
        self.assertIn("A valid token does not show that every claim the operation requires was present.", unknowns)
        hypotheses = section(report, "## Ranked hypotheses", "## Recommended investigation")
        self.assertIn("No model hypotheses were produced.", hypotheses)
        primary = select_primary(events)
        assert primary is not None
        self.assertIn("3596 ", format_model_input(primary))

    def test_human_prompt_separates_loggers_from_frames_and_a_completed_restart(self) -> None:
        text = REPORT_PROMPT.messages[0].prompt.template
        self.assertIn("logger, not an application frame", text)
        self.assertIn("the restart succeeded", text)
        self.assertIn("failed restart attempt", text)

    def test_aspnet_source_line_is_not_the_log_line(self) -> None:
        text, events = load("aspnet_core_crash_noisy.log")
        report = format_agent_report(Path("aspnet.log"), text, events)
        payload = json.loads(format_json_report(Path("aspnet.log"), text, events))
        thrown = payload["locations"][0]
        self.assertEqual(thrown["source_line"], 118)
        self.assertTrue(thrown["path"].endswith("KeyRingBuilder.cs"))
        self.assertNotEqual(thrown["log_line"], 118)
        failure = section(report, "## Primary failure", "## Failure mechanism")
        self.assertIn("source line 118", failure)
        self.assertNotIn("source line 3526", failure)
        self.assertEqual(thrown["role"], "throw_site")
        self.assertEqual(payload["incident"]["exception"], "System.NullReferenceException")
        self.assertEqual(payload["incident"]["language"], "C#")

    def test_django_innermost_frame_is_wsgi(self) -> None:
        text, events = load("django_rest_crash_noisy.log")
        payload = json.loads(format_json_report(Path("django.log"), text, events))
        innermost = [item for item in payload["locations"] if item["role"] == "innermost"]
        self.assertEqual(len(innermost), 1)
        self.assertEqual(innermost[0]["source_line"], 16)
        self.assertTrue(innermost[0]["path"].endswith("config/wsgi.py"))
        self.assertEqual(payload["incident"]["exception"], "ModuleNotFoundError")
        self.assertEqual(payload["incident"]["framework"], "Django")
        self.assertEqual(payload["incident"]["outcome"], "restart_attempted")
        self.assertEqual(payload["evidence"]["possible_explanations"], [])
        self.assertIn(
            "declared as a dependency",
            " ".join(payload["evidence"]["unknowns"]),
        )



class ReportAssemblyTest(unittest.TestCase):
    def test_model_input_omits_independent_events(self) -> None:
        events = extract_events((LOGS / "aspnet_core_crash_noisy.log").read_text())
        assert events is not None
        primary = select_primary(events)
        assert primary is not None
        model_input = format_model_input(primary)
        self.assertIn("Event E3524", model_input)
        self.assertIn("NullReferenceException", model_input)
        self.assertIn("confirmed 3460", model_input)
        self.assertIn("Main process exited", model_input)
        self.assertNotIn("2085", model_input)
        self.assertNotIn("3543", model_input)

    def test_compose_report_replaces_a_model_written_warning_section(self) -> None:
        narrative = "Main failure\nThe worker stopped (line 4).\n\nRelated events\nNone\n\nOther warnings\nThe model invented this."
        report = compose_report(narrative, [])
        self.assertIn("Main failure", report)
        self.assertNotIn("invented", report)
        self.assertTrue(report.strip().endswith("Other warnings\nNone"))

    def test_fallback_cites_extracted_lines_only(self) -> None:
        events = extract_events((LOGS / "aspnet_core_crash_noisy.log").read_text())
        assert events is not None
        primary = select_primary(events)
        assert primary is not None
        independent = independent_events(events, primary)
        report = fallback_report(primary, independent)
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
            status = main(["--dump-events", str(LOGS / "aspnet_core_crash_noisy.log")])
        self.assertEqual(status, 0)
        text = buffer.getvalue()
        model, _, reported = text.partition("# Reported by Python")
        self.assertIn("# Sent to the model", model)
        self.assertIn("E3524", model)
        self.assertIn("3460", model)
        self.assertNotIn("2085", model)
        self.assertIn("2085", reported)
        self.assertIn("3543", reported)


if __name__ == '__main__':
    unittest.main()
