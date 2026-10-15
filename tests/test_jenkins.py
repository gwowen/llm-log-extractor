"""Jenkins console logs: detection, stages, primary failure, and reports."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from log_and_order.cli import main
from log_and_order.incidents.jenkins import build_jenkins_report
from log_and_order.logs.extraction import extract_events
from log_and_order.logs.jenkins.analysis import analyse_jenkins
from log_and_order.pipeline import format_json_report

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"


def load(name: str) -> str:
    return (LOGS / name).read_text()


class DetectionTest(unittest.TestCase):
    def test_spring_log_is_not_jenkins(self) -> None:
        text = load("spring_boot_incident_noise.log")
        self.assertIsNone(analyse_jenkins(text))
        self.assertIsNotNone(extract_events(text))

    def test_generic_pipeline_words_are_not_jenkins(self) -> None:
        text = "\n".join(
            [
                "The Pipeline will Build and Deploy the service.",
                "Build started on the application host.",
                "Deploy finished without a Jenkins console.",
            ]
        )
        self.assertIsNone(analyse_jenkins(text))

    def test_one_finished_line_is_not_enough(self) -> None:
        self.assertIsNone(analyse_jenkins("Finished: FAILURE\n"))

    def test_success_and_truncated_logs_are_recognised(self) -> None:
        self.assertIsNotNone(analyse_jenkins(load("jenkins_pipeline_success.log")))
        self.assertIsNotNone(analyse_jenkins(load("jenkins_truncated.log")))
        self.assertIsNotNone(analyse_jenkins(load("jenkins_freestyle.log")))

    def test_malformed_markers_do_not_raise(self) -> None:
        text = "\n".join(
            [
                "Started by user casey",
                "[Pipeline] Start of Pipeline",
                "[Pipeline] {",
                "[Pipeline] {",
                "((((",
                "Finished: FAILURE",
            ]
        )
        analysis = analyse_jenkins(text)
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.build_result, "FAILURE")


class StageTest(unittest.TestCase):
    def test_nested_stage_is_the_failure_site(self) -> None:
        analysis = analyse_jenkins(load("jenkins_nested_stages.log"))
        assert analysis is not None
        package = next(stage for stage in analysis.stages if stage.name == "Package")
        self.assertEqual(package.parent, "Release")
        self.assertEqual(package.depth, 1)
        assert analysis.primary is not None
        self.assertEqual(analysis.primary.stage, "Package")
        self.assertEqual(analysis.primary.command, "python -m build")

    def test_parallel_output_without_a_branch_stays_uncertain(self) -> None:
        analysis = analyse_jenkins(load("jenkins_parallel_interleaved.log"))
        assert analysis is not None and analysis.primary is not None
        self.assertEqual(analysis.primary.stage_association, "uncertain")
        self.assertIsNone(analysis.primary.stage)
        self.assertEqual(analysis.primary.possible_stages, ["Unit tests", "Lint"])
        self.assertIsNone(analysis.primary.command)
        named = analysis.others[0]
        self.assertEqual(named.stage, "Unit tests")
        self.assertEqual(named.stage_association, "confirmed")

    def test_freestyle_has_no_stages(self) -> None:
        analysis = analyse_jenkins(load("jenkins_freestyle.log"))
        assert analysis is not None and analysis.primary is not None
        self.assertEqual(analysis.job_style, "freestyle")
        self.assertEqual(analysis.stages, [])
        self.assertEqual(analysis.primary.command, "make test")
        self.assertEqual(analysis.primary.source_file, "src/main.c")
        self.assertEqual(analysis.primary.source_line, 12)


class FailureSelectionTest(unittest.TestCase):
    def test_typescript_error_outranks_the_jenkins_status(self) -> None:
        text = load("jenkins_typescript_failure.log")
        analysis = analyse_jenkins(text)
        assert analysis is not None and analysis.primary is not None
        hit = analysis.primary
        self.assertEqual(hit.category, "BUILD")
        self.assertEqual(hit.stage, "Build")
        self.assertEqual(hit.command, "npm run build")
        self.assertEqual(hit.exit_code, 2)
        self.assertEqual(hit.source_file, "src/server.ts")
        self.assertEqual(hit.source_line, 42)
        self.assertNotEqual(hit.log_line, 42)
        self.assertIn("TS2322", load("jenkins_typescript_failure.log").splitlines()[hit.log_line - 1])
        self.assertNotIn("Finished:", hit.message)
        self.assertFalse(hit.symptom)

    def test_noisy_log_keeps_the_original_line_number(self) -> None:
        plain = analyse_jenkins(load("jenkins_noisy_build_failure.log"))
        stamped = analyse_jenkins(load("jenkins_noisy_build_failure_timestamped.log"))
        assert plain is not None and plain.primary is not None and plain.primary_event is not None
        assert stamped is not None and stamped.primary is not None and stamped.primary_event is not None
        self.assertEqual(plain.primary.log_line, 3694)
        self.assertEqual(stamped.primary.log_line, 3694)
        self.assertEqual(plain.primary_event.seed.timestamp, "")
        self.assertTrue(stamped.primary_event.seed.timestamp)
        self.assertEqual(stamped.primary.source_file, "src/services/tenantResolver.ts")
        self.assertEqual(stamped.primary.source_line, 147)
        self.assertEqual(len(plain.lines), len(load("jenkins_noisy_build_failure.log").splitlines()))

    def test_pytest_keeps_the_test_name_and_source_line(self) -> None:
        analysis = analyse_jenkins(load("jenkins_pytest_failure.log"))
        assert analysis is not None and analysis.primary is not None
        hit = analysis.primary
        self.assertEqual(hit.category, "TEST")
        self.assertEqual(hit.test_name, "tests/test_account.py::test_account_id")
        self.assertEqual(hit.source_file, "tests/test_account.py")
        self.assertEqual(hit.source_line, 27)
        self.assertIn("abc", hit.message)
        self.assertIn("xyz", hit.message)

    def test_tool_failures_use_their_categories(self) -> None:
        expected = {
            "jenkins_maven_failure.log": ("BUILD", "Compile", "/workspace/src/main/java/com/example/App.java", 42),
            "jenkins_docker_failure.log": ("DEPLOYMENT", "Container image", None, None),
            "jenkins_cloudformation_failure.log": ("DEPLOYMENT", "Deploy staging", None, None),
            "jenkins_missing_agent.log": ("INFRASTRUCTURE", None, None, None),
            "jenkins_timeout.log": ("INFRASTRUCTURE", "Import", None, None),
            "jenkins_shell_failure.log": ("INFRASTRUCTURE", "Migrate", None, None),
        }
        for name, (category, stage, path, line) in expected.items():
            analysis = analyse_jenkins(load(name))
            assert analysis is not None and analysis.primary is not None
            hit = analysis.primary
            self.assertEqual(hit.category, category, name)
            self.assertEqual(hit.stage, stage, name)
            self.assertEqual(hit.source_file, path, name)
            self.assertEqual(hit.source_line, line, name)
            self.assertNotIn("Finished:", hit.message)

    def test_timeout_is_not_the_finished_line(self) -> None:
        analysis = analyse_jenkins(load("jenkins_timeout.log"))
        assert analysis is not None and analysis.primary is not None
        self.assertEqual(analysis.build_result, "ABORTED")
        self.assertIn("timeout", analysis.primary.message.lower())

    def test_missing_agent_names_the_label(self) -> None:
        analysis = analyse_jenkins(load("jenkins_missing_agent.log"))
        assert analysis is not None and analysis.primary is not None
        self.assertIn("linux-builders", analysis.primary.message)
        self.assertIsNone(analysis.primary.command)

    def test_recovered_error_is_not_the_primary_failure(self) -> None:
        analysis = analyse_jenkins(load("jenkins_recovered_success.log"))
        assert analysis is not None
        self.assertIsNone(analysis.primary)
        self.assertEqual(analysis.build_result, "SUCCESS")
        self.assertEqual(len(analysis.recovered), 1)
        self.assertEqual(analysis.recovered[0].stage, "Migrate")

    def test_independent_stage_failures_both_remain(self) -> None:
        analysis = analyse_jenkins(load("jenkins_multiple_failures.log"))
        assert analysis is not None and analysis.primary is not None
        categories = {analysis.primary.category, *(item.category for item in analysis.others)}
        stages = {analysis.primary.stage, *(item.stage for item in analysis.others)}
        self.assertEqual(categories, {"BUILD", "TEST"})
        self.assertEqual(stages, {"Lint", "Unit tests"})
        self.assertNotIn("Finished:", analysis.primary.message)

    def test_truncated_log_has_no_invented_result(self) -> None:
        analysis = analyse_jenkins(load("jenkins_truncated.log"))
        assert analysis is not None and analysis.primary is not None
        self.assertIsNone(analysis.build_result)
        self.assertEqual(analysis.primary.source_line, 42)
        self.assertIsNone(analysis.primary.exit_code)

    def test_successful_pipeline_has_no_failure(self) -> None:
        analysis = analyse_jenkins(load("jenkins_pipeline_success.log"))
        assert analysis is not None
        self.assertEqual(analysis.build_result, "SUCCESS")
        self.assertIsNone(analysis.primary)
        self.assertEqual([stage.name for stage in analysis.stages], ["Checkout", "Test"])


class ReportTest(unittest.TestCase):
    def test_json_report_distinguishes_source_and_log_lines(self) -> None:
        text = load("jenkins_typescript_failure.log")
        analysis = analyse_jenkins(text)
        assert analysis is not None and analysis.primary is not None
        payload = json.loads(
            format_json_report(LOGS / "jenkins_typescript_failure.log", text, analysis.events, jenkins=analysis)
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["evidence"]["possible_explanations"], [])
        jenkins = payload["jenkins"]
        self.assertEqual(jenkins["log_type"], "jenkins")
        self.assertEqual(jenkins["build_result"], "FAILURE")
        self.assertEqual(jenkins["failed_stage"], "Build")
        self.assertEqual(jenkins["failure_category"], "BUILD")
        self.assertEqual(jenkins["failed_command"], "npm run build")
        self.assertEqual(jenkins["exit_code"], 2)
        primary = jenkins["primary_failure"]
        self.assertEqual(primary["source_file"], "src/server.ts")
        self.assertEqual(primary["source_line"], 42)
        self.assertIn(analysis.primary.log_line, primary["log_lines"])
        self.assertNotEqual(primary["source_line"], analysis.primary.log_line)
        location = payload["locations"][0]
        self.assertEqual(location["source_line"], 42)
        self.assertEqual(location["log_line"], analysis.primary.log_line)
        self.assertNotIn("jenkins", json.loads(
            format_json_report(
                LOGS / "spring_boot_incident_noise.log",
                load("spring_boot_incident_noise.log"),
                extract_events(load("spring_boot_incident_noise.log")) or [],
            )
        ))

    def test_human_report_quotes_the_diagnostic_and_redacts_secrets(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = main(["--no-llm", str(LOGS / "jenkins_shell_failure.log")])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        self.assertIn("Failure category: INFRASTRUCTURE", text)
        self.assertIn("command not found", text)
        self.assertIn("Exit code: 127", text)
        self.assertNotIn("supersecretvalue", text)
        self.assertNotIn("wJalrXUtnFEMI", text)
        self.assertNotIn("supersecrettoken", text)
        self.assertIn("[REDACTED]", text)

    def test_agent_report_includes_jenkins_fields(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = main(
                ["--format", "agent", "--no-llm", str(LOGS / "jenkins_typescript_failure.log")]
            )
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        self.assertIn("## Jenkins build", text)
        self.assertIn("## Executive summary", text)
        self.assertIn("src/server.ts", text)
        self.assertIn("source line 42", text)
        self.assertNotIn("The fix has been validated", text)

    def test_dump_events_accepts_a_jenkins_log(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = main(["--dump-events", str(LOGS / "jenkins_truncated.log")])
        self.assertEqual(status, 0)
        self.assertIn("Log type: Jenkins", stdout.getvalue())
        self.assertNotIn("Not a timestamped log.", stdout.getvalue())

    def test_uncertain_stage_is_omitted_from_the_report(self) -> None:
        text = load("jenkins_parallel_interleaved.log")
        analysis = analyse_jenkins(text)
        assert analysis is not None
        report = build_jenkins_report(Path("jenkins_parallel_interleaved.log"), analysis)
        assert report.jenkins is not None
        self.assertIsNone(report.jenkins.failed_stage)
        self.assertEqual(report.jenkins.stage_association, "uncertain")
        self.assertIsNone(report.jenkins.failed_command)

    def test_success_report_does_not_invent_a_failure(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = main(["--no-llm", str(LOGS / "jenkins_pipeline_success.log")])
        self.assertEqual(status, 0)
        text = stdout.getvalue()
        self.assertIn("Build result: SUCCESS", text)
        self.assertIn("No relevant errors or warnings were found.", text)
