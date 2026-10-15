"""Command line exit codes, output files, and dump mode."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from log_and_order.cli import main

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "example-logs"


class ReportCliTest(unittest.TestCase):
    def test_output_writes_the_agent_report_and_leaves_stdout_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "incident.md"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "--format",
                        "agent",
                        "--no-llm",
                        "--output",
                        str(destination),
                        str(LOGS / "spring_boot_incident_noise.log"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "")
            text = destination.read_text()
            self.assertIn("## Executive summary", text)
            self.assertIn("java.lang.NoSuchMethodError", text)

    def test_json_format_rejects_a_non_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("This is a meeting note.\nIt has no timestamps.\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main(["--format", "json", str(path)])
            self.assertEqual(status, 1)
            self.assertIn("timestamped log", stderr.getvalue())

    def test_empty_file_still_prints_the_empty_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.log"
            path.write_text("\n")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(["--format", "agent", str(path)])
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue().strip(), "The file is empty.")

    def test_dump_events_can_be_written_to_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "dump.txt"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "--dump-events",
                        "--format",
                        "json",
                        "--output",
                        str(destination),
                        str(LOGS / "aspnet_core_crash_noisy.log"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), "")
            text = destination.read_text()
            self.assertIn("# Sent to the model", text)
            self.assertIn("E3524", text)

    def test_missing_file_exits_with_an_error(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["/no/such/log-reader-file.log"])
        self.assertEqual(status, 1)
        self.assertIn("Not a file", stderr.getvalue())

    def test_binary_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.log"
            path.write_bytes(b"2026-10-15T10:00:00Z INFO app: ok\n\0")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main([str(path)])
            self.assertEqual(status, 1)
            self.assertIn("binary", stderr.getvalue())

    def test_no_llm_rejects_a_non_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("A meeting note without timestamps.\n")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                status = main(["--no-llm", str(path)])
            self.assertEqual(status, 1)
            self.assertIn("requires the model", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
