"""CI routing checks use Git snapshots and require no poker dependencies."""

import json
import itertools
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

from scripts.ci_scope import check_prose, route


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old_cwd)
        self.git("init", "-q")
        self.git("config", "user.email", "ci@example.invalid")
        self.git("config", "user.name", "CI fixture")
        self.git("config", "commit.gpgsign", "false")
        (self.root / "docs").mkdir()
        (self.root / "docs/report.md").write_text("# Initial\n")
        (self.root / "docs/rules.md").write_text("# Rules\n")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def scope(self, event_name="pull_request"):
        return route(event_name, {"pull_request": {"base": {"sha": self.base}}}, self.git("rev-parse", "HEAD"))

    def test_added_and_edited_markdown_use_prose_lane(self):
        (self.root / "docs/report.md").write_text("# Updated\n")
        (self.root / "readme.md").write_text("# Readme\n")
        head = self.commit()
        full, _, paths = self.scope()
        self.assertFalse(full)
        self.assertEqual(set(paths), {"docs/report.md", "readme.md"})
        check_prose(head, paths)

    def test_main_and_manual_stay_full_even_when_change_is_prose(self):
        (self.root / "docs/report.md").write_text("updated")
        self.commit()
        for event in ("push", "workflow_dispatch", "schedule", "unknown"):
            self.assertTrue(self.scope(event)[0])

    def test_code_configuration_fixtures_workflows_and_contracts_stay_full(self):
        for path in ("agent.py", "config.json", "fixture.csv", "pipeline.yml", "prompt.txt", "docs/rules.md"):
            with self.subTest(path=path):
                self.git("reset", "--hard", self.base)
                self.git("clean", "-fd")
                (self.root / path).write_text("changed")
                self.commit()
                self.assertTrue(self.scope()[0])

    def test_deletion_rename_and_symlink_stay_full(self):
        (self.root / "docs/report.md").unlink()
        self.commit()
        self.assertTrue(self.scope()[0])
        self.git("reset", "--hard", self.base)
        self.git("mv", "docs/report.md", "docs/renamed.md")
        self.commit()
        self.assertTrue(self.scope()[0])
        self.git("reset", "--hard", self.base)
        (self.root / "readme.md").symlink_to("docs/report.md")
        self.commit()
        self.assertTrue(self.scope()[0])

    def test_missing_invalid_and_unavailable_comparison_fail_to_full(self):
        head = self.git("rev-parse", "HEAD")
        for event in ({}, {"pull_request": None}, {"pull_request": {"base": {"sha": "--help"}}},
                      {"pull_request": {"base": {"sha": "1" * 40}}}):
            self.assertTrue(route("pull_request", event, head)[0])
        self.assertTrue(self.scope()[0])

    def test_executable_markdown_and_mode_changes_stay_full(self):
        path = self.root / "docs/report.md"
        path.chmod(0o755)
        self.base = self.commit()
        path.write_text("updated executable prose")
        self.commit()
        self.assertTrue(self.scope()[0])
        path.chmod(0o644)
        self.commit()
        self.assertTrue(self.scope()[0])

    def test_prose_merge_works_with_actual_depth_two_checkout(self):
        self.git("checkout", "-qb", "proposal")
        (self.root / "docs/report.md").write_text("updated prose")
        proposal = self.commit()
        self.git("checkout", "-qb", "integration", self.base)
        self.git("merge", "--no-ff", "-qm", "merge fixture", proposal)
        checkout = self.root / "shallow"
        self.git("clone", "--depth=2", "--branch=integration", self.root.as_uri(), str(checkout))
        os.chdir(checkout)
        self.assertEqual(self.git("rev-parse", "--is-shallow-repository"), "true")
        full, _, paths = self.scope()
        self.assertFalse(full)
        check_prose(self.git("rev-parse", "HEAD"), paths)

    def test_compares_merge_snapshot_not_just_latest_head_commit(self):
        self.git("checkout", "-qb", "proposal")
        (self.root / "code.py").write_text("changed code")
        self.commit()
        (self.root / "docs/report.md").write_text("later prose")
        proposal = self.commit()
        self.git("checkout", "-q", self.base)
        self.git("merge", "--no-ff", "-qm", "merge fixture", proposal)
        self.assertTrue(self.scope()[0])

    def test_reads_tracked_utf8_blob_not_untracked_working_file(self):
        (self.root / "docs/report.md").write_text("tracked prose")
        head = self.commit()
        (self.root / "docs/report.md").write_bytes(b"\xff")
        full, _, paths = self.scope()
        self.assertFalse(full)
        check_prose(head, paths)
        self.commit()
        with self.assertRaises(UnicodeDecodeError):
            check_prose(self.git("rev-parse", "HEAD"), paths)

    def test_cli_emits_fixed_output_and_summary_for_safe_fallback(self):
        script = Path(self.old_cwd / "scripts/ci_scope.py")
        event = self.root / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": "--help"}}}))
        output, summary = self.root / "output", self.root / "summary"
        result = subprocess.run([os.sys.executable, str(script)], check=True, capture_output=True, text=True,
                                env={**os.environ, "GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(event),
                                     "GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)})
        self.assertTrue(json.loads(result.stdout)["full"])
        self.assertEqual(output.read_text(), "full=true\n")
        self.assertIn("Lane: **full**", summary.read_text())

    def test_cli_prose_check_uses_actual_comparison_and_skips_no_runtime_change(self):
        script = self.old_cwd / "scripts/ci_scope.py"
        (self.root / "docs/report.md").write_text("updated prose")
        self.commit()
        event = self.root / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": self.base}}}))
        output = self.root / "output"
        env = {**os.environ, "GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(event),
               "GITHUB_OUTPUT": str(output)}
        result = subprocess.run([os.sys.executable, str(script)], env=env, check=True,
                                capture_output=True, text=True)
        self.assertFalse(json.loads(result.stdout)["full"])
        self.assertEqual(output.read_text(), "full=false\n")
        subprocess.run([os.sys.executable, str(script), "--check-prose"], env=env,
                       check=True, capture_output=True)

    def test_actual_final_gate_rejects_failed_cancelled_and_missing_checks(self):
        workflow = (self.old_cwd / ".github/workflows/tests.yml").read_text()
        step = workflow.split("      - name: Require the selected checks to pass", 1)[1]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        statuses = ("success", "failure", "cancelled", "skipped")
        for scope, required, full in itertools.product(statuses, ("true", "false", ""), statuses):
            with self.subTest(scope=scope, required=required, full=full):
                result = subprocess.run(["bash", "-e", "-c", script], capture_output=True,
                                        env={**os.environ, "SCOPE_RESULT": scope,
                                             "FULL_REQUIRED": required, "FULL_RESULT": full})
                expected = scope == "success" and (
                    (required == "true" and full == "success")
                    or (required == "false" and full == "skipped"))
                self.assertEqual(result.returncode == 0, expected)


if __name__ == "__main__":
    unittest.main()
