"""CLI orchestration contracts, including failure and resume boundaries."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from src.cli import parser
from src.runtime.workflow import Run, check_checkpoint, choose_groups, raw_profiles, invoke


class CliTests(unittest.TestCase):
    def test_run_requires_explicit_budget_and_gpu(self):
        base = ["run", "--pxrd", "observed.xy", "--smiles", "CCO", "--wavelength", "1.54184",
                "--checkpoint", "model.pt", "--checkpoint-sha256", "0" * 64,
                "--chemberta", "chemberta", "--bomlip-root", "engine", "--mace-model", "mace.model", "--output", "run"]
        with self.assertRaises(SystemExit):
            parser().parse_args(base + ["--gpu", "0"])
        with self.assertRaises(SystemExit):
            parser().parse_args(base + ["--count", "1"])
        self.assertEqual(parser().parse_args(base + ["--gpu", "0", "--count", "1"]).count, 1)

    def test_resume_rejects_changed_identity_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            def work(p):
                p.mkdir(); (p / "result").write_text("complete")
            run = Run(root, {"input": "a"})
            out = run.stage("one", work)
            run.lock.close()
            resumed = Run(root, {"input": "a"}, True)
            self.assertEqual(resumed.stage("one", lambda _: self.fail("should reuse")), out)
            (out / "result").write_text("changed")
            with self.assertRaises(ValueError): resumed.stage("one", work)
            resumed.lock.close()
            with self.assertRaises(ValueError): Run(root, {"input": "b"}, True)
            with self.assertRaises(ValueError): Run(root, {"input": "a"})

    def test_retry_preserves_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run = Run(root, {})
            def fail(p):
                p.mkdir(); (p / "partial").write_text("evidence"); raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError): run.stage("one", fail)
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "failed")
            def success(p):
                p.mkdir(); (p / "result").write_text("complete")
            second = run.stage("one", success)
            self.assertEqual(second.name, "one-002")
            self.assertTrue((root / "one-001/partial").exists())
            run.lock.close()

    def test_concurrent_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            run = Run(root, {})
            with self.assertRaises(ValueError): Run(root, {}, True)
            run.lock.close()

    def test_others_is_not_automatically_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "predictions.csv"
            p.write_text('sample_id,predicted_space_groups\none,14;Others;2\n')
            with self.assertRaisesRegex(ValueError, "Others"): choose_groups(p)
            self.assertEqual(choose_groups(p, [14, 2]), [14, 2])
            with self.assertRaises(ValueError): choose_groups(p, [14, 14])

    def test_failure_between_stages_is_recorded_and_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "shortlist"):
                with Run(root, {}) as run:
                    raise ValueError("invalid shortlist")
            self.assertEqual(json.loads((root / "run.json").read_text())["status"], "failed")
            with Run(root, {}, True):
                pass

    def test_engine_launch_does_not_prepend_predictor_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Run(Path(tmp) / "run", {}) as run:
                captured = []
                with patch("src.runtime.workflow.run_logged", side_effect=lambda command, **kwargs: captured.append((command, kwargs))):
                    invoke(run, "tools.generate_crystal_candidates", ["--help"], run.root / "generate", python=sys.executable)
                command, options = captured[0]
                result = subprocess.run(command, cwd=run.root, env=options["env"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--space-groups", result.stdout)
                self.assertIn("sys.path.append", command[2])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-lifecycle regression")
    def test_timeout_stops_nested_engine_process(self):
        from src.runtime.downstream_execution import run_logged
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = "import os,time; from pathlib import Path; Path('engine.pid').write_text(str(os.getpid())); time.sleep(60)"
            inner = ("import os,sys; from pathlib import Path; "
                     "from src.runtime.downstream_execution import run_logged; "
                     f"run_logged([sys.executable,'-c',{engine!r}],cwd=Path.cwd(),env=os.environ.copy(),log=Path('inner.log'),timeout=60)")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged([sys.executable, "-c", inner], cwd=root, env=env, log=root / "outer.log", timeout=2)
            pid = int((root / "engine.pid").read_text())
            try:
                self.assertFalse(Path(f"/proc/{pid}").exists(), "timed-out workflow left its engine alive")
            finally:
                if Path(f"/proc/{pid}").exists():
                    os.kill(pid, 15)

    def test_raw_dtw_input_retains_baseline_and_measured_support(self):
        from src.data.pxrd import preprocess_experimental
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "observed.xy"
            theta = np.arange(10, 50.001, .02)
            intensity = 7 + 50 * np.exp(-((theta - 20) / .1) ** 2)
            np.savetxt(p, np.column_stack([theta, intensity]))
            prepared, raw, start, stop = raw_profiles(p)
            np.testing.assert_array_equal(prepared, preprocess_experimental(theta, intensity))
            self.assertEqual(start, 400)
            self.assertGreater(raw[start:stop].min(), 6.99)
            self.assertEqual(raw[0], 0)
            self.assertLess(prepared.min(), .01)

    def test_zero_signal_and_wrong_model_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.xy"
            np.savetxt(p, [[10, 0], [20, 0]])
            with self.assertRaises(ValueError): raw_profiles(p)
            with self.assertRaises(ValueError): check_checkpoint(p, "0" * 64)


if __name__ == "__main__":
    unittest.main()
