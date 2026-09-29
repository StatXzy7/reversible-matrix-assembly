import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DemoTest(unittest.TestCase):
    def test_demo_runs_and_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "demo.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"demo failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertIn("DEMO PASSED", result.stdout)
        # It must print real inputs, operations and assertions.
        self.assertIn("Add", result.stdout)
        self.assertIn("matvec", result.stdout)
        self.assertIn("[assert:OK]", result.stdout)
        self.assertNotIn("[assert:FAIL]", result.stdout)


if __name__ == "__main__":
    unittest.main()
