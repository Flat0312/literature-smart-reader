from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AppStartupTests(unittest.TestCase):
    def test_app_import_does_not_eagerly_load_parse_pipeline(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import sys

            import app  # noqa: F401

            watched_modules = [
                "views.upload_view",
                "views.result_view",
                "services.paper_parse_service",
                "services.llm_service",
                "services.pdf_service",
            ]
            print("EAGER_MODULES=" + json.dumps([name for name in watched_modules if name in sys.modules]))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        marker_line = next(
            line for line in result.stdout.splitlines() if line.startswith("EAGER_MODULES=")
        )
        eager_modules = json.loads(marker_line.removeprefix("EAGER_MODULES="))
        self.assertEqual([], eager_modules)


if __name__ == "__main__":
    unittest.main()
