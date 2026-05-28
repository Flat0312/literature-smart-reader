from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _build_background_layer_html


class BackgroundLayerTests(unittest.TestCase):
    def test_background_uses_dense_collision_bubble_field(self) -> None:
        background_html = _build_background_layer_html()

        bubble_count = len(re.findall(r'class="pf-bubble ', background_html))
        self.assertGreaterEqual(bubble_count, 14)
        self.assertIn("pf-collision-wave", background_html)
        self.assertIn("pf-impact-a", background_html)
        self.assertIn("pf-impact-b", background_html)
        self.assertIn("pf-impact-c", background_html)
        self.assertIn("prefers-reduced-motion:reduce", background_html)
        self.assertNotIn("@keyframes b1", background_html)


if __name__ == "__main__":
    unittest.main()
