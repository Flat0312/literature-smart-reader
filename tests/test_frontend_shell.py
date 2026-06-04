from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from views.home_view import _build_home_hero_html
from views.result_view import _build_result_sections, _should_show_debug_info


class _DummyResult:
    structured_notice = "当前结果仅供辅助参考。"
    structured_debug = {"final_result": {"backend": "relay_precheck"}, "summary_debug": {"keyword_source": "strategy_a_explicit_zh"}}
    parse_status = "partial_success"
    title = "测试论文标题"
    file_name = "demo-paper.pdf"

    def warning_items(self) -> list[str]:
        return ["以下字段由 AI 补充识别：作者，建议结合原文核对。"]

    def filtered_keywords(self) -> list[str]:
        return ["关键词A", "关键词B"]

    def filtered_authors(self) -> list[str]:
        return ["作者A", "作者B"]

    def structured_field_items(self):
        return [
            (1, "研究问题", "研究问题内容", None),
            (2, "研究方法", "研究方法内容", None),
            (3, "核心结论", "核心结论内容", None),
        ]

    def structured_field_count(self) -> int:
        return 3

    def primary_summary_language(self) -> str:
        return "zh"

    def summary_text(self) -> str:
        return "这是主摘要。"

    def english_abstract_text(self) -> str:
        return ""

    def course_presentation_outline_items(self) -> list[str]:
        return ["汇报 1", "汇报 2"]

    def course_paper_outline_items(self) -> list[str]:
        return ["论文 1", "论文 2"]

    def literature_review_outline_items(self) -> list[str]:
        return ["综述 1", "综述 2"]

    def plain_language_summary_text(self) -> str:
        return "通俗摘要。"

    def method_explanation_text(self) -> str:
        return "方法说明。"

    def innovation_items(self) -> list[str]:
        return ["创新点 1"]

    def limitation_items(self) -> list[str]:
        return ["不足 1"]


class FrontendShellTests(unittest.TestCase):
    def test_home_hero_uses_editorial_shell_classes(self) -> None:
        hero_html = _build_home_hero_html()
        self.assertIn("pf-home-hero", hero_html)
        self.assertIn("home-hero__visual", hero_html)
        self.assertIn("demo-card", hero_html)
        self.assertIn("demo-pdf", hero_html)
        self.assertIn("demo-output", hero_html)
        self.assertNotIn("home-flow", hero_html)

    def test_result_sections_use_current_card_classes_and_parse_hints(self) -> None:
        hero_html, left_html, center_html, right_intro_html = _build_result_sections(_DummyResult())

        self.assertIn("pf-result-hero", hero_html)
        self.assertIn("rs-hero__meta", hero_html)
        self.assertIn("pf-panel--meta", left_html)
        self.assertIn("rs-label-stack--tips", left_html)
        self.assertIn("AI 补充识别", left_html)
        self.assertIn("pf-panel--main", center_html)
        self.assertIn("pf-panel--rail", right_intro_html)

    def test_debug_info_is_hidden_unless_explicitly_enabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_should_show_debug_info())

        with patch.dict("os.environ", {"SHOW_DEBUG_INFO": "1"}, clear=True):
            self.assertTrue(_should_show_debug_info())


if __name__ == "__main__":
    unittest.main()
