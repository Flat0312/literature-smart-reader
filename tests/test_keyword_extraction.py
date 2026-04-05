from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from models.paper_result import AUTHORS_DISPLAY_FALLBACK, NormalizedPaperParseResult, PaperResult
from services.llm_service import CourseSupportRequest, generate_course_support_material
from services.document_parse_service import parse_pdf_document
from services.metadata_service import (
    _extract_strategy_a_blocks,
    _extract_strategy_b_blocks,
    _extract_strategy_c_blocks,
    extract_authors_with_source,
    extract_keywords_with_source,
)
from services.paper_parse_service import ParsePipelineError, build_paper_result, parse_uploaded_pdf
from services.pdf_service import PdfExtractionResult, PdfPageDebug


class KeywordExtractionTests(unittest.TestCase):
    def without_relay(self):
        return patch.dict(
            os.environ,
            {"RELAY_API_KEY": "", "RELAY_BASE_URL": "", "RELAY_MODEL": ""},
            clear=False,
        )

    def build_pdf_result(self, text: str) -> PdfExtractionResult:
        page = PdfPageDebug(
            page_index=0,
            text=text,
            preview="sample",
            page_score=8.0,
            body_keywords_hit=["摘要", "关键词", "引言"],
            cover_keywords_hit=[],
            blocks=[],
        )
        return PdfExtractionResult(
            raw_text=text,
            body_text=text,
            pages=[page],
            body_start_page_index=0,
        )

    def test_strategy_a_structure_block(self) -> None:
        text = (
            "\u6458\u8981\uff1a...\n"
            "\u5173\u952e\u8bcd\uff1a\u6570\u5b57\u4eba\u6587\uff1b\u6c49\u670d\u6570\u5b57\u8d44\u6e90\uff1b"
            "\u77e5\u8bc6\u56fe\u8c31\uff1b\u6587\u732e\u5faa\u8bc1\uff1b\u8bc1\u636e\u94fe\n"
            "Abstract: ...\n"
        )
        blocks = _extract_strategy_a_blocks([text], language="zh")
        self.assertTrue(blocks)
        self.assertIn("\u6570\u5b57\u4eba\u6587", blocks[0][1])

    def test_strategy_b_line_match(self) -> None:
        text = (
            "\u6b63\u6587\u4e0a\u6587\n"
            "\u5173\u952e\u8bcd\uff1a\u6570\u5b57\u4eba\u6587\uff0c\u6c49\u670d\u6570\u5b57\u8d44\u6e90\uff0c"
            "\u77e5\u8bc6\u56fe\u8c31\uff0c\u6587\u732e\u5faa\u8bc1\uff0c\u8bc1\u636e\u94fe\n"
            "1 \u5f15\u8a00\n"
        )
        blocks = _extract_strategy_b_blocks([text], language="zh")
        self.assertTrue(blocks)
        self.assertIn("\u6587\u732e\u5faa\u8bc1", blocks[0][1])

    def test_space_delimited_keywords_with_bracket_label(self) -> None:
        text = (
            "\u6458\u8981\uff1a...\n"
            "\u3014\u5173\u952e\u8bcd\u3015 \u6587\u5316\u9057\u4ea7 \u6570\u5b57\u753b\u50cf \u5206\u7c7b\u4f53\u7cfb \u7cfb\u7edf\u6027\u4fdd\u62a4\n"
            "\u3014\u4e2d\u56fe\u6cd5\u5206\u7c7b\u53f7\u3015 G122\n"
            "\u3014\u5f15\u7528\u672c\u6587\u683c\u5f0f\u3015 xxx\n"
            "Abstract: ...\n"
        )
        result = extract_keywords_with_source(text, priority_text=text)
        self.assertEqual(
            result.keywords,
            ["\u6587\u5316\u9057\u4ea7", "\u6570\u5b57\u753b\u50cf", "\u5206\u7c7b\u4f53\u7cfb", "\u7cfb\u7edf\u6027\u4fdd\u62a4"],
        )
        self.assertEqual(len(result.keywords), 4)

    def test_strategy_c_abstract_neighborhood(self) -> None:
        text = (
            "\u6458 \u8981\uff1a\u672c\u6587\u7814\u7a76\u6570\u5b57\u8d44\u6e90\u3002\n"
            "\u4f5c\u8005\u5355\u4f4d\uff1a\u67d0\u5927\u5b66\n"
            "\u5173\u952e\u8bcd\uff1a\u6570\u5b57\u4eba\u6587\u3001\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u3001\u77e5\u8bc6\u56fe\u8c31\u3001"
            "\u6587\u732e\u5faa\u8bc1\u3001\u8bc1\u636e\u94fe\n"
            "1 \u5f15\u8a00\n"
        )
        blocks = _extract_strategy_c_blocks([text], language="zh")
        self.assertTrue(blocks)
        self.assertIn("\u8bc1\u636e\u94fe", blocks[0][1])

    def test_low_confidence_keyword_result(self) -> None:
        text = (
            "\u6458\u8981\uff1a\u672c\u6587\u7814\u7a76\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u3002\n"
            "\u5173\u952e\u8bcd\uff1a\u672c\u6587\u57fa\u4e8e\u77e5\u8bc6\u56fe\u8c31\u4e0e\u5faa\u8bc1\u65b9\u6cd5\u7cfb\u7edf\u5206\u6790\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u7684\u7ec4\u7ec7\u6a21\u5f0f\u4e0e\u5e94\u7528\u673a\u5236\n"
            "1 \u5f15\u8a00\n"
        )
        result = extract_keywords_with_source(text, priority_text=text)
        self.assertNotEqual(result.confidence, "\u9ad8")
        self.assertTrue(result.warnings)

    def test_pipeline_uses_same_keywords_array(self) -> None:
        page = PdfPageDebug(
            page_index=0,
            text=(
                "\u67d0\u67d0\u5b66\u62a5 ISSN 1000-0000\n"
                "\u57fa\u4e8e\u77e5\u8bc6\u56fe\u8c31\u7684\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u6587\u732e\u5faa\u8bc1\u53ca\u5e94\u7528\u7814\u7a76\n"
                "\u5b5f\u7e41\u723d\n"
                "\u6458\u8981\uff1a\u672c\u6587\u6784\u5efa\u8bc1\u636e\u94fe\u6a21\u578b\u3002\n"
                "\u5173\u952e\u8bcd\uff1a\u6570\u5b57\u4eba\u6587\uff1b\u6c49\u670d\u6570\u5b57\u8d44\u6e90\uff1b\u77e5\u8bc6\u56fe\u8c31\uff1b\u6587\u732e\u5faa\u8bc1\uff1b\u8bc1\u636e\u94fe\n"
                "1 \u5f15\u8a00\n"
            ),
            preview="sample",
            page_score=8.0,
            body_keywords_hit=["摘要", "关键词", "引言"],
            cover_keywords_hit=["issn"],
            blocks=[],
        )
        pdf_result = PdfExtractionResult(
            raw_text=page.text,
            body_text=page.text,
            pages=[page],
            body_start_page_index=0,
        )

        with self.without_relay():
            pipeline = parse_pdf_document(pdf_result)
        standard = pipeline.normalized_result.as_standard_dict()

        self.assertEqual(standard["keywords"], pipeline.normalized_result.filtered_keywords())
        self.assertEqual(len(standard["keywords"]), 5)

    def test_pipeline_prefers_chinese_abstract_over_english(self) -> None:
        page = PdfPageDebug(
            page_index=0,
            text=(
                "课程写作场景下的文献智读工具研究\n"
                "张三，李四\n"
                "摘要：本文从课程写作场景出发，讨论如何帮助学生更快完成单篇论文整理与课堂展示准备。\n"
                "关键词：课程写作；文献解读；课堂展示；学习辅助\n"
                "Abstract: This paper discusses how to support students with faster paper reading and presentation preparation.\n"
                "Keywords: course writing; literature reading; presentation\n"
                "1 引言\n"
            ),
            preview="sample",
            page_score=8.0,
            body_keywords_hit=["摘要", "关键词", "引言"],
            cover_keywords_hit=[],
            blocks=[],
        )
        pdf_result = PdfExtractionResult(
            raw_text=page.text,
            body_text=page.text,
            pages=[page],
            body_start_page_index=0,
        )

        with self.without_relay():
            pipeline = parse_pdf_document(pdf_result)

        self.assertIn("课程写作场景", pipeline.normalized_result.summary_text())
        self.assertEqual(pipeline.normalized_result.primary_summary_language(), "zh")
        self.assertIn("support students", pipeline.normalized_result.english_abstract_text())

    def test_pipeline_includes_authors_in_standard_result(self) -> None:
        page = PdfPageDebug(
            page_index=0,
            text=(
                "课程论文辅助中的文献整理方法研究\n"
                "张三，李四\n"
                "摘要：本文讨论文献整理流程。\n"
                "关键词：课程论文；文献整理；研究方法；课堂展示\n"
                "1 引言\n"
            ),
            preview="sample",
            page_score=8.0,
            body_keywords_hit=["摘要", "关键词", "引言"],
            cover_keywords_hit=[],
            blocks=[],
        )
        pdf_result = PdfExtractionResult(
            raw_text=page.text,
            body_text=page.text,
            pages=[page],
            body_start_page_index=0,
        )

        with self.without_relay():
            pipeline = parse_pdf_document(pdf_result)
        standard = pipeline.normalized_result.as_standard_dict()

        self.assertEqual(standard["authors"], ["张三", "李四"])
        self.assertEqual(pipeline.normalized_result.authors_text(), "张三、李四")

    def test_extract_authors_from_title_zone_with_institution(self) -> None:
        text = (
            "课程论文辅助中的文献整理方法研究\n"
            "张三1，李四2\n"
            "1. 某某大学信息管理学院 2. 某某大学档案学院\n"
            "摘要：本文讨论文献整理流程。\n"
        )
        result = extract_authors_with_source(text, title="课程论文辅助中的文献整理方法研究", priority_text=text)

        self.assertEqual(result.authors, ["张三", "李四"])
        self.assertEqual(result.source_kind, "rule_title_zone")
        self.assertIn("张三1，李四2", result.raw_candidates)

    def test_extract_authors_with_english_marker(self) -> None:
        text = (
            "A Study on Course Writing Support\n"
            "Authors: Alice Chen, Bob Li\n"
            "Abstract: This paper discusses course writing support.\n"
        )
        result = extract_authors_with_source(text, title="A Study on Course Writing Support", priority_text=text)

        self.assertEqual(result.authors, ["Alice Chen", "Bob Li"])
        self.assertEqual(result.source_kind, "rule_marker")

    def test_extract_authors_with_by_marker(self) -> None:
        text = (
            "Digital Heritage Research Support\n"
            "by Alice Chen and Bob Li\n"
            "Abstract: ...\n"
        )
        result = extract_authors_with_source(text, title="Digital Heritage Research Support", priority_text=text)

        self.assertEqual(result.authors, ["Alice Chen", "Bob Li"])
        self.assertEqual(result.source_kind, "rule_marker")

    def test_extract_authors_uses_llm_fallback_when_rules_fail(self) -> None:
        text = (
            "课程写作辅助研究\n"
            "某某大学信息管理学院\n"
            "摘要：本文讨论文献整理流程。\n"
        )
        mocked_result = type(
            "MockAuthorFallback",
            (),
            {
                "authors": ["张三", "李四"],
                "source_kind": "llm_fallback",
                "confidence": "low",
                "note": "",
                "debug_info": {"used_llm": True},
            },
        )()
        with patch("services.metadata_service.fallback_authors_with_llm", return_value=mocked_result):
            result = extract_authors_with_source(text, title="课程写作辅助研究", priority_text=text)

        self.assertEqual(result.authors, ["张三", "李四"])
        self.assertEqual(result.source_kind, "llm_fallback")
        self.assertEqual(result.confidence, "low")

    def test_keyword_result_dedupes_and_keeps_reasonable_count(self) -> None:
        text = (
            "摘要：本文聚焦课程写作中的文献整理过程。\n"
            "关键词：课程写作；文献解读；课程写作；文本分析；课堂展示；文献解读；研究综述；学习辅助\n"
            "1 引言\n"
        )
        result = extract_keywords_with_source(text, priority_text=text)

        self.assertEqual(
            result.keywords,
            ["课程写作", "文献解读", "文本分析", "课堂展示", "研究综述", "学习辅助"],
        )
        self.assertGreaterEqual(len(result.keywords), 3)
        self.assertLessEqual(len(result.keywords), 8)

    def test_course_support_fallback_is_stable_without_source_data(self) -> None:
        support = generate_course_support_material(CourseSupportRequest(title="", summary="", raw_text_preview=""))

        self.assertTrue(support.plain_language_summary)
        self.assertTrue(support.method_explanation)
        self.assertGreaterEqual(len(support.innovation_points), 2)
        self.assertGreaterEqual(len(support.limitation_points), 2)
        self.assertGreaterEqual(len(support.course_presentation_outline), 4)
        self.assertGreaterEqual(len(support.course_paper_outline), 4)
        self.assertGreaterEqual(len(support.literature_review_outline), 4)

    def test_build_paper_result_returns_success_when_core_fields_stable(self) -> None:
        pdf_result = self.build_pdf_result(
            "课程写作辅助中的文献解读研究\n"
            "张三，李四\n"
            "摘要：研究问题：本文关注课程写作场景下的单篇论文整理。研究方法：采用文本分析法梳理解析流程。核心结论：结构化阅读辅助可以提高课程展示准备效率。\n"
            "关键词：课程写作；文献解读；文本分析；课堂展示\n"
            "1 引言\n"
        )
        with self.without_relay():
            result = build_paper_result("sample.pdf", pdf_result)

        self.assertEqual(result.parse_status, "success")
        self.assertFalse(result.parse_errors)
        self.assertEqual(result.filtered_authors(), ["张三", "李四"])

    def test_build_paper_result_returns_partial_success_when_author_missing(self) -> None:
        pdf_result = self.build_pdf_result(
            "课程写作辅助中的文献解读研究\n"
            "摘要：本文关注课程写作场景下的单篇论文整理。\n"
            "关键词：课程写作；文献解读；文本分析；课堂展示\n"
            "1 引言\n"
        )
        with self.without_relay():
            result = build_paper_result("sample.pdf", pdf_result)

        self.assertEqual(result.parse_status, "partial_success")
        self.assertIn("本次解析未稳定识别作者。", result.warning_items())
        self.assertEqual(result.authors_text(), AUTHORS_DISPLAY_FALLBACK)

    def test_parse_uploaded_pdf_returns_structured_failure_feedback(self) -> None:
        with self.assertRaises(ParsePipelineError) as context:
            parse_uploaded_pdf("empty.pdf", b"")

        self.assertEqual(context.exception.category, "pdf_read_failed")
        self.assertEqual(context.exception.parse_feedback.get("status"), "failed")
        self.assertTrue(context.exception.parse_feedback.get("errors"))

    def test_parse_uploaded_pdf_classifies_text_extract_failure(self) -> None:
        with patch("services.paper_parse_service.extract_pdf_context", side_effect=ValueError("PDF 没有提取到可用文本。")):
            with self.assertRaises(ParsePipelineError) as context:
                parse_uploaded_pdf("empty.pdf", b"%PDF")

        self.assertEqual(context.exception.category, "text_extract_failed")
        self.assertEqual(context.exception.parse_feedback.get("error_stage"), "text_extract")

    def test_authors_fallback_text_is_stable_even_when_missing(self) -> None:
        result = PaperResult(file_name="sample.pdf", parsed_result=NormalizedPaperParseResult(title="test"))

        self.assertEqual(result.authors_text(), AUTHORS_DISPLAY_FALLBACK)

    def test_pipeline_emits_keyword_low_confidence_warning(self) -> None:
        page = PdfPageDebug(
            page_index=0,
            text=(
                "\u67d0\u67d0\u5b66\u62a5 ISSN 1000-0000\n"
                "\u57fa\u4e8e\u77e5\u8bc6\u56fe\u8c31\u7684\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u7814\u7a76\n"
                "\u6458\u8981\uff1a\u672c\u6587\u7814\u7a76\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u3002\n"
                "\u5173\u952e\u8bcd\uff1a\u672c\u6587\u57fa\u4e8e\u77e5\u8bc6\u56fe\u8c31\u4e0e\u5faa\u8bc1\u65b9\u6cd5"
                "\u7cfb\u7edf\u5206\u6790\u6c49\u670d\u6570\u5b57\u8d44\u6e90\u7684\u7ec4\u7ec7\u6a21\u5f0f\u4e0e\u5e94"
                "\u7528\u673a\u5236\n"
                "1 \u5f15\u8a00\n"
            ),
            preview="sample",
            page_score=8.0,
            body_keywords_hit=["摘要", "关键词", "引言"],
            cover_keywords_hit=["issn"],
            blocks=[],
        )
        pdf_result = PdfExtractionResult(
            raw_text=page.text,
            body_text=page.text,
            pages=[page],
            body_start_page_index=0,
        )

        with self.without_relay():
            pipeline = parse_pdf_document(pdf_result)
        warnings = pipeline.normalized_result.warning_items()

        self.assertTrue(
            any("\u5173\u952e\u8bcd" in warning and ("\u8f83\u4f4e" in warning or "\u56de\u9000" in warning) for warning in warnings)
        )

    @unittest.skipUnless(
        Path("D:/\u6587\u5316\u9057\u4ea7\u6570\u5b57\u753b\u50cf\uff1a\u7cfb\u7edf\u6027\u4fdd\u62a4\u89c6\u57df\u4e0b\u6211\u56fd\u6587\u5316\u9057\u4ea7\u5206\u7c7b\u4f53\u7cfb\u91cd\u6784_\u5f20\u4e39.pdf").exists(),
        "real regression pdf not found",
    )
    def test_real_pdf_space_delimited_keywords(self) -> None:
        pdf_path = Path("D:/\u6587\u5316\u9057\u4ea7\u6570\u5b57\u753b\u50cf\uff1a\u7cfb\u7edf\u6027\u4fdd\u62a4\u89c6\u57df\u4e0b\u6211\u56fd\u6587\u5316\u9057\u4ea7\u5206\u7c7b\u4f53\u7cfb\u91cd\u6784_\u5f20\u4e39.pdf")
        with self.without_relay():
            result = parse_uploaded_pdf(pdf_path.name, pdf_path.read_bytes())

        self.assertEqual(
            result.filtered_keywords(),
            ["\u6587\u5316\u9057\u4ea7", "\u6570\u5b57\u753b\u50cf", "\u5206\u7c7b\u4f53\u7cfb", "\u7cfb\u7edf\u6027\u4fdd\u62a4"],
        )
        self.assertEqual(len(result.filtered_keywords()), 4)
        self.assertEqual(result.as_standard_dict()["keywords"], result.filtered_keywords())


if __name__ == "__main__":
    unittest.main()
