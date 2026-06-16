from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from models.paper_result import AUTHORS_DISPLAY_FALLBACK, NormalizedPaperParseResult, PaperResult
from services.llm_service import (
    CourseSupportRequest,
    MetadataRecognitionPayload,
    MetadataRecognitionResult,
    RelaySettings,
    generate_course_support_material,
    recognize_metadata_with_llm,
)
from services.document_parse_service import parse_pdf_document
from services.metadata_service import (
    _extract_strategy_a_blocks,
    _extract_strategy_b_blocks,
    _extract_strategy_c_blocks,
    _is_valid_title_candidate,
    extract_authors_with_source,
    extract_keywords_with_source,
)
from services.paper_parse_service import ParsePipelineError, ParseRuntimeTracker, build_paper_result, parse_uploaded_pdf
from services.pdf_service import PdfExtractionResult, PdfPageDebug, extract_pdf_context
from services.structure_service import collect_structured_candidates
from services.structured_rewrite_service import rewrite_structured_fields
from services.summary_service import extract_abstracts


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

    def build_text_pdf_bytes(self, text: str) -> bytes:
        document = fitz.open()
        page = document.new_page()
        y = 72
        for line in text.splitlines():
            page.insert_text((72, y), line, fontsize=11)
            y += 18
        pdf_bytes = document.tobytes()
        document.close()
        return pdf_bytes

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

    def test_rejects_repeated_garbled_title_candidate(self) -> None:
        garbled_title = "欟" * 36 + " " + "榄" * 24

        self.assertFalse(_is_valid_title_candidate(garbled_title))

    def test_pipeline_uses_file_name_when_pdf_title_is_garbled(self) -> None:
        garbled_title = "欟" * 36 + " " + "榄" * 24
        page = PdfPageDebug(
            page_index=0,
            text=(
                f"{garbled_title}\n"
                "摘要：本文讨论人工智能研究前沿识别与领域演化分析。\n"
                "关键词：人工智能；研究前沿；领域演化；文献分析\n"
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
            result = build_paper_result("人工智能研究前沿识别与分析_基于领域全局演化研究视角_王日芬.pdf", pdf_result)

        self.assertNotIn("欟", result.title)
        self.assertIn("人工智能研究前沿识别与分析", result.title)
        self.assertIn("已使用文件名作为标题", " ".join(result.warning_items()))

    def test_pipeline_uses_cnki_file_name_when_extracted_title_is_body_fragment(self) -> None:
        body_fragment_title = "域， Small ， Garfield ， Persson 等知名学者陆续从不同角度研究和阐释它的内涵，形成了研究前沿是新兴主题 、 正"
        page = PdfPageDebug(
            page_index=0,
            text=(
                f"{body_fragment_title}\n"
                "摘要：本文讨论人工智能研究前沿识别与领域演化分析。\n"
                "关键词：人工智能；研究前沿；领域演化；文献分析\n"
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
            result = build_paper_result("人工智能研究前沿识别与分析_基于领域全局演化研究视角_王曰芬.pdf", pdf_result)

        self.assertEqual(result.title, "人工智能研究前沿识别与分析 基于领域全局演化研究视角")
        self.assertNotIn("王曰芬", result.title)
        self.assertIn("文件名中的论文标题", " ".join(result.warning_items()))

    def test_pipeline_uses_title_before_split_abstract_and_removes_garbled_summary(self) -> None:
        garbled_line = "欟" * 46
        page = PdfPageDebug(
            page_index=0,
            text=(
                f"{garbled_line}\n"
                "专题序: 多元研究视角下人工智能研究前沿识别与分析\n"
                "研究前沿的概念是1965 年由De S. Price 在“Science”上发表的论文中提出来的。\n"
                "王曰芬\n"
                "●王曰芬\n"
                "1，2，曹嘉君\n"
                "( 1. 南京理工大学经济管理学院，江苏 南京 210094)\n"
                "人工智能研究前沿识别与分析: 基于领域全局演化研究视角\n"
                "*\n"
                "摘\n"
                "要: ［目的/意义］在人工智能持续快速发展的背景下，借助于数据分析进行人工智能领域突变术语的识别。\n"
                "［方法/过程］以WoS 核心合集为数据源，利用突变检测算法识别出突变术语。\n"
                "［结果/结论］移动设备、能源消耗以及标准测试数据等研\n"
                "*\n"
                "本文为国家自然科学基金应急管理项目“人工智能领域研究前沿探测与决策支持”(项目编号: 61842602)\n"
                "和江苏省研究生科研与\n"
                "实践创新计划项目“基于数据科学的专家在线知识创新平台构建研究”(项目编号: KYCX18_0344)\n"
                "的成果之一。\n"
                f"{garbled_line}\n"
                "究发展得较快。［局限］主题术语可能存在分词误差，需追踪到实际文献中的主题词进行修正。\n"
                "关键词: 人工智能; 研究前沿; 突变检测; 突变术语; 前沿演进\n"
                "1 引言\n"
            ),
            preview="sample",
            page_score=10.0,
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
            result = build_paper_result("人工智能研究前沿识别与分析_基于领域全局演化研究视角_王曰芬.pdf", pdf_result)

        self.assertEqual(result.title, "人工智能研究前沿识别与分析: 基于领域全局演化研究视角")
        self.assertIn("研究发展得较快", result.summary_text())
        self.assertNotIn("专题序", result.title)
        self.assertFalse(any(char in result.summary_text() for char in "欟檪殏"))
        self.assertNotIn("项目编号", result.summary_text())
        self.assertNotIn("成果之一", result.summary_text())

    def test_structured_fields_extract_fullwidth_labeled_abstract(self) -> None:
        abstract = (
            "［目的/意义］在人工智能持续快速发展的背景下，借助于数据分析进行人工智能领域突变术语的识别，"
            "揭示人工智能领域的研究前沿及其态势的演变状况，以为科学研究与政策制定提供数据支撑和决策参考。"
            "［方法/过程］在对研究前沿综述的基础上，以WoS 核心合集为数据源，采集与处理人工智能研究的文献数据，"
            "利用突变检测算法识别出突变术语，从整体内容、突变持续区间、突变初始年限以及突变时间和词频相结合的角度"
            "进行研究前沿的识别与具体演进分析。"
            "［结果/结论］人工智能研究前沿由理论研究向技术方法和算法研究演变，整体上处于持续稳定的发展状态中; "
            "在学习模型和算法上出现新的发展思路，以智能应用为目标; 同时，移动设备、能源消耗以及标准测试数据等研究发展得较快。"
            "［局限］主题术语可能存在分词误差，需追踪到实际文献中的主题词进行修正。"
        )

        request = collect_structured_candidates(
            abstract,
            title="人工智能研究前沿识别与分析: 基于领域全局演化研究视角",
            chinese_abstract=abstract,
        )
        result = rewrite_structured_fields(request)

        self.assertTrue(request.debug_info["explicit_abstract_labels_found"])
        self.assertEqual(request.debug_info["candidate_source_strategy"], "cn_abstract_labels")
        self.assertIn("人工智能领域突变术语", result.research_question)
        self.assertIn("WoS 核心合集", result.research_method)
        self.assertIn("突变检测算法", result.research_method)
        self.assertIn("理论研究向技术方法和算法研究演变", result.core_conclusion)
        self.assertNotIn("局限", result.core_conclusion)

    def test_abstract_extraction_does_not_stop_inside_tongyi_punctuation(self) -> None:
        text = (
            "摘要:针对当前我国用水权改革认识不统一、改革面临诸多困难和障碍的现实局面,基于科斯三大\n"
            "定理,深入审视和剖析我国用水权初始分配确权、市场化交易、收储再配置“三部曲”式改革演变的\n"
            "理论逻辑,总结我国用水权制度演进的实践经验、理论依据与改革方向。研究发现:我国用水权改\n"
            "革的阶段性特征与科斯三大定理具有高度吻合性,初始分配确权、市场化交易和收储再配置是我国\n"
            "用水权制度和政策构成的三大核心,尤其是收储再配置机制具有“政府+市场”双重属性和“两手发\n"
            "力”特点,对活跃用水权市场和矫正初始分配结果具有重要的现实贡献。基于此,提出我国规范推\n"
            "进用水权改革、活跃用水权交易市场的相关政策建议。\n"
            "关键词:科斯三大定理;用水权交易;收储再配置;用水权制度演进\n"
        )

        abstract = extract_abstracts(text).chinese_abstract
        request = collect_structured_candidates(
            text,
            title="科斯理论视角下我国用水权制度演进与改革方向",
            chinese_abstract=abstract,
        )
        result = rewrite_structured_fields(request)

        self.assertIn("改革面临诸多困难和障碍", abstract)
        self.assertIn("政策建议", abstract)
        self.assertIn("当前我国用水权改革", result.research_question)
        self.assertIn("基于科斯三大", result.research_method)
        self.assertIn("收储再配置机制", result.core_conclusion)

    def test_abstract_extraction_keeps_decimal_or_values_and_prefers_conclusion_label(self) -> None:
        text = (
            "摘要：目的\n"
            "采用多因素Logistic 回归与随机森林模型分析中小学生近视的影响因素并对比两种模型结果，为近视防控提供多维度科学依据。方法\n"
            "于2019—2024 年，采用分层整群抽样方法，抽取北京市某区小学四年级至高中三年级共10 666名学生开展视力检查与问卷调查。结果\n"
            "Logistic 回归分析结果显示，高学段（初中OR=3.973，95%CI：3.519~4.484；高中OR=6.028，\n"
            "95%CI：5.299~6.858）、父母近视（OR=2.561，95%CI：2.318~2.830）、女生（OR=1.710，95%CI：1.554~1.882）、\n"
            "课间休息地点在教室内（OR=\n"
            "1.164，95%CI：1.041~1.303）是近视的显著危险因素。结论\n"
            "Logistic 回归模型识别出6 种危险因素，随机森林模型识别出5 种重要因素，两种模型均将学段、父母近视、做作业时间过长、父母提醒读写姿势4 种因素列为近视最突出的影响因素。\n"
            "关键词：随机森林模型；学生；近视；影响因素；多因素\n"
        )

        abstract = extract_abstracts(text).chinese_abstract
        request = collect_structured_candidates(
            text,
            title="中小学生近视影响因素的Logistic回归与随机森林模型对比研究",
            chinese_abstract=abstract,
        )
        result = rewrite_structured_fields(request)

        self.assertIn("1.164，95%CI", abstract)
        self.assertIn("两种模型均将学段", abstract)
        self.assertNotIn("OR=", result.core_conclusion)
        self.assertIn("随机森林模型识别出5 种重要因素", result.core_conclusion)

    def test_pdf_preflight_records_strategy_and_quality(self) -> None:
        pdf_bytes = self.build_text_pdf_bytes(
            "A Study on Literature Reading Support for Course Writing\n"
            "Alice Chen, Bob Li\n"
            "Abstract: This paper focuses on single-paper reading in course writing scenarios. "
            "It uses text analysis to organize the parsing workflow and discusses how structured reading "
            "support can improve classroom presentation preparation efficiency.\n"
            "Keywords: course writing; literature reading; text analysis; classroom presentation\n"
            "1 Introduction\n"
            "Course writing requires stable extraction of title, authors, abstract, keywords and methods, "
            "so the parser should evaluate PDF text quality before selecting the most reliable strategy.\n"
        )

        result = extract_pdf_context(pdf_bytes)

        self.assertIn(result.preflight.extraction_strategy, {"pymupdf_text", "pymupdf_blocks"})
        self.assertGreater(result.preflight.quality_score, 0)
        self.assertFalse(result.preflight.is_probably_scanned)
        self.assertTrue(result.extraction_attempts_debug)
        self.assertIn("Literature Reading Support", result.body_text)

    def test_pdf_preflight_reports_short_or_non_text_pdf(self) -> None:
        pdf_bytes = self.build_text_pdf_bytes("short")

        with self.assertRaises(ValueError) as context:
            extract_pdf_context(pdf_bytes)

        self.assertRegex(str(context.exception), r"(文本过少|没有提取到可用文本)")

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

    def test_metadata_recognition_success_keeps_debug_info(self) -> None:
        payload = MetadataRecognitionPayload(
            title="AI 补充标题",
            authors=["张三"],
            keywords=[],
            abstract_zh="",
            abstract_en="",
            note="基于首页识别作者。",
        )

        with patch("services.llm_service._load_relay_settings", return_value=RelaySettings("key", "https://relay.test", "model")):
            with patch("services.llm_service.OpenAI"):
                with patch("services.llm_service._metadata_recognition_with_responses", return_value=(payload, '{"authors":["张三"]}')):
                    result = recognize_metadata_with_llm(
                        title="课程写作辅助中的文献解读研究",
                        authors=[],
                        authors_confidence="none",
                        keywords=["课程写作"],
                        keyword_source_kind="strategy_a_explicit_zh",
                        abstract_zh="摘要原文",
                        abstract_en="",
                        front_text="课程写作辅助中的文献解读研究\n张三\n摘要：摘要原文，本文关注课堂报告写作场景下的文献解读流程。\n关键词：课程写作",
                    )

        self.assertIn("authors", result.fields_supplemented)
        self.assertIn("title", result.fields_supplemented)
        self.assertEqual(result.title, "AI 补充标题")
        self.assertTrue(result.debug_info.get("used_llm"))
        self.assertEqual(result.debug_info.get("attempted_path"), ["responses"])
        self.assertIn("raw_response_text", result.debug_info)

    def test_metadata_extract_stays_running_during_ai_metadata_recognition(self) -> None:
        pdf_result = self.build_pdf_result(
            "课程写作辅助中的文献解读研究\n"
            "摘要：本文关注课程写作场景下的单篇论文整理。\n"
            "关键词：课程写作；文献解读；文本分析；课堂展示\n"
            "1 引言\n"
        )
        progress_events: list[dict[str, object]] = []
        observed_status: list[str] = []

        def fake_recognize_metadata_with_llm(**_: object) -> MetadataRecognitionResult:
            latest_steps = progress_events[-1]["steps"]
            metadata_step = next(step for step in latest_steps if step["id"] == "metadata_extract")
            observed_status.append(str(metadata_step["status"]))
            return MetadataRecognitionResult(
                authors=["张三"],
                fields_supplemented=["authors"],
                debug_info={"used_llm": True},
            )

        tracker = ParseRuntimeTracker(progress_callback=progress_events.append)
        with patch("services.paper_parse_service.recognize_metadata_with_llm", side_effect=fake_recognize_metadata_with_llm):
            build_paper_result("sample.pdf", pdf_result, runtime_tracker=tracker)

        self.assertEqual(observed_status, ["running"])

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
