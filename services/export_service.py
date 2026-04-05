"""Local export helpers for parsed paper results."""

from __future__ import annotations

from dataclasses import dataclass

from models.paper_result import PaperResult
from utils.text_utils import reflow_text_for_display


@dataclass(slots=True)
class ExportPackage:
    markdown_text: str
    markdown_filename: str
    txt_text: str
    txt_filename: str


def build_export_package(result: PaperResult) -> ExportPackage:
    base_name = _build_export_basename(result)
    markdown_text = _build_markdown_text(result)
    txt_text = _build_txt_text(result)
    return ExportPackage(
        markdown_text=markdown_text,
        markdown_filename=f"{base_name}.md",
        txt_text=txt_text,
        txt_filename=f"{base_name}.txt",
    )


def _build_export_basename(result: PaperResult) -> str:
    return f"{result.export_file_stem()}_解析结果"


def _build_markdown_text(result: PaperResult) -> str:
    standard = result.as_standard_dict()
    lines = [
        "# 文献解析结果",
        "",
        f"- 文件名：{_display_text(result.file_name, '未记录文件名')}",
        f"- 论文标题：{standard['title']}",
        f"- 作者：{_format_inline_list(list(standard['authors']), separator='、')}",
        f"- 解析状态：已完成解析",
        f"- 主摘要语言：{_summary_language_label(str(standard['primary_summary_language']))}",
        "",
        "## 基础摘要",
        _display_text(str(standard["abstract_zh"]), result.summary_text()),
        "",
        "## 关键词",
        _format_keywords(list(standard["keywords"])),
        "",
    ]

    if result.english_abstract_text() and result.english_abstract_text() != result.summary_text():
        lines.extend(
            [
                "## English Abstract",
                reflow_text_for_display(result.english_abstract_text()),
                "",
            ]
        )

    lines.extend(
        [
            "## 结构化提取",
            "",
            "### 研究问题",
            str(standard["research_question"]),
            "",
            "### 研究方法",
            str(standard["research_method"]),
            "",
            "### 核心结论",
            str(standard["core_conclusion"]),
            "",
            "## AI 解读",
            "",
            "### 通俗摘要",
            str(standard["plain_language_summary"]),
            "",
            "### 研究方法说明",
            str(standard["method_explanation"]),
            "",
            "### 创新点分析",
            _format_markdown_list(list(standard["innovation_points"])),
            "",
            "### 不足分析",
            _format_markdown_list(list(standard["limitation_points"])),
            "",
            "## 写作输出",
            "",
            "### 课程汇报提纲",
            _format_markdown_list(list(standard["course_presentation_outline"])),
            "",
            "### 课程论文提纲",
            _format_markdown_list(list(standard["course_paper_outline"])),
            "",
            "### 文献综述基础框架",
            _format_markdown_list(list(standard["literature_review_outline"])),
        ]
    )

    preview = reflow_text_for_display(result.preview_text())
    if preview and preview != "暂无文本预览。":
        lines.extend(
            [
                "",
                "## 原文预览",
                preview,
            ]
        )

    lines.extend(
        [
            "",
            "## 说明",
            _display_text(
                result.structured_notice,
                "本文件由本地导出生成，适合直接保存和二次整理。",
            ),
        ]
    )
    warnings = [str(item) for item in standard["parse_warnings"]]
    if warnings:
        lines.extend(["", "## 解析提示", *warnings])
    return "\n".join(lines).strip() + "\n"


def _build_txt_text(result: PaperResult) -> str:
    standard = result.as_standard_dict()
    lines = [
        "文献解析结果",
        "============",
        "",
        f"文件名: {_display_text(result.file_name, '未记录文件名')}",
        f"论文标题: {standard['title']}",
        f"作者: {_format_inline_list(list(standard['authors']), separator='、')}",
        "解析状态: 已完成解析",
        f"主摘要语言: {_summary_language_label(str(standard['primary_summary_language']))}",
        "",
        "基础摘要",
        "--------",
        _display_text(str(standard["abstract_zh"]), result.summary_text()),
        "",
        "关键词",
        "------",
        _format_keywords(list(standard["keywords"]), separator="、"),
        "",
    ]

    if result.english_abstract_text() and result.english_abstract_text() != result.summary_text():
        lines.extend(
            [
                "English Abstract",
                "----------------",
                reflow_text_for_display(result.english_abstract_text()),
                "",
            ]
        )

    lines.extend(
        [
            "结构化提取",
            "--------",
            f"研究问题: {standard['research_question']}",
            f"研究方法: {standard['research_method']}",
            f"核心结论: {standard['core_conclusion']}",
            "",
            "AI 解读",
            "--------",
            f"通俗摘要: {standard['plain_language_summary']}",
            f"研究方法说明: {standard['method_explanation']}",
            "创新点分析:",
            _format_txt_list(list(standard["innovation_points"])),
            "",
            "不足分析:",
            _format_txt_list(list(standard["limitation_points"])),
            "",
            "写作输出",
            "--------",
            "课程汇报提纲:",
            _format_txt_list(list(standard["course_presentation_outline"])),
            "",
            "课程论文提纲:",
            _format_txt_list(list(standard["course_paper_outline"])),
            "",
            "文献综述基础框架:",
            _format_txt_list(list(standard["literature_review_outline"])),
        ]
    )

    preview = reflow_text_for_display(result.preview_text())
    if preview and preview != "暂无文本预览。":
        lines.extend(["", "原文预览", "--------", preview])

    lines.extend(
        [
            "",
            "说明",
            "----",
            _display_text(result.structured_notice, "本文件由本地导出生成，适合直接保存和二次整理。"),
        ]
    )
    warnings = [str(item) for item in standard["parse_warnings"]]
    if warnings:
        lines.extend(["", "解析提示", "--------", *warnings])
    return "\n".join(lines).strip() + "\n"


def _format_keywords(keywords: list[str], separator: str = " / ") -> str:
    cleaned = [keyword.strip() for keyword in keywords if keyword and keyword.strip()]
    if not cleaned:
        return "未提取到关键词"
    return separator.join(cleaned)


def _format_inline_list(items: list[str], separator: str = " / ") -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return "暂无内容"
    return separator.join(cleaned)


def _format_markdown_list(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return "- 暂无内容"
    return "\n".join(f"- {item}" for item in cleaned)


def _format_txt_list(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return "- 暂无内容"
    return "\n".join(f"- {item}" for item in cleaned)


def _summary_language_label(value: str) -> str:
    if value == "zh":
        return "中文摘要优先"
    if value == "en":
        return "英文摘要回退"
    return "未知"


def _display_text(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    return text if text else fallback
