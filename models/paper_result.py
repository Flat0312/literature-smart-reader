"""Standardized parsed paper result used by page rendering and exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


STRUCTURED_FIELD_LABELS = {
    "research_question": "研究问题",
    "research_method": "研究方法",
    "core_conclusion": "核心结论",
}

STRUCTURED_FIELD_FALLBACKS = {
    "research_question": "未从原文摘要中识别到研究问题",
    "research_method": "未从原文摘要中识别到研究方法",
    "core_conclusion": "未从原文摘要中识别到核心结论",
}

SUMMARY_DISPLAY_FALLBACK = "未能生成摘要内容。"
TEXT_PREVIEW_DISPLAY_FALLBACK = "暂无文本预览。"
KEYWORDS_DISPLAY_FALLBACK = "未提取到关键词"
AUTHORS_DISPLAY_FALLBACK = "未稳定识别作者"
PLAIN_LANGUAGE_SUMMARY_FALLBACK = "根据当前可识别内容，暂可归纳为：这篇论文围绕特定研究问题展开分析，建议结合原文摘要和结论部分进一步核对。"
METHOD_EXPLANATION_FALLBACK = "根据当前可识别内容，论文的方法信息仍较有限，建议结合原文的方法或研究设计部分补充核对。"
INNOVATION_POINTS_FALLBACK = [
    "根据当前可识别内容，论文对研究主题做了相对明确的聚焦，适合整理为课程汇报中的“研究对象与问题”。",
    "当前结果已提取出研究方法和核心结论，可作为课堂展示时概括论文贡献的基础。",
]
LIMITATION_POINTS_FALLBACK = [
    "根据当前可识别内容，部分方法细节和论证过程仍需回看正文，不能仅凭摘要做强结论判断。",
    "若要用于课程论文或综述写作，建议进一步核对样本、数据来源和结论适用范围。",
]
COURSE_PRESENTATION_OUTLINE_FALLBACK = [
    "研究背景与选题缘起：说明论文关注的主题和课程讨论价值。",
    "研究问题与研究对象：概括论文试图回答什么问题、聚焦什么对象。",
    "研究方法与分析路径：简述作者如何展开研究。",
    "主要发现、创新与不足：提炼可直接放入 PPT 或讲稿的核心要点。",
]
COURSE_PAPER_OUTLINE_FALLBACK = [
    "引言：交代研究主题、研究意义与论文基本信息。",
    "文献内容整理：梳理论文的研究问题、方法与核心结论。",
    "论文评价：从创新点与不足两个方面做课程化分析。",
    "可延展讨论：提出可继续比较或展开的相关问题。",
]
LITERATURE_REVIEW_OUTLINE_FALLBACK = [
    "研究主题与研究对象：明确这篇文献可归入的研究主题。",
    "方法路径与主要观点：记录作者采用的方法和形成的核心观点。",
    "创新与不足：整理后续综述写作时可比较的评价维度。",
    "可比较维度：标记与其他文献横向比较时可使用的切入点。",
]


@dataclass(slots=True)
class NormalizedPaperParseResult:
    title: str = ""
    abstract_zh: str = ""
    abstract_en: str = ""
    summary_fallback: str = ""
    authors: list[str] = field(default_factory=list)
    authors_source: str = ""
    authors_confidence: str = ""
    authors_raw_candidates: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    research_question: str = ""
    research_method: str = ""
    core_conclusion: str = ""
    plain_language_summary: str = ""
    method_explanation: str = ""
    innovation_points: list[str] = field(default_factory=list)
    limitation_points: list[str] = field(default_factory=list)
    course_presentation_outline: list[str] = field(default_factory=list)
    course_paper_outline: list[str] = field(default_factory=list)
    literature_review_outline: list[str] = field(default_factory=list)
    source_language: str = ""
    parse_warnings: list[str] = field(default_factory=list)
    keyword_source: str = ""
    structured_backend: str = ""
    structured_note: str = ""
    explicit_abstract_labels_found: bool = False
    llm_supplemented_fields: list[str] = field(default_factory=list)

    def filtered_authors(self) -> list[str]:
        return _compact_text_list(self.authors)

    def filtered_keywords(self) -> list[str]:
        return _compact_text_list(self.keywords)

    def summary_text(self) -> str:
        cleaned = (self.abstract_zh or self.abstract_en or self.summary_fallback or "").strip()
        if not cleaned:
            return SUMMARY_DISPLAY_FALLBACK
        return cleaned

    def primary_summary_language(self) -> str:
        if (self.abstract_zh or "").strip():
            return "zh"
        if (self.abstract_en or "").strip():
            return "en"
        return ""

    def english_abstract_text(self) -> str:
        cleaned = (self.abstract_en or "").strip()
        if cleaned and cleaned != (self.abstract_zh or "").strip():
            return cleaned
        return ""

    def authors_text(self) -> str:
        authors = self.filtered_authors()
        if not authors:
            return AUTHORS_DISPLAY_FALLBACK
        return "、".join(authors)

    def structured_field_text(self, field_name: str, fallback_text: str | None = None) -> str:
        raw_value = getattr(self, field_name, "")
        cleaned = (raw_value or "").strip()
        fallback = fallback_text or STRUCTURED_FIELD_FALLBACKS[field_name]
        if not cleaned or _is_low_quality_structured_text(cleaned):
            return fallback
        return cleaned

    def structured_field_items(self) -> list[tuple[str, str, str, bool]]:
        items: list[tuple[str, str, str, bool]] = []
        for field_name, label in STRUCTURED_FIELD_LABELS.items():
            resolved_text = self.structured_field_text(field_name)
            is_fallback = resolved_text == STRUCTURED_FIELD_FALLBACKS[field_name]
            items.append((field_name, label, resolved_text, is_fallback))
        return items

    def structured_field_count(self) -> int:
        return sum(0 if is_fallback else 1 for _, _, _, is_fallback in self.structured_field_items())

    def warning_items(self) -> list[str]:
        return _compact_text_list(self.parse_warnings)

    def plain_language_summary_text(self) -> str:
        cleaned = (self.plain_language_summary or "").strip()
        return cleaned or PLAIN_LANGUAGE_SUMMARY_FALLBACK

    def method_explanation_text(self) -> str:
        cleaned = (self.method_explanation or "").strip()
        return cleaned or METHOD_EXPLANATION_FALLBACK

    def innovation_items(self) -> list[str]:
        items = _compact_text_list(self.innovation_points)
        return items or list(INNOVATION_POINTS_FALLBACK)

    def limitation_items(self) -> list[str]:
        items = _compact_text_list(self.limitation_points)
        return items or list(LIMITATION_POINTS_FALLBACK)

    def course_presentation_outline_items(self) -> list[str]:
        items = _compact_text_list(self.course_presentation_outline)
        return items or list(COURSE_PRESENTATION_OUTLINE_FALLBACK)

    def course_paper_outline_items(self) -> list[str]:
        items = _compact_text_list(self.course_paper_outline)
        return items or list(COURSE_PAPER_OUTLINE_FALLBACK)

    def literature_review_outline_items(self) -> list[str]:
        items = _compact_text_list(self.literature_review_outline)
        return items or list(LITERATURE_REVIEW_OUTLINE_FALLBACK)

    def as_standard_dict(self) -> dict[str, object]:
        return {
            "title": (self.title or "未识别标题").strip() or "未识别标题",
            "abstract_zh": (self.abstract_zh or "").strip() or "未识别到中文摘要",
            "abstract_en": (self.abstract_en or "").strip() or "未识别到英文摘要",
            "authors": self.filtered_authors() or [AUTHORS_DISPLAY_FALLBACK],
            "authors_source": (self.authors_source or "").strip() or "none",
            "authors_confidence": (self.authors_confidence or "").strip() or "none",
            "authors_raw_candidates": _compact_text_list(self.authors_raw_candidates),
            "keywords": self.filtered_keywords() or [KEYWORDS_DISPLAY_FALLBACK],
            "research_question": self.structured_field_text("research_question"),
            "research_method": self.structured_field_text("research_method"),
            "core_conclusion": self.structured_field_text("core_conclusion"),
            "plain_language_summary": self.plain_language_summary_text(),
            "method_explanation": self.method_explanation_text(),
            "innovation_points": self.innovation_items(),
            "limitation_points": self.limitation_items(),
            "course_presentation_outline": self.course_presentation_outline_items(),
            "course_paper_outline": self.course_paper_outline_items(),
            "literature_review_outline": self.literature_review_outline_items(),
            "source_language": (self.source_language or "").strip() or "unknown",
            "primary_summary_language": self.primary_summary_language() or "unknown",
            "parse_warnings": self.warning_items(),
        }


@dataclass(slots=True)
class PaperResult:
    file_name: str
    parsed_result: NormalizedPaperParseResult
    structured_notice: str = ""
    structured_debug: dict[str, object] = field(default_factory=dict)
    raw_text: str = ""
    text_preview: str = ""
    parse_status: str = "success"
    parse_steps: list[dict[str, object]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    parse_timings: dict[str, int] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.parsed_result.title

    @property
    def source_language(self) -> str:
        return self.parsed_result.source_language

    def filtered_authors(self) -> list[str]:
        return self.parsed_result.filtered_authors()

    def filtered_keywords(self) -> list[str]:
        return self.parsed_result.filtered_keywords()

    def summary_text(self) -> str:
        return self.parsed_result.summary_text()

    def primary_summary_language(self) -> str:
        return self.parsed_result.primary_summary_language()

    def english_abstract_text(self) -> str:
        return self.parsed_result.english_abstract_text()

    def authors_text(self) -> str:
        return self.parsed_result.authors_text()

    def preview_text(self) -> str:
        cleaned = (self.text_preview or "").strip()
        return cleaned or TEXT_PREVIEW_DISPLAY_FALLBACK

    def structured_field_text(self, field_name: str, fallback_text: str | None = None) -> str:
        return self.parsed_result.structured_field_text(field_name, fallback_text=fallback_text)

    def structured_field_items(self) -> list[tuple[str, str, str, bool]]:
        return self.parsed_result.structured_field_items()

    def structured_field_count(self) -> int:
        return self.parsed_result.structured_field_count()

    def warning_items(self) -> list[str]:
        return self.parsed_result.warning_items()

    def plain_language_summary_text(self) -> str:
        return self.parsed_result.plain_language_summary_text()

    def method_explanation_text(self) -> str:
        return self.parsed_result.method_explanation_text()

    def innovation_items(self) -> list[str]:
        return self.parsed_result.innovation_items()

    def limitation_items(self) -> list[str]:
        return self.parsed_result.limitation_items()

    def course_presentation_outline_items(self) -> list[str]:
        return self.parsed_result.course_presentation_outline_items()

    def course_paper_outline_items(self) -> list[str]:
        return self.parsed_result.course_paper_outline_items()

    def literature_review_outline_items(self) -> list[str]:
        return self.parsed_result.literature_review_outline_items()

    def as_standard_dict(self) -> dict[str, object]:
        standard = self.parsed_result.as_standard_dict()
        standard["parse_status"] = (self.parse_status or "success").strip() or "success"
        standard["parse_warnings"] = self.warning_items()
        standard["parse_errors"] = _compact_text_list(self.parse_errors)
        return standard

    def has_meaningful_content(self) -> bool:
        if self.filtered_authors():
            return True
        if self.filtered_keywords():
            return True
        if self.structured_field_count() > 0:
            return True
        return self.summary_text() != SUMMARY_DISPLAY_FALLBACK

    def export_file_stem(self) -> str:
        base_name = Path(self.file_name or "").stem.strip()
        title = (self.title or "").strip()
        candidate = base_name or title or "文献"
        sanitized = re.sub(r'[\\/:*?"<>|]+', "_", candidate).strip(" ._")
        return sanitized or "文献"

    def parse_feedback_dict(self) -> dict[str, object]:
        return {
            "status": (self.parse_status or "success").strip() or "success",
            "steps": list(self.parse_steps),
            "warnings": self.warning_items(),
            "errors": _compact_text_list(self.parse_errors),
            "timings": dict(self.parse_timings),
        }


def _is_low_quality_structured_text(text: str) -> bool:
    ascii_count = sum(1 for character in text if character.isascii() and character.isalpha())
    zh_count = sum(1 for character in text if "\u4e00" <= character <= "\u9fff")
    if ascii_count > zh_count * 1.3 and ascii_count >= 18:
        return True
    return text == "暂未识别到高置信度结果。"


def _compact_text_list(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        compacted.append(cleaned)
    return compacted
