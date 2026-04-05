"""Unified four-stage parsing pipeline for Chinese academic papers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

from models.paper_result import NormalizedPaperParseResult
from services.llm_service import (
    RelayConfigError,
    RelayRequestError,
    get_relay_env_status,
    rewrite_structured_result,
)
from services.metadata_service import (
    KeywordExtractionResult,
    TitleExtractionResult,
    extract_keywords_with_source,
    extract_title_with_source,
)
from services.pdf_service import PdfExtractionResult
from services.paper_parse_service import build_paper_result
from services.structure_service import collect_structured_candidates
from services.structured_rewrite_service import rewrite_structured_fields
from services.summary_service import extract_abstract_sections
from utils.text_utils import build_preview, clean_extracted_text, is_noise_line, normalize_line, normalize_whitespace, sanitize_metadata_fragments

BODY_OPENING_PAGE_COUNT = 2
HEADER_FOOTER_LINE_COUNT = 2
TEXT_PREVIEW_LENGTH = 2200

ZH_KEYWORD_LABEL_PATTERN = r"(?:〔\s*(?:关键词|关键字)\s*〕|\[\s*(?:关键词|关键字)\s*\]|(?:关键词|关键字))\s*[：:]?"
EN_KEYWORD_LABEL_PATTERN = r"(?:keywords?|key words|index terms?)\s*[：:]?"
INTRODUCTION_BOUNDARY_PATTERN = r"(?:引言|绪论|问题提出|0[.、]?\s*引言|1[.、]?\s*引言|1[.、]|一、|第一章)"


@dataclass(slots=True)
class DocumentStructureBlocks:
    title_en: str = ""
    authors: str = ""
    abstract_zh: str = ""
    abstract_en: str = ""
    keywords_zh: str = ""
    keywords_en: str = ""
    body_start_label: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PrecleanedDocument:
    full_text: str = ""
    body_text: str = ""
    priority_text: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ParsePipelineResult:
    normalized_result: NormalizedPaperParseResult
    structured_notice: str
    title_result: TitleExtractionResult
    debug_info: dict[str, object] = field(default_factory=dict)
    cleaned_body_text: str = ""
    text_preview: str = ""


def parse_pdf_document(pdf_result: PdfExtractionResult) -> ParsePipelineResult:
    paper_result = build_paper_result("in_memory.pdf", pdf_result)
    title_result = TitleExtractionResult(
        title=paper_result.title,
        source_page_index=paper_result.structured_debug.get("pdf_debug", {}).get("title_source_page_index"),
        source_kind=str(paper_result.structured_debug.get("pdf_debug", {}).get("title_source", "")),
        debug_info={
            "candidates": paper_result.structured_debug.get("pdf_debug", {}).get("title_candidates", []),
        },
    )
    return ParsePipelineResult(
        normalized_result=paper_result.parsed_result,
        structured_notice=paper_result.structured_notice,
        title_result=title_result,
        debug_info=paper_result.structured_debug,
        cleaned_body_text=paper_result.raw_text,
        text_preview=paper_result.text_preview,
    )


def preclean_pdf_document(pdf_result: PdfExtractionResult) -> PrecleanedDocument:
    page_texts = [page.text for page in pdf_result.pages]
    repeated_lines = _detect_repeated_header_footer_lines(page_texts)
    full_text = _clean_page_texts(page_texts, repeated_lines)
    body_page_texts = page_texts[pdf_result.body_start_page_index:]
    body_text = _clean_page_texts(body_page_texts, repeated_lines) or full_text
    priority_text = _clean_page_texts(body_page_texts[:BODY_OPENING_PAGE_COUNT], repeated_lines)

    return PrecleanedDocument(
        full_text=full_text,
        body_text=body_text,
        priority_text=priority_text,
        debug_info={
            "repeated_header_footer_lines": sorted(repeated_lines),
            "body_start_page_index": pdf_result.body_start_page_index,
            "full_text_length": len(full_text),
            "body_text_length": len(body_text),
            "priority_text_length": len(priority_text),
        },
    )


def detect_document_structure(body_text: str, *, priority_text: str = "") -> DocumentStructureBlocks:
    candidate_texts = [text for text in (priority_text, body_text) if text.strip()]
    abstract_result = extract_abstract_sections(body_text, priority_text=priority_text)
    keywords_zh = _extract_keyword_block(candidate_texts, language="zh")
    keywords_en = _extract_keyword_block(candidate_texts, language="en")
    english_title = _extract_english_title(candidate_texts[0] if candidate_texts else body_text)
    authors = _extract_author_block(candidate_texts[0] if candidate_texts else body_text)
    body_start_label = _detect_body_start_label(body_text)

    return DocumentStructureBlocks(
        title_en=english_title,
        authors=authors,
        abstract_zh=abstract_result.chinese_abstract,
        abstract_en=abstract_result.english_abstract,
        keywords_zh=keywords_zh,
        keywords_en=keywords_en,
        body_start_label=body_start_label,
        debug_info={
            "source_language": abstract_result.source_language,
            "has_chinese_abstract": bool(abstract_result.chinese_abstract),
            "has_english_abstract": bool(abstract_result.english_abstract),
        },
    )


def _extract_structured_result(
    precleaned: PrecleanedDocument,
    title_result: TitleExtractionResult,
    structure_blocks: DocumentStructureBlocks,
    keyword_result: KeywordExtractionResult,
) -> tuple[NormalizedPaperParseResult, str, dict[str, object]]:
    structured_request = collect_structured_candidates(
        precleaned.body_text,
        title=title_result.title or "",
        priority_text=precleaned.priority_text,
        chinese_abstract=structure_blocks.abstract_zh,
    )
    llm_debug = {"env": get_relay_env_status()}
    explicit_abstract_labels_found = bool(structured_request.debug_info.get("explicit_abstract_labels_found", False))
    populated_rule_fields = sum(1 for candidate in structured_request.candidates.values() if candidate.text.strip())

    if explicit_abstract_labels_found or populated_rule_fields == 3:
        structured_result = rewrite_structured_fields(structured_request)
        structured_notice = "当前结构化字段优先来自中文摘要原文标签或规则抽取。"
        llm_debug = {**llm_debug, **structured_result.debug_info, "backend": "local_rule_priority"}
    else:
        try:
            structured_result = rewrite_structured_result(structured_request)
            supplemented_fields = structured_result.debug_info.get("supplemented_fields", [])
            if structured_result.backend == "relay_precheck":
                structured_notice = "当前未从中文摘要中识别到足够明确的结构化字段。"
            elif supplemented_fields:
                field_labels = "、".join(_field_label(field_name) for field_name in supplemented_fields)
                structured_notice = f"以下空缺字段由模型概括补充：{field_labels}；其余字段保留原文规则抽取结果。"
            else:
                structured_notice = "当前结构化字段优先保留原文规则抽取结果。"
            llm_debug = {**llm_debug, **structured_result.debug_info}
        except RelayConfigError as exc:
            structured_result = rewrite_structured_fields(structured_request)
            structured_notice = f"当前优先显示原文摘要/规则抽取结果：{exc}"
            llm_debug = {**llm_debug, "backend": "relay_config_error", "valid_return": False, "error": str(exc)}
        except RelayRequestError as exc:
            structured_result = rewrite_structured_fields(structured_request)
            structured_notice = f"当前优先显示原文摘要/规则抽取结果：{exc}"
            llm_debug = {**llm_debug, **structured_result.debug_info, "backend": "relay_request_error_local_fallback", "error": str(exc)}

    normalized_result = NormalizedPaperParseResult(
        title=title_result.title or "未识别标题",
        abstract_zh=structure_blocks.abstract_zh,
        abstract_en=structure_blocks.abstract_en,
        keywords=keyword_result.keywords,
        research_question=structured_result.research_question,
        research_method=structured_result.research_method,
        core_conclusion=structured_result.core_conclusion,
        source_language="zh" if structure_blocks.abstract_zh else "en" if structure_blocks.abstract_en else "",
        keyword_source=keyword_result.source_kind,
        structured_backend=structured_result.backend,
        structured_note=structured_result.note,
        explicit_abstract_labels_found=explicit_abstract_labels_found,
        llm_supplemented_fields=list(structured_result.debug_info.get("supplemented_fields", [])),
    )
    debug_info = {
        "candidate_debug": _build_candidate_debug(structured_request),
        "llm_debug": llm_debug,
        "final_result": {
            "research_question": structured_result.research_question,
            "research_method": structured_result.research_method,
            "core_conclusion": structured_result.core_conclusion,
            "backend": structured_result.backend,
            "confidence": structured_result.confidence,
            "note": structured_result.note,
        },
    }
    return normalized_result, structured_notice, debug_info


def _build_parse_warnings(
    normalized_result: NormalizedPaperParseResult,
    keyword_result: KeywordExtractionResult,
) -> list[str]:
    warnings: list[str] = []
    if not normalized_result.abstract_zh and normalized_result.abstract_en:
        warnings.append("未识别到中文摘要，当前使用英文 Abstract 作为备用摘要。")
    if not normalized_result.filtered_keywords():
        warnings.append("未识别到稳定关键词。")
    elif keyword_result.source_kind == "frequency_fallback":
        warnings.append("未定位到显式关键词区，当前关键词来自正文频次回退。")
    warnings.extend(keyword_result.warnings)
    if keyword_result.debug_info.get("low_confidence"):
        warnings.append("关键词提取置信度较低。")
    if normalized_result.llm_supplemented_fields:
        labels = "、".join(_field_label(field_name) for field_name in normalized_result.llm_supplemented_fields)
        warnings.append(f"结构化字段中的 {labels} 由模型概括补充。")
    if normalized_result.structured_field_count() == 0:
        warnings.append("未从原文摘要中识别到稳定的结构化字段。")
    return _dedupe_warnings(warnings)


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in warnings:
        normalized = normalize_line(warning)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _clean_page_texts(page_texts: list[str], repeated_lines: set[str]) -> str:
    cleaned_pages: list[str] = []
    for page_text in page_texts:
        lines = [normalize_whitespace(line) for line in page_text.splitlines()]
        cleaned_lines: list[str] = []
        blank_pending = False
        for raw_line in lines:
            line = normalize_line(raw_line)
            if not line:
                if cleaned_lines:
                    blank_pending = True
                continue
            if line in repeated_lines:
                continue
            line = sanitize_metadata_fragments(line)
            if not line or is_noise_line(line):
                continue
            if blank_pending and cleaned_lines:
                cleaned_lines.append("")
            cleaned_lines.append(line)
            blank_pending = False
        page_cleaned = clean_extracted_text("\n".join(cleaned_lines))
        if page_cleaned:
            cleaned_pages.append(page_cleaned)
    return clean_extracted_text("\n\n".join(cleaned_pages))


def _detect_repeated_header_footer_lines(page_texts: list[str]) -> set[str]:
    candidate_counter: Counter[str] = Counter()
    for page_text in page_texts:
        lines = [normalize_line(line) for line in page_text.splitlines() if normalize_line(line)]
        edge_lines = lines[:HEADER_FOOTER_LINE_COUNT] + lines[-HEADER_FOOTER_LINE_COUNT:]
        for line in edge_lines:
            if _is_header_footer_candidate(line):
                candidate_counter[line] += 1
    return {line for line, count in candidate_counter.items() if count >= 2}


def _is_header_footer_candidate(line: str) -> bool:
    if len(line) < 2 or len(line) > 80:
        return False
    if re.fullmatch(r"(?:第?\s*\d+\s*页|\d+\s*/\s*\d+|\d+)", line):
        return True
    if re.search(r"(doi|issn|cn\b|网络首发|available online|published online|收稿日期|修回日期|录用日期)", line, re.IGNORECASE):
        return True
    if re.search(r"(学报|期刊|杂志|journal|review)", line, re.IGNORECASE) and not re.search(r"[。！？!?；;]", line):
        return True
    if len(line) <= 32 and not re.search(r"[。！？!?；;]", line):
        return True
    return False


def _extract_keyword_block(candidate_texts: list[str], *, language: str) -> str:
    patterns = (
        [rf"{ZH_KEYWORD_LABEL_PATTERN}\s*(.+?)(?=(?:〔?\s*(?:中图法分类号|引用本文格式|英文标题|英文题名)\s*〕?|Abstract|abstract|英文摘要|Keywords?|key words|index terms?|{INTRODUCTION_BOUNDARY_PATTERN}|参考文献|$))"]
        if language == "zh"
        else [rf"{EN_KEYWORD_LABEL_PATTERN}\s*(.+?)(?=(?:摘要|中图法分类号|引用本文格式|{INTRODUCTION_BOUNDARY_PATTERN}|references?|$))"]
    )
    for text in candidate_texts:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            block = sanitize_metadata_fragments(match.group(1))
            block = normalize_whitespace(block)
            block = block.lstrip("〕】])} ").strip(" ：:；;，,、")
            if block:
                return block
    return ""


def _extract_author_block(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines()[:12] if normalize_line(line)]
    for line in lines[1:6]:
        if re.search(r"(摘要|abstract|关键词|关键字|keywords?)", line, re.IGNORECASE):
            break
        if re.search(r"(大学|学院|研究院|研究所|department|university|college)", line, re.IGNORECASE):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff·\s]{2,30}", line):
            return line
    return ""


def _extract_english_title(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines()[:18] if normalize_line(line)]
    for line in lines:
        if re.search(r"(摘要|关键词|引言)", line):
            break
        if re.search(r"[A-Za-z]{4,}", line) and re.search(r"\s", line):
            if 10 <= len(line) <= 180 and not re.search(r"(doi|issn|author|department|university|college)", line, re.IGNORECASE):
                return line
    return ""


def _detect_body_start_label(text: str) -> str:
    match = re.search(INTRODUCTION_BOUNDARY_PATTERN, text, re.IGNORECASE)
    return match.group(0) if match else ""


def _build_candidate_debug(structured_request) -> dict[str, object]:
    question_candidate = structured_request.candidates.get("research_question")
    method_candidate = structured_request.candidates.get("research_method")
    conclusion_candidate = structured_request.candidates.get("core_conclusion")
    field_debug = structured_request.debug_info.get("field_debug", {})
    return {
        "detected_title": structured_request.title,
        "candidate_source_strategy": structured_request.debug_info.get("candidate_source_strategy", ""),
        "explicit_abstract_labels_found": structured_request.debug_info.get("explicit_abstract_labels_found", False),
        "explicit_label_fields": structured_request.debug_info.get("explicit_label_fields", []),
        "priority_preview": structured_request.debug_info.get("priority_preview", ""),
        "abstract_priority_found": structured_request.debug_info.get("abstract_priority_found", False),
        "abstract_preview": structured_request.debug_info.get("abstract_preview", ""),
        "keywords_preview": structured_request.debug_info.get("keywords_preview", ""),
        "research_question_candidates": question_candidate.text if question_candidate else "",
        "research_method_candidates": method_candidate.text if method_candidate else "",
        "core_conclusion_candidates": conclusion_candidate.text if conclusion_candidate else "",
        "research_question_source": question_candidate.source_hint if question_candidate else "",
        "research_method_source": method_candidate.source_hint if method_candidate else "",
        "core_conclusion_source": conclusion_candidate.source_hint if conclusion_candidate else "",
        "research_question_filtered": field_debug.get("research_question", {}).get("filtered_out", []),
        "research_method_filtered": field_debug.get("research_method", {}).get("filtered_out", []),
        "core_conclusion_filtered": field_debug.get("core_conclusion", {}).get("filtered_out", []),
    }


def _field_label(field_name: str) -> str:
    mapping = {
        "research_question": "研究问题",
        "research_method": "研究方法",
        "core_conclusion": "核心结论",
    }
    return mapping.get(field_name, field_name)
