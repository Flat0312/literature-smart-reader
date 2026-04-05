"""Unified four-stage parsing pipeline for Chinese academic papers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from time import perf_counter

from config.settings import (
    STRUCTURED_LLM_MIN_ABSTRACT_CHARS,
    STRUCTURED_LLM_MIN_CANDIDATE_CHARS,
    TEXT_PREVIEW_LENGTH,
)
from models.paper_result import NormalizedPaperParseResult, PaperResult
from services.llm_service import (
    CourseSupportRequest,
    RelayConfigError,
    RelayRequestError,
    fallback_keywords_with_llm,
    generate_course_support_material,
    get_relay_env_status,
    rewrite_structured_result,
)
from services.metadata_service import (
    AuthorExtractionResult,
    KeywordExtractionResult,
    extract_authors_with_source,
    extract_keywords_with_source,
    extract_title_with_source,
)
from services.pdf_service import PdfExtractionResult, extract_pdf_context
from services.structure_service import collect_structured_candidates
from services.structured_rewrite_service import rewrite_structured_fields
from services.summary_service import extract_abstracts
from utils.text_utils import build_preview, clean_extracted_text, normalize_line, normalize_whitespace, reflow_text_for_display

HEADER_FOOTER_SCAN_LINES = 3
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+[.、]?\s*|[一二三四五六七八九十]+[、.]?\s*|[0-9]+\s+)?"
    r"(?:摘\s*要|摘要|关键词|关键字|英文摘要|abstract|keywords?|key words|index terms?|"
    r"引言|绪论|参考文献|结论|研究方法|研究结论)\b",
    re.IGNORECASE,
)
BODY_START_PATTERN = re.compile(r"^\s*(?:0|1|一|Ⅰ|I)?\s*[.、]?\s*(引言|绪论)\b", re.IGNORECASE)
PARSE_STEP_DEFINITIONS = [
    ("pdf_read", "正在读取 PDF"),
    ("text_extract", "正在提取文本"),
    ("metadata_extract", "正在识别标题/作者/关键词"),
    ("structured_extract", "正在生成摘要与结构化内容"),
    ("ai_interpretation", "正在生成 AI 解读"),
    ("writing_outline", "正在生成写作提纲"),
    ("completed", "解析完成"),
]


@dataclass(slots=True)
class ParseTextBundle:
    full_text: str
    body_text: str
    priority_text: str
    front_text: str
    repeated_artifacts: list[str] = field(default_factory=list)
    page_texts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructureBlocks:
    abstract_zh: str = ""
    abstract_en: str = ""
    display_summary: str = ""
    keywords_zh_text: str = ""
    keywords_en_text: str = ""
    authors: list[str] = field(default_factory=list)
    authors_source: str = ""
    authors_confidence: str = ""
    authors_raw_candidates: list[str] = field(default_factory=list)
    authors_cleaned: list[str] = field(default_factory=list)
    title_en: str = ""
    body_start_label: str = ""
    source_language: str = ""
    parse_warnings: list[str] = field(default_factory=list)
    debug_info: dict[str, object] = field(default_factory=dict)


class ParsePipelineError(RuntimeError):
    def __init__(self, *, category: str, stage: str, user_message: str, parse_feedback: dict[str, object] | None = None) -> None:
        super().__init__(user_message)
        self.category = category
        self.stage = stage
        self.user_message = user_message
        self.parse_feedback = parse_feedback or {}


class ParseRuntimeTracker:
    def __init__(self, progress_callback=None) -> None:
        self.progress_callback = progress_callback
        self.steps = [
            {
                "id": step_id,
                "label": label,
                "status": "pending",
                "detail": "",
                "duration_ms": 0,
            }
            for step_id, label in PARSE_STEP_DEFINITIONS
        ]
        self._step_index = {step["id"]: index for index, step in enumerate(self.steps)}
        self._started_at: dict[str, float] = {}
        self.timings: dict[str, int] = {}

    def start(self, step_id: str, detail: str = "") -> None:
        step = self._step(step_id)
        step["status"] = "running"
        step["detail"] = detail
        self._started_at[step_id] = perf_counter()
        self.emit(status="running")

    def complete(self, step_id: str, detail: str = "") -> None:
        step = self._step(step_id)
        step["status"] = "completed"
        step["detail"] = detail
        step["duration_ms"] = self._elapsed_ms(step_id)
        self.emit(status="running")

    def partial(self, step_id: str, detail: str = "") -> None:
        step = self._step(step_id)
        step["status"] = "partial"
        step["detail"] = detail
        step["duration_ms"] = self._elapsed_ms(step_id)
        self.emit(status="running")

    def fail(self, step_id: str, detail: str = "") -> None:
        step = self._step(step_id)
        step["status"] = "failed"
        step["detail"] = detail
        step["duration_ms"] = self._elapsed_ms(step_id)
        self.emit(status="failed", errors=[detail] if detail else [])

    def set_timing(self, key: str, started_at: float) -> None:
        self.timings[key] = max(int((perf_counter() - started_at) * 1000), 0)

    def feedback(
        self,
        *,
        status: str,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, object]:
        current_stage = next(
            (step["label"] for step in self.steps if step["status"] == "running"),
            "",
        )
        return {
            "status": status,
            "current_stage": current_stage,
            "steps": self.snapshot(),
            "warnings": _dedupe_items(warnings or []),
            "errors": _dedupe_items(errors or []),
            "timings": dict(self.timings),
        }

    def emit(
        self,
        *,
        status: str,
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(self.feedback(status=status, warnings=warnings, errors=errors))

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "id": step["id"],
                "label": step["label"],
                "status": step["status"],
                "detail": step["detail"],
                "duration_ms": step["duration_ms"],
            }
            for step in self.steps
        ]

    def _step(self, step_id: str) -> dict[str, object]:
        return self.steps[self._step_index[step_id]]

    def _elapsed_ms(self, step_id: str) -> int:
        started_at = self._started_at.get(step_id)
        if started_at is None:
            return 0
        return max(int((perf_counter() - started_at) * 1000), 0)


def parse_uploaded_pdf(file_name: str, file_bytes: bytes, progress_callback=None) -> PaperResult:
    tracker = ParseRuntimeTracker(progress_callback=progress_callback)
    total_started_at = perf_counter()

    tracker.start("pdf_read", "正在检查文件并载入 PDF 页面。")
    tracker.start("text_extract", "正在提取可复制文本。")
    pdf_started_at = perf_counter()
    try:
        pdf_result = extract_pdf_context(file_bytes)
    except ValueError as exc:
        message = str(exc)
        category = _classify_pdf_or_text_error(message)
        failed_step = "pdf_read" if category == "pdf_read_failed" else "text_extract"
        tracker.set_timing("pdf_read_ms", pdf_started_at)
        tracker.set_timing("text_extract_ms", pdf_started_at)
        tracker.fail(failed_step, message)
        parse_feedback = tracker.feedback(status="failed", errors=[message])
        parse_feedback["error_category"] = category
        parse_feedback["error_stage"] = failed_step
        raise ParsePipelineError(
            category=category,
            stage=failed_step,
            user_message=message,
            parse_feedback=parse_feedback,
        ) from exc

    tracker.set_timing("pdf_read_ms", pdf_started_at)
    tracker.set_timing("text_extract_ms", pdf_started_at)
    tracker.complete("pdf_read", f"PDF 已载入，共 {len(pdf_result.pages)} 页。")
    tracker.complete("text_extract", f"已提取正文窗口文本，共 {len(pdf_result.body_text)} 个字符。")

    try:
        result = build_paper_result(file_name, pdf_result, runtime_tracker=tracker)
    except ParsePipelineError:
        raise
    except Exception as exc:
        message = f"解析过程出现错误：{exc}"
        tracker.fail("structured_extract", message)
        parse_feedback = tracker.feedback(status="failed", errors=[message])
        parse_feedback["error_category"] = "structured_extract_failed"
        parse_feedback["error_stage"] = "structured_extract"
        raise ParsePipelineError(
            category="structured_extract_failed",
            stage="structured_extract",
            user_message=message,
            parse_feedback=parse_feedback,
        ) from exc

    tracker.set_timing("total_parse_ms", total_started_at)
    tracker.start("completed", "正在整理最终结果。")
    completion_message = "解析完成。"
    if result.parse_status == "partial_success":
        completion_message = "基础解析已完成，但仍有字段需要人工复核。"
        tracker.partial("completed", completion_message)
    else:
        tracker.complete("completed", completion_message)

    result.parse_steps = tracker.snapshot()
    result.parse_timings = dict(tracker.timings)
    if isinstance(result.structured_debug.get("final_result"), dict):
        result.structured_debug["final_result"]["parse_timings"] = dict(result.parse_timings)
        result.structured_debug["final_result"]["parse_status"] = result.parse_status
    if isinstance(result.structured_debug.get("pipeline_debug"), dict):
        result.structured_debug["pipeline_debug"]["parse_status"] = result.parse_status
    tracker.emit(
        status=result.parse_status,
        warnings=result.warning_items(),
        errors=result.parse_errors,
    )
    return result


def build_paper_result(
    file_name: str,
    pdf_result: PdfExtractionResult,
    *,
    runtime_tracker: ParseRuntimeTracker | None = None,
) -> PaperResult:
    tracker = runtime_tracker or ParseRuntimeTracker()

    text_started_at = perf_counter()
    text_bundle = build_parse_text_bundle(pdf_result)
    if "text_extract_ms" not in tracker.timings:
        tracker.set_timing("text_extract_ms", text_started_at)

    tracker.start("metadata_extract", "正在识别标题、作者和关键词。")
    metadata_started_at = perf_counter()
    try:
        title_result = extract_title_with_source(
            text_bundle.body_text,
            pdf_result.first_page_blocks,
            page_snapshots=pdf_result.pages,
            body_start_page_index=pdf_result.body_start_page_index,
        )
        title = title_result.title or "未识别标题"

        structure_blocks = extract_structure_blocks(
            text_bundle.priority_text,
            fallback_text=text_bundle.body_text,
            title_hint=title,
            front_text=text_bundle.front_text,
        )
        keyword_result = extract_keywords_with_source(
            text_bundle.body_text,
            title=title,
            priority_text=text_bundle.priority_text,
            chinese_keyword_block=structure_blocks.keywords_zh_text,
            english_keyword_block=structure_blocks.keywords_en_text,
        )
        keyword_result = _stabilize_keyword_result(
            keyword_result,
            title=title,
            priority_text=text_bundle.priority_text,
            body_text=text_bundle.body_text,
        )
    except Exception as exc:
        tracker.set_timing("metadata_extract_ms", metadata_started_at)
        tracker.fail("metadata_extract", "标题、作者或关键词识别失败。")
        parse_feedback = tracker.feedback(status="failed", errors=[f"标题、作者或关键词识别失败：{exc}"])
        parse_feedback["error_category"] = "metadata_extract_failed"
        parse_feedback["error_stage"] = "metadata_extract"
        raise ParsePipelineError(
            category="metadata_extract_failed",
            stage="metadata_extract",
            user_message=f"标题、作者或关键词识别失败：{exc}",
            parse_feedback=parse_feedback,
        ) from exc

    tracker.set_timing("metadata_extract_ms", metadata_started_at)
    metadata_detail = "已完成标题、作者和关键词识别。"
    if not structure_blocks.authors:
        metadata_detail = "标题和关键词已识别，作者仍未稳定识别。"
        tracker.partial("metadata_extract", metadata_detail)
    else:
        tracker.complete("metadata_extract", metadata_detail)

    tracker.start("structured_extract", "正在生成主摘要与结构化内容。")
    structured_started_at = perf_counter()
    structured_request = collect_structured_candidates(
        text_bundle.body_text,
        title=title,
        priority_text=text_bundle.priority_text,
        chinese_abstract=structure_blocks.abstract_zh,
    )
    llm_debug_seed = _build_llm_debug_seed(structured_request)
    structured_request.debug_info["llm_input_source"] = llm_debug_seed["llm_input_source"]
    structured_request.debug_info["abstract_fallback_enabled"] = llm_debug_seed["abstract_fallback_enabled"]

    structured_result, structured_notice, llm_debug = _resolve_structured_result(structured_request, llm_debug_seed)
    tracker.set_timing("structured_extract_ms", structured_started_at)
    tracker.complete("structured_extract", "主摘要与结构化字段已生成。")

    parsed_result = NormalizedPaperParseResult(
        title=title,
        abstract_zh=structure_blocks.abstract_zh,
        abstract_en=structure_blocks.abstract_en,
        summary_fallback=structure_blocks.display_summary,
        authors=structure_blocks.authors,
        authors_source=structure_blocks.authors_source,
        authors_confidence=structure_blocks.authors_confidence,
        authors_raw_candidates=structure_blocks.authors_raw_candidates,
        keywords=keyword_result.keywords,
        research_question=structured_result.research_question,
        research_method=structured_result.research_method,
        core_conclusion=structured_result.core_conclusion,
        source_language=structure_blocks.source_language or keyword_result.source_language or "unknown",
        keyword_source=keyword_result.source_kind,
        structured_backend=structured_result.backend,
        structured_note=structured_result.note,
        explicit_abstract_labels_found=bool(structured_request.debug_info.get("explicit_abstract_labels_found", False)),
        llm_supplemented_fields=list(structured_result.debug_info.get("supplemented_fields", [])),
    )

    tracker.start("ai_interpretation", "正在生成通俗摘要和方法说明。")
    llm_started_at = perf_counter()
    course_support_request = CourseSupportRequest(
        title=title,
        authors=structure_blocks.authors,
        summary=parsed_result.summary_text(),
        english_abstract=parsed_result.english_abstract_text(),
        keywords=parsed_result.filtered_keywords(),
        research_question=parsed_result.structured_field_text("research_question"),
        research_method=parsed_result.structured_field_text("research_method"),
        core_conclusion=parsed_result.structured_field_text("core_conclusion"),
        raw_text_preview=reflow_text_for_display(build_preview(text_bundle.body_text, TEXT_PREVIEW_LENGTH)),
    )
    course_support_warnings: list[str] = []
    try:
        course_support_result = generate_course_support_material(course_support_request)
    except Exception as exc:
        course_support_result = generate_course_support_material(CourseSupportRequest())
        course_support_result.note = f"课程写作辅助生成失败，已回退到保守结果：{exc}"
        course_support_result.backend = "local_rule_exception_fallback"
        course_support_warnings.append("AI 解读部分生成失败，已回退到保守结果。")

    tracker.set_timing("llm_generate_ms", llm_started_at)
    parsed_result.plain_language_summary = course_support_result.plain_language_summary
    parsed_result.method_explanation = course_support_result.method_explanation
    if course_support_warnings:
        tracker.partial("ai_interpretation", "AI 解读已回退到保守结果。")
    else:
        tracker.complete("ai_interpretation", "通俗摘要与方法说明已生成。")

    tracker.start("writing_outline", "正在整理课程写作提纲。")
    parsed_result.innovation_points = course_support_result.innovation_points
    parsed_result.limitation_points = course_support_result.limitation_points
    parsed_result.course_presentation_outline = course_support_result.course_presentation_outline
    parsed_result.course_paper_outline = course_support_result.course_paper_outline
    parsed_result.literature_review_outline = course_support_result.literature_review_outline
    if course_support_warnings:
        tracker.partial("writing_outline", "写作提纲已使用保守结果生成。")
    else:
        tracker.complete("writing_outline", "课程汇报、课程论文和文献综述提纲已生成。")

    parse_warnings = _build_parse_warnings(
        structure_blocks=structure_blocks,
        keyword_result=keyword_result,
        structured_result=structured_result,
        explicit_abstract_labels_found=bool(structured_request.debug_info.get("explicit_abstract_labels_found", False)),
        course_support_result=course_support_result,
        extra_warnings=course_support_warnings,
    )
    parsed_result.parse_warnings = parse_warnings

    structured_debug = {
        "pipeline_debug": {
            "stages": [
                "stage_1_preclean",
                "stage_2_structure_detection",
                "stage_3_rule_first_extraction",
                "stage_4_standardized_result",
            ],
            "parse_status": "partial_success" if parse_warnings else "success",
        },
        "preclean_debug": {
            "repeated_artifacts": text_bundle.repeated_artifacts,
            "front_preview": text_bundle.front_text[:300],
            "priority_preview": text_bundle.priority_text[:300],
            "body_preview": text_bundle.body_text[:300],
        },
        "pdf_debug": _build_pdf_debug(pdf_result, title_result),
        "structure_debug": structure_blocks.debug_info,
        "summary_debug": {
            "summary_source_language": structure_blocks.source_language,
            "has_chinese_abstract": bool(structure_blocks.abstract_zh.strip()),
            "has_english_abstract": bool(structure_blocks.abstract_en.strip()),
            "keyword_source": keyword_result.source_kind,
            "keyword_source_language": keyword_result.source_language,
            "keywords": keyword_result.keywords,
            "keyword_block_preview": keyword_result.raw_block[:240],
            "primary_summary_language": parsed_result.primary_summary_language(),
            "authors_source": structure_blocks.authors_source,
            "authors_confidence": structure_blocks.authors_confidence,
            "authors_raw_candidates": structure_blocks.authors_raw_candidates,
            "authors_cleaned": structure_blocks.authors_cleaned,
        },
        "candidate_debug": _build_candidate_debug(structured_request),
        "llm_debug": llm_debug,
        "course_support_debug": {
            "backend": course_support_result.backend,
            "note": course_support_result.note,
            **course_support_result.debug_info,
        },
    }

    parse_status = "partial_success" if parse_warnings else "success"
    result = PaperResult(
        file_name=file_name,
        parsed_result=parsed_result,
        structured_notice=structured_notice,
        structured_debug=structured_debug,
        raw_text=text_bundle.body_text,
        text_preview=reflow_text_for_display(build_preview(text_bundle.body_text, TEXT_PREVIEW_LENGTH)),
        parse_status=parse_status,
        parse_steps=tracker.snapshot(),
        parse_errors=[],
        parse_timings=dict(tracker.timings),
    )
    structured_debug["standard_result"] = result.as_standard_dict()
    structured_debug["final_result"] = {
        "research_question": structured_result.research_question,
        "research_method": structured_result.research_method,
        "core_conclusion": structured_result.core_conclusion,
        "backend": structured_result.backend,
        "confidence": structured_result.confidence,
        "note": structured_result.note,
        "low_confidence_reason": _build_final_low_confidence_reason(structured_result),
        "parse_warnings": result.warning_items(),
        "course_support_backend": course_support_result.backend,
        "parse_status": parse_status,
        "parse_timings": dict(tracker.timings),
    }

    if not result.has_meaningful_content():
        tracker.fail("structured_extract", "未提取到可展示的关键信息。")
        parse_feedback = tracker.feedback(
            status="failed",
            warnings=result.warning_items(),
            errors=["未提取到可展示的关键信息。"],
        )
        parse_feedback["error_category"] = "structured_extract_failed"
        parse_feedback["error_stage"] = "structured_extract"
        raise ParsePipelineError(
            category="structured_extract_failed",
            stage="structured_extract",
            user_message="解析结果为空，未提取到可展示的关键信息。请更换为可复制文本的 PDF 后重试。",
            parse_feedback=parse_feedback,
        )
    return result


def build_parse_text_bundle(pdf_result: PdfExtractionResult) -> ParseTextBundle:
    raw_page_texts = [normalize_whitespace(page.text) for page in pdf_result.pages if getattr(page, "text", "").strip()]
    repeated_artifacts = _detect_repeated_page_artifacts(raw_page_texts)
    cleaned_page_texts = [_preclean_page_text(page_text, repeated_artifacts) for page_text in raw_page_texts]
    cleaned_full_text = clean_extracted_text("\n\n".join(text for text in cleaned_page_texts if text.strip()))

    body_start = min(max(pdf_result.body_start_page_index, 0), max(len(cleaned_page_texts) - 1, 0))
    cleaned_body_text = clean_extracted_text(
        "\n\n".join(text for text in cleaned_page_texts[body_start:] if text.strip())
    ) or cleaned_full_text
    priority_text = clean_extracted_text("\n\n".join(cleaned_page_texts[body_start: body_start + 3]))
    front_text = clean_extracted_text("\n\n".join(cleaned_page_texts[:2])) or priority_text or cleaned_full_text

    return ParseTextBundle(
        full_text=cleaned_full_text,
        body_text=cleaned_body_text,
        priority_text=priority_text,
        front_text=front_text,
        repeated_artifacts=repeated_artifacts,
        page_texts=cleaned_page_texts,
    )


def extract_structure_blocks(
    priority_text: str,
    *,
    fallback_text: str = "",
    title_hint: str = "",
    front_text: str = "",
) -> StructureBlocks:
    source_text = fallback_text or priority_text
    abstract_result = extract_abstracts(source_text, priority_text=priority_text)
    author_source_text = front_text or priority_text or source_text
    authors_result = extract_authors_with_source(
        author_source_text,
        title=title_hint,
        priority_text=front_text or priority_text,
    )
    title_en = _extract_english_title(author_source_text)
    body_start_label = _extract_body_start_label(priority_text or source_text)
    keywords_zh_text = _extract_keyword_block(priority_text, language="zh") or _extract_keyword_block(source_text, language="zh")
    keywords_en_text = _extract_keyword_block(priority_text, language="en") or _extract_keyword_block(source_text, language="en")

    parse_warnings: list[str] = []
    if not abstract_result.chinese_abstract and abstract_result.english_abstract:
        parse_warnings.append("未识别中文摘要，已将英文 Abstract 作为备用字段。")
    if not keywords_zh_text and keywords_en_text:
        parse_warnings.append("未识别中文关键词，已将英文 Keywords 作为备用字段。")

    return StructureBlocks(
        abstract_zh=abstract_result.chinese_abstract,
        abstract_en=abstract_result.english_abstract,
        display_summary=abstract_result.display_abstract,
        keywords_zh_text=keywords_zh_text,
        keywords_en_text=keywords_en_text,
        authors=authors_result.authors,
        authors_source=authors_result.source_kind,
        authors_confidence=authors_result.confidence,
        authors_raw_candidates=authors_result.raw_candidates,
        authors_cleaned=authors_result.cleaned_authors,
        title_en=title_en,
        body_start_label=body_start_label,
        source_language=abstract_result.source_language,
        parse_warnings=parse_warnings,
        debug_info={
            "authors": authors_result.authors,
            "authors_source": authors_result.source_kind,
            "authors_confidence": authors_result.confidence,
            "authors_raw_candidates": authors_result.raw_candidates,
            "authors_cleaned": authors_result.cleaned_authors,
            "authors_debug": authors_result.debug_info,
            "title_en": title_en,
            "body_start_label": body_start_label,
            "abstract_source_language": abstract_result.source_language,
            "abstract_zh_preview": abstract_result.chinese_abstract[:300],
            "abstract_en_preview": abstract_result.english_abstract[:300],
            "display_summary_preview": abstract_result.display_abstract[:300],
            "keywords_zh_text": keywords_zh_text,
            "keywords_en_text": keywords_en_text,
        },
    )


def _resolve_structured_result(structured_request, llm_debug_seed: dict[str, object]):
    explicit_abstract_labels_found = bool(structured_request.debug_info.get("explicit_abstract_labels_found", False))
    populated_rule_fields = sum(
        1
        for candidate in structured_request.candidates.values()
        if candidate.text.strip()
    )

    if explicit_abstract_labels_found or populated_rule_fields == 3:
        structured_result = rewrite_structured_fields(structured_request)
        llm_debug = {
            "env": get_relay_env_status(),
            **llm_debug_seed,
            **structured_result.debug_info,
            "backend": "local_rule_priority",
        }
        return structured_result, "当前结构化字段优先来自中文摘要原文标签或规则抽取。", llm_debug

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
        llm_debug = {
            "env": get_relay_env_status(),
            **llm_debug_seed,
            **structured_result.debug_info,
        }
        return structured_result, structured_notice, llm_debug
    except RelayConfigError as exc:
        structured_result = rewrite_structured_fields(structured_request)
        llm_debug = {
            "env": get_relay_env_status(),
            **llm_debug_seed,
            "backend": "relay_config_error",
            "valid_return": False,
            "error": str(exc),
            "low_confidence_reason": str(exc),
        }
        return structured_result, f"当前优先显示原文摘要/规则抽取结果：{exc}", llm_debug
    except RelayRequestError as exc:
        structured_result = rewrite_structured_fields(structured_request)
        llm_debug = {
            "env": get_relay_env_status(),
            **llm_debug_seed,
            **structured_result.debug_info,
            "backend": "relay_request_error_local_fallback",
            "error": str(exc),
            "low_confidence_reason": structured_result.debug_info.get("low_confidence_reason", str(exc)),
        }
        return structured_result, f"当前优先显示原文摘要/规则抽取结果：{exc}", llm_debug


def _build_parse_warnings(
    *,
    structure_blocks: StructureBlocks,
    keyword_result,
    structured_result,
    explicit_abstract_labels_found: bool,
    course_support_result,
    extra_warnings: list[str] | None = None,
) -> list[str]:
    warnings = list(structure_blocks.parse_warnings)
    if not structure_blocks.authors:
        warnings.append("本次解析未稳定识别作者。")
    elif structure_blocks.authors_confidence == "low":
        warnings.append("作者识别置信度较低，请结合原文核对。")
    if not structure_blocks.abstract_zh and not structure_blocks.abstract_en:
        if structure_blocks.display_summary:
            warnings.append("未识别到显式摘要标签，当前主摘要来自正文 fallback 归纳。")
        else:
            warnings.append("未识别到可用摘要。")

    if not keyword_result.keywords:
        warnings.append("未识别到关键词。")
    elif keyword_result.source_language == "en":
        warnings.append("未识别中文关键词，当前关键词已回退到英文 Keywords。")
    elif keyword_result.source_language == "fallback":
        warnings.append("未识别显式关键词区块，当前关键词来自频次回退。")
    warnings.extend(keyword_result.warnings)
    if keyword_result.debug_info.get("low_confidence"):
        warnings.append("关键词提取置信度较低。")

    supplemented_fields = structured_result.debug_info.get("supplemented_fields", [])
    if supplemented_fields and not explicit_abstract_labels_found:
        warnings.append("部分结构化字段由模型概括补充。")
    course_support_skip_reason = str(course_support_result.debug_info.get("skip_reason", "")).strip()
    if (
        "exception" in course_support_result.backend
        or course_support_skip_reason == "source_text_too_short"
        or extra_warnings
    ):
        warnings.append("部分 AI 输出基于有限文本生成，请酌情复核。")
    warnings.extend(extra_warnings or [])
    return _dedupe_items(warnings)


def _detect_repeated_page_artifacts(page_texts: list[str]) -> list[str]:
    if len(page_texts) <= 1:
        return []

    line_counter: Counter[str] = Counter()
    for page_text in page_texts:
        lines = [normalize_line(line) for line in page_text.splitlines() if normalize_line(line)]
        if not lines:
            continue

        edge_lines = lines[:HEADER_FOOTER_SCAN_LINES] + lines[-HEADER_FOOTER_SCAN_LINES:]
        page_candidates = {line for line in edge_lines if _is_repeated_artifact_candidate(line)}
        line_counter.update(page_candidates)

    repeated_lines = [line for line, count in line_counter.items() if count >= 2]
    return sorted(repeated_lines)


def _is_repeated_artifact_candidate(line: str) -> bool:
    if len(line) < 4 or len(line) > 80:
        return False
    if SECTION_HEADING_PATTERN.match(line):
        return False
    if BODY_START_PATTERN.match(line):
        return False
    if re.search(r"(摘要|关键词|abstract|keywords?)\s*[：:]", line, re.IGNORECASE):
        return False
    if re.search(r"(issn|cn|doi|网络首发|首发时间|首发地址|期刊|journal|作者单位|通信作者)", line, re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:第?\s*\d+\s*页|\d+\s*/\s*\d+)", line):
        return True
    return False


def _preclean_page_text(page_text: str, repeated_artifacts: list[str]) -> str:
    repeated_set = {normalize_line(item) for item in repeated_artifacts}
    kept_lines: list[str] = []

    for raw_line in page_text.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            kept_lines.append("")
            continue
        normalized = normalize_line(line)
        if normalized in repeated_set:
            continue
        kept_lines.append(line)
    return clean_extracted_text("\n".join(kept_lines))


def _extract_keyword_block(text: str, *, language: str) -> str:
    if not text.strip():
        return ""

    label_pattern = (
        r"(?:〔\s*(?:关键词|关键字)\s*〕|\[\s*(?:关键词|关键字)\s*\]|(?:关键词|关键字))"
        if language == "zh"
        else r"(?:keywords?|key words|index terms?)"
    )
    stop_pattern = (
        r"(?:〔?\s*(?:中图法分类号|引用本文格式|英文标题|英文题名)\s*〕?|英文摘要|ABSTRACT|Abstract|abstract|Keywords?|key words|index terms?|引言|绪论|问题提出|参考文献|一、|0[.、]?\s*引言|1[.、]?\s*引言|第一章|$)"
        if language == "zh"
        else r"(?:摘要|中图法分类号|引用本文格式|引言|introduction|一、|1[.、]|第一章|references?|$)"
    )
    pattern = rf"{label_pattern}\s*[：:]?\s*(.+?)(?=(?:{stop_pattern}))"
    match = re.search(pattern, text[:6000], re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    block = normalize_whitespace(match.group(1))
    block = block.lstrip("〕】])} ").strip(" ：:；;，,、")
    return block


def _stabilize_keyword_result(
    keyword_result: KeywordExtractionResult,
    *,
    title: str,
    priority_text: str,
    body_text: str,
) -> KeywordExtractionResult:
    fallback_result = fallback_keywords_with_llm(
        title=title,
        priority_text=priority_text,
        raw_text=body_text,
        existing_keywords=keyword_result.keywords,
        current_source_kind=keyword_result.source_kind,
        current_confidence=keyword_result.confidence,
    )
    if fallback_result.source_kind != "llm_keyword_fallback" or not fallback_result.keywords:
        keyword_result.debug_info["llm_fallback"] = fallback_result.debug_info
        return keyword_result

    merged_warnings = _dedupe_items(
        [
            *keyword_result.warnings,
            "规则关键词结果不稳定，已使用模型对关键词做保守补充。",
        ]
    )
    return KeywordExtractionResult(
        keywords=fallback_result.keywords,
        source_language="zh",
        source_kind=fallback_result.source_kind,
        raw_block="；".join(fallback_result.keywords),
        confidence=fallback_result.confidence or keyword_result.confidence,
        warnings=merged_warnings,
        debug_info={
            **keyword_result.debug_info,
            "llm_fallback": fallback_result.debug_info,
            "llm_fallback_note": fallback_result.note,
        },
    )


def _extract_author_lines(text: str, title: str) -> list[str]:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    if not lines:
        return []

    normalized_title = normalize_line(title)
    title_index = 0
    for index, line in enumerate(lines[:12]):
        if normalized_title and (line == normalized_title or normalized_title in line or line in normalized_title):
            title_index = index
            continue

    authors: list[str] = []
    for line in lines[title_index + 1:title_index + 6]:
        if re.search(r"^(摘要|摘\s*要|关键词|关键字|abstract|keywords?|引言|绪论)", line, re.IGNORECASE):
            break
        if re.search(r"(大学|学院|研究院|研究所|department|university|college|institute)", line, re.IGNORECASE):
            continue
        if _looks_like_author_line(line):
            authors.extend(_split_author_line(line))
    return _dedupe_items(authors)[:6]


def _extract_english_title(text: str) -> str:
    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    for index, line in enumerate(lines):
        if re.match(r"^abstract\s*[：:]?$", line, re.IGNORECASE) and index > 0:
            candidate = lines[index - 1]
            if _looks_like_english_title(candidate):
                return candidate
    return ""


def _extract_body_start_label(text: str) -> str:
    for line in text.splitlines():
        normalized = normalize_line(line)
        if not normalized:
            continue
        if BODY_START_PATTERN.match(normalized):
            return normalized
    return ""


def _looks_like_author_line(text: str) -> bool:
    if len(text) < 2 or len(text) > 40:
        return False
    if re.search(r"[。！？；;:：]", text):
        return False
    if re.search(r"(基金项目|作者简介|通信作者|网络首发|doi|issn)", text, re.IGNORECASE):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z·•\s,，、]{2,40}", text))


def _split_author_line(text: str) -> list[str]:
    normalized = normalize_line(text)
    if not normalized:
        return []
    spaced_compact = re.sub(r"\s+", "", normalized)
    if " " in normalized and re.fullmatch(r"[\u4e00-\u9fff]+", spaced_compact) and 4 <= len(spaced_compact) <= 7:
        heuristic_names = _split_compact_chinese_author_names(spaced_compact)
        if heuristic_names:
            return heuristic_names
    if re.search(r"[,，、;/；]", normalized):
        raw_parts = re.split(r"[,，、;/；]+", normalized)
    else:
        raw_parts = [normalized]
        if normalized.count(" ") >= 1:
            space_parts = [item for item in re.split(r"\s+", normalized) if item]
            if len(space_parts) >= 2 and all(_looks_like_author_name(item) for item in space_parts):
                raw_parts = space_parts

    authors: list[str] = []
    for item in raw_parts:
        cleaned = normalize_line(item).strip("1234567890*†‡ ")
        if _looks_like_author_name(cleaned):
            authors.append(cleaned)
    return authors or [normalized]


def _looks_like_author_name(text: str) -> bool:
    if len(text) < 2 or len(text) > 20:
        return False
    if re.search(r"(大学|学院|研究院|研究所|department|university|college|institute)", text, re.IGNORECASE):
        return False
    if re.search(r"[。！？；;:：]", text):
        return False
    return bool(re.fullmatch(r"[\u4e00-\u9fffA-Za-z·•]{2,20}", text))


def _split_compact_chinese_author_names(text: str) -> list[str]:
    if len(text) == 4:
        return [text[:2], text[2:]]
    if len(text) == 5:
        return [text[:2], text[2:]]
    if len(text) == 6:
        return [text[:3], text[3:]]
    if len(text) == 7:
        return [text[:3], text[3:]]
    return []


def _looks_like_english_title(text: str) -> bool:
    if len(text) < 12 or len(text) > 140:
        return False
    if re.search(r"[\u4e00-\u9fff]", text):
        return False
    return len(re.findall(r"[A-Za-z]", text)) >= 10


def _dedupe_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = normalize_line(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


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
        "priority_pages": structured_request.debug_info.get("priority_pages", 0),
        "priority_preview": structured_request.debug_info.get("priority_preview", ""),
        "abstract_priority_found": structured_request.debug_info.get("abstract_priority_found", False),
        "abstract_fallback_available": structured_request.debug_info.get("abstract_fallback_available", False),
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
        "research_question_low_quality_reason": field_debug.get("research_question", {}).get("low_quality_reason", ""),
        "research_method_low_quality_reason": field_debug.get("research_method", {}).get("low_quality_reason", ""),
        "core_conclusion_low_quality_reason": field_debug.get("core_conclusion", {}).get("low_quality_reason", ""),
    }


def _build_llm_debug_seed(structured_request) -> dict[str, object]:
    rule_candidate_char_count = int(structured_request.debug_info.get("rule_candidate_char_count", 0))
    abstract_text = str(structured_request.debug_info.get("abstract_fallback_text", "")).strip()
    abstract_char_count = len(abstract_text)
    abstract_fallback_enabled = (
        abstract_char_count >= STRUCTURED_LLM_MIN_ABSTRACT_CHARS
        and rule_candidate_char_count < STRUCTURED_LLM_MIN_CANDIDATE_CHARS
    )

    if rule_candidate_char_count == 0 and abstract_fallback_enabled:
        llm_input_source = "摘要"
    elif rule_candidate_char_count > 0 and abstract_fallback_enabled:
        llm_input_source = "混合"
    elif rule_candidate_char_count > 0:
        llm_input_source = "正文候选"
    else:
        llm_input_source = "无可用输入"

    return {
        "rule_candidate_char_count": rule_candidate_char_count,
        "abstract_char_count": abstract_char_count,
        "abstract_fallback_enabled": abstract_fallback_enabled,
        "llm_input_source": llm_input_source,
    }


def _build_pdf_debug(pdf_result, title_result) -> dict[str, object]:
    body_window_pages = set(pdf_result.body_window_page_indices)
    pages = []
    for page in pdf_result.pages:
        pages.append(
            {
                "page_index": page.page_index,
                "page_preview_300": page.preview,
                "page_score": page.page_score,
                "body_keyword_hits": page.body_keywords_hit,
                "cover_keyword_hits": page.cover_keywords_hit,
                "is_selected_body_window": page.page_index in body_window_pages,
                "is_title_page": page.page_index == title_result.source_page_index,
            }
        )

    return {
        "version": "PDF 正文定位修复版",
        "pages": pages,
        "selected_body_start_page": pdf_result.body_start_page_index,
        "selected_body_pages": sorted(body_window_pages),
        "detected_title": title_result.title,
        "title_source_page_index": title_result.source_page_index,
        "title_source": title_result.source_kind,
        "title_candidates": title_result.debug_info.get("candidates", []),
    }


def _build_final_low_confidence_reason(structured_result) -> str:
    if structured_result.debug_info.get("low_confidence_reason"):
        return str(structured_result.debug_info.get("low_confidence_reason"))

    values = [
        structured_result.research_question.strip(),
        structured_result.research_method.strip(),
        structured_result.core_conclusion.strip(),
    ]
    if any(value == "暂未识别到高置信度结果。" for value in values):
        return "LLM 或预检查阶段直接返回了“暂未识别到高置信度结果”。"
    if structured_result.confidence == "低":
        return "最终结构化结果置信度为低。"
    return ""


def _classify_pdf_or_text_error(message: str) -> str:
    normalized = normalize_line(message)
    if re.search(r"(无法读取|格式正确|没有可解析页面|上传文件为空)", normalized):
        return "pdf_read_failed"
    if re.search(r"(没有提取到可用文本|文本过少|扫描版文档)", normalized):
        return "text_extract_failed"
    return "text_extract_failed"


def _field_label(field_name: str) -> str:
    mapping = {
        "research_question": "研究问题",
        "research_method": "研究方法",
        "core_conclusion": "核心结论",
    }
    return mapping.get(field_name, field_name)
