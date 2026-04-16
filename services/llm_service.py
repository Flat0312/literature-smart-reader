"""Relay-compatible LLM rewrite layer for structured paper summaries."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import logging

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

from config.settings import (
    COURSE_SUPPORT_LLM_MAX_OUTPUT_TOKENS,
    COURSE_SUPPORT_LLM_MIN_SOURCE_CHARS,
    KEYWORD_LLM_MAX_OUTPUT_TOKENS,
    MAX_KEYWORDS,
    MIN_KEYWORDS,
    STRUCTURED_LLM_MAX_OUTPUT_TOKENS,
    STRUCTURED_LLM_MIN_ABSTRACT_CHARS,
    STRUCTURED_LLM_MIN_CANDIDATE_CHARS,
    STRUCTURED_LLM_TIMEOUT_SECONDS,
    STRUCTURED_LOW_CONFIDENCE_TEXT,
)
from services.structured_rewrite_service import (
    StructuredRewriteRequest,
    StructuredRewriteResult,
    build_rewrite_prompt,
)
from utils.text_utils import compact_list, normalize_line, normalize_whitespace


@dataclass(slots=True)
class RelaySettings:
    api_key: str
    base_url: str
    model: str


class RelayConfigError(RuntimeError):
    """Raised when relay environment variables are missing."""


class RelayRequestError(RuntimeError):
    """Raised when all relay rewrite paths fail."""


class StructuredRewritePayload(BaseModel):
    """Unified schema returned by relay-backed rewrite calls."""

    research_question: str = Field(description="统一用简洁自然中文表述的研究问题。")
    research_method: str = Field(description="统一用简洁自然中文表述的研究方法。")
    core_conclusion: str = Field(description="统一用简洁自然中文表述的核心结论。")
    note: str = Field(description="简短说明结果依据或不确定性，统一使用中文。")


class KeywordFallbackPayload(BaseModel):
    """Structured keyword fallback payload returned by the relay backend."""

    keywords: list[str] = Field(description="3 到 8 个贴近原文的中文关键词。")
    note: str = Field(description="简短说明关键词依据或不确定性。")


class AuthorFallbackPayload(BaseModel):
    """Structured author fallback payload returned by the relay backend."""

    authors: list[str] = Field(description="仅返回论文作者名单；不确定时返回空数组。")
    note: str = Field(description="简短说明是否基于首页前部文本识别到作者。")


class CourseSupportPayload(BaseModel):
    """Structured course-writing support payload returned by the relay backend."""

    plain_language_summary: str = Field(description="面向大学生的通俗摘要，统一使用中文。")
    method_explanation: str = Field(description="对研究方法的解释性说明，统一使用中文。")
    innovation_points: list[str] = Field(description="2 到 4 条创新点分析。")
    limitation_points: list[str] = Field(description="2 到 4 条不足分析。")
    course_presentation_outline: list[str] = Field(description="适合课程汇报/PPT 的提纲。")
    course_paper_outline: list[str] = Field(description="适合课程论文前期整理的提纲。")
    literature_review_outline: list[str] = Field(description="适合文献综述基础框架整理的提纲。")
    note: str = Field(description="对输出依据或不确定性的简短说明。")


@dataclass(slots=True)
class KeywordFallbackResult:
    keywords: list[str] = field(default_factory=list)
    source_kind: str = ""
    confidence: str = ""
    note: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AuthorFallbackResult:
    authors: list[str] = field(default_factory=list)
    source_kind: str = ""
    confidence: str = ""
    note: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class CourseSupportRequest:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    summary: str = ""
    english_abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    research_question: str = ""
    research_method: str = ""
    core_conclusion: str = ""
    raw_text_preview: str = ""


@dataclass(slots=True)
class CourseSupportResult:
    plain_language_summary: str = ""
    method_explanation: str = ""
    innovation_points: list[str] = field(default_factory=list)
    limitation_points: list[str] = field(default_factory=list)
    course_presentation_outline: list[str] = field(default_factory=list)
    course_paper_outline: list[str] = field(default_factory=list)
    literature_review_outline: list[str] = field(default_factory=list)
    note: str = ""
    backend: str = "local_rule"
    debug_info: dict[str, object] = field(default_factory=dict)


def get_relay_env_status() -> dict[str, object]:
    return {
        "has_relay_api_key": bool(os.getenv("RELAY_API_KEY", "").strip()),
        "has_relay_base_url": bool(os.getenv("RELAY_BASE_URL", "").strip()),
        "has_relay_model": bool(os.getenv("RELAY_MODEL", "").strip()),
        "relay_model": os.getenv("RELAY_MODEL", "").strip(),
        "relay_base_url": os.getenv("RELAY_BASE_URL", "").strip(),
    }


def fallback_keywords_with_llm(
    *,
    title: str,
    priority_text: str,
    raw_text: str,
    existing_keywords: list[str],
    current_source_kind: str = "",
    current_confidence: str = "",
) -> KeywordFallbackResult:
    current_keywords = compact_list(existing_keywords, MAX_KEYWORDS)
    debug_info = {
        "env": get_relay_env_status(),
        "current_source_kind": current_source_kind,
        "current_confidence": current_confidence,
        "current_keywords": current_keywords,
        "used_llm": False,
        "attempted_path": [],
        "raw_response_text": "",
    }
    if not _needs_keyword_llm_fallback(current_keywords, current_source_kind, current_confidence):
        debug_info["skip_reason"] = "current_keywords_stable"
        return KeywordFallbackResult(
            keywords=current_keywords,
            source_kind=current_source_kind or "rule",
            confidence=current_confidence or _keyword_confidence_from_count(len(current_keywords)),
            note="当前关键词保留规则提取结果。",
            debug_info=debug_info,
        )

    source_text = _build_keyword_source_text(priority_text, raw_text)
    debug_info["source_text_length"] = len(source_text)
    if len(source_text) < 40:
        debug_info["skip_reason"] = "source_text_too_short"
        return KeywordFallbackResult(
            keywords=current_keywords,
            source_kind=current_source_kind or "rule",
            confidence=current_confidence or _keyword_confidence_from_count(len(current_keywords)),
            note="正文可用内容不足，未触发关键词模型兜底。",
            debug_info=debug_info,
        )

    try:
        settings = _load_relay_settings()
    except RelayConfigError as exc:
        debug_info["skip_reason"] = "relay_config_missing"
        debug_info["error"] = str(exc)
        return KeywordFallbackResult(
            keywords=current_keywords,
            source_kind=current_source_kind or "rule",
            confidence=current_confidence or _keyword_confidence_from_count(len(current_keywords)),
            note=str(exc),
            debug_info=debug_info,
        )

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=STRUCTURED_LLM_TIMEOUT_SECONDS,
    )
    responses_error: Exception | None = None
    try:
        debug_info["attempted_path"].append("responses")
        payload, raw_output = _keyword_fallback_with_responses(client, settings, title=title, source_text=source_text)
        debug_info["raw_response_text"] = raw_output
        keywords = _sanitize_keyword_payload(payload.keywords)
        if keywords:
            debug_info["used_llm"] = True
            return KeywordFallbackResult(
                keywords=keywords,
                source_kind="llm_keyword_fallback",
                confidence=_keyword_confidence_from_count(len(keywords)),
                note=(payload.note or "").strip() or "关键词由模型结合摘要和正文片段补充。",
                debug_info=debug_info,
            )
    except Exception as exc:
        responses_error = exc
        debug_info["responses_error"] = str(exc)

    try:
        debug_info["attempted_path"].append("chat.completions")
        payload, raw_output = _keyword_fallback_with_chat(client, settings, title=title, source_text=source_text)
        debug_info["raw_response_text"] = raw_output
        keywords = _sanitize_keyword_payload(payload.keywords)
        if keywords:
            debug_info["used_llm"] = True
            return KeywordFallbackResult(
                keywords=keywords,
                source_kind="llm_keyword_fallback",
                confidence=_keyword_confidence_from_count(len(keywords)),
                note=(payload.note or "").strip() or "关键词由模型结合摘要和正文片段补充。",
                debug_info=debug_info,
            )
    except Exception as exc:
        debug_info["chat_error"] = str(exc)
        if responses_error is not None:
            debug_info["error"] = "LLM 关键词兜底失败，继续保留规则关键词结果。"

    return KeywordFallbackResult(
        keywords=current_keywords,
        source_kind=current_source_kind or "rule",
        confidence=current_confidence or _keyword_confidence_from_count(len(current_keywords)),
        note="LLM 关键词兜底失败，继续保留规则关键词结果。",
        debug_info=debug_info,
    )


def fallback_authors_with_llm(
    *,
    title: str,
    front_text: str,
    raw_candidates: list[str] | None = None,
) -> AuthorFallbackResult:
    debug_info = {
        "env": get_relay_env_status(),
        "front_text_length": len(front_text or ""),
        "raw_candidates": list(raw_candidates or []),
        "used_llm": False,
        "attempted_path": [],
        "raw_response_text": "",
    }
    if len(normalize_whitespace(front_text or "")) < 20:
        debug_info["skip_reason"] = "front_text_too_short"
        return AuthorFallbackResult(
            authors=[],
            source_kind="none",
            confidence="none",
            note="首页前部文本不足，未触发作者模型兜底。",
            debug_info=debug_info,
        )

    try:
        settings = _load_relay_settings()
    except RelayConfigError as exc:
        debug_info["skip_reason"] = "relay_config_missing"
        debug_info["error"] = str(exc)
        return AuthorFallbackResult(
            authors=[],
            source_kind="none",
            confidence="none",
            note=str(exc),
            debug_info=debug_info,
        )

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=STRUCTURED_LLM_TIMEOUT_SECONDS,
    )
    responses_error: Exception | None = None
    try:
        debug_info["attempted_path"].append("responses")
        payload, raw_output = _author_fallback_with_responses(client, settings, title=title, front_text=front_text)
        debug_info["raw_response_text"] = raw_output
        authors = _sanitize_author_payload(payload.authors)
        if authors:
            debug_info["used_llm"] = True
            return AuthorFallbackResult(
                authors=authors,
                source_kind="llm_fallback",
                confidence="low",
                note=(payload.note or "").strip() or "作者由模型结合首页前部文本保守识别。",
                debug_info=debug_info,
            )
    except Exception as exc:
        responses_error = exc
        debug_info["responses_error"] = str(exc)

    try:
        debug_info["attempted_path"].append("chat.completions")
        payload, raw_output = _author_fallback_with_chat(client, settings, title=title, front_text=front_text)
        debug_info["raw_response_text"] = raw_output
        authors = _sanitize_author_payload(payload.authors)
        if authors:
            debug_info["used_llm"] = True
            return AuthorFallbackResult(
                authors=authors,
                source_kind="llm_fallback",
                confidence="low",
                note=(payload.note or "").strip() or "作者由模型结合首页前部文本保守识别。",
                debug_info=debug_info,
            )
    except Exception as exc:
        debug_info["chat_error"] = str(exc)
        if responses_error is not None:
            debug_info["error"] = "作者 LLM 兜底失败。"

    return AuthorFallbackResult(
        authors=[],
        source_kind="none",
        confidence="none",
        note="作者 LLM 兜底失败。",
        debug_info=debug_info,
    )


def generate_course_support_material(request: CourseSupportRequest) -> CourseSupportResult:
    local_result = _build_local_course_support_result(request)
    source_text = _build_course_support_source_text(request)
    debug_info = {
        "env": get_relay_env_status(),
        "source_text_length": len(source_text),
        "used_local_fallback": False,
        "attempted_path": [],
        "raw_response_text": "",
    }
    if len(source_text) < COURSE_SUPPORT_LLM_MIN_SOURCE_CHARS:
        local_result.backend = "local_rule"
        local_result.note = "当前可识别内容较少，课程写作辅助结果优先使用本地保守生成。"
        local_result.debug_info = {
            **debug_info,
            "used_local_fallback": True,
            "skip_reason": "source_text_too_short",
        }
        return local_result

    try:
        settings = _load_relay_settings()
    except RelayConfigError as exc:
        local_result.backend = "local_rule"
        local_result.note = "当前未配置模型服务，课程写作辅助结果使用本地保守生成。"
        local_result.debug_info = {
            **debug_info,
            "used_local_fallback": True,
            "skip_reason": "relay_config_missing",
            "error": str(exc),
        }
        return local_result

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=STRUCTURED_LLM_TIMEOUT_SECONDS,
    )
    responses_error: Exception | None = None
    try:
        debug_info["attempted_path"].append("responses")
        payload, raw_output = _course_support_with_responses(client, settings, request)
        debug_info["raw_response_text"] = raw_output
        return _merge_course_support_payload(local_result, payload, backend="relay_course_support", debug_info=debug_info)
    except Exception as exc:
        responses_error = exc
        debug_info["responses_error"] = str(exc)

    try:
        debug_info["attempted_path"].append("chat.completions")
        payload, raw_output = _course_support_with_chat(client, settings, request)
        debug_info["raw_response_text"] = raw_output
        return _merge_course_support_payload(local_result, payload, backend="relay_course_support_chat", debug_info=debug_info)
    except Exception as exc:
        debug_info["chat_error"] = str(exc)
        if responses_error is not None:
            debug_info["error"] = "课程写作辅助的 LLM 生成失败，已回退到本地保守生成。"

    local_result.backend = "local_rule"
    local_result.note = "课程写作辅助的 LLM 生成失败，已回退到本地保守生成。"
    local_result.debug_info = {**debug_info, "used_local_fallback": True}
    return local_result


def rewrite_structured_result(request: StructuredRewriteRequest) -> StructuredRewriteResult:
    source_context = _build_source_context(request)
    debug_info = {
        "env": get_relay_env_status(),
        "candidate_char_count": source_context["effective_input_char_count"],
        "rule_candidate_char_count": source_context["rule_candidate_char_count"],
        "abstract_char_count": source_context["abstract_char_count"],
        "abstract_fallback_enabled": source_context["abstract_fallback_enabled"],
        "llm_input_source": source_context["llm_input_source"],
        "attempted_path": [],
        "raw_response_text": "",
        "valid_return": False,
        "low_confidence_reason": "",
    }
    settings = _load_relay_settings()

    if (
        source_context["rule_candidate_char_count"] < STRUCTURED_LLM_MIN_CANDIDATE_CHARS
        and not source_context["abstract_fallback_enabled"]
    ):
        return StructuredRewriteResult(
            research_question=STRUCTURED_LOW_CONFIDENCE_TEXT,
            research_method=STRUCTURED_LOW_CONFIDENCE_TEXT,
            core_conclusion=STRUCTURED_LOW_CONFIDENCE_TEXT,
            confidence="低",
            note="候选内容不足，暂未识别到高置信度结果。",
            backend="relay_precheck",
            candidates=request.candidates,
            prompt_preview=build_rewrite_prompt(request),
            debug_info={
                **debug_info,
                "backend": "relay_precheck",
                "low_confidence_reason": _build_precheck_low_confidence_reason(source_context),
            },
        )

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=STRUCTURED_LLM_TIMEOUT_SECONDS,
    )

    responses_error: Exception | None = None
    try:
        debug_info["attempted_path"].append("responses")
        return _rewrite_with_responses(client, settings, request, debug_info=debug_info)
    except Exception as exc:
        responses_error = exc
        debug_info["responses_error"] = str(exc)
        logger.warning("[structured-llm] Responses API failed, fallback to chat.completions: %s", exc)

    try:
        debug_info["attempted_path"].append("chat.completions")
        return _rewrite_with_chat_completions(client, settings, request, debug_info=debug_info)
    except Exception as exc:
        debug_info["chat_error"] = str(exc)
        logger.warning("[structured-llm] Chat Completions fallback failed: %s", exc)
        if responses_error is not None:
            raise RelayRequestError("LLM 调用失败，已自动回退到规则结果。") from exc
        raise RelayRequestError("LLM 未返回有效结果，已自动回退到规则结果。") from exc


def _load_relay_settings() -> RelaySettings:
    api_key = os.getenv("RELAY_API_KEY", "").strip()
    base_url = os.getenv("RELAY_BASE_URL", "").strip()
    model = os.getenv("RELAY_MODEL", "").strip()
    missing_items = [
        env_name
        for env_name, value in (
            ("RELAY_API_KEY", api_key),
            ("RELAY_BASE_URL", base_url),
            ("RELAY_MODEL", model),
        )
        if not value
    ]
    if missing_items:
        missing_text = " / ".join(missing_items)
        raise RelayConfigError(f"缺少 {missing_text} 配置，已回退到规则结果。")

    return RelaySettings(api_key=api_key, base_url=base_url, model=model)


def _candidate_char_count(request: StructuredRewriteRequest) -> int:
    return _build_source_context(request)["effective_input_char_count"]


def _build_system_prompt() -> str:
    return (
        "你是学术论文结构化阅读助手。"
        "请仅依据给定候选摘录整理结果，不要补充外部事实，不要猜测。"
        "不允许把论文标题原样当成研究问题、研究方法或核心结论。"
        "最终输出必须是简洁、自然、完整的中文。"
        "每个字段控制在 1 到 2 句以内，优先贴近原文表达，不要自由扩写。"
        "若某字段已有非空候选，请优先保留其原文含义与表述，仅在明显不完整时做轻微整理。"
        "当正文候选不足但中文摘要可用时，可以基于中文摘要做最小必要的模型概括。"
        "如果需要模型补充概括，只能在 note 中标记“模型概括”，不要在字段正文里写“根据摘要推断”。"
        "删除网址、基金编号、项目编号、出版说明、网络首发说明、参考文献标号和无关片段。"
        "若证据不足，直接返回“暂未识别到高置信度结果”。"
        "输出 JSON 字段固定为 research_question、research_method、core_conclusion、note。"
    )


def _build_input(request: StructuredRewriteRequest) -> str:
    source_context = _build_source_context(request)
    lines = [
        "以下是规则阶段筛选出的候选内容，请据此做结构化重写。",
        f"论文标题：{request.title or '（未识别）'}",
        f"实际送入模型的文本来源：{source_context['llm_input_source']}",
    ]
    for field_name, label in (
        ("research_question", "研究问题候选"),
        ("research_method", "研究方法候选"),
        ("core_conclusion", "核心结论候选"),
    ):
        candidate = request.candidates.get(field_name)
        candidate_text = candidate.text if candidate else ""
        source_hint = candidate.source_hint if candidate else ""
        source_suffix = f"（来源：{source_hint}）" if source_hint else ""
        lines.append(f"{label}{source_suffix}：{candidate_text or '（空）'}")
    if source_context["abstract_fallback_enabled"]:
        lines.append(f"中文摘要补充文本：{source_context['abstract_fallback_text']}")
        lines.append("当正文候选为空或过短时，请仅做克制补全；不要在字段正文中写“根据摘要推断”。")
    lines.append("已有非空候选字段请尽量贴近原文；只有空字段再补充。")
    return "\n".join(lines)


def _rewrite_with_responses(
    client: OpenAI,
    settings: RelaySettings,
    request: StructuredRewriteRequest,
    *,
    debug_info: dict[str, object],
) -> StructuredRewriteResult:
    response = client.responses.parse(
        model=settings.model,
        instructions=_build_system_prompt(),
        input=_build_input(request),
        text_format=StructuredRewritePayload,
        temperature=0.2,
        max_output_tokens=STRUCTURED_LLM_MAX_OUTPUT_TOKENS,
        verbosity="low",
        store=False,
    )
    debug_info["raw_response_text"] = getattr(response, "output_text", "")
    parsed = response.output_parsed
    if parsed is None and getattr(response, "output_text", ""):
        parsed = _parse_payload_from_text(response.output_text)
    if parsed is None:
        raise ValueError("Responses API 未返回可解析结果。")
    return _build_result(request, parsed, backend="relay_responses", debug_info=debug_info)


def _rewrite_with_chat_completions(
    client: OpenAI,
    settings: RelaySettings,
    request: StructuredRewriteRequest,
    *,
    debug_info: dict[str, object],
) -> StructuredRewriteResult:
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_input(request)},
    ]
    request_kwargs = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": STRUCTURED_LLM_MAX_OUTPUT_TOKENS,
    }

    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        debug_info["chat_json_mode_error"] = str(exc)
        logger.warning("[structured-llm] chat.completions JSON mode unavailable, retry without response_format: %s", exc)
        completion = client.chat.completions.create(**request_kwargs)

    content = _extract_chat_content(completion)
    debug_info["raw_response_text"] = content
    parsed = _parse_payload_from_text(content)
    if parsed is None:
        raise ValueError("chat.completions 未返回可解析 JSON。")
    return _build_result(request, parsed, backend="relay_chat", debug_info=debug_info)


def _build_result(
    request: StructuredRewriteRequest,
    payload: StructuredRewritePayload,
    *,
    backend: str,
    debug_info: dict[str, object],
) -> StructuredRewriteResult:
    merged_fields, preserved_rule_fields, supplemented_fields = _merge_payload_with_rule_candidates(request, payload)
    note = _build_result_note(payload.note, preserved_rule_fields, supplemented_fields)
    return StructuredRewriteResult(
        research_question=merged_fields["research_question"],
        research_method=merged_fields["research_method"],
        core_conclusion=merged_fields["core_conclusion"],
        confidence="中",
        note=note,
        backend=backend,
        candidates=request.candidates,
        prompt_preview=build_rewrite_prompt(request),
        debug_info={
            **debug_info,
            "backend": backend,
            "valid_return": True,
            "parsed_payload": payload.model_dump(),
            "preserved_rule_fields": preserved_rule_fields,
            "supplemented_fields": supplemented_fields,
            "low_confidence_reason": _build_payload_low_confidence_reason(payload),
        },
    )


def _merge_payload_with_rule_candidates(
    request: StructuredRewriteRequest,
    payload: StructuredRewritePayload,
) -> tuple[dict[str, str], list[str], list[str]]:
    merged_fields: dict[str, str] = {}
    preserved_rule_fields: list[str] = []
    supplemented_fields: list[str] = []

    for field_name in ("research_question", "research_method", "core_conclusion"):
        candidate = request.candidates.get(field_name)
        candidate_text = (candidate.text if candidate else "").strip()
        payload_text = getattr(payload, field_name, "").strip()

        if candidate_text:
            merged_fields[field_name] = candidate_text
            preserved_rule_fields.append(field_name)
            continue

        if payload_text and payload_text != STRUCTURED_LOW_CONFIDENCE_TEXT:
            merged_fields[field_name] = payload_text
            supplemented_fields.append(field_name)
            continue

        merged_fields[field_name] = ""

    return merged_fields, preserved_rule_fields, supplemented_fields


def _build_result_note(
    payload_note: str,
    preserved_rule_fields: list[str],
    supplemented_fields: list[str],
) -> str:
    cleaned_note = (payload_note or "").strip()
    if supplemented_fields:
        supplemented_text = "、".join(_field_label(field_name) for field_name in supplemented_fields)
        if preserved_rule_fields:
            return f"模型概括仅补充了 {supplemented_text}，其余字段保留原文规则抽取结果。"
        return f"模型概括生成了 {supplemented_text}。"
    if preserved_rule_fields:
        return "当前结果优先保留原文规则抽取结果。"
    return cleaned_note


def _field_label(field_name: str) -> str:
    mapping = {
        "research_question": "研究问题",
        "research_method": "研究方法",
        "core_conclusion": "核心结论",
    }
    return mapping.get(field_name, field_name)


def _parse_payload_from_text(text: str) -> StructuredRewritePayload | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return StructuredRewritePayload.model_validate_json(cleaned)
    except ValidationError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return StructuredRewritePayload.model_validate_json(match.group(0))
        except ValidationError:
            return None


def _extract_chat_content(completion) -> str:
    if not completion.choices:
        return ""
    content = completion.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(getattr(item, "text", "")))
        return "".join(text_parts)
    return ""


def _build_payload_low_confidence_reason(payload: StructuredRewritePayload) -> str:
    values = [
        payload.research_question.strip(),
        payload.research_method.strip(),
        payload.core_conclusion.strip(),
    ]
    if all(value == STRUCTURED_LOW_CONFIDENCE_TEXT for value in values):
        return "模型返回的三个字段均为“暂未识别到高置信度结果”。"
    return ""


def _build_source_context(request: StructuredRewriteRequest) -> dict[str, object]:
    rule_candidate_char_count = sum(
        len(candidate.text.strip())
        for candidate in request.candidates.values()
        if candidate.text.strip()
    )
    abstract_text = str(request.debug_info.get("abstract_fallback_text", "")).strip()
    abstract_char_count = len(abstract_text)
    abstract_available = abstract_char_count >= STRUCTURED_LLM_MIN_ABSTRACT_CHARS
    abstract_fallback_enabled = abstract_available and rule_candidate_char_count < STRUCTURED_LLM_MIN_CANDIDATE_CHARS

    if rule_candidate_char_count == 0 and abstract_fallback_enabled:
        llm_input_source = "摘要"
    elif rule_candidate_char_count > 0 and abstract_fallback_enabled:
        llm_input_source = "混合"
    elif rule_candidate_char_count > 0:
        llm_input_source = "正文候选"
    else:
        llm_input_source = "无可用输入"

    effective_input_char_count = rule_candidate_char_count + (
        abstract_char_count if abstract_fallback_enabled else 0
    )
    return {
        "rule_candidate_char_count": rule_candidate_char_count,
        "abstract_char_count": abstract_char_count,
        "abstract_available": abstract_available,
        "abstract_fallback_enabled": abstract_fallback_enabled,
        "abstract_fallback_text": abstract_text,
        "llm_input_source": llm_input_source,
        "effective_input_char_count": effective_input_char_count,
    }


def _build_precheck_low_confidence_reason(source_context: dict[str, object]) -> str:
    rule_candidate_char_count = int(source_context["rule_candidate_char_count"])
    abstract_char_count = int(source_context["abstract_char_count"])

    if rule_candidate_char_count == 0 and abstract_char_count == 0:
        return "正文候选为空，且未提取到可用于中文摘要补充的文本。"
    if rule_candidate_char_count == 0:
        return (
            f"正文候选为空，中文摘要仅 {abstract_char_count} 个字符，"
            f"低于摘要补充阈值 {STRUCTURED_LLM_MIN_ABSTRACT_CHARS}。"
        )
    if abstract_char_count == 0:
        return (
            f"正文候选总长度仅 {rule_candidate_char_count} 个字符，"
            f"低于阈值 {STRUCTURED_LLM_MIN_CANDIDATE_CHARS}，且无可用中文摘要补充。"
        )
    return (
        f"正文候选总长度仅 {rule_candidate_char_count} 个字符，"
        f"摘要仅 {abstract_char_count} 个字符，两者均不足以触发结构化生成。"
    )


def _needs_keyword_llm_fallback(
    current_keywords: list[str],
    source_kind: str,
    confidence: str,
) -> bool:
    normalized_source = normalize_line(source_kind).lower()
    normalized_confidence = normalize_line(confidence)
    if not current_keywords:
        return True
    if normalized_confidence == "低":
        return True
    if normalized_source == "frequency_fallback":
        return True
    if len(current_keywords) > MAX_KEYWORDS:
        return True
    if len(current_keywords) < MIN_KEYWORDS and not any(
        key in normalized_source for key in ("strategy_a", "strategy_b", "strategy_c", "explicit")
    ):
        return True
    # Trigger LLM if any keyword looks like multiple words concatenated (too long)
    if any(len(kw) > 10 and re.search(r"[\u4e00-\u9fff]{10,}", kw) for kw in current_keywords):
        return True
    return False


def _build_keyword_source_text(priority_text: str, raw_text: str) -> str:
    preferred = normalize_whitespace(priority_text) if priority_text else ""
    if preferred:
        return preferred[:2400]
    return normalize_whitespace(raw_text)[:2400]


def _keyword_fallback_with_responses(client: OpenAI, settings: RelaySettings, *, title: str, source_text: str) -> tuple[KeywordFallbackPayload, str]:
    response = client.responses.parse(
        model=settings.model,
        instructions=_build_keyword_fallback_system_prompt(),
        input=_build_keyword_fallback_input(title=title, source_text=source_text),
        text_format=KeywordFallbackPayload,
        temperature=0.1,
        max_output_tokens=KEYWORD_LLM_MAX_OUTPUT_TOKENS,
        verbosity="low",
        store=False,
    )
    payload = response.output_parsed
    raw_output = getattr(response, "output_text", "")
    if payload is None and raw_output:
        payload = _parse_keyword_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("关键词兜底未返回可解析结果。")
    return payload, raw_output


def _keyword_fallback_with_chat(client: OpenAI, settings: RelaySettings, *, title: str, source_text: str) -> tuple[KeywordFallbackPayload, str]:
    completion = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": _build_keyword_fallback_system_prompt()},
            {"role": "user", "content": _build_keyword_fallback_input(title=title, source_text=source_text)},
        ],
        temperature=0.1,
        max_tokens=KEYWORD_LLM_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    raw_output = _extract_chat_content(completion)
    payload = _parse_keyword_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("关键词兜底未返回可解析 JSON。")
    return payload, raw_output


def _build_keyword_fallback_system_prompt() -> str:
    return (
        “你是学术论文关键词整理助手。”
        “请仅依据提供的论文标题和关键词区原文，整理 3 到 8 个最贴近原文的关键词。”
        “只能使用关键词区中明确出现的词，不得从摘要或正文中补充新词。”
        “优先保留原词，不要凭空发明术语，不要把完整句子当成关键词。”
        “如果原文只清晰出现 2 到 4 个关键词，也可以少量返回，不要硬凑数量。”
        “删除”摘要、作者、引言、本文、研究、结果”等泛化词。”
        “输出 JSON 字段固定为 keywords 和 note。”
    )


def _build_keyword_fallback_input(*, title: str, source_text: str) -> str:
    return "\n".join(
        [
            f"论文标题：{title or '（未识别）'}",
            "请从以下片段中整理关键词：",
            source_text,
        ]
    )


def _parse_keyword_payload_from_text(text: str) -> KeywordFallbackPayload | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return KeywordFallbackPayload.model_validate_json(cleaned)
    except ValidationError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return KeywordFallbackPayload.model_validate_json(match.group(0))
        except ValidationError:
            return None


def _sanitize_keyword_payload(keywords: list[str]) -> list[str]:
    cleaned_items: list[str] = []
    for item in keywords:
        cleaned = normalize_line(str(item)).strip("；;，,、/| ")
        if not cleaned:
            continue
        if len(cleaned) > 20:
            continue
        if re.search(r"[。！？；;]", cleaned):
            continue
        if re.fullmatch(r"(摘要|作者|引言|本文|研究|结果|方法|结论)", cleaned, re.IGNORECASE):
            continue
        cleaned_items.append(cleaned)
    return compact_list(cleaned_items, MAX_KEYWORDS)


def _keyword_confidence_from_count(count: int) -> str:
    if MIN_KEYWORDS <= count <= MAX_KEYWORDS:
        return "中"
    if count >= 1:
        return "低"
    return "低"


def _author_fallback_with_responses(
    client: OpenAI,
    settings: RelaySettings,
    *,
    title: str,
    front_text: str,
) -> tuple[AuthorFallbackPayload, str]:
    response = client.responses.parse(
        model=settings.model,
        instructions=_build_author_fallback_system_prompt(),
        input=_build_author_fallback_input(title=title, front_text=front_text),
        text_format=AuthorFallbackPayload,
        temperature=0.0,
        max_output_tokens=120,
        verbosity="low",
        store=False,
    )
    payload = response.output_parsed
    raw_output = getattr(response, "output_text", "")
    if payload is None and raw_output:
        payload = _parse_author_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("作者识别未返回可解析结果。")
    return payload, raw_output


def _author_fallback_with_chat(
    client: OpenAI,
    settings: RelaySettings,
    *,
    title: str,
    front_text: str,
) -> tuple[AuthorFallbackPayload, str]:
    completion = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": _build_author_fallback_system_prompt()},
            {"role": "user", "content": _build_author_fallback_input(title=title, front_text=front_text)},
        ],
        temperature=0.0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    raw_output = _extract_chat_content(completion)
    payload = _parse_author_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("作者识别未返回可解析 JSON。")
    return payload, raw_output


def _build_author_fallback_system_prompt() -> str:
    return (
        "你是论文首页作者识别助手。"
        "只能依据给定的论文标题和首页前部文本识别作者名单，不得从正文、摘要或其他部分补充。"
        "不要猜测，不要补充机构，不要返回解释，不要把单位、学校、邮箱、摘要、关键词当作作者。"
        "只返回在首页明确出现的作者姓名，如果无法稳定判断作者，请返回空数组。"
        "输出 JSON 字段固定为 authors 和 note。"
    )


def _build_author_fallback_input(*, title: str, front_text: str) -> str:
    return "\n".join(
        [
            f"论文标题：{title or '未识别标题'}",
            "以下是论文首页前部文本，请仅提取作者名单：",
            front_text[:1800],
        ]
    )


def _parse_author_payload_from_text(text: str) -> AuthorFallbackPayload | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return AuthorFallbackPayload.model_validate_json(cleaned)
    except ValidationError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return AuthorFallbackPayload.model_validate_json(match.group(0))
        except ValidationError:
            return None


def _sanitize_author_payload(authors: list[str]) -> list[str]:
    cleaned_items: list[str] = []
    for item in authors:
        cleaned = normalize_line(str(item))
        cleaned = re.sub(r"^(?:作者|author(?:s)?|by)\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip("，,、;；:：/| ")
        if not cleaned:
            continue
        if len(cleaned) > 32:
            continue
        if re.search(r"(大学|学院|学校|研究院|研究所|department|university|college|school|institute|摘要|关键词|abstract|keywords?|@)", cleaned, re.IGNORECASE):
            continue
        if re.fullmatch(r"[\u4e00-\u9fff·]{2,8}", cleaned):
            cleaned_items.append(cleaned)
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z.\-'\s]{1,30}", cleaned):
            tokens = [token for token in cleaned.split() if token]
            if len(tokens) >= 2:
                cleaned_items.append(" ".join(tokens))
    return compact_list(cleaned_items, 6)


def _build_course_support_source_text(request: CourseSupportRequest) -> str:
    lines = [
        f"论文标题：{request.title or '未识别标题'}",
        f"作者：{'、'.join(request.authors) if request.authors else '未识别作者信息'}",
        f"关键词：{'、'.join(request.keywords) if request.keywords else '未提取到关键词'}",
        f"主摘要：{request.summary or '未识别到摘要'}",
        f"研究问题：{request.research_question or '未识别'}",
        f"研究方法：{request.research_method or '未识别'}",
        f"核心结论：{request.core_conclusion or '未识别'}",
    ]
    if request.raw_text_preview:
        lines.extend(["正文预览：", request.raw_text_preview[:1800]])
    return "\n".join(lines)


def _course_support_with_responses(client: OpenAI, settings: RelaySettings, request: CourseSupportRequest) -> tuple[CourseSupportPayload, str]:
    response = client.responses.parse(
        model=settings.model,
        instructions=_build_course_support_system_prompt(),
        input=_build_course_support_input(request),
        text_format=CourseSupportPayload,
        temperature=0.2,
        max_output_tokens=COURSE_SUPPORT_LLM_MAX_OUTPUT_TOKENS,
        verbosity="low",
        store=False,
    )
    payload = response.output_parsed
    raw_output = getattr(response, "output_text", "")
    if payload is None and raw_output:
        payload = _parse_course_support_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("课程写作辅助未返回可解析结果。")
    return payload, raw_output


def _course_support_with_chat(client: OpenAI, settings: RelaySettings, request: CourseSupportRequest) -> tuple[CourseSupportPayload, str]:
    completion = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": _build_course_support_system_prompt()},
            {"role": "user", "content": _build_course_support_input(request)},
        ],
        temperature=0.2,
        max_tokens=COURSE_SUPPORT_LLM_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    raw_output = _extract_chat_content(completion)
    payload = _parse_course_support_payload_from_text(raw_output)
    if payload is None:
        raise ValueError("课程写作辅助未返回可解析 JSON。")
    return payload, raw_output


def _build_course_support_system_prompt() -> str:
    return (
        "你是面向大学生课程写作场景的论文解读助手。"
        "请仅依据提供的标题、作者、关键词、摘要、结构化字段和正文预览生成结果，不得虚构实验、数据、结论或论文未出现的研究对象。"
        "输出要适合课程汇报、讲稿和课程论文前期整理。"
        "当信息不足时，请明确使用保守表述，例如“根据当前可识别内容，暂可归纳为……”。"
        "innovation_points 与 limitation_points 各输出 2 到 4 条；outline 字段输出条目化中文短句。"
        "输出 JSON 字段固定为 plain_language_summary、method_explanation、innovation_points、limitation_points、course_presentation_outline、course_paper_outline、literature_review_outline、note。"
    )


def _build_course_support_input(request: CourseSupportRequest) -> str:
    return "\n".join(
        [
            f"论文标题：{request.title or '未识别标题'}",
            f"作者：{'、'.join(request.authors) if request.authors else '未识别作者信息'}",
            f"关键词：{'、'.join(request.keywords) if request.keywords else '未提取到关键词'}",
            f"主摘要：{request.summary or '未识别到摘要'}",
            f"研究问题：{request.research_question or '未识别'}",
            f"研究方法：{request.research_method or '未识别'}",
            f"核心结论：{request.core_conclusion or '未识别'}",
            f"正文预览：{request.raw_text_preview[:1800] if request.raw_text_preview else '无'}",
        ]
    )


def _parse_course_support_payload_from_text(text: str) -> CourseSupportPayload | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return CourseSupportPayload.model_validate_json(cleaned)
    except ValidationError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return CourseSupportPayload.model_validate_json(match.group(0))
        except ValidationError:
            return None


def _merge_course_support_payload(
    local_result: CourseSupportResult,
    payload: CourseSupportPayload,
    *,
    backend: str,
    debug_info: dict[str, object],
) -> CourseSupportResult:
    merged = CourseSupportResult(
        plain_language_summary=_sanitize_course_text(payload.plain_language_summary) or local_result.plain_language_summary,
        method_explanation=_sanitize_course_text(payload.method_explanation) or local_result.method_explanation,
        innovation_points=_sanitize_course_list(payload.innovation_points, local_result.innovation_points, minimum=2, maximum=4),
        limitation_points=_sanitize_course_list(payload.limitation_points, local_result.limitation_points, minimum=2, maximum=4),
        course_presentation_outline=_sanitize_course_list(payload.course_presentation_outline, local_result.course_presentation_outline, minimum=4, maximum=6),
        course_paper_outline=_sanitize_course_list(payload.course_paper_outline, local_result.course_paper_outline, minimum=4, maximum=6),
        literature_review_outline=_sanitize_course_list(payload.literature_review_outline, local_result.literature_review_outline, minimum=4, maximum=6),
        note=_sanitize_course_text(payload.note) or "课程写作辅助结果由模型结合结构化字段生成。",
        backend=backend,
        debug_info={
            **debug_info,
            "valid_return": True,
            "parsed_payload": payload.model_dump(),
        },
    )
    return merged


def _build_local_course_support_result(request: CourseSupportRequest) -> CourseSupportResult:
    innovation_points = _build_local_innovation_points(request)
    limitation_points = _build_local_limitation_points(request)
    return CourseSupportResult(
        plain_language_summary=_build_local_plain_language_summary(request),
        method_explanation=_build_local_method_explanation(request),
        innovation_points=innovation_points,
        limitation_points=limitation_points,
        course_presentation_outline=_build_course_presentation_outline(request, innovation_points, limitation_points),
        course_paper_outline=_build_course_paper_outline(request, innovation_points, limitation_points),
        literature_review_outline=_build_literature_review_outline(request, innovation_points, limitation_points),
        note="当前结果主要依据摘要、结构化字段和正文前部片段做保守整理。",
        backend="local_rule",
        debug_info={"valid_return": True},
    )


def _build_local_plain_language_summary(request: CourseSupportRequest) -> str:
    question = _meaningful_field(request.research_question)
    method = _meaningful_field(request.research_method)
    conclusion = _meaningful_field(request.core_conclusion)
    summary = _meaningful_summary(request.summary)
    topic = _topic_phrase(request)

    sentences: list[str] = []
    if question:
        sentences.append(f"这篇论文主要想回答的问题是：{_strip_terminal(question)}。")
    elif topic:
        sentences.append(f"这篇论文主要围绕{topic}展开讨论，适合作为课程写作中的单篇文献解读材料。")
    if method:
        sentences.append(f"作者主要通过{_strip_terminal(method)}来分析这一问题。")
    if conclusion:
        sentences.append(f"从当前可识别内容看，论文认为{_strip_terminal(conclusion)}。")
    elif summary:
        sentences.append(f"根据当前可识别内容，论文的核心内容可以概括为：{_compact_summary_fragment(summary)}。")

    merged = " ".join(sentences[:3]).strip()
    if merged:
        return merged
    return "根据当前可识别内容，暂可归纳为：这篇论文围绕特定研究问题展开分析，建议结合原文摘要和结论部分进一步核对。"


def _build_local_method_explanation(request: CourseSupportRequest) -> str:
    method = _meaningful_field(request.research_method)
    if not method:
        return "根据当前可识别内容，论文的方法信息仍较有限，建议结合原文的方法或研究设计部分补充核对。"
    method_core = _strip_terminal(method)
    return f"论文主要采用{method_core}。{_infer_method_reason(method_core)}"


def _build_local_innovation_points(request: CourseSupportRequest) -> list[str]:
    items: list[str] = []
    topic = _topic_phrase(request)
    question = _meaningful_field(request.research_question)
    method = _meaningful_field(request.research_method)
    conclusion = _meaningful_field(request.core_conclusion)

    if topic:
        items.append(f"选题聚焦在{topic}，有助于把课程讨论落到更具体的研究对象或研究情境上。")
    if question:
        items.append(f"论文把研究问题表达得较明确，便于在汇报中快速说明作者究竟要回答什么问题。")
    if method:
        items.append(f"作者使用{_strip_terminal(method)}展开分析，使研究不只停留在概念描述层面。")
    if conclusion:
        items.append(f"论文形成了相对清晰的核心结论，方便整理成课堂展示中的“主要发现”。")

    sanitized = _sanitize_course_list(items, [], minimum=2, maximum=4)
    if sanitized:
        return sanitized
    return [
        "根据当前可识别内容，论文对研究主题做了相对明确的聚焦，适合整理为课程汇报中的“研究对象与问题”。",
        "当前结果已提取出研究方法和核心结论，可作为课堂展示时概括论文贡献的基础。",
    ]


def _build_local_limitation_points(request: CourseSupportRequest) -> list[str]:
    items: list[str] = []
    method = _meaningful_field(request.research_method)
    conclusion = _meaningful_field(request.core_conclusion)

    if method:
        items.append("当前可识别结果已经说明了方法方向，但样本、步骤和数据处理细节仍需回看正文核对。")
    else:
        items.append("研究方法的细化步骤在当前可识别内容中仍不充分，课堂写作时需要补查原文的方法部分。")
    if conclusion:
        items.append("现有结论能够概括主要发现，但其适用范围和边界条件仍需要结合全文判断。")
    else:
        items.append("核心结论的证据链条在当前可识别内容中仍不完整，后续写作时需要补查结果与讨论部分。")
    items.append("当前输出主要依赖摘要和前部文本，如果要用于课程论文，建议与全文对读后再形成正式判断。")

    sanitized = _sanitize_course_list(items, [], minimum=2, maximum=4)
    if sanitized:
        return sanitized
    return [
        "根据当前可识别内容，部分方法细节和论证过程仍需回看正文，不能仅凭摘要做强结论判断。",
        "若要用于课程论文或综述写作，建议进一步核对样本、数据来源和结论适用范围。",
    ]


def _build_course_presentation_outline(
    request: CourseSupportRequest,
    innovation_points: list[str],
    limitation_points: list[str],
) -> list[str]:
    topic = _topic_phrase(request) or "论文主题"
    question = _meaningful_field(request.research_question) or "论文试图回答的核心问题"
    method = _meaningful_field(request.research_method) or "作者采用的主要分析路径"
    conclusion = _meaningful_field(request.core_conclusion) or "论文的主要发现"
    return [
        f"研究背景与选题缘起：说明{topic}为什么值得在课程中讨论。",
        f"研究问题：概括论文试图回答“{_strip_terminal(question)}”。",
        f"研究方法：简述作者如何通过{_strip_terminal(method)}展开分析。",
        f"主要发现：提炼论文的核心结论，即“{_strip_terminal(conclusion)}”。",
        f"创新与不足：结合已整理的 {len(innovation_points)} 条创新点和 {len(limitation_points)} 条不足做课堂讨论。",
        "可汇报结论：收束为“这篇论文对课程主题带来了什么启发”。",
    ]


def _build_course_paper_outline(
    request: CourseSupportRequest,
    innovation_points: list[str],
    limitation_points: list[str],
) -> list[str]:
    topic = _topic_phrase(request) or "该论文主题"
    method = _meaningful_field(request.research_method) or "论文的研究方法"
    conclusion = _meaningful_field(request.core_conclusion) or "论文的核心结论"
    return [
        f"引言：交代{topic}的研究背景、课程关联与选题意义。",
        "文献内容整理：概括论文的研究问题、研究对象和关键词。",
        f"方法与结论分析：围绕{_strip_terminal(method)}和“{_strip_terminal(conclusion)}”展开说明。",
        f"创新点分析：从已整理的 {len(innovation_points)} 条创新点中提炼可写入正文的评价。",
        f"不足与反思：结合 {len(limitation_points)} 条不足分析论文的局限与可改进处。",
        "可延展讨论：提出可与其他文献继续比较的议题或研究方向。",
    ]


def _build_literature_review_outline(
    request: CourseSupportRequest,
    innovation_points: list[str],
    limitation_points: list[str],
) -> list[str]:
    topic = _topic_phrase(request) or "该研究主题"
    question = _meaningful_field(request.research_question) or "论文关注的核心问题"
    method = _meaningful_field(request.research_method) or "作者采用的方法路径"
    return [
        f"研究主题定位：明确本文可归入的研究主题是{topic}。",
        f"研究对象与问题：记录论文围绕“{_strip_terminal(question)}”展开讨论。",
        f"方法路径：整理作者采用的{_strip_terminal(method)}以及相应分析思路。",
        "主要观点与发现：提炼论文可纳入综述的核心观点和结论。",
        f"创新与不足：汇总 {len(innovation_points)} 条创新点和 {len(limitation_points)} 条不足作为评价维度。",
        "可比较维度：标记后续与其他文献横向比较时可使用的主题、方法和结论切入点。",
    ]


def _sanitize_course_text(text: str) -> str:
    cleaned = normalize_whitespace(text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _sanitize_course_list(
    items: list[str],
    fallback: list[str],
    *,
    minimum: int,
    maximum: int,
) -> list[str]:
    cleaned_items: list[str] = []
    for item in items:
        cleaned = _sanitize_course_text(item)
        if not cleaned:
            continue
        if len(cleaned) > 120:
            cleaned = cleaned[:118].rstrip("，,；;。.!? ") + "。"
        cleaned_items.append(cleaned)
    compacted = compact_list(cleaned_items, maximum)
    if len(compacted) >= minimum:
        return compacted
    if fallback:
        fallback_items = compact_list([_sanitize_course_text(item) for item in fallback], maximum)
        if len(fallback_items) >= minimum:
            return fallback_items
    return compacted or fallback[:maximum]


def _meaningful_field(text: str) -> str:
    cleaned = _sanitize_course_text(text)
    if not cleaned:
        return ""
    if cleaned == STRUCTURED_LOW_CONFIDENCE_TEXT:
        return ""
    if cleaned.startswith("未从原文摘要中识别到"):
        return ""
    return cleaned


def _meaningful_summary(text: str) -> str:
    cleaned = _sanitize_course_text(text)
    if not cleaned or cleaned == "未能生成摘要内容。":
        return ""
    return cleaned


def _strip_terminal(text: str) -> str:
    return _sanitize_course_text(text).rstrip("。！？!?；;")


def _compact_summary_fragment(text: str) -> str:
    fragment = _strip_terminal(text)
    if len(fragment) <= 88:
        return fragment
    return fragment[:86].rstrip("，,；;。.!? ") + "……"


def _topic_phrase(request: CourseSupportRequest) -> str:
    title = _sanitize_course_text(request.title)
    if title and title != "未识别标题":
        if len(title) <= 28:
            return f"“{title}”"
        keywords = compact_list(request.keywords, 2)
        if keywords:
            return "、".join(keywords)
        return "该论文所讨论的主题"
    keywords = compact_list(request.keywords, 2)
    if keywords:
        return "、".join(keywords)
    return ""


def _infer_method_reason(method_text: str) -> str:
    lowered = method_text.lower()
    if re.search(r"(问卷|访谈|survey|interview)", lowered, re.IGNORECASE):
        return "这种方法更适合了解研究对象的看法、经验和实际反馈。"
    if re.search(r"(案例|case)", lowered, re.IGNORECASE):
        return "这种方法适合把复杂问题放进具体情境中展开说明。"
    if re.search(r"(回归|模型|实证|regression|model|empirical)", lowered, re.IGNORECASE):
        return "这种方法更适合分析变量关系，并为结论提供较清晰的证据支撑。"
    if re.search(r"(文本分析|文献分析|内容分析|text|document)", lowered, re.IGNORECASE):
        return "这种方法适合梳理概念、观点和论证路径，便于做课程化总结。"
    if re.search(r"(比较|compare)", lowered, re.IGNORECASE):
        return "这种方法有助于把不同对象或不同情境之间的差异展示得更清楚。"
    return "这种做法有助于说明作者如何组织分析过程，并把研究问题与结论连接起来。"
