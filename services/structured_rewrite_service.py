"""Local placeholder rewrite layer for structured extraction results.

This module keeps a stable interface for future LLM-based rewriting while the
current implementation stays fully local and rule based.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from dataclasses import dataclass, field
from typing import Callable

from utils.text_utils import (
    filter_noise_sentences,
    normalize_line,
    sanitize_metadata_fragments,
    split_sentences,
)

RewriteCallable = Callable[["StructuredRewriteRequest"], "StructuredRewriteResult | None"]

_FIELD_SENTENCE_LIMITS = {
    "research_question": 1,
    "research_method": 1,
    "core_conclusion": 1,
}

_FIELD_PROMPT_TITLES = {
    "research_question": "研究问题",
    "research_method": "研究方法",
    "core_conclusion": "核心结论",
}


@dataclass(slots=True)
class StructuredFieldCandidate:
    """Candidate text produced by the rule extractor."""

    field_name: str
    text: str
    source_kind: str = "rule"
    source_hint: str = ""
    score: float = 0.0
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredRewriteRequest:
    """Input envelope for structured-field rewriting."""

    raw_text: str = ""
    title: str = ""
    candidates: dict[str, StructuredFieldCandidate] = field(default_factory=dict)
    preferred_backend: str = "local_rule"
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredRewriteResult:
    """Output of the rewrite layer."""

    research_question: str = ""
    research_method: str = ""
    core_conclusion: str = ""
    confidence: str = ""
    note: str = ""
    backend: str = "local_rule"
    candidates: dict[str, StructuredFieldCandidate] = field(default_factory=dict)
    prompt_preview: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        return {
            "research_question": self.research_question,
            "research_method": self.research_method,
            "core_conclusion": self.core_conclusion,
            "confidence": self.confidence,
            "note": self.note,
        }


def build_structured_rewrite_request(
    *,
    raw_text: str,
    title: str = "",
    candidates: dict[str, StructuredFieldCandidate] | None = None,
    preferred_backend: str = "local_rule",
    debug_info: dict[str, object] | None = None,
) -> StructuredRewriteRequest:
    return StructuredRewriteRequest(
        raw_text=raw_text,
        title=title,
        candidates=_sanitize_candidates_for_rewrite(title, candidates or {}),
        preferred_backend=preferred_backend,
        debug_info=debug_info or {},
    )


def rewrite_structured_fields(
    request: StructuredRewriteRequest,
    rewriter: RewriteCallable | None = None,
) -> StructuredRewriteResult:
    """Rewrite structured fields locally or via an injected future backend."""

    if rewriter is not None:
        try:
            rewritten_result = rewriter(request)
        except Exception:
            rewritten_result = None
        if rewritten_result is not None:
            return _finalize_result(rewritten_result, request)

    if request.preferred_backend != "local_rule":
        # Future LLM backends can be plugged in through `rewriter`.
        pass

    rewritten: dict[str, str] = {}
    for field_name in _FIELD_SENTENCE_LIMITS:
        rewritten_text = _rewrite_single_field(field_name, request.candidates.get(field_name))
        rewritten[field_name] = rewritten_text

    return StructuredRewriteResult(
        research_question=rewritten["research_question"],
        research_method=rewritten["research_method"],
        core_conclusion=rewritten["core_conclusion"],
        confidence=_estimate_local_confidence(rewritten),
        note="当前结果基于原文规则抽取，未启用或未成功调用模型重写。",
        backend="local_rule",
        candidates=request.candidates,
        prompt_preview=build_rewrite_prompt(request),
        debug_info={
            "backend": "local_rule",
            "valid_return": any(bool(value) for value in rewritten.values()),
            "abstract_fallback_used": False,
            "abstract_fallback_fields": [],
            "llm_input_source": request.debug_info.get("llm_input_source", ""),
            "low_confidence_reason": _build_local_low_confidence_reason(
                rewritten,
                abstract_fallback_available=False,
                abstract_fallback_fields=[],
            ),
        },
    )


def build_rewrite_prompt(request: StructuredRewriteRequest) -> str:
    """Build a compact prompt for a future LLM rewrite backend."""

    lines = [
        "请将以下候选内容重写为更自然、完整、简洁的学术表达。",
        "要求：",
        "1. 保持原意，不新增事实。",
        "2. 分别输出研究问题、研究方法、核心结论。",
        "3. 删除编号、网址、基金编号、引用标记等杂质。",
        "4. 禁止输出论文标题或与标题高度相似的句子。",
    ]
    if request.title:
        lines.append(f"论文标题：{request.title}")

    for field_name in _FIELD_SENTENCE_LIMITS:
        label = _FIELD_PROMPT_TITLES[field_name]
        candidate = request.candidates.get(field_name)
        candidate_text = candidate.text if candidate else ""
        lines.append(f"{label}候选：{candidate_text or '（空）'}")
    abstract_fallback_text = str(request.debug_info.get("abstract_fallback_text", "")).strip()
    if abstract_fallback_text:
        lines.append(f"中文摘要补充：{abstract_fallback_text}")

    return "\n".join(lines)


def _rewrite_single_field(field_name: str, candidate: StructuredFieldCandidate | None) -> str:
    if candidate is None or not candidate.text.strip():
        return ""

    return finalize_structured_field_text(field_name, candidate.text)


def finalize_structured_field_text(field_name: str, text: str, title: str = "") -> str:
    if not text.strip():
        return ""

    text = filter_title_like_sentences(text, title)
    text = sanitize_metadata_fragments(text)
    if _is_english_heavy_text(text):
        return ""
    text = _strip_noise_markers(text)
    text = _strip_leading_labels(text)
    text = _normalize_spacing(text)

    sentences = filter_noise_sentences(split_sentences(text))
    if not sentences:
        return _finalize_text(text)

    sentences = _dedupe_sentences(sentences)
    sentences = [sentence for sentence in sentences if _is_readable_sentence(sentence)]
    sentences = _trim_sentences_for_field(field_name, sentences)
    return _finalize_text(" ".join(sentences))


def filter_title_like_sentences(text: str, title: str) -> str:
    normalized_title = _normalize_compare_text(title)
    if not text.strip() or not normalized_title:
        return text

    sentences = split_sentences(text)
    if not sentences:
        return "" if _is_title_like_sentence(text, title) else text

    kept = [sentence for sentence in sentences if not _is_title_like_sentence(sentence, title)]
    return " ".join(kept).strip()


def is_title_like_text(text: str, title: str) -> bool:
    return _is_title_like_sentence(text, title)


def _trim_sentences_for_field(field_name: str, sentences: list[str]) -> list[str]:
    if len(sentences) <= _FIELD_SENTENCE_LIMITS[field_name]:
        return sentences

    scored = [(score_sentence(field_name, sentence), index, sentence) for index, sentence in enumerate(sentences)]
    best_score, best_index, _ = max(scored, key=lambda item: item[0])

    selected = [sentences[best_index]]
    if len(selected) < _FIELD_SENTENCE_LIMITS[field_name] and best_index + 1 < len(sentences):
        next_sentence = sentences[best_index + 1]
        if score_sentence(field_name, next_sentence) >= max(0.9, best_score - 1.8):
            selected.append(next_sentence)
    if len(selected) < _FIELD_SENTENCE_LIMITS[field_name] and best_index > 0:
        prev_sentence = sentences[best_index - 1]
        if score_sentence(field_name, prev_sentence) >= max(0.9, best_score - 1.8):
            selected.insert(0, prev_sentence)
    return selected[: _FIELD_SENTENCE_LIMITS[field_name]]


def score_sentence(field_name: str, sentence: str) -> float:
    lowered = sentence.lower()
    score = 0.0

    if field_name == "research_question":
        if re.search(r"(旨在|目的|目标|探讨|关注|回答|问题|aims? to|seeks? to|investigates?|examines?|focuses? on)", lowered, re.IGNORECASE):
            score += 3.0
    elif field_name == "research_method":
        if re.search(r"(采用|通过|基于|构建|运用|结合|问卷|访谈|实验|案例|实证|回归|文本分析|survey|interview|experiment|case study|regression|method|model|framework|document analysis|rule-based)", lowered, re.IGNORECASE):
            score += 3.0
    elif field_name == "core_conclusion":
        if re.search(r"(结果表明|研究发现|研究表明|说明了|表明|发现|结论|results? show|findings? indicate|demonstrate|suggest)", lowered, re.IGNORECASE):
            score += 3.0

    if 18 <= len(sentence) <= 180:
        score += 1.0
    if re.search(r"\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}", sentence):
        score -= 2.0
    if re.search(r"https?://|www\.|doi", sentence, re.IGNORECASE):
        score -= 3.0
    if re.search(r"(参考文献|references?|基金项目|项目编号|收稿日期|修回日期|作者简介|通信作者)", sentence, re.IGNORECASE):
        score -= 2.5
    if sentence.endswith(("：", ":")):
        score -= 1.2
    return score


def _strip_noise_markers(text: str) -> str:
    text = re.sub(r"(?<!\w)(?:\[\d+(?:,\s*\d+)*\]|\[\d+\]|\【\d+\】|\(\d+\)|（\d+）)", " ", text)
    text = re.sub(r"[（(][^()（）]{0,40}\d{4}[a-z]?[^()（）]{0,40}[)）]", " ", text)
    text = re.sub(r"\[[JDMCRSNPA]+\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.)]?|\([一二三四五六七八九十]\))\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _strip_leading_labels(text: str) -> str:
    stripped = re.sub(
        r"^(?:研究问题|研究方法|研究结论|核心结论|结论|摘要|abstract|引言|introduction|method|methods|methodology)\s*(?:[:：]\s*)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"^[，,；;：:]\s*", "", stripped)


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([，。！？；：])", r"\1", text)
    text = re.sub(r"([，。！？；：])\s+", r"\1", text)
    text = re.sub(r"\s+([,!?;:])", r"\1", text)
    text = re.sub(r"([,!?;:])(?=[A-Za-z0-9])", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _dedupe_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for sentence in sentences:
        normalized = _strip_noise_markers(sentence)
        normalized = _strip_leading_labels(normalized)
        normalized = _normalize_spacing(normalized)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _finalize_text(text: str) -> str:
    text = _normalize_spacing(_strip_noise_markers(text))
    text = _strip_leading_labels(text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        return ""
    if text[-1] not in "。！？!?；;.":
        text += "。"
    return text


def _is_readable_sentence(sentence: str) -> bool:
    cleaned = _normalize_spacing(_strip_noise_markers(sentence))
    if len(cleaned) < 16:
        return False
    if len(cleaned) > 130:
        return False
    if re.search(r"https?://|www\.|doi", cleaned, re.IGNORECASE):
        return False
    if re.search(r"(参考文献|references?|基金项目|项目编号|收稿日期|作者简介)", cleaned, re.IGNORECASE):
        return False
    if _is_english_heavy_text(cleaned):
        return False
    return True


def _finalize_result(result: StructuredRewriteResult, request: StructuredRewriteRequest) -> StructuredRewriteResult:
    normalized_fields = {
        field_name: finalize_structured_field_text(field_name, getattr(result, field_name, ""), request.title)
        for field_name in _FIELD_SENTENCE_LIMITS
    }
    note = sanitize_metadata_fragments(getattr(result, "note", ""))
    if _is_english_heavy_text(note):
        note = ""
    note = _normalize_spacing(_strip_noise_markers(note))

    return StructuredRewriteResult(
        research_question=normalized_fields["research_question"],
        research_method=normalized_fields["research_method"],
        core_conclusion=normalized_fields["core_conclusion"],
        confidence=_normalize_confidence(getattr(result, "confidence", "")),
        note=note,
        backend=getattr(result, "backend", "local_rule"),
        candidates=request.candidates,
        prompt_preview=getattr(result, "prompt_preview", build_rewrite_prompt(request)),
        debug_info=dict(getattr(result, "debug_info", {})),
    )


def _estimate_local_confidence(rewritten: dict[str, str]) -> str:
    populated_count = sum(1 for value in rewritten.values() if value)
    if populated_count >= 3:
        return "中"
    if populated_count >= 1:
        return "低"
    return "低"


def _build_local_low_confidence_reason(
    rewritten: dict[str, str],
    *,
    abstract_fallback_available: bool = False,
    abstract_fallback_fields: list[str] | None = None,
) -> str:
    abstract_fallback_fields = abstract_fallback_fields or []
    populated_fields = [field_name for field_name, value in rewritten.items() if value.strip()]
    if len(populated_fields) == 3:
        return ""
    if not populated_fields:
        if abstract_fallback_available:
            return "规则回退阶段已尝试中文摘要补充，但仍未提取到可用结构化结果。"
        return "规则回退阶段未提取到可用结构化结果。"
    if abstract_fallback_fields:
        return (
            f"规则回退阶段通过中文摘要补充补全了 {', '.join(abstract_fallback_fields)}，"
            f"其余字段为空。"
        )
    return f"规则回退阶段仅保留了 {', '.join(populated_fields)}，其余字段为空。"


def _normalize_confidence(confidence: str) -> str:
    normalized = normalize_line(confidence).lower()
    mapping = {
        "high": "高",
        "medium": "中",
        "low": "低",
        "高": "高",
        "中": "中",
        "低": "低",
    }
    return mapping.get(normalized, "低")


def _sanitize_candidates_for_rewrite(
    title: str,
    candidates: dict[str, StructuredFieldCandidate],
) -> dict[str, StructuredFieldCandidate]:
    sanitized_candidates: dict[str, StructuredFieldCandidate] = {}
    for field_name, candidate in candidates.items():
        candidate_text = filter_title_like_sentences(candidate.text, title)
        candidate_text = sanitize_metadata_fragments(candidate_text)
        candidate_text = _normalize_spacing(_strip_noise_markers(candidate_text))
        sanitized_candidates[field_name] = StructuredFieldCandidate(
            field_name=field_name,
            text=candidate_text,
            source_kind=candidate.source_kind,
            source_hint=candidate.source_hint,
            score=candidate.score,
            debug_info=dict(candidate.debug_info),
        )
    return sanitized_candidates


def _is_title_like_sentence(sentence: str, title: str) -> bool:
    normalized_sentence = _normalize_compare_text(sentence)
    normalized_title = _normalize_compare_text(title)
    if not normalized_sentence or not normalized_title:
        return False
    if normalized_sentence == normalized_title:
        return True
    if (
        normalized_title in normalized_sentence
        and len(normalized_title) >= 8
        and len(normalized_sentence) <= len(normalized_title) + 6
    ):
        return True
    if normalized_sentence in normalized_title and len(normalized_sentence) >= max(8, int(len(normalized_title) * 0.6)):
        return True

    similarity = SequenceMatcher(None, normalized_sentence, normalized_title).ratio()
    if similarity >= 0.92 and len(normalized_sentence) <= int(len(normalized_title) * 1.15):
        return True

    sentence_tokens = _build_compare_units(normalized_sentence)
    title_tokens = _build_compare_units(normalized_title)
    if not sentence_tokens or not title_tokens:
        return False
    overlap_ratio = len(sentence_tokens & title_tokens) / max(len(title_tokens), 1)
    return overlap_ratio >= 0.88 and len(normalized_sentence) <= len(normalized_title) * 1.2


def _normalize_compare_text(text: str) -> str:
    lowered = normalize_line(text).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", lowered)


def _build_compare_units(text: str) -> set[str]:
    if not text:
        return set()
    if re.search(r"[\u4e00-\u9fff]", text):
        if len(text) <= 2:
            return {text}
        return {text[index:index + 2] for index in range(len(text) - 1)}
    tokens = re.findall(r"[a-z0-9]{2,}", text)
    return set(tokens) or {text}


def _is_english_heavy_text(text: str) -> bool:
    english_chars = len(re.findall(r"[A-Za-z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if english_chars < 24:
        return False
    return chinese_chars == 0 or english_chars > chinese_chars * 2.5
