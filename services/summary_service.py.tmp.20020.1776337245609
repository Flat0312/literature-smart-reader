"""Abstract extraction and fallback summary generation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from config.settings import SUMMARY_SENTENCE_MAX, SUMMARY_SENTENCE_MIN, SUMMARY_SOURCE_TEXT_LIMIT
from utils.text_utils import (
    clean_extracted_text,
    extract_chinese_phrases,
    extract_english_tokens,
    filter_noise_sentences,
    is_noise_line,
    is_noise_sentence,
    limit_sentences,
    normalize_whitespace,
    reflow_text_for_display,
    sanitize_metadata_fragments,
    split_sentences,
)

_ZH_ABSTRACT_PATTERNS = [
    r"(?:(?<=^)|(?<=[\n。；;]))(?:〔\s*摘\s*要\s*〕|〔\s*摘要\s*〕|\[\s*摘\s*要\s*\]|\[\s*摘要\s*\]|摘\s*要|(?<!英文)(?<!英 文)摘要)\s*[：:]?\s*(.+?)(?=(?:\s*(?:关键词|关键字)|英文摘要|ABSTRACT|Abstract|abstract|引言|绪论|一、|1[.、]|第一章|$))",
]

_EN_ABSTRACT_PATTERNS = [
    r"(?:(?<=^)|(?<=[\n。；;]))(?:英文摘要|abstract)\s*[：:]?\s*(.+?)(?=(?:keywords?|index terms?|引言|introduction|一、|1\.|第一章|references?|$))",
]

_ABSTRACT_INLINE_NOISE_PATTERNS = [
    r"本文系[^。；;\n]*[。；;]?",
    r"本研究受[^。；;\n]*[。；;]?",
    r"基金项目[^。；;\n]*[。；;]?",
    r"项目编号[^。；;\n]*[。；;]?",
    r"网络首发(?:时间|地址)?[^。；;\n]*[。；;]?",
    r"作者单位[^。；;\n]*[。；;]?",
    r"作者简介[^。；;\n]*[。；;]?",
    r"通信作者[^。；;\n]*[。；;]?",
    r"[（(]\s*项目编号[^）)]*[）)]",
    r"[（(]\s*基金项目[^）)]*[）)]",
]

_ABSTRACT_STOP_PATTERNS = [
    r"^(?:关键词|关键字|英文摘要|ABSTRACT|Abstract|abstract)\b",
    r"^\s*(?:引言|绪论|一、|1[.、]|第一章|参考文献|references?)",
]


@dataclass(slots=True)
class AbstractResult:
    chinese_abstract: str = ""
    english_abstract: str = ""
    display_abstract: str = ""
    source_language: str = ""
    debug_info: dict[str, object] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return self.display_abstract


def generate_summary(raw_text: str, priority_text: str = "") -> str:
    abstract_result = extract_abstracts(raw_text, priority_text=priority_text)
    if abstract_result.display_abstract:
        return abstract_result.display_abstract

    working_text = clean_extracted_text(raw_text)
    return _build_fallback_summary(working_text)


def extract_abstracts(raw_text: str, priority_text: str = "") -> AbstractResult:
    abstract_result = extract_abstract_sections(raw_text, priority_text=priority_text)
    if abstract_result.display_abstract:
        return abstract_result

    working_text = clean_extracted_text(raw_text)
    display_abstract = _build_fallback_summary(working_text)
    return AbstractResult(
        chinese_abstract="",
        english_abstract="",
        display_abstract=display_abstract,
        source_language="fallback" if display_abstract else "",
        debug_info={
            **abstract_result.debug_info,
            "summary_source_language": "fallback" if display_abstract else "",
        },
    )


def extract_abstract_sections(raw_text: str, priority_text: str = "") -> AbstractResult:
    working_text = clean_extracted_text(raw_text)
    priority_window = clean_extracted_text(priority_text) if priority_text else ""
    candidate_texts = [text for text in (priority_window, working_text[:SUMMARY_SOURCE_TEXT_LIMIT]) if text.strip()]

    chinese_abstract = _extract_priority_abstract(candidate_texts, language="zh")
    english_abstract = _extract_priority_abstract(candidate_texts, language="en")
    display_abstract = chinese_abstract or english_abstract
    source_language = "zh" if chinese_abstract else "en" if english_abstract else ""

    return AbstractResult(
        chinese_abstract=chinese_abstract,
        english_abstract=english_abstract,
        display_abstract=display_abstract,
        source_language=source_language,
        debug_info={
            "used_priority_window": bool(priority_window.strip()),
            "has_chinese_abstract": bool(chinese_abstract),
            "has_english_abstract": bool(english_abstract),
            "summary_source_language": source_language,
        },
    )


def _extract_priority_abstract(candidate_texts: list[str], *, language: str) -> str:
    for text in candidate_texts:
        abstract_text = _extract_abstract_section(text, language=language)
        if abstract_text:
            return abstract_text
    return ""


def _extract_abstract_section(raw_text: str, *, language: str) -> str:
    patterns = _ZH_ABSTRACT_PATTERNS if language == "zh" else _EN_ABSTRACT_PATTERNS
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        cleaned = _clean_abstract_block(match.group(1), language=language)
        if cleaned:
            return cleaned
    return ""


def _clean_abstract_block(abstract_text: str, *, language: str) -> str:
    cleaned_lines: list[str] = []

    for raw_line in abstract_text.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            continue
        if _is_abstract_stop_line(line):
            break
        line = _strip_abstract_noise(line)
        if not line or _is_abstract_noise_line(line):
            continue
        cleaned_lines.append(line)

    cleaned = reflow_text_for_display("\n".join(cleaned_lines)).strip()
    cleaned = _strip_abstract_noise(cleaned)
    cleaned = sanitize_metadata_fragments(cleaned)
    cleaned = normalize_whitespace(cleaned)
    cleaned = re.sub(r"([。！？；;])\s*[。！？；;]+", r"\1", cleaned)
    if language == "zh":
        cleaned = re.sub(r"^(?:摘\s*要)\s*[：:]\s*", "", cleaned)
    else:
        cleaned = re.sub(r"^(?:abstract)\s*[：:]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip("；;，, ")


def _strip_abstract_noise(text: str) -> str:
    cleaned = text
    for pattern in _ABSTRACT_INLINE_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = sanitize_metadata_fragments(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" *")


def _is_abstract_stop_line(line: str) -> bool:
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in _ABSTRACT_STOP_PATTERNS)


def _is_abstract_noise_line(line: str) -> bool:
    lowered = line.lower()
    if is_noise_line(line):
        return True
    if any(keyword in lowered for keyword in ("本文系", "项目编号", "网络首发", "作者单位", "作者简介", "通信作者")):
        return True
    if re.fullmatch(r"\*?\s*[A-Za-z]{1,8}\d{3,}[A-Za-z0-9-]*\s*", line):
        return True
    if re.fullmatch(r"第?\s*\d+\s*页", line):
        return True
    if re.fullmatch(r"\d+\s*/\s*\d+", line):
        return True
    if re.search(r"(大学|学院|研究院|department|university|college)", line, re.IGNORECASE) and len(line) <= 48:
        return True
    return False


def _build_fallback_summary(raw_text: str) -> str:
    source_text = _drop_leading_title_block(raw_text)[:SUMMARY_SOURCE_TEXT_LIMIT]
    sentences = filter_noise_sentences(split_sentences(source_text))
    if not sentences:
        return "未能生成摘要，请检查 PDF 是否包含可提取文本。"

    top_sentences = _rank_sentences(sentences[:18], source_text)
    if top_sentences:
        return " ".join(top_sentences)
    return limit_sentences(sentences, SUMMARY_SENTENCE_MIN, SUMMARY_SENTENCE_MAX) or "未能生成摘要，请检查 PDF 是否包含可提取文本。"


def _rank_sentences(sentences: list[str], source_text: str) -> list[str]:
    english_frequency = Counter(extract_english_tokens(source_text))
    chinese_frequency = extract_chinese_phrases(source_text)

    scored_sentences: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        if is_noise_sentence(sentence):
            continue

        score = 0.0
        sentence_length = max(len(sentence), 1)
        english_terms = extract_english_tokens(sentence)
        chinese_terms = extract_chinese_phrases(sentence)

        score += sum(english_frequency.get(term, 0) for term in english_terms)
        score += sum(chinese_frequency.get(term, 0) for term in chinese_terms)

        if re.search(r"(本文|本研究|研究表明|结果显示|结果发现|提出|表明|this paper|this study|results?)", sentence, re.IGNORECASE):
            score += 3.0
        if re.search(r"(摘要|abstract|引言|introduction)", sentence, re.IGNORECASE):
            score += 1.0
        if 18 <= sentence_length <= 140:
            score += 1.5
        if re.search(r"\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}", sentence):
            score -= 2.0
        if re.search(r"https?://|www\.|doi", sentence, re.IGNORECASE):
            score -= 3.0

        scored_sentences.append((score / sentence_length, index, sentence))

    if not scored_sentences:
        return []

    selected = sorted(scored_sentences, key=lambda item: item[0], reverse=True)[:SUMMARY_SENTENCE_MAX]
    selected = sorted(selected, key=lambda item: item[1])
    top_sentences = [_strip_section_label(sentence) for _, _, sentence in selected if sentence]

    if len(top_sentences) < SUMMARY_SENTENCE_MIN:
        return [_strip_section_label(sentence) for sentence in sentences[:SUMMARY_SENTENCE_MIN]]
    return top_sentences


def _drop_leading_title_block(text: str) -> str:
    paragraphs = [normalize_whitespace(paragraph) for paragraph in re.split(r"\n{2,}", text) if normalize_whitespace(paragraph)]
    if not paragraphs:
        return text

    first_block = paragraphs[0]
    if (
        len(paragraphs) >= 2
        and len(first_block) <= 120
        and not re.search(r"[。！？!?；;:：.]", first_block)
        and not re.search(r"(摘要|abstract|关键词|keywords?)", first_block, re.IGNORECASE)
    ):
        return "\n\n".join(paragraphs[1:])
    return text


def _strip_section_label(sentence: str) -> str:
    return re.sub(r"^(摘要|abstract|引言|introduction)\s*[:：]\s*", "", sentence, flags=re.IGNORECASE).strip()
