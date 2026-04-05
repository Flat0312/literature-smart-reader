"""Reusable text normalization and sentence utilities."""

from __future__ import annotations

import re
from collections import Counter

from config.settings import (
    CHINESE_STOP_PHRASES,
    ENGLISH_STOPWORDS,
    LOW_INFORMATION_LINE_LENGTH,
    LOW_INFORMATION_NON_TEXT_RATIO,
    METADATA_FRAGMENT_PATTERNS,
    METADATA_LINE_KEYWORDS,
    METADATA_LINE_PATTERNS,
    SUMMARY_NOISE_KEYWORDS,
    TEXT_PREVIEW_PARAGRAPHS,
)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = normalize_whitespace(text)
    text = text.replace(" ：", "：").replace(" ,", ",").replace(" .", ".")
    return text.strip(" |_-")


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_whitespace(text).replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if not cleaned:
        return []
    raw_sentences = re.split(r"(?<=[。！？!?；;.!?])\s+|(?<=[。！？!?；;])", cleaned)
    sentences = [normalize_line(sentence) for sentence in raw_sentences]
    return [sentence for sentence in sentences if len(sentence) >= 10]


def limit_sentences(sentences: list[str], min_count: int, max_count: int) -> str:
    if not sentences:
        return ""
    target = max(min_count, min(len(sentences), max_count))
    return " ".join(sentences[:target]).strip()


def build_preview(text: str, limit: int) -> str:
    paragraphs = [
        normalize_whitespace(paragraph)
        for paragraph in re.split(r"\n{2,}", text)
        if normalize_whitespace(paragraph)
    ]
    preview_parts: list[str] = []
    total_length = 0

    for paragraph in paragraphs:
        if is_noise_line(paragraph):
            continue
        preview_parts.append(paragraph)
        total_length += len(paragraph)
        if total_length >= limit or len(preview_parts) >= TEXT_PREVIEW_PARAGRAPHS:
            break

    cleaned = "\n\n".join(preview_parts).strip() or normalize_whitespace(text)
    cleaned = reflow_text_for_display(cleaned)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def reflow_text_for_display(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        if not line:
            _flush_reflow_lines(paragraphs, current_lines)
            continue
        if _is_display_standalone_line(line):
            _flush_reflow_lines(paragraphs, current_lines)
            paragraphs.append(line)
            continue
        if not current_lines:
            current_lines.append(line)
            continue
        if _should_break_display_paragraph(current_lines[-1], line):
            _flush_reflow_lines(paragraphs, current_lines)
        current_lines.append(line)

    _flush_reflow_lines(paragraphs, current_lines)
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def extract_english_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
        if token.lower() not in ENGLISH_STOPWORDS
    ]


def extract_chinese_phrases(text: str, min_length: int = 2, max_length: int = 4) -> Counter[str]:
    phrase_counter: Counter[str] = Counter()
    for chunk in re.findall(r"[\u4e00-\u9fff]{4,}", text):
        for size in range(min_length, max_length + 1):
            for index in range(len(chunk) - size + 1):
                phrase = chunk[index:index + size]
                if phrase in CHINESE_STOP_PHRASES:
                    continue
                if phrase.startswith(("的", "了", "和")) or phrase.endswith(("的", "了", "和")):
                    continue
                phrase_counter[phrase] += 1
    return phrase_counter


def compact_list(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for item in items:
        normalized = normalize_line(item).strip("；;，,、 ")
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        compacted.append(normalized)
        if len(compacted) >= limit:
            break
    return compacted


def clean_extracted_text(text: str) -> str:
    cleaned_lines: list[str] = []
    blank_pending = False

    for raw_line in text.splitlines():
        line = sanitize_metadata_fragments(raw_line)
        if not line:
            if cleaned_lines:
                blank_pending = True
            continue
        if is_noise_line(line):
            continue
        if blank_pending and cleaned_lines:
            cleaned_lines.append("")
        cleaned_lines.append(line)
        blank_pending = False

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return normalize_whitespace(cleaned_text)


def is_noise_line(line: str) -> bool:
    normalized = sanitize_metadata_fragments(line)
    if not normalized:
        return True
    lowered = normalized.lower()
    preserve_mixed_content = _should_preserve_mixed_content_line(normalized)

    if any(keyword in lowered for keyword in _iter_casefold_keywords(METADATA_LINE_KEYWORDS)) and not preserve_mixed_content:
        return True
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in METADATA_LINE_PATTERNS) and not preserve_mixed_content:
        return True
    if _is_low_information_line(normalized):
        return True
    return False


def is_noise_sentence(sentence: str) -> bool:
    normalized = sanitize_metadata_fragments(sentence)
    if not normalized:
        return True
    lowered = normalized.lower()

    if any(keyword in lowered for keyword in _iter_casefold_keywords(SUMMARY_NOISE_KEYWORDS)):
        return True
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in METADATA_LINE_PATTERNS):
        return True
    if re.fullmatch(r"[\d\s\-—–:：./()]+", normalized):
        return True
    if _is_low_information_line(normalized):
        return True
    return False


def filter_noise_sentences(sentences: list[str]) -> list[str]:
    filtered_sentences: list[str] = []
    for sentence in sentences:
        normalized = sanitize_metadata_fragments(sentence)
        if not normalized or is_noise_sentence(normalized):
            continue
        filtered_sentences.append(normalized)
    return filtered_sentences


def sanitize_metadata_fragments(text: str) -> str:
    sanitized = normalize_line(text)
    for pattern in METADATA_FRAGMENT_PATTERNS:
        sanitized = re.sub(pattern, " ", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"[，,;；]\s*$", "", sanitized)
    return normalize_line(sanitized)


def _should_preserve_mixed_content_line(text: str) -> bool:
    if re.search(r"(?:摘\s*要|摘要|abstract|研究目的|研究方法|研究结论|结果|结论)\s*[：:]", text, re.IGNORECASE):
        return True

    zh_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    sentence_like = bool(re.search(r"[。！？；;]", text))
    return zh_count >= 24 and sentence_like


def _is_low_information_line(text: str) -> bool:
    if len(text) > LOW_INFORMATION_LINE_LENGTH:
        return False
    alpha_count = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", text))
    non_text_count = len(re.findall(r"[^A-Za-z\u4e00-\u9fff\s]", text))
    digit_count = len(re.findall(r"\d", text))
    total = max(len(text), 1)
    non_text_ratio = (non_text_count + digit_count) / total

    if alpha_count <= 4 and non_text_ratio >= LOW_INFORMATION_NON_TEXT_RATIO:
        return True
    if re.fullmatch(r"[A-Za-z0-9\-_.:/()]+", text):
        return True
    return False


def _iter_casefold_keywords(keywords: set[str]) -> list[str]:
    return [keyword.lower() for keyword in keywords]


def _flush_reflow_lines(paragraphs: list[str], current_lines: list[str]) -> None:
    if not current_lines:
        return
    paragraphs.append(_join_display_lines(current_lines))
    current_lines.clear()


def _join_display_lines(lines: list[str]) -> str:
    merged = ""
    for line in lines:
        if not merged:
            merged = line
            continue
        if _needs_display_space(merged[-1], line[0]):
            merged = f"{merged} {line}"
        else:
            merged = f"{merged}{line}"
    return normalize_whitespace(merged)


def _needs_display_space(left_char: str, right_char: str) -> bool:
    if not right_char.isascii():
        return False
    if left_char.isascii() and left_char.isalnum() and right_char.isalnum():
        return True
    return left_char in ".!?:;" and right_char.isalnum()


def _should_break_display_paragraph(previous_line: str, current_line: str) -> bool:
    if re.match(r"^(摘要|abstract|关键词|关键字|keywords?|引言|introduction|结论|参考文献|references?)", current_line, re.IGNORECASE):
        return True
    if re.match(r"^\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.)])", current_line):
        return True
    return False


def _is_display_standalone_line(line: str) -> bool:
    if re.match(r"^(摘要|abstract|关键词|关键字|keywords?|引言|introduction|结论|参考文献|references?)", line, re.IGNORECASE):
        return True
    if re.search(r"(大学|学院|研究院|department|university|college)", line, re.IGNORECASE):
        return len(line) <= 40
    if line.endswith(("：", ":")) and len(line) <= 20:
        return True
    if len(line) <= 12 and not re.search(r"[。！？!?；;:：]", line):
        return True
    return False
