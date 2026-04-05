"""PDF extraction service based on PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import fitz

from config.settings import MIN_TEXT_LENGTH
from utils.text_utils import clean_extracted_text, normalize_line, normalize_whitespace

BODY_PAGE_KEYWORDS = (
    "摘要",
    "关键词",
    "引言",
    "研究方法",
    "研究设计",
    "结论",
    "参考文献",
    "作者简介",
)
BODY_PAGE_WEIGHTS = {
    "摘要": 4.2,
    "关键词": 3.6,
    "引言": 3.2,
    "研究方法": 3.0,
    "研究设计": 3.0,
    "结论": 2.8,
    "参考文献": 2.0,
    "作者简介": 1.2,
}
COVER_PAGE_KEYWORDS = (
    "网络首发",
    "出版确认",
    "编辑部",
    "期刊",
    "中国学术期刊",
    "排版定稿",
    "录用定稿",
    "光盘版",
    "杂志社",
)
COVER_PAGE_WEIGHTS = {
    "网络首发": 5.0,
    "出版确认": 3.2,
    "编辑部": 2.6,
    "期刊": 1.4,
    "中国学术期刊": 3.6,
    "排版定稿": 3.2,
    "录用定稿": 3.2,
    "光盘版": 2.4,
    "杂志社": 2.8,
}
STRONG_BODY_KEYWORDS = {"摘要", "关键词", "引言", "研究方法", "研究设计", "结论", "参考文献"}
PAGE_PREVIEW_LENGTH = 300
BODY_OPENING_PAGE_COUNT = 2


@dataclass(slots=True)
class PdfPageDebug:
    page_index: int
    text: str
    preview: str
    page_score: float
    body_keywords_hit: list[str] = field(default_factory=list)
    cover_keywords_hit: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class PdfExtractionResult:
    raw_text: str
    body_text: str
    pages: list[PdfPageDebug] = field(default_factory=list)
    body_start_page_index: int = 0

    @property
    def first_page_blocks(self) -> list[dict]:
        return self.pages[0].blocks if self.pages else []

    @property
    def page_texts(self) -> list[str]:
        return [page.text for page in self.pages]

    @property
    def body_window_page_indices(self) -> list[int]:
        if not self.pages:
            return []
        start_index = min(self.body_start_page_index, len(self.pages) - 1)
        end_index = min(start_index + BODY_OPENING_PAGE_COUNT, len(self.pages))
        return [page.page_index for page in self.pages[start_index:end_index]]


def extract_pdf_context(file_bytes: bytes) -> PdfExtractionResult:
    if not file_bytes:
        raise ValueError("上传文件为空，请重新选择 PDF。")

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - library specific
        raise ValueError("PDF 文件无法读取，请确认文件格式正确。") from exc

    if document.page_count == 0:
        raise ValueError("PDF 没有可解析页面。")

    raw_page_payloads: list[tuple[str, list[dict]]] = []
    try:
        for page_index, page in enumerate(document):
            page_text = normalize_whitespace(page.get_text("text"))
            page_dict = page.get_text("dict")
            page_blocks = page_dict.get("blocks", [])
            raw_page_payloads.append((page_text, page_blocks))
    finally:
        document.close()

    cleaned_page_texts = _strip_repeated_page_edge_noise([payload[0] for payload in raw_page_payloads])
    page_debug_items: list[PdfPageDebug] = []
    for page_index, ((_, page_blocks), cleaned_page_text) in enumerate(zip(raw_page_payloads, cleaned_page_texts)):
        page_debug_items.append(_build_page_debug(page_index, cleaned_page_text, page_blocks))

    full_text = clean_extracted_text("\n\n".join(page.text for page in page_debug_items if page.text))
    if not full_text:
        raise ValueError("PDF 没有提取到可用文本。")

    body_start_page_index = _select_body_start_page(page_debug_items)
    body_text = clean_extracted_text(
        "\n\n".join(page.text for page in page_debug_items[body_start_page_index:] if page.text)
    )
    working_text = body_text or full_text

    if len(working_text) < MIN_TEXT_LENGTH:
        raise ValueError("提取到的文本过少。当前版本仅支持可复制文本的 PDF，扫描版文档可能无法识别。")

    return PdfExtractionResult(
        raw_text=full_text,
        body_text=working_text,
        pages=page_debug_items,
        body_start_page_index=body_start_page_index,
    )


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, list[dict]]:
    pdf_context = extract_pdf_context(file_bytes)
    return pdf_context.body_text, pdf_context.first_page_blocks


def flatten_first_page_lines(first_page_blocks: list[dict]) -> list[dict]:
    lines: list[dict] = []
    for block in first_page_blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            texts = [normalize_line(span.get("text", "")) for span in spans if span.get("text", "").strip()]
            texts = [text for text in texts if text]
            if not texts:
                continue
            y_positions = [span.get("bbox", [0, 0, 0, 0])[1] for span in spans]
            font_sizes = [span.get("size", 0.0) for span in spans]
            lines.append(
                {
                    "text": " ".join(texts).strip(),
                    "y0": min(y_positions) if y_positions else 0.0,
                    "font_size": max(font_sizes) if font_sizes else 0.0,
                }
            )
    return sorted(lines, key=lambda item: (item["y0"], -item["font_size"]))


def _build_page_debug(page_index: int, page_text: str, blocks: list[dict]) -> PdfPageDebug:
    normalized_text = normalize_whitespace(page_text)
    body_hits = _match_keywords(normalized_text, BODY_PAGE_KEYWORDS)
    cover_hits = _match_keywords(normalized_text, COVER_PAGE_KEYWORDS)
    page_score = _score_page(normalized_text, body_hits, cover_hits)

    return PdfPageDebug(
        page_index=page_index,
        text=normalized_text,
        preview=_build_page_preview(normalized_text),
        page_score=round(page_score, 2),
        body_keywords_hit=body_hits,
        cover_keywords_hit=cover_hits,
        blocks=blocks,
    )


def _score_page(text: str, body_hits: list[str], cover_hits: list[str]) -> float:
    if not text.strip():
        return -5.0

    score = 0.0
    for keyword in body_hits:
        score += BODY_PAGE_WEIGHTS.get(keyword, 1.0)
    for keyword in cover_hits:
        score -= COVER_PAGE_WEIGHTS.get(keyword, 1.0)

    text_length = len(text)
    if text_length >= 1000:
        score += 2.0
    elif text_length >= 500:
        score += 1.2
    elif text_length >= 220:
        score += 0.5
    else:
        score -= 0.8

    if re.search(r"(摘要|abstract)\s*[:：]?", text, re.IGNORECASE):
        score += 2.6
    if re.search(r"(关键词|关键字|keywords?)\s*[:：]?", text, re.IGNORECASE):
        score += 2.2
    if re.search(r"(摘要|abstract).{0,240}(关键词|关键字|keywords?)", text, re.IGNORECASE | re.DOTALL):
        score += 3.4
    if re.search(r"(引言|introduction)", text, re.IGNORECASE):
        score += 1.8
    if re.search(r"(参考文献|references?)", text, re.IGNORECASE):
        score += 1.4
    if re.search(r"(网络首发论文|中国学术期刊|出版确认|排版定稿|录用定稿)", text, re.IGNORECASE):
        score -= 3.6
    if re.search(r"(目录|contents|目\s*录)", text, re.IGNORECASE):
        score -= 1.6

    return score


def _select_body_start_page(page_debug_items: list[PdfPageDebug]) -> int:
    if not page_debug_items:
        return 0

    for index, page in enumerate(page_debug_items):
        if _is_cover_dominant(page):
            continue

        has_strong_body_hit = bool(set(page.body_keywords_hit) & STRONG_BODY_KEYWORDS)
        next_score = page_debug_items[index + 1].page_score if index + 1 < len(page_debug_items) else float("-inf")
        if has_strong_body_hit or page.page_score >= 4.0:
            return index
        if page.page_score >= 0.0 and next_score >= 4.0 and len(page.text) >= 40:
            return index

    positive_candidates = [
        page.page_index
        for page in page_debug_items
        if page.page_score > 0.0 and not _is_cover_dominant(page)
    ]
    if positive_candidates:
        return positive_candidates[0]

    return max(
        range(len(page_debug_items)),
        key=lambda index: page_debug_items[index].page_score + _positive_next_page_score(page_debug_items, index),
    )


def _is_cover_dominant(page: PdfPageDebug) -> bool:
    strong_body_hits = set(page.body_keywords_hit) & STRONG_BODY_KEYWORDS
    return bool(page.cover_keywords_hit) and not strong_body_hits and page.page_score < 4.0


def _positive_next_page_score(page_debug_items: list[PdfPageDebug], index: int) -> float:
    if index + 1 >= len(page_debug_items):
        return 0.0
    return max(page_debug_items[index + 1].page_score, 0.0) * 0.45


def _match_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def _build_page_preview(text: str) -> str:
    preview = normalize_line(text)
    if len(preview) <= PAGE_PREVIEW_LENGTH:
        return preview
    return f"{preview[:PAGE_PREVIEW_LENGTH].rstrip()}..."


def _strip_repeated_page_edge_noise(page_texts: list[str]) -> list[str]:
    repeated_edge_lines = _collect_repeated_edge_lines(page_texts)
    cleaned_texts: list[str] = []
    for page_text in page_texts:
        lines = _split_page_lines(page_text)
        filtered_lines: list[str] = []
        for index, line in enumerate(lines):
            is_edge_line = index < 3 or index >= max(len(lines) - 3, 0)
            if is_edge_line and _should_drop_page_edge_line(line, repeated_edge_lines):
                continue
            filtered_lines.append(line)
        cleaned_texts.append(normalize_whitespace("\n".join(filtered_lines)))
    return cleaned_texts


def _collect_repeated_edge_lines(page_texts: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for page_text in page_texts:
        lines = _split_page_lines(page_text)
        edge_candidates = lines[:3] + lines[-3:]
        seen_in_page: set[str] = set()
        for line in edge_candidates:
            if line in seen_in_page or not _looks_like_edge_noise_candidate(line):
                continue
            seen_in_page.add(line)
            counts[line] = counts.get(line, 0) + 1
    repeat_threshold = 2 if len(page_texts) <= 4 else 3
    return {line for line, count in counts.items() if count >= repeat_threshold}


def _split_page_lines(page_text: str) -> list[str]:
    return [normalize_line(line) for line in page_text.splitlines() if normalize_line(line)]


def _looks_like_edge_noise_candidate(line: str) -> bool:
    if not line:
        return False
    if re.search(r"(摘要|关键词|关键字|abstract|keywords?|引言|绪论|结论|参考文献)", line, re.IGNORECASE):
        return False
    if len(line) <= 30:
        return True
    if re.search(r"(网络首发|issn|cn\s*\d|doi|www\.|http|学报|期刊|journal|vol\.?|no\.?)", line, re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:第?\s*\d+\s*页|\d+\s*/\s*\d+|\d+)", line):
        return True
    return False


def _should_drop_page_edge_line(line: str, repeated_edge_lines: set[str]) -> bool:
    if not line:
        return False
    if line in repeated_edge_lines:
        return True
    if re.fullmatch(r"(?:第?\s*\d+\s*页|\d+\s*/\s*\d+|\d+)", line):
        return True
    if re.search(r"(网络首发|issn|cn\s*\d|doi|www\.|http)", line, re.IGNORECASE):
        return True
    return False
