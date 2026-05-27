"""PDF extraction service based on PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import fitz

from config.settings import MIN_TEXT_LENGTH
from utils.text_utils import clean_extracted_text, looks_like_repeated_garbled_text, normalize_line, normalize_whitespace

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
LOW_TEXT_PAGE_CHAR_THRESHOLD = 40
LOW_TEXT_DENSITY_RATIO = 0.35
BLOCK_COLUMN_MIN_COUNT = 4
FULL_WIDTH_BLOCK_RATIO = 0.72


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
class PdfPreflightInfo:
    page_count: int = 0
    extraction_strategy: str = ""
    quality_score: float = 0.0
    is_encrypted: bool = False
    is_probably_scanned: bool = False
    is_low_text_density: bool = False
    is_garbled_heavy: bool = False
    has_images: bool = False
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PdfExtractionAttempt:
    strategy: str
    page_texts: list[str]
    page_blocks: list[list[dict]]
    quality_score: float
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PdfExtractionResult:
    raw_text: str
    body_text: str
    pages: list[PdfPageDebug] = field(default_factory=list)
    body_start_page_index: int = 0
    preflight: PdfPreflightInfo = field(default_factory=PdfPreflightInfo)
    extraction_attempts_debug: list[dict[str, object]] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

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
    if getattr(document, "needs_pass", False):
        document.close()
        raise ValueError("PDF 文件已加密或需要密码，当前无法解析。")

    raw_text_payloads: list[tuple[str, list[dict]]] = []
    raw_block_payloads: list[tuple[str, list[dict]]] = []
    image_counts: list[int] = []
    try:
        for page_index, page in enumerate(document):
            page_text = normalize_whitespace(page.get_text("text"))
            page_dict = page.get_text("dict")
            page_blocks = page_dict.get("blocks", [])
            block_text = normalize_whitespace(_extract_ordered_block_text(page))
            raw_text_payloads.append((page_text, page_blocks))
            raw_block_payloads.append((block_text, page_blocks))
            image_counts.append(len(page.get_images(full=True)))
    finally:
        document.close()

    attempts = [
        _build_extraction_attempt("pymupdf_text", raw_text_payloads, image_counts=image_counts),
        _build_extraction_attempt("pymupdf_blocks", raw_block_payloads, image_counts=image_counts),
    ]
    selected_attempt = _select_best_extraction_attempt(attempts)
    preflight = _build_preflight_info(
        page_count=len(raw_text_payloads),
        selected_attempt=selected_attempt,
        attempts=attempts,
        image_counts=image_counts,
    )

    cleaned_page_texts = _strip_repeated_page_edge_noise(selected_attempt.page_texts)
    page_debug_items: list[PdfPageDebug] = []
    for page_index, (page_blocks, cleaned_page_text) in enumerate(zip(selected_attempt.page_blocks, cleaned_page_texts)):
        page_debug_items.append(_build_page_debug(page_index, cleaned_page_text, page_blocks))

    full_text = clean_extracted_text("\n\n".join(page.text for page in page_debug_items if page.text))
    if not full_text:
        raise ValueError(_build_no_text_error(preflight))

    body_start_page_index = _select_body_start_page(page_debug_items)
    body_text = clean_extracted_text(
        "\n\n".join(page.text for page in page_debug_items[body_start_page_index:] if page.text)
    )
    working_text = body_text or full_text

    if len(working_text) < MIN_TEXT_LENGTH:
        raise ValueError(_build_short_text_error(preflight, len(working_text)))

    return PdfExtractionResult(
        raw_text=full_text,
        body_text=working_text,
        pages=page_debug_items,
        body_start_page_index=body_start_page_index,
        preflight=preflight,
        extraction_attempts_debug=_summarize_attempts(attempts),
        parse_warnings=preflight.warnings,
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


def _extract_ordered_block_text(page) -> str:
    page_width = float(page.rect.width or 0)
    raw_blocks = []
    for block in page.get_text("blocks"):
        if len(block) < 7:
            continue
        x0, y0, x1, y1, text, block_number, block_type = block[:7]
        if block_type != 0:
            continue
        cleaned = normalize_whitespace(str(text or ""))
        if not cleaned:
            continue
        raw_blocks.append(
            {
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": cleaned,
                "block_number": int(block_number),
            }
        )
    if not raw_blocks:
        return ""

    ordered_blocks = _order_text_blocks(raw_blocks, page_width=page_width)
    return "\n".join(block["text"] for block in ordered_blocks if block.get("text"))


def _order_text_blocks(blocks: list[dict[str, object]], *, page_width: float) -> list[dict[str, object]]:
    if page_width <= 0 or len(blocks) < BLOCK_COLUMN_MIN_COUNT:
        return sorted(blocks, key=lambda item: (float(item["y0"]), float(item["x0"])))

    full_width_blocks: list[dict[str, object]] = []
    column_blocks: list[dict[str, object]] = []
    for block in blocks:
        width = float(block["x1"]) - float(block["x0"])
        if width >= page_width * FULL_WIDTH_BLOCK_RATIO:
            full_width_blocks.append(block)
        else:
            column_blocks.append(block)

    left_blocks = [block for block in column_blocks if _block_center(block) < page_width * 0.52]
    right_blocks = [block for block in column_blocks if _block_center(block) >= page_width * 0.52]
    if len(left_blocks) < 2 or len(right_blocks) < 2:
        return sorted(blocks, key=lambda item: (float(item["y0"]), float(item["x0"])))

    header_blocks = [
        block
        for block in full_width_blocks
        if float(block["y0"]) <= min(float(item["y0"]) for item in column_blocks)
    ]
    footer_blocks = [block for block in full_width_blocks if block not in header_blocks]
    return [
        *sorted(header_blocks, key=lambda item: (float(item["y0"]), float(item["x0"]))),
        *sorted(left_blocks, key=lambda item: (float(item["y0"]), float(item["x0"]))),
        *sorted(right_blocks, key=lambda item: (float(item["y0"]), float(item["x0"]))),
        *sorted(footer_blocks, key=lambda item: (float(item["y0"]), float(item["x0"]))),
    ]


def _block_center(block: dict[str, object]) -> float:
    return (float(block["x0"]) + float(block["x1"])) / 2


def _build_extraction_attempt(
    strategy: str,
    raw_page_payloads: list[tuple[str, list[dict]]],
    *,
    image_counts: list[int],
) -> PdfExtractionAttempt:
    page_texts = [normalize_whitespace(payload[0]) for payload in raw_page_payloads]
    page_blocks = [payload[1] for payload in raw_page_payloads]
    diagnostics = _build_attempt_diagnostics(page_texts, image_counts=image_counts)
    warnings = _build_attempt_warnings(diagnostics)
    quality_score = _score_extraction_attempt(page_texts, diagnostics)
    return PdfExtractionAttempt(
        strategy=strategy,
        page_texts=page_texts,
        page_blocks=page_blocks,
        quality_score=round(quality_score, 2),
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _select_best_extraction_attempt(attempts: list[PdfExtractionAttempt]) -> PdfExtractionAttempt:
    return max(attempts, key=lambda attempt: (attempt.quality_score, attempt.diagnostics.get("total_text_chars", 0)))


def _build_attempt_diagnostics(page_texts: list[str], *, image_counts: list[int]) -> dict[str, object]:
    total_text = "\n".join(page_texts)
    page_count = len(page_texts)
    page_lengths = [len(text) for text in page_texts]
    pages_with_text = sum(1 for length in page_lengths if length >= LOW_TEXT_PAGE_CHAR_THRESHOLD)
    low_text_density = page_count > 0 and pages_with_text / page_count < LOW_TEXT_DENSITY_RATIO
    image_page_count = sum(1 for count in image_counts if count > 0)
    garbled_line_count = _count_garbled_lines(total_text)
    text_line_count = max(len([line for line in total_text.splitlines() if normalize_line(line)]), 1)
    garbled_ratio = garbled_line_count / text_line_count
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", total_text))
    latin_chars = len(re.findall(r"[A-Za-z]", total_text))
    body_hits = _match_keywords(total_text, BODY_PAGE_KEYWORDS)
    cover_hits = _match_keywords(total_text, COVER_PAGE_KEYWORDS)
    return {
        "total_text_chars": len(total_text),
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "low_text_density": low_text_density,
        "image_page_count": image_page_count,
        "has_images": image_page_count > 0,
        "garbled_line_count": garbled_line_count,
        "garbled_line_ratio": round(garbled_ratio, 3),
        "is_garbled_heavy": garbled_ratio >= 0.22,
        "chinese_chars": chinese_chars,
        "latin_chars": latin_chars,
        "body_keyword_hits": body_hits,
        "cover_keyword_hits": cover_hits,
        "has_abstract_signal": bool(re.search(r"(摘\s*要|摘要|abstract)", total_text, re.IGNORECASE)),
        "has_keyword_signal": bool(re.search(r"(关键词|关键字|keywords?|key words)", total_text, re.IGNORECASE)),
        "is_probably_scanned": low_text_density and image_page_count >= max(1, int(page_count * 0.5)),
        "page_lengths": page_lengths[:20],
    }


def _score_extraction_attempt(page_texts: list[str], diagnostics: dict[str, object]) -> float:
    total_text = "\n".join(page_texts)
    score = 0.0
    text_chars = int(diagnostics.get("total_text_chars", 0))
    score += min(text_chars / 600, 10.0)
    score += min(int(diagnostics.get("pages_with_text", 0)) * 0.8, 5.0)
    score += len(diagnostics.get("body_keyword_hits", [])) * 1.4
    if diagnostics.get("has_abstract_signal"):
        score += 3.0
    if diagnostics.get("has_keyword_signal"):
        score += 2.2
    if re.search(r"(引言|绪论|introduction)", total_text, re.IGNORECASE):
        score += 1.4
    if diagnostics.get("low_text_density"):
        score -= 4.0
    if diagnostics.get("is_probably_scanned"):
        score -= 5.0
    if diagnostics.get("is_garbled_heavy"):
        score -= 4.0
    score -= len(diagnostics.get("cover_keyword_hits", [])) * 0.8
    return score


def _build_attempt_warnings(diagnostics: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    if diagnostics.get("is_probably_scanned"):
        warnings.append("PDF 疑似扫描版或图片型文档，需要 OCR 才能稳定解析。")
    elif diagnostics.get("low_text_density"):
        warnings.append("PDF 可复制文本密度偏低，部分页面可能需要 OCR 回退。")
    if diagnostics.get("is_garbled_heavy"):
        warnings.append("PDF 提取文本疑似存在较多乱码，字段结果需要结合原文复核。")
    if not diagnostics.get("has_abstract_signal"):
        warnings.append("未在提取文本中稳定发现摘要标签。")
    return warnings


def _build_preflight_info(
    *,
    page_count: int,
    selected_attempt: PdfExtractionAttempt,
    attempts: list[PdfExtractionAttempt],
    image_counts: list[int],
) -> PdfPreflightInfo:
    diagnostics = dict(selected_attempt.diagnostics)
    diagnostics["attempts"] = _summarize_attempts(attempts)
    diagnostics["image_counts"] = image_counts[:20]
    warnings = _dedupe_warnings(selected_attempt.warnings)
    return PdfPreflightInfo(
        page_count=page_count,
        extraction_strategy=selected_attempt.strategy,
        quality_score=selected_attempt.quality_score,
        is_probably_scanned=bool(selected_attempt.diagnostics.get("is_probably_scanned")),
        is_low_text_density=bool(selected_attempt.diagnostics.get("low_text_density")),
        is_garbled_heavy=bool(selected_attempt.diagnostics.get("is_garbled_heavy")),
        has_images=bool(selected_attempt.diagnostics.get("has_images")),
        warnings=warnings,
        diagnostics=diagnostics,
    )


def _summarize_attempts(attempts: list[PdfExtractionAttempt]) -> list[dict[str, object]]:
    return [
        {
            "strategy": attempt.strategy,
            "quality_score": attempt.quality_score,
            "warnings": attempt.warnings,
            "diagnostics": {
                key: value
                for key, value in attempt.diagnostics.items()
                if key not in {"page_lengths"}
            },
            "page_lengths": attempt.diagnostics.get("page_lengths", []),
        }
        for attempt in attempts
    ]


def _build_no_text_error(preflight: PdfPreflightInfo) -> str:
    if preflight.is_probably_scanned or preflight.has_images:
        return "PDF 疑似扫描版或图片型文档，当前未提取到可复制文本；请启用 OCR 或换用可复制文本版 PDF。"
    return "PDF 没有提取到可用文本。"


def _build_short_text_error(preflight: PdfPreflightInfo, text_length: int) -> str:
    if preflight.is_probably_scanned or preflight.is_low_text_density:
        return f"提取到的文本过少（{text_length} 字符），PDF 疑似扫描版或图片型文档；请启用 OCR 或换用可复制文本版 PDF。"
    if preflight.is_garbled_heavy:
        return f"提取到的文本过少（{text_length} 字符）且疑似乱码较多，请换用质量更高的 PDF 或检查字体嵌入。"
    return f"提取到的文本过少（{text_length} 字符），暂无法稳定解析。"


def _count_garbled_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        normalized = normalize_line(line)
        if normalized and looks_like_repeated_garbled_text(normalized):
            count += 1
    return count


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
    if looks_like_repeated_garbled_text(line):
        return True
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
    if looks_like_repeated_garbled_text(line):
        return True
    if line in repeated_edge_lines:
        return True
    if re.fullmatch(r"(?:第?\s*\d+\s*页|\d+\s*/\s*\d+|\d+)", line):
        return True
    if re.search(r"(网络首发|issn|cn\s*\d|doi|www\.|http)", line, re.IGNORECASE):
        return True
    return False
