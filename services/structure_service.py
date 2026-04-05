"""Rule-based candidate collection for structured paper insights."""

from __future__ import annotations

import re

from services.structured_rewrite_service import (
    StructuredFieldCandidate,
    StructuredRewriteRequest,
    build_structured_rewrite_request,
    filter_title_like_sentences,
    is_title_like_text,
    rewrite_structured_fields,
)
from utils.text_utils import (
    clean_extracted_text,
    filter_noise_sentences,
    normalize_line,
    normalize_whitespace,
    sanitize_metadata_fragments,
    split_sentences,
)

SECTION_LABELS = {
    "abstract": ["摘 要", "摘要", "abstract"],
    "introduction": ["引言", "绪论", "问题提出", "introduction"],
    "research_question": ["研究问题", "研究目的", "研究目标", "问题界定", "research question", "research objective", "objective"],
    "research_method": ["研究方法", "研究设计", "研究思路", "研究路径", "research method", "methods", "methodology", "research design"],
    "core_conclusion": ["结论", "研究结论", "结语", "研究发现", "conclusion", "conclusions", "findings"],
    "results": ["结果分析", "结果讨论", "discussion", "results"],
    "keywords": ["关键词", "关键字", "keywords", "key words"],
    "references": ["参考文献", "references", "致谢", "acknowledgements"],
}

ABSTRACT_LABEL_GROUPS = {
    "research_question": ["目的/意义", "目的与意义", "研究目的", "目的", "意义"],
    "research_method": ["方法/过程", "方法与过程", "研究方法", "方法", "过程"],
    "core_conclusion": ["结果/结论", "结果与结论", "研究结论", "结果", "结论"],
}

FIELD_RULES = {
    "research_question": {
        "preferred_sections": ["abstract", "research_question", "research_method", "core_conclusion", "introduction_tail"],
        "cues": [
            r"本文旨在", r"本文关注", r"本文研究", r"本研究聚焦", r"本研究探讨", r"目的在于",
            r"本研究旨在", r"针对", r"围绕", r"聚焦于", r"以.+为例", r"从.+视角",
            r"aims? to", r"seeks? to", r"investigates?", r"examines?", r"focuses? on",
        ],
        "exclude": [
            r"采用", r"通过", r"使用", r"问卷", r"访谈", r"实验", r"回归", r"结果表明", r"研究发现",
            r"survey", r"interview", r"experiment", r"regression", r"results? show",
        ],
    },
    "research_method": {
        "preferred_sections": ["abstract", "research_method", "introduction_tail"],
        "cues": [
            r"采用", r"基于", r"构建", r"设计", r"结合", r"运用", r"通过",
            r"问卷", r"访谈", r"实验", r"案例", r"实证", r"回归", r"文本分析",
            r"\buses?\b", r"\badopts?\b", r"\bemploys?\b", r"\bapplies?\b",
        ],
        "exclude": [
            r"本文旨在", r"本文关注", r"本文研究", r"本研究聚焦", r"本研究探讨", r"目的在于",
            r"结果表明", r"研究发现", r"研究表明", r"conclusion", r"results? show",
        ],
    },
    "core_conclusion": {
        "preferred_sections": ["abstract", "core_conclusion", "results", "introduction_tail"],
        "cues": [
            r"研究发现", r"结果表明", r"结论认为", r"研究表明", r"可见", r"说明了", r"证明了",
            r"说明", r"表明", r"发现", r"results? show", r"findings? indicate", r"conclude",
        ],
        "exclude": [
            r"本文旨在", r"本文关注", r"本文研究", r"本研究聚焦", r"本研究探讨", r"目的在于",
            r"采用", r"通过", r"使用", r"问卷", r"访谈", r"实验", r"survey", r"interview",
        ],
    },
}

STOP_SECTIONS = {"research_question", "research_method", "core_conclusion", "results", "references", "keywords"}
PREDICATE_SIGNALS = [
    r"本文旨在", r"本文关注", r"本文研究", r"本研究聚焦", r"本研究探讨", r"目的在于", r"本研究旨在",
    r"采用", r"基于", r"构建", r"设计", r"结合", r"运用", r"通过", r"使用",
    r"研究发现", r"结果表明", r"结论认为", r"研究表明", r"可见", r"说明了", r"证明了",
    r"说明", r"表明", r"发现", r"认为", r"揭示", r"提出",
]
MAX_FILTER_DEBUG_ITEMS = 10
INTRODUCTION_TAIL_SENTENCES = 2
PRIORITY_PAGE_COUNT = 2


def collect_structured_candidates(
    raw_text: str,
    title: str = "",
    priority_text: str = "",
    chinese_abstract: str = "",
) -> StructuredRewriteRequest:
    working_text = clean_extracted_text(raw_text)
    prioritized_text = clean_extracted_text(priority_text) if priority_text else ""
    normalized_chinese_abstract = clean_extracted_text(chinese_abstract) if chinese_abstract else ""
    labeled_candidates, label_debug = _extract_labeled_abstract_candidates(normalized_chinese_abstract, title=title)
    if labeled_candidates:
        rule_candidate_char_count = sum(
            len(candidate.text.strip())
            for candidate in labeled_candidates.values()
            if candidate.text.strip()
        )
        return build_structured_rewrite_request(
            raw_text=working_text,
            title=title,
            candidates={
                field_name: labeled_candidates.get(
                    field_name,
                    StructuredFieldCandidate(field_name=field_name, text="", source_kind="empty", source_hint="", score=0.0),
                )
                for field_name in ("research_question", "research_method", "core_conclusion")
            },
            preferred_backend="local_rule",
            debug_info={
                "detected_title": title,
                "field_debug": label_debug,
                "priority_preview": normalized_chinese_abstract[:300],
                "priority_pages": 0,
                "abstract_priority_found": True,
                "abstract_priority_only": True,
                "abstract_fallback_text": normalized_chinese_abstract,
                "abstract_fallback_available": bool(normalized_chinese_abstract.strip()),
                "abstract_preview": normalized_chinese_abstract[:300],
                "keywords_preview": "",
                "candidate_source_strategy": "cn_abstract_labels",
                "rule_candidate_char_count": rule_candidate_char_count,
                "explicit_abstract_labels_found": True,
                "explicit_label_fields": [field_name for field_name, candidate in labeled_candidates.items() if candidate.text.strip()],
            },
        )

    paragraphs = _split_paragraphs(working_text)
    sections = _collect_sections(paragraphs)
    priority_sections = _collect_priority_sections(normalized_chinese_abstract or prioritized_text)
    sections = _merge_sections(sections, priority_sections)
    abstract_priority_only = bool(priority_sections.get("abstract"))
    if normalized_chinese_abstract:
        abstract_fallback_text = normalized_chinese_abstract
    elif priority_sections.get("abstract"):
        abstract_fallback_text = priority_sections.get("abstract", [""])[0]
    else:
        abstract_fallback_text = _extract_abstract_block(working_text)
    abstract_preview = abstract_fallback_text
    keywords_preview = priority_sections.get("keywords", [""])[0] if priority_sections.get("keywords") else ""

    candidates: dict[str, StructuredFieldCandidate] = {}
    field_debug: dict[str, object] = {}
    for field_name in ("research_question", "research_method", "core_conclusion"):
        candidate, debug_item = _extract_field(
            field_name,
            sections,
            title=title,
            abstract_priority_only=abstract_priority_only,
        )
        candidates[field_name] = candidate
        field_debug[field_name] = debug_item
    rule_candidate_char_count = sum(len(candidate.text.strip()) for candidate in candidates.values() if candidate.text.strip())

    return build_structured_rewrite_request(
        raw_text=working_text,
        title=title,
        candidates=candidates,
        preferred_backend="relay_llm",
        debug_info={
            "detected_title": title,
            "field_debug": field_debug,
            "priority_preview": prioritized_text[:300],
            "priority_pages": PRIORITY_PAGE_COUNT if prioritized_text else 0,
            "abstract_priority_found": abstract_priority_only,
            "abstract_priority_only": abstract_priority_only,
            "abstract_fallback_text": abstract_fallback_text,
            "abstract_fallback_available": bool(abstract_fallback_text.strip()),
            "abstract_preview": abstract_preview[:300],
            "keywords_preview": keywords_preview[:200],
            "candidate_source_strategy": "abstract_only" if abstract_priority_only else "section_fallback",
            "rule_candidate_char_count": rule_candidate_char_count,
            "explicit_abstract_labels_found": False,
            "explicit_label_fields": [],
        },
    )


def extract_structured_fields(raw_text: str, title: str = "") -> dict[str, str]:
    rewrite_request = collect_structured_candidates(raw_text, title=title)
    return rewrite_structured_fields(rewrite_request).as_dict()


def _extract_labeled_abstract_candidates(
    chinese_abstract: str,
    *,
    title: str = "",
) -> tuple[dict[str, StructuredFieldCandidate], dict[str, object]]:
    if not chinese_abstract.strip():
        return {}, {}

    matches = _find_abstract_label_matches(chinese_abstract)
    if not matches:
        return {}, {}

    candidates: dict[str, StructuredFieldCandidate] = {}
    debug: dict[str, object] = {}
    for index, match in enumerate(matches):
        field_name = match["field_name"]
        if field_name in candidates:
            continue
        next_start = matches[index + 1]["start"] if index + 1 < len(matches) else len(chinese_abstract)
        raw_value = chinese_abstract[match["end"]:next_start].strip(" ：:；;，,\n")
        candidate_text = _finalize_labeled_abstract_value(raw_value, field_name, title=title)
        if not candidate_text:
            continue
        candidates[field_name] = StructuredFieldCandidate(
            field_name=field_name,
            text=candidate_text,
            source_kind="abstract_label",
            source_hint=f"chinese_abstract:{match['label']}",
            score=10.0,
        )
        debug[field_name] = {
            "preferred_sections": ["chinese_abstract_label"],
            "selected_candidate": candidate_text,
            "selected_source": f"chinese_abstract:{match['label']}",
            "selected_score": 10.0,
            "filtered_out": [],
            "low_quality_reason": "",
            "label": match["label"],
            "abstract_priority_only": True,
        }
    return candidates, debug


def _find_abstract_label_matches(text: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for field_name, labels in ABSTRACT_LABEL_GROUPS.items():
        for label in labels:
            pattern = rf"(?:(?<=^)|(?<=[；;。.!?\n:：]))\s*(?:【|\[|\(|（)?\s*{re.escape(label)}\s*(?:】|\]|\)|）)?\s*[：:]?"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(
                    {
                        "field_name": field_name,
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

    matches.sort(key=lambda item: (int(item["start"]), -len(str(item["label"]))))
    deduped: list[dict[str, object]] = []
    occupied_ranges: list[tuple[int, int]] = []
    for item in matches:
        start = int(item["start"])
        end = int(item["end"])
        if any(not (end <= occupied_start or start >= occupied_end) for occupied_start, occupied_end in occupied_ranges):
            continue
        occupied_ranges.append((start, end))
        deduped.append(item)
    return deduped


def _finalize_labeled_abstract_value(text: str, field_name: str, *, title: str = "") -> str:
    cleaned = _clean_candidate_text(text)
    sentences = _prepare_sentences(cleaned)
    sentences = [sentence for sentence in sentences if not is_title_like_text(sentence, title)]
    if sentences:
        return _finalize_text(" ".join(sentences[:2]))
    return _finalize_text(cleaned)


def _extract_field(
    field_name: str,
    sections: dict[str, list[str]],
    title: str = "",
    abstract_priority_only: bool = False,
) -> tuple[StructuredFieldCandidate, dict[str, object]]:
    rules = FIELD_RULES[field_name]
    preferred_sections = ["abstract"] if abstract_priority_only else list(rules["preferred_sections"])
    debug_item = {
        "preferred_sections": preferred_sections,
        "selected_candidate": "",
        "selected_source": "",
        "filtered_out": [],
        "low_quality_reason": "",
        "abstract_priority_only": abstract_priority_only,
    }

    for section_name in preferred_sections:
        section_candidates: list[tuple[float, str]] = []
        for block in sections.get(section_name, []):
            candidate, score, rejected = _extract_from_block(block, field_name, section_name, title=title)
            debug_item["filtered_out"].extend(rejected)
            if candidate:
                section_candidates.append((score, candidate))
        if section_candidates:
            best_score, best_candidate = max(section_candidates, key=lambda item: item[0])
            debug_item["selected_candidate"] = best_candidate
            debug_item["selected_source"] = section_name
            debug_item["selected_score"] = round(best_score, 2)
            return (
                StructuredFieldCandidate(
                    field_name=field_name,
                    text=best_candidate,
                    source_kind="section",
                    source_hint=section_name,
                    score=best_score,
                ),
                debug_item,
            )

    debug_item["low_quality_reason"] = (
        "摘要区已定位，但未在摘要中找到满足线索词且通过标题过滤的完整候选句。"
        if abstract_priority_only
        else "未在优先章节中找到满足线索词且通过标题过滤的完整候选句。"
    )
    return (
        StructuredFieldCandidate(field_name=field_name, text="", source_kind="empty", source_hint="", score=0.0),
        debug_item,
    )


def _extract_from_block(
    text: str,
    field_name: str,
    section_name: str,
    title: str = "",
) -> tuple[str, float, list[dict[str, str]]]:
    sentences = _prepare_sentences(text)
    if not sentences:
        return "", 0.0, []
    if section_name == "introduction_tail" and len(sentences) > INTRODUCTION_TAIL_SENTENCES:
        sentences = sentences[-INTRODUCTION_TAIL_SENTENCES:]

    scored, rejected = _score_sentences(sentences, field_name, section_name=section_name, title=title)
    if not scored:
        return "", 0.0, rejected

    best_score, best_index = max(scored, key=lambda item: item[0])
    selected = [sentences[best_index]]
    if best_index + 1 < len(sentences):
        next_sentence = sentences[best_index + 1]
        next_score = next((score for score, idx in scored if idx == best_index + 1), 0.0)
        if next_score >= max(1.2, best_score - 1.6):
            selected.append(next_sentence)
    return _finalize_text(" ".join(selected)), best_score, rejected


def _score_sentences(
    sentences: list[str],
    field_name: str,
    section_name: str,
    title: str = "",
) -> tuple[list[tuple[float, int]], list[dict[str, str]]]:
    rules = FIELD_RULES[field_name]
    scored: list[tuple[float, int]] = []
    rejected: list[dict[str, str]] = []

    for index, sentence in enumerate(sentences):
        normalized = _strip_leading_label(sentence)
        filter_reason = _get_filter_reason(normalized, field_name, title)
        if filter_reason:
            _append_filter_reason(rejected, section_name, normalized, filter_reason)
            continue
        if _looks_like_other_field(normalized, field_name):
            _append_filter_reason(rejected, section_name, normalized, "更像其他结构化字段。")
            continue
        if not _matches_required_cue(normalized, field_name):
            _append_filter_reason(rejected, section_name, normalized, "未命中该字段的线索词。")
            continue

        score = 0.0
        for pattern in rules["cues"]:
            if re.search(pattern, normalized, re.IGNORECASE):
                score += 2.4
        for pattern in rules["exclude"]:
            if re.search(pattern, normalized, re.IGNORECASE):
                score -= 1.5

        if section_name == "abstract":
            score += 1.8
        elif section_name == "research_method":
            score += 1.5
        elif section_name in {"core_conclusion", "results"}:
            score += 1.5
        elif section_name == "introduction_tail":
            score += 0.4

        if re.search(r"(本文|本研究|this paper|this study)", normalized, re.IGNORECASE):
            score += 0.8
        if 18 <= len(normalized) <= 160:
            score += 0.7

        if score > 0.8:
            scored.append((score, index))
        else:
            _append_filter_reason(rejected, section_name, normalized, "字段得分不足。")

    return scored, rejected


def _collect_sections(paragraphs: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in SECTION_LABELS}
    sections["introduction_tail"] = []
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]
        matched_name, content = _match_section(paragraph)
        if not matched_name:
            if index <= 2 and re.search(r"(本文|本研究|this paper|this study)", paragraph, re.IGNORECASE):
                block = _finalize_text(paragraph)
                if block:
                    sections["abstract"].append(block)
            index += 1
            continue

        collected = [content] if content else []
        lookahead = index + 1
        while lookahead < len(paragraphs):
            next_paragraph = paragraphs[lookahead]
            next_name, _ = _match_section(next_paragraph)
            if next_name in STOP_SECTIONS:
                break
            collected.append(next_paragraph)
            if len(" ".join(collected)) >= 420:
                break
            lookahead += 1

        block = _finalize_text(" ".join(collected))
        if block:
            sections[matched_name].append(block)
            if matched_name == "introduction":
                tail_block = _tail_sentences(block, INTRODUCTION_TAIL_SENTENCES)
                if tail_block:
                    sections["introduction_tail"].append(tail_block)
        index = lookahead

    return sections


def _collect_priority_sections(priority_text: str) -> dict[str, list[str]]:
    empty_sections: dict[str, list[str]] = {name: [] for name in SECTION_LABELS}
    empty_sections["introduction_tail"] = []
    if not priority_text.strip():
        return empty_sections

    priority_sections = _collect_sections(_split_paragraphs(priority_text))
    abstract_block = _extract_abstract_block(priority_text)
    if abstract_block:
        priority_sections["abstract"] = _prepend_unique_blocks(priority_sections["abstract"], [abstract_block])
    keywords_block = _extract_keywords_block(priority_text)
    if keywords_block:
        priority_sections["keywords"] = _prepend_unique_blocks(priority_sections["keywords"], [keywords_block])
    return priority_sections


def _extract_abstract_block(text: str) -> str:
    if not text.strip():
        return ""

    patterns = [
        r"(?:摘\s*要|摘要)\s*[：:]?\s*(.+?)(?=(?:关键词|关键字|英文摘要|Abstract|abstract|引言|绪论|问题提出|研究方法|研究设计|结论|参考文献|一、|1[.、]|第一章|$))",
        r"(?:abstract)\s*[：:]?\s*(.+?)(?=(?:keywords?|index terms?|introduction|method(?:s|ology)?|conclusion|references|1\.|$))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        block = normalize_whitespace(_clean_candidate_text(match.group(1)))
        if block:
            return block
    return ""


def _extract_keywords_block(text: str) -> str:
    if not text.strip():
        return ""

    patterns = [
        r"(?:关键词|关键字)\s*[：:]\s*(.+)",
        r"(?:keywords?|key words)\s*[：:]\s*(.+)",
    ]
    text_window = text[:2200]
    for pattern in patterns:
        match = re.search(pattern, text_window, re.IGNORECASE)
        if not match:
            continue
        line = normalize_whitespace(match.group(1).splitlines()[0])
        if line:
            return line
    return ""


def _merge_sections(
    base_sections: dict[str, list[str]],
    priority_sections: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {
        section_name: list(blocks)
        for section_name, blocks in base_sections.items()
    }
    for section_name, priority_blocks in priority_sections.items():
        if not priority_blocks:
            continue
        merged[section_name] = _prepend_unique_blocks(merged.get(section_name, []), priority_blocks)
    return merged


def _prepend_unique_blocks(existing_blocks: list[str], incoming_blocks: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for block in incoming_blocks + existing_blocks:
        normalized = normalize_line(block)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(block)
    return merged


def _match_section(paragraph: str) -> tuple[str | None, str]:
    for section_name, labels in SECTION_LABELS.items():
        pattern = _build_heading_pattern(labels)
        if re.match(pattern, paragraph, re.IGNORECASE):
            content = re.sub(pattern, "", paragraph, count=1, flags=re.IGNORECASE).strip("：: ")
            return section_name, content
    return None, ""


def _prepare_sentences(text: str) -> list[str]:
    cleaned = _clean_candidate_text(text)
    sentences = split_sentences(cleaned)
    sentences = [_strip_leading_label(sentence) for sentence in sentences]
    sentences = filter_noise_sentences(sentences)
    return [sentence for sentence in sentences if _is_informative_sentence(sentence)]


def _clean_candidate_text(text: str) -> str:
    cleaned = sanitize_metadata_fragments(text)
    cleaned = re.sub(r"\[\d+(?:\s*[-,]\s*\d+)*\]|［\d+(?:\s*[-,]\s*\d+)*］", " ", cleaned)
    cleaned = re.sub(r"(?<!\w)[(（]\d+(?:\s*[-,]\s*\d+)*[)）]", " ", cleaned)
    cleaned = re.sub(r"[（(][^()（）]{0,40}\d{4}[a-z]?[^()（）]{0,40}[)）]", " ", cleaned)
    cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return normalize_whitespace(cleaned)


def _finalize_text(text: str) -> str:
    cleaned = _clean_candidate_text(text)
    cleaned = _strip_leading_label(cleaned)
    cleaned = re.sub(r"(见图\d+|见表\d+|如图\d+|如表\d+)$", "", cleaned)
    cleaned = normalize_whitespace(cleaned)
    return cleaned if _is_informative_sentence(cleaned) else ""


def _tail_sentences(text: str, count: int) -> str:
    sentences = _prepare_sentences(text)
    if not sentences:
        return ""
    return " ".join(sentences[-count:])


def _split_paragraphs(text: str) -> list[str]:
    heading_pattern = _build_heading_pattern(_all_labels(), anchored=False)
    normalized_text = normalize_whitespace(text)
    normalized_text = re.sub(rf"\n(?={heading_pattern})", "\n\n", normalized_text, flags=re.IGNORECASE)

    paragraphs: list[str] = []
    for block in re.split(r"\n{2,}", normalized_text):
        block = normalize_whitespace(block)
        if not block:
            continue
        lines = [normalize_whitespace(line) for line in block.splitlines() if normalize_whitespace(line)]
        if len(lines) <= 1:
            paragraphs.append(block)
            continue
        if _should_split_block_lines(lines):
            paragraphs.extend(lines)
        else:
            paragraphs.append(" ".join(lines))
    if not paragraphs:
        return []

    first_block = paragraphs[0]
    if (
        len(paragraphs) >= 2
        and len(first_block) <= 120
        and not re.search(r"[。！？!?；;:：.]", first_block)
        and not re.search(r"(摘要|abstract|关键词|keywords?)", first_block, re.IGNORECASE)
    ):
        return paragraphs[1:]
    return paragraphs


def _should_split_block_lines(lines: list[str]) -> bool:
    if len(lines) >= 3:
        return True
    if any(re.match(_build_heading_pattern(_all_labels(), anchored=True), line, re.IGNORECASE) for line in lines):
        return True
    if any(re.match(r"^\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.)]?|\([一二三四五六七八九十]\))", line) for line in lines):
        return True
    average_length = sum(len(line) for line in lines) / max(len(lines), 1)
    return average_length <= 180


def _is_informative_sentence(text: str) -> bool:
    normalized = normalize_line(text)
    if len(normalized) < 16:
        return False
    if len(normalized) > 180:
        return False
    if re.fullmatch(r"[\d\s\-—–:：./()]+", normalized):
        return False
    if re.search(r"(参考文献|references?|available online|doi|基金项目|项目编号)", normalized, re.IGNORECASE):
        return False
    return True


def _matches_required_cue(sentence: str, field_name: str) -> bool:
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in FIELD_RULES[field_name]["cues"])


def _looks_like_other_field(sentence: str, field_name: str) -> bool:
    current_matches = sum(
        1 for pattern in FIELD_RULES[field_name]["cues"] if re.search(pattern, sentence, re.IGNORECASE)
    )
    other_fields = [name for name in FIELD_RULES if name != field_name]
    for other_field in other_fields:
        matches = sum(1 for pattern in FIELD_RULES[other_field]["cues"] if re.search(pattern, sentence, re.IGNORECASE))
        if current_matches == 0 and matches >= 2:
            return True
    return False


def _get_filter_reason(sentence: str, field_name: str, title: str) -> str:
    normalized = normalize_line(sentence)
    if not normalized:
        return "句子为空。"
    if not _is_informative_sentence(normalized):
        return "过短片段或无完整句式。"
    if title and not filter_title_like_sentences(normalized, title).strip():
        return "与标题完全相同或高度相似。"
    if _looks_like_noun_phrase(normalized, field_name):
        return "更像标题或名词短语，缺少完整谓语结构。"
    return ""


def _looks_like_noun_phrase(sentence: str, field_name: str) -> bool:
    if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in PREDICATE_SIGNALS):
        return False
    if FIELD_RULES[field_name]["preferred_sections"] and _matches_required_cue(sentence, field_name):
        return False
    return len(sentence) <= 34


def _append_filter_reason(
    rejected: list[dict[str, str]],
    section_name: str,
    sentence: str,
    reason: str,
) -> None:
    if len(rejected) >= MAX_FILTER_DEBUG_ITEMS:
        return
    rejected.append(
        {
            "section": section_name,
            "text": sentence,
            "reason": reason,
        }
    )


def _strip_leading_label(text: str) -> str:
    stripped = re.sub(
        r"^(摘要|abstract|引言|introduction|研究问题|research question|研究方法|research method|结论|conclusion|结果|results?)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"^[，,；;：:]\s*", "", stripped)


def _build_heading_pattern(labels: list[str], anchored: bool = True) -> str:
    escaped_labels = "|".join(re.escape(label) for label in labels)
    prefix = "^" if anchored else ""
    return rf"{prefix}(?:\d+[.、]?\s*|[一二三四五六七八九十]+[、.]?\s*|\([一二三四五六七八九十]\)\s*)?(?:{escaped_labels})(?:\s*[:：]?\s*)"


def _all_labels() -> list[str]:
    labels: list[str] = []
    for section_labels in SECTION_LABELS.values():
        labels.extend(section_labels)
    return labels
