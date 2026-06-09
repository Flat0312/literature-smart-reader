"""Result page rendering — wide layout with dominant center reading column."""

from __future__ import annotations

from html import escape
import os
from textwrap import dedent

import streamlit as st

from config.settings import PAGE_HOME, PAGE_UPLOAD
from models.paper_result import AUTHORS_DISPLAY_FALLBACK, KEYWORDS_DISPLAY_FALLBACK, SUMMARY_DISPLAY_FALLBACK
from services.export_service import build_export_package
from utils.session import clear_analysis_result, get_analysis_result, reset_upload_flow, set_current_page
from utils.text_utils import reflow_text_for_display


def _build_result_package_html(result) -> str:
    structured_count = result.structured_field_count()
    items = [
        ("icon-coral", "📄", "论文速读卡", "标题、摘要、关键词与基础信息"),
        ("icon-blue", "🔍", "结构化解读", f"研究问题/方法/结论 已提取 {structured_count} 项"),
        ("icon-purple", "📝", "三类作业提纲", "课程汇报 / 课程论文 / 文献综述"),
        ("icon-green", "⬇️", "可导出成果", "支持 .md / .txt 导出"),
    ]
    cards = []
    for color_cls, icon, title, desc in items:
        cards.append(
            f'<div class="rs-package-item">'
            f'<span class="rs-package-icon {color_cls}">{icon}</span>'
            f'<div><strong>{escape(title)}</strong><p>{escape(desc)}</p></div>'
            f'</div>'
        )
    return _html_block(
        f"""
        <section class="rs-result-package topbar-green">
          <div class="rs-result-package__head">
            <h2>📦 学习成果包</h2>
          </div>
          <div class="rs-package-grid">{"".join(cards)}</div>
        </section>
        """
    )


def _build_structure_map_html(result) -> str:
    rq = result.structured_field_text("research_question", "")
    rm = result.structured_field_text("research_method", "")
    cc = result.structured_field_text("core_conclusion", "")
    innovations = result.innovation_items()
    limitations = result.limitation_items()
    use_hint = "可直接用于课程汇报、论文写作或文献综述"

    def _short(text: str, max_len: int = 60) -> str:
        text = (text or "").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"

    nodes = [
        ("❓", "研究问题", _short(rq) or "待识别"),
        ("🛠️", "研究方法", _short(rm) or "待识别"),
        ("🎯", "核心结论", _short(cc) or "待识别"),
        ("💡", "创新点", _short(innovations[0]) if innovations else "待识别"),
        ("⚠️", "不足之处", _short(limitations[0]) if limitations else "待识别"),
        ("✏️", "写作用途", use_hint),
    ]

    parts = []
    for i, (icon, label, text) in enumerate(nodes):
        parts.append(
            f'<div class="rs-map-node">'
            f'<span class="rs-map-node__icon">{icon}</span>'
            f'<span class="rs-map-node__label">{escape(label)}</span>'
            f'<span class="rs-map-node__text">{escape(text)}</span>'
            f'</div>'
        )
        if i < len(nodes) - 1:
            parts.append('<span class="rs-map-arrow">→</span>')

    return _html_block(
        f"""
        <section class="rs-structure-map topbar-indigo">
          <div class="rs-structure-map__head">
            <h2>🗺️ 论文结构地图</h2>
          </div>
          <div class="rs-map-flow">{"".join(parts)}</div>
        </section>
        """
    )


def _build_credibility_card_html(result) -> str:
    structured_debug = result.structured_debug if isinstance(result.structured_debug, dict) else {}
    summary_debug = structured_debug.get("summary_debug", {}) if isinstance(structured_debug.get("summary_debug"), dict) else {}
    final_result = structured_debug.get("final_result", {}) if isinstance(structured_debug.get("final_result"), dict) else {}

    summary_language = result.primary_summary_language()
    keyword_source = str(summary_debug.get("keyword_source", "")).strip()
    backend = str(final_result.get("backend", "")).strip()
    warnings = result.warning_items()

    summary_label = {"zh": "中文摘要原文", "en": "英文 Abstract 回退"}.get(summary_language, "正文 fallback 归纳")
    kw_label = _keyword_source_label(keyword_source) if keyword_source else "规则提取"
    backend_label = _fmt_backend(backend) if backend else "规则提取"

    cred_items = [
        ("摘要来源", summary_label),
        ("关键词来源", kw_label),
        ("结构化字段", backend_label),
    ]
    if warnings:
        cred_items.append(("复核提示", f"{len(warnings)} 项需人工核对"))

    rows = []
    for label, value in cred_items:
        rows.append(f'<div class="rs-cred-item"><span class="rs-cred-label">{escape(label)}</span>{escape(value)}</div>')

    note = ""
    ai_fields = result.parsed_result.ai_metadata_supplemented if hasattr(result, "parsed_result") else []
    if ai_fields:
        note = f'<p style="margin:0.4rem 0 0;font-size:0.72rem;color:var(--accent-3);font-weight:600;">⚠ AI 补充字段：{escape("、".join(ai_fields))}，建议结合原文核对。</p>'

    return _html_block(
        f"""
        <div class="rs-cred-card pf-panel--rail">
          <div class="rs-cred-card__head">
            <h3>🔍 可信度 / 复核提示</h3>
          </div>
          {"".join(rows)}
          {note}
        </div>
        """
    )


def _build_task_hint_html(icon: str, text: str) -> str:
    return _html_block(
        f"""
        <div class="rs-task-hint">
          <span class="rs-task-hint__icon">{icon}</span>
          <span>{escape(text)}</span>
        </div>
        """
    )


def _build_impact_card_html(title: str, icon: str, items: list[str], *, is_pro: bool) -> str:
    variant = "pro" if is_pro else "con"
    accent_cls = f"rs-impact-card--{variant}"
    num_cls = f"rs-impact-num--{variant}"

    if not items:
        body = f'<p class="rs-impact-empty">当前未能提取稳定{"创新点" if is_pro else "不足分析"}。</p>'
    else:
        rows = []
        for i, item in enumerate(items, 1):
            rows.append(
                f'<li class="rs-impact-item">'
                f'<span class="rs-impact-num {num_cls}">{i}</span>'
                f'<span>{escape(item.strip())}</span>'
                f'</li>'
            )
        body = f'<ul class="rs-impact-list">{"".join(rows)}</ul>'

    return _html_block(
        f"""
        <div class="rs-impact-card {accent_cls}">
          <div class="rs-impact-card__head">
            <h3>{icon} {escape(title)}</h3>
          </div>
          {body}
        </div>
        """
    )


def _build_steps_hint_html(steps: list[str]) -> str:
    parts = []
    for i, step in enumerate(steps):
        parts.append(
            f'<span class="rs-steps-hint__step">'
            f'<span class="rs-steps-hint__num">{i + 1}</span>'
            f'{escape(step)}'
            f'</span>'
        )
        if i < len(steps) - 1:
            parts.append('<span class="rs-steps-hint__arrow">→</span>')
    return _html_block(
        f'<div class="rs-steps-hint">{"".join(parts)}</div>'
    )


def _build_outline_count_html(count: int) -> str:
    return _html_block(
        f'<div class="rs-outline-count">'
        f'<span>📋</span>'
        f'已生成 <span class="rs-outline-count__num">{count}</span> 条提纲要点'
        f'</div>'
    )


def render_result_view() -> None:
    result = get_analysis_result()
    if result is None:
        st.warning("暂无解析结果，请先上传 PDF 并完成解析。")
        if st.button("前往上传页", type="primary"):
            set_current_page(PAGE_UPLOAD)
            st.rerun()
        return

    hero_html, package_html, left_html, center_html, right_intro_html = _build_result_sections(result)
    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown(package_html, unsafe_allow_html=True)
    if result.parse_status == "partial_success":
        st.warning("基础解析已完成，但仍有字段需要人工复核。请重点查看左侧“解析提示”和作者识别结果。")

    export_package = build_export_package(result)

    action_cols = st.columns([1.12, 1.12, 0.95, 0.95])
    with action_cols[0]:
        st.download_button(
            label="⬇️ 导出结果 .md",
            data=export_package.markdown_text,
            file_name=export_package.markdown_filename,
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with action_cols[1]:
        st.download_button(
            label="📄 导出结果 .txt",
            data=export_package.txt_text,
            file_name=export_package.txt_filename,
            mime="text/plain",
            use_container_width=True,
        )
    with action_cols[2]:
        if st.button("📤 重新上传", use_container_width=True):
            clear_analysis_result()
            reset_upload_flow(clear_file=True)
            set_current_page(PAGE_UPLOAD)
            st.rerun()
    with action_cols[3]:
        if st.button("🏠 返回首页", use_container_width=True):
            set_current_page(PAGE_HOME)
            st.rerun()

    st.caption("结果页按“基础信息 + 主阅读列 + 提纲输出区”组织，核心内容集中在中间主列，右侧用于按需查看提纲输出。")

    left_col, center_col, right_col = st.columns([0.92, 2.55, 1.18], gap="large")

    with left_col:
        st.markdown(left_html, unsafe_allow_html=True)

    with center_col:
        st.markdown(center_html, unsafe_allow_html=True)

    _outline_steps = {
        "presentation": ["浏览提纲", "按要点扩展", "制作 PPT"],
        "paper": ["确认论点", "按结构扩写", "补充文献引用"],
        "review": ["标注可比较维度", "收集同类文献", "组织综述框架"],
    }

    with right_col:
        st.markdown(right_intro_html, unsafe_allow_html=True)
        tab_presentation, tab_paper, tab_review = st.tabs(["课程汇报", "课程论文", "文献综述"])

        with tab_presentation:
            st.markdown(
                _build_task_hint_html("🎤", "适合直接拆成 PPT 页或课堂讲稿，快速定位汇报要点。"),
                unsafe_allow_html=True,
            )
            st.markdown(_build_steps_hint_html(_outline_steps["presentation"]), unsafe_allow_html=True)
            items_presentation = result.course_presentation_outline_items()
            st.markdown(_build_outline_count_html(len(items_presentation)), unsafe_allow_html=True)
            st.markdown(
                _build_outline_body_html("课程汇报提纲", items_presentation),
                unsafe_allow_html=True,
            )
            _copy_text_presentation = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items_presentation) if item.strip())
            if _copy_text_presentation:
                with st.expander("📋 复制提纲文本"):
                    st.code(_copy_text_presentation, language=None)

        with tab_paper:
            st.markdown(
                _build_task_hint_html("📝", "适合扩写为课程论文结构，围绕研究问题、方法、评价展开。"),
                unsafe_allow_html=True,
            )
            st.markdown(_build_steps_hint_html(_outline_steps["paper"]), unsafe_allow_html=True)
            items_paper = result.course_paper_outline_items()
            st.markdown(_build_outline_count_html(len(items_paper)), unsafe_allow_html=True)
            st.markdown(
                _build_outline_body_html("课程论文提纲", items_paper),
                unsafe_allow_html=True,
            )
            _copy_text_paper = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items_paper) if item.strip())
            if _copy_text_paper:
                with st.expander("📋 复制提纲文本"):
                    st.code(_copy_text_paper, language=None)

        with tab_review:
            st.markdown(
                _build_task_hint_html("📚", "适合作为文献综述材料卡，标注可比较维度和可引用观点。"),
                unsafe_allow_html=True,
            )
            st.markdown(_build_steps_hint_html(_outline_steps["review"]), unsafe_allow_html=True)
            items_review = result.literature_review_outline_items()
            st.markdown(_build_outline_count_html(len(items_review)), unsafe_allow_html=True)
            st.markdown(
                _build_outline_body_html("文献综述基础框架", items_review),
                unsafe_allow_html=True,
            )
            _copy_text_review = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items_review) if item.strip())
            if _copy_text_review:
                with st.expander("📋 复制提纲文本"):
                    st.code(_copy_text_review, language=None)

    if result.english_abstract_text() and result.english_abstract_text() != result.summary_text():
        with st.expander("English Abstract"):
            st.markdown(
                f'<div class="rs-body-text">{_fmt_multiline(reflow_text_for_display(result.english_abstract_text()))}</div>',
                unsafe_allow_html=True,
            )

    if result.raw_text:
        with st.expander("📄 查看提取全文"):
            st.markdown(
                f'<div class="rs-body-text">{_fmt_multiline(reflow_text_for_display(result.raw_text))}</div>',
                unsafe_allow_html=True,
            )
    if result.structured_debug and _should_show_debug_info():
        with st.expander("🔧 开发信息 · 解析调试数据"):
            st.json(result.structured_debug)


def _should_show_debug_info() -> bool:
    return os.getenv("SHOW_DEBUG_INFO", "").strip() == "1"


def _build_result_sections(result) -> tuple[str, str, str, str, str]:
    notice = (result.structured_notice or "").strip() or "当前结果仅供辅助参考，建议结合原文判断。"
    keywords = result.filtered_keywords()
    authors = result.filtered_authors()
    parse_hints = _build_parse_hints(result, result.warning_items())

    _chip_colors = ["chip-indigo", "chip-coral", "chip-blue", "chip-gold", "chip-green", "chip-purple"]
    keyword_display = (
        "".join(f'<span class="kw-chip {_chip_colors[i % len(_chip_colors)]}">{escape(keyword)}</span>' for i, keyword in enumerate(keywords))
        if keywords
        else f'<span class="kw-chip kw-chip--empty">{KEYWORDS_DISPLAY_FALLBACK}</span>'
    )
    authors_display = (
        "".join(f'<span class="author-chip {_chip_colors[i % len(_chip_colors)]}">{escape(author)}</span>' for i, author in enumerate(authors))
        if authors
        else f'<span class="author-chip author-chip--empty">{AUTHORS_DISPLAY_FALLBACK}</span>'
    )

    structured_cards = result.structured_field_items()
    structured_debug = result.structured_debug if isinstance(result.structured_debug, dict) else {}
    final_result = structured_debug.get("final_result", {}) if isinstance(structured_debug.get("final_result"), dict) else {}

    backend_label = _fmt_backend(str(final_result.get("backend", "")).strip())
    structured_ready = result.structured_field_count()
    keywords_count = len(keywords)
    summary_language = _summary_language(result.primary_summary_language())
    parse_status_label = "已完成解析" if result.parse_status == "success" else "部分成功，建议复核"

    title_text = escape(result.title or "未识别标题")
    file_name_text = escape(result.file_name or "未记录文件名")
    summary_text = _fmt_multiline(result.summary_text())

    hero_html = _html_block(
        f"""
        <section class="pf-result-hero">
          <div class="rs-hero">
            <p class="rs-hero__label">论文解析结果</p>
            <h1 class="rs-hero__title">{title_text}</h1>
            <p class="rs-hero__file">{file_name_text}</p>
            <div class="rs-hero__meta">
              <span class="rs-badge chip-coral">{escape(parse_status_label)}</span>
              <span class="rs-badge chip-blue">{backend_label}</span>
              <span class="rs-badge chip-green">{structured_ready}/3 字段</span>
              <span class="rs-badge chip-purple">{keywords_count} 关键词</span>
            </div>
          </div>
        </section>
        """
    )

    _field_accent = {"研究问题": "hl-coral", "研究方法": "hl-blue", "核心结论": "hl-green"}
    struct_cards_html = ""
    for _, label, content, _ in structured_cards:
        accent = _field_accent.get(label, "hl-indigo")
        struct_cards_html += f"""
<div class="rs-structured-row">
  <h3 class="rs-structured-row__title"><span class="{accent}">{_field_icon(label)} {escape(label)}</span></h3>
  <p class="rs-structured-row__body">{_fmt_multiline(content)}</p>
</div>"""

    left_html = _html_block(
        f"""
        <div class="rs-side-stack">
          <div class="rs-side-card pf-panel--meta">
            <h3 class="rs-side-card__title">文献信息</h3>
            <div class="rs-info-list">
              <div class="rs-info-row">
                <span class="rs-info-row__label">文件名</span>
                <span class="rs-info-row__value">{file_name_text}</span>
              </div>
              <div class="rs-info-row">
                <span class="rs-info-row__label">摘要语言</span>
                <span class="rs-info-row__value">{escape(summary_language)}</span>
              </div>
            </div>
            <div class="rs-label-stack">
              <span class="rs-mini-label">作者</span>
              <div class="rs-chip-cloud">{authors_display}</div>
            </div>
            <div class="rs-label-stack">
              <span class="rs-mini-label">关键词</span>
              <div class="rs-chip-cloud">{keyword_display}</div>
            </div>
            <div class="rs-label-stack rs-label-stack--tips">
              <span class="rs-mini-label">解析提示</span>
              <div class="rs-list-shell">{_fmt_list_html(parse_hints, empty_text="当前解析未返回额外提示。")}</div>
            </div>
          </div>
        </div>
        """
    )

    center_html = _html_block(
        f"""
        <div class="rs-main-stack">
          <section class="rs-main-module rs-main-module--summary pf-panel--main topbar-coral">
            <div class="rs-section__head">
              <h2 class="rs-section__title">主摘要</h2>
              <span class="rs-badge chip-coral">{escape(summary_language)}</span>
            </div>
            <div class="rs-summary-box rs-summary-box--main">
              <p class="rs-body-text rs-body-text--lead">{summary_text}</p>
            </div>
          </section>

          <section class="rs-main-module pf-panel--main topbar-blue">
            <div class="rs-section__head">
              <h2 class="rs-section__title">结构化提取</h2>
              <span class="rs-badge chip-blue">{structured_ready}/3</span>
            </div>
            <div class="rs-structured-list">
              {struct_cards_html}
            </div>
          </section>

          <section class="rs-main-module pf-panel--main topbar-purple">
            <div class="rs-section__head">
              <h2 class="rs-section__title">AI 解读与评价</h2>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">通俗摘要</h3>
              <p class="rs-body-text">{_fmt_multiline(result.plain_language_summary_text())}</p>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">研究方法说明</h3>
              <p class="rs-body-text">{_fmt_multiline(result.method_explanation_text())}</p>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">创新点</h3>
              <div class="rs-list-shell">{_fmt_list_html(result.innovation_items(), empty_text="当前未能提取稳定创新点。")}</div>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">不足之处</h3>
              <div class="rs-list-shell">{_fmt_list_html(result.limitation_items(), empty_text="当前未能提取稳定不足分析。")}</div>
            </div>
          </section>
        </div>
        """
    )

    package_html = _build_result_package_html(result)
    structure_map_html = _build_structure_map_html(result)
    credibility_html = _build_credibility_card_html(result)

    left_html = left_html + "\n" + credibility_html

    innovation_card_html = _build_impact_card_html("创新点", "💡", result.innovation_items(), is_pro=True)
    limitation_card_html = _build_impact_card_html("不足之处", "⚠️", result.limitation_items(), is_pro=False)

    center_html = _html_block(
        f"""
        <div class="rs-main-stack">
          <section class="rs-main-module pf-panel--main topbar-coral">
            <div class="rs-section__head">
              <h2 class="rs-section__title">主摘要</h2>
              <span class="rs-badge">{summary_language}</span>
              {escape(backend_label)}
            </div>
            <div class="rs-summary-box">
              <h4>摘要</h4>
              <p class="rs-body-text">{_fmt_multiline(reflow_text_for_display(result.summary_text()))}</p>
            </div>
          </section>

          <section class="rs-main-module pf-panel--main topbar-blue">
            <div class="rs-section__head">
              <h2 class="rs-section__title">结构化提取</h2>
              <span class="rs-badge chip-blue">{structured_ready}/3</span>
            </div>
            <div class="rs-structured-list">
              {struct_cards_html}
            </div>
          </section>

          {structure_map_html}

          <section class="rs-main-module pf-panel--main topbar-purple">
            <div class="rs-section__head">
              <h2 class="rs-section__title">AI 解读与评价</h2>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">通俗摘要</h3>
              <p class="rs-body-text">{_fmt_multiline(result.plain_language_summary_text())}</p>
            </div>
            <div class="rs-reading-card">
              <h3 class="rs-reading-card__title">研究方法说明</h3>
              <p class="rs-body-text">{_fmt_multiline(result.method_explanation_text())}</p>
            </div>
          </section>

          {innovation_card_html}

          {limitation_card_html}
        </div>
        """
    )

    right_intro_html = _html_block(
        """
        <div class="rs-rail-card rs-rail-card--intro pf-panel--rail">
          <h3 class="rs-rail-card__title">📝 作业模式工作台</h3>
          <p class="rs-writing-intro">选择你的作业场景，获取对应的结构化输出。每个模式都提供任务定位和可直接使用的提纲。</p>
        </div>
        """
    )
    return hero_html, package_html, left_html, center_html, right_intro_html


def _build_outline_body_html(title: str, items: list[str]) -> str:
    return _html_block(
        f"""
        <div class="rs-rail-card rs-rail-card--outline pf-panel--rail">
          <h4 class="rs-outline-panel__title">{escape(title)}</h4>
          <div class="rs-list-shell">{_fmt_list_html(items, ordered=True, empty_text="暂无内容。")}</div>
        </div>
        """
    )


def _field_icon(label: str) -> str:
    mapping = {
        "研究问题": "❓",
        "研究方法": "🛠️",
        "核心结论": "🎯",
    }
    return mapping.get(label, "🧩")


def _fmt_list_html(items: list[str], *, ordered: bool = False, empty_text: str = "暂无内容。") -> str:
    cleaned = [escape(item.strip()) for item in items if item and item.strip()]
    if not cleaned:
        return f'<p class="rs-empty-text">{escape(empty_text)}</p>'
    tag = "ol" if ordered else "ul"
    return f"<{tag} class=\"rs-list\">" + "".join(f"<li>{item}</li>" for item in cleaned) + f"</{tag}>"


def _build_parse_hints(result, warnings: list[str]) -> list[str]:
    if warnings:
        return warnings

    structured_debug = result.structured_debug if isinstance(result.structured_debug, dict) else {}
    summary_debug = structured_debug.get("summary_debug", {}) if isinstance(structured_debug.get("summary_debug"), dict) else {}
    final_result = structured_debug.get("final_result", {}) if isinstance(structured_debug.get("final_result"), dict) else {}

    summary_language = result.primary_summary_language()
    keyword_source = str(summary_debug.get("keyword_source", "")).strip()
    backend = str(final_result.get("backend", "")).strip()

    hints: list[str] = []
    if summary_language == "zh":
        hints.append("主摘要来自中文摘要原文。")
    elif summary_language == "en":
        hints.append("主摘要当前回退为英文 Abstract。")
    else:
        hints.append("主摘要当前来自正文 fallback 归纳。")

    if keyword_source:
        hints.append(f"关键词来源：{_keyword_source_label(keyword_source)}。")
    if backend:
        hints.append(f"结构化字段生成方式：{_fmt_backend(backend)}。")
    hints.append("当前未发现需要额外警示的解析异常。")
    return hints


def _fmt_multiline(text: str) -> str:
    return escape(text).replace("\n", "<br>")


def _fmt_backend(backend: str) -> str:
    if not backend:
        return "规则提取"
    if backend == "relay_precheck":
        return "候选预检"
    if "config_error" in backend:
        return "规则回退"
    if "request_error" in backend:
        return "请求回退"
    if "course_support" in backend:
        return "LLM 增强"
    if any(key in backend for key in ("chat", "response", "relay")):
        return "LLM 增强"
    return escape(backend)


def _summary_language(value: str) -> str:
    if value == "zh":
        return "中文摘要优先"
    if value == "en":
        return "英文摘要回退"
    return "摘要待补充"


def _keyword_source_label(value: str) -> str:
    mapping = {
        "strategy_a_explicit_zh": "显式关键词区",
        "strategy_a_block_zh": "显式关键词区",
        "strategy_b_line_zh": "关键词行匹配",
        "strategy_c_abstract_nearby_zh": "摘要邻近提取",
        "frequency_fallback": "正文频次回退",
        "llm_keyword_fallback": "模型保守补充",
    }
    return mapping.get(value, value)


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())
