"""Result page rendering — wide layout with dominant center reading column."""

from __future__ import annotations

from html import escape
from textwrap import dedent

import streamlit as st

from config.settings import PAGE_HOME, PAGE_UPLOAD
from models.paper_result import AUTHORS_DISPLAY_FALLBACK, KEYWORDS_DISPLAY_FALLBACK, SUMMARY_DISPLAY_FALLBACK
from services.export_service import build_export_package
from utils.session import clear_analysis_result, get_analysis_result, reset_upload_flow, set_current_page
from utils.text_utils import reflow_text_for_display


def render_result_view() -> None:
    result = get_analysis_result()
    if result is None:
        st.warning("暂无解析结果，请先上传 PDF 并完成解析。")
        if st.button("前往上传页", type="primary"):
            set_current_page(PAGE_UPLOAD)
            st.rerun()
        return

    hero_html, left_html, center_html, right_intro_html = _build_result_sections(result)
    st.markdown(hero_html, unsafe_allow_html=True)
    if result.parse_status == "partial_success":
        st.warning("基础解析已完成，但仍有字段需要人工复核。请重点查看左侧“解析提示”和作者识别结果。")

    export_package = build_export_package(result)

    action_cols = st.columns([1.12, 1.12, 0.95, 0.95])
    with action_cols[0]:
        st.download_button(
            label="导出结果 .md",
            data=export_package.markdown_text,
            file_name=export_package.markdown_filename,
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
    with action_cols[1]:
        st.download_button(
            label="导出结果 .txt",
            data=export_package.txt_text,
            file_name=export_package.txt_filename,
            mime="text/plain",
            use_container_width=True,
        )
    with action_cols[2]:
        if st.button("重新上传", use_container_width=True):
            clear_analysis_result()
            reset_upload_flow(clear_file=True)
            set_current_page(PAGE_UPLOAD)
            st.rerun()
    with action_cols[3]:
        if st.button("返回首页", use_container_width=True):
            set_current_page(PAGE_HOME)
            st.rerun()

    st.caption("结果页已按“辅助信息 + 主阅读列 + 写作输出区”的方式重排，核心内容集中在中间主列，右侧用于按需查看提纲输出。")

    left_col, center_col, right_col = st.columns([0.92, 2.55, 1.18], gap="large")

    with left_col:
        st.markdown(left_html, unsafe_allow_html=True)

    with center_col:
        st.markdown(center_html, unsafe_allow_html=True)

    with right_col:
        st.markdown(right_intro_html, unsafe_allow_html=True)
        tab_presentation, tab_paper, tab_review = st.tabs(["课程汇报", "课程论文", "文献综述"])

        with tab_presentation:
            st.markdown(
                _build_outline_body_html("课程汇报提纲", result.course_presentation_outline_items()),
                unsafe_allow_html=True,
            )

        with tab_paper:
            st.markdown(
                _build_outline_body_html("课程论文提纲", result.course_paper_outline_items()),
                unsafe_allow_html=True,
            )

        with tab_review:
            st.markdown(
                _build_outline_body_html("文献综述基础框架", result.literature_review_outline_items()),
                unsafe_allow_html=True,
            )

    if result.english_abstract_text() and result.english_abstract_text() != result.summary_text():
        with st.expander("English Abstract"):
            st.markdown(
                f'<div class="rs-body-text">{_fmt_multiline(reflow_text_for_display(result.english_abstract_text()))}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="result-expanders">', unsafe_allow_html=True)
    if result.raw_text:
        with st.expander("📄 查看提取全文"):
            st.markdown(
                f'<div class="rs-body-text">{_fmt_multiline(reflow_text_for_display(result.raw_text))}</div>',
                unsafe_allow_html=True,
            )
    if result.structured_debug:
        with st.expander("🔧 开发信息 · 解析调试数据"):
            st.json(result.structured_debug)
    st.markdown("</div>", unsafe_allow_html=True)


def _build_result_sections(result) -> tuple[str, str, str, str]:
    notice = (result.structured_notice or "").strip() or "当前结果仅供辅助参考，建议结合原文判断。"
    keywords = result.filtered_keywords()
    authors = result.filtered_authors()
    parse_hints = _build_parse_hints(result, result.warning_items())

    keyword_display = (
        "".join(f'<span class="kw-chip">{escape(keyword)}</span>' for keyword in keywords)
        if keywords
        else f'<span class="kw-chip kw-chip--empty">{KEYWORDS_DISPLAY_FALLBACK}</span>'
    )
    authors_display = (
        "".join(f'<span class="author-chip">{escape(author)}</span>' for author in authors)
        if authors
        else f'<span class="author-chip author-chip--empty">{AUTHORS_DISPLAY_FALLBACK}</span>'
    )

    structured_cards = result.structured_field_items()
    structured_debug = result.structured_debug if isinstance(result.structured_debug, dict) else {}
    final_result = structured_debug.get("final_result", {}) if isinstance(structured_debug.get("final_result"), dict) else {}

    backend_label = _fmt_backend(str(final_result.get("backend", "")).strip())
    structured_ready = result.structured_field_count()
    keywords_count = len(keywords)
    summary_language = _summary_language_label(result.primary_summary_language())
    parse_status_label = "已完成解析" if result.parse_status == "success" else "部分成功，建议复核"

    title_text = escape(result.title or "未识别标题")
    file_name_text = escape(result.file_name or "未记录文件名")
    summary_text = _fmt_multiline(result.summary_text())

    hero_html = _html_block(
        f"""
        <div class="rs-hero">
          <div class="rs-hero__inner">
            <div class="rs-hero__title-block">
              <p class="rs-hero__label">论文解析结果</p>
              <span class="rs-badge rs-badge--accent">{escape(parse_status_label)}</span>
              <p class="rs-hero__file">主阅读内容已集中到中间主列，右侧为按需切换的课程写作输出区。</p>
              <h1 class="rs-hero__title">{title_text}</h1>
              <p class="rs-hero__file">{file_name_text}</p>
            </div>
            <div class="rs-hero__stats">
              <div class="rs-stat">
                <span class="rs-stat__value">{backend_label}</span>
                <span class="rs-stat__label">解析方式</span>
              </div>
              <div class="rs-stat">
                <span class="rs-stat__value">{keywords_count}</span>
                <span class="rs-stat__label">关键词</span>
              </div>
              <div class="rs-stat">
                <span class="rs-stat__value">{structured_ready} / 3</span>
                <span class="rs-stat__label">结构化字段</span>
              </div>
              <div class="rs-stat">
                <span class="rs-stat__value">3 类</span>
                <span class="rs-stat__label">写作输出</span>
              </div>
            </div>
          </div>
        </div>
        """
    )

    struct_cards_html = ""
    for _, label, content, _ in structured_cards:
        struct_cards_html += f"""
<div class="rs-structured-item">
  <h3 class="rs-structured-item__title">{escape(label)}</h3>
  <p class="rs-structured-item__body">{_fmt_multiline(content)}</p>
</div>"""

    left_html = _html_block(
        f"""
        <div class="rs-side-stack">
          <div class="rs-side-card">
            <h3 class="rs-side-card__title">基础信息</h3>
            <div class="rs-info-list">
              <div class="rs-info-row">
                <span class="rs-info-row__label">文件名</span>
                <span class="rs-info-row__value">{file_name_text}</span>
              </div>
              <div class="rs-info-row">
                <span class="rs-info-row__label">识别标题</span>
                <span class="rs-info-row__value">{title_text}</span>
              </div>
              <div class="rs-info-row">
                <span class="rs-info-row__label">解析状态</span>
                <span class="rs-info-row__value">{escape(parse_status_label)}</span>
              </div>
              <div class="rs-info-row">
                <span class="rs-info-row__label">主摘要语言</span>
                <span class="rs-info-row__value">{escape(summary_language)}</span>
              </div>
              <div class="rs-info-row">
                <span class="rs-info-row__label">解析方式</span>
                <span class="rs-info-row__value">{backend_label}</span>
              </div>
            </div>
          </div>
          <div class="rs-side-card rs-side-card--soft">
            <h3 class="rs-side-card__title">作者、关键词与提示</h3>
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
          <section class="rs-main-module rs-main-module--summary">
            <div class="rs-section__head">
              <h2 class="rs-section__title">主摘要</h2>
              <span class="rs-badge">{escape(summary_language)}</span>
            </div>
            <div class="rs-summary-box rs-summary-box--main">
              <p class="rs-body-text rs-body-text--lead">{summary_text}</p>
            </div>
          </section>

          <section class="rs-main-module">
            <div class="rs-section__head">
              <h2 class="rs-section__title">结构化提取</h2>
              <span class="rs-badge rs-badge--accent">{structured_ready} / 3 字段</span>
            </div>
            <p class="rs-section__sub">{escape(notice)}</p>
            <div class="rs-structured-grid">
              {struct_cards_html}
            </div>
          </section>

          <section class="rs-main-module">
            <div class="rs-section__head">
              <h2 class="rs-section__title">AI 解读</h2>
              <span class="rs-badge rs-badge--accent">课程写作辅助</span>
            </div>
            <div class="rs-reading-grid">
              <div class="rs-reading-card">
                <h3 class="rs-reading-card__title">通俗摘要</h3>
                <p class="rs-body-text">{_fmt_multiline(result.plain_language_summary_text())}</p>
              </div>
              <div class="rs-reading-card">
                <h3 class="rs-reading-card__title">研究方法说明</h3>
                <p class="rs-body-text">{_fmt_multiline(result.method_explanation_text())}</p>
              </div>
            </div>
          </section>

          <section class="rs-main-module">
            <div class="rs-section__head">
              <h2 class="rs-section__title">创新与不足</h2>
              <span class="rs-badge">适合课程展示</span>
            </div>
            <div class="rs-reading-grid">
              <div class="rs-reading-card">
                <h3 class="rs-reading-card__title">创新点分析</h3>
                <div class="rs-list-shell">{_fmt_list_html(result.innovation_items(), empty_text="当前未能提取稳定创新点。")}</div>
              </div>
              <div class="rs-reading-card">
                <h3 class="rs-reading-card__title">不足分析</h3>
                <div class="rs-list-shell">{_fmt_list_html(result.limitation_items(), empty_text="当前未能提取稳定不足分析。")}</div>
              </div>
            </div>
          </section>
        </div>
        """
    )

    right_intro_html = _html_block(
        """
        <div class="rs-rail-card rs-rail-card--intro">
          <h3 class="rs-rail-card__title">写作输出区</h3>
          <p class="rs-writing-intro">右侧只保留提纲型输出。默认通过 tabs 切换课程汇报、课程论文和文献综述三类内容，避免整列过长与信息挤压。</p>
        </div>
        """
    )
    return hero_html, left_html, center_html, right_intro_html


def _build_outline_body_html(title: str, items: list[str]) -> str:
    return _html_block(
        f"""
        <div class="rs-rail-card rs-rail-card--outline">
          <h4 class="rs-outline-panel__title">{escape(title)}</h4>
          <div class="rs-list-shell">{_fmt_list_html(items, ordered=True, empty_text="暂无内容。")}</div>
        </div>
        """
    )


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
    return backend


def _summary_language_label(value: str) -> str:
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
