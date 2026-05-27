"""Streamlit entry for the Literature Smart Reader V1 app."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import logging
import os

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
load_dotenv()

# Load Streamlit secrets into env vars (for cloud deployment)
try:
    for _k, _v in st.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception as exc:
    logger.warning("Failed to load Streamlit secrets: %s", exc)

from config.settings import APP_ICON, APP_SUBTITLE, APP_TITLE, PAGE_HOME, PAGE_ORDER, PAGE_RESULT, PAGE_UPLOAD
from utils.session import get_current_page, init_session_state
from views.home_view import render_home_view
from views.result_view import render_result_view
from views.upload_view import render_upload_view


def load_global_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_navigation(current_page: str) -> None:
    page_labels = {
        PAGE_HOME: "01 首页",
        PAGE_UPLOAD: "02 上传",
        PAGE_RESULT: "03 结果",
    }
    items = []
    for page_name in PAGE_ORDER:
        active_class = "active" if page_name == current_page else ""
        aria_current = ' aria-current="page"' if page_name == current_page else ''
        items.append(f'<span class="pf-nav__item {active_class}"{aria_current}>{page_labels[page_name]}</span>')
    st.markdown(f'<nav aria-label="步骤导航" class="pf-nav">{"".join(items)}</nav>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_session_state()
    load_global_styles()

    st.markdown(
        _html_block(
            f"""
        <section class="pf-topbar">
            <div class="pf-topbar__brand">
                <div class="pf-topbar__icon">{APP_ICON}</div>
                <div class="pf-topbar__copy">
                    <p class="pf-topbar__eyebrow">LITERATURE SMART READER</p>
                    <h1>{APP_TITLE}</h1>
                    <p class="pf-topbar__subtitle">{APP_SUBTITLE}</p>
                </div>
            </div>
            <div class="pf-topbar__meta">
                <span class="pf-topbar__badge">课程写作场景</span>
                <span class="pf-topbar__badge">单篇 PDF 解析</span>
                <span class="pf-topbar__badge">结构化阅读辅助</span>
            </div>
        </section>
        """
        ),
        unsafe_allow_html=True,
    )

    current_page = get_current_page()
    render_navigation(current_page)

    if current_page == PAGE_HOME:
        render_home_view()
    elif current_page == PAGE_UPLOAD:
        render_upload_view()
    elif current_page == PAGE_RESULT:
        render_result_view()
    else:
        render_home_view()


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())


if __name__ == "__main__":
    main()
