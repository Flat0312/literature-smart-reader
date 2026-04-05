"""Home page rendering — 成品化改版 v3."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from config.settings import PAGE_UPLOAD
from utils.session import set_current_page


def render_home_view() -> None:
    # ── 整合 Hero：标题 + 副标题 + 能力点 + 主按钮 ──────────────────────────
    st.markdown(
        _html_block("""
        <div class="home-hero">
          <div class="home-hero__inner">
            <div class="home-hero__copy">
              <h1 class="home-hero__title">文献智读</h1>
              <p class="home-hero__sub">上传一篇论文 PDF，30 秒内获取标题、关键词、摘要与结构化解读。专为课程写作场景设计。</p>
              <div class="home-hero__caps">
                <div class="home-cap">
                  <span class="home-cap__icon">📄</span>
                  <div>
                    <strong>上传即解析</strong>
                    <p>支持单篇 PDF，可复制文本效果最佳</p>
                  </div>
                </div>
                <div class="home-cap">
                  <span class="home-cap__icon">🔍</span>
                  <div>
                    <strong>自动提取摘要与关键词</strong>
                    <p>结构化输出，信息一目了然</p>
                  </div>
                </div>
                <div class="home-cap">
                  <span class="home-cap__icon">📊</span>
                  <div>
                    <strong>研究脉络结构化</strong>
                    <p>研究问题 · 研究方法 · 核心结论</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # 主按钮（Streamlit 原生，保证可点击）
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("上传论文，开始解析 →", type="primary", use_container_width=True):
            set_current_page(PAGE_UPLOAD)
            st.rerun()


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())
