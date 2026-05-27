"""Home page rendering — 成品化改版 v3."""

from __future__ import annotations

from textwrap import dedent

import streamlit as st

from config.settings import PAGE_UPLOAD
from utils.session import set_current_page


def render_home_view() -> None:
    st.markdown(_build_home_hero_html(), unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📤 上传论文，开始解析", type="primary", use_container_width=True):
            set_current_page(PAGE_UPLOAD)
            st.rerun()


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())


def _build_home_hero_html() -> str:
    return _html_block(
        """
        <section class="pf-home-hero">
          <div class="home-hero__inner">
            <div class="home-hero__copy">
              <span class="home-hero__tag">📚 文献阅读工作台</span>
              <h1 class="home-hero__title">
                把一篇论文<br>
                变成一张可写作的
                <span class="pf-highlight">阅读地图</span>
              </h1>
              <p class="home-hero__sub">
                上传 PDF，自动整理摘要、关键词、研究问题、研究方法和课程写作提纲。
                视觉保持清爽利落，结构像一组可以直接使用的编辑卡片。
              </p>
              <div class="home-hero__caps">
                <div class="home-cap">
                  <span class="home-cap__icon">01</span>
                  <div>
                    <strong>上传即解析</strong>
                    <p>支持单篇 PDF，可复制文本效果最佳</p>
                  </div>
                </div>
                <div class="home-cap">
                  <span class="home-cap__icon">02</span>
                  <div>
                    <strong>自动提取摘要与关键词</strong>
                    <p>结构化输出，信息一目了然</p>
                  </div>
                </div>
                <div class="home-cap">
                  <span class="home-cap__icon">03</span>
                  <div>
                    <strong>研究脉络结构化</strong>
                    <p>研究问题 · 研究方法 · 核心结论</p>
                  </div>
                </div>
              </div>
            </div>
            <div class="home-hero__visual pf-hero-visual" aria-hidden="true">
              <div class="pf-map-board">
                <div class="pf-map-board__head">
                  <span>READING MAP</span>
                  <strong>论文阅读地图</strong>
                </div>
                <div class="pf-map-grid">
                  <div class="pf-map-document">
                    <div class="pf-map-document__top">
                      <span>摘要卡</span>
                      <em>ZH FIRST</em>
                    </div>
                    <strong>中文摘要优先</strong>
                    <p>先抓住研究问题、方法和结论，再把可写作的信息整理成清晰卡片。</p>
                    <div class="pf-map-chips">
                      <span>关键词</span>
                      <span>研究问题</span>
                      <span>核心结论</span>
                    </div>
                    <div class="pf-map-lines">
                      <i></i>
                      <i></i>
                      <i></i>
                    </div>
                  </div>
                  <div class="pf-map-index">
                    <div class="pf-map-step">
                      <b>01</b>
                      <div>
                        <strong>摘要</strong>
                        <span>核心观点</span>
                      </div>
                    </div>
                    <div class="pf-map-step">
                      <b>02</b>
                      <div>
                        <strong>结构化</strong>
                        <span>问题 · 方法</span>
                      </div>
                    </div>
                    <div class="pf-map-step">
                      <b>03</b>
                      <div>
                        <strong>输出</strong>
                        <span>课程写作</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="pf-map-output">
                  <span>写作出口</span>
                  <strong>汇报 · 课程论文 · 文献综述</strong>
                  <i></i>
                </div>
              </div>
            </div>
          </div>

        </section>
        """
    )
