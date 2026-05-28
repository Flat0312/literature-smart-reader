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
                把一篇<span class="hl-coral">论文</span><br>
                变成一张可写作的
                <span class="hl-blue">阅读地图</span>
              </h1>
              <p class="home-hero__sub">
                上传 PDF，自动整理<span class="hl-gold">摘要</span>、<span class="hl-indigo">关键词</span>、<span class="hl-green">研究问题</span>、研究方法和<span class="hl-purple">课程写作提纲</span>。
                视觉保持清爽利落，结构像一组可以直接使用的编辑卡片。
              </p>
              <div class="home-hero__caps">
                <div class="home-cap accent-coral">
                  <span class="home-cap__icon icon-coral">01</span>
                  <div>
                    <strong>上传即解析</strong>
                    <p>支持单篇 PDF，可复制文本效果最佳</p>
                  </div>
                </div>
                <div class="home-cap accent-blue">
                  <span class="home-cap__icon icon-blue">02</span>
                  <div>
                    <strong>自动提取摘要与关键词</strong>
                    <p>结构化输出，信息一目了然</p>
                  </div>
                </div>
                <div class="home-cap accent-purple">
                  <span class="home-cap__icon icon-purple">03</span>
                  <div>
                    <strong>研究脉络结构化</strong>
                    <p>研究问题 · 研究方法 · 核心结论</p>
                  </div>
                </div>
              </div>
            </div>
            <div class="home-hero__visual" aria-hidden="true">
              <div class="demo-card">
                <!-- Input: PDF -->
                <div class="demo-input">
                  <div class="demo-pdf">
                    <span class="demo-pdf__icon">📄</span>
                    <span class="demo-pdf__label">你的论文 PDF</span>
                  </div>
                  <div class="demo-arrow">
                    <span class="demo-arrow__line"></span>
                    <span class="demo-arrow__head">→</span>
                    <span class="demo-arrow__text">自动解析</span>
                  </div>
                </div>

                <!-- Output: 3 cards -->
                <div class="demo-output">
                  <div class="demo-out-card accent-coral">
                    <div class="demo-out-card__head">
                      <span class="demo-out-num icon-coral">01</span>
                      <strong>摘要</strong>
                    </div>
                    <p>一句话抓住论文核心</p>
                  </div>
                  <div class="demo-out-card accent-blue">
                    <div class="demo-out-card__head">
                      <span class="demo-out-num icon-blue">02</span>
                      <strong>关键词</strong>
                    </div>
                    <p>研究问题 · 方法 · 结论</p>
                  </div>
                  <div class="demo-out-card accent-purple">
                    <div class="demo-out-card__head">
                      <span class="demo-out-num icon-purple">03</span>
                      <strong>写作提纲</strong>
                    </div>
                    <p>汇报 · 论文 · 综述直接用</p>
                  </div>
                </div>

                <!-- Tagline -->
                <div class="demo-tagline">
                  <span class="demo-tag">✨ 上传一次，三张卡片搞定</span>
                </div>
              </div>
            </div>
          </div>

        </section>
        """
    )
