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


def load_global_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _build_background_layer_html() -> str:
    return _html_block(
        """
        <style>
        .pf-bg-layer{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;background:linear-gradient(135deg,rgba(255,255,255,.78),rgba(248,251,255,.56))}
        .pf-bubble{position:absolute;width:var(--s);height:var(--s);left:var(--x);top:var(--y);border-radius:50%;opacity:var(--o,.18);will-change:transform,filter,border-radius;background:radial-gradient(circle at 30% 25%,rgba(255,255,255,.92) 0 9%,rgba(255,255,255,.38) 10% 18%,var(--c) 45%,rgba(255,255,255,0) 76%);box-shadow:inset -24px -28px 46px rgba(20,33,62,.13),inset 18px 20px 38px rgba(255,255,255,.52),0 18px 44px rgba(44,62,98,.09);mix-blend-mode:multiply;filter:saturate(1.15) blur(.1px)}
        .pf-bubble::before{content:"";position:absolute;inset:14% 20% auto auto;width:28%;height:18%;border-radius:50%;background:rgba(255,255,255,.55);filter:blur(6px);transform:rotate(-24deg)}
        .pf-bubble::after{content:"";position:absolute;inset:8%;border-radius:inherit;border:1px solid rgba(255,255,255,.2)}
        .pf-bubble--1{--s:clamp(180px,22vw,360px);--x:3%;--y:6%;--c:rgba(255,107,122,.64);--o:.17;animation:pf-bubble-path-a 18s cubic-bezier(.42,0,.25,1) infinite}
        .pf-bubble--2{--s:clamp(140px,18vw,300px);--x:68%;--y:4%;--c:rgba(47,129,247,.58);--o:.16;animation:pf-bubble-path-b 21s cubic-bezier(.38,0,.28,1) infinite}
        .pf-bubble--3{--s:clamp(110px,14vw,230px);--x:45%;--y:18%;--c:rgba(253,185,39,.66);--o:.18;animation:pf-bubble-path-c 16s cubic-bezier(.4,0,.2,1) infinite}
        .pf-bubble--4{--s:clamp(130px,17vw,270px);--x:78%;--y:52%;--c:rgba(99,102,241,.57);--o:.15;animation:pf-bubble-path-d 23s cubic-bezier(.35,0,.25,1) infinite}
        .pf-bubble--5{--s:clamp(105px,13vw,210px);--x:8%;--y:62%;--c:rgba(16,185,129,.6);--o:.17;animation:pf-bubble-path-e 19s cubic-bezier(.4,0,.22,1) infinite}
        .pf-bubble--6{--s:clamp(150px,20vw,320px);--x:42%;--y:70%;--c:rgba(139,92,246,.55);--o:.14;animation:pf-bubble-path-f 25s cubic-bezier(.36,0,.25,1) infinite}
        .pf-bubble--7{--s:clamp(70px,9vw,150px);--x:22%;--y:28%;--c:rgba(14,165,233,.6);--o:.16;animation:pf-bubble-path-d 15s cubic-bezier(.37,0,.2,1) infinite reverse}
        .pf-bubble--8{--s:clamp(78px,10vw,170px);--x:58%;--y:42%;--c:rgba(244,114,182,.55);--o:.16;animation:pf-bubble-path-a 17s cubic-bezier(.4,0,.2,1) infinite reverse}
        .pf-bubble--9{--s:clamp(62px,8vw,130px);--x:86%;--y:25%;--c:rgba(251,146,60,.58);--o:.15;animation:pf-bubble-path-e 14s cubic-bezier(.38,0,.24,1) infinite}
        .pf-bubble--10{--s:clamp(84px,11vw,180px);--x:28%;--y:78%;--c:rgba(45,212,191,.58);--o:.14;animation:pf-bubble-path-b 20s cubic-bezier(.42,0,.2,1) infinite reverse}
        .pf-bubble--11{--s:clamp(54px,7vw,116px);--x:13%;--y:42%;--c:rgba(129,140,248,.6);--o:.14;animation:pf-bubble-path-c 13s cubic-bezier(.38,0,.26,1) infinite reverse}
        .pf-bubble--12{--s:clamp(60px,8vw,125px);--x:70%;--y:76%;--c:rgba(250,204,21,.56);--o:.14;animation:pf-bubble-path-f 18s cubic-bezier(.36,0,.24,1) infinite reverse}
        .pf-bubble--13{--s:clamp(42px,6vw,96px);--x:36%;--y:5%;--c:rgba(34,197,94,.52);--o:.13;animation:pf-bubble-path-e 12s cubic-bezier(.4,0,.22,1) infinite reverse}
        .pf-bubble--14{--s:clamp(48px,7vw,108px);--x:92%;--y:66%;--c:rgba(56,189,248,.5);--o:.13;animation:pf-bubble-path-c 15s cubic-bezier(.38,0,.22,1) infinite}
        .pf-bubble--15{--s:clamp(66px,8vw,140px);--x:51%;--y:6%;--c:rgba(248,113,113,.48);--o:.12;animation:pf-bubble-path-d 16s cubic-bezier(.36,0,.24,1) infinite reverse}
        .pf-bubble--16{--s:clamp(50px,7vw,112px);--x:1%;--y:83%;--c:rgba(168,85,247,.5);--o:.13;animation:pf-bubble-path-b 17s cubic-bezier(.42,0,.2,1) infinite}
        .pf-collision-wave{position:absolute;left:var(--x);top:var(--y);width:var(--s);height:var(--s);border-radius:50%;border:1px solid var(--c);opacity:0;transform:translate(-50%,-50%) scale(.25);filter:blur(.2px);will-change:transform,opacity}
        .pf-wave--1{--x:49%;--y:36%;--s:180px;--c:rgba(47,129,247,.24);animation:pf-impact-a 18s ease-out infinite}
        .pf-wave--2{--x:72%;--y:61%;--s:150px;--c:rgba(99,102,241,.24);animation:pf-impact-b 23s ease-out infinite}
        .pf-wave--3{--x:25%;--y:58%;--s:130px;--c:rgba(16,185,129,.22);animation:pf-impact-c 19s ease-out infinite}
        .pf-wave--4{--x:58%;--y:20%;--s:115px;--c:rgba(253,185,39,.2);animation:pf-impact-a 16s ease-out infinite reverse}
        .pf-wave--5{--x:35%;--y:78%;--s:145px;--c:rgba(139,92,246,.2);animation:pf-impact-b 25s ease-out infinite reverse}
        .pf-bg-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(10,10,10,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(10,10,10,.035) 1px,transparent 1px);background-size:60px 60px}
        .pf-bg-noise{position:absolute;inset:-50%;width:200%;height:200%;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.045'/%3E%3C/svg%3E");background-size:256px 256px;opacity:.38}
        @keyframes pf-bubble-path-a{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}22%{transform:translate3d(22vw,12vh,0) scale(1.04,.96)}34%{transform:translate3d(34vw,24vh,0) scale(1.17,.84);border-radius:47% 53% 55% 45%;filter:saturate(1.24) brightness(1.05)}46%{transform:translate3d(25vw,39vh,0) scale(.93,1.08)}72%{transform:translate3d(-4vw,49vh,0) scale(1.02)}}
        @keyframes pf-bubble-path-b{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}20%{transform:translate3d(-18vw,18vh,0) scale(.98,1.04)}37%{transform:translate3d(-28vw,30vh,0) scale(.84,1.16);border-radius:55% 45% 43% 57%;filter:saturate(1.22) brightness(1.04)}58%{transform:translate3d(-8vw,53vh,0) scale(1.08,.95)}80%{transform:translate3d(8vw,24vh,0) scale(.96,1.03)}}
        @keyframes pf-bubble-path-c{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}18%{transform:translate3d(14vw,16vh,0) scale(1.03)}42%{transform:translate3d(4vw,32vh,0) scale(1.14,.88);border-radius:45% 55% 50% 50%;filter:saturate(1.2) brightness(1.06)}64%{transform:translate3d(-22vw,24vh,0) scale(.9,1.1)}86%{transform:translate3d(-8vw,-2vh,0) scale(1.04,.98)}}
        @keyframes pf-bubble-path-d{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}26%{transform:translate3d(-15vw,-18vh,0) scale(.94,1.07)}44%{transform:translate3d(-33vw,-8vh,0) scale(1.16,.86);border-radius:50% 50% 57% 43%;filter:saturate(1.2) brightness(1.04)}67%{transform:translate3d(-20vw,16vh,0) scale(.95,1.06)}84%{transform:translate3d(5vw,5vh,0) scale(1.02)}}
        @keyframes pf-bubble-path-e{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}24%{transform:translate3d(24vw,-12vh,0) scale(1.03,.98)}39%{transform:translate3d(36vw,-3vh,0) scale(.86,1.16);border-radius:58% 42% 48% 52%;filter:saturate(1.22) brightness(1.05)}62%{transform:translate3d(18vw,14vh,0) scale(1.07,.94)}79%{transform:translate3d(-6vw,6vh,0) scale(.98,1.03)}}
        @keyframes pf-bubble-path-f{0%,100%{transform:translate3d(0,0,0) scale(1);border-radius:50%}21%{transform:translate3d(13vw,-20vh,0) scale(.97,1.04)}36%{transform:translate3d(24vw,-34vh,0) scale(1.15,.87);border-radius:46% 54% 56% 44%;filter:saturate(1.2) brightness(1.04)}57%{transform:translate3d(-9vw,-30vh,0) scale(.92,1.08)}78%{transform:translate3d(-28vw,-7vh,0) scale(1.04,.97)}}
        @keyframes pf-impact-a{0%,29%,51%,100%{opacity:0;transform:translate(-50%,-50%) scale(.18)}34%{opacity:.3;transform:translate(-50%,-50%) scale(.45)}47%{opacity:0;transform:translate(-50%,-50%) scale(1.65)}}
        @keyframes pf-impact-b{0%,33%,55%,100%{opacity:0;transform:translate(-50%,-50%) scale(.2)}38%{opacity:.26;transform:translate(-50%,-50%) scale(.5)}52%{opacity:0;transform:translate(-50%,-50%) scale(1.55)}}
        @keyframes pf-impact-c{0%,36%,58%,100%{opacity:0;transform:translate(-50%,-50%) scale(.2)}42%{opacity:.24;transform:translate(-50%,-50%) scale(.52)}56%{opacity:0;transform:translate(-50%,-50%) scale(1.45)}}
        @media(max-width:720px){.pf-bubble{opacity:calc(var(--o,.16) * .75)}.pf-bubble--11,.pf-bubble--12,.pf-bubble--13,.pf-bubble--14,.pf-bubble--15,.pf-bubble--16{display:none}.pf-collision-wave{display:none}}
        @media(prefers-reduced-motion:reduce){.pf-bubble,.pf-collision-wave,.pf-bg-grid,.pf-bg-noise{animation:none!important}.pf-collision-wave{display:none}.pf-bubble{transform:none!important}}
        </style>
        <div class="pf-bg-layer" aria-hidden="true">
          <div class="pf-bubble pf-bubble--1"></div>
          <div class="pf-bubble pf-bubble--2"></div>
          <div class="pf-bubble pf-bubble--3"></div>
          <div class="pf-bubble pf-bubble--4"></div>
          <div class="pf-bubble pf-bubble--5"></div>
          <div class="pf-bubble pf-bubble--6"></div>
          <div class="pf-bubble pf-bubble--7"></div>
          <div class="pf-bubble pf-bubble--8"></div>
          <div class="pf-bubble pf-bubble--9"></div>
          <div class="pf-bubble pf-bubble--10"></div>
          <div class="pf-bubble pf-bubble--11"></div>
          <div class="pf-bubble pf-bubble--12"></div>
          <div class="pf-bubble pf-bubble--13"></div>
          <div class="pf-bubble pf-bubble--14"></div>
          <div class="pf-bubble pf-bubble--15"></div>
          <div class="pf-bubble pf-bubble--16"></div>
          <div class="pf-collision-wave pf-wave--1"></div>
          <div class="pf-collision-wave pf-wave--2"></div>
          <div class="pf-collision-wave pf-wave--3"></div>
          <div class="pf-collision-wave pf-wave--4"></div>
          <div class="pf-collision-wave pf-wave--5"></div>
          <div class="pf-bg-grid"></div>
          <div class="pf-bg-noise"></div>
        </div>
        """
    )


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

    st.markdown(_build_background_layer_html(), unsafe_allow_html=True)

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
        from views.upload_view import render_upload_view

        render_upload_view()
    elif current_page == PAGE_RESULT:
        from views.result_view import render_result_view

        render_result_view()
    else:
        render_home_view()


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())


if __name__ == "__main__":
    main()
