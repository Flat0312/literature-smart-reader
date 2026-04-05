"""Upload and analysis page rendering."""

from __future__ import annotations

import hashlib
import json
from html import escape
from textwrap import dedent

import streamlit as st

from config.settings import PAGE_HOME, PAGE_RESULT
from services.paper_parse_service import PARSE_STEP_DEFINITIONS, ParsePipelineError, parse_uploaded_pdf
from utils.session import (
    clear_analysis_result,
    clear_parse_feedback,
    get_error_message,
    get_parse_feedback,
    get_upload_widget_key,
    mark_analysis_failed,
    mark_file_selected,
    mark_analysis_succeeded,
    reset_upload_flow,
    set_analysis_status,
    set_current_page,
    set_error_message,
    set_parse_feedback,
)


UPLOAD_STATUS_KEY = "upload_v3_status"
UPLOAD_PENDING_SIGNATURE_KEY = "upload_v3_pending_signature"
UPLOAD_LAST_SIGNATURE_KEY = "upload_v3_last_signature"


def render_upload_view() -> None:
    _init_upload_state()

    st.markdown(
        _html_block(
            """
            <div class="up-header">
              <h2 class="up-header__title">上传 PDF 文献</h2>
              <p class="up-header__sub">上传单篇论文 PDF 后，系统会按阶段完成文本提取、标题/作者/关键词识别、结构化摘要、AI 解读和写作提纲整理。</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        _html_block(
            """
            <div class="upload-workbench">
              <div class="upload-workbench__frame">
                <p class="section-kicker">FILE INPUT</p>
                <p class="hero-text">上传页现在会显示文件状态、解析阶段、失败原因和部分成功提示。即使作者或部分 AI 模块没有稳定识别，也不会把整次解析误报成完全失败。</p>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "选择或拖拽 PDF 文件",
        type=["pdf"],
        help="建议上传排版清晰、包含可复制文本的论文 PDF，大小不超过 20 MB。",
        label_visibility="collapsed",
        key=get_upload_widget_key(),
    )

    file_signature = _build_file_signature(uploaded_file)
    _sync_selection_state(uploaded_file, file_signature)
    stage = _get_stage(uploaded_file, file_signature)
    parse_feedback = get_parse_feedback()

    st.markdown(_build_file_chip_html(uploaded_file, stage), unsafe_allow_html=True)

    if stage == "ready" and uploaded_file is not None:
        st.markdown(
            _build_status_card_html(
                title="文件已载入待解析",
                body="文件已经选择成功，可以开始解析。系统将按阶段反馈 PDF 读取、文本提取、作者识别和写作输出生成情况。",
                modifier="ready",
            ),
            unsafe_allow_html=True,
        )

    if get_error_message():
        st.markdown(
            _html_block(
                f"""
                <div class="upload-alert upload-alert--error">
                  <strong>{escape(_error_heading(parse_feedback))}</strong>{escape(get_error_message())}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    if stage == "processing" and uploaded_file is not None:
        st.markdown(
            _build_status_card_html(
                title="正在解析中",
                body="系统正在按阶段读取 PDF、提取文本、识别标题/作者/关键词并生成结果，请稍候。",
                modifier="processing",
            ),
            unsafe_allow_html=True,
        )
        _handle_analysis(uploaded_file, file_signature)
        return

    if stage == "error":
        st.markdown(
            _build_status_card_html(
                title="解析未完成",
                body=_error_summary_text(parse_feedback),
                modifier="error",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(_build_stage_panel_html(parse_feedback), unsafe_allow_html=True)

    action_columns = st.columns([3, 1])
    with action_columns[0]:
        parse_label = "解析中..." if stage == "processing" else "开始解析"
        parse_clicked = st.button(
            parse_label,
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or stage == "processing",
        )
    with action_columns[1]:
        if st.button("返回首页", use_container_width=True):
            set_current_page(PAGE_HOME)
            st.rerun()

    if stage == "error":
        retry_columns = st.columns([2, 1])
        with retry_columns[0]:
            if st.button("重新选择文件", type="primary", use_container_width=True):
                _reset_uploader()
                st.rerun()
        with retry_columns[1]:
            if st.button("清除错误提示", use_container_width=True):
                reset_upload_flow(clear_file=False)
                st.session_state[UPLOAD_STATUS_KEY] = "idle"
                st.rerun()

    if parse_clicked and uploaded_file is not None and stage != "processing":
        if not uploaded_file.name.lower().endswith(".pdf"):
            feedback = _build_terminal_feedback(
                status="failed",
                errors=["文件类型不支持，请上传 PDF。"],
                category="pdf_read_failed",
                error_stage="pdf_read",
            )
            set_parse_feedback(feedback)
            mark_analysis_failed("文件类型不支持，请上传 PDF。", feedback)
            st.session_state[UPLOAD_STATUS_KEY] = "error"
            st.rerun()
        if not uploaded_file.getvalue():
            feedback = _build_terminal_feedback(
                status="failed",
                errors=["文件为空或无法读取，请重新上传。"],
                category="pdf_read_failed",
                error_stage="pdf_read",
            )
            set_parse_feedback(feedback)
            mark_analysis_failed("文件为空或无法读取，请重新上传。", feedback)
            st.session_state[UPLOAD_STATUS_KEY] = "error"
            st.rerun()

        st.session_state[UPLOAD_STATUS_KEY] = "processing"
        st.session_state[UPLOAD_PENDING_SIGNATURE_KEY] = file_signature
        set_analysis_status("parsing")
        set_error_message("")
        clear_parse_feedback()
        st.rerun()

    st.markdown(
        _html_block(
            """
            <p class="up-hint">
              上传状态会区分：文件已选择、待解析、处理中、解析完成、部分成功和解析失败。扫描版 PDF 或文本极少的文档仍可能在“文本提取”阶段失败。
            </p>
            """
        ),
        unsafe_allow_html=True,
    )


def _handle_analysis(uploaded_file, file_signature: str) -> None:
    clear_analysis_result()
    clear_parse_feedback()
    set_error_message("")

    progress_bar = st.progress(2)
    stage_placeholder = st.empty()
    summary_placeholder = st.empty()

    initial_feedback = _build_running_feedback()
    set_parse_feedback(initial_feedback)
    progress_bar.progress(_progress_from_feedback(initial_feedback))
    stage_placeholder.markdown(_build_stage_panel_html(initial_feedback), unsafe_allow_html=True)

    def on_progress(feedback: dict[str, object]) -> None:
        set_parse_feedback(feedback)
        progress_bar.progress(_progress_from_feedback(feedback))
        stage_placeholder.markdown(_build_stage_panel_html(feedback), unsafe_allow_html=True)
        summary_html = _build_feedback_summary_html(feedback)
        if summary_html:
            summary_placeholder.markdown(summary_html, unsafe_allow_html=True)
        else:
            summary_placeholder.empty()

    try:
        result = parse_uploaded_pdf(uploaded_file.name, uploaded_file.getvalue(), progress_callback=on_progress)
        print("[structured-debug]")
        print(json.dumps(result.structured_debug, ensure_ascii=False, indent=2))
    except ParsePipelineError as exc:
        feedback = exc.parse_feedback or get_parse_feedback()
        set_parse_feedback(feedback)
        mark_analysis_failed(exc.user_message, feedback)
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state.pop(UPLOAD_PENDING_SIGNATURE_KEY, None)
        st.session_state["upload_v3_last_error"] = exc.user_message
        st.rerun()
    except Exception as exc:  # pragma: no cover - unexpected runtime branch
        message = f"解析过程出现错误：{exc}"
        feedback = _build_terminal_feedback(
            status="failed",
            errors=[message],
            category="structured_extract_failed",
            error_stage="structured_extract",
        )
        set_parse_feedback(feedback)
        mark_analysis_failed(message, feedback)
        st.session_state[UPLOAD_STATUS_KEY] = "error"
        st.session_state.pop(UPLOAD_PENDING_SIGNATURE_KEY, None)
        st.session_state["upload_v3_last_error"] = message
        st.rerun()

    mark_analysis_succeeded(result)
    st.session_state[UPLOAD_STATUS_KEY] = "success"
    st.session_state.pop(UPLOAD_PENDING_SIGNATURE_KEY, None)
    st.session_state[UPLOAD_LAST_SIGNATURE_KEY] = file_signature
    st.session_state["upload_v3_last_file_name"] = uploaded_file.name
    st.session_state["upload_v3_last_file_size"] = uploaded_file.size
    st.session_state["upload_v3_last_error"] = ""
    set_current_page(PAGE_RESULT)
    st.rerun()


def _sync_selection_state(uploaded_file, file_signature: str) -> None:
    if uploaded_file is None:
        if st.session_state.get(UPLOAD_STATUS_KEY) != "processing":
            st.session_state[UPLOAD_STATUS_KEY] = "idle"
            set_analysis_status("initial")
            clear_parse_feedback()
        st.session_state.pop(UPLOAD_PENDING_SIGNATURE_KEY, None)
        return

    last_signature = str(st.session_state.get(UPLOAD_LAST_SIGNATURE_KEY, ""))
    pending_signature = str(st.session_state.get(UPLOAD_PENDING_SIGNATURE_KEY, ""))

    if file_signature != last_signature:
        clear_analysis_result()
        clear_parse_feedback()
        set_error_message("")
        mark_file_selected(name=uploaded_file.name, size=uploaded_file.size, signature=file_signature)
        st.session_state[UPLOAD_LAST_SIGNATURE_KEY] = file_signature
        st.session_state["upload_v3_last_file_name"] = uploaded_file.name
        st.session_state["upload_v3_last_file_size"] = uploaded_file.size
        if pending_signature != file_signature:
            st.session_state[UPLOAD_STATUS_KEY] = "ready"


def _get_stage(uploaded_file, file_signature: str) -> str:
    if uploaded_file is None:
        return "idle"
    if st.session_state.get(UPLOAD_PENDING_SIGNATURE_KEY) == file_signature:
        return "processing"
    if get_error_message():
        return "error"
    return str(st.session_state.get(UPLOAD_STATUS_KEY, "ready"))


def _reset_uploader() -> None:
    reset_upload_flow(clear_file=True)
    clear_parse_feedback()
    st.session_state[UPLOAD_STATUS_KEY] = "idle"
    st.session_state.pop(UPLOAD_PENDING_SIGNATURE_KEY, None)
    st.session_state[UPLOAD_LAST_SIGNATURE_KEY] = ""


def _build_file_chip_html(uploaded_file, stage: str) -> str:
    if uploaded_file is None:
        return _html_block(
            """
            <div class="upload-file-chip">
              <strong>尚未选择文件</strong>
              <span>请先上传一篇论文 PDF，再点击“开始解析”。</span>
              <em>等待用户上传文件</em>
            </div>
            """
        )

    size_kb = uploaded_file.size / 1024 if uploaded_file.size else 0
    status_text = {
        "ready": "文件已选择 · 待解析",
        "processing": "正在解析",
        "success": "最近一次解析已完成",
        "error": "解析失败",
    }.get(stage, "准备解析")
    return _html_block(
        f"""
        <div class="upload-file-chip upload-file-chip--active">
          <strong>已选择文件</strong>
          <span>{escape(uploaded_file.name)}</span>
          <em>{size_kb:.1f} KB · {status_text}</em>
        </div>
        """
    )


def _build_stage_panel_html(feedback: dict[str, object]) -> str:
    steps = feedback.get("steps", []) if isinstance(feedback, dict) else []
    if not isinstance(steps, list):
        steps = []
    if not steps:
        steps = _build_running_feedback()["steps"]

    rows: list[str] = []
    for step in steps:
        step_id = escape(str(step.get("id", "")))
        label = escape(str(step.get("label", "")))
        status = escape(str(step.get("status", "pending")))
        detail = escape(str(step.get("detail", "")))
        duration_ms = int(step.get("duration_ms", 0) or 0)
        duration_text = f"{duration_ms} ms" if duration_ms > 0 else ""
        rows.append(
            f"""
            <div class="upload-stage upload-stage--{status}" data-step="{step_id}">
              <span class="upload-stage__dot"></span>
              <div class="upload-stage__content">
                <strong>{label}</strong>
                <span>{detail or _stage_status_text(status)}</span>
              </div>
              <em>{escape(duration_text)}</em>
            </div>
            """
        )

    warnings = feedback.get("warnings", []) if isinstance(feedback.get("warnings"), list) else []
    errors = feedback.get("errors", []) if isinstance(feedback.get("errors"), list) else []
    meta_parts: list[str] = []
    if warnings:
        meta_parts.append(f"warnings {len(warnings)}")
    if errors:
        meta_parts.append(f"errors {len(errors)}")
    timings_html = _build_timing_line_html(feedback)
    meta_text = " · ".join(meta_parts) if meta_parts else "当前正在执行解析流程。"
    return _html_block(
        f"""
        <div class="upload-stage-panel">
          <div class="upload-stage-panel__head">
            <strong>解析阶段进度</strong>
            <span>{escape(str(feedback.get("status", "running")))}</span>
          </div>
          <div class="upload-stage-panel__body">
            {''.join(rows)}
          </div>
          <div class="upload-stage-panel__foot">
            <span>{escape(meta_text)}</span>
            <span>{timings_html}</span>
          </div>
        </div>
        """
    )


def _build_feedback_summary_html(feedback: dict[str, object]) -> str:
    status = str(feedback.get("status", ""))
    warnings = [str(item) for item in feedback.get("warnings", []) if str(item).strip()]
    errors = [str(item) for item in feedback.get("errors", []) if str(item).strip()]
    if status == "failed" and errors:
        return _html_block(
            f"""
            <div class="upload-inline-note upload-inline-note--error">
              <strong>{escape(_error_heading(feedback))}</strong>
              <span>{escape(errors[0])}</span>
            </div>
            """
        )
    if status == "partial_success" and warnings:
        return _html_block(
            f"""
            <div class="upload-inline-note upload-inline-note--warn">
              <strong>部分成功</strong>
              <span>{escape(warnings[0])}</span>
            </div>
            """
        )
    return ""


def _build_status_card_html(*, title: str, body: str, modifier: str) -> str:
    icon = {
        "ready": "✓",
        "processing": "⏳",
        "error": "!",
    }.get(modifier, "•")
    return _html_block(
        f"""
        <div class="upload-state-card upload-state-card--{escape(modifier)}">
          <div class="upload-state-card__icon">{icon}</div>
          <div class="upload-state-card__body">
            <strong>{escape(title)}</strong>
            <span>{escape(body)}</span>
          </div>
        </div>
        """
    )


def _progress_from_feedback(feedback: dict[str, object]) -> int:
    steps = feedback.get("steps", []) if isinstance(feedback.get("steps"), list) else []
    if not steps:
        return 2
    completed_count = sum(1 for step in steps if step.get("status") in {"completed", "partial"})
    if str(feedback.get("status", "")) == "failed":
        completed_count += 1
    ratio = max(completed_count / max(len(steps), 1), 0.02)
    return max(min(int(ratio * 100), 100), 2)


def _build_running_feedback() -> dict[str, object]:
    return {
        "status": "running",
        "steps": [
            {
                "id": step_id,
                "label": label,
                "status": "running" if index == 0 else "pending",
                "detail": "等待执行" if index != 0 else "准备开始解析。",
                "duration_ms": 0,
            }
            for index, (step_id, label) in enumerate(PARSE_STEP_DEFINITIONS)
        ],
        "warnings": [],
        "errors": [],
        "timings": {},
    }


def _build_terminal_feedback(
    *,
    status: str,
    errors: list[str],
    category: str,
    error_stage: str,
) -> dict[str, object]:
    feedback = _build_running_feedback()
    feedback["status"] = status
    feedback["errors"] = errors
    feedback["error_category"] = category
    feedback["error_stage"] = error_stage
    for step in feedback["steps"]:
        if step["id"] == error_stage:
            step["status"] = "failed"
            step["detail"] = errors[0]
            break
        step["status"] = "completed" if step["id"] != "completed" else "pending"
        step["detail"] = "已完成。"
    return feedback


def _build_timing_line_html(feedback: dict[str, object]) -> str:
    timings = feedback.get("timings", {})
    if not isinstance(timings, dict) or not timings:
        return "timing unavailable"
    visible_pairs: list[str] = []
    for key in ("pdf_read_ms", "text_extract_ms", "metadata_extract_ms", "structured_extract_ms", "llm_generate_ms", "total_parse_ms"):
        value = timings.get(key)
        if not isinstance(value, int) or value <= 0:
            continue
        visible_pairs.append(f"{key.replace('_ms', '')}: {value}ms")
    return " · ".join(visible_pairs) if visible_pairs else "timing unavailable"


def _error_heading(feedback: dict[str, object]) -> str:
    mapping = {
        "pdf_read_failed": "PDF 读取失败：",
        "text_extract_failed": "文本提取失败：",
        "metadata_extract_failed": "基础字段识别失败：",
        "structured_extract_failed": "结构化提取失败：",
        "llm_failed": "LLM 调用失败：",
        "export_failed": "导出失败：",
    }
    category = str(feedback.get("error_category", "")).strip()
    return mapping.get(category, "解析失败：")


def _error_summary_text(feedback: dict[str, object]) -> str:
    errors = [str(item) for item in feedback.get("errors", []) if str(item).strip()]
    if errors:
        return errors[0]
    return "系统未能完成本次解析，请根据下方阶段信息查看失败位置并重新尝试。"


def _stage_status_text(status: str) -> str:
    mapping = {
        "pending": "等待执行。",
        "running": "正在执行。",
        "completed": "已完成。",
        "partial": "已完成，但建议人工复核。",
        "failed": "执行失败。",
    }
    return mapping.get(status, "等待执行。")


def _build_file_signature(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.md5(file_bytes).hexdigest()
    return f"{uploaded_file.name}|{uploaded_file.size}|{digest}"


def _init_upload_state() -> None:
    defaults = {
        UPLOAD_STATUS_KEY: "idle",
        UPLOAD_PENDING_SIGNATURE_KEY: "",
        UPLOAD_LAST_SIGNATURE_KEY: "",
        "upload_v3_last_file_name": "",
        "upload_v3_last_file_size": 0,
        "upload_v3_last_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _html_block(markup: str) -> str:
    normalized = dedent(markup).strip()
    return "\n".join(line for line in normalized.splitlines() if line.strip())
