"""Session state helpers for Streamlit page flow."""

from __future__ import annotations

import streamlit as st

from config.settings import (
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_INITIAL,
    ANALYSIS_STATUS_SELECTED,
    ANALYSIS_STATUS_SUCCESS,
    PAGE_HOME,
)
from models.paper_result import PaperResult


def init_session_state() -> None:
    defaults = {
        "current_page": PAGE_HOME,
        "analysis_result": None,
        "error_message": "",
        "analysis_status": ANALYSIS_STATUS_INITIAL,
        "parse_feedback": {"status": "idle", "steps": [], "warnings": [], "errors": [], "timings": {}},
        "parse_requested": False,
        "upload_widget_nonce": 0,
        "selected_file_name": "",
        "selected_file_size": 0,
        "selected_file_signature": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_page() -> str:
    return st.session_state.get("current_page", PAGE_HOME)


def set_current_page(page_name: str) -> None:
    st.session_state["current_page"] = page_name


def get_analysis_result() -> PaperResult | None:
    return st.session_state.get("analysis_result")


def set_analysis_result(result: PaperResult | None) -> None:
    st.session_state["analysis_result"] = result


def clear_analysis_result() -> None:
    st.session_state["analysis_result"] = None
    st.session_state["error_message"] = ""


def get_parse_feedback() -> dict[str, object]:
    feedback = st.session_state.get("parse_feedback")
    if isinstance(feedback, dict):
        return feedback
    return {"status": "idle", "steps": [], "warnings": [], "errors": [], "timings": {}}


def set_parse_feedback(feedback: dict[str, object] | None) -> None:
    st.session_state["parse_feedback"] = feedback or {"status": "idle", "steps": [], "warnings": [], "errors": [], "timings": {}}


def clear_parse_feedback() -> None:
    set_parse_feedback(None)


def set_error_message(message: str) -> None:
    st.session_state["error_message"] = message


def get_error_message() -> str:
    return st.session_state.get("error_message", "")


def get_analysis_status() -> str:
    return st.session_state.get("analysis_status", ANALYSIS_STATUS_INITIAL)


def set_analysis_status(status: str) -> None:
    st.session_state["analysis_status"] = status


def is_parse_requested() -> bool:
    return bool(st.session_state.get("parse_requested", False))


def set_parse_requested(requested: bool) -> None:
    st.session_state["parse_requested"] = requested


def get_upload_widget_key() -> str:
    return f"pdf_upload_{st.session_state.get('upload_widget_nonce', 0)}"


def bump_upload_widget_nonce() -> None:
    st.session_state["upload_widget_nonce"] = int(st.session_state.get("upload_widget_nonce", 0)) + 1


def get_selected_file_meta() -> dict[str, object]:
    return {
        "name": st.session_state.get("selected_file_name", ""),
        "size": int(st.session_state.get("selected_file_size", 0) or 0),
        "signature": st.session_state.get("selected_file_signature", ""),
    }


def set_selected_file_meta(*, name: str = "", size: int = 0, signature: str = "") -> None:
    st.session_state["selected_file_name"] = name
    st.session_state["selected_file_size"] = int(size or 0)
    st.session_state["selected_file_signature"] = signature


def reset_upload_flow(*, clear_file: bool = False) -> None:
    st.session_state["error_message"] = ""
    st.session_state["analysis_status"] = ANALYSIS_STATUS_INITIAL
    clear_parse_feedback()
    st.session_state["parse_requested"] = False
    set_selected_file_meta()
    if clear_file:
        bump_upload_widget_nonce()


def mark_file_selected(*, name: str, size: int, signature: str) -> None:
    set_selected_file_meta(name=name, size=size, signature=signature)
    st.session_state["analysis_status"] = ANALYSIS_STATUS_SELECTED
    st.session_state["error_message"] = ""
    clear_parse_feedback()
    st.session_state["parse_requested"] = False


def mark_analysis_failed(message: str, parse_feedback: dict[str, object] | None = None) -> None:
    st.session_state["analysis_status"] = ANALYSIS_STATUS_FAILED
    st.session_state["parse_requested"] = False
    st.session_state["error_message"] = message
    set_parse_feedback(parse_feedback)


def mark_analysis_succeeded(result: PaperResult) -> None:
    st.session_state["analysis_result"] = result
    st.session_state["analysis_status"] = ANALYSIS_STATUS_SUCCESS
    st.session_state["parse_requested"] = False
    st.session_state["error_message"] = ""
    set_parse_feedback(result.parse_feedback_dict())
