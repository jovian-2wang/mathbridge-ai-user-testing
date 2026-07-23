import os
import sys
from dotenv import load_dotenv

load_dotenv()

import streamlit as st


def _read_setting(*names: str, default: str | None = None) -> str | None:
    """Read a setting from environment first, then Streamlit secrets."""
    for name in names:
        value = os.getenv(name)
        if value:
            return str(value)

        try:
            value = st.secrets.get(name, None)
        except Exception:
            value = None

        if value:
            return str(value)

    return default


def _sync_streamlit_secrets_to_env() -> None:
    """Expose Streamlit Cloud secrets to code that uses os.getenv()."""
    keys = [
        "OPENAI_API_KEY",
        "MODEL_NAME",
        "LLM_MODEL",
        "VISUAL_MODEL",
        "VISUAL_PLANNER_MODEL",
        "USE_LLM_VISUAL",
        "EMBEDDING_MODEL",
        "MATHBRIDGE_USE_SKILL_EMBEDDINGS",
        "MATHBRIDGE_EMBEDDING_MODEL",
        "MATHBRIDGE_AUTH_COOKIE_KEY",
        "MATHBRIDGE_AUTH_COOKIE_NAME",
        "MATHBRIDGE_AUTH_COOKIE_DAYS",
        "MATHBRIDGE_PUBLIC_TESTING_MODE",
        "MATHBRIDGE_SHOW_DEMO_SHORTCUTS",
        "MATHBRIDGE_SUPPORT_EMAIL",
    ]

    for key in keys:
        if os.getenv(key):
            continue

        try:
            value = st.secrets.get(key, None)
        except Exception:
            value = None

        if value is not None:
            os.environ[key] = str(value)


_sync_streamlit_secrets_to_env()

from config import APP_TITLE, APP_SUBTITLE  # noqa: E402
from ui import (  # noqa: E402
    class_overview,
    landing,
    login,
    parent,
    student,
    teacher,
)


VIEW_LABELS = {
    "landing": "Overview",
    "student": "Student Chat",
    "teacher": "Teacher Dashboard",
    "class_overview": "Class Overview",
    "parent": "Parent Summary",
}

LABEL_TO_KEY = {
    label: key
    for key, label in VIEW_LABELS.items()
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧮",
    layout="wide",
)

st.sidebar.caption(
    f"Python: {sys.version.split()[0]} | "
    f"Streamlit: {st.__version__}"
)

st.sidebar.caption(
    f"Model: {_read_setting('LLM_MODEL', 'MODEL_NAME', default='not set')}"
)

visual_model = _read_setting(
    "VISUAL_MODEL",
    "VISUAL_PLANNER_MODEL",
    default=None,
)
if not visual_model:
    visual_flag = _read_setting("USE_LLM_VISUAL", default=None)
    visual_model = f"enabled={visual_flag}" if visual_flag else "not set"

st.sidebar.caption(f"Visual planner: {visual_model}")

login.require_login()
login.render_user_panel()

if not _read_setting("OPENAI_API_KEY", default=""):
    st.error(
        "OPENAI_API_KEY not set. Copy `.env.example` to `.env` "
        "and add your key, or set it in Streamlit Cloud Secrets."
    )
    st.stop()

st.title(f"🧮 {APP_TITLE}")
st.caption(APP_SUBTITLE)

if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view in LABEL_TO_KEY:
    st.session_state.view = LABEL_TO_KEY[st.session_state.view]

if st.session_state.view not in VIEW_LABELS:
    st.session_state.view = "landing"

role = st.session_state.role

ROLE_VIEWS = {
    "Student": [
        "landing",
        "student",
    ],
    "Teacher": [
        "landing",
        "teacher",
        "class_overview",
    ],
    "Parent": [
        "landing",
        "parent",
    ],
}

allowed_views = ROLE_VIEWS.get(role, ["landing"])

if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view in LABEL_TO_KEY:
    st.session_state.view = LABEL_TO_KEY[st.session_state.view]

if st.session_state.view not in allowed_views:
    st.session_state.view = allowed_views[0]

view_options = [
    VIEW_LABELS[view_key]
    for view_key in allowed_views
]

current_label = VIEW_LABELS[st.session_state.view]

selected_label = st.radio(
    "nav",
    view_options,
    index=view_options.index(current_label),
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.view = LABEL_TO_KEY[selected_label]

st.divider()

view = st.session_state.view

if view == "landing":
    landing.render()
elif view == "student":
    student.render()
elif view == "teacher":
    teacher.render()
elif view == "class_overview":
    class_overview.render()
elif view == "parent":
    parent.render()
