import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from config import APP_TITLE, APP_SUBTITLE
from ui import (
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
login.require_login()
login.render_user_panel()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY not set. Copy `.env.example` to `.env` "
        "and add your key."
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