import json

import streamlit as st

from ai.insights import generate_parent_summary
from memory.student_memory import load_student


_BOX = '''
<div style="background:{bg};border-radius:16px;padding:20px;min-height:170px;">
  <h4 style="margin:0 0 8px">{header}</h4>
  <p style="color:#374151;margin:0;line-height:1.55;">{body}</p>
</div>
'''


def _child_to_student_id(child) -> str | None:
    '''Convert a linked child record into a student memory id.

    Supported formats:
    - {"name": "Alex", "memory_file": "alex.json"}
    - {"student_id": "alex"}
    - "alex"
    - "alex.json"
    '''
    if isinstance(child, dict):
        if child.get("student_id"):
            return str(child["student_id"]).strip()

        memory_file = str(child.get("memory_file", "")).strip()
        if memory_file.endswith(".json"):
            return memory_file[:-5]

        if child.get("name"):
            return str(child["name"]).strip().lower()

    if isinstance(child, str):
        value = child.strip()
        if value.endswith(".json"):
            return value[:-5]
        return value.lower()

    return None


def _get_linked_child_ids() -> list[str]:
    children = st.session_state.get("children") or []

    child_ids = []
    for child in children:
        student_id = _child_to_student_id(child)

        if student_id and student_id != "demo_users":
            child_ids.append(student_id)

    return list(dict.fromkeys(child_ids))


@st.cache_data(show_spinner=False, ttl=600)
def _cached_parent_summary(student_json: str, student_id: str) -> dict:
    return generate_parent_summary(
        json.loads(student_json),
        student_id=student_id,
    )


def _student_json_for_agent(student: dict) -> str:
    compact = {
        "student_id": student.get("student_id"),
        "name": student.get("name"),
        "current_topic": student.get("current_topic"),
        "current_signals": student.get("current_signals") or {},
        "signals_history": (student.get("signals_history") or [])[-5:],
        "mastery": student.get("mastery") or {},
        "mastery_history": (student.get("mastery_history") or [])[-5:],
        "sessions": (student.get("sessions") or [])[-3:],
        "weekly_summary": student.get("weekly_summary") or {},
        "context_profile": student.get("context_profile") or student.get("learner_profile") or {},
    }

    return json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
    )


def render():
    student_ids = _get_linked_child_ids()

    if not student_ids:
        st.warning(
            "No child is linked to this parent account. "
            "Add a children field for this account in data/memory/demo_users.json."
        )
        st.code(
            '''"children": [
  {
    "name": "Alex",
    "memory_file": "alex.json"
  }
]''',
            language="json",
        )
        return

    student_records = {
        student_id: load_student(student_id)
        for student_id in student_ids
    }

    active_student_id = st.session_state.get(
        "active_student_id",
        st.session_state.get("student_id", student_ids[0]),
    )

    if active_student_id in student_ids:
        default_index = student_ids.index(active_student_id)
    else:
        default_index = 0

    selected_student_id = st.selectbox(
        "Select a child",
        options=student_ids,
        index=default_index,
        format_func=lambda student_id: (
            student_records[student_id].get(
                "name",
                student_id.capitalize(),
            )
        ),
        key="parent_child_selector",
    )

    student_id = selected_student_id
    student = student_records[student_id]
    report = _cached_parent_summary(
        _student_json_for_agent(student),
        student_id,
    )

    st.caption(
        f"Viewing report from: data/memory/{student_id}.json"
    )

    st.subheader("Parent View: Weekly Learning Summary")
    st.caption("AI-generated summary based on recent tutoring activity")
    st.write(report["headline"])
    st.write("")
    focus_plain_language = report.get("focus_plain_language")

    if focus_plain_language:
        st.info(focus_plain_language)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            _BOX.format(
                bg="#dcfce7",
                header="✅ What went well",
                body=_clean_text(report["what_went_well"]),
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            _BOX.format(
                bg="#fef9c3",
                header="🎯 Next practice focus",
                body=_clean_text(report["needs_support"]),
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            _BOX.format(
                bg="#dbeafe",
                header="🏠 Try at home",
                body=_clean_text(report["try_at_home"]),
            ),
            unsafe_allow_html=True,
        )


    home_practice = report.get("contextualized_home_practice") or {}

    if home_practice:
        st.write("")
        st.subheader("Contextualized Home Practice")
        st.caption(
            "A short home activity adapted from the learner context, recent math focus, and family-friendly examples."
        )

        hp_col1, hp_col2, hp_col3 = st.columns([1, 1.4, 1.2])

        with hp_col1:
            st.markdown(
                _BOX.format(
                    bg="#f0fdf4",
                    header="🌎 Home-friendly context",
                    body=_clean_text(home_practice.get("home_context")),
                ),
                unsafe_allow_html=True,
            )

        with hp_col2:
            st.markdown(
                _BOX.format(
                    bg="#eef2ff",
                    header="💬 Parent question",
                    body=_clean_text(home_practice.get("suggested_parent_question")),
                ),
                unsafe_allow_html=True,
            )

        with hp_col3:
            st.markdown(
                _BOX.format(
                    bg="#fff7ed",
                    header="👂 What to listen for",
                    body=_clean_text(home_practice.get("what_to_listen_for")),
                ),
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("**Why this practice was suggested**")
            st.write(_clean_text(home_practice.get("why_this_context")))
            encouragement_prompt = _clean_text(
                home_practice.get("encouragement_prompt")
            )
            if encouragement_prompt:
                st.info(encouragement_prompt)

    st.markdown(
        f'''
        <div style="background:#f9fafb;border:1px solid #d1d5db;border-radius:16px;
                    padding:20px;margin-top:20px;">
          <h4 style="margin:0 0 8px">💛 Encouragement Note</h4>
          <p style="color:#374151;margin:0;line-height:1.55;">{_clean_text(report["encouragement"])}</p>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("What this summary is based on"):
        st.write(report["basis"])


def _clean_text(value) -> str:
    text = str(value or "").strip()

    replacements = {
        "Â": "",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "—",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())
