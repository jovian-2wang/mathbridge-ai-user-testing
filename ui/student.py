import inspect
import re
import os
import time
import json
from ai.grounding_agent import generate_grounding_panel
import streamlit as st
import streamlit.components.v1 as components
from ai.signals import extract_signals
from ai.tutor_chain import get_tutor_response_payload_with_context
from ai.visual_planner import plan_visual
from ai.visual_renderer import create_visual
from ai.deterministic_visuals import plan_deterministic_visual
from memory.student_memory import (
    get_context_profile,
    list_student_ids,
    load_student,
    save_session,
    update_context_profile,
    update_mastery,
    update_signals,
    update_weekly_summary,
)
from ai.practice_generator import generate_similar_practice_problem
from ai.contextualization import build_contextualization_context, summarize_context_used

def _is_similar_problem_request(user_text: str) -> bool:
    text = str(user_text or "").lower()
    return (
        "similar practice problem" in text
        or "similar problem" in text
    )



def _similar_practice_response() -> str:
    current_problem = str(
        st.session_state.get("current_problem_text", "")
    ).strip()

    signals = st.session_state.get("signals", {}) or {}
    concept = str(signals.get("concept") or "")

    seen_problems = _collect_seen_problem_texts()

    result = generate_similar_practice_problem(
        current_problem=current_problem,
        concept=concept,
        seen_problems=seen_problems,
    )

    problem = result.get("problem", "")
    student_prompt = result.get(
        "student_prompt",
        "What expression would you write to start solving this?",
    )

    if problem:
        st.session_state.current_problem_text = problem
        st.session_state.hint_level = 0
        st.session_state.problem_solved = False
        return (
            "Here is a similar practice problem:\n\n"
            f"{problem}\n\n"
            f"{student_prompt}"
        )

    return student_prompt

def _extract_similar_problem_text(assistant_text: str) -> str:
    lines = [
        line.strip()
        for line in str(assistant_text or "").splitlines()
        if line.strip()
    ]

    selected = []

    for line in lines:
        lower = line.lower()

        if lower.startswith("here is a similar practice problem"):
            continue

        if lower.startswith((
            "how would",
            "how can",
            "what operation",
            "what should",
            "what is the first",
            "how could",
        )):
            break

        if any(ch.isdigit() for ch in line):
            selected.append(line)

    if selected:
        return " ".join(selected).strip()

    return str(assistant_text or "").strip()



def _csv_from_list(value) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())

    if value is None:
        return ""

    return str(value)


def _list_from_csv(value: str) -> list[str]:
    seen = set()
    items = []

    for item in str(value or "").split(","):
        text = item.strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        items.append(text)

    return items


def _options_with_current(options: list[str], current: str) -> list[str]:
    merged = list(options)

    if current and current not in merged:
        merged.append(current)

    return merged


def _render_contextualization_setup(student_id: str, student: dict) -> dict:
    """
    Product-facing setup for the MathBridge contextualization layer.

    This is more than an interests form: it captures learner context,
    course scope, real-world context, and instructional strategy.
    """
    profile = get_context_profile(student)
    learner = profile.get("learner", {})
    curriculum_scope = profile.get("curriculum_scope", {})
    strategy = profile.get("contextualization_strategy", {})

    with st.expander("🧭 Contextualization Setup", expanded=False):
        st.caption(
            "Configure the learner, course scope, real-life contexts, and "
            "instructional strategy used by the contextualization layer."
        )

        with st.form(f"context_profile_form_{student_id}"):
            st.markdown("#### Learner Profile")

            c1, c2, c3 = st.columns(3)

            with c1:
                grade_level = st.text_input(
                    "Grade level",
                    value=str(learner.get("grade_level", "6")),
                )

            with c2:
                language_options = _options_with_current(
                    [
                        "English",
                        "Chinese",
                        "Bilingual English/Chinese",
                        "Spanish",
                    ],
                    str(learner.get("language_preference", "English")),
                )
                language_preference = st.selectbox(
                    "Language preference",
                    options=language_options,
                    index=language_options.index(
                        str(learner.get("language_preference", "English"))
                    ),
                )

            with c3:
                confidence_options = _options_with_current(
                    [
                        "low",
                        "medium",
                        "high",
                    ],
                    str(learner.get("math_confidence", "medium")),
                )
                math_confidence = st.selectbox(
                    "Math confidence",
                    options=confidence_options,
                    index=confidence_options.index(
                        str(learner.get("math_confidence", "medium"))
                    ),
                )

            c4, c5 = st.columns(2)

            with c4:
                reading_level = st.text_input(
                    "Reading level",
                    value=str(learner.get("reading_level", "Grade 6")),
                )

            with c5:
                style_options = [
                    "visual",
                    "step-by-step",
                    "real-life examples",
                    "challenge-based",
                    "slow pacing",
                    "concise explanations",
                ]
                current_styles = [
                    item
                    for item in learner.get("learning_style", [])
                    if item in style_options
                ]
                learning_style = st.multiselect(
                    "Learning style",
                    options=style_options,
                    default=current_styles or [
                        "visual",
                        "step-by-step",
                        "real-life examples",
                    ],
                )

            st.markdown("#### Real-Life Contexts")

            interests_text = st.text_area(
                "Student interests / hobbies",
                value=_csv_from_list(profile.get("interests", [])),
                help="Comma-separated. Example: basketball, baking, games",
            )

            contexts_text = st.text_area(
                "Preferred real-life contexts",
                value=_csv_from_list(profile.get("real_world_contexts", [])),
                help=(
                    "Comma-separated. Example: shopping, food, sports, "
                    "classroom supplies"
                ),
            )

            st.markdown("#### Learning Needs and Curriculum Scope")

            learning_needs_text = st.text_area(
                "Learning needs",
                value=_csv_from_list(profile.get("learning_needs", [])),
                help="Comma-separated. Example: word problems, fractions, unit rates",
            )

            c6, c7 = st.columns(2)

            with c6:
                curriculum_grade = st.text_input(
                    "Curriculum grade",
                    value=str(curriculum_scope.get("grade", "6")),
                )

            with c7:
                curriculum_unit = st.text_input(
                    "Curriculum unit",
                    value=str(
                        curriculum_scope.get(
                            "unit",
                            "Ratios, rates, and fraction reasoning",
                        )
                    ),
                )

            current_objective = st.text_area(
                "Current learning objective",
                value=str(
                    curriculum_scope.get(
                        "current_objective",
                        "Use division reasoning, visual models, and unit-rate "
                        "thinking to solve Grade 6 math problems.",
                    )
                ),
            )

            st.markdown("#### Contextualization Strategy")

            tone = st.text_input(
                "Tone",
                value=str(
                    strategy.get("tone", "friendly Socratic tutor")
                ),
            )

            example_style = st.text_area(
                "Example style",
                value=str(
                    strategy.get(
                        "example_style",
                        "Connect abstract math to familiar real-life scenarios "
                        "when it naturally supports the problem.",
                    )
                ),
            )

            visual_style = st.text_area(
                "Visual style",
                value=str(
                    strategy.get(
                        "visual_style",
                        "Use diagrams before abstract equations.",
                    )
                ),
            )

            avoid_text = st.text_area(
                "Avoid",
                value=_csv_from_list(
                    strategy.get(
                        "avoid",
                        [
                            "revealing final answers too early",
                            "forced personalization",
                            "overly abstract explanations",
                        ],
                    )
                ),
                help="Comma-separated guardrails.",
            )

            submitted = st.form_submit_button(
                "💾 Save Contextualization Profile",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            updated_profile = {
                "learner": {
                    "grade_level": grade_level,
                    "language_preference": language_preference,
                    "reading_level": reading_level,
                    "math_confidence": math_confidence,
                    "learning_style": learning_style,
                },
                "interests": _list_from_csv(interests_text),
                "real_world_contexts": _list_from_csv(contexts_text),
                "learning_needs": _list_from_csv(learning_needs_text),
                "curriculum_scope": {
                    "grade": curriculum_grade,
                    "unit": curriculum_unit,
                    "current_objective": current_objective,
                },
                "contextualization_strategy": {
                    "tone": tone,
                    "example_style": example_style,
                    "visual_style": visual_style,
                    "avoid": _list_from_csv(avoid_text),
                },
            }

            student = update_context_profile(
                student_id,
                updated_profile,
            )
            st.success("Contextualization profile saved.")

        context_used = summarize_context_used(
            student,
            current_problem=st.session_state.get(
                "current_problem_text",
                "",
            ),
        )

        with st.container(border=True):
            st.markdown("#### Context Engine Preview")

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("**Learner context**")
                st.caption(context_used.get("learner_context", "—"))

                st.markdown("**Real-life context**")
                st.caption(context_used.get("real_life_context", "—"))

            with p2:
                st.markdown("**Curriculum context**")
                st.caption(context_used.get("curriculum_context", "—"))

                st.markdown("**Learning needs**")
                st.caption(context_used.get("learning_needs", "—"))

            with st.expander("Prompt-ready context preview", expanded=False):
                st.code(
                    build_contextualization_context(
                        student=student,
                        current_problem=st.session_state.get(
                            "current_problem_text",
                            "",
                        ),
                        curriculum_context="",
                        signals=st.session_state.get("signals", {}),
                    ),
                    language="text",
                )

    return student



def render():
    _init_state()

    student_ids = get_allowed_student_ids()

    if not student_ids:
        load_student("alex")
        student_ids = ["alex"]


    if st.session_state.active_student_id not in student_ids:
        st.session_state.active_student_id = student_ids[0]
        st.session_state.student_id = student_ids[0]


    if (
        "student_selector_v2" not in st.session_state
        or st.session_state.student_selector_v2 not in student_ids
    ):
        st.session_state.student_selector_v2 = (
            st.session_state.active_student_id
        )

    student_records = {
        sid: load_student(sid)
        for sid in student_ids
    }

    if st.session_state.get("role") == "Student":
        selected_student_id = student_ids[0]
        student_name = student_records[selected_student_id].get(
            "name",
            selected_student_id.capitalize(),
        )
        st.caption(f"Student Profile: {student_name}")
    else:
        default_index = (
            student_ids.index(st.session_state.active_student_id)
            if st.session_state.active_student_id in student_ids
            else 0
        )

        selected_student_id = st.selectbox(
            "Student Profile",
            options=student_ids,
            index=default_index,
            format_func=lambda student_id: student_records[student_id].get(
                "name",
                student_id.capitalize(),
            ),
            key="student_selector_v2",
            on_change=_on_student_change,
        )

        st.session_state.active_student_id = selected_student_id
        st.session_state.student_id = selected_student_id


    student_id = st.session_state.active_student_id
    student = load_student(student_id)
    if st.session_state.get("student_runtime_owner") != student_id:
        st.session_state.chat_history = []
        st.session_state.signals = {}
        st.session_state.curriculum_evidence = {}
        st.session_state.session_saved = False
        st.session_state.hint_level = 0
        st.session_state.current_problem_text = ""
        st.session_state.problem_solved = False
        st.session_state.current_topic = None
        st.session_state.support_request_count = 0
        st.session_state.student_runtime_owner = student_id


    st.caption(
        f"Current data file: data/memory/{student_id}.json"
    )

    student = _render_contextualization_setup(
        student_id,
        student,
    )

    col_chat, col_signals = st.columns([2.2, 1])

    with col_chat:
        signals = st.session_state.get("signals") or {}
        topic = (
            signals.get("concept")
            or student.get("current_topic")
            or "Fractions"
        )

        st.subheader("Student Chat: Math Support")
        signals = st.session_state.get("signals") or student.get("signals") or {}

        topic = (
            st.session_state.get("current_topic")
            or signals.get("concept")
            or student.get("current_topic")
            or "Fractions"
        )
        st.markdown(
            f"**Today's Topic:** {topic} &nbsp;&nbsp; "
            "`Aligned with class lesson`"
        )

        
        st.write("")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                visual_type = message.get("visual_type", "none")
                visual_data = message.get("visual_data", {})
                needs_visual = bool(message.get("needs_visual", False))

                if not isinstance(visual_data, dict):
                    visual_data = {}

                if message.get("role") == "assistant" and os.getenv("DEBUG_VISUAL", "0") == "1":
                    with st.expander("DEBUG visual payload", expanded=False):
                        st.write("needs_visual:", needs_visual)
                        st.write("visual_type:", visual_type)
                        st.write("visual_data keys:", list(visual_data.keys()))
                        st.write("reason:", message.get("reason"))
                        st.write("reveals_answer:", message.get("reveals_answer"))

                if needs_visual and visual_type == "llm_svg":
                    svg = visual_data.get("svg", "")
                    caption = visual_data.get(
                        "caption",
                        "AI-generated visual explanation",
                    )

                    if svg:
                        _render_svg(svg, height=360)
                        st.caption(caption)
                    else:
                        st.warning("DEBUG: llm_svg was requested, but SVG is empty.")

                elif needs_visual and visual_type not in {"none", "llm_svg"}:
                    image = create_visual(
                        visual_type,
                        visual_data,
                    )

                    if image is not None:
                        st.image(
                            image,
                            caption=message.get(
                                "image_caption",
                                "Visual explanation",
                            ),
                            use_container_width=True,
                        )
        # Students request help naturally through the chat input.
        # Keep the practice-generation control centered and separate from
        # hint/support counting, so Hint usage reflects authentic typed requests.
        _left_spacer, b_similar, _right_spacer = st.columns([1, 1.2, 1])

        with b_similar:
            if st.button(
                "🔄 Similar problem",
                use_container_width=True,
            ):
                _reply(
                    "Give me a similar practice problem."
                )

        st.write("")

        if st.button(
            "✅ End Session & Update Reports",
            use_container_width=True,
            type="primary",
        ):
            _end_session()

        if st.session_state.get("session_saved"):
            if st.button(
                "🆕 Start New Session",
                use_container_width=True,
            ):
                _start_new_session()

    with col_signals:
        _render_signals()
        _render_curriculum_evidence()

    if prompt := st.chat_input(
        "Type your answer or question here..."
    ):
        _reply(prompt)

def get_allowed_student_ids():
    role = st.session_state.get("role")
    memory_file = st.session_state.get("memory_file")
    username = st.session_state.get("username")

    if role == "Student":
        if memory_file:
            return [memory_file.replace(".json", "")]
        if username:
            return [username]
        return ["alex"]

    return list_student_ids()

def _init_state():

    if "active_student_id" not in st.session_state:
        st.session_state.active_student_id = (
            st.session_state.get("student_id", "alex")
        )


    st.session_state.student_id = (
        st.session_state.active_student_id
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "signals" not in st.session_state:
        st.session_state.signals = {}

    if "session_saved" not in st.session_state:
        st.session_state.session_saved = False

    if "student_session_cache" not in st.session_state:
        st.session_state.student_session_cache = {}

    if "curriculum_evidence" not in st.session_state:
        st.session_state.curriculum_evidence = {}

    if "hint_level" not in st.session_state:
        st.session_state.hint_level = 0

    if "current_problem_text" not in st.session_state:
        st.session_state.current_problem_text = ""
    if "problem_solved" not in st.session_state:
        st.session_state.problem_solved = False

    if "context_used" not in st.session_state:
        st.session_state.context_used = {}

    if "support_request_count" not in st.session_state:
        st.session_state.support_request_count = 0


def _save_student_session_state(student_id: str):
    """Save the selected student's temporary browser state."""
    st.session_state.student_session_cache[student_id] = {
        "chat_history": [
            dict(message)
            for message in st.session_state.get(
                "chat_history",
                [],
            )
        ],
        "curriculum_evidence": dict(
            st.session_state.get("curriculum_evidence", {})
        ),
        "signals": dict(
            st.session_state.get("signals", {})
        ),
        "session_saved": bool(
            st.session_state.get(
                "session_saved",
                False,
            )
        ),
        "hint_level": int(
            st.session_state.get(
                "hint_level",
                0,
            )
        ),
        "current_problem_text": str(
            st.session_state.get(
                "current_problem_text",
                "",
            )
        ),
        "problem_solved": bool(
            st.session_state.get(
                "problem_solved",
                False,
            )
        ),
        "context_used": dict(
            st.session_state.get("context_used", {})
        ),
        "support_request_count": int(
            st.session_state.get("support_request_count", 0)
        ),
    }


def _on_student_change():
    """
    Called automatically before Streamlit reruns the page
    after the selector changes.
    """
    old_student_id = st.session_state.active_student_id
    new_student_id = st.session_state.student_selector_v2

    if old_student_id == new_student_id:
        return


    _save_student_session_state(old_student_id)


    new_state = st.session_state.student_session_cache.get(
        new_student_id,
        {},
    )

    st.session_state.chat_history = [
        dict(message)
        for message in new_state.get(
            "chat_history",
            [],
        )
    ]
    
    st.session_state.signals = dict(
        new_state.get("signals", {})
    )

    st.session_state.curriculum_evidence = dict(
        new_state.get("curriculum_evidence", {})
    )

    st.session_state.session_saved = bool(
        new_state.get(
            "session_saved",
            False,
        )
    )

    st.session_state.hint_level = int(
        new_state.get(
            "hint_level",
            0,
        )
    )

    st.session_state.current_problem_text = str(
        new_state.get(
            "current_problem_text",
            "",
        )
    )
    st.session_state.problem_solved = bool(
        new_state.get("problem_solved", False)
    )

    st.session_state.context_used = dict(
        new_state.get("context_used", {})
    )

    st.session_state.support_request_count = int(
        new_state.get("support_request_count", 0)
    )

    st.session_state.active_student_id = new_student_id
    st.session_state.student_id = new_student_id



_HINT_CONTROL_PROMPTS = {
    "give me one hint, but do not give the final answer.",
    "give me a similar practice problem.",
}

def _collect_seen_problem_texts() -> list[str]:
    seen = []

    current_problem = st.session_state.get("current_problem_text", "")
    if current_problem:
        seen.append(str(current_problem))

    for message in st.session_state.get("chat_history", []):
        content = str(message.get("content", ""))

        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            if "?" in line and (
                "how many" in line.lower()
                or "divided" in line.lower()
                or "unit rate" in line.lower()
                or "per" in line.lower()
            ):
                seen.append(line)

    return seen

def _current_problem_text() -> str:
    return str(
        st.session_state.get("current_problem_text", "")
    ).lower()


def _problem_mentions_fraction() -> bool:
    text = _current_problem_text()

    return bool(
        re.search(
            r"\d+/\d+|fraction|half|third|quarter|fourth|eighth|decimal",
            text,
        )
    )

def _problem_is_measurement_division() -> bool:
    text = _current_problem_text()

    return bool(
        re.search(
            r"\bhow many groups of\b|\bhow many sets of\b|\bgroups of\s+\d+\b|\bsets of\s+\d+\b|\bcan be made from\b",
            text,
        )
    )



def _normalize_for_problem_tracking(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )
def _is_bare_division_expression(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*-?\d+(?:\.\d+)?\s*/\s*-?\d+(?:\.\d+)?\s*",
            str(text or ""),
        )
    )


def _division_expression_as_words(text: str) -> str:
    match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*",
        str(text or ""),
    )

    if not match:
        return str(text or "")

    left, right = match.groups()
    return (
        f"The student entered the division expression {left} ÷ {right}. "
        f"Treat it as {left} divided by {right}. "
        "Do not ask the student to write the division expression again. "
        "Guide the student to estimate or compute the quotient step by step without immediately giving the final answer."
    )


def _is_control_prompt(user_text: str) -> bool:
    normalized = _normalize_for_problem_tracking(user_text)
    return normalized in _HINT_CONTROL_PROMPTS

def _infer_concept_from_text(text: str) -> str | None:
    lowered = str(text or "").lower()

    if re.search(
        r"\bcoordinate\b|\bcoordinates\b|\bpoint\s*\(|\bx-coordinate\b|\by-coordinate\b|\bquadrant\b|\bcoordinate plane\b",
        lowered,
    ):
        return "coordinate plane"

    if re.search(
        r"\bratios?\b|\bratio\b|\bcompare two quantities\b|\b\d+\s*:\s*\d+\b",
        lowered,
    ):
        return "ratios"

    if re.search(r"\bunit rate\b|\bper\b|\bper one\b|\bper hour\b|\bper item\b", lowered):
        return "unit rates"

    if re.search(r"\bfraction\b|\bfractions\b|\b\d+\s*/\s*\d+\b", lowered):
        return "fractions"

    if re.search(r"\bdecimal\b|\bdecimals\b|\bpercent\b|\bpercentage\b", lowered):
        return "fraction-decimal relationships"

    if re.search(r"\bdivision\b|\bdivide\b|\bdivided\b|\b÷\b|\bhow many groups\b", lowered):
        return "division reasoning"

    return None
def _looks_like_explicit_new_problem_prompt(user_text: str) -> bool:
    text = str(user_text or "").strip()
    lowered = text.lower()

    if not text:
        return False

    explicit_starters = (
        "what is",
        "evaluate",
        "solve",
        "find",
        "calculate",
        "simplify",
        "how many",
        "how much",
    )

    if lowered.startswith(explicit_starters):
        return True

    if lowered.startswith("if ") and "?" in lowered:
        return True

    if text.endswith("?") and len(text) >= 8:
        return True

    return False
def _looks_like_full_math_problem(user_text: str) -> bool:
    text = str(user_text or "").strip()
    lowered = text.lower()

    if len(text) < 8:
        return False

    # Explicit classroom problem verbs.
    problem_markers = (
        "what is",
        "evaluate",
        "solve",
        "find",
        "calculate",
        "simplify",
        "how many",
        "how much",
        "cost",
        "when",
        "if",
    )

    if any(marker in lowered for marker in problem_markers):
        return True

    # Expressions / equations with variables or operators.
    has_math_symbol = bool(
        re.search(r"[+\-*/÷=]", text)
    )

    has_variable = bool(
        re.search(r"\b[a-zA-Z]\b", text)
    )

    has_number = bool(
        re.search(r"\d", text)
    )

    if has_math_symbol and has_number:
        return True

    if has_variable and has_number and ("=" in text or "+" in text or "-" in text):
        return True

    if text.endswith("?"):
        return True

    return False

def _looks_like_new_student_problem(user_text: str) -> bool:
    """
    Detect whether the student typed a new problem/question.

    This is only for Socratic state tracking. It does not decide the math
    concept, visual type, or curriculum source.
    """

    if _is_control_prompt(user_text):
        return False
    if _looks_like_full_math_problem(user_text):
        return True

    normalized = _normalize_for_problem_tracking(user_text)
    if _infer_concept_from_text(normalized):
        return True
    if _is_bare_division_expression(user_text):
        return True
    if len(normalized) < 8:
        return False

    if "?" in normalized:
        return True

    problem_starters = (
        "what is",
        "how many",
        "how much",
        "why",
        "find",
        "solve",
        "calculate",
        "a car",
        "a student",
        "a recipe",
        "if ",
    )

    return any(starter in normalized for starter in problem_starters)


def _similarity(a: str, b: str) -> float:
    import difflib

    a_norm = _normalize_for_problem_tracking(a)
    b_norm = _normalize_for_problem_tracking(b)

    if not a_norm or not b_norm:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()


def _maybe_reset_hint_level_for_new_problem(user_text: str) -> int:
    """
    Return the hint level to use for the current reply.

    New typed problems reset hint_level to 0 so a previous problem does not
    unlock answer-revealing visuals for a new problem.
    """

    current_problem = st.session_state.get(
        "current_problem_text",
        "",
    )

    if not _looks_like_new_student_problem(user_text):
        return int(st.session_state.get("hint_level", 0))

    if not current_problem:
        st.session_state.current_problem_text = user_text
        st.session_state.hint_level = 0
        st.session_state.problem_solved = False
        return 0
    

    if _similarity(user_text, current_problem) < 0.55:
        st.session_state.current_problem_text = user_text
        st.session_state.hint_level = 0
        st.session_state.problem_solved = False
        return 0
    

    return int(st.session_state.get("hint_level", 0))


def _is_new_problem_turn(user_text: str, previous_problem_text: str) -> bool:
    """Return True only when this turn should start a new problem.

    This prevents a freshly typed problem from being sent to the answer
    evaluator as if it were the student's answer to itself.
    """

    if not _looks_like_new_student_problem(user_text):
        return False

    if not previous_problem_text:
        return True

    return _similarity(user_text, previous_problem_text) < 0.55


def _effective_user_text_for_agent(user_text: str) -> str:
    """
    Build the text sent to the tutor / retriever / visual planner.

    If the student is responding to an existing problem, include that problem
    as context so short replies like "3" are not treated as a new standalone
    math problem.
    """

    if _is_bare_division_expression(user_text):
        return _division_expression_as_words(user_text)

    current_problem = st.session_state.get(
        "current_problem_text",
        "",
    )

    if current_problem and _is_control_prompt(user_text):
        return (
            "Current math problem:\n"
            f"{current_problem}\n\n"
            "Student request:\n"
            f"{user_text}"
        )

    if current_problem and not _looks_like_new_student_problem(user_text):
        return (
            "Current math problem:\n"
            f"{current_problem}\n\n"
            "Student response:\n"
            f"{user_text}\n\n"
            "Respond to the student in the context of the current problem."
        )

    return user_text

def _append_user_turn_once(content: str) -> None:
    history = st.session_state.setdefault("chat_history", [])
    if (
        history
        and history[-1].get("role") == "user"
        and history[-1].get("content") == content
    ):
        return

    history.append({
        "role": "user",
        "content": content,
    })
def _is_hint_or_next_step_request(user_text: str) -> bool:
    normalized = _normalize_for_problem_tracking(user_text)
    return normalized in {
        "give me one hint, but do not give the final answer.",
    }


def _looks_like_answer_attempt_turn(user_text: str) -> bool:
    text = str(user_text or "").strip()
    lowered = text.lower()

    if not text:
        return False

    if _is_control_prompt(text):
        return False

    help_markers = (
        "hint",
        "help",
        "explain",
        "why",
        "i don't understand",
        "i dont understand",
        "not sure",
        "show me",
        "next step",
        "similar problem",
    )

    if any(marker in lowered for marker in help_markers):
        return False

    # Plain numeric answers: 23, =23, 4, 4.0, -2
    if re.fullmatch(r"=?\s*-?\d+(?:\.\d+)?\s*", text):
        return True

    # Numeric answers with units: 23 dollars, 4 groups, 5 notebooks
    if re.fullmatch(
        r"=?\s*-?\d+(?:\.\d+)?\s*[a-zA-Z ]{0,24}",
        text,
    ):
        return True

    # Fraction answers: 1/2, =3/4, 1 1/6
    if re.fullmatch(
        r"=?\s*(?:\d+\s+)?\d+\s*/\s*\d+\s*",
        text,
    ):
        return True

    # Algebraic answers: x=5, a = 5, 5x, 23a
    if re.fullmatch(
        r"[a-zA-Z]\s*=\s*-?\d+(?:\.\d+)?",
        text,
    ):
        return True

    if re.fullmatch(
        r"-?\d+(?:\.\d+)?\s*[a-zA-Z]",
        text,
    ):
        return True

    answer_starters = (
        "the answer is",
        "answer is",
        "it is",
        "it's",
        "i think",
        "i got",
        "my answer is",
    )

    if any(lowered.startswith(starter) for starter in answer_starters):
        return True

    # Short expression answer, such as 4*5+3 or 4×5+3=23.
    if len(text) <= 60 and any(ch.isdigit() for ch in text):
        if any(op in text for op in ["=", "+", "-", "*", "×", "÷"]):
            return True

    return False


def _looks_like_correct_confirmation(text: str) -> bool:
    normalized = str(text or "").lower()

    negative_markers = (
        "not correct",
        "isn't correct",
        "incorrect",
        "not quite",
        "try again",
        "too small",
        "too large",
        "too many",
        "too few",
    )

    if any(marker in normalized for marker in negative_markers):
        return False

    correct_markers = (
        "your answer is correct",
        "that is correct",
        "that's correct",
        "which is correct",
        "is correct.",
        "correct!",
        "exactly right",
        "that's exactly right",
        "that’s exactly right",
        "you found that",
        "nice work",
        "great job",
        "you got it",
        "well done",
    )

    return any(marker in normalized for marker in correct_markers)

def _fast_signals_from_turn(
    user_text: str,
    response_payload: dict,
    retrieval: dict,
    current_hint_level: int,
) -> dict:
    """
    Lightweight per-turn signals for fast demo performance.

    Full signal extraction can still run when the session is ended.
    """
    previous = st.session_state.get("signals", {}) or {}

    text_for_concept = " ".join([
    str(st.session_state.get("current_problem_text", "")),
    str(user_text or ""),
    str(response_payload.get("answer", "")),
    str(retrieval.get("context", "")),
    ])

    detected_concept = _infer_concept_from_text(text_for_concept)

    concept = (
        response_payload.get("concept")
        or retrieval.get("concept")
        or retrieval.get("detected_concept")
        or detected_concept
        or st.session_state.get("current_topic")
        or "New math topic"
    )

    if _is_bare_division_expression(
        st.session_state.get("current_problem_text", "") or user_text
    ):
        concept = "dividing whole numbers by whole numbers"

    support_count = int(st.session_state.get("support_request_count", 0))

    if len(st.session_state.get("chat_history", [])) >= 4:
        engagement = "Moderate"
    else:
        engagement = previous.get("engagement", "Low")

    next_support = (
        response_payload.get("next_support")
        or previous.get("next_support")
        or "Use the tutor's latest question as the next support step."
    )   

    return {
        "concept": concept,
        "misconception": previous.get("misconception") or "—",
        "hints_used": f"{support_count} student help request(s)",
        "engagement": engagement,
        "next_support": next_support,
    }



def _render_svg(svg: str, height: int = 360) -> None:
    components.html(
        svg,
        height=height,
        scrolling=False,
    )

def _looks_like_division_problem_text(text: str) -> bool:
    text = str(text or "").strip().lower()

    if not text:
        return False

    return bool(
        re.search(
            r"\d+(?:\.\d+)?\s*(?:/|÷)\s*\d+(?:\.\d+)?",
            text,
        )
        or re.search(
            r"\d+(?:\.\d+)?\s+divided\s+by\s+\d+(?:\.\d+)?",
            text,
        )
    )

def _reply(user_text: str):
    timings = {}
    total_start = time.perf_counter()

    if _is_similar_problem_request(user_text):
        t0 = time.perf_counter()

        st.session_state.session_saved = False
        _append_user_turn_once(user_text)

        assistant_text = _similar_practice_response()

        similar_problem_text = _extract_similar_problem_text(
            assistant_text
        )

        if similar_problem_text:
            st.session_state.current_problem_text = similar_problem_text
            st.session_state.problem_solved = False
            st.session_state.hint_level = 0

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_text,
            "needs_visual": False,
            "visual_type": "none",
            "visual_data": {},
        })
        student_record_for_context = load_student(
            st.session_state.active_student_id
        )
        st.session_state.context_used = summarize_context_used(
            student_record_for_context,
            current_problem=st.session_state.get(
                "current_problem_text",
                "",
            ),
            selected_context="similar practice problem",
        )

        _save_student_session_state(st.session_state.active_student_id)

        timings["similar_problem"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start
        timings["unaccounted"] = 0.0

        st.session_state.last_timings = dict(timings)
        print("TIMINGS:", timings)

        st.rerun()
        return

    if (
        st.session_state.get("problem_solved")
        and _is_hint_or_next_step_request(user_text)
    ):
        t0 = time.perf_counter()

        st.session_state.session_saved = False
        _append_user_turn_once(user_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "You already solved this problem correctly. "
                "Try a similar practice problem, or type a new math problem to keep practicing."
            ),
            "needs_visual": False,
            "visual_type": "none",
            "visual_data": {},
        })

        _save_student_session_state(st.session_state.active_student_id)

        timings["solved_guard"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start
        timings["unaccounted"] = 0.0

        st.session_state.last_timings = dict(timings)
        print("TIMINGS:", timings)

        st.rerun()
        return

    # -----------------------------
    # Preprocess / centralized turn router
    # -----------------------------
    t0 = time.perf_counter()

    student_id = st.session_state.active_student_id

    previous_problem_text = str(
        st.session_state.get("current_problem_text", "")
    ).strip()

    is_support_only_turn = _is_control_prompt(user_text)

    looks_like_explicit_new_problem = (
        _looks_like_explicit_new_problem_prompt(user_text)
    )

    looks_like_answer_attempt = _looks_like_answer_attempt_turn(
        user_text
    )

    looks_like_new_problem = _is_new_problem_turn(
        user_text,
        previous_problem_text,
    )

    # -------------------------------------------------
    # Central turn router.
    # Every user input is classified exactly once here.
    # -------------------------------------------------
    if is_support_only_turn:
        turn_type = "support"

    elif looks_like_explicit_new_problem:
        turn_type = "new_problem"
    
    elif previous_problem_text:
        turn_type = "response_to_current_problem"

    elif previous_problem_text and looks_like_answer_attempt:
        turn_type = "answer_attempt"

    elif looks_like_new_problem:
        turn_type = "new_problem"

    else:
        turn_type = "normal"

    is_new_problem_turn = turn_type == "new_problem"
    force_answer_attempt = False
    is_response_to_current_problem = turn_type == "response_to_current_problem"

    if is_new_problem_turn:
        st.session_state.current_problem_text = user_text.strip()
        st.session_state.problem_solved = False
        st.session_state.hint_level = 0

    current_problem_text = str(
        st.session_state.get("current_problem_text", "")
    ).strip()

    current_hint_level = int(
        st.session_state.get("hint_level", 0)
    )

    if is_new_problem_turn:
        current_hint_level = 0

    if is_response_to_current_problem:
        agent_user_text = (
            f"Current math problem:\n{current_problem_text}\n\n"
            f"Student message:\n{user_text}\n\n"
            "Respond flexibly in the context of the current problem. "
            "If this is an answer attempt, give feedback. If it is a request "
            "for translation, explanation, clarification, platform help, or a "
            "different language, help the student naturally while preserving the learning goal."
        )

    elif is_support_only_turn and current_problem_text:
        agent_user_text = (
            f"Current math problem:\n{current_problem_text}\n\n"
            f"Student request:\n{user_text}"
        )

    elif is_new_problem_turn and _is_bare_division_expression(user_text):
        agent_user_text = _division_expression_as_words(user_text)

    else:
        agent_user_text = user_text

    st.session_state.session_saved = False

    _append_user_turn_once(user_text)

    timings["preprocess"] = time.perf_counter() - t0



    # -----------------------------
    # Tutor chain
    # -----------------------------
    t0 = time.perf_counter()

    with st.spinner("Thinking..."):
        tutor_params = inspect.signature(
            get_tutor_response_payload_with_context
        ).parameters

        kwargs = {}

        if "hint_level" in tutor_params:
            kwargs["hint_level"] = current_hint_level

        active_problem_for_evaluator = (
            ""
            if is_new_problem_turn
            else st.session_state.get("current_problem_text", "")
        )
        if "current_problem_text" in tutor_params:
            kwargs["current_problem_text"] = active_problem_for_evaluator

        if "force_answer_attempt" in tutor_params:
            kwargs["force_answer_attempt"] = force_answer_attempt

        if "contextualization_context" in tutor_params:
            student_record_for_context = load_student(student_id)
            kwargs["contextualization_context"] = (
                build_contextualization_context(
                    student=student_record_for_context,
                    current_problem=current_problem_text or user_text,
                    curriculum_context="",
                    signals=st.session_state.get("signals", {}),
                )
            )

        response_payload, retrieval = (
            get_tutor_response_payload_with_context(
                agent_user_text,
                st.session_state.chat_history[:-1],
                **kwargs,
            )
        )

    timings["tutor_chain"] = time.perf_counter() - t0
    tutor_inner_timings = response_payload.get("_timings", {})

    if isinstance(tutor_inner_timings, dict):
        for name, seconds in tutor_inner_timings.items():
            if isinstance(seconds, (int, float)):
                timings[f"tutor_{name}"] = seconds

    answer = response_payload.get(
        "answer",
        "Let's work through this together.",
    )

    student_intent = str(
        response_payload.get("student_intent") or ""
    ).strip().lower()

    is_answer_attempt_turn = bool(
        response_payload.get("is_answer_attempt", False)
    )

    answer_is_correct = bool(
        response_payload.get("is_correct_answer", False)
    )

    if not answer_is_correct:
        answer_is_correct = _looks_like_correct_confirmation(answer)

    if answer_is_correct:
        st.session_state.problem_solved = True

    # Hint usage is counted from the tutoring pipeline outcome, not from a
    # hard-coded phrase list. A typed student turn counts as help use when:
    # - the student is working inside an existing problem,
    # - the turn is not a new problem,
    # - the evaluator/tutor did not classify it as an answer attempt, and
    # - the response did not confirm a correct answer.
    # This covers natural messages like "how?", "I have no idea",
    # "can you explain", or a language/accessibility request, while keeping
    # correct answers out of the count.
    counted_as_support_request = bool(
        previous_problem_text
        and turn_type in {"response_to_current_problem", "support"}
        and not is_new_problem_turn
        and not is_answer_attempt_turn
        and not answer_is_correct
        and not _is_similar_problem_request(user_text)
    )

    if counted_as_support_request:
        st.session_state.support_request_count = int(
            st.session_state.get("support_request_count", 0)
        ) + 1

    # Store the routed outcome on the latest user turn so the signal agent can
    # count authentic help use without relying on keyword matching.
    for message in reversed(st.session_state.get("chat_history", [])):
        if message.get("role") == "user":
            message["student_intent"] = student_intent or turn_type
            message["turn_type"] = turn_type
            message["counted_as_support_request"] = counted_as_support_request
            break

    # -----------------------------
    # Visual planner
    # -----------------------------
    t0 = time.perf_counter()

    is_support_only_turn = _is_control_prompt(user_text)
    is_answer_attempt_turn = bool(
        response_payload.get("is_answer_attempt", False)
    )

    if answer_is_correct or force_answer_attempt or is_answer_attempt_turn:
        visual_plan = {
            "needs_visual": False,
            "visual_type": "none",
            "visual_data": {},
            "reason": (
                "No new visual is needed for correct-answer confirmation "
                "or answer-attempt feedback."
            ),
            "reveals_answer": False,
        }
    else:
        # Visual planner should see the math problem, not the wrapped evaluator text.
        visual_user_text = current_problem_text or user_text

    

        contextualization_context_for_visual = kwargs.get(
            "contextualization_context",
            "",
        )

        deterministic_visual_plan = plan_deterministic_visual(
            problem_text=visual_user_text,
            concept=(
                response_payload.get("concept")
                or retrieval.get("concept")
                or retrieval.get("detected_concept")
                or st.session_state.get("signals", {}).get("concept", "")
            ),
            hint_level=current_hint_level,
            contextualization_context=contextualization_context_for_visual,
        )

        if deterministic_visual_plan.get("needs_visual"):
            visual_plan = deterministic_visual_plan
        else:
            visual_plan = plan_visual(
                user_text=visual_user_text,
                tutor_reply=answer,
                curriculum_context=retrieval.get("context", ""),
                hint_level=current_hint_level,
            )

    timings["visual_plan"] = time.perf_counter() - t0

    visual_type = visual_plan.get("visual_type", "none")
    visual_data = visual_plan.get("visual_data", {})
    needs_visual = bool(visual_plan.get("needs_visual", False))

    if not isinstance(visual_data, dict):
        visual_data = {}

    student_record_for_context = load_student(student_id)
    context_used = summarize_context_used(
        student_record_for_context,
        current_problem=current_problem_text or user_text,
    )

    if retrieval.get("retrieval_method") or retrieval.get("matches"):
        context_used["curriculum_context"] = (
            f"{context_used.get('curriculum_context', 'Grade 6 math')} · "
            f"retrieval: {retrieval.get('retrieval_method', 'available')}"
        )

    context_used["learning_state"] = (
        f"Concept: {st.session_state.get('signals', {}).get('concept', 'not available')}; "
        f"support requests: {st.session_state.get('support_request_count', 0)}; "
        f"turn type: {turn_type}."
    )
    visual_template = visual_plan.get("template")
    base_context_decision = context_used.get(
        "personalization_decision",
        "Use context only when it clarifies the math.",
    )

    if visual_plan.get("generated_by") == "Deterministic Visual Template":
        context_used["visual_context"] = (
            f"Reliable deterministic visual template used: {visual_template}."
        )
        context_used["personalization_decision"] = (
            f"{base_context_decision} The tutor was given the learner profile, "
            "course scope, and live learning state. The visual was selected "
            "from a deterministic Grade 6 template for lower latency and "
            "more stable demo behavior."
        )
    else:
        context_used["visual_context"] = "LLM visual planner fallback available."
        context_used["personalization_decision"] = (
            f"{base_context_decision} The tutor was given the learner profile, "
            "course scope, and live learning state, and should preserve "
            "Socratic flow without forcing personalization."
        )
    st.session_state.context_used = dict(context_used)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "needs_visual": needs_visual,
        "visual_type": visual_type,
        "visual_data": visual_data,
        "reason": visual_plan.get("reason", ""),
        "reveals_answer": visual_plan.get("reveals_answer", False),
        "context_used": context_used,
    }

    # -----------------------------
    # Visual validation
    # -----------------------------
    t0 = time.perf_counter()

    if needs_visual and visual_type not in {"none", "llm_svg"}:
        test_image = create_visual(
            visual_type,
            visual_data,
        )

        if test_image is None:
            assistant_message["needs_visual"] = False
            assistant_message["visual_type"] = "none"
            assistant_message["visual_data"] = {}
        else:
            assistant_message["image_caption"] = "Visual explanation"

    timings["visual_validate"] = time.perf_counter() - t0
  

    st.session_state.chat_history.append(
        assistant_message
    )

    st.session_state.hint_level = min(
        current_hint_level + 1,
        3,
    )

    retrieval_matches = retrieval.get("matches", []) or []
    retrieval_context = str(retrieval.get("context", "") or "").strip()
    retrieval_method = str(
        retrieval.get("retrieval_method")
        or retrieval.get("method")
        or ""
    ).lower()

    should_update_curriculum_evidence = bool(
        retrieval_matches
        or (
            retrieval_context
            and not retrieval_method.startswith("skipped")
        )
    )

    if should_update_curriculum_evidence:
        st.session_state.curriculum_evidence = retrieval

    # -----------------------------
    # Signal extraction
    # -----------------------------
    t0 = time.perf_counter()

    use_fast_signals = os.getenv(
        "MATHBRIDGE_FAST_SIGNALS",
        "1"
    ) == "1"

    if use_fast_signals:
        signals = _fast_signals_from_turn(
            user_text=user_text,
            response_payload=response_payload,
            retrieval=retrieval,
            current_hint_level=current_hint_level,
        )
    else:
        student_record = load_student(student_id)

        signals = extract_signals(
            st.session_state.chat_history,
            curriculum_context=retrieval.get("context", ""),
            prior_signals_history=student_record.get("signals_history", []),
        )

    st.session_state.signals = signals

    if signals.get("concept"):
        st.session_state.current_topic = signals["concept"]

    update_signals(student_id, signals)

    timings["signal_extraction"] = time.perf_counter() - t0

    # -----------------------------
    # Save state
    # -----------------------------
    t0 = time.perf_counter()

    _save_student_session_state(student_id)

    timings["save_state"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - total_start

    timed_sum = sum(
        value
        for key, value in timings.items()
        if key != "total"
    )

    timings["unaccounted"] = max(
        0.0,
        timings["total"] - timed_sum,
    )

    timings["is_answer_attempt"] = 1.0 if response_payload.get("is_answer_attempt") else 0.0
    timings["is_correct_answer"] = 1.0 if response_payload.get("is_correct_answer") else 0.0
    st.session_state.last_timings = dict(timings)
    print("TIMINGS:", timings)

    st.rerun()

def _end_session():
    student_id = st.session_state.active_student_id

    chat_history = st.session_state.get(
        "chat_history",
        [],
    )
    signals = st.session_state.get(
        "signals",
        {},
    )
    if os.getenv("MATHBRIDGE_FAST_SIGNALS", "1") == "1":
        student_record = load_student(student_id)

        signals = extract_signals(
            chat_history,
            curriculum_context=st.session_state.get(
                "curriculum_evidence",
                {},
            ).get("context", ""),
            prior_signals_history=student_record.get(
                "signals_history",
                [],
            ),
        )

    st.session_state.signals = signals
    update_signals(student_id, signals)

    if not chat_history:
        st.warning(
            "There is no conversation to save yet."
        )
        return

    if not signals:
        st.warning(
            "No learning signals are available yet. "
            "Please complete at least one tutoring interaction."
        )
        return

    if st.session_state.get("session_saved"):
        st.info(
            "This session has already been saved."
        )
        return

    mastery_update = update_mastery(
        student_id,
        signals,
    )

    save_session(
        student_id,
        {
            "topic": signals.get(
                "concept",
                "Not recorded",
            ),
            "engagement": signals.get(
                "engagement",
                "Not recorded",
            ),
            "signals": signals,
            "chat_history": chat_history,
            "mastery_update": mastery_update,
        },
    )

    update_weekly_summary(student_id)

    st.session_state.session_saved = True

    _save_student_session_state(student_id)

    if mastery_update.get("updated"):
        st.success(
            f"Session saved for {student_id.capitalize()}. "
            f"{mastery_update['skill']}: "
            f"{mastery_update['old_score']}% → "
            f"{mastery_update['new_score']}% "
            f"({float(mastery_update['delta']):+g})"
        )
    else:
        st.success(
            f"Session saved for {student_id.capitalize()}. "
            "No mastery category matched the detected concept."
        )


def _start_new_session():
    student_id = st.session_state.active_student_id

    st.session_state.chat_history = []
    st.session_state.signals = {}
    st.session_state.curriculum_evidence = {}
    st.session_state.session_saved = False
    st.session_state.hint_level = 0
    st.session_state.current_problem_text = ""
    st.session_state.problem_solved = False
    st.session_state.context_used = {}
    st.session_state.support_request_count = 0
    _save_student_session_state(student_id)

    st.rerun()



def _render_context_used():
    student_id = st.session_state.get("active_student_id", "alex")

    try:
        student = load_student(student_id)
    except Exception:
        student = {}

    context_used = st.session_state.get("context_used") or summarize_context_used(
        student,
        current_problem=st.session_state.get("current_problem_text", ""),
    )

    st.markdown("**Context Used**")

    rows = [
        ("Learner context", context_used.get("learner_context", "—")),
        ("Problem context", context_used.get("problem_context", "—")),
        ("Context source", context_used.get("context_source", "—")),
        ("Real-life context", context_used.get("real_life_context", "—")),
        (
            "Learner preferences available",
            context_used.get("learner_preferences_available", "—"),
        ),
        ("Curriculum context", context_used.get("curriculum_context", "—")),
        ("Learning needs", context_used.get("learning_needs", "—")),
        ("Learning state", context_used.get("learning_state", "—")),
        ("Visual context", context_used.get("visual_context", "—")),
        (
            "Personalization decision",
            context_used.get("personalization_decision", "—"),
        ),
    ]

    for label, value in rows:
        st.caption(f"{label}: {value}")

    st.divider()

def _render_signals():
    signals = st.session_state.get(
        "signals",
        {},
    )

    st.subheader("Behind the Scenes")
    st.caption(
        "Signals captured from the session:"
    )

    rows = [
        (
            "Concept",
            signals.get(
                "concept",
                "No current session",
            ),
        ),
        (
            "Possible misconception",
            signals.get("misconception") or "—",
        ),
        (
            "Hint usage",
            signals.get(
                "hints_used",
                "0 student help request(s)",
            ),
        ),
        (
            "Engagement",
            signals.get(
                "engagement",
                "Starting session",
            ),
        ),
        (
            "Next support",
            signals.get("next_support") or "—",
        ),
    ]

    for label, value in rows:
        st.markdown(f"**{label}**")
        st.caption(str(value))
        st.divider()

    _render_context_used()

    if os.getenv("DEBUG_TIMING", "0") == "1":
        st.markdown("**DEBUG timing**")

    timings = st.session_state.get("last_timings", {})

    if not timings:
        st.caption("No timing data yet. Send one student message first.")
    else:
        ordered_names = [
            "total",
            "tutor_chain",
            "tutor_total",
            "tutor_retrieval",
            "tutor_answer_evaluator",
            "tutor_answer_evaluator_skipped",
            "tutor_answer_evaluator_llm_fallback",
            "tutor_hint_agent",
            "tutor_tutor_llm",
            "visual_plan",
            "visual_validate",
            "signal_extraction",
            "save_state",
            "preprocess",
            "unaccounted",
            "similar_problem",
            "solved_guard",
            "is_answer_attempt",
            "is_correct_answer",
            "tutor_has_current_problem",
            "tutor_force_answer_attempt",
            "tutor_answer_evaluator_not_answer",
        ]

        for name in ordered_names:
            if name in timings:
                st.caption(f"{name}: {timings[name]:.2f}s")

    st.divider()

def _shorten_text(text: str, max_chars: int = 260) -> str:
    text = " ".join(str(text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."

def _clean_grounding_excerpt(text: str, max_chars: int = 220) -> str:
    """
    Clean raw curriculum excerpts before showing them in the right panel.

    This only cleans display artifacts:
    - bullet-list hyphens like "- 180 is..."
    - LaTeX fragments like \\underline{\\hspace{1in}}
    - repeated separators
    - very noisy symbol-heavy fragments

    It does NOT change retrieval, scoring, or matching.
    """
    cleaned = " ".join(str(text or "").split())

    if not cleaned:
        return ""

    # Drop separator clutter from extracted curriculum files.
    cleaned = re.sub(r"=+", " ", cleaned)

    # Remove common LaTeX commands but keep readable words/numbers.
    cleaned = re.sub(
        r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?",
        " ",
        cleaned,
    )

    # Remove leftover braces from LaTeX fragments.
    cleaned = cleaned.replace("{", " ").replace("}", " ")

    # Remove image-description fragments if they appear in raw curriculum text.
    cleaned = re.sub(
        r"\[image description:[^\]]*\]",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove bullet hyphens like "- 180 is..." but keep real negatives like "-2".
    cleaned = re.sub(
        r"(^|\s)-\s+(?=[A-Za-z0-9])",
        r"\1",
        cleaned,
    )

    # Remove bullet dots if extracted as "• text".
    cleaned = re.sub(
        r"(^|\s)[•]\s+",
        r"\1",
        cleaned,
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;:,.")

    if not cleaned:
        return ""

    # If the excerpt is mostly symbols/numbers, it is not readable enough
    # for the right-side curriculum panel.
    letters = len(re.findall(r"[A-Za-z]", cleaned))
    symbols = len(re.findall(r"[^A-Za-z0-9\s.,;:!?'-]", cleaned))

    if len(cleaned) >= 40:
        letter_ratio = letters / max(len(cleaned), 1)

        if letter_ratio < 0.35 or symbols > 18:
            return ""

    return _shorten_text(cleaned, max_chars=max_chars)


def _current_detected_concept() -> str:
    signals = st.session_state.get("signals", {}) or {}
    return str(signals.get("concept") or "").lower()






def _is_relevant_grounding(match: dict) -> bool:
    """
    UI-level quality gate.

    Hide weak or pedagogically mismatched curriculum references.
    The grounding panel should not force a lesson if retrieval confidence is low.
    """
    concept = _current_detected_concept()

    score = match.get(
        "similarity",
        match.get("score", None),
    )

    # Do not show low-confidence grounding as if it were useful.
    if isinstance(score, (int, float)) and score < 0.28:
        return False

    source = str(match.get("source", "")).lower()
    excerpt = str(match.get("excerpt", "")).lower()
    text = f"{source} {excerpt}"
    text = text.replace("_", " ").replace("-", " ")
    # If the current problem is not about fractions, do not show fraction-only
# curriculum lessons in the main grounding panel.
    if not _problem_mentions_fraction():
        fraction_only_terms = (
            "unit fraction",
            "unit fractions",
            "non unit fraction",
            "non unit fractions",
            "fraction of a group",
            "fractional length",
            "dividing by unit and non unit fractions",
            "dividing fractions",
            "divide fractions",
            "fraction division",
            "multiply by reciprocal",
            "reciprocal",
        )
        
        if any(term in text for term in fraction_only_terms):
            return False


    # Whole-number division / equal-sharing / measurement-division.
    if "division with whole numbers" in concept or "whole number" in concept:
        allowed_terms = (
            "meaning of division",
            "meanings of division",
            "division situation",
            "division situations",
            "how many groups",
            "how much in each group",
            "equal group",
            "equal groups",
            "each group",
            "shared equally",
            "quotient",
            "groups of",
            "size of each group",
        )

        blocked_terms = (
            "fraction of a group",
            "unit fraction",
            "unit fractions",
            "non unit fraction",
            "non unit fractions",
            "dividing by unit and non unit fractions",
            "dividing fractions",
            "divide fractions",
            "fraction division",
            "multiply by reciprocal",
            "reciprocal",
            "fractional length",
            "triangle",
            "prism",
            "volume",
            "decimal",
        )

        if any(term in text for term in blocked_terms):
            return False

        return any(term in text for term in allowed_terms)

    if "unit rate" in concept:
        allowed_terms = (
            "unit rate",
            "rate",
            "per one",
            "per hour",
            "per item",
            "for each",
            "how much in each",
        )
        return any(term in text for term in allowed_terms)

    if "fraction" in concept and "division" in concept:
        blocked_terms = (
            "volume",
            "prism",
            "triangle",
        )
        return not any(term in text for term in blocked_terms)

    return True


def _infer_grounding_focus(source: str, excerpt: str) -> str:
    text = f"{source} {excerpt}".lower()
    concept = _current_detected_concept()

    if _problem_is_measurement_division():
        if "how many groups" in text or "number of groups" in text:
            return (
                "Understanding measurement division: finding how many groups "
                "can be made when the group size is known."
            )

        return (
            "Supporting measurement-division reasoning: the total amount and "
            "group size are known, and the unknown is the number of groups."
        )

    if "division with whole numbers" in concept or "whole number" in concept:
        if "how much in each group" in text or "each group" in text:
            return "Understanding equal sharing: finding how many items go in one group."

        if "meaning" in text and "division" in text:
            return "Interpreting the division situation before calculating."

        if "how many groups" in text or "number of groups" in text:
            return (
                "Related division model: reasoning about groups. "
                "This is supporting context, but the current problem asks for the amount in each group."
            )

        return "Supporting whole-number division reasoning."

    if "unit rate" in concept:
        return "Finding an amount per one unit."

    if "fraction" in concept and "division" in concept:
        return "Connecting division reasoning with fraction models."

    if "ratio" in concept:
        return "Using ratios to compare two quantities."

    return "Closest available curriculum context for the current math question."


def _clean_source_name(source: str) -> str:
    source = str(source or "Unknown source")
    source = source.replace(".txt", "")
    source = source.replace("grade6_unit4_", "")
    source = source.replace("grade6_", "")
    source = source.replace("lesson", "Lesson ")
    source = source.replace("_", " ")
    source = source.replace("-", " ")
    source = " ".join(source.split())
    return source.title()

def _stable_json(data: dict) -> str:
    return json.dumps(
        data or {},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_grounding_panel(
    current_problem: str,
    detected_concept: str,
    evidence_json: str,
) -> dict:
    evidence = json.loads(evidence_json)

    return generate_grounding_panel(
        current_problem=current_problem,
        detected_concept=detected_concept,
        retrieval_evidence=evidence,
    )

def _confidence_from_score(score) -> str:
    if not isinstance(score, (int, float)):
        return "moderate"

    if score < 0.28:
        return "low"

    if score < 0.45:
        return "moderate"

    return "high"


def _direct_grounding_panel(evidence: dict) -> dict:
    matches = evidence.get("matches", []) or []
    selected = []

    for match in matches:
        if not _is_relevant_grounding(match):
            continue

        source = match.get("source", "Unknown source")
        excerpt = match.get("excerpt", "")
        score = match.get(
            "dense_similarity",
            match.get(
                "similarity",
                match.get("score", None),
            ),
        )
        if isinstance(score, (int, float)) and score < 0.45:
            continue


        selected.append(
            {
                "source": source,
                "score": score,
                "confidence": _confidence_from_score(score),
                "focus": _infer_grounding_focus(
                    source=source,
                    excerpt=excerpt,
                ),
                "why_this_matches": _clean_grounding_excerpt(
                    excerpt,
                    max_chars=220,
                ),
            }
        )

        if len(selected) >= 2:
            break

    if not selected:
        return {
            "generated_by": "Direct Retrieval Display",
            "overall_status": "no_alignment",
            "message": (
                "No high-confidence curriculum reference matched this turn. "
                "Raw retrieved candidates are available below."
            ),
            "selected_references": [],
            "retrieval_method": evidence.get(
                "retrieval_method",
                "unknown",
            ),
        }

    return {
        "generated_by": "Direct Retrieval Display",
        "overall_status": "strong_alignment",
        "message": (
            "Here are curriculum references selected directly from retrieval "
            "for faster demo performance."
        ),
        "selected_references": selected,
        "retrieval_method": evidence.get(
            "retrieval_method",
            "unknown",
        ),
    }



def _render_curriculum_evidence():
    evidence = st.session_state.get("curriculum_evidence", {}) or {}
    matches = evidence.get("matches", [])

    st.subheader("Curriculum Grounding")
    st.caption("Curriculum references selected from retrieval evidence.")

    if not matches:
        st.info("No curriculum source matched the latest message yet.")
        return

    signals = st.session_state.get("signals", {}) or {}
    detected_concept = str(signals.get("concept") or "")

    current_problem = str(
        st.session_state.get("current_problem_text", "")
    )

    use_grounding_agent = os.getenv(
        "MATHBRIDGE_USE_GROUNDING_AGENT",
        "0",
    ) == "1"

    if use_grounding_agent:
        evidence_json = _stable_json(evidence)

        grounding = _cached_grounding_panel(
            current_problem=current_problem,
            detected_concept=detected_concept,
            evidence_json=evidence_json,
        )
    else:
        grounding = _direct_grounding_panel(evidence)

    method = (
        evidence.get("retrieval_method")
        or evidence.get("method")
        or grounding.get("retrieval_method")
        or "unknown"
    )

    st.markdown(f"**Retrieval method:** {method}")
    st.caption(
        f"Grounding generated by: "
        f"{grounding.get('generated_by', 'Direct Retrieval Display')}"
    )

    status = grounding.get("overall_status", "weak_alignment")
    message = grounding.get("message", "")

    if status == "no_alignment":
        st.warning(message)

        with st.expander("Show raw retrieved candidates", expanded=False):
            for index, match in enumerate(matches[:3], start=1):
                source = _clean_source_name(
                    match.get("source", "Unknown source")
                )
                score = match.get(
                    "dense_similarity",
                    match.get(
                        "similarity",
                        match.get("score", None),
                    ),
                )

                if isinstance(score, (int, float)):
                    score_text = f"{score:.3f}"
                else:
                    score_text = "—"

                st.markdown(f"**{index}. {source}**")
                st.caption(f"Raw retrieval score: {score_text}")

        return

    if status == "weak_alignment":
        st.info(message)
    else:
        st.success(message)

    selected = grounding.get("selected_references", [])

    for index, item in enumerate(selected[:2], start=1):
        source = _clean_source_name(
            item.get("source", "Unknown source")
        )
        confidence = item.get("confidence", "moderate")
        focus = item.get("focus", "")
        why = item.get("why_this_matches", "")
        score = item.get("score", None)

        with st.container(border=True):
            st.markdown(f"**{index}. {source}**")

            if isinstance(score, (int, float)):
                st.caption(
                    f"Confidence: {confidence} · score {score:.3f}"
                )
            else:
                st.caption(f"Confidence: {confidence}")

            st.markdown("**Focus**")
            st.write(focus)

            if why:
                st.markdown("**Why this matches**")
                st.write(why)

    st.caption(
        "Note: curriculum grounding is selected from retrieved curriculum "
        "context; it is not the final answer source."
    )