import streamlit as st


_HERO_STYLE = """
<style>
.mathbridge-hero {
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 28px 30px;
    background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 58%, #ecfeff 100%);
    margin-bottom: 20px;
}
.mathbridge-eyebrow {
    color: #2563eb;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-size: 0.78rem;
    margin-bottom: 8px;
}
.mathbridge-title {
    font-size: 2.2rem;
    line-height: 1.1;
    font-weight: 800;
    color: #111827;
    margin: 0 0 10px 0;
}
.mathbridge-subtitle {
    font-size: 1.05rem;
    color: #374151;
    line-height: 1.6;
    max-width: 980px;
    margin: 0;
}
.mathbridge-pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}
.mathbridge-pill {
    border-radius: 999px;
    border: 1px solid #bfdbfe;
    background: #ffffffcc;
    padding: 7px 12px;
    color: #1e3a8a;
    font-size: 0.86rem;
    font-weight: 650;
}
.mathbridge-card {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 18px;
    background: white;
    min-height: 156px;
}
.mathbridge-card h4 {
    margin: 0 0 8px 0;
    font-size: 1.0rem;
    color: #111827;
}
.mathbridge-card p {
    margin: 0;
    color: #4b5563;
    line-height: 1.55;
    font-size: 0.92rem;
}
.mathbridge-step {
    border-left: 4px solid #3b82f6;
    background: #f8fafc;
    border-radius: 14px;
    padding: 13px 15px;
    margin-bottom: 10px;
}
.mathbridge-step strong {
    color: #111827;
}
.mathbridge-muted {
    color: #6b7280;
    font-size: 0.92rem;
}
.mathbridge-codebox {
    background: #0f172a;
    color: #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9rem;
    line-height: 1.55;
    white-space: pre-wrap;
}
</style>
"""


FEATURES = [
    (
        "Learner Context Profile",
        "Collects grade level, language preference, confidence, learning style, interests, real-life contexts, and learning needs.",
    ),
    (
        "Curriculum Grounding",
        "Retrieves Grade 6 curriculum evidence so tutoring stays aligned with the class lesson instead of becoming a generic chatbot.",
    ),
    (
        "Socratic Tutoring + Evaluator",
        "Guides students step by step, checks answer attempts, accepts equivalent forms, and stops hinting once the student is correct.",
    ),
    (
        "Reliable Visual Templates",
        "Uses deterministic diagrams for common ratio, unit-rate, division, fraction, and coordinate-plane problems, with LLM visuals as fallback.",
    ),
    (
        "Learning Signals",
        "Tracks concept, misconception, hint usage, engagement, next support, and mastery movement across sessions.",
    ),
    (
        "Multi-Stakeholder Insights",
        "Turns the same learning record into student support, teacher actions, parent home practice, and class grouping strategies.",
    ),
]


DEMO_PROBLEMS = [
    "A car travels 240 miles in 4 hours. What is the unit rate?",
    "The ratio of students who walk to total students is 10:15. What does the ratio mean?",
    "If 5 notebooks cost $15, what is the cost of one notebook?",
    "How many 1/8 pieces fit in 3/4?",
]


SHOWCASE_STEPS = [
    (
        "1. Configure learner context",
        "Open Student Chat and review the Contextualization Setup: learning style, interests, preferred contexts, and current learning needs.",
    ),
    (
        "2. Ask a curriculum-aligned problem",
        "Use a unit-rate or ratio problem. The tutor should keep the response Socratic and connect to the problem context or learner profile when useful.",
    ),
    (
        "3. Inspect the explanation layer",
        "Point to the visual diagram, Curriculum Grounding, and Context Used panel to show that the system is explainable rather than black-box.",
    ),
    (
        "4. Submit an answer",
        "Enter a correct answer such as 60, 60 miles, or 60 miles per hour. The evaluator should confirm the answer and restate the complete unit.",
    ),
    (
        "5. Update reports",
        "End the session so signals, mastery history, teacher insight, parent practice, and class overview update from the same student record.",
    ),
    (
        "6. Show the ecosystem",
        "Switch to Teacher Dashboard, Parent Summary, and Class Overview to show the full contextualized learning loop.",
    ),
]


def _feature_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class=\"mathbridge-card\">
          <h4>{title}</h4>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class=\"mathbridge-step\">
          <strong>{title}</strong><br>
          <span class=\"mathbridge-muted\">{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _nav_buttons() -> None:
    role = st.session_state.get("role", "")

    if role == "Student":
        if st.button("Open Student Chat", use_container_width=True, type="primary"):
            st.session_state.view = "student"
            st.rerun()
        return

    if role == "Teacher":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Open Teacher Dashboard", use_container_width=True, type="primary"):
                st.session_state.view = "teacher"
                st.rerun()
        with col2:
            if st.button("Open Class Overview", use_container_width=True):
                st.session_state.view = "class_overview"
                st.rerun()
        return

    if role == "Parent":
        if st.button("Open Parent Summary", use_container_width=True, type="primary"):
            st.session_state.view = "parent"
            st.rerun()


def render():
    st.markdown(_HERO_STYLE, unsafe_allow_html=True)

    st.markdown(
        """
        <section class="mathbridge-hero">
          <div class="mathbridge-eyebrow">Showcase Demo · Contextualized Learning Platform</div>
          <h1 class="mathbridge-title">MathBridge AI: Contextualized, Curriculum-Grounded Math Support</h1>
          <p class="mathbridge-subtitle">
            MathBridge AI adapts Socratic math tutoring using learner profiles, Grade 6 curriculum grounding,
            real-life contexts, visual explanations, answer evaluation, and learning analytics for students,
            teachers, parents, and class-level planning.
          </p>
          <div class="mathbridge-pill-row">
            <span class="mathbridge-pill">Learner Context</span>
            <span class="mathbridge-pill">Curriculum Grounding</span>
            <span class="mathbridge-pill">Socratic Tutor</span>
            <span class="mathbridge-pill">Stable Visuals</span>
            <span class="mathbridge-pill">Teacher + Parent Insights</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    _nav_buttons()

    st.divider()

    st.subheader("What this demo is designed to prove")
    st.caption(
        "A polished vertical slice: controlled curriculum scope, full learning loop, and product-like reporting."
    )

    cols = st.columns(3)
    for index, (title, body) in enumerate(FEATURES):
        with cols[index % 3]:
            _feature_card(title, body)

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Recommended demo flow")
        for title, body in SHOWCASE_STEPS:
            _step(title, body)

    with right:
        st.subheader("Fast test problems")
        st.caption("Use these to show answer evaluation, context selection, and visual templates.")
        st.markdown(
            "\n".join([f"- {problem}" for problem in DEMO_PROBLEMS])
        )

        st.markdown("#### One-sentence project positioning")
        st.markdown(
            """
            <div class="mathbridge-card">
              <p><strong>MathBridge AI is not only a math chatbot.</strong> It is a contextualization layer that connects learner context, curriculum objectives, live learning signals, and real-world examples into one adaptive tutoring workflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("System architecture at a glance")
    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)

    with arch_col1:
        _feature_card(
            "Input Context",
            "Learner profile, current problem, curriculum scope, previous sessions, and current learning signals.",
        )
    with arch_col2:
        _feature_card(
            "AI + Deterministic Engines",
            "Tutor agent, answer evaluator, hint agent, retrieval layer, context selector, and deterministic visual templates.",
        )
    with arch_col3:
        _feature_card(
            "Student Support",
            "Socratic explanations, context-aware hints, stable diagrams, and correctness feedback.",
        )
    with arch_col4:
        _feature_card(
            "Actionable Reporting",
            "Teacher insights, parent home practice, class grouping strategy, and downloadable class summary.",
        )

