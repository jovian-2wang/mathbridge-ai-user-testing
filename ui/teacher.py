import json

import streamlit as st

from ai.insights import generate_teacher_insights
from ai.contextualization import get_context_profile
from memory.student_memory import load_student, list_student_ids


EXCLUDED_STUDENT_IDS = {"demo_users", "skill_buckets"}


def _format_delta(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value or "0")

    if numeric.is_integer():
        return f"{int(numeric):+d}"

    return f"{numeric:+.1f}"


def get_visible_student_ids():
    """Return real student memory ids only, excluding account/config files."""
    return [
        student_id
        for student_id in list_student_ids()
        if student_id not in EXCLUDED_STUDENT_IDS
    ]


@st.cache_data(show_spinner=False, ttl=600)
def _cached_teacher_insights(student_json: str) -> dict:
    return generate_teacher_insights(
        json.loads(student_json)
    )


def _student_json_for_agent(student: dict) -> str:
    """
    Cache key for the Teacher Insight Agent.

    Keep only fields needed for teacher-facing analysis so Streamlit reruns do
    not call the LLM unnecessarily.
    """
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
        "context_profile": get_context_profile(student),
    }

    return json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
    )


def _safe_join(values, fallback: str = "—") -> str:
    if isinstance(values, str):
        values = [
            item.strip()
            for item in values.split(",")
            if item.strip()
        ]
    elif not isinstance(values, list):
        values = []

    cleaned = [
        str(item).strip()
        for item in values
        if str(item).strip()
    ]

    return ", ".join(cleaned) if cleaned else fallback


def _render_delta_badge(delta_text: str, delta_value) -> None:
    try:
        numeric = float(delta_value)
    except (TypeError, ValueError):
        st.info(delta_text)
        return

    if numeric > 0:
        st.success(delta_text)
    elif numeric < 0:
        st.error(delta_text)
    else:
        st.info(delta_text)


def _render_contextualized_teaching_insight(
    student: dict,
    teacher_insights: dict,
) -> None:
    profile = get_context_profile(student)
    learner = profile.get("learner", {})
    strategy = profile.get("contextualization_strategy", {})

    insight = teacher_insights.get("contextualized_teaching_insight") or {}

    st.subheader("Contextualized Teaching Insight")
    st.caption(
        "How the learner context can shape the next explanation, example, or support move."
    )

    top1, top2, top3 = st.columns(3)

    with top1:
        with st.container(border=True):
            st.caption("Learner Context")
            st.markdown(
                f"**Grade {learner.get('grade_level', '6')} · "
                f"{learner.get('math_confidence', 'medium')} confidence**"
            )
            st.write(_safe_join(learner.get("learning_style")))

    with top2:
        with st.container(border=True):
            st.caption("Real-Life Contexts")
            st.markdown("**Preferred examples**")
            st.write(_safe_join(profile.get("real_world_contexts")))

    with top3:
        with st.container(border=True):
            st.caption("Learning Needs")
            st.markdown("**Current support targets**")
            st.write(_safe_join(profile.get("learning_needs")))

    with st.container(border=True):
        st.markdown("#### Recommended Contextualization Move")

        summary = insight.get("summary") or (
            "Use the learner profile to choose familiar examples while keeping the original curriculum objective."
        )
        st.info(summary)

        recommended_contexts = insight.get("recommended_contexts") or profile.get("real_world_contexts", [])
        if recommended_contexts:
            st.markdown("**Recommended contexts**")
            st.write(_safe_join(recommended_contexts))

        support_style = insight.get("support_style") or strategy.get("visual_style")
        if support_style:
            st.markdown("**Support style**")
            st.write(support_style)

        next_teacher_move = insight.get("next_teacher_move")
        if next_teacher_move:
            st.markdown("**Next teacher move**")
            st.write(next_teacher_move)



def render():
    student_ids = get_visible_student_ids()

    if not student_ids:
        load_student("alex")
        student_ids = ["alex"]

    student_records = {
        student_id: load_student(student_id)
        for student_id in student_ids
    }

    active_student_id = st.session_state.get("student_id", "alex")

    if active_student_id in student_ids:
        default_index = student_ids.index(active_student_id)
    else:
        default_index = 0

    selected_student_id = st.selectbox(
        "Select a student",
        options=student_ids,
        index=default_index,
        format_func=lambda student_id: (
            student_records[student_id].get(
                "name",
                student_id.capitalize(),
            )
        ),
        key="teacher_student_id",
    )

    student_id = selected_student_id
    student = student_records[student_id]

    if student_id == active_student_id:
        live_signals = st.session_state.get("signals") or {}
    else:
        live_signals = {}

    saved_signals = student.get("current_signals") or {}
    signals = live_signals or saved_signals

    signals_history = student.get("signals_history", [])
    mastery = student.get("mastery", {})
    mastery_history = student.get("mastery_history", [])
    student_name = student.get("name", student_id.capitalize())

    topic = signals.get("concept") or student.get(
        "current_topic",
        "No topic recorded",
    )
    engagement = signals.get("engagement") or "No recent data"

    if mastery_history:
        latest_mastery = mastery_history[-1]
        latest_skill = latest_mastery.get(
            "skill",
            "Latest skill",
        )
        latest_score = latest_mastery.get(
            "new_score",
            "—",
        )
        latest_delta = latest_mastery.get("delta", 0)

        trend_status = f"{latest_skill}: {latest_score}%"
        trend_delta = f"{_format_delta(latest_delta)} points"
    else:
        trend_status = "No mastery updates yet"
        trend_delta = None

    if mastery:
        support_skill = min(mastery, key=mastery.get)
    else:
        support_skill = "No mastery data"

    teacher_insights = _cached_teacher_insights(
        _student_json_for_agent(student)
    )

    st.subheader(f"Teacher View: {student_name}")
    st.caption(
        "Live and saved learning signals from the student's tutoring activity."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.caption("Topic Focus")
            st.markdown(f"### {topic}")

    with col2:
        with st.container(border=True):
            st.caption("Mastery Trend")
            st.markdown(f"### {trend_status}")
            if trend_delta:
                _render_delta_badge(trend_delta, latest_delta)

    with col3:
        with st.container(border=True):
            st.caption("Support Needed")
            st.markdown(f"### {support_skill}")

    with col4:
        with st.container(border=True):
            st.caption("Engagement")
            st.markdown(f"### {engagement}")

    st.divider()

    st.subheader("Latest Learning Signal")

    if signals:
        misconception = signals.get("misconception")
        next_support = signals.get("next_support")
        hints_used = signals.get("hints_used", "Not recorded")

        signal_col1, signal_col2 = st.columns(2)

        with signal_col1:
            with st.container(border=True):
                st.markdown("#### Possible Misconception")
                if misconception:
                    st.warning(str(misconception))
                else:
                    st.success("No clear misconception detected.")

                st.markdown("#### Hint Usage")
                st.write(str(hints_used))

        with signal_col2:
            with st.container(border=True):
                st.markdown("#### Suggested Next Support")
                if next_support:
                    st.info(str(next_support))
                else:
                    st.write("No recommendation recorded yet.")

                st.markdown("#### Current Engagement")
                st.write(str(engagement))
    else:
        st.info(
            "No learning signals have been recorded yet. "
            "Ask the student to complete a tutoring interaction first."
        )

    st.divider()

    _render_contextualized_teaching_insight(student, teacher_insights)

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Student Mastery Map")

        if mastery:
            for skill, score in mastery.items():
                safe_score = max(0, min(int(score), 100))

                if safe_score >= 75:
                    status_icon = "🟢"
                    status_text = "Proficient"
                elif safe_score >= 50:
                    status_icon = "🟡"
                    status_text = "Developing"
                else:
                    status_icon = "🔴"
                    status_text = "Needs support"

                st.markdown(
                    f"**{status_icon} {skill} — {safe_score}%**"
                )
                st.progress(safe_score / 100)
                st.caption(status_text)
        else:
            st.info("No mastery scores are available.")

    with right:
        st.subheader("AI Teacher Insights")

        generated_by = teacher_insights.get(
            "generated_by",
            "Teacher Insight Agent",
        )
        st.caption(generated_by)

        summary = teacher_insights.get("summary")
        priority = teacher_insights.get("priority")

        if summary:
            st.info(summary)

        if priority:
            with st.container(border=True):
                st.markdown("#### Priority Focus")
                st.write(priority)

        action_plan = teacher_insights.get("action_plan", [])

        if action_plan:
            for insight in action_plan:
                with st.container(border=True):
                    st.markdown(
                        f"**{insight.get('title', 'Suggested action')}**"
                    )
                    st.write(insight.get("body", "—"))
        else:
            st.info("No teacher insights are available yet.")

        small_grouping = teacher_insights.get("small_grouping")
        watch_for = teacher_insights.get("watch_for")

        if small_grouping:
            with st.container(border=True):
                st.markdown("#### Differentiation / Grouping")
                st.write(small_grouping)

        if watch_for:
            with st.container(border=True):
                st.markdown("#### Watch Next")
                st.write(watch_for)

    st.divider()
    st.subheader("Recent Signal History")

    if signals_history:
        recent_history = list(reversed(signals_history[-5:]))

        for index, record in enumerate(recent_history, start=1):
            timestamp = record.get("timestamp", "Time not recorded")
            concept = record.get("concept", "Unknown concept")

            with st.expander(
                f"{index}. {concept} — {timestamp}",
                expanded=index == 1,
            ):
                st.markdown(
                    f"**Misconception:** "
                    f"{record.get('misconception') or 'None detected'}"
                )
                st.markdown(
                    f"**Hint usage:** "
                    f"{record.get('hints_used', 'Not recorded')}"
                )
                st.markdown(
                    f"**Engagement:** "
                    f"{record.get('engagement', 'Not recorded')}"
                )
                st.markdown(
                    f"**Next support:** "
                    f"{record.get('next_support') or 'Not recorded'}"
                )
    else:
        st.caption("No historical signals have been saved yet.")

    st.divider()
    st.subheader("Recent Mastery Updates")

    st.caption(
        "These scores are preliminary estimates based on tutoring "
        "interactions, not formal assessment results."
    )

    if mastery_history:
        recent_mastery_updates = list(
            reversed(mastery_history[-5:])
        )

        for update in recent_mastery_updates:
            skill = update.get("skill", "Unknown skill")
            old_score = update.get("old_score", "—")
            new_score = update.get("new_score", "—")
            delta = update.get("delta", 0)
            timestamp = update.get(
                "timestamp",
                "Time not recorded",
            )
            concept = update.get(
                "concept",
                "Concept not recorded",
            )
            evidence = update.get("evidence", [])
            match_info = update.get("match_info", {})

            with st.container(border=True):
                st.markdown(
                    f"### {skill}: {old_score}% → "
                    f"{new_score}% ({_format_delta(delta)})"
                )

                st.caption(timestamp)
                st.markdown(f"**Detected concept:** {concept}")

                if match_info:
                    method = match_info.get("method", "unknown")
                    confidence = match_info.get("confidence")
                    if confidence is not None:
                        st.caption(
                            f"Skill match: {method}, confidence {confidence:.2f}"
                        )
                    else:
                        st.caption(f"Skill match: {method}")

                if evidence:
                    st.markdown("**Evidence used:**")

                    for item in evidence:
                        st.markdown(f"- {item}")
                else:
                    st.write("No supporting evidence recorded.")
    else:
        st.info(
            "No mastery score updates have been recorded yet. "
            "Complete and save a student session first."
        )
