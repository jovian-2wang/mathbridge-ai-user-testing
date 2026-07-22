import json

import pandas as pd
import streamlit as st

from ai.class_insights import generate_class_overview_insights
from ai.contextualization import (
    summarize_class_context_patterns,
    summarize_context_profile_for_class,
)
from memory.student_memory import list_student_ids, load_student
EXCLUDED_STUDENT_IDS = {"demo_users", "skill_buckets"}

def _has_misconception(value) -> bool:
    """Return True only when a meaningful misconception exists."""
    if value is None:
        return False

    text = str(value).strip().lower()

    no_misconception_values = {
        "",
        "none",
        "null",
        "n/a",
        "no",
        "—",
        "-",
        "none detected",
        "no misconception",
        "no clear misconception detected",
        "no clear misconception detected.",
    }

    if text in no_misconception_values:
        return False

    if text.startswith("no clear misconception"):
        return False

    return True


def _clean_text(value, fallback="—") -> str:
    text = str(value or "").strip()

    if not text:
        return fallback

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

def _safe_number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_number(row, *keys, default=0):
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "—", "-"):
            return _safe_number(value, default=default)
    return default


def _row_text(row, *keys, default=""):
    for key in keys:
        value = row.get(key)
        if value not in (None, "", "—", "-"):
            return str(value)
    return default

def _split_items(value) -> list[str]:
    text = str(value or "").strip()
    if not text or text in {"—", "-"}:
        return []
    return [
        item.strip()
        for item in text.replace(";", ",").split(",")
        if item.strip()
    ]


def _join_top_items(items: list[str], limit: int = 4) -> str:
    if not items:
        return "No pattern yet"
    return ", ".join(items[:limit])


def _local_contextualized_group_suggestions(rows: list[dict]) -> list[dict]:
    """Deterministic backup if the Class Overview Agent returns no groups."""
    buckets: dict[str, list[dict]] = {}

    for row in rows:
        score = _row_number(
            row,
            "Lowest mastery score",
            default=100,
        )
        support_status = _row_text(
            row,
            "Support status",
            default="",
        ).lower()

        if score >= 75 and "needs" not in support_status and "developing" not in support_status:
            continue

        skill = _row_text(
            row,
            "Current topic",
            "Lowest mastery area",
            default="Current math focus",
        )
        buckets.setdefault(skill, []).append(row)

    suggestions = []

    for skill, group_rows in list(buckets.items())[:4]:
        student_names = [
            str(row.get("Student", "Student"))
            for row in group_rows
        ]

        contexts = []
        styles = []
        for row in group_rows:
            contexts.extend(_split_items(row.get("Preferred contexts")))
            styles.extend(_split_items(row.get("Learning style")))

        seen_contexts = list(dict.fromkeys(contexts))
        seen_styles = list(dict.fromkeys(styles))

        recommended_context = (
            ", ".join(seen_contexts[:2])
            if seen_contexts
            else "familiar classroom or home examples"
        )
        support_style = (
            ", ".join(seen_styles[:2])
            if seen_styles
            else "visual, step-by-step"
        )

        suggestions.append(
            {
                "group_name": f"{skill} Context Group",
                "students": student_names,
                "shared_skill_need": skill,
                "recommended_context": recommended_context,
                "teacher_move": (
                    f"Use a {recommended_context} scenario with {support_style} support. "
                    "Ask students to identify the quantities before they calculate."
                ),
                "why_this_context": (
                    "This group shares a current support need and has overlapping learner-context preferences."
                ),
            }
        )

    return suggestions


def _status_from_score(score):
    if score is None:
        return "No data"

    if score < 50:
        return "🔴 Needs support"

    if score < 75:
        return "🟡 Developing"

    return "🟢 Proficient"


def _latest_signal(student: dict) -> dict:
    signals = student.get("current_signals") or {}

    if signals.get("concept"):
        return signals

    history = student.get("signals_history") or []

    if history:
        return history[-1]

    return {}


def _student_row(student_id: str) -> tuple[dict, dict]:
    student = load_student(student_id)

    name = student.get(
        "name",
        student_id.capitalize(),
    )

    signals = _latest_signal(student)
    mastery = student.get("mastery") or {}
    sessions = student.get("sessions") or []

    topic = _clean_text(
        signals.get("concept")
        or student.get("current_topic")
        or "No topic recorded"
    )

    engagement = _clean_text(
        signals.get("engagement"),
        "No recent data",
    )

    misconception = signals.get("misconception")
    misconception_text = (
        _clean_text(misconception)
        if _has_misconception(misconception)
        else "—"
    )

    next_support = _clean_text(
        signals.get("next_support"),
        "No next support recorded",
    )

    if mastery:
        valid_mastery = {
            skill: max(0.0, min(100.0, float(score)))
            for skill, score in mastery.items()
            if isinstance(score, (int, float))
        }
    else:
        valid_mastery = {}

    if valid_mastery:
        lowest_skill = min(
            valid_mastery,
            key=valid_mastery.get,
        )
        lowest_score = round(valid_mastery[lowest_skill])
        average_mastery = round(
            sum(valid_mastery.values())
            / len(valid_mastery)
        )
    else:
        lowest_skill = "No mastery data"
        lowest_score = None
        average_mastery = None

    context_summary = summarize_context_profile_for_class(student)

    row = {
        "Student": name,
        "Current topic": topic,
        "Engagement": engagement,
        "Sessions": len(sessions),
        "Average mastery": average_mastery,
        "Lowest mastery area": lowest_skill,
        "Lowest mastery score": lowest_score,
        "Support status": _status_from_score(lowest_score),
        "Latest misconception": misconception_text,
        "Recommended next support": next_support,
        "Learning style": context_summary.get("learning_style", "—"),
        "Math confidence": context_summary.get("math_confidence", "—"),
        "Preferred contexts": context_summary.get("preferred_contexts", "—"),
        "Contextual learning needs": context_summary.get("learning_needs", "—"),
        "Contextualization move": context_summary.get("contextualization_move", "—"),
    }

    return row, valid_mastery

@st.cache_data(show_spinner=False, ttl=600)
def _cached_class_overview_insights(class_json: str) -> dict:
    return generate_class_overview_insights(
        json.loads(class_json)
    )


def _class_json_for_agent(
    rows: list[dict],
    mastery_values_by_skill: dict,
) -> str:
    skill_averages = []

    for skill, scores in mastery_values_by_skill.items():
        if not scores:
            continue

        skill_averages.append(
            {
                "skill": skill,
                "average_mastery": round(sum(scores) / len(scores)),
                "student_count": len(scores),
            }
        )

    compact = {
        "students": rows,
        "skill_averages": sorted(
            skill_averages,
            key=lambda item: item["average_mastery"],
        ),
        "note": (
            "These are tutoring signals and mastery estimates, "
            "not formal assessment results."
        ),
    }

    return json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
    )

def render():
    st.subheader("Class Overview")
    st.caption(
        "A class-wide snapshot of mastery, engagement, "
        "learning activity, and support needs."
    )

    student_ids = [
        student_id
        for student_id in list_student_ids()
        if student_id not in EXCLUDED_STUDENT_IDS
    ]

    if not student_ids:
        st.info("No student profiles are available yet.")
        return

    rows = []
    mastery_values_by_skill = {}

    for student_id in student_ids:
        row, valid_mastery = _student_row(student_id)
        rows.append(row)

        for skill, score in valid_mastery.items():
            mastery_values_by_skill.setdefault(
                skill,
                [],
            ).append(score)

    class_df = pd.DataFrame(rows)

    students_needing_support = sum(
        1
        for row in rows
        if _row_number(
            row,
            "Lowest mastery score",
            "Lowest score",
            default=100,
        ) < 50
    )

    active_students = sum(
        1
        for row in rows
        if _row_text(row, "Engagement").strip().lower()
        in {"active", "engaged", "persistent"}
    )

    misconception_alerts = sum(
        1
        for row in rows
        if _has_misconception(
            _row_text(row, "Latest misconception", default="—")
        )
    )

    total_sessions = sum(
        int(_row_number(row, "Sessions", default=0))
        for row in rows
    )

    average_class_mastery_values = [
        _row_number(row, "Average mastery", default=0)
        for row in rows
        if row.get("Average mastery") is not None
    ]

    average_class_mastery = (
        round(
            sum(average_class_mastery_values)
            / len(average_class_mastery_values)
        )
        if average_class_mastery_values
        else None
    )

    class_insights = _cached_class_overview_insights(
        _class_json_for_agent(
            rows,
            mastery_values_by_skill,
        )
    )
    # ---------------------------------------------------------
    # Class-level metrics
    # ---------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with st.container(border=True):
            st.caption("Students")
            st.markdown(f"## {len(student_ids)}")

    with col2:
        with st.container(border=True):
            st.caption("Need Support")
            st.markdown(f"## {students_needing_support}")

    with col3:
        with st.container(border=True):
            st.caption("Active Students")
            st.markdown(f"## {active_students}")

    with col4:
        with st.container(border=True):
            st.caption("Saved Sessions")
            st.markdown(f"## {total_sessions}")

    with col5:
        with st.container(border=True):
            st.caption("Avg. Mastery")
            st.markdown(
                f"## {average_class_mastery}%"
                if average_class_mastery is not None
                else "## —"
            )

    class_context_patterns = summarize_class_context_patterns(rows)

    st.divider()
    st.subheader("Class Context Snapshot")
    st.caption(
        "Common learner-context patterns that can guide examples, grouping, and support moves."
    )

    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

    with ctx_col1:
        with st.container(border=True):
            st.caption("Common real-life contexts")
            st.markdown(
                "**"
                + _join_top_items(
                    class_context_patterns.get("common_contexts", []),
                    limit=4,
                )
                + "**"
            )

    with ctx_col2:
        with st.container(border=True):
            st.caption("Learning style pattern")
            st.markdown(
                "**"
                + _join_top_items(
                    class_context_patterns.get("common_learning_styles", []),
                    limit=4,
                )
                + "**"
            )

    with ctx_col3:
        with st.container(border=True):
            st.caption("Shared learning needs")
            st.markdown(
                "**"
                + _join_top_items(
                    class_context_patterns.get("common_learning_needs", []),
                    limit=4,
                )
                + "**"
            )

    st.divider()
    st.subheader("AI Class Insights")
    st.caption(class_insights.get("generated_by", "Class Overview Agent"))

    class_summary = class_insights.get("class_summary")
    priority_focus = class_insights.get("priority_focus")
    watch_next = class_insights.get("watch_next")

    if class_summary:
        st.info(class_summary)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        with st.container(border=True):
            st.markdown("#### Priority Focus")
            st.write(priority_focus or "No class-level priority generated yet.")

    with insight_col2:
        with st.container(border=True):
            st.markdown("#### Watch Next")
            st.write(watch_next or "No watch item generated yet.")

    st.divider()
    st.subheader("Contextualized Grouping Strategy")
    st.caption(
        "Small-group teaching moves that combine skill needs with learner-context patterns."
    )

    contextualized_groups = (
        class_insights.get("contextualized_groups", [])
        or _local_contextualized_group_suggestions(rows)
    )

    if contextualized_groups:
        for group in contextualized_groups[:4]:
            with st.container(border=True):
                st.markdown(
                    f"#### {group.get('group_name', 'Context group')}"
                )

                students = group.get("students", [])
                if isinstance(students, list):
                    student_text = ", ".join(students)
                else:
                    student_text = str(students)

                st.write(f"**Students:** {student_text}")

                shared_need = group.get("shared_skill_need")
                if shared_need:
                    st.write(f"**Shared skill need:** {shared_need}")

                recommended_context = group.get("recommended_context")
                if recommended_context:
                    st.write(f"**Recommended context:** {recommended_context}")

                teacher_move = group.get("teacher_move")
                if teacher_move:
                    st.info(teacher_move)

                why_this_context = group.get("why_this_context")
                if why_this_context:
                    st.caption("Why this context:")
                    st.write(why_this_context)
    else:
        st.info(
            "No contextualized grouping strategy is currently recommended."
        )

    st.divider()
    # ---------------------------------------------------------
    # Student table
    # ---------------------------------------------------------
    st.subheader("Student Learning Snapshot")

    support_filter = st.multiselect(
        "Filter by support status",
        options=[
            "🔴 Needs support",
            "🟡 Developing",
            "🟢 Proficient",
            "No data",
        ],
        default=[],
    )

    filtered_df = class_df

    if support_filter:
        filtered_df = class_df[
            class_df["Support status"].isin(
                support_filter
            )
        ]

    summary_columns = [
        "Student",
        "Current topic",
        "Engagement",
        "Sessions",
        "Average mastery",
        "Lowest mastery area",
        "Lowest mastery score",
        "Support status",
        "Latest misconception",
    ]

    display_df = filtered_df[
        [
            column
            for column in summary_columns
            if column in filtered_df.columns
        ]
    ].copy()

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
    )

    with st.expander("Detailed support recommendations", expanded=False):
        for _, row in filtered_df.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{row.get('Student', 'Student')}**"
                )
                st.write(
                    "**Current topic:** "
                    + str(row.get("Current topic", "—"))
                )
                st.write(
                    "**Recommended next support:** "
                    + str(row.get("Recommended next support", "—"))
                )
                st.write(
                    "**Preferred contexts:** "
                    + str(row.get("Preferred contexts", "—"))
                )
                st.write(
                    "**Learning style:** "
                    + str(row.get("Learning style", "—"))
                )
                st.write(
                    "**Contextualization move:** "
                    + str(row.get("Contextualization move", "—"))
                )

    # ---------------------------------------------------------
    # Average mastery by skill
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Average Mastery by Skill")

    average_rows = []

    for skill, scores in mastery_values_by_skill.items():
        if not scores:
            continue

        average_rows.append(
            {
                "Skill": skill,
                "Average mastery": round(
                    sum(scores) / len(scores)
                ),
            }
        )

    if average_rows:
        average_df = pd.DataFrame(
            average_rows
        ).sort_values(
            "Average mastery",
            ascending=True,
        )

        average_df["Skill label"] = average_df["Skill"].apply(
            lambda text: str(text)[:18] + "..."
            if len(str(text)) > 18
            else str(text)
        )

        st.bar_chart(
            average_df.set_index("Skill label")[["Average mastery"]]
        )
    else:
        st.info("No mastery data is available yet.")

    # ---------------------------------------------------------
    # Suggested small groups
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Suggested Small Groups")

    small_groups = class_insights.get("small_groups", [])

    if small_groups:
        for group in small_groups:
            with st.container(border=True):
                st.markdown(
                    f"**{group.get('group_name', 'Suggested group')}**"
                )

                students = group.get("students", [])

                if isinstance(students, list):
                    student_text = ", ".join(students)
                else:
                    student_text = str(students)

                st.write(f"Students: {student_text}")

                reason = group.get("reason")
                if reason:
                    st.caption("Why this group:")
                    st.write(reason)

                suggested_action = group.get("suggested_action")
                if suggested_action:
                    st.caption("Suggested action:")
                    st.write(suggested_action)
    else:
        st.info(
            "No AI-generated small groups are currently recommended."
        )

    # ---------------------------------------------------------
    # Misconception alerts
    # ---------------------------------------------------------
    alerts = class_insights.get("misconception_alerts", [])

    st.divider()
    st.subheader(
        f"Misconception Alerts ({len(alerts)})"
    )

    if alerts:
        for alert in alerts:
            with st.container(border=True):
                student = alert.get("student", "Student")
                issue = alert.get("issue", "Possible issue")
                evidence = alert.get("evidence", "")
                next_step = alert.get("next_step", "")

                st.markdown(f"**{student}**")
                st.warning(issue)

                if evidence:
                    st.caption("Evidence")
                    st.write(evidence)

                if next_step:
                    st.caption("Next teacher move")
                    st.write(next_step)
    else:
        st.success(
            "No AI-generated misconception alerts are currently flagged."
        )

    # ---------------------------------------------------------
    # CSV export
    # ---------------------------------------------------------
    st.divider()

    csv_data = class_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Class Summary CSV",
        data=csv_data,
        file_name="mathbridge_class_summary.csv",
        mime="text/csv",
        width="stretch",
    )