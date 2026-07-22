import json
import os
import re
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


_CLASS_OVERVIEW_PROMPT = """\
You are the MathBridge Class Overview Agent.

Use the class-level student records to generate teacher-facing class insights.

You are given:
- each student's current topic
- engagement
- sessions
- average mastery
- lowest skill and score
- latest misconception
- recommended next support
- average mastery by skill
- each student's contextualization profile summary:
  learning style, math confidence, preferred real-life contexts, and learning needs

Your job:
- Generate a concise class summary.
- Identify the most important class-level teaching priority.
- Suggest small groups based on current learning needs, not only the lowest mastery score.
- Add contextualized grouping moves: for each group, recommend a real-life context and teacher move that fits the students' learner profiles.
- Avoid mismatches. For example, do not group a student under "Unit rates" if the current evidence is about whole-number division unless you explain why.
- Do not force a preferred context when the math problem already has a clear scenario.
- Generate misconception alerts only when the issue is meaningful and actionable.
- Do not overstate certainty. These are tutoring signals, not formal assessments.
- Keep teacher language practical and concise.
- Return ONLY valid JSON.

Return this exact JSON shape:
{
  "class_summary": "one concise summary of the class pattern",
  "priority_focus": "one class-level teaching priority",
  "small_groups": [
    {
      "group_name": "short group focus",
      "students": ["Student name"],
      "reason": "why these students are grouped together",
      "suggested_action": "one concrete teacher action"
    }
  ],
  "contextualized_groups": [
    {
      "group_name": "short group focus",
      "students": ["Student name"],
      "shared_skill_need": "shared math need",
      "recommended_context": "real-life context that fits this group",
      "teacher_move": "one concrete classroom move",
      "why_this_context": "why this context is useful for this group"
    }
  ],
  "misconception_alerts": [
    {
      "student": "Student name",
      "issue": "brief issue",
      "evidence": "what signal suggests this",
      "next_step": "one teacher move"
    }
  ],
  "watch_next": "what the teacher should monitor next"
}
"""


def _safe_json_loads(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except Exception:
        first = text.find("{")
        last = text.rfind("}")

        if first != -1 and last > first:
            try:
                payload = json.loads(text[first:last + 1])
            except Exception:
                payload = {}
        else:
            payload = {}

    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        raw_items = re.split(r"[,;/]", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]

    cleaned = []
    seen = set()

    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in {"—", "-"}:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

    return cleaned


def _fallback_contextualized_groups(
    class_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    class_context = class_context or {}
    students = class_context.get("students", [])
    if not isinstance(students, list):
        return []

    buckets: dict[str, list[dict[str, Any]]] = {}

    for row in students:
        if not isinstance(row, dict):
            continue

        name = str(row.get("Student") or row.get("name") or "").strip()
        if not name:
            continue

        skill = str(
            row.get("Current topic")
            or row.get("Lowest mastery area")
            or "current math focus"
        ).strip()

        if not skill or skill in {"—", "No topic recorded"}:
            skill = "current math focus"

        score = row.get("Lowest mastery score")
        support_status = str(row.get("Support status") or "").lower()

        include = (
            isinstance(score, (int, float)) and score < 75
        ) or "needs" in support_status or "developing" in support_status

        if not include:
            continue

        buckets.setdefault(skill, []).append(row)

    groups = []

    for skill, rows in list(buckets.items())[:4]:
        student_names = [
            str(row.get("Student") or row.get("name") or "").strip()
            for row in rows
            if str(row.get("Student") or row.get("name") or "").strip()
        ]

        context_counter = Counter()
        style_counter = Counter()

        for row in rows:
            context_counter.update(_as_list(row.get("Preferred contexts")))
            context_counter.update(_as_list(row.get("Real-life contexts")))
            style_counter.update(_as_list(row.get("Learning style")))

        common_contexts = [
            context
            for context, _count in context_counter.most_common(2)
        ] or ["familiar classroom or home examples"]

        common_styles = [
            style
            for style, _count in style_counter.most_common(2)
        ] or ["visual", "step-by-step"]

        recommended_context = ", ".join(common_contexts)
        support_style = ", ".join(common_styles)

        groups.append(
            {
                "group_name": f"{skill} Context Group",
                "students": student_names,
                "shared_skill_need": skill,
                "recommended_context": recommended_context,
                "teacher_move": (
                    f"Start with a {recommended_context} scenario, use a {support_style} explanation, "
                    "then ask students to name the quantities before calculating."
                ),
                "why_this_context": (
                    "This group has similar support needs and overlapping learner-context preferences, "
                    "so the teacher can use one familiar scenario while keeping the math goal consistent."
                ),
            }
        )

    return groups


def _fallback_class_overview(
    class_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "class_summary": (
            "AI class insights are currently unavailable. "
            "Use the class snapshot table to review recent learning signals."
        ),
        "priority_focus": "Review current topics, hint usage, and lowest mastery areas manually.",
        "small_groups": [],
        "contextualized_groups": _fallback_contextualized_groups(class_context),
        "misconception_alerts": [],
        "watch_next": (
            "Watch whether students can begin similar problems independently "
            "or continue to need repeated hints."
        ),
        "generated_by": "Fallback class overview",
    }


def _normalize_class_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    small_groups = payload.get("small_groups", [])
    if not isinstance(small_groups, list):
        small_groups = []

    clean_groups = []
    for item in small_groups[:4]:
        if not isinstance(item, dict):
            continue

        students = item.get("students", [])
        if not isinstance(students, list):
            students = [str(students)]

        students = [
            str(student).strip()
            for student in students
            if str(student).strip()
        ]

        group_name = str(item.get("group_name") or "").strip()
        reason = str(item.get("reason") or "").strip()
        suggested_action = str(item.get("suggested_action") or "").strip()

        if group_name and students and suggested_action:
            clean_groups.append(
                {
                    "group_name": group_name,
                    "students": students,
                    "reason": reason,
                    "suggested_action": suggested_action,
                }
            )

    contextualized_groups = payload.get("contextualized_groups", [])
    if not isinstance(contextualized_groups, list):
        contextualized_groups = []

    clean_context_groups = []
    for item in contextualized_groups[:4]:
        if not isinstance(item, dict):
            continue

        students = item.get("students", [])
        if not isinstance(students, list):
            students = [str(students)]

        students = [
            str(student).strip()
            for student in students
            if str(student).strip()
        ]

        group_name = str(item.get("group_name") or "").strip()
        shared_skill_need = str(item.get("shared_skill_need") or "").strip()
        recommended_context = str(item.get("recommended_context") or "").strip()
        teacher_move = str(item.get("teacher_move") or "").strip()
        why_this_context = str(item.get("why_this_context") or "").strip()

        if group_name and students and teacher_move:
            clean_context_groups.append(
                {
                    "group_name": group_name,
                    "students": students,
                    "shared_skill_need": shared_skill_need,
                    "recommended_context": recommended_context,
                    "teacher_move": teacher_move,
                    "why_this_context": why_this_context,
                }
            )

    alerts = payload.get("misconception_alerts", [])
    if not isinstance(alerts, list):
        alerts = []

    clean_alerts = []
    for item in alerts[:5]:
        if not isinstance(item, dict):
            continue

        student = str(item.get("student") or "").strip()
        issue = str(item.get("issue") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        next_step = str(item.get("next_step") or "").strip()

        if student and issue and next_step:
            clean_alerts.append(
                {
                    "student": student,
                    "issue": issue,
                    "evidence": evidence,
                    "next_step": next_step,
                }
            )

    return {
        "class_summary": str(payload.get("class_summary") or "").strip(),
        "priority_focus": str(payload.get("priority_focus") or "").strip(),
        "small_groups": clean_groups,
        "contextualized_groups": clean_context_groups,
        "misconception_alerts": clean_alerts,
        "watch_next": str(payload.get("watch_next") or "").strip(),
    }


def generate_class_overview_insights(
    class_context: dict[str, Any],
) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_class_overview(class_context)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_CLASS_OVERVIEW_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        class_context,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        payload = _safe_json_loads(str(response.content))
        normalized = _normalize_class_payload(payload)

        if not normalized:
            return _fallback_class_overview(class_context)

        if not normalized.get("contextualized_groups"):
            normalized["contextualized_groups"] = _fallback_contextualized_groups(
                class_context
            )

        normalized["generated_by"] = "Class Overview Agent"
        return normalized

    except Exception:
        return _fallback_class_overview(class_context)
