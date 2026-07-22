import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from ai.contextualization import get_context_profile


_TEACHER_PROMPT = """\
You are the MathBridge Teacher Insight Agent.

Use the student's recent tutoring signals, mastery history, and session data to
generate concise teacher-facing instructional insights.

Rules:
- Be specific to the student's latest concept and mastery evidence.
- Do not overstate certainty. These are tutoring signals, not formal assessment.
- Do not copy old static insights if they no longer match the current topic.
- Prefer actionable instructional guidance.
- Keep language professional and concise.
- Return ONLY valid JSON.

Return this exact shape:
{
  "summary": "one sentence summary of the current learning pattern",
  "priority": "one short priority area",
  "action_plan": [
    {"title": "short title", "body": "actionable teacher step"},
    {"title": "short title", "body": "actionable teacher step"},
    {"title": "short title", "body": "actionable teacher step"}
  ],
  "contextualized_teaching_insight": {
    "summary": "one sentence explaining how the learner context should shape instruction",
    "recommended_contexts": ["real-life context 1", "real-life context 2"],
    "support_style": "how the teacher should frame the next explanation",
    "next_teacher_move": "one specific classroom or tutoring move"
  },
  "small_grouping": "suggestion for grouping or differentiation",
  "watch_for": "what the teacher should monitor next"
}

Contextualization rules:
- Use the student's context_profile when available.
- Connect instructional recommendations to learner style, interests, real-world contexts, and learning needs.
- Do not force a hobby/context when it would distract from the math.
- Keep recommendations teacher-actionable, not generic personalization language.
"""


_PARENT_PROMPT = """\
You are the MathBridge Parent Summary Agent.

Use the student's recent tutoring signals, mastery history, and session data to
write a parent-friendly weekly learning summary.

The audience is a parent or caregiver, not a math teacher.

Important rules:
- Keep the headline and focus consistent with the most recent problem type.
- If the recent problem asks how many groups can be made, describe it as
  "figuring out how many groups can be made when each group has the same size,"
  not as "sharing things into equal groups."
- Prioritize the most recent tutoring signal and current concept.
- Do not mix in older topics unless they clearly explain a trend.
- Translate math concepts into plain language.
  Example:
  "division with whole numbers" -> "figuring out how many equal groups can be made"
  or "sharing a total into equal groups."
- Do not overstate certainty. These are tutoring signals, not a formal assessment.
- Keep the tone warm, calm, and encouraging.
- Avoid sounding like a diagnosis.
- Avoid teacher jargon such as "measurement division", "mastery", "intervention",
  or "misconception" unless explained in simple words.
- The "try_at_home" field must include ONE concrete everyday math question the parent can directly ask.
- The home question must use specific numbers and objects.
- Do not say only "try a similar problem" or "explain one problem"; give the actual problem.
- Do not provide the final answer.
- After the question, add one short parent instruction such as "Ask Alex to explain what each number means before solving."
- The activity should feel like a short everyday conversation, not a worksheet.
- Return ONLY valid JSON.

Contextualized home-practice rules:
- Use context_profile when available, but keep the activity natural for the current math topic.
- Prefer family-friendly everyday contexts: food, shopping, sports, travel, classroom supplies, or the problem's own scenario.
- Do not force a hobby if it distracts from the math.
- The home practice must be short enough for a parent to ask during a normal conversation.
- The suggested_parent_question must include numbers and a question mark.
- The question should NOT include the answer.
- what_to_listen_for should describe the reasoning parents should hear, not just the final answer.

Return this exact shape:
{
  "headline": "one warm sentence about the child's recent learning",
  "focus_plain_language": "one sentence explaining the current math focus in parent-friendly language",
  "what_went_well": "specific parent-friendly description of what the child did well",
  "needs_support": "gentle description of what the child is still practicing",
  "try_at_home": "one concrete question a parent can ask the child at home",
  "contextualized_home_practice": {
    "home_context": "short context label, such as shopping, snacks, sports, travel, or classroom supplies",
    "suggested_parent_question": "one concrete parent question with numbers and objects",
    "what_to_listen_for": "what reasoning the parent should listen for",
    "encouragement_prompt": "one sentence parent can say to encourage reasoning",
    "why_this_context": "why this context fits the learner and recent math focus"
  },
  "encouragement": "warm encouragement note",
  "basis": "brief explanation of what recent tutoring data this summary used"
}
"""


def _parse_json(raw_content: str) -> dict[str, Any]:
    cleaned = str(raw_content or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            try:
                payload = json.loads(cleaned[first:last + 1])
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    return payload if isinstance(payload, dict) else {}


def _compact_student_context(student: dict) -> dict:
    sessions = student.get("sessions") or []
    signals_history = student.get("signals_history") or []
    mastery_history = student.get("mastery_history") or []

    compact_sessions = []
    for session in sessions[-3:]:
        compact_sessions.append(
            {
                "date": session.get("date"),
                "topic": session.get("topic"),
                "engagement": session.get("engagement"),
                "signals": session.get("signals"),
                "mastery_update": session.get("mastery_update"),
            }
        )

    return {
        "student_id": student.get("student_id"),
        "name": student.get("name"),
        "current_topic": student.get("current_topic"),
        "current_signals": student.get("current_signals") or {},
        "recent_signals_history": signals_history[-5:],
        "mastery": student.get("mastery") or {},
        "recent_mastery_history": mastery_history[-5:],
        "recent_sessions": compact_sessions,
        "weekly_summary": student.get("weekly_summary") or {},
        "context_profile": get_context_profile(student),
    }


def _llm_json(prompt: str, student: dict) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {}

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    context = _compact_student_context(student)

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=json.dumps(
                    context,
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )

    return _parse_json(str(response.content))


def generate_teacher_insights(student: dict) -> dict[str, Any]:
    """
    Generate teacher-facing instructional insights.

    The agent uses LLM reasoning over recent signals and mastery history. If the
    LLM is unavailable, the function falls back to a deterministic summary so the
    app still runs.
    """

    try:
        payload = _llm_json(_TEACHER_PROMPT, student)
        normalized = _normalize_teacher_payload(payload)

        if normalized:
            if not normalized.get("contextualized_teaching_insight"):
                normalized["contextualized_teaching_insight"] = _fallback_contextualized_teacher_insight(student)
            normalized["generated_by"] = "Teacher Insight Agent"
            return normalized

    except Exception:
        pass

    fallback = _fallback_teacher_insights(student)
    fallback["generated_by"] = "Fallback teacher summary"
    return fallback


def generate_parent_summary(student: dict, student_id: str | None = None) -> dict[str, str]:
    """
    Generate a parent-friendly learning summary.

    The agent uses LLM reasoning over recent signals and mastery history. If the
    LLM is unavailable, the function falls back to a deterministic summary.
    """

    try:
        payload = _llm_json(_PARENT_PROMPT, student)
        normalized = _normalize_parent_payload(payload)

        if normalized:
            if not normalized.get("contextualized_home_practice"):
                normalized["contextualized_home_practice"] = (
                    _fallback_contextualized_home_practice(student)
                )
            normalized["generated_by"] = "Parent Summary Agent"
            return normalized

    except Exception:
        pass

    fallback = _fallback_parent_summary(student, student_id=student_id)
    fallback["generated_by"] = "Fallback parent summary"
    return fallback


def _normalize_teacher_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    action_plan = payload.get("action_plan", [])
    if not isinstance(action_plan, list):
        action_plan = []

    clean_actions = []
    for item in action_plan[:4]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "Suggested action").strip()
        body = str(item.get("body") or "").strip()
        if body:
            clean_actions.append(
                {
                    "title": title,
                    "body": body,
                }
            )

    if not clean_actions:
        return {}

    return {
        "summary": str(payload.get("summary") or "").strip(),
        "priority": str(payload.get("priority") or "").strip(),
        "action_plan": clean_actions,
        "contextualized_teaching_insight": _normalize_contextualized_teacher_insight(
            payload.get("contextualized_teaching_insight")
        ),
        "small_grouping": str(payload.get("small_grouping") or "").strip(),
        "watch_for": str(payload.get("watch_for") or "").strip(),
    }


def _normalize_contextualized_teacher_insight(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    recommended_contexts = value.get("recommended_contexts", [])
    if isinstance(recommended_contexts, str):
        recommended_contexts = [
            item.strip()
            for item in recommended_contexts.split(",")
            if item.strip()
        ]
    elif not isinstance(recommended_contexts, list):
        recommended_contexts = []

    cleaned_contexts = [
        str(item).strip()
        for item in recommended_contexts[:4]
        if str(item).strip()
    ]

    result = {
        "summary": str(value.get("summary") or "").strip(),
        "recommended_contexts": cleaned_contexts,
        "support_style": str(value.get("support_style") or "").strip(),
        "next_teacher_move": str(value.get("next_teacher_move") or "").strip(),
    }

    if not any(result.values()):
        return {}

    return result


def _fallback_contextualized_teacher_insight(student: dict) -> dict[str, Any]:
    profile = get_context_profile(student)
    learner = profile.get("learner", {})
    contexts = profile.get("real_world_contexts", []) or []
    needs = profile.get("learning_needs", []) or []
    strategy = profile.get("contextualization_strategy", {})

    learning_style = learner.get("learning_style") or []
    if isinstance(learning_style, list):
        learning_style_text = ", ".join(str(item) for item in learning_style if str(item).strip())
    else:
        learning_style_text = str(learning_style)

    context_text = ", ".join(contexts[:3]) or "familiar real-life examples"
    needs_text = ", ".join(needs[:3]) or "the current math topic"
    visual_style = strategy.get("visual_style") or "use diagrams before abstract equations"
    example_style = strategy.get("example_style") or "connect math to real-life scenarios"

    return {
        "summary": (
            f"Use {context_text} to make the next explanation feel relevant while targeting {needs_text}."
        ),
        "recommended_contexts": contexts[:4],
        "support_style": (
            f"Use a {learning_style_text or 'step-by-step'} approach; {visual_style}."
        ),
        "next_teacher_move": (
            f"Frame the next problem with one familiar context, then ask the student to explain the math relationship before computing. {example_style}"
        ),
    }



def _normalize_contextualized_home_practice(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    result = {
        "home_context": str(value.get("home_context") or "").strip(),
        "suggested_parent_question": str(
            value.get("suggested_parent_question") or ""
        ).strip(),
        "what_to_listen_for": str(
            value.get("what_to_listen_for") or ""
        ).strip(),
        "encouragement_prompt": str(
            value.get("encouragement_prompt") or ""
        ).strip(),
        "why_this_context": str(value.get("why_this_context") or "").strip(),
    }

    # The parent question should be concrete enough for a family to use.
    question = result["suggested_parent_question"]
    if _try_at_home_is_too_vague(question):
        return {}

    if not result["home_context"]:
        result["home_context"] = "everyday math"

    if not result["what_to_listen_for"]:
        result["what_to_listen_for"] = (
            "Listen for whether the child explains what each number means before calculating."
        )

    if not result["encouragement_prompt"]:
        result["encouragement_prompt"] = (
            "Praise the explanation process, not just the final answer."
        )

    if not result["why_this_context"]:
        result["why_this_context"] = (
            "This home practice uses recent tutoring signals and the learner context profile."
        )

    return result

def _try_at_home_is_too_vague(text: str) -> bool:
    lowered = str(text or "").lower()

    vague_phrases = (
        "one similar problem",
        "a similar problem",
        "explain one problem",
        "practice a problem",
        "try a problem",
    )

    has_question = "?" in lowered
    has_number = bool(re.search(r"\d+", lowered))

    return (
        any(phrase in lowered for phrase in vague_phrases)
        or not has_question
        or not has_number
    )




def _normalize_parent_payload(payload: dict[str, Any]) -> dict[str, str]:
    required = [
        "headline",
        "what_went_well",
        "needs_support",
        "try_at_home",
        "encouragement",
        "basis",
    ]

    if not isinstance(payload, dict):
        return {}

    result = {}

    for key in required:
        value = str(payload.get(key) or "").strip()
        if not value:
            return {}
        if key == "try_at_home" and _try_at_home_is_too_vague(value):
            return {}
        
        result[key] = value
    focus = str(payload.get("focus_plain_language") or "").strip()
    if focus:
        result["focus_plain_language"] = focus

    home_practice = _normalize_contextualized_home_practice(
        payload.get("contextualized_home_practice")
    )
    if home_practice:
        result["contextualized_home_practice"] = home_practice

    return result


def _fallback_teacher_insights(student: dict) -> dict[str, Any]:
    name = student.get("name", "The student")
    signals = student.get("current_signals") or {}
    topic = (
        signals.get("concept")
        or student.get("current_topic")
        or "the current topic"
    )

    return {
        "summary": (
            f"AI-generated teacher insights are currently unavailable. "
            f"{name}'s latest recorded topic is {topic}."
        ),
        "priority": "Review the latest tutoring signals manually.",
        "action_plan": [
            {
                "title": "Check recent work",
                "body": (
                    "Review the student's latest tutoring interaction, hint usage, "
                    "and saved misconception notes before deciding next instruction."
                ),
            },
            {
                "title": "Ask for explanation",
                "body": (
                    "Have the student explain what each number in the problem means "
                    "before solving."
                ),
            },
        ],
        "contextualized_teaching_insight": _fallback_contextualized_teacher_insight(student),
        "small_grouping": (
            "Use teacher judgment to group this student with peers working on a similar concept."
        ),
        "watch_for": (
            "Watch whether the student can start the next problem independently."
        ),
    }




def _fallback_contextualized_home_practice(student: dict) -> dict[str, str]:
    profile = get_context_profile(student)
    name = student.get("name", "your child")
    signals = student.get("current_signals") or {}
    topic = str(
        signals.get("concept")
        or student.get("current_topic")
        or "the current math topic"
    ).lower()

    contexts = profile.get("real_world_contexts") or []
    context = contexts[0] if contexts else "everyday math"

    if "unit rate" in topic or "rate" in topic:
        question = "If 4 packs of snacks cost $12, how much does one pack cost?"
        listen = (
            "Listen for dividing the total cost by the number of packs and naming the unit, such as dollars per pack."
        )
        home_context = "shopping / snacks"
    elif "ratio" in topic:
        question = "If there are 2 red game pieces for every 3 blue game pieces, what does 2 to 3 compare?"
        listen = (
            "Listen for explaining the two quantities being compared, not just reading the numbers."
        )
        home_context = "games / objects"
    elif "fraction" in topic:
        question = "If a recipe uses 3/4 cup of flour and you make half the recipe, what amount of flour would you think about?"
        listen = (
            "Listen for using the fraction meaning and explaining the parts before calculating."
        )
        home_context = "food / recipe"
    elif "division" in topic:
        question = "If 24 stickers are shared equally into 6 groups, how many stickers go in each group?"
        listen = (
            "Listen for explaining the total, the number of groups, and what one group means."
        )
        home_context = "stickers / sharing"
    else:
        question = "If 5 notebooks cost $15, how much does one notebook cost?"
        listen = (
            "Listen for identifying what each number means and choosing an operation before calculating."
        )
        home_context = context

    return {
        "home_context": home_context,
        "suggested_parent_question": question,
        "what_to_listen_for": listen,
        "encouragement_prompt": (
            f"Tell {name}: I like how you explained your thinking before focusing on the answer."
        ),
        "why_this_context": (
            "This activity connects the latest learning focus with a familiar everyday situation from the learner context profile."
        ),
    }

def _fallback_parent_summary(
    student: dict,
    student_id: str | None = None,
) -> dict[str, str]:
    name = student.get("name", (student_id or "student").capitalize())
    signals = student.get("current_signals") or {}

    topic = (
        signals.get("concept")
        or student.get("current_topic")
        or "the current math topic"
    )

    home_practice = _fallback_contextualized_home_practice(student)

    return {
        "headline": f"{name} recently worked on {topic}.",
        "focus_plain_language": (
            f"The current focus is practicing {topic} in a way that connects to everyday reasoning."
        ),
        "what_went_well": (
            f"{name} participated in a tutoring session and practiced explaining math steps."
        ),
        "needs_support": (
            "The AI parent summary is currently unavailable, so this is a brief fallback "
            "based only on saved tutoring signals."
        ),
        "try_at_home": (
            f"{home_practice['suggested_parent_question']} "
            f"Ask {name} to explain what each number means before solving."
        ),
        "contextualized_home_practice": home_practice,
        "encouragement": (
            f"Keep encouraging {name} to explain their thinking step by step."
        ),
        "basis": (
            "This fallback summary used the latest saved topic, tutoring signals, "
            "and learner context profile. It is not a full AI-generated parent report."
        ),
    }

