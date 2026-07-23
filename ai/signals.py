import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


_PROMPT = """\
You are the MathBridge Student Signal Agent.

Analyze the latest math tutoring interaction and extract learning signals.
Use the conversation, curriculum context, prior signals, and behavioral features.
Do not rely on simple keyword matching.
Do not guess a misconception unless it is visible in the student's words.

Return ONLY valid JSON — no explanation, no markdown fences.

Signal extraction rules:
- "concept" should be the specific Grade 6 math concept currently practiced.
  Examples: "dividing fractions", "ratio reasoning", "unit rate", "fraction-decimal equivalence".
- Do NOT label a fraction-division problem as "fraction comparison" unless the student is actually comparing which fraction is larger.
- "misconception" should be null if the student has not made a clear mistake.
- "hints_used" is retained for compatibility, but it means typed student support requests, not button clicks. Use the provided behavioral features.
- "engagement" must follow this rubric:
  - Active: the student attempts a step, gives reasoning, answers a guiding question, or asks a meaningful follow-up.
  - Moderate: the student responds or requests help but gives little reasoning.
  - Low: the student gives very short replies, repeats the same request, or does not attempt a step.
  - Needs scaffold: the student says they do not understand, asks for the answer, gives up, or shows repeated confusion.
- "next_support" should be the next pedagogical move for the teacher/tutor.
  It should match the concept and the student's latest state.
- "evidence" should briefly cite the interaction pattern used to infer the signal.

Return this exact JSON shape:
{
  "concept": "main math concept being practiced",
  "misconception": "specific misconception observed, or null",
  "hints_used": "X student help request(s)",
  "engagement": "Active | Moderate | Low | Needs scaffold",
  "next_support": "suggested next pedagogical move, or null",
  "evidence": ["brief evidence item 1", "brief evidence item 2"]
}
"""


def _safe_json_loads(raw_content: str) -> dict[str, Any]:
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
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")

        if first_brace != -1 and last_brace > first_brace:
            try:
                payload = json.loads(cleaned[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    return payload if isinstance(payload, dict) else {}


def _is_user_message(message: dict) -> bool:
    return message.get("role") in {"user", "student", "human"}


def _is_assistant_message(message: dict) -> bool:
    return message.get("role") in {"assistant", "ai", "tutor"}


def _count_support_requests(chat_history: list[dict]) -> int:
    """Count authentic student support requests.

    The current UI stores a model-classified ``student_intent`` on user turns,
    so the counter does not need a long phrase list. Legacy conversations that
    do not have this metadata are left uncounted here and can still be
    interpreted by the LLM signal agent from the conversation text.
    """
    support_intents = {
        "learning_support",
        "language_support",
        "platform_help",
        "clarification",
    }

    count = 0

    for message in chat_history:
        if not _is_user_message(message):
            continue

        # New UI turns may explicitly mark whether the latest user turn
        # was counted as a real hint/support request. This prevents correct
        # answers or answer attempts from inflating Hint usage.
        if "counted_as_support_request" in message:
            if bool(message.get("counted_as_support_request")):
                count += 1
            continue

        intent = str(message.get("student_intent") or "").strip().lower()

        if intent in support_intents:
            count += 1

    return count


def _build_behavior_features(chat_history: list[dict]) -> dict[str, Any]:
    user_messages = [
        str(message.get("content", "")).strip()
        for message in chat_history
        if _is_user_message(message)
    ]

    assistant_messages = [
        str(message.get("content", "")).strip()
        for message in chat_history
        if _is_assistant_message(message)
    ]

    support_count = _count_support_requests(chat_history)
    latest_user_message = user_messages[-1] if user_messages else ""

    average_user_message_length = (
        sum(len(message.split()) for message in user_messages) / len(user_messages)
        if user_messages
        else 0
    )

    very_short_user_turns = sum(
        1
        for message in user_messages
        if len(message.split()) <= 3
    )

    return {
        "support_requests": support_count,
        "latest_user_message": latest_user_message,
        "average_user_message_length": round(average_user_message_length, 2),
        "user_turn_count": len(user_messages),
        "assistant_turn_count": len(assistant_messages),
        "very_short_user_turns": very_short_user_turns,
    }


def _clean_optional_text(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in {
        "",
        "null",
        "none",
        "n/a",
        "no misconception",
        "none detected",
        "no clear misconception detected",
        "—",
    }:
        return None

    return text


def _normalize_signal_payload(
    payload: dict[str, Any],
    behavior_features: dict[str, Any],
) -> dict[str, Any]:
    support_count = int(behavior_features.get("support_requests", 0))

    concept = str(
        payload.get("concept") or "No clear concept detected"
    ).strip()

    misconception = _clean_optional_text(
        payload.get("misconception")
    )

    engagement = str(
        payload.get("engagement") or "Moderate"
    ).strip()

    allowed_engagement = {
        "Active",
        "Moderate",
        "Low",
        "Needs scaffold",
    }

    if engagement not in allowed_engagement:
        engagement = "Moderate"

    next_support = _clean_optional_text(
        payload.get("next_support")
    )

    evidence = payload.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = []

    clean_evidence = [
        str(item).strip()
        for item in evidence
        if str(item).strip()
    ]

    if not clean_evidence:
        latest_message = behavior_features.get("latest_user_message", "")
        if latest_message:
            clean_evidence.append(
                f"Latest student message: {latest_message}"
            )
        clean_evidence.append(
            f"Support requests counted from behavior features: {support_count}."
        )

    return {
        "concept": concept,
        "misconception": misconception,
        "hints_used": f"{support_count} student help request(s)",
        "engagement": engagement,
        "next_support": next_support,
        "evidence": clean_evidence[:3],
        "generated_by": "Student Signal Agent",
    }


def _fallback_signals(
    chat_history: list[dict],
    behavior_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Safe fallback so the app still records a usable signal if the LLM call fails.
    This fallback avoids concept keyword classification and keeps the result conservative.
    """
    latest_user_message = behavior_features.get("latest_user_message", "")
    support_count = int(behavior_features.get("support_requests", 0))
    average_length = float(
        behavior_features.get("average_user_message_length", 0)
    )

    if support_count >= 3:
        engagement = "Needs scaffold"
    elif average_length <= 3:
        engagement = "Low"
    elif support_count >= 1:
        engagement = "Moderate"
    else:
        engagement = "Active"

    return {
        "concept": "No clear concept detected",
        "misconception": None,
        "hints_used": f"{support_count} student help request(s)",
        "engagement": engagement,
        "next_support": (
            "Ask the student to explain what the problem is asking before "
            "choosing an operation."
        ),
        "evidence": [
            "Fallback signal summary used because the Student Signal Agent was unavailable.",
            f"Latest student message: {latest_user_message}" if latest_user_message else "No latest student message recorded.",
        ],
        "generated_by": "Fallback signal summary",
    }


def extract_signals(
    chat_history: list[dict],
    curriculum_context: str = "",
    prior_signals_history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Extract structured learning signals from a tutoring interaction.

    This replaces hard-coded signal collection with an LLM-powered
    Student Signal Agent. The deterministic code only computes behavioral
    features and provides a safe fallback.
    """
    if len(chat_history) < 2:
        return {}

    recent = chat_history[-8:]
    prior_signals_history = prior_signals_history or []
    recent_prior_signals = prior_signals_history[-3:]

    conversation = "\n".join(
        f"{message.get('role', '').upper()}: {message.get('content', '')}"
        for message in recent
    )

    behavior_features = _build_behavior_features(chat_history)

    diagnostic_input = {
        "conversation": conversation,
        "curriculum_context": curriculum_context or "No curriculum context provided.",
        "prior_signals": recent_prior_signals,
        "behavior_features": behavior_features,
    }

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_signals(chat_history, behavior_features)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        diagnostic_input,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        payload = _safe_json_loads(str(response.content))
        normalized = _normalize_signal_payload(
            payload,
            behavior_features,
        )

        return normalized

    except Exception:
        return _fallback_signals(chat_history, behavior_features)