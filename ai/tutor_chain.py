import json
import os
import re
from typing import Any
from ai.answer_evaluator import evaluate_student_answer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ai.hint_agent import generate_socratic_hint
from ai.rag import retrieve_curriculum_context
from config import LLM_MODEL, LLM_TEMPERATURE
import time

_SYSTEM = """\
You are MathBridge, a patient and encouraging Grade 6 math tutor.

Default student-facing language is clear, natural English. If the student explicitly asks for another language, translation, or bilingual support, respond in that language or bilingually.

You are also a flexible learning-platform assistant, not only a math-answer machine. The student may ask how to use the platform, ask for translation, ask what a word means, ask for a simpler explanation, ask about the current page, or make a short conversational learning request. Handle these naturally while keeping the learning goal in view.

Core tutoring goal:
- Help the student think through the next step instead of immediately giving
  the final answer.
- Use a Socratic approach: one short hint, one guiding question, or one
  concrete next step at a time.

Current hint level: {hint_level}

Socratic rules:
- Use a Socratic tutoring style.
- Do NOT reveal the final numeric answer in the first response.
- Do NOT give the full procedure or shortcut rule in the first response.
- For the first 1–2 tutor turns, ask the student to reason about the meaning of the problem before giving an algorithm.
- For fraction division, do not immediately say "multiply by the reciprocal" unless the student has already tried a step or hint_level is at least 2.
- Give only one small hint at a time.
- End every response with one guiding question or one concrete next step.
- If hint_level is 0 or 1, focus on conceptual understanding.
- If hint_level is 2, you may introduce the next operation but still do not compute the final answer.
- If hint_level is 3 or higher, you may give a full explanation.
- If the student only asks for a hint, steps, or another explanation, do not say "Great", "Exactly", or "You're on the right track" unless the student actually made an attempt.
- If the student has not attempted a step yet, start with a neutral phrase such as "Here is one hint" or "Let's try one small step."
- Praise only when the student's message contains reasoning, an attempted answer, or a correct intermediate step.
- Do not praise the student unless the student actually made an attempt or gave a reasoning step.
Actionable hint rules:
- A hint must be useful enough for the student to do the next step.
- Do not only ask "what operation should you use?" repeatedly.
- For equal-sharing division, guide the student toward a division expression and then a nearby multiplication or decomposition strategy.
- For example, for "180 apples divided into 4 boxes":
  - Early hint: "This asks for apples in each box. Write 180 ÷ 4."
  - Next hint: "Think of 180 as 160 + 20. 160 ÷ 4 = 40. What is 20 ÷ 4?"
  - Later hint: "Combine the two parts. What is 40 plus the amount from 20 ÷ 4?"
- Do not state the final answer until hint_level is 3 or higher.

Flexible platform-use guidelines:
- If the student asks to translate, rephrase, simplify, or switch language, do that request directly and continue with one helpful next step. Do not scold the student for not answering the math problem.
- If the student asks how to use the app, what a section means, or what to do next, answer as a product tutor and guide them to the appropriate next action.
- If the request is clearly outside learning or platform support, answer briefly if harmless, then invite the student back to the math or learning task.
- Preserve the current math problem when the student asks a follow-up about it.

Tutoring guidelines:
- Use language appropriate for a Grade 6 student.
- Celebrate effort and gently redirect mistakes.
- Use concrete examples such as fraction bars, tape diagrams, number lines,
  measuring cups, and equal groups.
- When curriculum notes are provided, align the hint or question with them.
- If curriculum notes are not relevant, do not force them into the answer.
- Keep the response concise, usually 2 to 4 sentences.
- End every response with either one question or one concrete next step.

Contextualization rules:
- Use the provided contextualization profile to adapt examples, wording, and scaffolds when it naturally supports the math.
- Prefer the student's real-life contexts and learning style for examples, but do not force personalization.
- Keep the original mathematical structure and curriculum objective unchanged.
- Do not reveal final answers early just because an example is personalized.
- If the learner profile is not useful for the current problem, give a normal curriculum-aligned Socratic hint.

Visual-response guidelines:
- Set "needs_visual" to true only when a visual would meaningfully improve
  understanding.
- For fraction-division questions, prefer the visual type
  "fraction_division_bar".
- For "fraction_division_bar", visual_data must contain:
  "dividend": a plain fraction or whole-number string, such as "3/4" or "2"
  "divisor": a plain fraction or whole-number string, such as "1/8" or "3"
- Do not invent numbers that are not supported by the student's question or
  the curriculum context.
- If no visual is needed, use "visual_type": "none" and an empty visual_data
  object.

Return ONLY valid JSON in exactly this shape:
{
  "answer": "A short Socratic hint, guiding question, platform-help response, translation, or student-facing explanation. Match the student's requested language when appropriate. Do not reveal the final numeric answer when hint_level is below 3 unless the student is asking a non-math platform/language question.",
  "student_intent": "math_problem | answer_attempt | learning_support | language_support | platform_help | general_conversation",
  "needs_visual": true,
  "visual_type": "fraction_division_bar",
  "visual_data": {
    "dividend": "3/4",
    "divisor": "1/8"
  }
}

{curriculum_context}
"""


_ALLOWED_VISUAL_TYPES = {
    "none",
    "fraction_division_bar",
}


def _estimate_hint_level(history: list[dict]) -> int:
    """
    Estimate how much help the student has already received in the current
    chat. This keeps the Socratic behavior backward-compatible with the
    existing UI, which currently calls this module without passing a separate
    hint counter.
    """
    if not history:
        return 0

    assistant_turns = sum(
        1
        for msg in history
        if msg.get("role") in {"assistant", "ai"}
    )

    return max(0, min(assistant_turns, 3))


def _build_messages(
    user_message: str,
    history: list[dict],
    curriculum_context: str,
    hint_level: int | None = None,
    contextualization_context: str = "",
):
    context_sections = []

    if curriculum_context:
        context_sections.append(
            f"Relevant curriculum notes:\n{curriculum_context}"
        )

    if contextualization_context:
        context_sections.append(
            f"Contextualization notes:\n{contextualization_context}"
        )

    context_block = "\n\n".join(context_sections)

    if hint_level is None:
        hint_level = _estimate_hint_level(history)

    hint_level = max(0, min(int(hint_level), 3))

    system_prompt = (
        _SYSTEM
        .replace("{curriculum_context}", context_block)
        .replace("{hint_level}", str(hint_level))
    )

    messages = [SystemMessage(content=system_prompt)]

    for msg in history:
        if msg["role"] == "user":
            messages.append(
                HumanMessage(content=msg["content"])
            )
        else:
            messages.append(
                AIMessage(content=msg["content"])
            )

    messages.append(HumanMessage(content=user_message))

    return messages


def _parse_json_response(
    raw_content: str,
    hint_level: int = 0,
    latest_user_text: str = "",
    recent_chat: list[dict] | None = None,
) -> dict[str, Any]:
    
    """Parse the model response and safely fall back to text-only output."""

    cleaned = raw_content.strip()

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
                payload = json.loads(
                    cleaned[first_brace:last_brace + 1]
                )
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}

    answer = str(
        payload.get("answer") or raw_content
    ).strip()

    if not answer:
        answer = "Let's work through this together. What is one step you could try first?"

    visual_type = str(
        payload.get("visual_type", "none")
    ).strip()

    if visual_type not in _ALLOWED_VISUAL_TYPES:
        visual_type = "none"

    visual_data = payload.get("visual_data", {})
    if not isinstance(visual_data, dict):
        visual_data = {}

    needs_visual = bool(
        payload.get("needs_visual", False)
    )

    if visual_type == "none":
        needs_visual = False
        visual_data = {}

    student_intent = str(
        payload.get("student_intent") or "math_tutoring"
    ).strip()

    payload = {
    "answer": answer,
    "student_intent": student_intent,
    "needs_visual": needs_visual,
    "visual_type": visual_type,
    "visual_data": visual_data,
    }

    return _apply_socratic_guard(
        payload,
        hint_level,
        latest_user_text,
        recent_chat=recent_chat,
    )

def _safe_socratic_fallback(hint_level: int) -> str:
    if hint_level <= 0:
        return (
            "Let's try one small step first. "
            "What quantities do you see in the problem, and what are they asking you to find?"
        )

    if hint_level == 1:
        return (
            "Here is one hint: think about what one group or one unit would represent. "
            "What operation might help you compare the total amount to one unit?"
        )

    return (
        "Now choose the operation that matches the situation, but do not compute it yet. "
        "Can you write the expression you would use?"
    )


def _looks_answer_revealing(answer: str, hint_level: int) -> bool:
    if hint_level >= 3:
        return False

    text = str(answer or "").lower()

    answer_phrases = [
        "the answer is",
        "final answer",
        "so the answer",
        "therefore the answer",
        "the result is",
        "it equals",
        "equals",
    ]

    if any(phrase in text for phrase in answer_phrases):
        return True

    if hint_level < 2 and "multiply by the reciprocal" in text:
        return True

    if hint_level < 2 and "reciprocal" in text and "fraction" in text:
        return True

    # Avoid early responses that show a completed numeric equation, but do not
    # block ordinary explanatory symbols or translated concept statements.
    if re.search(r"\b\d+(?:\.\d+)?\s*=\s*\d+(?:\.\d+)?\b", text):
        return True

    return False

def _remove_unearned_praise(answer: str, latest_user_text: str) -> str:
    """
    Remove praise such as 'Great!' when the student only clicked a support
    button and did not actually attempt reasoning.
    """
    user_text = str(latest_user_text or "").lower()

    support_only_markers = (
        "give me one hint",
        "help me take the next step",
        "give me a similar practice problem",
        "without giving the final answer",
    )

    is_support_only = any(
        marker in user_text
        for marker in support_only_markers
    )

    if not is_support_only:
        return answer

    answer = re.sub(
        r"^\s*(great|exactly|nice work|good job|you're right|you are right)[!,.]?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    if not answer:
        return "Let's try one small step. What operation would match this situation?"

    return answer

def _is_support_only_request(text: str) -> bool:
    lowered = str(text or "").lower()

    markers = (
        "give me one hint",
        "help me take the next step",
        "give me a similar practice problem",
        "without giving the final answer",
        "student request:",
    )

    return any(marker in lowered for marker in markers)


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
def _find_current_problem_for_check(
    history: list[dict] | None,
    current_user_message: str = "",
) -> str:
    help_like_phrases = (
        "hint",
        "next step",
        "similar problem",
        "help me",
        "do not give",
        "give me one",
        "take the next step",
    )

    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue

        content = str(msg.get("content", "")).strip()
        lowered = content.lower()

        if not content:
            continue

        if content == str(current_user_message).strip():
            continue

        if any(phrase in lowered for phrase in help_like_phrases):
            continue

        # IMPORTANT:
        # Short division expressions like "12/3" are problems,
        # not student answers. Do not skip them.
        if _looks_like_division_problem_text(content):
            return content

        # Short numeric messages like "4" or "15" are usually answers,
        # not original problems.
        if len(content) <= 8 and any(ch.isdigit() for ch in content):
            continue

        return content

    return ""

def _looks_like_answer_attempt_message(
    user_message: str,
    current_problem: str,
) -> bool:
    text = str(user_message or "").strip().lower()

    if not current_problem:
        return False

    if not text:
        return False

    if _is_support_only_request(text):
        return False

    # Common answer forms: 3, 4, 15, 7.5, 3 groups, 4 apples.
    if re.fullmatch(
        r"-?\d+(?:\.\d+)?\s*(groups?|boxes?|bags?|apples?|items?|units?)?",
        text,
    ):
        return True

    if re.fullmatch(
        r"(?:answer|the answer is|it is|it's)\s*:?\s*-?\d+(?:\.\d+)?",
        text,
    ):
        return True

    return False



def _extract_problem_text(text: str) -> str:
    """
    ui/student.py may send:
    Current math problem:
    ...
    Student request:
    ...
    """
    raw = str(text or "")

    if "Current math problem:" in raw and "Student request:" in raw:
        start = raw.find("Current math problem:") + len("Current math problem:")
        end = raw.find("Student request:")
        return raw[start:end].strip()

    return raw.strip()


def _extract_first_two_numbers(text: str) -> tuple[int | None, int | None]:
    numbers = re.findall(r"\b\d+\b", str(text or ""))

    if len(numbers) < 2:
        return None, None

    return int(numbers[0]), int(numbers[1])


def _looks_like_equal_sharing_problem(text: str) -> bool:
    lowered = str(text or "").lower()

    return bool(
        re.search(
            r"\b(divide|divided|split|share|shared|put)\b.*\b(into|among|between|across)\b",
            lowered,
        )
        or re.search(
            r"\b(each box|each group|per box|per group|equal groups|equally)\b",
            lowered,
        )
        or (
            re.search(r"\bboxes?|groups?|friends?|students?|people|bags?\b", lowered)
            and re.search(r"\bdivide|divided|split|share|shared|into|among\b", lowered)
        )
    )

def _looks_like_measurement_division_problem(text: str) -> bool:
    lowered = str(text or "").lower()

    return bool(
        re.search(
            r"\bhow many groups of\b|\bhow many sets of\b|\bhow many boxes of\b|\bhow many bags of\b",
            lowered,
        )
        or re.search(
            r"\bgroups of\s+\d+\b|\bsets of\s+\d+\b|\bboxes of\s+\d+\b|\bbags of\s+\d+\b",
            lowered,
        )
        or re.search(
            r"\bhow many\b.*\bcan be made from\b",
            lowered,
        )
    )


def _extract_measurement_division_numbers(text: str) -> tuple[int | None, int | None]:
    """
    For questions like:
    'How many groups of 4 apples can be made from 180 apples?'

    Return:
    total = 180
    group_size = 4
    """
    raw = str(text or "").lower()

    group_size_match = re.search(
        r"\b(?:groups|sets|boxes|bags)\s+of\s+(\d+)\b",
        raw,
    )

    all_numbers = [int(n) for n in re.findall(r"\b\d+\b", raw)]

    if not all_numbers:
        return None, None

    group_size = None

    if group_size_match:
        group_size = int(group_size_match.group(1))

    if group_size is None and len(all_numbers) >= 2:
        group_size = all_numbers[0]

    total = None

    if len(all_numbers) >= 2:
        # In "groups of 4 from 180", the larger later number is usually total.
        candidates = [n for n in all_numbers if n != group_size]

        if candidates:
            total = max(candidates)

    return total, group_size
def _is_similar_problem_request(text: str) -> bool:
    lowered = str(text or "").lower()

    return (
        "similar practice problem" in lowered
        or "similar problem" in lowered
    )


def _similar_problem_scaffold(latest_user_text: str) -> str | None:
    """
    Generate a small related practice problem instead of continuing the
    current problem.
    """
    problem_text = _extract_problem_text(latest_user_text)

    if _looks_like_measurement_division_problem(problem_text):
        return (
            "Here is a similar practice problem:\n\n"
            "How many groups of 5 oranges can be made from 20 oranges?\n\n"
            "Think about what each number means first: 20 is the total number "
            "of oranges, and 5 is the number of oranges in each group. "
            "What division expression would you write?"
        )

    if _looks_like_equal_sharing_problem(problem_text):
        return (
            "Here is a similar practice problem:\n\n"
            "20 oranges are divided equally into 5 bags. "
            "How many oranges go in each bag?\n\n"
            "Think about what each number means first: 20 is the total number "
            "of oranges, and 5 is the number of equal bags. "
            "What division expression would you write?"
        )

    return (
        "Here is a similar practice problem:\n\n"
        "20 oranges are divided equally into 5 bags. "
        "How many oranges go in each bag?\n\n"
        "What division expression would you write first?"
    )

def _actionable_support_scaffold(
    latest_user_text: str,
    hint_level: int,
    recent_chat: list[dict] | None = None,
) -> str | None:
    """
    LLM-powered Socratic hint scaffold.

    The model chooses the teaching strategy. Deterministic code only extracts
    the current problem and routes support-only requests to the hint agent.
    """
    if not _is_support_only_request(latest_user_text):
        return None

    problem_text = _extract_problem_text(latest_user_text)

    result = generate_socratic_hint(
        current_problem=problem_text,
        student_request=latest_user_text,
        hint_level=hint_level,
        recent_chat=recent_chat or [],
    )

    hint = str(result.get("hint") or "").strip()

    if not hint:
        return None

    return hint



def _numeric_answer_feedback_scaffold(
    latest_user_text: str,
    history: list[dict] | None = None,
) -> str | None:
    user_text = str(latest_user_text or "").strip()

    if not re.fullmatch(r"\d+(\.\d+)?", user_text):
        return None

    student_answer = float(user_text)

    # Find the most recent original problem from history.
    problem_text = ""
    for message in reversed(history or []):
        if message.get("role") == "user":
            candidate = str(message.get("content", ""))
            if not re.fullmatch(r"\d+(\.\d+)?", candidate.strip()):
                problem_text = candidate
                break

    if not _looks_like_measurement_division_problem(problem_text):
        return None

    total, group_size = _extract_measurement_division_numbers(problem_text)

    if total is None or group_size is None or group_size == 0:
        return None

    product = student_answer * group_size

    if abs(product - total) < 1e-9:
        return (
            f"Yes — if there are {int(student_answer)} groups and each group has "
            f"{group_size} apples, then {int(student_answer)} × {group_size} = {total}. "
            f"So your answer matches the total. What does {int(student_answer)} represent in the story?"
        )

    if product < total:
        remaining = total - product
        return (
            f"Let's check {int(student_answer)}. "
            f"If there are {int(student_answer)} groups and each group has {group_size} apples, "
            f"that uses {int(student_answer)} × {group_size} = {int(product)} apples. "
            f"That is less than {total}, so {int(student_answer)} groups is too small. "
            f"There are still {int(remaining)} apples left. "
            f"How many more groups of {group_size} can you make from {int(remaining)}?"
        )

    return (
        f"Let's check {int(student_answer)}. "
        f"{int(student_answer)} × {group_size} = {int(product)}, which is more than {total}. "
        f"So {int(student_answer)} groups is too many. "
        f"What smaller number of groups could you try?"
    )


    
def _apply_socratic_guard(
    payload: dict[str, Any],
    hint_level: int,
    latest_user_text: str = "",
    recent_chat: list[dict] | None = None,
) -> dict[str, Any]:
    answer = str(payload.get("answer") or "").strip()

    answer = _remove_unearned_praise(
        answer,
        latest_user_text,
    )
    similar_problem = _similar_problem_scaffold(
        latest_user_text,
    )

    if _is_similar_problem_request(latest_user_text):
        payload["answer"] = similar_problem
        payload["needs_visual"] = False
        payload["visual_type"] = "none"
        payload["visual_data"] = {}
        return payload


    support_scaffold = _actionable_support_scaffold(
        latest_user_text,
        hint_level,
        recent_chat=recent_chat,
    )

    # If we have a reliable deterministic scaffold for this kind of problem,
    # use it. This prevents vague repeated hints like "choose the operation."
    if support_scaffold and hint_level < 4:
        payload["answer"] = support_scaffold
        payload["needs_visual"] = False
        payload["visual_type"] = "none"
        payload["visual_data"] = {}
        return payload

    intent = str(payload.get("student_intent") or "").lower()

    if intent not in {"language_support", "platform_help", "general_conversation"}:
        if _looks_answer_revealing(answer, hint_level):
            answer = _safe_socratic_fallback(hint_level)

    payload["answer"] = answer

    if hint_level < 3:
        payload["needs_visual"] = False
        payload["visual_type"] = "none"
        payload["visual_data"] = {}

    return payload




def _parse_number_like(text: str) -> float | None:
    """Parse a short student numeric answer such as 4, 4 groups, or 1/2."""
    raw = str(text or "").strip().lower()

    if not raw:
        return None

    raw = re.sub(
        r"^(?:answer|the answer is|it is|it's)\s*:?\s*",
        "",
        raw,
    ).strip()

    # Keep only the first simple number or fraction. This handles
    # answers such as "4 groups" without needing the unit label.
    fraction_match = re.match(
        r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\b",
        raw,
    )
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator

    number_match = re.match(r"^(-?\d+(?:\.\d+)?)\b", raw)
    if not number_match:
        return None

    return float(number_match.group(1))


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))

    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text


def _extract_simple_division_numbers(problem_text: str) -> tuple[float | None, float | None, str]:
    """
    Extract total and divisor/group count from common Grade 6 division prompts.

    Returns (total, divisor, mode), where mode is one of:
    - "expression": 12/3 or 12 divided by 3
    - "measurement": how many groups of 3 can be made from 12
    - "sharing": 12 apples divided into 3 boxes
    """
    text = _extract_problem_text(problem_text)
    lowered = str(text or "").lower()

    expression_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:/|÷)\s*(-?\d+(?:\.\d+)?)",
        lowered,
    )
    if not expression_match:
        expression_match = re.search(
            r"(-?\d+(?:\.\d+)?)\s+divided\s+by\s+(-?\d+(?:\.\d+)?)",
            lowered,
        )

    if expression_match:
        return (
            float(expression_match.group(1)),
            float(expression_match.group(2)),
            "expression",
        )

    if _looks_like_measurement_division_problem(lowered):
        total, group_size = _extract_measurement_division_numbers(lowered)
        if total is not None and group_size not in {None, 0}:
            return float(total), float(group_size), "measurement"

    if _looks_like_equal_sharing_problem(lowered):
        numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", lowered)]
        if len(numbers) >= 2 and numbers[1] != 0:
            return numbers[0], numbers[1], "sharing"

    return None, None, ""


def _deterministic_division_answer_payload(
    current_problem: str,
    latest_student_message: str,
) -> dict[str, Any] | None:
    """
    Fast, reliable answer checking for simple division cases.

    This catches cases where the LLM answer evaluator is too conservative and
    says a short number like "4" is not an answer attempt. It is intentionally
    narrow: if the problem is not a simple division prompt, return None and let
    the normal evaluator / tutor chain handle it.
    """
    student_value = _parse_number_like(latest_student_message)
    if student_value is None:
        return None

    total, divisor, mode = _extract_simple_division_numbers(current_problem)
    if total is None or divisor in {None, 0}:
        return None

    expected = total / divisor
    is_correct = abs(student_value - expected) < 1e-6

    total_text = _format_number(total)
    divisor_text = _format_number(divisor)
    student_text = _format_number(student_value)
    expected_text = _format_number(expected)

    if is_correct:
        if mode == "measurement":
            feedback = (
                f"Correct — {student_text} groups of {divisor_text} make {total_text}. "
                f"Nice work. What does {student_text} represent in the story?"
            )
        elif mode == "sharing":
            feedback = (
                f"Correct — if {total_text} is shared into {divisor_text} equal groups, "
                f"each group gets {student_text}. Nice work."
            )
        else:
            feedback = (
                f"Correct — {divisor_text} × {student_text} = {total_text}, "
                f"so {total_text} ÷ {divisor_text} = {student_text}. Nice work."
            )
    else:
        product = student_value * divisor
        product_text = _format_number(product)

        if product < total:
            feedback = (
                f"Let's check {student_text}. {student_text} × {divisor_text} = {product_text}, "
                f"which is less than {total_text}. Try a larger number."
            )
        elif product > total:
            feedback = (
                f"Let's check {student_text}. {student_text} × {divisor_text} = {product_text}, "
                f"which is more than {total_text}. Try a smaller number."
            )
        else:
            feedback = (
                f"Let's check {student_text}. The quotient should be {expected_text}. "
                f"Can you use multiplication to verify it?"
            )

    return {
        "answer": feedback,
        "student_intent": "answer_attempt",
        "needs_visual": False,
        "visual_type": "none",
        "visual_data": {},
        "is_answer_attempt": True,
        "is_correct_answer": is_correct,
        "should_stop_hinting": is_correct,
    }

def get_tutor_response_payload_with_context(
    user_message: str,
    history: list[dict],
    hint_level: int | None = None,
    current_problem_text: str | None = None,
    force_answer_attempt: bool = False,
    contextualization_context: str = "",
) -> tuple[dict[str, Any], dict]:
    """
    Return a structured tutor response and curriculum retrieval metadata.
    """
    timings = {}
    total_start = time.perf_counter()

    if hint_level is None:
        hint_level = _estimate_hint_level(history)

    is_support_only = _is_support_only_request(user_message)

    # ---------------------------------------------------------
    # Support-only shortcut
    # Hint / similar-problem clicks are not answer attempts.
    # ---------------------------------------------------------
    if is_support_only:
        t0 = time.perf_counter()

        problem_text = _extract_problem_text(user_message)

        hint_result = generate_socratic_hint(
            current_problem=problem_text,
            student_request=user_message,
            hint_level=hint_level,
            recent_chat=history or [],
            contextualization_context=contextualization_context,
        )

        hint_text = str(
            hint_result.get("hint") or ""
        ).strip()

        if not hint_text:
            hint_text = _safe_socratic_fallback(hint_level)

        retrieval = {
            "context": "",
            "matches": [],
            "retrieval_method": "skipped_for_support_only_request",
        }

        response_payload = {
            "answer": hint_text,
            "student_intent": "learning_support",
            "needs_visual": False,
            "visual_type": "none",
            "visual_data": {},
            "is_answer_attempt": False,
            "is_correct_answer": False,
            "should_stop_hinting": False,
        }

        timings["has_current_problem"] = 0.0
        timings["force_answer_attempt"] = 0.0
        timings["retrieval"] = 0.0
        timings.setdefault("answer_evaluator", 0.0)
        timings.setdefault("answer_evaluator_skipped", 1.0)
        timings["hint_agent"] = time.perf_counter() - t0
        timings["tutor_llm"] = 0.0
        timings["total"] = time.perf_counter() - total_start
        response_payload["_timings"] = dict(timings)

        return response_payload, retrieval

    # ---------------------------------------------------------
    # Find active current problem
    # ---------------------------------------------------------
    # current_problem_text has three meanings:
    # - None: caller did not provide a problem, so infer from history.
    # - "": caller explicitly says this is a new problem, so do not reuse old history.
    # - non-empty string: use that active problem for answer checking.
    if current_problem_text is None:
        current_problem_for_check = _find_current_problem_for_check(
            history,
            current_user_message=user_message,
        )
    else:
        current_problem_for_check = str(current_problem_text).strip()

    timings["has_current_problem"] = (
        1.0 if current_problem_for_check else 0.0
    )
    timings["force_answer_attempt"] = (
        1.0 if force_answer_attempt else 0.0
    )

    # ---------------------------------------------------------
    # General answer-evaluator route
    # ---------------------------------------------------------
    # Important:
    # If there is an active current problem, any non-control student turn
    # should be evaluated first. The evaluator decides whether it is a number,
    # fraction, unit answer, equation, explanation, geometry description,
    # partial answer, help request, or unrelated text.
    should_try_answer_evaluator = bool(
        current_problem_for_check
        and not is_support_only
    )

    if should_try_answer_evaluator:
        deterministic_payload = _deterministic_division_answer_payload(
            current_problem=current_problem_for_check,
            latest_student_message=str(user_message),
        )

        if deterministic_payload is not None:
            timings["retrieval"] = 0.0
            timings["answer_evaluator"] = 0.0
            timings["answer_evaluator_skipped"] = 1.0
            timings["answer_evaluator_deterministic"] = 1.0
            timings["answer_evaluator_not_answer"] = 0.0
            timings["tutor_llm"] = 0.0
            timings["total"] = time.perf_counter() - total_start
            deterministic_payload["_timings"] = dict(timings)

            return deterministic_payload, {
                "context": "",
                "matches": [],
                "retrieval_method": "skipped_for_deterministic_answer_check",
            }

        t0 = time.perf_counter()

        answer_check = evaluate_student_answer(
            current_problem=current_problem_for_check,
            latest_student_message=str(user_message),
            recent_chat=history,
        )

        timings["retrieval"] = 0.0
        timings["answer_evaluator"] = time.perf_counter() - t0
        timings["answer_evaluator_skipped"] = 0.0
        timings.setdefault("answer_evaluator_deterministic", 0.0)

        raw_is_answer_attempt = bool(
            answer_check.get("is_answer_attempt", False)
        )
        is_correct = bool(
            answer_check.get("is_correct", False)
        )
    

        # The deterministic checker above handles short numeric answers.
        # If the LLM evaluator still says this is not an answer attempt, do not
        # force a generic incorrect response; fall through to the normal tutor.
        is_answer_attempt = bool(raw_is_answer_attempt)

        timings["answer_evaluator_not_answer"] = (
            0.0 if raw_is_answer_attempt else 1.0
        )

        if is_answer_attempt:
            feedback = str(
                answer_check.get("feedback") or ""
            ).strip()

            if not feedback:
                if is_correct:
                    feedback = "Correct. Nice work."
                else:
                    feedback = (
                        "Let's check that response. "
                        "What fact, model, or reasoning step could you use to verify it?"
                    )

            response_payload = {
                "answer": feedback,
                "student_intent": "answer_attempt",
                "needs_visual": False,
                "visual_type": "none",
                "visual_data": {},
                "is_answer_attempt": True,
                "is_correct_answer": is_correct,
                "should_stop_hinting": is_correct,
            }

            timings["tutor_llm"] = 0.0
            timings["total"] = time.perf_counter() - total_start
            response_payload["_timings"] = dict(timings)

            return response_payload, {
                "context": "",
                "matches": [],
                "retrieval_method": "skipped_for_answer_attempt",
            }

    # Fall through to normal tutoring when there is no active answer attempt.
    timings.setdefault("answer_evaluator_not_answer", 1.0)

    # ---------------------------------------------------------
    # New problem / real tutoring turn
    # ---------------------------------------------------------
    t0 = time.perf_counter()

    retrieval = retrieve_curriculum_context(user_message)
    curriculum_context = retrieval.get("context", "")

    timings["retrieval"] = time.perf_counter() - t0
    timings.setdefault("answer_evaluator", 0.0)
    timings.setdefault("answer_evaluator_skipped", 1.0)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    messages = _build_messages(
        user_message=user_message,
        history=history,
        curriculum_context=curriculum_context,
        hint_level=hint_level,
        contextualization_context=contextualization_context,
    )

    t0 = time.perf_counter()

    raw_content = llm.invoke(messages).content

    response_payload = _parse_json_response(
        raw_content,
        hint_level=hint_level,
        latest_user_text=user_message,
        recent_chat=history,
    )

    response_payload.setdefault("student_intent", "math_tutoring")
    response_payload.setdefault("is_answer_attempt", False)
    response_payload.setdefault("is_correct_answer", False)
    response_payload.setdefault("should_stop_hinting", False)

    timings["tutor_llm"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - total_start
    response_payload["_timings"] = dict(timings)

    return response_payload, retrieval



def get_tutor_response_with_context(user_input, chat_history=None):
    payload, _retrieval = get_tutor_response_payload_with_context(
        user_input,
        chat_history or [],
    )
    return payload.get(
        "answer",
        "Let's work through this together.",
    )


def get_tutor_response(user_input, chat_history=None):
    return get_tutor_response_with_context(
        user_input,
        chat_history or [],
    )