import json
import os
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, LLM_TEMPERATURE



_PROMPT = """\
You are the MathBridge Socratic Hint Agent.

Your job is to give ONE useful next hint for a Grade 6 math student.

Rules:
- Do not give the final answer in early hints.
- Do not repeat vague questions like "what operation should you use?" if the student already needs help.
- Give a concrete next step that helps the student make progress.
- Choose the strategy based on the actual numbers and wording.
- Do not force one fixed strategy.
- Use the student's current problem, recent chat, and hint level.

Concept definition rules:
- On the first hint for a new concept, briefly define the key math idea before giving the next step.
- The definition should be generated from the current problem, curriculum context, and recent chat.
- Do not use a hard-coded definition table.
- Keep the definition to one short sentence.
- If useful, include one simple "For example, ..." sentence using easier numbers or a simpler situation.
- The example must explain the concept but must not solve the exact current problem.
- After the definition/example, ask the student to apply the idea to their own problem.
- Do not repeat the definition in every later hint unless the student seems confused about the concept.

Formula and relationship rules:
- When the current problem naturally depends on a standard formula or relationship, briefly name or state the relevant formula.
- Use only ONE relevant formula or relationship at a time.
- Do not dump a formula list.
- The formula should support reasoning, not replace reasoning.
- If the formula would directly reveal the final answer, state the relationship first and ask the student to apply it.

Example scaffold rules:
- When useful, include one short "For example, ..." sentence.
- The example should use simpler or parallel numbers.
- The example must show the same reasoning pattern.
- The example must NOT solve the exact current problem.
- After the example, ask the student to apply the same idea to their own problem.

Hint-level behavior:
- If hint_level is 0 or 1, include a short concept definition when helpful.
- If hint_level is 2 or higher, focus more on the next actionable step instead of repeating the definition.
- If the student has already attempted an answer, respond to that attempt rather than restarting from the definition.

Contextualization rules:
- Use the contextualization context when it naturally helps the student understand the current problem.
- Prefer familiar real-life contexts, interests, and learning style from the profile.
- Do not force personalization if it would make the math unclear.
- Keep the math structure and curriculum objective unchanged.

Style rules:
- Keep the hint concise and friendly.
- Ask exactly one follow-up question at the end.
- Return ONLY valid JSON.

Return this JSON shape:
{
  "hint": "one Socratic hint",
  "strategy": "brief name of the chosen strategy",
  "reveals_final_answer": false
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


def _fallback_hint(current_problem: str) -> str:
    return (
        "Let's identify what the numbers mean first. "
        "For example, in a sharing or grouping problem, one number tells the total and another tells the group size or number of groups. "
        "Which number in your problem is the total?"
    )

def _parse_json_response(raw_content: str) -> dict[str, Any]:
    cleaned = str(raw_content or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$",
                         "",
                         cleaned)

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




def generate_socratic_hint(
    current_problem: str = "",
    student_request: str | int = "",
    hint_level: int = 0,
    recent_chat: list[dict[str, Any]] | None = None,
    curriculum_context: str = "",
    contextualization_context: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    LLM-powered Socratic hint agent.

    This version is intentionally compatible with older tutor_chain calls, such as:
      generate_socratic_hint(problem_text, hint_level, recent_chat=...)
      generate_socratic_hint(problem_text, student_request, hint_level=...)
    """

    # Compatibility: sometimes tutor_chain passes hint_level as the second positional arg.
    if isinstance(student_request, int):
        hint_level = student_request
        student_request = ""

    problem_text = str(
        current_problem
        or kwargs.get("problem_text", "")
        or kwargs.get("latest_user_text", "")
    ).strip()

    request_text = str(
        student_request
        or kwargs.get("student_request", "")
        or kwargs.get("user_text", "")
    ).strip()

    recent_chat = recent_chat or kwargs.get("history", []) or []
    contextualization_context = str(
        contextualization_context
        or kwargs.get("contextualization_context", "")
        or ""
    ).strip()

    fallback = {
        "hint": (
            "Think about the relationship or formula that connects the numbers in the problem. "
            "For example, try a simpler case with smaller numbers first. "
            "What same step could you apply to this problem?"
        ),
        "strategy": "fallback scaffold",
        "reveals_final_answer": False,
    }

    if not os.getenv("OPENAI_API_KEY"):
        return fallback

    payload = {
        "current_problem": problem_text,
        "student_request": request_text,
        "hint_level": hint_level,
        "recent_chat": recent_chat[-6:] if isinstance(recent_chat, list) else [],
        "curriculum_context": str(curriculum_context or "")[:2000],
        "contextualization_context": contextualization_context[:2500],
        "instruction": (
            "Generate the next useful Socratic hint. "
            "Use the actual numbers and wording. "
            "If this is an early hint for the concept, include a one-sentence concept definition. "
            "When helpful, add a short 'For example' scaffold with simpler or parallel numbers. "
            "Do not solve the exact problem unless the student has already given the correct answer. "
            "Ask exactly one follow-up question at the end."
        ),
    }

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        response = llm.invoke(
            [
                SystemMessage(content=_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        parsed = _parse_json_response(str(response.content))

        hint = str(parsed.get("hint", "")).strip()

        if not hint:
            return fallback

        return {
            "hint": hint,
            "strategy": str(parsed.get("strategy", "LLM Socratic scaffold")).strip(),
            "reveals_final_answer": bool(parsed.get("reveals_final_answer", False)),
        }

    except Exception as exc:
        print("HINT AGENT ERROR:", repr(exc))
        return fallback