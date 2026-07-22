import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


_PROMPT = """\
You are the MathBridge Practice Problem Agent.

Generate ONE new Grade 6 math practice problem that is similar in mathematical structure
to the current problem, but uses a different context and different numbers.

Important rules:
- Do NOT simply reword the same problem.
- Do NOT repeat any problem from the already_seen_problems list.
- Keep the same underlying skill.
- Use student-friendly numbers.
- Use a new real-world context when possible.
- Do NOT give the final answer.
- Do NOT include a solution.
- Ask one Socratic follow-up question about how to start.
- If the original problem is measurement division, keep measurement division.
- If the original problem is equal sharing division, keep equal sharing division.
- If the original problem is unit rate, keep unit rate.
- Avoid fractions unless the original problem uses fractions.

Return ONLY valid JSON with this schema:
{
  "problem": "new practice problem",
  "problem_type": "measurement_division | equal_sharing | unit_rate | fraction_division | other",
  "student_prompt": "one Socratic question asking how the student would start",
  "reason": "brief explanation of why this problem is structurally similar"
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


def _normalize_key(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _is_duplicate(candidate: str, seen_problems: list[str]) -> bool:
    candidate_key = _normalize_key(candidate)

    if not candidate_key:
        return True

    for problem in seen_problems:
        if candidate_key == _normalize_key(problem):
            return True

    return False


def _looks_like_solution(text: str) -> bool:
    lowered = str(text or "").lower()

    forbidden_phrases = (
        "the answer is",
        "final answer",
        "solution:",
        "answer:",
        "so the answer",
    )

    return any(phrase in lowered for phrase in forbidden_phrases)


def _fallback_message() -> dict[str, str]:
    """
    Safety fallback only. This is not the main generation path.
    """
    return {
        "problem": "",
        "problem_type": "other",
        "student_prompt": (
            "I could not generate a new practice problem right now. "
            "Please try again."
        ),
        "reason": "Practice Problem Agent fallback.",
        "generated_by": "fallback",
    }


def generate_similar_practice_problem(
    current_problem: str,
    concept: str = "",
    seen_problems: list[str] | None = None,
    max_attempts: int = 3,
) -> dict[str, str]:
    seen_problems = seen_problems or []

    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_message()

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.75,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    for attempt in range(max_attempts):
        user_payload = {
            "current_problem": current_problem,
            "detected_concept": concept,
            "already_seen_problems": seen_problems[-12:],
            "attempt_number": attempt + 1,
            "diversity_instruction": (
                "Use a different object/context and different numbers from the current problem. "
                "Examples of possible contexts include classroom supplies, snacks, books, tickets, stickers, sports, garden items, or travel."
            ),
        }

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_PROMPT),
                    HumanMessage(
                        content=json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                ]
            )

            payload = _safe_json_loads(str(response.content))

            problem = str(payload.get("problem") or "").strip()
            student_prompt = str(payload.get("student_prompt") or "").strip()
            problem_type = str(payload.get("problem_type") or "other").strip()
            reason = str(payload.get("reason") or "").strip()

            if not problem:
                continue

            if _is_duplicate(problem, seen_problems + [current_problem]):
                continue

            if _looks_like_solution(problem) or _looks_like_solution(student_prompt):
                continue

            if not student_prompt:
                student_prompt = "What expression would you write to start solving this?"

            return {
                "problem": problem,
                "problem_type": problem_type,
                "student_prompt": student_prompt,
                "reason": reason,
                "generated_by": "Practice Problem Agent",
            }

        except Exception:
            continue

    return _fallback_message()