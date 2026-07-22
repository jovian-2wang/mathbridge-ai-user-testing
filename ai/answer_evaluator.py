import json
import os
import re
from typing import Any
from config import LLM_MODEL
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


ANSWER_EVALUATOR_PROMPT = """
You are a K-12 math answer evaluator.

Your job is to evaluate the student's latest message against the current math problem.

Use mathematical meaning, not exact wording.

Decide whether the latest student message is:
1. an answer attempt,
2. a help request,
3. a new question,
4. or unrelated text.

If it is an answer attempt, judge whether it correctly answers the current problem.
The student may answer using numbers, words, units, equations, reasoning, or a description.
Accept mathematically equivalent forms when the meaning matches the problem.

Use the current problem and recent chat history to infer what quantity the student is trying to answer.
Do not require the same wording as the expected solution.
Unit tolerance rules:
- If the student's numeric value is correct but the unit is missing, incomplete, or slightly imprecise, still mark the answer as correct.
- In the feedback, restate the full correct unit.
- Example: if the problem asks for miles per hour and the student writes "60" or "60 miles", treat it as correct and say "Correct — the unit rate is 60 miles per hour."
- Example: if the problem asks for cost per notebook and the student writes "3" or "$3", treat it as correct and restate "$3 per notebook."
- Only mark the answer incorrect if the numeric value is wrong, the unit contradicts the problem, or the response answers a different quantity.

If the answer is correct:
- confirm briefly,
- do not continue hinting,
- do not ask another checking question.

If the answer is incorrect or incomplete:
- do not reveal the final answer,
- give one small checking question or one small next step.

Return only valid JSON in this exact shape:

{
  "is_answer_attempt": true or false,
  "is_correct": true or false,
  "should_stop_hinting": true or false,
  "feedback": "student-facing response",
  "reason": "brief explanation for the system"
}
"""


def _parse_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            try:
                data = json.loads(text[first:last + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return data if isinstance(data, dict) else {}


def evaluate_student_answer(
    current_problem: str,
    latest_student_message: str,
    recent_chat: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback = {
        "is_answer_attempt": False,
        "is_correct": False,
        "confidence": "low",
        "feedback": "",
        "should_stop_hinting": False,
    }

    if not os.getenv("OPENAI_API_KEY"):
        return fallback

    payload = {
        "current_problem": current_problem,
        "latest_student_message": latest_student_message,
        "recent_chat": recent_chat[-8:] if isinstance(recent_chat, list) else [],
        "instruction": (
            "Check whether the latest student message is a final answer attempt. "
            "If it is correct, confirm it and explain briefly. "
            "If it is incorrect, give a short corrective scaffold."
        ),
    }

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        response = llm.invoke(
            [
                SystemMessage(content=ANSWER_EVALUATOR_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )

        data = _parse_json(str(response.content))

        return {
            "is_answer_attempt": bool(data.get("is_answer_attempt", False)),
            "is_correct": bool(data.get("is_correct", False)),
            "confidence": str(data.get("confidence", "low")),
            "feedback": str(data.get("feedback", "")).strip(),
            "should_stop_hinting": bool(data.get("should_stop_hinting", False)),
        }

    except Exception as exc:
        print("ANSWER EVALUATOR ERROR:", repr(exc))
        return fallback