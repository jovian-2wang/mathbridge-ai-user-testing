import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


_PROMPT = """\
You are the MathBridge Curriculum Grounding Agent.

Your job is to decide which retrieved curriculum references should be shown
as grounding evidence for the current tutoring response.

You are given:
- the current student problem
- the detected learning concept
- RAG retrieved curriculum candidates

Important selection rules:
- Do not blindly trust retrieval order.
- Select references that are pedagogically aligned with the current problem.
- If the match is not exact but the candidate supports the same underlying
  concept, you may select it with confidence="low" or confidence="moderate".
- If candidates are weak or partial, use a soft message such as
  "Closest related curriculum connections are shown below." Do not say that
  nothing was found unless there are truly no candidates.
- Do not display raw long excerpts.
- Generate a concise focus statement for each selected reference.
- Keep the panel language supportive and teacher/student friendly, not like
  a system error.
- Distinguish division meanings carefully:
  1. Equal sharing: total amount and number of groups are known; unknown is
     amount in each group.
  2. Measurement division: total amount and group size are known; unknown is
     number of groups.
- Avoid fraction-only references for whole-number division unless they support
  the same division meaning or general reasoning.
- Avoid geometry references unless the current problem is geometry, area,
  coordinate plane, perimeter, volume, or spatial reasoning.
- Return ONLY valid JSON.

Do NOT use phrases like:
- "No retrieved references directly address..."
- "No references found..."
- "Nothing matches..."
- "The search failed..."

Use one of these overall_status values:
- strong_alignment: selected references closely match the current problem.
- weak_alignment: selected references are related but not exact.
- no_alignment: curriculum context is limited for this exact prompt.

Return this exact JSON shape:
{
  "problem_type": "measurement_division | equal_sharing | unit_rate | fraction_division | ratio | geometry | coordinate_plane | area | other",
  "overall_status": "strong_alignment | weak_alignment | no_alignment",
  "message": "short supportive message for the grounding panel",
  "selected_references": [
    {
      "source": "source filename or title",
      "confidence": "high | moderate | low",
      "focus": "student-friendly explanation of what this source supports",
      "why_this_matches": "brief explanation tied to the current problem",
      "score": 0.0
    }
  ]
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


def _compact_matches(matches: list[dict], limit: int = 5) -> list[dict]:
    compact = []

    for match in matches[:limit]:
        compact.append(
            {
                "source": match.get("source", ""),
                "score": match.get("similarity", match.get("score", None)),
                "retrieval_method": match.get("retrieval_method", ""),
                "excerpt": str(match.get("excerpt", ""))[:700],
            }
        )

    return compact


def _score_to_confidence(score: Any) -> str:
    try:
        value = float(score)
    except Exception:
        return "low"

    if value >= 0.55:
        return "high"
    if value >= 0.35:
        return "moderate"
    return "low"


def _soft_message(overall_status: str, has_selected: bool) -> str:
    if overall_status == "strong_alignment" and has_selected:
        return (
            "Selected curriculum references closely support the current math idea."
        )

    if overall_status == "weak_alignment" and has_selected:
        return (
            "Closest related curriculum connections are shown below. "
            "They support the same general concept, even if they do not match the problem exactly."
        )

    return (
        "Curriculum context is limited for this exact prompt. "
        "The tutor response is based mainly on the student's current work and the nearest available concept support."
    )


def _fallback_grounding(retrieval_method: str = "unknown") -> dict[str, Any]:
    return {
        "problem_type": "other",
        "overall_status": "no_alignment",
        "message": _soft_message("no_alignment", False),
        "selected_references": [],
        "retrieval_method": retrieval_method,
        "generated_by": "fallback",
    }


def _normalize_message(message: str, overall_status: str, has_selected: bool) -> str:
    text = str(message or "").strip()
    lowered = text.lower()

    blocked = (
        "no retrieved references directly address",
        "no references found",
        "nothing matches",
        "search failed",
        "no strong curriculum alignment was found",
    )

    if not text or any(phrase in lowered for phrase in blocked):
        return _soft_message(overall_status, has_selected)

    return text


def _fallback_selected_from_matches(matches: list[dict], limit: int = 1) -> list[dict]:
    """
    If the LLM rejects everything but retrieval did return candidates, show at most
    one closest available reference with low confidence. This avoids a harsh
    "nothing found" panel while still being honest about limited alignment.
    """
    clean = []

    for match in matches[:limit]:
        source = str(match.get("source", "")).strip()
        if not source:
            continue

        score = match.get("similarity", match.get("score", None))
        clean.append(
            {
                "source": source,
                "confidence": _score_to_confidence(score),
                "focus": (
                    "Closest available curriculum context for the current math reasoning."
                ),
                "why_this_matches": (
                    "This reference is shown as nearby support, not as an exact match."
                ),
                "score": score,
            }
        )

    return clean


def generate_grounding_panel(
    current_problem: str,
    detected_concept: str,
    retrieval_evidence: dict,
) -> dict[str, Any]:
    matches = retrieval_evidence.get("matches", [])
    retrieval_method = (
        retrieval_evidence.get("retrieval_method")
        or retrieval_evidence.get("method")
        or "unknown"
    )

    if not matches:
        return _fallback_grounding(retrieval_method)

    if not os.getenv("OPENAI_API_KEY"):
        fallback_selected = _fallback_selected_from_matches(matches, limit=1)
        return {
            "problem_type": "other",
            "overall_status": "weak_alignment" if fallback_selected else "no_alignment",
            "message": _soft_message(
                "weak_alignment" if fallback_selected else "no_alignment",
                bool(fallback_selected),
            ),
            "selected_references": fallback_selected,
            "retrieval_method": retrieval_method,
            "generated_by": "fallback",
        }

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    payload = {
        "current_problem": current_problem,
        "detected_concept": detected_concept,
        "retrieval_method": retrieval_method,
        "retrieved_candidates": _compact_matches(matches),
    }

    try:
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

        parsed = _safe_json_loads(str(response.content))

        selected = parsed.get("selected_references", [])
        if not isinstance(selected, list):
            selected = []

        clean_selected = []
        score_by_source = {
            str(match.get("source", "")).strip(): match.get(
                "similarity", match.get("score", None)
            )
            for match in matches
        }

        for item in selected[:2]:
            if not isinstance(item, dict):
                continue

            source = str(item.get("source", "")).strip()
            focus = str(item.get("focus", "")).strip()
            why = str(item.get("why_this_matches", "")).strip()
            confidence = str(item.get("confidence", "moderate")).strip().lower()
            score = item.get("score", None)

            if score is None and source in score_by_source:
                score = score_by_source[source]

            if confidence not in {"high", "moderate", "low"}:
                confidence = _score_to_confidence(score)

            if source and focus:
                clean_selected.append(
                    {
                        "source": source,
                        "confidence": confidence,
                        "focus": focus,
                        "why_this_matches": why,
                        "score": score,
                    }
                )

        overall_status = str(
            parsed.get("overall_status") or "weak_alignment"
        ).strip()

        if overall_status not in {
            "strong_alignment",
            "weak_alignment",
            "no_alignment",
        }:
            overall_status = "weak_alignment"

        if not clean_selected:
            clean_selected = _fallback_selected_from_matches(matches, limit=1)
            overall_status = "weak_alignment" if clean_selected else "no_alignment"

        if overall_status == "strong_alignment":
            # Do not claim strong alignment if all selected references are low confidence.
            if not any(item.get("confidence") in {"high", "moderate"} for item in clean_selected):
                overall_status = "weak_alignment"

        message = _normalize_message(
            str(parsed.get("message") or "").strip(),
            overall_status,
            bool(clean_selected),
        )

        return {
            "problem_type": str(parsed.get("problem_type") or "other"),
            "overall_status": overall_status,
            "message": message,
            "selected_references": clean_selected,
            "retrieval_method": retrieval_method,
            "generated_by": "Curriculum Grounding Agent",
        }

    except Exception:
        fallback_selected = _fallback_selected_from_matches(matches, limit=1)
        return {
            "problem_type": "other",
            "overall_status": "weak_alignment" if fallback_selected else "no_alignment",
            "message": _soft_message(
                "weak_alignment" if fallback_selected else "no_alignment",
                bool(fallback_selected),
            ),
            "selected_references": fallback_selected,
            "retrieval_method": retrieval_method,
            "generated_by": "fallback",
        }


def build_grounding_panel_copy(
    selected_refs,
    retrieval_method="embedding-first hybrid",
):
    """
    Optional helper for UI copy. Kept for compatibility.
    """
    refs = selected_refs or []

    def _get_score(ref):
        for key in ("score", "similarity"):
            value = ref.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    top_score = max((_get_score(r) for r in refs), default=0.0)

    if top_score >= 0.55:
        status = "strong"
        summary = "Selected references closely support the current math idea."
        helper = (
            "These lessons align well with the tutor response and support the same math idea."
        )
    elif top_score >= 0.35:
        status = "related"
        summary = "Closest related curriculum connections are shown below."
        helper = (
            "These references support the same general concept, even if they do not match the problem exactly."
        )
    else:
        status = "limited"
        summary = "Curriculum context is limited for this exact prompt."
        helper = (
            "The references below are the nearest available lesson connections. "
            "The tutor response is based mainly on the student's current work and overall concept understanding."
        )

    return {
        "status": status,
        "retrieval_method": retrieval_method,
        "summary": summary,
        "helper": helper,
        "references": refs,
    }
