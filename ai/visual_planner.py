import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


DEBUG_VISUAL = os.getenv("DEBUG_VISUAL", "0") == "1"


def _debug_visual(title: str, value: Any) -> None:
    if not DEBUG_VISUAL:
        return

    print("\n" + "=" * 20 + f" {title} " + "=" * 20)

    try:
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            print(value)
    except Exception:
        print(value)


_PROMPT = """\
You are the MathBridge Visual Drawing Agent.

Your job is to decide whether a visual would help the student, and if so,
generate one polished, student-friendly SVG diagram.

You are NOT limited to a fixed set of visual types.
Choose the most appropriate visual for the current tutoring step.

Before creating a visual, first decide what kind of mathematical idea the student is asking about.

Do not assume that a slash fraction such as 1/2 or 3/4 means a division-operation problem.

Use a division tape diagram only when the student explicitly asks about dividing, splitting, sharing, equal groups, how many groups, how many in each group, or how many fit.

If the student is explaining that fractions and decimals represent the same value, use a fraction-decimal equivalence visual, a number line, or a paired model instead of a division tape diagram.

If the visual type is uncertain, return show_visual=false rather than drawing a misleading visual.

Possible visuals include:
- coordinate plane
- tape / bar model
- equal groups
- array / area model
- rectangle or triangle geometry diagram
- fraction model
- number line
- simple labeled sketch
- unit-rate or ratio table

You must use the student message, tutor reply, curriculum context, and hint level.

Core teaching rules:
- Preserve the Socratic approach.
- Match the tutor's current hint or next step.
- If the student is still asking for hints or next steps, the visual may show structure,
  labels, and relationships, but should NOT directly compute or reveal the final answer.
- Do not reveal the final numeric answer unless the student has already attempted an answer
  or the tutor is clearly reviewing the answer.
- If a visual would not help, return show_visual=false.

SVG quality rules:
- Use simple, valid, self-contained SVG only.
- Do not use external images, scripts, animation, foreignObject, iframes, or links.
- Keep the SVG self-contained.
- Use width="720" height="360" viewBox="0 0 720 360".
- Leave at least 36px padding from every canvas edge.
- Keep ALL shapes, labels, and annotations fully inside the canvas.
- Never place labels so close to the border that they may be clipped.
- Use a light background or no background.
- Use dark lines and at most 2-3 accent colors.
- Make the diagram look polished, balanced, and classroom-friendly.
- Avoid overlapping labels, clipped words, crowded text, or text touching borders.
- Use readable font sizes and clear spacing.
- Keep text INSIDE the figure minimal.
- Put the main explanation in the caption, not as long sentences inside the SVG.
- The caption should be one short sentence shown below the figure.
- Prefer clarity over decoration.
- Avoid fake “input box” looking shapes unless absolutely necessary.
- Avoid truncated labels such as “3 equ...” or cropped words.
- If a label does not fit comfortably inside the shape, place it outside the shape but still inside the SVG canvas.
- Use short labels only, such as "base = 10 cm", "height = 4 cm", "x = -2", "y = 6", "each part = y".
For tape diagrams or bar models:
- Place labels on separate vertical levels.
- Do not place "Total", "Each part", or unknown-value labels on top of each other.
- Put the total label above or below the whole bar.
- Put the unknown-per-part label at least 28 pixels away from the total label.
- Prefer simple labels and fewer text elements.



SVG readability rules:
- Do not label every segment when there are more than 6 parts.
- For many equal groups, label only the total, the group size, and 2-3 representative groups.
- Avoid putting text directly under every partition line.
- Keep all text at least 18 px apart horizontally.
- If the number of groups is large, use an ellipsis or representative groups instead of drawing every label.
- Never let labels overlap.
Mathematical accuracy rules:
- Do not invent a specific number of groups unless it is given in the problem or derived in the tutor reply.
- If the quotient is unknown, label the group count with "?" or "unknown number of groups".
- For expressions like 356 ÷ 13, show "groups of 13" and "Total = 356", but do not label the diagram as "5 groups" unless the tutor is explicitly testing 5 groups.
- For large quotients, use representative groups plus "..." instead of pretending the drawing shows the exact number of groups.
- If showing an estimate, label it clearly, such as "try 10 groups", "try 20 groups", or "13 × ? ≈ 356".

Visual guidance by problem type:
1) Coordinate plane / ordered pairs
- Draw both x-axis and y-axis clearly.
- Label the axes.
- Plot the point clearly.
- Use dashed guide lines when helpful.
- Label only the needed coordinate value when the tutor is asking about x-coordinate or y-coordinate.
- Keep the diagram visually centered and easy to read.
- Avoid overlapping the point label, axis labels, and coordinate guide labels.


2) Equal groups / unknown factor / one-step equations such as 3y = 24
- Prefer a tape diagram or bar model.
- Draw one long bar split into the required number of equal parts.
- Show the total clearly above the full bar, e.g. "Total = 24".
- Label each equal part with the unknown, e.g. "y".
- Optionally show a short label such as "3 equal parts".
- Do NOT replace the unknown with the final answer during early hinting.
- Do NOT clutter the bar with long instructional text.
- Use the caption for the explanation.

3) Rectangle / area model
- Draw a clean rectangle with visually reasonable proportions.
- Place side labels near the correct edges.
- If the task is still a hint step, show dimensions and structure,
  but avoid writing the final numeric area if that would reveal the answer too early.
- If useful, lightly shade the shape.

4) Triangle area
- Draw a triangle clearly and completely within the canvas.
- Show the base and a perpendicular height.
- Label the base and height close to the relevant segments.
- Keep triangle vertices comfortably away from canvas edges.
- Keep the height label fully visible and horizontal unless vertical placement is clearly better.
- If you show a right-angle marker, make it small and readable.
- Avoid too much text inside the figure.
- If still in hinting mode, emphasize the structure (base, height, half of rectangle idea)
  without directly giving the final answer too early.
- For triangle diagrams, prefer a clean layout with the base near the bottom, the height clearly marked, and labels separated from the lines enough to remain readable.

5) Fraction / part-whole / number line
- Use clean partitions.
- Keep labels simple and aligned.
- Avoid visual clutter.

6) Circle / radius / diameter diagrams
Teaching principle:
- A radius is one segment from the center of the circle to the circle.
- A diameter is one straight segment across the full circle through the center.
- A diameter is made of two radii.
- If the radius is given and the diameter is unknown during a hint step, show the structure without computing the final answer too early.

Drawing layout:
- Draw one clean circle, centered on the canvas.
- Mark the center clearly with a small dot and label it "O".
- Draw exactly one straight diameter line through the center, from one side of the circle to the other.
- Visually split the diameter into two radius segments at the center point.
- Label one or both half-segments as "r = given value" when the radius is given.
- Put the diameter label outside the circle or below the full line, such as "d = ?" or "diameter = two radii".
- Do not put "r", "d", "7 cm", and "O" all on the same horizontal line.
- Do not overlap radius labels with the diameter line or the center label.
- Keep labels at least 24 px away from each other.
- Use one color for the diameter line and a second subtle color only if it helps distinguish the two radii.
- Do not draw duplicate diameter lines on top of each other.
- Do not reveal the final numeric diameter unless the student has already attempted an answer or the tutor is reviewing the answer.

Good hint-style layout:
- Circle centered.
- Horizontal diameter line through center.
- "r = 7 cm" label above the left or right half.
- "d = ?" label below the full diameter line.
- Caption explains: "The diameter is made of two radii."

7）For ratio and comparison problems:
- Prefer ONE clean tape diagram instead of multiple repeated diagrams.
- Show each quantity as equal-sized unit blocks so the ratio is visually countable.
- If the ratio is a:b, draw a equal blocks for the first quantity and b equal blocks for the second quantity.
- Use aligned rows with the same unit width for both quantities.
- Put the quantity labels on the left (for example, Flour and Milk).
- Put the measured amounts clearly near the bars (for example, 2 cups and 3 cups).
- Add a clear takeaway statement below the diagram, such as: "Ratio of flour to milk = 2 : 3".
- Do not include unfinished lines, empty blanks, or decorative marks that are not explained.
- Avoid duplicate diagrams that restate the same information.
- Keep the layout centered, balanced, and easy to read for a Grade 6 student.
- Use simple colors, clear outlines, and enough spacing so labels do not overlap.

Decision rules:
- If the tutor response is about coordinate points, geometry, area, equal groups,
  part-whole reasoning, fractions, or similar structure, a visual is often helpful.
- If the tutor response is purely conversational or a visual would not add value,
  return show_visual=false.

Return ONLY valid JSON in this exact shape:
{
  "show_visual": false,
  "svg": "",
  "caption": "short visual caption",
  "reason": "brief reason",
  "reveals_answer": false
}
"""
VISUAL_SELECTION_RULES = """
Choose the visual form based on the math situation.

1. For discrete countable objects (such as balls, apples, books, stickers, students, marbles, shapes, or items that are counted one by one):
- Prefer repeated object visuals, equal-size unit boxes, dot groups, or icon rows.
- Do NOT use long comparison bars as the primary visual.
- Show one unit per object when the counts are small enough to display clearly.
- If the numbers are small integers (for example 10 or less), draw each object or unit explicitly.
- Label each row clearly.
- Make the count visually obvious.
- End with a short statement such as "Red balls : Blue balls = 3 : 5".

2. For continuous quantities, part-whole relationships, or amount comparison problems:
- Prefer tape diagrams, bar models, or partitioned rectangles.
- Use equal-sized parts where appropriate.

3. Avoid duplicate visuals that restate the same information.
- Generate one clean, finished diagram unless a second diagram adds genuinely new understanding.

4. The visual should match Grade 6 pedagogy:
- clear
- centered
- uncluttered
- labeled
- easy to count
- no decorative arrows unless they help reasoning
- no unfinished lines or meaningless comparison arrows
"""
RATIO_VISUAL_RULES = """
For ratio problems:
- If the compared things are discrete objects, show two aligned rows of equal-size units.
- Example structure:
  Row 1: Red balls -> 3 equal units
  Row 2: Blue balls -> 5 equal units
- Use color to distinguish the two categories.
- Keep the two rows aligned so students can compare by counting.
- Add a short title above the diagram.
- Add a final statement below the diagram, such as:
  "Ratio of red balls to blue balls = 3 : 5"
- Do not draw both a top and bottom version of the same ratio.
- Do not use a generic "Compare" arrow unless necessary.
- Do not replace countable objects with oversized empty rectangles containing only numbers.
"""
VISUAL_STYLE_RULES = """
Visual quality rules:
- Prefer one polished diagram over multiple rough diagrams.
- Use balanced spacing and large readable labels.
- Keep text from overlapping with shapes.
- For discrete objects, prefer rows of repeated units rather than abstract bars.
- For small integer counts, make the count directly visible in the picture.
- Add one concise caption below the diagram explaining what the student should notice.
For geometry diagrams:
- Separate object labels, measurement labels, and unknown labels onto different vertical levels.
- Never place two measurement labels directly on top of the same line segment.
- If a line has multiple labels, put one above the line and one below the line.
- Keep center-point labels away from measurement labels.
"""


_REPAIR_PROMPT = """\
You are repairing an SVG math diagram for a Grade 6 tutoring platform.
When repairing SVGs, prioritize readability: remove crowded repeated labels, keep only essential labels, and ensure no text overlaps.

You will receive:
- the original student/tutor context
- the current SVG
- a list of quality issues

Your task:
- preserve the same mathematical intent
- preserve the same Socratic level (do not reveal the final answer early)
- return an improved SVG and caption

Repair requirements:
- Return valid self-contained SVG only
- Use width="720" height="360" viewBox="0 0 720 360"
- Keep all shapes, text, and labels fully inside the canvas
- Leave at least 36px margin from every edge
- Avoid cropped text, overlapping labels, and awkward spacing
- Keep text inside the SVG short
- Put explanation in the caption, not inside the figure
- Make the figure look neat, balanced, and classroom-friendly
- If the current SVG is a geometry figure, ensure the important lines and labels are complete and readable

Return ONLY valid JSON:
{
  "svg": "<svg ...>...</svg>",
  "caption": "short visual caption"
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


def _clean_svg(svg: str) -> str:
    """
    Security cleanup only.
    This does not decide the math content or visual type.
    """
    svg = str(svg or "").strip()

    if not svg:
        return ""

    first = svg.find("<svg")
    last = svg.rfind("</svg>")

    if first == -1 or last == -1:
        return ""

    svg = svg[first:last + len("</svg>")]

    blocked_patterns = [
        r"<script[\s\S]*?</script>",
        r"<foreignObject[\s\S]*?</foreignObject>",
        r"on\w+\s*=",
        r"javascript:",
        r"xlink:href\s*=",
        r"href\s*=",
        r"<iframe[\s\S]*?</iframe>",
    ]

    for pattern in blocked_patterns:
        svg = re.sub(
            pattern,
            "",
            svg,
            flags=re.IGNORECASE,
        )
  

    
    # Keep the UI layout stable even when the model forgets explicit size.
    if "viewBox" not in svg[:160]:
        svg = re.sub(
            r"<svg\b",
            '<svg viewBox="0 0 720 360"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )

    if "width" not in svg[:160]:
        svg = re.sub(
            r"<svg\b",
            '<svg width="720"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )

    if "height" not in svg[:180]:
        svg = re.sub(
            r"<svg\b",
            '<svg height="360"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )

    if "xmlns" not in svg[:220]:
        svg = re.sub(
            r"<svg\b",
            '<svg xmlns="http://www.w3.org/2000/svg"',
            svg,
            count=1,
            flags=re.IGNORECASE,
        )

    return svg.strip()

def _extract_svg_text_nodes(svg: str) -> list[dict]:
    nodes = []

    pattern = re.compile(
        r"<text\b([^>]*)>(.*?)</text>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for match in pattern.finditer(str(svg or "")):
        attrs = match.group(1)
        text = re.sub(r"<[^>]+>", "", match.group(2))
        text = " ".join(text.split()).strip()

        x_match = re.search(r'\bx=["\']?(-?\d+(?:\.\d+)?)', attrs)
        y_match = re.search(r'\by=["\']?(-?\d+(?:\.\d+)?)', attrs)

        if not text or not x_match or not y_match:
            continue

        nodes.append(
            {
                "text": text.lower(),
                "x": float(x_match.group(1)),
                "y": float(y_match.group(1)),
            }
        )

    return nodes



def _svg_quality_issues(svg: str) -> list[str]:
    """
    Return only serious SVG quality/safety issues.

    Keep this check lightweight. Do not trigger a second LLM repair pass
    just because the SVG has many labels, long text, or slightly different
    dimensions. Those are usually acceptable in Streamlit.
    """
    issues: list[str] = []
    s = str(svg or "").strip()

    if not s:
        return ["empty svg"]

    lowered = s.lower()

    # Basic structure checks.
    if "<svg" not in lowered or "</svg>" not in lowered:
        issues.append("invalid svg wrapper")

    if "viewbox=" not in lowered:
        issues.append("missing viewBox")

    # Safety checks. _clean_svg should already remove these, so if any remain,
    # repair is justified.
    unsafe_fragments = (
        "<script",
        "foreignobject",
        "<iframe",
        "javascript:",
        "onload=",
        "onclick=",
        "onmouseover=",
    )

    for frag in unsafe_fragments:
        if frag in lowered:
            issues.append("unsafe svg content remains")
            break

    # Known broken text fragments from bad SVG generations.
    # Do NOT include "..." here because ellipsis can be valid in tape diagrams.
    suspicious_fragments = (
        "3 equ",
        "eight =",
    )
    

    for frag in suspicious_fragments:
        if frag in lowered:
            issues.append(f"suspicious svg fragment: {frag}")
            break

    return issues

def _repair_svg_with_llm(
    llm: ChatOpenAI,
    user_payload: dict[str, Any],
    current_svg: str,
    current_caption: str,
    issues: list[str],
) -> tuple[str, str]:
    repair_input = {
        "context": user_payload,
        "current_svg": current_svg,
        "current_caption": current_caption,
        "issues": issues,
    }

    response = llm.invoke(
        [
            SystemMessage(content=_REPAIR_PROMPT),
            HumanMessage(
                content=json.dumps(
                    repair_input,
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )

    repaired = _parse_json(str(response.content))
    repaired_svg = _clean_svg(str(repaired.get("svg", "")))
    repaired_caption = str(repaired.get("caption", "")).strip()

    return repaired_svg, repaired_caption

def _safe_no_visual(reason: str) -> dict[str, Any]:
    return {
        "needs_visual": False,
        "visual_type": "none",
        "visual_data": {},
        "reason": reason,
        "reveals_answer": False,
    }


def _normalize_visual_payload(
    payload: dict[str, Any],
    hint_level: int,
) -> dict[str, Any]:
    show_visual = bool(
        payload.get("show_visual", payload.get("should_generate", False))
    )
    reveals_answer = bool(payload.get("reveals_answer", False))
    reason = str(payload.get("reason", "")).strip()
    caption = str(payload.get("caption", "")).strip()

    svg = _clean_svg(
        str(payload.get("svg", ""))
    )

    if not show_visual:
        return _safe_no_visual(
            reason or "Visual agent decided no visual was needed."
        )

    if not svg:
        return _safe_no_visual(
            "Visual agent did not return a valid SVG."
        )

    if hint_level < 3 and reveals_answer:
        return _safe_no_visual(
            "Answer-revealing visual was suppressed during early Socratic scaffolding."
        )

    return {
        "needs_visual": True,
        "visual_type": "llm_svg",
        "visual_data": {
            "svg": svg,
            "caption": caption or "AI-generated visual explanation",
        },
        "reason": reason,
        "reveals_answer": reveals_answer,
    }

def _parse_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _local_whole_number_division_plan(
    user_text: str,
    tutor_reply: str,
) -> dict | None:
    text = f"{user_text}\n{tutor_reply}"
    lower_text = text.lower()

   
    fraction_markers = (
        "unit fraction",
        "non-unit fraction",
        "non unit fraction",
        "reciprocal",
        "fraction division",
        "divide fractions",
        "dividing fractions",
    )

    if any(marker in lower_text for marker in fraction_markers):
        return None

    patterns = [
        r"\b(-?\d+(?:\.\d+)?)\s*÷\s*(-?\d+(?:\.\d+)?)\b",
        r"\b(-?\d+(?:\.\d+)?)\s+divided\s+by\s+(-?\d+(?:\.\d+)?)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        total = _parse_number(match.group(1))
        group_size = _parse_number(match.group(2))

        if total is None or group_size is None:
            continue

        if total <= 0 or group_size <= 0:
            continue

        return {
            "needs_visual": True,
            "visual_type": "whole_number_division_tape",
            "visual_data": {
                "total": total,
                "group_size": group_size,
                "group_count": None,
                "mode": "measurement_division",
            },
            "reason": (
                "Local deterministic whole-number division tape diagram."
            ),
            "reveals_answer": False,
        }

    return None

def plan_visual(
    user_text: str,
    tutor_reply: str,
    curriculum_context: str = "",
    hint_level: int = 0,
) -> dict[str, Any]:
    local_plan = _local_whole_number_division_plan(
        user_text=user_text,
        tutor_reply=tutor_reply,
    )

    if local_plan is not None:
        return local_plan
    """
    LLM-generated visual planning.

    The model decides whether to draw and directly generates SVG.
    Deterministic code only cleans unsafe SVG and prevents early answer leakage.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return _safe_no_visual(
            "Visual drawing agent is unavailable because OPENAI_API_KEY is not set."
        )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=7,
        max_retries=0,
    )

    user_payload = {
        "student_text": user_text,
        "tutor_reply": tutor_reply,
        "curriculum_context": curriculum_context[:3000],
        "hint_level": hint_level,
        "instruction": (
            "Generate a visual only if it helps the student reason. "
            "Do not use a fixed visual type. Choose the best diagram yourself and draw the SVG directly. "
            "Keep the figure clean and polished. Keep all shapes and labels fully inside the canvas. "
            "Use short labels in the SVG and put longer explanation in the caption."
        ),
    }

    _debug_visual("VISUAL AGENT INPUT", user_payload)

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

        _debug_visual("VISUAL AGENT RAW OUTPUT", response.content)

        payload = _parse_json(
            str(response.content)
        )
        _debug_visual("VISUAL AGENT PARSED PAYLOAD", payload)

        normalized = _normalize_visual_payload(
            payload,
            hint_level,
        )

        if normalized.get("needs_visual"):
            svg = normalized.get("visual_data", {}).get("svg", "")
            caption = normalized.get("visual_data", {}).get("caption", "")
            issues = _svg_quality_issues(svg)

            _debug_visual("VISUAL SVG QUALITY ISSUES", issues)

            if issues:
                repaired_svg, repaired_caption = _repair_svg_with_llm(
                    llm=llm,
                    user_payload=user_payload,
                    current_svg=svg,
                    current_caption=caption,
                    issues=issues,
                )

                if repaired_svg:
                    normalized["visual_data"]["svg"] = repaired_svg
                    if repaired_caption:
                        normalized["visual_data"]["caption"] = repaired_caption

        _debug_visual("VISUAL AGENT NORMALIZED PAYLOAD", normalized)

        return normalized

    except Exception as exc:
        _debug_visual("VISUAL AGENT ERROR", repr(exc))

        return _safe_no_visual(
            f"Visual drawing agent failed safely: {exc}"
        )
