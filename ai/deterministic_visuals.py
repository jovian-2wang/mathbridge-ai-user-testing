import html
import os
import re
from typing import Any

SVG_WIDTH = 720
SVG_HEIGHT = 360


def plan_deterministic_visual(
    problem_text: str = "",
    concept: str = "",
    hint_level: int = 0,
    contextualization_context: str = "",
) -> dict[str, Any]:
    """
    Reliable visual templates for high-confidence Grade 6 problem types.

    These templates intentionally avoid computing the final answer. They show
    the structure of the problem so the tutor can remain Socratic while the
    demo stays visually stable and low-latency.
    """
    if os.getenv("MATHBRIDGE_USE_DETERMINISTIC_VISUALS", "1") != "1":
        return _none("Deterministic visuals disabled by environment.")

    problem = str(problem_text or "").strip()
    concept_text = str(concept or "").strip().lower()
    combined = f"{problem} {concept_text}".lower()

    if not problem:
        return _none("No problem text available for deterministic visual.")

    if _looks_like_fraction_addition(combined):
        visual = _fraction_addition_visual(problem)
        if visual:
            return visual

    if _looks_like_fraction_division(combined):
        visual = _fraction_division_visual(problem)
        if visual:
            return visual

    if _looks_like_unit_rate(combined):
        visual = _unit_rate_visual(problem)
        if visual:
            return visual

    if _looks_like_ratio(combined):
        visual = _ratio_visual(problem)
        if visual:
            return visual

    if _looks_like_coordinate_problem(combined):
        visual = _coordinate_visual(problem)
        if visual:
            return visual

    if _looks_like_division_reasoning(combined):
        visual = _division_reasoning_visual(problem)
        if visual:
            return visual

    return _none("No high-confidence deterministic template matched.")


def _none(reason: str) -> dict[str, Any]:
    return {
        "needs_visual": False,
        "visual_type": "none",
        "visual_data": {},
        "reason": reason,
        "reveals_answer": False,
        "generated_by": "Deterministic Visual Templates",
    }


def _visual(svg: str, caption: str, reason: str, template: str) -> dict[str, Any]:
    return {
        "needs_visual": True,
        "visual_type": "llm_svg",
        "visual_data": {
            "svg": svg,
            "caption": caption,
        },
        "reason": reason,
        "reveals_answer": False,
        "generated_by": "Deterministic Visual Template",
        "template": template,
    }


def _escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _num_text(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+\s*/\s*\d+|\d+(?:\.\d+)?", str(text or ""))


def _first_noun_after_number(text: str, number: str, fallback: str = "items") -> str:
    pattern = re.escape(str(number).strip()) + r"\s+([A-Za-z][A-Za-z-]{1,24})"
    match = re.search(pattern, text)
    if not match:
        return fallback
    word = match.group(1).strip().lower()
    if word in {"and", "or", "of", "per", "for", "cost", "costs", "total", "divided"}:
        return fallback
    return word



def _clean_label(label: str, fallback: str = "Quantity") -> str:
    text = str(label or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.split(
        r"\b(?:is|are|was|were|equals?|compares?|compare|with|for every|to find|what|how)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:?")
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text).strip(" ,.;:?")

    if not text:
        text = fallback

    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4])

    if len(text) > 30:
        text = text[:27].rstrip() + "..."

    return text[:1].upper() + text[1:]


def _extract_ratio_labels(text: str) -> tuple[str, str]:
    raw = str(text or "")

    patterns = [
        r"ratio\s+of\s+(?P<a>.+?)\s+to\s+(?P<b>.+?)\s+(?:is|=|equals?|compares?|compare)\b",
        r"(?P<a_count>\d+)\s+(?P<a>[A-Za-z][A-Za-z\s-]{1,40}?)\s+(?:to|and)\s+(?P<b_count>\d+)\s+(?P<b>[A-Za-z][A-Za-z\s-]{1,40})",
        r"(?P<a_count>\d+)\s+(?P<a>[A-Za-z][A-Za-z\s-]{1,40}?)\s+for\s+every\s+(?P<b_count>\d+)\s+(?P<b>[A-Za-z][A-Za-z\s-]{1,40})",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            first = _clean_label(match.group("a"), "Part A")
            second = _clean_label(match.group("b"), "Part B")
            if first.lower() != second.lower():
                return first, second

    lowered = raw.lower()
    if "students" in lowered and "walk" in lowered:
        return "Students who walk", "Total students"
    if "red ball" in lowered and "blue ball" in lowered:
        return "Red balls", "Blue balls"
    if "boys" in lowered and "girls" in lowered:
        return "Boys", "Girls"
    if "cats" in lowered and "dogs" in lowered:
        return "Cats", "Dogs"
    if "wins" in lowered and "losses" in lowered:
        return "Wins", "Losses"

    return "First quantity", "Second quantity"


def _extract_problem_item_label(text: str, fallback: str = "items") -> str:
    lowered = str(text or "").lower()
    candidates = [
        "students", "notebooks", "books", "tickets", "stickers",
        "apples", "oranges", "cookies", "balls", "pencils",
    ]
    for candidate in candidates:
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            return candidate
    return fallback

def _extract_unit_rate_parts(text: str) -> dict[str, Any] | None:
    clean = str(text or "")
    lowered = clean.lower()

    # Cost-per-item: "5 notebooks cost $15" or "$15 for 5 notebooks".
    cost_patterns = [
        r"(?P<count>\d+(?:\.\d+)?)\s+(?P<item>[A-Za-z][A-Za-z-]{1,24})[^\n?.]{0,90}?\b(?:cost|costs|are|is|total(?:\s+cost)?(?:\s+is)?)\s*\$?(?P<total>\d+(?:\.\d+)?)",
        r"\$?(?P<total>\d+(?:\.\d+)?)\s+(?:for|on)\s+(?P<count>\d+(?:\.\d+)?)\s+(?P<item>[A-Za-z][A-Za-z-]{1,24})",
    ]

    for pattern in cost_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            count = float(match.group("count"))
            total = _num_text(match.group("total"))
            item = (match.groupdict().get("item") or "items").lower()
            if count > 1:
                return {
                    "kind": "cost",
                    "count": int(count) if count.is_integer() else count,
                    "total": total,
                    "item": item,
                    "total_label": f"Total cost = ${total}",
                    "parts_label": f"Items = {_num_text(str(count))} {item}",
                    "box_unknown": "$ ?",
                    "part_label": f"1 {_singular(item)}",
                    "question_label": f"Cost per {_singular(item)} = ?",
                    "caption_unit": f"{item}",
                }

    # Distance/time or rate-per-time: "240 miles in 4 hours",
    # "96 points in 48 minutes", "120 kilometers for 2 hours".
    rate_patterns = [
        r"(?P<total>\d+(?:\.\d+)?)\s+(?P<total_unit>miles?|kilometers?|km|meters?|yards?|laps?|points?|pages?|words?)\b[^\n?.]{0,90}?\b(?:in|for|over|during)\s+(?P<count>\d+(?:\.\d+)?)\s+(?P<part_unit>hours?|hrs?|minutes?|mins?|seconds?|days?|games?|quarters?|weeks?)\b",
        r"(?:travels?|drives?|walks?|runs?|scores?|earns?|reads?)\s+(?P<total>\d+(?:\.\d+)?)\s+(?P<total_unit>miles?|kilometers?|km|meters?|yards?|laps?|points?|pages?|words?)\b[^\n?.]{0,90}?\b(?:in|for|over|during)\s+(?P<count>\d+(?:\.\d+)?)\s+(?P<part_unit>hours?|hrs?|minutes?|mins?|seconds?|days?|games?|quarters?|weeks?)\b",
    ]

    for pattern in rate_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            count = float(match.group("count"))
            total = _num_text(match.group("total"))
            total_unit = _normalize_unit(match.group("total_unit"))
            part_unit = _normalize_unit(match.group("part_unit"))
            if count > 1:
                return {
                    "kind": "per_time",
                    "count": int(count) if count.is_integer() else count,
                    "total": total,
                    "item": part_unit,
                    "total_label": f"Total {total_unit} = {total} {total_unit}",
                    "parts_label": f"Total time/parts = {_num_text(str(count))} {part_unit}",
                    "box_unknown": f"? {total_unit}",
                    "part_label": f"1 {_singular(part_unit)}",
                    "question_label": f"{_capitalize_unit(total_unit)} per {_singular(part_unit)} = ?",
                    "caption_unit": total_unit,
                }


    # Duration first, outcome later: "game lasts 48 minutes and the team scores 96 points".
    reverse_rate_patterns = [
        r"(?P<count>\d+(?:\.\d+)?)\s+(?P<part_unit>hours?|hrs?|minutes?|mins?|seconds?|days?|games?|quarters?|weeks?)\b[^\n?.]{0,110}?\b(?:scores?|earns?|gets?|reads?|travels?|drives?|walks?|runs?)\s+(?P<total>\d+(?:\.\d+)?)\s+(?P<total_unit>miles?|kilometers?|km|meters?|yards?|laps?|points?|pages?|words?)\b",
    ]

    for pattern in reverse_rate_patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            count = float(match.group("count"))
            total = _num_text(match.group("total"))
            total_unit = _normalize_unit(match.group("total_unit"))
            part_unit = _normalize_unit(match.group("part_unit"))
            if count > 1:
                return {
                    "kind": "per_time",
                    "count": int(count) if count.is_integer() else count,
                    "total": total,
                    "item": part_unit,
                    "total_label": f"Total {total_unit} = {total} {total_unit}",
                    "parts_label": f"Total time/parts = {_num_text(str(count))} {part_unit}",
                    "box_unknown": f"? {total_unit}",
                    "part_label": f"1 {_singular(part_unit)}",
                    "question_label": f"{_capitalize_unit(total_unit)} per {_singular(part_unit)} = ?",
                    "caption_unit": total_unit,
                }

    # Conservative fallback for prompts that explicitly ask for unit rate but
    # use simple "A in B" phrasing not covered above.
    nums = _numbers(clean)
    if "unit rate" in lowered and len(nums) >= 2 and all("/" not in n for n in nums[:2]):
        total = _num_text(nums[0])
        count_value = float(nums[1])
        if count_value > 1:
            noun_total = _first_noun_after_number(clean, nums[0], "units")
            noun_part = _first_noun_after_number(clean, nums[1], "parts")
            return {
                "kind": "generic",
                "count": int(count_value) if count_value.is_integer() else count_value,
                "total": total,
                "item": noun_part,
                "total_label": f"Total = {total} {noun_total}",
                "parts_label": f"Equal parts = {_num_text(str(count_value))} {noun_part}",
                "box_unknown": "?",
                "part_label": f"1 {_singular(noun_part)}",
                "question_label": f"One-part value = ?",
                "caption_unit": noun_total,
            }

    return None


def _normalize_unit(unit: str) -> str:
    text = str(unit or "").strip().lower()
    aliases = {
        "hrs": "hours",
        "hr": "hour",
        "mins": "minutes",
        "min": "minute",
        "km": "kilometers",
    }
    return aliases.get(text, text)


def _capitalize_unit(unit: str) -> str:
    text = str(unit or "units").strip()
    return text[:1].upper() + text[1:]



def _looks_like_unit_rate(text: str) -> bool:
    return bool(
        "unit rate" in text
        or "per item" in text
        or "per one" in text
        or "per hour" in text
        or "per minute" in text
        or "miles per" in text
        or "points per" in text
        or "cost per" in text
        or "price per" in text
        or "for each" in text
        or re.search(r"\b\d+\s+[a-z]+s?\s+cost", text)
        or re.search(r"\$\d+(?:\.\d+)?\s+for\s+\d+", text)
        or re.search(r"\b\d+(?:\.\d+)?\s+(?:miles?|kilometers?|km|meters?|points?|pages?)\b[^.?!]{0,90}\b(?:in|for|over)\s+\d+(?:\.\d+)?\s+(?:hours?|minutes?|games?|days?)\b", text)
        or re.search(r"\b\d+(?:\.\d+)?\s+(?:hours?|minutes?|games?|days?)\b[^.?!]{0,110}\b(?:scores?|earns?|gets?|reads?|travels?|drives?|walks?|runs?)\s+\d+(?:\.\d+)?\s+(?:miles?|kilometers?|km|meters?|points?|pages?)\b", text)
    )


def _unit_rate_visual(problem: str) -> dict[str, Any] | None:
    parts = _extract_unit_rate_parts(problem)
    if not parts:
        return None

    count = parts["count"]
    total = parts["total"]
    item = str(parts.get("item") or "parts")

    try:
        count_float = float(count)
    except (TypeError, ValueError):
        return None

    if count_float <= 1:
        return None

    # Only draw individual boxes when the part count is reasonable. For large
    # denominators like 48 minutes, show a compact sample of equal parts.
    display_count = int(min(max(round(count_float), 2), 6))
    count_text = _num_text(str(count))

    diagram_w = 560
    gap = 10
    box_w = min(96, max(68, (diagram_w - (display_count - 1) * gap) / display_count))
    box_h = 66
    total_w = display_count * box_w + (display_count - 1) * gap
    start_x = (SVG_WIDTH - total_w) / 2
    y = 154

    boxes = []
    for index in range(display_count):
        x = start_x + index * (box_w + gap)
        boxes.append(
            f'<rect x="{x:.1f}" y="{y}" width="{box_w:.1f}" height="{box_h}" rx="10" fill="#e6fffa" stroke="#2c7a7b" stroke-width="2" />'
        )
        boxes.append(
            f'<text x="{x + box_w / 2:.1f}" y="{y + 28}" text-anchor="middle" font-size="17" font-weight="700" fill="#234e52">{_escape(parts["box_unknown"])}</text>'
        )
        boxes.append(
            f'<text x="{x + box_w / 2:.1f}" y="{y + 52}" text-anchor="middle" font-size="13" fill="#2c7a7b">{_escape(parts["part_label"])}</text>'
        )

    if count_float > display_count:
        boxes.append(
            f'<text x="{start_x + total_w + 25:.1f}" y="{y + 42}" text-anchor="middle" font-size="24" font-weight="700" fill="#2c7a7b">...</text>'
        )
        count_note = f"showing {display_count} of {count_text} equal parts"
    else:
        count_note = f"{count_text} equal parts"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Unit rate tape diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="40" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Unit Rate Structure</text>
  <text x="360" y="72" text-anchor="middle" font-size="16" fill="#4a5568">Split the total into equal parts, then reason about one part.</text>

  <rect x="96" y="94" width="250" height="38" rx="10" fill="#f7fafc" stroke="#cbd5e0" />
  <rect x="374" y="94" width="250" height="38" rx="10" fill="#f7fafc" stroke="#cbd5e0" />
  <text x="221" y="119" text-anchor="middle" font-size="15" font-weight="700" fill="#2d3748">{_escape(parts["total_label"])}</text>
  <text x="499" y="119" text-anchor="middle" font-size="15" font-weight="700" fill="#2d3748">{_escape(parts["parts_label"])}</text>

  {''.join(boxes)}

  <line x1="{start_x:.1f}" y1="{y + box_h + 26}" x2="{start_x + total_w:.1f}" y2="{y + box_h + 26}" stroke="#2c7a7b" stroke-width="3" />
  <text x="{start_x + total_w / 2:.1f}" y="{y + box_h + 58}" text-anchor="middle" font-size="21" font-weight="700" fill="#dd6b20">{_escape(parts["question_label"])}</text>
  <text x="360" y="332" text-anchor="middle" font-size="13" fill="#718096">Template: {_escape(count_note)}. The visual shows structure without calculating the final answer.</text>
</svg>'''

    return _visual(
        svg=svg,
        caption=(
            f"A unit-rate tape diagram showing {count_text} equal parts and a total of {total}, prompting the student to reason about one part."
        ),
        reason="Deterministic unit-rate visual selected for a rate-per-one problem with non-overlapping labels.",
        template="unit_rate_tape_polished",
    )



def _singular(word: str) -> str:
    word = str(word or "item").strip().lower()
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word or "item"


def _looks_like_ratio(text: str) -> bool:
    return bool(
        "ratio" in text
        or re.search(r"\b\d+\s*:\s*\d+\b", text)
        or re.search(r"\bfor every\b", text)
    )


def _extract_ratio(text: str) -> tuple[int, int] | None:
    match = re.search(r"\b(\d+)\s*:\s*(\d+)\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    match = re.search(r"\b(\d+)\s+[A-Za-z]+s?\s+for every\s+(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    nums = [n for n in _numbers(text) if "/" not in n]
    if len(nums) >= 2:
        a, b = int(float(nums[0])), int(float(nums[1]))
        if 0 < a <= 20 and 0 < b <= 20:
            return a, b

    return None


def _ratio_visual(problem: str) -> dict[str, Any] | None:
    ratio = _extract_ratio(problem)
    if not ratio:
        return None
    a, b = ratio
    label_a, label_b = _extract_ratio_labels(problem)

    max_dots = 10
    a_dots = min(a, max_dots)
    b_dots = min(b, max_dots)

    def dots(row_y: int, count: int, fill: str) -> str:
        parts = []
        for index in range(count):
            x = 300 + index * 28
            parts.append(f'<circle cx="{x}" cy="{row_y}" r="10" fill="{fill}" stroke="#2d3748" stroke-width="1" />')
        if count >= max_dots:
            parts.append(f'<text x="{300 + count * 28 + 8}" y="{row_y + 6}" font-size="20" fill="#2d3748">...</text>')
        return "".join(parts)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Ratio comparison diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Ratio Structure</text>
  <text x="360" y="78" text-anchor="middle" font-size="16" fill="#4a5568">Compare two quantities without changing their relationship.</text>
  <rect x="70" y="112" width="580" height="78" rx="12" fill="#f7fafc" stroke="#cbd5e0" />
  <rect x="70" y="204" width="580" height="78" rx="12" fill="#f7fafc" stroke="#cbd5e0" />
  <text x="175" y="158" text-anchor="middle" font-size="15" font-weight="700" fill="#2d3748">{_escape(label_a)}</text>
  <text x="175" y="250" text-anchor="middle" font-size="15" font-weight="700" fill="#2d3748">{_escape(label_b)}</text>
  {dots(150, a_dots, '#90cdf4')}
  {dots(242, b_dots, '#fbb6ce')}
  <text x="360" y="315" text-anchor="middle" font-size="20" font-weight="700" fill="#2d3748">Ratio = {_escape(a)} : {_escape(b)}</text>
  <text x="360" y="340" text-anchor="middle" font-size="13" fill="#718096">Use equivalent groups or a ratio table to reason before calculating.</text>
</svg>'''

    return _visual(
        svg=svg,
        caption=f"A ratio diagram comparing {label_a.lower()} and {label_b.lower()} as {a}:{b} without changing the quantities into an answer.",
        reason="Deterministic ratio visual selected and labeled from the problem context.",
        template="ratio_dots_context_labeled",
    )


def _looks_like_fraction_addition(text: str) -> bool:
    lowered = str(text or "").lower()
    fractions = re.findall(r"\d+\s*/\s*\d+", lowered)

    if len(fractions) < 2:
        return False

    if any(word in lowered for word in ["divide", "divided", "division", "÷", "groups of", "fit in"]):
        return False

    return bool(
        "+" in lowered
        or "add" in lowered
        or "sum" in lowered
        or "plus" in lowered
    )


def _fraction_addition_visual(problem: str) -> dict[str, Any] | None:
    fractions = _extract_fractions(problem)
    if len(fractions) < 2:
        return None

    first = fractions[0]
    second = fractions[1]
    n1, d1 = _parse_fraction(first)
    n2, d2 = _parse_fraction(second)

    if not d1 or not d2:
        return None

    d1 = max(2, min(int(d1), 12))
    d2 = max(2, min(int(d2), 12))
    n1 = max(0, min(int(n1 or 0), d1))
    n2 = max(0, min(int(n2 or 0), d2))

    start_x = 120
    bar_w = 480
    cell_h = 42
    y1 = 128
    y2 = 208

    def bar(y: int, numerator: int, denominator: int, label: str, fill: str) -> str:
        cell_w = bar_w / denominator
        cells = []
        for index in range(denominator):
            x = start_x + index * cell_w
            cell_fill = fill if index < numerator else "#ffffff"
            cells.append(
                f'<rect x="{x:.2f}" y="{y}" width="{cell_w:.2f}" height="{cell_h}" fill="{cell_fill}" stroke="#2b6cb0" stroke-width="2" />'
            )
        return (
            f'<text x="{start_x - 22}" y="{y + 27}" text-anchor="end" font-size="18" font-weight="700" fill="#2d3748">{_escape(label)}</text>'
            + "".join(cells)
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Fraction addition strip diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Fraction Addition Structure</text>
  <text x="360" y="78" text-anchor="middle" font-size="16" fill="#4a5568">Use equal-size parts before adding fractions.</text>
  {bar(y1, n1, d1, first, "#bee3f8")}
  {bar(y2, n2, d2, second, "#c6f6d5")}
  <text x="360" y="292" text-anchor="middle" font-size="22" font-weight="700" fill="#dd6b20">Rewrite with a common denominator, then add.</text>
  <text x="360" y="328" text-anchor="middle" font-size="13" fill="#718096">The visual shows the two addends and the need for equal-sized parts without calculating the final sum.</text>
</svg>"""

    return _visual(
        svg=svg,
        caption=f"A fraction-strip diagram showing {first} and {second}; the student should rewrite them with equal-size parts before adding.",
        reason="Deterministic fraction-addition visual selected for adding fractions.",
        template="fraction_addition_strips",
    )

def _looks_like_division_reasoning(text: str) -> bool:
    lowered = str(text or "").lower()

    # Do not treat ordinary fractions like 2/3 + 1/6 as whole-number
    # division. Slash notation can mean a fraction, not a division prompt.
    has_fraction = bool(re.search(r"\d+\s*/\s*\d+", lowered))
    has_explicit_division_language = bool(
        "division reasoning" in lowered
        or "divided by" in lowered
        or "divide" in lowered
        or "shared equally" in lowered
        or "split equally" in lowered
        or "equal groups" in lowered
        or "how many groups" in lowered
        or "groups of" in lowered
        or "÷" in lowered
    )

    if has_fraction and not has_explicit_division_language:
        return False

    return bool(has_explicit_division_language) and not _looks_like_fraction_division(lowered)


def _extract_division_parts(text: str) -> tuple[str, str] | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|÷)\s*(\d+(?:\.\d+)?)", text)
    if match:
        return _num_text(match.group(1)), _num_text(match.group(2))

    match = re.search(r"(\d+(?:\.\d+)?)\s+divided\s+by\s+(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return _num_text(match.group(1)), _num_text(match.group(2))

    nums = [n for n in _numbers(text) if "/" not in n]
    if len(nums) >= 2:
        return _num_text(nums[0]), _num_text(nums[1])

    return None


def _division_reasoning_visual(problem: str) -> dict[str, Any] | None:
    parts = _extract_division_parts(problem)
    if not parts:
        return None
    total, groups = parts
    item_label = _extract_problem_item_label(problem, "items")
    try:
        group_count = int(float(groups))
    except ValueError:
        group_count = 4
    group_count = max(2, min(group_count, 8))

    start_x = 110
    y = 145
    box_w = 64
    box_h = 56
    gap = 10
    total_w = group_count * box_w + (group_count - 1) * gap

    boxes = []
    for index in range(group_count):
        x = start_x + index * (box_w + gap)
        boxes.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" fill="#e6fffa" stroke="#2c7a7b" stroke-width="2" />'
        )
        boxes.append(
            f'<text x="{x + box_w / 2}" y="{y + 35}" text-anchor="middle" font-size="20" font-weight="700" fill="#2c7a7b">?</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Division reasoning equal groups diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Division Reasoning</text>
  <text x="360" y="78" text-anchor="middle" font-size="16" fill="#4a5568">Use equal groups to understand what the quotient represents.</text>
  {''.join(boxes)}
  <line x1="{start_x}" y1="{y + box_h + 36}" x2="{start_x + total_w}" y2="{y + box_h + 36}" stroke="#2c7a7b" stroke-width="3" />
  <text x="{start_x + total_w / 2}" y="{y + box_h + 68}" text-anchor="middle" font-size="20" font-weight="700" fill="#2d3748">Total = {_escape(total)}</text>
  <text x="{start_x + total_w / 2}" y="{y + box_h + 100}" text-anchor="middle" font-size="22" font-weight="700" fill="#dd6b20">Each group = ?</text>
  <text x="360" y="334" text-anchor="middle" font-size="13" fill="#718096">The diagram shows {_escape(groups)} equal groups of {_escape(item_label)} without calculating the quotient.</text>
</svg>'''

    return _visual(
        svg=svg,
        caption=f"An equal-groups diagram showing a total of {total} {item_label} split into {groups} groups, with each group left unknown.",
        reason="Deterministic division visual selected for an equal-sharing or division-reasoning problem.",
        template="division_equal_groups",
    )


def _looks_like_fraction_division(text: str) -> bool:
    has_fraction = bool(re.search(r"\d+\s*/\s*\d+", text))
    if not has_fraction:
        return False

    return bool(
        "fraction division" in text
        or "divide" in text
        or "divided" in text
        or "division" in text
        or "÷" in text
        or "fit in" in text
        or "fit inside" in text
        or "pieces fit" in text
        or "groups of" in text
        or re.search(r"\d+\s*/\s*\d+\s*/\s*\d+", text)
    )


def _extract_fractions(text: str) -> list[str]:
    return [re.sub(r"\s+", "", item) for item in re.findall(r"\d+\s*/\s*\d+", text)]


def _fraction_division_visual(problem: str) -> dict[str, Any] | None:
    fractions = _extract_fractions(problem)
    if not fractions:
        return None

    whole_fraction = fractions[0]
    group_fraction = fractions[1] if len(fractions) > 1 else "unit fraction"

    numerator, denominator = _parse_fraction(whole_fraction)
    if denominator is None:
        denominator = 8
        numerator = 4
    denominator = max(2, min(denominator, 12))
    numerator = max(0, min(numerator or 0, denominator))

    start_x = 120
    y = 150
    cell_w = 42
    cell_h = 54
    cells = []
    for index in range(denominator):
        x = start_x + index * cell_w
        fill = "#bee3f8" if index < numerator else "#ffffff"
        cells.append(
            f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#2b6cb0" stroke-width="2" />'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Fraction division bar diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Fraction Division Model</text>
  <text x="360" y="78" text-anchor="middle" font-size="16" fill="#4a5568">Think about how many equal-size pieces fit inside the shaded amount.</text>
  {''.join(cells)}
  <text x="{start_x + denominator * cell_w / 2}" y="{y + cell_h + 42}" text-anchor="middle" font-size="20" font-weight="700" fill="#2d3748">Shaded amount: {_escape(whole_fraction)}</text>
  <text x="{start_x + denominator * cell_w / 2}" y="{y + cell_h + 76}" text-anchor="middle" font-size="22" font-weight="700" fill="#dd6b20">Groups of {_escape(group_fraction)} = ?</text>
  <text x="360" y="332" text-anchor="middle" font-size="13" fill="#718096">The bar shows the dividend and asks how many divisor-sized groups fit.</text>
</svg>'''

    return _visual(
        svg=svg,
        caption=f"A fraction bar showing {whole_fraction} as the amount being divided into groups of {group_fraction}.",
        reason="Deterministic fraction-division visual selected for a fraction division problem.",
        template="fraction_division_bar",
    )


def _parse_fraction(text: str) -> tuple[int | None, int | None]:
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(text or ""))
    if not match:
        return None, None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        return None, None
    return numerator, denominator


def _looks_like_coordinate_problem(text: str) -> bool:
    return bool(
        "coordinate" in text
        or "coordinate plane" in text
        or re.search(r"\((-?\d+)\s*,\s*(-?\d+)\)", text)
    )


def _coordinate_visual(problem: str) -> dict[str, Any] | None:
    points = re.findall(r"\((-?\d+)\s*,\s*(-?\d+)\)", problem)
    if not points and "coordinate" not in problem.lower():
        return None

    # Keep the coordinate plane inside the 720 x 360 SVG.
    # The previous version changed the axis range to -20..20 while keeping
    # spacing at 22 px, so many grid lines and points were drawn outside the
    # SVG viewBox. This version expands the coordinate range while shrinking
    # spacing dynamically so values greater than 6 still stay visible.
    parsed_points: list[tuple[int, int]] = []
    for px, py in points[:3]:
        try:
            parsed_points.append((int(px), int(py)))
        except ValueError:
            continue

    max_abs_coordinate = max(
        [abs(value) for point in parsed_points for value in point],
        default=6,
    )

    axis_limit = max(6, min(12, max_abs_coordinate))

    origin_x = 360
    origin_y = 190
    plot_left = 214
    plot_right = 506
    plot_top = 54
    plot_bottom = 326

    max_x_spacing = (plot_right - origin_x) / axis_limit
    max_y_spacing = min(origin_y - plot_top, plot_bottom - origin_y) / axis_limit
    spacing = max(10, min(22, int(min(max_x_spacing, max_y_spacing))))

    x_min = origin_x - axis_limit * spacing
    x_max = origin_x + axis_limit * spacing
    y_min = origin_y - axis_limit * spacing
    y_max = origin_y + axis_limit * spacing

    grid = []
    for i in range(-axis_limit, axis_limit + 1):
        x = origin_x + i * spacing
        y = origin_y - i * spacing

        stroke = "#dbe4ef" if i == 0 else "#edf2f7"
        stroke_width = "2" if i == 0 else "1"

        grid.append(
            f'<line x1="{x}" y1="{y_min}" x2="{x}" y2="{y_max}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" />'
        )
        grid.append(
            f'<line x1="{x_min}" y1="{y}" x2="{x_max}" y2="{y}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" />'
        )

    plotted = []
    for idx, (x_val, y_val) in enumerate(parsed_points, start=1):
        if -axis_limit <= x_val <= axis_limit and -axis_limit <= y_val <= axis_limit:
            x = origin_x + x_val * spacing
            y = origin_y - y_val * spacing

            # Keep the point label inside the plotted grid area.
            label_x = min(max(x + 12, x_min + 8), x_max - 92)
            label_y = y - 10
            if label_y < y_min + 18:
                label_y = y + 24
            if label_y > y_max - 8:
                label_y = y - 14

            plotted.append(
                f'<circle cx="{x}" cy="{y}" r="7" fill="#dd6b20" '
                f'stroke="#7c2d12" stroke-width="1.5" />'
            )
            plotted.append(
                f'<text x="{label_x}" y="{label_y}" font-size="14" '
                f'font-weight="700" fill="#dd6b20">P{idx} ({_escape(x_val)}, {_escape(y_val)})</text>'
            )

    axis_note = f"Grid shown from -{axis_limit} to {axis_limit}."

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Coordinate plane diagram">
  <rect width="720" height="360" fill="#ffffff" />
  <text x="360" y="34" text-anchor="middle" font-size="24" font-weight="700" fill="#1a202c">Coordinate Plane</text>
  {''.join(grid)}
  <line x1="{x_min}" y1="{origin_y}" x2="{x_max}" y2="{origin_y}" stroke="#2d3748" stroke-width="2.2" />
  <line x1="{origin_x}" y1="{y_min}" x2="{origin_x}" y2="{y_max}" stroke="#2d3748" stroke-width="2.2" />
  <text x="{x_max + 8}" y="{origin_y + 5}" font-size="15" fill="#2d3748">x</text>
  <text x="{origin_x + 7}" y="{y_min - 6}" font-size="15" fill="#2d3748">y</text>
  {''.join(plotted)}
  <text x="360" y="342" text-anchor="middle" font-size="13" fill="#718096">Use the first number for x and the second number for y. {_escape(axis_note)}</text>
</svg>"""

    return _visual(
        svg=svg,
        caption="A coordinate-plane template for identifying x- and y-values in an ordered pair.",
        reason="Deterministic coordinate-plane visual selected with a dynamically scaled grid so values greater than 6 stay visible.",
        template="coordinate_plane_scaled",
    )
