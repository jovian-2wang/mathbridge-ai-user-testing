import re
from fractions import Fraction
from math import ceil, gcd
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []

    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            ]
        )

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)

    return ImageFont.load_default()


def _parse_fraction(value: Any) -> Fraction:
    text = str(value).strip()

    if not text:
        raise ValueError("A fraction value is required.")

    return Fraction(text)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)

    return f"{value.numerator}/{value.denominator}"


def create_visual(
    visual_type: str,
    visual_data: dict[str, Any],
) -> Image.Image | None:
    """
    Create an accurate instructional visual and return a PIL Image.
    The image is not saved to disk.
    """

    try:
        if visual_type == "fraction_division_bar":
            return _create_fraction_division_bar(
                dividend=_parse_fraction(
                    visual_data.get("dividend")
                ),
                divisor=_parse_fraction(
                    visual_data.get("divisor")
                ),
            )

        if visual_type == "ratio_bar":
            return _create_ratio_bar(
                label_a=str(visual_data.get("label_a", "Part A")),
                value_a=int(visual_data.get("value_a", 0)),
                label_b=str(visual_data.get("label_b", "Part B")),
                value_b=int(visual_data.get("value_b", 0)),
            )

        if visual_type == "unit_rate_table":
            # New explicit schema:
            #   total_quantity / quantity_label / total_units / unit_label
            # Legacy schema is still supported as a fallback.
            return _create_unit_rate_table(
                total_quantity=float(
                    visual_data.get(
                        "total_quantity",
                        visual_data.get("total_value", 0),
                    )
                ),
                total_units=float(visual_data.get("total_units", 1)),
                quantity_label=str(
                    visual_data.get(
                        "quantity_label",
                        visual_data.get("value_label", "miles"),
                    )
                ),
                unit_label=str(
                    visual_data.get(
                        "unit_label",
                        visual_data.get("item_label", "units"),
                    )
                ),
            )

        if visual_type == "whole_number_division_tape":
            return _create_whole_number_division_tape(
                total=float(visual_data.get("total", 0)),
                group_size=float(visual_data.get("group_size", 1)),
                group_count=visual_data.get("group_count"),
                mode=str(visual_data.get("mode", "measurement_division")),
            )


        if visual_type == "fraction_decimal_bar":
            return _create_fraction_decimal_bar(
                fraction_value=_parse_fraction(
                    visual_data.get("fraction")
                ),
                decimal_value=float(visual_data.get("decimal", 0)),
            )

    except (ValueError, ZeroDivisionError, TypeError):
        return None

    return None


def _create_fraction_division_bar(
    dividend: Fraction,
    divisor: Fraction,
) -> Image.Image | None:
    if dividend < 0 or divisor <= 0:
        return None

    common_denominator = _lcm(
        dividend.denominator,
        divisor.denominator,
    )

    dividend_units_fraction = (
        dividend * common_denominator
    )
    divisor_units_fraction = (
        divisor * common_denominator
    )

    if (
        dividend_units_fraction.denominator != 1
        or divisor_units_fraction.denominator != 1
    ):
        return None

    dividend_units = dividend_units_fraction.numerator
    divisor_units = divisor_units_fraction.numerator

    if divisor_units <= 0:
        return None

    whole_count = max(1, ceil(float(dividend)))
    total_units = whole_count * common_denominator

    # Keep the first version readable and prevent excessively large images.
    if total_units > 48 or common_denominator > 24:
        return None

    image_width = 1200
    margin_x = 80
    title_y = 45
    bar_y = 190
    bar_height = 170

    image = Image.new(
        "RGB",
        (image_width, 520),
        "white",
    )
    draw = ImageDraw.Draw(image)

    title_font = _font(38, bold=True)
    body_font = _font(27)
    result_font = _font(31, bold=True)

    dividend_text = _format_fraction(dividend)
    divisor_text = _format_fraction(divisor)
    quotient = dividend / divisor
    quotient_text = _format_fraction(quotient)

    title = (
        f"Visual explanation: "
        f"{dividend_text} ÷ {divisor_text}"
    )
    draw.text(
        (margin_x, title_y),
        title,
        fill="black",
        font=title_font,
    )

    available_width = image_width - (2 * margin_x)
    cell_width = available_width / total_units

    for index in range(total_units):
        x0 = margin_x + (index * cell_width)
        x1 = margin_x + ((index + 1) * cell_width)

        is_in_dividend = index < dividend_units
        fill = "#B8D8F0" if is_in_dividend else "#F3F3F3"

        draw.rectangle(
            [x0, bar_y, x1, bar_y + bar_height],
            fill=fill,
            outline="#6E6E6E",
            width=2,
        )

    # Draw whole-number boundaries.
    for whole_index in range(whole_count + 1):
        x = (
            margin_x
            + whole_index
            * common_denominator
            * cell_width
        )
        draw.line(
            [x, bar_y, x, bar_y + bar_height],
            fill="black",
            width=5,
        )

    # Outline each divisor-sized group inside the dividend.
    group_start = 0
    while group_start < dividend_units:
        group_end = min(
            group_start + divisor_units,
            dividend_units,
        )

        x0 = margin_x + (group_start * cell_width)
        x1 = margin_x + (group_end * cell_width)

        draw.rectangle(
            [
                x0,
                bar_y - 10,
                x1,
                bar_y + bar_height + 10,
            ],
            outline="#C84B31",
            width=5,
        )

        group_start += divisor_units

    note = (
        f"Each red outline represents a group of "
        f"{divisor_text}."
    )
    draw.text(
        (margin_x, 400),
        note,
        fill="black",
        font=body_font,
    )

    result = (
        f"{dividend_text} contains "
        f"{quotient_text} groups of {divisor_text}."
    )
    draw.text(
        (margin_x, 448),
        result,
        fill="black",
        font=result_font,
    )

    return image
def _create_ratio_bar(
    label_a: str,
    value_a: int,
    label_b: str,
    value_b: int,
) -> Image.Image | None:
    if value_a <= 0 or value_b <= 0:
        return None

    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)

    title_font = _font(38, bold=True)
    body_font = _font(27)
    result_font = _font(31, bold=True)

    margin_x = 80
    bar_y = 170
    bar_height = 80
    total = value_a + value_b
    cell_count = total
    available_width = 1000
    cell_width = available_width / cell_count

    draw.text(
        (margin_x, 40),
        f"Visual explanation: ratio of {label_a} to {label_b}",
        fill="black",
        font=title_font,
    )

    current_x = margin_x
    for _ in range(value_a):
        draw.rectangle(
            [current_x, bar_y, current_x + cell_width, bar_y + bar_height],
            fill="#B8D8F0",
            outline="black",
            width=2,
        )
        current_x += cell_width

    for _ in range(value_b):
        draw.rectangle(
            [current_x, bar_y, current_x + cell_width, bar_y + bar_height],
            fill="#F7C59F",
            outline="black",
            width=2,
        )
        current_x += cell_width

    draw.text(
        (margin_x, 280),
        f"{label_a}: {value_a}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (margin_x + 260, 280),
        f"{label_b}: {value_b}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (margin_x, 340),
        f"Ratio = {value_a}:{value_b}",
        fill="black",
        font=result_font,
    )

    return image
def _clean_unit_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"^1\s+", "", text)
    return text or "unit"


def _format_labeled_amount(
    amount: float,
    label: str,
) -> str:
    label = _clean_unit_label(label)

    if amount == int(amount):
        amount_text = str(int(amount))
    else:
        amount_text = f"{amount:g}"

    # Simple singular/plural cleanup for demo labels such as hours/miles/items.
    display_label = label
    if amount == 1 and display_label.endswith("s"):
        display_label = display_label[:-1]
    elif amount != 1 and not display_label.endswith("s"):
        display_label = display_label + "s"

    return f"{amount_text} {display_label}"


def _create_unit_rate_table(
    total_quantity: float,
    total_units: float,
    quantity_label: str,
    unit_label: str,
) -> Image.Image | None:
    """
    Render a deterministic unit-rate table.

    Example:
    total_quantity=180, quantity_label="miles",
    total_units=3, unit_label="hours"

    The table should read:
    3 hours -> 180 miles
    1 hour  -> 60 miles
    Unit rate = 60 miles per hour
    """

    if total_units <= 0:
        return None

    unit_rate = total_quantity / total_units

    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)

    title_font = _font(38, bold=True)
    body_font = _font(27)
    result_font = _font(31, bold=True)

    quantity_label = _clean_unit_label(quantity_label)
    unit_label = _clean_unit_label(unit_label)
    singular_unit = unit_label[:-1] if unit_label.endswith("s") else unit_label

    draw.text(
        (80, 40),
        "Visual explanation: unit rate",
        fill="black",
        font=title_font,
    )

    draw.text(
        (80, 92),
        f"Find how many {quantity_label} there are for 1 {singular_unit}.",
        fill="#4B5563",
        font=body_font,
    )

    # table
    x0, y0 = 120, 170
    col1 = 440
    col2 = 820
    row_h = 80

    # header
    draw.rectangle([x0, y0, col1, y0 + row_h], outline="black", width=3, fill="#EAF2F8")
    draw.rectangle([col1, y0, col2, y0 + row_h], outline="black", width=3, fill="#EAF2F8")
    draw.text((150, y0 + 22), "Units", fill="black", font=body_font)
    draw.text((485, y0 + 22), "Quantity", fill="black", font=body_font)

    # total row
    draw.rectangle([x0, y0 + row_h, col1, y0 + 2 * row_h], outline="black", width=3)
    draw.rectangle([col1, y0 + row_h, col2, y0 + 2 * row_h], outline="black", width=3)
    draw.text(
        (150, y0 + row_h + 22),
        _format_labeled_amount(total_units, unit_label),
        fill="black",
        font=body_font,
    )
    draw.text(
        (485, y0 + row_h + 22),
        _format_labeled_amount(total_quantity, quantity_label),
        fill="black",
        font=body_font,
    )

    # unit row
    draw.rectangle([x0, y0 + 2 * row_h, col1, y0 + 3 * row_h], outline="black", width=3)
    draw.rectangle([col1, y0 + 2 * row_h, col2, y0 + 3 * row_h], outline="black", width=3)
    draw.text(
        (150, y0 + 2 * row_h + 22),
        _format_labeled_amount(1, unit_label),
        fill="black",
        font=body_font,
    )
    draw.text(
        (485, y0 + 2 * row_h + 22),
        _format_labeled_amount(unit_rate, quantity_label),
        fill="black",
        font=body_font,
    )

    draw.text(
        (120, 430),
        f"Unit rate = {unit_rate:.2f} {quantity_label} per {singular_unit}",
        fill="black",
        font=result_font,
    )

    return image

def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _create_whole_number_division_tape(
    total: float,
    group_size: float,
    group_count: Any = None,
    mode: str = "measurement_division",
) -> Image.Image | None:
    """
    Deterministic tape diagram for whole-number division.

    Measurement division:
        total and group size are known; number of groups is unknown.
        Example: 365 ÷ 23 means "How many groups of 23 are in 365?"

    Equal sharing:
        total and number of groups are known; group size is unknown.
    """

    if total <= 0 or group_size <= 0:
        return None

    known_group_count = _optional_float(group_count)

    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)

    title_font = _font(38, bold=True)
    body_font = _font(27)
    small_font = _font(23)
    result_font = _font(31, bold=True)

    total_text = _format_number(total)
    group_size_text = _format_number(group_size)

    draw.text(
        (80, 40),
        "Visual explanation: division tape diagram",
        fill="black",
        font=title_font,
    )

    draw.text(
        (80, 92),
        f"Think of {total_text} as groups of {group_size_text}.",
        fill="#4B5563",
        font=body_font,
    )

    bar_x = 120
    bar_y = 210
    bar_width = 900
    bar_height = 90

    if known_group_count and known_group_count > 0 and known_group_count <= 8:
        display_parts = int(known_group_count)
        show_exact_count = True
    else:
        display_parts = 6
        show_exact_count = False

    cell_width = bar_width / display_parts

    for index in range(display_parts):
        x0 = bar_x + index * cell_width
        x1 = bar_x + (index + 1) * cell_width

        draw.rectangle(
            [x0, bar_y, x1, bar_y + bar_height],
            fill="#D9F2F2",
            outline="#1F6F68",
            width=3,
        )

        if show_exact_count or index < display_parts - 1:
            label = group_size_text
        else:
            label = "..."

        draw.text(
            (x0 + cell_width / 2 - 16, bar_y + 28),
            label,
            fill="#1F6F68",
            font=body_font,
        )

    # Total label above the tape.
    total_label = f"Total = {total_text}"
    draw.text(
        (bar_x + 330, bar_y - 52),
        total_label,
        fill="#1F6F68",
        font=result_font,
    )

    # Unknown / known group count below the tape.
    if show_exact_count:
        group_label = f"{display_parts} groups of {group_size_text}"
    else:
        group_label = f"Number of groups of {group_size_text} = ?"

    draw.text(
        (bar_x + 260, bar_y + bar_height + 35),
        group_label,
        fill="#1F6F68",
        font=result_font,
    )

    # Estimation support, without giving the final answer.
    draw.text(
        (80, 430),
        (
            f"Use multiples of {group_size_text} to estimate how many "
            f"groups fit into {total_text}."
        ),
        fill="black",
        font=small_font,
    )

    return image


def _create_fraction_decimal_bar(
    fraction_value: Fraction,
    decimal_value: float,
) -> Image.Image | None:
    if fraction_value < 0 or fraction_value > 1:
        return None

    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)

    title_font = _font(38, bold=True)
    body_font = _font(27)
    result_font = _font(31, bold=True)

    draw.text(
        (80, 40),
        "Visual explanation: fraction and decimal",
        fill="black",
        font=title_font,
    )

    bar_x = 100
    bar_y = 180
    bar_width = 900
    bar_height = 80

    denominator = fraction_value.denominator
    numerator = fraction_value.numerator
    cell_width = bar_width / denominator

    for i in range(denominator):
        x0 = bar_x + i * cell_width
        x1 = bar_x + (i + 1) * cell_width
        fill = "#B8D8F0" if i < numerator else "#F3F3F3"
        draw.rectangle(
            [x0, bar_y, x1, bar_y + bar_height],
            fill=fill,
            outline="black",
            width=2,
        )

    draw.text(
        (100, 300),
        f"Fraction: {numerator}/{denominator}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (100, 350),
        f"Decimal: {decimal_value}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (100, 420),
        f"{numerator}/{denominator} = {decimal_value}",
        fill="black",
        font=result_font,
    )

    return image
