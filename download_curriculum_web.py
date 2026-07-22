from dataclasses import dataclass
from pathlib import Path
import re
import time

import requests
from bs4 import BeautifulSoup, NavigableString


PROJECT_ROOT = Path(__file__).resolve().parent
CURRICULUM_DIR = PROJECT_ROOT / "data" / "curriculum"

BASE_URL = (
    "https://access.openupresources.org/"
    "curricula/our6-8math/en/grade-6/unit-4"
)


@dataclass
class LessonInfo:
    number: int
    title: str
    filename: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/lesson-{self.number}/index.html"


LESSONS = [
    LessonInfo(
        1,
        "Size of Divisor and Size of Quotient",
        "grade6_unit4_lesson1_size_of_divisor.txt",
    ),
    LessonInfo(
        2,
        "Meanings of Division",
        "grade6_unit4_lesson2_meanings_of_division.txt",
    ),
    LessonInfo(
        3,
        "Interpreting Division Situations",
        "grade6_unit4_lesson3_interpreting_division_situations.txt",
    ),
    LessonInfo(
        4,
        "How Many Groups? Part 1",
        "grade6_unit4_lesson4_how_many_groups_part1.txt",
    ),
    LessonInfo(
        5,
        "How Many Groups? Part 2",
        "grade6_unit4_lesson5_how_many_groups_part2.txt",
    ),
    LessonInfo(
        6,
        "Diagrams to Find the Number of Groups",
        "grade6_unit4_lesson6_diagrams_to_find_number_of_groups.txt",
    ),
    LessonInfo(
        7,
        "What Fraction of a Group?",
        "grade6_unit4_lesson7_what_fraction_of_a_group.txt",
    ),
    LessonInfo(
        8,
        "How Much in Each Group? Part 1",
        "grade6_unit4_lesson8_how_much_in_each_group_part1.txt",
    ),
    LessonInfo(
        9,
        "How Much in Each Group? Part 2",
        "grade6_unit4_lesson9_how_much_in_each_group_part2.txt",
    ),
    LessonInfo(
        10,
        "Dividing by Unit and Non-Unit Fractions",
        "grade6_unit4_lesson10_dividing_by_unit_and_non_unit_fractions.txt",
    ),
    LessonInfo(
        11,
        "Using an Algorithm to Divide Fractions",
        "grade6_unit4_lesson11_algorithm_to_divide_fractions.txt",
    ),
    LessonInfo(
        12,
        "Fractional Lengths",
        "grade6_unit4_lesson12_fractional_lengths.txt",
    ),
    LessonInfo(
        13,
        "Rectangles with Fractional Side Lengths",
        "grade6_unit4_lesson13_rectangles_fractional_side_lengths.txt",
    ),
    LessonInfo(
        14,
        "Fractional Lengths in Triangles and Prisms",
        "grade6_unit4_lesson14_fractional_lengths_triangles_prisms.txt",
    ),
    LessonInfo(
        15,
        "Volume of Prisms",
        "grade6_unit4_lesson15_volume_of_prisms.txt",
    ),
    LessonInfo(
        16,
        "Solving Problems Involving Fractions",
        "grade6_unit4_lesson16_solving_problems_involving_fractions.txt",
    ),
    LessonInfo(
        17,
        "Fitting Boxes into Boxes",
        "grade6_unit4_lesson17_fitting_boxes_into_boxes.txt",
    ),
]


def preserve_math_text(soup: BeautifulSoup) -> None:
    """Keep math expressions before removing script tags."""

    for script in list(soup.find_all("script")):
        script_type = script.get("type", "").lower()

        if "math/tex" in script_type:
            latex = script.get_text(" ", strip=True)

            if latex:
                script.replace_with(
                    NavigableString(f" {latex} ")
                )

    for math_tag in list(soup.find_all("math")):
        annotation = math_tag.find(
            "annotation",
            attrs={
                "encoding": re.compile(
                    r"tex",
                    re.IGNORECASE,
                )
            },
        )

        if annotation:
            latex = annotation.get_text(" ", strip=True)
            math_tag.replace_with(
                NavigableString(f" {latex} ")
            )

    for image in list(soup.find_all("img")):
        alt_text = image.get("alt", "").strip()

        if alt_text:
            image.replace_with(
                NavigableString(
                    f"\n[Image description: {alt_text}]\n"
                )
            )
        else:
            image.decompose()


def clean_latex_fractions(text: str) -> str:
    """Convert simple LaTeX fractions into readable text."""

    text = re.sub(
        r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1/\2",
        text,
    )

    text = re.sub(
        r"\\(?:d?frac)\s*([0-9])\s*([0-9])",
        r"\1/\2",
        text,
    )

    return text


def clean_text(text: str) -> str:
    """Clean encoding issues and make curriculum text readable."""

    replacements = {
        "\u00a0": " ",
        "Â": "",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "—",
        "�": "",
        r"\div": "÷",
        r"\times": "×",
        r"\cdot": "·",
        r"\boldcdot": "·",
        r"\le": "≤",
        r"\ge": "≥",
        r"\lt": "<",
        r"\gt": ">",
        r"\!": "",
        r"\left": "",
        r"\right": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = clean_latex_fractions(text)

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_lesson_text(html: str, lesson: LessonInfo) -> str:
    soup = BeautifulSoup(html, "html.parser")

    preserve_math_text(soup)

    for tag in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "noscript",
        ]
    ):
        tag.decompose()

    content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
    )

    if content is None:
        raise RuntimeError(
            f"Could not locate lesson content for lesson {lesson.number}."
        )

    text = content.get_text(
        separator="\n",
        strip=True,
    )

    text = clean_text(text)

    start_patterns = [
        f"Lesson {lesson.number} {lesson.title}",
        f"Lesson {lesson.number}: {lesson.title}",
        f"Lesson {lesson.number}",
    ]

    for marker in start_patterns:
        start = text.find(marker)

        if start != -1:
            text = text[start:]
            break

    return text


def download_lesson(lesson: LessonInfo) -> str:
    response = requests.get(
        lesson.url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "MathTutorCurriculumPrototype/1.0"
            )
        },
        timeout=30,
    )

    response.raise_for_status()

    html = response.content.decode(
        "utf-8",
        errors="replace",
    )

    return extract_lesson_text(html, lesson)


def clear_curriculum_folder() -> None:
    """Delete old temporary TXT files from the RAG curriculum folder."""

    CURRICULUM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in CURRICULUM_DIR.glob("*.txt"):
        path.unlink()
        print(f"Removed old curriculum file: {path.name}")


def write_lesson_file(
    lesson: LessonInfo,
    lesson_text: str,
) -> Path:
    output_file = CURRICULUM_DIR / lesson.filename

    header = (
        "Open Up Resources 6-8 Math\n"
        "Grade 6, Unit 4: Dividing Fractions\n"
        f"Lesson {lesson.number}: {lesson.title}\n"
        "Resource Type: Student Curriculum\n"
        f"Source: {lesson.url}\n"
        "\n"
        + "=" * 70
        + "\n\n"
    )

    output_file.write_text(
        header + lesson_text,
        encoding="utf-8",
    )

    return output_file


def main() -> None:
    print("Preparing Grade 6 Unit 4 curriculum files...")
    clear_curriculum_folder()

    successful_files = []
    failed_lessons = []

    for lesson in LESSONS:
        print(
            f"Downloading Lesson {lesson.number}: "
            f"{lesson.title}"
        )

        try:
            lesson_text = download_lesson(lesson)
            output_file = write_lesson_file(
                lesson,
                lesson_text,
            )

            successful_files.append(output_file)
            print(f"Saved: {output_file.name}")

        except Exception as error:
            failed_lessons.append(
                (lesson.number, lesson.title, str(error))
            )
            print(
                f"Failed Lesson {lesson.number}: "
                f"{lesson.title}"
            )
            print(error)

        time.sleep(0.3)

    print("\nFinished.")
    print(f"Successful files: {len(successful_files)}")
    print(f"Failed lessons: {len(failed_lessons)}")

    if failed_lessons:
        print("\nFailed lesson details:")
        for number, title, error in failed_lessons:
            print(f"- Lesson {number}: {title}")
            print(f"  {error}")

    print("\nCurrent curriculum files:")
    for path in sorted(CURRICULUM_DIR.glob("*.txt")):
        print(f"- {path.name}")


if __name__ == "__main__":
    main()