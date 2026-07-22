import json
import re
from typing import Any


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    cleaned: list[str] = []
    seen = set()

    for item in items:
        text = str(item or "").strip()
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

    return cleaned



PROBLEM_CONTEXT_RULES = [
    {
        "key": "school_classroom",
        "label": "school / classroom",
        "patterns": [
            r"\bstudents?\b",
            r"\bclass(?:room)?\b",
            r"\bteacher\b",
            r"\bnotebooks?\b",
            r"\bpencils?\b",
            r"\bbooks?\b",
            r"\bdesks?\b",
            r"\bwalk(?:s|ed|ing)? to school\b",
        ],
    },
    {
        "key": "shopping_money",
        "label": "shopping / money",
        "patterns": [
            r"\$\s*\d+",
            r"\bcosts?\b",
            r"\bprices?\b",
            r"\bbuy(?:s|ing)?\b",
            r"\bbought\b",
            r"\bstores?\b",
            r"\bsale\b",
            r"\btickets?\b",
        ],
    },
    {
        "key": "food_baking",
        "label": "food / baking",
        "patterns": [
            r"\brecipe\b",
            r"\bbak(?:e|ing|ed)\b",
            r"\bcups?\b",
            r"\bflour\b",
            r"\bsugar\b",
            r"\bpizza\b",
            r"\bcake\b",
            r"\bcupcakes?\b",
            r"\bsnacks?\b",
            r"\bapples?\b",
            r"\boranges?\b",
        ],
    },
    {
        "key": "sports_games",
        "label": "sports / games",
        "patterns": [
            r"\bbasketball\b",
            r"\bsoccer\b",
            r"\bfootball\b",
            r"\bplayers?\b",
            r"\bteams?\b",
            r"\bpoints?\b",
            r"\bscores?\b",
            r"\bballs?\b",
        ],
    },
    {
        "key": "travel_distance",
        "label": "travel / distance",
        "patterns": [
            r"\bmiles?\b",
            r"\bkilometers?\b",
            r"\bkm\b",
            r"\bhours?\b",
            r"\bspeed\b",
            r"\bper hour\b",
            r"\btravel(?:s|ed|ing)?\b",
            r"\btrip\b",
        ],
    },
    {
        "key": "digital_games",
        "label": "games / digital items",
        "patterns": [
            r"\blevels?\b",
            r"\bcoins?\b",
            r"\bgems?\b",
            r"\broblox\b",
            r"\bvideo game\b",
            r"\bgames?\b",
        ],
    },
]


def detect_problem_context(problem_text: str = "") -> dict[str, Any]:
    """
    Detect the real-life context already present in the current problem.

    Contextualization should first respect the problem's own scenario. Learner
    preferences are used mainly when the problem is abstract or underspecified.
    """
    text = str(problem_text or "").strip()
    lowered = text.lower()

    if not lowered:
        return {
            "has_problem_context": False,
            "context_key": "abstract_math",
            "context_label": "abstract math",
            "matched_terms": [],
            "source": "none",
        }

    # Money/cost language is a strong contextual signal. Prioritize it over
    # item nouns like notebooks, which could otherwise look like classroom-only
    # context even when the problem is really about price per item.
    if re.search(r"\$\s*\d+|\bcosts?\b|\bprices?\b|\bbuy(?:s|ing)?\b|\bbought\b", lowered):
        return {
            "has_problem_context": True,
            "context_key": "shopping_money",
            "context_label": "shopping / money",
            "matched_terms": ["cost or money language"],
            "source": "problem",
        }

    for rule in PROBLEM_CONTEXT_RULES:
        matched_terms: list[str] = []
        for pattern in rule["patterns"]:
            if re.search(pattern, lowered):
                display = pattern
                display = display.replace(r"\b", "")
                display = display.replace("(?:s|ed|ing)?", "")
                display = display.replace("(?:room)?", "")
                display = display.replace("?", "")
                display = display.replace("\\", "")
                matched_terms.append(display)

        if matched_terms:
            return {
                "has_problem_context": True,
                "context_key": rule["key"],
                "context_label": rule["label"],
                "matched_terms": matched_terms[:5],
                "source": "problem",
            }

    return {
        "has_problem_context": False,
        "context_key": "abstract_math",
        "context_label": "abstract math",
        "matched_terms": [],
        "source": "learner_profile",
    }

def get_context_profile(student: dict | None) -> dict:
    """
    Read a student's contextualization profile without importing memory code.

    The memory layer owns persistence. This helper is intentionally lightweight
    so tutor, hint, visual, teacher, and UI code can share one context format.
    """
    if not isinstance(student, dict):
        student = {}

    profile = student.get("context_profile")
    if not isinstance(profile, dict):
        profile = student.get("learner_profile")

    if not isinstance(profile, dict):
        profile = {}

    learner = profile.get("learner")
    if not isinstance(learner, dict):
        learner = {}

    curriculum_scope = profile.get("curriculum_scope")
    if not isinstance(curriculum_scope, dict):
        curriculum_scope = {}

    strategy = profile.get("contextualization_strategy")
    if not isinstance(strategy, dict):
        strategy = {}

    return {
        "learner": {
            "grade_level": str(learner.get("grade_level") or "6"),
            "language_preference": str(
                learner.get("language_preference") or "English"
            ),
            "reading_level": str(learner.get("reading_level") or "Grade 6"),
            "math_confidence": str(learner.get("math_confidence") or "medium"),
            "learning_style": _as_list(
                learner.get("learning_style")
            ) or ["visual", "step-by-step", "real-life examples"],
        },
        "interests": _as_list(
            profile.get("interests")
        ) or ["basketball", "baking", "games"],
        "real_world_contexts": _as_list(
            profile.get("real_world_contexts")
        ) or ["shopping", "food", "sports", "classroom supplies"],
        "learning_needs": _as_list(
            profile.get("learning_needs")
        ) or ["word problems", "fractions", "unit rates"],
        "curriculum_scope": {
            "grade": str(curriculum_scope.get("grade") or "6"),
            "unit": str(
                curriculum_scope.get("unit")
                or "Ratios, rates, and fraction reasoning"
            ),
            "current_objective": str(
                curriculum_scope.get("current_objective")
                or "Use Grade 6 reasoning to solve the current math problem."
            ),
        },
        "contextualization_strategy": {
            "tone": str(strategy.get("tone") or "friendly Socratic tutor"),
            "example_style": str(
                strategy.get("example_style")
                or "Connect math to familiar real-life scenarios when helpful."
            ),
            "visual_style": str(
                strategy.get("visual_style")
                or "Use diagrams before abstract equations."
            ),
            "avoid": _as_list(
                strategy.get("avoid")
            ) or [
                "revealing final answers too early",
                "forced personalization",
                "overly abstract explanations",
            ],
        },
    }


def build_contextualization_context(
    student: dict | None,
    current_problem: str = "",
    curriculum_context: str = "",
    signals: dict | None = None,
) -> str:
    """
    Convert learner/course/context data into compact prompt-ready text.

    Later tutor, hint, visual, and teacher agents can consume this single block
    instead of each building their own preference prompt.
    """
    profile = get_context_profile(student)
    signals = signals or {}

    learner = profile["learner"]
    curriculum_scope = profile["curriculum_scope"]
    strategy = profile["contextualization_strategy"]

    problem_context = detect_problem_context(current_problem)

    lines = [
        "Contextualization profile:",
        f"- Grade / reading level: Grade {learner['grade_level']} / {learner['reading_level']}",
        f"- Language preference: {learner['language_preference']}",
        f"- Math confidence: {learner['math_confidence']}",
        f"- Learning style: {', '.join(learner['learning_style'])}",
        f"- Interests: {', '.join(profile['interests'])}",
        f"- Preferred real-life contexts: {', '.join(profile['real_world_contexts'])}",
        f"- Learning needs: {', '.join(profile['learning_needs'])}",
        f"- Curriculum scope: Grade {curriculum_scope['grade']}, {curriculum_scope['unit']}",
        f"- Current objective: {curriculum_scope['current_objective']}",
        f"- Strategy tone: {strategy['tone']}",
        f"- Example style: {strategy['example_style']}",
        f"- Visual style: {strategy['visual_style']}",
        f"- Avoid: {', '.join(strategy['avoid'])}",
        "",
        "Context selection rule:",
        f"- Detected problem context: {problem_context['context_label']} ({problem_context['source']})",
        "- If the problem already has a clear real-life scenario, keep that scenario and do not force a different learner-preference context.",
        "- If the problem is abstract or underspecified, choose a familiar context from the learner profile.",
        "",
        "Use this context to adapt examples, wording, visuals, and scaffolds only when it naturally supports the math.",
        "Keep the tutoring curriculum-aligned, student-friendly, and Socratic.",
        "Do not force personalization if it would make the math less clear.",
    ]

    concept = signals.get("concept")
    engagement = signals.get("engagement")
    misconception = signals.get("misconception")
    next_support = signals.get("next_support")

    if concept or engagement or misconception or next_support:
        lines.extend(
            [
                "",
                "Live learning state:",
                f"- Detected concept: {concept or 'not available'}",
                f"- Engagement: {engagement or 'not available'}",
                f"- Possible misconception: {misconception or 'none flagged'}",
                f"- Suggested next support: {next_support or 'not available'}",
            ]
        )

    if current_problem:
        lines.extend(
            [
                "",
                "Current problem:",
                str(current_problem)[:800],
            ]
        )

    if curriculum_context:
        lines.extend(
            [
                "",
                "Curriculum grounding summary:",
                str(curriculum_context)[:1200],
            ]
        )

    return "\n".join(lines)


def summarize_context_used(
    student: dict | None,
    current_problem: str = "",
    selected_context: str | None = None,
) -> dict[str, Any]:
    """
    Return a compact UI/analytics summary of context selection.

    Round 5 makes the selection explicit: problem context wins when present;
    learner preferences fill in when the problem is abstract.
    """
    profile = get_context_profile(student)
    learner = profile["learner"]
    curriculum_scope = profile["curriculum_scope"]
    problem_context = detect_problem_context(current_problem)

    profile_contexts = profile["real_world_contexts"]
    profile_context_text = ", ".join(profile_contexts) or "general real-life examples"

    if selected_context:
        real_world_context = selected_context
        context_source = "explicit selection"
        decision = (
            f"Used the selected context '{selected_context}' for this support move while keeping the Grade 6 objective aligned."
        )
    elif problem_context.get("has_problem_context"):
        real_world_context = problem_context["context_label"]
        context_source = "problem scenario"
        decision = (
            f"Used the problem's own {problem_context['context_label']} context instead of forcing a learner-preference context. "
            f"Learner preferences remain available for examples if the problem becomes abstract."
        )
    else:
        real_world_context = profile_contexts[0] if profile_contexts else "general real-life examples"
        context_source = "learner profile"
        decision = (
            f"The problem is abstract or underspecified, so the tutor may use the learner's preferred context '{real_world_context}' if it clarifies the math."
        )

    return {
        "learner_context": (
            f"Grade {learner['grade_level']}, "
            f"{', '.join(learner['learning_style'])}"
        ),
        "problem_context": problem_context["context_label"],
        "context_source": context_source,
        "real_life_context": real_world_context,
        "learner_preferences_available": profile_context_text,
        "curriculum_context": (
            f"Grade {curriculum_scope['grade']} · {curriculum_scope['unit']}"
        ),
        "learning_needs": ", ".join(profile["learning_needs"]),
        "personalization_decision": decision,
        "current_problem": str(current_problem or "")[:240],
    }

def profile_to_json(profile: dict | None) -> str:
    return json.dumps(
        get_context_profile({"context_profile": profile or {}}),
        ensure_ascii=False,
        indent=2,
    )

def summarize_context_profile_for_class(student: dict | None) -> dict[str, Any]:
    """Compact learner-context summary for class overview rows and agents."""
    profile = get_context_profile(student)
    learner = profile["learner"]
    curriculum_scope = profile["curriculum_scope"]
    strategy = profile["contextualization_strategy"]

    return {
        "learning_style": ", ".join(learner["learning_style"]),
        "math_confidence": learner["math_confidence"],
        "language_preference": learner["language_preference"],
        "reading_level": learner["reading_level"],
        "preferred_contexts": ", ".join(profile["real_world_contexts"]),
        "interests": ", ".join(profile["interests"]),
        "learning_needs": ", ".join(profile["learning_needs"]),
        "curriculum_scope": (
            f"Grade {curriculum_scope['grade']} · {curriculum_scope['unit']}"
        ),
        "contextualization_move": (
            f"{strategy['example_style']} Visual approach: {strategy['visual_style']}"
        ),
    }


def summarize_class_context_patterns(
    context_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Summarize common learner-context patterns across a class."""
    context_rows = context_rows or []

    def _count_items(key: str) -> list[str]:
        counts: dict[str, int] = {}
        for row in context_rows:
            for item in _as_list(row.get(key)):
                counts[item] = counts.get(item, 0) + 1
        return [
            item
            for item, _count in sorted(
                counts.items(),
                key=lambda pair: (-pair[1], pair[0].lower()),
            )
        ]

    return {
        "common_contexts": _count_items("preferred_contexts")[:5],
        "common_learning_styles": _count_items("learning_style")[:5],
        "common_learning_needs": _count_items("learning_needs")[:5],
    }
