import json
import os
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import MEMORY_DIR


SKILL_BUCKETS_PATH = MEMORY_DIR / "skill_buckets.json"
SKILL_MATCH_CACHE_PATH = MEMORY_DIR.parent / ".cache" / "skill_bucket_embeddings.npz"
SKILL_MATCH_META_PATH = MEMORY_DIR.parent / ".cache" / "skill_bucket_embeddings.json"

EMBEDDING_MODEL = os.getenv(
    "MATHBRIDGE_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
USE_SKILL_EMBEDDINGS = os.getenv(
    "MATHBRIDGE_USE_SKILL_EMBEDDINGS",
    "1",
).strip().lower() not in {"0", "false", "no"}

SKILL_MATCH_THRESHOLD = 0.32


EXCLUDED_MEMORY_FILES = {
    "demo_users.json",
    "skill_buckets.json",
}


ENGAGEMENT_WEIGHTS = {
    "active": 1.0,
    "engaged": 1.0,
    "persistent": 1.0,
    "moderate": 0.35,
    "steady": 0.35,
    "needs scaffold": -0.6,
    "needs scaffolding": -0.6,
    "frustrated": -0.8,
    "disengaged": -1.0,
    "low": -1.0,
}

ENGAGEMENT_HISTORY_WINDOW = 4


def _clean_list(value) -> list[str]:
    """Normalize comma-separated strings or lists into a clean string list."""
    if value is None:
        return []

    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    cleaned = []
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


def _default_context_profile() -> dict:
    """
    Default learner/course/context profile for the contextualization layer.

    This is intentionally broader than a simple preference form. It mirrors
    the demo goal: adapt tutoring from learner context, curriculum scope,
    real-life contexts, and instructional strategy.
    """
    return {
        "learner": {
            "grade_level": "6",
            "language_preference": "English",
            "reading_level": "Grade 6",
            "math_confidence": "medium",
            "learning_style": [
                "visual",
                "step-by-step",
                "real-life examples",
            ],
        },
        "interests": [
            "basketball",
            "baking",
            "games",
        ],
        "real_world_contexts": [
            "shopping",
            "food",
            "sports",
            "classroom supplies",
        ],
        "learning_needs": [
            "word problems",
            "fractions",
            "unit rates",
        ],
        "curriculum_scope": {
            "grade": "6",
            "unit": "Ratios, rates, and fraction reasoning",
            "current_objective": (
                "Use division reasoning, visual models, and unit-rate "
                "thinking to solve Grade 6 math problems."
            ),
        },
        "contextualization_strategy": {
            "tone": "friendly Socratic tutor",
            "example_style": (
                "Connect abstract math to familiar real-life scenarios when "
                "it naturally supports the problem."
            ),
            "visual_style": "Use diagrams before abstract equations.",
            "avoid": [
                "revealing final answers too early",
                "forced personalization",
                "overly abstract explanations",
            ],
        },
    }


def get_context_profile(student: dict | None) -> dict:
    """
    Return a normalized contextualization profile.

    Existing older JSON files may not contain context_profile yet. This helper
    provides a stable default and also accepts the earlier learner_profile name
    if it exists.
    """
    default = _default_context_profile()

    if not isinstance(student, dict):
        student = {}

    raw = student.get("context_profile")

    # Backward compatibility with an earlier simple preference-layer name.
    if not isinstance(raw, dict):
        raw = student.get("learner_profile")

    if not isinstance(raw, dict):
        raw = {}

    learner = {
        **default["learner"],
        **(raw.get("learner") if isinstance(raw.get("learner"), dict) else {}),
    }

    learner["learning_style"] = _clean_list(
        learner.get("learning_style")
    ) or list(default["learner"]["learning_style"])

    curriculum_scope = {
        **default["curriculum_scope"],
        **(
            raw.get("curriculum_scope")
            if isinstance(raw.get("curriculum_scope"), dict)
            else {}
        ),
    }

    strategy = {
        **default["contextualization_strategy"],
        **(
            raw.get("contextualization_strategy")
            if isinstance(raw.get("contextualization_strategy"), dict)
            else {}
        ),
    }

    strategy["avoid"] = _clean_list(
        strategy.get("avoid")
    ) or list(default["contextualization_strategy"]["avoid"])

    return {
        "learner": learner,
        "interests": _clean_list(
            raw.get("interests", default["interests"])
        ) or list(default["interests"]),
        "real_world_contexts": _clean_list(
            raw.get("real_world_contexts", default["real_world_contexts"])
        ) or list(default["real_world_contexts"]),
        "learning_needs": _clean_list(
            raw.get("learning_needs", default["learning_needs"])
        ) or list(default["learning_needs"]),
        "curriculum_scope": curriculum_scope,
        "contextualization_strategy": strategy,
    }


def update_context_profile(student_id: str, context_profile: dict) -> dict:
    """
    Persist the learner/course/context profile for one student.
    """
    student = load_student(student_id)

    student["context_profile"] = get_context_profile(
        {
            "context_profile": context_profile,
        }
    )

    _write(student_id, student)
    return student



DEFAULT_SKILL_BUCKETS = {
    "Fraction comparison": {
        "description": "Compare fractions and reason about which fraction is larger or smaller.",
        "examples": [
            "compare fractions",
            "which fraction is larger",
            "which fraction is smaller",
            "comparing fractions with different denominators",
            "use common denominators to compare fractions",
        ],
    },
    "Equivalent fractions": {
        "description": "Recognize and generate equivalent fractions.",
        "examples": [
            "equivalent fractions",
            "rewrite fractions with the same value",
            "generate equivalent fractions",
            "rename fractions",
            "same value fractions",
        ],
    },
    "Fraction addition": {
        "description": "Add and subtract fractions using common denominators.",
        "examples": [
            "fraction addition",
            "adding fractions",
            "subtracting fractions",
            "common denominator",
            "least common denominator",
            "least common multiple",
        ],
    },
    "Dividing fractions": {
        "description": "Divide fractions using visual models, measurement division, and reciprocal reasoning.",
        "examples": [
            "dividing fractions",
            "division of fractions",
            "divide fractions",
            "how many unit fraction groups fit inside a fraction",
            "multiply by reciprocal",
            "reciprocal algorithm for fraction division",
            "measurement division with fractions",
            "counting one-eighth pieces in three-fourths",
        ],
    },
    "Ratios": {
        "description": "Understand, simplify, and reason about ratios and equivalent ratios.",
        "examples": [
            "ratio",
            "ratios",
            "equivalent ratios",
            "simplify a ratio",
            "compare two quantities",
            "part to part comparison",
            "ratio table",
        ],
    },
    "Unit rates": {
        "description": "Find rates per one unit, such as miles per hour or price per item.",
        "examples": [
            "unit rate",
            "unit rates",
            "rate per one",
            "miles per hour",
            "price per item",
            "cost per one",
            "distance per hour",
            "quantity per unit",
            "for each",
            "per hour",
        ],
    },
    "Fraction-decimal relationships": {
        "description": "Connect fractions and decimals as equivalent representations.",
        "examples": [
            "fractions and decimals",
            "fraction to decimal",
            "decimal equivalent",
            "convert fraction to decimal",
            "convert decimal to fraction",
            "tenths and hundredths",
            "1/2 equals 0.5",
            "3/4 equals 0.75",
        ],
    },
    "Data and statistics": {
        "description": "Analyze data using measures such as mean, median, mode, and range.",
        "examples": [
            "mode",
            "mean",
            "median",
            "range",
            "average",
            "data set",
            "frequency",
            "most common value",
        ],
    },
    "Word problems": {
        "description": "Represent and solve real-world math problems with equations or models.",
        "examples": [
            "word problem",
            "story problem",
            "real-world problem",
            "write an equation for the situation",
            "model a situation",
            "solve a contextual problem",
        ],
    },
}


def _student_file_path(student_id: str) -> Path:
    return MEMORY_DIR / f"{student_id}.json"


def _memory_json_fingerprint() -> tuple[tuple[str, int, int], ...]:
    """Return a lightweight fingerprint for real student memory files."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    entries = []

    for path in sorted(MEMORY_DIR.glob("*.json")):
        if path.name in EXCLUDED_MEMORY_FILES:
            continue

        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        entries.append(
            (
                path.name,
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        )

    return tuple(entries)


@lru_cache(maxsize=128)
def _load_student_raw_cached(
    student_id: str,
    mtime_ns: int,
    size: int,
) -> str:
    """Read a student JSON file only when its mtime/size changes."""
    path = _student_file_path(student_id)
    return path.read_text(encoding="utf-8")


def load_student(student_id: str) -> dict:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _student_file_path(student_id)

    if not path.exists():
        data = _default(student_id)
        _write(student_id, data)
        return data

    stat = path.stat()
    raw = _load_student_raw_cached(
        student_id,
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )

    return json.loads(raw)


@lru_cache(maxsize=16)
def _list_student_ids_cached(
    fingerprint: tuple[tuple[str, int, int], ...],
) -> tuple[str, ...]:
    """Return real student ids without re-reading unchanged JSON files."""
    student_ids = []

    for filename, _mtime_ns, _size in fingerprint:
        path = MEMORY_DIR / filename

        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                continue

            if (
                "student_id" in data
                or "sessions" in data
                or "mastery" in data
                or "current_signals" in data
            ):
                student_ids.append(path.stem)

        except Exception:
            continue

    return tuple(sorted(student_ids))


def list_student_ids():
    """
    Return only real student memory files.

    The directory fingerprint keeps Streamlit reruns from reopening every
    unchanged student JSON file. Writes clear the cache immediately.
    """
    return list(
        _list_student_ids_cached(
            _memory_json_fingerprint()
        )
    )


def save_session(student_id: str, session_data: dict):
    student = load_student(student_id)

    sessions = student.setdefault("sessions", [])
    sessions.append(
        {
            "date": str(date.today()),
            **session_data,
        }
    )

    _write(student_id, student)


def update_signals(student_id: str, signals: dict):
    student = load_student(student_id)

    stabilized_signals = _stabilize_engagement_signals(
        student=student,
        signals=signals,
    )

    student["current_signals"] = stabilized_signals

    history = student.setdefault("signals_history", [])
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **stabilized_signals,
        }
    )

    _write(student_id, student)


def update_weekly_summary(student_id: str):
    """Generate a parent-facing summary from the latest saved signals."""
    student = load_student(student_id)
    signals = student.get("current_signals") or {}

    if not signals:
        return

    name = student.get("name", student_id.capitalize())

    concept = (
        signals.get("concept")
        or student.get("current_topic")
        or "the current math topic"
    )

    misconception = signals.get("misconception")
    next_support = signals.get("next_support")
    engagement = str(signals.get("engagement", "")).lower()

    if "active" in engagement or "engaged" in engagement:
        what_went_well = (
            f"{name} stayed engaged while practicing {concept} "
            "and continued working through the problem with support."
        )
    else:
        what_went_well = (
            f"{name} practiced {concept} and used tutoring support "
            "to work through the problem."
        )

    if misconception:
        needs_support = (
            f"Based on this session, {name} may need more practice with: "
            f"{misconception}."
        )
    else:
        needs_support = (
            f"No clear misconception was detected in the latest interaction. "
            f"Continue checking understanding of {concept} with more examples."
        )

    try_at_home = (
        next_support
        or f"Ask {name} to explain each step aloud while practicing {concept}."
    )

    student["weekly_summary"] = {
        "topics": concept,
        "what_went_well": what_went_well,
        "needs_support": needs_support,
        "try_at_home": try_at_home,
        "encouragement": (
            f"Great persistence, {name}! Keep encouraging careful reasoning "
            "and step-by-step explanations. This summary is based on recent "
            "tutoring interactions, not a formal assessment."
        ),
    }

    _write(student_id, student)


def update_mastery(
    student_id: str,
    signals: dict | None = None,
) -> dict:
    """
    Update one mastery skill after a completed tutoring session.

    This is a preliminary heuristic estimate, not a formal assessment.
    """
    student = load_student(student_id)

    if signals is None:
        signals = student.get("current_signals") or {}
    else:
        signals = _stabilize_engagement_signals(
            student=student,
            signals=signals,
        )

    concept = str(signals.get("concept") or "").strip()
    mastery = student.setdefault("mastery", {})

    if not concept:
        return {
            "updated": False,
            "reason": "No concept was recorded.",
        }

    skill, match_info = _match_mastery_skill(concept, mastery)

    if skill is None:
        return {
            "updated": False,
            "reason": f"No mastery category matched concept: {concept}",
            "match_info": match_info,
        }

    # If the skill bucket exists but this student's JSON does not yet have it,
    # initialize it conservatively at 50.
    old_score = int(mastery.get(skill, 50))

    if skill not in mastery:
        mastery[skill] = old_score

    delta, evidence = _calculate_mastery_delta(signals)

    if match_info:
        method = match_info.get("method", "unknown")
        confidence = match_info.get("confidence")
        if confidence is not None:
            evidence.append(
                f"Skill matched by {method} with confidence {confidence:.2f}."
            )
        else:
            evidence.append(
                f"Skill matched by {method}."
            )

    new_score = round(max(0.0, min(100.0, old_score + delta)), 1)
    if float(new_score).is_integer():
        new_score = int(new_score)
    mastery[skill] = new_score

    history = student.setdefault("mastery_history", [])
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "concept": concept,
            "skill": skill,
            "old_score": old_score,
            "delta": delta,
            "new_score": new_score,
            "evidence": evidence,
            "match_info": match_info,
        }
    )

    _write(student_id, student)

    return {
        "updated": True,
        "concept": concept,
        "skill": skill,
        "old_score": old_score,
        "delta": delta,
        "new_score": new_score,
        "evidence": evidence,
        "match_info": match_info,
    }


@lru_cache(maxsize=8)
def _load_skill_buckets_cached(
    mtime_ns: int,
    size: int,
) -> dict:
    """Load and normalize skill buckets only when the file changes."""
    try:
        loaded = json.loads(
            SKILL_BUCKETS_PATH.read_text(encoding="utf-8")
        )

        if isinstance(loaded, dict) and loaded:
            return _normalize_skill_buckets(loaded)

    except Exception:
        pass

    return DEFAULT_SKILL_BUCKETS


def _load_skill_buckets() -> dict:
    """
    Load skill buckets from data/memory/skill_buckets.json.

    If the file is missing or malformed, use a built-in default so the app
    remains stable during demos. The normalized result is cached by file
    mtime/size so update_mastery does not re-read the same config on each run.
    """

    if SKILL_BUCKETS_PATH.exists():
        try:
            stat = SKILL_BUCKETS_PATH.stat()
            return _load_skill_buckets_cached(
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except Exception:
            pass

    return DEFAULT_SKILL_BUCKETS


def _normalize_skill_buckets(raw: dict) -> dict:
    normalized = {}

    for skill, value in raw.items():
        skill_name = str(skill).strip()
        if not skill_name:
            continue

        if isinstance(value, dict):
            description = str(value.get("description", "")).strip()
            examples = value.get("examples", [])
        elif isinstance(value, list):
            description = ""
            examples = value
        else:
            description = str(value)
            examples = []

        clean_examples = [
            str(example).strip()
            for example in examples
            if str(example).strip()
        ]

        normalized[skill_name] = {
            "description": description,
            "examples": clean_examples,
        }

    return normalized or DEFAULT_SKILL_BUCKETS


def _bucket_text(skill: str, bucket: dict) -> str:
    examples = bucket.get("examples", [])
    description = bucket.get("description", "")

    return "\n".join(
        [
            f"Skill: {skill}",
            f"Description: {description}",
            "Examples:",
            *[f"- {example}" for example in examples],
        ]
    )


def _match_mastery_skill(
    concept: str,
    mastery: dict,
) -> tuple[str | None, dict]:
    """
    Map the LLM concept description to a mastery skill bucket.

    Preferred path:
    - Use embeddings to compare the Diagnostician Agent's concept string to
      skill bucket descriptions/examples.

    Safe fallback:
    - Use lightweight token/fuzzy overlap if embeddings are unavailable.
    """

    skill_buckets = _load_skill_buckets()

    if not skill_buckets:
        return None, {
            "method": "none",
            "reason": "No skill buckets are available.",
        }

    embedding_match = _match_skill_by_embedding(
        concept=concept,
        skill_buckets=skill_buckets,
    )

    if embedding_match is not None:
        skill, confidence = embedding_match
        return skill, {
            "method": "embedding",
            "confidence": confidence,
        }

    fallback_match = _match_skill_by_fallback(
        concept=concept,
        skill_buckets=skill_buckets,
        mastery=mastery,
    )

    if fallback_match is not None:
        skill, confidence, method = fallback_match
        return skill, {
            "method": method,
            "confidence": confidence,
        }

    return None, {
        "method": "fallback",
        "reason": "No skill bucket was similar enough.",
    }


def _match_skill_by_embedding(
    concept: str,
    skill_buckets: dict,
) -> tuple[str, float] | None:
    if not USE_SKILL_EMBEDDINGS:
        return None

    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        skills = list(skill_buckets.keys())
        bucket_texts = [
            _bucket_text(skill, skill_buckets[skill])
            for skill in skills
        ]

        bucket_embeddings = _load_or_build_skill_embeddings(
            skills=skills,
            bucket_texts=bucket_texts,
        )

        if bucket_embeddings is None:
            return None

        embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        query_text = f"Student concept: {concept}"
        query_embedding = np.array(
            embedder.embed_query(query_text),
            dtype="float32",
        )

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            return None

        query_embedding = query_embedding / query_norm
        scores = bucket_embeddings @ query_embedding

        best_index = int(np.argmax(scores))
        confidence = float(scores[best_index])

        if confidence < SKILL_MATCH_THRESHOLD:
            return None

        return skills[best_index], confidence

    except Exception:
        return None


def _load_or_build_skill_embeddings(
    skills: list[str],
    bucket_texts: list[str],
) -> np.ndarray | None:
    MEMORY_DIR.parent.joinpath(".cache").mkdir(parents=True, exist_ok=True)

    fingerprint_payload = {
        "model": EMBEDDING_MODEL,
        "skills": skills,
        "bucket_texts": bucket_texts,
    }
    fingerprint = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    if SKILL_MATCH_CACHE_PATH.exists() and SKILL_MATCH_META_PATH.exists():
        try:
            meta = json.loads(
                SKILL_MATCH_META_PATH.read_text(encoding="utf-8")
            )

            if meta.get("fingerprint") == fingerprint:
                data = np.load(SKILL_MATCH_CACHE_PATH)
                embeddings = data["embeddings"].astype("float32")
                return _normalize_matrix(embeddings)
        except Exception:
            pass

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    embeddings = np.array(
        embedder.embed_documents(bucket_texts),
        dtype="float32",
    )

    np.savez_compressed(
        SKILL_MATCH_CACHE_PATH,
        embeddings=embeddings,
    )

    SKILL_MATCH_META_PATH.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "model": EMBEDDING_MODEL,
                "skills": skills,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return _normalize_matrix(embeddings)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )
    norms[norms == 0] = 1.0
    return matrix / norms


def _match_skill_by_fallback(
    concept: str,
    skill_buckets: dict,
    mastery: dict,
) -> tuple[str, float, str] | None:
    concept_tokens = set(_tokenize_skill_text(concept))

    if not concept_tokens:
        return None

    best_skill = None
    best_score = 0.0

    for skill, bucket in skill_buckets.items():
        bucket_tokens = set(
            _tokenize_skill_text(
                _bucket_text(skill, bucket)
            )
        )

        if not bucket_tokens:
            continue

        overlap = len(concept_tokens & bucket_tokens)
        coverage = overlap / max(1, len(concept_tokens))
        jaccard = overlap / max(1, len(concept_tokens | bucket_tokens))
        fuzzy = SequenceMatcher(
            None,
            concept.lower(),
            skill.lower(),
        ).ratio()

        # Prefer exact existing mastery label overlap when available, but still
        # allow new Grade 6 buckets like Unit rates to be initialized.
        label_bonus = 0.15 if skill in mastery else 0.0

        score = (0.65 * coverage) + (0.25 * fuzzy) + (0.10 * jaccard) + label_bonus

        if score > best_score:
            best_score = score
            best_skill = skill

    if best_skill is None:
        return None

    # Lower threshold than embedding because this is fallback and the buckets
    # include explicit examples.
    if best_score < 0.42:
        return None

    return best_skill, float(best_score), "fallback_token_fuzzy"


def _tokenize_skill_text(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("÷", " divide ")
    text = text.replace("×", " times ")
    text = text.replace("-", " ")

    raw_tokens = re.findall(r"\d+/\d+|[a-zA-Z0-9]+", text)

    stop_words = {
        "a", "an", "the", "is", "are", "am", "to", "of", "in", "on",
        "at", "for", "from", "by", "with", "and", "or", "but", "as",
        "such", "using", "use", "student", "concept", "skill",
    }

    aliases = {
        "fractions": "fraction",
        "dividing": "divide",
        "division": "divide",
        "divided": "divide",
        "ratios": "ratio",
        "rates": "rate",
        "quarters": "quarter",
        "eighths": "eighth",
        "halves": "half",
        "miles": "mile",
        "hours": "hour",
        "decimals": "decimal",
        "problems": "problem",
        "groups": "group",
        "comparing": "compare",
        "comparison": "compare",
        "equivalent": "equivalent",
        "average": "mean",
    }

    tokens = []

    for token in raw_tokens:
        token = token.strip().lower()
        token = aliases.get(token, token)

        if token in stop_words:
            continue

        if len(token) > 4 and token.endswith("ing"):
            token = token[:-3]

        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]

        token = aliases.get(token, token)

        if len(token) < 2 and not token.isdigit():
            continue

        tokens.append(token)

    return tokens


def _canonical_engagement_label(label: str | None) -> str:
    text = str(label or "").strip().lower()

    if not text:
        return "Moderate"

    if any(token in text for token in ("active", "engaged", "persistent")):
        return "Active"

    if any(
        token in text
        for token in (
            "low",
            "needs scaffold",
            "needs scaffolding",
            "frustrated",
            "disengaged",
        )
    ):
        return "Low"

    if "moderate" in text or "steady" in text:
        return "Moderate"

    return "Moderate"


def _engagement_score(label: str | None) -> float:
    canonical = _canonical_engagement_label(label).lower()
    return ENGAGEMENT_WEIGHTS.get(canonical, 0.0)


def _engagement_label_from_score(score: float) -> str:
    if score >= 0.55:
        return "Active"
    if score <= -0.45:
        return "Low"
    return "Moderate"


def _recent_engagement_labels(student: dict) -> list[str]:
    history = student.get("signals_history") or []

    labels: list[str] = []
    for record in history[-ENGAGEMENT_HISTORY_WINDOW:]:
        if not isinstance(record, dict):
            continue
        label = record.get("raw_engagement") or record.get("engagement")
        if label:
            labels.append(_canonical_engagement_label(label))

    return labels


def _smoothed_engagement(
    current_label: str | None,
    recent_labels: list[str] | None = None,
) -> tuple[str, float]:
    labels = [
        _canonical_engagement_label(label)
        for label in list(recent_labels or [])[-ENGAGEMENT_HISTORY_WINDOW:]
    ]
    labels.append(_canonical_engagement_label(current_label))

    weights = [0.15, 0.20, 0.25, 0.40]
    weights = weights[-len(labels):]

    total_weight = sum(weights) or 1.0
    score = sum(
        _engagement_score(label) * weight
        for label, weight in zip(labels, weights)
    ) / total_weight

    return _engagement_label_from_score(score), round(float(score), 3)


def _stabilize_engagement_signals(
    student: dict,
    signals: dict | None,
) -> dict:
    stabilized = dict(signals or {})

    raw_engagement = stabilized.get("engagement") or "Moderate"
    recent_labels = _recent_engagement_labels(student)
    smoothed_label, smoothed_score = _smoothed_engagement(
        current_label=raw_engagement,
        recent_labels=recent_labels,
    )

    stabilized["raw_engagement"] = _canonical_engagement_label(raw_engagement)
    stabilized["engagement"] = smoothed_label
    stabilized["engagement_score"] = smoothed_score

    return stabilized


def _calculate_mastery_delta(
    signals: dict,
) -> tuple[float, list[str]]:
    """
    Produce a conservative score change from -3 to +3.

    Engagement is treated as a smoothed score rather than a single hard label,
    so Moderate has a small effect and one noisy turn does not swing mastery.
    """
    delta = 0.0
    evidence = []

    misconception = signals.get("misconception")
    has_misconception = _has_meaningful_misconception(misconception)

    if has_misconception:
        delta -= 2.0
        evidence.append("A possible misconception was detected.")
    else:
        delta += 2.0
        evidence.append("No clear misconception was detected.")

    hint_count = _extract_hint_count(signals.get("hints_used"))

    if hint_count == 0:
        delta += 1.0
        evidence.append("The student completed the interaction without hints.")
    elif hint_count is not None and hint_count >= 3:
        delta -= 1.0
        evidence.append(
            f"The student requested substantial support ({hint_count} hints)."
        )
    elif hint_count is not None:
        evidence.append(
            f"The student used {hint_count} hint(s)."
        )

    try:
        engagement_score = float(
            signals.get("engagement_score")
            if signals.get("engagement_score") is not None
            else _engagement_score(signals.get("engagement"))
        )
    except (TypeError, ValueError):
        engagement_score = _engagement_score(signals.get("engagement"))

    engagement_label = _engagement_label_from_score(engagement_score)

    if engagement_score >= 0.55:
        delta += 1.0
        evidence.append("The student remained actively engaged.")
    elif engagement_score >= 0.20:
        delta += 0.5
        evidence.append("The student showed steady moderate engagement.")
    elif engagement_score <= -0.45:
        delta -= 1.0
        evidence.append("The student may need additional support.")
    else:
        evidence.append("Engagement was neutral in this session.")

    evidence.append(
        f"Engagement was smoothed as {engagement_label} "
        f"(score {engagement_score:.2f})."
    )

    delta = round(max(-3.0, min(3.0, delta)), 1)

    if float(delta).is_integer():
        delta = int(delta)

    return delta, evidence


def _extract_hint_count(value) -> int | None:
    if value is None:
        return None

    match = re.search(r"\d+", str(value))

    if not match:
        return None

    return int(match.group())


def _has_meaningful_misconception(value) -> bool:
    if value is None:
        return False

    text = str(value).strip().lower()

    if not text:
        return False

    no_misconception_values = {
        "none",
        "null",
        "n/a",
        "no",
        "—",
        "no misconception",
        "none detected",
        "no clear misconception detected",
        "no clear misconception detected.",
    }

    if text in no_misconception_values:
        return False

    if text.startswith("no clear misconception"):
        return False

    return True


def _clear_memory_caches():
    _load_student_raw_cached.cache_clear()
    _list_student_ids_cached.cache_clear()
    _load_skill_buckets_cached.cache_clear()


def _write(student_id: str, data: dict):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _student_file_path(student_id)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    _clear_memory_caches()


def _default(student_id: str) -> dict:
    return {
        "student_id": student_id,
        "name": student_id.capitalize(),
        "current_topic": "Fractions",
        "context_profile": _default_context_profile(),
        "sessions": [],
        "mastery_history": [],
        "current_signals": {},
        "signals_history": [],
        "mastery": {
            "Fraction comparison": 85,
            "Equivalent fractions": 48,
            "Fraction addition": 35,
            "Dividing fractions": 50,
            "Ratios": 50,
            "Unit rates": 50,
            "Fraction-decimal relationships": 50,
            "Data and statistics": 50,
            "Word problems": 58,
        },
        "weekly_summary": {
            "topics": "fractions: comparison and addition",
            "what_went_well": (
                "Improved at identifying which fraction is larger "
                "using drawings."
            ),
            "needs_support": (
                "Still needs practice understanding equivalent fractions."
            ),
            "try_at_home": (
                "Use pizza slices, drawings, or measuring cups "
                "to compare equal parts."
            ),
            "encouragement": (
                "Great persistence this week! Encourage your child "
                "to explain their thinking out loud."
            ),
        },
        "insights": [
            {
                "title": "Misconception Alert",
                "body": (
                    "Student may be adding numerators and denominators "
                    "directly when adding fractions."
                ),
            },
            {
                "title": "Instructional Suggestion",
                "body": (
                    "Use visual area models before moving to symbolic "
                    "fraction addition."
                ),
            },
            {
                "title": "Grouping Suggestion",
                "body": (
                    "Consider small-group practice with students showing "
                    "similar misconception patterns."
                ),
            },
        ],
    }
