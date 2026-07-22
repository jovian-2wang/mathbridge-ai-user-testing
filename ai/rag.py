from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import OrderedDict
import numpy as np
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_DIR = PROJECT_ROOT / "data" / "curriculum"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"

TOP_K = 3
MAX_EXCERPT_CHARS = 900
MAX_CONTEXT_CHARS = 3500
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 250

EMBEDDING_MODEL = os.getenv("MATHBRIDGE_EMBEDDING_MODEL", "text-embedding-3-small")
USE_EMBEDDINGS = os.getenv("MATHBRIDGE_USE_EMBEDDINGS", "1").strip().lower() not in {"0", "false", "no"}

BM25_WEIGHT = 0.25
DENSE_WEIGHT = 0.75
WEAK_DENSE_THRESHOLD = 0.28

_QUERY_EMBEDDING_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_MAX_QUERY_EMBEDDING_CACHE_SIZE = 256
_INTENT_PROTOTYPE_CENTROID_CACHE: dict[str, dict[str, np.ndarray]] = {}

# Intent detection now uses query embeddings against short intent prototypes
# instead of maintaining separate regex classifiers in the RAG path.
_INTENT_SIMILARITY_THRESHOLD = 0.31
_INTENT_RELATIVE_MARGIN = 0.045

_INTENT_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "equal_sharing": (
        "A total amount is shared equally into a fixed number of groups.",
        "Find how much is in each group after splitting equally.",
        "Divide objects among people, boxes, bags, or groups equally.",
        "Partitive division where the number of groups is known and the amount per group is unknown.",
    ),
    "measurement_division": (
        "Find how many groups of a given size fit into a total.",
        "Find how many times the divisor or group size fits in the dividend.",
        "Measurement division asking how many groups can be made.",
        "Count the number of groups when the size of each group is known.",
    ),
    "unit_rate": (
        "Find a rate per one unit.",
        "Find miles per hour, cost per item, dollars per pound, or amount for each one.",
        "Compare two quantities as a rate and scale to one unit.",
        "Unit rate problem involving per, for each, speed, or price per item.",
    ),
    "fractions": (
        "Reason about fractions, unit fractions, equivalent fractions, or fraction operations.",
        "Add, subtract, multiply, divide, or compare fractions.",
        "Use numerators, denominators, halves, thirds, fourths, quarters, or eighths.",
        "Understand fraction values and fraction diagrams.",
    ),
    "geometry": (
        "Reason about geometric measurement such as area, perimeter, volume, surface area, radius, diameter, or angles.",
        "Use shapes such as rectangles, triangles, circles, prisms, cubes, or polygons.",
        "Find length, width, height, side length, edge length, radius, diameter, area, perimeter, or volume.",
    ),
    "coordinate_plane": (
        "Read or plot an ordered pair on the coordinate plane.",
        "Identify x-coordinate, y-coordinate, quadrant, origin, x-axis, or y-axis.",
        "Determine the quadrant of a point using signs of x and y coordinates.",
    ),
    "whole_number_division": (
        "Interpret or compute division of whole numbers.",
        "Use dividend, divisor, quotient, remainder, groups, or sharing with whole numbers.",
        "Represent whole-number division using equal groups or tape diagrams.",
    ),
}

STOP_WORDS = {
    "a", "an", "the",
    "i", "me", "my", "you", "your", "we", "our",
    "is", "are", "am", "be", "being", "been",
    "do", "does", "did",
    "can", "could", "would", "should",
    "how", "what", "why", "when", "where",
    "to", "of", "in", "on", "at", "for", "from",
    "by", "with", "and", "or", "but",
    "this", "that", "these", "those",
    "there", "here",
    "using", "use",
    "find", "show", "explain",
}

ALIASES = {
    "divided": "divide",
    "dividing": "divide",
    "division": "divide",
    "divide": "divide",
    "fractions": "fraction",
    "fractional": "fraction",
    "groups": "group",
    "grouping": "group",
    "meanings": "mean",
    "meaning": "mean",
    "means": "mean",
    "diagrams": "diagram",
    "rectangles": "rectangle",
    "triangles": "triangle",
    "prisms": "prism",
    "lengths": "length",
    "sides": "side",
    "boxes": "box",
    "fitting": "fit",
    "problems": "problem",
    "situations": "situation",
    "quotients": "quotient",
    "divisors": "divisor",
    "dividends": "dividend",
    "quarters": "quarter",
    "eighths": "eighth",
    "halves": "half",
    "cups": "cup",
    "rates": "rate",
    "miles": "mile",
    "hours": "hour",
    "prices": "price",
    "costs": "cost",
}


FRACTION_WORD_EXPANSIONS = {
    "half": "1/2 one half",
    "halves": "1/2 one half",
    "third": "1/3 one third",
    "thirds": "1/3 one third",
    "quarter": "1/4 one fourth",
    "quarters": "1/4 one fourth",
    "fourth": "1/4 one fourth",
    "fourths": "1/4 one fourth",
    "eighth": "1/8 one eighth",
    "eighths": "1/8 one eighth",
    "tenth": "1/10 one tenth",
    "tenths": "1/10 one tenth",
}


def _expand_query_for_retrieval(query: str) -> str:
    """
    Add light math-aware retrieval hints without changing the student's wording.

    This does not solve the problem. It only rewrites common Grade 6 wording
    into curriculum-search language so embedding retrieval can find better
    lesson chunks.
    """

    normalized = _normalize_text(query)
    expansions = []

    for word, expansion in FRACTION_WORD_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            expansions.append(expansion)

    # Measurement division phrasing:
    # "How many quarter cups are in 3/4 cup?" means 3/4 ÷ 1/4.
    if re.search(r"\bhow many\b", normalized) and re.search(
        r"\bin\b|\bfit\b|\binside\b",
        normalized,
    ):
        expansions.append(
            "fraction division measurement division count groups how many groups fit inside"
        )

    if re.search(r"\bfit\b|\bfits\b|\bfitting\b|\binside\b", normalized):
        expansions.append(
            "count groups group size divisor dividend quotient"
        )

    # Equal-sharing / equal-groups whole-number division:
    # "180 apples divided into 4 boxes" means how many in each group.
    equal_group_pattern = (
        re.search(
            r"\b(divide|divided|split|share|shared|put)\b.*\b(into|among|between|across)\b",
            normalized,
        )
        or re.search(
            r"\b(each box|each group|per box|per group|equal groups|equally)\b",
            normalized,
        )
    )

    object_group_pattern = (
        re.search(
            r"\b(apples?|oranges?|items?|students?|people|friends?|boxes?|bags?|groups?)\b",
            normalized,
        )
        and re.search(
            r"\b(divide|divided|split|share|shared|into|among|each)\b",
            normalized,
        )
    )

    if equal_group_pattern or object_group_pattern:
        expansions.append(
            "division situation equal groups equal sharing how many in each group amount per group quotient"
        )
        expansions.append(
            "interpret division word problem write a division equation"
        )

    # Reciprocal algorithm phrasing.
    if "reciprocal" in normalized or re.search(r"\bflip\b|\bflipping\b", normalized):
        expansions.append(
            "fraction division algorithm multiply by reciprocal inverse divide by fraction"
        )

    # Unit-rate phrasing.
    if re.search(r"\bunit rate\b|\bper\b|\bfor each\b|\beach\b", normalized):
        expansions.append("unit rate per one for each equivalent rate")

    if re.search(r"\bmile\b|\bmiles\b", normalized) and re.search(
        r"\bhour\b|\bhours\b",
        normalized,
    ):
        expansions.append("unit rate miles per hour speed rate")

    if re.search(r"\bcost\b|\bprice\b|\bdollar\b|\bdollars\b", normalized):
        expansions.append("unit rate price per item cost per one")

    if not expansions:
        return query

    return f"{query}\n\nRetrieval hints: {' '.join(expansions)}"


@dataclass(frozen=True)
class CurriculumChunk:
    chunk_id: str
    source: str
    text: str
    searchable_text: str


_INDEX_STATE: dict[str, Any] = {
    "fingerprint": None,
    "chunks": [],
    "bm25": None,
    "dense_embeddings": None,
    "dense_available": False,
    "dense_error": "",
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("÷", " divide ")
    text = text.replace("×", " times ")
    text = text.replace("·", " times ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def _stem_token(token: str) -> str:
    token = token.lower().strip()

    if token in ALIASES:
        return ALIASES[token]

    if len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
        if len(token) >= 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]

    return ALIASES.get(token, token)


def _tokenize(text: str) -> list[str]:
    text = _normalize_text(text)
    raw_tokens = re.findall(r"\d+/\d+|[a-zA-Z0-9]+", text)

    tokens = []
    for token in raw_tokens:
        token = _stem_token(token)
        if token in STOP_WORDS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.append(token)

    return tokens


def _curriculum_fingerprint() -> str:
    if not CURRICULUM_DIR.exists():
        return "missing"

    pieces = []
    for path in sorted(CURRICULUM_DIR.rglob("*.txt")):
        stat = path.stat()
        pieces.append({
            "name": str(path.relative_to(CURRICULUM_DIR)),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })

    payload = json.dumps(pieces, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_into_chunks(text: str) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", text).strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        if len(current) + 2 + len(paragraph) <= CHUNK_MAX_CHARS:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            overlap = current[-CHUNK_OVERLAP_CHARS:]
            current = f"{overlap}\n\n{paragraph}"

    if current:
        chunks.append(current)

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_MAX_CHARS + CHUNK_OVERLAP_CHARS:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            end = min(len(chunk), start + CHUNK_MAX_CHARS)
            final_chunks.append(chunk[start:end].strip())
            if end == len(chunk):
                break
            start = max(0, end - CHUNK_OVERLAP_CHARS)

    return [chunk for chunk in final_chunks if chunk.strip()]


def _load_curriculum_chunks() -> list[CurriculumChunk]:
    if not CURRICULUM_DIR.exists():
        return []

    chunks = []
    for path in sorted(CURRICULUM_DIR.rglob("*.txt")):
        relative_source = str(path.relative_to(CURRICULUM_DIR))

        document_text = path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk_text in enumerate(_split_into_chunks(document_text)):
            chunk_id = f"{relative_source}::chunk_{index:03d}"
            searchable_title = relative_source.replace("/", " ").replace("_", " ")

            searchable_text = f"{searchable_title}\n\n{chunk_text}"

            chunks.append(CurriculumChunk(
                chunk_id=chunk_id,
                source=relative_source,
                text=chunk_text,
                searchable_text=searchable_text,
            ))

    return chunks


def _embedding_cache_paths(fingerprint: str) -> tuple[Path, Path]:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", EMBEDDING_MODEL)
    stem = f"curriculum_embeddings_{safe_model}_{fingerprint[:12]}"
    return CACHE_DIR / f"{stem}.npz", CACHE_DIR / f"{stem}.json"


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _load_or_build_dense_embeddings(
    chunks: list[CurriculumChunk],
    fingerprint: str,
) -> tuple[np.ndarray | None, str]:
    if not USE_EMBEDDINGS:
        return None, "Embeddings disabled by MATHBRIDGE_USE_EMBEDDINGS."

    if not os.getenv("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY is not set; using BM25 fallback."

    if not chunks:
        return None, "No curriculum chunks found."

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _embedding_cache_paths(fingerprint)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if npz_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("fingerprint") == fingerprint
                and meta.get("model") == EMBEDDING_MODEL
                and meta.get("chunk_ids") == chunk_ids
            ):
                data = np.load(npz_path)
                embeddings = data["embeddings"].astype("float32")
                return _normalize_matrix(embeddings), ""
        except Exception:
            pass

    try:
        embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        texts = [chunk.searchable_text[:6000] for chunk in chunks]
        embeddings = np.array(embedder.embed_documents(texts), dtype="float32")

        meta_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": EMBEDDING_MODEL,
                    "chunk_ids": chunk_ids,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        np.savez_compressed(npz_path, embeddings=embeddings)

        return _normalize_matrix(embeddings), ""

    except Exception as exc:
        return None, f"Embedding retrieval failed; using BM25 fallback. Error: {exc}"


def _ensure_index() -> None:
    fingerprint = _curriculum_fingerprint()

    if _INDEX_STATE["fingerprint"] == fingerprint:
        return

    chunks = _load_curriculum_chunks()
    tokenized_corpus = [_tokenize(chunk.searchable_text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    dense_embeddings, dense_error = _load_or_build_dense_embeddings(
        chunks=chunks,
        fingerprint=fingerprint,
    )

    _INDEX_STATE.update({
        "fingerprint": fingerprint,
        "chunks": chunks,
        "bm25": bm25,
        "dense_embeddings": dense_embeddings,
        "dense_available": dense_embeddings is not None,
        "dense_error": dense_error,
    })


def _rank_by_scores(scores: np.ndarray) -> list[int]:
    if scores.size == 0:
        return []

    ranked = np.argsort(-scores)
    return [int(index) for index in ranked if float(scores[index]) > 0]



def _normalize_query_cache_key(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _cached_embed_query(embeddings, query: str) -> list[float]:
    """
    Cache query/prototype embeddings during the Streamlit server process.

    This avoids calling the embedding API repeatedly for the same query during
    reruns, repeated hints, or repeated grounding retrieval.
    """
    key = _normalize_query_cache_key(query)

    if not key:
        return []

    if key in _QUERY_EMBEDDING_CACHE:
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        return _QUERY_EMBEDDING_CACHE[key]

    vector = embeddings.embed_query(query)

    _QUERY_EMBEDDING_CACHE[key] = vector
    _QUERY_EMBEDDING_CACHE.move_to_end(key)

    if len(_QUERY_EMBEDDING_CACHE) > _MAX_QUERY_EMBEDDING_CACHE_SIZE:
        _QUERY_EMBEDDING_CACHE.popitem(last=False)

    return vector


def _as_unit_vector(values: list[float] | np.ndarray) -> np.ndarray | None:
    vector = np.array(values, dtype="float32")
    norm = float(np.linalg.norm(vector))

    if norm == 0.0:
        return None

    return vector / norm


def _get_intent_prototype_centroids(embedder) -> dict[str, np.ndarray]:
    """Build one normalized centroid vector per intent, cached by model."""
    cache_key = EMBEDDING_MODEL

    if cache_key in _INTENT_PROTOTYPE_CENTROID_CACHE:
        return _INTENT_PROTOTYPE_CENTROID_CACHE[cache_key]

    prototype_inputs: list[str] = []
    prototype_intents: list[str] = []

    for intent, prototypes in _INTENT_PROTOTYPES.items():
        for prototype in prototypes:
            prototype_inputs.append(
                f"math problem intent {intent}: {prototype}"
            )
            prototype_intents.append(intent)

    if not prototype_inputs:
        return {}

    vectors = np.array(
        embedder.embed_documents(prototype_inputs),
        dtype="float32",
    )

    grouped: dict[str, list[np.ndarray]] = {}
    for intent, vector in zip(prototype_intents, vectors):
        unit_vector = _as_unit_vector(vector)
        if unit_vector is not None:
            grouped.setdefault(intent, []).append(unit_vector)

    centroids: dict[str, np.ndarray] = {}
    for intent, intent_vectors in grouped.items():
        centroid = np.mean(
            np.stack(intent_vectors, axis=0),
            axis=0,
        )
        centroid_unit = _as_unit_vector(centroid)
        if centroid_unit is not None:
            centroids[intent] = centroid_unit

    _INTENT_PROTOTYPE_CENTROID_CACHE[cache_key] = centroids
    return centroids


def _embedding_intent_scores(query: str) -> dict[str, float]:
    """
    Score the query against embedded intent prototypes.

    This replaces the old RAG-side hand-written intent regex with a lightweight
    embedding prototype classifier. It only affects curriculum reranking; tutor
    behavior, answer evaluation, and visual generation are not changed.
    """
    if not USE_EMBEDDINGS or not os.getenv("OPENAI_API_KEY"):
        return {}

    query = str(query or "").strip()
    if not query:
        return {}

    try:
        embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        query_vector = _as_unit_vector(
            _cached_embed_query(embedder, query)
        )
        if query_vector is None:
            return {}

        prototype_centroids = _get_intent_prototype_centroids(embedder)
        if not prototype_centroids:
            return {}

        return {
            intent: float(np.dot(query_vector, centroid))
            for intent, centroid in prototype_centroids.items()
        }

    except Exception as exc:
        _INDEX_STATE["dense_error"] = (
            "Intent prototype embedding failed; using lexical safety fallback. "
            f"Error: {exc}"
        )
        return {}


def _lexical_intent_fallback(query: str) -> set[str]:
    """Fallback only for offline / embedding-unavailable runs."""
    text = _normalize_text(query)
    intents: set[str] = set()

    if re.search(
        r"\b(divide|divided|split|share|shared|put)\b.*\b(into|among|between|across)\b",
        text,
    ) or re.search(
        r"\b(each box|each group|per box|per group|in each|how many.*each)\b",
        text,
    ):
        intents.add("equal_sharing")

    if re.search(
        r"\bhow many groups\b|\bhow many.*fit\b|\bhow many.*in\b|\bgroups of\b|\bsize of each group\b",
        text,
    ):
        intents.add("measurement_division")

    if re.search(r"\bunit rate\b|\bper\b|\bfor each\b", text):
        intents.add("unit_rate")

    if re.search(r"\d+/\d+|\bfraction\b|\bhalf\b|\bthird\b|\bquarter\b|\beighth\b", text):
        intents.add("fractions")

    if re.search(r"\bvolume|prism|area|surface|edge|length|width|height|radius|diameter|perimeter\b", text):
        intents.add("geometry")

    if re.search(r"\bcoordinate|coordinates|point\s*\(|x\s*coordinate|y\s*coordinate|quadrant|axis\b", text):
        intents.add("coordinate_plane")

    return intents

def _dense_scores(query: str) -> np.ndarray | None:
    dense_embeddings = _INDEX_STATE.get("dense_embeddings")
    if dense_embeddings is None:
        return None

    try:
        embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        query_embedding = np.array(_cached_embed_query(embedder, query), dtype="float32")
        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            return None

        query_embedding = query_embedding / query_norm
        return dense_embeddings @ query_embedding

    except Exception as exc:
        _INDEX_STATE["dense_available"] = False
        _INDEX_STATE["dense_error"] = f"Query embedding failed; using BM25 fallback. Error: {exc}"
        return None

def _normalize_scores(scores: np.ndarray | None) -> np.ndarray:
    if scores is None or scores.size == 0:
        return np.array([], dtype="float32")

    scores = scores.astype("float32")
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score == min_score:
        return np.zeros_like(scores, dtype="float32")

    return (scores - min_score) / (max_score - min_score)

def _looks_like_bare_numeric_division(query: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:/|÷|divide|divided by)\s*\d+(?:\.\d+)?\b",
            str(query or "").lower(),
        )
    )

def _detect_query_intents(query: str) -> set[str]:
    """
    Detect broad math intent for curriculum reranking.

    The primary path is embedding-prototype classification. Regex is retained
    only as an offline fallback and for a few exact notation-derived guards.
    """
    scores = _embedding_intent_scores(query)

    if scores:
        best_score = max(scores.values())
        cutoff = max(
            _INTENT_SIMILARITY_THRESHOLD,
            best_score - _INTENT_RELATIVE_MARGIN,
        )
        intents = {
            intent
            for intent, score in scores.items()
            if score >= cutoff
        }
    else:
        intents = _lexical_intent_fallback(query)

    text = _normalize_text(query)

    # Exact notation guards are deterministic, cheap, and do not replace the
    # embedding prototype classifier. They only add obvious secondary labels.
    if _looks_like_bare_numeric_division(query):
        intents.add("measurement_division")
        intents.add("whole_number_division")

    if re.search(r"\d+/\d+", text):
        intents.add("fractions")

    if re.search(r"\bcoordinate|coordinates|point\s*\(|x\s*coordinate|y\s*coordinate|quadrant|axis\b", text):
        intents.add("coordinate_plane")

    # Coordinate plane is a graphing topic, but it should not be treated as
    # measurement geometry for the RAG geometry boost. Keep it separate.
    if "coordinate_plane" in intents and "geometry" in intents:
        geometry_score = scores.get("geometry", 0.0) if scores else 0.0
        coordinate_score = scores.get("coordinate_plane", 0.0) if scores else 1.0
        if coordinate_score >= geometry_score:
            intents.discard("geometry")

    return intents

def _intent_adjustment(chunk: CurriculumChunk, intents: set[str]) -> float:
    """
    Small reranking adjustment to avoid obvious topic mismatches.
    Embeddings still dominate; this only prevents lexical false positives such
    as matching 'boxes' to geometry lessons or ordinary equal-sharing problems
    to fraction-division algorithm lessons.
    """
    text = _normalize_text(chunk.searchable_text)
    score = 0.0
    if "whole_number_division" in intents or "measurement_division" in intents:
        if any(
            phrase in text
            for phrase in (
                "whole number division",
                "dividing whole numbers",
                "dividend",
                "divisor",
                "quotient",
                "size of divisor",
                "meaning of division",
                "division situation",
                "how many groups",
                "groups of",
                "remainder",
            )
        ):
            score += 0.35

        if any(
            phrase in text
            for phrase in (
                "unit fraction",
                "non unit fraction",
                "multiply by reciprocal",
                "reciprocal",
                "dividing fractions",
                "divide fractions",
                "fraction division",
                "solving problems involving fractions",
            )
        ):
            score -= 0.45

    if "equal_sharing" in intents:
        if any(
            phrase in text
            for phrase in (
                "meaning of division",
                "meanings of division",
                "division situation",
                "division situations",
                "how much in each group",
                "how many in each group",
                "how many groups",
                "equal group",
                "equal groups",
                "each group",
                "sharing",
                "shared equally",
                "quotient",
                "division equation",
                "interpret division",
            )
        ):
            score += 0.30

        # Strongly avoid geometry false positives for box/container wording.
        if "geometry" not in intents and any(
            term in text
            for term in (
                "volume",
                "prism",
                "surface area",
                "edge length",
                "length width height",
            )
        ):
            score -= 0.40

        # If the student question is ordinary whole-number equal sharing,
        # fraction algorithm lessons should not dominate the grounding panel.
        if "fractions" not in intents:
            if any(
                phrase in text
                for phrase in (
                    "unit fraction",
                    "non unit fraction",
                    "multiply by reciprocal",
                    "reciprocal",
                    "dividing fractions",
                    "divide fractions",
                    "fraction division",
                )
            ):
                score -= 0.30

    if "unit_rate" in intents:
        if any(
            phrase in text
            for phrase in (
                "unit rate",
                "per one",
                "per hour",
                "per item",
                "for each",
                "rate",
            )
        ):
            score += 0.20

    return score


def _clean_excerpt(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    # If the excerpt starts in the middle of a sentence because of chunk
    # overlap, trim to the first likely sentence boundary.
    if text and text[0].islower():
        match = re.search(r"(?<=[.!?])\s+[A-Z0-9]", text)
        if match:
            text = text[match.start() + 1:].strip()

    # Remove broken leading fragments.
    text = re.sub(
        r"^[a-z]{1,12}\s+",
        "",
        text,
    ).strip()

    return text


def _make_semantic_excerpt(document_text: str, query_tokens: list[str]) -> str:
    """
    Create a cleaner preview. Prefer lines with query overlap, but avoid
    returning huge blocks or overlap fragments that begin mid-sentence.
    """
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in document_text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    query_set = set(query_tokens)
    scored_lines = []

    for index, line in enumerate(lines):
        clean_line = _clean_excerpt(line)

        if not clean_line:
            continue

        line_tokens = set(_tokenize(clean_line))
        overlap = len(query_set & line_tokens)

        title_bonus = 1 if index <= 2 else 0
        division_bonus = 1 if any(
            phrase in _normalize_text(clean_line)
            for phrase in (
                "division situation",
                "how much in each group",
                "how many groups",
                "equal groups",
                "shared equally",
            )
        ) else 0

        score = overlap + title_bonus + division_bonus
        scored_lines.append((score, index, clean_line))

    if not scored_lines:
        excerpt = _clean_excerpt(" ".join(lines[:3]))
    else:
        scored_lines.sort(reverse=True)
        best_index = scored_lines[0][1]

        start = max(0, best_index - 1)
        end = min(len(lines), best_index + 3)

        excerpt = _clean_excerpt(" ".join(lines[start:end]))

    if len(excerpt) > MAX_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_EXCERPT_CHARS].rstrip() + "..."

    return excerpt



def _matched_terms(query_tokens: list[str], chunk: CurriculumChunk) -> list[str]:
    chunk_tokens = set(_tokenize(chunk.searchable_text))
    matched = [token for token in query_tokens if token in chunk_tokens]
    return sorted(set(matched))[:12]


def _make_excerpt(document_text: str, matched_terms: list[str]) -> str:
    lines = [line.strip() for line in document_text.splitlines() if line.strip()]

    if not lines:
        return ""

    matched_set = set(matched_terms)
    selected_lines = []

    for index, line in enumerate(lines):
        line_tokens = set(_tokenize(line))
        if line_tokens & matched_set:
            start = max(0, index - 1)
            end = min(len(lines), index + 4)
            selected_lines.extend(lines[start:end])
            break

    if not selected_lines:
        selected_lines = lines[:8]

    excerpt = " ".join(selected_lines)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()

    if len(excerpt) > MAX_EXCERPT_CHARS:
        excerpt = excerpt[:MAX_EXCERPT_CHARS].rstrip() + "..."

    return excerpt


def retrieve_curriculum_context(
    query: str,
    top_k: int = TOP_K,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> dict:
    """
    Retrieve curriculum notes for a student question using embedding-first
    hybrid retrieval.

    Strategy:
    1. Expand the student question into curriculum-search language.
    2. Use embeddings as the primary semantic signal.
    3. Use BM25 as a secondary lexical signal.
    4. Apply a small topic-mismatch guard for obvious false positives.
    5. Return cleaner metadata for the Curriculum Grounding panel.
    """

    expanded_query = _expand_query_for_retrieval(query)
    query_tokens = _tokenize(expanded_query)

    if not query_tokens:
        return {
            "context": "",
            "matches": [],
            "retrieval_method": "none",
            "dense_available": False,
            "weak_match": True,
        }

    _ensure_index()

    chunks: list[CurriculumChunk] = _INDEX_STATE["chunks"]
    bm25: BM25Okapi | None = _INDEX_STATE["bm25"]

    if not chunks or bm25 is None:
        return {
            "context": "",
            "matches": [],
            "retrieval_method": "none",
            "dense_available": False,
            "weak_match": True,
        }

    bm25_scores = np.array(
        bm25.get_scores(query_tokens),
        dtype="float32",
    )
    bm25_norm = _normalize_scores(bm25_scores)

    dense = _dense_scores(expanded_query)
    dense_available = dense is not None

    if dense_available:
        dense_norm = _normalize_scores(dense)

        combined_scores = (
            DENSE_WEIGHT * dense_norm
            + BM25_WEIGHT * bm25_norm
        )

        retrieval_method = "embedding-first hybrid"
    else:
        combined_scores = bm25_norm
        retrieval_method = "bm25 fallback"

    intents = _detect_query_intents(expanded_query)

    for index, chunk in enumerate(chunks):
        combined_scores[index] += _intent_adjustment(
            chunk,
            intents,
        )

    combined_scores = np.clip(
        combined_scores,
        0.0,
        None,
    )

    ranked_indices = _rank_by_scores(combined_scores)

    if not ranked_indices:
        return {
            "context": "",
            "matches": [],
            "retrieval_method": retrieval_method,
            "dense_available": dense_available,
            "dense_error": _INDEX_STATE.get("dense_error", ""),
            "weak_match": True,
        }

    matches = []
    seen_sources = set()

    for index in ranked_indices:
        chunk = chunks[index]

        if chunk.source in seen_sources:
            continue

        seen_sources.add(chunk.source)

        matched = _matched_terms(
            query_tokens=query_tokens,
            chunk=chunk,
        )

        excerpt = _make_semantic_excerpt(
            document_text=chunk.text,
            query_tokens=query_tokens,
        )

        dense_similarity = (
            float(dense[index])
            if dense_available and dense is not None
            else None
        )

        display_score = (
            dense_similarity
            if dense_similarity is not None
            else float(combined_scores[index])
        )

        matches.append({
            "source": chunk.source,
            "score": round(float(display_score), 3),
            "combined_score": round(float(combined_scores[index]), 3),
            "dense_similarity": (
                round(float(dense_similarity), 3)
                if dense_similarity is not None
                else None
            ),
            "bm25_score": round(float(bm25_scores[index]), 3),
            "retrieval_method": retrieval_method,
            "matched_terms": matched,
            "excerpt": excerpt,
        })

        if len(matches) >= top_k:
            break

    best_score = matches[0].get("dense_similarity")
    if best_score is None:
        best_score = matches[0].get("score", 0)

    weak_match = (
        isinstance(best_score, (int, float))
        and best_score < WEAK_DENSE_THRESHOLD
    )

    context_parts = []
    for match in matches:
        context_parts.append(
            f"Source: {match['source']}\n"
            f"Retrieval method: {match['retrieval_method']}\n"
            f"Similarity: {match.get('dense_similarity') or match.get('score')}\n"
            f"Excerpt:\n{match['excerpt']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "..."

    return {
        "context": context,
        "matches": matches,
        "retrieval_method": retrieval_method,
        "dense_available": dense_available,
        "dense_error": _INDEX_STATE.get("dense_error", ""),
        "weak_match": weak_match,
    }



def get_curriculum_context(
    query: str,
    top_k: int = 3,
    max_chars: int = 800,
) -> str:
    """
    Backward-compatible helper.

    Old code expects only a string, so we keep this function.
    """

    result = retrieve_curriculum_context(
        query=query,
        top_k=top_k,
        max_chars=max_chars,
    )
    return result["context"]
