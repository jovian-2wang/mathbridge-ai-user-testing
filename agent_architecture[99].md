# MathBridge AI — Multi-Agent Architecture Plan

_Last updated: 2026-06-22_

This document describes how to evolve MathBridge from its current single-chain
design into a multi-agent system. It is grounded in the actual code as of the
date above (see `review.md` for the audit it is built on) and is intended as a
development guide for v2.

The guiding principle: **specialized agents behind a thin coordinator** —
introduce LLM agents only where the current hardcoding is the limiting factor,
and keep deterministic code where determinism is a feature (mastery math,
image rendering).

---

## Current State (v1) — what actually exists today

Two LLM call sites, one keyword retriever, four hand-coded PIL renderers, and a
deterministic mastery heuristic. No shared state across calls; no agent
framework; no embeddings.

| Concern | File | Implementation today | Model |
|---|---|---|---|
| Tutoring reply | `ai/tutor_chain.py:165-190` | Single `ChatOpenAI.invoke` with a system prompt that requires a JSON payload `{answer, needs_visual, visual_type, visual_data}` | `gpt-4.1-mini` @ T=0.3 (`config.py:11-12`) |
| Curriculum grounding | `ai/rag.py` | Hand-written BM25-like keyword scorer over `data/curriculum/*.txt` — stemmer + `ALIASES` dict + 9 hand-tuned `concept_boosts` | — (no model) |
| Visual planning | `ai/tutor_chain.py:13-54` + `ui/student.py:283-405` | LLM emits visual type/data, but a **regex layer** in the UI overrides it for ratio / unit-rate / fraction-decimal / fraction-division patterns | LLM + regex |
| Visual rendering | `ai/visual_renderer.py` | 4 hand-coded PIL renderers: `fraction_division_bar`, `ratio_bar`, `unit_rate_table`, `fraction_decimal_bar` | — (no model) |
| Signal extraction | `ai/signals.py` | One `ChatOpenAI.invoke` on the last 6 chat turns; returns `{concept, misconception, hints_used, engagement, next_support}` | `gpt-4.1-mini` @ T=0 |
| Mastery update | `memory/student_memory.py:122-312` | Concept → 1 of 4 fixed skill buckets via keyword table, then integer delta in `[-3, +3]` | — (deterministic) |
| Teacher dashboard | `ui/teacher.py` | Pure read of `current_signals`, `signals_history`, `mastery`, `mastery_history`, plus 3 **seeded** insights in `memory/student_memory.py:398-420` | — (no model) |
| Parent weekly summary | `memory/student_memory.py:61-120` | Template-stitched strings from the latest signal dict | — (no model) |

**Limiting factors:**

1. **Diagnostician is blind.** `ai/signals.py` sees only chat — not curriculum,
   not history. Misconceptions that span sessions are invisible.
2. **Grounding is brittle.** `ai/rag.py` cannot handle paraphrase or synonymy
   beyond what is hand-listed in `ALIASES` / `concept_boosts`.
3. **Visual planning is split-brain.** The LLM only knows about
   `fraction_division_bar`; the regex layer knows about three other types.
   Adding a new visual means editing both the prompt and the regex layer.
4. **Teacher and parent views regurgitate data.** No cross-session analysis,
   no longitudinal pattern detection.
5. **Tutor reveals answers.** The system prompt says "guide with questions"
   but never forbids giving the final answer; the JSON schema literally asks
   for an `answer` on every turn.

---

## Target Architecture (v2)

```
                       ┌──────────────────────────────────┐
  Student Input  ──▶   │            Coordinator           │  (plain Python, not a framework)
                       │  retrieve → diagnose → respond   │
                       └──┬──────────────┬──────────┬─────┘
                          │              │          │
                ┌─────────▼──────┐  ┌────▼──────┐  ┌▼─────────────┐
                │  Retriever     │  │   Tutor   │  │ Visual       │
                │  Agent         │  │   Agent   │  │ Planner      │
                │ (BM25+dense)   │  │ (Socratic)│  │ Agent        │
                └────────────────┘  └────┬──────┘  └──┬───────────┘
                                         │            │
                                         │       ┌────▼──────────┐
                                         │       │ PIL Renderer  │  ← deterministic, code
                                         │       │ (+ VLM critic)│
                                         │       └───────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Diagnostician Agent │  ← async after each turn
                              │ (signals + concept) │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Mastery Updater    │  ← deterministic, code
                              │  (delta in [-3,+3]) │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐         ┌──────────────────┐
                              │ Persistent Store    │ ◀───── │ Teacher Insight  │  on-demand
                              │ (data/memory/*.json)│         │ Agent            │
                              └──────────┬──────────┘         └──────────────────┘
                                         │
                                         │                    ┌──────────────────┐
                                         └──────────────────▶ │ Parent Summary   │  weekly
                                                              │ Agent            │
                                                              └──────────────────┘
```

Solid arrows are real-time per-turn flow. Teacher / parent agents run on-demand
(dashboard open, session end, weekly digest), not per turn.

---

## Components

### Retriever Agent (replaces `ai/rag.py`)
- **Role:** choose retrieval strategy and return top-K curriculum chunks.
- **Tools:**
  - `bm25_search(query, k)` — `rank_bm25` over pre-tokenized chunks.
  - `dense_search(query, k)` — cosine over a numpy matrix of chunk embeddings
    (`text-embedding-3-small`, 1536-dim, ~100 chunks today).
  - `lookup_lesson(n)` — metadata filter on `lesson_id`.
  - `hybrid_search(query, k)` — Reciprocal Rank Fusion of BM25 + dense
    (recommended default).
- **Triggers:** every student turn, before the tutor.
- **Inputs:** student turn + last 2 chat turns.
- **Outputs:** `{context: str, matches: [{source, score, matched_terms, excerpt}]}`
  — same shape as today so the curriculum-evidence sidebar
  (`ui/student.py:583-614`) keeps working.
- **Model:** `gpt-4.1-mini` or `claude-haiku-4-5` (fast). For the v0 of this
  agent, skip the LLM and just default to `hybrid_search` — that already beats
  current quality.

### Tutor Agent (refactor of `ai/tutor_chain.py`)
- **Role:** Socratic math tutoring. Asks a question or gives a single hint;
  reveals the answer only after a configurable number of unsuccessful hints.
- **Inputs:** conversation history, current problem, retriever output,
  hint counter from state.
- **Outputs:** `{message, asks_question: bool, reveals_answer: bool}` —
  drop the visual fields from this schema (they move to the Visual Planner).
- **Model:** `gpt-4.1-mini` (latency-sensitive). Promote to
  `claude-sonnet-4-6` if pedagogical quality lags.
- **Key prompt changes vs. v1:**
  - "Never reveal the final answer unless `hint_level >= 3` or the student
    has just stated a correct solution."
  - "End every turn with either a question or a concrete next step the
    student can attempt."
  - Remove the visual fields from the required JSON.

### Visual Planner Agent (replaces the regex fallback in `ui/student.py:283-405`)
- **Role:** decide whether a visual would help and, if so, pick the visual
  type and fill in `visual_data`.
- **Inputs:** student turn, tutor's planned reply, diagnostician's `concept`.
- **Outputs:** `{needs_visual: bool, visual_type: str, visual_data: dict}`.
- **Allowed `visual_type` values:** the union of what the renderer supports —
  `none`, `fraction_division_bar`, `ratio_bar`, `unit_rate_table`,
  `fraction_decimal_bar`. Today only `fraction_division_bar` is in the prompt.
- **Model:** `gpt-4.1-mini`.
- **Why an agent and not regex:** new visual types are added by extending the
  prompt + adding a renderer; no parallel regex layer to maintain.

### PIL Renderer (keep `ai/visual_renderer.py`)
- **Role:** deterministic image drawing. **Not an agent** — pure code.
- **Why:** math visuals must be exact (a `1/8` cell really is `1/8` of the
  bar). An LLM image model would hallucinate proportions.

### Optional: VLM Visual Critic
- **Role:** verify the rendered image matches its caption before sending to
  the student. Catches "wrong divisor count" / "wrong shading" bugs.
- **Triggers:** post-render; cheap because images are small.
- **Inputs:** the PNG bytes + the caption + the original `visual_data`.
- **Outputs:** `{matches_caption: bool, reason: str}`. On failure, drop the
  visual and let the tutor reply without it.
- **Model:** vision-capable LLM — e.g. `claude-opus-4-7` or
  `gpt-4.1-mini` if it has vision support at the time of build. Skip in v2.0
  and add in v2.1 if the planner agent + renderer combination produces
  occasional bad outputs.

### Diagnostician Agent (replaces `ai/signals.py`)
- **Role:** extract structured learning signals from the latest interaction,
  with awareness of curriculum and prior sessions.
- **Triggers:** after every tutor turn (can run async — does not block the
  student-facing reply).
- **Inputs:**
  - last 6 chat messages
  - the retriever's chunks for this turn
  - last 3 entries of the student's `signals_history`
- **Outputs:** same JSON shape as `ai/signals.py:11-17` today, so
  `update_signals` and the teacher view keep working unchanged:
  ```json
  {
    "concept": "...",
    "misconception": "... or null",
    "hints_used": "X hint(s)",
    "engagement": "Active | Moderate | Low | Needs scaffold",
    "next_support": "... or null"
  }
  ```
- **Model:** `gpt-4.1-mini` @ T=0 — same as today, just richer inputs.
  Promote to `claude-sonnet-4-6` only if misconception detection misses
  subtle cases.

### Mastery Updater (mostly deterministic — one exception)
- **Role:** map LLM `concept` → skill bucket, then apply integer delta.
- **Delta math stays deterministic.** The integer-delta logic in
  `memory/student_memory.py:250-312` is correct and explainable. An LLM
  here introduces variance with no quality gain. Keep it.
- **Bucket matching is the exception.** The keyword table in
  `_match_mastery_skill` (`memory/student_memory.py:185-247`) is brittle
  and silently no-ops when the LLM `concept` doesn't match — see
  `review.md` § 9. Replace it with **embedding similarity to bucket
  descriptions**, not a full LLM agent:
  - For each skill bucket, author a one-sentence description.
  - At classify time: embed `concept` with `text-embedding-3-small`,
    cosine against the bucket descriptions, argmax above a threshold,
    else `None`.
  - Deterministic (same concept → same bucket every call), ~5 ms,
    reuses the embedding model already used by the retriever.
- **Bucket definitions move out of code.** The seeded 4-skill dict in
  `_default()` (`memory/student_memory.py:374-379`) becomes a JSON file
  (`data/skill_buckets.json` or similar) the teacher can edit. The
  current 4 buckets are also misaligned with the curriculum on disk
  (Unit 4 is fraction division, but there is no fraction-division
  bucket); add buckets to match the actual lessons in
  `data/curriculum/`.
- **Do not add a "Skill Taxonomist" agent that invents new buckets at
  runtime.** Schema drift makes teacher dashboards unstable. If
  bucket-discovery is desired later, gate it behind an approval step
  in the teacher UI.

### Teacher Insight Agent (replaces seeded `insights` in `memory/student_memory.py:398-420`)
- **Role:** produce grounded, longitudinal observations across the
  student's sessions — patterns the deterministic dashboard cannot see.
- **Triggers:** teacher opens the dashboard OR session ends.
- **Inputs:** full `signals_history`, full `mastery_history`, recent
  `sessions`, `learning_context`.
- **Outputs:** list of `{title, body, severity}` insights, written to
  `student["insights"]` (replaces the seeded list).
- **Model:** `gpt-4.1-mini` for v2.0; promote to `claude-sonnet-4-6` if
  pedagogical nuance is lacking.
- **Cache:** results until new signals arrive; do not re-run on every page
  load.

### Parent Summary Agent (replaces `update_weekly_summary` in `memory/student_memory.py:61-120`)
- **Role:** plain-English weekly digest. Focuses on effort and one
  take-home activity. No jargon, no scores.
- **Triggers:** weekly cron OR parent opens the dashboard if no summary
  exists for the current week.
- **Inputs:** weekly window of `signals_history` + `mastery_history`,
  `learning_context`.
- **Outputs:** the same `weekly_summary` dict shape used today
  (`topics`, `what_went_well`, `needs_support`, `try_at_home`,
  `encouragement`) so the parent UI is unchanged.
- **Model:** `gpt-4.1-mini`.

---

## Coordinator — plain Python, not a framework

For ~5 agents with simple flow, a function beats LangGraph. Adopt
LangGraph only when you actually need conditional routing, checkpointing,
or cross-thread persistence.

```python
# ai/coordinator.py — sketch
def handle_student_turn(student_id: str, user_text: str, history: list[dict]) -> dict:
    retrieval = retriever_agent.retrieve(user_text, history)        # tool-using agent or hybrid_search()
    tutor_reply = tutor_agent.reply(user_text, history, retrieval)  # {message, asks_question, reveals_answer}
    visual_plan = visual_planner_agent.plan(                        # {needs_visual, visual_type, visual_data}
        user_text=user_text,
        tutor_reply=tutor_reply,
        concept=session_state.last_concept,
    )
    image = create_visual(visual_plan["visual_type"], visual_plan["visual_data"]) \
            if visual_plan["needs_visual"] else None

    # Async — does not block the student-facing reply
    enqueue(diagnostician_agent.extract, history + [tutor_reply], retrieval, student_id)

    return {
        "answer": tutor_reply["message"],
        "image": image,
        "retrieval": retrieval,
    }
```

The async queue (`enqueue(...)`) can be a `concurrent.futures.ThreadPoolExecutor`
in dev; promote to a real task queue (RQ / Celery) only when load demands it.

---

## Per-Turn Latency Budget

Keep **synchronous, per-turn** agents to 2: retriever + tutor.

| Stage | Sync/Async | Model | Why |
|---|---|---|---|
| Retriever | sync | small LLM or pure code | needed before tutor |
| Tutor | sync | small LLM | this is the user-facing reply |
| Visual planner | sync if needed inline; otherwise async | small LLM | renderer needs its output, but rendering is cheap |
| PIL render | sync | — | local, ~10–50 ms |
| Diagnostician | **async** | small LLM @ T=0 | result is consumed by next turn / dashboard, not by current reply |
| Mastery updater | sync after diagnostician | — | trivial code |
| VLM critic | async (or skip in v2.0) | VLM | block only if you choose to gate visuals on it |
| Teacher / Parent agents | on-demand | small LLM | never on the chat path |

---

## Shared State

State stays in `data/memory/<student_id>.json` (the current store) — no
framework checkpointer yet. Add fields:

```python
# Conceptual shape, not a TypedDict requirement
{
  # existing
  "name": str,
  "current_topic": str,
  "current_signals": dict,
  "signals_history": list[dict],
  "mastery": dict[str, int],          # 0..100
  "mastery_history": list[dict],
  "sessions": list[dict],
  "insights": list[dict],             # populated by Teacher Insight Agent
  "weekly_summary": dict,             # populated by Parent Summary Agent

  # new for v2
  "hint_level": int,                  # incremented by tutor; reset on new problem
  "learning_context": "home" | "classroom",
  "grade_level": str,                 # e.g. "Grade 6"
  "curriculum_standard": str | None,  # e.g. "Common Core"
  "last_concept": str | None,         # passed from diagnostician → visual planner next turn
}
```

---

## Learning-Context Customization

Both reporter agents adapt their suggestions based on `learning_context`.

|  | `"classroom"` | `"home"` |
|---|---|---|
| Teacher insight | Intervention strategies, group activities, curriculum-standard codes | Homeschool-friendly activities, online resource links, next-session plan |
| Parent summary | "What to practice at home tonight" (supplement) | "What clicked and what to try tomorrow" (parent is the teacher) |
| Tone | Professional, pedagogical | Warm, practical, jargon-free |

`learning_context` is set once at session start and passed through state. No
branching inside agent logic — only prompts differ.

---

## Model Choices (current generation)

All models listed below are current as of 2026-06-21. Per-agent overrides
should live in `config.py` (or env vars), not hardcoded in agent files.

| Component | Recommended default | Promote to | Reason |
|---|---|---|---|
| Retriever Agent | `gpt-4.1-mini` or skip-LLM-use-hybrid | `claude-haiku-4-5` | Routing across 3 tools; latency-sensitive |
| Tutor Agent | `gpt-4.1-mini` | `claude-sonnet-4-6` | Real-time; Socratic prompt is the main lever |
| Visual Planner | `gpt-4.1-mini` | — | Pure structured-output task |
| Visual Critic (optional) | `claude-opus-4-7` (vision) | — | Needs vision; opus tier is overkill but safe |
| Diagnostician | `gpt-4.1-mini` @ T=0 | `claude-sonnet-4-6` | Structured JSON; misconception nuance |
| Teacher Insight | `gpt-4.1-mini` | `claude-sonnet-4-6` | Longitudinal pedagogical reasoning |
| Parent Summary | `gpt-4.1-mini` | — | Simple plain-English prose |
| Mastery Updater (delta math) | — | — | Deterministic; not a model |
| Mastery Updater (bucket match) | `text-embedding-3-small` | — | Deterministic given fixed bucket text; replaces brittle keyword table |
| PIL Renderer | — | — | Deterministic; not a model |

Embedding model for retrieval: `text-embedding-3-small` (cheap, already in
the OpenAI stack). Alternative: `BAAI/bge-small-en-v1.5` locally via
`sentence-transformers` if avoiding a second OpenAI dependency matters.

---

## Migration Path

Build incrementally. Each step is independently shippable; the system stays
working after every step.

1. **Hybrid retrieval (1 file).** Replace `ai/rag.py` internals with BM25
   (`rank_bm25`) + dense (`text-embedding-3-small`) + RRF. Keep the public
   `retrieve_curriculum_context(query) -> {context, matches}` signature so
   the tutor and the sidebar (`ui/student.py:583-614`) are untouched.
   Build the index once at startup; cache in memory.
2. **Diagnostician sees curriculum + history (1 file).** Extend
   `ai/signals.py` to accept and use the retriever's chunks and the last
   3 `signals_history` entries. JSON shape unchanged → no UI changes.
3. **Tutor stops giving answers (1 file).** Edit `_SYSTEM` in
   `ai/tutor_chain.py:13-54`: add a "never reveal answer before hint_level
   ≥ 3" rule and a `hint_level` counter in state. Remove visual fields
   from the required JSON.
4. **Visual planner agent.** Move visual-type/data selection out of
   `ui/student.py:283-405` into a new `ai/visual_planner.py`. Delete the
   regex block. Visual planner is allowed to emit all 4 renderer types.
5. **Teacher insight agent.** Add `ai/insights.py`. Wire `ui/teacher.py`
   to call it on dashboard open; cache result on the student record.
   Remove the seeded `insights` list from
   `memory/student_memory.py:398-420`.
6. **Parent summary agent.** Replace `update_weekly_summary` body in
   `memory/student_memory.py:61-120` with an LLM call; keep the same
   output dict shape.
6a. **Embedding-based skill-bucket matching.** Move the 4-skill dict out
   of `_default()` into `data/skill_buckets.json`, expand to match the
   curriculum (add at minimum `Fraction division`), and replace the
   keyword table in `_match_mastery_skill` with cosine similarity
   against bucket descriptions. Reuses the embedding index from
   step 1. See `review.md` § 9.
7. **Coordinator.** Introduce `ai/coordinator.py` and route the student
   chat through it. At this point the per-call instantiation of
   `ChatOpenAI` in `ai/tutor_chain.py:175` and `ai/signals.py:30` can
   move to module-level singletons.
8. **(Optional) VLM critic.** Add only if step 4 produces occasional
   wrong visuals in real usage.
9. **(Optional) LangGraph + checkpointing.** Adopt only if you need
   cross-session resumption or conditional routing more complex than
   `if reveals_answer: ...`.

---

## What NOT to do

- **Don't convert mastery delta math to an agent.** Determinism here is a
  feature for parents and teachers ("why did the score drop?"). The
  **bucket-matching** step is the only LLM-adjacent piece, and even there
  the recommendation is embedding cosine, not a generative call — same
  concept must always map to the same bucket.
- **Don't let an agent invent new skill buckets at runtime.** Schema
  drift breaks teacher dashboards. New buckets are a curriculum change
  and belong in `data/skill_buckets.json` (or a future admin UI), not in
  model output.
- **Don't generate visuals with an image model.** Math visuals require
  exact proportions; PIL is correct.
- **Don't adopt LangGraph / CrewAI on day one.** The flow is linear with
  one async branch. A function is clearer than a graph until it isn't.
- **Don't run teacher / parent agents on the chat path.** They are
  on-demand by design.
- **Don't keep the regex fallback in `ui/student.py:283-405` after the
  visual planner agent ships.** Two sources of truth for visual selection
  is the v1 bug that we are fixing.

---

## Dependencies to Add

```
rank_bm25>=0.2.2        # step 1
numpy>=1.26              # step 1 (likely already pulled in by PIL stack)
openai>=1.40             # already present transitively via langchain-openai
# optional, only if going local for embeddings:
sentence-transformers>=2.7
# optional, only if/when step 9 lands:
langgraph>=0.2.0
```

---

## Cross-Reference

- Audit of the current codebase: `review.md` (questions 1–6).
- Recommendation that this plan implements: `review.md` § 7
  ("Would a multi-agent architecture make this more intelligent?").
