# MathBridge AI — Multi-Agent Architecture Plan

This document describes the target architecture for evolving MathBridge from its current single-chain design into a multi-agent system. It is intended as a development guide for anyone building v2.

---

## Current State (v1)

Two standalone LLM calls, no shared state:

- `ai/tutor_chain.py` — tutoring response using `ChatOpenAI` + LangChain message objects
- `ai/signals.py` — post-conversation signal extraction, returns structured JSON
- `ai/rag.py` — keyword-based curriculum retrieval (no embeddings yet)

---

## Target Architecture

```
                    ┌──────────────────────────────────┐
  User Input ──▶    │         Supervisor Agent          │
                    │  (routes based on session state)  │
                    └──┬──────────┬─────────────────────┘
                       │          │
              ┌────────▼──┐  ┌────▼────────────┐
              │   Tutor   │  │ Problem Generator│
              │   Agent   │  │     Agent        │
              └────────┬──┘  └────┬────────────┘
                       │          │
              ┌────────▼──────────▼──────┐
              │   Hint Escalation Agent  │
              │   (activates when stuck) │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │   Signals Analyst Agent  │  ← runs after every tutor turn
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │  Curriculum Planner Agent │  ← updates mastery map
              └──────┬──────────┬─────────┘
                     │          │
            ┌────────▼──┐  ┌────▼──────────┐
            │  Teacher  │  │    Parent     │
            │ Reporter  │  │   Reporter    │
            └───────────┘  └───────────────┘
```

Reporters are **not agents** — they are chains (no tool use, no routing). They run on-demand, not after every turn.

---

## Components

### Supervisor Agent
- **Role:** Reads session state and decides which agent to invoke next.
- **Triggers:** Every user message.
- **Routing logic:**
  - Student appears stuck (signals: `engagement = "Needs scaffold"`) → Hint Escalation
  - Session just started or topic changed → Problem Generator
  - Normal conversational turn → Tutor Agent
- **Model:** Fast and cheap — `gpt-4o-mini` or `claude-haiku-4-5`. Routing decisions don't need a strong model.
- **Implementation:** LangGraph conditional edges from this node.

---

### Tutor Agent
- **Role:** Socratic math tutoring. Guides students to answers through questions and hints. Never gives answers directly.
- **Triggers:** Routed by Supervisor on normal turns.
- **Inputs:** Conversation history, current problem, curriculum context from RAG.
- **Model:** `gpt-4o-mini` or `claude-haiku-4-5` — latency matters here (real-time response).
- **Current file:** `ai/tutor_chain.py` — logic can migrate directly into a LangGraph node.

---

### Problem Generator Agent
- **Role:** Creates a new math problem appropriate to the student's current level and topic.
- **Triggers:** Start of session, topic change, or after a problem is solved.
- **Inputs:** `mastery` map and `grade_level` from state; curriculum context from RAG.
- **Outputs:** A problem string written into `state["current_problem"]`.
- **Model:** `gpt-4o-mini` or `claude-haiku-4-5`.
- **Note:** This is new functionality — does not exist in v1.

---

### Hint Escalation Agent
- **Role:** Generates a sequence of progressive hints (level 1 → level 2 → partial reveal) when a student is stuck.
- **Triggers:** Supervisor detects `engagement = "Needs scaffold"` in signals, or student explicitly asks for help multiple times.
- **Inputs:** Current problem, conversation history, hint level counter.
- **Outputs:** One hint at the appropriate level; increments hint counter in state.
- **Model:** `gpt-4o-mini` or `claude-haiku-4-5`.
- **Note:** In v1 this logic lives inside the Tutor Agent prompt. Extracting it here gives finer control and avoids the tutor giving away answers too early.

---

### Signals Analyst Agent
- **Role:** Analyzes recent conversation turns and extracts structured learning signals.
- **Triggers:** After every Tutor Agent turn (automatic edge in the graph).
- **Inputs:** Last 6 messages from conversation history.
- **Outputs:** Structured dict appended to `state["signals_history"]`:
  ```json
  {
    "concept": "addition with regrouping",
    "misconception": "student adds digits without carrying",
    "hints_used": "2 hints requested",
    "engagement": "Needs scaffold",
    "next_support": "try a visual number line"
  }
  ```
- **Model:** `gpt-4o` or `claude-sonnet-4-6` — needs reliable structured JSON output and subtle misconception detection.
- **Current file:** `ai/signals.py` — logic migrates directly; add accumulation into `signals_history`.

---

### Curriculum Planner Agent
- **Role:** Maintains a per-concept mastery map and decides what to focus on next.
- **Triggers:** After Signals Analyst updates signals.
- **Inputs:** Full `signals_history`, current `mastery` map.
- **Outputs:** Updated `mastery` dict; optionally sets `state["next_topic"]` when mastery threshold is reached.
- **Model:** `gpt-4o` or `claude-sonnet-4-6`.
- **Note:** This is new functionality. In the short term it can use simple heuristics (e.g., mastery += 0.1 per clean solve). Promote to LLM-driven when the logic becomes complex.

---

## Reporters

Reporters are LangChain chains, not agents. They have no tools, make no routing decisions, and do not run in the real-time loop. They read accumulated state and produce formatted prose on demand.

### Teacher Reporter
- **Triggered by:** Teacher opening the dashboard or end-of-session summary request.
- **Inputs:** Full `signals_history`, `mastery` map, `grade_level`, `curriculum_standard`, `learning_context`.
- **Output:** A structured session report with misconceptions, intervention suggestions, and curriculum alignment.
- **Model:** `gpt-4o` or `claude-sonnet-4-6` — pedagogical nuance and curriculum references justify a stronger model.

### Parent Reporter
- **Triggered by:** Parent viewing the progress tab or weekly digest.
- **Inputs:** Simplified signals summary, `mastery` map, `learning_context`.
- **Output:** Short, friendly prose. Focuses on effort and attitude. One take-home activity suggestion.
- **Model:** `gpt-4o-mini` or `claude-haiku-4-5` — simpler language generation, no jargon needed.

---

## Learning Context Customization

Both reporters adapt their suggestions based on `learning_context` in state.

| | `"classroom"` | `"home"` |
|---|---|---|
| Teacher report | Intervention strategies, group activities, curriculum standard codes | Homeschool-friendly activities, online resource links, next session plan |
| Parent report | "What to practice at home tonight" (short supplement) | "What clicked and what to try tomorrow" (homeschool parent is the teacher) |
| Tone | Professional, pedagogical | Warm, practical, jargon-free |

The `learning_context` field is set once at the start of a session and flows through state. No branching needed inside agent logic — only the reporter prompts differ.

Additional state fields that enable customization:
```python
learning_context: Literal["home", "classroom"]
grade_level: str                  # e.g. "Grade 3"
curriculum_standard: str | None   # e.g. "Common Core", "Singapore Math", "IB PYP"
```

---

## Shared State (LangGraph)

```python
from typing import TypedDict, Annotated, Literal
import operator

class TutorState(TypedDict):
    # Conversation
    messages: Annotated[list, operator.add]   # full chat history
    current_problem: str | None

    # Analysis
    signals_history: Annotated[list, operator.add]   # one entry per turn
    mastery: dict[str, float]                        # concept → 0.0–1.0
    next_topic: str | None

    # Routing
    next_agent: str
    hint_level: int                # 0 = no hints yet; increments per hint turn

    # Session config
    learning_context: Literal["home", "classroom"]
    grade_level: str
    curriculum_standard: str | None
```

---

## LangGraph Wiring (outline)

```python
from langgraph.graph import StateGraph, END

builder = StateGraph(TutorState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("tutor", tutor_node)
builder.add_node("problem_gen", problem_gen_node)
builder.add_node("hints", hint_node)
builder.add_node("signals", signals_node)
builder.add_node("curriculum", curriculum_node)

builder.set_entry_point("supervisor")

# Supervisor routes to one of three agents
builder.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {"tutor": "tutor", "problem_gen": "problem_gen", "hints": "hints"},
)

# After any agent responds, always run signals then curriculum
builder.add_edge("tutor", "signals")
builder.add_edge("hints", "signals")
builder.add_edge("problem_gen", "tutor")   # gen a problem, then tutor presents it
builder.add_edge("signals", "curriculum")
builder.add_edge("curriculum", END)

graph = builder.compile(checkpointer=...)  # add SQLite/Postgres checkpointer for persistence
```

Reporters are invoked outside the graph, by the Streamlit UI layer, passing the final state as input.

---

## Model Summary

| Component | Recommended Model | Reason |
|---|---|---|
| Supervisor | `gpt-4o-mini` / `claude-haiku-4-5` | Routing only, fast |
| Tutor Agent | `gpt-4o-mini` / `claude-haiku-4-5` | Real-time, latency-sensitive |
| Problem Generator | `gpt-4o-mini` / `claude-haiku-4-5` | Straightforward generation |
| Hint Escalation | `gpt-4o-mini` / `claude-haiku-4-5` | Simple scaffolded output |
| Signals Analyst | `gpt-4o` / `claude-sonnet-4-6` | Structured JSON, subtle misconception detection |
| Curriculum Planner | `gpt-4o` / `claude-sonnet-4-6` | Reasoning over mastery history |
| Teacher Reporter | `gpt-4o` / `claude-sonnet-4-6` | Pedagogical nuance, curriculum references |
| Parent Reporter | `gpt-4o-mini` / `claude-haiku-4-5` | Simple prose, no jargon |

All models are swappable per node — configure via `config.py` or environment variables, not hardcoded.

---

## Migration Path

Build incrementally — do not rewrite v1 all at once.

1. **Wrap v1 in LangGraph** — move `tutor_chain.py` and `signals.py` into nodes that read/write `TutorState`. No logic changes yet.
2. **Add Supervisor** — start with simple if/else routing on signals output. Promote to LLM-driven later.
3. **Extract Hint Escalation** — pull hint logic out of the tutor prompt into its own node. Add `hint_level` counter to state.
4. **Add Problem Generator** — new node; feed into the tutor node before presenting to student.
5. **Add Curriculum Planner** — start with heuristic mastery scoring. Replace with LLM-driven planning once signal history is rich enough.
6. **Build Reporters** — add Teacher and Parent reporter chains, wired to the Streamlit dashboard layer.
7. **Add persistence** — attach a LangGraph checkpointer (SQLite for dev, Postgres for prod) to survive across sessions.

---

## Dependencies to Add

```
langgraph>=0.2.0
langchain-anthropic>=0.1.0   # if switching to Claude models
```

Everything else (`langchain`, `langchain-core`, `langchain-openai`) is already in `requirements.txt`.
