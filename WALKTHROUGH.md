# MathBridge AI — Internal Walkthrough

A beginner-friendly, step-by-step guide to what actually happens when a student types a message.

---

## Table of Contents

1. [Big Picture Diagram](#1-big-picture-diagram)
2. [Step-by-Step: What Happens After a Student Submits a Message](#2-step-by-step-what-happens-after-a-student-submits-a-message)
3. [Files and Functions Called in Order](#3-files-and-functions-called-in-order)
4. [What LangChain Actually Does](#4-what-langchain-actually-does)
5. [How Streamlit session_state and rerun Work](#5-how-streamlit-session_state-and-rerun-work)
6. [Where Memory Is Updated](#6-where-memory-is-updated)
7. [Where Future Agents Could Be Inserted](#7-where-future-agents-could-be-inserted)

---

## 1. Big Picture Diagram

```
Student types a message
        │
        ▼
┌───────────────────┐
│   app.py          │  Streamlit entry point — routes to the right view
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ ui/student.py     │  Renders chat UI, captures input, calls _reply()
└───────────────────┘
        │
        ├──── OpenAI Call #1 ──────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌───────────────────┐     ┌─────────────┐     ┌───────────────────────────┐
│ ai/tutor_chain.py │────▶│  ai/rag.py  │     │  OpenAI gpt-4.1-mini      │
│ get_tutor_response│     │ (find notes)│     │  (generate tutor reply)   │
└───────────────────┘     └─────────────┘     └───────────────────────────┘
        │
        ├──── OpenAI Call #2 ──────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌───────────────────┐                         ┌───────────────────────────┐
│ ai/signals.py     │                         │  OpenAI gpt-4.1-mini      │
│ extract_signals() │                         │  (analyze the conversation)│
└───────────────────┘                         └───────────────────────────┘
        │
        ▼
┌───────────────────┐
│  session_state    │  In-memory store updated; UI re-renders via st.rerun()
└───────────────────┘
```

---

## 2. Step-by-Step: What Happens After a Student Submits a Message

### Step 1 — Streamlit receives the input

The student types in the chat box at the bottom of the screen (or clicks a quick-action button like "Give me a hint"). Streamlit detects this and calls `_reply()` in `ui/student.py`.

```python
# ui/student.py:39-40
if prompt := st.chat_input("Type your answer or question here..."):
    _reply(prompt, student_id)
```

### Step 2 — The user message is saved immediately

Before doing anything else, the user's message is appended to `chat_history` in `session_state`. This is the running log of the conversation for this browser session.

```python
# ui/student.py:51
st.session_state.chat_history.append({"role": "user", "content": user_text})
```

### Step 3 — Tutor chain fetches curriculum context (RAG)

`get_tutor_response()` is called. Its first action is to search the local `.txt` files in `data/curriculum/` for paragraphs that match keywords from the student's message. This is the **retrieval** part of Retrieval-Augmented Generation.

```
Student says: "how do I add fractions?"
         │
         ▼
  Split into words: {"how", "do", "i", "add", "fractions"}
         │
         ▼
  Scan every paragraph in every .txt file
  Score = number of matching words
         │
         ▼
  Return top-3 matching paragraphs (max 800 chars total)
```

If no curriculum files match, an empty string is returned and the LLM answers from its own training knowledge.

### Step 4 — OpenAI Call #1: Generate the tutor reply

LangChain assembles a list of messages and sends them to OpenAI:

```
[SystemMessage]   ← the tutor persona + any curriculum notes
[HumanMessage]    ← oldest user message in history
[AIMessage]       ← oldest assistant reply
[HumanMessage]    ← ...
[AIMessage]       ← ...
[HumanMessage]    ← the NEW message just submitted  ← always last
```

OpenAI returns a short, friendly response (2–4 sentences, per the system prompt rules). That response string is returned to `_reply()`.

### Step 5 — The assistant reply is saved

The tutor's response is appended to `chat_history`, right after the user message.

```python
# ui/student.py:54
st.session_state.chat_history.append({"role": "assistant", "content": response})
```

### Step 6 — OpenAI Call #2: Extract learning signals

`extract_signals()` is called with the full `chat_history`. It takes the last 6 messages, formats them as a plain text transcript, and sends them to OpenAI with `temperature=0` (no creativity — we want a consistent structured output).

The model is asked to return **only** a JSON object describing:
- The math concept being practiced
- Any misconception spotted
- How many hints were requested
- Engagement level
- Suggested next teaching move

```python
# ai/signals.py:30-43
llm = ChatOpenAI(model=LLM_MODEL, temperature=0, ...)
response = llm.invoke([SystemMessage(...), HumanMessage(conversation)])
return json.loads(response.content)
```

### Step 7 — session_state is updated and the page reruns

The signals dict is saved to `session_state.signals`, then `st.rerun()` is called. This tells Streamlit to re-execute `app.py` from the top, which causes the chat history and signals panel to redraw with the new data.

```python
# ui/student.py:55-56
st.session_state.signals = extract_signals(st.session_state.chat_history)
st.rerun()
```

---

## 3. Files and Functions Called in Order

| # | File | Function | What it does |
|---|------|----------|--------------|
| 1 | `app.py` | *(top level)* | Loads env, sets page config, routes to `student.render()` |
| 2 | `ui/student.py` | `render()` | Draws the page; detects chat input |
| 3 | `ui/student.py` | `_reply()` | Orchestrates the whole turn |
| 4 | `memory/student_memory.py` | `load_student()` | Reads student JSON profile from disk |
| 5 | `ai/tutor_chain.py` | `get_tutor_response()` | Builds and sends the tutor prompt |
| 6 | `ai/rag.py` | `get_curriculum_context()` | Keyword search over curriculum `.txt` files |
| 7 | `ai/tutor_chain.py` | *(LangChain invoke)* | OpenAI Call #1 — tutor reply |
| 8 | `ai/signals.py` | `extract_signals()` | OpenAI Call #2 — JSON learning signals |
| 9 | `ui/student.py` | `st.rerun()` | Forces Streamlit to redraw the whole page |

---

## 4. What LangChain Actually Does

LangChain might sound complex, but in this app it does two simple things:

### Thing 1: Wraps OpenAI in a clean Python object

Instead of writing raw HTTP requests to the OpenAI API, LangChain provides `ChatOpenAI`, a class that handles authentication, retries, and response parsing.

```python
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, api_key="...")
response = llm.invoke(list_of_messages)
print(response.content)  # the reply text
```

### Thing 2: Provides typed message classes

LangChain defines three message types that map directly to the OpenAI chat format:

| LangChain class | OpenAI role | Used for |
|----------------|------------|---------|
| `SystemMessage` | `system` | Tutor persona, rules, curriculum notes |
| `HumanMessage` | `user` | Student messages |
| `AIMessage` | `assistant` | Previous tutor replies |

The app manually builds this list every turn — there is no LangChain "chain" or "memory" object in use. The history is assembled from `session_state.chat_history` by hand.

```
messages = [SystemMessage(...)]          ← always first
for each past message in history:
    if user → append HumanMessage
    if assistant → append AIMessage
messages.append(HumanMessage(new_input)) ← always last
```

That list is handed to `llm.invoke()` and LangChain serializes it into the JSON format OpenAI expects.

---

## 5. How Streamlit `session_state` and `rerun` Work

### The core concept

Streamlit is unusual: **every user interaction re-runs the entire Python script from top to bottom.** The way data survives between those re-runs is `st.session_state` — a dictionary that lives as long as the browser tab is open.

```
User clicks button
       │
       ▼
Python script runs from line 1 of app.py
       │
       ▼
session_state still has all the data from the previous run
       │
       ▼
UI is redrawn with the current state
```

### What lives in session_state for this app

| Key | Type | Contains |
|-----|------|---------|
| `view` | string | Which tab is active (`"student"`, `"teacher"`, etc.) |
| `chat_history` | list of dicts | All messages: `[{"role": "user", "content": "..."}]` |
| `signals` | dict | Latest learning signals JSON from `extract_signals()` |

### Why `st.rerun()` is called explicitly

After `_reply()` finishes, the new messages are already in `session_state`. But Streamlit is mid-render — it's still in the function that handled the button click. Calling `st.rerun()` immediately discards the rest of the current render and starts the script fresh, so the new chat bubbles appear right away rather than waiting for the next natural interaction.

```
_reply() finishes
    ├── chat_history has new user + assistant messages
    ├── signals has new JSON
    └── st.rerun() ──▶ script restarts from app.py line 1
                            │
                            ▼
                      render() reads session_state
                      and draws the updated chat
```

---

## 6. Where Memory Is Updated

There are two kinds of "memory" in this app that should not be confused:

### Kind 1: In-session memory (`session_state`)

`chat_history` in `session_state` is the conversation memory **within one browser session**. It is lost when the tab closes. It is how the tutor "remembers" what was said earlier in the same chat.

- Updated in: `ui/student.py:_reply()` — lines 51 and 54
- Cleared when: the browser tab closes or the user refreshes

### Kind 2: Persistent memory (JSON files on disk)

Each student has a JSON file at `data/memory/{student_id}.json`. This stores the student profile, mastery percentages, session history, and the latest signals.

- **Read** on every page load: `memory/student_memory.load_student()` — called at the top of `render()` in `ui/student.py:10`
- **Written** by `memory/student_memory.save_session()` — saves a completed session to the sessions list
- **Written** by `memory/student_memory.update_signals()` — overwrites `current_signals` with the latest extracted signals

> **Note:** As of the current implementation, `update_signals()` and `save_session()` exist but are **not yet called** after each turn. The signals are stored in `session_state` for display but not persisted to disk automatically. This is an obvious place to add a call in `_reply()`.

```
data/memory/
    alex.json       ← one file per student_id
    maya.json
    ...
```

---

## 7. Where Future Agents Could Be Inserted

The `_reply()` function in `ui/student.py` is the central pipeline. Each step below is a clean insertion point:

```python
def _reply(user_text: str, student_id: str):
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    # ── INSERT POINT A ──────────────────────────────────────────────
    # Intent classifier agent: detect if the student is frustrated,
    # off-topic, or asking something outside math — before hitting OpenAI.

    response = get_tutor_response(user_text, st.session_state.chat_history[:-1])

    # ── INSERT POINT B ──────────────────────────────────────────────
    # Safety / tone filter agent: check the response before showing it
    # to the student. Rewrite if too complex or discouraging.

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.signals = extract_signals(st.session_state.chat_history)

    # ── INSERT POINT C ──────────────────────────────────────────────
    # Persistence agent: call update_signals() and save_session() here
    # to write signals to disk after every turn.

    # ── INSERT POINT D ──────────────────────────────────────────────
    # Alert agent: if signals["misconception"] is not null, trigger a
    # notification to the teacher dashboard or log it for review.

    st.rerun()
```

Other expansion points:

| Location | What to add |
|----------|------------|
| `ai/rag.py` | Replace keyword search with FAISS + sentence-transformers for semantic retrieval |
| `ai/tutor_chain.py` | Add a difficulty-scaling agent that adjusts vocabulary based on mastery scores |
| `memory/student_memory.py` | Add a summarization agent that condenses old sessions into a running profile narrative |
| `app.py` | Add a routing agent that decides which view to show based on the student's current state |

---

*This document reflects the code as of the initial prototype. Check `PROJECT_NOTES.md` for design decisions and `config.py` for model and path configuration.*

---

## 8. Teacher Dashboard

The teacher dashboard lives in `ui/teacher.py` and is rendered when `session_state.view == "Teacher Dashboard"`. It pulls from two sources at the same time: the persisted student JSON file on disk and the live signals in `session_state` from the current chat session.

### Data sources

```
load_student(student_id)          ← disk: data/memory/{student_id}.json
st.session_state.get("signals")   ← in-memory: updated after every student turn
```

The two sources complement each other. The JSON file has the full mastery history; `session_state.signals` has what just happened in the current chat. If the teacher is watching while a student is actively chatting, the dashboard reflects that live activity without a page refresh.

### Layout: four metric tiles

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Topic Focus  │ Mastery Trend│Support Needed│  Engagement  │
│ (live signal │ (hardcoded   │ (lowest-score│ (live signal │
│  or profile) │  stub value) │  skill)      │  or default) │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

- **Topic Focus** — `signals["concept"]` if a session is running, otherwise `student["current_topic"]` from the JSON profile.
- **Mastery Trend** — currently a static placeholder (`"Improving", "+12%"`). This is the natural place to compute a real trend from `student["sessions"]`.
- **Support Needed** — the skill with the lowest mastery percentage, found with `min(mastery, key=mastery.get)`.
- **Engagement** — `signals["engagement"]` from the live session, defaulting to `"Moderate"`.

### Layout: two-column detail panel

Below the metrics, the page splits into two columns:

**Left — Student Mastery Map** (`ui/teacher.py:28-33`)

Iterates over every entry in `student["mastery"]` and renders a labeled progress bar. The colored dot is determined by the score:

| Score | Dot | Meaning |
|-------|-----|---------|
| ≥ 75% | 🟢 | Proficient |
| 50–74% | 🟡 | Developing |
| < 50% | 🔴 | Needs support |

**Right — Key Insights & Suggested Actions** (`ui/teacher.py:36-48`)

Starts from the `insights` list in the student JSON, then prepends a live item if the current session produced a detected misconception:

```python
if signals.get("misconception"):
    insights.insert(0, {
        "title": "Live Signal: Misconception Detected",
        "body": signals["misconception"],
    })
```

This means the most urgent, real-time finding always appears at the top. Each insight renders in a bordered card with a bold title and a body paragraph.

### Full data flow for the teacher dashboard

```
Student chats in ui/student.py
        │
        ▼
extract_signals() → session_state.signals
        │
        ▼
Teacher navigates to "Teacher Dashboard"
        │
        ├── load_student("alex") ──▶ data/memory/alex.json
        │                              mastery, insights, current_topic
        │
        └── session_state.signals ──▶ concept, misconception, engagement
                │
                ▼
        Metric tiles + Mastery Map + Insights panel rendered
```

### Improvements and future expansion

**Known gaps in the current implementation**

- The **Mastery Trend** tile is hardcoded to `"Improving, +12%"` (`ui/teacher.py:16`). It should compute a real delta by comparing the mastery scores from the most recent session against the previous one, both of which are already stored in `student["sessions"]`.
- The **student selector** is hardcoded to `"alex"` via `session_state.get("student_id", "alex")`. There is no UI to switch between students — a teacher with a full class cannot see anyone else. A `st.selectbox` populated from the filenames in `data/memory/` would fix this with a few lines.
- Signals are **only live for the current session**. If the teacher opens the dashboard after the student has logged off, `session_state.signals` is empty and the live misconception card never appears. Calling `update_signals()` and `save_session()` inside `_reply()` (see Insert Point C in section 7) would write signals to disk so they survive across sessions.

**Near-term additions worth considering**

| Addition | Where to add it | What it unlocks |
|----------|----------------|-----------------|
| Real mastery trend delta | `ui/teacher.py:16` — compute from `student["sessions"]` | Accurate progress tracking instead of a stub |
| Multi-student selector | `ui/teacher.py` top — `st.selectbox` over `data/memory/` filenames | Teachers can browse their whole class |
| Persist signals to disk | `ui/student.py:_reply()` — call `update_signals()` after `extract_signals()` | Historical signal data; dashboard works after session ends |
| Session history chart | New panel in `ui/teacher.py` — `st.line_chart` over mastery over time | Visual trend across weeks, not just a number |
| Alert badge | `ui/teacher.py` — check `signals["misconception"]` on load, show `st.warning` | Teacher sees urgent issues at a glance without reading the whole insights panel |
| Class-wide summary | New `ui/class_overview.py` — aggregate mastery across all JSON files | Principal or department view across all students |

---

## 9. Parent Dashboard

The parent dashboard lives in `ui/parent.py`. Its audience is a family member rather than an educator, so everything is simplified: no raw percentages, no pedagogical jargon, no live signals.

### Data source

```python
student = load_student(student_id)
summary = student.get("weekly_summary", {})
```

The entire display is driven by the `weekly_summary` block inside the student JSON. There is no dependency on `session_state.signals` — the parent view is deliberately a snapshot, not a live feed.

### Layout

```
"{name} practiced {topics} this week."

┌──────────────────┬──────────────────┬──────────────────┐
│ ✅ What went well│ 🎯 Needs support │ 🏠 Try at home   │
│  (green card)    │  (yellow card)   │  (blue card)     │
└──────────────────┴──────────────────┴──────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 💛 Encouragement Note                                   │
└─────────────────────────────────────────────────────────┘
```

Each colored card is a raw HTML `<div>` injected via `st.markdown(..., unsafe_allow_html=True)`. The template is defined once at the top of the file as the `_BOX` string (`ui/parent.py:4-9`) and re-used for all three columns.

### Fields read from `weekly_summary`

| JSON key | Card | Default if missing |
|----------|------|--------------------|
| `what_went_well` | Green — What went well | `"—"` |
| `needs_support` | Yellow — Needs support | `"—"` |
| `try_at_home` | Blue — Try at home | `"—"` |
| `encouragement` | Encouragement note | `"Keep up the great work!"` |
| `topics` | Intro sentence | `"fractions: comparison and addition"` |

### Comparison: teacher vs. parent dashboard

| Aspect | Teacher Dashboard | Parent Dashboard |
|--------|------------------|-----------------|
| File | `ui/teacher.py` | `ui/parent.py` |
| Primary audience | Educator | Family |
| Live signals used | Yes — concept, misconception, engagement | No |
| Data format | Mastery percentages, color-coded bars | Plain sentences, colored cards |
| Update cadence | Reflects every student turn via session_state | Reflects last saved weekly_summary JSON |
| Where data comes from | `student["mastery"]` + `session_state.signals` | `student["weekly_summary"]` |

### Improvements and future expansion

**Known gaps in the current implementation**

- `weekly_summary` is **static data written by hand** into the student JSON. Nothing in the app generates it automatically. Every field (`what_went_well`, `needs_support`, `try_at_home`, `encouragement`) will stay stale until someone edits the file manually.
- There is **no multi-child support**. The student selector defaults to `"alex"` and there is no UI to switch. A parent with more than one child has no way to view the other.
- The `topics` intro sentence at the top (`ui/parent.py:20`) falls back to `"fractions: comparison and addition"` if missing — a visible placeholder that should never reach a real user.

**Near-term additions worth considering**

| Addition | Where to add it | What it unlocks |
|----------|----------------|-----------------|
| Auto-generate `weekly_summary` with an LLM | New `ai/weekly_summary.py` — call after `save_session()` at week boundary | Parent view always reflects real activity, no manual editing |
| Child switcher | `ui/parent.py` top — `st.selectbox` over `data/memory/` filenames | Families with multiple children can switch between them |
| Session streak / time-on-task | Add to the intro paragraph — compute from `student["sessions"]` | Simple engagement metric parents can understand at a glance |
| Shareable report | `ui/parent.py` — `st.download_button` exporting the summary as a PDF or plain text | Parents can forward the summary to a tutor or print it |
| Notification hook | After LLM generates new `weekly_summary` — send an email or push notification | Parents are told proactively when the summary updates, rather than having to log in |

---

*This document reflects the code as of the initial prototype. Check `PROJECT_NOTES.md` for design decisions and `config.py` for model and path configuration.*
