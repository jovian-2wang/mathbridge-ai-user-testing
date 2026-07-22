# MathBridge AI — Project Notes

**Last updated:** 2026-05-26
**Status:** Prototype scaffold complete — ready for first run

---

## Overview

MathBridge AI is a connected math learning companion that turns a single student tutoring session into three different views: a student chat interface, a teacher analytics dashboard, and a plain-language parent summary.

UI/UX target: `Finders Foundry_2026/mock prototype.html`

---

## Architecture

```
Math_tutor_app/
├── app.py                    # Entry point. Nav routing via session_state.view
├── config.py                 # Paths, model name, color palette (matches mockup)
├── requirements.txt
├── .env.example
│
├── ui/                       # Streamlit views — display only, no AI logic
│   ├── landing.py            # Role-selection cards
│   ├── student.py            # Chat interface + live signals sidebar
│   ├── teacher.py            # Mastery map + insights panel
│   └── parent.py             # Weekly summary + encouragement note
│
├── ai/                       # All LangChain / OpenAI logic
│   ├── tutor_chain.py        # Builds message list → calls OpenAI → returns response
│   ├── rag.py                # Keyword retrieval over data/curriculum/*.txt
│   └── signals.py            # Extracts misconceptions/engagement after each turn
│
├── memory/
│   └── student_memory.py     # Load/save per-student JSON in data/memory/
│
└── data/
    ├── curriculum/           # Drop .txt files here to extend the RAG corpus
    │   └── fractions_grade4.txt
    └── memory/               # Auto-created; one JSON file per student
```

### Data flow (student turn)

```
User types → student.py._reply()
  → tutor_chain.get_tutor_response()
      → rag.get_curriculum_context()   (keyword search over .txt files)
      → ChatOpenAI.invoke()            (OpenAI gpt-4.1-mini)
  → signals.extract_signals()          (second OpenAI call, temp=0)
  → session_state updated → st.rerun()
```

### Key design choices

| Concern | Decision | Rationale |
|---|---|---|
| LLM | `gpt-4.1-mini` via `langchain-openai` | Single import; model swappable in `config.py` |
| RAG | Keyword search over `.txt` files | No embedding model required in prototype phase |
| Memory | JSON files per student | Simple, inspectable, no database dependency |
| Routing | `session_state.view` + `st.rerun()` | Single `app.py`; avoids Streamlit multipage complexity |
| Signal extraction | Separate Claude call after each turn | Clean separation; can be disabled to cut API cost |
| Venv | `math/` (Python 3.11, already exists in repo) | Reuses existing environment |

---

## Implemented Features

### Student view
- [x] Chat interface with `st.chat_message` bubbles
- [x] Three quick-action buttons: Give me a hint / Show steps / Similar problem
- [x] `st.chat_input` for free-form typing
- [x] "Behind the Scenes" sidebar showing live signals (concept, misconception, hints used, engagement, next support)
- [x] RAG-augmented system prompt (curriculum context injected per turn)
- [x] Encouraging, concise tutor persona (2–4 sentence response target)

### Teacher dashboard
- [x] Four metric tiles: Topic Focus, Mastery Trend, Support Needed, Engagement
- [x] Student Mastery Map with color-coded progress bars (🟢 ≥75%, 🟡 ≥50%, 🔴 <50%)
- [x] Key Insights & Suggested Actions panel (loaded from student memory)
- [x] Live signal injection: detected misconceptions appear as a top insight card

### Parent summary
- [x] Weekly summary loaded from student memory
- [x] Three colored boxes: What went well / Needs support / Try at home
- [x] Encouragement note section

### Infrastructure
- [x] `config.py` with all paths and the mockup color palette
- [x] `student_memory.py` with load/save/default scaffold (auto-creates `data/memory/<id>.json`)
- [x] `.env` / `ANTHROPIC_API_KEY` check with friendly error on missing key
- [x] `fractions_grade4.txt` curriculum seed document for RAG

---

## Pending Tasks

### Short-term (next session)
- [ ] **Run and smoke-test** the full app end-to-end with a real OpenAI API key
- [ ] **Student selector** — add a name/ID input widget so the teacher and parent views can switch between students (currently hardcoded to `"alex"`)
- [ ] **Persist signals to memory** — after each session, call `student_memory.update_signals()` and `save_session()` so the teacher dashboard shows real data over time
- [ ] **Mastery score updater** — increment/decrement mastery scores based on signal output from each session

### Medium-term
- [ ] **Upgrade RAG to vector search** — swap `ai/rag.py` keyword retrieval for FAISS + `sentence-transformers` (`all-MiniLM-L6-v2`) once more curriculum docs are added
- [ ] **Expand curriculum corpus** — add `.txt` files for other grade 4 topics (multiplication, division, geometry, measurement)
- [ ] **Parent summary generator** — add `ai/summary.py` to auto-generate the weekly summary from accumulated session signals instead of using static defaults
- [ ] **Teacher insight generator** — add `ai/insights.py` to generate fresh insights from the week's signals on demand
- [ ] **Session history in student view** — show "Previous sessions" count or last topic studied

### Future / agent expansion hooks
- [ ] **Multi-student teacher view** — list all students, filter by mastery level or misconception type
- [ ] **Hint tracking** — count hint button presses per session and store in memory
- [ ] **Progress over time charts** — use `st.line_chart` to show mastery score trends across sessions
- [ ] **Export** — teacher can download a CSV of class mastery data
- [ ] **Agent framework** — replace `tutor_chain.py` with a LangChain agent that can call tools (e.g., generate a new problem, look up a definition)

---

## Known Issues

| # | Issue | Severity | Notes |
|---|---|---|---|
| 1 | Signals sidebar renders with placeholder data until first chat turn | Low | Expected for cold start; acceptable in prototype |
| 2 | Teacher dashboard mastery scores are static defaults until `save_session()` is wired up | Medium | Data flow is scaffolded; just needs the session-end hook |
| 3 | `extract_signals()` makes a second OpenAI call every turn — doubles API cost | Low | Acceptable for prototype; add a `SIGNALS_ENABLED` flag in `config.py` to toggle off |
| 4 | Student ID is hardcoded to `"alex"` across all views | Medium | Need a UI widget for student selection |
| 5 | `st.chat_input` is sticky at page bottom — conflicts visually with the two-column layout on narrow screens | Low | Streamlit limitation; document for stakeholders |
| 6 | Keyword RAG has no relevance threshold — very short queries may return unrelated paragraphs | Low | Add a minimum score filter (`score >= 2`) if noise becomes a problem |

---

## How to Run

```bash
# 1. Set up your API key
copy .env.example .env
# Edit .env and paste your OPENAI_API_KEY

# 2. Install dependencies (one-time if langchain-openai is missing)
math\Scripts\pip.exe install langchain-openai python-dotenv

# 3. Launch
math\Scripts\streamlit.exe run app.py
```

---

## Adding Content

**New curriculum topic** — drop a `.txt` file in `data/curriculum/`. The RAG picks it up automatically on the next query.

**New student** — call `load_student("student_name")` anywhere; the JSON is auto-created with default data.

**New view** — add `ui/newview.py` with a `render()` function, import it in `app.py`, add it to `VIEWS`/`VIEW_KEYS`.
