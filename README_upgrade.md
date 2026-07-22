# MathBridge AI

**Contextualized, curriculum-grounded math tutoring for students, teachers, parents, and class-level instruction.**

MathBridge AI is a polished showcase demo of a K–12 math support platform. The current version focuses on selected Grade 6 math topics to demonstrate the core framework: learner profile setup, curriculum grounding, Socratic tutoring, stable visual explanations, answer evaluation, learning-signal tracking, teacher insights, parent home practice, and class-level contextualized grouping.

---

## 1. Project Overview

MathBridge AI is designed to make math tutoring more adaptive and explainable. Instead of giving every student the same generic explanation, the system uses a **contextualization layer** that combines:

- learner profile and preferences
- real-life contexts and interests
- curriculum scope and learning objectives
- recent tutoring signals
- mastery estimates and misconceptions

The goal is to provide support that is both **personalized** and **curriculum-aligned**. The student receives Socratic guidance and visual explanations; teachers, parents, and class-level views receive summaries built from the same learning evidence.

---

## 2. Core Demo Claim

MathBridge AI demonstrates a contextualized learning loop:

```text
Learner profile + curriculum grounding
        ↓
Contextualized Socratic tutoring
        ↓
Answer evaluation + visual explanation
        ↓
Learning signals and mastery updates
        ↓
Teacher insight + parent home practice + class grouping
```

This makes the platform more than a chatbot. It is a structured educational system that connects student interaction data to actionable support for different stakeholders.

---

## 3. Key Features

### Student Experience

- Contextualization setup for learner profile, interests, learning style, real-life contexts, and learning needs.
- Socratic tutoring that guides students step by step rather than immediately revealing answers.
- Answer evaluation that checks student responses across different answer formats.
- Hybrid visual explanations for common Grade 6 math problem types.
- Behind-the-scenes panel showing curriculum grounding, context used, learning signals, and runtime information.

### Teacher Experience

- Teacher dashboard with student learning signals, mastery trends, and support recommendations.
- Contextualized teaching insight that connects learner context to practical classroom moves.
- Class overview with class-level metrics, small-group suggestions, and misconception alerts.
- Contextualized grouping strategy based on shared skill needs and real-life context patterns.

### Parent Experience

- Weekly learning summary written in parent-friendly language.
- What went well, next practice focus, and encouragement notes.
- Contextualized home practice that turns recent math focus into family-friendly examples.

---

## 4. Contextualization Layer

The contextualization layer is the main feature of MathBridge AI.

Students can enter or edit basic learning information, including:

- grade level
- language preference
- reading level
- math confidence
- learning style
- interests
- preferred real-life contexts
- learning needs
- curriculum scope

The tutor then uses this context carefully. If the problem already contains a real-life scenario, the system preserves that scenario. If the problem is abstract or underspecified, the system can use the learner profile to choose a familiar example.

Example:

```text
Problem: A car travels 240 miles in 4 hours. What is the unit rate?
Detected context: travel / distance
System behavior: preserve the travel context instead of forcing an unrelated preference.
```

Example:

```text
Problem: What does the ratio 10:15 mean?
Detected context: abstract math
System behavior: use learner preferences, such as sports, food, or shopping, when helpful.
```

This avoids forced personalization while still making math more connected to students' lives.

---

## 5. Curriculum Grounding

MathBridge AI includes curriculum-grounded support for selected Grade 6 topics. The current demo intentionally focuses on representative math areas rather than full curriculum coverage.

Current representative topics include:

- ratios
- unit rates
- fraction division
- whole-number division reasoning
- word problems
- coordinate-plane reasoning

The tutoring system retrieves and displays curriculum-related grounding so the response is not only personalized, but also connected to the intended math objective.

---

## 6. Hybrid Visual Explanation System

MathBridge AI uses a two-path visual system.

```text
Student problem
      ↓
Visual routing / problem-type detection
      ↓
If common Grade 6 type is detected:
      deterministic SVG visual template
Else:
      LLM visual planner fallback
```

### Deterministic visual templates

For common Grade 6 problem types, the system uses stable SVG templates. This improves speed and reliability compared with relying only on LLM-generated diagrams.

Supported deterministic visual types include:

- unit-rate tape diagrams
- ratio comparison diagrams
- division equal-group diagrams
- fraction division bar diagrams
- coordinate-plane diagrams

### LLM visual planner fallback

For open-ended or unmatched problems, the original LLM visual planner remains available as a fallback. This keeps the system flexible while prioritizing stable diagrams for high-frequency demo scenarios.

### Context-aware visual labels

The contextualization layer also improves visual labels. For example, a ratio problem can display labels such as:

```text
Students who walk / Total students
Red balls / Blue balls
```

instead of generic labels like:

```text
Part A / Part B
```

---

## 7. System Architecture

High-level flow:

```text
app.py
  ├── ui/login.py
  ├── ui/landing.py
  ├── ui/student.py
  │     ├── ai/tutor_chain.py
  │     ├── ai/answer_evaluator.py
  │     ├── ai/hint_agent.py
  │     ├── ai/deterministic_visuals.py
  │     ├── ai/visual_planner.py
  │     ├── ai/contextualization.py
  │     └── memory/student_memory.py
  ├── ui/teacher.py
  │     └── ai/insights.py
  ├── ui/parent.py
  │     └── ai/insights.py
  └── ui/class_overview.py
        └── ai/class_insights.py
```

Core data flow:

```text
Student interaction
      ↓
Tutor response + answer evaluation + visual explanation
      ↓
Learning signals are extracted
      ↓
Student memory is updated
      ↓
Teacher, parent, and class views summarize the same evidence
```

---

## 8. Demo Accounts

The demo includes role-based login.

Typical demo accounts:

```text
Student accounts: alex, liam, maya
Teacher account: teacher_demo
Parent accounts: parent_demo, alex_parent, liam_parent, maya_parent
Default password: 1234
```

The app also includes a demo password reset tab for testing.

---

## 9. How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 10. Recommended Demo Flow

### Step 1: Login page

Show that the platform supports separate student, teacher, and parent experiences.

### Step 2: Overview page

Explain the core idea:

```text
MathBridge AI is a contextualized, curriculum-grounded tutoring platform.
```

### Step 3: Student page

Open the contextualization setup and show the learner profile:

- interests
- real-life contexts
- learning style
- learning needs
- curriculum scope

Then ask a math question.

Recommended test problem:

```text
A car travels 240 miles in 4 hours. What is the unit rate?
```

Expected student answer:

```text
60 miles
```

The system should mark the response correct and restate the full unit rate.

### Step 4: Behind the Scenes

Show the right-side panel:

- curriculum grounding
- context used
- learning signals
- visual generation path
- timing information

### Step 5: End Session & Update Reports

Save the tutoring session so teacher, parent, and class views update.

### Step 6: Teacher Dashboard

Show contextualized teaching insight based on student signals and learner context.

### Step 7: Parent Summary

Show contextualized home practice and parent-friendly language.

### Step 8: Class Overview

Show class-level contextualized grouping strategy.

---

## 11. Recommended Test Problems

### Unit Rate

```text
A car travels 240 miles in 4 hours. What is the unit rate?
```

Answer:

```text
60 miles per hour
```

### Cost Per Item

```text
If 5 notebooks cost $15, what is the cost of one notebook?
```

Answer:

```text
3 dollars
```

### Ratio Meaning

```text
The ratio of students who walk to total students is 10:15. What does the ratio mean?
```

Answer:

```text
2:3
```

### Parent-Friendly Unit Rate Example

```text
If a basketball game lasts 48 minutes and the team scores 96 points, how many points do they score per minute?
```

Answer:

```text
2 points per minute
```

---

## 12. Planning and Development Scope

### Planning Stage

The current demo focuses on a selected set of Grade 6 topics rather than full curriculum coverage. This keeps the scope focused on proving the core framework:

- contextualized learner profile
- curriculum grounding
- Socratic tutoring
- answer evaluation
- stable visual explanations
- student memory and learning signals
- teacher, parent, and class-level insights

### Development Stage

Future development can expand curriculum coverage after the core framework is stable. This includes:

- adding more Grade 6 units
- expanding to broader K–12 math topics
- adding more curriculum files
- adding more problem types
- building more deterministic visual templates
- improving retrieval evaluation
- refining contextualized practice generation
- strengthening longitudinal mastery tracking

This staged approach makes the current version a strong proof of concept while leaving a clear path for continued development.

---

## 13. Current Limitations

- Curriculum coverage currently focuses on selected Grade 6 topics.
- Deterministic visuals cover common problem types; open-ended problems still use the LLM visual planner fallback.
- Mastery values and learning signals are useful for demo and formative support, but they are not formal assessment results.
- Parent and teacher insights are generated from recent tutoring interactions and should be interpreted as support recommendations.

---

## 14. Future Work

Potential next steps include:

- broader curriculum ingestion and topic coverage
- more robust problem-type routing
- richer assessment and progress reports
- stronger teacher controls for assigning practice
- improved student onboarding questionnaires
- more visual templates for geometry, statistics, and algebra-readiness topics
- classroom-level planning tools for differentiated instruction

---

## 15. Suggested Screenshots for Submission

Recommended screenshots:

1. Login page with role-based demo entry.
2. Overview page showing the contextualized learning loop.
3. Student contextualization setup.
4. Student tutoring example with visual explanation.
5. Behind-the-scenes context-used panel.
6. Teacher contextualized teaching insight.
7. Parent contextualized home practice.
8. Class overview contextualized grouping strategy.

Avoid using screenshots where visual labels overlap or the app is still loading.

---

## 16. One-Sentence Summary

MathBridge AI is a contextualized, curriculum-grounded math tutoring demo that connects student profile, Socratic support, visual explanations, answer evaluation, and learning signals across student, teacher, parent, and class-level views.
