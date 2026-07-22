# MathBridge AI Demo Script

## Positioning

**MathBridge AI is a contextualized, curriculum-grounded math tutoring platform.** It adapts Socratic support using learner profiles, Grade 6 curriculum objectives, real-life contexts, visual explanations, answer evaluation, and learning analytics.

The demo is designed to show a polished vertical slice rather than a generic chatbot.

## Core demo claim

MathBridge combines four kinds of context:

1. **Learner context**: grade level, confidence, language preference, learning style, interests, real-life contexts, and learning needs.
2. **Curriculum context**: Grade 6 topic, retrieved lesson evidence, and skill bucket alignment.
3. **Problem context**: the real-world scenario already present in the student's problem.
4. **Learning state context**: concept, misconception, hint usage, engagement, mastery movement, and next support.

## Suggested live demo flow

### 1. Student side: Contextualization Setup

Login as a student such as Liam or Maya.

Show the **Contextualization Setup** panel:

- Learner profile
- Real-life contexts
- Learning needs
- Curriculum scope
- Contextualization strategy

Say:

> This is the contextualization layer. It tells the tutor who the learner is, what contexts are familiar, and how support should be framed.

### 2. Student side: Ask a math problem

Use one of these problems:

```text
A car travels 240 miles in 4 hours. What is the unit rate?
```

or

```text
The ratio of students who walk to total students is 10:15. What does the ratio mean?
```

Point out:

- The tutor responds Socratically instead of immediately giving the answer.
- The visual explanation appears with a stable template.
- The right panel shows **Context Used** and **Curriculum Grounding**.

### 3. Answer evaluation

For the car problem, enter:

```text
60 miles
```

Expected behavior:

- The answer evaluator should accept it as mathematically correct.
- The feedback should restate the complete unit: **60 miles per hour**.

Say:

> The evaluator focuses on mathematical meaning. It can accept incomplete but correct student wording and then restate the precise unit.

### 4. End session and update reports

Click **End Session & Update Reports**.

Explain that this saves:

- chat history
- current signals
- engagement
- mastery update
- teacher/parent/class reporting evidence

### 5. Teacher Dashboard

Open the Teacher Dashboard.

Show:

- learning signal cards
- mastery map
- AI Teacher Insights
- **Contextualized Teaching Insight**

Say:

> The teacher does not just see a score. The dashboard explains which contexts and support styles may help the student next.

### 6. Parent Summary

Open Parent Summary.

Show:

- What went well
- Next practice focus
- Try at home
- **Contextualized Home Practice**

Say:

> The parent view converts tutoring signals into a home-friendly activity with a question, what to listen for, and why that practice was suggested.

### 7. Class Overview

Open Class Overview.

Show:

- AI Class Insights
- **Contextualized Grouping Strategy**
- Student Learning Snapshot
- Average Mastery by Skill

Say:

> The class view groups students not only by score, but by shared learning need and useful real-life contexts.

## Short project summary

MathBridge AI demonstrates how a tutoring system can move from isolated chat responses to an explainable learning ecosystem:

```text
Learner Profile
→ Curriculum Grounding
→ Contextualized Socratic Tutor
→ Stable Visual Explanations
→ Answer Evaluation
→ Learning Signals
→ Teacher / Parent / Class Insights
```

## Good demo problems

```text
A car travels 240 miles in 4 hours. What is the unit rate?
```

```text
If 5 notebooks cost $15, what is the cost of one notebook?
```

```text
The ratio of students who walk to total students is 10:15. What does the ratio mean?
```

```text
The ratio of red balls to blue balls is 3:4. What does the ratio mean?
```

```text
How many 1/8 pieces fit in 3/4?
```

## Final wording for email or README

MathBridge AI is a contextualized, curriculum-grounded math tutoring platform. It combines learner profiles, curriculum retrieval, Socratic tutoring, stable visual templates, answer evaluation, and learning analytics to generate personalized support for students and actionable insights for teachers and parents.
