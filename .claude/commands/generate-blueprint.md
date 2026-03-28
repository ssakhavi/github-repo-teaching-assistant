You are the Blueprint Generator for the GitHub Training Assistant.

Generate a detailed lesson blueprint for the NEXT lesson. The blueprint drives all downstream generation.

## Step 1: Determine Repository

**If $ARGUMENTS is provided:** use it as the repo name. Verify `input/<repo_name>.txt` exists.

**Otherwise:** Use the Glob tool to list `input/*.txt` files.
- No files: stop — "No repo txt files found in input/."
- One file: use its stem automatically.
- Multiple files: list them and ask: "Run /generate-blueprint <repo-name>"

All paths use `training/<repo_name>/` as the base.

## Step 2: Determine the Next Lesson Number

Use the Glob tool to list all directories matching `training/<repo_name>/lessons/lesson_*`.
- No directories: next lesson is **1**.
- Otherwise: next lesson is (highest existing N) + 1.

## Step 3: Read the Curriculum Plan

Read `training/<repo_name>/curriculum.md`. Find the entry for **Lesson N** and note:
- Structural focus, key files, core concepts, what NOT to cover, exercise ideas, MCQ topics, prerequisites

If `curriculum.md` does not exist, stop:
> "curriculum.md not found. Run /generate-curriculum first, or /next-lesson which runs it automatically."

## Step 4: Read Relevant Repository Code

Read `input/<repo_name>.txt` — only the files listed in the Lesson N "Key files to read" entry. Do not read the whole file. Understand those files in enough detail to write accurate lesson content, exercises, and MCQ questions.

## Step 5: Read Previous Results (if any)

Use the Glob tool to find `training/<repo_name>/results/lesson_*_results.json`. If any exist, read the most recent.

**Interpret performance signals:**

Coding examples:
- `passed: true` → understood
- `attempts > 1` and `passed: true` → struggled; consider brief reinforcement
- `passed: false` → **flag for reinforcement**
- `attempts == 0` (skipped) → **treat as "learner does not know this concept"** — strongest signal

MCQ questions:
- `correct: true` → knows this fact
- `answer == "E"` (I don't know) → **treat as "learner explicitly does not know this"**
- `correct: false` and `answer != "E"` → wrong guess; mild signal

**Interpret written feedback** (`feedback.examples` and `feedback.mcq` fields):
- If either field is non-null, read it carefully. The learner has expressed something directly.
- Extract specific confusion points, questions, or topics they want more of.
- Treat explicit confusion ("I didn't understand X") as equivalent to a skipped exercise — that concept needs reinforcement.
- Treat explicit interest ("I'd like more on Y") as a signal to weight that topic in the next lesson if it fits the curriculum.
- Treat questions about things not yet covered ("what about Z?") as a note to ensure Z appears in a future lesson — record it in Progression Notes.

**Calibration:**
- ≥ 80% MCQ correct (excluding E) AND all examples passed AND no confusion in feedback: follow curriculum as-is
- Any examples skipped OR any "E" answers OR any expressed confusion: add a **Reinforcement** section
- > 30% "E"/wrong OR > 1 skipped OR substantial confusion in feedback: consider covering the same curriculum module more deeply rather than advancing

Also read `training/<repo_name>/results/lesson_0_background.json` to remind yourself of the learner's experience level.

## Step 6: Write the Blueprint

Create `training/<repo_name>/lessons/lesson_N/` and write `blueprint.md`:

```markdown
# Lesson N Blueprint: [Title from curriculum]

## Lesson Number
N

## Repository
<repo_name>

## Topic Title
[From curriculum]

## Lesson Overview
[2–3 sentences: what this covers, why it matters, how it builds on previous]

## Learning Objectives
1. [Specific, measurable — tied to structural focus]
2. [Specific objective]
3. [Specific objective]

## Key Concepts
[From curriculum "Core concepts to teach" — with one-line explanations citing actual file/function]
- **[Concept]**: [explanation]
(4–8 concepts)

## Lesson Content Outline
1. [Section title] — [description]
2. [Section title] — [description]
3. [Section title] — [description]
4. Summary and key takeaways

## Code Focus
[From curriculum "Key files to read"]
- `path/to/file.py` — [what to highlight]
- `ClassName.method()` — [what to explain]

## What NOT to Cover
[From curriculum "What NOT to cover" — topics reserved for future lessons]
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- [concept/module] → covered in Lesson X
- [concept/module] → covered in Lesson Y

## Example Exercise Ideas
[From curriculum "Exercise ideas" — make these concrete]

### Exercise 1: [Title] — fill_in_blank
- **Concept tested:** [from this lesson only]
- **Function to base it on:** [exact function from Code Focus files]
- **What blanks to use:** [description of logic to omit]

### Exercise 2: [Title] — write_whole
- **Concept tested:** [from this lesson only]
- **Description for learner:** [what they implement]
- **Function to base it on:** [exact function/pattern]

## MCQ Topic Areas
[From curriculum "MCQ topics"]
1. [Specific testable fact]
2. [Another fact]
3. [Conceptual "why" question]
4. [Code-reading question]
5. [Application question]

## Prior Knowledge Available
[What learner knows from previous lessons, from curriculum prerequisites]
- [Concept from lesson N-1]
- [Concept from lesson N-2 if relevant]
(or "None — first lesson")

## Performance Notes
[Only if previous results exist]
- Examples skipped (attempts == 0): [list exercise IDs and their concepts]
- "I don't know" MCQ answers: [list question IDs and topics]
- Adjustments made: [what was reinforced or slowed down]
(or "No previous results — first lesson")

## User Feedback
[Only if previous results contain non-null feedback fields]
- Exercises feedback: "[verbatim text from feedback.examples]"
- MCQ feedback: "[verbatim text from feedback.mcq]"
- Concepts to reinforce based on feedback: [list]
- Topics the learner is curious about: [list — to plan for future lessons if not yet covered]
(or "No written feedback provided")

## Progression Notes
[Next lesson per curriculum; what this lesson sets up]
```

Confirm the file path and the lesson number and title generated.
