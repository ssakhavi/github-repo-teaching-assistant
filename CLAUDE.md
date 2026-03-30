# GitHub Training Assistant — Claude Code Instructions

## Project Overview

An AI-powered CLI tool that generates interactive lessons from any GitHub repository. Claude Code skills (slash commands) generate structured lesson content; a Python TUI presents it interactively.

## Package Manager

Always use `uv` — never `pip` or `pip install`.

```bash
uv add <package>       # add dependency
uv run python <file>   # run scripts
```

## Linting, Formatting, and Type Checking

Run after every code change:

```bash
uv run ruff check --fix src/   # lint and auto-fix
uv run ruff format src/        # format
uv run ty check src/           # type check
```

All three commands must pass with no errors before considering a change done.

## Directory Structure

```
input/                          # Drop <repo-name>.txt files here (git-ingest output)
                                # Also drop cv.pdf / resume.txt / cover_letter.pdf here for KYU
training/<repo-name>/
  curriculum.md                 # Lesson sequence plan
  lessons/lesson_N/
    blueprint.md                # Lesson plan (drives all generation)
    lesson.md                   # Paginated lesson text
    examples.json               # Coding exercises
    mcq.json                    # Multiple choice questions
  results/
    lesson_0_background.json    # Learner background profile
    kyu.md                      # Know Your User profile (optional, from CV/resume)
    lesson_N_results.json       # Per-lesson performance + feedback
src/
  terminal_interface.py         # Main TUI (rich + readchar)
.claude/commands/               # Claude Code slash command definitions
```

## Slash Commands

| Command | Description |
|---|---|
| `/next-lesson [repo]` | Full orchestration: curriculum → blueprint → 3 parallel subagents |
| `/generate-blueprint [repo]` | Generate blueprint for next lesson |
| `/generate-curriculum [repo]` | Generate full curriculum from background + repo txt |
| `/generate-lesson [repo]` | Generate lesson.md from blueprint |
| `/generate-mcq [repo]` | Generate mcq.json from blueprint |
| `/generate-examples [repo]` | Generate examples.json from blueprint |
| `/generate-kyu [repo]` | Synthesise CV/resume/cover letter + Lesson 0 background into a personalised kyu.md profile |

## Running the TUI

```bash
# Ingest a repo first
gitingest https://github.com/user/repo --output input/my-repo.txt

# Start the lesson (triggers background assessment on first run)
uv run python src/terminal_interface.py
uv run python src/terminal_interface.py <repo_name>
uv run python src/terminal_interface.py <repo_name> <lesson_number>
```

## Workflow

1. User drops `<repo>.txt` into `input/`
2. `uv run python src/terminal_interface.py <repo>` — runs background assessment (lesson 0) if not yet done
2a. (Optional) Drop CV/resume/cover letter into `input/` (e.g. `cv.pdf`, `resume.txt`) → run `/generate-kyu` → creates `training/<repo>/results/kyu.md` for personalised lessons
3. `/next-lesson` in Claude Code — generates full lesson (curriculum if needed → blueprint → lesson + MCQ + examples in parallel)
4. TUI presents: paginated lesson → coding exercises → MCQ quiz → feedback collection → summary
5. Results saved to `training/<repo>/results/lesson_N_results.json`
6. Repeat from step 3 — blueprint reads prior results for adaptive calibration

## Key Design Principles

- **Progressive scope**: MCQ and exercises never reference concepts not yet taught. Each blueprint has a "What NOT to Cover" section.
- **Adaptive calibration**: Blueprint generator reads prior results. Skipped exercises (`attempts == 0`) and "I don't know" MCQ answers (`answer == "E"`) trigger reinforcement in the next lesson.
- **Written feedback loop**: TUI collects free-text feedback after exercises and MCQ; stored in results JSON; blueprint generator reads and acts on it.
- **Multi-repo support**: All paths rooted at `training/<repo-name>/`; repo name derived from `input/<repo>.txt` stem.

## Results JSON Format

```json
{
  "repo": "fastapi",
  "lesson": 1,
  "started_at": "2024-01-01T00:00:00",
  "examples": [
    {"id": 1, "passed": true, "attempts": 2, "skipped": false}
  ],
  "mcq": [
    {"id": 1, "answer": "B", "correct": true, "i_dont_know": false}
  ],
  "feedback": {
    "examples": "I didn't understand why X works this way",
    "mcq": null
  }
}
```

## Dependencies

- `rich>=14.3.3` — Terminal UI (panels, markdown, syntax highlighting, tables)
- `readchar>=4.2.1` — Single-keypress input
