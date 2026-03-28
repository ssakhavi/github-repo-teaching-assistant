# Lesson 4 Blueprint: Hidden States and Transitions (The B Matrix)

## Lesson Number
4

## Repository
infer-actively-pymdp-8a5edab282632443

## Topic Title
Hidden States and Transitions (The B Matrix)

## Lesson Overview
This lesson introduces the B matrix — the transition model that describes how hidden states change over time in response to actions. Before covering the B matrix itself, the lesson opens with a dedicated clarification of 2D and 3D list indexing in Python, directly addressing the learner's stated confusion about matrix operations. Once indexing is clear, B follows naturally: it is a 3D structure where each "action slice" is a 2D matrix whose columns are probability distributions over the next hidden state.

## Learning Objectives
1. Correctly index a 2D list using `matrix[row][col]` and a 3D list using `matrix[i][j][k]`.
2. Explain what the B matrix represents: the probability of the next hidden state given the current state and a chosen action.
3. Interpret the shape of a B tensor: `(num_next_states, num_current_states, num_actions)`.
4. Read `B[0][:, :, u]` as the transition matrix for action `u`.
5. Construct a simple B matrix as a Python list-of-lists-of-lists without any library imports.

## Key Concepts

- **Matrix indexing (clarification)**: In Python, a 2D matrix stored as a list-of-lists is accessed as `matrix[row][col]`. The first index selects the row (outer list), the second selects the column (inner list). For example, `M = [[0.9, 0.1], [0.1, 0.9]]`, then `M[0][1]` = 0.1 (row 0, column 1). This is the same pattern used in all pymdp exercises.
- **3D indexing**: A 3D structure (list of list of list) is accessed as `matrix[i][j][k]`. For the B matrix, the convention is `B_f[next_state][current_state][action]`. So `B_f[1][0][1]` means: "given we are in state 0 and take action 1, the probability of ending up in state 1".
- **Transition model (B matrix)**: B encodes how the world changes when the agent takes an action. For each factor f, `B[f]` is a 3D tensor with shape `(num_states_f, num_states_f, num_controls_f)`. The first axis is the next state, the second is the current state, and the third is the action.
- **Action slice**: `B[f][:, :, u]` is the transition matrix for action `u` — a 2D matrix. Reading a column of this matrix: `B[f][:, s, u]` is the distribution over next states when you are in state `s` and take action `u`. This column must sum to 1 (Lesson 2/3 rule applies here too).
- **Control factors**: A hidden state factor is "controllable" if actions can change its state. `num_controls = [4, 1]` means factor 0 has 4 possible actions (controllable), factor 1 has only 1 action — meaning it evolves on its own and the agent can't directly influence it.
- **Stay/move transitions**: A common pattern — action 0 = stay in current state (identity transition), action 1 = move to another state. The identity transition for state 0 looks like `B[f][:, 0, 0] = [1.0, 0.0]` (100% probability of staying in state 0).
- **`random_B_array(key, num_states, num_controls)`**: Creates a list of random B tensors — one per factor — where each column of each action-slice is a valid distribution. Requires a JAX PRNG key.
- **`create_controllable_B(num_states, num_controls)`**: Creates a deterministic B where each action directly sets the next state. Useful for simple grid-world environments.

## Lesson Content Outline
1. **Matrix indexing clarification** — Explicitly correct the misconception from Lesson 3 feedback: `matrix[row][col]` for 2D, `matrix[i][j][k]` for 3D. Use a tiny concrete example before any B matrix content.
2. **What is the B matrix?** — The question it answers ("what state will I be in?"), the analogy (a rulebook for every action), how it extends A from "observation model" to "transition model".
3. **Reading B: shape and action slices** — The `(next_state, current_state, action)` axis layout, reading `B[0][:, :, 0]` as the first action's transition matrix, the "columns sum to 1" rule.
4. **Building B: stay/move example** — A concrete 2-state 2-action B matrix built by hand as a Python list, `random_B_array` and `create_controllable_B` shown with real source code.
5. **Summary and what's next** — Recap, indexing rule summary, preview of Lesson 5 (C and D: preferences and priors).

## Code Focus
- `pymdp/utils.py` — `random_B_array` function
- `pymdp/utils.py` — `create_controllable_B` function

## What NOT to Cover
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- C vector (preferences) → covered in Lesson 5
- D vector (prior beliefs) → covered in Lesson 5
- Distribution class → covered in Lesson 6
- State inference / FPI → covered in Lesson 7
- Policy inference / EFE → covered in Lesson 8
- Environment code → covered in Lessons 9–10
- Dirichlet distribution math → covered in Lesson 11
- MCTS → covered in Lesson 12
- `resolve_b_dependencies` or `B_dependencies` parameter — too advanced
- `B_action_dependencies` — too advanced
- equinox, JIT, vmap — not needed yet
- Multi-factor B with cross-factor dependencies — keep to single-factor examples

## Example Exercise Ideas

### Exercise 1: Index into a 3D B matrix — fill_in_blank
- **Concept tested:** 3D list indexing `B[next_state][current_state][action]` — directly addresses the matrix indexing confusion
- **Function:** `def get_transition_prob(B_matrix, next_state, current_state, action)`
- **Template:** All variables named, ONE blank is `return B_matrix[next_state][current_state][action]`
- **Pre-computed in template:** Nothing needed — the blank IS the return, it's a single lookup
- **Use only Python built-ins** (list of list of list)
- **Tests (3):**
  - A 2×2×2 B matrix (2 states, 2 actions); verify `get_transition_prob(B, 0, 0, 0)` returns 1.0 (stay-in-state-0 action)
  - Same B; verify `get_transition_prob(B, 1, 0, 1)` returns 1.0 (move-from-0-to-1 action)
  - Same B; verify `get_transition_prob(B, 0, 0, 1)` returns 0.0 (move action doesn't stay)

### Exercise 2: Check if a B action-slice column is valid — fill_in_blank
- **Concept tested:** Columns of a B action-slice must sum to 1 (same "columns sum to 1" rule from Lessons 2–3)
- **Function:** `def column_sums_to_one(B_action_slice, current_state)`
- **B_action_slice** is a 2D list-of-lists `[next_state][current_state]`; we check whether the column at `current_state` sums to ~1.0
- **Template:** `n_next = len(B_action_slice)` already computed; ONE blank is `col_sum = sum(B_action_slice[ns][current_state] for ns in range(n_next))`; return statement already provided: `return abs(col_sum - 1.0) < 0.001`
- **Tests (3):**
  - `[[1.0, 0.0], [0.0, 1.0]]` column 0 → True (identity matrix, first column sums to 1)
  - `[[0.8, 0.3], [0.2, 0.7]]` column 0 → True
  - `[[0.8, 0.3], [0.3, 0.7]]` column 0 → False (0.8 + 0.3 = 1.1)

## MCQ Topic Areas
1. What does the B matrix represent in a POMDP? (transition model — how states change with actions)
2. In Python, how do you access row 1, column 2 of a 2D list `M`? (M[1][2]) — directly targets the indexing confusion
3. What is the shape of `B[0]` if `num_states = [3]` and `num_controls = [2]`? (3, 3, 2)
4. What does `B[0][:, :, 0]` represent? (the transition matrix for action 0 — an action slice)
5. Why must each column of a B action-slice sum to 1? (it is a conditional probability distribution over the next state)
6. What does `num_controls = [4, 1]` tell you about which factors are controllable? (factor 0 has 4 actions; factor 1 has only 1 action and is not meaningfully controllable)
7. What is the difference between `random_B_array` and `create_controllable_B`? (random vs deterministic transitions)

## Prior Knowledge Available
- Lesson 1: Active inference vocabulary, POMDP components, perception-action loop
- Lesson 2: Arrays as distributions, norm_dist, lists of arrays for factors, JAX vs NumPy
- Lesson 3: A matrix as observation model, `A[m].shape = (num_obs[m], *num_states)`, columns of A sum to 1, `list_array_uniform`, `list_array_zeros`

## Performance Notes
**Lesson 3 results summary:**
- MCQ: 7/8 correct (87.5%) — consistent pattern, strong conceptual grasp
- MCQ Q3 wrong (answered B): "What does `A[0][:, 2]` represent?" — code reading / slice notation question
- Exercise 1: attempted once, failed (110 seconds)
- Exercise 2: attempted once, failed (165 seconds)
- 0/8 total exercises passed across all 3 lessons — consistent failure on coding

**Written feedback from exercises:** *"I think I have a wrong understanding of some of the matrix operations."*

**Calibration decisions for Lesson 4:**
1. **Matrix indexing is the lesson priority**: The first content page must explain `matrix[row][col]` and `matrix[i][j][k]` explicitly, with a tiny worked example BEFORE any B matrix material. This directly addresses the learner's stated confusion.
2. **Exercise 1 must test pure indexing**: Just `return B_matrix[next_state][current_state][action]` — a single lookup with no arithmetic. If the learner can't pass this, we know indexing itself is the blocker.
3. **Only ONE blank per exercise**, with all intermediate values pre-computed.
4. **2 exercises only**.
5. **MCQ Q2** must directly test `M[row][col]` notation to reinforce the indexing lesson.
6. **Lesson text** must show the step-by-step breakdown of `M[0][1]` before ever showing `B[0][:, :, 0]`.

## User Feedback
**From coding exercises (Lesson 3):** "I think I have a wrong understanding of some of the matrix operations."
**Interpretation:** The learner cannot correctly index into a 2D or 3D list. All three coding exercises to date have involved `matrix[row][col]` operations. This must be treated as the root cause of all exercise failures. The entire lesson must be reframed around building indexing intuition, with B as the motivating context.
**Topics the learner is curious about:** No additional curiosity signals provided.

## Progression Notes
Lesson 5 introduces the C vector (preferences) and D vector (prior beliefs). These are simpler structures — 1D arrays — so they don't involve the 2D/3D indexing complexity of A and B. This is intentionally a gentler lesson after the heavier indexing work in Lessons 3 and 4.
