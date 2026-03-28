# Lesson 2 Blueprint: Python Arrays and JAX Basics

## Lesson Number
2

## Repository
infer-actively-pymdp-8a5edab282632443

## Topic Title
Python Arrays and JAX Basics

## Lesson Overview
This lesson introduces the fundamental data structure underpinning all of pymdp: the probability distribution as a numerical array. We move gently from Python lists to NumPy arrays to JAX arrays, and arrive at the single most important utility function in the library: `norm_dist`. The goal is to build rock-solid intuition for what a probability distribution looks like in code and why normalisation matters — without rushing into any matrix-building yet.

## Learning Objectives
1. Explain the difference between a Python list, a NumPy array, and a JAX array.
2. State what it means for an array to be a valid probability distribution (non-negative, sums to 1).
3. Apply `norm_dist(dist)` to normalise an unnormalised array.
4. Explain why pymdp stores multiple distributions as a Python list (one element per factor) rather than one big 2D array.

## Key Concepts

- **NumPy array**: A fast numerical array from the `numpy` library. pymdp uses `numpy` for some utilities and the legacy backend.
- **JAX array**: A NumPy-compatible array that can run on GPUs/TPUs and supports automatic differentiation. pymdp's modern backend uses `jax.numpy` (imported as `jnp`). From a user's perspective, `jnp.array([1, 2, 3])` works almost identically to `np.array([1, 2, 3])`.
- **Probability distribution**: A 1D array of non-negative numbers that sum to exactly 1.0. For example, `[0.3, 0.5, 0.2]` is a valid distribution over 3 outcomes.
- **Unnormalised distribution**: An array with non-negative values that don't yet sum to 1. For example, `[3, 5, 2]` represents the same relative probabilities as `[0.3, 0.5, 0.2]` but must be normalised first.
- **`norm_dist(dist)`**: The core utility in `pymdp/utils.py`. It normalises an array by dividing each element by the array's total sum: `dist / dist.sum(0)`. This is the most-called function in all of pymdp.
- **`list_array_norm_dist(dist_list)`**: Applies `norm_dist` to every array in a Python list. Used when you have one distribution per hidden-state factor.
- **Factorised representation**: pymdp stores multi-factor distributions as a Python list — one array per factor — rather than a giant 2D matrix. This keeps each factor independent and avoids the combinatorial explosion that comes from multiplying all factors together. A list `[D_factor_1, D_factor_2]` is a factorised prior; each element is a small 1D array.
- **Epistemic value revisited**: In Lesson 1 you saw that agents are naturally curious. Under the hood, curiosity is computed from probability distributions — an agent tracks a distribution over where it might be, and chooses actions that make that distribution more peaked (less uncertain). Everything in pymdp starts with distributions, so this lesson is where it all begins.

## Lesson Content Outline
1. **From Python lists to NumPy/JAX arrays** — What they are, why we use them instead of lists, what `jax.numpy` means in practice.
2. **What is a probability distribution?** — The "must sum to 1" rule, the "must be non-negative" rule, examples with real numbers.
3. **Normalising with `norm_dist`** — How the function works, why we normalise (connecting random initialisation to valid distributions), the actual code in `pymdp/utils.py`.
4. **Factorised distributions: why a list?** — The combinatorial explosion problem, why pymdp stores one array per factor, `list_array_norm_dist` as the list-level version.
5. **Summary and what's next** — Recap, key takeaways, preview of Lesson 3 (the A matrix).

## Code Focus
- `pymdp/utils.py` lines `norm_dist` and `list_array_norm_dist` (lines ~17290–17318 in the repo txt)
- `pymdp/utils.py` import block to show `import jax`, `from jax import numpy as jnp`, `import numpy as np`

## What NOT to Cover
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- A matrix construction or shape (`random_A_array`) → covered in Lesson 3
- B matrix → covered in Lesson 4
- C and D vectors → covered in Lesson 5
- Distribution class (`pymdp/distribution.py`) → covered in Lesson 6
- State inference / FPI → covered in Lesson 7
- Policy inference / EFE → covered in Lesson 8
- Environment code → covered in Lessons 9–10
- Dirichlet distributions or learning → covered in Lesson 11
- MCTS → covered in Lesson 12
- JAX JIT, vmap, grad, or tracing internals — too advanced
- `random_A_array`, `random_B_array`, `random_factorized_categorical` — too early
- `dtype=object` numpy object arrays — only in legacy; don't introduce here
- `validate_normalization` — too advanced for this lesson
- `equinox` — not needed yet

## Example Exercise Ideas

### Exercise 1: Implement `norm_dist` yourself — fill_in_blank
- **Concept tested:** How normalisation works (divide by sum)
- **Function:** `def normalise(values: list) -> list` — takes a Python list of positive numbers, returns a new list where each element is divided by the total sum
- **Template:** Function signature + the sum computation already done, blank is the return expression using list comprehension
- **Tests:** Check that output sums to 1.0, check individual values, check it works for a list of length 2
- **Why it must be simple:** The learner couldn't complete any Lesson 1 exercises. Only one blank, and the sum is already computed for them.

### Exercise 2: Check if a distribution is valid — fill_in_blank
- **Concept tested:** What makes a valid probability distribution
- **Function:** `def is_valid_distribution(values: list) -> bool` — returns True if all values are >= 0 and the sum is approximately 1.0 (within 0.001)
- **Template:** Two conditions scaffolded in, blanks are just the two boolean expressions
- **Tests:** True for [0.3, 0.5, 0.2], False for [1, 2, 3], False for [-0.1, 0.6, 0.5]
- **Scaffolding:** Keep very minimal — one blank per condition, both conditions already named with a comment

## MCQ Topic Areas
1. What is the key property that makes an array a valid probability distribution? (sums to 1, non-negative)
2. What does `norm_dist(dist)` return? (dist divided by its sum)
3. What is `jax.numpy` in relation to `numpy`? (near-identical API, can run on GPU/TPU)
4. Why does pymdp store multiple distributions as a list rather than one 2D array? (factorised; combinatorial explosion avoided)
5. If `dist = [3, 1]`, what does `norm_dist(dist)` produce? ([0.75, 0.25])
6. What does "epistemic value" mean at a conceptual level? (reinforcement of Q6 from Lesson 1 — reducing uncertainty about hidden states)
7. What does `list_array_norm_dist` do compared to `norm_dist`? (applies norm_dist to each array in a list)

## Prior Knowledge Available
- Vocabulary from Lesson 1: hidden states, observations, factors, modalities, POMDP, active inference, perception-action loop, A/B/C/D matrix names (not details), generative model, epistemic value.

## Performance Notes
**Lesson 1 results summary:**
- MCQ: 9/10 correct — strong conceptual understanding of vocabulary.
- MCQ Q6 (epistemic value / curiosity): answered "I don't know" — this concept needs reinforcement.
- Exercise 1: attempted once, failed — struggled to write working Python code.
- Exercises 2 & 3: both skipped (attempts == 0) — effectively "doesn't know" these exercises.
- All 3 exercises failed, none passed.

**Calibration decisions for Lesson 2:**
1. Exercises must be much simpler than Lesson 1. Only one blank per exercise. The blank should be a short expression (one line), not a full block of logic.
2. Pre-compute intermediate values in the template (e.g. give `total = sum(values)` already) so the learner just fills in the final expression.
3. Limit to 2 exercises (not 3) — fewer but achievable.
4. Reinforce epistemic value in Lesson 2 content (Key Concepts section includes a "Epistemic value revisited" item connecting distributions to curiosity).

## User Feedback
**From coding exercises:** "I need more time to grasp the contents. One by one."
**Interpretation:** The learner wants a slower pace with concepts introduced individually, not in groups. Lesson 2 focuses on a single concept (probability distributions and normalisation) instead of layering multiple ideas. The lesson outline is deliberately linear and avoids tangents.

## Progression Notes
Lesson 3 introduces the A matrix — the observation likelihood. It builds directly on Lesson 2: the A matrix is a 2D array whose columns are probability distributions, so the "must sum to 1" rule learned here applies immediately.
