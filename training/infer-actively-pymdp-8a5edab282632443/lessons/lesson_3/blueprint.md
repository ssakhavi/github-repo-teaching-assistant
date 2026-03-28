# Lesson 3 Blueprint: Building an Observation Model (The A Matrix)

## Lesson Number
3

## Repository
infer-actively-pymdp-8a5edab282632443

## Topic Title
Building an Observation Model (The A Matrix)

## Lesson Overview
This lesson introduces the first and most important matrix in a POMDP: the A matrix, also called the observation likelihood or observation model. We build on Lesson 2's key insight — that probability distributions are arrays that sum to 1 — by showing that the A matrix is simply a collection of those distributions arranged as a 2D (or multi-dimensional) array. Each column is a distribution that answers one question: "If the world is in this hidden state, what would I observe?" By the end, the learner can read an A matrix, understand its shape, and know how to construct one using pymdp utilities.

## Learning Objectives
1. Explain in plain language what the A matrix represents in a POMDP.
2. Interpret the shape of an A matrix tensor given `num_obs` and `num_states`.
3. Read a column of A and state what it means ("given hidden state X, the probability of each observation is...").
4. Create a random A matrix using `random_A_array` from `pymdp/utils.py`.
5. Create a uniform A matrix using `list_array_uniform`.

## Key Concepts

- **Observation model (A matrix)**: The A matrix encodes how the agent's observations depend on the hidden state of the world. If the hidden state is "location 2", A tells you the probability of seeing each possible observation given that location. It is also called the "likelihood" because it tells you how likely each observation is under each state.
- **Observation modality**: A type of sensory input. For example, an agent might receive two types of observation simultaneously — a visual signal and an auditory signal. Each modality has its own A sub-matrix: `A[0]` for modality 0, `A[1]` for modality 1.
- **`num_obs`**: A Python list where each element is the number of distinct observations for one modality. For example, `num_obs = [3, 2]` means modality 0 has 3 possible observations and modality 1 has 2.
- **Hidden state factor**: A dimension of the hidden state. The world might have multiple independent hidden variables — for example, an agent's location AND the colour of a cue. Each is a "factor". `num_states = [4, 2]` means factor 0 has 4 possible states and factor 1 has 2.
- **Shape of `A[m]`**: For modality `m`, the shape is `(num_obs[m], num_states[0], num_states[1], ...)`. The first axis is the observation axis; the remaining axes index the hidden state factors. So `A[0].shape = (3, 4)` means modality 0, 3 observations, 4 hidden states.
- **Columns must sum to 1**: Each "column" of A (a slice along the hidden state axes) is a probability distribution over observations. This is the Lesson 2 rule: distributions sum to 1. `A[0][:, s]` must sum to 1 for every hidden state `s`.
- **`random_A_array(key, num_obs, num_states)`**: Creates a list of random A tensors — one per modality — where each column is a valid (Dirichlet-sampled) probability distribution. Requires a JAX PRNG key.
- **`list_array_uniform(shape_list)`**: Creates a list of uniform distributions for each given shape. Each element of the result is `norm_dist(jnp.ones(shape))` — i.e. every outcome is equally likely. Useful as a neutral starting point when you don't want to bias the agent.
- **`list_array_zeros(shape_list)`**: Creates a list of all-zero arrays for each shape. You then fill in the values manually and normalise. Used when you know the exact observation mapping.
- **JAX vs NumPy (reinforcement)**: Recall from Lesson 2 Q3: `jax.numpy` (imported as `jnp`) provides nearly the same API as `numpy` but can run on GPU/TPU. pymdp's modern utilities (`random_A_array`, `list_array_uniform`) all return JAX arrays, not NumPy arrays.

## Lesson Content Outline
1. **What is the A matrix?** — The question it answers, the analogy (sensor model, likelihood), why it sits between hidden states and observations.
2. **Reading A: shape and what each axis means** — `num_obs`, `num_states`, axis layout, reading `A[0][:, s]` as "the distribution over observations given hidden state s".
3. **Building A with `random_A_array`** — The real function signature, what a JAX PRNG key is (briefly), the returned list, how to inspect shape.
4. **Building A with `list_array_uniform` and `list_array_zeros`** — When you want full control; the `list_array_zeros` + fill-in pattern; how `list_array_uniform` gives a neutral start.
5. **Summary and what's next** — Recap, columns-sum-to-1 reinforcement, preview of Lesson 4 (B matrix: how hidden states transition).

## Code Focus
- `pymdp/utils.py` — `random_A_array` function (lines ~17432–17470 in the repo txt)
- `pymdp/utils.py` — `list_array_uniform` function (lines ~17559–17576)
- `pymdp/utils.py` — `list_array_zeros` function (lines ~17579+)

## What NOT to Cover
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- B matrix (transition model) → covered in Lesson 4
- C vector (preferences) → covered in Lesson 5
- D vector (prior beliefs) → covered in Lesson 5
- Distribution class → covered in Lesson 6
- State inference / FPI / belief updating → covered in Lesson 7
- Policy inference / EFE → covered in Lesson 8
- Environment code → covered in Lessons 9–10
- Dirichlet distribution math internals → covered in Lesson 11
- MCTS → covered in Lesson 12
- `resolve_a_dependencies` or `A_dependencies` parameter — too advanced for this lesson
- `validate_normalization` — too advanced
- `equinox`, JIT, vmap, grad — not needed yet
- `factor_dot` or anything in `pymdp/maths.py` — save for later
- Multi-modality A with cross-factor dependencies — one simple case first

## Example Exercise Ideas

### Exercise 1: Read an A matrix column — fill_in_blank
- **Concept tested:** Reading A as a conditional distribution (columns sum to 1)
- **Function:** `def read_A_column(A_matrix, hidden_state_index)` — takes a 2D list-of-lists (representing A) and a hidden state index; returns the column as a list (i.e. the probability of each observation given that hidden state)
- **Template:** `n_obs = len(A_matrix)` already given; the ONE blank is `return [A_matrix[obs][hidden_state_index] for obs in range(n_obs)]`
- **Use only Python built-ins** (lists of lists, range, len)
- **Tests (3):** Correct column for index 0, correct column for index 1, correct length

### Exercise 2: Check if A matrix columns are valid — fill_in_blank
- **Concept tested:** The rule that every column of A must sum to 1
- **Function:** `def all_columns_valid(A_matrix)` — takes a 2D list-of-lists; returns True if all columns sum to approximately 1.0 (within 0.001)
- **Template:** `n_states = len(A_matrix[0])` and `n_obs = len(A_matrix)` already given; loop scaffold already written with one blank for the column-sum check
- **The ONE blank:** `col_sum = sum(A_matrix[obs][s] for obs in range(n_obs))`
- **Tests (4):** Valid A (columns sum to 1), invalid A (one column off), single-column A, all-uniform A

## MCQ Topic Areas
1. What does the A matrix encode in a POMDP? (the probability of each observation given each hidden state)
2. What is an "observation modality"? (a distinct type of sensory input; agent can have multiple)
3. Given `num_obs = [3]` and `num_states = [4]`, what is `A[0].shape`? (3, 4)
4. Why must each column of A sum to 1? (it is a conditional probability distribution — reinforces Lesson 2)
5. What does `list_array_uniform` return compared to `random_A_array`? (uniform vs. random distributions)
6. What does `A[0][:, 2]` represent if `num_obs = [3]` and `num_states = [4]`? (probabilities of each of 3 observations when in hidden state 2)
7. JAX vs NumPy reinforcement (Q3 from Lesson 2 was wrong): What does `jax.numpy` (jnp) have in common with `numpy`? (nearly identical API, but can run on GPU/TPU — not a different language, not slower)

## Prior Knowledge Available
- Lesson 1: Active inference vocabulary, POMDP components named, perception-action loop
- Lesson 2: Arrays as probability distributions, normalisation (norm_dist), lists of arrays for multiple factors, JAX vs NumPy basics (Q3 was wrong — reinforce here)

## Performance Notes
**Lesson 2 results summary:**
- MCQ: 7/8 correct (87.5%) — strong overall
- MCQ Q3 (JAX vs NumPy relationship): answered **incorrectly** — this concept needs direct reinforcement
- Exercise 1: attempted once, failed (69 seconds — quick attempt)
- Exercise 2: attempted once, failed (982 seconds — tried for 16 minutes)
- Both exercises attempted (no skips) — engagement is improving

**Calibration decisions for Lesson 3:**
1. Exercises: still 0/2 passing after two lessons. Keep only ONE blank per exercise, with all intermediate values pre-computed. The exercises should use only Python built-ins (list of lists). Do NOT use numpy or jax in the exercise code — the learner can't install/use them in the sandbox.
2. Generate only 2 exercises (matching Lesson 2 — the learner needs achievable targets).
3. Include a MCQ question that directly reinforces the JAX vs NumPy concept (Q3 from Lesson 2).
4. The lesson text should explicitly connect columns of A to the Lesson 2 concept of probability distributions.

## User Feedback
No written feedback provided for Lesson 2.

## Progression Notes
Lesson 4 introduces the B matrix — the transition model. The B matrix follows the same pattern as A: its "slices" (indexed by action) are also collections of probability distributions, where each column is a distribution over next states. The "columns must sum to 1" rule learned here applies directly.
