# Lesson 5 Blueprint: Preferences and Priors (The C and D Vectors)

## Lesson Number
5

## Repository
infer-actively-pymdp-8a5edab282632443

## Topic Title
Preferences and Priors (The C and D Vectors)

## Lesson Overview
This lesson completes the generative model by introducing the two remaining components: C (what the agent prefers to observe) and D (what the agent believes about where it starts). After covering A (observation model), B (transitions), you now add preferences and initial beliefs — making the model capable of "wanting" outcomes and having a prior expectation about the world. By the end of this lesson you'll have assembled a complete POMDP generative model: A + B + C + D together.

## Learning Objectives
1. Explain what the C vector encodes and why higher values mean stronger preference
2. Build a C vector using `obj_array_uniform` (no preference) and by setting custom log-probability values manually
3. Explain what the D vector encodes and construct one using `random_single_categorical` or manually as a probability distribution

## Key Concepts
- **C vector (preferences)**: An object array of 1-D arrays — one per observation modality — where each value expresses how strongly the agent prefers that observation outcome (`pymdp/utils.py: obj_array_uniform`)
- **Higher C = more preferred**: The agent will act to make high-C outcomes happen; values can be any real number (log-probability scale)
- **C as log-probability**: C values are not raw probabilities; they sit on a log scale so a value of 0 = neutral, positive = desired, negative = undesired
- **`obj_array_uniform(num_obs)`**: Creates a C array with all zeros — "no preference" — using the shape list `num_obs` (`pymdp/utils.py` line 23933)
- **D vector (prior beliefs over initial hidden states)**: An object array of 1-D probability distributions — one per hidden state factor — that encodes what the agent believes about its starting state before it has seen anything (`pymdp/utils.py: random_single_categorical`)
- **`random_single_categorical(shape_list)`**: Generates a random normalised 1-D categorical distribution for each factor and returns them in an object array (`pymdp/utils.py` line 24067)
- **D must sum to 1**: Each sub-array of D is a probability distribution over initial states — just like each column of A sums to 1, each D[f] must sum to 1
- **The complete generative model — A + B + C + D**: Once you have all four components, you have everything needed to define a POMDP agent's beliefs about the world, its goals, and its starting point

## Lesson Content Outline
1. Quick B Matrix Recap — revisit num_controls, action slices, and which factors are controllable (reinforcement for questions missed in Lesson 4)
2. Introducing C — What Does the Agent Want? — motivation, log-probability scale, shape matching num_obs
3. Building C in Practice — `obj_array_uniform` for neutral C, setting custom values, interpreting `C[0] = [0, 0, 2.0]`
4. Introducing D — Where Does the Agent Think It Starts? — motivation, shape matching num_states, connection to the prior
5. Building D in Practice — `random_single_categorical`, setting D manually, verifying it sums to 1
6. The Complete Generative Model — bringing A + B + C + D together, what you now have, a brief preview of what inference does with these four components
7. Summary and key takeaways

## Code Focus
- `pymdp/utils.py` line 23933 — `obj_array_uniform(shape_list)`: creates uniform (zero-preference) C arrays; each sub-array is normalised `np.ones(shape)`, so all outcomes equally likely/preferred
- `pymdp/utils.py` line 24067 — `random_single_categorical(shape_list)`: creates random normalised 1-D categoricals for D; normalises `np.random.rand(shape_i)` via `norm_dist`
- `pymdp/utils.py` line 23905 — `obj_array_zeros(shape_list)`: creates zero-filled object arrays; useful when building C manually from scratch
- `pymdp/utils.py` line 23933 — note that `obj_array_uniform` calls `norm_dist(np.ones(shape))` — emphasise that this returns 1/N for each outcome, not zeros

## What NOT to Cover
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- How C is used in Expected Free Energy (EFE) computation → covered in Lesson 8
- How D is updated during learning → covered in Lesson 11
- Policy inference math or `infer_policies()` → covered in Lesson 8
- The Distribution class (`pymdp/distribution.py`) → covered in Lesson 6
- Equinox, JAX tracing, or JIT compilation → out of scope for this tier
- Dirichlet distributions → covered in Lesson 11

## Example Exercise Ideas

### Exercise 1: Build a C Vector — fill_in_blank
- **Concept tested:** Creating a C vector manually using `obj_array_zeros` and assigning preferences for a specific outcome
- **Function to base it on:** `obj_array_zeros` in `pymdp/utils.py`
- **What blanks to use:** The call to `obj_array_zeros` to allocate the array, and the line that sets the preferred outcome to a positive log-probability value (e.g., 2.0)
- **Setup:** `num_obs = [3, 2]` — two modalities, first has 3 outcomes, second has 2; agent prefers outcome index 2 in modality 0

### Exercise 2: Build and Validate a D Vector — write_whole
- **Concept tested:** Creating a D prior distribution list (one array per factor) and verifying each sub-array sums to ~1.0
- **Description for learner:** Write `check_D_valid(D)` that returns True if every element of the object array D is a valid probability distribution (non-negative values, sums to approximately 1.0)
- **Function to base it on:** Pattern of `check_A_valid` / `check_B_valid` from prior lessons; uses `np.allclose(np.sum(d), 1.0)` and `np.all(d >= 0)`

## MCQ Topic Areas
1. What does the C vector represent in pymdp? (factual)
2. If `C[0] = [0, 0, 2.0]`, which outcome does the agent prefer and why? (conceptual — log scale)
3. Why are C values on a log-probability scale rather than a plain probability scale? (conceptual)
4. What does `obj_array_uniform(num_obs)` produce for C, and what does it mean for the agent's preferences? (code reading)
5. What does the D vector represent? (factual)
6. Why must each D[f] sum to 1? (conceptual — it's a probability distribution)
7. What is the full set of matrices/vectors that define a POMDP generative model in pymdp? (integration / A+B+C+D)

## Prior Knowledge Available
- Lesson 4: B matrix shape `(num_states, num_states, num_controls)`, action slices, controllable vs non-controllable factors, `random_B_matrix`
- Lesson 3: A matrix — observation model, `num_obs`, `num_states`, conditional distributions
- Lesson 2: Arrays as probability distributions, normalisation, `norm_dist`, object arrays
- Lesson 1: What an "agent" is, the observe → infer → act loop, POMDPs at a high level

## Performance Notes
- Examples skipped (attempts == 0): None — both examples passed on first attempt
- "I don't know" MCQ answers: Q6 — 1 question explicitly unknown
- Wrong guesses (not E): Q4, Q5, Q7 — 3 questions answered incorrectly with a wrong guess
- MCQ score: 4/8 = 50% — significantly below the 80% threshold
- Calibration decision: Examples were solid (both passed, first attempt), suggesting procedural/coding understanding is good. MCQ struggles are conceptual — specifically around the later B matrix questions (control factors, num_controls interpretation, action slices). Add a reinforcement section at the start of Lesson 5 that re-explains `num_controls`, what the third dimension of B means, and which factors are controllable, before introducing C and D.

## User Feedback
No written feedback provided.

## Progression Notes
- Next lesson per curriculum: **Lesson 6 — The Distribution Class** (`pymdp/distribution.py`)
- Lesson 5 sets up: The learner will now have a complete A + B + C + D model. Lesson 6 wraps this in the Distribution class so dimensions have human-readable names — a natural next step that makes the whole model easier to inspect and reason about.
- B matrix reinforcement in this lesson directly prepares the learner for policy inference (Lesson 8), where num_controls and action slices are used heavily.
