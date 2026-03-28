# Curriculum Plan — infer-actively-pymdp

## Repository Overview

`pymdp` is a Python library for simulating **Active Inference** agents in discrete state spaces, modelled as Partially-Observed Markov Decision Processes (POMDPs). Written in JAX (with a legacy NumPy layer), it lets researchers and engineers build cognitive agents that perceive, learn, and act by minimising free energy — a principled Bayesian framework originally developed in computational neuroscience. The library is published in the Journal of Open Source Software and is production-stable (v1.0.0).

## Learner Profile

- **Programming experience:** Beginner (< 1 year)
- **Codebase comfort:** 1/5
- **Learning goal:** Learn the technology
- **Special interests:** What types of problems can be solved with this repository
- **Pacing:** Small, focused lessons with heavy explanations and lots of scaffolding. Concepts are introduced one at a time and each lesson builds directly on the last. 12 lessons total — more lessons, less per lesson.

## Lesson Map

| #  | Title                                    | Structural Focus              | Key Files/Modules                        |
|----|------------------------------------------|-------------------------------|------------------------------------------|
| 1  | What is pymdp and Active Inference?      | README, docs, motivation      | README.md, setup.cfg, docs/index.rst     |
| 2  | Python Arrays and JAX Basics             | pymdp/utils.py foundations    | pymdp/utils.py                           |
| 3  | Building an Observation Model (A Matrix) | Generative model — A          | pymdp/utils.py, pymdp/maths.py           |
| 4  | Hidden States and Transitions (B Matrix) | Generative model — B          | pymdp/utils.py                           |
| 5  | Preferences and Priors (C and D)         | Generative model — C, D       | pymdp/utils.py                           |
| 6  | The Distribution Class                   | pymdp/distribution.py         | pymdp/distribution.py                   |
| 7  | Belief Updating: Where Am I?             | pymdp/inference.py, algos.py  | pymdp/inference.py, pymdp/algos.py       |
| 8  | Policy Inference: What Should I Do?      | pymdp/control.py              | pymdp/control.py                         |
| 9  | Environments: The World the Agent Lives In | pymdp/envs/env.py, tmaze.py | pymdp/envs/env.py, pymdp/envs/tmaze.py   |
| 10 | The Full Active Inference Loop           | pymdp/envs/rollout.py         | pymdp/envs/rollout.py                    |
| 11 | Learning: Updating Beliefs Over Time     | pymdp/learning.py             | pymdp/learning.py                        |
| 12 | Planning and Sophisticated Inference     | pymdp/planning/               | pymdp/planning/si.py, pymdp/planning/mcts.py |

---

## Detailed Lesson Plans

### Lesson 1: What is pymdp and Active Inference?

- **Structural focus:** Repository overview, motivation, real-world use cases
- **Key files to read:** `README.md`, `setup.cfg`, `docs/index.rst`, `paper/paper.md`
- **Core concepts to teach:**
  - What is Active Inference and the Free Energy Principle?
  - What is a POMDP (Partially-Observed Markov Decision Process)?
  - What kinds of problems does pymdp solve? (perception, decision-making, curiosity-driven exploration)
  - What is an "agent" in this context?
  - The high-level loop: observe → infer → act → repeat
  - How pymdp relates to neuroscience and AI
  - How to install pymdp (`pip install inferactively-pymdp`)
  - The two backends: JAX (modern, `pymdp/`) and legacy NumPy (`pymdp/legacy/`)
- **What NOT to cover:** Any math beyond intuition; A/B/C/D matrices in detail; JAX specifics; inference algorithms; environment code; Dirichlet distributions
- **Exercise ideas:**
  - fill_in_blank: Write a function that returns a list of three real-world domains where active inference agents could be applied (as strings)
  - fill_in_blank: Write a function that checks whether a string is a valid pymdp backend name ("jax" or "legacy")
- **MCQ topics:**
  - What problem domain does pymdp address?
  - What is a POMDP?
  - What does "active inference" mean at a high level?
  - What two backends does pymdp support?
  - What is the core observation-action loop?
- **Prerequisites:** None

---

### Lesson 2: Python Arrays and JAX Basics

- **Structural focus:** `pymdp/utils.py` — core utility functions and data structures
- **Key files to read:** `pymdp/utils.py` (lines 1–200)
- **Core concepts to teach:**
  - What is NumPy and why arrays matter
  - What is JAX and how it differs from NumPy (`jax.numpy`)
  - 1D arrays as probability distributions (they must sum to 1)
  - Normalising a distribution: `norm_dist(dist)` in `pymdp/utils.py`
  - Lists of arrays and why pymdp uses them (factorised distributions)
  - `list_array_norm_dist()` — normalising a list of distributions
  - What is `dtype=object` and why pymdp uses object arrays (legacy) vs lists (JAX)
  - The concept of "factors" — splitting one big distribution into smaller independent ones
- **What NOT to cover:** A/B/C/D matrices themselves; inference algorithms; the Distribution class; control functions; JAX tracing/JIT; equinox
- **Exercise ideas:**
  - fill_in_blank: Implement `norm_dist` — divide an array by its sum
  - write_whole: Write a function that takes a list of arrays and returns True if all of them are valid probability distributions (all non-negative, each sums to ~1.0)
- **MCQ topics:**
  - What does normalising a probability distribution do?
  - Why does pymdp store multiple distributions as a list rather than one big array?
  - What does `jax.numpy` have in common with `numpy`?
  - What does it mean for a distribution to "sum to 1"?
  - What is a "factor" in the context of a factorised distribution?
- **Prerequisites:** Lesson 1 (motivation and vocabulary)

---

### Lesson 3: Building an Observation Model (The A Matrix)

- **Structural focus:** The A matrix — how observations relate to hidden states
- **Key files to read:** `pymdp/utils.py` (`random_A_matrix`, `obj_array_zeros`, `obj_array_uniform`), `docs/notebooks/pymdp_fundamentals.ipynb` (A matrix section)
- **Core concepts to teach:**
  - What is the observation model (A matrix)?
  - Observation modalities — different types of sensory input
  - Hidden state factors — the things the world can be in
  - Shape of A[m]: `(num_obs_m, num_states_1, num_states_2, ...)`
  - How to read A: "given these hidden states, what is the probability of this observation?"
  - Building A with `utils.random_A_matrix(num_obs, num_states)`
  - Initialising A to zeros with `obj_array_zeros(A_shapes)` and filling it manually
  - Why each column of A must sum to 1 (it's a conditional distribution)
  - `num_obs` and `num_states` lists — what they represent
- **What NOT to cover:** B matrix; C/D vectors; inference using A; learning / updating A; factor_dot or maths internals; equinox or JIT
- **Exercise ideas:**
  - fill_in_blank: Given `num_obs = [2]` and `num_states = [3]`, create an A matrix where hidden state 0 → observation 0 with prob 0.9, hidden state 1 → observation 0 with prob 0.5, hidden state 2 → observation 1 with prob 0.8
  - write_whole: Write a function `check_A_valid(A_matrix)` that returns True if every column of a 2D array sums to approximately 1.0
- **MCQ topics:**
  - What does the A matrix encode in a POMDP?
  - Why must each column of A sum to 1?
  - What is an "observation modality"?
  - What does `A[0].shape` tell you?
  - What does `num_obs = [3, 5]` mean?
- **Prerequisites:** Lesson 2 (arrays, normalisation, lists of distributions)

---

### Lesson 4: Hidden States and Transitions (The B Matrix)

- **Structural focus:** The B matrix — how states transition over time with actions
- **Key files to read:** `pymdp/utils.py` (`random_B_matrix`, `resolve_b_dependencies`), `docs/notebooks/pymdp_fundamentals.ipynb` (B matrix section)
- **Core concepts to teach:**
  - What is the transition model (B matrix)?
  - Actions (control states) and control factors
  - Shape of B[f]: `(num_states_f, num_states_f, num_controls_f)`
  - How to read B: "given I'm in state s and take action u, what state will I be in?"
  - Building B with `utils.random_B_matrix(num_states, num_controls)`
  - `num_controls` list — what it represents
  - Identity transitions (the world doesn't change with that action)
  - `resolve_b_dependencies` — which factors influence which
- **What NOT to cover:** The C or D matrices; learning/updating B; inference or policy selection; MCTS or planning; equinox/JIT
- **Exercise ideas:**
  - fill_in_blank: Create a simple B matrix for one factor with 2 states and 2 actions where action 0 = stay, action 1 = move to the other state
  - write_whole: Write a function `check_B_valid(B_matrix)` that verifies every column of each action-slice sums to 1.0
- **MCQ topics:**
  - What does the B matrix represent?
  - What are "control factors"?
  - What does `B[0][:, :, 0]` represent (the first action slice)?
  - What does `num_controls = [4, 1, 1]` imply about which factors are controllable?
  - Why must each column of a B action-slice sum to 1?
- **Prerequisites:** Lesson 3 (A matrix, observation model, hidden states, num_states)

---

### Lesson 5: Preferences and Priors (The C and D Vectors)

- **Structural focus:** The C vector (preferences) and D vector (prior state beliefs)
- **Key files to read:** `pymdp/utils.py` (`obj_array_uniform`, `obj_array_zeros`, `random_single_categorical`), `docs/notebooks/pymdp_fundamentals.ipynb` (C and D sections)
- **Core concepts to teach:**
  - What is the C vector (preferences over observations)?
  - Higher C values = more preferred outcomes
  - C as a log-probability: why values can be negative
  - Building C with `obj_array_uniform(num_obs)` (no preferences) vs custom values
  - What is the D vector (prior beliefs over hidden states)?
  - D as a probability distribution over initial states
  - Building D with `random_single_categorical(num_states)` or manually
  - The full generative model: A + B + C + D together
  - What you now have: a complete (but random) POMDP agent model
- **What NOT to cover:** How C is used in EFE computation; how D is updated; policy inference math; the Distribution class; equinox/JAX tracing
- **Exercise ideas:**
  - fill_in_blank: Create a C vector for 2 observation modalities where the first modality has 3 outcomes (prefer outcome index 2) and the second modality has 2 outcomes (no preference)
  - write_whole: Write a function `summarise_model(num_obs, num_states, num_controls)` that returns a dict with keys "modalities", "factors", "control_factors"
- **MCQ topics:**
  - What does the C vector represent?
  - What does D represent?
  - If C[0] = [0, 0, 2.0] what does the agent prefer?
  - Why is D initialised as a probability distribution (summing to 1)?
  - What is the full set of matrices that define a POMDP generative model in pymdp?
- **Prerequisites:** Lesson 4 (B matrix, actions, control factors)

---

### Lesson 6: The Distribution Class

- **Structural focus:** `pymdp/distribution.py` — named-dimension wrapper
- **Key files to read:** `pymdp/distribution.py`
- **Core concepts to teach:**
  - The problem with raw arrays: hard-to-read indices
  - What the `Distribution` class does: gives dimensions human-readable names
  - `event` dict — the dimensions that vary within one distribution
  - `batch` dict — dimensions that index separate distributions
  - Constructing a Distribution: `Distribution(event={"color": ["red", "blue"]}, ...)`
  - Reading values: `dist.get(event={"color": "red"})`
  - Setting values: `dist.set(event={"color": "red"}, values=0.9)`
  - `dist.data` — the underlying numpy array
  - When to use Distribution vs plain arrays
- **What NOT to cover:** Inference or control using Distribution; the likelihoods.py module; Distribution in multi-modality setups; equinox or JAX integration
- **Exercise ideas:**
  - fill_in_blank: Create a `Distribution` for a coin with event `{"side": ["heads", "tails"]}` and set `heads` to 0.7 and `tails` to 0.3
  - write_whole: Write a function `make_uniform_distribution(event_dict)` that creates a Distribution with all values set to 1/N where N is the number of outcomes for each event dimension
- **MCQ topics:**
  - What problem does the Distribution class solve compared to raw arrays?
  - What is the `event` parameter in `Distribution.__init__`?
  - How do you read the value for a specific named outcome?
  - What does `dist.data` return?
  - What is the difference between `event` and `batch` parameters?
- **Prerequisites:** Lesson 5 (full generative model picture, arrays, distributions)

---

### Lesson 7: Belief Updating — Where Am I?

- **Structural focus:** `pymdp/inference.py` and `pymdp/algos.py` — fixed-point iteration
- **Key files to read:** `pymdp/inference.py` (first 100 lines), `pymdp/algos.py` (`run_vanilla_fpi`)
- **Core concepts to teach:**
  - The inference problem: given an observation, what hidden state am I in?
  - Bayesian belief updating at an intuitive level
  - What is Fixed Point Iteration (FPI)?
  - The posterior `qs` — a list of arrays over hidden state factors
  - `run_vanilla_fpi(A, obs, prior)` — inputs and outputs
  - What `obs` looks like: a list of integers (one per modality)
  - How the prior `D` seeds the inference
  - Iterating to convergence: why one pass is sometimes not enough
  - `num_iter` parameter
- **What NOT to cover:** Marginal Message Passing (MMP); policy inference; control functions; learning; the full Agent class; equinox/JIT tracing details; backward pass / gradients
- **Exercise ideas:**
  - fill_in_blank: Given a simple 2-state, 1-modality model, call `run_vanilla_fpi` with a given A, obs, and prior and return the resulting posterior
  - write_whole: Write a function `most_likely_state(qs_factor)` that takes a 1D probability array and returns the index of the highest value
- **MCQ topics:**
  - What does state inference compute?
  - What is the input `obs` to `run_vanilla_fpi`?
  - What does the posterior `qs` represent?
  - What does the prior `prior` parameter seed?
  - What does "fixed point" mean in FPI?
- **Prerequisites:** Lesson 3 (A matrix), Lesson 5 (D prior), Lesson 2 (array basics)

---

### Lesson 8: Policy Inference — What Should I Do?

- **Structural focus:** `pymdp/control.py` — expected free energy and policy selection
- **Key files to read:** `pymdp/control.py` (first 150 lines)
- **Core concepts to teach:**
  - What is a policy? (a sequence of actions)
  - Expected Free Energy (EFE) — a score for each policy
  - Epistemic value: how much will this policy reduce my uncertainty? (curiosity)
  - Instrumental value: how much will this policy bring preferred outcomes?
  - Why active inference agents are naturally curious
  - `infer_policies(qs, A, B, C)` — what it returns
  - The policy posterior `q_pi` — a distribution over policies
  - `sample_action(q_pi, num_controls)` — sampling an action
  - Why higher EFE = better policy
- **What NOT to cover:** MMP-based planning; MCTS; sophisticated inference; Bayesian model reduction; learning the C vector; exact EFE math derivations
- **Exercise ideas:**
  - fill_in_blank: Given a 1D array of EFE values (one per policy), write a function that converts them to a softmax probability distribution
  - write_whole: Write a function `greedy_policy(efe_values)` that returns the index of the policy with the highest EFE value
- **MCQ topics:**
  - What does Expected Free Energy measure?
  - What is the difference between epistemic and instrumental value?
  - What does `q_pi` represent?
  - Why is an active inference agent naturally curious?
  - What is a "policy" in this context?
- **Prerequisites:** Lesson 7 (state inference, qs posterior), Lesson 5 (C preferences), Lesson 4 (B matrix, actions)

---

### Lesson 9: Environments — The World the Agent Lives In

- **Structural focus:** `pymdp/envs/env.py` and `pymdp/envs/tmaze.py`
- **Key files to read:** `pymdp/envs/env.py`, `pymdp/envs/tmaze.py` (first 100 lines)
- **Core concepts to teach:**
  - What is an "environment" in pymdp?
  - The `Env` base class and its API: `reset()` and `step(action)`
  - Observations returned by `step()` — list of indices per modality
  - The T-Maze environment as a concrete example
  - What the T-Maze represents: an agent foraging for reward through cues
  - How the environment is separate from the agent's generative model
  - Creating an environment instance
  - The "generative process" vs the "generative model" distinction
- **What NOT to cover:** Rollout utilities; the cue-chaining or graph-world environments; building custom environments; MCTS or planning on top of environments
- **Exercise ideas:**
  - fill_in_blank: Create a TMazeEnv instance, call `reset()`, and return the observation
  - write_whole: Write a function `run_one_step(env, action)` that calls `env.step(action)` and returns the observation
- **MCQ topics:**
  - What two methods does the `Env` base class require?
  - What does `step(action)` return?
  - What is the difference between the "generative process" and the "generative model"?
  - What does `reset()` do?
  - What scenario does the T-Maze environment model?
- **Prerequisites:** Lesson 4 (actions, control factors), Lesson 3 (observation modalities)

---

### Lesson 10: The Full Active Inference Loop

- **Structural focus:** `pymdp/envs/rollout.py` — the complete perception-action cycle
- **Key files to read:** `pymdp/envs/rollout.py`
- **Core concepts to teach:**
  - The perception-action cycle: observe → infer states → infer policies → act → repeat
  - How to wire together: Env + A + B + C + D + FPI + policy inference
  - `rollout()` function — inputs, outputs, what it does
  - Time steps and episode length
  - Recording history: observations, beliefs, actions over time
  - What a typical simulation result looks like
  - How to visualise or inspect results
  - Why this loop is the "heart" of active inference
- **What NOT to cover:** Multi-agent rollouts; learning during rollout; MCTS or SI planning; cue-chaining specifics; JAX `vmap`-based batching
- **Exercise ideas:**
  - fill_in_blank: Given a pre-built agent and environment, call `rollout()` for 5 time steps and return the list of actions taken
  - write_whole: Write a function `count_correct_observations(obs_history, target_obs)` that counts how many times a specific observation appeared across time steps
- **MCQ topics:**
  - What are the four steps of the active inference loop?
  - What does the `rollout()` function return?
  - Why do we separate the generative process (Env) from the generative model (A, B, C, D)?
  - What does the agent do after receiving an observation?
  - What does "episode" mean in this context?
- **Prerequisites:** Lessons 7–9 (inference, policy, environment)

---

### Lesson 11: Learning — Updating Beliefs Over Time

- **Structural focus:** `pymdp/learning.py` — Dirichlet parameter updates
- **Key files to read:** `pymdp/learning.py` (first 120 lines)
- **Core concepts to teach:**
  - The difference between inference (updating beliefs) and learning (updating model parameters)
  - What is a Dirichlet distribution? (a distribution over distributions)
  - Why Dirichlet distributions are used for A and B
  - `update_likelihood_dirichlet(pA, A, obs, qs)` — updating the observation model
  - `update_transition_dirichlet(pB, B, actions, qs, qs_prev)` — updating the transition model
  - Accumulating counts: how the agent learns from experience
  - When to learn vs when to keep the model fixed
- **What NOT to cover:** MCTS or sophisticated inference; Bayesian model reduction (BMR); model comparison; multi-agent learning; JAX-specific gradient-based learning
- **Exercise ideas:**
  - fill_in_blank: Write a function `dirichlet_to_likelihood(pA)` that normalises a Dirichlet concentration parameter array along axis=0 to produce a valid likelihood
  - write_whole: Write a function `update_counts(count_matrix, obs_index, state_index)` that increments a 2D count matrix at position [obs_index, state_index] and returns the updated matrix
- **MCQ topics:**
  - What is the difference between inference and learning in pymdp?
  - What is a Dirichlet distribution used for?
  - What does it mean to "accumulate counts"?
  - Which pymdp matrices can be learned: A, B, C, or D?
  - What happens to the likelihood A after many observations of the same state-observation pair?
- **Prerequisites:** Lesson 10 (full loop), Lesson 3 (A matrix), Lesson 4 (B matrix)

---

### Lesson 12: Planning with MCTS and Sophisticated Inference

- **Structural focus:** `pymdp/planning/` — MCTS and sophisticated inference
- **Key files to read:** `pymdp/planning/si.py`, `pymdp/planning/mcts.py` (first 80 lines each)
- **Core concepts to teach:**
  - Why standard policy inference can be limited (exponentially many long-horizon policies)
  - What is Monte Carlo Tree Search (MCTS)?
  - How MCTS is used in pymdp for deep planning
  - What is "Sophisticated Inference" (SI)?
  - SI: recursive future planning using counterfactual reasoning
  - When to use MCTS vs SI vs standard FPI-based policy inference
  - What kinds of problems benefit from planning (sparse reward, long horizons)
  - Real-world problems solvable with these techniques
- **What NOT to cover:** Implementation details of MCTS tree expansion; JAX `vmap` batching internals; MCTX library internals; multi-agent scenarios; Bayesian model reduction
- **Exercise ideas:**
  - fill_in_blank: Write a function `is_planning_useful(horizon, num_actions)` that returns True if `horizon * num_actions > 20` (simplified heuristic for when planning is valuable)
  - write_whole: Write a function `describe_problem_type(horizon, num_states)` that returns "shallow" if horizon <= 3, "medium" if horizon <= 10, else "deep planning required"
- **MCQ topics:**
  - Why does deep planning become hard with standard policy inference?
  - What does MCTS stand for and what does it do?
  - What is Sophisticated Inference?
  - What type of problems benefit most from planning?
  - How does pymdp's planning relate to the full active inference loop?
- **Prerequisites:** Lesson 10 (full loop), Lesson 8 (policy inference, EFE)

---

## Progression Rationale

The curriculum is structured in three natural tiers. The first tier (Lessons 1–5) builds the complete generative model piece by piece — always answering the question "what is this matrix for?" before introducing any math. Starting with the big-picture problem (what is active inference and what problems does it solve?) directly addresses the learner's stated interest. Each of the four model components (A, B, C, D) gets its own lesson, which is a deliberate choice for a beginner-level learner: rushing through all four at once would overwhelm someone with low codebase comfort.

The second tier (Lessons 6–10) introduces the computational machinery — the Distribution class for readable model construction, state inference via FPI, policy inference via EFE, environments, and finally the full closed loop. The rollout lesson in position 10 acts as the synthesis lesson where everything learned so far is assembled into a working simulation, giving the learner a clear "I built something" moment. The third tier (Lessons 11–12) covers the more advanced topics of learning and planning, which naturally build on the foundation of the loop and are the features that make pymdp uniquely powerful for real cognitive modelling problems.
