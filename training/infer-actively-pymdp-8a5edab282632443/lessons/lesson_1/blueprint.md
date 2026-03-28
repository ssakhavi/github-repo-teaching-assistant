# Lesson 1 Blueprint: What is pymdp and Active Inference?

## Lesson Number
1

## Repository
infer-actively-pymdp

## Topic Title
What is pymdp and Active Inference?

## Lesson Overview
This first lesson answers the learner's core question: "What types of problems can be solved with this repository?" We cover the big picture — what Active Inference is, what POMDPs are, and why pymdp exists — entirely at the conceptual level with no math. It establishes the vocabulary and motivation that every subsequent lesson will build on.

## Learning Objectives
1. Explain in plain language what pymdp does and what problem it solves.
2. Describe what a POMDP is and name its four main components.
3. Give at least two real-world examples of problems solvable with active inference.
4. Distinguish between the JAX and legacy NumPy backends.

## Key Concepts

- **Active Inference**: A framework where an agent perceives and acts in the world by minimising "free energy" — essentially, it tries to keep its model of the world accurate and achieve preferred outcomes simultaneously. Originally developed in computational neuroscience by Karl Friston.
- **Free Energy Principle**: The idea that intelligent agents (biological or artificial) minimise surprise about their sensory observations. Curiosity and goal-seeking behaviour emerge naturally from this single principle.
- **POMDP (Partially-Observed Markov Decision Process)**: A mathematical model of an agent in an uncertain world. The world has hidden states the agent can't see directly; it can only observe noisy signals and must infer what's happening.
- **The four POMDP components**: A (observation model), B (transition model), C (preferences), D (initial state prior). Each lesson 3–5 dives into one of these.
- **The perception-action loop**: At each time step the agent (1) receives an observation, (2) infers its current hidden state, (3) picks an action, (4) acts and repeats. This loop is at the heart of all pymdp simulations.
- **Epistemic value (curiosity)**: Agents in pymdp are naturally curious — they seek out actions that reduce uncertainty about the world. This is an automatic by-product of free energy minimisation, not a hand-engineered reward.
- **JAX backend vs legacy NumPy**: The modern `pymdp/` folder uses JAX for GPU/TPU acceleration and clean functional style. The `pymdp/legacy/` folder uses NumPy and is kept for backward-compatibility. This course focuses on the JAX backend.
- **Generative model vs generative process**: The generative model is the agent's internal belief about how the world works (A, B, C, D matrices). The generative process is the actual external environment. They are separate and don't have to match perfectly.

## Lesson Content Outline
1. **The problem pymdp solves** — Why do we need a library like this? From MATLAB to open-source Python; the gap pymdp fills; who uses it and why.
2. **Active inference in plain English** — Free energy, surprise minimisation, and why curiosity and goal-seeking emerge for free.
3. **What is a POMDP?** — Hidden states, partial observability, the four components (A/B/C/D) named but not yet detailed. The mouse-in-a-maze analogy.
4. **The perception-action loop** — The five-step cycle every pymdp simulation follows; how A/B/C/D fit into each step.
5. **Summary and what's next** — Real-world domains solvable with pymdp; how the course will build up from here.

## Code Focus
- `README.md` — The motivation, the "pymdp in action" epistemic chaining demo description, and the quick-start code snippet showing Agent, infer_states, infer_policies, sample_action
- `setup.cfg` — Package description ("solving MDPs with Active Inference"), dependencies (jax, equinox, mctx — no explanation needed beyond noting they are math/ML libraries)
- `docs/index.rst` — Official description: "discrete space and time, POMDP generative model class, modular and flexible"
- `paper/paper.md` — Statement of Need section: real-world applications (cognitive neuroscience, control theory, reinforcement learning, psychopathology, social cognition, engineering)

## What NOT to Cover
Do NOT reference these concepts anywhere in the lesson, exercises, or MCQ:
- NumPy arrays / JAX arrays in detail → covered in Lesson 2
- A matrix shape or construction → covered in Lesson 3
- B matrix → covered in Lesson 4
- C and D vectors → covered in Lesson 5
- Distribution class → covered in Lesson 6
- State inference (FPI) algorithms → covered in Lesson 7
- Policy inference / EFE math → covered in Lesson 8
- Environment code (Env class, TMaze, rollout) → covered in Lessons 9–10
- Dirichlet distributions or learning → covered in Lesson 11
- MCTS or sophisticated inference → covered in Lesson 12
- Any code syntax beyond high-level pseudocode / reading existing snippets

## Example Exercise Ideas

### Exercise 1: Match the term to its meaning — fill_in_blank
- **Concept tested:** POMDP vocabulary and the perception-action loop
- **Function to base it on:** A pure-Python dictionary lookup (no pymdp import needed)
- **What blanks to use:** Complete a function that maps four POMDP component names ("A", "B", "C", "D") to one-line descriptions by filling in the blank values in a pre-written dict

### Exercise 2: Real-world problem classifier — fill_in_blank
- **Concept tested:** Understanding what kinds of problems active inference solves
- **Function to base it on:** A simple string classification function
- **What blanks to use:** Complete a function that takes a problem domain string and returns True if it is a domain where active inference has been applied (e.g., "cognitive neuroscience", "robotics", "reinforcement learning"), False for clearly out-of-scope ones (e.g., "image compression", "sorting algorithms")

## MCQ Topic Areas
1. What problem domain does pymdp address? (reinforcement learning / cognitive modelling / web development / image processing)
2. What does POMDP stand for and what is the key characteristic of a POMDP?
3. What does "active inference" mean at a high level? (curiosity + goal-seeking from one principle vs hand-engineered rewards)
4. What are the two backends in pymdp and which is the modern one?
5. What is the core observation-action loop in pymdp? (order of steps)
6. What is "epistemic value" in the context of active inference?
7. How does pymdp's API relate to OpenAI Gym?

## Prior Knowledge Available
None — this is the first lesson.

## Performance Notes
No previous results — first lesson.

## User Feedback
No written feedback provided.

## Progression Notes
Lesson 2 introduces NumPy and JAX arrays, normalisation, and the utility functions in `pymdp/utils.py`. The vocabulary built in Lesson 1 (hidden states, factors, modalities, distributions) will be used immediately in Lesson 2 when working with arrays as probability distributions.
