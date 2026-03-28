# Lesson 4: Hidden States and Transitions (The B Matrix)

**Repo:** infer-actively-pymdp-8a5edab282632443
**Lesson:** 4 of 12

## What you'll learn

In this lesson you're going to tackle something that has been tripping you up in the exercises: **how Python indexes into 2D and 3D lists**. You'll start there — with small, concrete examples — before touching any pymdp code at all. Once that's solid, you'll meet the B matrix: the part of an active inference model that describes how the world changes when you take an action.

## Why it matters

Every single pymdp structure — A, B, and the ones coming in later lessons — is a multi-dimensional list or array. If you can't correctly read `M[row][col]` or `M[i][j][k]`, every coding exercise will feel like a mystery. After this lesson, indexing won't be a blocker anymore. And the B matrix itself is the "rulebook" that tells your agent: *if I'm in this state and I take this action, where do I end up?* Without B, your agent can't plan.

---

## Page 2: Matrix Indexing — Let's Fix This First

Your feedback from Lesson 3 was direct: *"I think I have a wrong understanding of some of the matrix operations."* That's exactly what this page addresses. Let's fix it completely before any B matrix content.

### 2D indexing: `matrix[row][col]`

A 2D matrix in Python is a **list of lists**. The outer list holds the rows; each inner list holds the columns within that row.

```python
M = [
    [0.9, 0.1],   # row 0
    [0.1, 0.9],   # row 1
]
```

To get a single value, you use two square-bracket lookups in sequence:

- `M[0]` gives you the whole first row: `[0.9, 0.1]`
- `M[0][1]` gives you row 0, column 1: **`0.1`**
- `M[1][0]` gives you row 1, column 0: **`0.1`**
- `M[1][1]` gives you row 1, column 1: **`0.9`**

A concrete trick to remember: **the first index picks the outer list (the row), the second index picks within that row (the column).** You can always check your answer by asking: "did I peel off the outer layer first?"

### 3D indexing: `matrix[i][j][k]`

A 3D structure is a list of lists of lists — just one more layer deep. You peel off three layers with three bracket pairs.

```python
cube = [
    # i=0 (first "slice")
    [[1, 2],   # j=0
     [3, 4]],  # j=1

    # i=1 (second "slice")
    [[5, 6],   # j=0
     [7, 8]],  # j=1
]
```

Let's trace through a lookup step by step:

- `cube[1]` → the second slice: `[[5, 6], [7, 8]]`
- `cube[1][0]` → row 0 inside that slice: `[5, 6]`
- `cube[1][0][1]` → column 1 inside that row: **`6`**

So `cube[i][j][k]` means: pick slice `i`, then row `j` within that slice, then column `k` within that row. Three indices, three peeling steps.

That's the complete rule. The B matrix is just a 3D structure — so you'll use `B[next_state][current_state][action]` — but now you know exactly how to read it.

---

## Page 3: What Is the B Matrix?

Now that indexing is clear, let's talk about what B actually represents.

### The question B answers

The A matrix (from Lesson 3) answered: "what am I likely to observe given my current hidden state?" B answers a different question: **"what hidden state will I be in next, given my current state and the action I take?"**

Think of B as a rulebook with one page per action. Flip to the page for "move left," and you'll find a table that tells you: if you start in state 0, you'll end up in state 1 with probability 0.95. If you start in state 1, you'll stay there with probability 0.8. Every entry is a conditional probability.

### The analogy: a traffic rulebook

Imagine you're driving in a grid. "State" is your current location. "Action" is a direction you can drive. B is the map of where each direction takes you from each location. If the roads are deterministic, some entries will be 1.0 and the rest 0.0. If the roads are slippery (stochastic), the probabilities spread out.

### How B extends A

- **A** maps: `(hidden state) → observation`
- **B** maps: `(current state, action) → next state`

A is about *perceiving* the world. B is about *predicting how your actions change* the world. Together they give the agent a full generative model: it knows what it would observe from any state (A) and how it gets from one state to another (B).

### B is a list of tensors — one per factor

Just like `A` is a list of modality-specific arrays, `B` is a list of factor-specific tensors. `B[f]` is the transition model for hidden state factor `f`. In the simplest case (one factor), you just work with `B[0]`.

---

## Page 4: Reading B — Shape and Action Slices

### The shape convention

For a single factor with `num_states` states and `num_controls` actions, `B[f]` has shape:

```
(num_next_states, num_current_states, num_actions)
```

So if you have 3 states and 2 actions, `B[0]` has shape `(3, 3, 2)`. The axes are always ordered: **next state first, current state second, action third**.

A quick memory aid: the axis order mirrors the *question* B answers — "where am I going (next), given where I am (current) and what I do (action)?"

### Reading a single entry

Using the indexing rule from Page 2:

```python
B[0][next_state][current_state][action]
```

For example, `B[0][1][0][1]` means:

- `B[0]` → factor 0's transition tensor
- `[1]` → next state is 1
- `[0]` → current state is 0
- `[1]` → action is 1

Translation: *"the probability of transitioning to state 1, given you're currently in state 0 and you take action 1."*

### Action slices: `B[f][:, :, u]`

To see the full transition matrix for a single action `u`, you take a slice across all next states and all current states:

```python
B[0][:, :, u]   # shape: (num_states, num_states)
```

This 2D slice is what you'd call the "transition matrix for action u". Its columns are the important part: **column `s` tells you the full distribution over next states when you're currently in state `s` and take action `u`**.

### The "columns sum to 1" rule

You already know this from the A matrix. The same rule applies here: each column of a B action-slice must sum to 1. Why? Because `B[0][:, s, u]` is a probability distribution over the *next* state — it lists all the ways the world can go, and those must add up to 100%.

```python
# For a 2-state, 2-action B:
action_0_slice = B[0][:, :, 0]   # shape (2, 2)
# column 0 of this slice:
col_0 = [action_0_slice[0][0], action_0_slice[1][0]]
sum(col_0)   # must be 1.0
```

---

## Page 5: Building B — Stay/Move Example and pymdp Helpers

### A hand-built B matrix

Let's build a minimal B from scratch: 2 hidden states, 2 actions.

- **Action 0 = "stay"**: no matter what state you're in, you stay there.
- **Action 1 = "move"**: if you're in state 0 you go to state 1; if you're in state 1 you go to state 0.

In list-of-lists form (shape `[next_state][current_state]` for each action slice):

```python
# Action slice 0 (stay): identity matrix
stay = [
    [1.0, 0.0],   # next=0: prob 1.0 from state 0, prob 0.0 from state 1
    [0.0, 1.0],   # next=1: prob 0.0 from state 0, prob 1.0 from state 1
]

# Action slice 1 (move): swap matrix
move = [
    [0.0, 1.0],   # next=0: prob 0.0 from state 0 (you moved away), prob 1.0 from state 1
    [1.0, 0.0],   # next=1: prob 1.0 from state 0 (you moved here), prob 0.0 from state 1
]

# Full B tensor as a 3D list: B[next][current][action]
B_manual = [
    [[1.0, 0.0], [0.0, 1.0]],   # next=0
    [[0.0, 1.0], [1.0, 0.0]],   # next=1
]
```

Verify: `B_manual[1][0][1]` should be 1.0 (move action takes you from state 0 to state 1). Trace it: outer index 1 → next state 1; index 0 → current state 0; index 1 → action 1. Result: **1.0**. Correct.

### `random_B_array` — randomized transitions

pymdp provides `random_B_array` for when you want valid but random B tensors:

```python
key = jr.PRNGKey(0)
B = random_B_array(key, num_states=[3], num_controls=[2])
# B[0].shape == (3, 3, 2)
# Every column of every action slice sums to 1
```

Under the hood, it samples each column from a Dirichlet distribution (a distribution over distributions), then moves the "next state" axis to the front. The result is always a valid set of transition matrices — all columns sum to 1 — but the specific probabilities are random. You'd use this when prototyping and you don't care about the exact transitions yet.

### `create_controllable_B` — deterministic control

When you want an agent that can *directly* jump to any state by choosing the right action, use `create_controllable_B`:

```python
B = create_controllable_B(num_states=[4], num_controls=[4])
# B[0].shape == (4, 4, 4)
# B[0][u][s][u] == 1.0 for any s (action u always takes you to state u)
# B[0][i][s][u] == 0.0 for i != u
```

The logic: for action `u`, every current state `s` transitions with probability 1.0 to state `u`. The agent has perfect control — the state you end up in is exactly the action you chose. This is useful for simple grid worlds where "take action 2" means "go to cell 2."

### Control factors and `num_controls`

`num_controls = [4, 1]` means:

- Factor 0 has **4 possible actions** — the agent can meaningfully choose here.
- Factor 1 has **1 action** — there's only one "action," so the agent can't influence this factor. It evolves on its own.

A factor with `num_controls = 1` is called an **uncontrollable factor**. Its B still has the same shape convention, but with a third axis of size 1.

---

## Page 6: Summary

## Summary

In this lesson we covered:

- **Matrix indexing**: `matrix[row][col]` peels the outer list first (row), then the inner list (column). For 3D: `matrix[i][j][k]` peels three layers. This is the foundational skill for every pymdp operation.
- **What the B matrix is**: the transition model — it encodes the probability of the next hidden state given the current state and a chosen action. Its shape for factor `f` is `(num_next_states, num_current_states, num_actions)`.
- **Reading B entries and action slices**: `B[f][next][current][action]` gives a single probability. `B[f][:, :, u]` gives the full transition matrix for action `u`, whose columns must each sum to 1.
- **Building B by hand**: a 2-state, 2-action stay/move B matrix — and how to verify entries using the indexing rule.
- **pymdp helpers**: `random_B_array` creates valid random B tensors using Dirichlet sampling; `create_controllable_B` creates deterministic B tensors where the action directly selects the next state.

### Key takeaways

1. Always trace indexing step by step: outer index first, then work inward.
2. B's axis order — `(next, current, action)` — mirrors the question it answers.
3. Columns of any action slice must sum to 1, just like columns of A.
4. `num_controls = [4, 1]` means one controllable factor (4 actions) and one uncontrollable factor (1 "action").

### What's next

**Lesson 5** introduces the **C vector** (the agent's preferences over observations) and the **D vector** (prior beliefs over the initial hidden state). These are both 1D structures — simpler than the 2D/3D matrices you've been working with — so Lesson 5 will feel more straightforward after the indexing work you've done here.
