# Lesson 5: Preferences and Priors (The C and D Vectors)

**Repo:** infer-actively-pymdp-8a5edab282632443
**Lesson:** 5 of the curriculum

## What You'll Learn

In this lesson you'll complete the four-component generative model by adding the two remaining pieces: **C** (what your agent *wants* to observe) and **D** (what your agent *believes* about where it starts). You already have A (the observation model) and B (the transition model). Once you add C and D, you'll have a fully specified POMDP that can express goals and prior beliefs.

Before jumping in, we'll also revisit a couple of B matrix concepts — specifically `num_controls` and what the third dimension of B really means — since those came up as trouble spots in the last lesson's quiz. Getting that solid now will pay off when you reach policy inference later.

## Why This Matters

A generative model without C or D is like a compass without a target: it can model the world but it has no opinion about *where to go* or *where it came from*. C gives the agent desires; D gives it a starting worldview. Together, A + B + C + D form everything pymdp needs to run the full inference-and-action loop. By the end of this lesson you'll be able to hand those four components to an `Agent` object and have a working POMDP agent.

---

## B Matrix Recap: num_controls and Controllable Factors

Before moving on, let's cement what the B matrix's third dimension means, because it's a common source of confusion.

Recall that `B[f]` has shape `(num_states[f], num_states[f], num_controls[f])`. The third dimension — `num_controls[f]` — is the number of *actions* available for factor `f`. Each action slice `B[f][:, :, a]` is a transition matrix that says "if the agent picks action `a`, here is the probability of moving from any state to any other state in factor `f`."

```python
import numpy as np
from pymdp.utils import random_B_matrix

num_states   = [3, 2]   # factor 0 has 3 states, factor 1 has 2 states
num_controls = [3, 1]   # factor 0 has 3 actions, factor 1 has only 1 action

B = random_B_matrix(num_states, num_controls)

print(B[0].shape)  # (3, 3, 3) — 3 actions for factor 0
print(B[1].shape)  # (2, 2, 1) — 1 "action" for factor 1
```

When `num_controls[f] == 1` there is only one action slice, meaning the agent cannot influence that factor — it is **non-controllable**. The transition happens on its own regardless of what the agent does. When `num_controls[f] > 1` the agent can pick among multiple transitions, making that factor **controllable**.

A quick way to remember this: the third dimension of B tells you *how many choices* the agent has for that factor. One choice = no real control; multiple choices = genuine agency over that part of the world.

---

## Introducing C — What Does the Agent Want?

The **C vector** is how you give your agent goals. It is an object array with one sub-array per observation modality, and each sub-array has one value per possible observation outcome. The value answers the question: *how much does the agent prefer this outcome?*

Here's the key design choice in pymdp: **C values live on a log-probability scale**. That means:

- `C[m][o] = 0.0` → neutral — the agent is indifferent to this outcome
- `C[m][o] = 2.0` → desired — the agent will actively seek this outcome
- `C[m][o] = -2.0` → undesired — the agent will try to avoid this outcome

Why logs instead of raw probabilities? Probability values are constrained to [0, 1] and must sum to 1, which makes them awkward as preference scores. Log-probability values are unconstrained real numbers, so you can freely express "strongly want," "strongly avoid," and "don't care" without worrying about normalisation. The underlying math uses these values when evaluating how attractive a sequence of future observations would be.

The shape of C must align with `num_obs`: `C[m]` must have exactly `num_obs[m]` entries — one preference value for each possible outcome in modality `m`.

```python
# An agent with 2 observation modalities:
# modality 0 has 3 outcomes (e.g. "low", "medium", "high")
# modality 1 has 2 outcomes (e.g. "not at goal", "at goal")
num_obs = [3, 2]
```

---

## Building C in Practice

There are two common ways to build C: start from "no preference" and customise, or allocate zeros and fill manually.

**Option 1: Start from uniform (no preference)**

`obj_array_uniform(num_obs)` is the quick way to say "the agent is indifferent about everything." Under the hood it calls `norm_dist(np.ones(shape))` for each modality, producing `1/N` for each outcome — a uniform distribution.

```python
from pymdp.utils import obj_array_uniform

num_obs = [3, 2]
C = obj_array_uniform(num_obs)

print(C[0])  # [0.333, 0.333, 0.333]
print(C[1])  # [0.5,   0.5  ]
```

Wait — didn't we say C should be on a log scale? `obj_array_uniform` actually returns a *normalised distribution* (values sum to 1), not log values. In practice, pymdp accepts C in either form: uniform C is a reasonable "no preference" starting point. When you want to express *actual* preferences you typically work on the log scale by setting values directly.

**Option 2: Allocate zeros and set preferences manually**

`obj_array_zeros(shape_list)` gives you a blank object array filled with zeros. You then set specific entries to express preferences:

```python
from pymdp.utils import obj_array_zeros

num_obs = [3, 2]
C = obj_array_zeros(num_obs)

# Modality 0: strongly prefer outcome 2 ("high")
C[0][2] = 2.0

# Modality 1: strongly prefer outcome 1 ("at goal")
C[1][1] = 3.0

print(C[0])  # [0.0, 0.0, 2.0]
print(C[1])  # [0.0, 3.0]
```

When you see `C[0] = [0, 0, 2.0]`, the interpretation is direct: the agent is indifferent to outcomes 0 and 1, and has a preference strength of 2.0 for outcome 2. The higher the number, the stronger the pull toward that outcome. Negative values push the agent away. Zero means "I don't care."

---

## Introducing D — Where Does the Agent Think It Starts?

The **D vector** encodes the agent's *prior beliefs about its initial hidden state* — what it believes is true about the world before it has made any observations. It is an object array with one sub-array per hidden state factor, and each sub-array is a proper probability distribution over that factor's states.

Think of D as the agent answering the question: "Before I've seen anything, where do I think I am?"

- If the agent has no idea, D should be uniform: equal probability for each state.
- If the agent knows it always starts in a specific state, D should be a one-hot vector (all probability on one state).
- Any valid probability distribution in between is fine.

The shape requirement mirrors A and B: `D[f]` must have exactly `num_states[f]` entries, one for each possible state in factor `f`. And like any probability distribution, each `D[f]` must:

1. Have all non-negative values
2. Sum to exactly 1.0

This is the same requirement as columns of A or slices of B — pymdp is consistent about enforcing the probability simplex.

---

## Building D in Practice

**Option 1: Random D**

`random_single_categorical(shape_list)` generates a random normalised probability distribution for each factor. It calls `norm_dist(np.random.rand(shape_i))` internally, which draws uniform random values and then divides by their sum to get a valid distribution:

```python
from pymdp.utils import random_single_categorical
import numpy as np

num_states = [3, 2]
D = random_single_categorical(num_states)

print(D[0])            # e.g. [0.42, 0.21, 0.37] — random, sums to 1
print(D[1])            # e.g. [0.61, 0.39]
print(np.sum(D[0]))    # 1.0
print(np.sum(D[1]))    # 1.0
```

**Option 2: Manually specified D**

For more control, build D as a list of arrays and construct the object array yourself:

```python
from pymdp.utils import obj_array_zeros
import numpy as np

num_states = [3, 2]
D = obj_array_zeros(num_states)

# Factor 0: agent believes it starts in state 0 with certainty (one-hot)
D[0] = np.array([1.0, 0.0, 0.0])

# Factor 1: agent is completely uncertain between states 0 and 1
D[1] = np.array([0.5, 0.5])

# Always verify your distributions are valid
for f, d in enumerate(D):
    assert np.allclose(np.sum(d), 1.0), f"D[{f}] does not sum to 1!"
    assert np.all(d >= 0),             f"D[{f}] has negative values!"

print("D is valid.")
```

The `np.allclose` check (rather than exact `==`) is important because floating-point arithmetic can produce results like `0.9999999999` instead of exactly `1.0`. `allclose` uses a small tolerance so those tiny errors don't cause false failures.

---

## The Complete Generative Model: A + B + C + D

You now have all four components. Here's what each one contributes:

| Component | Shape | What it encodes |
|-----------|-------|-----------------|
| **A** | `(num_obs[m], *num_states)` per modality | How hidden states produce observations |
| **B** | `(num_states[f], num_states[f], num_controls[f])` per factor | How actions move the agent between states |
| **C** | `(num_obs[m],)` per modality | What observations the agent prefers |
| **D** | `(num_states[f],)` per factor | What state the agent believes it starts in |

Putting them together in code:

```python
from pymdp.utils import (
    random_A_matrix,
    random_B_matrix,
    obj_array_zeros,
    random_single_categorical,
)
import numpy as np

num_obs      = [3, 2]   # 2 modalities
num_states   = [4, 2]   # 2 factors
num_controls = [4, 1]   # factor 0 controllable, factor 1 not

# Observation model
A = random_A_matrix(num_obs, num_states)

# Transition model
B = random_B_matrix(num_states, num_controls)

# Preferences — strongly prefer outcome 2 in modality 0
C = obj_array_zeros(num_obs)
C[0][2] = 2.0

# Prior over initial states — uniform uncertainty
D = random_single_categorical(num_states)

print("Generative model complete:")
print(f"  A shapes: {[A[m].shape for m in range(len(num_obs))]}")
print(f"  B shapes: {[B[f].shape for f in range(len(num_states))]}")
print(f"  C shapes: {[C[m].shape for m in range(len(num_obs))]}")
print(f"  D shapes: {[D[f].shape for f in range(len(num_states))]}")
```

At this point, you have a fully specified generative model. A describes the world's observability, B describes its dynamics, C describes what the agent cares about, and D describes where the agent thinks it begins. These four matrices are the complete specification an `Agent` in pymdp needs. The next step — which comes in Lesson 6 — is looking at a higher-level wrapper that attaches human-readable names to all these dimensions, making the model easier to inspect.

---

## Summary

In this lesson we covered:

- **B matrix reinforcement**: `num_controls[f]` sets the number of action slices in `B[f]`; factors with one control action are non-controllable
- **The C vector**: an object array of log-scale preference values — zero is neutral, positive is desired, negative is undesired — with one sub-array per observation modality
- **Building C**: use `obj_array_uniform` for no preference, or `obj_array_zeros` plus manual assignment for custom preferences
- **The D vector**: an object array of probability distributions over initial hidden states — one sub-array per hidden state factor, each must sum to 1
- **Building D**: use `random_single_categorical` for a random prior, or build manually with `obj_array_zeros` and NumPy arrays

**Key takeaways:**

1. C lives on a log scale — higher values mean stronger preference, there is no upper bound, and you don't need to normalise
2. D is a genuine probability distribution — it must be non-negative and sum to 1 for each factor
3. A + B + C + D together constitute the complete generative model for a POMDP agent in pymdp

**Preview of Lesson 6:**

Next lesson introduces the `Distribution` class from `pymdp/distribution.py`. This wrapper adds human-readable names to the dimensions of your A, B, C, and D arrays, so instead of remembering that axis 0 of A[1] corresponds to "at goal" you can query it by name. It makes debugging and inspection far more intuitive, especially as your models grow more complex.
