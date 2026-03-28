# Lesson 3: Building an Observation Model (The A Matrix)

**Repository:** infer-actively-pymdp-8a5edab282632443
**Lesson:** 3 of 12

## What You'll Learn

In this lesson you'll meet the A matrix — the observation model at the heart of every POMDP agent. You'll understand what it represents, how to read its shape, and how to build one using pymdp's utility functions.

## Why It Matters

An agent that can't connect hidden states to observations is blind. The A matrix is the agent's sensor model: it answers the question "if the world is in state *s*, what do I expect to see?" Without A you can't do perception — you can't update beliefs, reason about uncertainty, or act intelligently.

## Connecting Back to Lesson 2

In Lesson 2 you learned that a probability distribution is just an array whose entries sum to 1. You built them, normalised them, and stored them in lists. Here's the key insight for Lesson 3: **the A matrix is just a collection of those distributions, arranged side by side**.

Each *column* of A is exactly the kind of distribution you built in Lesson 2 — it gives you the probability of every possible observation for one specific hidden state. You already know how columns work. Now you're going to see how pymdp packages them.

---

## What Is the A Matrix?

Think of the A matrix as your agent's sensor manual. For every possible hidden state of the world, it tells the agent: "here are the probabilities of each observation you might receive."

Formally, `A[m][obs, state]` is the probability of observation `obs` given hidden state `state`, for modality `m`. If that sounds abstract, a concrete example makes it click.

Imagine your agent can be in one of four rooms (hidden states 0–3). In each room it receives a colour signal. The A matrix for this setup might look like:

```
         state 0  state 1  state 2  state 3
obs red    0.9      0.1      0.0      0.2
obs green  0.05     0.8      0.3      0.1
obs blue   0.05     0.1      0.7      0.7
```

Read column 0: if the agent is in room 0, it sees red with probability 0.9, green with 0.05, blue with 0.05. That column sums to 1.0 — just like every distribution you built in Lesson 2.

The A matrix is also called the **observation likelihood** because it tells you how *likely* each observation is given each state.

### Observation Modalities

An agent can receive more than one type of sensory signal at a time. Each signal type is called an **observation modality**. You might have a visual signal and an auditory signal simultaneously. Each modality gets its own sub-matrix: `A[0]` for modality 0, `A[1]` for modality 1, and so on.

In pymdp, `A` is always a Python **list** — one element per modality. Even when you only have one modality, `A` is still a list: `A[0]` is the single tensor.

---

## Reading A: Shape and Axes

### `num_obs` and `num_states`

You describe your problem to pymdp using two lists:

- **`num_obs`**: how many distinct observations each modality can produce. `num_obs = [3, 2]` means modality 0 has 3 possible observations, modality 1 has 2.
- **`num_states`**: how many distinct hidden states each factor has. `num_states = [4]` means one hidden state factor with 4 possible states.

### The Shape Rule

For modality `m`, the shape of `A[m]` follows a simple rule:

```
A[m].shape == (num_obs[m], num_states[0], num_states[1], ...)
```

The **first axis** is always the observation axis. The **remaining axes** index the hidden state factors.

So with `num_obs = [3]` and `num_states = [4]`:

```python
A[0].shape  # → (3, 4)
# 3 possible observations, 4 possible hidden states
```

### Slicing a Column

To read the distribution for hidden state 2, you slice along the state axis:

```python
A[0][:, 2]
# → an array of length 3
# → the probability of each observation when the agent is in state 2
```

That slice is a distribution, so it sums to 1. That's the Lesson 2 rule applied directly: `A[0][:, 2].sum() == 1.0`.

With two hidden state factors (`num_states = [4, 2]`), the shape becomes `(3, 4, 2)` and you'd read a column as `A[0][:, s0, s1]` — still a distribution summing to 1.

---

## Building A with `random_A_array`

### JAX vs NumPy — A Quick Reminder

Before looking at the code, a quick note since this tripped up the MCQ in Lesson 2: `jax.numpy` (imported as `jnp`) is **not** a different language or a slower NumPy. It provides nearly the **identical API** as NumPy — the same function names, the same indexing — but it can run on GPU and TPU. When you see `jnp.ones(...)` in pymdp, that's simply `numpy.ones(...)` written for JAX. The result is a JAX array, not a NumPy array, but it looks and behaves the same in most contexts.

### The Function

Here is the actual `random_A_array` from `pymdp/utils.py`:

```python
def random_A_array(
    key: Array,
    num_obs: int | Sequence[int],
    num_states: int | Sequence[int],
    A_dependencies: list[list[int]] | None = None,
) -> list[Array]:
    num_obs    = [num_obs]    if isinstance(num_obs, int)    else num_obs
    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_modalities = len(num_obs)

    A_dependencies = resolve_a_dependencies(len(num_states), num_modalities, A_dependencies)

    keys = jr.split(key, num_modalities)
    A = []
    for m, n_o in enumerate(num_obs):
        lagging_dimensions = tuple(num_states[idx] for idx in A_dependencies[m])
        A_m = jr.dirichlet(keys[m], alpha=jnp.ones(n_o), shape=lagging_dimensions)
        A.append(jnp.moveaxis(A_m, -1, 0))
    return A
```

### What's Happening

1. **`key`** is a JAX PRNG key — a small array that seeds the random number generator. You create one with `jax.random.PRNGKey(0)`. pymdp requires you to pass this explicitly so that computations are reproducible and JAX-compatible.
2. The function loops over modalities. For each one, it samples a Dirichlet distribution — a random distribution over distributions — which guarantees that each column sums to 1.
3. The `jnp.moveaxis` call moves the observation axis to the front so the final shape is `(num_obs[m], *state_dims)`, matching the shape rule you just learned.
4. The result is a **list** of JAX arrays, one per modality.

### Calling It

```python
import jax
from pymdp.utils import random_A_array

key = jax.random.PRNGKey(42)
A = random_A_array(key, num_obs=[3], num_states=[4])

print(A[0].shape)   # → (3, 4)
print(A[0][:, 0])   # → random distribution summing to ~1.0
```

The randomness is useful for testing or for seeding a model that you'll later update with real data. Every column is guaranteed to be a valid probability distribution.

---

## Building A with `list_array_uniform` and `list_array_zeros`

Sometimes you don't want a random A matrix — you want to specify it yourself. pymdp gives you two helpers for that.

### `list_array_uniform`

Here is the actual source:

```python
def list_array_uniform(shape_list: Sequence[Sequence[int]]) -> list[Array]:
    arr = []
    for shape in shape_list:
        arr.append(norm_dist(jnp.ones(shape)))
    return arr
```

It creates a list of uniform distributions — every observation is equally likely for every hidden state. Notice that it calls `norm_dist` (which you saw in Lesson 2!) on `jnp.ones(shape)`. An array of ones divided by its sum gives equal probability to each entry.

```python
from pymdp.utils import list_array_uniform

A = list_array_uniform(shape_list=[(3, 4)])
print(A[0].shape)     # → (3, 4)
print(A[0][:, 0])     # → [0.333, 0.333, 0.333]
```

This is your neutral starting point when you haven't decided how observations depend on states. Every state looks the same to the agent — no information at all.

### `list_array_zeros`

Here is the actual source:

```python
def list_array_zeros(shape_list: Sequence[Sequence[int]]) -> list[Array]:
    arr = []
    for shape in shape_list:
        arr.append(jnp.zeros(shape))
    return arr
```

This gives you a blank slate — an array of zeros with the right shape. You fill in the values yourself, then normalise each column with `norm_dist`.

```python
from pymdp.utils import list_array_zeros, norm_dist

A = list_array_zeros(shape_list=[(3, 4)])
# A[0] is all zeros — fill in column 0 manually
A[0] = A[0].at[0, 0].set(1.0)  # state 0 always gives observation 0
A[0] = A[0].at[1, 1].set(1.0)  # state 1 always gives observation 1
# ... fill remaining columns, then normalise
```

Use `list_array_zeros` when you know the exact observation mapping — for example, when you're modelling a grid world where each location deterministically produces a unique signal.

### Which One to Use?

| You want... | Use... |
|---|---|
| A random A for testing | `random_A_array` |
| A neutral, uninformative A | `list_array_uniform` |
| A hand-crafted, precise A | `list_array_zeros` + fill in + `norm_dist` |

---

## Summary

In this lesson we covered:

- **The A matrix is the observation model**: it encodes how likely each observation is for every hidden state. It sits between the hidden world and the agent's senses.
- **Shape rule**: `A[m].shape == (num_obs[m], num_states[0], ...)` — observations on the first axis, hidden state factors on the rest.
- **Columns must sum to 1**: every column of A is a probability distribution, exactly like the arrays you built in Lesson 2. `A[0][:, s].sum() == 1.0` for every state `s`.
- **`random_A_array`**: builds a random, valid A using Dirichlet sampling and JAX's PRNG. Returns a list of JAX arrays.
- **`list_array_uniform`**: builds a maximally uninformative A where every observation is equally likely. Internally calls `norm_dist(jnp.ones(shape))`.
- **`list_array_zeros`**: gives you a blank array to fill in yourself when you know the exact mapping.
- **JAX vs NumPy**: `jnp` is nearly identical to `numpy` in API — same function names, same indexing — but runs on GPU/TPU. pymdp's utilities all return JAX arrays.

## What's Next: Lesson 4 — The B Matrix

Now that you understand the A matrix, you're ready for the **B matrix** — the transition model. B answers a different question: "if the world is in state *s* and the agent takes action *a*, what is the distribution over next states?" Just like A, each "slice" of B (indexed by action) is a collection of probability distributions where every column sums to 1. The lesson-2 rule carries forward again.
