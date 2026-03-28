# Lesson 2: Python Arrays and JAX Basics

**Repo:** infer-actively-pymdp-8a5edab282632443

Welcome back. In this lesson you will take your first step into the actual code of pymdp. Everything in the library — every belief, every curiosity signal, every decision — is built on one simple idea: a probability distribution stored as a numerical array.

By the end of this lesson you will be able to:

1. Explain the difference between a Python list and a NumPy/JAX array.
2. Describe what makes an array a valid probability distribution.
3. Read and explain the `norm_dist` function from `pymdp/utils.py`.
4. Understand why pymdp stores multiple distributions as a Python *list* of arrays, not one big 2D array.

That is the full agenda — four ideas, covered one at a time. No rushing.

---

## From Python Lists to NumPy and JAX Arrays

You already know Python lists. They are flexible containers that can hold anything:

```python
my_list = [0.3, 0.5, 0.2]
```

Lists are great for general programming, but they are slow for maths. When you need to add, multiply, or divide thousands of numbers at once, Python loops are too sluggish.

That is why scientists use **NumPy**. NumPy gives you an `ndarray` — a *numerical* array that does maths in compiled C code, not slow Python. You import it like this:

```python
import numpy as np

my_array = np.array([0.3, 0.5, 0.2])
```

Now `my_array.sum()`, `my_array * 2`, and similar operations are fast and concise.

**JAX** goes one step further. It provides `jax.numpy` — an almost identical API to NumPy — but the arrays can run on GPUs and TPUs. That matters for big active-inference agents. From your perspective as a user, `jnp.array(...)` behaves just like `np.array(...)`.

Look at the very top of `pymdp/utils.py`:

```python
import jax
from jax import numpy as jnp, random as jr
import numpy as np
```

Both are imported. pymdp uses JAX arrays for its modern backend, and `jnp` is just a shorthand for `jax.numpy`. When you see `jnp.array([1, 2, 3])`, think of it as exactly the same as `np.array([1, 2, 3])` — just with the option to run on a GPU.

The key takeaway: **a JAX array is a NumPy array with superpowers. The maths you already know applies.**

---

## What Is a Probability Distribution?

A probability distribution is an array of numbers that answers the question: *how likely is each possible outcome?*

There are exactly two rules it must follow:

1. **Every value must be non-negative** — a probability can never be below zero.
2. **All values must sum to exactly 1.0** — something always happens; the total probability is 100%.

Here are some examples:

```python
valid     = [0.3, 0.5, 0.2]   # sums to 1.0, all >= 0  ✓
also_valid = [1.0, 0.0, 0.0]  # sums to 1.0, all >= 0  ✓

invalid_1 = [1, 2, 3]         # sums to 6, not 1        ✗
invalid_2 = [-0.1, 0.6, 0.5]  # has a negative value    ✗
```

In the context of pymdp, a distribution might represent: *where does the agent believe it currently is?* If there are three possible locations, the distribution has three numbers — one probability per location.

The second rule ("sums to 1") is not optional. If you compute anything with an invalid distribution, the results are meaningless. pymdp is very careful to normalise distributions before using them, and that is exactly what the next page is about.

---

## Normalising with `norm_dist`

Suppose you start with an unnormalised array — values that capture the *relative* likelihood of each outcome but do not yet sum to 1:

```python
raw = jnp.array([3.0, 5.0, 2.0])
# sum = 10.0  — not a valid distribution yet
```

To turn this into a valid distribution, you divide every element by the total sum:

```
3/10 = 0.3
5/10 = 0.5
2/10 = 0.2
```

That operation is so fundamental in pymdp that it has its own function: `norm_dist`. Here is the full, real implementation from `pymdp/utils.py`:

```python
def norm_dist(dist: Array) -> Array:
    """Normalizes a Categorical probability distribution.

    Parameters
    ----------
    dist: Array
        Unnormalized Categorical distribution.

    Returns
    -------
    Array
        Normalized distribution.
    """
    return dist / dist.sum(0)
```

That is the entire function body: one line, `dist / dist.sum(0)`.

- `dist.sum(0)` computes the sum along axis 0 (for a 1D array that is simply the total of all elements).
- `dist / ...` divides every element by that total in one vectorised step — no loop needed.

**Why normalise?** Because in practice you often build distributions from counts, random numbers, or guesses — none of which are guaranteed to sum to 1. `norm_dist` is the bridge between "raw numbers" and "valid probability distribution". It is the most frequently called function in the entire library.

---

## Factorised Distributions: Why a List?

Now you know what a single probability distribution looks like. But a real active-inference agent has *multiple* hidden-state factors — for example, one factor for its *location* and another for its *hand position*. How should you store two distributions at once?

The naive answer is a 2D array — one row per factor. The problem: if factor 1 has 4 states and factor 2 has 10 states, a joint representation would need **4 × 10 = 40 entries**. Add a third factor with 5 states and you need **200 entries**. This combinatorial explosion grows exponentially as you add factors.

pymdp avoids this with a much simpler idea: **store one array per factor in a plain Python list**.

```python
# Two independent factors — stored separately
D_location     = jnp.array([0.25, 0.25, 0.25, 0.25])  # 4 locations, uniform prior
D_hand         = jnp.array([0.1, 0.1, 0.8])            # 3 hand positions

D = [D_location, D_hand]   # just a Python list
```

Each factor stays small. No explosion. Each element of the list is an independent distribution and you can reason about them separately.

When you need to normalise every distribution in such a list at once, pymdp provides `list_array_norm_dist`:

```python
def list_array_norm_dist(dist_list: list[Array]) -> list[Array]:
    return jtu.tree_map(lambda dist: norm_dist(dist), dist_list)
```

`jtu.tree_map` is a JAX utility that applies a function to every leaf of a nested structure — here, to every array in the list. The result is a new list where every distribution is normalised. Think of it as "run `norm_dist` on each item in the list".

This pattern — a **Python list of JAX arrays** — appears everywhere in pymdp. Once you recognise it, the rest of the codebase becomes much easier to read.

---

## Summary

In this lesson we covered:

- **NumPy vs. JAX arrays** — both are fast numerical arrays; `jax.numpy` is almost identical to `numpy` but can run on GPUs. pymdp imports both.
- **Valid probability distributions** — an array where all values are non-negative and the total sum equals 1.0.
- **`norm_dist(dist)`** — one line (`dist / dist.sum(0)`) that turns any non-negative array into a valid distribution. The most-called function in pymdp.
- **Factorised representation** — pymdp stores multi-factor distributions as a Python list (one array per factor) to avoid the combinatorial explosion of a joint representation.
- **`list_array_norm_dist`** — applies `norm_dist` to every array in such a list using `jtu.tree_map`.

**Key takeaway:** Every belief, prior, and likelihood in pymdp is just a normalised array. Once you are comfortable with `norm_dist` and the list-of-arrays pattern, you are ready for the rest of the library.

**Next lesson — Lesson 3:** You will meet the **A matrix** — the observation likelihood. Each column of the A matrix is a probability distribution over observations given a hidden state. The "must sum to 1" rule you learned here applies directly: pymdp normalises each column of A using exactly the tools from this lesson.
