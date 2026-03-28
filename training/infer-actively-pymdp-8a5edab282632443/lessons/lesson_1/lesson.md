# Lesson 1: What is pymdp and Active Inference?

**Repository:** infer-actively-pymdp
**Lesson:** 1 of 12

Welcome to the first lesson on `pymdp` — a Python package for building and simulating active inference agents in discrete environments. By the end of this lesson you will have a clear mental model of what problem this library solves, why it was built, and the key vocabulary you will use throughout every lesson that follows.

Here's what we'll cover today:

1. The problem pymdp was built to solve — and where it came from
2. Active inference in plain English — what it means to "minimise free energy"
3. What a POMDP is, and why it's the right model for uncertain environments
4. The perception-action loop — the five-step heartbeat of every pymdp simulation

If any of these terms feel unfamiliar right now, that's fine. By the end of this lesson they'll all click into place.

---

## The Problem pymdp Was Built to Solve

Before `pymdp` existed, researchers who wanted to simulate active inference agents had one real option: a MATLAB toolbox called `DEM`, buried inside a neuroimaging package called `SPM`. This was powerful software, but it had serious limitations. It required a paid MATLAB license. Its core logic lived in a single, monolithic function — `spm_MDP_VB_X.m` — that was difficult to read, modify, or extend. And it was designed for neuroscientists, not software engineers.

As active inference grew in popularity — moving from computational neuroscience into robotics, control theory, reinforcement learning, and even social cognition — the community needed something better. Researchers needed an open-source, Pythonic, modular tool they could actually build on.

That's exactly what `pymdp` is. Its own `setup.cfg` describes it simply: *"A Python package for solving Markov Decision Processes with Active Inference."*

The library was developed by researchers at the Max Planck Institute, VERSES Research Lab, Oxford, UCL, and others. Most of the low-level mathematical operations were ported from `SPM`'s MATLAB routines, benchmarked, and validated to match their original counterparts — but now available for anyone to use, extend, and contribute to.

The result is a library that lets you define an intelligent agent, give it a model of the world, and watch it perceive, reason, and act — all in a few lines of Python.

---

## Active Inference in Plain English

Active inference sounds abstract, but the core idea is surprisingly intuitive. Here it is in one sentence:

> An active inference agent is always trying to minimise its own surprise about the world.

That's it. "Surprise" here has a specific technical meaning — it's related to the difference between what the agent expects to observe and what it actually observes. This quantity is called **free energy**. Minimising free energy is the single unifying objective that drives everything an agent does: perceiving, acting, and even learning.

Why does this matter? Because it means you don't have to hand-engineer separate mechanisms for curiosity, exploration, and goal-seeking. They all emerge naturally from this one principle.

- **Curiosity** emerges because an agent with high uncertainty about the world can reduce its surprise by actively gathering information. This is called **epistemic value** — the value of an action in terms of how much it reduces uncertainty.
- **Goal-seeking** emerges because the agent also has preferences about what it expects to observe. It treats preferred outcomes as things it "predicts" will happen, and then acts to make those predictions come true.

The README's "epistemic chaining" demo illustrates this beautifully: a simulated mouse forages through a chain of cues to find hidden food. Nobody programmed "follow the cues." The agent does it automatically, because each cue reduces uncertainty and gets it closer to its preferred outcome.

This framework was originally developed by neuroscientist Karl Friston as an account of how the brain works — explaining both perception and action as two sides of the same coin: inference. The name "active inference" captures exactly this: inference that is *active*, i.e. the agent doesn't just passively observe the world, it takes actions to confirm its predictions.

---

## What is a POMDP?

Now that you understand the goal, you need to understand the stage on which pymdp agents perform. That stage is a **Partially-Observed Markov Decision Process**, or **POMDP**.

Think of a mouse navigating a maze. The mouse cannot see the entire maze at once — it can only see the corridor immediately in front of it. The true layout of the maze is the **hidden state**: it exists, it matters, but the mouse can't observe it directly. All the mouse has are **observations** — the sights and sounds it receives at each moment — which give it partial, noisy hints about where it actually is.

This is the essence of a POMDP:
- The world has **hidden states** the agent cannot see directly.
- The agent receives **observations** that are generated by those hidden states.
- The agent must **infer** what the hidden state probably is, given its observations.
- The agent can take **actions** that change the hidden state over time.

In `pymdp`, every POMDP is specified by four components, each represented by a matrix or vector. You'll study each of these in depth in Lessons 3–5, but for now, just learn the names:

| Label | Name | What it represents |
|-------|------|--------------------|
| **A** | Observation model | How hidden states generate observations |
| **B** | Transition model | How actions change hidden states over time |
| **C** | Preferences | Which observations the agent prefers (its "goals") |
| **D** | Initial state prior | What the agent believes about the world before it starts |

The **A/B/C/D** vocabulary is the universal language of pymdp. You'll see it everywhere — in function names, documentation, and paper figures. Getting comfortable with these four letters is one of the most important things you can do right now.

---

## The Perception-Action Loop

Now you know what a POMDP is. But how does an agent actually *run* inside one? The answer is the **perception-action loop** — a five-step cycle that repeats at every time step of a simulation.

Here's the loop:

```
1. Receive an observation from the environment
2. Infer the current hidden state (using A and D)
3. Evaluate possible future actions (using B and C)
4. Sample an action from the best policy
5. Execute the action; the environment transitions to a new state
   → Go back to step 1
```

Each step maps directly onto the A/B/C/D components:

- **Step 2 (infer_states)** uses the **A** matrix — it asks: "Given what I just observed, what hidden state am I probably in?"
- **Step 3 (infer_policies)** uses **B** and **C** — it asks: "If I take each possible action, where will I end up, and how much do I prefer that outcome?"
- **Step 4 (sample_action)** picks an action based on the policy probabilities.

You can see this loop reflected directly in the quick-start code from the README:

```python
from pymdp.agent import Agent

# ... (A, B, C matrices defined earlier)
agent = Agent(A=A, B=B, C=C, batch_size=1)

# Step 1: receive an observation
observation = [jnp.array([1]), jnp.array([4])]

# Step 2: infer the current hidden state
qs, info = agent.infer_states(observation, empirical_prior=agent.D, return_info=True)

# Step 3: evaluate policies
q_pi, neg_efe = agent.infer_policies(qs)

# Step 4: sample an action
action = agent.sample_action(q_pi, rng_key=action_keys[1:])
```

Don't worry about what `jnp.array`, `empirical_prior`, or `batch_size` mean yet — those details are for Lessons 2 and beyond. What matters right now is the *shape* of the loop: observe, infer state, infer policy, act, repeat. This four-method sequence — `infer_states` → `infer_policies` → `sample_action` — is the heartbeat of every pymdp simulation you will ever write.

---

## Summary

In this lesson we covered:

- **Why pymdp exists:** It's the first open-source Python library for active inference in discrete state spaces, replacing a difficult-to-use MATLAB toolbox and making the framework accessible to researchers and engineers across many disciplines.
- **Active inference:** Agents minimise free energy — a measure of surprise about observations. Curiosity (epistemic value) and goal-seeking emerge automatically from this single principle, without hand-engineering.
- **POMDPs:** The mathematical model pymdp uses to represent uncertain environments. Hidden states, noisy observations, and actions are the core ingredients. The four components A, B, C, D specify the entire generative model.
- **The perception-action loop:** At every time step, an agent observes → infers state → evaluates policies → acts. The three pymdp methods `infer_states`, `infer_policies`, and `sample_action` implement each of these steps.
- **Two backends:** The modern JAX-based backend (`pymdp/`) and a legacy NumPy backend (`pymdp/legacy/`). This course focuses on the JAX backend.

**Key takeaway:** Everything in pymdp — every matrix, every function, every tutorial — is an elaboration of one idea: an agent with a model of the world (A/B/C/D) running the perception-action loop to minimise surprise.

**Next lesson:** We'll get hands-on with the data structures pymdp uses under the hood. Lesson 2 introduces JAX arrays, probability distributions, and the utility functions in `pymdp/utils.py` that you'll use to build the A, B, C, and D components from scratch.
