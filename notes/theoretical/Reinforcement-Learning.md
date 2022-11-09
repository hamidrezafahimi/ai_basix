### Summary 

- [Basic Concept](#section-id-3)
  - [What is Reinforcement Learning](#section-id-5)
  - [Mathematics of RL: Markov Decision Process (MDP)](#section-id-40)
  - [About Deep Reinforcement Learning](#section-id-57)
- [RL Taxonomy](#section-id-66)
  - [Environment-Model-Based RL](#section-id-76)
  - [Policy-Based RL](#section-id-101)
  - [Value-Based RL](#section-id-116)
  




<div id='section-id-3'/>

# Basic Concept

<div id='section-id-5'/>

## What is Reinforcement Learning

The structure of reinforcement leaning paradigm is based on *reward* and *punishment*. In a reinforcement learning problem, always:

(There is an) **ACTION** --> (done by an *AGENT* on an *ENVIRONMENT*, changing a) **STATE (OBSERVATIONS)** --> (resulting a) **REWARD** (based on a certain criteria, which is given to the agent with a number of steps delay) 


<p align="center">
  <img src="../rl_loop.png"/>
</p>


**Markov State:** Is a state in which all the useful information of the history lies. A state is a markov state if and only if:

$$
  p(S_{t+1}|S_t) = p(S_{t+1}|S_1, S_2, ..., S_t)
$$

I.e., future is independent of past, given present.

The concept of *total reward* in the mentioned structure is shown in the following:

<p align="center">
  <img src="../rl_total_reward.png"/>
</p>

To avoid an infinite total reward:

$$
  R_t = \gamma^0r_t + \gamma^1r_{t+1} + \gamma^2r_{t+2} + ...
$$

The objective in a reinforcement learning problem, is to **select actions to maximize the future reward** (above).


<div id='section-id-40'/>

## Mathematics of RL: Markov Decision Process (MDP)

An MDP is defined by five properties: (S, A, R, P, $\gamma$)

- Satets (S)
- Actions (A)
- Rewards (R)
- Probabilities (P)
- Discount Factor (0<$\gamma$<1)

The following graph is an example an MDP. For each state, there are allowed actions. For each allowed action, there is resulting states with certain probabilities for each.

<p align="center">
  <img src="../mdp-graph.png"/>
</p>


<div id='section-id-57'/>

## About Deep Reinforcement Learning

Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="../drl.png"/>
</p>


<div id='section-id-66'/>

# RL Taxonomy

First of all, here are some examples of RL methods devided by two major fields, Model-Based or Model-Free:

<p align="center">
  <img src="../rl-taxonomy.png"/>
</p>

The main taxonomy of RLs, is based on *What an RL Agent May Learn*. There are three major fields:

<div id='section-id-76'/>

## Environment-Model-Based RL

The question that the agent is to learn its anwer is: What will happen to the environment in the next step? (= Environment Model)

There are to types for this:

Approach:

<p align="left">
  <img src="../model-based-rl.png"/>
</p>


- A **Transition Model** predicts what is going to happen for known state and action.

$$
p^a_{ss'} = p[S'=s' | S=s, A=a]
$$

- A **Rewards Model** predicts what reward is going to be obtained for known state and action.

$$
R^a_s = E[R| S=s, A=a]
$$

<div id='section-id-101'/>

## Policy-Based RL

The question that the agent is to learn its anwer is: What decision I have to make in a particular situation?

<p align="center">
  <img src="../policy.png"/>
</p>

Approach:

<p align="left">
  <img src="../policy-based-rl.png"/>
</p>


<div id='section-id-116'/>

## Value-Based RL

The question that the agent is to learn its anwer is: How much a state/action is likely to reward me in the future?

<p align="center">
  <img src="../val_func.png"/>
</p>

Approach:

<p align="left">
  <img src="../value-based-rl.png"/>
</p>



