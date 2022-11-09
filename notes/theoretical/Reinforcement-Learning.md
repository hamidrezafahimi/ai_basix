### Summary 

- [Basic Concepts](#section-id-3)
  - [What is Reinforcement Learning](#section-id-5)
  - [Mathematics of RL: Markov Decision Process (MDP)](#section-id-40)
  - [About Deep Reinforcement Learning](#section-id-57)
  - [RL Taxonomy](#section-id-66)
    - [Model-Based or Model-Free](#section-id-68)
    - [Based on the Type of the Output](#section-id-78)
- [Review of Major RL Methods](#section-id-90)
  - [Environment-Model-Based RL Methods (Brief Overview)](#section-id-92)
  - [Deep Q-Learning (DQN): A Value-Based RL Method](#section-id-119)
    - [What is a Value-Based RL Method?](#section-id-121)
    - [DQN Overview](#section-id-139)
    - [Algorithm](#section-id-157)
    - [$psilongreedy policy for taking actions](#section-id-190)
    - [Limitations](#section-id-202)
  - [Policy Gradients: A Policy-Based RL Method](#section-id-209)
    - [Mathematical Base](#section-id-230)
    - [Algorithm](#section-id-241)
  - [Actor-Critic Algorithm](#section-id-248)
  




<div id='section-id-3'/>

# Basic Concepts

<div id='section-id-5'/>

## What is Reinforcement Learning

The structure of reinforcement leaning paradigm is based on *reward* and *punishment*. In a reinforcement learning problem, always:

(There is an) **ACTION** --> (done by an *AGENT* on an *ENVIRONMENT*, changing a) **STATE (OBSERVATIONS)** --> (resulting a) **REWARD** (based on a certain criteria, which is given to the agent with a number of steps delay) 


<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl_loop.png"/>
</p>


**Markov State:** Is a state in which all the useful information of the history lies. A state is a markov state if and only if:

$$
  p(S_{t+1}|S_t) = p(S_{t+1}|S_1, S_2, ..., S_t)
$$

I.e., future is independent of past, given present.

The concept of *total reward* in the mentioned structure is shown in the following:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl_total_reward.png"/>
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
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/mdp-graph.png"/>
</p>


<div id='section-id-57'/>

## About Deep Reinforcement Learning

Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/drl.png"/>
</p>


<div id='section-id-66'/>

## RL Taxonomy

<div id='section-id-68'/>

### Model-Based or Model-Free

First of all, here are some examples of RL methods devided by two major fields, Model-Based or Model-Free:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl-taxonomy.png"/>
</p>

The main taxonomy of RLs, is based on *What an RL Agent May Learn*. There are three major fields:

<div id='section-id-78'/>

### Based on the Type of the Output

An RL solution can be classified based on its nature:

- A **Value-Based RL** predicts the reward of quantized actions

- A **Policy-Based RL** generates a probability distribution on action space (may be continuous), indicating the best action point

- A **Model-Based RL** approximates a model for the environment and predicts the environment's behavior.



<div id='section-id-90'/>

# Review of Major RL Methods

<div id='section-id-92'/>

## Environment-Model-Based RL Methods (Brief Overview)

In this methods, a *model of the environment* is to be learned.

The question that the agent is to learn its answer is: What will happen to the environment in the next step? (= Environment Model)

There are to types for this:

Approach:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/model-based-rl.png"/>
</p>


- A **Transition Model** predicts what is going to happen for known state and action.

$$
p^a_{ss'} = p[S'=s' | S=s, A=a]
$$

- A **Rewards Model** predicts what reward is going to be obtained for known state and action.

$$
R^a_s = E[R| S=s, A=a]
$$

<div id='section-id-119'/>

## Deep Q-Learning (DQN): A Value-Based RL Method

<div id='section-id-121'/>

### What is a Value-Based RL Method?

In this methods, a *value function* is to be learned, i.e., an NN is trained as an approximation of a *value function*.

The question that the agent is to learn its anwer is: How much a state/action is likely to reward me in the future?

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/val_func.png"/>
</p>

Approach:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/value-based-rl.png"/>
</p>

Some major algorithm in this area are reviewed in the followings:

<div id='section-id-139'/>

### DQN Overview

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/q-learning.png"/>
</p>

As shown in graph, a *Q* function (approximated with an NN) gives *Expected Total Reward* (value) of a each possible action *a* taken in an input state *s*:

$$
Q(s,a) = E[R_t]
$$

So a policy to take the actions which maximize the Q function, is required:

$$
\pi^* = argmax_a Q(s,a)
$$

<div id='section-id-157'/>

### Algorithm

* Conceptual major steps::

- In initial state, Q = 0
- For each state *s*, find Q for all possible actions.
- Choose the action for which:

$$
a = \pi^*
$$

* pseudo code:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/q_learning_pseudo_code.png"/>
</p>

* Optimization approach is based on Gradient Descent

* Q-Update functon:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/value-based-q-update-func.png"/>
</p>

* Loss function: The loss function is based on MSE:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/q_learning_loss_function.png"/>
</p>


<div id='section-id-190'/>

### $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1/$\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 


<div id='section-id-202'/>

### Limitations

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.


<div id='section-id-209'/>

## Policy Gradients: A Policy-Based RL Method

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_RL.png"/>
</p>

In this methods, a *policy* is to be learned. A network is trained so that it generates probability for more reward for each action. The output of such network, is a probability distribution on the action space which its maxima determines the best choice.

The question that the agent is to learn its anwer is: What decision I have to make in a particular situation?

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy.png"/>
</p>

Approach:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy-based-rl.png"/>
</p>


<div id='section-id-230'/>

### Mathematical Base

In this method, a *Policy Gradient* is defined with a mathematical gradient ascend approach, to maximize a reward based on a policy. The following figures describe how this Policy Gradient is obtained.

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_1.png"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_2.png"/>
</p>

<div id='section-id-241'/>

### Algorithm

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_rl_alg.png"/>
</p>


<div id='section-id-248'/>

## Actor-Critic Algorithm

This method is a compound of *Q-Learning* and *Policy Gradients*:

- The *Actor* is the policy; It determines the action
- The *Critic* is the Q-function; It evaluates the action
