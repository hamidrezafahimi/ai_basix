### Summary 

- [Basic Concepts](#section-id-3)
  - [What is Reinforcement Learning](#section-id-5)
  - [Markov_State](#section-id-17)
  - [Mathematics of RL: Markov Decision Process (MDP)](#section-id-42)
  - [What is a Policy?](#section-id-60)
  - [Main Steps to Solve an RL Problem](#section-id-77)
  - [RL Taxonomy](#section-id-100)
    - [Model-Based or Model-Free](#section-id-102)
    - [Based on the Type of the Output](#section-id-112)
      - [Value-Based-RL](#section-id-116)
      - [Policy-Based-RL](#section-id-119)
      - [Model-Based-RL](#section-id-122)
- [Traditional RL vs. DRL](#section-id-127)
  - [Bellman Functions: The Traditional Way to Find the Optimal Policy](#section-id-141)
    - [Value_Function](#section-id-145)
    - [Q_Function](#section-id-167)
  - [Deep Reinforcement Learning](#section-id-183)
- [Review of Major DRL Methods](#section-id-198)
  - [Environment-Model-Based RL Methods (Brief Overview)](#section-id-200)
  - [Deep Q-Learning (DQN): A Value-Based RL Method](#section-id-227)
    - [What is a Value-Based RL Method?](#section-id-229)
    - [DQN Overview](#section-id-242)
    - [Algorithm](#section-id-260)
    - [$psilongreedy policy for taking actions](#section-id-293)
    - [Limitations](#section-id-305)
  - [Policy Gradients: A Policy-Based RL Method](#section-id-312)
    - [Mathematical Base](#section-id-333)
    - [Algorithm](#section-id-344)
  - [Actor-Critic Algorithm](#section-id-351)
  




<div id='section-id-3'/>

# Basic Concepts

<div id='section-id-5'/>

## What is Reinforcement Learning

The structure of reinforcement leaning paradigm is based on *reward* and *punishment*. In a reinforcement learning problem, always:

(There is an) **ACTION** --> (done by an *AGENT* on an *ENVIRONMENT*, changing a) **STATE (OBSERVATIONS)** --> (resulting a) **REWARD** (based on a certain criteria, which is given to the agent with a number of steps delay) 


<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl_loop.png"/>
</p>


<div id='section-id-17'/>

## Markov_State

A *Markov State* Is a state in which all the useful information of the history lies. A state is a markov state if and only if:

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


<div id='section-id-42'/>

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



<div id='section-id-60'/>

## What is a Policy?

A *Policy* is a "state -> action" function, determining what are the choices of action in each state. A policy is designated by the designer. An example:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_function.png"/>
</p>

In the above, the table cells are the states and movement in 4 directions is the action.

*There is a proof for the fact that one or multiple policies exist in an RL problem.*

A policy may be deterministic or nondeterministic:
- A *deterministic policy* determines exactly what action must be performed in each state
- A *nondeterministic policy* gives a probability distribution over the possible action, claiming the possibility to be the optimal case, for each action. 


<div id='section-id-77'/>

## Main Steps to Solve an RL Problem

These are the major parts of a reinforcement learning problem:

1- *Agent* - What we're going to train, which experiments variable states through time
2- *Environment* - Everything in the world, except out solution and agent
3- *Reward Function* - Is obtained based on the design, as a result of each action the agent performs
4- *Training Algorithm* - Is done in the same time as testing is done

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl_block_diagram.png"/>
</p>

In an RL problem, one must:

1- Define exactly what is the agent and what is the environment
2- Define the *Action* and *State* vectors. What actions are possible in each state?
3- How can actions change the environment and how is the reward calculated? (based on single or multiple criterias - fastness? or accuracy? or ...)
4- With what algorithm the model parameters must be optimized?

*The goal in an RL problem, is to get to an optimal policy.*


<div id='section-id-100'/>

## RL Taxonomy

<div id='section-id-102'/>

### Model-Based or Model-Free

First of all, here are some examples of RL methods devided by two major fields, Model-Based or Model-Free:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl-taxonomy.png"/>
</p>

The main taxonomy of RLs, is based on *What an RL Agent May Learn*. There are three major fields:

<div id='section-id-112'/>

### Based on the Type of the Output

An RL solution can be classified based on its nature:

<div id='section-id-116'/>

#### Value-Based-RL
- A *Value-Based RL* returns the rewards: There is a function predicting the reward of quantized states/states-and-actions (-> value-function/Q-function)

<div id='section-id-119'/>

#### Policy-Based-RL
- A *Policy-Based RL* returns the actions: It generates a probability distribution on action space (may be continuous), indicating the best action point

<div id='section-id-122'/>

#### Model-Based-RL
- A *Model-Based RL* approximates a model for the environment and predicts the environment's behavior.



<div id='section-id-127'/>

# Traditional RL vs. DRL

The traditional RL has been introduced since 1950s. But it has not been used in real-world problems until 2000s. The major issue was:

*The traditional RL methods where based on descritized space of states and action. They where not useful in real problems with continuous nature or numerous descritized states and actions.*

In 2000s, Deep Reinforcement Learning (DRL) was introduced; And in real-world problems, it worked really good! 

Here is the main difference between the traditional RL and DRL:

- In the traditional RL, an optimal policy is obtained based on *bellman functions*
- In DRL, the function to find the optimal policy is estimated with a deep neural network


<div id='section-id-141'/>

## Bellman Functions: The Traditional Way to Find the Optimal Policy

As follows, there are two utilities to get to an optimal policy: *Value-Function* and *Q-Function*. Notice the term $\pi$ in the two following equations. It means the calculation is related to a specific policy.

<div id='section-id-145'/>

### Value_Function

Starting at each state, there are lots of possible state trajectories (*episodes*) to get to desired target state. The expected value for are possibilities, is th *value* returned by a value function. For a simple 4-state problem, the values returned by a value function may be like the following. Knowing the aforementioned value for each state, helps deciding about the next step at each initial step. For example, in the following, considering the `S2` as a goal, starting at the state `S0`, is more optimal to go to `S1` rather than `S3`.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/value_func.png"/>
</p>

Bellman states a mathematical expression for the value function:

$$
v_\pi(s) = E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s]
$$

Leading to:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/val_func.png"/>
</p>

The roll of $\gamma$ is explained [previously](#markovstate)

<div id='section-id-167'/>

### Q_Function

A Q-function retuens the expected value of possible rewards for all episodes, starting at a *each specific state* and *taking each specific action*.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/Q_function.png"/>
</p>

*NOTE:* A value function can be determined having a Q-function. But the opposite is not true.

Bellman states a mathematical expression for the value function:

$$
q_\pi(s, a) = E_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
$$

<div id='section-id-183'/>

## Deep Reinforcement Learning

With help of deep neural networks (DNNs), the (numerous-descritized- or) continuous-state/action-space problems can be solved. Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/drl.png"/>
</p>

There are three main paradigms in DRL:

* **Critic-Only:** Estimating a value function using a DNN [just as described](#Value-Based-RL). The output of such networks is as wide as the whole action space (for the input state), claiming a value for each action (for the input state).
* **Actor-Only:** Direct estimation of policy function using a DNN [just as described](#Policy-Based-RL). The output of sunch network is a single action
* **Actor-Critic**


<div id='section-id-198'/>

# Review of Major DRL Methods

<div id='section-id-200'/>

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

<div id='section-id-227'/>

## Deep Q-Learning (DQN): A Value-Based RL Method

<div id='section-id-229'/>

### What is a Value-Based RL Method?

In this methods, a *value function* is to be learned, i.e., an NN is trained as an approximation of a *value function*.

The question that the agent is to learn its anwer is: How much a state/action is likely to reward me in the future? The function's mathematical expression is declared [here](#Value_Function)

Approach:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/value-based-rl.png"/>
</p>


<div id='section-id-242'/>

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

<div id='section-id-260'/>

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


<div id='section-id-293'/>

### $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1/$\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 


<div id='section-id-305'/>

### Limitations

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.


<div id='section-id-312'/>

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


<div id='section-id-333'/>

### Mathematical Base

In this method, a *Policy Gradient* is defined with a mathematical gradient ascend approach, to maximize a reward based on a policy. The following figures describe how this Policy Gradient is obtained.

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_1.png"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_2.png"/>
</p>

<div id='section-id-344'/>

### Algorithm

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_rl_alg.png"/>
</p>


<div id='section-id-351'/>

## Actor-Critic Algorithm

This method is a compound of *Q-Learning* and *Policy Gradients*:

- The *Actor* is the policy; It determines the action
- The *Critic* is the Q-function; It evaluates the action
