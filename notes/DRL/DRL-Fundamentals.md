### Summary 

- [Basic Concepts](#section-id-2)
  - [What is Reinforcement Learning](#section-id-4)
  - [Markov_State](#section-id-16)
  - [Mathematics of RL: Markov Decision Process (MDP)](#section-id-52)
  - [What is a Policy?](#section-id-70)
  - [Main Steps to Solve an RL Problem](#section-id-87)
  - [RL Taxonomy](#section-id-110)
    - [Value_Based_RL](#section-id-128)
    - [Policy_Based_RL](#section-id-137)
    - [Model_Based_RL](#section-id-156)
  - [Traditional RL vs. DRL](#section-id-184)
- [RL Mathematics](#section-id-215)
  - [Value-Based RL Mathematics: Bellman Functions](#section-id-217)
    - [Value_Function](#section-id-221)
    - [Q_Function](#section-id-243)
  - [Policy-Based RL Mathematics](#section-id-260)
- [DRL Leraning Methods](#section-id-290)
  - [Monte-Carlo Methods](#section-id-294)
    - [Training Algorithm](#section-id-299)
    - [Samples](#section-id-307)
  - [Temporal Difference](#section-id-315)
  



<div id='section-id-2'/>

# Basic Concepts

<div id='section-id-4'/>

## What is Reinforcement Learning

The structure of reinforcement leaning paradigm is based on *reward* and *punishment*. In a reinforcement learning problem, always:

(There is an) **ACTION** --> (done by an *AGENT* on an *ENVIRONMENT*, changing a) **STATE (OBSERVATIONS)** --> (resulting a) **REWARD** (based on a certain criteria, which is given to the agent with a number of steps delay) 


<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/rl_loop.png?raw=true", width="600"/>
</p>


<div id='section-id-16'/>

## Markov_State

A *Markov State* Is a state in which all the useful information of the history lies. A state is a markov state if and only if:

$$
  p(S_{t+1}|S_t) = p(S_{t+1}|S_1, S_2, ..., S_t)
$$

I.e., future is independent of past, given present.

The concept of *total reward* in the mentioned structure is shown in the following:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/rl_total_reward.png?raw=true", width="600"/>
</p>

To avoid an infinite total reward:

$$
  R_t = \gamma^0r_t + \gamma^1r_{t+1} + \gamma^2r_{t+2} + ...
$$

The above, is called the *Discounted Sum of Future Rewards* (Sometimes I abbreviate it by *DSoFR*). 
During training phase a DRL solution, the above statement is calculated for all time-steps of each episode, to be used in the training. A schmatics of this algorithm is depicted in the following:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/G_calc_in_PG.png?raw=true", width="600"/>
</p>

Discounting the future rewards until the end (like above) has two advantages:
1. You can control the future rewards so that it doesn't sum up to infinity in future, so that you have limited reward values needed for the optimization of network parameters. (If there is no limitation for rewards, than no maximization of reward is valid)
2. There are uncertainties aboutreceiving the future rewards, as much as the distance to current instant increases. So the parameter $\gamma$ leads to consideration of this.

The objective in a reinforcement learning problem, is to **select actions to maximize the future reward** (above).


<div id='section-id-52'/>

## Mathematics of RL: Markov Decision Process (MDP)

An MDP is defined by five properties: (S, A, R, P, $\gamma$)

- Satets (S)
- Actions (A)
- Rewards (R)
- Probabilities (P)
- Discount Factor (0<$\gamma$<1)

The following graph is an example an MDP. For each state, there are allowed actions. For each allowed action, there is resulting states with certain probabilities for each.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/mdp-graph.png?raw=true", width="600"/>
</p>



<div id='section-id-70'/>

## What is a Policy?

A *Policy* is a "state -> action" function, determining what are the choices of action in each state, i.e. A policy is a probability distribution which the agent uses to pick actions. It is actually the solution of s reinforcement learning problem. An example:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/policy_function.png?raw=true", width="600"/>
</p>

In the above, the table cells are the states and movement in 4 directions is the action.

*There is a proof for the fact that one or multiple policies exist in an RL problem.*

A policy may be deterministic or nondeterministic:
- A *deterministic policy* determines exactly what action must be performed in each state
- A *nondeterministic policy* gives a probability distribution over the possible action, claiming the possibility to be the optimal case, for each action. 


<div id='section-id-87'/>

## Main Steps to Solve an RL Problem

These are the major parts of a reinforcement learning problem:

1- *Agent* - What we're going to train, which experiments variable states through time
2- *Environment* - Everything in the world, except out solution and agent
3- *Reward Function* - Is obtained based on the design, as a result of each action the agent performs
4- *Training Algorithm* - Is done in the same time as testing is done

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/rl_block_diagram.png?raw=true", width="600"/>
</p>

In an RL problem, one must:

1- Define exactly what is the agent and what is the environment
2- Define the *Action* and *State* vectors. What actions are possible in each state?
3- How can actions change the environment and how is the reward calculated? (based on single or multiple criterias - fastness? or accuracy? or ...)
4- With what algorithm the model parameters must be optimized?

*The goal in an RL problem, is to get to an optimal policy.*


<div id='section-id-110'/>

## RL Taxonomy

This is the main taxonomy of RL methods:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/major_rl_taxonomy.png?raw=true", width="600"/>
</p>

In the above, the *model* means a *State Ttansition Function* that predict the next states and rewards during the next transition. Here are some examples of RL methods devided by two major fields, Model-Based or Model-Free:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/rl-taxonomy.png?raw=true", width="600"/>
</p>

The main taxonomy of RLs, is based on *What an RL Agent May Learn*. There are three major fields:

An RL solution can be classified based on its nature:

<div id='section-id-128'/>

### Value_Based_RL

A *Value-Based RL* is to find the rewards: There is a function predicting the reward of quantized states/states-and-actions (-> value-function/Q-function)

In this methods, an NN is trained as an approximation of a *value function*.

The question that the agent is to learn its anwer is: How much a state/action is likely to reward me in the future? The function's mathematical expression is declared [here](#Value_Function)


<div id='section-id-137'/>

### Policy_Based_RL

A *Policy-Based RL* is to find the actions: It generates a probability distribution on action space (may be continuous), indicating the best action point. In this methods, a *policy* is to be learned. A network is trained so that it generates probability for more reward for each action. The output of such network, is a probability distribution on the action space which its maxima determines the best choice.

The question that the agent is to learn its anwer is: What decision I have to make in a particular situation?

A policy may be *deterministic*:

$$
a = \pi(s)
$$

Or *stochastic*:

$$
\pi(a|s) = p[A=a|S=s]
$$


<div id='section-id-156'/>

### Model_Based_RL

A *Model-Based RL* approximates a model for the environment and predicts the environment's behavior. In this methods, a *model of the environment* is to be learned.

The question that the agent is to learn its answer is: What will happen to the environment in the next step? (= Environment Model)

There are to types for this:

Approach:

- Learn the environment model
- Plan using the model
- Update the model often
- Re-plan often


- A **Transition Model** predicts what is going to happen for known state and action.

$$
p^a_{ss'} = p[S'=s' | S=s, A=a]
$$

- A **Rewards Model** predicts what reward is going to be obtained for known state and action.

$$
R^a_s = E[R| S=s, A=a]
$$

<div id='section-id-184'/>

## Traditional RL vs. DRL

The traditional RL has been introduced since 1950s. But it has not been used in real-world problems until 2000s. The major issue was:

*The traditional RL methods where based on descritized space of states and action. They where not useful in real problems with continuous nature or numerous descritized states and actions.*

In 2000s, Deep Reinforcement Learning (DRL) was introduced; And in real-world problems, it worked really good! 

Here is the main difference between the traditional RL and DRL:

- In the traditional RL, an optimal policy is obtained based on *bellman functions*
- In DRL, the function to find the optimal policy is estimated with a deep neural network

With help of deep neural networks (DNNs), the (numerous-descritized- or) continuous-state/action-space problems can be solved. Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/drl.png?raw=true", width="600"/>
</p>

There are three main paradigms in DRL:

* **Critic-Only:** Estimating a value function using a DNN [just as described](#Value-Based-RL). The output of such networks is as wide as the whole action space (for the input state), claiming a value for each action (for the input state).

* **Actor-Only:** Direct estimation of policy function using a DNN [just as described](#Policy_Based_RL). The output of sunch network is a single action

* **Actor-Critic**
This method is a compound of value-based approaches (*Q-Learning*) and Policy-Based approaches (*Policy Gradients*):

- The *Actor* is the policy; It determines the action
- The *Critic* is the Q-function; It evaluates the action

<div id='section-id-215'/>

# RL Mathematics

<div id='section-id-217'/>

## Value-Based RL Mathematics: Bellman Functions

As follows, there are two utilities to get to an optimal policy: *Value-Function* and *Q-Function*. Notice the term $\pi$ in the two following equations. It means the calculation is related to a specific policy.

<div id='section-id-221'/>

### Value_Function

Starting at each state, there are lots of possible state trajectories (*episodes*) to get to desired target state. The expected value for are possibilities, is th *value* returned by a value function. For a simple 4-state problem, the values returned by a value function may be like the following. Knowing the aforementioned value for each state, helps deciding about the next step at each initial step. For example, in the following, considering the `S2` as a goal, starting at the state `S0`, is more optimal to go to `S1` rather than `S3`.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/value_func.png?raw=true", width="600"/>
</p>

Bellman states a mathematical expression for the value function:

$$
v_\pi(s) = E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s]
$$

Leading to:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/val_func.png?raw=true", width="600"/>
</p>

The roll of $\gamma$ is explained [previously](#markovstate)

<div id='section-id-243'/>

### Q_Function

A Q-function retuens the expected value of possible rewards for all episodes, starting at a *each specific state* and *taking each specific action*.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/Q_function.png?raw=true", width="600"/>
</p>

*NOTE:* A value function can be determined having a Q-function. But the opposite is not true.

Bellman states a mathematical expression for the value function:

$$
q_\pi(s, a) = E_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
$$


<div id='section-id-260'/>

## Policy-Based RL Mathematics

 A *Policy Gradient* is defined with a gradient ascend approach, to maximize a reward based on a policy. 

For each policy $\pi$, there are a set of aprameters $\theta_t$ (network weights in DRL) 

$$
\Pi = {\pi_\theta, \theta \in R^m}
$$

For which a reward function is defined:

$$
r(\tau)
$$

And so a cost function $J(\theta_t)$: 

$$
J(\theta) = E_\pi[r(\tau)]
$$

The main objective is to maximize the reward (= the expected value of possible rewards - the cost function). To do so, the parameters must be updated based on a *Gradient Ascend* approach:

$$
\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)
$$



<div id='section-id-290'/>

# DRL Leraning Methods

There are different learning strategies in DRL, each subjected in one of the following sections.

<div id='section-id-294'/>

## Monte-Carlo Methods

In these methods, the training (updating the network parameters) happens after playing a game, i.e., each single episode. A full set of states, actions, and rewards are given to the training-handler function after a full game is been played. These data are iterated and used then, to update the network.


<div id='section-id-299'/>

### Training Algorithm 

The following figure demonstrates the algorithm of training the DRL network in monte-carlo methods as a block diagram:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/monte_carlo_schematics.png?raw=true", width="600"/>
</p>

<div id='section-id-307'/>

### Samples

The followings are some samples of monte-carlo based DRL methods:

- Policy-Gradients (PG)
- ...


<div id='section-id-315'/>

## Temporal Difference

... In these methods, the training procedure is done step-by-step, after each time-step within an episode.
