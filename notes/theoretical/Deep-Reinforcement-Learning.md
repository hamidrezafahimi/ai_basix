### Summary 

- [Basic Concepts](#section-id-3)
  - [What is Reinforcement Learning](#section-id-5)
  - [Markov_State](#section-id-17)
  - [Mathematics of RL: Markov Decision Process (MDP)](#section-id-42)
  - [What is a Policy?](#section-id-60)
  - [Main Steps to Solve an RL Problem](#section-id-77)
  - [RL Taxonomy](#section-id-100)
    - [Value_Based_RL](#section-id-112)
    - [Policy_Based_RL](#section-id-121)
    - [Model_Based_RL](#section-id-140)
  - [Traditional RL vs. DRL](#section-id-168)
- [RL Mathematics](#section-id-199)
  - [Value-Based RL Mathematics: Bellman Functions](#section-id-201)
    - [Value_Function](#section-id-205)
    - [Q_Function](#section-id-227)
  - [Policy-Based RL Mathematics](#section-id-244)
    - [More Detailed Math](#section-id-272)
    - [Policy-Based RL: A Model-Free RL](#section-id-281)
- [Review of Major DRL Methods](#section-id-298)
  - [Deep Q-Learning (DQN): A Value-Based RL Method](#section-id-300)
    - [Traditional Q-Learning Algorithm](#section-id-304)
    - [Raw DQN Algorithm](#section-id-326)
    - [Revised (Practical) DQN](#section-id-350)
      - [Issue-1 In Loss Function: Q](#section-id-354)
      - [Issue-2 In Loss Function: target](#section-id-362)
      - [$psilongreedy policy for taking actions](#section-id-370)
      - [Revised (Final) DQN Algorithm](#section-id-382)
      - [Loss function](#section-id-389)
    - [DQN Architecture](#section-id-398)
    - [Limitations of DQN](#section-id-407)
  - [A Policy-Based RL Method](#section-id-414)
    - [PG Algorithm](#section-id-421)
    - [A DDPG Sample](#section-id-429)
  




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

Here are some examples of RL methods devided by two major fields, Model-Based or Model-Free:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/rl-taxonomy.png"/>
</p>

The main taxonomy of RLs, is based on *What an RL Agent May Learn*. There are three major fields:

An RL solution can be classified based on its nature:

<div id='section-id-112'/>

### Value_Based_RL

A *Value-Based RL* is to find the rewards: There is a function predicting the reward of quantized states/states-and-actions (-> value-function/Q-function)

In this methods, an NN is trained as an approximation of a *value function*.

The question that the agent is to learn its anwer is: How much a state/action is likely to reward me in the future? The function's mathematical expression is declared [here](#Value_Function)


<div id='section-id-121'/>

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


<div id='section-id-140'/>

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

<div id='section-id-168'/>

## Traditional RL vs. DRL

The traditional RL has been introduced since 1950s. But it has not been used in real-world problems until 2000s. The major issue was:

*The traditional RL methods where based on descritized space of states and action. They where not useful in real problems with continuous nature or numerous descritized states and actions.*

In 2000s, Deep Reinforcement Learning (DRL) was introduced; And in real-world problems, it worked really good! 

Here is the main difference between the traditional RL and DRL:

- In the traditional RL, an optimal policy is obtained based on *bellman functions*
- In DRL, the function to find the optimal policy is estimated with a deep neural network

With help of deep neural networks (DNNs), the (numerous-descritized- or) continuous-state/action-space problems can be solved. Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/drl.png"/>
</p>

There are three main paradigms in DRL:

* **Critic-Only:** Estimating a value function using a DNN [just as described](#Value-Based-RL). The output of such networks is as wide as the whole action space (for the input state), claiming a value for each action (for the input state).

* **Actor-Only:** Direct estimation of policy function using a DNN [just as described](#Policy_Based_RL). The output of sunch network is a single action

* **Actor-Critic**
This method is a compound of value-based approaches (*Q-Learning*) and Policy-Based approaches (*Policy Gradients*):

- The *Actor* is the policy; It determines the action
- The *Critic* is the Q-function; It evaluates the action

<div id='section-id-199'/>

# RL Mathematics

<div id='section-id-201'/>

## Value-Based RL Mathematics: Bellman Functions

As follows, there are two utilities to get to an optimal policy: *Value-Function* and *Q-Function*. Notice the term $\pi$ in the two following equations. It means the calculation is related to a specific policy.

<div id='section-id-205'/>

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

<div id='section-id-227'/>

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


<div id='section-id-244'/>

## Policy-Based RL Mathematics

In this method, a *Policy Gradient* is defined with a mathematical gradient ascend approach, to maximize a reward based on a policy. 

In other words, for each member policy $\pi_\theta$ of a parameterized policy class:

$$
\Pi = {\pi_\theta, \theta \in R^m}
$$

There is a reward function:

$$
r(\tau)
$$

The main objective is to maximize the reward (= the expected value of possible rewards - the cost function):

$$
J(\theta) = E_\pi[r(\tau)]
$$

So, the parameters must be updated based on a *Gradient Ascend* approach:

$$
\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)
$$

<div id='section-id-272'/>

### More Detailed Math

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_1.png"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_2.png"/>
</p>

<div id='section-id-281'/>

### Policy-Based RL: A Model-Free RL

Breaking the above equations, is done in the the followings:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/j_brake_1.png"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/j_brake_2.png"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/j_brake_3.png"/>
</p>

The last expression shows no effect of environment in J function. Thus, the optimization based on this method is totally independent of the enviroment.


<div id='section-id-298'/>

# Review of Major DRL Methods

<div id='section-id-300'/>

## Deep Q-Learning (DQN): A Value-Based RL Method

*Q-Learning* is the most important sample of Value-Based reinforcement learning. First the traditional approach is presented. Then the solution of the same problem in DNN is reviewed.

<div id='section-id-304'/>

### Traditional Q-Learning Algorithm

The traditional Q-Learning algorithm is almost a dynamic programming approach. A *Q* function gives *Expected Total Reward* (value) of a each possible action *a* taken in an input state *s*:

$$
Q(s,a) = E[R_t]
$$

So a policy to take the actions which maximize the Q function, is required:

$$
\pi^* = argmax_a Q(s,a)
$$

The pseudo code for the traditional Q-learning algorithm is depicted in the following. 

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/traditional_Q_learning.png"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=497s)

<div id='section-id-326'/>

### Raw DQN Algorithm

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/q-learning.png"/>
</p>

As shown in graph, a *Q* function is approximated with a DNN to retuen a value for each possible action *a* taken in an input state *s*. In the following figure, note that the function Q(s,a) is approximated with a neural network. Also, notice the loss function and back propagation step in the last two lines.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/dql.png"/>
</p>

The **optimization approach** is based on Gradient Descent

The **loss function** is based on MSE:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/q_learning_loss_function.png"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=875s).



<div id='section-id-350'/>

### Revised (Practical) DQN

According to the the last section, two major issues exist in the problem. Next sections presents solutions and additional refinements to make a DQN converge.

<div id='section-id-354'/>

#### Issue-1 In Loss Function: Q

Calculating the loss function, a feedback is obtained to change the Q function, with Q function changing itself!

**SOLUTION: Mirror Network**
There are to calls to Q() function (network) in the above algorithm. A copy of the network is saved and its weight are freezed in the beginning of a specific period of iterations (e.g. 100 iterations). The call to $Q_\theta(s,a)$ (2nd call in the algorithm) is made for the network which its weights are being updated online. But the call to $Q_k(s',a')$ is made for the freezed network. After each period, the freezed network is updated with the online network.


<div id='section-id-362'/>

#### Issue-2 In Loss Function: target

The main assumption in solving the bellman equation, is that the network inputs are independent of each other (IID). Calculating the loss function, a feedback is obtained while the inputs to the network are not IID.

**SOLUTION: Experience Replay**
From an initial state, all the transitions ("action->state"s) for a noon-trained network are generated and buffered in a memory called an *Experience Replay*. A mini-batch of these data is used in training. This waye the inputs are IID because they are generated with a non-trained network. The experience replay is actually as same as the *dataset* which is known in neural network training.


<div id='section-id-370'/>

#### $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1 / $\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 


<div id='section-id-382'/>

#### Revised (Final) DQN Algorithm

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/revised_dqn.png"/>
</p>


<div id='section-id-389'/>

#### Loss function

The actual loss function considered in a DQN is *Huber Loss*. The green curve in the following figure:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/dqn_loss_func.png"/>
</p>


<div id='section-id-398'/>

### DQN Architecture

A DQ-Network (DQN) is created of to convolutional and two dense layers, as follows:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/dqn_architecture.png"/>
</p>


<div id='section-id-407'/>

### Limitations of DQN

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.


<div id='section-id-414'/>

## A Policy-Based RL Method

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_RL.png"/>
</p>


<div id='section-id-421'/>

### PG Algorithm

A policy gradient (PG) algorithm may be implemented like the follwoing:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_rl_alg.png"/>
</p>

<div id='section-id-429'/>

### A DDPG Sample

A sample of Deep Deterministic Policy Gradient (DDPG) is the follwoing network. This network gets the history of stock and gives descision for investing on 12 different items. The action space is basically continuous and value-based approaches won't work in such a case.

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/stock.png"/>
</p>

