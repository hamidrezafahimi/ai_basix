### Summary 

- [Deep Q-Learning (DQN): A Value-Based RL Method](#section-id-2)
  - [Traditional Q-Learning Algorithm](#section-id-6)
  - [Raw DQN Algorithm](#section-id-29)
  - [Revised (Practical) DQN](#section-id-60)
    - [Modification1:Double_DQN](#section-id-64)
    - [Modification2:Replay_Buffer](#section-id-81)
    - [Modification 3: $psilongreedy policy for taking actions](#section-id-89)
    - [Final Revised DQN Algorithm](#section-id-102)
    - [Loss function](#section-id-118)
    - [Consideration of Movement in Environment](#section-id-127)
    - [Implementation Hints](#section-id-132)
  - [Limitations of DQN](#section-id-147)
  - [Different Types of DQN](#section-id-154)
    - [Simple DQN](#section-id-158)
    - [DQN with Experience Replay](#section-id-162)
    - [Double DQN](#section-id-168)
    - [Dueling DQN](#section-id-172)
    - [Dueling Double DQN](#section-id-175)
    - [Noisy DQN](#section-id-178)
  



<div id='section-id-2'/>

# Deep Q-Learning (DQN): A Value-Based RL Method

*Q-Learning* is the most important sample of Value-Based reinforcement learning. Just like all the value-based RL methods, a mapping of *state/actions* to *rewards* is approximated. First the traditional approach is presented. Then the solution of the same problem in DNN is reviewed.

<div id='section-id-6'/>

## Traditional Q-Learning Algorithm

The traditional Q-Learning algorithm is almost a dynamic programming approach. 
In this method, a table with inputs: "action - space" and the output given by a Q-function is generated. The *Q* function gives *Expected Total Reward* (value) of each possible action *a* taken in an input state *s*:

$$
Q(s,a) = E[R_t]
$$

So a policy to take the actions which maximize the Q function, is required:

$$
\pi^* = argmax_a Q(s,a)
$$

The pseudo code for the traditional Q-learning algorithm is depicted in the following. 

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/traditional_Q_learning.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=497s)

<div id='section-id-29'/>

## Raw DQN Algorithm

As mentioned before, the traditional Q-Learning algorithm works by keeping (real-time generation of) a table. This method only works for environments with discrete and limited number of states and actions. To work with continuous state spaces or huge number of discrete states, the only practical way to solve the problem is to use a deep neural network as the approximation of the Q-function, returning a set of discrete values each corresponding to a specific action.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/q-learning.png?raw=true", width="600"/>
</p>

During the training, the DQN agent uses the difference with maximal action and current action (-?-) as the loss function to update the network parameters in real-time.
In the following figure, note that the function Q(s,a) is approximated with a neural network. Also, notice the loss function and back propagation step in the last two lines.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/dql.png?raw=true", width="600"/>
</p>

The **optimization approach** is based on Gradient Descent:

$$
\theta_{k+1} = \theta_k + \alpha \nabla_\theta L
$$

The **loss function** is based on MSE:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/q_learning_loss_function.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=875s).



<div id='section-id-60'/>

## Revised (Practical) DQN

According to the the last section, two major issues exist in the problem. Next sections presents solutions and additional refinements to make a DQN converge.

<div id='section-id-64'/>

### Modification1:Double_DQN

**Issue-1 In Loss Function (Q):** Calculating the loss function, a feedback is obtained to change the Q function, with Q function changing itself!

**SOLUTION: Mirror Network**

There are two calls to Q() function (network) in the above algorithm. A copy of the network is saved and its weight are freezed in the beginning of a specific period of iterations (e.g. 100 iterations). After each period, the freezed network is updated with the online network.

- *The Evaluation Network* (being updated online): This network is used to evaluate the current state and see which action to take. The call to $Q_\theta(s,a)$ (2nd call in the algorithm) is made for this network. 
- *The Target Network* (being updated periodically): This network is used to calculate the value of maximal actions during the learning step. The call to $Q_k(s',a')$ is made for the freezed network. The weights of this network is updated periodically with the evaluation network so that the estimates of the maximal actions get more accurate. 

The following visualizes the difference between the two networks:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/dqn_target_eval.png?raw=true", width="600"/>
</p>

<div id='section-id-81'/>

### Modification2:Replay_Buffer

**Issue-2 In Loss Function (target):** The main assumption in solving the bellman equation, is that the network inputs are independent of each other (IID). Calculating the loss function, a feedback is obtained while the inputs to the network are not IID.

**SOLUTION: Experience Replay**
From an initial state, all the transitions ("action->state"s) for a non-trained network are generated and buffered in a memory called an *Experience Replay*. A mini-batch of these data is used in training. This waye the inputs are IID because they are generated with a non-trained network. The experience replay is actually as same as the *dataset* which is known in neural network training.


<div id='section-id-89'/>

### Modification 3: $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1 / $\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 

*NOTE:* You probably don't want to let the $\epsilon$ become zero. Because it's important to always test the agent's model of the environment.

<div id='section-id-102'/>

### Final Revised DQN Algorithm

There is a good explanation of this algorithm [here](https://www.youtube.com/watch?v=4GhH3d9NsIc&t=348s).

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/revised_dqn.png?raw=true", width="600"/>
</p>

The following flowchart shows a brief review of how a DQN (or DDQN) is trained. 

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/DQN_algorithm.png?raw=true", width="600"/>
</p>

Remember that the *target* is the direction in which we want the weights of the DNN change.

<div id='section-id-118'/>

### Loss function

The actual loss function considered in a DQN is *Huber Loss*. The green curve in the following figure:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/dqn_loss_func.png?raw=true", width="600"/>
</p>


<div id='section-id-127'/>

### Consideration of Movement in Environment

To understand the effect of changes in the environment, consider an environment in which the states contain changes through time. If the states in such environment are defined in particular data packs (e.g. images),to train the agent how to choose actions when there are changes in the environment, the training is done on a batch of data packs (images) instead of a single pack (image).


<div id='section-id-132'/>

### Implementation Hints

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/dqn_implementation_hints.png?raw=true", width="600"/>
</p>

<!-- ## DQN Architecture

A DQ-Network (DQN) is created of two convolutional and two dense layers, as follows:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/dqn_architecture.png?raw=true", width="600"/>
</p> -->


<div id='section-id-147'/>

## Limitations of DQN

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.


<div id='section-id-154'/>

## Different Types of DQN

There are different supplements each of which, if added to a DQN, making a new type of solution with custom properties. Thus, a DQN solution may be one of the followings. Note that the [previously shown flowchart](https://github.com/hamidrezafahimi/ai_basix/blob/master/figs/DQN_algorithm.png) depicts enough info to understand the algorithm for the following fisrt 3-items.

<div id='section-id-158'/>

### Simple DQN

It is only the main idea of Q-Learning implemented such that the Q-function is approximated with a deep neural network. [Here]() is the raw algorithm for DQN.

<div id='section-id-162'/>

### DQN with Experience Replay 

[code sample](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dqn_agent.py)

The most simple implementations of DQN consider the first modification which was reviewed [above](#Modification2:Replay_Buffer). 

<div id='section-id-168'/>

### Double DQN

In a double DQN, the modification 2 (reviewed [above](#Modification1:Double_DQN)) is taken into the consideration.

<div id='section-id-172'/>

### Dueling DQN


<div id='section-id-175'/>

### Dueling Double DQN


<div id='section-id-178'/>

### Noisy DQN
