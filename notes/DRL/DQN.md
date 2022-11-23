### Summary 

- [Deep Q-Learning (DQN): A Value-Based RL Method](#section-id-2)
  - [Traditional Q-Learning Algorithm](#section-id-6)
  - [Raw DQN Algorithm](#section-id-28)
  - [Revised (Practical) DQN](#section-id-52)
    - [Issue-1 In Loss Function: Q](#section-id-56)
    - [Issue-2 In Loss Function: target](#section-id-64)
    - [$psilongreedy policy for taking actions](#section-id-72)
    - [Revised (Final) DQN Algorithm](#section-id-84)
    - [Loss function](#section-id-91)
  - [DQN Architecture](#section-id-100)
  - [Limitations of DQN](#section-id-109)
  



<div id='section-id-2'/>

# Deep Q-Learning (DQN): A Value-Based RL Method

*Q-Learning* is the most important sample of Value-Based reinforcement learning. First the traditional approach is presented. Then the solution of the same problem in DNN is reviewed.

<div id='section-id-6'/>

## Traditional Q-Learning Algorithm

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
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/traditional_Q_learning.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=497s)

<div id='section-id-28'/>

## Raw DQN Algorithm

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/q-learning.png?raw=true", width="600"/>
</p>

As shown in graph, a *Q* function is approximated with a DNN to retuen a value for each possible action *a* taken in an input state *s*. In the following figure, note that the function Q(s,a) is approximated with a neural network. Also, notice the loss function and back propagation step in the last two lines.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/dql.png?raw=true", width="600"/>
</p>

The **optimization approach** is based on Gradient Descent

The **loss function** is based on MSE:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/q_learning_loss_function.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=875s).



<div id='section-id-52'/>

## Revised (Practical) DQN

According to the the last section, two major issues exist in the problem. Next sections presents solutions and additional refinements to make a DQN converge.

<div id='section-id-56'/>

### Issue-1 In Loss Function: Q

Calculating the loss function, a feedback is obtained to change the Q function, with Q function changing itself!

**SOLUTION: Mirror Network**
There are to calls to Q() function (network) in the above algorithm. A copy of the network is saved and its weight are freezed in the beginning of a specific period of iterations (e.g. 100 iterations). The call to $Q_\theta(s,a)$ (2nd call in the algorithm) is made for the network which its weights are being updated online. But the call to $Q_k(s',a')$ is made for the freezed network. After each period, the freezed network is updated with the online network.


<div id='section-id-64'/>

### Issue-2 In Loss Function: target

The main assumption in solving the bellman equation, is that the network inputs are independent of each other (IID). Calculating the loss function, a feedback is obtained while the inputs to the network are not IID.

**SOLUTION: Experience Replay**
From an initial state, all the transitions ("action->state"s) for a noon-trained network are generated and buffered in a memory called an *Experience Replay*. A mini-batch of these data is used in training. This waye the inputs are IID because they are generated with a non-trained network. The experience replay is actually as same as the *dataset* which is known in neural network training.


<div id='section-id-72'/>

### $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1 / $\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 


<div id='section-id-84'/>

### Revised (Final) DQN Algorithm

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/revised_dqn.png?raw=true", width="600"/>
</p>


<div id='section-id-91'/>

### Loss function

The actual loss function considered in a DQN is *Huber Loss*. The green curve in the following figure:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/dqn_loss_func.png?raw=true", width="600"/>
</p>


<div id='section-id-100'/>

## DQN Architecture

A DQ-Network (DQN) is created of to convolutional and two dense layers, as follows:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/dqn_architecture.png?raw=true", width="600"/>
</p>


<div id='section-id-109'/>

## Limitations of DQN

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.

