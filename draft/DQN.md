
# Deep Q-Learning (DQN): A Value-Based RL Method

*Q-Learning* is one of the most important samples of Value-Based reinforcement learning. Just like all the value-based RL methods, a mapping of *state/actions* to *rewards* is approximated. The [Tradiotional Q-Learning Algorithm](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/RL/Q-learning.md) has been reviewd separately. Here, the solution of the same problem in DNN is reviewed.

## Raw DQN Algorithm

As mentioned before, the traditional Q-Learning algorithm works by keeping (real-time generation of) a table. This method only works for environments with discrete and limited number of states and actions. To work with continuous state spaces or huge number of discrete states, the only practical way to solve the problem is to use a deep neural network as the approximation of the Q-function, returning a set of discrete values each corresponding to a specific action.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/q-learning.png?raw=true", width="600"/>
</p>

During the training, the DQN agent uses the difference with maximal action and current action (-?-) as the loss function to update the network parameters in real-time.
In the following figure, note that the function Q(s,a) is approximated with a neural network. Also, notice the loss function and back propagation step in the last two lines.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dql.png?raw=true", width="600"/>
</p>

The **optimization approach** is based on Gradient Descent:

$$
\theta_{k+1} = \theta_k + \alpha \nabla_\theta L
$$

The **loss function** is based on MSE:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/q_learning_loss_function.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=875s).



## Revised (Practical) DQN

According to the the last section, two major issues exist in the problem. Next sections presents solutions and additional refinements to make a DQN converge.

### Modification1:Double_DQN

**Issue-1 In Loss Function (Q):** Calculating the loss function, a feedback is obtained to change the Q function, with Q function changing itself!

**SOLUTION: Mirror Network**

There are two calls to Q() function (network) in the above algorithm. A copy of the network is saved and its weight are freezed in the beginning of a specific period of iterations (e.g. 100 iterations). After each period, the freezed network is updated with the online network.

- *The Evaluation Network* (being updated online): This network is used to evaluate the current state and see which action to take. The call to $Q_\theta(s,a)$ (2nd call in the algorithm) is made for this network. 
- *The Target Network* (being updated periodically): This network is used to calculate the value of maximal actions during the learning step. The call to $Q_k(s',a')$ is made for the freezed network. The weights of this network is updated periodically with the evaluation network so that the estimates of the maximal actions get more accurate. 

The following visualizes the difference between the two networks:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dqn_target_eval.png?raw=true", width="600"/>
</p>

### Modification2:Replay_Buffer

**Issue-2 In Loss Function (target):** The main assumption in solving the bellman equation, is that the network inputs are independent of each other (IID). Calculating the loss function, a feedback is obtained while the inputs to the network are not IID.

**SOLUTION: Experience Replay**
From an initial state, all the transitions ("action->state"s) for a non-trained network are generated and buffered in a memory called an *Experience Replay*. A mini-batch of these data is used in training. This waye the inputs are IID because they are generated with a non-trained network. The experience replay is actually as same as the *dataset* which is known in neural network training.


### Modification 3: $\epsilon$-greedy policy for taking actions

- **In the beginning of training**, no info exists about environment. So *all possible actions are taken* (-> exploration)

- **In the middle of training**, 
  * The best action is taken with probability 1 / $\epsilon$
  * A random action is taken with probability $\epsilon$
  * Reduce $\epsilon$ to zero as training goes on

- **The more closed to the end of the training**, *the better actions are chosen* (-> exploitation) 

*NOTE:* You probably don't want to let the $\epsilon$ become zero. Because it's important to always test the agent's model of the environment.

### Final Revised DQN Algorithm

There is a good explanation of this algorithm [here](https://www.youtube.com/watch?v=4GhH3d9NsIc&t=348s).

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/revised_dqn.png?raw=true", width="600"/>
</p>

The following flowchart shows a brief review of how a DQN (or DDQN, or Dueling DDQN) is trained. 

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/DQN_algorithm.png?raw=true", width="600"/>
</p>

With consideration of the following table, the above flowchart can be generalized to DDQN and Dueling DDQN. 

|                |                DQN                |                            DDQN                            |                        Dueling DDQN                        |
|----------------|:---------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
| Two Networks   | Q_target and Q_eval  are the same | Q_eval is Assigned to Q_target in a  Specific Set of Steps | Q_eval is Assigned to Q_target in a  Specific Set of Steps |
| Network Output | (Q) = (V) = (A)                   | (Q) = (V) = (A)                                            | (Q) = (V)+(A)                                              |

Remember that the *target* is the direction in which we want the weights of the DNN change.

### Loss function

The actual loss function considered in a DQN is *Huber Loss*. The green curve in the following figure:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dqn_loss_func.png?raw=true", width="600"/>
</p>


### Consideration of Movement in Environment

To understand the effect of changes in the environment, consider an environment in which the states contain changes through time. If the states in such environment are defined in particular data packs (e.g. images),to train the agent how to choose actions when there are changes in the environment, the training is done on a batch of data packs (images) instead of a single pack (image).


### Implementation Hints

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dqn_implementation_hints.png?raw=true", width="600"/>
</p>

<!-- ## DQN Architecture

A DQ-Network (DQN) is created of two convolutional and two dense layers, as follows:

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dqn_architecture.png?raw=true", width="600"/>
</p> -->


## Limitations of DQN

DQN works in problems with descritized action space. If the actions are to be taken from a continuous range of cases, this is not gonna work!

DQN doesn't learn stochastic policies, because the outputs are provided deterministically from the Q-function, i.e., the outputs are the value for the actions, not the probability for the optimality of the actions in terms of reward. This is obtained using a *Policy-Based RL*.


## Different Types of DQN

There are different supplements each of which, if added to a DQN, making a new type of solution with custom properties. Thus, a DQN solution may be one of the followings. Note that the [previously shown flowchart](https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/DQN_algorithm.png) depicts enough info to understand the algorithm for the following fisrt 3-items.

### Simple DQN

It is only the main idea of Q-Learning implemented such that the Q-function is approximated with a deep neural network. [Here]() is the raw algorithm for DQN.

### DQN with Experience Replay 

[code sample](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dqn_agent.py)

The most simple implementations of DQN consider the first modification which was reviewed [above](#Modification2:Replay_Buffer). 

### Double DQN

[code sample](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_ddqn_agent.py)

In a double DQN, the modification 2 (reviewed [above](#Modification1:Double_DQN)) is taken into the consideration.

### Dueling DQN

[code sample](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dueling_ddqn_agent.py)
[reference for text](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df)

The Q-values that we have been discussing so far correspond to how good it is to take a certain action given a certain state. This can be written as Q(s,a). This action given state can actually be decomposed into two more fundamental notions of value. The first is the value function V(s), which says simple how good it is to be in any given state. The second is the advantage function A(a), which tells how much better taking a certain action would be compared to the others. We can then think of Q as being the combination of V and A. More formally:

$$
Q(s,a) = V(s) + A(a)
$$

Why we use such an architecture: imagine sitting outside in a park watching the sunset. It is beautiful, and highly rewarding to be sitting there. No action needs to be taken, and it doesn’t really make sense to think of the value of sitting there as being conditioned on anything beyond the environmental state you are in. We can achieve more robust estimates of state value by decoupling it from the necessity of being attached to specific actions.

In the following figure, the above demonstrates a regular DQN with a single stream for Q-values, and below, deonstrates a Dueling DQN where the value and advantage are calculated separately and then combined only at the final layer into a Q value.

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/dueling_dqn_architecture.png?raw=true", width="600"/>
</p>


### Noisy DQN

...
