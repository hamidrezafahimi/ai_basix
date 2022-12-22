

## Policy-Based RL Methods

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/policy_based_RL.png?raw=true", width="600"/>
</p>


## Algorithm

A conceptual representation of policy gradient (PG) algorithm would be:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/policy_based_rl_alg.png?raw=true", width="200"/>
</p>

Practically, the algorithm is implemented like the following:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/policy_based_rl_alg_2.png?raw=true", width="600"/>
</p>


## Terminology

### Variables

- *A* or *a* - Action
- *t* - Time step (index)
- *S* or *s* - State
- $\theta$ - A set parameters subjected to a specific policy
- $\Pi$ - A set of probable policies
- $\pi$ - A single policy
- *J* - The cost function to be maximized during the training


### Parameters

- *G* or *R* - Discounted sum of rewards that follow each time-step. It tells the agent how much reward is received after each time-step.
- $\alpha$ - Learning rate - The *Gradient Descent (Ascent!)* parameter - Determines how much does the parameters change in each update.
- $\gamma$ - Discount factor - Determines how much the agent cares about more future rewards. 


## Mathematics

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


### More Detailed Math

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/policy_1.png?raw=true", width="600"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/policy_2.png?raw=true", width="600"/>
</p>

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/reinfoece_math_1.png?raw=true", width="400"/>
</p>

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/reinfoece_math_2.png?raw=true", width="400"/>
</p>

So here is the gradient ascent parameter-update method:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/reinfoece_math_3.png?raw=true", width="300"/>
</p>


### Policy-Based RL: A Model-Free RL

Breaking the above equations, is done in the the followings:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/j_brake_1.png?raw=true", width="600"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/j_brake_2.png?raw=true", width="600"/>
</p>
<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/j_brake_3.png?raw=true", width="600"/>
</p>

The last expression shows no effect of environment in J function. Thus, the optimization based on this method is totally independent of the enviroment.





## Refinements

[reference](https://www.youtube.com/watch?v=A_2U6Sx67sE&list=PL-9x0_FO_lgkwi8ES611NsV-cjYaH_nLa&index=1)

There are two major problems in this method:

- After one episode, the parameters must be reinitialized and the training must be done from scratch. How to handle the variations between episodes so that the performance of the agent is impoved?

-> *SOLUTION:* The training is done on a *batch* of episodes. The update is done then, considering the stochastic features of the results gained on different episodes in a batch of episodes.

- ... (I just didn't quite understand the second one!)