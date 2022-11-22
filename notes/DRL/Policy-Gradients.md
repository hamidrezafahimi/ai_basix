### Summary 

  - [Policy-Based RL Methods](#section-id-3)
  - [Algorithm](#section-id-10)
  - [Terminology](#section-id-25)
    - [Variables](#section-id-27)
    - [Parameters](#section-id-38)
  - [Mathematics](#section-id-46)
  - [Refinements](#section-id-65)
  




<div id='section-id-3'/>

## Policy-Based RL Methods

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_RL.png"/>
</p>


<div id='section-id-10'/>

## Algorithm

A policy gradient (PG) algorithm may be implemented like the follwoing:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_rl_alg.png"/>
</p>

The algorithm can be stated in other words:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/policy_based_rl_alg_2.png"/>
</p>


<div id='section-id-25'/>

## Terminology

<div id='section-id-27'/>

### Variables

- *A* or *a* - Action
- *t* - Time step (index)
- *S* or *s* - State
- *$\theta$* - A set parameters subjected to a specific policy
- *$\Pi$* - A set of probable policies
- *$\pi$* - A single policy
- *J* - The const function to be maximized during the training


<div id='section-id-38'/>

### Parameters

- ** - Discounted sum of rewards that follow each time-step. It tells the agent how much reward is received after each time-step.
- *$\alpha$* - Learning rate - The *Gradient Descent (Ascent!)* parameter - Determines how much does the parameters change in each update.
- *$\gamma$* - Discount factor - Determines how much the agent cares about more future rewards. 
- ** - 


<div id='section-id-46'/>

## Mathematics

Check the basic mathematics of policy-based RL methods [here]().

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/reinfoece_math_1.png"/>
</p>

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/reinfoece_math_2.png"/>
</p>

So here is the gradient ascent parameter-update method:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/notes/figs/reinfoece_math_3.png"/>
</p>


<div id='section-id-65'/>

## Refinements

[reference](https://www.youtube.com/watch?v=A_2U6Sx67sE&list=PL-9x0_FO_lgkwi8ES611NsV-cjYaH_nLa&index=1)

There are two major problems in this method:

- After one episode, the parameters must be reinitialized and the training must be done from scratch. How to handle the variations between episodes so that the performance of the agent is impoved?

-> *SOLUTION:* The training is done on a *batch* of episodes. The update is done then, considering the stochastic features of the results gained on different episodes in a batch of episodes.

- ... (I just didn't quite understand the second one!)
