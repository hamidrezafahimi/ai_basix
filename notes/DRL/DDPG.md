### Summary 

  - [Policy-Based RL Methods](#section-id-3)
    - [Policy Gradients (Reinforce)](#section-id-10)
    - [DDPG](#section-id-19)
  




<div id='section-id-3'/>

## Policy-Based RL Methods

<p align="center">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/policy_based_RL.png?raw=true", width="600"/>
</p>


<div id='section-id-10'/>

### Policy Gradients (Reinforce)

A policy gradient (PG) algorithm may be implemented like the follwoing:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/policy_based_rl_alg.png?raw=true", width="600"/>
</p>


<div id='section-id-19'/>

### DDPG

A sample of Deep Deterministic Policy Gradient (DDPG) is the follwoing network. This network gets the history of stock and gives descision for investing on 12 different items. The action space is basically continuous and value-based approaches won't work in such a case.

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ann_basix/blob/master/figs/stock.png?raw=true", width="600"/>
</p>

