### Summary 

  - [What is Reinforcement Learning](#section-id-4)
      - [Markov State:](#section-id-16)
    - [Deep Reinforcement Learing](#section-id-38)
  





<div id='section-id-4'/>

## What is Reinforcement Learning

The structure of reinforcement leaning paradigm is based on *reward* and *punishment*. In a reinforcement learning problem, always:

(There is an) **ACTION** --> (done by an *AGENT* on an *ENVIRONMENT*, changing a) **STATE (OBSERVATIONS)** --> (resulting a) **REWARD** (based on a certain criteria, which is given to the agent with a number of steps delay) 

The goal in a reinforcement learning problem, is to **select actions to maximize the future reward**.

<p align="center">
  <img src="../../rl_loop.png"/>
</p>

<div id='section-id-16'/>

#### Markov State:
Is a state in which all the useful information of the history lies. A state is a markov state if and only if:

$$
  p(S_{t+1}|S_t) = p(S_{t+1}|S_1, S_2, ..., S_t)
$$

I.e., future is independent of past, given present.

The concept of *total reward* in the mentioned structure is shown in the following:

<p align="center">
  <img src="../../rl_total_reward.png"/>
</p>

To avoid an infinite total reward:

$$
  R_t = \gamma^0r_t + \gamma^1r_{t+1} + \gamma^2r_{t+2} + ...
$$


<div id='section-id-38'/>

### Deep Reinforcement Learing

Two major benefits of deep learning, **Representing** and **Comprehending** data, can be added to the major benefit of reinforcement learning, which is the **Action Ability** on an obtained understanding.

<p align="center">
  <img src="../../drl.png"/>
</p>
