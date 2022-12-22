### Summary 

  



This is a theoretical review of Q-Learning as a reinforcement learning method. To breifly check out the basics, take a look at the [higher-level theory not]().

*Q-Learning* is one of the most important samples of Value-Based reinforcement learning.
In this method, a table with inputs: "action - space" and the output given by a Q-function is generated. The *Q* function gives *Expected Total Future Reward* (value) of each possible action *a* taken in an input state *s*. So a policy to take the actions which maximize the Q function, is required:

$$
\pi^* = argmax_a Q(s,a)
$$ -->

Reviewing the main structure of Reinforcement Learning,

<p align="center">
  <!-- <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/traditional_Q_learning.png?raw=true", width="600"/> -->
</p>

And also taking a brief look at the structure of value-based RL methods,

<p align="center">
  <!-- <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/traditional_Q_learning.png?raw=true", width="600"/> -->
</p>

The pseudo code for the traditional Q-learning algorithm is depicted in the following:

<p align="left">
  <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/traditional_Q_learning.png?raw=true", width="600"/>
</p>

For more details refer [here](https://www.youtube.com/watch?v=D3b50jrKzcc&t=497s)

After all, the following conceptual graph shows exactly what is going on in a Q-Learning procedure. 

<p align="center">
  <!-- <img src="https://github.com/hamidrezafahimi/ai_basix/blob/master/data/figs/traditional_Q_learning.png?raw=true", width="600"/> -->
</p>
