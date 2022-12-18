
This can be both a starting point for the development of artificial neural network algorithms, and a handbook containing introductory to advanced tools and solutions in this field. I have recently started documenting everything I have learned (and am learning) in the area. Soon I'll develope it much more. 

## Overall Review of Context

Explanations about the context of each directory:

First of all, my theoretic summaries are placed in the `notes/`. There are different text files each subjected to a specific topic with classification of the context in a table of contents at the beginning of each.

Elementary coding instructions based on different platforms are are placed in the `samples/` directory.If you are not much familiar to the basic concepts, in `samples/`, there are a bunch of code examples related to different known ANN developement platforms. So the scripts are classified based on the different platforms. For instance, I suggest the `samples/tensorflow/keras/` for a basic introduction to the artificial neural network design and implementation.

After all, the main helpful sample projects exist in `projects/`. Despite the `samples/` directory, the scripts are classified based on the main topic of each. Withing the sample codes, baisc instruction subjected to the specific topic are given in comments. 

The references for this context are *a lot*. I've tried to give a link to the reference in each code or note.


## Learning Roadmap

The following is the best order for one to study the stuff. Actually this a table of contents ordered such that I personally have learned the stuff. I mean, this order is just a suggestion. Do whatever you like!
<!-- In cases that there are numbered scripts within folders, read them in order. Otherwise, read what you need. -->

1. **MLP**: [Theory](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/MLP) - [Samples](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/MLP)

2. **Custom Networks in Keras**: [Samples](https://github.com/hamidrezafahimi/ai_basix/blob/master/platforms/tensorflow/keras)

3. **Unsupervised Learning with SOM**: [Theory](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/SOM)

4. **DRL**: [Theory](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/DRL) - [Samples](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL)


### Learning Roadmap: DRL

For DRL documents, go to directory: `notes/DRL/`. The order of documents (`.md` files) is as follows:

1. [Fundamentals](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/DRL/DRL-Fundamentals.md)
2. [DQN](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/DRL/DQN.md)
3. [PG](https://github.com/hamidrezafahimi/ai_basix/blob/master/notes/DRL/Policy-Gradients.md)

After reviewing the notes, for DRL scripts, go to directory: `samples/DRL/`.The order of codes (`.py` files) is as follows. This is how I dicumented the codes and the code comments are continued in this order:

1. [PG Agent (tf2)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/reinforce/tf2_policy_gradient_agent.py) (---> 
[network](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/reinforce/tf2_policy_gradient_network.py) -
[training](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/reinforce/tf2_policy_gradient_training.py)) 
(You can also take a look at 
[the same sample in keras](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/reinforce/keras_policy_gradient_agent.py))

2. [DQN Agent (tf2)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dqn_agent.py) (---> [trainging](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dqn_training.py))

3. [DQN Agent with CNN (keras)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_ddqn_cnn_agent.py)  (Make sure to see the [trainging](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_ddqn_cnn_training.py) code)

4. [DQN Agent (keras)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_dqn_agent.py)  (---> [trainging](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_dqn_training.py) code)

5. [DDQN Agent (keras)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_ddqn_agent.py)  (---> [trainging](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/keras_ddqn_training.py) code)

6. [Dueling DDQN Agent (tf2)](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dueling_ddqn_agent.py)  (---> [trainging](https://github.com/hamidrezafahimi/ai_basix/blob/master/samples/DRL/DQN/tf2_dueling_ddqn_training.py) code)

<!-- ## Summary of Resources

There are two major concepts covered:

### ANN Theory

Summary manuscripts (in the `notes` folder) resulted by passing these courses:

- An ANN course I had in 2020 in Amirkabir University of Technology - Dr Safabakhsh

- This [youtube video course](https://www.youtube.com/playlist?list=PLQY2H8rRoyvxWE6bWx8XiMvyZFgg_25Q_)


### ANN Coding Samples - Tensorflow

A bunch of code samples exist which are my done homeworks during the ANN-Coding courses I have passed:

- Tensorflow-2 - Ahmad Asadi (Teaching Assistant Class for ANN course, safabakhsh, Amirkabir University of Technology) - [This Link](https://www.youtube.com/watch?v=1e6rEiTqPsk&list=PLQYKiL8d05Znurc2AwuXQC8mmD5-WeSPm)

### OpenAI gym

There are also samples of using the OpenAI gym environments for DRL. Sources:

- This tutorial: [youtube link](https://www.youtube.com/watch?v=Mut_u40Sqz4) - [github link](https://github.com/nicknochnack/ReinforcementLearningCourse) -->