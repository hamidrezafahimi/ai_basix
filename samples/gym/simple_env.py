
import gym
import time

environment_name = "CartPole-v0"
env = gym.make(environment_name,  render_mode="rgb_array")

# There are two spaces: "the observation space" and "the action space"
# 
# This returns the type of observation space:
# env.observation_space
# This returns the type of action space:
# env.action_space


episodes = 5
for episode in range(1, episodes+1):

    # The following returns a set of observations from the environment:

    state = env.reset()

    # The `done` flag shows the end of an episode:

    done = False
    score = 0 
    
    for t in range(200):
    
        # The function `render` leads to a graphical representation of the environment:
    
        env.render()

        # The term `.action_space.sample()` returns a (random) action sample from the action space

        action = env.action_space.sample()

        # The function `step` gets the action as the input and returns the next data generated in the
        # environment. There are 4 outputs given by the function: `next states` - `resulted reward` 
        # - `is the episode done?` - `additional information` 

        n_state, reward, done, d, info = env.step(action)

        # The reward is to be accumulated through various "action/observation"s

        score+=reward

        if done:
            break

    print('Episode:{} Score:{}'.format(episode, score))

# The function `close` closes the python pop-up showing the environment - resulted previously by the
# function `render`

env.close()