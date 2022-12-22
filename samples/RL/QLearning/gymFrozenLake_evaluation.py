import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import sys
sys.path.insert(1, '../../../projects/env/gym_frozenLake_GUI')
import envGUICreation

# This is a Q-Learning sample, based on the 'FrozenLake' environment of OpenAI Gym

# To access each "LINK", read the "README.md" in the current folder

# The theroretical background for the algorithm is given in LINK-1
# LINk-2 gives a good demonstration of the algorithm
# Rference for the code: LINK-3

env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            break
        state = new_state
env.close()