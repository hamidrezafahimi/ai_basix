
import numpy as np
import gym

env = gym.make("FrozenLake-v0")

episodes = 5
for episode in range(1, episodes+1):

    state = env.reset()

    done = False
    score = 0 
    
    for t in range(200):
    
        env.render()

        action = env.action_space.sample()

        n_state, reward, done, info = env.step(action)

        score+=reward

        if done:
            break

    print('Episode:{} Score:{}'.format(episode, score))

env.close()