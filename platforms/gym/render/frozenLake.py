
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v0', desc=generate_random_map(size=8))


# Running a single episode:
print(env.observation_space)
state = env.reset()
done = False
score = 0 

while not done:

    env.render()
    action = env.action_space.sample()

    n_state, reward, done, info = env.step(action)
    print(n_state)
    score+=reward

    if done:
        break

print('Score:{}'.format(score))

env.close()