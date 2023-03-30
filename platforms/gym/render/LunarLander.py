
import gym
# from gym.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('LunarLander-v2')


# Running a single episode:
print(env.observation_space)
state = env.reset()
done = False
score = 0 

for k in range(100):
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