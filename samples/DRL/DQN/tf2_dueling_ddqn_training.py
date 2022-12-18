import gym
import numpy as np
from tf2_dueling_ddqn_agent import Agent
import sys
sys.path.insert(1, '../../../modules')
# from dqn_utils import plotLearning
# from utils import makeModelSaveDir
import drl
import gen
import time


agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=[8])

drl.temporalDifferenceTrain(envName='LunarLander-v2', agent=agent, n_games=100, 
                            modelFolderName="dueling_ddqn")

# if __name__ == '__main__':
#     env = gym.make('LunarLander-v2')
#     agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
#                   batch_size=64, input_dims=[8])
#     n_games = 20
#     ddqn_scores = []
#     eps_history = []

#     for i in range(n_games):
#         t0 = time.time()
#         done = False
#         score = 0
#         observation = env.reset()
#         while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             score += reward
#             agent.store_transition(observation, action, reward, observation_, done)
#             observation = observation_
#             agent.learn()
#         eps_history.append(agent.epsilon)

#         ddqn_scores.append(score)

#         avg_score = np.mean(ddqn_scores[-100:])
#         print('episode: ', i,'score: %.2f' % score,
#               ' average score %.2f' % avg_score, ' time: %.2f' % (time.time() - t0))

#     agent.save_model(saveDir)
#     filename = 'lunarlander-dueling_ddqn.png'

#     x = [i+1 for i in range(n_games)]
#     plotLearning(x, ddqn_scores, eps_history, filename)