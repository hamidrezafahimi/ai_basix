import os
# from io import StringIO
import numpy as np
# import gym
# from gym.envs.toy_text.frozen_lake import generate_random_map
import sys
sys.path.insert(1, '../../../projects/env/gym_frozenLake_GUI')
sys.path.insert(2, '../../../modules')
sys.path.insert(3, '../../../samples/DRL/DQN')
import envGUICreation as egc
import gen
from tf2_dueling_ddqn_agent import Agent
import drl
import time


logMode = False

def log(index, done, action, nState, eps):
    
    global logDir
    if logDir is None:
        return
    with open(logDir + "log.txt".format(index), 'a+') as f:
        f.write("\n\n\n\nstep_{}:\n\n".format(index))
        # f.write(np.array2string(qtable))
        f.write("\n\ndone: {}".format(done))
        f.write("\naction: {}".format(action))
        f.write("\nnew state: {}".format(nState))
        f.write("\nepsilon: {}".format(eps))


def logAction(action, print=True):

    if action == 0:
        log = "left"
    elif action == 1:
        log = "down"
    elif action == 2:
        log = "right"
    elif action == 3:
        log = "up"
    
    if print:
        print("action:  " + log)
    return log


logDirFromRoot = "/data/logs/dqn_frozenLake/"
gen.resetDir(gen.getRootDir()+logDirFromRoot)

logDir = "../../.."+logDirFromRoot
# logDir = None

env = egc.GymGraphicalFrozenLake(envSize=(4,4), delay=0.001, show=False, logMode=logMode)



modelFolderName = "dueling_ddqn"

agent = Agent(lr=0.005, gamma=0.99, n_actions=4, epsilon=1.0, epsilon_dec=1e-2,
                  batch_size=16, input_dims=[2])

# drl.temporalDifferenceTrain(envName='LunarLander-v2', agent=agent, n_games=100, 
#                             modelFolderName="dueling_ddqn")

saveDir = gen.makeModelSaveDir(modelFolderName)
logDir = gen.makeLogSaveDir(modelFolderName + "_log")

ddqn_scores = []
eps_history = []
total_episodes = 200            # Total episodes
max_steps = 50

for i in range(total_episodes):
    t0 = time.time()
    done = False
    score = 0
    obs_1d, observation = env.reset()
    
    logDir = "../../../data/logs/q_learning_frozenLake/episode_{}/".format(i)
    if not os.path.exists(logDir):
        os.mkdir(logDir)

    step = 0
    # while not done:
    for step in range(max_steps):

        # print("state: ", observation)
        if logMode:
            # actionStr = logAction(action, print=False)
            # log(step+1, done, actionStr, observation_, agent.epsilon)
            env.logConsole()

        action = agent.choose_action(observation)
        obs_1d_, reward, done, info, observation_ = env.step(action, obs_1d)
        env.saveImages(logDir)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        obs_1d = obs_1d_
        agent.learn()
        if done == True: 
            break
        # step += 1
    
    eps_history.append(agent.epsilon)
    ddqn_scores.append(score)

    avg_score = np.mean(ddqn_scores[-100:])
    print('episode: ', i,'score: %.2f' % score,
            ' average score %.2f' % avg_score, ' time: %.2f' % (time.time() - t0))

agent.save_model(saveDir)
filename = logDir + "lunarlander-dueling_ddqn.png"

# x = [i+1 for i in range(n_games)]
# plotLearning(x, ddqn_scores, eps_history, filename)



















