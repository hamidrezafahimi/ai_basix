import os
# from io import StringIO
import numpy as np
# import gym
# from gym.envs.toy_text.frozen_lake import generate_random_map
import sys
sys.path.insert(1, '../../../projects/env/gym_frozenLake_GUI')
sys.path.insert(2, '../../../modules')
import envGUICreation as egc
import gen
# import time



logMode = True

def log(index, done, action, nState, eps, dir):
    
    # global logDir
    # if logDir is None:
    #     return
    with open(dir + "log.txt".format(index), 'a+') as f:
        f.write("\n\n\n\nstep_{}:\n\n".format(index))
        f.write(np.array2string(qtable))
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


projectFolder = "q_learning_frozenLake_new"
logDirFromRoot = "/data/logs/" + projectFolder
gen.resetDir(gen.getRootDir()+logDirFromRoot)

logDir = "../../.."+logDirFromRoot
# logDir = None

env = egc.GymGraphicalFrozenLake(envSize=(4,4), delay=0.001, show=False, logMode=logMode)

# This is a Q-Learning sample, based on the 'FrozenLake' environment of OpenAI Gym

action_size = env.env.action_space.n
state_size = env.env.observation_space.n

qtable = np.zeros((state_size, action_size))

total_episodes = 200            # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 50                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# List of rewards
rewards = []

def epsilonGreedyPolicy(eps):

    global env
    # Choose an action a in the current world state (s)
    # First we randomize a number
    exp_exp_tradeoff = np.random.uniform(0, 1)

    # If this number > greater than epsilon --> exploitation (taking the biggest Q value for
    # this state)
    if exp_exp_tradeoff > eps:
        act = np.argmax(qtable[state,:])

    # Else doing a random choice --> exploration
    else:
        act = env.env.action_space.sample()

    return act


def updateEpsilon(episode):

    global min_epsilon, max_epsilon, decay_rate
    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)



for episode in range(total_episodes):
    # Reset the environment
    state, _ = env.reset()
    # step = 0
    done = False
    total_rewards = 0
    
    lgDir = logDir+"/episode_{}/".format(episode)
    if not os.path.exists(lgDir):
        os.mkdir(lgDir)
    
    # env.logDir = logDir

    for step in range(max_steps):
        
        action = epsilonGreedyPolicy(epsilon)
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info, _ = env.step(action, state)
        env.saveImages(lgDir)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        
        actionStr = logAction(action, print=False)
        log(step+1, done, actionStr, new_state, epsilon, dir=lgDir)

        # If done (if we're dead) : finish episode
        env.logConsole()
        if done == True: 
            break
        
    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = updateEpsilon(episode)
    rewards.append(total_rewards)

    print("episode: {} - score: {}".format(episode, total_rewards))

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)