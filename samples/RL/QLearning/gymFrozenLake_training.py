# from io import StringIO
import numpy as np
# import gym
# from gym.envs.toy_text.frozen_lake import generate_random_map
import sys
sys.path.insert(1, '../../../projects/env/gym_frozenLake_GUI')
import envGUICreation as egc

# This is a Q-Learning sample, based on the 'FrozenLake' environment of OpenAI Gym

#  ... sparate folders for different episodes! ...


# To access each "LINK", read the "README.md" in the current folder

# The theroretical background for the algorithm is given in LINK-1
# LINk-2 gives a good demonstration of the algorithm
# Rference for the code: LINK-3
logDir = "../../../data/logs/q_learning_frozenLake/"
logDir = None
# logData = True

# if logData:

env = egc.GymGraphicalFrozenLake(envSize=(8,8), delay=0.1, saveDir=logDir)

action_size = env.env.action_space.n
state_size = env.env.observation_space.n

# env = gym.make('FrozenLake-v0', desc=generate_random_map(size=8))
# egc = envGUICreation.GymFrozenLakeGUICreator((8,8))
# sys.stdout = buffer = StringIO()

qtable = np.zeros((state_size, action_size))

total_episodes = 3            # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
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



def log(index, logDir=None):
    
    if logDir is None:
        return
    with open(logDir + "log_{}.txt".format(index), 'w') as f:
        f.write(np.array2string(qtable))
    
    # env.save()


for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    # step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        
        action = epsilonGreedyPolicy(epsilon)
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
    episode += 1
    log(episode)
    # Reduce epsilon (because we need less and less exploration)
    epsilon = updateEpsilon(episode)
    rewards.append(total_rewards)

    print("episode: {} - score: {}".format(episode, total_rewards))

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)