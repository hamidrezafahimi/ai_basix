import gym
import matplotlib.pyplot as plt
import numpy as np
import gen
import time

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    
    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def justPlayGym(envName, n_games, agent = None):

    env = gym.make(envName)

    for i in range(n_games):
        
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation) if not agent is None else env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            score += reward
            env.render()
            observation = observation_

        print('episode: ', i,'score: %.2f' % score)


def TDTrain(envName, agent, n_games, modelFolderName, filename):

    saveDir = gen.makeModelSaveDir(modelFolderName)
    logDir = gen.makeLogSaveDir(modelFolderName + "_log")

    env = gym.make(envName)
    ddqn_scores = []
    eps_history = []

    for i in range(n_games):
        t0 = time.time()
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            # print('before learn')
            agent.learn()
            # print('after learn')
        
        eps_history.append(agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[-100:])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score, ' time: %.2f' % (time.time() - t0))

    agent.save_model(saveDir)
    fileDir = logDir + filename

    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)
