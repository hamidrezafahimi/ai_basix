import numpy as np
from dqn_keras import Agent
from utils import plotLearning, make_env

"""
Training a DDQN agent with observations as images with motion - keras
"""

# To access each "LINK", read the "README.md" in the current folder.
# To understand how this algorithm works, check the figure shown in LINK-3

if __name__ == '__main__':
    
    env = make_env('PongNoFrameskip-v4')
    filename = 'PongNoFrameskip-v4.png'
    num_games = 500
    
    # Defining the agent with its hyper-parameters:
    # gamma, epsilon, and alpha: What programmer says
    # input_dims: Is given as (4, 80, 80). It means: A stack of 4 sequential frames is fed to the
    # network as state (not a single image; Because a sense of motion in the environment is to be
    # provided), each of which with a size of 80x80
    # n_actions: Depends to the environment
    # ... (Why don't you take a look at LINK-11? There, everything is obvious)
    # 
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0001,
                  input_dims=(4,80,80), n_actions=6, mem_size=25000,
                  eps_min=0.02, batch_size=32, replace=1000, eps_dec=1e-5)

    # If running in an evaluation mode: True
    # If running in a training mode: False
    # 
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    # This is for saving the models with the best score:
    # Set it to the worst score, if better than worse, save and set it again
    # 
    best_score = -21

    scores, eps_history = [], []
    n_steps = 0

    # TOPIC: (DRL/DQN) DQN Training Algorithm
    # Just correlate this with the algorithm demonstrated in LINK-3 to figure out what is going on
    # 
    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
            else:
                env.render()
            observation = observation_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.3f' % avg_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)
        if avg_score > best_score:
            agent.save_models()
            print('avg score %.2f better than best score %.2f, saving model' % (
                  avg_score, best_score))
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plot_learning_curve(x, scores, eps_history, filename)