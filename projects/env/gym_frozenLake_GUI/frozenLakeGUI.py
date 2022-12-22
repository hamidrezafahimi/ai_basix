from io import StringIO
import sys
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import envGUICreation as eguic


def cleanLog(inputLog):

    inputLog = inputLog.replace('[41m', '')
    inputLog = inputLog.replace('[0m', '')
    outputLog = inputLog.replace('', '')
    return outputLog

if __name__ == '__main__':

    env = gym.make('FrozenLake-v0', desc=generate_random_map(size=8))
    egc = eguic.EnvGUICreator((8,8))
    sys.stdout = buffer = StringIO()

    state = env.reset()
    done = False
    score = 0 
    out = env.render()

    log = buffer.getvalue()
    log = cleanLog(log)

    egc.make(log)

    while not done:

        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        
        egc.step(n_state)
        egc.show()

        score+=reward

        if done:
            break

    print('Score:{}'.format(score))

    env.close()