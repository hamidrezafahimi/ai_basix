from io import StringIO
import numpy as np 
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import sys
sys.path.insert(1, '../../../modules')
import gen
import time
import os


class GridEnvGUICreator:
    def __init__(self, envSize=(4,4), delay=0.5, show=True):
        self.vision = np.zeros([envSize[0],envSize[1],3],dtype=np.uint8)
        self.logDir = None
        self.footPrint = (0, 100, 0)
        self.agent = (0, 255, 0)
        self.obstacle = (0, 0, 0)
        # self.start = (255, 0, 0)
        self.end = (0, 0, 255)

        self.statesNum = envSize[0] * envSize[1]
        self.envRows = envSize[0]
        self.envSize = envSize

        self.lastPoseIdx = 0

        self.delay = delay
        self.iteration = 0
        self.show = show


    def render(self, show=True):

        save = not self.logDir is None

        if not self.show and not save:
            return
        
        img = plt.imshow(self.vision)
        img.set_cmap('hot')
        plt.axis('off')

        if self.show:
            plt.pause(self.delay)

        if save:
            plt.savefig(self.logDir + "step_{}".format(self.iteration))
            self.logDir = None


    def fillPixel(self, idx, val):
        row, col = self.getRowCol(idx)
        self.vision[row, col] = val

    def getRowCol(self, idx):
        row, col = divmod(idx, self.envRows)
        return row, col

    def make(self, envStr):

        data = np.uint8(np.random.uniform(low=0.65, high=0.95, size=self.envSize)*255)
        self.vision[:, :, 2] = data
        self.vision[:, :, 1] = data
        self.vision[:, :, 0] = data

        k = 0
        # print("envStr")
        for ch in envStr:
            if ch == "S":
                k += 1

            if ch == "F":
                k += 1

            elif ch == "H":
                self.fillPixel(k, self.obstacle)
                k += 1

            elif ch == "G":
                self.fillPixel(k, self.end)
                k += 1

        self.originalVision = np.copy(self.vision)


    def step(self, poseIdx):

        # if self.lastPoseIdx != 0:
        self.fillPixel(self.lastPoseIdx, self.footPrint)

        self.fillPixel(poseIdx, self.agent)
        
        # if poseIdx == 0:
        #     self.fillPixel(0, self.start)

        self.lastPoseIdx = poseIdx
        self.iteration += 1
    

    def reset(self):
        self.vision = np.copy(self.originalVision)
        self.lastPoseIdx = 0
        self.iteration = 0


if __name__ == '__main__':

    guic = EnvGUICreator((8,8))

    envData = "SFFFFFFF\
            FFFFFFFF\
            FFFHFFFF\
            FFFFFHFF\
            FFFHFFFF\
            FHHFFFHF\
            FHFFHFHF\
            FFFHFFFG"

    guic.make(envData)
    guic.show()


class GymGraphicalFrozenLake(GridEnvGUICreator):

    def __init__(self, envSize=(4, 4), delay=0.5, visualize=True, show=True, logMode=True):
        super().__init__(envSize, delay, show=show)
        self.visualize = visualize
        self.logMode = logMode
        # self.logDir = None
        # self.show = show
        # if not logDir is None:
        if logMode or show:
            sys.stdout = self.buffer = StringIO()
        self.logFile = "log.txt"
        if os.path.exists(self.logFile):
            os.remove(self.logFile)
        open(self.logFile, 'x')
        # sys.stdout = gen.TerminalLog(file)
        self.env = gym.make('FrozenLake-v0', desc=generate_random_map(size=envSize[0]),
                             is_slippery=False)
        self.made = False
        self.numStates = self.env.observation_space.n
        self.resetRewardMap()


    def resetRewardMap(self):
        self.rewardMap = np.zeros(self.numStates)
        self.rewardMap[self.numStates-1] = 100


    def cleanLog(self, inputLog):
        inputLog = inputLog.replace('[41m', '')
        inputLog = inputLog.replace('[0m', '')
        outputLog = inputLog.replace('', '')
        return outputLog

    
    def saveImages(self, dir):
        self.logDir = dir


    def logConsole(self):
        log = self.buffer.getvalue()
        with open(self.logFile, 'w') as file:
            file.write(log)
        return log


    def reset(self):
        state = self.env.reset()
        state2d = self.get2dState(state)
        self.resetRewardMap()
        # if not self.logDir is None:

        # print("log: ", log)
        if not (self.made or self.show or self.logMode):
            pass
        elif not self.made:
            self.env.render()
            log = self.logConsole()
            super().make(self.cleanLog(log))
            self.made = True
        else:
            super().reset()
        return state, state2d


    def step(self, action, state):
        n_state, gymReward, done, info = self.env.step(action)
        super().step(n_state)
        super().render(self.visualize)

        reward = self.reward(state, n_state, done, gymReward)
        
        n_state2d = self.get2dState(n_state)
        return n_state, reward, done, info, n_state2d


    def get2dState(self, state):
        row, col = super().getRowCol(state)
        return np.array([row, col])


    def reward(self, state, newState, done, gymReward):
        reward = self.rewardMap[newState]
        self.rewardMap[newState] -= 1
        # This is a modification on the original gym env. If not desired, disable it:
        if done and gymReward == 0:
            reward -= 100
        # print(state)
        # print(newState)
        if not done and newState == state:
            reward -= 2
        return reward

    def close(self):
        self.env.close()