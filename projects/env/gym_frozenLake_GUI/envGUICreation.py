from io import StringIO
import numpy as np 
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import sys


class GridEnvGUICreator:
    def __init__(self, envSize=(4,4), delay=0.5):
        self.vision = np.zeros([envSize[0],envSize[1],3],dtype=np.uint8)

        self.footPrint = (0, 100, 0)
        self.agent = (0, 255, 0)
        self.obstacle = (0, 0, 0)
        # self.start = (255, 0, 0)
        self.end = (0, 0, 255)

        self.statesNum = envSize[0] * envSize[1]
        self.envRows = envSize[0]

        self.lastPoseIdx = 0

        self.delay = delay
        self.iteration = 0


    def render(self, show=True, saveDir=None):

        img = plt.imshow(self.vision)
        img.set_cmap('hot')
        plt.axis('off')
        
        if show:
            plt.pause(self.delay)
        
        if not saveDir is None:
            plt.savefig(saveDir + "step_{}".format(self.iteration))


    def fillPixel(self, idx, val):
        row, col = divmod(idx, self.envRows)
        self.vision[row, col] = val


    def make(self, envStr):

        data = np.uint8(np.random.uniform(low=0.65, high=0.95, size=(8,8))*255)
        self.vision[:, :, 2] = data
        self.vision[:, :, 1] = data
        self.vision[:, :, 0] = data

        k = 0
        for ch in envStr:
            # if ch == "S":
            #     self.fillPixel(k, self.start)
            #     k += 1

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

    def __init__(self, envSize=(4, 4), delay=0.5, visualize=True, saveDir=None):
        super().__init__(envSize, delay)
        self.visualize = visualize
        self.saveDir = saveDir
        sys.stdout = self.buffer = StringIO()
        self.env = gym.make('FrozenLake-v0', desc=generate_random_map(size=envSize[0]))
        self.made = False


    def cleanLog(self, inputLog):
        inputLog = inputLog.replace('[41m', '')
        inputLog = inputLog.replace('[0m', '')
        outputLog = inputLog.replace('', '')
        return outputLog


    def reset(self):
        state = self.env.reset()
        self.env.render()
        log = self.buffer.getvalue()
        if not self.made:
            super().make(self.cleanLog(log))
            self.made = True
        else:
            super().reset()
        return state


    def step(self, action):
        n_state, reward, done, info = self.env.step(action)
        super().step(n_state)
        super().render(self.visualize, self.saveDir)
        return n_state, reward, done, info


    def close(self):
        self.env.close()