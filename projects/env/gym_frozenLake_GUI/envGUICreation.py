import numpy as np 
import matplotlib.pyplot as plt

class EnvGUICreator:
    def __init__(self, envSize=(4,4), delay=0.5):
        self.vision = np.zeros([envSize[0],envSize[1],3],dtype=np.uint8)

        self.footPrint = (0, 100, 0)
        self.agent = (0, 255, 0)
        self.obstacle = (0, 0, 0)
        self.start = (255, 0, 0)
        self.end = (0, 0, 255)

        self.statesNum = envSize[0] * envSize[1]
        self.envRows = envSize[0]

        self.lastPoseIdx = 0

        self.delay = delay


    def show(self):
        img = plt.imshow(self.vision)
        img.set_cmap('hot')
        plt.axis('off')
        plt.pause(self.delay)


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
            if ch == "S":
                self.fillPixel(k, self.start)
                k += 1

            elif ch == "F":
                k += 1

            elif ch == "H":
                self.fillPixel(k, self.obstacle)
                k += 1

            elif ch == "G":
                self.fillPixel(k, self.end)
                k += 1


    def step(self, poseIdx):

        if self.lastPoseIdx != 0:
            self.fillPixel(self.lastPoseIdx, self.footPrint)

        self.fillPixel(poseIdx, self.agent)
        
        if poseIdx == 0:
            self.fillPixel(0, self.start)

        self.lastPoseIdx = poseIdx


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