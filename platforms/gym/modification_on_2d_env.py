import matplotlib.pyplot as plt
import numpy as np
import gym

# To access each "LINK", read the "README.md" in the current folder.

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


# TOPIC: (DRL) Modification on a "gym 2d env" For DRL Training
# The following 5 classes handle 5 functionalities needed in training an agent based on a 2d
# gym environment (2d means the observations are images)
# The sample environment on which the code is tested is 'PongNoFrameskip-v4'
# The deployment of the code is placed in LINK-1

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):

        new_frame = np.reshape(frame, frame.shape).astype(np.float32)

        new_frame = 0.299*new_frame[:, :, 0] + 0.587*new_frame[:, :, 1] + \
            0.114*new_frame[:, :, 2]

        new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)

        return new_frame.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.observation_space.shape[-1],
                                                       self.observation_space.shape[0],
                                                       self.observation_space.shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


# TAG-1
# TOPIC: (GYM) How to Convert an Ordinary Environment's States to State-Buffers for Motion Sensing
# For a sense of motion, a state must be defined as a stack of sequential states rather than a 
# single one. This class modifies a gyn env such that its state becomes a sequence as such
# 
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)
