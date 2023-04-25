import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from collections import namedtuple, deque
from copy import deepcopy, copy

class QNetwork(nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(QNetwork, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(), 
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs, bias=True))
        
        # Set to GPU if cuda is specified
        if self.device == 'cuda':
            self.network.cuda()
            
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.learning_rate)
        
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.get_greedy_action(state)
        return action
    
    def get_greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.network(state_t)



class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, 
                                   replace=False)
        # Use asterisk operator to unpack deque 
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in



class DQNAgent:
    
    def __init__(self, env, network, buffer, epsilon=0.05, batch_size=32):
        
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 195 # Avg reward before CartPole is "solved"
        self.initialize()
    
    def take_step(self, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            action = self.network.get_action(self.s_0, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        self.rewards += r
        self.buffer.append(self.s_0, action, r, done, s_1)
        self.s_0 = s_1.copy()
        if done:
            self.s_0 = env.reset()
        return done
        
    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=10000, 
              batch_size=32,
              network_update_frequency=4,
              network_sync_frequency=2000):
        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
            
        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)
                    
                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        ep, mean_rewards), end="")
                    
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break
                        
    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1,1).to(
            device=self.network.device)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)
        
        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0 # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss
    
    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        
    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    buffer = experienceReplayBuffer(memory_size=10000, burn_in=1000)
    dqn = QNetwork(env, learning_rate=1e-3)
    agent = DQNAgent(env, dqn, buffer)
    agent.train(max_episodes=5000, network_update_frequency=1, 
                network_sync_frequency=2500)
