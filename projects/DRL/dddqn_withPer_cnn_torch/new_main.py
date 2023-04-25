import os
DISPLAY = False
if not DISPLAY:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys
sys.path.append('game/')
import flappy_wrapped as game
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import collections


env = game.GameState()

#Dueling DQN
class DDQN(nn.Module):
    def __init__(self,input_shape,nactions):
        super(DDQN,self).__init__()
        self.nactions = nactions
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=2,stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fca = nn.Sequential(
            nn.Linear( conv_out_size, 512),
            nn.ReLU(),
            nn.Linear( 512, nactions )
        )
        
        self.fcv = nn.Sequential(
            nn.Linear(conv_out_size,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
        
    def _get_conv_out(self,shape):
        o = self.conv( torch.zeros(1,*shape) )
        return int(np.prod(o.size()))
    
    def forward(self,x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        action_v = self.fca(conv_out)
        value_v = self.fcv(conv_out).expand(x.size(0), self.nactions)
        return value_v + action_v - action_v.mean(1).unsqueeze(1).expand(x.size(0), self.nactions)
    

class ExperienceBuffer():
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.priority = collections.deque(maxlen=capacity)
    
    def clear(self):
        self.buffer.clear()
        self.priority.clear()
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self,exp,p):
        self.buffer.append(exp)
        self.priority.append(p)
        
    def sample(self,batch_size):
        probs = np.array(self.priority)/sum(np.array(self.priority))
        indices = np.random.choice( range(len(self.buffer)), batch_size, p = probs)
        states,actions,rewards,dones,next_states = zip(*[ self.buffer[idx] for idx in indices ])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),\
    np.array(dones,dtype=np.uint8), np.array(next_states)


MAX_EPISODES = 10000
STATE_DIM = 4
INITIAL_SKIP = [0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1]
ACTIONS = [0,1]
EXPERIENCE_BUFFER_SIZE = 2000
GAMMA = 0.99
EPSILON_START = 1
EPSILON_FINAL = 0.001
EPSILON_DECAY_FRAMES = (10**4)/3
MEAN_GOAL_REWARD = 10
BATCH_SIZE = 32
MIN_EXP_BUFFER_SIZE = 500
SYNC_TARGET_FRAMES = 30
LEARNING_RATE = 1e-4
SKIP_FRAME = 2


class Agent():

    def __init__(self):
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        self.net = DDQN( (STATE_DIM,84,84), len(ACTIONS) ).to(self.device)
        self.tgt_net = DDQN( (STATE_DIM,84,84), len(ACTIONS) ).to(self.device)
        self.exp_buffer = ExperienceBuffer(EXPERIENCE_BUFFER_SIZE)
        self.epsilon = 1
        self.optimizer = optim.Adam(self.net.parameters(),lr=LEARNING_RATE)

    def act(self, _state, game_id):
        self.epsilon = max( EPSILON_FINAL , EPSILON_START - game_id/EPSILON_DECAY_FRAMES )
        if np.random.random() < self.epsilon:
            action = np.random.choice(ACTIONS)
        else:
            state_v = torch.tensor(np.array([_state],copy=False),dtype=torch.float32).to(self.device)
            action = int(torch.argmax(self.net(state_v)))
        
        return action

    def remember(self, state, next_state, state_reward):

        if len(next_state)==STATE_DIM and len(state)==STATE_DIM:
            #PER - Prioritized Experience Replay
            o = self.net( torch.tensor( np.array([state]),dtype=torch.float32).to(self.device)).to('cpu').detach().numpy()[0][action]
            e = float(torch.max(self.tgt_net( torch.tensor( np.array([next_state]),dtype=torch.float32).to(self.device))))
            p = abs(o-e)+0.0001
            self.exp_buffer.append((state.copy(), action, int(state_reward), done, next_state.copy()),p)

    def learn(self):
        if len(self.exp_buffer) < MIN_EXP_BUFFER_SIZE:
            return
        
        self.optimizer.zero_grad()
        batch = self.exp_buffer.sample(BATCH_SIZE)
        loss_t = self.calc_loss(batch, device=self.device)
        loss_t.backward()
        self.optimizer.step()
    
    def calc_loss(self, batch, device='cpu'):
        states,actions,rewards,dones,next_states = batch
        
        states_v = torch.tensor(states,dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions,dtype=torch.long).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        dones_v = torch.ByteTensor(dones).to(device)
        next_states_v = torch.tensor(next_states,dtype=torch.float32).to(device)
        
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_action_values = self.tgt_net(next_states_v).max(1)[0]
        next_state_action_values[dones_v] = 0.0
        next_state_action_values = next_state_action_values.detach() 
        
        expected_values = rewards_v +  next_state_action_values * GAMMA
        return nn.MSELoss()(state_action_values,expected_values)

    def update_target_net(self, game_id):
        if game_id % SYNC_TARGET_FRAMES == 0:
            self.tgt_net.load_state_dict(self.net.state_dict())


agent = Agent()


KERNEL = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
def processFrame(frame):
    frame = frame[55:288,0:400] #crop image
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to black and white
    frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
    _ , frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
    #frame = cv2.blur(frame,(5,5))
    frame = cv2.filter2D(frame,-1,KERNEL)
    #frame = cv2.Canny(frame,100,200)
    frame = frame.astype(np.float64)/255.0
    return frame


def reset(_state, _next_state):
    global env
    
    state_reward = 0
    _state.clear()
    _next_state.clear()
    
    for i in INITIAL_SKIP[:-7]:
        frame, reward, done = env.frame_step(i)
        state_reward+=reward
        if done:
            reset(env, _state, _next_state)
    frame = processFrame(frame)
    _state.append(frame)
    _next_state.append(frame)

    for i in INITIAL_SKIP[-7:-5]:
        frame,reward,done = env.frame_step(i)
        state_reward+=reward
        if done:
            reset(env, _state, _next_state)
    frame = processFrame(frame)
    _state.append(frame)
    _next_state.append(frame)
    
    for i in INITIAL_SKIP[-5:-3]:
        frame,reward,done = env.frame_step(i)
        state_reward+=reward
        if done:
            reset(env, _state, _next_state)
    frame = processFrame(frame)
    _state.append(frame)
    _next_state.append(frame)
    
    for i in INITIAL_SKIP[-3:-1]:
        frame,reward,done = env.frame_step(i)
        state_reward+=reward
        if done:
            reset(env, _state, _next_state)
    frame = processFrame(frame)
    _state.append(frame)
    _next_state.append(frame)

    return state_reward

def step(action, state, next_state):

    global env, agent
    state_reward = 0
    frame,reward,done = env.frame_step(action)
    state_reward += reward
    for _ in range(SKIP_FRAME):
        frame,reward,done =  env.frame_step(action)
        state_reward += reward
        if done:
            break

    frame = processFrame(frame)
    next_state.append(frame)

    agent.remember(state, next_state, state_reward)
    state.append(frame)

    return int(state_reward), done


def log():
    global agent, best_mean_reward, last_mean
    mean_reward = np.mean(scores[-100:])
    if game_id%5 == 0:
        print("GAME : {} | EPSILON : {:.4f} | MEAN REWARD : {}".format( game_id, agent.epsilon, mean_reward ))
    if best_mean_reward < mean_reward:
        best_mean_reward = mean_reward
        
        if best_mean_reward - last_mean >= 0.1:
            torch.save(agent.net.state_dict(),'checkpoints/flappy_best_model.dat')
            print("REWARD {} -> {}. Model Saved".format(last_mean,mean_reward))
            last_mean = best_mean_reward
    
    return mean_reward >= MEAN_GOAL_REWARD

    
best_mean_reward = float('-inf')
last_mean = float('-inf')

if __name__ == '__main__':

    state = collections.deque(maxlen = STATE_DIM)
    next_state= collections.deque(maxlen = STATE_DIM)

    scores = []
    for game_id in range(MAX_EPISODES):
        reward = reset(state, next_state)
        done = False

        while not done:
            action = agent.act(state, game_id)
            reward, done = step(action, state, next_state)
            agent.learn()

            # 'reward != 0' means crossing a hole, which is equal to winning a game! Thus we assume 
            # starting a new episode with no reset in the next step of the internal loop
            if reward != 0:
                game_id += 1
                scores.append(reward)
                agent.update_target_net(game_id)
                learned = log()

        if learned:
            print("Learned in {} Games.".format(game_id))
            break
