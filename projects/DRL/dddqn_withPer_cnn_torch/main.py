
import os
import pygame
DISPLAY = False
if not DISPLAY:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
sys.path.append('game/')
import flappy_wrapped as game
import cv2
import numpy as np
import time
# %matplotlib inline
import matplotlib.pyplot as plt
import argparse

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

import torch
import torch.nn as nn
import torch.optim as optim

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
    

ACTIONS = [0,1]
EXPERIENCE_BUFFER_SIZE = 2000
STATE_DIM = 4
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
INITIAL_SKIP = [0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1]

import collections
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




class Agent():
    def __init__(self,env,buffer,state_buffer_size = STATE_DIM):
        self.env = env
        self.exp_buffer = buffer
        self.state = collections.deque(maxlen = STATE_DIM)
        self.next_state= collections.deque(maxlen = STATE_DIM)
        self._reset()
        
    def _reset(self):
        self.total_rewards = 0
        self.state.clear()
        self.next_state.clear()
        
        for i in INITIAL_SKIP[:-7]:
            frame,reward,done = self.env.frame_step(i)
            self.total_rewards+=reward
            if done:
                self._reset()
        frame = processFrame(frame)
        self.state.append(frame)
        self.next_state.append(frame)

        for i in INITIAL_SKIP[-7:-5]:
            frame,reward,done = self.env.frame_step(i)
            self.total_rewards+=reward
            if done:
                self._reset()
        frame = processFrame(frame)
        self.state.append(frame)
        self.next_state.append(frame)
        
        for i in INITIAL_SKIP[-5:-3]:
            frame,reward,done = self.env.frame_step(i)
            self.total_rewards+=reward
            if done:
                self._reset()
        frame = processFrame(frame)
        self.state.append(frame)
        self.next_state.append(frame)
        
        for i in INITIAL_SKIP[-3:-1]:
            frame,reward,done = self.env.frame_step(i)
            self.total_rewards+=reward
            if done:
                self._reset()
        frame = processFrame(frame)
        self.state.append(frame)
        self.next_state.append(frame)
    
    def step(self,net,tgt_net,epsilon=0.9,device='cpu'):
        self.total_rewards = 0
        if np.random.random() < epsilon:
            action = np.random.choice(ACTIONS)
        else:
            state_v = torch.tensor(np.array([self.state],copy=False),dtype=torch.float32).to(device)
            action = int(torch.argmax(net(state_v)))
       
        frame,reward,done = self.env.frame_step(action)
        self.total_rewards += reward
        for _ in range(SKIP_FRAME):
                frame,reward,done =  self.env.frame_step(action)
                self.total_rewards += reward
                if done:
                    break
                    
        frame = processFrame(frame)
        self.next_state.append(frame)
        
        if len(self.next_state)==STATE_DIM and len(self.state)==STATE_DIM:
            #PER - Prioritized Experience Replay
            o = net( torch.tensor( np.array([self.state]),dtype=torch.float32).to(device)).to('cpu').detach().numpy()[0][action]
            e = float(torch.max(tgt_net( torch.tensor( np.array([self.next_state]),dtype=torch.float32).to(device))))
            p = abs(o-e)+0.0001
            self.exp_buffer.append((self.state.copy(),action,int(self.total_rewards),done,self.next_state.copy()),p)
        
        self.state.append(frame)
        
        end_reward = int(self.total_rewards)
        if done:
            self._reset()
            # print("Lost the episode")
            # input("enter 3 ...")
        return end_reward

def calc_loss(batch,net,tgt_net,device='cpu'):
    states,actions,rewards,dones,next_states = batch
    
    states_v = torch.tensor(states,dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions,dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    dones_v = torch.ByteTensor(dones).to(device)
    next_states_v = torch.tensor(next_states,dtype=torch.float32).to(device)
    
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_action_values = tgt_net(next_states_v).max(1)[0]
    next_state_action_values[dones_v] = 0.0
    next_state_action_values = next_state_action_values.detach() 
    
    expected_values = rewards_v +  next_state_action_values * GAMMA
    return nn.MSELoss()(state_action_values,expected_values)


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=False, help="Model file to load" ,dest="model")

if __name__ == '__main__':

    args = parser.parse_args()

    all_losses = []
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

    #Double Dueling DQN

    net = DDQN( (STATE_DIM,84,84), len(ACTIONS) ).to(device)
    tgt_net = DDQN( (STATE_DIM,84,84), len(ACTIONS) ).to(device)

    # If there is an already trained model, load it. Otherwise, initiate from the first
    if not args.model is None:
        net.load_state_dict(torch.load(args.model))
        tgt_net.load_state_dict(net.state_dict())

    env = game.GameState()
    buffer = ExperienceBuffer(EXPERIENCE_BUFFER_SIZE)
    agent = Agent(env,buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(),lr=LEARNING_RATE)

    total_rewards = []
    best_mean_reward = float('-inf')
    last_mean = float('-inf')
    game_id = 0
    while True:
        epsilon = max( EPSILON_FINAL , EPSILON_START - game_id/EPSILON_DECAY_FRAMES )
        # If you have loaded an already trained model, comment the above and set the epsilon 
        # manually like the following
        # epsilon = EPSILON_FINAL

        reward = agent.step(net,tgt_net,epsilon,device=device)
        # print(reward)
        # input("enter 1 ...")
        if reward != 0:
            game_id += 1
            total_rewards.append(reward)
            # print(total_rewards[len(total_rewards)-1])
            # print(game_id)
            # input("enter 2 ...")# print(total_rewards)
            mean_reward = np.mean(total_rewards[-100:])
            if game_id%5 == 0:
                print("GAME : {} | EPSILON : {:.4f} | MEAN REWARD : {}".format( game_id, epsilon, mean_reward ))
            if best_mean_reward < mean_reward:
                best_mean_reward = mean_reward
                
                if best_mean_reward - last_mean >= 0.1:
                    torch.save(net.state_dict(),'checkpoints/flappy_best_model.dat')
                    print("REWARD {} -> {}. Model Saved".format(last_mean,mean_reward))
                    last_mean = best_mean_reward

            if game_id % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
                
            if mean_reward >= MEAN_GOAL_REWARD:
                print("Learned in {} Games.".format(game_id))
                break
        
        if len(buffer) < MIN_EXP_BUFFER_SIZE:
            continue
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch,net,tgt_net,device=device)
        all_losses.append(float(loss_t))
        loss_t.backward()
        optimizer.step()

