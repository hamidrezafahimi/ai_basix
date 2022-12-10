import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

"""
Implementation of Simple DQN Network
"""

# To access each "LINK", read the "README.md" in the current folder.

class ReplayBuffer():
    def __init__(self, max_size, input_dims):

        # Size of the buffer
        # 
        self.mem_size = max_size             
        
        self.mem_cntr = 0
        # 
        # Keeps track of our first unsaved memory - Used in buffering the earliest memory

        # Definition of the state memory - '*' means unpack (e.g. elements of a list) in python
        # 
        self.state_memory = np.zeros((self.mem_size, *input_dims), 
                                    dtype=np.float32)

        # The memory for the states next to the states stored in 'state_memory'  (-?-)
        # 
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        
        # reward and action memory
        # 
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        # 
        # To keep track of our 'done' flags. (TAG-1) We want to set all terminal states to zero


    def store_transition(self, state, action, reward, state_, done):

        # What is the index for our first unoccupied memory? If the predefined maximum size for 
        # the agents memory is full, then get your ass to the beginning indices.
        # 
        index = self.mem_cntr % self.mem_size

        # Insert the data in the proper place
        # 
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        # 
        # (TAG-3) We're going to multiply the state whit this last guy, so that the terminal 
        # states become zero (TAG-1). If you don't understand what is going on, take a look at
        # TAG-2.

        # I stored this memory cell. Remember!
        # 
        self.mem_cntr += 1


    # Have you filled the agent's memory enough? If yes, sample a buffer with the predefined 
    # batch_size, among all the memory cells you've filled so far. If no (happens in the initial 
    # steps), just sample from the currently filled memory cells. Not initial zero values!
    # 
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # Generate an array with size: 'batch_size' while its elements are numbers in range(max_mem). 
        # Also, 'replace=False' means that don't take a memory index twice
        # 
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Now sample your random batch from the memory indexes you got
        # 
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


# TOPIC: (DRL/DQN) Build the Deep Q-Network in tf2
# 
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu'),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(n_actions, activation=None)])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model


# TOPIC: (DRL/DQN) A Simple DQN Agent - tf2
# 
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        
        self.model_file = fname

        # Action labels
        # 
        self.action_space = [i for i in range(n_actions)]
        
        # Hyper-parameters of the DQN solution
        # 
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Epsilon's decrement: The step of its change
        # 
        self.eps_dec = epsilon_dec

        # The limit for the epsilon so that it doesn't become zero (Refer to LINK-1 to understand why)
        # 
        self.eps_min = epsilon_end

        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)


    # An interface between the agent and its memory
    # 
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    # TAG-4
    # TOPIC: (DRL) Epsilon-Greedy Action Selection
    # 
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    # 
    # TOPIC: (DRL) Practical Difference Between Value-Based and Policy-Based DRL
    # Compare the above with the LINK-12/TAG-2 to find out what is the practical difference 
    # between a value-based DRL method (like DQN) and a policy-based DRL method (like PG)


    def learn(self):

        # If not enough obsevration and stuff stored yet, wait until the minimum batch size is 
        # provided
        # 
        if self.memory.mem_cntr < self.batch_size:
            return
        # 
        # Otherwise:

        # TOPIC: (DRL/DQN) Simple DQN Learning Algorithm
        # To get a better understanding of how the agent is trained in the following, look at the
        # flowchart demonstrated in LINK-3. 
        
        # Get a sample batch of data to be used in learning. This is training with the experience
        # replay:
        # 
        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 
        # Provide all possible indexes in range of the selected random replay buffer

        # Call the network to get the outputs (Q-values) to be used in next formulations
        # 
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)
        # 
        # If it was DDQN, the 'q_next' (which is used in calculation of target T in LINK-3) would 
        # be obtained from the target network (called Q_next in LINK-4). Take a look at LINK-5 
        # for a better understanding.

        # (TAG-2) This is the implementation of the if-else structure seen in the basic algorithm
        # of DQN (demonstrated in LINK-2)
        # 
        q_target = np.copy(q_eval)
        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones
        # 
        # Fill the target array such that: In each row - related to index of a randomly selected set 
        # of data within the replay buffer - and column - related to the index of action, store the
        # calculated target value (based on the T formula in LINK-3). That means:
        # For each data (state, ...) in our batch, update the action the agent actually took based on
        # The target formula

        # Updating the network's parameters based on the gradient ascent approach - One to last 
        # line the algorithm demonstrated in LINK-2
        # 
        self.q_eval.train_on_batch(states, q_target)

        # Updating the epsilon value such that it never get lower than a particular value
        # 
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min


    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)



