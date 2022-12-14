from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

# This is a DRL agent, developed based on keras (not tf2!)

# To access each "LINK", read the "README.md" in the current folder.

# NOTE: The last DRL code sample has been the LINK-11 (agent) and LINK-10 (training). Thus, the 
# code comments are continued from there


class ReplayBuffer(object):

    def __init__(self, max_size, input_shape, n_actions, discrete=False):
    
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
        self.discrete = discrete
        # 
        # This is for incorporating both discrete and continuous action-space cases in the agent
        # TOPIC: (DRL/GEN) When is One-Hot Encoding Necessary?
        # In DRL, a "one-hot encoding" is performed when the action-space (the outputs of the 
        # neural network) is discrete; Not continuous.


    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            # TOPIC: (DRL/GEN) Performing One-Hot Encoding on the Network Outputs
            # This is same as LINK-14/TAG-1
            # 
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1


    # To what it is, take a look at LINK-6/TAG-5
    # 
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model



class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5'):

        # TAG-1
        # Definition of the discrete actions with labels as separated integer numbers
        # 
        self.action_space = [i for i in range(n_actions)]

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):

        # This is for preserving compatability for the agent in order to handle batch training as well
        # as a feedforward pass for a single state. To avoid defining to different functions for two 
        # different tasks. Note that the input 'state' to this function is used in other places of 
        # the main program (for batch training so its shape is proper for that). Whereas in this
        # function, the feedforward pass to the network is desired.
        # 
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            # TOPIC: Reverse of One-Hot Encoding
            # The following converts the one-hot-encoded 'action' array to an integer-encoded
            # representation (it means: each action is specified with an integer number as the 
            # action-apace was previously defined in TAG-1)
            # How the 'action_indices' is calculated below, is practically similar to 'actions'
            # variable in LINK-6/TAG-2.
            # 
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_eval = self.q_eval.predict(state)

            q_next = self.q_eval.predict(new_state)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                                  self.gamma*np.max(q_next, axis=1)*done

            _ = self.q_eval.fit(state, q_target, verbose=0)
            # 
            # This is how to perform the one to last line in the algorithm demonstrated in LINK-2,
            # in keras

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)