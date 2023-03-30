import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np

# This is a Dueling DDQN agent, developed on  tf2

# To access each "LINK", read the "README.md" in the current folder.

# NOTE: The last DRL code sample has been the LINK-17 (agent) and LINK-18 (training). Thus, the 
# code comments are continued from there

# TOPIC: (DRL/DQN) Dueling DQN Network Architecture
# 
class DuelingDeepQNetwork(keras.Model):
  
    def __init__(self, n_actions, fc1_dims, fc2_dims):
  
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation='relu')
        self.dense2 = keras.layers.Dense(fc2_dims, activation='relu')
  
        # These two are wha t make this a "Dueling Deep Q-Network":
        # The parallel two last layers explained in LINK-1 and depicted in LINK-19
        # 
        self.V = keras.layers.Dense(1, activation=None)
        # 
        # Just a single output! Because it is the value of a states (or a set of states). 
        # 
        self.A = keras.layers.Dense(n_actions, activation=None)
        # 
        # This is: For this action I think the future reward is this, and for that action I think the 
        # future reward is that, ..., and so on.
        # 
        # And remember that no activation function is needed in both parallel output layers. Because
        # we only want the raw values for actions or states


    # TAG-1
    # Formally, in a Dueling DQN, the call function is used in training. For choosing actions it's
    # better to use another function. Because the action values are the outputs of the layer A. Not Q!
    # 
    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        # In LINK-1 it has been discussed that the output of a Dueling Deep Q-Network is caculated 
        # this way:
        # 
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        # 
        # But this is not just an ordinary summation. 
        # Actually, this is a linear combination of V and A. We subtract the average value of A. That
        # is because just doing 'V+A' is not identifiable because you can add any constant to it, so 
        # it will give you the same relative results

        return Q


    # As mentioned in TAG-1, we have a separated function in Dueling DQN to get the advantages 
    # subjected to each action
    # 
    def advantage(self, state):

        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A



class ReplayBuffer():

    def __init__(self, max_size, input_shape):

        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones



# TOPIC: (DRL/DQN) A Dueling DQN Agent 
# 
class Agent():

    def __init__(self, lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=[8], epsilon_dec=1e-3, eps_end=0.01, 
                 mem_size=100000, fc1_dims=128,
                 fc2_dims=128, replace=100, fname='dueling_ddqn_model.h5'):

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.model_file = fname

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def choose_action(self, observation):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action


    # TOPIC: (DRL/DQN) DDQN Learning Algorithm - tf2
    # 
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train function anyway
        # also, no obvious way to copy tensors in tf2?
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        # The result of this is actually the same as LINK-17/TAG-1. The difference in its appearence 
        # is because of the wat tf2 handles the indexing
        # 
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1


    def save_model(self, dir):
        self.q_eval.save(dir, save_format="tf")


    def load_model(self, dir):
        self.q_eval = load_model(dir)
