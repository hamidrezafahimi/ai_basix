from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np

# This is a DRL agent, developed based on keras (not tf2!), with a CNN mind architecture, to beat
# the openAI gym's "pong" game

# To access each "LINK", read the "README.md" in the current folder.

# NOTE: The last DRL code sample has been the LINK-6 (agent) and LINK-7 (training). Thus, the 
# code comments are continued from there

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        # TOPIC: (DRL/GEN) State-Memory for Environment with Motion
        # The state memory is a 4D array: 
        #   - 1st dimension is the size specified for the agent's memory ('mem_size')
        #   - 2nd to 4th dimensions are related to shape of a single state
        #   * A single state for a 2D env with motion is: A sequence of images in order to give
        #     the agent a sense of motion. Thus:
        #       - The 2nd dimension: The number of stacked frames taken from the env in each 
        #         time-step
        #       - The 3rd and 4th dimensions: Specifying the image shape
        # Look here at TAG-2 for a better understanding of the concept of "state" in this problem
        # Look at LINK-13/TAG-1 to see how a gym environment is modified to convert the 
        # single-states to state-batches
        # 
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                      dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                          dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)


    def store_transition(self, state, action, reward, state_, done):

        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        
        self.terminal_memory[index] = done
        # 
        # This is a definition for the memory of 'done' flags, different from LINK-6/TAG-3. 
        # Notice that we are not using the "one-hot encoding" for the actions in this program

        self.mem_cntr += 1


    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# TOPIC: (DRL/DQN) CNN Network for agent with input states as images
# 
def build_dqn(lr, n_actions, input_dims, fc1_dims):

    model = Sequential()

    # TOPIC: (DRL/DQN) The Structure of a Convolutional Layer in DQN
    # Take a look at LINK-8 for theretical backgrounds and LINK-9 for more descriptions and 
    # details about the structure of the following layer
    # TOPIC: (DRL/DQN) Training Agent with Time-Varying Environment and CNN Brain
    # In the following network, a batch of stacked frames (Here with a conventional size of 4) 
    # will be given to the network so that a sense of motion is provided for the agent.
    # (TAG-1)
    # 
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                     input_shape=(*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                     data_format='channels_first'))
    # 
    # The 'input_dims' in the next lyers will be inferred after it is specified in the first 
    # layer

    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model



# TOPIC: (DRL/DQN) A Simple DDQN Agent - keras
# 
class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace,
                 input_dims, eps_dec=0.996,  eps_min=0.01,
                 mem_size=1000000, q_eval_fname='q_eval.h5',
                 q_target_fname='q_next.h5'):
        
        # The basic understanding for the followings is provided previously in the coments in 
        # LINK-6
        # 
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        # 
        # TAG-2
        # TOPIC: Input 2D (Image) Observarion to a DRL Agent in Environment with Motion
        # The parameter 'input_dim', is passed by the main training execution program (LINK-10)
        # and used in declaration of the input to the agent's NN (TAG-1) and also here in 
        # declaration of the state memory (TAG-2) 
        # The value for this parameter is given as (4, 80, 80) in LINK-10. It means: A buffer of
        # 4 sequential frames is fed to the network as state (not a single image; Because a 
        # sense of motion in the environment is to be provided), each of which with a size of 
        # 80x80 
        
        # In DDQN, these two are necessary for copying the evaluation network to the target 
        # network after a period of steps
        # 
        self.replace = replace
        self.learn_step = 0
        # 
        # The last one is a counter specifying how many times the 'learn' function is been 
        # called. So after a certain number of steps (= times that the 'learn' function is been
        # called), the evaluation network is copied to the target network

        # In normal DQN, we defined a single network used in both evaluation and as target:
        # 
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 512)
        # 
        # But in DDQN, as clarified in LINK-3 and LINK-5, there is a separated "target network":
        # 
        self.q_next = build_dqn(alpha, n_actions, input_dims, 512)


    # This is dedicated to DDQN rather than DQN. Update the target network with the evaluation 
    # network after a certain number of steps
    # 
    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    # Just like LINK-6/TAG-4
    # 
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action


    # TOPIC: (DRL/DQN) DDQN Learning Algorithm - keras
    # What is not commented has been described previously in LINK-6
    # 
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            # LINK-3 and LINK-5 are useful to review the program's path through the following 
            # lines of code

            q_eval = self.q_eval.predict(state)

            q_next = self.q_next.predict(new_state)

            """
            Thanks to Maximus-Kranic for pointing out this subtle bug.
            q_next[done] = 0.0 works in Torch; it sets q_next to 0
            for every index that done == 1. The behavior is different in
            Keras, as you can verify by printing out q_next to the terminal
            when done.any() == 1.
            Despite this, the agent still manages to learn. Odd.
            The correct implementation in Keras is to use q_next * (1-done)
            q_next[done] = 0.0
            q_target = q_eval[:]
            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + \
                                        self.gamma*np.max(q_next,axis=1)
            """
            # A description for the following is given in LINK-6/TAG-2
            # 
            q_target = q_eval[:]
            indices = np.arange(self.batch_size)
            q_target[indices, action] = reward + \
                                    self.gamma*np.max(q_next, axis=1)*(1 - done)
            # 
            # Compare the 'q_next' with LINK-6 and correlate it with LINK-5 to figure out the exact
            # difference of DQN and DDQN
            
            self.q_eval.train_on_batch(state, q_target)

            self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
            
            self.learn_step += 1


    def save_models(self):
        
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)
        print('... saving models ...')


    def load_models(self):
        
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_nexdt = load_model(self.q_target_model_file)
        print('... loading models ...')