from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

# ** Implementation of a DRL model with the "policy gradients" algorithm in keras (not tensorflow!) **

# To access each "LINK", read the "README.md" in the current folder.

class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 fname='reinforce.h5'):

        # The `Policy Gradient` method's parameters:
        # 
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0

        # Parameters of the deep neural network using which the policy is approximated:
        # 
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size

        # Number of probable actions (discrete):
        # 
        self.n_actions = n_actions

        # Memory parameters for the agent - Simple case of training with single episode (episode batch
        # size = 1):
        # 
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # Two separate networks - Subjected in the issue reviewed in LINK-1.
        # The model returning from the following funtion is the network demonstrated in LINK-2.
        # 
        self.policy, self.predict = self.build_policy_network()
        
        # The labels for actions
        # 
        self.action_space = [i for i in range(n_actions)]

        self.model_file = fname


    def build_policy_network(self):

        advantages = Input(shape=[1])

        # I have given examples so far (in LINK-3, LINK-4, and LINK-5). This is just an ordinary model
        # design in keras:
        # 
        input = Input(shape=(self.input_dims,))
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        # 
        # Remember that the last layer is going to give a probabilty distribution function (what a 
        # policy function actually returns). Thus, a 'softmax' activation function is required (
        # -> LINK-6).

        # TOPIC: (DRL) A Custom Loss Function in Keras (no tf)
        # The desired loss function for this policy-function-estimator network is not present in 
        # keras. Thus, it is defined here. No other part of the class needs access to this function. 
        # So a nested function is defined.
        # About inputs: (-?-)
        # 
        def custom_loss(y_true, y_pred):

            # We don't want to deal with log(0). So we use the 'clip' function:
            # 
            out = K.clip(y_pred, 1e-8, 1-1e-8)

            # Loss function's formula: log-likelihood (-?-):
            # 
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        # There are two inputs given to a PG network. The first one is - as a rule - the states. 
        # The second one, is the rewards. Here, we define the the rewards as a variable named 
        # 'advantages'. Take a look at LINK-7.
        # 
        policy = Model(input=[input, advantages], output=[probs])

        policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)

        # TOPIC: Defining the mirror network in DRL
        # Not that, during training the first network, the weights are to be updated online. While 
        # the second network (mirror) is just to be used to do the predictions. So, the inputs to two 
        # networks are different. 
        # 
        predict = Model(input=[input], output=[probs])

        return policy, predict


    # This function handles the feedforward pass in the 'predict' model (the mirrored model). It 
    # receives a set of states (observations) and outputs an action:
    # 
    def choose_action(self, observation):

        # During the training, the observation contains the whole history of the states, whereas the 
        # prediction only needs a single state-set as input to the 'predict' network. So the states in
        # the last observation are separated to "predict", and then "choose" an action.
        # 
        state = observation[np.newaxis, :]

        # Get the predicted action-space probability distribution, giving a set of states:
        # The last 0-index is because the function returns a tuple, first element of which is the 
        # output probability distribution (output of last layer: softmax)
        # 
        probabilities = self.predict.predict(state)[0]

        # Selecting the action given the output probability distribution:
        # The following, selects an action among the probable actions (action space) with a given 
        # probability
        # 
        action = np.random.choice(self.action_space, p=probabilities)

        return action


    # This is the interface with the memory. It saves the states, actions and rewards:
    # 
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)


    # Here, we iterate over the history of each episode and calculate the return values related to 
    # each time step. So the weights are to be tunned in the increasing direction of the cost function
    # 
    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        # TOPIC: Performing One-Hot Encoding on the Network Outputs
        # Each single-action (referred to a time-step) is one-hot encoded. For more details, refer to
        # LINK-8
        # 
        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        # The "discounted rewards from now on" 'R_t' in formulations of LINK-9, and 'G_t' in 
        # formulations of LINK-10 is calculated for each time-step and saved in an array each element
        # of which correlated to a time-step
        # 
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        # TOPIC: Zero-Based Normalization on Network Inputs
        # A zero-based normalization is done on the discounted rewards. That's because generally, the
        # rewards may be much different; Thus must be normalized
        # 
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std
        # 
        # The second line is to avoid "devide-by-zero" situation

        # TOPIC: Network Inputs in DRL
        # Notice that the network which is being trained online receives two inputs: time history of 
        # states (observations) and rewards. Despite the mirrored network which is used in evaluation 
        # and only receives the states
        # 
        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)