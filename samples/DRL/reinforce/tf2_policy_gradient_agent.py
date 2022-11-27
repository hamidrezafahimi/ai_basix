import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from tf2_policy_gradient_network import PolicyGradientNetwork

class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256):

        # The `Policy Gradient` method's parameters:
        # 
        self.gamma = gamma
        self.lr = alpha

        # Number of probable actions (discrete):
        # 
        self.n_actions = n_actions

        # Memory parameters for the agent - Simple case of training with single episode (episode batch
        # size = 1):
        # 
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        # Create and compile the network for model to be trained:
        # Check the module 'tf2_policy_gradient_network'. There, the network is defined.
        # 
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    # This function handles the feedforward pass in the 'predict' model (the mirrored model). It 
    # receives a set of states (observations) and outputs an action:
    # 
    def choose_action(self, observation):

        # TAG-1
        # TOPIC: (GEN) The Input to tf2's Dense Layers
        # - The input to tf2's Dense layers has to be 2D. So create a nested list like below.
        # - The input to tf2's Dense layers has to be of tensor type. So convert it like below.
        # 
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        # TOPIC: (DRL) Using a Network in a Single Time-Step (forward pass):
        # Get the predicted action-space probability distribution (output of last layer: softmax),
        # giving a set of states:
        # 
        probs = self.policy(state)

        # TOPIC: (DRL) Probabilistic Action Selection 
        # Selecting the action given the output probability distribution:
        # Select an action among the probable actions (action space) with a given probability
        # 
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        # TOPIC: (DRL) Passing Network Output (action) to OpenAI Gym
        # Notice that the final resulting action (the agent's output) must be a 1D numpy array. 
        # Despite the input to the network's Dense layers which was tf2's 2D tensor. Thus, a 
        # conversion like below is needed.
        # 
        return action.numpy()[0]

    # Store the memory of current episode:
    # 
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    # Iterate over the history of the current episode, calculate and store the discounted rewards (R_t
    # in LINK-9) related to each time-step, and complete the mathematics related to PG algorithm (
    # pointed in LINK-9 and LINK-10) until updating the network parameters (e.g. with gradient ascent)
    # 
    def learn(self):

        # The reason for the following is same as explained in TAG-1
        # 
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)

        # 'rewards' are not going to be passed directly to the network (-?- Why? What about the 
        # mirrored network?)
        rewards = np.array(self.reward_memory)

        # TOPIC: (DRL) How to Calculate Discounted Sum of Future Rewards (DSoFR) from Now On (G_t or R_t)
        # The "discounted rewards from now on" 'R_t' in formulations of LINK-9, and 'G_t' in 
        # formulations of LINK-10 is calculated for each time-step and saved in an array each element
        # of which correlated to a time-step
        # For a better understanding, take a look at the figure demonstrating a schematics of the
        # following block's algorithm in LINK-12
        # Notice that, algorithmically, the following block of code is the 'one line to the end' in 
        # the algorithm demonstrated in LINK-13
        # 
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # The following 4 blocks is the calculation of loss and implementation of the mathematics of 
        # PG method to train the network, based on the algorithm and mathematical formulas presented 
        # in LINK-10
        # 
        with tf.GradientTape() as tape:
            loss = 0

            # The loop iterates over all "DSoFR"s and "state"s to calculate the non-gradiented form of
            # last term in the last line in the algorithm shown in LINK-13
            # 
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        # Calculate the gradient to obtain the final form of last term in the last line of the 
        # algorithm shown in LINK-13
        # 
        gradient = tape.gradient(loss, self.policy.trainable_variables)

        # Apply the gradient and update the parameters of the network: Complete form of last line in 
        # the algorithm shown in LINK-13
        # 
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []