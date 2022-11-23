import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

# ** Design of a DRL network for the "policy gradients" algorithm in tensorflow-2 **

# To access each "LINK", read the "README.md" in the current folder.

# This is simply making a custom model in tf2. For more details, refer to LINK-11
# 
class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')
        # 
        # The output of the network is the "policy". Therefore, we call the last layer 'pi'.
        # Remember that according to the fact that the output policy-function (LINK-9) is a 
        # "probability distribution", a softmax activation function is necessary at the end.


    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)

        return pi