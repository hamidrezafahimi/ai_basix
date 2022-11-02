
import tensorflow as tf
from tensorflow import keras

"""
KEYWORDS:

Defining Custom Layers (task)

reduce_sum (function)
assign_add (function)
"""

# Defining Custom Layer

# We are to create a layer such that gets an input and accumulates it with previous inputs.
# To do so, create a class like below. Note that it must inherit from keras.layers.Layer class


class CustomSum(keras.layers.Layer):

    # The first funcion that must be wrriten is the constructor of the class
    # 'input_units' is the number of columns (of input) on which the layer is to sum.
    
    def __init__(self, input_units):

        # **Calling parent's constructor:** - This is necessary

        super(CustomSum, self).__init__()

        # **Defining class (layer object's) variables**
        # Another convention is to put the following line in a function named 'build' and call it here.
        # In the following, note that variables must be defined as keras variables
        # "trainable = False" Prevents the variable from changing during training (TF variables
        # are homogeneous with network weights and change during training)

        self.res = tf.Variable(initial_value=tf.zeros(
            shape=(input_units,)), trainable=False)

    # The function 'call' must be overwrriten from the parent. This is the one who is called whenever
    # inputs are given to the layer

    def call(self, inputs):

        # Now we are to accumulate he 'res' value with the inputs.
        # Note that for all operations tf built-ins must be used instead of normal operators. That's
        # because all the actions in system must be performed in a graph-based structure. tf's
        # built-in functions contain the graphical structure with themselves.
        # "assign_add" is equal to +=
        # "reduce_sum" returns sum of elements on the axis

        self.res.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.res


# **Simple Test:**

# input:

x = tf.ones((2, 2))
print(x)

layer = CustomSum(2)
y = layer(x)
print(y.numpy())

# '.numpy()' -> Just make it numpy array to be simpler to read