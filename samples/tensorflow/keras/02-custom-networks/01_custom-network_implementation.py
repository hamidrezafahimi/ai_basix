
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
from re import X
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Note: tensorflow 2.x is required! Be careful to add the correct packages


"""
KEYWORDS:

"The 2nd Method to Create the Model Object: Custom Model" (local topic)
"""

# **1 - Dataset Preparation**
# Prepare the dataset:

fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Preprocess data:

X_train = X_train/255.0
X_test = X_test/255.0


# **2 - Model Design**


# How to design a custom model (example: an ensambeled NN model):

# To shape a topology of sub-models and create a custom model as a directed graph, two task must
# be done:
# 1st task: Create each node (sub-model) of the graph (done in TAG-1)
# 2nd task: Determine the directed connections between the models  (done in TAG-2). This second
# task is done when declaring the input(s) of each node.

# We are to design a custom (ensambeled) NN model. This is the shape of the graph to be designed:
#
#                       | [MLP Net] ->|
#                       | [MLP Net] ->|
# [28*28 Image input] ->| [MLP Net] ->|-> [A Combiner single layer]
#                       | [MLP Net] ->|
#                       | [MLP Net] ->|

# Definition of an input shape - which will be used in future

inp = keras.Input(shape=(28, 28))

# Defining five similar MLP models

mini_models = []
for i in range(5):
    
    # (TAG-1) Creation of predefined nodes (each parallel MLP node): 
    
    model = keras.models.Sequential([
        
        # Note that you can also put names on layers:
        
        keras.layers.Flatten(input_shape=(28, 28), name="input_layer"),
        keras.layers.Dense(units=128, activation="relu", name="hidden_layer"),
        keras.layers.Dense(10, name="output_layer")
    ])
    
    mini_models.append(
        
        # (TAG-2) Declaring the input(s) for each node
        
        model(inp)
    )

# (TAG-1) Creation of custom nodes (the final sequential combiner node - this is just two layers):
# The following, is the "The 2nd Method to Create the Model Object: Custom Model". In this method:
# 1. each layer is defined manually. Its input can be a 

combiner = keras.layers.average(mini_models)
output_layer = keras.layers.Softmax()

# 2. The object (= output) of each defined layer is given to the next layer

output_value = output_layer(combiner)

# 3. The final custom model (which can be integration of custom and predefined models) is created 
# using the `keras.Model` module, entering only the input layer object (from keras.Input) and output 
# object (return value of last layer - which contains the whole history of node connections):

model = keras.Model(inputs=inp, outputs=output_value)

model.summary()
keras.utils.plot_model(model, show_shapes=True,
                       show_layer_names=True, expand_nested=True)


# **3 - Model Compile**

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=["accuracy"])


# **4 - Model Training**

# Training with a validation dataset

model.fit(X_train, Y_train, epochs=3, validation_split=0.1)

# NOTE: Calling the 'fit' function sequentially results the nexts calls continue on the output net
# of the previous calls


# **5 - Model Evaluation**

# Evaluation of the trined model on 'test dataset'
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)


# **6 - Using the Model**

# First, the input data must be prepared:

X = X_test[0]       # The first picture of test dataset
plt.imshow(X)

Y = Y_test[0]       # The first label of test dataset
print(Y)

# Use the trained network (give input and get output):

predicted_distr = model.predict(np.array([X]))
print(predicted_distr)

# a single input output
predicted_class = np.argmax(predicted_distr)
print(predicted_class)
