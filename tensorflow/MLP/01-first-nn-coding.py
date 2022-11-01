
import os
from re import X
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import tensorflow as tf
# Note: tensorflow 2.x is required! Be careful to add the correct package
from tensorflow import keras

import numpy as np

"""
KEYWORDS:

optimizer (concept - function argument)
loss (concept - function argument)
metrics (concept - function argument)

model (object)
compile (function)
fit(function)
"""

# **1 - Dataset Preparation**

# Prepare the dataset:

fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), _ = fashion_mnist.load_data()

# Check what is loaded:
print(X_train.shape, Y_train.shape)


# Preprocess data:
X_train = X_train/255.0


# **2 - Model Design**

# This is one of the ways to create the model object:
model = keras.models.Sequential([
                                 # This first layer makes a row-vector from the input image:
                                 keras.layers.Flatten(input_shape=(28,28)),
                                 # This is the hidden layer:
                                 keras.layers.Dense(units=128, activation="relu"),
                                 # This is the output layer - the probability for 10 classes to
                                 # find out what probability is more (so the input belongs to 
                                 # that class):
                                 keras.layers.Dense(10)
])


# **3 - Model Compile**

# compile the model before training
# These three must be given to the 'compile' function
# optimizer : Updates the weights (the optimization method). The mathematical optimization 
# (-> back-propagation) procedure is handled with this guy. There are less than 10 other optimizers 
# available (gradien descent, stochastic gadient descent, RMS prob, ada delta, etc.). 
# loss : Type of the loss function that is to be optimized
# metrics : What is monitored to evaluate the net
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=30)
