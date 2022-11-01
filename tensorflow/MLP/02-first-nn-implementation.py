
import os
from re import X
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import tensorflow as tf
# Note: tensorflow 2.x is required! Be careful to add the correct package
from tensorflow import keras

import numpy as np
from matplotlib import pyplot as plt

"""
KEYWORDS:

logits (concept)
softmax (activation function)
sparse_categorical_crossentropy (loss function)

test dataset (concept)
training dataset (concept)
validation dataset (concept)

evaluate (function)
predict (function)

summary (function)
keras.utils.plot_model (function)
"""

# **1 - Dataset Preparation**

# Prepare the dataset:

fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# Preprocess data:
X_train = X_train/255.0
X_test = X_test/255.0

# Show samples of dataset:
plt.figure(figsize=(10,10))
for i in range (25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(Y_train[i])
plt.show()
# (TAG-1) Looking to the pictures, the 10 labels are scalar values between 0 to 9. There are no 10-element 
# vectors and probibility distributions.

# **2 - Model Design**

# This is one of the ways to create the model object:
model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=(28,28)),
                                 keras.layers.Dense(units=128, activation="relu"),
                                 keras.layers.Dense(10)

                                 # (TAG-2) The output of this net is not a probability theory: It is 
                                 # the result of multiplying weights on input - so its sum is not 
                                 # necessarily equal to 1. This type of outputs are called 'logits' 
                                 # (the net outputs before a softmax layer)
                                 # (TAG-3) ... Defining a 'softmax' activation function to convert the 
                                 # output to probibility distribution

                                #  keras.layers.Dense(10, activation="softmax")
])


# Creating models from models:
# put a (probably trained) model inside another model:

proba_model = keras.models.Sequential([
                                       model,
                                       keras.layers.Softmax()                                                                    
                                      ]    
)

# (TAG-3) The above model doesn't need to be compiled. Because a 'softmax' layer only normalizes the 
# output; It doesn't change the weights (because it has no weights!)

# To check properties of the designed model:

model.summary()

# To get visual data about the designed model:

keras.utils.plot_model(model)


# **3 - Model Compile**

# 1st method to compile model object:
# In this method, all the input objects are defined with their default parameters

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Te function 'sparse_categorical_crossentropy' gets two probability distribution inputs, and 
# calculates the similarity between them. 
# Two given inputs to this loss function: 
# 1- labels: the desired values to be generated for each input. 
# 2- The output of the net. 
# These two are to be compared. Looking to the (TAG-1) and (TAG-2), none of the inputs are 
# probability distributions. 
# (About 1) This function makes probability distribution from a non-probability-distribution label 
# input This way: Knowing the number of the classes, it converts a label '9' to a vector
# '[0,0,0,0,0,0,0,0,0,1]' (sum = 1, numel = 10)
# (About 2) It is not so. There are 2 ways to fix this: 1- (TAG-3) ... 2- The following (-> TAG-4):

# 2nd method to compile the model object:
# In this method, the input objects are defined manually with desired parameters

# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),

              # In the above, the learning rate '0.1' is too big and '0.001' is too small. It is 
              # so important to know how to set such parameters optimally
              # (TAG-4) In the following, setting the 'from_logits=True' warns the loss function 
              # object to convert the inputs to probibility distribution

            #   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #   metrics=["accuracy"])


# Finally, this is the good way to solve the current problem:

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])


# **4 - Model Training**

# Training with no validation dataset ('fit' does both backward- and forward-pass):

# model.fit(X_train, Y_train, epochs=10)

# Training with a validation dataset
# A 'validation dataset' is a dataset which is used to evaluate the training performance during the 
# 'fit' procedure, but its results are not used to update the weights
# The term 'validation_split = 0.1' means that 10% of the 'training dataset' (the main dataset which 
# the net is trained with) is to be used as validation dataset

model.fit(X_train, Y_train, epochs=30, validation_split=0.1)

# NOTE: Calling the 'fit' function sequentially results the nexts calls continue on the output net 
# of the previous calls


# **5 - Model Evaluation**

# Evaluation of the trined model on 'test dataset'
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)


# **6 - Using the Model**

# First, theinput data must be prepared:

X = X_test[0]       # The first picture of test dataset
plt.imshow(X)

Y = Y_test[0]       # The first label of tst dataset
print(Y)

# Use the trained network (give input and get output):
# I use the above-defined 'proba_model' to use the trained net. Because it ahs a softmax layer in 
# its output and so its results are understandable
# predict : performs only the forward-pass on any given input
# 1- To get the output on a single input (Note that the input shape is important and can be source
# of some errors; Thus, the 'np.array()' function is used):

predicted_distr = proba_model.predict(np.array([X]))
print(predicted_distr)

predicted_class = np.argmax(predicted_distr)                                    # a single input output
print(predicted_class)