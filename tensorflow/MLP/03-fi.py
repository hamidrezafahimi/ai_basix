
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


# **2 - Model Design**

# Note that you can also put names on layers:

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape=(28,28), name="input_layer"),
                                 keras.layers.Dense(units=128, activation="relu", name="hidden_layer"),
                                 keras.layers.Dense(10, name="output_layer")
                                    ])

proba_model = keras.models.Sequential([
                                       model,
                                       keras.layers.Softmax()                                                                    
                                      ]    
)

# To check properties of the designed model:

model.summary()

# To get visual data about the designed model:

keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

# The above flags state that: show shapes of arrays - show names of layers - show nested nets, 
# if exist

# **3 - Model Compile**

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])


# **4 - Model Training**

# Training with a validation dataset

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

predicted_distr = proba_model.predict(np.array([X]))
print(predicted_distr)

predicted_class = np.argmax(predicted_distr)             # a single input output
print(predicted_class)