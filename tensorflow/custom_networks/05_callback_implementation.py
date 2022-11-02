
import numpy as np
from tensorflow import keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""
KEYWORDS:

"""

# **1 - Dataset Preparation**

fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0


# **2 - Model Design**

inp = keras.Input(shape=(28, 28))
mini_models = []
for i in range(5):
    
    model = keras.models.Sequential([
        
        keras.layers.Flatten(input_shape=(28, 28), name="input_layer"),
        keras.layers.Dense(units=128, activation="relu", name="hidden_layer"),
        keras.layers.Dense(10, name="output_layer")
    ])
    
    mini_models.append(
        
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

# First, theinput data must be prepared:

X = X_test[0]       # The first picture of test dataset
plt.imshow(X)

Y = Y_test[0]       # The first label of tst dataset
print(Y)

# Use the trained network (give input and get output):

predicted_distr = model.predict(np.array([X]))
print(predicted_distr)

# a single input output
predicted_class = np.argmax(predicted_distr)
print(predicted_class)
