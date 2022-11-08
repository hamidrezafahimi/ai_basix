
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tensorflow import keras

import custom_callback

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

combiner = keras.layers.average(mini_models)
output_layer = keras.layers.Softmax()
output_value = output_layer(combiner)
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

# Using a custom callback in the training procedure:

# Two different cases for callback are implemented in the `custom_callback` module:

# (TAG-1):

# es_callback = custom_callback.EarlyStoppingCallback()

# (TAG-2):

es_callback = custom_callback.EarlyStoppingCallback(patience=5)


model.fit(X_train, Y_train, epochs=1000, validation_split=0.1, callbacks=[es_callback])


# **5 - Model Evaluation**

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)


# **6 - Using the Model**

X = X_test[0]       # The first picture of test dataset
Y = Y_test[0]       # The first label of tst dataset

predicted_distr = model.predict(np.array([X]))
predicted_class = np.argmax(predicted_distr)

print(predicted_distr)
print(predicted_class)
