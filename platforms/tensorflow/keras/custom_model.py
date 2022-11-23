
import custom_layer
from tensorflow import keras
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


"""
KEYWORDS:

Defining Custom Models (task)

"""

# Defining custom model

# We are to create a cusom model. To do so, create a class like below. Note that it must inherit
# from keras.models.Model class


class CustomModel(keras.models.Model):

    # For custom models, the layers (graph nodes) are defined in the constructor
    
    def __init__(self, units):
        super(CustomModel, self).__init__()
        self.l1 = custom_layer.CustomSum(units)
        self.l2 = custom_layer.CustomSum(units)
        self.l3 = custom_layer.CustomSum(units)
        self.l4 = keras.layers.Dense(units, activation="relu")

    # For custom models, the inputs (graph connections) are defined in the 'call' function

    def call(self, inputs):
        
        x1 = self.l1(inputs)
        x2 = self.l2(inputs)
        x3 = self.l3(x1)
        x3 = self.l3(x2)
        output = self.l4(x3)
        return output


# Such user-defined model, can be trained and used on a similar structure like other for models:

# model = CustomModel(2)
# model.compile(optimizer, loss, metrices)
# model.fit()
# model.evaluate()
# model.predict()