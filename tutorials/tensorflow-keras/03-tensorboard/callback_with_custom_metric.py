
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
import tensorflow as tf
import numpy as np

"""
KEYWORDS:

Creating Callback (task)

set_weights (function)
get_weights (function)
"""

# **Defining a custom metric to be monitored, in a callback**

# In addition to typical metrics, user-defined parameters from different types (scalar, image, 
# distribution, or histogram) can be monitored in tensorboard.
# We are to define a custom metric in our callback class so that tensorboard can monitor it. To do so,
# we select the parameter `wait` in the following class.


class EarlyStoppingCallback(keras.callbacks.Callback):

    def __init__(self, patience=0):

        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience

        # To a specific data in the `logs` folder in which the tensoroard is able to read, the 
        # following object must be created:

        self.writer = tf.summary.create_file_writer("/home/hamid/w/nn_samples/logs")


    def on_train_begin(self, logs=None):

        self.best = np.Inf
        self.wait = 0
        self.stopped_epoch = 0


    def on_epoch_end(self, epoch, logs=None):

        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best):
            self.best = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            print("\nwait mode, step: %d\n" % self.wait)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

        # The method to write a specific data as mentioned is as follows:

        with self.writer.as_default():
            tf.summary.scalar("wait", self.wait, step=epoch)

            # The above line causes wrting data. It means: 
            # In writer object, create a `summary`` from type `scalar`, name it `wait` (the vertical
            # axis for a monitored scalar) and monitor it through `epoch` numbers (the horizontal axis).
        # The following is to make sure that everything in the wrter object's buffer will be written:

        self.writer.flush()



    def on_train_end(self, logs=None):

        if self.stopped_epoch > 0:
            print("epoch: %d: early stopping" % self.stopped_epoch)
