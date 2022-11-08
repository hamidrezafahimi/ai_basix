
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

# **Defining custom callback**

# We are to create a callback. To do so, create a class like below. Note that it must inherit
# from keras.callbacks.Callback class
# In a callback class, to execute an action on a special event, a member function with the same
# name as the event - with same inputs as its main format - must be defined.
# We are to program a callback which stops the training procedure as a specific metric (here:
# accuracy) arives to a particular value
# This code is going to be programmed in 2 versions:
# 
# - (TAG-1) - simpler - idea 1: stop training after the first time that accuracy starts decreasing
# - (TAG-2) - more developed - stop training after the first time that validation loss starts increases
#             for a number (patience) of epochs:


# - (TAG-1):

class EarlyStoppingCallback(keras.callbacks.Callback):

    def __init__(self):
        super(EarlyStoppingCallback, self).__init__()
        self.prev_accuracy = -np.Inf

    # In the following,
    # 'epoch' : The number of the ended epoch
    # 'logs' : A dictionary containing the values of all declared metrics. 'loss' always exists in it.

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get("accuracy")
        
        # In the above, "accuracy" or "loss" and ... must be the metrics string declared in 
        # model.compile() entry
        # The following algorithm: As soon as the accuracy starts to decrease, stop the 'fit' funtion
        # "model" attribute is set on the parent class when the callback in introduced to the model
        # "stop_training" stops training immediately when true
        
        if np.less(current_accuracy, self.prev_accuracy):
            self.model.stop_training = True
        else:
            self.prev_accuracy = current_accuracy


# - (TAG-2):

class EarlyStoppingCallback(keras.callbacks.Callback):

    def __init__(self, patience=0):

        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience

        # "patience" : number of epochs that we would wait for the loss to gets back decreasing, 
        # and then if not, stop training

    # The following function will be called in the initialization of each call to function `fit()`


    def on_train_begin(self, logs=None):

        self.best = np.Inf
        self.wait = 0
        self.stopped_epoch = 0

        # In the above,
        # "best" : The best seen loss 
        # "wait" : Number of epochs that we are waiting for the loss to get back decreasing
        # "stopped_epoch" : The number of epoch that last stop has happened (through different 
        # calls to `fit()` function)


    def on_epoch_end(self, epoch, logs=None):

        # The following algorithm: If the val_loss metric (loss on validation dataset) decreases, save
        # the weights of the network and reset the best loss. Otherwise, wait until patience parameter
        # value is reached. If val_loss is not decreased for a number of epochs as the patience 
        # parameter states, stop training while resetting the weights on the best case values.

        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best):
            self.best = current_loss
            self.wait = 0

            self.best_weights = self.model.get_weights()
            
            # (In the above,) Save the weights in the best case according to loss value
            # "get_weights()" returns all the network weights
        
        else:
            self.wait += 1
            print("\nwait mode, step: %d\n" % self.wait)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

                # "set_weights()" sets network (specified by .model) weights on input
    

    def on_train_end(self, logs=None):

        if self.stopped_epoch > 0:
            print("epoch: %d: early stopping" % self.stopped_epoch)
