
from tensorflow import keras
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


"""
KEYWORDS:

Creating Callback (task)

"""

# Defining custom model

# We are to create a callback. To do so, create a class like below. Note that it must inherit
# from keras.callbacks.Callback class


class EarlyStoppingCallback(keras.callbacks.Callback):

    # def __init__(self, patience=0):
    def __init__(self):
        super(EarlyStoppingCallback, self).__init__()
        # self.patience = patience
        # ".patience" : number of epochs that we would wait for the loss to gets back decreasing, and then if not, stop training

    # In a callback class, to execute an action on a special event, a member function with the same
    # name as the event - with same inputs as its main format - must be defined.

    # def on_train_begin(self, logs=None):
    #     self.best = np.Inf
    #     # ".best" is the prev_accuracy (which its value is -np.Inf in case of idea 1)
    #     self.wait = 0
    #     # ".wait" : number of epochs that we are waiting for the loss to gets back decreasing
    #     self.stopped_epoch = 0

    # We are to program a callback which stops the training procedure as a specific metric (here:
    # accuracy) arives to a particular value

    # idea 1: stop training after the first time that accuracy starts decreasing:

    # In the following,
    # 'epoch' : the number of the epoch
    # 'logs' : A dictionary containing the values of all declared metrics. 'loss' always exists in it.

    def on_epoch_end(self, epoch, logs=None):
        
        current_accuracy = logs.get("accuracy")
        
        # In the above, "accuracy" or "loss" and ... must be the metrics string declared in 
        # model.compile() entry
        
        # The following algorithm: As soon as the accuracy starts to decrease, stop the 'fit' funtion
        
        if np.less(current_accuracy, self.prev_accuracy):
            self.model.stop_training = True
            
            # "model" attribute is set on the parent class when the callback in introduced to the model
            # "stop_training" stops training immediately when true
        else:
            self.prev_accuracy = current_accuracy

    # idea 2: stop training after the first time that validation loss starts increases for a number (patience) of epochs:

    # def on_epoch_end(self, epoch, logs=None):
    #     current_loss = logs.get("val_loss")
    #     if np.less(current_loss, self.best):
    #         self.best = current_loss
    #         self.wait = 0
    #         # save the weights in case that loss is decreasing
    #         self.best_weights = self.model.get_weights()
    #         # ".get_weights()" returns all the network weights
    #     else:
    #         self.wait += 1
    #         print("\nwait mode, step: %d\n" % self.wait)
    #         if self.wait >= self.patience:
    #             self.stopped_epoch = epoch
    #             self.model.stop_training = True
    #             self.model.set_weights(self.best_weights)
    #             # ".set_weights()" sets network (specified by .model) weights on input

    # def on_train_end(self, logs=None):
    #     if self.stopped_epoch > 0:
    #         print("epoch: %d: early stopping" % self.stopped_epoch)
