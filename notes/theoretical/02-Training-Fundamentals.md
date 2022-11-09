




## Network Architecture Modification

### Bad Gradients in Network Weights

The change in network weights must be reasonable. Unchanges weights or diverged weights both are not desired. Looking at the changes of network weights through sequential layers of a network, to bad things may happen:

#### A Vanishing Gradient:

A vanishing gradient is a case in which the change in network weights decreases while going backward into the input of network. Mathemattically, the reason for this phenomena is an exponential increase in the order of the errors propagated as result of the back propagation formula.

#### An Exploding Gradient:

An exploding gradient is a case in which the change in network weights increases while going backward into the input of network. Such that the weights of the network in the input layers become totally stochastic. A probable reason for that, may be the high number produced by the derivative a the activation function (as it is present in back propagation formula)


### Overfitting and Underfitting

#### Overfitting:
Is the case that loss decreases on training dataset while remains constant or increases on validation dataset during the training epochs.


### Fixing Utilities

There are a bunch of mathematical solutions as utilities provided to fix the network architecture problems. Each must be used in a case in which it's proper.

#### Regularizers:
Consider a single weight in a layer with high value (higher than the other weights of the same network), given a high value as input. This particular weight leads to anormal effects and shocks in network behavior. A regularizer is used to limit the network weights so that no such problem happens.


## Metrics

Metrics are the monitored parameters during, or after a network training procedure.

### Loss


### Accuracy


### Convergence Speed

The convergence speed is the number of epochs in which the convergende has happened. Note that it has no connection to time. That's because the calculation time may be decreased with use of better processors or etc. Whereas a criteria which is independent of hardware is needed to evaluate the performance of a particular network architecture.

### Confusion Matrix

One of the benefits of a confusion matrix is that it tells us about the network performance about the classes with the highest occurrence rate - which is much useful in analyzing the network's behavior.