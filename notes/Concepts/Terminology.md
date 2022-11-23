### Summary 

    - [One-Hot Encoding](#section-id-4)
  





<div id='section-id-4'/>

### One-Hot Encoding

In the output of an NN, there are possible labels for the given input. Assume that these are the labels:

```
[0,1,2,3]
```

In one-hot encoding, there is an output vector with a known shape (same as the shape of the label-space). 

- If the output label is `0`, then the output vector will be:

```
[1,0,0,0]
```

- If the output label is `1`, then the output vector will be:

```
[0,1,0,0]
```

- If the output label is `2`, then the output vector will be:

```
[0,0,1,0]
```

- If the output label is `3`, then the output vector will be:

```
[0,0,0,1]
```

So, presenting one-hot encoded outputs, the shape of the output is as same as shape of the label-space, rather than the defined shape of a single output. 

Note that, when the outputs are presented as probability distribution with the same shape of the label-space, replacing the maximum probabilities with one and the rest with zero is one-hot encoding.
