# Artificial Neuronal Network

This Maven module provides functionality for a neural network.

### Examples

given the following ANN configuration

```java
NeuronalNetwork ann = NeuronalNetwork.builder()
    .learningRate(0.001)
    .addLayer(80, 1, ActivationFunction.SIGMOID)
    .addLayer(80, 80, ActivationFunction.SIGMOID)
    .addLayer(80, 80, ActivationFunction.SIGMOID)
    .addLayer(80, 80, ActivationFunction.SIGMOID)
    .addLayer(80, 80, ActivationFunction.SIGMOID)
    .addLayer(1, 80, ActivationFunction.IDENTITY)
    .build();
```

We can use it to approximate `cosine(x)` function for values from 0 to 15
