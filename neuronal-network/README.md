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

![cos](https://github.com/jmorla/ML-samples/assets/31217110/6fa6ab82-7308-41ea-a459-8cd475f1e4d7)

![cos2](https://github.com/jmorla/ML-samples/assets/31217110/008f1287-625d-4359-9450-629cfc3f25d8)


