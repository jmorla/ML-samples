package com.jmorla;

import java.util.Arrays;
import java.util.Random;

/**
 * Represents a layer in a neural network.
 * 
 * <p>This class handles the initialization of the layer with specified number of neurons and inputs,
 * as well as the forward pass operation using the specified activation function.</p>
 * 
 * @author Jorge L. Morla
 * @version 1.0
 */
public class Layer {

    private final ActivationFunction activationFunction;
    private final double[][] weights;
    private final double[] biases;
    private double[] currentOutput;
    private double[] input;

    /**
     * Constructs a layer with the specified number of neurons, inputs, and activation function.
     * 
     * @param numberOfNeurons the number of neurons in the layer
     * @param numberOfInputs the number of inputs to each neuron (excluding the bias)
     * @param activation the activation function for the layer
     */
    public Layer (int numberOfNeurons, int numberOfInputs, ActivationFunction activation) {
        // weights = new double[numberOfNeurons][numberOfInputs + 1];
        weights = new double[numberOfNeurons][numberOfInputs];
        biases = new double[numberOfNeurons];
        this.activationFunction = activation;
        initializateWeights();
    }

    private void initializateWeights() {
        Random random = new Random();
        for(int i = 0; i < weights.length; i ++) {
            for(int j = 0; j < weights[i].length; j++)  {
                weights[i][j] = random.nextGaussian();
            }
            biases[i] = random.nextGaussian();
        }
    }

    /**
     * Performs the forward pass operation on the input data.
     * 
     * <p>This method applies the weights to the input, including the bias, and then applies the activation function.</p>
     * 
     * @param input the input data to the layer
     * @return the output of the layer after applying the weights and activation function
     */
    public double[] passFoward(double[] input) {
        this.input = input;
        var weighted = MatrixUtils.multiply(weights, input);
        var z = MatrixUtils.sum(weighted, biases);
        currentOutput = MatrixUtils.applyFunction(z, activationFunction.function);
        return currentOutput;
    }

    /**
     * Performs the backpropagation operation to update weights based on delta.
     * 
     * @param delta the gradient of the loss with respect to the output of this layer
     * @param learningRate the learning rate for weight updates
     * @return the gradient of the loss with respect to the input of this layer
     */
    public double[] backpropagate(double[] delta, double learningRate) {
        var dActivation = MatrixUtils.applyFunction(currentOutput, 
        activationFunction.derivative);

        delta = MatrixUtils.multiply(delta, dActivation);

        for(int i = 0; i < weights.length; i ++) {
            for(int j = 0; j < weights[i].length; j++) {
                weights[i][j] -= learningRate * delta[i] * input[j];
            }
            biases[i] -= learningRate * delta[i];
        }

        return delta;
    }

    public double[][] getWeights() {
        return this.weights;
    }

    @Override
    public String toString() {
        return "Layer [ weights=" + Arrays.deepToString(weights) + ", biases="
                + Arrays.toString(biases) + ", currentOutput=" + Arrays.toString(currentOutput) + ", input="
                + Arrays.toString(input) + "]";
    }
}
