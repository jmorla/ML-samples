package com.jmorla;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

/**
 * @author Jorge L. Morla
 * @since 18-06-2024
 */
public class NeuronalNetwork {

    private final double learningRate;
    private final List<Layer> layers;

    private NeuronalNetwork(double learningRate, List<Layer> layers) {
        this.learningRate = learningRate;
        this.layers = layers;
    }

    public static Builder builder() {
        return new Builder();
    }

    public double[] predict(double[] x) {
        var input = x;
        for (var layer : layers) {
            input = layer.passFoward(input);
        }
        return input;
    }

    public double[] fit(double[] x, double[] y) {
        var py = predict(x); 
        if(y.length != py.length) {
            throw new IllegalArgumentException("array sizes does not match y=[%d] and py=[%d]"
            .formatted(y.length, py.length));
        }
        double[] error = new double[y.length];
        for(var i = 0; i < y.length; i ++) {
            error[i] = py[i] - y[i];
        }

        ListIterator<Layer> iterator = layers.listIterator(layers.size());

        Layer layer; // current layer

        if(!iterator.hasPrevious())
            return null;

        // for the input layer
        layer = iterator.previous();
        var delta = layer.backpropagate(error, learningRate);

        // backprop for hidden layers
        while(iterator.hasPrevious()) {
            var weightsT = MatrixUtils.transpose(layer.getWeights());
            layer = iterator.previous();
            delta = layer.backpropagate(MatrixUtils.multiply(weightsT, delta), learningRate);
        }

        return py;
    }


    @Override
    public String toString() {
        return "NeuronalNetwork [learningRate=" + learningRate + ", layers=" + layers + "]";
    }



    public static final class Builder {
        private double learningRate;
        private List<Layer> layers = new ArrayList<>();

        public Builder learningRate(double rate) {
            this.learningRate = rate;
            return this;
        }

        public Builder addLayer(int numberOfNeurons, int numberOfInputs, ActivationFunction activation) {
            layers.add(new Layer(numberOfNeurons, numberOfInputs, activation));
            return this;
        }

        public NeuronalNetwork build() {
            return new NeuronalNetwork(learningRate, layers);
        }

    }
}
