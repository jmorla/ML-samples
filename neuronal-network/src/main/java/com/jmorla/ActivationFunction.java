package com.jmorla;

import java.util.function.DoubleUnaryOperator;

public class ActivationFunction {

    public static final ActivationFunction SIGMOID = new ActivationFunction("sigmoid",
        x -> 1 / (1 + Math.exp(-x)),
        x -> x * (1 - x)
    );
    public static final ActivationFunction IDENTITY = new ActivationFunction("identity",
        x -> x, 
        x -> 1.0
    );

    public static final ActivationFunction RELU = new ActivationFunction("relu",
        x -> Math.max(0, x), 
        x -> x > 0 ? 1.0 : 0.0
    );
    
    public final DoubleUnaryOperator function;
    public final DoubleUnaryOperator derivative;

    private ActivationFunction(String name, DoubleUnaryOperator function, DoubleUnaryOperator derivative) {
        this.function = function;
        this.derivative = derivative;
    }
}
