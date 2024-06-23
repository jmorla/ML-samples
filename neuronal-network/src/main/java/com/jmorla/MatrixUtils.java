package com.jmorla;

import java.util.function.DoubleUnaryOperator;

public class MatrixUtils {

    public static double[] sum(double[] A, double[] B) {
        // Check if both arrays are of the same length
        if (A.length != B.length) {
            throw new IllegalArgumentException("Both arrays must have the same length");
        }

        // Create an array to store the result
        double[] result = new double[A.length];

        // Perform element-wise multiplication
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] + B[i];
        }

        return result;
    }

    public static double[] multiply(double[] A, double[] B) {
        // Check if both arrays are of the same length
        if (A.length != B.length) {
            throw new IllegalArgumentException("Both arrays must have the same length");
        }

        // Create an array to store the result
        double[] result = new double[A.length];

        // Perform element-wise multiplication
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] * B[i];
        }

        return result;
    }

    // Method to multiply two matrices
    public static double[][] multiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;
        double[][] result = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // Method to transpose a matrix
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // Method to apply a function element-wise to a matrix
    public static double[][] applyFunction(double[][] matrix, DoubleUnaryOperator function) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = function.applyAsDouble(matrix[i][j]);
            }
        }
        return result;
    }

    // Method to apply a function element-wise to a vector
    public static double[] applyFunction(double[] vector, DoubleUnaryOperator function) {
        int rows = vector.length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = function.applyAsDouble(vector[i]);
        }
        return result;
    }

    // Method to multiply a matrix with a vector
    public static double[] multiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        if (vector.length != cols) {
            throw new IllegalArgumentException("Matrix columns must match vector length.");
        }
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
}
