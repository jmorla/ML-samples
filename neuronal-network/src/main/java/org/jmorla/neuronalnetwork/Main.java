package org.jmorla.neuronalnetwork;

import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleUnaryOperator;

import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.XYSeries.XYSeriesRenderStyle;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class Main {

    private static final Random r = new Random();
    // private static final DoubleUnaryOperator cos = (x) -> Math.cos(x) * Math.sin(x) * Math.cos(x + 1);
    private static final DoubleUnaryOperator cos = (x) -> Math.cos(x);
    public static void main(String[] args) {
        NeuronalNetwork ann = NeuronalNetwork.builder()
                .learningRate(0.001)
                .addLayer(80, 1, ActivationFunction.SIGMOID)
                .addLayer(80, 80, ActivationFunction.SIGMOID)
                .addLayer(80, 80, ActivationFunction.SIGMOID)
                .addLayer(80, 80, ActivationFunction.SIGMOID)
                .addLayer(80, 80, ActivationFunction.SIGMOID)
                .addLayer(1, 80, ActivationFunction.IDENTITY)
                .build();

        // Define the number of points
        int numPoints = 200;

        // Create arrays to hold the x and y values for cos(x) and sin(x)
        double[] xData = new double[numPoints];
        double[] yCosData = new double[numPoints];
        double[] modelData = new double[numPoints];

        // Generate the data
        for (int i = 0; i < numPoints; i++) {
            double x = 20.0 * i / (numPoints - 1); // Values from 0 to 10
            xData[i] = x;
            yCosData[i] = cos.applyAsDouble(x) ;//+ r.nextDouble(-0.1, 0.1);
            modelData[i] = ann.predict(new double[] { x })[0];
        }

        // Create a chart
        XYChart chart = new XYChartBuilder().width(800).height(600).title("Cosine & prediction Chart").xAxisTitle("X")
                .yAxisTitle("Y").build();

        // Add the cosine scatter data to the chart
        chart.addSeries("cos(x)", xData, yCosData).setXYSeriesRenderStyle(XYSeriesRenderStyle.Scatter);

        // Add the sine line data to the chart
        var serie = chart.addSeries("g(x)", xData, modelData);
        serie.setMarker(SeriesMarkers.NONE);
        serie.setXYSeriesRenderStyle(XYSeriesRenderStyle.Line);
        serie.setLineWidth(4f);

        chart.getStyler().setYAxisMin(-1.0); // Minimum y-axis value
        chart.getStyler().setYAxisMax( 1.0);  // Maximum y-axis value

        chart.getStyler().setXAxisMax(15.0);
        chart.getStyler().setXAxisMin(0.0);

        var wrapper = new SwingWrapper<>(chart);
        trainModel(ann, numPoints, chart, () -> wrapper.repaintChart());
        // Show the chart
        wrapper.displayChart();

    }

    private static void trainModel(NeuronalNetwork ann, int numPoints,  XYChart chart, Runnable callback) {
        var executor = Executors.newSingleThreadScheduledExecutor();
        executor.schedule(() -> {
            while(true) {
                for(int i = 0; i < 400; i ++) {
                    var x = r.nextDouble(0, 15);
                    ann.fit(new double[] { x }, new double[] { cos.applyAsDouble(x) });
                }
    
                double[] xData = new double[numPoints];
                double[] modelData = new double[numPoints];
                for (int i = 0; i < numPoints; i++) {
                    double x1 = 15.0 * i / (numPoints - 1);
                    xData[i] = x1;
                    modelData[i] = ann.predict(new double[] { x1 })[0];
                }
                chart.updateXYSeries("g(x)", xData, modelData, null);
                callback.run();
            }
        }, 2, TimeUnit.SECONDS);

    }
}
