import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double sigmoid_derivative(double x) {
        return x * (1 - x);
    }

    public static void main(String[] args) {
        List<List<Double>> training_inputs = new ArrayList<>();
        training_inputs.add(List.of(0.0, 0.0));
        training_inputs.add(List.of(0.0, 1.0));
        training_inputs.add(List.of(1.0, 0.0));
        training_inputs.add(List.of(1.0, 1.0));

        List<List<Double>> training_outputs = new ArrayList<>();
        training_outputs.add(List.of(0.0));
        training_outputs.add(List.of(1.0));
        training_outputs.add(List.of(1.0));
        training_outputs.add(List.of(0.0));

        int num_inputs = 2;
        int num_hidden = 2;
        int num_outputs = 1;

        Random random = new Random();
        List<List<Double>> hidden_weights = new ArrayList<>();
        List<Double> hidden_biases = new ArrayList<>();
        List<List<Double>> output_weights = new ArrayList<>();
        List<Double> output_biases = new ArrayList<>();

        for (int i = 0; i < num_inputs; i++) {
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < num_hidden; j++) {
                row.add(random.nextDouble() * 2 - 1);
            }
            hidden_weights.add(row);
        }

        for (int i = 0; i < num_hidden; i++) {
            hidden_biases.add(random.nextDouble() * 2 - 1);
            List<Double> row = new ArrayList<>();
            for (int j = 0; j < num_outputs; j++) {
                row.add(random.nextDouble() * 2 - 1);
            }
            output_weights.add(row);
        }

        for (int i = 0; i < num_outputs; i++) {
            output_biases.add(random.nextDouble() * 2 - 1);
        }

        int num_epochs = 10000;
        double learning_rate = 0.1;

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (int i = 0; i < training_inputs.size(); ++i) {
                List<Double> hidden_layer = new ArrayList<>();
                for (int j = 0; j < num_hidden; ++j) {
                    double weighted_sum = 0;
                    for (int k = 0; k < num_inputs; ++k) {
                        weighted_sum += training_inputs.get(i).get(k) * hidden_weights.get(k).get(j);
                    }
                    hidden_layer.add(sigmoid(weighted_sum + hidden_biases.get(j)));
                }

                List<Double> output_layer = new ArrayList<>();
                for (int j = 0; j < num_outputs; ++j) {
                    double weighted_sum = 0;
                    for (int k = 0; k < num_hidden; ++k) {
                        weighted_sum += hidden_layer.get(k) * output_weights.get(k).get(j);
                    }
                    output_layer.add(sigmoid(weighted_sum + output_biases.get(j)));
                }

                List<Double> output_errors = new ArrayList<>();
                for (int j = 0; j < num_outputs; ++j) {
                    output_errors.add((training_outputs.get(i).get(j) - output_layer.get(j)) * sigmoid_derivative(output_layer.get(j)));
                }

                List<Double> hidden_errors = new ArrayList<>();
                for (int j = 0; j < num_hidden; ++j) {
                    double error_sum = 0;
                    for (int k = 0; k < num_outputs; ++k) {
                        error_sum += output_errors.get(k) * output_weights.get(j).get(k);
                    }
                    hidden_errors.add(error_sum * sigmoid_derivative(hidden_layer.get(j)));
                }

                for (int j = 0; j < num_outputs; ++j) {
                    for (int k = 0; k < num_hidden; ++k) {
                        output_weights.get(k).set(j, output_weights.get(k).get(j) + learning_rate * output_errors.get(j) * hidden_layer.get(k));
                    }
                    output_biases.set(j, output_biases.get(j) + learning_rate * output_errors.get(j));
                }

                for (int j = 0; j < num_hidden; ++j) {
                    for (int k = 0; k < num_inputs; ++k) {
                        hidden_weights.get(k).set(j, hidden_weights.get(k).get(j) + learning_rate * hidden_errors.get(j) * training_inputs.get(i).get(k));
                    }
                    hidden_biases.set(j, hidden_biases.get(j) + learning_rate * hidden_errors.get(j));
                }
            }
        }

        for (int i = 0; i < training_inputs.size(); ++i) {
            List<Double> hidden_layer = new ArrayList<>();
            for (int j = 0; j < num_hidden; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < num_inputs; ++k) {
                    weighted_sum += training_inputs.get(i).get(k) * hidden_weights.get(k).get(j);
                }
                hidden_layer.add(sigmoid(weighted_sum + hidden_biases.get(j)));
            }

            List<Double> output_layer = new ArrayList<>();
            for (int j = 0; j < num_outputs; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < num_hidden; ++k) {
                    weighted_sum += hidden_layer.get(k) * output_weights.get(k).get(j);
                }
                output_layer.add(sigmoid(weighted_sum + output_biases.get(j)));
            }

            System.out.printf("Input: %.0f %.0f, Output: %.4f%n",
                    training_inputs.get(i).get(0),
                    training_inputs.get(i).get(1),
                    output_layer.get(0));
        }
    }
}
