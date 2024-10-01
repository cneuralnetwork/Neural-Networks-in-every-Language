#include <iostream>
#include <vector>
#include <cmath>

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

int main() {
    // Training data (XOR gate)
    std::vector<std::vector<double>> training_inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> training_outputs = {
        {0},
        {1},
        {1},
        {0}
    };

    // Network architecture (2 input, 2 hidden, 1 output)
    int num_inputs = 2;
    int num_hidden = 2;
    int num_outputs = 1;

    // Weights and biases (randomly initialized)
    std::vector<std::vector<double>> hidden_weights(num_inputs, std::vector<double>(num_hidden));
    std::vector<double> hidden_biases(num_hidden);
    std::vector<std::vector<double>> output_weights(num_hidden, std::vector<double>(num_outputs));
    std::vector<double> output_biases(num_outputs);

    // Training loop
    int num_epochs = 10000;
    double learning_rate = 0.1;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int i = 0; i < training_inputs.size(); ++i) {
            // Forward pass
            std::vector<double> hidden_layer(num_hidden);
            for (int j = 0; j < num_hidden; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < num_inputs; ++k) {
                    weighted_sum += training_inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = sigmoid(weighted_sum + hidden_biases[j]);
            }

            std::vector<double> output_layer(num_outputs);
            for (int j = 0; j < num_outputs; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < num_hidden; ++k) {
                    weighted_sum += hidden_layer[k] * output_weights[k][j];
                }
                output_layer[j] = sigmoid(weighted_sum + output_biases[j]);
            }

            // Backpropagation
            std::vector<double> output_errors(num_outputs);
            for (int j = 0; j < num_outputs; ++j) {
                output_errors[j] = (training_outputs[i][j] - output_layer[j]) * sigmoid_derivative(output_layer[j]);
            }

            std::vector<double> hidden_errors(num_hidden);
            for (int j = 0; j < num_hidden; ++j) {
                double error_sum = 0;
                for (int k = 0; k < num_outputs; ++k) {
                    error_sum += output_errors[k] * output_weights[j][k];
                }
                hidden_errors[j] = error_sum * sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and biases
            for (int j = 0; j < num_outputs; ++j) {
                for (int k = 0; k < num_hidden; ++k) {
                    output_weights[k][j] += learning_rate * output_errors[j] * hidden_layer[k];
                }
                output_biases[j] += learning_rate * output_errors[j];
            }

            for (int j = 0; j < num_hidden; ++j) {
                for (int k = 0; k < num_inputs; ++k) {
                    hidden_weights[k][j] += learning_rate * hidden_errors[j] * training_inputs[i][k];
                }
                hidden_biases[j] += learning_rate * hidden_errors[j];
            }
        }
    }

    // Test the network
    for (int i = 0; i < training_inputs.size(); ++i) {
        std::vector<double> output_layer(num_outputs);
        // ... (forward pass to get output_layer)
        std::cout << "Input: " << training_inputs[i][0] << " " << training_inputs[i][1] 
                  << ", Output: " << output_layer[0] << std::endl;
    }

    return 0;
}

