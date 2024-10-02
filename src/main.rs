use rand::Rng;
use std::f64;
/*
 * Do cargo run to run the program
 * On windows, you need wsl to run this
 */

// Activation function (sigmoid)
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}

// Derivative of sigmoid
fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn main() {
    // Training data (XOR gate)
    let training_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let training_outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Network architecture (2 input, 2 hidden, 1 output)
    let num_inputs = 2;
    let num_hidden = 2;
    let num_outputs = 1;

    // Initialize weights and biases randomly
    let mut rng = rand::thread_rng();
    let mut hidden_weights = vec![vec![0.0; num_hidden]; num_inputs];
    let mut hidden_biases = vec![0.0; num_hidden];
    let mut output_weights = vec![vec![0.0; num_outputs]; num_hidden];
    let mut output_biases = vec![0.0; num_outputs];

    for i in 0..num_inputs {
        for j in 0..num_hidden {
            hidden_weights[i][j] = rng.gen_range(-1.0..1.0);
        }
    }
    for i in 0..num_hidden {
        hidden_biases[i] = rng.gen_range(-1.0..1.0);
        for j in 0..num_outputs {
            output_weights[i][j] = rng.gen_range(-1.0..1.0);
        }
    }
    for i in 0..num_outputs {
        output_biases[i] = rng.gen_range(-1.0..1.0);
    }

    // Training loop
    let num_epochs = 10000;
    let learning_rate = 0.1;
    for _ in 0..num_epochs {
        for i in 0..training_inputs.len() {
            // Forward pass
            let mut hidden_layer = vec![0.0; num_hidden];
            for j in 0..num_hidden {
                let mut weighted_sum = 0.0;
                for k in 0..num_inputs {
                    weighted_sum += training_inputs[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = sigmoid(weighted_sum + hidden_biases[j]);
            }

            let mut output_layer = vec![0.0; num_outputs];
            for j in 0..num_outputs {
                let mut weighted_sum = 0.0;
                for k in 0..num_hidden {
                    weighted_sum += hidden_layer[k] * output_weights[k][j];
                }
                output_layer[j] = sigmoid(weighted_sum + output_biases[j]);
            }

            // Backpropagation
            let mut output_errors = vec![0.0; num_outputs];
            for j in 0..num_outputs {
                output_errors[j] = (training_outputs[i][j] - output_layer[j]) * sigmoid_derivative(output_layer[j]);
            }

            let mut hidden_errors = vec![0.0; num_hidden];
            for j in 0..num_hidden {
                let mut error_sum = 0.0;
                for k in 0..num_outputs {
                    error_sum += output_errors[k] * output_weights[j][k];
                }
                hidden_errors[j] = error_sum * sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and biases
            for j in 0..num_outputs {
                for k in 0..num_hidden {
                    output_weights[k][j] += learning_rate * output_errors[j] * hidden_layer[k];
                }
                output_biases[j] += learning_rate * output_errors[j];
            }

            for j in 0..num_hidden {
                for k in 0..num_inputs {
                    hidden_weights[k][j] += learning_rate * hidden_errors[j] * training_inputs[i][k];
                }
                hidden_biases[j] += learning_rate * hidden_errors[j];
            }
        }
    }

    // Test the network
    println!("Testing the neural network:");
    for i in 0..training_inputs.len() {
        let mut hidden_layer = vec![0.0; num_hidden];
        for j in 0..num_hidden {
            let mut weighted_sum = 0.0;
            for k in 0..num_inputs {
                weighted_sum += training_inputs[i][k] * hidden_weights[k][j];
            }
            hidden_layer[j] = sigmoid(weighted_sum + hidden_biases[j]);
        }

        let mut output_layer = vec![0.0; num_outputs];
        for j in 0..num_outputs {
            let mut weighted_sum = 0.0;
            for k in 0..num_hidden {
                weighted_sum += hidden_layer[k] * output_weights[k][j];
            }
            output_layer[j] = sigmoid(weighted_sum + output_biases[j]);
        }

        println!("Input: {:?}, Output: {:.4}", training_inputs[i], output_layer[0]);
    }
}