#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple NN based on XOR function

// Setting weights
double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

// Sigmoid function
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

// Derivative of sigmoid
double dsigmoid(double x) { return x * (1 - x); }

// Shuffle function to help neural net learn faster
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numOutputs 1
#define HiddenNodes 2
#define TrainingSet 4

int main(void) {
    const double lr = 0.1f;
    
    double hiddenLayer[HiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[HiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][HiddenNodes];
    double outputWeights[HiddenNodes][numOutputs];

    double training_inputs[TrainingSet][numInputs] = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };

    double training_outputs[TrainingSet][numOutputs] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    // Initialize weights and biases
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < HiddenNodes; j++) {
            hiddenWeights[i][j] = init_weights(); // Initializing weights to input layers
        }
    }
    for (int i = 0; i < HiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weights(); // Initializing output weights as random weights
        }
    }
    for (int i = 0; i < HiddenNodes; i++) {
        hiddenLayerBias[i] = init_weights(); // Initializing hidden layer biases
    }
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weights(); // Initializing output biases
    }

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 10000;

    // Open a file for writing results
    FILE *outputFile = fopen("nn_output.txt", "w");
    if (!outputFile) {
        fprintf(stderr, "Could not open file for writing.\n");
        return 1;
    }

    // Training the neural network for n number of epochs
    for (int epochs = 0; epochs < numberOfEpochs; epochs++) {
        shuffle(trainingSetOrder, TrainingSet);
        
        for (int x = 0; x < TrainingSet; x++) {
            int i = trainingSetOrder[x];

            // Forward pass
            // Computing hidden layer activation
            for (int j = 0; j < HiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            // Computing output layer activation
            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < HiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            // Print output and write to file
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Input: (%g, %g)    Output: %g    Expected Output: %g\n",
                     training_inputs[i][0], training_inputs[i][1],
                     outputLayer[0], training_outputs[i][0]);
            fputs(buffer, outputFile);

            // Backpropagation
            // Change in output weights
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dsigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights
            double deltaHidden[HiddenNodes];
            for (int j = 0; j < HiddenNodes; j++) {
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k]; // Fixed indexing
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
            }

            // Applying the changes in output weights
            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < HiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            // Applying changes in hidden weights
            for (int j = 0; j < HiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr; // Fixed indexing
                }
            }
        }
    }

    // Write final weights and biases to the output file
    fputs("Final Hidden Weights:\n", outputFile);
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < HiddenNodes; j++) {
            snprintf(buffer, sizeof(buffer), "%g ", hiddenWeights[i][j]);
            fputs(buffer, outputFile);
        }
        fputs("\n", outputFile);
    }

    fputs("Final Hidden Biases:\n", outputFile);
    for (int i = 0; i < HiddenNodes; i++) {
        snprintf(buffer, sizeof(buffer), "%g ", hiddenLayerBias[i]);
        fputs(buffer, outputFile);
    }
    fputs("\n", outputFile);

    fputs("Final Output Weights:\n", outputFile);
    for (int i = 0; i < HiddenNodes; i++) {
        for (int j = 0; j < numOutputs; j++) {
            snprintf(buffer, sizeof(buffer), "%g ", outputWeights[i][j]);
            fputs(buffer, outputFile);
        }
        fputs("\n", outputFile);
    }

    fputs("Final Output Biases:\n", outputFile);
    for (int i = 0; i < numOutputs; i++) {
        snprintf(buffer, sizeof(buffer), "%g ", outputLayerBias[i]);
        fputs(buffer, outputFile);
    }
    fputs("\n", outputFile);

    // Close the output file
    fclose(outputFile);
    return 0;
}
