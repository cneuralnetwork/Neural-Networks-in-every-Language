import 'dart:math';

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

class Matrix {
  int rows, cols;
  late List<List<double>> data;

  Matrix(this.rows, this.cols) {
    data = List.generate(rows, (_) => List<double>.filled(cols, 0.0));
  }

  void randomize(Random rng) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = rng.nextDouble() * 2 - 1;
      }
    }
  }

  Matrix transpose() {
    Matrix result = Matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[j][i] = data[i][j];
      }
    }
    return result;
  }

  Matrix multiply(Matrix other) {
    if (cols != other.rows) {
      throw Exception('Incompatible matrix dimensions');
    }
    Matrix result = Matrix(rows, other.cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        double sum = 0.0;
        for (int k = 0; k < cols; k++) {
          sum += data[i][k] * other.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  void add(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw Exception('Incompatible matrix dimensions');
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] += other.data[i][j];
      }
    }
  }

  void subtract(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw Exception('Incompatible matrix dimensions');
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] -= other.data[i][j];
      }
    }
  }

  void multiplyScalar(double scalar) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] *= scalar;
      }
    }
  }

  void apply(double Function(double) f) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = f(data[i][j]);
      }
    }
  }
}

class NeuralNetwork {
  int inputSize, hiddenSize, outputSize;
  double learningRate;
  late Matrix weightsIH, weightsHO, biasH, biasO;

  NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize, this.learningRate) {
    weightsIH = Matrix(hiddenSize, inputSize);
    weightsHO = Matrix(outputSize, hiddenSize);
    biasH = Matrix(hiddenSize, 1);
    biasO = Matrix(outputSize, 1);
  }

  List<Matrix> forward(List<double> inputArray) {
    Matrix inputs = Matrix(inputArray.length, 1);
    for (int i = 0; i < inputArray.length; i++) {
      inputs.data[i][0] = inputArray[i];
    }

    Matrix hidden = weightsIH.multiply(inputs);
    hidden.add(biasH);
    hidden.apply(sigmoid);

    Matrix outputs = weightsHO.multiply(hidden);
    outputs.add(biasO);
    outputs.apply(sigmoid);

    return [outputs, hidden, inputs];
  }

  void train(List<double> inputArray, List<double> targetArray) {
    List<Matrix> results = forward(inputArray);
    Matrix outputs = results[0], hidden = results[1], inputs = results[2];

    Matrix targets = Matrix(targetArray.length, 1);
    for (int i = 0; i < targetArray.length; i++) {
      targets.data[i][0] = targetArray[i];
    }

    Matrix outputErrors = Matrix(outputSize, 1);
    for (int i = 0; i < outputSize; i++) {
      outputErrors.data[i][0] = targets.data[i][0] - outputs.data[i][0];
    }

    Matrix hiddenErrors = weightsHO.transpose().multiply(outputErrors);

    Matrix gradientsO = Matrix(outputSize, 1);
    for (int i = 0; i < outputSize; i++) {
      gradientsO.data[i][0] = outputErrors.data[i][0] * sigmoidDerivative(outputs.data[i][0]);
    }
    gradientsO.multiplyScalar(learningRate);

    Matrix weightsHODeltas = gradientsO.multiply(hidden.transpose());
    weightsHO.add(weightsHODeltas);
    biasO.add(gradientsO);

    Matrix gradientsH = Matrix(hiddenSize, 1);
    for (int i = 0; i < hiddenSize; i++) {
      gradientsH.data[i][0] = hiddenErrors.data[i][0] * sigmoidDerivative(hidden.data[i][0]);
    }
    gradientsH.multiplyScalar(learningRate);

    Matrix weightsIHDeltas = gradientsH.multiply(inputs.transpose());
    weightsIH.add(weightsIHDeltas);
    biasH.add(gradientsH);
  }
}

void main() {
  Random rng = Random();

  NeuralNetwork nn = NeuralNetwork(2, 4, 1, 0.1);

  nn.weightsIH.randomize(rng);
  nn.weightsHO.randomize(rng);
  nn.biasH.randomize(rng);
  nn.biasO.randomize(rng);

  List<List<double>> xorInputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ];
  List<double> xorOutputs = [0, 1, 1, 0];

  int epochs = 100000;
  for (int i = 0; i < epochs; i++) {
    for (int j = 0; j < xorInputs.length; j++) {
      nn.train(xorInputs[j], [xorOutputs[j]]);
    }
    if (i % 10000 == 0) {
      print('$i epochs done');
    }
  }

  print('Testing XOR outputs:');
  for (List<double> input in xorInputs) {
    Matrix result = nn.forward(input)[0];
    print('Input: ${input[0]} ${input[1]}, Output: ${result.data[0][0].toStringAsFixed(4)}');
  }
}