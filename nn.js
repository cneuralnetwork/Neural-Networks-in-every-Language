const epsilon = 1e-15;
const sigmoid = x => 1 / (1 + Math.exp(-x));
const sigmoidDerivative = x => Math.max(epsilon, x * (1 - x));

const create = (rows, cols) => Array.from(
  { length: rows },
  () => Array.from({ length: cols }, () => (Math.random() - 0.5) * 0.1)
);

const dot = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);

const train = (inputs, targets, hiddenSize = 2, epochs = 100000, learningRate = 0.1) => {
  let w1 = create(inputs[0].length, hiddenSize);
  let w2 = create(hiddenSize, 1);
  let bias1 = Array(hiddenSize).fill(0);
  let bias2 = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalerr = 0;

    inputs.forEach((input, j) => {
      const hidden = w1.map((w, k) => sigmoid(dot(input, w) + bias1[k]));
      const output = sigmoid(dot(hidden, w2.map(w => w[0])) + bias2);
      const outputerr = targets[j][0] - output;
      totalerr += Math.abs(outputerr);
      const outputdel = outputerr * sigmoidDerivative(output);
      const hiddenErrors = w2.map(w => w[0] * outputdel);
      const hiddendels = hiddenErrors.map((err, k) => err * sigmoidDerivative(hidden[k]));
      w2 = w2.map((w, k) => [w[0] + learningRate * outputdel * hidden[k]]);
      bias2 += learningRate * outputdel;
      w1 = w1.map((w, k) => w.map((v, l) => v + learningRate * hiddendels[k] * input[l]));
      bias1 = bias1.map((b, k) => b + learningRate * hiddendels[k]);
    });

    if (epoch % 1000 === 0) {
      console.log(`epoch ${epoch}: total error = ${totalerr}`);
    }
  }

  return input => {
    const hidden = w1.map((w, k) => sigmoid(dot(input, w) + bias1[k]));
    return sigmoid(dot(hidden, w2.map(w => w[0])) + bias2);
  };
};

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];

const model = train(inputs, targets);
inputs.forEach(input => console.log(`input: ${input}, output: ${model(input).toFixed(4)}`));
