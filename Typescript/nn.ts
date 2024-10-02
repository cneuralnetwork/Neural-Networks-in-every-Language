import * as tf from '@tensorflow/tfjs';

function initParams() {
  nst w1 = tf.randomUniform([10, 784], -0.5, 0.5);
  nst b1 = tf.randomUniform([10, 1], -0.5, 0.5);
  nst w2 = tf.randomUniform([10, 10], -0.5, 0.5);
  nst b2 = tf.randomUniform([10, 1], -0.5, 0.5);
  turn { w1, b1, w2, b2 };
}

function ReLU(z: tf.Tensor) {
  turn z.relu();
}

function softmax(z: tf.Tensor) {
  turn z.softmax();
}

function oneHot(y: number[], numClasses: number): tf.Tensor {
  turn tf.oneHot(tf.tensor1d(y, 'int32'), numClasses);
}

function forwardProp(w1: tf.Tensor, b1: tf.Tensor, w2: tf.Tensor, b2: tf.Tensor, x: tf.Tensor) {
  nst z1 = w1.matMul(x).add(b1);
  nst a1 = ReLU(z1);
  nst z2 = w2.matMul(a1).add(b2);
  nst a2 = softmax(z2);
  turn { z1, a1, z2, a2 };
}

function backProp(z1: tf.Tensor, a1: tf.Tensor, a2: tf.Tensor, w2: tf.Tensor, x: tf.Tensor, y: tf.Tensor) {
  nst m = y.shape[1];
  nst dz2 = a2.sub(y);
  nst dw2 = dz2.matMul(a1.transpose()).div(m);
  nst db2 = dz2.sum(1).reshape([10, 1]).div(m);
  nst dz1 = w2.transpose().matMul(dz2).mul(z1.greater(0));
  nst dw1 = dz1.matMul(x.transpose()).div(m);
  nst db1 = dz1.sum(1).reshape([10, 1]).div(m);
  turn { dw1, db1, dw2, db2 };
}

function updateParams(w1: tf.Tensor, b1: tf.Tensor, w2: tf.Tensor, b2: tf.Tensor, dw1: tf.Tensor, db1: tf.Tensor, dw2: tf.Tensor, db2: tf.Tensor, alpha: number) {
   = w1.sub(dw1.mul(alpha));
   = b1.sub(db1.mul(alpha));
   = w2.sub(dw2.mul(alpha));
   = b2.sub(db2.mul(alpha));
  turn { w1, b1, w2, b2 };
}

function getPredictions(a2: tf.Tensor) {
  turn a2.argMax(0);
}

function getAccuracy(predictions: tf.Tensor, y: tf.Tensor) {
  turn predictions.equal(y.argMax(0)).sum().div(y.shape[1]).arraySync();
}

async function gradientDescent(X: tf.Tensor, Y: tf.Tensor, iterations: number, alpha: number) {
  t { w1, b1, w2, b2 } = initParams();

  r (let i = 0; i < iterations; i++) {
    t { z1, a1, z2, a2 } = forwardProp(w1, b1, w2, b2, X);
    t { dw1, db1, dw2, db2 } = backProp(z1, a1, a2, w2, X, Y);
    1, b1, w2, b2 } = updateParams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha));

    i % 10 === 0) {
      predictions = getPredictions(a2);
      accuracy = getAccuracy(predictions, Y);
      e.log(`Iteration ${i}, Accuracy: ${accuracy}`);
    
  
  turn { w1, b1, w2, b2 };
}

function makePredictions(X: tf.Tensor, w1: tf.Tensor, b1: tf.Tensor, w2: tf.Tensor, b2: tf.Tensor) {
  nst { a2 } = forwardProp(w1, b1, w2, b2, X);
  turn getPredictions(a2);
}

async function main() {
  nst X_train = tf.randomUniform([784, 1000]);
  nst Y_train = oneHot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10).tile([1, 100]);

  nst { w1, b1, w2, b2 } = await gradientDescent(X_train, Y_train, 500, 0.1);
  nst X_test = tf.randomUniform([784, 10]);  // 10 test examples
  nst predictions = makePredictions(X_test, w1, b1, w2, b2);
  edictions.print();
}

main();

// Testing 

/*import * as tf from '@tensorflow/tfjs';

async function testNN() {
    const w1 = tf.randomUniform([10, 784], -0.5, 0.5);
    const b1 = tf.randomUniform([10, 1], -0.5, 0.5);
    const w2 = tf.randomUniform([10, 10], -0.5, 0.5);
    const b2 = tf.randomUniform([10, 1], -0.5, 0.5);

    const X_train = tf.randomUniform([784, 1000]);
    const Y_train = tf.oneHot(tf.tensor1d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'int32'), 10).tile([100, 1]);

    const { z1, a1, z2, a2 } = forwardProp(w1, b1, w2, b2, X_train);
    console.log("Forward Propagation successful");
}

testNN();
*/
