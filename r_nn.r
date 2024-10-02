library(testthat)

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
  sigmoid(x) * (1 - sigmoid(x))
}

NeuralNetwork <- setRefClass(
  "NeuralNetwork",
  fields = list(
    input_size = "numeric",
    hidden_size = "numeric",
    output_size = "numeric",
    learning_rate = "numeric",
    weights_ih = "matrix",
    weights_ho = "matrix",
    bias_h = "matrix",
    bias_o = "matrix"
  ),
  methods = list(
    initialize = function(input_size, hidden_size, output_size, learning_rate = 0.01) {
      input_size <<- input_size
      hidden_size <<- hidden_size
      output_size <<- output_size
      learning_rate <<- learning_rate
      weights_ih <<- matrix(rnorm(input_size * hidden_size, mean = 0, sd = 0.5), nrow = hidden_size, ncol = input_size)
      weights_ho <<- matrix(rnorm(hidden_size * output_size, mean = 0, sd = 0.5), nrow = output_size, ncol = hidden_size)
      bias_h <<- matrix(rnorm(hidden_size, mean = 0, sd = 0.5), nrow = hidden_size, ncol = 1)
      bias_o <<- matrix(rnorm(output_size, mean = 0, sd = 0.5), nrow = output_size, ncol = 1)
    },
    
    forward = function(input_array) {
      inputs <- matrix(input_array, ncol = 1)
      hidden <- weights_ih %*% inputs + bias_h
      hidden <- apply(hidden, c(1, 2), sigmoid)
      output <- weights_ho %*% hidden + bias_o
      output <- apply(output, c(1, 2), sigmoid)
      list(output = output, hidden = hidden, inputs = inputs)
    },
    
    train = function(input_array, target_array) {
      results <- forward(input_array)
      outputs <- results$output
      hidden <- results$hidden
      inputs <- results$inputs
      targets <- matrix(target_array, ncol = 1)
      output_errors <- targets - outputs
      hidden_errors <- t(weights_ho) %*% output_errors
      gradients_o <- output_errors * apply(outputs, c(1, 2), sigmoid_derivative)
      gradients_o <- gradients_o * learning_rate
      weights_ho_deltas <- gradients_o %*% t(hidden)
      weights_ho <<- weights_ho + weights_ho_deltas
      bias_o <<- bias_o + gradients_o
      gradients_h <- hidden_errors * apply(hidden, c(1, 2), sigmoid_derivative)
      gradients_h <- gradients_h * learning_rate
      weights_ih_deltas <- gradients_h %*% t(inputs)
      weights_ih <<- weights_ih + weights_ih_deltas
      bias_h <<- bias_h + gradients_h
    }
  )
)

test_that("Neural Network initialization works", {
  nn <- NeuralNetwork$new(2, 4, 1)
  expect_equal(dim(nn$weights_ih), c(4, 2))
  expect_equal(dim(nn$weights_ho), c(1, 4))
  expect_equal(dim(nn$bias_h), c(4, 1))
  expect_equal(dim(nn$bias_o), c(1, 1))
})

test_that("Forward propagation produces expected output shape", {
  nn <- NeuralNetwork$new(2, 4, 1)
  input <- c(0.5, 0.8)
  output <- nn$forward(input)
  expect_equal(dim(output$output), c(1, 1))
})

set.seed(123)
xor_inputs <- matrix(c(0,0, 0,1, 1,0, 1,1), ncol=2, byrow=TRUE)
xor_outputs <- c(0, 1, 1, 0)
nn <- NeuralNetwork$new(input_size=2, hidden_size=4, output_size=1)
epochs <- 100000
for(i in 1:epochs) {
  for(j in 1:nrow(xor_inputs)) {
    nn$train(xor_inputs[j,], xor_outputs[j])
  }
  if (i %% 10000 == 0) {
    cat(i, " epochs done\n")
  }
}

cat("Testing XOR outputs:\n")
for(i in 1:nrow(xor_inputs)) {
  result <- nn$forward(xor_inputs[i,])$output
  cat(sprintf("Input: %d %d, Output: %.4f\n", 
              xor_inputs[i,1], xor_inputs[i,2], result))
}





