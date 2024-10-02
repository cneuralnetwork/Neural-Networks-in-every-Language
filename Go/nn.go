package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	s := sigmoid(x)
	return s * (1.0 - s)
}

type Matrix struct {
	Rows, Cols int
	Data       [][]float64
}

func NewMatrix(rows, cols int) *Matrix {
	m := &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([][]float64, rows),
	}
	for i := range m.Data {
		m.Data[i] = make([]float64, cols)
	}
	return m
}

func (m *Matrix) Randomize(rng *rand.Rand) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = rng.Float64()*2 - 1
		}
	}
}

func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Multiply(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic("Incompatible matrix dimensions")
	}
	result := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result
}

func (m *Matrix) Add(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Incompatible matrix dimensions")
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] += other.Data[i][j]
		}
	}
}

func (m *Matrix) Subtract(other *Matrix) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic("Incompatible matrix dimensions")
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] -= other.Data[i][j]
		}
	}
}

func (m *Matrix) MultiplyScalar(scalar float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] *= scalar
		}
	}
}

func (m *Matrix) Apply(f func(float64) float64) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = f(m.Data[i][j])
		}
	}
}

type NeuralNetwork struct {
	InputSize    int
	HiddenSize   int
	OutputSize   int
	LearningRate float64
	WeightsIH    *Matrix
	WeightsHO    *Matrix
	BiasH        *Matrix
	BiasO        *Matrix
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *NeuralNetwork {
	nn := &NeuralNetwork{
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
		OutputSize:   outputSize,
		LearningRate: learningRate,
		WeightsIH:    NewMatrix(hiddenSize, inputSize),
		WeightsHO:    NewMatrix(outputSize, hiddenSize),
		BiasH:        NewMatrix(hiddenSize, 1),
		BiasO:        NewMatrix(outputSize, 1),
	}
	return nn
}

func (nn *NeuralNetwork) Forward(inputArray []float64) []*Matrix {
	inputs := NewMatrix(len(inputArray), 1)
	for i, v := range inputArray {
		inputs.Data[i][0] = v
	}

	hidden := nn.WeightsIH.Multiply(inputs)
	hidden.Add(nn.BiasH)
	hidden.Apply(sigmoid)

	outputs := nn.WeightsHO.Multiply(hidden)
	outputs.Add(nn.BiasO)
	outputs.Apply(sigmoid)

	return []*Matrix{outputs, hidden, inputs}
}

func (nn *NeuralNetwork) Train(inputArray, targetArray []float64) {
	results := nn.Forward(inputArray)
	outputs, hidden, inputs := results[0], results[1], results[2]

	targets := NewMatrix(len(targetArray), 1)
	for i, v := range targetArray {
		targets.Data[i][0] = v
	}

	outputErrors := NewMatrix(nn.OutputSize, 1)
	for i := 0; i < nn.OutputSize; i++ {
		outputErrors.Data[i][0] = targets.Data[i][0] - outputs.Data[i][0]
	}

	hiddenErrors := nn.WeightsHO.Transpose().Multiply(outputErrors)

	gradientsO := NewMatrix(nn.OutputSize, 1)
	for i := 0; i < nn.OutputSize; i++ {
		gradientsO.Data[i][0] = outputErrors.Data[i][0] * sigmoidDerivative(outputs.Data[i][0])
	}
	gradientsO.MultiplyScalar(nn.LearningRate)

	weightsHODeltas := gradientsO.Multiply(hidden.Transpose())
	nn.WeightsHO.Add(weightsHODeltas)
	nn.BiasO.Add(gradientsO)

	gradientsH := NewMatrix(nn.HiddenSize, 1)
	for i := 0; i < nn.HiddenSize; i++ {
		gradientsH.Data[i][0] = hiddenErrors.Data[i][0] * sigmoidDerivative(hidden.Data[i][0])
	}
	gradientsH.MultiplyScalar(nn.LearningRate)

	weightsIHDeltas := gradientsH.Multiply(inputs.Transpose())
	nn.WeightsIH.Add(weightsIHDeltas)
	nn.BiasH.Add(gradientsH)
}

func main() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	nn := NewNeuralNetwork(2, 4, 1, 0.1)

	nn.WeightsIH.Randomize(rng)
	nn.WeightsHO.Randomize(rng)
	nn.BiasH.Randomize(rng)
	nn.BiasO.Randomize(rng)

	xorInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	xorOutputs := []float64{0, 1, 1, 0}

	epochs := 100000
	for i := 0; i < epochs; i++ {
		for j := range xorInputs {
			nn.Train(xorInputs[j], []float64{xorOutputs[j]})
		}
		if i%10000 == 0 {
			fmt.Printf("%d epochs done\n", i)
		}
	}

	fmt.Println("Testing XOR outputs:")
	for _, input := range xorInputs {
		result := nn.Forward(input)[0]
		fmt.Printf("Input: %v %v, Output: %.4f\n", input[0], input[1], result.Data[0][0])
	}
}
