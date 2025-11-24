{-# LANGUAGE BangPatterns #-}

-- XOR Neural Network in Haskell
-- Architecture: 2 inputs, 2 hidden neurons, 1 output
-- Compile: ghc -O2 -o nn nn.hs
-- Run: ./nn

module Main where

import System.Random
import Text.Printf
import Control.Monad (foldM)

-- Type aliases for clarity
type Vector = [Double]
type Matrix = [Vector]
type Layer = Vector
type Weights = Matrix
type Biases = Vector

-- Network state
data Network = Network
  { hiddenWeights :: !Weights
  , hiddenBiases  :: !Biases
  , outputWeights :: !Weights
  , outputBiases  :: !Biases
  } deriving (Show)

-- Activation functions
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

sigmoidDerivative :: Double -> Double
sigmoidDerivative y = y * (1.0 - y)

-- Vector operations
dotProduct :: Vector -> Vector -> Double
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

matrixVectorMult :: Matrix -> Vector -> Vector
matrixVectorMult matrix vec = map (`dotProduct` vec) matrix

vectorAdd :: Vector -> Vector -> Vector
vectorAdd = zipWith (+)

scalarVectorMult :: Double -> Vector -> Vector
scalarVectorMult s = map (* s)

applyToVector :: (Double -> Double) -> Vector -> Vector
applyToVector = map

-- Initialize random weights and biases
randomWeights :: Int -> Int -> StdGen -> (Matrix, StdGen)
randomWeights rows cols gen = 
  let (weights, gen') = foldr step ([], gen) [1..rows]
      step _ (acc, g) = 
        let (row, g') = foldr getWeight ([], g) [1..cols]
            getWeight _ (r, g'') = 
              let (w, g''') = randomR (-0.5, 0.5) g''
              in (w:r, g''')
        in (row:acc, g')
  in (weights, gen')

randomBiases :: Int -> StdGen -> (Vector, StdGen)
randomBiases size gen = 
  foldr step ([], gen) [1..size]
  where
    step _ (acc, g) = 
      let (b, g') = randomR (-0.5, 0.5) g
      in (b:acc, g')

-- Initialize network with random weights
initNetwork :: StdGen -> Network
initNetwork gen = 
  let (hw, gen1) = randomWeights 2 2 gen
      (hb, gen2) = randomBiases 2 gen1
      (ow, gen3) = randomWeights 2 1 gen2
      (ob, _)    = randomBiases 1 gen3
  in Network hw hb ow ob

-- Forward pass
forwardPass :: Network -> Vector -> (Layer, Layer)
forwardPass net input =
  let hiddenWeighted = matrixVectorMult (hiddenWeights net) input
      hiddenPre = vectorAdd hiddenWeighted (hiddenBiases net)
      hiddenLayer = applyToVector sigmoid hiddenPre
      
      outputWeighted = matrixVectorMult (outputWeights net) hiddenLayer
      outputPre = vectorAdd outputWeighted (outputBiases net)
      outputLayer = applyToVector sigmoid outputPre
  in (hiddenLayer, outputLayer)

-- Backpropagation and weight update
updateWeights :: Double -> Vector -> Vector -> Network -> (Layer, Layer) -> Network
updateWeights learningRate input target net (hiddenLayer, outputLayer) =
  let -- Output layer errors
      outputErrors = zipWith (\t o -> (t - o) * sigmoidDerivative o) target outputLayer
      
      -- Hidden layer errors
      hiddenErrors = zipWith (\h j -> 
        let errorSum = sum $ zipWith (\err weights -> err * (weights !! j)) 
                                     outputErrors (outputWeights net)
        in errorSum * sigmoidDerivative h) hiddenLayer [0..]
      
      -- Update output weights and biases
      newOutputWeights = zipWith (\row j ->
        zipWith (\w k -> w + learningRate * (outputErrors !! k) * (hiddenLayer !! j))
                row [0..]) (outputWeights net) [0..]
      
      newOutputBiases = zipWith (\b err -> b + learningRate * err) 
                                (outputBiases net) outputErrors
      
      -- Update hidden weights and biases
      newHiddenWeights = zipWith (\row j ->
        zipWith (\w k -> w + learningRate * (hiddenErrors !! k) * (input !! j))
                row [0..]) (hiddenWeights net) [0..]
      
      newHiddenBiases = zipWith (\b err -> b + learningRate * err) 
                                (hiddenBiases net) hiddenErrors
  
  in Network newHiddenWeights newHiddenBiases newOutputWeights newOutputBiases

-- Train on single sample
trainSample :: Double -> (Vector, Vector) -> Network -> Network
trainSample learningRate (input, target) net =
  let (hiddenLayer, outputLayer) = forwardPass net input
  in updateWeights learningRate input target net (hiddenLayer, outputLayer)

-- Train for one epoch
trainEpoch :: Double -> [(Vector, Vector)] -> Network -> Network
trainEpoch learningRate samples net = foldl (flip (trainSample learningRate)) net samples

-- Train for multiple epochs
trainNetwork :: Int -> Double -> [(Vector, Vector)] -> Network -> Network
trainNetwork numEpochs learningRate samples net =
  foldl (\n _ -> trainEpoch learningRate samples n) net [1..numEpochs]

-- Test the network
testNetwork :: Network -> [(Vector, Vector)] -> IO ()
testNetwork net samples = do
  putStrLn "\nTraining complete! Testing network:\n"
  mapM_ testSample samples
  where
    testSample (input, target) =
      let (_, output) = forwardPass net input
      in printf "Input: %.1f %.1f, Output: %.6f, Expected: %.1f\n"
           (input !! 0) (input !! 1) (output !! 0) (target !! 0)

-- Main program
main :: IO ()
main = do
  putStrLn "Training XOR Neural Network in Haskell..."
  
  -- Training data for XOR
  let trainingData = [ ([0.0, 0.0], [0.0])
                     , ([0.0, 1.0], [1.0])
                     , ([1.0, 0.0], [1.0])
                     , ([1.0, 1.0], [0.0])
                     ]
  
  -- Initialize network
  gen <- getStdGen
  let net = initNetwork gen
  
  -- Training parameters
  let numEpochs = 10000
      learningRate = 0.1
  
  -- Train the network
  let !trainedNet = trainNetwork numEpochs learningRate trainingData net
  
  -- Test the network
  testNetwork trainedNet trainingData
