using System;
class NeuralNetwork
{
    static double Sigmoid(double x)=>1.0/(1.0+Math.Exp(-x));
    static double SigmoidDerivative(double x)=>x*(1-x);
    static void Main()
    {
        double[][] trainingInputs={
            new double[]{0,0},
            new double[]{0,1},
            new double[]{1,0},
            new double[]{1,1}
        };
        double[][] trainingOutputs={
            new double[]{0},
            new double[]{1},
            new double[]{1},
            new double[]{0}
        };
        int numInputs=2,numHidden=2,numOutputs=1;
        Random rand=new Random();
        double[,] hiddenWeights=new double[numInputs,numHidden];
        double[] hiddenBiases=new double[numHidden];
        double[,] outputWeights=new double[numHidden,numOutputs];
        double[] outputBiases=new double[numOutputs];
        for(int i=0;i<numInputs;i++)
            for(int j=0;j<numHidden;j++)
                hiddenWeights[i,j]=rand.NextDouble()*2-1;
        for(int j=0;j<numHidden;j++)
            hiddenBiases[j]=rand.NextDouble()*2-1;
        for(int i=0;i<numHidden;i++)
            for(int j=0;j<numOutputs;j++)
                outputWeights[i,j]=rand.NextDouble()*2-1;
        for(int j=0;j<numOutputs;j++)
            outputBiases[j]=rand.NextDouble()*2-1;
        int epochs=10000;
        double learningRate=0.1;
        for(int epoch=0;epoch<epochs;epoch++)
        {
            for(int i=0;i<trainingInputs.Length;i++)
            {
                double[] hiddenLayer=new double[numHidden];
                for(int j=0;j<numHidden;j++)
                {
                    double sum=0;
                    for(int k=0;k<numInputs;k++)
                        sum+=trainingInputs[i][k]*hiddenWeights[k,j];
                    hiddenLayer[j]=Sigmoid(sum+hiddenBiases[j]);
                }
                double[] outputLayer=new double[numOutputs];
                for(int j=0;j<numOutputs;j++)
                {
                    double sum=0;
                    for(int k=0;k<numHidden;k++)
                        sum+=hiddenLayer[k]*outputWeights[k,j];
                    outputLayer[j]=Sigmoid(sum+outputBiases[j]);
                }
                double[] outputErrors=new double[numOutputs];
                for(int j=0;j<numOutputs;j++)
                    outputErrors[j]=(trainingOutputs[i][j]-outputLayer[j])*SigmoidDerivative(outputLayer[j]);
                double[] hiddenErrors=new double[numHidden];
                for(int j=0;j<numHidden;j++)
                {
                    double errorSum=0;
                    for(int k=0;k<numOutputs;k++)
                        errorSum+=outputErrors[k]*outputWeights[j,k];
                    hiddenErrors[j]=errorSum*SigmoidDerivative(hiddenLayer[j]);
                }
                for(int j=0;j<numOutputs;j++)
                {
                    for(int k=0;k<numHidden;k++)
                        outputWeights[k,j]+=learningRate*outputErrors[j]*hiddenLayer[k];
                    outputBiases[j]+=learningRate*outputErrors[j];
                }
                for(int j=0;j<numHidden;j++)
                {
                    for(int k=0;k<numInputs;k++)
                        hiddenWeights[k,j]+=learningRate*hiddenErrors[j]*trainingInputs[i][k];
                    hiddenBiases[j]+=learningRate*hiddenErrors[j];
                }
            }
        }
        Console.WriteLine("Trained XOR Neural Network:");
        for(int i=0;i<trainingInputs.Length;i++)
        {
            double[] hiddenLayer=new double[numHidden];
            for(int j=0;j<numHidden;j++)
            {
                double sum=0;
                for(int k=0;k<numInputs;k++)
                    sum+=trainingInputs[i][k]*hiddenWeights[k,j];
                hiddenLayer[j]=Sigmoid(sum+hiddenBiases[j]);
            }
            double[] outputLayer=new double[numOutputs];
            for(int j=0;j<numOutputs;j++)
            {
                double sum=0;
                for(int k=0;k<numHidden;k++)
                    sum+=hiddenLayer[k]*outputWeights[k,j];
                outputLayer[j]=Sigmoid(sum+outputBiases[j]);
            }
            Console.WriteLine($"{trainingInputs[i][0]} XOR {trainingInputs[i][1]} = {Math.Round(outputLayer[0],4)}");
        }
    }
}
