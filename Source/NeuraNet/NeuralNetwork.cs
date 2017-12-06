using System.Collections.Generic;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;

using NeuraNet.NetworkLayout;

namespace NeuraNet
{
    /// <summary>
    /// Neural network implementation that can be trained to recognise 'patterns' by learning from examples.
    /// </summary>
    public class NeuralNetwork
    {
        private readonly IEnumerable<Layer> layers;
        private readonly Layer firstHiddenLayer;
        private readonly Layer outputLayer;

        /// <summary>
        /// Instantiates a new neural network with the layout provided by the specified <paramref name="layoutProvider"/>.
        /// </summary>
        /// <param name="layoutProvider">Provides the layout of the network</param>
        public NeuralNetwork(INetworkLayoutProvider layoutProvider)
        {
            layers = layoutProvider.GetLayers();
            firstHiddenLayer = layers.First();
            outputLayer = layers.Last();
        }

        /// <summary>
        /// Returns the layers of the network
        /// </summary>
        public IEnumerable<Layer> GetLayers()
        {
            return layers;
        }

        /// <summary>
        /// Queries the network for the result of the given <paramref name="input"/>.
        /// </summary>
        public double[] Query(double[] input)
        {
            return firstHiddenLayer.FeedForward(input).ToArray();
        }

        /// <summary>
        /// Train the network using the specified <paramref name="trainingExamples"/>.
        /// </summary>
        /// <param name="trainingExamples">The list of examples that will train the network.</param>
        /// <param name="numberOfEpochs">
        /// The number of epochs to use for the training. Each epoch means one forward pass and one backward pass of
        /// all the training examples
        /// </param>
        /// <returns>The mean cost for the examples in the last epoch</returns>
        public double Train(TrainingExample[] trainingExamples, int numberOfEpochs)
        {
            double meanCost = 0;

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                double costSumForAllExamples = 0.0;

                int currentExample = 1;
                foreach (TrainingExample example in trainingExamples)
                {
                    costSumForAllExamples += Train(example.Input, example.ExpectedOutput);

                    meanCost = costSumForAllExamples / currentExample;

                    currentExample++;
                }
            }

            return meanCost;
        }

        private double Train(double[] input, Vector<double> targetOutput)
        {
            Vector<double> networkOutput = firstHiddenLayer.FeedForward(input);

            Vector<double> costDerivative = networkOutput - targetOutput;
            outputLayer.BackPropagate(costDerivative);

            return CalculateCost(targetOutput, networkOutput);
        }

        /// <summary>
        /// The training algorithm has done a good job if it can find weights and biases for which the quadratic cost
        /// is close to 0. It's not doing so well when the cost is large, because that would mean that
        /// targetOutput - networkOutput is not close to the output for a large number of inputs.
        /// </summary>
        private double CalculateCost(Vector<double> targetOutput, Vector<double> networkOutput)
        {
            double Squared(double value) => (value * value);

            return 0.5 * ((networkOutput - targetOutput).Map(Squared).Sum());
        }
    }
}