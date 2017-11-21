using MathNet.Numerics.LinearAlgebra;

using NeuraNet.Activations;
using NeuraNet.NetworkLayout;

namespace NeuraNet
{
    /// <summary>
    /// Represents one layer in a <see cref="NeuralNetwork"/>. The layer receives inputs from the previous layer in
    /// the network. Based on this input it calculates an output that serves as the input for the next layer in the network.
    /// </summary>
    public class Layer
    {
        private Layer previousLayer;
        private Layer nextLayer;

        public IActivation OutputActivation { get; }

        internal Matrix<double> Weights { get; }
        internal Vector<double> Biases { get; }
        private Vector<double> inputs;
        private Vector<double> z;

        internal Matrix<double> WeightGradients { get; private set; }
        internal Vector<double> BiasGradients { get; private set; }

        public Layer(int numberOfNeuronsInPreviousLayer, int numberOfNeurons, ILayerInitializer layerInitializer,
            IActivation outputActivation)
        {
            OutputActivation = outputActivation;

            Weights = Matrix<double>.Build.Dense(numberOfNeuronsInPreviousLayer, numberOfNeurons, layerInitializer.GetWeight);
            Biases = Vector<double>.Build.Dense(numberOfNeurons, layerInitializer.GetBias);
        }

        /// <summary>
        /// Connects the current layer to the specified <paramref name="previous"/> and <paramref name="next"/>.
        /// A proper connection between the layers is required for the feedforward and backpropagation algorithms.
        /// </summary>
        public void ConnectTo(Layer previous, Layer next)
        {
            previousLayer = previous;
            nextLayer = next;
        }

        /// <summary>
        /// Calculates the current layer's output values based on the specified <paramref name="inputs"/>, the current
        /// <see cref="Weights"/> and <see cref="Biases"/> and the used <see cref="OutputActivation"/> algorithm.
        /// The output is then passed on to the <see cref="nextLayer"/>. If there is no next layer the output values are
        /// the output of the entire network.
        /// </summary>
        public Vector<double> FeedForward(double[] inputs)
        {
            return FeedForward(Vector<double>.Build.DenseOfArray(inputs));
        }

        private Vector<double> FeedForward(Vector<double> inputs)
        {
            this.inputs = inputs;

            z = (inputs * Weights) + Biases;
            Vector<double> outputs = OutputActivation.Transform(z);

            return (nextLayer != null) ? nextLayer.FeedForward(outputs) : outputs;
        }
    }
}
