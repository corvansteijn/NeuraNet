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

        /// <summary>
        /// Propagates the network output error backwards through the network by calculating the gradients for the current layer.
        /// If the current layer has a <see cref="previousLayer"/> the <see cref="PreviousLayerActivationGradients"/> will be
        /// propagated backwards to that layer, so that eventually the gradients will be calculated for all layers in the network.
        /// </summary>
        /// <param name="costDerivative">Derivative of the cost with respect to the output activation of the current layer</param>
        public void BackPropagate(Vector<double> costDerivative)
        {
            CalculateGradients(costDerivative);
        }

        /// <summary>
        /// Calculates the gradient for the current layer based on the gradients and input weights of the next layer
        /// in the neural network.
        /// </summary>
        /// <param name="delC_delA">Derivative of cost w.r.t. the hidden layer output</param>
        /// <remarks>
        /// Gradients are a measure of how far off, and in what direction (positive or negative) the current layer's 
        /// output values are.
        /// </remarks>
        private void CalculateGradients(Vector<double> delC_delA)
        {
            Vector<double> delA_delZ = OutputActivation.Derivative(z);
            Vector<double> nodeDeltas = delA_delZ.PointwiseMultiply(delC_delA);

            WeightGradients = CalculateWeightGradients(nodeDeltas);
            BiasGradients = CalculateBiasGradients(nodeDeltas);
        }

        private Matrix<double> CalculateWeightGradients(Vector<double> nodeDeltas)
        {
            Vector<double> delZ_delW = inputs;
            return delZ_delW.OuterProduct(nodeDeltas);
        }

        private Vector<double> CalculateBiasGradients(Vector<double> nodeDeltas)
        {
            const int delZ_delB = 1;
            return delZ_delB * nodeDeltas;
        }
    }
}
