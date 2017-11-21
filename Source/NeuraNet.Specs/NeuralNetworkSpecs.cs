using System.Collections.Generic;
using System.Linq;

using FluentAssertions;

using NeuraNet.Activations;
using NeuraNet.NetworkLayout;
using NeuraNet.Specs.Builders;

using Xunit;

namespace NeuraNet.Specs
{
    public class NeuralNetworkSpecs
    {
        private class TwoLayerNetworkProvider : NetworkLayoutProvider
        {
            private readonly ILayerInitializer hiddenLayerInitializer = new LayerInitializerBuilder()
                .UsingWeights(new[,]
                {
                    { 0.001, 0.002, 0.003, 0.004 },
                    { 0.005, 0.006, 0.007, 0.008 },
                    { 0.009, 0.010, 0.011, 0.012 }
                })
                .UsingBiases(new[]
                {
                    0.013,
                    0.014,
                    0.015,
                    0.016
                })
                .Build();

            private readonly ILayerInitializer outputLayerInitializer = new LayerInitializerBuilder()
                .UsingWeights(new[,]
                {
                    { 0.017, 0.018 },
                    { 0.019, 0.020 },
                    { 0.021, 0.022 },
                    { 0.023, 0.024 }
                })
                .UsingBiases(new[]
                {
                    0.025,
                    0.026
                })
                .Build();

            public TwoLayerNetworkProvider()
            {
                layers = new List<Layer>
                {
                    new Layer(3, 4, hiddenLayerInitializer, new SigmoidActivation()),
                    new Layer(4, 2, outputLayerInitializer, new SigmoidActivation())
                };

                ConnectLayers();
            }
        }

        [Fact]
        public void When_querying_a_network_that_has_known_weights_and_biases_it_should_yield_the_expected_output()
        {
            // Arrange
            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            double[] networkOutput = network.Query(new[] { 1.0, -2.0, 3.0 });

            // Assert
            networkOutput[0].Should().BeApproximately(0.5164, 0.00005);
            networkOutput[1].Should().BeApproximately(0.5172, 0.00005);
        }

        [Fact]
        public void When_training_the_network_for_a_single_epoch_it_should_return_the_cost()
        {
            // Arrange
            const double learningRate = 0.5;
            const double momentum = 0.0;

            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            double cost = network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1, learningRate, momentum);

            // Assert
            cost.Should().BeApproximately(0.1418, 0.00005);
        }

        [Fact]
        public void When_training_the_network_for_a_single_epoch_it_should_calculate_the_gradient_vectors()
        {
            // Arrange
            const double learningRate = 0.5;
            const double momentum = 0.0;

            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1, learningRate, momentum);

            // Assert
            Layer hiddenLayer = network.GetLayers().First();
            Layer outputLayer = network.GetLayers().Last();

            double[] outputWeightGradients = outputLayer.WeightGradients.ToColumnMajorArray();
            outputWeightGradients[0].Should().BeApproximately(0.04983553, 0.000000005);
            outputWeightGradients[1].Should().BeApproximately(0.04990912, 0.000000005);
            outputWeightGradients[2].Should().BeApproximately(0.04998271, 0.000000005);
            outputWeightGradients[3].Should().BeApproximately(0.05005629, 0.000000005);
            outputWeightGradients[4].Should().BeApproximately(-0.04556976, 0.000000005);
            outputWeightGradients[5].Should().BeApproximately(-0.04563706, 0.000000005);
            outputWeightGradients[6].Should().BeApproximately(-0.04570435, 0.000000005);
            outputWeightGradients[7].Should().BeApproximately(-0.04577163, 0.000000005);

            var outputBiasGradients = outputLayer.BiasGradients.ToArray();
            outputBiasGradients[0].Should().BeApproximately(0.098150, 0.0000005);
            outputBiasGradients[1].Should().BeApproximately(-0.089749, 0.0000005);

            var outputInputGradients = outputLayer.PreviousLayerActivationGradients.ToArray();
            outputInputGradients[0].Should().BeApproximately(0.00005307, 0.000000005);
            outputInputGradients[1].Should().BeApproximately(0.00006988, 0.000000005);
            outputInputGradients[2].Should().BeApproximately(0.00008668, 0.000000005);
            outputInputGradients[3].Should().BeApproximately(0.00010348, 0.000000005);

            double[] hiddenWeightGradients = hiddenLayer.WeightGradients.ToColumnMajorArray();
            hiddenWeightGradients[0].Should().BeApproximately(0.00001326, 0.0000005);
            hiddenWeightGradients[1].Should().BeApproximately(-0.00002653, 0.0000005);
            hiddenWeightGradients[2].Should().BeApproximately(0.00003980, 0.0000005);
            hiddenWeightGradients[3].Should().BeApproximately(0.00001746, 0.0000005);
            hiddenWeightGradients[4].Should().BeApproximately(-0.00003493, 0.0000005);
            hiddenWeightGradients[5].Should().BeApproximately(0.00005239, 0.0000005);
            hiddenWeightGradients[6].Should().BeApproximately(0.00002166, 0.0000005);
            hiddenWeightGradients[7].Should().BeApproximately(-0.00004332, 0.0000005);
            hiddenWeightGradients[8].Should().BeApproximately(0.00006499, 0.0000005);
            hiddenWeightGradients[9].Should().BeApproximately(0.00002586, 0.0000005);
            hiddenWeightGradients[10].Should().BeApproximately(-0.00005172, 0.0000005);
            hiddenWeightGradients[11].Should().BeApproximately(0.00007758, 0.0000005);

            double[] hiddenBiasGradients = hiddenLayer.BiasGradients.ToArray();
            hiddenBiasGradients[0].Should().BeApproximately(0.000013265, 0.0000000005);
            hiddenBiasGradients[1].Should().BeApproximately(0.000017464, 0.0000000005);
            hiddenBiasGradients[2].Should().BeApproximately(0.000021662, 0.0000000005);
            hiddenBiasGradients[3].Should().BeApproximately(0.000025860, 0.0000000005);

            hiddenLayer.PreviousLayerActivationGradients.Should().BeNull();
        }

        [Fact]
        public void When_training_the_network_for_a_single_epoch_it_should_update_the_weights_and_biases_correctly()
        {
            // Arrange
            const double learningRate = 0.5;
            const double momentum = 0.0;

            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1, learningRate, momentum);

            // Assert
            Layer hiddenLayer = network.GetLayers().First();
            Layer outputLayer = network.GetLayers().Last();

            var updatedOutputWeights = outputLayer.Weights.ToColumnMajorArray();
            updatedOutputWeights[0].Should().BeApproximately(-0.00791776, 0.000000005);
            updatedOutputWeights[1].Should().BeApproximately(-0.00595456, 0.000000005);
            updatedOutputWeights[2].Should().BeApproximately(-0.00399135, 0.000000005);
            updatedOutputWeights[3].Should().BeApproximately(-0.00202815, 0.000000005);
            updatedOutputWeights[4].Should().BeApproximately(0.04078488, 0.000000005);
            updatedOutputWeights[5].Should().BeApproximately(0.04281853, 0.000000005);
            updatedOutputWeights[6].Should().BeApproximately(0.04485217, 0.000000005);
            updatedOutputWeights[7].Should().BeApproximately(0.04688582, 0.000000005);

            var updatedOutputBiases = outputLayer.Biases.ToArray();
            updatedOutputBiases[0].Should().BeApproximately(-0.02407493, 0.000000005);
            updatedOutputBiases[1].Should().BeApproximately(0.07087427, 0.000000005);

            var updatedHiddenWeights = hiddenLayer.Weights.ToColumnMajorArray();
            updatedHiddenWeights[0].Should().BeApproximately(0.00099337, 0.000000005);
            updatedHiddenWeights[1].Should().BeApproximately(0.00501327, 0.000000005);
            updatedHiddenWeights[2].Should().BeApproximately(0.00898010, 0.000000005);
            updatedHiddenWeights[3].Should().BeApproximately(0.00199127, 0.000000005);
            updatedHiddenWeights[4].Should().BeApproximately(0.00601746, 0.000000005);
            updatedHiddenWeights[5].Should().BeApproximately(0.00997380, 0.000000005);
            updatedHiddenWeights[6].Should().BeApproximately(0.00298917, 0.000000005);
            updatedHiddenWeights[7].Should().BeApproximately(0.00702166, 0.000000005);
            updatedHiddenWeights[8].Should().BeApproximately(0.01096751, 0.000000005);
            updatedHiddenWeights[9].Should().BeApproximately(0.00398707, 0.000000005);
            updatedHiddenWeights[10].Should().BeApproximately(0.00802586, 0.000000005);
            updatedHiddenWeights[11].Should().BeApproximately(0.01196121, 0.000000005);

            var updatedHiddenBiases = hiddenLayer.Biases.ToArray();
            updatedHiddenBiases[0].Should().BeApproximately(0.01299337, 0.000000005);
            updatedHiddenBiases[1].Should().BeApproximately(0.01399127, 0.000000005);
            updatedHiddenBiases[2].Should().BeApproximately(0.01498917, 0.000000005);
            updatedHiddenBiases[3].Should().BeApproximately(0.01598707, 0.000000005);
        }
    }
}