using System.Collections.Generic;

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
            var network = new NeuralNetworkBuilder()
                .Using(new TwoLayerNetworkProvider())
                .Build();

            // Act
            double cost = network.Train(new[]
            {
                new TrainingExample(new[] { 1.0, -2.0, 3.0 }, new[] { 0.1234, 0.8766 })
            }, 1);

            // Assert
            cost.Should().BeApproximately(0.1418, 0.00005);
        }
    }
}