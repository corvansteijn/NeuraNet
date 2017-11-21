using System;
using System.Collections.Generic;
using System.Linq;

using NeuraNet.Activations;
using Newtonsoft.Json;

namespace NeuraNet.Serialization
{
    public class NetworkJsonConverter
    {
        public string Serialize(NeuralNetwork network)
        {
            var jsonObject = new NetworkJson();
            foreach (Layer layer in network.GetLayers())
            {
                var layerJson = new LayerJson()
                {
                    Weights = layer.Weights.ToArray(),
                    Biases = layer.Biases.ToArray(),
                    Activation = GetActivationName(layer.OutputActivation)
                };

                jsonObject.Add(layerJson);
            }

            return JsonConvert.SerializeObject(jsonObject, Formatting.Indented);
        }

        public NeuralNetwork Deserialize(string json)
        {
            var networkJson = JsonConvert.DeserializeObject<NetworkJson>(json);
            var layers = networkJson.Layers
                .Select(layerJson => new Layer(layerJson.Weights, layerJson.Biases, GetActivation(layerJson.Activation)))
                .ToList();

            ConnectLayers(layers);

            return new NeuralNetwork(layers);
        }

        private IActivation GetActivation(string activationName)
        {
            switch (activationName)
            {
                case "Sigmoid":
                    return new SigmoidActivation();

                case "Tanh":
                    return new HyperbolicTangentActivation();

                case "ReLU":
                    return new RectifiedLinearUnitActivation();

                case "Softplus":
                    return new SoftplusActivation();

                default:
                    throw new ArgumentException($"{activationName} is not a known activation function");
            }
        }

        private string GetActivationName(IActivation activationFunction)
        {
            if (activationFunction is SigmoidActivation)
            {
                return "Sigmoid";
            }

            if (activationFunction is HyperbolicTangentActivation)
            {
                return "Tanh";
            }

            if (activationFunction is RectifiedLinearUnitActivation)
            {
                return "ReLU";
            }

            if (activationFunction is SoftplusActivation)
            {
                return "Softplus";
            }

            throw new ArgumentException($"{activationFunction} is not a known name for an activation function");
        }

        // REFACTOR: This connecting of layers is duplicated in NetworkLayoutProvider and here.
        private void ConnectLayers(List<Layer> layers)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                Layer previous = (i > 0) ? layers[i - 1] : null;
                Layer next = (i < (layers.Count - 1)) ? layers[i + 1] : null;

                layers[i].ConnectTo(previous, next);
            }
        }
    }
}
