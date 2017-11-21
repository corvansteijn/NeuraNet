using System.Collections.Generic;

namespace NeuraNet.Serialization
{
    public class NetworkJson
    {
        public List<LayerJson> Layers { get; set; } = new List<LayerJson>();

        public void Add(LayerJson layer)
        {
            Layers.Add(layer);
        }
    }
}
