using System;

namespace NeuraNet.Activations
{
    public class HyperbolicTangentActivation : Activation
    {
        public override string Name => "Tanh";

        protected override double Transform(double value)
        {
            return Math.Tanh(value);
        }

        protected override double Derivative(double value)
        {
            return (1 - value) * (1 + value);
        }
    }
}
