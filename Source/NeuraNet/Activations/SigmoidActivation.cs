using System;

namespace NeuraNet.Activations
{
    public class SigmoidActivation : Activation
    {
        protected override double Transform(double value)
        {
            return (1 / (1 + Math.Exp(-value)));
        }

        protected override double Derivative(double value)
        {
            return value * (1 - value);
        }
    }
}