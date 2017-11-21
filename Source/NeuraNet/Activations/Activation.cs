using MathNet.Numerics.LinearAlgebra;

namespace NeuraNet.Activations
{
    public abstract class Activation : IActivation
    {
        public abstract string Name { get; }

        public Vector<double> Transform(Vector<double> values)
        {
            return values.Map(Transform, Zeros.Include);
        }

        public Vector<double> Derivative(Vector<double> values)
        {
            return values.Map(Derivative, Zeros.Include);
        }

        protected abstract double Transform(double value);

        protected abstract double Derivative(double value);
    }
}
