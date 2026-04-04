using System;

namespace LLM
{
    /// <summary>
    /// Common interface for a learnable parameter: weight matrix + gradient + Adam state.
    ///
    /// Provided implementations:
    ///   <see cref="LLM.Parameter"/>           – CPU (managed float[,] matrices)
    ///   <see cref="LLM_GPU.GpuParameter"/>    – GPU (ILGPU GPU-resident buffers)
    /// </summary>
    public interface IParameter : IDisposable
    {
        /// <summary>Number of rows in the weight matrix.</summary>
        int Rows { get; }

        /// <summary>Number of columns in the weight matrix.</summary>
        int Cols { get; }

        /// <summary>Reset the gradient to zero before a new forward/backward pass.</summary>
        void ZeroGrad();

        /// <summary>Apply one Adam optimiser step to this parameter.</summary>
        void Update(float lr, float beta1, float beta2, float eps, int adamStep);

        /// <summary>
        /// Return a flat copy of the gradient values (row-major).
        /// Used by gradient clipping to compute the global L2 norm.
        /// </summary>
        float[] GetGradientFlat();

        /// <summary>
        /// Scale every element of the gradient by <paramref name="scale"/>.
        /// Used by gradient clipping after the norm exceeds the threshold.
        /// </summary>
        void ScaleGradient(float scale);

        /// <summary>Return the weight values as a flat row-major array.</summary>
        float[] GetWeightsFlat();

        /// <summary>Overwrite the weight values from a flat row-major array.</summary>
        void LoadWeightsFlat(float[] data);

        /// <summary>Return the Adam first-moment (M) values as a flat row-major array.</summary>
        float[] GetMFlat();

        /// <summary>Overwrite the Adam first-moment (M) values from a flat row-major array.</summary>
        void LoadMFlat(float[] data);

        /// <summary>Return the Adam second-moment (V) values as a flat row-major array.</summary>
        float[] GetVFlat();

        /// <summary>Overwrite the Adam second-moment (V) values from a flat row-major array.</summary>
        void LoadVFlat(float[] data);
    }
}
