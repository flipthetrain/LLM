using System;
using ILGPU;
using ILGPU.Runtime;
using LLM;

using A1 = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace LLM_GPU
{
    /// <summary>
    /// GPU-resident learnable parameter: weight + gradient + Adam state.
    ///
    /// All four buffers live on the GPU.  The Adam update and gradient clipping
    /// are performed entirely on the GPU (no round-trip to CPU per step).
    /// Weight initialisation is done on the CPU and uploaded once.
    /// </summary>
    internal sealed class GpuParameter : IParameter
    {
        // ── GPU buffers ───────────────────────────────────────────────────────
        public readonly GpuMatrix Weight;
        public readonly GpuMatrix Gradient;
        public readonly GpuMatrix M;   // Adam first moment
        public readonly GpuMatrix V;   // Adam second moment

        public int Rows => Weight.Rows;
        public int Cols => Weight.Cols;
        private int N   => Rows * Cols;

        // ── kernel delegates ──────────────────────────────────────────────────
        private static Action<Index1D, A1, A1, A1, A1, float, float, float, float, float, float, int>? _adamKernel;
        private static Action<Index1D, A1, float, int>?                                                 _clipKernel;

        // ── constructors ─────────────────────────────────────────────────────
        /// <summary>
        /// Allocate buffers and initialise Weight with N(0, initStd).
        /// Gradient, M, V are zero-initialised.
        /// </summary>
        public GpuParameter(int rows, int cols, Random rng, float initStd = 0.02f)
        {
            Weight   = new GpuMatrix(rows, cols);
            Gradient = new GpuMatrix(rows, cols);
            M        = new GpuMatrix(rows, cols);
            V        = new GpuMatrix(rows, cols);

            // Gradient/M/V stay as zero (Allocate1D zeroes memory on CUDA;
            // we explicitly zero on other backends for safety).
            Gradient.Zero();
            M.Zero();
            V.Zero();

            if (initStd != 0f)
            {
                var flat = new float[rows * cols];
                NormalInit(rng, flat, 0f, initStd);
                Weight.UploadFlat(flat);
            }
            else
            {
                Weight.Zero();
            }
        }

        private static void NormalInit(Random rng, float[] buf, float mean, float std)
        {
            for (int i = 0; i < buf.Length; i++)
            {
                double u1 = 1.0 - rng.NextDouble();
                double u2 = 1.0 - rng.NextDouble();
                double z  = Math.Sqrt(-2.0 * Math.Log(u1))
                            * Math.Cos(2.0 * Math.PI * u2);
                buf[i] = mean + (float)z * std;
            }
        }

        /// <summary>Xavier / Glorot uniform initialisation.</summary>
        public static GpuParameter Xavier(int rows, int cols, Random rng)
        {
            var p     = new GpuParameter(rows, cols, rng, initStd: 0f);
            float lim = MathF.Sqrt(6f / (rows + cols));
            var flat  = new float[rows * cols];
            for (int i = 0; i < flat.Length; i++)
                flat[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * lim;
            p.Weight.UploadFlat(flat);
            return p;
        }

        public static GpuParameter Zeros(int rows, int cols)
        {
            return new GpuParameter(rows, cols, new Random(0), initStd: 0f);
        }

        public static GpuParameter Ones(int rows, int cols)
        {
            var p = Zeros(rows, cols);
            p.Weight.Fill(1f);
            return p;
        }

        // ── gradient management ───────────────────────────────────────────────
        public void ZeroGrad() => Gradient.Zero();

        // ── Adam update ───────────────────────────────────────────────────────
        /// <summary>
        /// Apply one Adam step entirely on the GPU.
        /// Also zeros the gradient buffer in the same kernel pass.
        /// </summary>
        public void Update(float lr, float beta1, float beta2, float eps, int adamStep)
        {
            float bc1 = 1f - MathF.Pow(beta1, adamStep);
            float bc2 = 1f - MathF.Pow(beta2, adamStep);

            _adamKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<
                    Index1D,
                    A1, A1, A1, A1,
                    float, float, float, float, float, float,
                    int>(Kernels.AdamKernel);

            _adamKernel(N,
                Weight.View, Gradient.View, M.View, V.View,
                lr, beta1, beta2, eps, bc1, bc2,
                N);
        }

        /// <inheritdoc/>
        public float[] GetGradientFlat() => Gradient.DownloadFlat();

        /// <inheritdoc/>
        public void ScaleGradient(float scale)
        {
            _clipKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, float, int>(
                    Kernels.GradClipScaleKernel);
            _clipKernel(N, Gradient.View, scale, N);
        }

        /// <inheritdoc/>
        public float[] GetWeightsFlat() => Weight.DownloadFlat();

        /// <inheritdoc/>
        public void LoadWeightsFlat(float[] data) => Weight.UploadFlat(data);

        /// <inheritdoc/>
        public float[] GetMFlat() => M.DownloadFlat();

        /// <inheritdoc/>
        public void LoadMFlat(float[] data) => M.UploadFlat(data);

        /// <inheritdoc/>
        public float[] GetVFlat() => V.DownloadFlat();

        /// <inheritdoc/>
        public void LoadVFlat(float[] data) => V.UploadFlat(data);

        public override string ToString() => $"GpuParameter[{Rows}×{Cols}]";

        public void Dispose()
        {
            Weight.Dispose();
            Gradient.Dispose();
            M.Dispose();
            V.Dispose();
        }
    }
}
