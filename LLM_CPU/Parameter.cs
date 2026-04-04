using LLM_CPU;
using System;

namespace LLM
{
    /// <summary>
    /// A learnable parameter – a weight matrix paired with its gradient and
    /// Adam optimiser state.
    ///
    /// Bundling weight + gradient + moments into one object keeps the optimiser
    /// update loop simple: iterate over a flat list of Parameter objects and call
    /// Update() on each one.
    ///
    /// Lifecycle:
    ///   1. Created by a layer (Embedding, MultiHeadAttention, …) during model init.
    ///   2. Forward pass reads Weight.Data.
    ///   3. Backward pass writes into Gradient.Data (accumulates via AddInPlace).
    ///   4. Optimiser calls Update() to apply Adam and then ZeroGrad() to reset.
    /// </summary>
    public sealed class Parameter : IParameter
    {
        // ── weight and gradient ───────────────────────────────────────────────────

        /// <summary>The actual learnable values, shape [Rows × Cols].</summary>
        public Matrix Weight { get; }

        /// <summary>
        /// Accumulated gradient for one mini-batch / sequence.
        /// Must be reset to zero before each forward/backward cycle by calling
        /// ZeroGrad().
        /// </summary>
        public Matrix Gradient { get; }

        // ── Adam optimiser state ──────────────────────────────────────────────────

        /// <summary>
        /// First moment estimate (exponential moving average of gradients).
        /// Also called the "momentum" term.
        ///   m_t = β₁·m_{t-1} + (1−β₁)·g_t
        /// Initialised to zero; bias-corrected before each parameter update.
        /// </summary>
        public Matrix M { get; }

        /// <summary>
        /// Second moment estimate (exponential moving average of squared gradients).
        /// Also called the "RMS" or "adaptive" term.
        ///   v_t = β₂·v_{t-1} + (1−β₂)·g_t²
        /// Initialised to zero; bias-corrected before each parameter update.
        /// </summary>
        public Matrix V { get; }

        // ── shape ─────────────────────────────────────────────────────────────────

        /// <summary>Number of rows in the weight matrix.</summary>
        public int Rows => Weight.Rows;

        /// <summary>Number of columns in the weight matrix.</summary>
        public int Cols => Weight.Cols;

        // ── constructors ─────────────────────────────────────────────────────────

        /// <summary>
        /// Create a parameter and initialise the weight from a normal distribution
        /// (mean=0, std=<paramref name="initStd"/>).  Gradient, M, V start at zero.
        /// </summary>
        public Parameter(int rows, int cols, Random rng, float initStd = 0.02f)
        {
            Weight   = new Matrix(rows, cols);
            Gradient = new Matrix(rows, cols);
            M        = new Matrix(rows, cols);
            V        = new Matrix(rows, cols);

            Weight.NormalInit(rng, mean: 0f, std: initStd);
        }

        /// <summary>
        /// Create a parameter and initialise with Xavier / Glorot uniform sampling.
        /// Preferred for attention projection matrices where fanIn and fanOut are known.
        /// </summary>
        public static Parameter Xavier(int rows, int cols, Random rng)
        {
            var p = new Parameter(rows, cols, rng, initStd: 0f); // skip normal init
            p.Weight.XavierInit(rng, fanIn: rows, fanOut: cols);
            return p;
        }

        /// <summary>
        /// Create a parameter whose weight is all zeros (used for bias vectors,
        /// layer-norm β, etc.).
        /// </summary>
        public static Parameter Zeros(int rows, int cols)
        {
            // Pass a dummy rng – normal init is called with std=0, giving 0 values.
            var p = new Parameter(rows, cols, new Random(0), initStd: 0f);
            return p;
        }

        /// <summary>
        /// Create a parameter whose weight is all ones (used for layer-norm γ so
        /// the network starts as the identity transform).
        /// </summary>
        public static Parameter Ones(int rows, int cols)
        {
            var p = Zeros(rows, cols);
            p.Weight.Fill(1f);
            return p;
        }

        // ── gradient management ───────────────────────────────────────────────────

        /// <summary>
        /// Reset the gradient to zero before the next forward/backward pass.
        /// Must be called at the start of every training step.
        /// </summary>
        public void ZeroGrad() => Gradient.Zero();

        // ── Adam update ───────────────────────────────────────────────────────────

        /// <summary>
        /// Apply one step of the Adam optimiser to this parameter.
        ///
        /// Adam (Adaptive Moment Estimation) combines momentum (first moment) with
        /// per-parameter adaptive learning rates (second moment):
        ///
        ///   m̂ = m / (1 − β₁ᵗ)   ← bias-corrected first moment
        ///   v̂ = v / (1 − β₂ᵗ)   ← bias-corrected second moment
        ///   θ ← θ − α · m̂ / (√v̂ + ε)
        ///
        /// The bias correction terms (1 − βᵗ) compensate for the zero initialisation
        /// of m and v: at t=1 they are small, so the correction is large, keeping the
        /// effective step size close to the learning rate.
        ///
        /// Reference: Kingma &amp; Ba (2015), "Adam: A Method for Stochastic Optimization."
        /// </summary>
        /// <param name="lr">Learning rate α (e.g. 3e-4).</param>
        /// <param name="beta1">Decay rate for first moment (default 0.9).</param>
        /// <param name="beta2">Decay rate for second moment (default 0.999).</param>
        /// <param name="eps">Small constant to prevent division by zero (default 1e-8).</param>
        /// <param name="adamStep">Global step count t, starting at 1.</param>
        public void Update(float lr, float beta1, float beta2, float eps, int adamStep)
        {
            // Bias-correction denominators
            float bc1 = 1f - MathF.Pow(beta1, adamStep);   // 1 − β₁ᵗ
            float bc2 = 1f - MathF.Pow(beta2, adamStep);   // 1 − β₂ᵗ

            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                {
                    float g = Gradient.Data[i, j];   // gradient for this element

                    // Update first moment (momentum)
                    M.Data[i, j] = beta1 * M.Data[i, j] + (1f - beta1) * g;

                    // Update second moment (adaptive scale)
                    V.Data[i, j] = beta2 * V.Data[i, j] + (1f - beta2) * g * g;

                    // Bias-corrected estimates
                    float mHat = M.Data[i, j] / bc1;
                    float vHat = V.Data[i, j] / bc2;

                    // Apply update: subtract a step proportional to m̂ / (√v̂ + ε)
                    Weight.Data[i, j] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                }
        }

        /// <inheritdoc/>
        public float[] GetGradientFlat()
        {
            var flat = new float[Rows * Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    flat[i * Cols + j] = Gradient.Data[i, j];
            return flat;
        }

        /// <inheritdoc/>
        public void ScaleGradient(float scale)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Gradient.Data[i, j] *= scale;
        }

        /// <inheritdoc/>
        public float[] GetWeightsFlat()
        {
            var flat = new float[Rows * Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    flat[i * Cols + j] = Weight.Data[i, j];
            return flat;
        }

        /// <inheritdoc/>
        public void LoadWeightsFlat(float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Weight.Data[i, j] = data[i * Cols + j];
        }

        /// <inheritdoc/>
        public float[] GetMFlat()
        {
            var flat = new float[Rows * Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    flat[i * Cols + j] = M.Data[i, j];
            return flat;
        }

        /// <inheritdoc/>
        public void LoadMFlat(float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    M.Data[i, j] = data[i * Cols + j];
        }

        /// <inheritdoc/>
        public float[] GetVFlat()
        {
            var flat = new float[Rows * Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    flat[i * Cols + j] = V.Data[i, j];
            return flat;
        }

        /// <inheritdoc/>
        public void LoadVFlat(float[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    V.Data[i, j] = data[i * Cols + j];
        }

        /// <summary>CPU parameter has no unmanaged resources.</summary>
        public void Dispose() { }

        public override string ToString() => $"Parameter[{Rows}×{Cols}]";
    }
}
