using LLM_CPU;
using System;
using System.Threading.Tasks;

namespace LLM
{
    /// <summary>
    /// Layer Normalisation (Ba et al., 2016).
    ///
    /// ─── What it does ────────────────────────────────────────────────────────────
    /// For each position in the sequence independently, layer norm normalises the
    /// d_model-dimensional feature vector to have zero mean and unit variance, then
    /// applies a learnable per-feature scale (γ) and shift (β):
    ///
    ///   μ     = (1/D) · Σ_d x_d                ← mean over feature dimension
    ///   σ²    = (1/D) · Σ_d (x_d − μ)²         ← variance over feature dimension
    ///   x̂_d  = (x_d − μ) / √(σ² + ε)          ← normalised value
    ///   y_d   = γ_d · x̂_d + β_d               ← learned rescaling
    ///
    /// ─── Why it helps ────────────────────────────────────────────────────────────
    /// • Keeps activations in a well-behaved range as depth increases.
    /// • Reduces sensitivity to weight initialisation.
    /// • Unlike Batch Norm, it normalises over features (not the batch), so it works
    ///   the same at batch size 1 and is trivially auto-regressive.
    ///
    /// ─── Placement ───────────────────────────────────────────────────────────────
    /// We use pre-norm (norm applied before attention/FFN, not after), as in GPT-2.
    /// Pre-norm stabilises training for deep models.
    ///
    /// ─── Backward pass derivation ────────────────────────────────────────────────
    /// Given upstream gradient dL/dy, and letting g_d = dL/dy_d · γ_d:
    ///
    ///   dL/dγ_d = Σ_t  dL/dy[t,d] · x̂[t,d]
    ///   dL/dβ_d = Σ_t  dL/dy[t,d]
    ///
    /// For each position t (treating it as a D-dimensional normalisation):
    ///   dL/dx[t,d] = (1/σ_t) · ( g[t,d] − mean_d(g[t,:]) − x̂[t,d]·mean_d(g[t,:]·x̂[t,:]) )
    ///
    /// where g[t,d] = dL/dy[t,d] · γ_d   (upstream gradient after the γ multiplication).
    /// </summary>
    public sealed class LayerNorm : ILayer<Matrix>
    {
        // ── configuration ─────────────────────────────────────────────────────────

        /// <summary>Feature dimension D being normalised (= EmbeddingDim).</summary>
        private readonly int _dim;

        /// <summary>Small constant ε to avoid dividing by zero when variance is 0.</summary>
        private const float Eps = 1e-5f;

        // ── learnable parameters ──────────────────────────────────────────────────

        /// <summary>
        /// Per-feature scale γ (gamma).
        /// Shape: [1 × dim].  Initialised to 1 so that the layer starts as identity.
        /// </summary>
        public Parameter Gamma { get; }

        /// <summary>
        /// Per-feature shift β (beta).
        /// Shape: [1 × dim].  Initialised to 0.
        /// </summary>
        public Parameter Beta { get; }

        // ── forward-pass cache (for backward) ────────────────────────────────────

        /// <summary>
        /// Normalised values x̂, shape [seqLen × dim].
        /// Cached because the backward pass needs x̂ to compute gradients for γ
        /// and for x.
        /// </summary>
        private Matrix? _cachedXHat;

        /// <summary>
        /// Per-position standard deviation σ_t = √(σ²_t + ε), length = seqLen.
        /// Cached to avoid recomputing the square root in the backward pass.
        /// </summary>
        private float[]? _cachedSigma;

        /// <summary>Sequence length from the last forward call.</summary>
        private int _cachedSeqLen;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Initialise layer norm with dim-dimensional γ=1 and β=0.
        /// </summary>
        /// <param name="dim">Feature dimension (EmbeddingDim).</param>
        public LayerNorm(int dim)
        {
            _dim  = dim;
            Gamma = Parameter.Ones(1, dim);   // γ = 1 → identity at initialisation
            Beta  = Parameter.Zeros(1, dim);  // β = 0
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Apply layer normalisation to each row of the input independently.
        ///
        /// Input shape:  [seqLen × dim]
        /// Output shape: [seqLen × dim]   (same shape)
        /// </summary>
        public Matrix Forward(Matrix x)
        {
            ArgumentNullException.ThrowIfNull(x);
            int T = x.Rows;   // sequence length (number of positions to normalise)
            int D = x.Cols;   // feature dimension

            if (D != _dim)
                throw new ArgumentException($"LayerNorm dim mismatch: expected {_dim}, got {D}");

            var xHat   = new Matrix(T, D);
            var sigma  = new float[T];
            var output = new Matrix(T, D);

            // Each position t is normalised independently — fully parallel.
            Parallel.For(0, T, t =>
            {
                // ── Step 1: compute mean μ ─────────────────────────────────────
                float mean = 0f;
                for (int d = 0; d < D; d++)
                    mean += x.Data[t, d];
                mean /= D;

                // ── Step 2: compute variance σ² ────────────────────────────────
                float variance = 0f;
                for (int d = 0; d < D; d++)
                {
                    float diff = x.Data[t, d] - mean;
                    variance += diff * diff;
                }
                variance /= D;

                // ── Step 3: σ = √(σ² + ε) ─────────────────────────────────────
                float sig = MathF.Sqrt(variance + Eps);
                sigma[t] = sig;

                // ── Step 4: normalise, cache x̂, apply γ and β ─────────────────
                for (int d = 0; d < D; d++)
                {
                    float normalised   = (x.Data[t, d] - mean) / sig;
                    xHat.Data[t, d]    = normalised;
                    output.Data[t, d]  = Gamma.Weight.Data[0, d] * normalised
                                       + Beta.Weight.Data[0, d];
                }
            });

            _cachedXHat   = xHat;
            _cachedSigma  = sigma;
            _cachedSeqLen = T;

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Backpropagate through layer normalisation.
        ///
        /// Accumulates gradients into Gamma.Gradient and Beta.Gradient and returns
        /// the gradient with respect to the input x.
        ///
        /// Key formula (per position t, derived in the class doc above):
        ///
        ///   g[t,d]  = dL/dy[t,d] · γ_d
        ///   ḡ[t]    = (1/D) · Σ_d g[t,d]        ← mean of g over features
        ///   ḡx[t]   = (1/D) · Σ_d g[t,d]·x̂[t,d] ← mean of g·x̂ over features
        ///
        ///   dL/dx[t,d] = (1/σ_t) · ( g[t,d] − ḡ[t] − x̂[t,d]·ḡx[t] )
        /// </summary>
        /// <param name="dOut">
        /// Upstream gradient dL/dy, shape [seqLen × dim].
        /// </param>
        /// <returns>
        /// Gradient dL/dx, shape [seqLen × dim].
        /// </returns>
        public Matrix Backward(Matrix dOut)
        {
            if (_cachedXHat is null || _cachedSigma is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int T = _cachedSeqLen;
            int D = _dim;

            var dX = new Matrix(T, D);

            // Each position t is independent for dX.
            // γ/β gradients sum over t, so we accumulate into thread-local arrays
            // and merge at the end to avoid write races.
            Parallel.For(0, T,
                () => (new float[D], new float[D]),   // (localDGamma, localDBeta)
                (t, _, local) =>
                {
                    float sig    = _cachedSigma[t];
                    float gMean  = 0f;
                    float gXMean = 0f;

                    for (int d = 0; d < D; d++)
                    {
                        float g = dOut.Data[t, d] * Gamma.Weight.Data[0, d];
                        gMean  += g;
                        gXMean += g * _cachedXHat.Data[t, d];
                        local.Item1[d] += dOut.Data[t, d] * _cachedXHat.Data[t, d]; // dγ
                        local.Item2[d] += dOut.Data[t, d];                            // dβ
                    }
                    gMean  /= D;
                    gXMean /= D;

                    for (int d = 0; d < D; d++)
                    {
                        float g = dOut.Data[t, d] * Gamma.Weight.Data[0, d];
                        dX.Data[t, d] = (g - gMean - _cachedXHat.Data[t, d] * gXMean) / sig;
                    }

                    return local;
                },
                local =>
                {
                    // Merge thread-local γ/β gradient contributions into shared params.
                    // Lock covers only D additions — cheap relative to per-position work.
                    lock (Gamma.Gradient)
                    {
                        for (int d = 0; d < D; d++)
                        {
                            Gamma.Gradient.Data[0, d] += local.Item1[d];
                            Beta.Gradient.Data[0, d]  += local.Item2[d];
                        }
                    }
                });

            return dX;
        }

        // ── parameter access ──────────────────────────────────────────────────────

        /// <summary>Yield γ and β for the optimiser.</summary>
        public System.Collections.Generic.IEnumerable<IParameter> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void Dispose()
        {
            Gamma.Dispose();
            Beta.Dispose();
        }

        public override string ToString() => $"LayerNorm(dim={_dim})";
    }
}
