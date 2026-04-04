using LLM_CPU;
using System;
using System.Collections.Generic;

namespace LLM
{
    /// <summary>
    /// Position-wise Feed-Forward Network (FFN).
    ///
    /// ─── Purpose ─────────────────────────────────────────────────────────────────
    /// The attention sub-layer lets tokens communicate; the FFN sub-layer lets each
    /// position transform its own representation independently.  Together they form
    /// the two halves of a transformer block.
    ///
    /// ─── Architecture ────────────────────────────────────────────────────────────
    /// Two linear layers with a GELU activation in between (as in GPT-2):
    ///
    ///   hidden = GELU( X · W1 + b1 )     [T × d_ff]   ← expand
    ///   output = hidden · W2 + b2         [T × D]      ← project back
    ///
    ///   W1 : [D   × d_ff]   (expand from d_model to d_ff = 4·d_model)
    ///   W2 : [d_ff × D  ]   (project back to d_model)
    ///
    /// "Position-wise" means the same two-layer MLP is applied to every token
    /// position independently (they share the same W1, W2 but don't communicate
    /// across positions here – that's the job of attention).
    ///
    /// ─── Backward pass ───────────────────────────────────────────────────────────
    ///   Let h  = X·W1 + b1       (pre-activation)
    ///   Let h' = GELU(h)         (post-activation)
    ///   Let y  = h'·W2 + b2      (output)
    ///
    ///   Given dL/dy:
    ///     dL/dW2 = h'ᵀ · dL/dy
    ///     dL/db2 = Σ_t dL/dy[t,:]
    ///     dL/dh' = dL/dy · W2ᵀ
    ///     dL/dh  = dL/dh' ⊙ GELU'(h)    (element-wise product with GELU derivative)
    ///     dL/dW1 = Xᵀ · dL/dh
    ///     dL/db1 = Σ_t dL/dh[t,:]
    ///     dL/dX  = dL/dh · W1ᵀ
    /// </summary>
    public sealed class FeedForward : ILayer<Matrix>
    {
        // ── configuration ─────────────────────────────────────────────────────────

        private readonly int _dModel;   // input/output dimension (D)
        private readonly int _dFF;      // hidden dimension (d_ff, typically 4·D)

        // ── learnable parameters ──────────────────────────────────────────────────

        /// <summary>First linear weight W1, shape [D × d_ff].  Expands the representation.</summary>
        public Parameter W1 { get; }

        /// <summary>First linear bias b1, shape [1 × d_ff].</summary>
        public Parameter B1 { get; }

        /// <summary>Second linear weight W2, shape [d_ff × D].  Projects back to d_model.</summary>
        public Parameter W2 { get; }

        /// <summary>Second linear bias b2, shape [1 × D].</summary>
        public Parameter B2 { get; }

        // ── forward-pass cache ────────────────────────────────────────────────────

        /// <summary>Input X from the last forward pass, shape [T × D].</summary>
        private Matrix? _cachedX;

        /// <summary>
        /// Pre-activation values h = X·W1 + b1, shape [T × d_ff].
        /// Needed in backward to compute GELU'(h).
        /// </summary>
        private Matrix? _cachedPreActivation;

        /// <summary>
        /// Post-activation values h' = GELU(h), shape [T × d_ff].
        /// Needed in backward to compute dL/dW2.
        /// </summary>
        private Matrix? _cachedPostActivation;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Build the FFN layer with Xavier-initialised weights and zero biases.
        /// </summary>
        /// <param name="dModel">Residual stream dimension D.</param>
        /// <param name="dFF">Hidden dimension d_ff (typically 4·dModel).</param>
        /// <param name="rng">Random number generator for initialisation.</param>
        public FeedForward(int dModel, int dFF, Random rng)
        {
            _dModel = dModel;
            _dFF    = dFF;

            // Xavier / Glorot uniform for the two projection matrices.
            // The scale √(6/(fanIn+fanOut)) keeps gradient variance stable.
            W1 = Parameter.Xavier(dModel, dFF,    rng);
            W2 = Parameter.Xavier(dFF,    dModel, rng);

            // Biases start at zero
            B1 = Parameter.Zeros(1, dFF);
            B2 = Parameter.Zeros(1, dModel);
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Apply the two-layer MLP to every token position.
        ///
        /// Input:  X of shape [T × D]
        /// Output: shape [T × D]
        ///
        /// Computation:
        ///   h  = X·W1 + b1        [T × d_ff]   (expand)
        ///   h' = GELU(h)          [T × d_ff]   (non-linearity)
        ///   y  = h'·W2 + b2       [T × D]      (project back)
        /// </summary>
        public Matrix Forward(Matrix x)
        {
            _cachedX = x;

            // ── Layer 1: expand ────────────────────────────────────────────────────
            // h = x · W1 + b1   shape [T × d_ff]
            float[] b1Arr = GetBias(B1, _dFF);
            Matrix pre = Matrix.Dot(x, W1.Weight).AddBias(b1Arr);
            _cachedPreActivation = pre;

            // ── GELU activation ────────────────────────────────────────────────────
            Matrix post = pre.GELU();
            _cachedPostActivation = post;

            // ── Layer 2: project back ──────────────────────────────────────────────
            // y = post · W2 + b2   shape [T × D]
            float[] b2Arr = GetBias(B2, _dModel);
            Matrix output = Matrix.Dot(post, W2.Weight).AddBias(b2Arr);

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Backpropagate through the FFN.
        ///
        /// Accumulates gradients into W1, B1, W2, B2 and returns dL/dX.
        ///
        /// Full derivation:
        ///   y  = h'·W2 + b2         → dW2 = h'ᵀ·dY,   db2 = Σ dY,  dh' = dY·W2ᵀ
        ///   h' = GELU(h)             → dh  = dh' ⊙ GELU'(h)
        ///   h  = X·W1 + b1           → dW1 = Xᵀ·dh,   db1 = Σ dh,   dX = dh·W1ᵀ
        /// </summary>
        /// <param name="dOut">Upstream gradient dL/dy, shape [T × D].</param>
        /// <returns>Gradient dL/dX, shape [T × D].</returns>
        public Matrix Backward(Matrix dOut)
        {
            ArgumentNullException.ThrowIfNull(dOut);
            if (_cachedX is null || _cachedPreActivation is null || _cachedPostActivation is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int T = _cachedX.Rows;

            // ── Layer 2 backward ──────────────────────────────────────────────────
            // y = h' · W2 + b2
            // dL/dW2 = h'ᵀ · dOut
            W2.Gradient.AddInPlace(Matrix.Dot(_cachedPostActivation.Transpose(), dOut));
            // dL/dB2 = Σ_t dOut[t,:]
            float[] db2 = dOut.SumOverRows();
            for (int j = 0; j < _dModel; j++) B2.Gradient.Data[0, j] += db2[j];

            // dL/dh' = dOut · W2ᵀ   [T × d_ff]
            Matrix dPost = Matrix.Dot(dOut, W2.Weight.Transpose());

            // ── GELU backward ─────────────────────────────────────────────────────
            // h' = GELU(h)
            // dL/dh = dL/dh' ⊙ GELU'(h)   (element-wise)
            Matrix geluGrad = _cachedPreActivation.GELUGrad();   // [T × d_ff]
            Matrix dPre     = Matrix.Mul(dPost, geluGrad);        // [T × d_ff]

            // ── Layer 1 backward ──────────────────────────────────────────────────
            // h = X · W1 + b1
            // dL/dW1 = Xᵀ · dPre
            W1.Gradient.AddInPlace(Matrix.Dot(_cachedX.Transpose(), dPre));
            // dL/dB1 = Σ_t dPre[t,:]
            float[] db1 = dPre.SumOverRows();
            for (int j = 0; j < _dFF; j++) B1.Gradient.Data[0, j] += db1[j];

            // dL/dX = dPre · W1ᵀ   [T × D]
            Matrix dX = Matrix.Dot(dPre, W1.Weight.Transpose());

            return dX;
        }

        // ── helpers ───────────────────────────────────────────────────────────────

        private static float[] GetBias(Parameter p, int len)
        {
            float[] bias = new float[len];
            for (int j = 0; j < len; j++) bias[j] = p.Weight.Data[0, j];
            return bias;
        }

        // ── parameter access ──────────────────────────────────────────────────────

        /// <summary>Yield all learnable parameters for the optimiser.</summary>
        public IEnumerable<IParameter> Parameters()
        {
            yield return W1; yield return B1;
            yield return W2; yield return B2;
        }

        public void Dispose()
        {
            W1.Dispose();
            B1.Dispose();
            W2.Dispose();
            B2.Dispose();
        }

        public override string ToString() =>
            $"FeedForward(d_model={_dModel}, d_ff={_dFF})";
    }
}
