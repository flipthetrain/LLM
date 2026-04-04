using LLM;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LLM_CPU
{
    /// <summary>
    /// Multi-Head Causal Self-Attention (Vaswani et al., 2017 – "Attention Is All You Need").
    ///
    /// ─── Concept ─────────────────────────────────────────────────────────────────
    /// Self-attention allows every position in the sequence to "look at" every other
    /// position when computing its output.  By running h parallel attention "heads"
    /// in lower-dimensional subspaces and then concatenating, the model can
    /// simultaneously capture many different types of relationships.
    ///
    /// ─── Causal masking ──────────────────────────────────────────────────────────
    /// In a language model, position t may only attend to positions ≤ t (i.e. past
    /// and present, not future).  We enforce this by adding −∞ to all "future"
    /// attention logits before the softmax, making them zero probability.
    ///
    /// ─── Notation ────────────────────────────────────────────────────────────────
    ///   T   = sequence length
    ///   D   = d_model (EmbeddingDim)
    ///   H   = number of heads (NumHeads)
    ///   d_k = D / H  (HeadDim)
    ///
    /// ─── Forward pass ────────────────────────────────────────────────────────────
    ///   Q  = X · Wq  [T×D]       — queries
    ///   K  = X · Wk  [T×D]       — keys
    ///   V  = X · Wv  [T×D]       — values
    ///
    ///   For head h (h = 0…H-1), columns h·d_k … (h+1)·d_k:
    ///     Q_h, K_h, V_h  ∈ [T×d_k]
    ///     scores_h = Q_h · K_hᵀ / √d_k + mask   [T×T]
    ///     A_h      = softmax(scores_h)            [T×T]
    ///     out_h    = A_h · V_h                   [T×d_k]
    ///
    ///   out    = concat(out_0, …, out_{H-1})     [T×D]
    ///   result = out · Wo + bo                   [T×D]
    ///
    /// ─── Backward pass ───────────────────────────────────────────────────────────
    ///   (Detailed derivation is inline in the Backward() method.)
    /// </summary>
    public sealed class MultiHeadAttention : ILayer<Matrix>
    {
        // ── configuration ─────────────────────────────────────────────────────────

        private readonly TransformerConfig _cfg;

        /// <summary>Per-head dimension: d_k = EmbeddingDim / NumHeads.</summary>
        private readonly int _headDim;

        // ── projection parameters ─────────────────────────────────────────────────

        /// <summary>Query projection weight Wq, shape [D × D].</summary>
        public Parameter Wq { get; }
        /// <summary>Key projection weight Wk, shape [D × D].</summary>
        public Parameter Wk { get; }
        /// <summary>Value projection weight Wv, shape [D × D].</summary>
        public Parameter Wv { get; }
        /// <summary>Output projection weight Wo, shape [D × D].</summary>
        public Parameter Wo { get; }

        /// <summary>Query bias bq, shape [1 × D].</summary>
        public Parameter Bq { get; }
        /// <summary>Key bias bk, shape [1 × D].</summary>
        public Parameter Bk { get; }
        /// <summary>Value bias bv, shape [1 × D].</summary>
        public Parameter Bv { get; }
        /// <summary>Output bias bo, shape [1 × D].</summary>
        public Parameter Bo { get; }

        // ── forward-pass cache ────────────────────────────────────────────────────

        /// <summary>Input X from the last forward pass, shape [T × D].</summary>
        private Matrix? _cachedX;

        /// <summary>Full Q, K, V matrices [T × D] before head splitting.</summary>
        private Matrix? _cachedQ, _cachedK, _cachedV;

        /// <summary>Per-head query slices, each [T × d_k].</summary>
        private Matrix[]? _cachedQh;
        /// <summary>Per-head key slices, each [T × d_k].</summary>
        private Matrix[]? _cachedKh;
        /// <summary>Per-head value slices, each [T × d_k].</summary>
        private Matrix[]? _cachedVh;

        /// <summary>
        /// Per-head attention weight matrices after softmax, each [T × T].
        /// These are probabilities (all non-negative, each row sums to 1).
        /// Needed in the backward pass to compute dL/d(pre-softmax scores).
        /// </summary>
        private Matrix[]? _cachedAttnWeights;

        /// <summary>Per-head attention outputs, each [T × d_k].</summary>
        private Matrix[]? _cachedHeadOutputs;

        /// <summary>Concatenated head outputs before Wo, shape [T × D].</summary>
        private Matrix? _cachedConcat;

        // ── causal mask ───────────────────────────────────────────────────────────

        /// <summary>
        /// Causal (lower-triangular) mask for sequence length T.
        /// mask[i, j] = 0      if j ≤ i  (position i can attend to j)
        /// mask[i, j] = −∞     if j > i  (future positions are blocked)
        ///
        /// Adding this to the raw attention scores before softmax zeroes out
        /// attention to future tokens (exp(−∞) = 0).
        ///
        /// Recomputed whenever the sequence length changes.
        /// </summary>
        private Matrix? _causalMask;
        private int     _cachedMaskLen;

        // ── RoPE frequency tables ─────────────────────────────────────────────────

        /// <summary>Whether RoPE is enabled (set from config).</summary>
        private readonly bool _useRoPE;

        /// <summary>
        /// cos(θ_i · pos) table, shape [ContextLength × HeadDim/2].
        /// Pre-computed once so forward/backward pay no trig cost.
        /// </summary>
        private readonly float[,]? _ropeFreqCos;

        /// <summary>sin(θ_i · pos) table, shape [ContextLength × HeadDim/2].</summary>
        private readonly float[,]? _ropeFreqSin;

        // ── KV-Cache (inference only) ─────────────────────────────────────────────

        /// <summary>Per-head key cache, each [ContextLength × dk]. Null until first ForwardCached call.</summary>
        private Matrix[]? _kvCacheK;

        /// <summary>Per-head value cache, each [ContextLength × dk].</summary>
        private Matrix[]? _kvCacheV;

        /// <summary>Number of tokens currently stored in the KV-cache.</summary>
        private int _kvLen;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Create a multi-head attention layer and initialise all weight matrices.
        /// </summary>
        public MultiHeadAttention(TransformerConfig cfg, Random rng)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            _cfg     = cfg;
            _headDim = cfg.HeadDim;
            _useRoPE = cfg.UseRoPE;

            int D = cfg.EmbeddingDim;

            // Projection weights: [D × D]  (small std to keep outputs near zero initially)
            Wq = new Parameter(D, D, rng, initStd: 0.02f);
            Wk = new Parameter(D, D, rng, initStd: 0.02f);
            Wv = new Parameter(D, D, rng, initStd: 0.02f);
            Wo = new Parameter(D, D, rng, initStd: 0.02f);

            // Biases: [1 × D], zero-initialised
            Bq = Parameter.Zeros(1, D);
            Bk = Parameter.Zeros(1, D);
            Bv = Parameter.Zeros(1, D);
            Bo = Parameter.Zeros(1, D);

            // ── RoPE frequency tables ──────────────────────────────────────────────
            if (_useRoPE)
            {
                int halfDk = _headDim / 2;
                int maxLen = cfg.ContextLength;
                _ropeFreqCos = new float[maxLen, halfDk];
                _ropeFreqSin = new float[maxLen, halfDk];
                for (int pos = 0; pos < maxLen; pos++)
                {
                    for (int i = 0; i < halfDk; i++)
                    {
                        // θ_i(pos) = pos / 10000^(2i / HeadDim)
                        float theta = pos / MathF.Pow(10000f, 2f * i / _headDim);
                        _ropeFreqCos[pos, i] = MathF.Cos(theta);
                        _ropeFreqSin[pos, i] = MathF.Sin(theta);
                    }
                }
            }
        }

        // ── causal mask construction ──────────────────────────────────────────────

        /// <summary>
        /// Build or return the cached causal mask for a given sequence length.
        /// The mask is reused across forward calls with the same length.
        /// </summary>
        private Matrix GetCausalMask(int T)
        {
            if (_causalMask != null && _cachedMaskLen == T)
                return _causalMask;

            _causalMask = new Matrix(T, T);
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    // Allow attending to positions ≤ i; block positions > i.
                    _causalMask.Data[i, j] = (j <= i) ? 0f : float.NegativeInfinity;

            _cachedMaskLen = T;
            return _causalMask;
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Compute multi-head causal self-attention.
        ///
        /// Input:  X of shape [T × D]
        /// Output: same shape [T × D]
        ///
        /// Each output position t is a weighted sum of value projections of all
        /// positions s ≤ t, with weights determined by the query-key similarity.
        /// </summary>
        public Matrix Forward(Matrix x)
        {
            ArgumentNullException.ThrowIfNull(x);
            int T = x.Rows;   // sequence length
            int D = _cfg.EmbeddingDim;
            int H = _cfg.NumHeads;
            int dk = _headDim;

            // Cache input for backward
            _cachedX = x;

            // ── 1. Project X to Q, K, V with learnable weights + biases ──────────
            // Q = X·Wq + bq,  K = X·Wk + bk,  V = X·Wv + bv
            // Shape of each: [T × D]
            Matrix Q = Matrix.Dot(x, Wq.Weight).AddBias(GetBias(Bq));
            Matrix K = Matrix.Dot(x, Wk.Weight).AddBias(GetBias(Bk));
            Matrix V = Matrix.Dot(x, Wv.Weight).AddBias(GetBias(Bv));

            _cachedQ = Q;
            _cachedK = K;
            _cachedV = V;

            // ── 2. Get the causal mask for this sequence length ───────────────────
            Matrix mask = GetCausalMask(T);

            // ── 3. Per-head attention ─────────────────────────────────────────────
            // Allocate per-head caches
            _cachedQh          = new Matrix[H];
            _cachedKh          = new Matrix[H];
            _cachedVh          = new Matrix[H];
            _cachedAttnWeights = new Matrix[H];
            _cachedHeadOutputs = new Matrix[H];

            var headOutputs = new Matrix[H];  // will be concatenated

            float scale = _cfg.AttentionScale;   // 1/√d_k

            // Each head reads non-overlapping column slices of Q/K/V and writes to its
            // own index in the per-head arrays — no shared writes, fully parallel.
            Parallel.For(0, H, h =>
            {
                // Slice columns for this head: columns [h·dk, (h+1)·dk)
                int colStart = h * dk;
                int colEnd   = (h + 1) * dk;

                Matrix Qh = Q.SliceCols(colStart, colEnd);   // [T × dk]
                Matrix Kh = K.SliceCols(colStart, colEnd);   // [T × dk]
                Matrix Vh = V.SliceCols(colStart, colEnd);   // [T × dk]

                // Apply RoPE in-place at positions 0..T-1 (training always starts at 0)
                if (_useRoPE)
                {
                    ApplyRoPE(Qh, 0);
                    ApplyRoPE(Kh, 0);
                }

                _cachedQh[h] = Qh;
                _cachedKh[h] = Kh;
                _cachedVh[h] = Vh;

                // ── Scaled dot-product attention ──────────────────────────────────
                // scores = Q_h · K_hᵀ / √d_k   [T × T]
                Matrix scores = Matrix.Dot(Qh, Kh.Transpose()).Scale(scale);

                // Add causal mask: future positions become −∞ → exp(−∞) = 0
                scores = Matrix.Add(scores, mask);

                // Softmax converts scores into a proper probability distribution
                Matrix attn = scores.Softmax();   // [T × T]

                // ── Weighted sum of values ────────────────────────────────────────
                // out_h = A_h · V_h   [T × dk]
                Matrix outH = Matrix.Dot(attn, Vh);

                // Cache everything needed for backward
                _cachedAttnWeights[h] = attn;
                _cachedHeadOutputs[h] = outH;
                headOutputs[h]        = outH;
            });

            // ── 4. Concatenate head outputs and project ───────────────────────────
            // concat: [T × D],  result = concat · Wo + bo
            Matrix concat = Matrix.ConcatCols(headOutputs);
            _cachedConcat = concat;

            Matrix result = Matrix.Dot(concat, Wo.Weight).AddBias(GetBias(Bo));

            return result;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Backpropagate through multi-head attention.
        ///
        /// Given dL/d(result) (the upstream gradient from the layer above),
        /// compute and accumulate:
        ///   • dL/dWo, dL/dBo
        ///   • dL/dWq, dL/dWk, dL/dWv  (and their biases)
        ///
        /// Return dL/dX so the gradient can flow back to the layer norm.
        ///
        /// Step-by-step derivation:
        ///
        ///   result = concat · Wo + bo
        ///   → dL/dWo    = concatᵀ · dResult
        ///   → dL/dBo    = Σ_t dResult[t, :]   (sum over sequence)
        ///   → dL/dconcat = dResult · Woᵀ
        ///
        ///   For each head h:
        ///     out_h = A_h · V_h
        ///     → dL/dA_h   = dConcat_h · V_hᵀ
        ///     → dL/dV_h   = A_hᵀ · dConcat_h
        ///
        ///     A_h = softmax(scores_h)  [causal mask adds a constant → no grad]
        ///     → dL/dScores_h = softmax_backward(A_h, dA_h)
        ///
        ///     scores_h = Q_h · K_hᵀ / √d_k
        ///     → dL/dQ_h = dScores_h · K_h / √d_k
        ///     → dL/dK_h = dScores_hᵀ · Q_h / √d_k
        ///
        ///   Accumulate head-sliced gradients into full [T×D] dQ, dK, dV:
        ///     dQ[:, h·dk:(h+1)·dk] += dQ_h
        ///     dK[:, h·dk:(h+1)·dk] += dK_h
        ///     dV[:, h·dk:(h+1)·dk] += dV_h
        ///
        ///   Q = X · Wq + bq, K = X · Wk + bk, V = X · Wv + bv
        ///   → dL/dWq = Xᵀ · dQ,  dL/dBq = Σ_t dQ[t,:]
        ///   → dL/dWk = Xᵀ · dK,  dL/dBk = Σ_t dK[t,:]
        ///   → dL/dWv = Xᵀ · dV,  dL/dBv = Σ_t dV[t,:]
        ///   → dL/dX  = dQ · Wqᵀ + dK · Wkᵀ + dV · Wvᵀ
        /// </summary>
        public Matrix Backward(Matrix dResult)
        {
            ArgumentNullException.ThrowIfNull(dResult);
            if (_cachedX is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int T  = _cachedX.Rows;
            int D  = _cfg.EmbeddingDim;
            int H  = _cfg.NumHeads;
            int dk = _headDim;

            // ── 4. Gradient through output projection ─────────────────────────────
            // result = concat · Wo + bo
            // dL/dWo    = concatᵀ · dResult
            // dL/dBo    = sum over rows of dResult
            // dL/dconcat = dResult · Woᵀ
            Wo.Gradient.AddInPlace(Matrix.Dot(_cachedConcat!.Transpose(), dResult));
            float[] dBoArr = dResult.SumOverRows();
            for (int j = 0; j < D; j++) Bo.Gradient.Data[0, j] += dBoArr[j];

            Matrix dConcat = Matrix.Dot(dResult, Wo.Weight.Transpose());   // [T × D]

            // ── 3. Per-head backward ──────────────────────────────────────────────
            // dQ, dK, dV are the full [T × D] gradient matrices for the projections.
            var dQ = new Matrix(T, D);
            var dK = new Matrix(T, D);
            var dV = new Matrix(T, D);

            float scale = _cfg.AttentionScale;   // 1/√d_k

            // Each head writes to columns [h·dk, (h+1)·dk) of dQ/dK/dV — non-overlapping,
            // so AddSliceCols is safe to call in parallel without locks.
            Parallel.For(0, H, h =>
            {
                int colStart = h * dk;

                // Gradient of the head slice of dConcat → dOut_h [T × dk]
                Matrix dOutH = dConcat.SliceCols(colStart, colStart + dk);

                // out_h = A_h · V_h
                // dL/dA_h  = dOut_h · V_hᵀ     [T × T]
                // dL/dV_h  = A_hᵀ · dOut_h     [T × dk]
                Matrix dAh = Matrix.Dot(dOutH, _cachedVh![h].Transpose());
                Matrix dVh = Matrix.Dot(_cachedAttnWeights![h].Transpose(), dOutH);

                // A_h = softmax(scores_h + mask)
                // The mask has no parameters (it is a constant), so the gradient
                // flows straight through to the pre-mask scores.
                // dL/dScores_h = softmax_backward(A_h, dA_h)   [T × T]
                Matrix dScores = Matrix.SoftmaxBackward(_cachedAttnWeights[h], dAh);

                // scores_h = Q_h · K_hᵀ · scale
                // dL/dQ_h = dScores · K_h · scale          [T × dk]
                // dL/dK_h = dScoresᵀ · Q_h · scale         [T × dk]
                Matrix dQh = Matrix.Dot(dScores, _cachedKh![h]).Scale(scale);
                Matrix dKh = Matrix.Dot(dScores.Transpose(), _cachedQh![h]).Scale(scale);

                // RoPE is an orthogonal transform; its gradient is the inverse rotation.
                if (_useRoPE)
                {
                    ApplyRoPEInverse(dQh, 0);
                    ApplyRoPEInverse(dKh, 0);
                }

                // Scatter gradients back into the full dQ, dK, dV matrices.
                // Writes target columns [h·dk, (h+1)·dk) — no overlap between heads.
                dQ.AddSliceCols(dQh, colStart);
                dK.AddSliceCols(dKh, colStart);
                dV.AddSliceCols(dVh, colStart);
            });

            // ── 1-2. Gradient through Q, K, V projection ─────────────────────────
            // Q = X·Wq + bq  →  dWq = Xᵀ·dQ,  dBq = Σ dQ,  dX_contribution = dQ·Wqᵀ
            Matrix Xt = _cachedX.Transpose();   // [D × T]

            Wq.Gradient.AddInPlace(Matrix.Dot(Xt, dQ));
            Wk.Gradient.AddInPlace(Matrix.Dot(Xt, dK));
            Wv.Gradient.AddInPlace(Matrix.Dot(Xt, dV));

            // Bias gradients (sum over sequence positions)
            float[] dBqArr = dQ.SumOverRows();
            float[] dBkArr = dK.SumOverRows();
            float[] dBvArr = dV.SumOverRows();
            for (int j = 0; j < D; j++)
            {
                Bq.Gradient.Data[0, j] += dBqArr[j];
                Bk.Gradient.Data[0, j] += dBkArr[j];
                Bv.Gradient.Data[0, j] += dBvArr[j];
            }

            // Gradient w.r.t. input X:
            //   dL/dX = dQ·Wqᵀ + dK·Wkᵀ + dV·Wvᵀ
            Matrix dX = Matrix.Dot(dQ, Wq.Weight.Transpose());
            dX.AddInPlace(Matrix.Dot(dK, Wk.Weight.Transpose()));
            dX.AddInPlace(Matrix.Dot(dV, Wv.Weight.Transpose()));

            return dX;
        }

        // ── RoPE helpers ──────────────────────────────────────────────────────────

        /// <summary>
        /// Apply Rotary Positional Encoding in-place to <paramref name="m"/> [T × dk].
        /// Each row t is at absolute sequence position startPos + t.
        ///
        /// For each pair (2i, 2i+1):
        ///   x'_{2i}   = x_{2i}·cos(θ) − x_{2i+1}·sin(θ)
        ///   x'_{2i+1} = x_{2i}·sin(θ) + x_{2i+1}·cos(θ)
        /// </summary>
        private void ApplyRoPE(Matrix m, int startPos)
        {
            int T      = m.Rows;
            int halfDk = _headDim / 2;
            for (int t = 0; t < T; t++)
            {
                int pos = startPos + t;
                for (int i = 0; i < halfDk; i++)
                {
                    float x0  = m.Data[t, 2 * i];
                    float x1  = m.Data[t, 2 * i + 1];
                    float cos = _ropeFreqCos![pos, i];
                    float sin = _ropeFreqSin![pos, i];
                    m.Data[t, 2 * i]     = x0 * cos - x1 * sin;
                    m.Data[t, 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }

        /// <summary>
        /// Inverse RoPE: rotate with −θ (negate sin).
        /// RoPE is orthogonal so R(−θ) = R(θ)ᵀ = R(θ)⁻¹.
        /// Used in backward to un-rotate the gradient w.r.t. rotated Q/K back
        /// to the gradient w.r.t. the pre-rotation Q/K.
        /// </summary>
        private void ApplyRoPEInverse(Matrix m, int startPos)
        {
            int T      = m.Rows;
            int halfDk = _headDim / 2;
            for (int t = 0; t < T; t++)
            {
                int pos = startPos + t;
                for (int i = 0; i < halfDk; i++)
                {
                    float x0  =  m.Data[t, 2 * i];
                    float x1  =  m.Data[t, 2 * i + 1];
                    float cos =  _ropeFreqCos![pos, i];
                    float sin = -_ropeFreqSin![pos, i];   // negated → inverse rotation
                    m.Data[t, 2 * i]     = x0 * cos - x1 * sin;
                    m.Data[t, 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        }

        // ── KV-Cache inference ────────────────────────────────────────────────────

        /// <summary>
        /// Clear all KV caches.  Call before starting a new generation sequence.
        /// </summary>
        public void ClearKVCache()
        {
            _kvCacheK = null;
            _kvCacheV = null;
            _kvLen    = 0;
        }

        /// <summary>
        /// Cached-attention forward pass for inference-time generation.
        ///
        /// On the first call (prefill, <paramref name="posOffset"/> = 0):
        ///   • Process all prompt tokens at once, populating the KV cache.
        ///   • Applies the standard lower-triangular causal mask.
        ///
        /// On subsequent calls (decode, <paramref name="posOffset"/> = prompt.Length + generated):
        ///   • Processes one new token, appending its K/V to the cache.
        ///   • No masking needed: the single query can attend to all cached positions.
        ///
        /// This method does NOT update the forward-pass training cache
        /// (_cachedQ, _cachedAttnWeights, etc.) and is not differentiable.
        /// Use <see cref="Forward"/> during training.
        /// </summary>
        public Matrix ForwardCached(Matrix x, int posOffset)
        {
            ArgumentNullException.ThrowIfNull(x);
            int newT = x.Rows;
            int D    = _cfg.EmbeddingDim;
            int H    = _cfg.NumHeads;
            int dk   = _headDim;

            // Project to Q, K, V
            Matrix Q = Matrix.Dot(x, Wq.Weight).AddBias(GetBias(Bq));
            Matrix K = Matrix.Dot(x, Wk.Weight).AddBias(GetBias(Bk));
            Matrix V = Matrix.Dot(x, Wv.Weight).AddBias(GetBias(Bv));

            // Allocate KV cache on first use
            if (_kvCacheK == null)
            {
                _kvCacheK = new Matrix[H];
                _kvCacheV = new Matrix[H];
                for (int h = 0; h < H; h++)
                {
                    _kvCacheK[h] = new Matrix(_cfg.ContextLength, dk);
                    _kvCacheV[h] = new Matrix(_cfg.ContextLength, dk);
                }
                _kvLen = 0;
            }

            float scale      = _cfg.AttentionScale;
            int   totalLen   = posOffset + newT;   // attended sequence length after this step
            var   headOutputs = new Matrix[H];

            Parallel.For(0, H, h =>
            {
                int colStart = h * dk;
                Matrix Qh = Q.SliceCols(colStart, colStart + dk);   // [newT × dk]
                Matrix Kh = K.SliceCols(colStart, colStart + dk);   // [newT × dk]
                Matrix Vh = V.SliceCols(colStart, colStart + dk);   // [newT × dk]

                // Apply RoPE at the actual absolute positions for this token batch
                if (_useRoPE)
                {
                    ApplyRoPE(Qh, posOffset);
                    ApplyRoPE(Kh, posOffset);
                }

                // Write new K, V into the cache at positions [posOffset, posOffset+newT)
                for (int t = 0; t < newT; t++)
                    for (int d = 0; d < dk; d++)
                    {
                        _kvCacheK![h].Data[posOffset + t, d] = Kh.Data[t, d];
                        _kvCacheV![h].Data[posOffset + t, d] = Vh.Data[t, d];
                    }

                // Slice the full relevant portion of the cache: [totalLen × dk]
                Matrix Kfull = _kvCacheK![h].SliceRows(0, totalLen);
                Matrix Vfull = _kvCacheV![h].SliceRows(0, totalLen);

                // scores = Qh · Kfull^T · scale   [newT × totalLen]
                Matrix scores = Matrix.Dot(Qh, Kfull.Transpose()).Scale(scale);

                // Causal masking for prefill (newT > 1): row i attends to positions ≤ posOffset+i
                if (newT > 1)
                {
                    for (int i = 0; i < newT; i++)
                        for (int j = posOffset + i + 1; j < totalLen; j++)
                            scores.Data[i, j] = float.NegativeInfinity;
                }
                // For single-token decode (newT==1) no masking needed:
                // the one query attends to all totalLen cached positions, all valid.

                Matrix attn = scores.Softmax();
                headOutputs[h] = Matrix.Dot(attn, Vfull);
            });

            _kvLen = totalLen;

            Matrix concat = Matrix.ConcatCols(headOutputs);
            return Matrix.Dot(concat, Wo.Weight).AddBias(GetBias(Bo));
        }

        // ── helpers ───────────────────────────────────────────────────────────────

        /// <summary>
        /// Extract a bias Parameter's values as a flat float[] for AddBias().
        /// </summary>
        private static float[] GetBias(Parameter p)
        {
            float[] bias = new float[p.Cols];
            for (int j = 0; j < p.Cols; j++) bias[j] = p.Weight.Data[0, j];
            return bias;
        }

        // ── parameter access ──────────────────────────────────────────────────────

        /// <summary>Enumerate all learnable parameters for the optimiser.</summary>
        public IEnumerable<IParameter> Parameters()
        {
            yield return Wq; yield return Wk; yield return Wv; yield return Wo;
            yield return Bq; yield return Bk; yield return Bv; yield return Bo;
        }

        public void Dispose()
        {
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
            Bq.Dispose(); Bk.Dispose(); Bv.Dispose(); Bo.Dispose();
        }

        public override string ToString() =>
            $"MultiHeadAttention(d={_cfg.EmbeddingDim}, heads={_cfg.NumHeads}, d_k={_headDim})";
    }
}
