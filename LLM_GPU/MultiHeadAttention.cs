using System;
using System.Collections.Generic;
using LLM;

namespace LLM_GPU
{
    /// <summary>
    /// GPU multi-head causal self-attention.
    /// Mirrors the CPU implementation but uses GpuMatrix operations throughout.
    /// Intermediate matrices created during forward are cached for the backward pass
    /// and disposed at the start of the next forward call.
    /// </summary>
    internal sealed class GpuMultiHeadAttention : ILayer<GpuMatrix>
    {
        private readonly TransformerConfig _cfg;
        private readonly int _headDim;

        // Projection parameters
        public readonly GpuParameter Wq, Wk, Wv, Wo;
        public readonly GpuParameter Bq, Bk, Bv, Bo;

        // Forward-pass cache
        private GpuMatrix?   _cachedX;
        private GpuMatrix?   _cachedQ, _cachedK, _cachedV;
        private GpuMatrix[]? _cachedQh, _cachedKh, _cachedVh;
        private GpuMatrix[]? _cachedAttnWeights, _cachedHeadOutputs;
        private GpuMatrix?   _cachedConcat;

        // Causal mask cache
        private GpuMatrix? _causalMask;
        private int        _cachedMaskLen;

        public GpuMultiHeadAttention(TransformerConfig cfg, Random rng)
        {
            _cfg     = cfg;
            _headDim = cfg.HeadDim;
            int D    = cfg.EmbeddingDim;

            Wq = new GpuParameter(D, D, rng, 0.02f);
            Wk = new GpuParameter(D, D, rng, 0.02f);
            Wv = new GpuParameter(D, D, rng, 0.02f);
            Wo = new GpuParameter(D, D, rng, 0.02f);
            Bq = GpuParameter.Zeros(1, D);
            Bk = GpuParameter.Zeros(1, D);
            Bv = GpuParameter.Zeros(1, D);
            Bo = GpuParameter.Zeros(1, D);
        }

        // ── causal mask ───────────────────────────────────────────────────────
        private GpuMatrix GetCausalMask(int T)
        {
            if (_causalMask != null && _cachedMaskLen == T)
                return _causalMask;

            _causalMask?.Dispose();
            var flat = new float[T * T];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    flat[i * T + j] = (j <= i) ? 0f : float.NegativeInfinity;
            _causalMask = new GpuMatrix(T, T);
            _causalMask.UploadFlat(flat);
            _cachedMaskLen = T;
            return _causalMask;
        }

        // ── forward pass ──────────────────────────────────────────────────────
        public GpuMatrix Forward(GpuMatrix x)
        {
            DisposeFwdCache();

            int T  = x.Rows;
            int D  = _cfg.EmbeddingDim;
            int H  = _cfg.NumHeads;
            int dk = _headDim;

            _cachedX = x;

            // Q = x·Wq + Bq, K = x·Wk + Bk, V = x·Wv + Bv
            GpuMatrix Q, K, V;
            {
                using var tmp = GpuMatrix.Dot(x, Wq.Weight);
                Q = tmp.AddBias(Bq.Weight);
            }
            {
                using var tmp = GpuMatrix.Dot(x, Wk.Weight);
                K = tmp.AddBias(Bk.Weight);
            }
            {
                using var tmp = GpuMatrix.Dot(x, Wv.Weight);
                V = tmp.AddBias(Bv.Weight);
            }
            _cachedQ = Q; _cachedK = K; _cachedV = V;

            GpuMatrix mask = GetCausalMask(T);
            float scale    = _cfg.AttentionScale;

            _cachedQh          = new GpuMatrix[H];
            _cachedKh          = new GpuMatrix[H];
            _cachedVh          = new GpuMatrix[H];
            _cachedAttnWeights = new GpuMatrix[H];
            _cachedHeadOutputs = new GpuMatrix[H];
            var headOutputs    = new GpuMatrix[H];

            for (int h = 0; h < H; h++)
            {
                int cs = h * dk, ce = (h + 1) * dk;

                GpuMatrix Qh = Q.SliceCols(cs, ce);
                GpuMatrix Kh = K.SliceCols(cs, ce);
                GpuMatrix Vh = V.SliceCols(cs, ce);
                _cachedQh[h] = Qh;
                _cachedKh[h] = Kh;
                _cachedVh[h] = Vh;

                // scores = Qh · KhT * scale + mask
                GpuMatrix scores;
                {
                    using var KhT    = Kh.Transpose();
                    using var rawDot = GpuMatrix.Dot(Qh, KhT);
                    using var scaled = rawDot.Scale(scale);
                    scores = GpuMatrix.Add(scaled, mask);
                }

                GpuMatrix attn = scores.Softmax();
                scores.Dispose();
                GpuMatrix outH = GpuMatrix.Dot(attn, Vh);

                _cachedAttnWeights[h] = attn;
                _cachedHeadOutputs[h] = outH;
                headOutputs[h]        = outH;
            }

            _cachedConcat = GpuMatrix.ConcatCols(headOutputs);

            GpuMatrix result;
            {
                using var tmp = GpuMatrix.Dot(_cachedConcat, Wo.Weight);
                result = tmp.AddBias(Bo.Weight);
            }
            return result;
        }

        // ── backward pass ─────────────────────────────────────────────────────
        public GpuMatrix Backward(GpuMatrix dResult)
        {
            if (_cachedX is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int T  = _cachedX.Rows;
            int D  = _cfg.EmbeddingDim;
            int H  = _cfg.NumHeads;
            int dk = _headDim;

            // ── 4. Output projection backward ────────────────────────────────
            // dWo += concatT · dResult
            {
                using var concatT = _cachedConcat!.Transpose();
                using var grad    = GpuMatrix.Dot(concatT, dResult);
                Wo.Gradient.AddInPlace(grad);
            }
            // dBo += sum_rows(dResult)  — CPU round-trip (D floats)
            AccumulateBiasGrad(Bo, dResult, D);

            // dConcat = dResult · WoT
            GpuMatrix dConcat;
            {
                using var WoT = Wo.Weight.Transpose();
                dConcat = GpuMatrix.Dot(dResult, WoT);
            }

            // ── 3. Per-head backward ──────────────────────────────────────────
            using var dQ = new GpuMatrix(T, D);
            using var dK = new GpuMatrix(T, D);
            using var dV = new GpuMatrix(T, D);
            dQ.Zero(); dK.Zero(); dV.Zero();

            float scale = _cfg.AttentionScale;

            for (int h = 0; h < H; h++)
            {
                int cs = h * dk;
                using var dOutH = dConcat.SliceCols(cs, cs + dk);

                // dA_h = dOutH · VhT
                GpuMatrix dAh;
                {
                    using var VhT = _cachedVh![h].Transpose();
                    dAh = GpuMatrix.Dot(dOutH, VhT);
                }

                // dV_h = AttnT · dOutH
                GpuMatrix dVh;
                {
                    using var AttnT = _cachedAttnWeights![h].Transpose();
                    dVh = GpuMatrix.Dot(AttnT, dOutH);
                }

                // dScores = softmax_backward(Attn, dAh)
                using var dScores = GpuMatrix.SoftmaxBackward(_cachedAttnWeights[h], dAh);
                dAh.Dispose();

                // dQ_h = dScores · Kh * scale
                GpuMatrix dQh;
                {
                    using var tmp = GpuMatrix.Dot(dScores, _cachedKh![h]);
                    dQh = tmp.Scale(scale);
                }

                // dK_h = dScoresT · Qh * scale
                GpuMatrix dKh;
                {
                    using var dScoresT = dScores.Transpose();
                    using var tmp      = GpuMatrix.Dot(dScoresT, _cachedQh![h]);
                    dKh = tmp.Scale(scale);
                }

                using (dQh) using (dKh) using (dVh)
                {
                    dQ.AddSliceCols(dQh, cs);
                    dK.AddSliceCols(dKh, cs);
                    dV.AddSliceCols(dVh, cs);
                }
            }

            dConcat.Dispose();

            // ── 1-2. Projection backward ──────────────────────────────────────
            using var Xt = _cachedX.Transpose();

            {
                using var g = GpuMatrix.Dot(Xt, dQ);
                Wq.Gradient.AddInPlace(g);
            }
            {
                using var g = GpuMatrix.Dot(Xt, dK);
                Wk.Gradient.AddInPlace(g);
            }
            {
                using var g = GpuMatrix.Dot(Xt, dV);
                Wv.Gradient.AddInPlace(g);
            }

            AccumulateBiasGrad(Bq, dQ, D);
            AccumulateBiasGrad(Bk, dK, D);
            AccumulateBiasGrad(Bv, dV, D);

            // dX = dQ·WqT + dK·WkT + dV·WvT
            GpuMatrix dX;
            {
                using var WqT = Wq.Weight.Transpose();
                dX = GpuMatrix.Dot(dQ, WqT);
            }
            {
                using var WkT = Wk.Weight.Transpose();
                using var t   = GpuMatrix.Dot(dK, WkT);
                dX.AddInPlace(t);
            }
            {
                using var WvT = Wv.Weight.Transpose();
                using var t   = GpuMatrix.Dot(dV, WvT);
                dX.AddInPlace(t);
            }

            return dX;
        }

        // ── helpers ───────────────────────────────────────────────────────────
        /// <summary>
        /// Accumulate sum_rows(dMat) into bias.Gradient via CPU round-trip.
        /// Small enough (D floats) that the transfer cost is negligible.
        /// </summary>
        private static void AccumulateBiasGrad(GpuParameter bias, GpuMatrix dMat, int D)
        {
            float[] sum  = dMat.SumOverRows();            // syncs GPU
            float[] flat = bias.Gradient.DownloadFlat();
            for (int j = 0; j < D; j++) flat[j] += sum[j];
            bias.Gradient.UploadFlat(flat);
        }

        // ── parameter access ──────────────────────────────────────────────────
        public IEnumerable<IParameter> Parameters()
        {
            yield return Wq; yield return Wk; yield return Wv; yield return Wo;
            yield return Bq; yield return Bk; yield return Bv; yield return Bo;
        }

        // ── forward cache disposal ─────────────────────────────────────────────
        private void DisposeFwdCache()
        {
            // Q, K, V own these buffers
            _cachedQ?.Dispose(); _cachedQ = null;
            _cachedK?.Dispose(); _cachedK = null;
            _cachedV?.Dispose(); _cachedV = null;

            if (_cachedQh != null)
                foreach (var m in _cachedQh) m?.Dispose();
            if (_cachedKh != null)
                foreach (var m in _cachedKh) m?.Dispose();
            if (_cachedVh != null)
                foreach (var m in _cachedVh) m?.Dispose();
            if (_cachedAttnWeights != null)
                foreach (var m in _cachedAttnWeights) m?.Dispose();
            if (_cachedHeadOutputs != null)
                foreach (var m in _cachedHeadOutputs) m?.Dispose();
            _cachedConcat?.Dispose(); _cachedConcat = null;
        }

        public void Dispose()
        {
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
            Bq.Dispose(); Bk.Dispose(); Bv.Dispose(); Bo.Dispose();
            DisposeFwdCache();
            _causalMask?.Dispose();
        }
    }
}
