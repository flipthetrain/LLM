using LLM;
using System;

namespace LLM_CPU
{
    /// <summary>
    /// Combines token embeddings and positional encodings into the initial
    /// residual stream fed into the first transformer block.
    ///
    /// ─── Token embeddings ───────────────────────────────────────────────────────
    /// Each integer token ID is looked up in a table of shape [VocabSize × d_model].
    /// This table is a learnable parameter: the network discovers which vector
    /// representation is most useful for each token.
    ///
    /// ─── Positional encodings ───────────────────────────────────────────────────
    /// Because the self-attention operation is position-agnostic (it sees a set, not
    /// a sequence), we must inject information about where each token sits in the
    /// sequence.  We use fixed sinusoidal encodings (Vaswani et al., 2017):
    ///
    ///   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    ///   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    ///
    /// • Different frequencies let the model attend to absolute and relative positions.
    /// • Because sin/cos are periodic and linear combinations of sinusoids of different
    ///   frequencies can represent relative offsets, the model can generalise to
    ///   unseen sequence lengths.
    /// • Fixed encodings do not add learnable parameters.
    ///
    /// Alternative: GPT-2 uses learned positional embeddings (another lookup table).
    /// We use sinusoidal here for clarity and to avoid the extra parameter.
    ///
    /// ─── Forward pass ───────────────────────────────────────────────────────────
    ///   output[t, :] = TokenEmbedding[tokens[t], :] + PE[t, :]
    ///
    /// ─── Backward pass ──────────────────────────────────────────────────────────
    ///   Gradient flows into the token embedding table: for each position t,
    ///   row tokens[t] of the embedding table receives the gradient from that position.
    ///   The positional encodings are fixed so their gradient is discarded.
    /// </summary>
    public sealed class Embedding : IEmbeddingLayer<Matrix>
    {
        // ── configuration ─────────────────────────────────────────────────────────

        /// <summary>Model configuration (vocabulary size, embedding dim, etc.).</summary>
        private readonly TransformerConfig _cfg;

        // ── learnable parameters ──────────────────────────────────────────────────

        /// <summary>
        /// Token embedding lookup table.
        /// Shape: [VocabSize × EmbeddingDim].
        /// Row i is the d_model-dimensional vector for token ID i.
        /// </summary>
        public Parameter TokenEmbedding { get; }

        // ── fixed buffers ─────────────────────────────────────────────────────────

        /// <summary>
        /// Pre-computed sinusoidal positional encodings.
        /// Shape: [ContextLength × EmbeddingDim].
        /// Row t contains the encoding for position t (0-indexed).
        /// Computed once in the constructor and reused for every forward pass.
        /// </summary>
        private readonly Matrix _posEncoding;

        // ── forward-pass cache (needed for backward) ──────────────────────────────

        /// <summary>
        /// Token IDs from the most recent forward pass.
        /// Needed in backward to know which rows of TokenEmbedding to update.
        /// </summary>
        private int[]? _cachedTokenIds;

        /// <summary>Sequence length of the most recent forward pass.</summary>
        private int _cachedSeqLen;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Initialise the token embedding table and pre-compute positional encodings.
        /// </summary>
        /// <param name="cfg">Model configuration.</param>
        /// <param name="rng">Random number generator for weight initialisation.</param>
        public Embedding(TransformerConfig cfg, Random rng)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            _cfg = cfg;

            // Token embedding: GPT-2 uses std=0.02; use smaller for tiny models.
            TokenEmbedding = new Parameter(cfg.VocabSize, cfg.EmbeddingDim, rng, initStd: 0.02f);

            // Pre-compute sinusoidal positional encodings (fixed, not learned)
            _posEncoding = ComputeSinusoidalEncoding(cfg.ContextLength, cfg.EmbeddingDim);
        }

        // ── sinusoidal encoding ───────────────────────────────────────────────────

        /// <summary>
        /// Compute the sinusoidal positional encoding matrix once.
        ///
        /// For each position pos in [0, maxLen) and each dimension pair (2i, 2i+1):
        ///
        ///   PE[pos, 2i]   = sin( pos / 10000^(2i / d_model) )
        ///   PE[pos, 2i+1] = cos( pos / 10000^(2i / d_model) )
        ///
        /// The base 10000 means that the highest-frequency sinusoid cycles every
        /// 2π ≈ 6.28 positions, while the lowest-frequency cycles only every
        /// 10000^(2·(d/2−1)/d) ≈ 10000 positions.
        ///
        /// This wide range of frequencies gives the model a rich positional signal
        /// at every scale from token-to-token to thousands of tokens apart.
        /// </summary>
        private static Matrix ComputeSinusoidalEncoding(int maxLen, int d)
        {
            var pe = new Matrix(maxLen, d);

            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int i = 0; i < d / 2; i++)
                {
                    // Frequency: 1 / 10000^(2i/d)
                    // Using logarithm for numerical stability: exp(-2i/d * ln(10000))
                    float freq = MathF.Exp(-2f * i / d * MathF.Log(10000f));

                    float angle = pos * freq;
                    pe.Data[pos, 2 * i]     = MathF.Sin(angle);   // even dimensions: sin
                    pe.Data[pos, 2 * i + 1] = MathF.Cos(angle);   // odd  dimensions: cos
                }

                // Handle the case where d is odd (extra last dimension gets sin only)
                if (d % 2 != 0)
                {
                    float freq = MathF.Exp(-(d - 1f) / d * MathF.Log(10000f));
                    pe.Data[pos, d - 1] = MathF.Sin(pos * freq);
                }
            }
            return pe;
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Convert a sequence of token IDs into a dense embedding matrix.
        ///
        /// Steps:
        ///   1. Look up each token ID in the embedding table (a no-op gather).
        ///   2. Add the sinusoidal positional encoding for each position.
        ///
        /// Returns a matrix of shape [seqLen × EmbeddingDim].
        /// </summary>
        /// <param name="tokenIds">Array of integer token IDs, length = seqLen.</param>
        public Matrix Forward(int[] tokenIds)
        {
            ArgumentNullException.ThrowIfNull(tokenIds);
            int seqLen = tokenIds.Length;
            if (seqLen > _cfg.ContextLength)
                throw new ArgumentException(
                    $"Sequence length {seqLen} exceeds ContextLength {_cfg.ContextLength}");

            // Cache for backward pass
            _cachedTokenIds = tokenIds;
            _cachedSeqLen   = seqLen;

            int d = _cfg.EmbeddingDim;
            var output = new Matrix(seqLen, d);

            bool useRoPE = _cfg.UseRoPE;
            for (int t = 0; t < seqLen; t++)
            {
                int id = tokenIds[t];
                for (int j = 0; j < d; j++)
                {
                    float tokenEmb = TokenEmbedding.Weight.Data[id, j];
                    // RoPE encodes position inside attention; no additive PE needed.
                    output.Data[t, j] = useRoPE ? tokenEmb
                                                : tokenEmb + _posEncoding.Data[t, j];
                }
            }

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Propagate the gradient from the residual stream back to the token
        /// embedding table.
        ///
        /// The positional encoding is fixed (not learnable) so its gradient is
        /// silently discarded.
        ///
        /// The token embedding table is a simple lookup (gather), so its backward
        /// is a scatter-add:
        ///   For each position t,
        ///     dL / d(TokenEmbedding[tokens[t], :]) += dL / d(output[t, :])
        ///
        /// Multiple positions can reference the same token ID; their gradients
        /// are accumulated (added together).
        ///
        /// This method does NOT return dL/d(input) because the "input" is discrete
        /// token IDs, which are not differentiable.
        /// </summary>
        /// <param name="grad">
        /// Upstream gradient of the loss w.r.t. the embedding layer output.
        /// Shape: [seqLen × EmbeddingDim].
        /// </param>
        public void Backward(Matrix grad)
        {
            ArgumentNullException.ThrowIfNull(grad);
            if (_cachedTokenIds is null)
                throw new InvalidOperationException("Backward called before Forward.");

            int seqLen = _cachedSeqLen;
            int d      = _cfg.EmbeddingDim;

            for (int t = 0; t < seqLen; t++)
            {
                int id = _cachedTokenIds[t];
                for (int j = 0; j < d; j++)
                {
                    // Scatter-add: accumulate the gradient at the corresponding
                    // row of the embedding table.
                    TokenEmbedding.Gradient.Data[id, j] += grad.Data[t, j];
                }
            }
        }

        // ── parameter access ──────────────────────────────────────────────────────

        /// <summary>Yield all learnable parameters for the optimiser to update.</summary>
        public System.Collections.Generic.IEnumerable<IParameter> Parameters()
        {
            yield return TokenEmbedding;
        }

        public void Dispose()
        {
            TokenEmbedding.Dispose();
        }

        public override string ToString() =>
            $"Embedding(vocab={_cfg.VocabSize}, d={_cfg.EmbeddingDim})";
    }
}
