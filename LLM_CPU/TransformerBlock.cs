using LLM_CPU;
using System;
using System.Collections.Generic;

namespace LLM
{
    /// <summary>
    /// One transformer block (a single "layer" of the transformer).
    ///
    /// ─── Architecture: Pre-Norm ──────────────────────────────────────────────────
    /// We use the pre-norm design popularised by GPT-2 and most modern LLMs:
    ///
    ///   x = x + Attention( LayerNorm1(x) )     ← attention sub-layer + residual
    ///   x = x + FFN( LayerNorm2(x) )           ← FFN sub-layer + residual
    ///
    /// The original "Attention Is All You Need" paper (Vaswani et al., 2017) placed
    /// layer norms after the sub-layer (post-norm).  Pre-norm is now preferred because:
    ///   • Gradients flow more cleanly through the residual connection.
    ///   • Training is more stable without a warm-up learning rate schedule.
    ///   • Empirically reaches the same quality with fewer steps.
    ///
    /// ─── Residual connections ────────────────────────────────────────────────────
    /// Each sub-layer wraps its output in a residual (skip) connection:
    ///   output = input + f(input)
    ///
    /// Benefits:
    ///   • Gradients can flow directly from the loss to early layers through the
    ///     identity shortcut, alleviating the vanishing-gradient problem.
    ///   • The network only needs to learn the "correction" f(input) on top of the
    ///     existing representation, which is easier.
    ///   • Allows very deep networks (100+ layers) to train stably.
    ///
    /// ─── Forward pass ────────────────────────────────────────────────────────────
    ///   normed1  = LayerNorm1(x)
    ///   attnOut  = Attention(normed1)
    ///   x        = x + attnOut          ← first residual
    ///   normed2  = LayerNorm2(x)
    ///   ffnOut   = FFN(normed2)
    ///   x        = x + ffnOut           ← second residual
    ///
    /// ─── Backward pass ────────────────────────────────────────────────────────────
    /// The gradient of the residual connection is simply addition (the identity):
    ///   dL/d(input) = dL/d(output) + dL/d(f_out) · df/d(input)
    ///
    /// In code: the skip-path gradient is simply the incoming dOut (copied through),
    /// and the main-path gradient flows through FFN/Attention and their LayerNorms.
    /// Both are summed at the residual junction.
    /// </summary>
    public sealed class TransformerBlock : ILayer<Matrix>
    {
        // ── sub-layers ────────────────────────────────────────────────────────────

        /// <summary>Layer norm applied to the input before the attention sub-layer.</summary>
        public ILayer<Matrix> Norm1 { get; }

        /// <summary>Multi-head causal self-attention.</summary>
        public ILayer<Matrix> Attention { get; }

        /// <summary>Layer norm applied to the input before the FFN sub-layer.</summary>
        public ILayer<Matrix> Norm2 { get; }

        /// <summary>Position-wise feed-forward network.</summary>
        public ILayer<Matrix> FFN { get; }

        // ── forward-pass cache ────────────────────────────────────────────────────

        /// <summary>
        /// Input x entering this block, shape [T × D].
        /// Needed for the backward residual skip path.
        /// </summary>
        private Matrix? _cachedX;

        /// <summary>
        /// Output of LayerNorm1, shape [T × D].
        /// Needed for the backward pass of Norm1 (Norm1.Backward returns dL/d(x),
        /// which flows back through the first residual).
        /// </summary>
        private Matrix? _cachedNormed1;

        /// <summary>
        /// x after the first residual ( x + attnOut ), shape [T × D].
        /// This is the input to Norm2 in the forward pass and the point where
        /// the second residual branches off in the backward pass.
        /// </summary>
        private Matrix? _cachedAfterAttn;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Build one transformer block.
        /// </summary>
        /// <param name="cfg">Model configuration.</param>
        /// <param name="rng">Random number generator for weight initialisation.</param>
        public TransformerBlock(TransformerConfig cfg, Random rng)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            int D = cfg.EmbeddingDim;

            Norm1     = new LayerNorm(D);
            Attention = new MultiHeadAttention(cfg, rng);
            Norm2     = new LayerNorm(D);
            FFN       = new FeedForward(D, cfg.FFNDim, rng);
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Run one transformer block.
        ///
        /// Input:  x of shape [T × D]
        /// Output: same shape [T × D]
        ///
        /// The two residual additions mean that the block is learning a correction
        /// to add to the stream, rather than a complete replacement.
        /// </summary>
        public Matrix Forward(Matrix x)
        {
            _cachedX = x;

            // ── Attention sub-layer ────────────────────────────────────────────────
            // 1. Normalise the input
            Matrix normed1 = Norm1.Forward(x);
            _cachedNormed1 = normed1;

            // 2. Self-attention on the normalised input
            Matrix attnOut = Attention.Forward(normed1);

            // 3. Add residual: mix the attention output back into the stream
            Matrix afterAttn = Matrix.Add(x, attnOut);   // x + Attention(Norm1(x))
            _cachedAfterAttn = afterAttn;

            // ── FFN sub-layer ──────────────────────────────────────────────────────
            // 4. Normalise the updated stream
            Matrix normed2 = Norm2.Forward(afterAttn);

            // 5. Feed-forward network on the normalised stream
            Matrix ffnOut = FFN.Forward(normed2);

            // 6. Add second residual
            Matrix output = Matrix.Add(afterAttn, ffnOut);   // x' + FFN(Norm2(x'))

            return output;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Backpropagate through the transformer block.
        ///
        /// Given dL/d(output), return dL/d(input x).
        ///
        /// Following the computation graph in reverse:
        ///
        ///   output = afterAttn + ffnOut
        ///   → dL/d(afterAttn) = dOut   (residual: gradient copies through addition)
        ///   → dL/d(ffnOut)    = dOut
        ///
        ///   ffnOut  = FFN(normed2)
        ///   normed2 = Norm2(afterAttn)
        ///   → dL/d(normed2)   = FFN.Backward(dL/d(ffnOut))
        ///   → dL/d(afterAttn) += Norm2.Backward(dL/d(normed2))   ← added to residual
        ///
        ///   afterAttn = x + attnOut        [first residual]
        ///   → dL/d(x)       starts as dL/d(afterAttn)   (skip path)
        ///   → dL/d(attnOut) =           dL/d(afterAttn)
        ///
        ///   attnOut  = Attention(normed1)
        ///   normed1  = Norm1(x)
        ///   → dL/d(normed1) = Attention.Backward(dL/d(attnOut))
        ///   → dL/d(x)      += Norm1.Backward(dL/d(normed1))       ← added to skip
        /// </summary>
        /// <param name="dOut">Upstream gradient dL/d(output), shape [T × D].</param>
        /// <returns>Gradient dL/d(x), shape [T × D].</returns>
        public Matrix Backward(Matrix dOut)
        {
            if (_cachedX is null || _cachedAfterAttn is null)
                throw new InvalidOperationException("Backward called before Forward.");

            // ── Second residual: output = afterAttn + ffnOut ──────────────────────
            // Both branches receive dOut as their gradient (addition splits the gradient).
            // dL/d(afterAttn) from the residual skip path = dOut
            // dL/d(ffnOut) = dOut

            // ── FFN backward ──────────────────────────────────────────────────────
            // ffnOut  = FFN( Norm2(afterAttn) )
            // dL/d(normed2)   = FFN.Backward(dOut)
            Matrix dNormed2   = FFN.Backward(dOut);

            // dL/d(afterAttn) from FFN path = Norm2.Backward(dL/d(normed2))
            Matrix dAfterAttn = Norm2.Backward(dNormed2);

            // Add the skip-path gradient for the second residual
            // (output = afterAttn + ffnOut → d/d(afterAttn) = 1)
            dAfterAttn.AddInPlace(dOut);

            // ── First residual: afterAttn = x + attnOut ───────────────────────────
            // Both branches receive dAfterAttn as their gradient.
            // dL/d(attnOut)   = dAfterAttn
            // dL/d(x) from skip path = dAfterAttn

            // ── Attention backward ────────────────────────────────────────────────
            // attnOut = Attention( Norm1(x) )
            // dL/d(normed1) = Attention.Backward(dAfterAttn)
            Matrix dNormed1 = Attention.Backward(dAfterAttn);

            // dL/d(x) from attention path = Norm1.Backward(dL/d(normed1))
            Matrix dXFromAttn = Norm1.Backward(dNormed1);

            // Combine skip path and attention-path gradients at the first residual
            Matrix dX = Matrix.Add(dAfterAttn, dXFromAttn);

            return dX;
        }

        // ── cached inference forward ──────────────────────────────────────────────

        /// <summary>
        /// KV-Cache forward pass for inference.
        /// Mirrors the training <see cref="Forward"/> but delegates to
        /// <see cref="MultiHeadAttention.ForwardCached"/> so that K/V pairs
        /// are accumulated across generation steps without re-processing the prompt.
        ///
        /// This method does not store a backward cache and is NOT differentiable.
        /// </summary>
        public Matrix ForwardCached(Matrix x, int posOffset)
        {
            var attn = (MultiHeadAttention)Attention;

            // ── Attention sub-layer (cached) ───────────────────────────────────────
            Matrix normed1  = Norm1.Forward(x);
            Matrix attnOut  = attn.ForwardCached(normed1, posOffset);
            Matrix afterAttn = Matrix.Add(x, attnOut);

            // ── FFN sub-layer ──────────────────────────────────────────────────────
            Matrix normed2 = Norm2.Forward(afterAttn);
            Matrix ffnOut  = FFN.Forward(normed2);

            return Matrix.Add(afterAttn, ffnOut);
        }

        /// <summary>Clear the KV cache on the attention layer.</summary>
        public void ClearKVCache() => ((MultiHeadAttention)Attention).ClearKVCache();

        // ── parameter access ──────────────────────────────────────────────────────

        /// <summary>Enumerate all learnable parameters in this block for the optimiser.</summary>
        public IEnumerable<IParameter> Parameters()
        {
            foreach (var p in Norm1.Parameters())     yield return p;
            foreach (var p in Attention.Parameters()) yield return p;
            foreach (var p in Norm2.Parameters())     yield return p;
            foreach (var p in FFN.Parameters())       yield return p;
        }

        public void Dispose()
        {
            Norm1.Dispose();
            Attention.Dispose();
            Norm2.Dispose();
            FFN.Dispose();
        }

        public override string ToString() => "TransformerBlock";
    }
}
