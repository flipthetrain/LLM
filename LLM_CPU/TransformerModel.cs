using LLM_CPU;
using System;
using System.Collections.Generic;

namespace LLM
{
    /// <summary>
    /// Full GPT-style autoregressive language model.
    ///
    /// ─── Overview ─────────────────────────────────────────────────────────────────
    /// Architecture (decoder-only transformer, as in GPT / GPT-2):
    ///
    ///   tokens  ──► Embedding ──► [Block_0 → Block_1 → … → Block_{L-1}] ──► FinalNorm ──► Projection ──► logits
    ///
    ///   Embedding  : converts token IDs to d_model vectors + sinusoidal position encodings
    ///   Block_i    : pre-norm transformer block (LayerNorm → Attention → residual,
    ///                                            LayerNorm → FFN → residual)
    ///   FinalNorm  : LayerNorm on the last hidden state (stabilises the output)
    ///   Projection : linear map [d_model → VocabSize] that produces raw logit scores
    ///
    /// The model predicts the next token at every position simultaneously (teacher
    /// forcing during training), which is the standard language-modelling objective.
    ///
    /// ─── Parameter count ─────────────────────────────────────────────────────────
    /// With default settings (vocab≈100, d=128, H=4, L=4, d_ff=512, ctx=128):
    ///   Embedding  : 100·128          ≈   12 800
    ///   Per block  : 4·128² + 2·128·512 + 4·128 ≈  199 680
    ///   4 blocks   :                  ≈  798 720
    ///   FinalNorm  :               256
    ///   Projection : 128·100         ≈   12 800
    ///   Total      :                 ≈  824 576   (~800 K parameters)
    ///
    /// ─── Training objective ──────────────────────────────────────────────────────
    /// Cross-entropy language modelling loss:
    ///   L = −(1/T) · Σ_t  log P(token_{t+1} | tokens_{0..t})
    ///
    /// The model sees tokens[0..T-1] as input and predicts tokens[1..T] as targets.
    /// Causal masking inside attention enforces the autoregressive constraint.
    ///
    /// ─── Inference (text generation) ─────────────────────────────────────────────
    /// Given a prompt, repeatedly:
    ///   1. Forward pass → logits at the last position
    ///   2. Sample the next token from the logit distribution (temperature + top-k)
    ///   3. Append and repeat until the desired length is reached
    /// </summary>
    public sealed class TransformerModel : ITransformerModel
    {
        // ── configuration ─────────────────────────────────────────────────────────

        /// <summary>All hyper-parameters for this model instance.</summary>
        public TransformerConfig Config { get; }

        // ── layers ────────────────────────────────────────────────────────────────

        /// <summary>Converts integer token IDs → dense embeddings.</summary>
        public IEmbeddingLayer<Matrix> EmbeddingLayer { get; }

        /// <summary>The stack of transformer blocks (the "depth" of the model).</summary>
        public ILayer<Matrix>[] Blocks { get; }

        /// <summary>
        /// Final layer norm applied to the output of the last block before
        /// the linear projection.  Ensures the values fed to the vocabulary
        /// head are well-normalised.
        /// </summary>
        public ILayer<Matrix> FinalNorm { get; }

        /// <summary>
        /// Linear projection from d_model to VocabSize.
        /// Produces un-normalised log-probability scores (logits) for each token.
        /// Shape: [d_model × VocabSize].
        ///
        /// In some implementations (GPT-2) the token embedding matrix is reused
        /// here ("weight tying") to save parameters and improve generalisation.
        /// We keep them separate for clarity.
        /// </summary>
        public Parameter OutputProjection { get; }

        /// <summary>Output projection bias, shape [1 × VocabSize].</summary>
        public Parameter OutputBias { get; }

        // ── forward-pass cache ────────────────────────────────────────────────────

        /// <summary>Embeddings output before the first block, shape [T × D].</summary>
        private Matrix? _cachedEmbedOut;

        /// <summary>
        /// Outputs of each block, indexed 0 … NumLayers-1.
        /// _cachedBlockOuts[i] is the output of Block i.
        /// Used in the backward pass to feed each block its upstream gradient.
        /// </summary>
        private Matrix[]? _cachedBlockOuts;

        /// <summary>Output of FinalNorm, shape [T × D].</summary>
        private Matrix? _cachedNormed;

        // ── random number generator ───────────────────────────────────────────────

        /// <summary>
        /// Shared RNG used for stochastic sampling during text generation.
        /// Seeded at model construction so sampling is reproducible.
        /// </summary>
        private readonly Random _rng;

        // ── constructor ───────────────────────────────────────────────────────────

        /// <summary>
        /// Build and randomly initialise the full transformer model.
        /// </summary>
        /// <param name="cfg">Model configuration (must pass cfg.Validate() first).</param>
        /// <param name="rng">Random number generator for weight initialisation.</param>
        public TransformerModel(TransformerConfig cfg, Random rng)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            cfg.Validate();
            Config = cfg;
            _rng   = rng;

            EmbeddingLayer   = new Embedding(cfg, rng);
            Blocks           = new TransformerBlock[cfg.NumLayers];
            for (int i = 0; i < cfg.NumLayers; i++)
                Blocks[i] = new TransformerBlock(cfg, rng);

            FinalNorm        = new LayerNorm(cfg.EmbeddingDim);
            OutputProjection = new Parameter(cfg.EmbeddingDim, cfg.VocabSize, rng, initStd: 0.02f);
            OutputBias       = Parameter.Zeros(1, cfg.VocabSize);
        }

        // ── ITransformerModel ─────────────────────────────────────────────────────

        /// <inheritdoc/>
        public void Save(string path) => ModelSerializer.Save(path, Config, AllParameters());

        /// <inheritdoc/>
        public void Load(string path) => ModelSerializer.Load(path, Config, AllParameters());

        /// <inheritdoc/>
        public void SaveCheckpoint(string path, int epoch, int adamStep, int innerStep) =>
            ModelSerializer.SaveCheckpoint(path, Config, AllParameters(), epoch, adamStep, innerStep);

        /// <inheritdoc/>
        public (int epoch, int adamStep, int innerStep) LoadCheckpoint(string path) =>
            ModelSerializer.LoadCheckpoint(path, Config, AllParameters());

        /// <summary>Zero gradients, then forward → cross-entropy → backward → clip → Adam.</summary>
        public float TrainStep(int[] input, int[] targets, int adamStep)
        {
            ZeroAllGradients();
            float loss = AccumulateStep(input, targets);
            ClipAndUpdate(adamStep);
            return loss;
        }

        /// <inheritdoc/>
        public float Evaluate(int[] input, int[] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            Matrix logits = Forward(input);
            return CrossEntropyOnly(logits, targets);
        }

        /// <inheritdoc/>
        public float AccumulateStep(int[] input, int[] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            Matrix logits = Forward(input);
            float  loss   = CrossEntropyGrad(logits, targets, out Matrix dLogits);
            Backward(dLogits);
            return loss;
        }

        /// <inheritdoc/>
        public void ScaleAllGradients(float scale)
        {
            foreach (var p in AllParameters())
                p.ScaleGradient(scale);
        }

        /// <inheritdoc/>
        public void ClipAndUpdate(int adamStep)
        {
            ClipGradients();
            UpdateAllParameters(adamStep, Config.LearningRate);
        }

        /// <inheritdoc/>
        public void ClipAndUpdate(int adamStep, float lr)
        {
            ClipGradients();
            UpdateAllParameters(adamStep, lr);
        }

        /// <inheritdoc/>
        public void ClearKVCache()
        {
            foreach (var block in Blocks)
                ((TransformerBlock)block).ClearKVCache();
        }

        public void Dispose()
        {
            EmbeddingLayer.Dispose();
            foreach (var block in Blocks)
                block.Dispose();
            FinalNorm.Dispose();
            OutputProjection.Dispose();
            OutputBias.Dispose();
        }

        private static float CrossEntropyOnly(Matrix logits, int[] targets)
        {
            int   T     = logits.Rows;
            int   V     = logits.Cols;
            float total = 0f;

            for (int t = 0; t < T; t++)
            {
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (logits.Data[t, j] > maxVal) maxVal = logits.Data[t, j];

                float sumExp = 0f;
                for (int j = 0; j < V; j++)
                    sumExp += MathF.Exp(logits.Data[t, j] - maxVal);

                float prob = MathF.Exp(logits.Data[t, targets[t]] - maxVal) / sumExp;
                total -= MathF.Log(Math.Max(prob, 1e-8f));
            }

            return total / T;
        }

        private static float CrossEntropyGrad(Matrix logits, int[] targets, out Matrix dLogits)
        {
            int T      = logits.Rows;
            int V      = logits.Cols;
            dLogits    = new Matrix(T, V);
            float total = 0f;
            float invT  = 1f / T;

            for (int t = 0; t < T; t++)
            {
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (logits.Data[t, j] > maxVal) maxVal = logits.Data[t, j];

                float sumExp = 0f;
                for (int j = 0; j < V; j++)
                {
                    float e = MathF.Exp(logits.Data[t, j] - maxVal);
                    dLogits.Data[t, j] = e;
                    sumExp += e;
                }

                int correct = targets[t];
                for (int j = 0; j < V; j++)
                {
                    float prob = dLogits.Data[t, j] / sumExp;
                    dLogits.Data[t, j] = prob;
                    if (j == correct) total -= MathF.Log(Math.Max(prob, 1e-8f));
                }

                dLogits.Data[t, correct] -= 1f;
                for (int j = 0; j < V; j++) dLogits.Data[t, j] *= invT;
            }

            return total / T;
        }

        // ── forward pass ──────────────────────────────────────────────────────────

        /// <summary>
        /// Run a full forward pass of the model.
        ///
        /// Input:  integer token IDs of length T (T ≤ ContextLength)
        /// Output: logit matrix of shape [T × VocabSize]
        ///
        /// logits[t, :] are the un-normalised scores for the next token after
        /// position t.  Apply softmax to convert to probabilities.
        /// </summary>
        public Matrix Forward(int[] tokenIds)
        {
            // ── 1. Convert token IDs to embeddings ────────────────────────────────
            Matrix x = EmbeddingLayer.Forward(tokenIds);
            _cachedEmbedOut = x;

            // ── 2. Pass through each transformer block in sequence ────────────────
            _cachedBlockOuts = new Matrix[Config.NumLayers];
            for (int i = 0; i < Config.NumLayers; i++)
            {
                x = Blocks[i].Forward(x);
                _cachedBlockOuts[i] = x;
            }

            // ── 3. Final layer normalisation ──────────────────────────────────────
            Matrix normed = FinalNorm.Forward(x);
            _cachedNormed = normed;

            // ── 4. Project to vocabulary logits ───────────────────────────────────
            // logits = normed · W_out + b_out   [T × VocabSize]
            float[] bOut = GetBias(OutputBias, Config.VocabSize);
            Matrix logits = Matrix.Dot(normed, OutputProjection.Weight).AddBias(bOut);

            return logits;
        }

        // ── backward pass ─────────────────────────────────────────────────────────

        /// <summary>
        /// Backpropagate through the full model.
        ///
        /// Takes the gradient of the loss w.r.t. the logits
        /// (computed externally in the training loop) and propagates it all the
        /// way back, accumulating gradients in every parameter.
        ///
        /// Call ZeroAllGradients() before Forward + Backward, then call
        /// UpdateAllParameters() to apply one Adam step.
        /// </summary>
        /// <param name="dLogits">dL/d(logits), shape [T × VocabSize].</param>
        public void Backward(Matrix dLogits)
        {
            ArgumentNullException.ThrowIfNull(dLogits);
            if (_cachedNormed is null)
                throw new InvalidOperationException("Backward called before Forward.");

            // ── 4. Output projection backward ────────────────────────────────────
            // logits = normed · W_out + b_out
            // dL/dW_out = normedᵀ · dLogits
            // dL/db_out = Σ_t dLogits[t,:]
            // dL/d(normed) = dLogits · W_outᵀ
            OutputProjection.Gradient.AddInPlace(
                Matrix.Dot(_cachedNormed.Transpose(), dLogits));

            float[] dBiasOut = dLogits.SumOverRows();
            for (int j = 0; j < Config.VocabSize; j++)
                OutputBias.Gradient.Data[0, j] += dBiasOut[j];

            Matrix dX = Matrix.Dot(dLogits, OutputProjection.Weight.Transpose());

            // ── 3. Final layer norm backward ──────────────────────────────────────
            dX = FinalNorm.Backward(dX);

            // ── 2. Blocks backward (reverse order) ───────────────────────────────
            for (int i = Config.NumLayers - 1; i >= 0; i--)
                dX = Blocks[i].Backward(dX);

            // ── 1. Embedding backward ─────────────────────────────────────────────
            // The embedding layer has no further upstream to propagate to,
            // so we just scatter the gradients into the token embedding table.
            EmbeddingLayer.Backward(dX);
        }

        // ── gradient management ───────────────────────────────────────────────────

        /// <summary>
        /// Zero all parameter gradients before a new forward/backward pass.
        /// Must be called at the beginning of every training step.
        /// </summary>
        public void ZeroAllGradients()
        {
            foreach (var p in AllParameters())
                p.ZeroGrad();
        }

        // ── optimiser step ────────────────────────────────────────────────────────

        /// <summary>
        /// Apply one Adam optimiser step to all parameters.
        /// Call after Backward() to update the weights.
        /// </summary>
        /// <param name="step">Global training step (1-indexed) for bias correction.</param>
        /// <param name="lr">Learning rate to use for this step.</param>
        public void UpdateAllParameters(int step, float lr)
        {
            float beta1 = Config.Beta1;
            float beta2 = Config.Beta2;
            float eps   = Config.AdamEps;

            foreach (var p in AllParameters())
                p.Update(lr, beta1, beta2, eps, step);
        }

        // ── gradient clipping ─────────────────────────────────────────────────────

        /// <summary>
        /// Clip gradients by global L2 norm to prevent exploding gradients.
        ///
        /// Algorithm:
        ///   1. Compute the L2 norm of ALL gradients concatenated into one vector.
        ///   2. If that norm exceeds <see cref="TransformerConfig.GradClip"/>,
        ///      scale every gradient element down by (GradClip / norm).
        ///
        /// This keeps the training step size bounded while preserving gradient
        /// direction, unlike per-element clipping which can distort the direction.
        /// </summary>
        public void ClipGradients()
        {
            float maxNorm = Config.GradClip;

            // ── Step 1: compute global gradient norm ──────────────────────────────
            float sumSq = 0f;
            foreach (var p in AllParameters())
            {
                float[] g = p.GetGradientFlat();
                foreach (float v in g) sumSq += v * v;
            }
            float norm = MathF.Sqrt(sumSq);

            // ── Step 2: rescale if over the threshold ─────────────────────────────
            if (norm > maxNorm)
            {
                float scale = maxNorm / norm;
                foreach (var p in AllParameters())
                    p.ScaleGradient(scale);
            }
        }

        // ── text generation ───────────────────────────────────────────────────────

        /// <summary>
        /// Generate <paramref name="numTokens"/> new tokens given a prompt.
        ///
        /// Uses temperature-scaled sampling with optional top-k filtering:
        ///   1. Forward pass → logits at the last position.
        ///   2. Divide logits by temperature (lower = more confident, higher = more random).
        ///   3. (Optional) Zero out all logits except the top-k highest.
        ///   4. Softmax → probability distribution.
        ///   5. Sample from that distribution.
        ///   6. Append sampled token and repeat.
        ///
        /// The context window is kept at most ContextLength tokens long
        /// (older tokens are dropped from the left).
        /// </summary>
        /// <param name="promptIds">Token IDs of the prompt.</param>
        /// <param name="numTokens">How many new tokens to generate.</param>
        /// <param name="temperature">
        /// Sampling temperature (1.0 = unmodified, &lt;1 = sharper, &gt;1 = flatter).
        /// </param>
        /// <param name="topK">
        /// If &gt; 0, only sample from the top-K most likely tokens.
        /// topK = 1 is greedy decoding.
        /// </param>
        public int[] Generate(int[] promptIds, int numTokens, float temperature = 1.0f, int topK = 40)
        {
            ArgumentNullException.ThrowIfNull(promptIds);
            ClearKVCache();

            // Trim prompt to ContextLength
            int[] prompt = promptIds.Length > Config.ContextLength
                ? promptIds[(promptIds.Length - Config.ContextLength)..]
                : promptIds;

            var generated = new int[numTokens];

            // ── Prefill: process full prompt, populating KV caches ────────────────
            Matrix logits = ForwardCached(prompt, 0);
            int posOffset = prompt.Length;

            for (int step = 0; step < numTokens; step++)
            {
                float[] lastLogits = logits.GetRow(logits.Rows - 1);   // last position

                if (temperature != 1.0f)
                    for (int i = 0; i < lastLogits.Length; i++)
                        lastLogits[i] /= temperature;

                if (topK > 0 && topK < lastLogits.Length)
                    ApplyTopKMask(lastLogits, topK);

                int nextToken = SampleFromLogits(lastLogits);
                generated[step] = nextToken;

                if (step == numTokens - 1) break;

                // ── Decode: single-token step using cached K/V ─────────────────────
                if (posOffset >= Config.ContextLength) break;   // context window full
                logits = ForwardCached(new[] { nextToken }, posOffset);
                posOffset++;
            }

            return generated;
        }

        /// <summary>
        /// Inference-only forward pass that uses per-attention-head KV caches.
        /// Call <see cref="ClearKVCache"/> before each new generation sequence.
        /// </summary>
        private Matrix ForwardCached(int[] tokenIds, int posOffset)
        {
            Matrix x = EmbeddingLayer.Forward(tokenIds);

            for (int i = 0; i < Config.NumLayers; i++)
                x = ((TransformerBlock)Blocks[i]).ForwardCached(x, posOffset);

            Matrix normed = FinalNorm.Forward(x);
            float[] bOut  = GetBias(OutputBias, Config.VocabSize);
            return Matrix.Dot(normed, OutputProjection.Weight).AddBias(bOut);
        }

        /// <summary>
        /// Zero out (set to float.NegativeInfinity) all logit positions that are
        /// not in the top-k highest-scoring positions.
        /// </summary>
        private static void ApplyTopKMask(float[] logits, int k)
        {
            // Find the k-th largest value via a partial sort
            float threshold = FindKthLargest(logits, k);
            for (int i = 0; i < logits.Length; i++)
                if (logits[i] < threshold)
                    logits[i] = float.NegativeInfinity;
        }

        /// <summary>
        /// Return the k-th largest value in an array (1-indexed: k=1 is the max).
        /// Uses a simple linear scan approach suitable for small vocabulary sizes.
        /// </summary>
        private static float FindKthLargest(float[] arr, int k)
        {
            // Copy, sort descending, return the k-th element
            float[] copy = (float[])arr.Clone();
            Array.Sort(copy);
            Array.Reverse(copy);
            return copy[k - 1];
        }

        /// <summary>
        /// Sample a token index from un-normalised logits using the softmax
        /// distribution and a random draw.
        ///
        /// Algorithm (categorical sampling):
        ///   1. Compute softmax probabilities.
        ///   2. Draw u ∼ Uniform(0, 1).
        ///   3. Walk through the probability distribution until the cumulative
        ///      probability exceeds u; return that index.
        ///      (Equivalent to inverse-CDF sampling.)
        /// </summary>
        private int SampleFromLogits(float[] logits)
        {
            int V = logits.Length;

            // Softmax with numerical stability
            float maxLogit = float.NegativeInfinity;
            for (int i = 0; i < V; i++)
                if (logits[i] > maxLogit) maxLogit = logits[i];

            float sum = 0f;
            float[] probs = new float[V];
            for (int i = 0; i < V; i++)
            {
                probs[i] = MathF.Exp(logits[i] - maxLogit);
                sum += probs[i];
            }
            for (int i = 0; i < V; i++) probs[i] /= sum;

            // Inverse-CDF sampling
            double u = _rng.NextDouble();
            double cumulative = 0.0;
            for (int i = 0; i < V - 1; i++)
            {
                cumulative += probs[i];
                if (u <= cumulative) return i;
            }
            return V - 1;   // fallback: return last token if rounding pushes u past 1
        }

        // ── parameter enumeration ─────────────────────────────────────────────────

        /// <summary>
        /// Enumerate every learnable parameter in the model in a consistent order.
        /// Used by ZeroAllGradients(), ClipGradients(), and UpdateAllParameters().
        /// </summary>
        public IEnumerable<IParameter> AllParameters()
        {
            foreach (var p in EmbeddingLayer.Parameters()) yield return p;
            foreach (var block in Blocks)
                foreach (var p in block.Parameters()) yield return p;
            foreach (var p in FinalNorm.Parameters()) yield return p;
            yield return OutputProjection;
            yield return OutputBias;
        }

        /// <summary>
        /// Count total number of scalar learnable parameters.
        /// Useful for a quick sanity check after model construction.
        /// </summary>
        public long CountParameters()
        {
            long total = 0;
            foreach (var p in AllParameters())
                total += p.Rows * p.Cols;
            return total;
        }

        public long ParameterCount => CountParameters();

        // ── helpers ───────────────────────────────────────────────────────────────

        private static float[] GetBias(Parameter p, int len)
        {
            float[] bias = new float[len];
            for (int j = 0; j < len; j++) bias[j] = p.Weight.Data[0, j];
            return bias;
        }

        public override string ToString() =>
            $"TransformerModel({Config}) | {CountParameters():N0} parameters";
    }
}
