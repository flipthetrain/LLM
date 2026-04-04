using System;
using System.Collections.Generic;
using LLM;

namespace LLM_GPU
{
    /// <summary>
    /// Full GPU GPT-style language model.
    ///
    /// Forward/backward are entirely GPU-side.
    /// The only CPU↔GPU transfers per training step are:
    ///   1. Upload token IDs (tiny).
    ///   2. Download the logit matrix for the last training token (T × VocabSize).
    ///   3. Upload the loss gradient matrix back.
    ///   4. Download the scalar loss for logging.
    /// </summary>
    public sealed class GpuTransformerModel : ITransformerModel
    {
        public TransformerConfig Config { get; }
        internal readonly IEmbeddingLayer<GpuMatrix> EmbeddingLayer;
        internal readonly ILayer<GpuMatrix>[]        Blocks;
        internal readonly ILayer<GpuMatrix>          FinalNorm;
        internal readonly GpuParameter          OutputProjection;
        internal readonly GpuParameter          OutputBias;

        private GpuMatrix?   _cachedNormed;
        private readonly Random _rng;

        public GpuTransformerModel(TransformerConfig cfg, Random rng)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            GpuContext.Initialize();   // must come first — all GPU allocations below depend on it
            cfg.Validate();
            Config           = cfg;
            _rng             = rng;
            EmbeddingLayer   = new GpuEmbedding(cfg, rng);
            Blocks           = new GpuTransformerBlock[cfg.NumLayers];
            for (int i = 0; i < cfg.NumLayers; i++)
                Blocks[i] = new GpuTransformerBlock(cfg, rng);
            FinalNorm        = new GpuLayerNorm(cfg.EmbeddingDim);
            OutputProjection = new GpuParameter(cfg.EmbeddingDim, cfg.VocabSize, rng,
                                                initStd: 0.02f);
            OutputBias       = GpuParameter.Zeros(1, cfg.VocabSize);
        }

        // ── ITransformerModel ─────────────────────────────────────────────────

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
            using GpuMatrix logits = Forward(input);
            return CrossEntropyOnly(logits, targets, Config.VocabSize);
        }

        /// <inheritdoc/>
        public float AccumulateStep(int[] input, int[] targets)
        {
            ArgumentNullException.ThrowIfNull(targets);
            using GpuMatrix logits = Forward(input);
            float loss = CrossEntropyGrad(logits, targets, Config.VocabSize, out GpuMatrix dLogits);
            try { Backward(dLogits); }
            finally { dLogits.Dispose(); }
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
            UpdateAllParameters(adamStep);
        }

        /// <inheritdoc/>
        public void ClipAndUpdate(int adamStep, float lr)
        {
            ClipGradients();
            UpdateAllParameters(adamStep, lr);
        }

        /// <inheritdoc/>
        public void ClearKVCache() { /* GPU generate does not use a KV cache */ }

        private static float CrossEntropyOnly(GpuMatrix logits, int[] targets, int vocabSize)
        {
            GpuContext.Sync();
            float[] flat  = logits.DownloadFlat();
            int     T     = logits.Rows;
            int     V     = vocabSize;
            float   total = 0f;

            for (int t = 0; t < T; t++)
            {
                int   tOff   = t * V;
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (flat[tOff + j] > maxVal) maxVal = flat[tOff + j];

                float sumExp = 0f;
                for (int j = 0; j < V; j++)
                    sumExp += MathF.Exp(flat[tOff + j] - maxVal);

                float prob = MathF.Exp(flat[tOff + targets[t]] - maxVal) / sumExp;
                total -= MathF.Log(Math.Max(prob, 1e-8f));
            }

            return total / T;
        }

        private static float CrossEntropyGrad(
            GpuMatrix logits, int[] targets, int vocabSize, out GpuMatrix dLogits)
        {
            GpuContext.Sync();
            float[] flat     = logits.DownloadFlat();
            int T            = logits.Rows;
            int V            = vocabSize;
            var gradFlat     = new float[T * V];
            float total      = 0f;
            float invT       = 1f / T;

            for (int t = 0; t < T; t++)
            {
                int tOff = t * V;
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < V; j++)
                    if (flat[tOff + j] > maxVal) maxVal = flat[tOff + j];

                float sumExp = 0f;
                for (int j = 0; j < V; j++)
                {
                    float e = MathF.Exp(flat[tOff + j] - maxVal);
                    gradFlat[tOff + j] = e;
                    sumExp += e;
                }

                int correct = targets[t];
                for (int j = 0; j < V; j++)
                {
                    float prob = gradFlat[tOff + j] / sumExp;
                    gradFlat[tOff + j] = prob;
                    if (j == correct) total -= MathF.Log(Math.Max(prob, 1e-8f));
                }

                gradFlat[tOff + correct] -= 1f;
                for (int j = 0; j < V; j++) gradFlat[tOff + j] *= invT;
            }

            dLogits = new GpuMatrix(T, V);
            dLogits.UploadFlat(gradFlat);
            return total / T;
        }

        // ── forward pass ──────────────────────────────────────────────────────
        internal GpuMatrix Forward(int[] tokenIds)
        {
            GpuMatrix x = EmbeddingLayer.Forward(tokenIds);

            for (int i = 0; i < Config.NumLayers; i++)
            {
                x = Blocks[i].Forward(x);
            }

            _cachedNormed?.Dispose();
            _cachedNormed = FinalNorm.Forward(x);
            x.Dispose();

            // logits = normed · Wout + bout
            GpuMatrix logits;
            {
                using var tmp = GpuMatrix.Dot(_cachedNormed, OutputProjection.Weight);
                logits = tmp.AddBias(OutputBias.Weight);
            }
            return logits;
        }

        // ── backward pass ─────────────────────────────────────────────────────
        internal void Backward(GpuMatrix dLogits)
        {
            if (_cachedNormed is null)
                throw new InvalidOperationException("Backward called before Forward.");

            // Output projection backward
            {
                using var normedT = _cachedNormed.Transpose();
                using var g       = GpuMatrix.Dot(normedT, dLogits);
                OutputProjection.Gradient.AddInPlace(g);
            }
            AccumulateBiasGrad(OutputBias, dLogits, Config.VocabSize);

            GpuMatrix dX;
            {
                using var WoutT = OutputProjection.Weight.Transpose();
                GpuMatrix preFinal = GpuMatrix.Dot(dLogits, WoutT);
                dX = FinalNorm.Backward(preFinal);
                preFinal.Dispose();
            }

            for (int i = Config.NumLayers - 1; i >= 0; i--)
            {
                GpuMatrix prev = Blocks[i].Backward(dX);
                dX.Dispose();
                dX = prev;
            }

            EmbeddingLayer.Backward(dX);
            dX.Dispose();
        }

        // ── gradient management ───────────────────────────────────────────────
        public void ZeroAllGradients()
        {
            foreach (var p in AllParameters()) p.ZeroGrad();
        }

        public void ClipGradients()
        {
            float maxNorm = Config.GradClip;

            // Compute global gradient L2 norm on CPU (download each gradient).
            // Total gradient data ~3 MB for the default model; acceptable once/step.
            GpuContext.Sync();
            double sumSq = 0.0;
            foreach (var p in AllParameters())
            {
                float[] g = p.GetGradientFlat();
                foreach (float v in g) sumSq += v * (double)v;
            }

            float norm = (float)Math.Sqrt(sumSq);
            if (norm > maxNorm)
            {
                float scale = maxNorm / norm;
                foreach (var p in AllParameters())
                    p.ScaleGradient(scale);
            }
        }

        public void UpdateAllParameters(int step) =>
            UpdateAllParameters(step, Config.LearningRate);

        public void UpdateAllParameters(int step, float lr)
        {
            float beta1 = Config.Beta1;
            float beta2 = Config.Beta2;
            float eps   = Config.AdamEps;
            foreach (var p in AllParameters())
                p.Update(lr, beta1, beta2, eps, step);
        }

        // ── text generation ───────────────────────────────────────────────────
        public int[] Generate(
            int[] promptIds, int numTokens,
            float temperature = 1.0f, int topK = 40)
        {
            var context   = new System.Collections.Generic.List<int>(promptIds);
            var generated = new int[numTokens];

            for (int step = 0; step < numTokens; step++)
            {
                int start = Math.Max(0, context.Count - Config.ContextLength);
                int[] input = context.GetRange(start, context.Count - start).ToArray();

                using var logits = Forward(input);
                float[] lastLogits = logits.GetRow(input.Length - 1);

                if (temperature != 1.0f)
                    for (int i = 0; i < lastLogits.Length; i++)
                        lastLogits[i] /= temperature;

                if (topK > 0 && topK < lastLogits.Length)
                    ApplyTopKMask(lastLogits, topK);

                int nextToken = SampleFromLogits(lastLogits);
                generated[step] = nextToken;
                context.Add(nextToken);
            }

            return generated;
        }

        // ── helpers ───────────────────────────────────────────────────────────
        private static void AccumulateBiasGrad(GpuParameter bias, GpuMatrix dMat, int dim)
        {
            float[] sum  = dMat.SumOverRows();
            float[] flat = bias.Gradient.DownloadFlat();
            for (int j = 0; j < dim; j++) flat[j] += sum[j];
            bias.Gradient.UploadFlat(flat);
        }

        private static void ApplyTopKMask(float[] logits, int k)
        {
            float[] copy = (float[])logits.Clone();
            Array.Sort(copy);
            Array.Reverse(copy);
            float threshold = copy[k - 1];
            for (int i = 0; i < logits.Length; i++)
                if (logits[i] < threshold)
                    logits[i] = float.NegativeInfinity;
        }

        private int SampleFromLogits(float[] logits)
        {
            int V = logits.Length;
            float maxV = float.NegativeInfinity;
            for (int i = 0; i < V; i++)
                if (logits[i] > maxV) maxV = logits[i];

            float sum   = 0f;
            float[] probs = new float[V];
            for (int i = 0; i < V; i++)
            {
                probs[i] = MathF.Exp(logits[i] - maxV);
                sum += probs[i];
            }
            for (int i = 0; i < V; i++) probs[i] /= sum;

            double u = _rng.NextDouble();
            double cumulative = 0.0;
            for (int i = 0; i < V - 1; i++)
            {
                cumulative += probs[i];
                if (u <= cumulative) return i;
            }
            return V - 1;
        }

        // ── parameter enumeration ─────────────────────────────────────────────
        internal IEnumerable<IParameter> AllParameters()
        {
            foreach (var p in EmbeddingLayer.Parameters()) yield return p;
            foreach (var block in Blocks)
                foreach (var p in block.Parameters()) yield return p;
            foreach (var p in FinalNorm.Parameters()) yield return p;
            yield return OutputProjection;
            yield return OutputBias;
        }

        public long CountParameters()
        {
            long total = 0;
            foreach (var p in AllParameters()) total += p.Rows * p.Cols;
            return total;
        }

        public long ParameterCount => CountParameters();

        public override string ToString() =>
            $"GpuTransformerModel({Config}) | {CountParameters():N0} parameters";

        public void Dispose()
        {
            EmbeddingLayer.Dispose();
            foreach (var b in Blocks) b.Dispose();
            FinalNorm.Dispose();
            OutputProjection.Dispose();
            OutputBias.Dispose();
            _cachedNormed?.Dispose();
            GpuContext.Shutdown();     // mirror of Initialize() in the constructor
        }
    }
}
