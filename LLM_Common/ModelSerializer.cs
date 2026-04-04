using System;
using System.Collections.Generic;
using System.IO;

namespace LLM
{
    /// <summary>
    /// Saves and loads trained model weights to/from a binary file.
    ///
    /// File format:
    ///   [int32]  magic     = 0x4C4C4D01
    ///   [int32]  version   = 1
    ///   [int32]  VocabSize
    ///   [int32]  EmbeddingDim
    ///   [int32]  NumHeads
    ///   [int32]  NumLayers
    ///   [int32]  FFNDim
    ///   [int32]  ContextLength
    ///   [int32]  paramCount
    ///   for each parameter:
    ///     [int32]         rows
    ///     [int32]         cols
    ///     [rows*cols * float32]  weights (row-major)
    /// </summary>
    public static class ModelSerializer
    {
        /// <summary>
        /// Magic number written at the start of every weights file.
        /// The first three bytes spell 'L','L','M' in ASCII (0x4C, 0x4C, 0x4D);
        /// the low byte (0x01) identifies this as a weights-only file.
        /// </summary>
        private const int MagicWeights    = 0x4C4C4D01;

        /// <summary>
        /// Magic number written at the start of every checkpoint file.
        /// Shares the 'L','L','M' prefix with <see cref="MagicWeights"/>;
        /// the low byte (0x02) distinguishes it as a full training checkpoint.
        /// </summary>
        private const int MagicCheckpoint = 0x4C4C4D02;

        /// <summary>Format version written into every weights file.</summary>
        private const int Version = 1;

        /// <summary>
        /// Saves model weights to a binary file.
        /// Only the weight tensors are written — Adam state is not included.
        /// </summary>
        /// <param name="path">Destination file path. Created or overwritten.</param>
        /// <param name="cfg">Model architecture configuration written into the file header.</param>
        /// <param name="parameters">All trainable parameters whose weights will be serialised.</param>
        public static void Save(string path, TransformerConfig cfg, IEnumerable<IParameter> parameters)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var w  = new BinaryWriter(fs);

            w.Write(MagicWeights);
            w.Write(Version);
            w.Write(cfg.VocabSize);
            w.Write(cfg.EmbeddingDim);
            w.Write(cfg.NumHeads);
            w.Write(cfg.NumLayers);
            w.Write(cfg.FFNDim);
            w.Write(cfg.ContextLength);

            // Collect parameters so we can write the count first.
            var paramList = new List<IParameter>(parameters);
            w.Write(paramList.Count);

            foreach (var p in paramList)
            {
                w.Write(p.Rows);
                w.Write(p.Cols);
                float[] flat = p.GetWeightsFlat();
                foreach (float v in flat)
                    w.Write(v);
            }
        }

        /// <summary>
        /// Reads only the architecture header from a saved weights or checkpoint file
        /// and overwrites the corresponding fields in <paramref name="cfg"/>.
        /// Training-only settings (LearningRate, Epochs, etc.) are left unchanged.
        /// Accepts both weights files (<see cref="MagicWeights"/>) and checkpoint files
        /// (<see cref="MagicCheckpoint"/>).
        /// </summary>
        /// <param name="path">Path to the weights or checkpoint file to read.</param>
        /// <param name="cfg">Configuration object whose architecture fields are overwritten.</param>
        /// <exception cref="InvalidDataException">
        /// Thrown when the file magic number is not recognised.
        /// </exception>
        public static void ReadConfig(string path, TransformerConfig cfg)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var r  = new BinaryReader(fs);

            int magic = r.ReadInt32();
            if (magic != MagicWeights && magic != MagicCheckpoint)
                throw new InvalidDataException($"Not a valid model or checkpoint file (bad magic: 0x{magic:X8}).");

            int version = r.ReadInt32();  // version field exists in both formats

            cfg.VocabSize     = r.ReadInt32();
            cfg.EmbeddingDim  = r.ReadInt32();
            cfg.NumHeads      = r.ReadInt32();
            cfg.NumLayers     = r.ReadInt32();
            cfg.FFNDim        = r.ReadInt32();
            cfg.ContextLength = r.ReadInt32();
        }

        /// <summary>
        /// Loads model weights from a weights file or a checkpoint file.
        /// When given a checkpoint file the Adam M/V moments and training metadata are
        /// silently skipped — only the weight tensors are restored.
        /// This allows a checkpoint to be used directly for inference without conversion.
        /// </summary>
        /// <param name="path">Path to a weights file or checkpoint file.</param>
        /// <param name="cfg">
        /// Architecture configuration used to validate that the file matches the current model.
        /// </param>
        /// <param name="parameters">
        /// All trainable parameters whose weights will be populated from the file.
        /// </param>
        /// <exception cref="InvalidDataException">
        /// Thrown when the file magic number is not recognised, the architecture does not match,
        /// or the parameter count or shapes differ from the model.
        /// </exception>
        public static void Load(string path, TransformerConfig cfg, IEnumerable<IParameter> parameters)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var r  = new BinaryReader(fs);

            int magic = r.ReadInt32();
            bool isCheckpoint = magic == MagicCheckpoint;
            if (magic != MagicWeights && !isCheckpoint)
                throw new InvalidDataException($"Not a valid model or checkpoint file (bad magic: 0x{magic:X8}).");

            int version = r.ReadInt32();

            int vocabSize     = r.ReadInt32();
            int embeddingDim  = r.ReadInt32();
            int numHeads      = r.ReadInt32();
            int numLayers     = r.ReadInt32();
            int ffnDim        = r.ReadInt32();
            int contextLength = r.ReadInt32();

            if (vocabSize    != cfg.VocabSize    ||
                embeddingDim != cfg.EmbeddingDim  ||
                numHeads     != cfg.NumHeads      ||
                numLayers    != cfg.NumLayers     ||
                ffnDim       != cfg.FFNDim        ||
                contextLength!= cfg.ContextLength)
            {
                throw new InvalidDataException(
                    $"Model file configuration does not match the current model.\n" +
                    $"  File:    VocabSize={vocabSize}, EmbeddingDim={embeddingDim}, " +
                    $"NumHeads={numHeads}, NumLayers={numLayers}, FFNDim={ffnDim}, ContextLength={contextLength}\n" +
                    $"  Current: VocabSize={cfg.VocabSize}, EmbeddingDim={cfg.EmbeddingDim}, " +
                    $"NumHeads={cfg.NumHeads}, NumLayers={cfg.NumLayers}, FFNDim={cfg.FFNDim}, ContextLength={cfg.ContextLength}");
            }

            // Checkpoint files have epoch/step metadata before paramCount — skip them.
            if (isCheckpoint)
            {
                r.ReadInt32();  // epoch
                r.ReadInt32();  // adamStep
                if (version >= 2) r.ReadInt32();  // innerStep
            }

            int paramCount = r.ReadInt32();
            var paramList  = new List<IParameter>(parameters);

            if (paramCount != paramList.Count)
                throw new InvalidDataException(
                    $"Parameter count mismatch: file has {paramCount}, model has {paramList.Count}.");

            foreach (var p in paramList)
            {
                int rows = r.ReadInt32();
                int cols = r.ReadInt32();
                if (rows != p.Rows || cols != p.Cols)
                    throw new InvalidDataException(
                        $"Parameter shape mismatch: file has [{rows}×{cols}], model expects [{p.Rows}×{p.Cols}].");

                int n    = rows * cols;
                var flat = new float[n];
                for (int i = 0; i < n; i++)
                    flat[i] = r.ReadSingle();
                p.LoadWeightsFlat(flat);

                // Checkpoint files store M and V after weights — skip them.
                if (isCheckpoint)
                {
                    r.BaseStream.Seek((long)n * 2 * sizeof(float), SeekOrigin.Current);
                }
            }
        }
        /// <summary>
        /// Saves a full training checkpoint: weights, Adam first and second moments,
        /// and the exact position in the training loop so that training can resume
        /// without replaying any already-applied gradient updates.
        ///
        /// <para>Checkpoint file format (version 2):</para>
        /// <code>
        ///   [int32]  magic          = 0x4C4C4D02
        ///   [int32]  version        = 2
        ///   [int32]  VocabSize
        ///   [int32]  EmbeddingDim
        ///   [int32]  NumHeads
        ///   [int32]  NumLayers
        ///   [int32]  FFNDim
        ///   [int32]  ContextLength
        ///   [int32]  epoch          (0-indexed epoch active at save time)
        ///   [int32]  adamStep       (total Adam steps completed)
        ///   [int32]  innerStep      (next inner-loop step to run within epoch)
        ///   [int32]  paramCount
        ///   for each parameter:
        ///     [int32]              rows
        ///     [int32]              cols
        ///     [rows*cols * float]  weights
        ///     [rows*cols * float]  M  (Adam first moment)
        ///     [rows*cols * float]  V  (Adam second moment)
        /// </code>
        /// </summary>
        /// <param name="path">Destination file path. Created or overwritten.</param>
        /// <param name="cfg">Model architecture configuration written into the header.</param>
        /// <param name="parameters">All trainable parameters to serialise.</param>
        /// <param name="completedEpoch">0-indexed epoch that was active when the checkpoint was saved.</param>
        /// <param name="adamStep">Total number of Adam optimiser steps completed so far.</param>
        /// <param name="innerStep">
        /// Next inner-loop step index to resume from within <paramref name="completedEpoch"/>.
        /// Pass <c>stepsPerEpoch</c> for an epoch-end checkpoint (signals the epoch is fully done).
        /// </param>
        public static void SaveCheckpoint(string path, TransformerConfig cfg, IEnumerable<IParameter> parameters, int completedEpoch, int adamStep, int innerStep)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var w  = new BinaryWriter(fs);

            w.Write(MagicCheckpoint);
            w.Write(2);           // version
            w.Write(cfg.VocabSize);
            w.Write(cfg.EmbeddingDim);
            w.Write(cfg.NumHeads);
            w.Write(cfg.NumLayers);
            w.Write(cfg.FFNDim);
            w.Write(cfg.ContextLength);
            w.Write(completedEpoch);
            w.Write(adamStep);
            w.Write(innerStep);

            var paramList = new List<IParameter>(parameters);
            w.Write(paramList.Count);

            foreach (var p in paramList)
            {
                w.Write(p.Rows);
                w.Write(p.Cols);
                foreach (float v in p.GetWeightsFlat()) w.Write(v);
                foreach (float v in p.GetMFlat())       w.Write(v);
                foreach (float v in p.GetVFlat())       w.Write(v);
            }
        }

        /// <summary>
        /// Loads a training checkpoint, restoring weights and Adam first/second moments
        /// so that training can resume exactly where it left off.
        /// Both version 1 (no inner step) and version 2 (with inner step) files are accepted.
        /// </summary>
        /// <param name="path">Path to the checkpoint file.</param>
        /// <param name="cfg">
        /// Architecture configuration used to validate that the checkpoint matches the model.
        /// </param>
        /// <param name="parameters">All trainable parameters to restore.</param>
        /// <returns>
        /// A tuple of <c>(epoch, adamStep, innerStep)</c>:
        /// <list type="bullet">
        ///   <item><term>epoch</term><description>0-indexed epoch that was active when saved.</description></item>
        ///   <item><term>adamStep</term><description>Total Adam steps completed at save time.</description></item>
        ///   <item><term>innerStep</term><description>
        ///     Next inner-loop step to run within <c>epoch</c>, or <c>-1</c> for version 1
        ///     checkpoints where the inner step was not stored (caller should derive it).
        ///   </description></item>
        /// </list>
        /// </returns>
        /// <exception cref="InvalidDataException">
        /// Thrown when the magic number is wrong, the version is unsupported, the architecture
        /// does not match, or parameter count/shapes differ.
        /// </exception>
        public static (int epoch, int adamStep, int innerStep) LoadCheckpoint(string path, TransformerConfig cfg, IEnumerable<IParameter> parameters)
        {
            ArgumentNullException.ThrowIfNull(cfg);
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var r  = new BinaryReader(fs);

            int magic = r.ReadInt32();
            if (magic != MagicCheckpoint)
                throw new InvalidDataException($"Not a valid checkpoint file (bad magic: 0x{magic:X8}).");

            int version = r.ReadInt32();
            if (version != 1 && version != 2)
                throw new InvalidDataException($"Unsupported checkpoint version {version}.");

            int vocabSize     = r.ReadInt32();
            int embeddingDim  = r.ReadInt32();
            int numHeads      = r.ReadInt32();
            int numLayers     = r.ReadInt32();
            int ffnDim        = r.ReadInt32();
            int contextLength = r.ReadInt32();

            if (vocabSize     != cfg.VocabSize     ||
                embeddingDim  != cfg.EmbeddingDim  ||
                numHeads      != cfg.NumHeads      ||
                numLayers     != cfg.NumLayers     ||
                ffnDim        != cfg.FFNDim        ||
                contextLength != cfg.ContextLength)
            {
                throw new InvalidDataException(
                    $"Checkpoint configuration does not match the current model.\n" +
                    $"  File:    VocabSize={vocabSize}, EmbeddingDim={embeddingDim}, " +
                    $"NumHeads={numHeads}, NumLayers={numLayers}, FFNDim={ffnDim}, ContextLength={contextLength}\n" +
                    $"  Current: VocabSize={cfg.VocabSize}, EmbeddingDim={cfg.EmbeddingDim}, " +
                    $"NumHeads={cfg.NumHeads}, NumLayers={cfg.NumLayers}, FFNDim={cfg.FFNDim}, ContextLength={cfg.ContextLength}");
            }

            int completedEpoch = r.ReadInt32();
            int adamStep       = r.ReadInt32();
            int innerStep      = version >= 2 ? r.ReadInt32() : -1;  // -1 = v1, no inner step stored
            int paramCount     = r.ReadInt32();

            var paramList = new List<IParameter>(parameters);
            if (paramCount != paramList.Count)
                throw new InvalidDataException(
                    $"Parameter count mismatch: checkpoint has {paramCount}, model has {paramList.Count}.");

            foreach (var p in paramList)
            {
                int rows = r.ReadInt32();
                int cols = r.ReadInt32();
                if (rows != p.Rows || cols != p.Cols)
                    throw new InvalidDataException(
                        $"Parameter shape mismatch: checkpoint has [{rows}×{cols}], model expects [{p.Rows}×{p.Cols}].");

                int n = rows * cols;
                var weights = new float[n];
                var mFlat   = new float[n];
                var vFlat   = new float[n];
                for (int i = 0; i < n; i++) weights[i] = r.ReadSingle();
                for (int i = 0; i < n; i++) mFlat[i]   = r.ReadSingle();
                for (int i = 0; i < n; i++) vFlat[i]   = r.ReadSingle();
                p.LoadWeightsFlat(weights);
                p.LoadMFlat(mFlat);
                p.LoadVFlat(vFlat);
            }

            return (completedEpoch, adamStep, innerStep);
        }
    }
}
