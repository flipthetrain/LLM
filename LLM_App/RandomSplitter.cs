using System;
using System.Collections.Generic;

namespace LLM_App
{
    /// <summary>
    /// Splits the token array into fixed-size chunks, randomly assigns
    /// <paramref name="valFraction"/> of them to validation and the rest to training.
    /// Uses a fixed seed so the split is reproducible across runs.
    /// </summary>
    internal sealed class RandomSplitter : ICorpusSplitter
    {
        private readonly int    _chunkSize;
        private readonly double _valFraction;

        /// <param name="chunkSize">
        /// Size of each chunk in tokens. Should match ContextLength so each chunk
        /// produces a valid training or validation example.
        /// </param>
        public RandomSplitter(int chunkSize, double valFraction = 0.1)
        {
            if (valFraction <= 0 || valFraction >= 1)
                throw new ArgumentOutOfRangeException(nameof(valFraction), "Must be between 0 and 1 exclusive.");
            _chunkSize   = chunkSize;
            _valFraction = valFraction;
        }

        public (int[] Train, int[] Validation) Split(int[] allTokens)
        {
            int numChunks = allTokens.Length / _chunkSize;

            // Build shuffled index list with fixed seed for reproducibility
            int[] indices = new int[numChunks];
            for (int i = 0; i < numChunks; i++) indices[i] = i;

            var rng = new Random(42);
            for (int i = numChunks - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            int valChunks   = (int)(numChunks * _valFraction);
            int trainChunks = numChunks - valChunks;

            var trainTokens = new List<int>(trainChunks * _chunkSize);
            var valTokens   = new List<int>(valChunks   * _chunkSize);

            for (int i = 0; i < trainChunks; i++)
            {
                int start = indices[i] * _chunkSize;
                for (int j = 0; j < _chunkSize; j++)
                    trainTokens.Add(allTokens[start + j]);
            }

            for (int i = trainChunks; i < numChunks; i++)
            {
                int start = indices[i] * _chunkSize;
                for (int j = 0; j < _chunkSize; j++)
                    valTokens.Add(allTokens[start + j]);
            }

            return (trainTokens.ToArray(), valTokens.ToArray());
        }
    }
}
