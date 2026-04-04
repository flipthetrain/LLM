using System;

namespace LLM_App
{
    /// <summary>
    /// Holds out the last <paramref name="valFraction"/> of the token array as validation.
    /// The model never sees this text during training.
    /// </summary>
    internal sealed class TailSplitter : ICorpusSplitter
    {
        private readonly double _valFraction;

        public TailSplitter(double valFraction = 0.1)
        {
            if (valFraction <= 0 || valFraction >= 1)
                throw new ArgumentOutOfRangeException(nameof(valFraction), "Must be between 0 and 1 exclusive.");
            _valFraction = valFraction;
        }

        public (int[] Train, int[] Validation) Split(int[] allTokens)
        {
            int valCount   = (int)(allTokens.Length * _valFraction);
            int trainCount = allTokens.Length - valCount;

            int[] train = new int[trainCount];
            int[] val   = new int[valCount];
            Array.Copy(allTokens, 0,          train, 0, trainCount);
            Array.Copy(allTokens, trainCount, val,   0, valCount);
            return (train, val);
        }
    }
}
