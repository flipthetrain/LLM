using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// Unigram Language Model tokenizer, trained from scratch on a text corpus
    /// (Kudo 2018 – "Subword Regularization").  Used by T5, ALBERT, mBART.
    ///
    /// Unlike BPE / WordPiece which build up a vocabulary by merging,
    /// Unigram starts with a large seed vocabulary and prunes it down:
    ///
    ///   1. Seed: collect all substrings up to length <see cref="MaxSeedPieceLen"/>
    ///      from the ▁-normalised corpus; keep the most frequent ones.
    ///   2. EM: run several Expectation-Maximisation iterations —
    ///        E-step: find the Viterbi-optimal segmentation of each word.
    ///        M-step: re-estimate log P(token) from segmentation counts.
    ///   3. Prune: for each non-essential token compute how much the corpus
    ///      log-likelihood would decrease if the token were removed.
    ///      Remove the least-damaging tokens until the target vocab size is met.
    ///   4. Repeat EM + Prune until done; run a final EM pass.
    ///
    /// Encoding uses the Viterbi algorithm on the ▁-normalised word to find
    /// the maximum-probability segmentation under the learned unigram model.
    ///
    /// Word-boundary convention: ▁ (U+2581) is prepended to the first character
    /// of every word (same as SentencePiece / <see cref="SentencePieceTokenizer"/>).
    /// Decoding replaces ▁ with a space and strips the leading space.
    /// </summary>
    public sealed class UnigramTokenizer : ITokenizer
    {
        private const char   Sp             = '\u2581';  // ▁
        private const int    MaxSeedPieceLen = 16;
        private const int    SeedFactor      = 10;       // seed vocab = target × SeedFactor
        private const int    EmIterPerRound  = 5;
        private const double PrunePercent    = 0.20;     // remove 20 % of vocab per round

        public const string UnkToken = "<unk>";
        public const string BosToken = "<s>";
        public const string EosToken = "</s>";

        private readonly string[]               _idToToken;
        private readonly Dictionary<string, int> _tokenToId;
        private readonly double[]               _logProb;    // log P(token[id])
        private readonly int                    _maxLen;     // max token length in chars

        public int VocabSize => _idToToken.Length;
        public int UnknownId { get; }

        /// <param name="corpus">Full training text.</param>
        /// <param name="targetVocabSize">
        /// Final vocabulary size including the three special tokens.
        /// Typical range: 500–32 000.
        /// </param>
        public UnigramTokenizer(string corpus, int targetVocabSize = 1000)
        {
            if (string.IsNullOrEmpty(corpus))
                throw new ArgumentException("Corpus must not be empty.", nameof(corpus));

            Dictionary<string, int> wordFreq = CountWords(corpus);

            // ── 1. Build seed vocabulary ───────────────────────────────────────
            // For each word, collect all substrings of its ▁-normalised form.
            int targetTokens = targetVocabSize - 3;   // reserve 3 for special tokens
            int seedSize     = Math.Max(targetTokens * SeedFactor, targetTokens + 100);

            var subFreq = new Dictionary<string, long>();
            foreach (var (word, freq) in wordFreq)
            {
                string ws = Sp + word;                        // "▁hello"
                for (int s = 0; s < ws.Length; s++)
                    for (int len = 1; len <= Math.Min(MaxSeedPieceLen, ws.Length - s); len++)
                    {
                        string piece = ws.Substring(s, len);
                        subFreq.TryGetValue(piece, out long cur);
                        subFreq[piece] = cur + freq;
                    }
            }

            // Always keep every single char (and ▁+char) for full coverage.
            var logProbs  = new Dictionary<string, double>();
            foreach (var (piece, _) in subFreq)
                if (IsSingleChar(piece)) logProbs[piece] = 0.0;

            // Fill up to seedSize with most-frequent substrings.
            var byFreq = new List<KeyValuePair<string, long>>(subFreq);
            byFreq.Sort((a, b) => b.Value.CompareTo(a.Value));
            foreach (var kv in byFreq)
            {
                if (logProbs.Count >= seedSize) break;
                if (!logProbs.ContainsKey(kv.Key)) logProbs[kv.Key] = 0.0;
            }

            // Uniform initialisation.
            double initLp = -Math.Log(logProbs.Count);
            foreach (string k in new List<string>(logProbs.Keys)) logProbs[k] = initLp;

            // ── 2. EM + Prune loop ─────────────────────────────────────────────
            Dictionary<string, long>? lastCounts = null;

            while (logProbs.Count > targetTokens)
            {
                // ── EM iterations ──────────────────────────────────────────────
                for (int em = 0; em < EmIterPerRound; em++)
                {
                    var counts = new Dictionary<string, long>(logProbs.Count);
                    long total = 0;

                    foreach (var (word, freq) in wordFreq)
                    {
                        string ws  = Sp + word;
                        List<string> seg = ViterbiSegment(ws, logProbs, MaxSeedPieceLen);
                        foreach (string tok in seg)
                        {
                            counts.TryGetValue(tok, out long c);
                            counts[tok] = c + freq;
                            total += freq;
                        }
                    }

                    if (total > 0)
                    {
                        double logTotal = Math.Log(total);
                        foreach (string tok in new List<string>(logProbs.Keys))
                            logProbs[tok] = (counts.TryGetValue(tok, out long c) && c > 0)
                                ? Math.Log(c) - logTotal
                                : double.NegativeInfinity;
                    }
                    lastCounts = counts;
                }

                // Remove never-used tokens (keep single chars).
                foreach (string tok in new List<string>(logProbs.Keys))
                    if (logProbs[tok] == double.NegativeInfinity && !IsSingleChar(tok))
                        logProbs.Remove(tok);

                if (logProbs.Count <= targetTokens) break;

                // ── Pruning step ───────────────────────────────────────────────
                int targetNow = Math.Max(
                    (int)(logProbs.Count * (1.0 - PrunePercent)),
                    targetTokens);

                var losses = new List<(string tok, double loss)>();
                foreach (var (tok, lp) in logProbs)
                {
                    if (IsSingleChar(tok)) continue;          // always keep

                    double fallback = ViterbiLogProb(tok, logProbs, MaxSeedPieceLen, exclude: tok);
                    long count = (lastCounts != null && lastCounts.TryGetValue(tok, out long c))
                        ? c : 0L;

                    // loss_increase ≥ 0: bigger means this token is more important.
                    double loss = (count > 0 && fallback > double.NegativeInfinity)
                        ? (double)count * (lp - fallback)
                        : 0.0;
                    losses.Add((tok, loss));
                }

                // Remove least-important tokens first (ascending by loss_increase).
                losses.Sort((a, b) => a.loss.CompareTo(b.loss));
                int removeN = logProbs.Count - targetNow;
                for (int i = 0; i < removeN && i < losses.Count; i++)
                    logProbs.Remove(losses[i].tok);

                if (logProbs.Count <= targetTokens) break;
            }

            // ── 3. Final EM pass to re-estimate log-probs ─────────────────────
            {
                var counts = new Dictionary<string, long>(logProbs.Count);
                long total = 0;
                foreach (var (word, freq) in wordFreq)
                {
                    string ws = Sp + word;
                    foreach (string tok in ViterbiSegment(ws, logProbs, MaxSeedPieceLen))
                    {
                        counts.TryGetValue(tok, out long c);
                        counts[tok] = c + freq;
                        total += freq;
                    }
                }
                if (total > 0)
                {
                    double logTotal = Math.Log(total);
                    foreach (string tok in new List<string>(logProbs.Keys))
                        logProbs[tok] = (counts.TryGetValue(tok, out long c) && c > 0)
                            ? Math.Log(c) - logTotal
                            : double.NegativeInfinity;
                }
                // Remove any remaining zero-count non-essential tokens.
                foreach (string tok in new List<string>(logProbs.Keys))
                    if (logProbs[tok] == double.NegativeInfinity && !IsSingleChar(tok))
                        logProbs.Remove(tok);
            }

            // ── 4. Assemble final vocab ────────────────────────────────────────
            // Sort tokens by log-probability descending (most likely first).
            var tokenList = new List<string>(logProbs.Keys);
            tokenList.Sort((a, b) => logProbs[b].CompareTo(logProbs[a]));

            var vocabList = new List<string> { UnkToken, BosToken, EosToken };
            var tokenToId = new Dictionary<string, int>
            { [UnkToken] = 0, [BosToken] = 1, [EosToken] = 2 };
            UnknownId = 0;
            foreach (string tok in tokenList)
                if (!tokenToId.ContainsKey(tok))
                { tokenToId[tok] = vocabList.Count; vocabList.Add(tok); }

            _idToToken = vocabList.ToArray();
            _tokenToId = tokenToId;

            // Build the log-prob array aligned to vocab IDs.
            _logProb = new double[vocabList.Count];
            for (int i = 3; i < vocabList.Count; i++)
                _logProb[i] = logProbs.TryGetValue(vocabList[i], out double lp)
                    ? lp
                    : double.NegativeInfinity;
            // Special tokens get -inf so Viterbi never places them.
            _logProb[0] = _logProb[1] = _logProb[2] = double.NegativeInfinity;

            _maxLen = MaxSeedPieceLen + 1;  // +1 for the ▁ prefix char
        }

        // ── encoding ──────────────────────────────────────────────────────────

        public int[] Encode(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            var result = new List<int>();
            foreach (string word in SplitWords(text))
            {
                string ws = Sp + word;
                List<int> ids = ViterbiInfer(ws);
                if (ids.Count == 0) result.Add(UnknownId);
                else foreach (int id in ids) result.Add(id);
            }
            return result.ToArray();
        }

        // ── decoding ──────────────────────────────────────────────────────────

        public string Decode(int[] ids)
        {
            ArgumentNullException.ThrowIfNull(ids);
            var sb = new StringBuilder();
            foreach (int id in ids) sb.Append(DecodeToken(id));
            string s = sb.ToString();
            if (s.Length > 0 && s[0] == ' ') s = s.Substring(1);
            return s;
        }

        /// <summary>
        /// Single-token surface form.  ▁ → space; BOS/EOS → empty string.
        /// </summary>
        public string DecodeToken(int id)
        {
            if (id < 0 || id >= _idToToken.Length) return UnkToken;
            string tok = _idToToken[id];
            if (tok == BosToken || tok == EosToken) return "";
            if (tok == UnkToken) return " [UNK]";
            return tok.Replace(Sp, ' ');
        }

        // ── vocab display ─────────────────────────────────────────────────────

        public void PrintVocab()
        {
            int show = Math.Min(VocabSize, 64);
            Console.WriteLine($"UnigramTokenizer vocabulary ({VocabSize} tokens, showing first {show}):");
            for (int i = 0; i < show; i++)
            {
                string label = i >= 3
                    ? $"{_idToToken[i],-10} ({_logProb[i]:F3})"
                    : $"{_idToToken[i],-10}";
                Console.Write($"  {i,4}: {label,-20}");
                if ((i + 1) % 3 == 0) Console.WriteLine();
            }
            if (show % 3 != 0) Console.WriteLine();
            Console.WriteLine();
        }

        public override string ToString() => $"UnigramTokenizer(VocabSize={VocabSize})";

        // ── vocab persistence ─────────────────────────────────────────────────

        /// <summary>
        /// Save format (UTF-8):
        ///   Line 0: "UnigramTokenizer"
        ///   Lines 1+: &lt;escaped_token&gt;\t&lt;log_probability&gt;  (line index = token ID)
        /// Log-probability is "-Infinity" for special tokens; otherwise "R" round-trip format.
        /// </summary>
        public void SaveVocab(string path)
        {
            using var w = new StreamWriter(path, false, Encoding.UTF8);
            w.WriteLine(nameof(UnigramTokenizer));
            for (int i = 0; i < _idToToken.Length; i++)
            {
                string lpStr = double.IsNegativeInfinity(_logProb[i])
                    ? "-Infinity"
                    : _logProb[i].ToString("R", System.Globalization.CultureInfo.InvariantCulture);
                w.WriteLine($"{TokenizerIO.EscapeToken(_idToToken[i])}\t{lpStr}");
            }
        }

        internal static UnigramTokenizer LoadFrom(StreamReader reader)
        {
            var tokens   = new List<string>();
            var logProbs = new List<double>();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                int tab = line.IndexOf('\t', StringComparison.Ordinal);
                if (tab < 0) continue;
                tokens.Add(TokenizerIO.UnescapeToken(line.Substring(0, tab)));
                string lpStr = line.Substring(tab + 1);
                logProbs.Add(lpStr == "-Infinity"
                    ? double.NegativeInfinity
                    : double.Parse(lpStr, System.Globalization.CultureInfo.InvariantCulture));
            }
            return new UnigramTokenizer(tokens.ToArray(), logProbs.ToArray());
        }

        private UnigramTokenizer(string[] idToToken, double[] logProb)
        {
            _idToToken = idToToken;
            _logProb   = logProb;
            _tokenToId = new Dictionary<string, int>(idToToken.Length);
            UnknownId  = 0;
            for (int i = 0; i < idToToken.Length; i++)
                _tokenToId[idToToken[i]] = i;

            int maxLen = 1;
            for (int i = 0; i < idToToken.Length; i++)
                if (idToToken[i].Length > maxLen) maxLen = idToToken[i].Length;
            _maxLen = maxLen;
        }

        // ── private helpers ───────────────────────────────────────────────────

        /// <summary>
        /// Viterbi segmentation used during training (takes the live logProbs dict).
        /// Returns the token strings of the best-probability segmentation.
        /// </summary>
        private static List<string> ViterbiSegment(
            string                    str,
            Dictionary<string, double> logProbs,
            int                        maxLen)
        {
            int n = str.Length;
            var dp   = new double[n + 1];
            var back = new int[n + 1];   // length of best token ending at position i
            for (int i = 1; i <= n; i++) dp[i] = double.NegativeInfinity;

            for (int i = 0; i < n; i++)
            {
                if (i > 0 && dp[i] == double.NegativeInfinity) continue;
                for (int len = 1; len <= Math.Min(maxLen, n - i); len++)
                {
                    string piece = str.Substring(i, len);
                    if (logProbs.TryGetValue(piece, out double lp) &&
                        lp > double.NegativeInfinity)
                    {
                        double score = dp[i] + lp;
                        if (score > dp[i + len]) { dp[i + len] = score; back[i + len] = len; }
                    }
                }
            }

            if (dp[n] == double.NegativeInfinity) return new List<string>();

            var result = new List<string>();
            int pos = n;
            while (pos > 0) { int len = back[pos]; result.Add(str.Substring(pos - len, len)); pos -= len; }
            result.Reverse();
            return result;
        }

        /// <summary>
        /// Viterbi log-probability used for loss_increase computation during pruning.
        /// When <paramref name="exclude"/> is set that token is skipped.
        /// </summary>
        private static double ViterbiLogProb(
            string                    str,
            Dictionary<string, double> logProbs,
            int                        maxLen,
            string?                    exclude = null)
        {
            int n = str.Length;
            var dp = new double[n + 1];
            for (int i = 1; i <= n; i++) dp[i] = double.NegativeInfinity;

            for (int i = 0; i < n; i++)
            {
                if (i > 0 && dp[i] == double.NegativeInfinity) continue;
                for (int len = 1; len <= Math.Min(maxLen, n - i); len++)
                {
                    string piece = str.Substring(i, len);
                    if (piece == exclude) continue;
                    if (logProbs.TryGetValue(piece, out double lp) &&
                        lp > double.NegativeInfinity)
                    {
                        double score = dp[i] + lp;
                        if (score > dp[i + len]) dp[i + len] = score;
                    }
                }
            }
            return dp[n];
        }

        /// <summary>Viterbi segmentation at inference time using the stored ID/logProb arrays.</summary>
        private List<int> ViterbiInfer(string str)
        {
            int n = str.Length;
            var dp   = new double[n + 1];
            var back = new int[n + 1];
            for (int i = 1; i <= n; i++) dp[i] = double.NegativeInfinity;

            for (int i = 0; i < n; i++)
            {
                if (i > 0 && dp[i] == double.NegativeInfinity) continue;
                for (int len = 1; len <= Math.Min(_maxLen, n - i); len++)
                {
                    string piece = str.Substring(i, len);
                    if (_tokenToId.TryGetValue(piece, out int tid) &&
                        _logProb[tid] > double.NegativeInfinity)
                    {
                        double score = dp[i] + _logProb[tid];
                        if (score > dp[i + len]) { dp[i + len] = score; back[i + len] = len; }
                    }
                }
            }

            if (dp[n] == double.NegativeInfinity) return new List<int>();

            var result = new List<int>();
            int pos = n;
            while (pos > 0)
            {
                int len = back[pos];
                string tok = str.Substring(pos - len, len);
                result.Add(_tokenToId.TryGetValue(tok, out int id) ? id : UnknownId);
                pos -= len;
            }
            result.Reverse();
            return result;
        }

        /// <summary>Returns true for single-character tokens that must be kept for full coverage.</summary>
        private static bool IsSingleChar(string tok) =>
            tok.Length == 1 || (tok.Length == 2 && tok[0] == Sp);

        private static IEnumerable<string> SplitWords(string text)
        {
            int start = -1;
            for (int i = 0; i <= text.Length; i++)
            {
                bool ws = i == text.Length || char.IsWhiteSpace(text[i]);
                if (!ws && start < 0) start = i;
                else if (ws && start >= 0)
                { yield return text.Substring(start, i - start); start = -1; }
            }
        }

        private static Dictionary<string, int> CountWords(string corpus)
        {
            var freq = new Dictionary<string, int>();
            foreach (string w in SplitWords(corpus))
            { freq.TryGetValue(w, out int c); freq[w] = c + 1; }
            return freq;
        }
    }
}
