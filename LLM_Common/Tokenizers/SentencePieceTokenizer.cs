using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// SentencePiece-style BPE tokenizer (as used in LLaMA / Mistral / Gemma).
    ///
    /// Differences from <see cref="BpeTokenizer"/>:
    ///   • Word boundaries are marked with the prefix character ▁ (U+2581) instead of
    ///     a "</w>" suffix.  "hello world" → ["▁h","e","l","l","o","▁w","o","r","l","d"].
    ///   • No whitespace pre-tokenisation beyond splitting into words — the ▁ character
    ///     itself encodes where spaces were, so the vocabulary is language-agnostic.
    ///   • Special tokens &lt;unk&gt; (0), &lt;s&gt; (1), &lt;/s&gt; (2) are prepended to the vocab.
    ///   • Decoding: replace every ▁ with a space and strip the leading space.
    ///
    /// Training applies BPE merges on the ▁-normalised word representations so that
    /// common word beginnings (▁hello, ▁the, ▁to …) become single tokens.
    /// </summary>
    public sealed class SentencePieceTokenizer : ITokenizer
    {
        private const char Sp = '\u2581';   // ▁ – SentencePiece word-boundary marker

        public const string UnkToken = "<unk>";
        public const string BosToken = "<s>";
        public const string EosToken = "</s>";

        private readonly string[]                          _idToToken;
        private readonly Dictionary<string, int>           _tokenToId;
        private readonly Dictionary<(string, string), int> _mergeRank;

        public int VocabSize => _idToToken.Length;
        public int UnknownId { get; }

        /// <param name="corpus">Full training text.</param>
        /// <param name="numMerges">
        /// BPE merge operations to perform.  Final vocab ≈ (unique chars) + 3 special
        /// tokens + numMerges.  Typical range: 500–50 000.
        /// </param>
        public SentencePieceTokenizer(string corpus, int numMerges = 1000)
        {
            if (string.IsNullOrEmpty(corpus))
                throw new ArgumentException("Corpus must not be empty.", nameof(corpus));

            Dictionary<string, int> wordFreq = CountWords(corpus);

            // ── 1. Initial ▁-normalised char representations ───────────────────
            var wordTokens = new Dictionary<string, List<string>>(wordFreq.Count);
            var charSet    = new SortedSet<string>();
            foreach (string word in wordFreq.Keys)
            {
                List<string> toks = SpSplit(word);
                wordTokens[word]  = toks;
                foreach (string t in toks) charSet.Add(t);
            }

            // ── 2. Build initial vocabulary ────────────────────────────────────
            // Special tokens occupy IDs 0–2; sorted chars follow.
            var vocabList = new List<string> { UnkToken, BosToken, EosToken };
            var tokenToId = new Dictionary<string, int>
            {
                [UnkToken] = 0, [BosToken] = 1, [EosToken] = 2
            };
            UnknownId = 0;
            foreach (string t in charSet)
                if (!tokenToId.ContainsKey(t))
                { tokenToId[t] = vocabList.Count; vocabList.Add(t); }

            // ── 3. BPE merge loop ──────────────────────────────────────────────
            var mergeRank = new Dictionary<(string, string), int>();
            for (int step = 0; step < numMerges; step++)
            {
                var pairFreq = new Dictionary<(string, string), int>();
                foreach (var (word, freq) in wordFreq)
                {
                    List<string> toks = wordTokens[word];
                    for (int i = 0; i < toks.Count - 1; i++)
                    {
                        var pair = (toks[i], toks[i + 1]);
                        pairFreq.TryGetValue(pair, out int cur);
                        pairFreq[pair] = cur + freq;
                    }
                }
                if (pairFreq.Count == 0) break;

                (string, string) best  = default;
                int               bestF = -1;
                foreach (var (pair, freq) in pairFreq)
                    if (freq > bestF) { bestF = freq; best = pair; }

                string merged = best.Item1 + best.Item2;
                mergeRank[best] = step;

                if (!tokenToId.ContainsKey(merged))
                { tokenToId[merged] = vocabList.Count; vocabList.Add(merged); }

                foreach (string word in wordFreq.Keys)
                {
                    List<string> toks = wordTokens[word];
                    int i = 0;
                    while (i < toks.Count - 1)
                    {
                        if (toks[i] == best.Item1 && toks[i + 1] == best.Item2)
                        { toks[i] = merged; toks.RemoveAt(i + 1); }
                        else i++;
                    }
                }
            }

            _idToToken = vocabList.ToArray();
            _tokenToId = tokenToId;
            _mergeRank = mergeRank;
        }

        // ── encoding ──────────────────────────────────────────────────────────

        public int[] Encode(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            var result = new List<int>();
            foreach (string word in SplitWords(text))
            {
                int[] ids = TokenizeWord(word);
                foreach (int id in ids) result.Add(id);
            }
            return result.ToArray();
        }

        private int[] TokenizeWord(string word)
        {
            List<string> tokens = SpSplit(word);
            while (tokens.Count > 1)
            {
                int bestIdx = -1, bestRank = int.MaxValue;
                for (int i = 0; i < tokens.Count - 1; i++)
                    if (_mergeRank.TryGetValue((tokens[i], tokens[i + 1]), out int r) && r < bestRank)
                    { bestRank = r; bestIdx = i; }

                if (bestIdx == -1) break;
                tokens[bestIdx] = tokens[bestIdx] + tokens[bestIdx + 1];
                tokens.RemoveAt(bestIdx + 1);
            }

            int[] ids = new int[tokens.Count];
            for (int i = 0; i < tokens.Count; i++)
                ids[i] = _tokenToId.TryGetValue(tokens[i], out int id) ? id : UnknownId;
            return ids;
        }

        // ── decoding ──────────────────────────────────────────────────────────

        public string Decode(int[] ids)
        {
            ArgumentNullException.ThrowIfNull(ids);
            var sb = new StringBuilder();
            foreach (int id in ids) sb.Append(DecodeToken(id));
            // ▁ marks word starts; after joining, replace with space and trim the leading one.
            string s = sb.ToString();
            if (s.Length > 0 && s[0] == ' ') s = s.Substring(1);
            return s;
        }

        /// <summary>
        /// Returns the surface form of a single token for streaming generation.
        /// ▁ is replaced with a space so output reads naturally.
        /// BOS/EOS tokens return an empty string.
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
            Console.WriteLine($"SentencePieceTokenizer vocabulary ({VocabSize} tokens, showing first {show}):");
            for (int i = 0; i < show; i++)
            {
                Console.Write($"  {i,4}: {_idToToken[i],-14}");
                if ((i + 1) % 4 == 0) Console.WriteLine();
            }
            if (show % 4 != 0) Console.WriteLine();
            Console.WriteLine();
        }

        public override string ToString() => $"SentencePieceTokenizer(VocabSize={VocabSize})";

        // ── vocab persistence ─────────────────────────────────────────────────

        /// <summary>
        /// Save format (UTF-8):
        ///   Line 0:   "SentencePieceTokenizer"
        ///   Line 1:   "VOCAB"
        ///   Lines …:  one escaped token per line  (line index - 2 = token ID)
        ///   Then:     "MERGES"
        ///   Lines …:  &lt;escaped_left&gt;\t&lt;escaped_right&gt;  (in rank order, rank 0 first)
        /// </summary>
        public void SaveVocab(string path)
        {
            using var w = new StreamWriter(path, false, Encoding.UTF8);
            w.WriteLine(nameof(SentencePieceTokenizer));
            w.WriteLine("VOCAB");
            foreach (string tok in _idToToken)
                w.WriteLine(TokenizerIO.EscapeToken(tok));
            w.WriteLine("MERGES");
            var merges = new List<((string, string) pair, int rank)>(_mergeRank.Count);
            foreach (var kv in _mergeRank) merges.Add((kv.Key, kv.Value));
            merges.Sort((a, b) => a.rank.CompareTo(b.rank));
            foreach (var (pair, _) in merges)
                w.WriteLine($"{TokenizerIO.EscapeToken(pair.Item1)}\t{TokenizerIO.EscapeToken(pair.Item2)}");
        }

        internal static SentencePieceTokenizer LoadFrom(StreamReader reader)
        {
            // Expect "VOCAB" header
            reader.ReadLine();
            var tokens = new List<string>();
            string? line;
            while ((line = reader.ReadLine()) != null && line != "MERGES")
                tokens.Add(TokenizerIO.UnescapeToken(line));

            var mergeRank = new Dictionary<(string, string), int>();
            int rank = 0;
            while ((line = reader.ReadLine()) != null)
            {
                int tab = line.IndexOf('\t', StringComparison.Ordinal);
                if (tab < 0) continue;
                string left  = TokenizerIO.UnescapeToken(line.Substring(0, tab));
                string right = TokenizerIO.UnescapeToken(line.Substring(tab + 1));
                mergeRank[(left, right)] = rank++;
            }
            return new SentencePieceTokenizer(tokens.ToArray(), mergeRank);
        }

        private SentencePieceTokenizer(string[] idToToken, Dictionary<(string, string), int> mergeRank)
        {
            _idToToken = idToToken;
            _mergeRank = mergeRank;
            _tokenToId = new Dictionary<string, int>(idToToken.Length);
            UnknownId  = 0;
            for (int i = 0; i < idToToken.Length; i++)
                _tokenToId[idToToken[i]] = i;
        }

        // ── private helpers ───────────────────────────────────────────────────

        /// <summary>
        /// Initial SentencePiece split: prepend ▁ to the first character of each word.
        /// "hello" → ["▁h", "e", "l", "l", "o"]
        /// </summary>
        private static List<string> SpSplit(string word)
        {
            var tokens = new List<string>(word.Length);
            tokens.Add(Sp + word[0].ToString());   // ▁ + first char
            for (int i = 1; i < word.Length; i++)
                tokens.Add(word[i].ToString());
            return tokens;
        }

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
