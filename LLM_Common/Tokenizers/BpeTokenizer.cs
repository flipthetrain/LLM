using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// Byte Pair Encoding (BPE) tokenizer, trained from scratch on a text corpus.
    ///
    /// Algorithm (Sennrich et al., 2015 – "Neural Machine Translation of Rare Words
    /// with Subword Units"):
    ///   1. Pre-tokenise the corpus into whitespace-separated words.
    ///   2. Represent each word as individual characters with an end-of-word marker
    ///      appended to the last character (e.g. "hi" → ["h", "i&lt;/w&gt;"]).
    ///   3. Count every adjacent token pair, weighted by word frequency.
    ///   4. Merge the most-frequent pair into a single new token and record the rule.
    ///   5. Repeat for <paramref name="numMerges"/> steps or until no pairs remain.
    ///
    /// Encoding applies the learned merge rules in rank order (lowest rank = first
    /// learned = highest priority), greedily, per word.
    ///
    /// Decoding joins token strings; end-of-word markers become trailing spaces.
    /// </summary>
    public sealed class BpeTokenizer : ITokenizer
    {
        private const string Eow = "</w>";   // end-of-word suffix

        private readonly string[]                          _idToToken;
        private readonly Dictionary<string, int>           _tokenToId;
        private readonly Dictionary<(string, string), int> _mergeRank;

        public int VocabSize => _idToToken.Length;
        public int UnknownId { get; }

        /// <summary>
        /// Train a BPE tokenizer on <paramref name="corpus"/>.
        /// </summary>
        /// <param name="corpus">Full training text.</param>
        /// <param name="numMerges">
        /// Number of merge operations.  The final vocabulary size is approximately
        /// (unique characters) + 1 (&lt;UNK&gt;) + numMerges.
        /// Typical range: 500–50 000.
        /// </param>
        public BpeTokenizer(string corpus, int numMerges = 1000)
        {
            if (string.IsNullOrEmpty(corpus))
                throw new ArgumentException("Corpus must not be empty.", nameof(corpus));

            // ── 1. Word frequencies ────────────────────────────────────────────
            Dictionary<string, int> wordFreq = CountWords(corpus);

            // ── 2. Initial character-level word representations ────────────────
            var wordTokens = new Dictionary<string, List<string>>(wordFreq.Count);
            var charSet    = new SortedSet<string>();
            foreach (string word in wordFreq.Keys)
            {
                List<string> toks = CharSplit(word);
                wordTokens[word]  = toks;
                foreach (string t in toks) charSet.Add(t);
            }

            var vocabList = new List<string> { "<UNK>" };
            var tokenToId = new Dictionary<string, int> { ["<UNK>"] = 0 };
            foreach (string t in charSet)
            {
                tokenToId[t] = vocabList.Count;
                vocabList.Add(t);
            }

            // ── 3. BPE merge loop ──────────────────────────────────────────────
            var mergeRank = new Dictionary<(string, string), int>();
            for (int step = 0; step < numMerges; step++)
            {
                // Count adjacent pair frequencies weighted by word frequency
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

                // Find the most frequent pair
                (string, string) best  = default;
                int               bestF = -1;
                foreach (var (pair, freq) in pairFreq)
                    if (freq > bestF) { bestF = freq; best = pair; }

                string merged = best.Item1 + best.Item2;
                mergeRank[best] = step;

                if (!tokenToId.ContainsKey(merged))
                {
                    tokenToId[merged] = vocabList.Count;
                    vocabList.Add(merged);
                }

                // Apply merge to all word representations
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
            UnknownId  = 0;
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
            List<string> tokens = CharSplit(word);

            // Apply merges greedily in rank order (lowest rank = first learned = highest priority)
            while (tokens.Count > 1)
            {
                int bestIdx  = -1;
                int bestRank = int.MaxValue;
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
            return sb.ToString().TrimEnd();
        }

        /// <summary>
        /// Returns the surface form of a single token.
        /// End-of-word markers are replaced by a trailing space so that streaming
        /// output reads naturally (e.g. "hello</w>" → "hello ").
        /// </summary>
        public string DecodeToken(int id)
        {
            if (id < 0 || id >= _idToToken.Length) return "<UNK>";
            string t = _idToToken[id];
            return t.EndsWith(Eow, StringComparison.Ordinal)
                ? string.Concat(t.AsSpan(0, t.Length - Eow.Length), " ")
                : t;
        }

        // ── vocab display ─────────────────────────────────────────────────────

        public void PrintVocab()
        {
            int show = Math.Min(VocabSize, 64);
            Console.WriteLine($"BpeTokenizer vocabulary ({VocabSize} tokens, showing first {show}):");
            for (int i = 0; i < show; i++)
            {
                Console.Write($"  {i,4}: {_idToToken[i],-14}");
                if ((i + 1) % 4 == 0) Console.WriteLine();
            }
            if (show % 4 != 0) Console.WriteLine();
            Console.WriteLine();
        }

        public override string ToString() => $"BpeTokenizer(VocabSize={VocabSize})";

        // ── vocab persistence ─────────────────────────────────────────────────

        /// <summary>
        /// Save format (UTF-8):
        ///   Line 0:   "BpeTokenizer"
        ///   Line 1:   "VOCAB"
        ///   Lines …:  one escaped token per line  (line index - 2 = token ID)
        ///   Then:     "MERGES"
        ///   Lines …:  &lt;escaped_left&gt;\t&lt;escaped_right&gt;  (in rank order, rank 0 first)
        /// </summary>
        public void SaveVocab(string path)
        {
            using var w = new StreamWriter(path, false, Encoding.UTF8);
            w.WriteLine(nameof(BpeTokenizer));
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

        internal static BpeTokenizer LoadFrom(StreamReader reader)
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
            return new BpeTokenizer(tokens.ToArray(), mergeRank);
        }

        private BpeTokenizer(string[] idToToken, Dictionary<(string, string), int> mergeRank)
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
        /// Split a word into its initial BPE character-level representation.
        /// The end-of-word marker is appended to the last character token.
        /// "hello" → ["h", "e", "l", "l", "o&lt;/w&gt;"]
        /// </summary>
        private static List<string> CharSplit(string word)
        {
            var tokens = new List<string>(word.Length);
            for (int i = 0; i < word.Length; i++)
                tokens.Add(i == word.Length - 1
                    ? word[i].ToString() + Eow
                    : word[i].ToString());
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
                {
                    yield return text.Substring(start, i - start);
                    start = -1;
                }
            }
        }

        private static Dictionary<string, int> CountWords(string corpus)
        {
            var freq = new Dictionary<string, int>();
            foreach (string w in SplitWords(corpus))
            {
                freq.TryGetValue(w, out int cur);
                freq[w] = cur + 1;
            }
            return freq;
        }
    }
}
