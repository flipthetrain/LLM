using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// WordPiece tokenizer, trained from scratch on a text corpus (BERT style).
    ///
    /// Key differences from BPE:
    ///   • Continuation sub-words carry a "##" prefix  (e.g. "##llo").
    ///   • Merge criterion: score(a,b) = freq(ab) / (freq(a) × freq(b)),
    ///     favouring pairs that are more likely together than by chance.
    ///   • Special tokens [PAD] [UNK] [CLS] [SEP] [MASK] are prepended to the vocab.
    ///   • Encoding uses longest-match-first per word; pieces after the first get "##".
    ///   • A word whose first character is absent from the vocab encodes as [UNK].
    /// </summary>
    public sealed class WordPieceTokenizer : ITokenizer
    {
        private const string Cont = "##";  // continuation-piece prefix

        public const string PadToken  = "[PAD]";
        public const string UnkToken  = "[UNK]";
        public const string ClsToken  = "[CLS]";
        public const string SepToken  = "[SEP]";
        public const string MaskToken = "[MASK]";

        private readonly string[]               _idToToken;
        private readonly Dictionary<string, int> _tokenToId;

        public int VocabSize => _idToToken.Length;
        public int UnknownId { get; }

        /// <summary>
        /// Train a WordPiece tokenizer on <paramref name="corpus"/>.
        /// </summary>
        /// <param name="corpus">Full training text.</param>
        /// <param name="targetVocabSize">
        /// Target vocabulary size.  Training stops when this is reached or no more
        /// merges are possible.  Typical range: 3 000–32 000.
        /// </param>
        public WordPieceTokenizer(string corpus, int targetVocabSize = 1000)
        {
            if (string.IsNullOrEmpty(corpus))
                throw new ArgumentException("Corpus must not be empty.", nameof(corpus));

            // ── 1. Word frequencies ────────────────────────────────────────────
            Dictionary<string, int> wordFreq = CountWords(corpus);

            // ── 2. Initial vocabulary ──────────────────────────────────────────
            // Special tokens first, then sorted characters.
            // Non-initial characters within a word carry the "##" prefix.
            var vocabList = new List<string> { PadToken, UnkToken, ClsToken, SepToken, MaskToken };
            var tokenToId = new Dictionary<string, int>();
            for (int i = 0; i < vocabList.Count; i++) tokenToId[vocabList[i]] = i;
            UnknownId = tokenToId[UnkToken];

            var charSet = new SortedSet<string>();
            foreach (string word in wordFreq.Keys)
                for (int i = 0; i < word.Length; i++)
                    charSet.Add(i == 0 ? word[i].ToString() : Cont + word[i]);

            foreach (string ch in charSet)
                if (!tokenToId.ContainsKey(ch))
                { tokenToId[ch] = vocabList.Count; vocabList.Add(ch); }

            // ── 3. Initial word representations ───────────────────────────────
            var wordTokens = new Dictionary<string, List<string>>(wordFreq.Count);
            foreach (string word in wordFreq.Keys)
                wordTokens[word] = PieceSplit(word);

            // ── 4. WordPiece merge loop ────────────────────────────────────────
            while (vocabList.Count < targetVocabSize)
            {
                // Count individual token and pair frequencies
                var tokenFreq = new Dictionary<string, long>();
                var pairFreq  = new Dictionary<(string, string), long>();

                foreach (var (word, freq) in wordFreq)
                {
                    List<string> toks = wordTokens[word];
                    foreach (string t in toks)
                    {
                        tokenFreq.TryGetValue(t, out long tf);
                        tokenFreq[t] = tf + freq;
                    }
                    for (int i = 0; i < toks.Count - 1; i++)
                    {
                        var pair = (toks[i], toks[i + 1]);
                        pairFreq.TryGetValue(pair, out long pf);
                        pairFreq[pair] = pf + freq;
                    }
                }
                if (pairFreq.Count == 0) break;

                // Find pair with highest WordPiece score: freq(ab) / (freq(a) * freq(b))
                (string, string) best      = ("", "");
                double            bestScore = double.NegativeInfinity;
                foreach (var (pair, freq) in pairFreq)
                {
                    long fa = tokenFreq.TryGetValue(pair.Item1, out long v1) ? v1 : 1L;
                    long fb = tokenFreq.TryGetValue(pair.Item2, out long v2) ? v2 : 1L;
                    double score = (double)freq / ((double)fa * fb);
                    if (score > bestScore) { bestScore = score; best = pair; }
                }

                // Merged token: strip "##" from the second piece's prefix.
                // The result inherits the prefix status of the first piece.
                string bSuffix = best.Item2.StartsWith(Cont, StringComparison.Ordinal)
                    ? best.Item2.Substring(Cont.Length)
                    : best.Item2;
                string merged = best.Item1 + bSuffix;

                if (!tokenToId.ContainsKey(merged))
                { tokenToId[merged] = vocabList.Count; vocabList.Add(merged); }

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
            var  ids   = new List<int>();
            int  start = 0;
            bool first = true;

            while (start < word.Length)
            {
                int end     = word.Length;
                int foundId = -1;

                // Longest-match scan from the current position
                while (end > start)
                {
                    string piece     = word.Substring(start, end - start);
                    string candidate = first ? piece : Cont + piece;
                    if (_tokenToId.TryGetValue(candidate, out int tid)) { foundId = tid; break; }
                    end--;
                }

                if (foundId < 0)
                    return new[] { UnknownId };   // whole word → [UNK]

                ids.Add(foundId);
                start = end;
                first = false;
            }

            return ids.ToArray();
        }

        // ── decoding ──────────────────────────────────────────────────────────

        public string Decode(int[] ids)
        {
            ArgumentNullException.ThrowIfNull(ids);
            var  sb    = new StringBuilder();
            bool first = true;
            foreach (int id in ids)
            {
                if (id < 0 || id >= _idToToken.Length) continue;
                string tok = _idToToken[id];

                if (tok == PadToken || tok == ClsToken ||
                    tok == SepToken || tok == MaskToken) continue;

                if (tok == UnkToken)
                {
                    if (!first) sb.Append(' ');
                    sb.Append("[UNK]");
                    first = false;
                    continue;
                }

                if (tok.StartsWith(Cont, StringComparison.Ordinal))
                    sb.Append(tok.AsSpan(Cont.Length));
                else
                {
                    if (!first) sb.Append(' ');
                    sb.Append(tok);
                    first = false;
                }
            }
            return sb.ToString();
        }

        /// <summary>
        /// Decode one token to its surface form for streaming generation.
        /// Continuation pieces ("##xxx") return "xxx".
        /// Word-initial pieces return " xxx" (leading space separates words).
        /// Special tokens return an empty string.
        /// </summary>
        public string DecodeToken(int id)
        {
            if (id < 0 || id >= _idToToken.Length) return UnkToken;
            string tok = _idToToken[id];
            if (tok == PadToken || tok == ClsToken ||
                tok == SepToken || tok == MaskToken) return "";
            if (tok == UnkToken) return " [UNK]";
            if (tok.StartsWith(Cont, StringComparison.Ordinal))
                return tok.Substring(Cont.Length);
            return " " + tok;
        }

        // ── vocab display ─────────────────────────────────────────────────────

        public void PrintVocab()
        {
            int show = Math.Min(VocabSize, 64);
            Console.WriteLine($"WordPieceTokenizer vocabulary ({VocabSize} tokens, showing first {show}):");
            for (int i = 0; i < show; i++)
            {
                Console.Write($"  {i,4}: {_idToToken[i],-14}");
                if ((i + 1) % 4 == 0) Console.WriteLine();
            }
            if (show % 4 != 0) Console.WriteLine();
            Console.WriteLine();
        }

        public override string ToString() => $"WordPieceTokenizer(VocabSize={VocabSize})";

        // ── vocab persistence ─────────────────────────────────────────────────

        /// <summary>
        /// Save format (UTF-8):
        ///   Line 0: "WordPieceTokenizer"
        ///   Lines 1+: one escaped token per line  (line index - 1 = token ID)
        /// WordPiece encoding is longest-match-first so no merge rules need saving.
        /// </summary>
        public void SaveVocab(string path)
        {
            using var w = new StreamWriter(path, false, Encoding.UTF8);
            w.WriteLine(nameof(WordPieceTokenizer));
            foreach (string tok in _idToToken)
                w.WriteLine(TokenizerIO.EscapeToken(tok));
        }

        internal static WordPieceTokenizer LoadFrom(StreamReader reader)
        {
            var tokens = new List<string>();
            string? line;
            while ((line = reader.ReadLine()) != null)
                tokens.Add(TokenizerIO.UnescapeToken(line));
            return new WordPieceTokenizer(tokens.ToArray());
        }

        private WordPieceTokenizer(string[] idToToken)
        {
            _idToToken = idToToken;
            _tokenToId = new Dictionary<string, int>(idToToken.Length);
            for (int i = 0; i < idToToken.Length; i++)
                _tokenToId[idToToken[i]] = i;
            UnknownId = _tokenToId.TryGetValue(UnkToken, out int uid) ? uid : 0;
        }

        // ── private helpers ───────────────────────────────────────────────────

        /// <summary>
        /// Initial WordPiece split: first character as-is, remaining characters
        /// with "##" prefix.  "hello" → ["h", "##e", "##l", "##l", "##o"]
        /// </summary>
        private static List<string> PieceSplit(string word)
        {
            var toks = new List<string>(word.Length);
            for (int i = 0; i < word.Length; i++)
                toks.Add(i == 0 ? word[i].ToString() : Cont + word[i]);
            return toks;
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
