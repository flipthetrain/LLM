using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// Character-level tokenizer.
    ///
    /// Maps every unique character in the training corpus to a small integer ID.
    /// ID 0 is reserved for the &lt;UNK&gt; token (characters unseen during training).
    /// Vocabulary size = distinct characters + 1.
    ///
    /// Pros:  zero external dependencies; deterministic; smallest possible vocab.
    /// Cons:  long token sequences per sentence; the model must learn spelling from scratch.
    /// </summary>
    public sealed class CharTokenizer : ITokenizer
    {
        private readonly char[]                _idToChar;
        private readonly Dictionary<char, int> _charToId;

        public int UnknownId { get; }
        public int VocabSize  => _idToChar.Length;

        /// <summary>
        /// Build the vocabulary from <paramref name="corpus"/>.
        /// Collects all distinct characters, sorts them for determinism, and
        /// reserves ID 0 for &lt;UNK&gt;.
        /// </summary>
        public CharTokenizer(string corpus)
        {
            if (string.IsNullOrEmpty(corpus))
                throw new ArgumentException("Corpus must not be empty.", nameof(corpus));

            var seen = new SortedSet<char>();
            foreach (char c in corpus) seen.Add(c);

            _idToChar    = new char[seen.Count + 1];
            _charToId    = new Dictionary<char, int>(seen.Count + 1);
            _idToChar[0] = '\0';
            UnknownId    = 0;

            int id = 1;
            foreach (char c in seen)
            {
                _idToChar[id] = c;
                _charToId[c]  = id;
                id++;
            }
        }

        // ── ITokenizer ────────────────────────────────────────────────────────

        public int[] Encode(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            var ids = new int[text.Length];
            for (int i = 0; i < text.Length; i++)
                ids[i] = _charToId.TryGetValue(text[i], out int tid) ? tid : UnknownId;
            return ids;
        }

        public string Decode(int[] ids)
        {
            ArgumentNullException.ThrowIfNull(ids);
            var sb = new StringBuilder(ids.Length);
            foreach (int id in ids)
            {
                if (id >= 0 && id < _idToChar.Length)
                    sb.Append(_idToChar[id] == '\0' ? '?' : _idToChar[id]);
                else
                    sb.Append('?');
            }
            return sb.ToString();
        }

        public string DecodeToken(int id)
        {
            if (id < 0 || id >= _idToChar.Length || _idToChar[id] == '\0') return "?";
            return _idToChar[id].ToString();
        }

        public void PrintVocab()
        {
            Console.WriteLine($"CharTokenizer vocabulary ({VocabSize} tokens):");
            for (int i = 0; i < _idToChar.Length; i++)
            {
                char   c       = _idToChar[i];
                string display = c switch
                {
                    '\0' => "<UNK>",
                    '\n' => "\\n",
                    '\r' => "\\r",
                    '\t' => "\\t",
                    ' '  => "<SP>",
                    _    => c.ToString()
                };
                Console.Write($"  {i,3}: {display}");
                if ((i + 1) % 8 == 0) Console.WriteLine();
            }
            Console.WriteLine();
        }

        public override string ToString() => $"CharTokenizer(VocabSize={VocabSize})";

        // ── vocab persistence ─────────────────────────────────────────────────

        /// <summary>
        /// Save format (UTF-8):
        ///   Line 0: "CharTokenizer"
        ///   Lines 1+: one escaped token per line  (line index = token ID)
        /// </summary>
        public void SaveVocab(string path)
        {
            using var w = new StreamWriter(path, false, Encoding.UTF8);
            w.WriteLine(nameof(CharTokenizer));
            foreach (char c in _idToChar)
                w.WriteLine(TokenizerIO.EscapeToken(c.ToString()));
        }

        internal static CharTokenizer LoadFrom(StreamReader reader)
        {
            var chars = new List<char>();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                string tok = TokenizerIO.UnescapeToken(line);
                chars.Add(tok.Length > 0 ? tok[0] : '\0');
            }
            return new CharTokenizer(chars.ToArray());
        }

        private CharTokenizer(char[] idToChar)
        {
            _idToChar = idToChar;
            _charToId = new Dictionary<char, int>(idToChar.Length);
            UnknownId = 0;
            for (int i = 0; i < idToChar.Length; i++)
                if (idToChar[i] != '\0')
                    _charToId[idToChar[i]] = i;
        }
    }
}
