using System;
using System.IO;
using System.Text;

namespace LLM.Tokenizers
{
    /// <summary>
    /// Shared helpers for saving and loading tokenizer vocabulary files.
    ///
    /// File format (UTF-8 text):
    ///   Line 0  : tokenizer type name  (e.g. "UnigramTokenizer")
    ///   Lines 1+: tokenizer-specific   (see each SaveVocab implementation)
    ///
    /// Token text is escaped so that any character (including TAB, LF, backslash)
    /// can be stored safely as a single line.
    /// </summary>
    public static class TokenizerIO
    {
        // ── text escaping ─────────────────────────────────────────────────────

        internal static string EscapeToken(string tok)
        {
            var sb = new StringBuilder(tok.Length + 4);
            foreach (char c in tok)
            {
                switch (c)
                {
                    case '\\': sb.Append("\\\\"); break;
                    case '\t': sb.Append("\\t");  break;
                    case '\n': sb.Append("\\n");  break;
                    case '\r': sb.Append("\\r");  break;
                    case '\0': sb.Append("\\0");  break;
                    default:   sb.Append(c);       break;
                }
            }
            return sb.ToString();
        }

        internal static string UnescapeToken(string tok)
        {
            if (!tok.Contains('\\', StringComparison.Ordinal)) return tok;   // fast path

            var sb = new StringBuilder(tok.Length);
            for (int i = 0; i < tok.Length; i++)
            {
                if (tok[i] == '\\' && i + 1 < tok.Length)
                {
                    i++;
                    switch (tok[i])
                    {
                        case '\\': sb.Append('\\'); break;
                        case 't':  sb.Append('\t'); break;
                        case 'n':  sb.Append('\n'); break;
                        case 'r':  sb.Append('\r'); break;
                        case '0':  sb.Append('\0'); break;
                        default:   sb.Append('\\'); sb.Append(tok[i]); break;
                    }
                }
                else sb.Append(tok[i]);
            }
            return sb.ToString();
        }

        // ── factory ───────────────────────────────────────────────────────────

        /// <summary>
        /// Load a tokenizer from a <c>.vocab</c> file.
        /// The first line of the file identifies the tokenizer type.
        /// </summary>
        public static ITokenizer LoadVocab(string path)
        {
            using var reader = new StreamReader(path, Encoding.UTF8);
            string? typeLine = reader.ReadLine()?.Trim();
            return typeLine switch
            {
                nameof(CharTokenizer)          => CharTokenizer.LoadFrom(reader),
                nameof(BpeTokenizer)           => BpeTokenizer.LoadFrom(reader),
                nameof(WordPieceTokenizer)     => WordPieceTokenizer.LoadFrom(reader),
                nameof(SentencePieceTokenizer) => SentencePieceTokenizer.LoadFrom(reader),
                nameof(UnigramTokenizer)       => UnigramTokenizer.LoadFrom(reader),
                _ => throw new InvalidOperationException(
                         $"Unknown tokenizer type in vocab file: '{typeLine}'")
            };
        }
    }
}
