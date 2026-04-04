using System;
using System.Threading.Tasks;

namespace LLM_CPU
{
    /// <summary>
    /// Core 2-D matrix class used for every numerical computation in the transformer.
    ///
    /// All weights, activations, and gradients are represented as Matrix objects.
    /// Data is stored in row-major order: element (row, col) lives at Data[row, col].
    ///
    /// Design choices:
    ///   • float precision – fast enough for a demo, matches GPU conventions.
    ///   • No SIMD / BLAS – pure C# so the reader can follow every operation.
    ///   • In-place helpers (AddInPlace, Zero) reduce allocations during training.
    /// </summary>
    public sealed class Matrix
    {
        // ── storage ──────────────────────────────────────────────────────────────

        /// <summary>Raw storage in [rows, cols] layout.</summary>
        public float[,] Data { get; }

        /// <summary>Number of rows (first dimension).</summary>
        public int Rows { get; }

        /// <summary>Number of columns (second dimension).</summary>
        public int Cols { get; }

        // ── constructors ─────────────────────────────────────────────────────────

        /// <summary>Allocate a zero-filled matrix of the given shape.</summary>
        public Matrix(int rows, int cols)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
            Rows = rows;
            Cols = cols;
            Data = new float[rows, cols];
        }

        /// <summary>Wrap an existing 2-D array without copying.</summary>
        public Matrix(float[,] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            Rows = data.GetLength(0);
            Cols = data.GetLength(1);
        }

        // ── element access ───────────────────────────────────────────────────────

        /// <summary>Convenient bracket access: matrix[row, col].</summary>
        public float this[int row, int col]
        {
            get => Data[row, col];
            set => Data[row, col] = value;
        }

        // ── matrix multiplication ─────────────────────────────────────────────────

        /// <summary>
        /// Standard matrix multiply: C = A · B.
        ///
        /// Algorithm: C[i,j] = Σ_k  A[i,k] * B[k,j]
        ///
        /// Dimensions: A is [m×k], B is [k×n], result is [m×n].
        /// This operation is O(m·k·n) – the dominant cost in transformer forward passes.
        /// </summary>
        public static Matrix Dot(Matrix a, Matrix b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            if (a.Cols != b.Rows)
                throw new InvalidOperationException(
                    $"Matrix dimension mismatch for Dot: [{a.Rows}×{a.Cols}] · [{b.Rows}×{b.Cols}]");

            int m = a.Rows, k = a.Cols, n = b.Cols;
            var result = new Matrix(m, n);

            // Each row i of the result depends only on row i of A and all of B,
            // so rows are fully independent and safe to compute in parallel.
            Parallel.For(0, m, i =>
            {
                for (int l = 0; l < k; l++)        // iterate over shared dimension first
                {                                   // (cache-friendly inner loop over j)
                    float aIL = a.Data[i, l];
                    for (int j = 0; j < n; j++)
                        result.Data[i, j] += aIL * b.Data[l, j];
                }
            });
            return result;
        }

        // ── transpose ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Return the transpose: result[j, i] = this[i, j].
        /// Shape changes from [R×C] to [C×R].
        /// Used extensively in backpropagation (e.g., dW = xᵀ · dOut).
        /// </summary>
        public Matrix Transpose()
        {
            var result = new Matrix(Cols, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[j, i] = Data[i, j];
            return result;
        }

        // ── element-wise arithmetic ───────────────────────────────────────────────

        /// <summary>Element-wise addition: C[i,j] = A[i,j] + B[i,j].</summary>
        public static Matrix Add(Matrix a, Matrix b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            AssertSameShape(a, b, "Add");
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] + b.Data[i, j];
            return result;
        }

        /// <summary>Element-wise subtraction: C[i,j] = A[i,j] − B[i,j].</summary>
        public static Matrix Sub(Matrix a, Matrix b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            AssertSameShape(a, b, "Sub");
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
            return result;
        }

        /// <summary>
        /// Element-wise (Hadamard) product: C[i,j] = A[i,j] · B[i,j].
        /// Used when multiplying gradient signals by cached activations.
        /// </summary>
        public static Matrix Mul(Matrix a, Matrix b)
        {
            ArgumentNullException.ThrowIfNull(a);
            ArgumentNullException.ThrowIfNull(b);
            AssertSameShape(a, b, "Mul");
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] * b.Data[i, j];
            return result;
        }

        /// <summary>Scalar multiplication: every element scaled by <paramref name="s"/>.</summary>
        public Matrix Scale(float s)
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = Data[i, j] * s;
            return result;
        }

        // ── bias addition ─────────────────────────────────────────────────────────

        /// <summary>
        /// Broadcast-add a 1-D bias vector to every row of this matrix.
        ///
        /// bias.Length must equal Cols.
        /// result[i, j] = this[i, j] + bias[j].
        ///
        /// Equivalent to the bias term in a linear layer: y = xW + b.
        /// </summary>
        public Matrix AddBias(float[] bias)
        {
            ArgumentNullException.ThrowIfNull(bias);
            if (bias.Length != Cols)
                throw new ArgumentException(
                    $"Bias length {bias.Length} does not match matrix columns {Cols}");

            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = Data[i, j] + bias[j];
            return result;
        }

        // ── activation functions ──────────────────────────────────────────────────

        /// <summary>
        /// Apply softmax independently to each row.
        ///
        /// Formula:   softmax(x)_i = exp(x_i − max(x)) / Σ_j exp(x_j − max(x))
        ///
        /// Subtracting the row maximum before exponentiation is the standard
        /// numerical-stability trick: it prevents overflow (e.g. exp(1000)) while
        /// leaving the final ratio unchanged.
        ///
        /// Used in the attention mechanism to turn raw scores into a probability
        /// distribution over keys.
        /// </summary>
        public Matrix Softmax()
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
            {
                // Step 1: find the maximum value in this row for stability
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < Cols; j++)
                    if (Data[i, j] > maxVal) maxVal = Data[i, j];

                // Step 2: compute exp(x − max) and accumulate the sum
                float sum = 0f;
                for (int j = 0; j < Cols; j++)
                {
                    float e = MathF.Exp(Data[i, j] - maxVal);
                    result.Data[i, j] = e;
                    sum += e;
                }

                // Step 3: normalise so each row sums to 1
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] /= sum;
            }
            return result;
        }

        /// <summary>
        /// Backward pass for row-wise softmax.
        ///
        /// Given the softmax output S and the upstream gradient dOut, we need dIn:
        ///
        ///   dIn[i,j] = S[i,j] · ( dOut[i,j]  −  Σ_k S[i,k]·dOut[i,k] )
        ///
        /// Derivation sketch:
        ///   dy_j/dx_i = y_j(δ_ij − y_i)   (Jacobian of softmax)
        ///   dL/dx_i   = Σ_j dL/dy_j · dy_j/dx_i
        ///             = Σ_j dL/dy_j · y_j(δ_ij − y_i)
        ///             = dL/dy_i · y_i  − y_i · Σ_j dL/dy_j · y_j
        ///             = y_i · ( dL/dy_i − dot(dL/dy, y) )
        /// </summary>
        public static Matrix SoftmaxBackward(Matrix softmaxOut, Matrix dOut)
        {
            ArgumentNullException.ThrowIfNull(softmaxOut);
            ArgumentNullException.ThrowIfNull(dOut);
            AssertSameShape(softmaxOut, dOut, "SoftmaxBackward");
            var dIn = new Matrix(softmaxOut.Rows, softmaxOut.Cols);
            for (int i = 0; i < softmaxOut.Rows; i++)
            {
                // dot product:  Σ_k S[i,k] · dOut[i,k]
                float dot = 0f;
                for (int k = 0; k < softmaxOut.Cols; k++)
                    dot += softmaxOut.Data[i, k] * dOut.Data[i, k];

                for (int j = 0; j < softmaxOut.Cols; j++)
                    dIn.Data[i, j] = softmaxOut.Data[i, j] * (dOut.Data[i, j] - dot);
            }
            return dIn;
        }

        /// <summary>
        /// GELU (Gaussian Error Linear Unit) activation – element-wise.
        ///
        /// Exact definition:  GELU(x) = x · Φ(x)
        ///   where Φ is the CDF of N(0,1).
        ///
        /// Tanh approximation (used here, same as GPT-2):
        ///   GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
        ///
        /// GELU is preferred over ReLU in transformers because:
        ///   • It is smooth (differentiable everywhere).
        ///   • Its stochastic interpretation regularises training.
        ///   • Empirically better perplexity on language tasks.
        /// </summary>
        public Matrix GELU()
        {
            // Pre-computed constant: √(2/π)
            const float Sqrt2OverPi = 0.7978845608028654f;
            // Coefficient from the tanh approximation
            const float Coeff = 0.044715f;

            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                {
                    float x = Data[i, j];
                    // inner argument to tanh
                    float inner = Sqrt2OverPi * (x + Coeff * x * x * x);
                    result.Data[i, j] = 0.5f * x * (1f + MathF.Tanh(inner));
                }
            return result;
        }

        /// <summary>
        /// Element-wise derivative of GELU for backpropagation.
        ///
        /// d/dx GELU(x)
        ///   = 0.5·(1 + tanh(k)) + 0.5·x·sech²(k)·dk/dx
        ///
        /// where k = √(2/π)·(x + 0.044715·x³)
        ///       dk/dx = √(2/π)·(1 + 3·0.044715·x²)
        ///
        /// In the backward pass we multiply this element-wise by the upstream gradient.
        /// </summary>
        public Matrix GELUGrad()
        {
            const float Sqrt2OverPi = 0.7978845608028654f;
            const float Coeff = 0.044715f;

            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                {
                    float x = Data[i, j];
                    float inner = Sqrt2OverPi * (x + Coeff * x * x * x);
                    float tanhInner = MathF.Tanh(inner);
                    float sech2 = 1f - tanhInner * tanhInner;       // sech²(k) = 1 − tanh²(k)
                    float dInnerDx = Sqrt2OverPi * (1f + 3f * Coeff * x * x);
                    result.Data[i, j] = 0.5f * (1f + tanhInner) + 0.5f * x * sech2 * dInnerDx;
                }
            return result;
        }

        // ── reduction operations ──────────────────────────────────────────────────

        /// <summary>Sum all elements and return a scalar.</summary>
        public float Sum()
        {
            float sum = 0f;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    sum += Data[i, j];
            return sum;
        }

        /// <summary>
        /// Sum each column across all rows, returning an array of length Cols.
        ///
        /// result[j] = Σ_i this[i, j]
        ///
        /// Used to compute the gradient of a bias vector:
        ///   dL/db[j] = Σ_i dL/dy[i, j]   (sum over the batch / sequence dimension).
        /// </summary>
        public float[] SumOverRows()
        {
            var result = new float[Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[j] += Data[i, j];
            return result;
        }

        // ── slicing and concatenation ─────────────────────────────────────────────

        /// <summary>
        /// Extract columns [startCol, endCol) into a new matrix.
        ///
        /// Shape: [Rows × (endCol − startCol)].
        /// Used to split the combined QKV or head-concatenated output into
        /// per-head slices.
        /// </summary>
        public Matrix SliceCols(int startCol, int endCol)
        {
            int numCols = endCol - startCol;
            var result = new Matrix(Rows, numCols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < numCols; j++)
                    result.Data[i, j] = Data[i, startCol + j];
            return result;
        }

        /// <summary>Extract a contiguous row slice [rowStart, rowEnd).</summary>
        public Matrix SliceRows(int rowStart, int rowEnd)
        {
            int numRows = rowEnd - rowStart;
            var result = new Matrix(numRows, Cols);
            for (int i = 0; i < numRows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = Data[rowStart + i, j];
            return result;
        }

        /// <summary>
        /// Write <paramref name="src"/> into the column range [startCol, startCol+src.Cols)
        /// of this matrix in-place.
        ///
        /// The inverse of SliceCols; used during gradient accumulation to scatter
        /// per-head gradients back into the combined [T × d_model] gradient matrix.
        /// </summary>
        public void AddSliceCols(Matrix src, int startCol)
        {
            ArgumentNullException.ThrowIfNull(src);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < src.Cols; j++)
                    Data[i, startCol + j] += src.Data[i, j];
        }

        /// <summary>
        /// Horizontally concatenate an array of matrices along the column axis.
        ///
        /// All matrices must have the same number of rows.
        /// result = [M0 | M1 | M2 | …]
        ///
        /// Used to combine the outputs from all attention heads:
        ///   concat([head_0, head_1, …, head_h]) → [T × d_model]
        /// </summary>
        public static Matrix ConcatCols(Matrix[] matrices)
        {
            ArgumentNullException.ThrowIfNull(matrices);
            int rows = matrices[0].Rows;
            int totalCols = 0;
            foreach (var m in matrices)
            {
                if (m.Rows != rows)
                    throw new ArgumentException("All matrices must have the same row count for ConcatCols");
                totalCols += m.Cols;
            }

            var result = new Matrix(rows, totalCols);
            int offset = 0;
            foreach (var m in matrices)
            {
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < m.Cols; j++)
                        result.Data[i, offset + j] = m.Data[i, j];
                offset += m.Cols;
            }
            return result;
        }

        // ── in-place operations ───────────────────────────────────────────────────

        /// <summary>
        /// Add <paramref name="other"/> element-wise into this matrix in-place.
        ///
        /// Used for gradient accumulation across multiple backward paths
        /// (e.g. the residual connection in a transformer block adds gradients
        /// from the main path and the skip path together).
        /// </summary>
        public void AddInPlace(Matrix other)
        {
            ArgumentNullException.ThrowIfNull(other);
            AssertSameShape(this, other, "AddInPlace");
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] += other.Data[i, j];
        }

        /// <summary>Set every element to zero (fast path for gradient reset).</summary>
        public void Zero()
        {
            // Buffer.BlockCopy is the fastest way to clear a managed float array.
            Buffer.BlockCopy(Data, 0, Data, 0, 0); // dummy call to suppress analysis
            Array.Clear(Data, 0, Data.Length);
        }

        /// <summary>Return a deep copy of this matrix.</summary>
        public Matrix Clone()
        {
            var result = new Matrix(Rows, Cols);
            Buffer.BlockCopy(Data, 0, result.Data, 0, Data.Length * sizeof(float));
            return result;
        }

        // ── initialisation ────────────────────────────────────────────────────────

        /// <summary>
        /// Xavier / Glorot uniform initialisation.
        ///
        /// Samples uniformly from [-limit, +limit] where:
        ///   limit = √(6 / (fanIn + fanOut))
        ///
        /// Motivation: keeps the variance of activations and gradients roughly
        /// constant across layers so that neither explodes nor vanishes during
        /// the first forward/backward pass.
        ///
        /// Reference: Glorot &amp; Bengio (2010), "Understanding the difficulty of
        /// training deep feedforward neural networks."
        /// </summary>
        public void XavierInit(Random rng, int fanIn, int fanOut)
        {
            ArgumentNullException.ThrowIfNull(rng);
            float limit = MathF.Sqrt(6f / (fanIn + fanOut));
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = (float)(rng.NextDouble() * 2.0 - 1.0) * limit;
        }

        /// <summary>
        /// Normal (Gaussian) initialisation using the Box-Muller transform.
        ///
        /// Box-Muller converts two independent uniform samples u1, u2 ∈ (0,1]
        /// into a standard-normal sample z:
        ///   z = √(−2·ln u1) · cos(2π·u2)
        ///
        /// Then the sample is scaled: x = mean + z·std.
        ///
        /// GPT-2 uses std = 0.02 for most weight matrices.
        /// </summary>
        public void NormalInit(Random rng, float mean = 0f, float std = 0.02f)
        {
            ArgumentNullException.ThrowIfNull(rng);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                {
                    // Box-Muller: generate two uniform samples (avoid exact 0 for log)
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    Data[i, j] = mean + (float)z * std;
                }
        }

        /// <summary>Set every element to <paramref name="value"/>.</summary>
        public void Fill(float value)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = value;
        }

        // ── misc utilities ────────────────────────────────────────────────────────

        /// <summary>
        /// Clip every element into [min, max].
        ///
        /// Applied during gradient clipping to prevent exploding gradients:
        /// large gradient values can destabilise training, so we hard-cap them.
        /// </summary>
        public Matrix Clip(float min, float max)
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result.Data[i, j] = Math.Clamp(Data[i, j], min, max);
            return result;
        }

        /// <summary>
        /// Return the index of the largest value in row <paramref name="row"/>.
        /// Used during greedy decoding (argmax over the vocabulary logits).
        /// </summary>
        public int ArgMaxRow(int row)
        {
            int best = 0;
            float bestVal = Data[row, 0];
            for (int j = 1; j < Cols; j++)
                if (Data[row, j] > bestVal) { bestVal = Data[row, j]; best = j; }
            return best;
        }

        /// <summary>
        /// Copy one row of this matrix into a newly allocated float array.
        /// Useful for extracting the logit vector at a specific sequence position.
        /// </summary>
        public float[] GetRow(int row)
        {
            var arr = new float[Cols];
            for (int j = 0; j < Cols; j++) arr[j] = Data[row, j];
            return arr;
        }

        /// <summary>Human-readable shape string for debugging.</summary>
        public override string ToString() => $"Matrix[{Rows}×{Cols}]";

        // ── private helpers ───────────────────────────────────────────────────────

        private static void AssertSameShape(Matrix a, Matrix b, string op)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new InvalidOperationException(
                    $"{op}: shape mismatch [{a.Rows}×{a.Cols}] vs [{b.Rows}×{b.Cols}]");
        }
    }
}
