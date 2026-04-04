using System;
using ILGPU;
using ILGPU.Runtime;

using A1 = ILGPU.Runtime.ArrayView1D<float, ILGPU.Stride1D.Dense>;

namespace LLM_GPU
{
    /// <summary>
    /// GPU-resident 2-D float matrix backed by a flat 1-D MemoryBuffer.
    /// Row-major layout: element [row, col] lives at buffer[row * Cols + col].
    ///
    /// Operations mirror LLM.Matrix but execute on the GPU via static kernels.
    /// Kernels are loaded lazily (??= pattern) and cached in static fields;
    /// ILGPU caches the compiled PTX/SPIRV internally.
    /// </summary>
    internal sealed class GpuMatrix : IDisposable
    {
        // ── storage ───────────────────────────────────────────────────────────
        public readonly MemoryBuffer1D<float, Stride1D.Dense> Buffer;
        public readonly int Rows;
        public readonly int Cols;

        /// <summary>1-D view over the GPU buffer, used in kernel calls.</summary>
        public A1 View => Buffer.View;

        private int N => Rows * Cols;

        // ── cached kernel delegates ───────────────────────────────────────────
        private static Action<Index2D, A1, A1, A1, int, int, int>?  _dotKernel;
        private static Action<Index1D, A1, A1, int, int>?            _transposeKernel;
        private static Action<Index1D, A1, A1, A1, int>?             _addKernel;
        private static Action<Index1D, A1, A1, A1, int>?             _subKernel;
        private static Action<Index1D, A1, A1, A1, int>?             _mulKernel;
        private static Action<Index1D, A1, A1, float, int>?          _scaleKernel;
        private static Action<Index2D, A1, A1, A1, int, int>?        _addBiasKernel;
        private static Action<Index1D, A1, A1, int>?                 _addInPlaceKernel;
        private static Action<Index1D, A1, int>?                     _zeroKernel;
        private static Action<Index1D, A1, float, int>?              _fillKernel;
        private static Action<Index1D, A1, A1, int>?                 _copyKernel;
        private static Action<Index1D, A1, int, int>?                _softmaxKernel;
        private static Action<Index1D, A1, A1, A1, int, int>?        _softmaxBackwardKernel;
        private static Action<Index1D, A1, A1, int>?                 _geluKernel;
        private static Action<Index1D, A1, A1, int>?                 _geluGradKernel;
        private static Action<Index1D, A1, A1, int, int>?            _sumOverRowsKernel;
        private static Action<Index2D, A1, A1, int, int, int, int>?  _sliceColsKernel;
        private static Action<Index2D, A1, A1, int, int, int, int>?  _scatterAddColsKernel;

        // ── constructors ─────────────────────────────────────────────────────
        public GpuMatrix(int rows, int cols)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
            Rows   = rows;
            Cols   = cols;
            Buffer = GpuContext.Accelerator.Allocate1D<float>(rows * cols);
        }

        // ── CPU ↔ GPU transfers ───────────────────────────────────────────────
        /// <summary>Upload a [rows, cols] CPU array to a new GpuMatrix.</summary>
        public static GpuMatrix FromCpu(float[,] data)
        {
            int rows = data.GetLength(0);
            int cols = data.GetLength(1);
            var m    = new GpuMatrix(rows, cols);
            var flat = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    flat[i * cols + j] = data[i, j];
            m.Buffer.CopyFromCPU(flat);
            return m;
        }

        /// <summary>Download to a CPU [rows, cols] array. Syncs the GPU first.</summary>
        public float[,] ToCpu()
        {
            GpuContext.Sync();
            var flat   = Buffer.GetAsArray1D();
            var result = new float[Rows, Cols];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[i, j] = flat[i * Cols + j];
            return result;
        }

        /// <summary>Download one row as a float[]. Syncs the GPU first.</summary>
        public float[] GetRow(int row)
        {
            GpuContext.Sync();
            var flat   = Buffer.GetAsArray1D();
            var result = new float[Cols];
            for (int j = 0; j < Cols; j++)
                result[j] = flat[row * Cols + j];
            return result;
        }

        // ── matrix multiply ───────────────────────────────────────────────────
        public static GpuMatrix Dot(GpuMatrix a, GpuMatrix b)
        {
            if (a.Cols != b.Rows)
                throw new InvalidOperationException(
                    $"Dot: [{a.Rows}×{a.Cols}] · [{b.Rows}×{b.Cols}]");
            int m = a.Rows, k = a.Cols, n = b.Cols;
            var c = new GpuMatrix(m, n);
            _dotKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index2D, A1, A1, A1, int, int, int>(
                    Kernels.DotKernel);
            _dotKernel(new Index2D(m, n), a.View, b.View, c.View, m, k, n);
            return c;
        }

        // ── transpose ─────────────────────────────────────────────────────────
        public GpuMatrix Transpose()
        {
            var c = new GpuMatrix(Cols, Rows);
            _transposeKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int, int>(
                    Kernels.TransposeKernel);
            _transposeKernel(N, View, c.View, Rows, Cols);
            return c;
        }

        // ── element-wise operations ───────────────────────────────────────────
        public static GpuMatrix Add(GpuMatrix a, GpuMatrix b)
        {
            var c = new GpuMatrix(a.Rows, a.Cols);
            _addKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, A1, int>(Kernels.AddKernel);
            _addKernel(a.N, a.View, b.View, c.View, a.N);
            return c;
        }

        public static GpuMatrix Sub(GpuMatrix a, GpuMatrix b)
        {
            var c = new GpuMatrix(a.Rows, a.Cols);
            _subKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, A1, int>(Kernels.SubKernel);
            _subKernel(a.N, a.View, b.View, c.View, a.N);
            return c;
        }

        public static GpuMatrix Mul(GpuMatrix a, GpuMatrix b)
        {
            var c = new GpuMatrix(a.Rows, a.Cols);
            _mulKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, A1, int>(Kernels.MulKernel);
            _mulKernel(a.N, a.View, b.View, c.View, a.N);
            return c;
        }

        public GpuMatrix Scale(float s)
        {
            var c = new GpuMatrix(Rows, Cols);
            _scaleKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, float, int>(
                    Kernels.ScaleKernel);
            _scaleKernel(N, View, c.View, s, N);
            return c;
        }

        /// <summary>
        /// Broadcast-add the bias (shape [1, Cols]) to every row.
        /// bias.Cols must equal this.Cols.
        /// </summary>
        public GpuMatrix AddBias(GpuMatrix bias)
        {
            var c = new GpuMatrix(Rows, Cols);
            _addBiasKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index2D, A1, A1, A1, int, int>(
                    Kernels.AddBiasKernel);
            _addBiasKernel(new Index2D(Rows, Cols), View, bias.View, c.View, Rows, Cols);
            return c;
        }

        public void AddInPlace(GpuMatrix other)
        {
            _addInPlaceKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int>(
                    Kernels.AddInPlaceKernel);
            _addInPlaceKernel(N, View, other.View, N);
        }

        // ── flat CPU ↔ GPU helpers ────────────────────────────────────────────
        /// <summary>Upload a flat float[] (length = Rows*Cols) to the GPU buffer.</summary>
        public void UploadFlat(float[] data) => Buffer.CopyFromCPU(data);

        /// <summary>Download the GPU buffer to a flat float[]. Syncs the GPU first.</summary>
        public float[] DownloadFlat()
        {
            GpuContext.Sync();
            return Buffer.GetAsArray1D();
        }

        // ── buffer utilities ──────────────────────────────────────────────────
        public void Zero()
        {
            _zeroKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, int>(Kernels.ZeroKernel);
            _zeroKernel(N, View, N);
        }

        public void Fill(float val)
        {
            _fillKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, float, int>(Kernels.FillKernel);
            _fillKernel(N, View, val, N);
        }

        public GpuMatrix Clone()
        {
            var c = new GpuMatrix(Rows, Cols);
            _copyKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int>(Kernels.CopyKernel);
            _copyKernel(N, View, c.View, N);
            return c;
        }

        // ── activation functions ──────────────────────────────────────────────
        public GpuMatrix Softmax()
        {
            var c = Clone();   // softmax kernel works in-place on the copy
            _softmaxKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, int, int>(
                    Kernels.SoftmaxKernel);
            _softmaxKernel(Rows, c.View, Rows, Cols);
            return c;
        }

        public static GpuMatrix SoftmaxBackward(GpuMatrix softmaxOut, GpuMatrix dOut)
        {
            var dIn = new GpuMatrix(softmaxOut.Rows, softmaxOut.Cols);
            _softmaxBackwardKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, A1, int, int>(
                    Kernels.SoftmaxBackwardKernel);
            _softmaxBackwardKernel(
                softmaxOut.Rows,
                softmaxOut.View, dOut.View, dIn.View,
                softmaxOut.Rows, softmaxOut.Cols);
            return dIn;
        }

        public GpuMatrix GELU()
        {
            var c = new GpuMatrix(Rows, Cols);
            _geluKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int>(Kernels.GELUKernel);
            _geluKernel(N, View, c.View, N);
            return c;
        }

        public GpuMatrix GELUGrad()
        {
            var c = new GpuMatrix(Rows, Cols);
            _geluGradKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int>(
                    Kernels.GELUGradKernel);
            _geluGradKernel(N, View, c.View, N);
            return c;
        }

        // ── reductions ────────────────────────────────────────────────────────
        /// <summary>
        /// Sum each column over all rows. Returns a [1, Cols] GpuMatrix
        /// that lives on the GPU (no CPU transfer).
        /// </summary>
        public GpuMatrix SumOverRowsGpu()
        {
            var result = new GpuMatrix(1, Cols);
            result.Zero();
            _sumOverRowsKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index1D, A1, A1, int, int>(
                    Kernels.SumOverRowsKernel);
            _sumOverRowsKernel(Cols, View, result.View, Rows, Cols);
            return result;
        }

        /// <summary>
        /// Sum each column over all rows. Returns a CPU float[] (syncs GPU).
        /// </summary>
        public float[] SumOverRows()
        {
            using var gpuResult = SumOverRowsGpu();
            GpuContext.Sync();
            return gpuResult.Buffer.GetAsArray1D();
        }

        // ── column slicing ────────────────────────────────────────────────────
        public GpuMatrix SliceCols(int startCol, int endCol)
        {
            int numCols = endCol - startCol;
            var c = new GpuMatrix(Rows, numCols);
            _sliceColsKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index2D, A1, A1, int, int, int, int>(
                    Kernels.SliceColsKernel);
            _sliceColsKernel(
                new Index2D(Rows, numCols),
                View, c.View,
                Rows, Cols, numCols, startCol);
            return c;
        }

        /// <summary>
        /// Scatter-add src into dst columns [startCol, startCol + src.Cols).
        /// Used to write per-head gradients back into the full [T × D] matrix.
        /// </summary>
        public void AddSliceCols(GpuMatrix src, int startCol)
        {
            _scatterAddColsKernel ??= GpuContext.Accelerator
                .LoadAutoGroupedStreamKernel<Index2D, A1, A1, int, int, int, int>(
                    Kernels.ScatterAddColsKernel);
            _scatterAddColsKernel(
                new Index2D(src.Rows, src.Cols),
                src.View, View,
                src.Rows, src.Cols, Cols, startCol);
        }

        /// <summary>Horizontal concatenation of matrices along the column axis.</summary>
        public static GpuMatrix ConcatCols(GpuMatrix[] matrices)
        {
            int rows      = matrices[0].Rows;
            int totalCols = 0;
            foreach (var m in matrices) totalCols += m.Cols;

            var result = new GpuMatrix(rows, totalCols);
            result.Zero();
            int offset = 0;
            foreach (var m in matrices)
            {
                result.AddSliceCols(m, offset);
                offset += m.Cols;
            }
            return result;
        }

        // ── misc ──────────────────────────────────────────────────────────────
        /// <summary>
        /// Return the argmax column index in the given row. Syncs and downloads.
        /// </summary>
        public int ArgMaxRow(int row)
        {
            GpuContext.Sync();
            var flat    = Buffer.GetAsArray1D();
            int best    = 0;
            float bestV = flat[row * Cols];
            for (int j = 1; j < Cols; j++)
            {
                float v = flat[row * Cols + j];
                if (v > bestV) { bestV = v; best = j; }
            }
            return best;
        }

        public override string ToString() => $"GpuMatrix[{Rows}×{Cols}]";

        public void Dispose() => Buffer.Dispose();
    }
}
