namespace LLM_App
{
    /// <summary>
    /// Runtime settings loaded from the <c>AppConfig</c> section of <c>appsettings.json</c>.
    /// Every property can be overridden at the command line using the corresponding
    /// kebab-case flag (e.g. <c>--train-file</c> overrides <see cref="TrainFile"/>).
    /// </summary>
    internal sealed class AppConfig
    {
        /// <summary>
        /// Action to perform.  Valid values: <c>Train</c>, <c>Prompt</c>.
        /// </summary>
        public string Action    { get; set; } = "Train";

        /// <summary>
        /// Target vocabulary size for the tokenizer when training from scratch.
        /// Larger vocabularies capture more meaningful subwords but increase model size.
        /// Recommended range: 2 000–8 000 for a corpus of ~5 M characters.
        /// </summary>
        public int    VocabSize { get; set; } = 4000;

        /// <summary>
        /// Corpus split strategy for creating a held-out validation set.
        /// Valid values:
        /// <list type="bullet">
        ///   <item><term>None</term><description>No validation; train on all tokens.</description></item>
        ///   <item><term>Tail</term><description>Hold out the last <see cref="ValidationFraction"/> of the corpus.</description></item>
        ///   <item><term>Random</term><description>Randomly assign <see cref="ValidationFraction"/> of chunks to validation.</description></item>
        /// </list>
        /// </summary>
        public string ValidationSplit    { get; set; } = "Tail";

        /// <summary>
        /// Fraction of tokens to hold out for validation (e.g. <c>0.1</c> = 10%).
        /// Must be in the open interval (0, 1) when <see cref="ValidationSplit"/> is not <c>None</c>.
        /// </summary>
        public double ValidationFraction { get; set; } = 0.1;

        /// <summary>
        /// Training stop condition.  Valid values:
        /// <list type="bullet">
        ///   <item><term>Epochs</term><description>Train for exactly <c>TransformerConfig:Epochs</c> epochs.</description></item>
        ///   <item><term>Patience</term><description>Stop early if validation loss does not improve for <see cref="Patience"/> consecutive epochs.</description></item>
        ///   <item><term>EarlyStopping</term><description>Stop early if validation loss improvement falls below <see cref="MinDeltaLoss"/> for <see cref="Patience"/> consecutive epochs.</description></item>
        /// </list>
        /// Patience and EarlyStopping require a validation split.
        /// </summary>
        public string TrainingMode  { get; set; } = "EarlyStopping";

        /// <summary>
        /// Maximum number of consecutive epochs without sufficient validation-loss improvement
        /// before training stops.  Used by <c>Patience</c> and <c>EarlyStopping</c> modes.
        /// Must be greater than zero when either of those modes is active.
        /// </summary>
        public int    Patience      { get; set; } = 5;

        /// <summary>
        /// Minimum improvement in validation loss that resets the patience counter.
        /// Used by <c>EarlyStopping</c> mode only.  Must be &gt;= 0.
        /// </summary>
        public double MinDeltaLoss  { get; set; } = 0.001;

        /// <summary>
        /// Compute backend.  Valid values: <c>CPU</c>, <c>GPU</c>.
        /// GPU requires a CUDA-capable device and the ILGPU runtime.
        /// </summary>
        public string Backend   { get; set; } = "CPU";

        /// <summary>
        /// Path to the plain-text corpus file used for training.
        /// Required when <see cref="Action"/> is <c>Train</c>.
        /// </summary>
        public string TrainFile { get; set; } = "";

        /// <summary>
        /// Path where model weights are written after training completes.
        /// Required when <see cref="Action"/> is <c>Train</c>.
        /// A vocabulary file is saved alongside it at <c>&lt;SaveFile&gt;.vocab</c>,
        /// and a CSV loss log at <c>&lt;SaveFile&gt;.csv</c>.
        /// </summary>
        public string SaveFile  { get; set; } = "";

        /// <summary>
        /// Path to a weights file or checkpoint file to load before training or inference.
        /// Both file types are accepted; checkpoint Adam state is discarded when loading
        /// for inference.  Empty string means start from randomly initialised weights.
        /// </summary>
        public string LoadFile  { get; set; } = "";

        /// <summary>
        /// Path to redirect stderr (runtime error output).
        /// Empty string leaves stderr on the console.
        /// Configuration validation errors always go to the console regardless of this setting.
        /// </summary>
        public string ErrorFile { get; set; } = "";

        /// <summary>
        /// Save a crash-recovery checkpoint every this many minutes of wall-clock training time.
        /// The checkpoint stores weights, Adam state, and the exact training position so that
        /// training can resume mid-epoch without replaying any completed gradient updates.
        /// Set to <c>0</c> to save only at the end of each epoch.
        /// Must be &gt;= 0.
        /// </summary>
        public double CheckpointEveryMinutes { get; set; } = 60.0;

        /// <summary>
        /// Maximum number of tokens to generate per response in Prompt mode.
        /// Must be greater than zero.
        /// </summary>
        public int    MaxTokens   { get; set; } = 200;

        /// <summary>
        /// Strategy for compacting the context window when it exceeds <c>TransformerConfig:ContextLength</c> tokens.
        /// Valid values:
        /// <list type="bullet">
        ///   <item><term>FIFO</term><description>Drop the oldest tokens first.</description></item>
        ///   <item><term>SlidingWindow</term><description>Preserve the first <see cref="AnchorFraction"/> of the context (e.g. a system prompt) and FIFO the rest.</description></item>
        /// </list>
        /// </summary>
        public string ContextCompaction { get; set; } = "FIFO";

        /// <summary>
        /// Fraction of <c>TransformerConfig:ContextLength</c> to preserve at the start of the
        /// context when <see cref="ContextCompaction"/> is <c>SlidingWindow</c> (e.g. <c>0.2</c> = 20%).
        /// The remainder of the window is managed with FIFO eviction.
        /// Must be in the open interval (0, 1).
        /// </summary>
        public float  AnchorFraction   { get; set; } = 0.2f;

        /// <summary>
        /// Softmax temperature applied during token sampling in Prompt mode.
        /// Values below 1.0 make the distribution sharper (more deterministic);
        /// values above 1.0 make it flatter (more random).
        /// Must be greater than zero.
        /// </summary>
        public float  Temperature { get; set; } = 0.8f;

        /// <summary>
        /// Top-K filter applied before sampling: only the K most probable tokens are
        /// considered at each step.  Set to <c>0</c> to disable (sample from the full vocabulary).
        /// Must be &gt;= 0.
        /// </summary>
        public int    TopK        { get; set; } = 15;
    }
}
