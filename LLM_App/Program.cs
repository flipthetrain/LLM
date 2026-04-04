using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using LLM;
using LLM.Tokenizers;
using LLM_GPU;
using Microsoft.Extensions.Configuration;

namespace LLM_App
{
    /// <summary>
    /// Entry point for the GPT-style transformer.
    /// All settings are read from appsettings.json (AppConfig + TransformerConfig sections).
    /// </summary>
    internal static class Program
    {
        /// <summary>
        /// Application entry point. Loads configuration from <c>appsettings.json</c>,
        /// applies any CLI overrides, validates the result, then dispatches to either
        /// the training loop or the interactive prompt loop.
        /// </summary>
        /// <param name="args">
        /// Command-line arguments in <c>--key value</c> form.
        /// Pass <c>--help</c> or <c>-h</c> to print all available options and exit.
        /// </param>
        static void Main(string[] args)
        {
            if (args.Length > 0 && (args[0] == "--help" || args[0] == "-h"))
            {
                PrintHelp();
                return;
            }

            Console.Clear();

            // ── Load appsettings.json then overlay CLI args ───────────────────────
            IConfiguration configuration;
            try
            {
                configuration = new ConfigurationBuilder()
                    .SetBasePath(AppContext.BaseDirectory)
                    .AddJsonFile("appsettings.json", optional: false)
                    .AddCommandLine(args, BuildSwitchMappings())
                    .Build();
            }
            catch (FormatException ex)
            {
                Console.WriteLine($"Unknown argument: {ex.Message}");
                Console.WriteLine("Run with --help to see all available options.");
                return;
            }

            var appCfg = new AppConfig();
            configuration.GetSection("AppConfig").Bind(appCfg);

            bool    useGpu     = appCfg.Backend.Equals("GPU",  StringComparison.OrdinalIgnoreCase);
            bool    isTrain    = appCfg.Action.Equals("Train",  StringComparison.OrdinalIgnoreCase);
            bool    isPrompt   = appCfg.Action.Equals("Prompt", StringComparison.OrdinalIgnoreCase);
            string? corpusPath = string.IsNullOrEmpty(appCfg.TrainFile) ? null : appCfg.TrainFile;
            string? savePath   = string.IsNullOrEmpty(appCfg.SaveFile)  ? null : appCfg.SaveFile;
            string? loadPath   = string.IsNullOrEmpty(appCfg.LoadFile)  ? null : appCfg.LoadFile;
            string? errorFile  = string.IsNullOrEmpty(appCfg.ErrorFile) ? null : appCfg.ErrorFile;

            // ── Validate config — always writes to Console.Out so errors are visible
            //    even when ErrorFile redirects stderr ────────────────────────────────
            var configErrors = ValidateConfig(appCfg);
            if (configErrors.Count > 0)
            {
                Console.WriteLine("appsettings.json has the following errors — please correct them and restart:");
                Console.WriteLine();
                foreach (string err in configErrors)
                    Console.WriteLine($"  • {err}");
                return;
            }

            // ── Redirect stderr if an error file is configured ────────────────────
            using StreamWriter? errorWriter = errorFile != null
                ? new StreamWriter(errorFile, append: false) { AutoFlush = true }
                : null;
            if (errorWriter != null)
                Console.SetError(errorWriter);

            // ── 1. Load corpus (only needed when training or when no saved vocab) ──
            string? vocabPath = loadPath != null ? loadPath + ".vocab" : null;
            bool    hasVocab  = vocabPath != null && File.Exists(vocabPath);

            string corpus     = "";
            string corpusFile = corpusPath ?? "sample_corpus.txt";
            if (corpusPath != null || !hasVocab)
            {
                if (!File.Exists(corpusFile))
                {
                    string hint = hasVocab ? "" : $"  (or provide --load <file> with a saved .vocab)";
                    Console.WriteLine($"Error: corpus file '{corpusFile}' not found.{hint}");
                    return;
                }
                corpus = File.ReadAllText(corpusFile);
                Console.WriteLine($"Corpus: '{corpusFile}'  ({corpus.Length:N0} characters)");
            }

            // ── 2. Tokeniser ──────────────────────────────────────────────────
            ITokenizer tokenizer;
            if (hasVocab)
            {
                tokenizer = TokenizerIO.LoadVocab(vocabPath!);
                Console.WriteLine($"Tokenizer: loaded from '{vocabPath}'  ({tokenizer.VocabSize} tokens)");
            }
            else
            {
                //ITokenizer tokenizer = new CharTokenizer(corpus);
                //ITokenizer tokenizer = new BpeTokenizer(corpus, numMerges: 500);
                //ITokenizer tokenizer = new WordPieceTokenizer(corpus, targetVocabSize: 500);
                //ITokenizer tokenizer = new SentencePieceTokenizer(corpus, numMerges: 500);
                tokenizer = new UnigramTokenizer(corpus, targetVocabSize: appCfg.VocabSize);
                Console.WriteLine($"Tokenizer: trained  ({tokenizer.VocabSize} tokens)");
            }

            int[] allTokens = corpusPath != null ? tokenizer.Encode(corpus) : [];
            if (corpusPath != null)
                Console.WriteLine($"Total tokens: {allTokens.Length:N0}");
            Console.WriteLine();

            // ── 3. Model configuration ────────────────────────────────────────
            var cfg = LoadConfig(configuration);
            // If loading saved weights, overwrite architecture settings from the file
            // so the model is built to match — training-only settings stay from appsettings.json.
            if (loadPath != null)
            {
                ModelSerializer.ReadConfig(loadPath, cfg);
                Console.WriteLine($"Architecture loaded from '{loadPath}'.");
            }
            else
            {
                cfg.VocabSize = tokenizer.VocabSize;   // set by the tokeniser when training fresh
            }

            var rng = cfg.Seed >= 0 ? new Random(cfg.Seed) : new Random();

            // ── 4. Strategy selection ─────────────────────────────────────────
            using ITransformerModel model = useGpu
                ? new GpuTransformerModel(cfg, rng)
                : new TransformerModel(cfg, rng);

            PrintConfigSummary(cfg, model, useGpu, isTrain, appCfg);
            Console.WriteLine();

            // ── Validation split (after cfg so ContextLength is known) ────────
            int[] trainTokens = allTokens;
            int[] valTokens = [];

            if (corpusPath != null && !appCfg.ValidationSplit.Equals("None", StringComparison.OrdinalIgnoreCase))
            {
                ICorpusSplitter splitter = appCfg.ValidationSplit.Equals("Random", StringComparison.OrdinalIgnoreCase)
                    ? new RandomSplitter(cfg.ContextLength, appCfg.ValidationFraction)
                    : new TailSplitter(appCfg.ValidationFraction);

                (trainTokens, valTokens) = splitter.Split(allTokens);
                Console.WriteLine($"Validation split : {appCfg.ValidationSplit}  ({appCfg.ValidationFraction * 100:F0}%)  |  " +
                                  $"train={trainTokens.Length:N0}  val={valTokens.Length:N0} tokens");
                Console.WriteLine();
            }

            // ── 5. Load weights or resume from checkpoint ─────────────────────
            string? checkpointPath = savePath != null ? savePath + ".checkpoint" : null;
            int resumeEpoch     = 0;
            int resumeStep      = 0;
            int resumeInnerStep = 0;   // -1 = v1 checkpoint (no inner step stored)

            if (isTrain && checkpointPath != null && File.Exists(checkpointPath))
            {
                Console.WriteLine($"Checkpoint found: '{checkpointPath}' — resuming training.");
                int rawInner;
                (resumeEpoch, resumeStep, rawInner) = model.LoadCheckpoint(checkpointPath);
                resumeInnerStep = rawInner;  // -1 signals v1 format
                string innerDesc = rawInner >= 0 ? $", inner step {rawInner}" : "";
                Console.WriteLine($"Resuming from epoch {resumeEpoch + 1}, Adam step {resumeStep}{innerDesc}.");
                Console.WriteLine();
            }
            else if (loadPath != null)
            {
                Console.WriteLine($"Loading weights from '{loadPath}'...");
                model.Load(loadPath);
                Console.WriteLine("Weights loaded.");
                Console.WriteLine();
            }

            // ── 6. Train ──────────────────────────────────────────────────────
            if (isTrain)
            {
                string? csvPath = savePath != null ? savePath + ".csv" : null;
                bool savedDuringTraining = Train(model, tokenizer, trainTokens, valTokens, cfg, appCfg, csvPath, savePath, checkpointPath, resumeEpoch, resumeStep, resumeInnerStep);

                // Epochs mode: save at end. Patience/EarlyStopping: already saved best weights during training.
                if (savePath != null && !savedDuringTraining)
                {
                    Console.WriteLine($"Saving weights to '{savePath}'...");
                    model.Save(savePath);
                }

                if (savePath != null)
                {
                    string vp = savePath + ".vocab";
                    tokenizer.SaveVocab(vp);
                    Console.WriteLine($"Weights saved → '{savePath}'  Vocab → '{vp}'");
                }

                // Training completed successfully — delete the crash-recovery checkpoint.
                if (checkpointPath != null && File.Exists(checkpointPath))
                {
                    File.Delete(checkpointPath);
                    Console.WriteLine($"Checkpoint deleted: '{checkpointPath}'");
                }
            }

            // ── 7. Prompt (interactive loop) ──────────────────────────────────
            if (isPrompt)
            {
                Console.WriteLine("Interactive prompt  (Ctrl-C to exit)");
                Console.WriteLine();

                var contextIds  = new System.Collections.Generic.List<int>();
                int maxCtx      = cfg.ContextLength;
                int anchorCount = (int)(maxCtx * appCfg.AnchorFraction);
                bool slidingWindow = appCfg.ContextCompaction.Equals("SlidingWindow", StringComparison.OrdinalIgnoreCase);

                while (true)
                {
                    Console.Write("> ");
                    string? input = Console.ReadLine();
                    if (input == null) break;   // EOF / Ctrl-C redirect

                    if (string.IsNullOrWhiteSpace(input)) continue;

                    // Append user input to context
                    contextIds.AddRange(tokenizer.Encode(input));

                    // Compact context if over the window limit
                    if (contextIds.Count > maxCtx)
                    {
                        if (slidingWindow && anchorCount > 0)
                        {
                            // Preserve the first anchorCount tokens, FIFO the rest
                            int overflow   = contextIds.Count - maxCtx;
                            int fifoStart  = anchorCount;
                            int removeFrom = fifoStart;
                            int removeCount = Math.Min(overflow, contextIds.Count - fifoStart);
                            if (removeCount > 0)
                                contextIds.RemoveRange(removeFrom, removeCount);
                        }
                        else
                        {
                            // FIFO: drop oldest tokens
                            contextIds.RemoveRange(0, contextIds.Count - maxCtx);
                        }
                    }

                    int[] generated = model.Generate(
                        [.. contextIds],
                        numTokens:   appCfg.MaxTokens,
                        temperature: appCfg.Temperature,
                        topK:        appCfg.TopK);

                    Console.WriteLine();
                    foreach (int id in generated)
                        Console.Write(tokenizer.DecodeToken(id));
                    Console.WriteLine();
                    Console.WriteLine();

                    // Append generated tokens to context for next turn
                    contextIds.AddRange(generated);
                }
            }

        }   // using model → Dispose() here; GpuTransformerModel shuts down the GPU context

        // ── config summary ────────────────────────────────────────────────────
        /// <summary>
        /// Prints a formatted summary of the active configuration to stdout,
        /// including architecture, parameter count, file paths, and training or
        /// prompt settings depending on the current action.
        /// </summary>
        /// <param name="cfg">Transformer architecture and training hyper-parameters.</param>
        /// <param name="model">Instantiated model, used to report total parameter count.</param>
        /// <param name="useGpu"><c>true</c> if the GPU backend is active.</param>
        /// <param name="isTrain"><c>true</c> when the action is Train; controls which sections are printed.</param>
        /// <param name="appCfg">Runtime application settings (files, splits, training mode, etc.).</param>
        private static void PrintConfigSummary(TransformerConfig cfg, ITransformerModel model, bool useGpu, bool isTrain, AppConfig appCfg)
        {
            Console.WriteLine("=== Configuration ===");
            Console.WriteLine();

            // ── Architecture (shown for both Train and Prompt) ─────────────────
            Console.WriteLine("  Architecture");
            Console.WriteLine($"    Backend        : {(useGpu ? "GPU" : "CPU")}");
            Console.WriteLine($"    Vocab size     : {cfg.VocabSize:N0}");
            Console.WriteLine($"    Embedding dim  : {cfg.EmbeddingDim}");
            Console.WriteLine($"    Attention heads: {cfg.NumHeads}  (head dim: {cfg.HeadDim})");
            Console.WriteLine($"    Layers         : {cfg.NumLayers}");
            Console.WriteLine($"    FFN dim        : {cfg.FFNDim}");
            Console.WriteLine($"    Context length : {cfg.ContextLength}");
            Console.WriteLine($"    Parameters     : {model.ParameterCount:N0}");

            if (isTrain)
            {
                // ── Parameter breakdown ────────────────────────────────────────
                int    D = cfg.EmbeddingDim;
                int    V = cfg.VocabSize;
                int    F = cfg.FFNDim;
                int    L = cfg.NumLayers;

                long embedding   = (long)V * D;
                long perLayerLN  = 2L * (D + D);                        // 2 LayerNorms × (scale + bias)
                long perLayerAttn= 4L * ((long)D * D + D);              // Q/K/V/O weight + bias
                long perLayerFFN = (long)D * F + F + (long)F * D + D;   // W1+b1 + W2+b2
                long perLayer    = perLayerLN + perLayerAttn + perLayerFFN;
                long allLayers   = perLayer * L;
                long finalLN     = D + D;
                long outputProj  = (long)D * V + V;

                Console.WriteLine();
                Console.WriteLine("  Parameter breakdown");
                Console.WriteLine($"    Embedding table  : {V:N0} × {D} = {embedding:N0}");
                Console.WriteLine($"    Per layer × {L}");
                Console.WriteLine($"      LayerNorms (×2): 2 × ({D}+{D}) = {perLayerLN:N0}");
                Console.WriteLine($"      Attn Q/K/V/O   : 4 × ({D}×{D}+{D}) = {perLayerAttn:N0}");
                Console.WriteLine($"      FFN W1+W2      : {D}×{F}+{F} + {F}×{D}+{D} = {perLayerFFN:N0}");
                Console.WriteLine($"      Layer total    : {perLayer:N0}  ×{L} = {allLayers:N0}");
                Console.WriteLine($"    Final LayerNorm  : {D}+{D} = {finalLN:N0}");
                Console.WriteLine($"    Output proj      : {D}×{V}+{V} = {outputProj:N0}");
                Console.WriteLine($"    Total            : {embedding:N0} + {allLayers:N0} + {finalLN:N0} + {outputProj:N0} = {model.ParameterCount:N0}");
            }

            if (isTrain)
            {
                // ── Files ──────────────────────────────────────────────────────
                Console.WriteLine();
                Console.WriteLine("  Files");
                Console.WriteLine($"    Train corpus   : {appCfg.TrainFile}");
                Console.WriteLine($"    Save weights   : {appCfg.SaveFile}");
                if (!string.IsNullOrEmpty(appCfg.LoadFile))
                    Console.WriteLine($"    Load weights   : {appCfg.LoadFile}");
                if (!string.IsNullOrEmpty(appCfg.ErrorFile))
                    Console.WriteLine($"    Error log      : {appCfg.ErrorFile}");

                // ── Validation ─────────────────────────────────────────────────
                Console.WriteLine();
                Console.WriteLine("  Validation");
                Console.WriteLine($"    Split mode     : {appCfg.ValidationSplit}");
                if (!appCfg.ValidationSplit.Equals("None", StringComparison.OrdinalIgnoreCase))
                    Console.WriteLine($"    Hold-out       : {appCfg.ValidationFraction * 100:F0}%");

                // ── Training ───────────────────────────────────────────────────
                Console.WriteLine();
                Console.WriteLine("  Training");
                Console.WriteLine($"    Mode           : {appCfg.TrainingMode}");
                Console.WriteLine($"    Epochs (max)   : {cfg.Epochs}");
                if (!appCfg.TrainingMode.Equals("Epochs", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine($"    Patience       : {appCfg.Patience}");
                    if (appCfg.TrainingMode.Equals("EarlyStopping", StringComparison.OrdinalIgnoreCase))
                        Console.WriteLine($"    Min delta loss : {appCfg.MinDeltaLoss}");
                }
                Console.WriteLine($"    Learning rate  : {cfg.LearningRate}  (warmup {cfg.WarmupSteps} steps → cosine → {cfg.MinLearningRate})");
                Console.WriteLine($"    RoPE           : {cfg.UseRoPE}");
                Console.WriteLine($"    Adam β1/β2/ε   : {cfg.Beta1} / {cfg.Beta2} / {cfg.AdamEps}");
                Console.WriteLine($"    Grad clip      : {cfg.GradClip}");
                Console.WriteLine($"    Accumulation   : {cfg.AccumulationSteps}x  (eff. batch: {cfg.AccumulationSteps * cfg.ContextLength:N0} tokens)");
                Console.WriteLine($"    Seed           : {(cfg.Seed >= 0 ? cfg.Seed.ToString(System.Globalization.CultureInfo.InvariantCulture) : "random")}");
                if (cfg.SampleEvery > 0)
                    Console.WriteLine($"    Sample every   : {cfg.SampleEvery} epochs  (prompt: \"{cfg.SamplePrompt}\")");
            }
            else
            {
                // ── Prompt ─────────────────────────────────────────────────────
                Console.WriteLine();
                Console.WriteLine("  Prompt");
                Console.WriteLine($"    Load weights   : {appCfg.LoadFile}");
                Console.WriteLine($"    Max tokens     : {appCfg.MaxTokens}");
                Console.WriteLine($"    Temperature    : {appCfg.Temperature}");
                Console.WriteLine($"    Top-K          : {appCfg.TopK}");
                Console.WriteLine($"    Compaction     : {appCfg.ContextCompaction}");
                if (appCfg.ContextCompaction.Equals("SlidingWindow", StringComparison.OrdinalIgnoreCase))
                    Console.WriteLine($"    Anchor         : {appCfg.AnchorFraction * 100:F0}%  ({(int)(cfg.ContextLength * appCfg.AnchorFraction)} tokens)");
                if (!string.IsNullOrEmpty(appCfg.ErrorFile))
                    Console.WriteLine($"    Error log      : {appCfg.ErrorFile}");
            }
        }

        // ── training loop ─────────────────────────────────────────────────────
        /// <summary>
        /// Runs the main training loop, including gradient accumulation, learning-rate
        /// scheduling, optional validation, time-based checkpointing, and early stopping.
        /// </summary>
        /// <param name="model">The transformer model to train.</param>
        /// <param name="tokenizer">Tokenizer used to decode samples during periodic generation.</param>
        /// <param name="trainTokens">Flat array of token IDs used for training.</param>
        /// <param name="valTokens">
        /// Flat array of token IDs held out for validation.  May be empty when
        /// <see cref="AppConfig.ValidationSplit"/> is <c>None</c>.
        /// </param>
        /// <param name="cfg">Transformer architecture and training hyper-parameters.</param>
        /// <param name="appCfg">Runtime application settings (training mode, patience, files, etc.).</param>
        /// <param name="csvPath">
        /// Optional path for a CSV loss log written after every epoch.
        /// <c>null</c> disables logging.
        /// </param>
        /// <param name="savePath">
        /// Optional path to write the best weights during Patience/EarlyStopping training.
        /// <c>null</c> disables mid-training saves.
        /// </param>
        /// <param name="checkpointPath">
        /// Optional path for the crash-recovery checkpoint file.
        /// <c>null</c> disables checkpointing.
        /// </param>
        /// <param name="startEpoch">0-indexed epoch to begin from when resuming.</param>
        /// <param name="startStep">Total Adam steps already completed when resuming.</param>
        /// <param name="startInnerStep">
        /// Inner-loop step within <paramref name="startEpoch"/> to resume from.
        /// <c>-1</c> signals a version-1 checkpoint; the value is derived from
        /// <paramref name="startStep"/> and <paramref name="startEpoch"/> instead.
        /// </param>
        /// <returns>
        /// <c>true</c> if the best weights were saved during training
        /// (Patience or EarlyStopping mode); <c>false</c> in Epochs mode where weights
        /// are saved by the caller after the loop completes.
        /// </returns>
        private static bool Train(
            ITransformerModel model,
            ITokenizer        tokenizer,
            int[]             trainTokens,
            int[]             valTokens,
            TransformerConfig cfg,
            AppConfig         appCfg,
            string?           csvPath,
            string?           savePath,
            string?           checkpointPath  = null,
            int               startEpoch      = 0,
            int               startStep       = 0,
            int               startInnerStep  = 0)
        {
            int T          = cfg.ContextLength;
            int step       = startStep;
            int maxOffset  = trainTokens.Length - T - 1;   // last valid start index
            // Number of steps per epoch = number of non-overlapping chunks that fit.
            // Random sampling still visits the same total token budget per epoch.
            int stepsPerEpoch  = maxOffset / T + 1;
            int totalAdamSteps = stepsPerEpoch * cfg.Epochs;

            // Where to resume within the start epoch.
            // v2 checkpoints store the exact inner step; v1 checkpoints (startInnerStep = -1)
            // fall back to deriving it from the Adam step count.
            int innerStart = startInnerStep >= 0
                ? startInnerStep
                : startStep - startEpoch * stepsPerEpoch;

            var rng = new Random();   // separate RNG so training seed doesn't affect sampling

            Console.WriteLine("=== Training ===");
            string accumInfo = cfg.AccumulationSteps > 1
                ? $"  |  Accumulation: {cfg.AccumulationSteps}x (eff. batch: {cfg.AccumulationSteps * T} tokens)"
                : "";
            string lrInfo = cfg.WarmupSteps > 0
                ? $"  |  LR: {cfg.LearningRate} (warmup {cfg.WarmupSteps} steps → cosine → {cfg.MinLearningRate})"
                : $"  |  LR: {cfg.LearningRate} (cosine → {cfg.MinLearningRate})";
            Console.WriteLine($"Epochs: {cfg.Epochs}  |  Context: {T}  |  Steps/epoch: {stepsPerEpoch}{lrInfo}{accumInfo}");
            if (startEpoch > 0 || innerStart > 0)
                Console.WriteLine($"Resuming: epoch {startEpoch + 1}, inner step {innerStart + 1}/{stepsPerEpoch} (Adam step {startStep}).");
            Console.WriteLine();

            double checkpointIntervalSecs = appCfg.CheckpointEveryMinutes * 60.0;
            var    checkpointTimer        = Stopwatch.StartNew();  // wall-clock time since last checkpoint

            // ── CSV log ───────────────────────────────────────────────────────────
            StreamWriter? csv = null;
            if (csvPath != null)
            {
                csv = new StreamWriter(csvPath, append: false) { AutoFlush = true };
                csv.WriteLine("epoch,step,train_loss,train_ppl,val_loss,val_ppl,epoch_elapsed_s,total_elapsed_s,total_est_s");
            }

            // ── Early-stopping state ──────────────────────────────────────────────
            bool   modeEpochs    = appCfg.TrainingMode.Equals("Epochs",        StringComparison.OrdinalIgnoreCase);
            bool   modePatience  = appCfg.TrainingMode.Equals("Patience",      StringComparison.OrdinalIgnoreCase);
            float  bestValLoss   = float.MaxValue;
            int    patienceLeft  = appCfg.Patience;
            bool   stopEarly     = false;
            bool   savedWeights  = false;

            double totalEpochSeconds = 0.0;
            var    epochTimer        = new Stopwatch();

            for (int epoch = startEpoch; epoch < cfg.Epochs && !stopEarly; epoch++)
            {
                float epochLoss = 0f;
                int   numChunks = 0;
                epochTimer.Restart();
                double lastReportTime = 0.0;  // epochTimer seconds at last console update

                int   accumN  = Math.Max(1, cfg.AccumulationSteps);
                int[] input   = new int[T];
                int[] target  = new int[T];

                int sStart = (epoch == startEpoch) ? innerStart : 0;
                for (int s = sStart; s < stepsPerEpoch; s++)
                {
                    model.ZeroAllGradients();
                    float accumLoss = 0f;
                    for (int a = 0; a < accumN; a++)
                    {
                        FillChunk(trainTokens, rng.Next(0, maxOffset + 1), T, input, target);
                        accumLoss += model.AccumulateStep(input, target);
                    }
                    if (accumN > 1)
                        model.ScaleAllGradients(1f / accumN);
                    step++;
                    float lr   = cfg.ComputeLR(step, totalAdamSteps);
                    model.ClipAndUpdate(step, lr);
                    float loss = accumLoss / accumN;

                    epochLoss += loss;
                    numChunks++;

                    // ── Time-based mid-epoch checkpoint ───────────────────────
                    if (checkpointPath != null
                        && checkpointIntervalSecs > 0
                        && checkpointTimer.Elapsed.TotalSeconds >= checkpointIntervalSecs)
                    {
                        model.SaveCheckpoint(checkpointPath, epoch, step, s + 1);
                        checkpointTimer.Restart();
                    }

                    double nowSecs = epochTimer.Elapsed.TotalSeconds;
                    if (nowSecs - lastReportTime >= 1.0 || s == stepsPerEpoch - 1)
                    {
                        lastReportTime = nowSecs;
                        float  pct          = (s + 1) * 100f / stepsPerEpoch;
                        float  runningAvg   = epochLoss / numChunks;
                        int    barWidth     = 25;
                        int    filled       = (int)(pct / 100f * barWidth);
                        string bar          = new string('█', filled) + new string('░', barWidth - filled);
                        double epochElapsed  = nowSecs;
                        double epochEst      = pct > 0 ? epochElapsed * 100.0 / pct : 0;
                        double totalElapsed  = totalEpochSeconds + epochElapsed;
                        double totalEst      = (totalEpochSeconds + epochEst) / (epoch + 1) * cfg.Epochs;
                        string patienceTag = modeEpochs ? "" : $"  patience={patienceLeft}/{appCfg.Patience}";
                        Console.Write(
                            $"\r  Epoch {epoch + 1,3}/{cfg.Epochs}  step {s + 1,6}/{stepsPerEpoch}  " +
                            $"| [{bar}] {pct,6:F2}%  loss={runningAvg:F4}  " +
                            $"epoch_elapsed={FormatDuration(epochElapsed)}  epoch_est={FormatDuration(epochEst)}  " +
                            $"total_elapsed={FormatDuration(totalElapsed)}  total_est={FormatDuration(totalEst)}{patienceTag}   ");
                    }
                }

                epochTimer.Stop();
                double epochSecs = epochTimer.Elapsed.TotalSeconds;
                totalEpochSeconds += epochSecs;

                float  avgLoss      = epochLoss / numChunks;
                float  perplexity   = MathF.Exp(avgLoss);
                float  pctComplete  = (epoch + 1) * 100f / cfg.Epochs;
                double avgEpochSecs = totalEpochSeconds / (epoch + 1);
                double estTotalSecs = avgEpochSecs * cfg.Epochs;

                string epochPatienceTag = modeEpochs ? "" : $"  patience={patienceLeft}/{appCfg.Patience}  best_val={bestValLoss:F4}";
                // Overwrite the progress line with the final epoch summary
                Console.WriteLine(
                    $"\rEpoch {epoch + 1,3}/{cfg.Epochs}  step={step}  " +
                    $"| {pctComplete:F2}%  loss={avgLoss:F4}  ppl={perplexity:F1}  " +
                    $"epoch_elapsed={FormatDuration(epochSecs)}  epoch_est={FormatDuration(epochSecs)}  " +
                    $"total_elapsed={FormatDuration(totalEpochSeconds)}  total_est={FormatDuration(estTotalSecs)}{epochPatienceTag}  ");

                // ── Validation pass ───────────────────────────────────────────
                float valLoss = 0f;
                float valPpl  = 0f;
                if (valTokens.Length > T + 1)
                {
                    float valTotal = 0f;
                    int   valChunks = 0;
                    int[] valInput  = new int[T];
                    int[] valTarget = new int[T];
                    for (int offset = 0; offset + T + 1 <= valTokens.Length; offset += T)
                    {
                        FillChunk(valTokens, offset, T, valInput, valTarget);
                        valTotal += model.Evaluate(valInput, valTarget);
                        valChunks++;
                    }
                    valLoss = valChunks > 0 ? valTotal / valChunks : 0f;
                    valPpl  = MathF.Exp(valLoss);

                    float improvement = bestValLoss - valLoss;
                    bool  improved    = improvement > (float)appCfg.MinDeltaLoss;

                    if (improved)
                    {
                        bestValLoss  = valLoss;
                        patienceLeft = appCfg.Patience;
                        if (savePath != null) { model.Save(savePath); savedWeights = true; }
                    }
                    else
                    {
                        patienceLeft--;
                    }

                    Console.WriteLine($"  val_loss={valLoss:F4}  val_ppl={valPpl:F1}");

                    if (!modeEpochs && patienceLeft <= 0)
                    {
                        Console.WriteLine(modePatience
                            ? $"  >> Stopping: val_loss did not improve for {appCfg.Patience} consecutive epochs."
                            : $"  >> Stopping: val_loss improvement < MinDeltaLoss ({appCfg.MinDeltaLoss}) for {appCfg.Patience} consecutive epochs.");
                        stopEarly = true;
                    }
                }

                // ── Save crash-recovery checkpoint at epoch end ───────────────
                if (checkpointPath != null)
                {
                    model.SaveCheckpoint(checkpointPath, epoch, step, stepsPerEpoch);
                    checkpointTimer.Restart();
                }

                csv?.WriteLine($"{epoch + 1},{step},{avgLoss:F6},{perplexity:F4},{valLoss:F6},{valPpl:F4},{epochSecs:F1},{totalEpochSeconds:F1},{estTotalSecs:F1}");

                if (cfg.SampleEvery > 0 && (epoch + 1) % cfg.SampleEvery == 0)
                {
                    Console.Write($"  Sample: \"{cfg.SamplePrompt}");
                    int[] ids = model.Generate(
                        tokenizer.Encode(cfg.SamplePrompt),
                        numTokens:   80,
                        temperature: 0.9f,
                        topK:        15);
                    foreach (int id in ids)
                        Console.Write(tokenizer.DecodeToken(id));
                    Console.WriteLine("\"");
                }
            }

            csv?.Dispose();
            return savedWeights;
        }

        // ── timing helper ─────────────────────────────────────────────────────
        /// <summary>
        /// Formats a duration in seconds as <c>DD:HH:MM:SS</c>, clamping each component
        /// to a maximum of 99 to keep display width fixed.
        /// </summary>
        /// <param name="totalSeconds">Elapsed time in seconds. Negative values are treated as zero.</param>
        /// <returns>A fixed-width string in <c>DD:HH:MM:SS</c> format.</returns>
        private static string FormatDuration(double totalSeconds)
        {
            if (totalSeconds < 0) totalSeconds = 0;
            long s       = (long)totalSeconds;
            long days    = Math.Min(s / 86400,       99);
            long hours   = Math.Min((s % 86400) / 3600, 99);
            long minutes = Math.Min((s % 3600)  / 60,   99);
            long seconds = Math.Min(s % 60,            99);
            return $"{days:D2}:{hours:D2}:{minutes:D2}:{seconds:D2}";
        }

        // ── training helpers ──────────────────────────────────────────────────
        /// <summary>
        /// Fills pre-allocated <paramref name="input"/> and <paramref name="target"/> arrays
        /// with a contiguous chunk of tokens starting at <paramref name="offset"/>.
        /// The target is the input shifted right by one position (next-token prediction).
        /// </summary>
        /// <param name="tokens">Source token array (full corpus or split).</param>
        /// <param name="offset">Start index within <paramref name="tokens"/>.</param>
        /// <param name="T">Context length; number of tokens to copy.</param>
        /// <param name="input">Output array of length <paramref name="T"/> filled with input tokens.</param>
        /// <param name="target">Output array of length <paramref name="T"/> filled with target tokens.</param>
        private static void FillChunk(int[] tokens, int offset, int T, int[] input, int[] target)
        {
            for (int i = 0; i < T; i++)
            {
                input[i]  = tokens[offset + i];
                target[i] = tokens[offset + i + 1];
            }
        }

        // ── config loading ────────────────────────────────────────────────────
        /// <summary>
        /// Binds the <c>TransformerConfig</c> section of the merged configuration
        /// (appsettings.json overlaid with any CLI overrides) to a new
        /// <see cref="TransformerConfig"/> instance.
        /// </summary>
        /// <param name="configuration">The fully built <see cref="IConfiguration"/> to read from.</param>
        /// <returns>A <see cref="TransformerConfig"/> populated from the configuration.</returns>
        private static TransformerConfig LoadConfig(IConfiguration configuration)
        {
            var cfg = new TransformerConfig();
            configuration.GetSection("TransformerConfig").Bind(cfg);
            Console.WriteLine("Config: 'appsettings.json'");
            return cfg;
        }

        // ── CLI switch mappings ───────────────────────────────────────────────
        /// <summary>
        /// Builds the switch-mapping dictionary used by
        /// <see cref="Microsoft.Extensions.Configuration.CommandLineConfigurationExtensions.AddCommandLine"/>
        /// to translate kebab-case CLI flags (e.g. <c>--train-file</c>) into the
        /// dotted configuration key paths expected by the binder
        /// (e.g. <c>AppConfig:TrainFile</c>).
        /// </summary>
        /// <returns>
        /// A case-insensitive dictionary mapping every supported CLI flag to its
        /// configuration key path.
        /// </returns>
        private static Dictionary<string, string> BuildSwitchMappings() =>
            new(StringComparer.OrdinalIgnoreCase)
            {
                // ── AppConfig ──────────────────────────────────────────────────
                { "--action",               "AppConfig:Action"              },
                { "--backend",              "AppConfig:Backend"             },
                { "--train-file",           "AppConfig:TrainFile"           },
                { "--save-file",            "AppConfig:SaveFile"            },
                { "--load-file",            "AppConfig:LoadFile"            },
                { "--error-file",           "AppConfig:ErrorFile"           },
                { "--checkpoint-every",     "AppConfig:CheckpointEveryMinutes" },
                { "--vocab-size",           "AppConfig:VocabSize"           },
                { "--validation-split",     "AppConfig:ValidationSplit"     },
                { "--validation-fraction",  "AppConfig:ValidationFraction"  },
                { "--training-mode",        "AppConfig:TrainingMode"        },
                { "--patience",             "AppConfig:Patience"            },
                { "--min-delta-loss",       "AppConfig:MinDeltaLoss"        },
                { "--max-tokens",           "AppConfig:MaxTokens"           },
                { "--temperature",          "AppConfig:Temperature"         },
                { "--top-k",                "AppConfig:TopK"                },
                { "--context-compaction",   "AppConfig:ContextCompaction"   },
                { "--anchor-fraction",      "AppConfig:AnchorFraction"      },
                // ── TransformerConfig ──────────────────────────────────────────
                { "--embedding-dim",        "TransformerConfig:EmbeddingDim"    },
                { "--num-heads",            "TransformerConfig:NumHeads"        },
                { "--num-layers",           "TransformerConfig:NumLayers"       },
                { "--ffn-dim",              "TransformerConfig:FFNDim"          },
                { "--context-length",       "TransformerConfig:ContextLength"   },
                { "--epochs",               "TransformerConfig:Epochs"          },
                { "--learning-rate",        "TransformerConfig:LearningRate"    },
                { "--warmup-steps",         "TransformerConfig:WarmupSteps"     },
                { "--min-learning-rate",    "TransformerConfig:MinLearningRate" },
                { "--use-rope",             "TransformerConfig:UseRoPE"         },
                { "--beta1",                "TransformerConfig:Beta1"           },
                { "--beta2",                "TransformerConfig:Beta2"           },
                { "--adam-eps",             "TransformerConfig:AdamEps"         },
                { "--grad-clip",            "TransformerConfig:GradClip"        },
                { "--accumulation-steps",   "TransformerConfig:AccumulationSteps" },
                { "--sample-every",         "TransformerConfig:SampleEvery"     },
                { "--sample-prompt",        "TransformerConfig:SamplePrompt"    },
                { "--seed",                 "TransformerConfig:Seed"            },
            };

        // ── help text ─────────────────────────────────────────────────────────
        /// <summary>
        /// Prints usage information and a full list of CLI options to stdout, then returns.
        /// Invoked when <c>--help</c> or <c>-h</c> is the first argument.
        /// </summary>
        private static void PrintHelp()
        {
            Console.WriteLine("Usage: LLM_App [options]");
            Console.WriteLine();
            Console.WriteLine("All options override the corresponding value in appsettings.json.");
            Console.WriteLine("String values with spaces must be quoted: --sample-prompt \"Hello world\"");
            Console.WriteLine();
            Console.WriteLine("AppConfig options:");
            Console.WriteLine("  --action               <string>  Train or Prompt");
            Console.WriteLine("  --backend              <string>  CPU or GPU");
            Console.WriteLine("  --train-file           <path>    Corpus file for training");
            Console.WriteLine("  --save-file            <path>    Where to save model weights");
            Console.WriteLine("  --load-file            <path>    Weights to load before training/prompting");
            Console.WriteLine("  --error-file           <path>    Redirect stderr to this file");
            Console.WriteLine("  --checkpoint-every     <float>   Save checkpoint every N minutes (0 = epoch end only, default: 60)");
            Console.WriteLine("  --vocab-size           <int>     Tokenizer vocabulary size");
            Console.WriteLine("  --validation-split     <string>  None, Tail, or Random");
            Console.WriteLine("  --validation-fraction  <float>   Hold-out fraction (e.g. 0.1 = 10%)");
            Console.WriteLine("  --training-mode        <string>  Epochs, Patience, or EarlyStopping");
            Console.WriteLine("  --patience             <int>     Consecutive epochs without improvement before stopping");
            Console.WriteLine("  --min-delta-loss       <float>   Min val_loss improvement to reset patience");
            Console.WriteLine("  --max-tokens           <int>     Max tokens to generate per prompt response");
            Console.WriteLine("  --temperature          <float>   Sampling temperature (lower = more focused)");
            Console.WriteLine("  --top-k                <int>     Top-K sampling filter (0 = disabled)");
            Console.WriteLine("  --context-compaction   <string>  FIFO or SlidingWindow");
            Console.WriteLine("  --anchor-fraction      <float>   SlidingWindow anchor fraction (e.g. 0.2 = 20%)");
            Console.WriteLine();
            Console.WriteLine("TransformerConfig options:");
            Console.WriteLine("  --embedding-dim        <int>     Embedding / residual stream dimension");
            Console.WriteLine("  --num-heads            <int>     Number of attention heads");
            Console.WriteLine("  --num-layers           <int>     Number of transformer layers");
            Console.WriteLine("  --ffn-dim              <int>     Feed-forward hidden dimension");
            Console.WriteLine("  --context-length       <int>     Context window size in tokens");
            Console.WriteLine("  --epochs               <int>     Max training epochs");
            Console.WriteLine("  --learning-rate        <float>   Peak Adam learning rate");
            Console.WriteLine("  --warmup-steps         <int>     LR warmup steps (0 = no warmup)");
            Console.WriteLine("  --min-learning-rate    <float>   Minimum LR at end of cosine decay");
            Console.WriteLine("  --use-rope             <bool>    Use RoPE positional encoding (true/false)");
            Console.WriteLine("  --beta1                <float>   Adam β1 (gradient momentum decay)");
            Console.WriteLine("  --beta2                <float>   Adam β2 (adaptive scaling decay)");
            Console.WriteLine("  --adam-eps             <float>   Adam ε (division stability constant)");
            Console.WriteLine("  --grad-clip            <float>   Gradient clipping L2 norm");
            Console.WriteLine("  --accumulation-steps   <int>     Gradient accumulation steps");
            Console.WriteLine("  --sample-every         <int>     Print a sample every N epochs (0 = off)");
            Console.WriteLine("  --sample-prompt        <string>  Prompt used for training samples");
            Console.WriteLine("  --seed                 <int>     RNG seed for weight init (-1 = random)");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  dotnet run -- --action Prompt --load-file weights.bin");
            Console.WriteLine("  dotnet run -- --backend GPU --epochs 50 --learning-rate 1e-4");
            Console.WriteLine("  dotnet run -- --train-file corpus.txt --save-file model.bin --embedding-dim 256 --num-layers 4");
        }

        // ── config validation ─────────────────────────────────────────────────
        /// <summary>
        /// Validates all fields of <paramref name="cfg"/> for correctness and
        /// internal consistency, collecting every problem into a list rather than
        /// stopping at the first error so the user sees all issues at once.
        /// </summary>
        /// <param name="cfg">The application configuration to validate.</param>
        /// <returns>
        /// A list of human-readable error messages.  An empty list means the
        /// configuration is valid.
        /// </returns>
        private static List<string> ValidateConfig(AppConfig cfg)
        {
            var errors = new List<string>();

            // ── Action ────────────────────────────────────────────────────────
            bool isTrain  = cfg.Action.Equals("Train",  StringComparison.OrdinalIgnoreCase);
            bool isPrompt = cfg.Action.Equals("Prompt", StringComparison.OrdinalIgnoreCase);
            if (!isTrain && !isPrompt)
                errors.Add($"Action must be 'Train' or 'Prompt' (got '{cfg.Action}').");

            // ── Backend ───────────────────────────────────────────────────────
            if (!cfg.Backend.Equals("CPU", StringComparison.OrdinalIgnoreCase) &&
                !cfg.Backend.Equals("GPU", StringComparison.OrdinalIgnoreCase))
                errors.Add($"Backend must be 'CPU' or 'GPU' (got '{cfg.Backend}').");

            // ── Files ─────────────────────────────────────────────────────────
            if (isTrain)
            {
                if (string.IsNullOrEmpty(cfg.TrainFile))
                    errors.Add("TrainFile is required when Action=Train.");
                else if (!File.Exists(cfg.TrainFile))
                    errors.Add($"TrainFile not found: '{cfg.TrainFile}'.");

                if (string.IsNullOrEmpty(cfg.SaveFile))
                    errors.Add("SaveFile is required when Action=Train.");
                else
                {
                    string? saveDir = Path.GetDirectoryName(Path.GetFullPath(cfg.SaveFile));
                    if (saveDir != null && !Directory.Exists(saveDir))
                        errors.Add($"SaveFile directory does not exist: '{saveDir}'.");
                }
            }

            if (isPrompt)
            {
                if (string.IsNullOrEmpty(cfg.LoadFile))
                    errors.Add("LoadFile is required when Action=Prompt.");
                else
                {
                    if (!File.Exists(cfg.LoadFile))
                        errors.Add($"LoadFile not found: '{cfg.LoadFile}'.");
                    if (!File.Exists(cfg.LoadFile + ".vocab"))
                        errors.Add($"LoadFile vocab not found: '{cfg.LoadFile}.vocab'.");
                }
            }

            if (!string.IsNullOrEmpty(cfg.LoadFile) && isTrain)
            {
                if (!File.Exists(cfg.LoadFile))
                    errors.Add($"LoadFile not found: '{cfg.LoadFile}'.");
            }

            // ── VocabSize ─────────────────────────────────────────────────────
            if (cfg.VocabSize <= 0)
                errors.Add($"VocabSize must be > 0 (got {cfg.VocabSize}).");

            // ── ValidationSplit ───────────────────────────────────────────────
            bool splitNone   = cfg.ValidationSplit.Equals("None",   StringComparison.OrdinalIgnoreCase);
            bool splitTail   = cfg.ValidationSplit.Equals("Tail",   StringComparison.OrdinalIgnoreCase);
            bool splitRandom = cfg.ValidationSplit.Equals("Random", StringComparison.OrdinalIgnoreCase);
            if (!splitNone && !splitTail && !splitRandom)
                errors.Add($"ValidationSplit must be 'None', 'Tail', or 'Random' (got '{cfg.ValidationSplit}').");
            if (!splitNone && (cfg.ValidationFraction <= 0.0 || cfg.ValidationFraction >= 1.0))
                errors.Add($"ValidationFraction must be in (0, 1) (got {cfg.ValidationFraction}).");

            // ── TrainingMode ──────────────────────────────────────────────────
            bool modeEpochs   = cfg.TrainingMode.Equals("Epochs",        StringComparison.OrdinalIgnoreCase);
            bool modePatience = cfg.TrainingMode.Equals("Patience",      StringComparison.OrdinalIgnoreCase);
            bool modeEarly    = cfg.TrainingMode.Equals("EarlyStopping", StringComparison.OrdinalIgnoreCase);
            if (!modeEpochs && !modePatience && !modeEarly)
                errors.Add($"TrainingMode must be 'Epochs', 'Patience', or 'EarlyStopping' (got '{cfg.TrainingMode}').");

            if (isTrain && (modePatience || modeEarly) && splitNone)
                errors.Add($"TrainingMode={cfg.TrainingMode} requires ValidationSplit to be 'Tail' or 'Random'.");

            if ((modePatience || modeEarly) && cfg.Patience <= 0)
                errors.Add($"Patience must be > 0 (got {cfg.Patience}).");

            if (modeEarly && cfg.MinDeltaLoss < 0.0)
                errors.Add($"MinDeltaLoss must be >= 0 (got {cfg.MinDeltaLoss}).");

            if (cfg.CheckpointEveryMinutes < 0.0)
                errors.Add($"CheckpointEveryMinutes must be >= 0 (got {cfg.CheckpointEveryMinutes}).");

            // ── Prompt settings ───────────────────────────────────────────────
            if (cfg.MaxTokens <= 0)
                errors.Add($"MaxTokens must be > 0 (got {cfg.MaxTokens}).");

            if (cfg.Temperature <= 0f)
                errors.Add($"Temperature must be > 0 (got {cfg.Temperature}).");

            if (cfg.TopK < 0)
                errors.Add($"TopK must be >= 0 (got {cfg.TopK}).");

            bool compFifo   = cfg.ContextCompaction.Equals("FIFO",          StringComparison.OrdinalIgnoreCase);
            bool compSlide  = cfg.ContextCompaction.Equals("SlidingWindow",  StringComparison.OrdinalIgnoreCase);
            if (!compFifo && !compSlide)
                errors.Add($"ContextCompaction must be 'FIFO' or 'SlidingWindow' (got '{cfg.ContextCompaction}').");

            if (compSlide && (cfg.AnchorFraction <= 0f || cfg.AnchorFraction >= 1f))
                errors.Add($"AnchorFraction must be in (0, 1) when ContextCompaction=SlidingWindow (got {cfg.AnchorFraction}).");

            return errors;
        }

    }
}
