//! Model-backed performance harness for the on-device `local_gguf` engine.
//!
//! This is a measurement tool, not a Criterion microbenchmark: it drives the real
//! llama.cpp/Metal engine against a real GGUF file and reports the numbers that
//! matter for the 16 GB M1 target (cold/warm load, time to first token, full
//! suggestion latency, decode throughput, peak resident memory and scheduler
//! responsiveness).
//!
//! It is gated on `LOCAL_GGUF_TEST_MODEL`, so it skips cleanly when no model is
//! present (for example on CI). Run it release-mode, unsandboxed (Metal is
//! unavailable inside the agent sandbox):
//!
//! ```text
//! LOCAL_GGUF_TEST_MODEL=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//!   cargo run --release -p local_gguf --features metal --example local_gguf_benchmark
//! ```
//!
//! Optional environment variables:
//! - `LOCAL_GGUF_BENCH_ITERATIONS` (default 5): full-suggestion runs per context.
//! - `LOCAL_GGUF_BENCH_OUTPUT_TOKENS` (default 32): tokens per full suggestion.
//! - `LOCAL_GGUF_BENCH_CONTEXT_SIZE` (default 2048): engine context window.

use local_gguf::{
    CancellationToken, Completion, LocalGgufEngine, LocalGgufError, QWEN_FIM_STOP_TOKENS,
    build_qwen_fim_prompt, is_supported,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

/// A single measured scenario.
struct ScenarioResult {
    label: String,
    prompt_tokens: usize,
    first_token_ms: f64,
    full: LatencyStats,
    full_rss_mib: f64,
}

struct LatencyStats {
    samples: Vec<f64>,
    generated_tokens: usize,
    decode_tokens_per_second: f64,
}

impl LatencyStats {
    fn median(&self) -> f64 {
        median(&self.samples)
    }

    fn min(&self) -> f64 {
        self.samples.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Measures the largest gap between scheduled ticks of a background thread while
/// the engine works. This is a proxy for whether on-device inference starves the
/// rest of the process (in the app, the blocking work runs on a worker thread, so
/// the UI thread only awaits).
struct Responsiveness {
    stop: Arc<AtomicBool>,
    max_gap_ms: Arc<Mutex<f64>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Responsiveness {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let max_gap_ms = Arc::new(Mutex::new(0.0f64));
        let thread = {
            let stop = stop.clone();
            let max_gap_ms = max_gap_ms.clone();
            thread::Builder::new()
                .name("local_gguf_bench_heartbeat".to_string())
                .spawn(move || {
                    let tick = Duration::from_millis(2);
                    let mut last = Instant::now();
                    while !stop.load(Ordering::Relaxed) {
                        thread::sleep(tick);
                        let now = Instant::now();
                        let gap_ms = (now - last).as_secs_f64() * 1000.0;
                        last = now;
                        let mut max_gap_ms = max_gap_ms
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if gap_ms > *max_gap_ms {
                            *max_gap_ms = gap_ms;
                        }
                    }
                })
                .ok()
        };
        Self {
            stop,
            max_gap_ms,
            thread,
        }
    }

    fn finish(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                eprintln!("warning: responsiveness thread panicked");
            }
        }
        *self
            .max_gap_ms
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    smol::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = model_path() else {
        eprintln!(
            "LOCAL_GGUF_TEST_MODEL is not set; skipping the benchmark.\n\
             Set it to a Qwen2.5-Coder GGUF to run the harness."
        );
        return Ok(());
    };

    if !is_supported() {
        eprintln!(
            "the native on-device engine is not available in this build/target; skipping the benchmark."
        );
        return Ok(());
    }

    let iterations = env_usize("LOCAL_GGUF_BENCH_ITERATIONS", 5);
    let output_tokens = env_u32("LOCAL_GGUF_BENCH_OUTPUT_TOKENS", 32);
    let context_size = env_u32(
        "LOCAL_GGUF_BENCH_CONTEXT_SIZE",
        local_gguf::DEFAULT_CONTEXT_SIZE,
    );

    let file_size = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);

    println!("# local_gguf benchmark\n");
    println!("- model: `{}`", model_path.display());
    println!(
        "- model size: {:.2} GiB",
        file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("- context size: {context_size} tokens");
    println!("- output cap: {output_tokens} tokens");
    println!("- iterations per scenario: {iterations}");
    println!("- baseline peak RSS: {:.1} MiB\n", peak_rss_mib());

    let responsiveness = Responsiveness::start();

    let engine = LocalGgufEngine::new();

    // Cold load: the first load in this process, reading the weights from disk.
    let cold_load_start = Instant::now();
    engine.load(model_path.clone(), context_size).await?;
    let cold_load = cold_load_start.elapsed();
    let cold_load_rss = peak_rss_mib();

    // Warm reload: unload, then load again with the file already in the page cache
    // and the Metal backend already initialised.
    engine.unload();
    let warm_load_start = Instant::now();
    engine.load(model_path, context_size).await?;
    let warm_load = warm_load_start.elapsed();

    // One discarded completion so the first measured request does not pay for any
    // lazy Metal pipeline compilation.
    let _ = engine
        .complete(
            short_prompt(),
            qwen_stop_tokens(),
            1,
            CancellationToken::new(),
        )
        .await?;

    let scenarios = [
        ("short context", short_prompt()),
        ("long context (~950 tokens)", long_prompt()),
    ];
    let mut results = Vec::new();
    for (label, prompt) in scenarios {
        results.push(measure_scenario(&engine, label, prompt, output_tokens, iterations).await?);
    }

    let cancel = measure_cancellation(&engine).await?;

    let max_gap_ms = responsiveness.finish();

    print_table(&results);
    println!("\n- cold model load: {:.3} s", cold_load.as_secs_f64());
    println!("- warm reload: {:.3} s", warm_load.as_secs_f64());
    println!("- cold load peak RSS: {cold_load_rss:.1} MiB");
    println!(
        "- final peak RSS: {:.1} MiB ({:.2} GiB)",
        peak_rss_mib(),
        peak_rss_mib() / 1024.0
    );
    println!("- scheduler responsiveness: max tick gap {max_gap_ms:.1} ms while generating");
    println!(
        "- cancellation: requested at {:.0} ms, engine stopped after {:.1} ms",
        cancel.requested_after.as_secs_f64() * 1000.0,
        cancel.observed_latency.as_secs_f64() * 1000.0
    );

    Ok(())
}

async fn measure_scenario(
    engine: &LocalGgufEngine,
    label: &str,
    prompt: String,
    output_tokens: u32,
    iterations: usize,
) -> Result<ScenarioResult, LocalGgufError> {
    // Warm up this prompt's batch shape first, so the one-time Metal graph
    // reservation for the prompt length is not attributed to the first-token
    // measurement below.
    let _ = engine
        .complete(
            prompt.clone(),
            qwen_stop_tokens(),
            output_tokens,
            CancellationToken::new(),
        )
        .await?;

    // Time to first token: a one-token request measures prefill plus a single
    // decode step, which is exactly the user-visible latency to the first token.
    let first_token_start = Instant::now();
    let first_token = engine
        .complete(
            prompt.clone(),
            qwen_stop_tokens(),
            1,
            CancellationToken::new(),
        )
        .await?;
    let first_token_ms = first_token_start.elapsed().as_secs_f64() * 1000.0;

    let mut samples = Vec::with_capacity(iterations);
    let mut generated_tokens = 0;
    for _ in 0..iterations {
        let start = Instant::now();
        let completion = engine
            .complete(
                prompt.clone(),
                qwen_stop_tokens(),
                output_tokens,
                CancellationToken::new(),
            )
            .await?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        generated_tokens = completion.generated_tokens;
    }
    let full_rss_mib = peak_rss_mib();

    let median_ms = median(&samples);
    let decode_tokens_per_second = if median_ms > first_token_ms && generated_tokens > 1 {
        (generated_tokens - 1) as f64 / ((median_ms - first_token_ms) / 1000.0)
    } else {
        f64::NAN
    };

    Ok(ScenarioResult {
        label: label.to_string(),
        prompt_tokens: first_token.prompt_tokens,
        first_token_ms,
        full: LatencyStats {
            samples,
            generated_tokens,
            decode_tokens_per_second,
        },
        full_rss_mib,
    })
}

struct CancellationResult {
    requested_after: Duration,
    observed_latency: Duration,
}

/// Starts a long generation, cancels it from another thread, and measures how long
/// the engine takes to observe cancellation between decode steps.
async fn measure_cancellation(
    engine: &LocalGgufEngine,
) -> Result<CancellationResult, LocalGgufError> {
    let requested_after = Duration::from_millis(150);
    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let canceller = thread::spawn(move || {
        thread::sleep(requested_after);
        cancel_token.cancel();
    });

    let start = Instant::now();
    let completion: Completion = engine
        .complete(long_prompt(), qwen_stop_tokens(), 4096, token)
        .await?;
    let observed_latency = start.elapsed().saturating_sub(requested_after);

    if canceller.join().is_err() {
        eprintln!("warning: canceller thread panicked");
    }

    debug_assert_eq!(completion.stop, local_gguf::CompletionStop::Cancelled);

    Ok(CancellationResult {
        requested_after,
        observed_latency,
    })
}

fn print_table(results: &[ScenarioResult]) {
    println!(
        "| scenario | prompt tokens | tokens generated | first token (ms) | full suggestion p50 (ms) | full min/max (ms) | decode tok/s | peak RSS (MiB) |"
    );
    println!("|---|---|---|---|---|---|---|---|");
    for result in results {
        println!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} / {:.1} | {:.1} | {:.0} |",
            result.label,
            result.prompt_tokens,
            result.full.generated_tokens,
            result.first_token_ms,
            result.full.median(),
            result.full.min(),
            result.full.max(),
            result.full.decode_tokens_per_second,
            result.full_rss_mib,
        );
    }
}

fn model_path() -> Option<PathBuf> {
    std::env::var_os("LOCAL_GGUF_TEST_MODEL").map(PathBuf::from)
}

fn qwen_stop_tokens() -> Vec<String> {
    QWEN_FIM_STOP_TOKENS
        .iter()
        .map(|token| token.to_string())
        .collect()
}

fn short_prompt() -> String {
    // A one-line completion; the model reliably writes a whole function body, so
    // the "full suggestion" timing covers more than a couple of tokens.
    build_qwen_fim_prompt(
        "def fibonacci(n):\n",
        "\n\nfor i in range(10):\n    print(fibonacci(i))\n",
    )
}

/// A prompt of roughly 950 tokens, to compare prefill cost against the short case.
///
/// The app caps the fill-in-the-middle excerpt at `FIM_CONTEXT_TOKENS` (512) in
/// `crates/edit_prediction/src/fim.rs`, so this deliberately overshoots that budget
/// to show how prefill scales.
fn long_prompt() -> String {
    let mut prefix = String::from(
        "# Data helpers\n\n\ndef clamp(value, low, high):\n    \"\"\"Clamp value into [low, high].\"\"\"\n    return max(low, min(value, high))\n\n\n",
    );
    for index in 0..30 {
        prefix.push_str(&format!(
            "def scale_{index}(value):\n    \"\"\"Scale value by {index}.\"\"\"\n    return value * {index}\n\n\n"
        ));
    }
    prefix.push_str(
        "def average(values):\n    \"\"\"Return the arithmetic mean of values.\"\"\"\n    ",
    );
    build_qwen_fim_prompt(&prefix, "\n\n\nprint(average([1, 2, 3]))\n")
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// The process's peak resident set size, in MiB.
///
/// On macOS `ru_maxrss` is reported in bytes (Linux reports KiB), which is what
/// makes it usable directly for the "peak RSS" column here.
fn peak_rss_mib() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` writes to the pointer we pass and does not retain it.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return f64::NAN;
    }
    // SAFETY: `getrusage` returned success, so the value is initialised.
    let usage = unsafe { usage.assume_init() };
    let bytes = usage.ru_maxrss.max(0) as f64;
    bytes / (1024.0 * 1024.0)
}
