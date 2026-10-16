//! On-device GGUF inference for edit predictions.
//!
//! This crate runs a quantized GGUF model (for example
//! `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`) entirely inside the host process using
//! `llama.cpp` with the Metal backend on Apple silicon. It never opens a network
//! socket, never spawns a helper process, and never requires an account or API
//! key.
//!
//! Blocking inference runs on a single dedicated worker thread; callers drive it
//! through [`LocalGgufEngine`], which serializes requests and can be shared
//! between threads.
//!
//! The native engine is compiled only on Apple silicon with the `metal` feature
//! (see [`is_supported`]). Everywhere else the crate builds a stub that reports
//! [`LocalGgufError::Unsupported`], so it keeps compiling on every platform —
//! including when the `metal` feature is enabled by `cargo --all-features`,
//! because the native dependency is declared for Apple silicon only.

mod engine;

use flume::Sender;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::JoinHandle,
};
use thiserror::Error;

/// Default context window, in tokens, used when a caller does not choose one.
pub const DEFAULT_CONTEXT_SIZE: u32 = 2048;

/// Stop sequences for the Qwen fill-in-the-middle prompt format.
///
/// These are model *special tokens*: the engine resolves them to vocabulary ids
/// instead of matching raw text.
pub const QWEN_FIM_STOP_TOKENS: [&str; 5] = [
    "<|endoftext|>",
    "<|fim_pad|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
];

/// Whether the native on-device engine is available in this build.
///
/// This is true only on Apple silicon with the `metal` feature. The feature alone
/// is not enough: the llama.cpp dependency is declared for that target only, so on
/// every other target the crate falls back to its stub even if the feature is
/// enabled (for example by `cargo --all-features`).
pub const fn is_supported() -> bool {
    cfg!(all(
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

/// Builds a Qwen fill-in-the-middle prompt from the code before and after the
/// cursor. The `<|fim_*|>` markers are model special tokens.
pub fn build_qwen_fim_prompt(prefix: &str, suffix: &str) -> String {
    format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
}

/// Errors returned by the local GGUF engine.
#[derive(Debug, Error)]
pub enum LocalGgufError {
    /// The engine was not compiled in for this platform (no Apple-silicon `metal`
    /// engine in this build).
    #[error("local GGUF inference is not supported on this platform")]
    Unsupported,
    /// No model is loaded; call [`LocalGgufEngine::load`] first.
    #[error("no local GGUF model is loaded")]
    ModelNotLoaded,
    /// The requested model file does not exist.
    #[error("model file not found: {0}")]
    ModelNotFound(PathBuf),
    /// The model file exists but could not be loaded (corrupt or unsupported).
    #[error("failed to load model {path}: {message}")]
    ModelLoad {
        /// The path that failed to load.
        path: PathBuf,
        /// The underlying loader error.
        message: String,
    },
    /// The Metal/GPU backend could not be initialized.
    #[error("local GGUF backend unavailable: {0}")]
    BackendUnavailable(String),
    /// The machine ran out of memory while loading or generating.
    #[error("out of memory while running local GGUF inference")]
    OutOfMemory,
    /// The prompt (or the requested output) does not fit the context window.
    #[error("prompt of {prompt_tokens} tokens does not fit in a {context_size}-token context")]
    PromptTooLong {
        /// Number of tokens in the tokenized prompt.
        prompt_tokens: usize,
        /// The configured context window.
        context_size: usize,
    },
    /// Generation failed for another reason.
    #[error("local GGUF generation failed: {0}")]
    Generation(String),
    /// The worker thread is no longer running.
    #[error("local GGUF worker stopped unexpectedly")]
    WorkerStopped,
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStop {
    /// A stop token or end-of-generation token was produced.
    StopToken,
    /// `max_tokens` was reached.
    MaxTokens,
    /// Generation was cancelled.
    Cancelled,
}

/// A completed generation.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The generated prediction text.
    pub text: String,
    /// Monotonically increasing id of the request that produced this result.
    /// Newer requests always have larger ids, so a caller can drop stale results.
    pub generation: u64,
    /// Why generation stopped.
    pub stop: CompletionStop,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
}

/// A cancellation signal shared between a caller and the inference worker.
///
/// Cancellation is observed between decode steps, so an in-flight request stops
/// promptly without discarding work the worker has not started on.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a fresh, uncancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Safe to call from any thread.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Commands sent to the worker thread.
// The non-`reply` fields of `Complete` are only read by the metal worker.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) enum Command {
    /// Load `model_path`, replacing any currently loaded model.
    Load {
        model_path: PathBuf,
        context_size: u32,
        reply: Sender<Result<(), LocalGgufError>>,
    },
    /// Run a single fill-in-the-middle completion against the loaded model.
    Complete {
        prompt: String,
        stop_tokens: Vec<String>,
        max_tokens: u32,
        cancellation: CancellationToken,
        generation: u64,
        reply: Sender<Result<Completion, LocalGgufError>>,
    },
    /// Drop the loaded model and release its memory.
    Unload,
    /// Stop the worker thread.
    Shutdown,
}

/// Sends a worker reply, tolerating a caller that has already given up.
pub(crate) fn send_reply<T>(sender: Sender<T>, value: T) {
    if sender.send(value).is_err() {
        log::debug!("local GGUF: caller dropped before the worker replied");
    }
}

/// A handle to the single-flight on-device inference worker.
///
/// The engine owns exactly one worker thread, so blocking inference never runs
/// on the caller's executor. It is cheap to share behind an [`Arc`].
pub struct LocalGgufEngine {
    commands: Sender<Command>,
    next_generation: AtomicU64,
    shutdown: CancellationToken,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Default for LocalGgufEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalGgufEngine {
    /// Starts the worker thread.
    ///
    /// This does not load a model or touch the GPU; that happens on the first
    /// [`LocalGgufEngine::load`].
    pub fn new() -> Self {
        let (commands, receiver) = flume::unbounded();
        let shutdown = CancellationToken::new();
        let worker = engine::spawn_worker(receiver, shutdown.clone());
        Self {
            commands,
            next_generation: AtomicU64::new(0),
            shutdown,
            worker: Mutex::new(worker),
        }
    }

    /// Loads `model_path`, reusing a resident model with the same path and context size.
    ///
    /// Call [`LocalGgufEngine::unload`] first if the file was replaced at the same path.
    pub async fn load(&self, model_path: PathBuf, context_size: u32) -> Result<(), LocalGgufError> {
        if self.shutdown.is_cancelled() {
            return Err(LocalGgufError::WorkerStopped);
        }
        let (reply, response) = flume::bounded(1);
        self.commands
            .send(Command::Load {
                model_path,
                context_size,
                reply,
            })
            .map_err(|_| LocalGgufError::WorkerStopped)?;
        response
            .recv_async()
            .await
            .map_err(|_| LocalGgufError::WorkerStopped)?
    }

    /// Runs a fill-in-the-middle completion against the loaded model.
    ///
    /// `prompt` should already contain the model's fill-in-the-middle markers;
    /// they are parsed as special tokens rather than raw text. `stop_tokens` are
    /// likewise resolved to special-token ids. The returned [`Completion`] carries
    /// a `generation` id so callers can ignore results superseded by a newer
    /// request.
    pub async fn complete(
        &self,
        prompt: String,
        stop_tokens: Vec<String>,
        max_tokens: u32,
        cancellation: CancellationToken,
    ) -> Result<Completion, LocalGgufError> {
        if self.shutdown.is_cancelled() {
            return Err(LocalGgufError::WorkerStopped);
        }
        let _cancel_on_drop = CancelOnDrop(cancellation.clone());
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let (reply, response) = flume::bounded(1);
        self.commands
            .send(Command::Complete {
                prompt,
                stop_tokens,
                max_tokens,
                cancellation,
                generation,
                reply,
            })
            .map_err(|_| LocalGgufError::WorkerStopped)?;
        response
            .recv_async()
            .await
            .map_err(|_| LocalGgufError::WorkerStopped)?
    }

    /// Drops the loaded model and frees its memory.
    ///
    /// Best effort: if the worker is mid-request, the unload takes effect after
    /// that request finishes or is cancelled.
    pub fn unload(&self) {
        if self.shutdown.is_cancelled() {
            return;
        }
        if self.commands.send(Command::Unload).is_err() {
            log::warn!("local GGUF: worker is no longer running; unload ignored");
        }
    }

    /// Stops inference and joins the worker, even when other callers retain a handle.
    pub fn shutdown(&self) {
        self.shutdown.cancel();
        let mut worker = self.worker.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(worker_handle) = worker.take() else {
            return;
        };
        if self.commands.send(Command::Shutdown).is_err() {
            log::debug!("local GGUF: worker already stopped before shutdown");
        }
        if worker_handle.join().is_err() {
            log::error!("local GGUF worker thread panicked");
        }
    }
}

impl Drop for LocalGgufEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_qwen_fim_prompt_with_special_markers() {
        let prompt = build_qwen_fim_prompt("def f():\n", "\nprint(f())\n");
        assert_eq!(
            prompt,
            "<|fim_prefix|>def f():\n<|fim_suffix|>\nprint(f())\n<|fim_middle|>"
        );
    }

    #[test]
    fn cancellation_token_is_shared_between_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn shutdown_joins_the_worker_with_other_handles_alive() {
        let engine = Arc::new(LocalGgufEngine::new());
        let other_handle = engine.clone();
        engine.shutdown();
        engine.shutdown();

        assert!(
            other_handle
                .worker
                .lock()
                .expect("worker lock poisoned")
                .is_none()
        );
        assert!(matches!(
            smol::block_on(other_handle.load(PathBuf::from("unused.gguf"), 512)),
            Err(LocalGgufError::WorkerStopped)
        ));
        assert!(matches!(
            smol::block_on(other_handle.complete(
                String::new(),
                Vec::new(),
                1,
                CancellationToken::new(),
            )),
            Err(LocalGgufError::WorkerStopped)
        ));
    }

    #[test]
    fn stop_tokens_are_special_markers() {
        assert!(QWEN_FIM_STOP_TOKENS.contains(&"<|endoftext|>"));
        for token in QWEN_FIM_STOP_TOKENS {
            assert!(token.starts_with("<|") && token.ends_with("|>"), "{token}");
        }
    }

    #[test]
    fn reports_whether_the_engine_is_compiled_in() {
        // The value depends on the active features and the target; this asserts
        // the accessor is usable at compile time and only reports support on the
        // Apple-silicon target with the `metal` feature enabled.
        assert_eq!(
            is_supported(),
            cfg!(all(
                feature = "metal",
                target_os = "macos",
                target_arch = "aarch64"
            ))
        );
    }
}
