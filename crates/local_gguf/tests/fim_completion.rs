//! End-to-end tests for the on-device engine.
//!
//! The model-backed tests are skipped unless `LOCAL_GGUF_TEST_MODEL` points at a
//! local Qwen2.5-Coder GGUF, so CI without a model still passes. Everything here
//! requires the `metal` feature and never starts a server.

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use local_gguf::{
    CancellationToken, CompletionStop, LocalGgufEngine, LocalGgufError, QWEN_FIM_STOP_TOKENS,
    build_qwen_fim_prompt,
};
use std::{path::PathBuf, time::Instant};

#[test]
fn cancelled_load_is_reused_until_unloaded() -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = test_model_path() else {
        return Ok(());
    };
    let directory = tempfile::tempdir()?;
    let model_link = directory.path().join("model.gguf");
    std::os::unix::fs::symlink(model_path.canonicalize()?, &model_link)?;

    smol::block_on(async {
        let engine = LocalGgufEngine::new();
        let mut loading = Box::pin(engine.load(model_link.clone(), 2048));
        assert!(smol::future::poll_once(&mut loading).await.is_none());
        drop(loading);

        engine
            .complete(
                build_qwen_fim_prompt("def add(a, b):\n    ", "\n"),
                Vec::new(),
                1,
                CancellationToken::new(),
            )
            .await?;
        std::fs::remove_file(&model_link)?;

        engine.load(model_link.clone(), 2048).await?;
        engine.unload();
        assert!(matches!(
            engine.load(model_link, 2048).await,
            Err(LocalGgufError::ModelNotFound(_))
        ));
        Ok(())
    })
}

#[test]
fn shutdown_interrupts_generation_with_other_handles_alive()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = test_model_path() else {
        return Ok(());
    };
    smol::block_on(async {
        let engine = std::sync::Arc::new(LocalGgufEngine::new());
        engine.load(model_path, 2048).await?;
        let other_handle = engine.clone();
        let mut completion = Box::pin(other_handle.complete(
            build_qwen_fim_prompt("def add(a, b):\n    ", "\n"),
            Vec::new(),
            2048,
            CancellationToken::new(),
        ));
        assert!(smol::future::poll_once(&mut completion).await.is_none());

        engine.shutdown();
        assert!(matches!(
            completion.await,
            Ok(local_gguf::Completion {
                stop: CompletionStop::Cancelled,
                ..
            }) | Err(LocalGgufError::WorkerStopped)
        ));
        Ok(())
    })
}

fn test_model_path() -> Option<PathBuf> {
    std::env::var_os("LOCAL_GGUF_TEST_MODEL").map(PathBuf::from)
}

/// `</s>` tokenizes to `</`, `s`, `>` in Qwen-family vocabularies. Those
/// sub-token ids must not be used as stop ids, or generation stops on ordinary
/// code containing `s` or `>`.
#[test]
fn multi_token_stop_sequence_does_not_change_generation() -> Result<(), Box<dyn std::error::Error>>
{
    let Some(model_path) = test_model_path() else {
        eprintln!("LOCAL_GGUF_TEST_MODEL is not set; skipping the stop-sequence test");
        return Ok(());
    };

    smol::block_on(async {
        let engine = LocalGgufEngine::new();
        engine.load(model_path, 2048).await?;

        let prompt = "fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n".to_string();
        let with_stop = engine
            .complete(
                prompt.clone(),
                vec!["</s>".to_string()],
                64,
                CancellationToken::new(),
            )
            .await?;
        let without_stop = engine
            .complete(prompt, Vec::new(), 64, CancellationToken::new())
            .await?;

        assert_eq!(
            with_stop.text, without_stop.text,
            "a multi-token stop sequence must only match its full text"
        );
        Ok(())
    })
}

#[test]
fn missing_model_reports_model_not_found() -> Result<(), Box<dyn std::error::Error>> {
    smol::block_on(async {
        let engine = LocalGgufEngine::new();
        let result = engine
            .load(PathBuf::from("/nonexistent/local-gguf-test.gguf"), 512)
            .await;
        assert!(
            matches!(&result, Err(LocalGgufError::ModelNotFound(_))),
            "expected ModelNotFound, got {result:?}"
        );
        Ok(())
    })
}

#[test]
fn completes_fim_middle_from_a_local_model() -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = test_model_path() else {
        eprintln!("LOCAL_GGUF_TEST_MODEL is not set; skipping the real-model FIM test");
        return Ok(());
    };

    smol::block_on(async {
        let engine = LocalGgufEngine::new();

        let load_start = Instant::now();
        engine.load(model_path, 2048).await?;
        eprintln!("local_gguf: model loaded in {:?}", load_start.elapsed());

        let prompt = build_qwen_fim_prompt("def fib(n):\n", "\nprint(fib(10))\n");
        let stop_tokens = QWEN_FIM_STOP_TOKENS
            .iter()
            .map(|token| token.to_string())
            .collect();

        let start = Instant::now();
        let completion = engine
            .complete(prompt, stop_tokens, 64, CancellationToken::new())
            .await?;
        eprintln!(
            "local_gguf: {} tokens in {:?} (stop={:?}), text:\n{}",
            completion.generated_tokens,
            start.elapsed(),
            completion.stop,
            completion.text
        );

        assert_eq!(completion.generation, 1);
        assert!(completion.prompt_tokens > 0);
        assert!(completion.generated_tokens > 0);
        assert_ne!(completion.stop, CompletionStop::Cancelled);

        let text = completion.text.to_ascii_lowercase();
        assert!(!text.trim().is_empty(), "expected a non-empty insertion");
        assert!(
            text.contains("return") || text.contains("n ==") || text.contains("n=="),
            "implausible Fibonacci insertion: {:?}",
            completion.text
        );
        Ok(())
    })
}

#[test]
fn cancellation_and_supersession_drop_stale_requests() -> Result<(), Box<dyn std::error::Error>> {
    let Some(model_path) = test_model_path() else {
        eprintln!("LOCAL_GGUF_TEST_MODEL is not set; skipping the cancellation test");
        return Ok(());
    };

    smol::block_on(async {
        let engine = LocalGgufEngine::new();
        engine.load(model_path, 2048).await?;

        // A request whose token was already cancelled must not produce an edit.
        let stale_token = CancellationToken::new();
        stale_token.cancel();
        let stale = engine
            .complete(
                build_qwen_fim_prompt("def add(a, b):\n", "\n"),
                Vec::new(),
                64,
                stale_token,
            )
            .await?;
        assert_eq!(stale.stop, CompletionStop::Cancelled);
        assert_eq!(stale.generated_tokens, 0);

        // The next request has a strictly newer generation and does produce text.
        let fresh = engine
            .complete(
                build_qwen_fim_prompt("def add(a, b):\n", "\n"),
                Vec::new(),
                64,
                CancellationToken::new(),
            )
            .await?;
        assert!(fresh.generation > stale.generation);
        assert_ne!(fresh.stop, CompletionStop::Cancelled);
        assert!(fresh.generated_tokens > 0);

        // After unloading, completions report that no model is loaded. The
        // unload is enqueued before the completion, so ordering is deterministic.
        engine.unload();
        let result = engine
            .complete(
                "anything".to_string(),
                Vec::new(),
                8,
                CancellationToken::new(),
            )
            .await;
        assert!(
            matches!(&result, Err(LocalGgufError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
        Ok(())
    })
}
