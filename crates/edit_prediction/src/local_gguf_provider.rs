//! Transport glue for the in-process `local_gguf` edit prediction provider.
//!
//! Prediction providers build their model-specific prompts and hand them here. This module
//! owns the process-wide [`LocalGgufEngine`], which loads the configured model
//! on demand and keeps it resident between predictions. It never touches the
//! network, an API key, or a cloud endpoint.

use anyhow::{Context as _, Result, anyhow};
use language::{EditPredictionPromptFormat, language_settings::LocalGgufEditPredictionSettings};
use local_gguf::{CancellationToken, CompletionStop, DEFAULT_CONTEXT_SIZE, LocalGgufEngine};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

static ENGINE: OnceLock<Mutex<Option<Arc<LocalGgufEngine>>>> = OnceLock::new();

fn engine_slot() -> &'static Mutex<Option<Arc<LocalGgufEngine>>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

fn shared_engine() -> Arc<LocalGgufEngine> {
    let mut slot = engine_slot().lock().unwrap_or_else(PoisonError::into_inner);
    slot.get_or_insert_with(|| Arc::new(LocalGgufEngine::new()))
        .clone()
}

pub(crate) fn shutdown() {
    let engine = engine_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if let Some(engine) = engine {
        // Retain the stopped handle so a late request cannot create another
        // Metal context after the quit hook has joined the worker.
        engine.shutdown();
    }
}

pub(crate) fn unload() {
    let engine = engine_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if let Some(engine) = engine {
        engine.unload();
    }
}

/// Runs a completion on the on-device engine.
///
/// Returns `Ok(None)` when the request was cancelled before producing any text,
/// so callers can drop it without reporting an empty prediction.
pub(crate) async fn complete(
    settings: &LocalGgufEditPredictionSettings,
    prompt: String,
    stop_tokens: Vec<String>,
    cancellation: CancellationToken,
) -> Result<Option<(String, String)>> {
    if !local_gguf::is_supported() {
        return Err(anyhow!(
            "on-device GGUF inference is not available in this build"
        ));
    }

    let Some(model_path) = settings.model_path.as_deref() else {
        return Err(anyhow!(
            "the Local GGUF provider is selected but no model is configured; \
             set `edit_predictions.local_gguf.model_path` to a GGUF file"
        ));
    };

    let engine = shared_engine();
    let prompt_format = crate::local_gguf_models::prompt_format_for_model_path(Some(model_path));
    let is_next_edit = matches!(
        prompt_format,
        EditPredictionPromptFormat::Sweep | EditPredictionPromptFormat::Zeta(_)
    );
    let context_size = if is_next_edit {
        8192
    } else {
        DEFAULT_CONTEXT_SIZE
    };
    engine
        .load(model_path.to_path_buf(), context_size)
        .await
        .with_context(|| {
            format!(
                "failed to load the local GGUF model at {}",
                model_path.display()
            )
        })?;

    let completion = engine
        .complete(
            prompt,
            stop_tokens,
            settings.max_output_tokens,
            cancellation,
        )
        .await?;

    if completion.stop == CompletionStop::Cancelled {
        return Ok(None);
    }
    // A partial rewrite would delete the ungenerated tail of the editable region.
    if is_next_edit && completion.stop == CompletionStop::MaxTokens {
        log::debug!("local gguf: discarding a truncated next-edit prediction");
        return Ok(None);
    }

    log::debug!(
        "local gguf: generated {} tokens from a {}-token prompt",
        completion.generated_tokens,
        completion.prompt_tokens
    );

    Ok(Some((
        completion.text,
        format!("local-gguf-{}", completion.generation),
    )))
}

/// Runs one bounded completion to check that the configured
/// model actually loads and generates on this machine.
///
/// The normal prediction path only logs load and Metal failures, so a user can
/// see "Model installed" while every suggestion silently fails. This runs a
/// single small completion and returns the generated text, or the error, so the
/// settings page can report the real state inline. The output cap is clamped so a
/// self-test can never run long.
pub(crate) async fn self_test(settings: &LocalGgufEditPredictionSettings) -> Result<String> {
    const SELF_TEST_MAX_TOKENS: u32 = 16;

    let settings = LocalGgufEditPredictionSettings {
        model_path: settings.model_path.clone(),
        max_output_tokens: settings.max_output_tokens.clamp(1, SELF_TEST_MAX_TOKENS),
        prediction_debounce: settings.prediction_debounce,
    };

    let format =
        crate::local_gguf_models::prompt_format_for_model_path(settings.model_path.as_deref());
    let (prompt, stop_tokens) = match format {
        EditPredictionPromptFormat::Sweep => (
            crate::sweep_prompt::build_prompt(&crate::sweep_prompt::SweepPromptInput {
                file_path: std::path::Path::new("test.py").into(),
                original_window: "def add(a, b):\n    pass\n".into(),
                current_window: "def add(a, b):\n    return a\n".into(),
                recent_changes: Vec::new(),
                related_files: Vec::new(),
            }),
            &["<|file_sep|>", "</s>"][..],
        ),
        EditPredictionPromptFormat::Zeta(_) => (
            "<[fim-suffix]>\n<[fim-prefix]><filename>test.py\n<|marker_1|>\ndef add(a, b):\n    return a<|user_cursor|>\n<|marker_2|>\n<[fim-middle]>".into(),
            zeta_prompt::stop_tokens_for_format(zeta_prompt::ZetaFormat::V0318SeedMultiRegions),
        ),
        _ => (
            local_gguf::build_qwen_fim_prompt(
                "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    ",
                "\n\nprint(add(1, 2))\n",
            ),
            local_gguf::QWEN_FIM_STOP_TOKENS.as_slice(),
        ),
    };
    let stop_tokens = stop_tokens.iter().map(|token| token.to_string()).collect();

    match complete(&settings, prompt, stop_tokens, CancellationToken::new()).await? {
        Some((text, _request_id)) => Ok(text),
        None => Ok(String::new()),
    }
}

/// A test-only completion function used in place of the real engine.
///
/// It receives the model prompt, the stop tokens and the request's cancellation
/// token, so tests can observe prompts and simulate cancellation.
#[cfg(test)]
pub(crate) type TestLocalGgufCompletionFn = Arc<
    dyn Fn(String, Vec<String>, CancellationToken) -> TestLocalGgufCompletionFuture + Send + Sync,
>;

/// The future returned by a [`TestLocalGgufCompletionFn`].
#[cfg(test)]
pub(crate) type TestLocalGgufCompletionFuture =
    futures::future::BoxFuture<'static, Result<Option<(String, String)>>>;

/// A test-only override that replaces the on-device engine with a fake
/// completion.
///
/// Editor-wiring tests install this as a GPUI global (which is scoped to a
/// single [`gpui::App`]) so the local prediction path can run end-to-end without a
/// model file or a GPU.
#[cfg(test)]
pub(crate) struct TestLocalGgufCompletion(pub TestLocalGgufCompletionFn);

#[cfg(test)]
impl gpui::Global for TestLocalGgufCompletion {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unload_without_a_loaded_model_is_a_no_op() {
        // No engine has been created in this process, so this must not spawn a
        // worker or panic.
        unload();
    }

    #[test]
    fn self_test_without_a_configured_model_reports_an_error() {
        let settings = LocalGgufEditPredictionSettings {
            model_path: None,
            max_output_tokens: 16,
            prediction_debounce: Default::default(),
        };

        // No model is configured, so this must fail before any engine or GPU work
        // is attempted (which is what makes it safe to run without a model file).
        let result = futures::executor::block_on(self_test(&settings));
        assert!(result.is_err(), "expected an error, got {result:?}");
    }
}
