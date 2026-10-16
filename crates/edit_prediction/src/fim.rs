use crate::{
    EditPredictionId, EditPredictionInputs, EditPredictionModelInput, cursor_excerpt,
    local_gguf_provider,
    open_ai_compatible::{self, load_open_ai_compatible_api_key_if_needed},
    prediction::EditPredictionResult,
};
use anyhow::{Context as _, Result, anyhow};
use gpui::{App, AppContext as _, Entity, Task, http_client};
use language::{
    Anchor, Buffer, BufferSnapshot, EditPredictionPromptFormat, ToOffset, ToPoint as _,
    ZetaVersion,
    language_settings::{
        LocalGgufEditPredictionSettings, OpenAiCompatibleEditPredictionSettings,
        all_language_settings,
    },
};
use local_gguf::CancellationToken;
use settings::EditPredictionProvider;
use std::{path::Path, sync::Arc, time::Instant};
use zeta_prompt::{Zeta2PromptInput, compute_editable_and_context_ranges};

const FIM_CONTEXT_TOKENS: usize = 512;

struct FimRequestOutput {
    request_id: String,
    edits: Vec<(std::ops::Range<Anchor>, Arc<str>)>,
    editable_range: std::ops::Range<Anchor>,
    snapshot: BufferSnapshot,
    inputs: Zeta2PromptInput,
    buffer: Entity<Buffer>,
}

/// Where completion text comes from.
///
/// Remote providers keep using the OpenAI-compatible HTTP transport; the local
/// provider dispatches to the in-process engine instead, so it never builds an
/// HTTP client, loads an API key, or sends source code anywhere.
pub(crate) enum CompletionTransport {
    Remote {
        provider: EditPredictionProvider,
        settings: OpenAiCompatibleEditPredictionSettings,
        api_key: Option<Arc<str>>,
        http_client: Arc<dyn http_client::HttpClient>,
    },
    Local {
        settings: LocalGgufEditPredictionSettings,
    },
    /// A test double for the on-device engine, installed as a GPUI global by
    /// editor-wiring tests that must run without a model file or a GPU.
    #[cfg(test)]
    LocalTest {
        completion: local_gguf_provider::TestLocalGgufCompletionFn,
    },
}

impl CompletionTransport {
    /// Sends the prompt, returning the completion text and a request id.
    ///
    /// `Ok(None)` means the request was cancelled before producing any text.
    pub(crate) async fn send(
        self,
        prompt: String,
        stop_tokens: Vec<String>,
    ) -> Result<Option<(String, String)>> {
        match self {
            CompletionTransport::Remote {
                provider,
                settings,
                api_key,
                http_client,
            } => {
                let (text, request_id) = open_ai_compatible::send_custom_server_request(
                    provider,
                    &settings,
                    prompt,
                    settings.max_output_tokens,
                    stop_tokens,
                    api_key,
                    &http_client,
                )
                .await?;
                Ok(Some((text, request_id)))
            }
            CompletionTransport::Local { settings } => {
                // Cancelling on drop aborts the generation if this request is
                // superseded (its task is dropped) before it completes.
                let cancellation = CancellationToken::new();
                let _cancel_on_drop = CancelOnDrop(cancellation.clone());
                local_gguf_provider::complete(&settings, prompt, stop_tokens, cancellation).await
            }
            #[cfg(test)]
            CompletionTransport::LocalTest { completion } => {
                let cancellation = CancellationToken::new();
                let _cancel_on_drop = CancelOnDrop(cancellation.clone());
                completion(prompt, stop_tokens, cancellation).await
            }
        }
    }
}

/// Resolves the transport for the local provider.
///
/// In tests, a GPUI global can substitute a fake completion so the local prediction
/// path is exercised without a model file or a GPU.
#[cfg(test)]
pub(crate) fn local_gguf_transport(
    settings: LocalGgufEditPredictionSettings,
    cx: &App,
) -> CompletionTransport {
    if let Some(test) = cx.try_global::<local_gguf_provider::TestLocalGgufCompletion>() {
        return CompletionTransport::LocalTest {
            completion: test.0.clone(),
        };
    }
    CompletionTransport::Local { settings }
}

#[cfg(not(test))]
pub(crate) fn local_gguf_transport(
    settings: LocalGgufEditPredictionSettings,
    _cx: &App,
) -> CompletionTransport {
    CompletionTransport::Local { settings }
}

/// Cancels an in-flight generation when the request that owns it is dropped,
/// which is how a superseded or aborted prediction is interrupted.
struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub fn request_prediction(
    EditPredictionModelInput {
        buffer,
        snapshot,
        position,
        events,
        trigger,
        ..
    }: EditPredictionModelInput,
    prompt_format: EditPredictionPromptFormat,
    cx: &mut App,
) -> Task<Result<Option<EditPredictionResult>>> {
    // `all_language_settings` borrows `cx`, so copy the pieces we need before
    // taking a mutable borrow for the API key lookup below.
    let (provider, remote_settings, local_gguf_settings) = {
        let settings = &all_language_settings(None, cx).edit_predictions;
        let remote_settings = match settings.provider {
            EditPredictionProvider::Ollama => settings.ollama.clone(),
            EditPredictionProvider::OpenAiCompatibleApi => settings.open_ai_compatible_api.clone(),
            _ => None,
        };
        (
            settings.provider,
            remote_settings,
            settings.local_gguf.clone(),
        )
    };

    let transport = match provider {
        EditPredictionProvider::Ollama | EditPredictionProvider::OpenAiCompatibleApi => {
            let Some(remote_settings) = remote_settings else {
                return Task::ready(Err(anyhow!("Unsupported edit prediction provider for FIM")));
            };
            CompletionTransport::Remote {
                provider,
                settings: remote_settings,
                api_key: load_open_ai_compatible_api_key_if_needed(provider, cx),
                http_client: cx.http_client(),
            }
        }
        EditPredictionProvider::LocalGguf => local_gguf_transport(local_gguf_settings, cx),
        _ => {
            return Task::ready(Err(anyhow!("Unsupported edit prediction provider for FIM")));
        }
    };

    let full_path: Arc<Path> = snapshot
        .file()
        .map(|file| file.full_path(cx))
        .unwrap_or_else(|| "untitled".into())
        .into();

    let cursor_point = position.to_point(&snapshot);
    let request_start = cx.background_executor().now();

    let result = cx.background_spawn(async move {
        let cursor_offset = cursor_point.to_offset(&snapshot);
        let (excerpt_point_range, excerpt_offset_range, cursor_offset_in_excerpt) =
            cursor_excerpt::compute_cursor_excerpt(&snapshot, cursor_offset);
        let cursor_excerpt: Arc<str> = snapshot
            .text_for_range(excerpt_point_range.clone())
            .collect::<String>()
            .into();
        let syntax_ranges =
            cursor_excerpt::compute_syntax_ranges(&snapshot, cursor_offset, &excerpt_offset_range);
        let (editable_range, _) = compute_editable_and_context_ranges(
            &cursor_excerpt,
            cursor_offset_in_excerpt,
            &syntax_ranges,
            FIM_CONTEXT_TOKENS,
            0,
        );

        let inputs = Zeta2PromptInput {
            events,
            related_files: Some(Vec::new()),
            active_buffer_diagnostics: Vec::new(),
            cursor_offset_in_excerpt: cursor_offset - excerpt_offset_range.start,
            cursor_path: full_path.clone(),
            excerpt_start_row: Some(excerpt_point_range.start.row),
            cursor_excerpt,
            excerpt_ranges: Default::default(),
            syntax_ranges: None,
            in_open_source_repo: false,
            can_collect_data: false,
            repo_url: None,
        };

        let editable_text = &inputs.cursor_excerpt[editable_range.clone()];
        let cursor_in_editable = cursor_offset_in_excerpt.saturating_sub(editable_range.start);
        let prefix = editable_text[..cursor_in_editable].to_string();
        let suffix = editable_text[cursor_in_editable..].to_string();
        let prompt = format_fim_prompt(prompt_format, &prefix, &suffix);
        let stop_tokens = fim_stop_tokens(prompt_format)
            .iter()
            .map(|token| token.to_string())
            .collect();

        let Some((response_text, request_id)) = transport.send(prompt, stop_tokens).await? else {
            // The request was cancelled; there is nothing to show.
            return anyhow::Ok(None);
        };

        let response_received_at = Instant::now();

        log::debug!(
            "fim: completion received ({:.2}s)",
            (response_received_at - request_start).as_secs_f64()
        );

        let completion: Arc<str> = clean_fim_completion(&response_text).into();
        let edits = if completion.is_empty() {
            vec![]
        } else {
            let anchor = snapshot.anchor_after(cursor_offset);
            vec![(anchor..anchor, completion)]
        };

        let editable_range = snapshot.anchor_range_inside(
            (excerpt_offset_range.start + editable_range.start)
                ..(excerpt_offset_range.start + editable_range.end),
        );

        anyhow::Ok(Some(FimRequestOutput {
            request_id,
            edits,
            editable_range,
            snapshot,
            inputs,
            buffer,
        }))
    });

    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        let Some(output) = result.await.context("fim edit prediction failed")? else {
            return anyhow::Ok(None);
        };
        anyhow::Ok(Some(
            EditPredictionResult::new(
                EditPredictionId(output.request_id.into()),
                &output.buffer,
                &output.snapshot,
                output.edits.into(),
                None,
                Some(output.editable_range),
                EditPredictionInputs::V2(output.inputs),
                None,
                trigger,
                cx.background_executor().now() - request_start,
                cx,
            )
            .await,
        ))
    })
}

/// Infers the FIM prompt format from an Ollama/OpenAI-compatible model name.
/// Returns `None` if the model isn't a known FIM-capable model.
pub fn infer_prompt_format(model: &str) -> Option<EditPredictionPromptFormat> {
    let model_base = model.split(':').next().unwrap_or(model);

    Some(match model_base {
        "zeta2" => EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2),
        "zeta2.1" => EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2_1),
        model_base if model_base.to_ascii_lowercase().contains("sweep-next-edit") => {
            EditPredictionPromptFormat::Sweep
        }
        "codellama" | "code-llama" => EditPredictionPromptFormat::CodeLlama,
        "starcoder" | "starcoder2" | "starcoderbase" => EditPredictionPromptFormat::StarCoder,
        "deepseek-coder" | "deepseek-coder-v2" => EditPredictionPromptFormat::DeepseekCoder,
        "qwen2.5-coder" | "qwen-coder" | "qwen" => EditPredictionPromptFormat::Qwen,
        "codegemma" => EditPredictionPromptFormat::CodeGemma,
        "codestral" | "mistral" => EditPredictionPromptFormat::Codestral,
        "glm" | "glm-4" | "glm-4.5" => EditPredictionPromptFormat::Glm,
        _ => {
            return None;
        }
    })
}

fn format_fim_prompt(
    prompt_format: EditPredictionPromptFormat,
    prefix: &str,
    suffix: &str,
) -> String {
    match prompt_format {
        EditPredictionPromptFormat::CodeLlama => {
            format!("<PRE> {prefix} <SUF>{suffix} <MID>")
        }
        EditPredictionPromptFormat::StarCoder => {
            format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>")
        }
        EditPredictionPromptFormat::DeepseekCoder => {
            format!("<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>")
        }
        EditPredictionPromptFormat::Qwen | EditPredictionPromptFormat::CodeGemma => {
            format!("<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>")
        }
        EditPredictionPromptFormat::Codestral => {
            format!("[SUFFIX]{suffix}[PREFIX]{prefix}")
        }
        EditPredictionPromptFormat::Glm => {
            format!("<|code_prefix|>{prefix}<|code_suffix|>{suffix}<|code_middle|>")
        }
        _ => {
            format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>")
        }
    }
}

// OpenAI-compatible servers such as vLLM reject requests with more than 4 stop
// sequences, so each format may only list up to 4 tokens.
fn fim_stop_tokens(prompt_format: EditPredictionPromptFormat) -> &'static [&'static str] {
    match prompt_format {
        EditPredictionPromptFormat::CodeLlama => &["<PRE>", "<SUF>", "<MID>"],
        EditPredictionPromptFormat::DeepseekCoder => {
            &["<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"]
        }
        EditPredictionPromptFormat::Qwen => &[
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ],
        EditPredictionPromptFormat::CodeGemma => &[
            "<|file_separator|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
        ],
        EditPredictionPromptFormat::Codestral => &["[PREFIX]", "[SUFFIX]"],
        EditPredictionPromptFormat::Glm => &[
            "<|endoftext|>",
            "<|code_prefix|>",
            "<|code_suffix|>",
            "<|code_middle|>",
        ],
        _ => &[
            "<|endoftext|>",
            "<fim_prefix>",
            "<fim_middle>",
            "<fim_suffix>",
        ],
    }
}

fn clean_fim_completion(response: &str) -> String {
    let mut result = response.to_string();

    let end_tokens = [
        "<|endoftext|>",
        "<|file_separator|>",
        "<|fim_pad|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<fim_prefix>",
        "<fim_middle>",
        "<fim_suffix>",
        "<PRE>",
        "<SUF>",
        "<MID>",
        "[PREFIX]",
        "[SUFFIX]",
    ];

    for token in &end_tokens {
        if let Some(pos) = result.find(token) {
            result.truncate(pos);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_prompt_format_matches_known_model_families() {
        assert_eq!(
            infer_prompt_format("qwen2.5-coder:3b"),
            Some(EditPredictionPromptFormat::Qwen)
        );
        assert_eq!(
            infer_prompt_format("codellama:7b"),
            Some(EditPredictionPromptFormat::CodeLlama)
        );
        assert_eq!(
            infer_prompt_format("deepseek-coder-v2:16b"),
            Some(EditPredictionPromptFormat::DeepseekCoder)
        );
        assert_eq!(
            infer_prompt_format("starcoder2:3b"),
            Some(EditPredictionPromptFormat::StarCoder)
        );
        assert_eq!(
            infer_prompt_format("codestral:latest"),
            Some(EditPredictionPromptFormat::Codestral)
        );
        assert_eq!(
            infer_prompt_format("glm-4:9b"),
            Some(EditPredictionPromptFormat::Glm)
        );
    }

    #[test]
    fn infer_prompt_format_matches_zeta_and_sweep() {
        assert_eq!(
            infer_prompt_format("zeta2"),
            Some(EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2))
        );
        assert_eq!(
            infer_prompt_format("zeta2.1"),
            Some(EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2_1))
        );
        assert_eq!(
            infer_prompt_format("my-sweep-next-edit-v1"),
            Some(EditPredictionPromptFormat::Sweep)
        );
    }

    const FIM_PROMPT_FORMATS: [EditPredictionPromptFormat; 7] = [
        EditPredictionPromptFormat::CodeLlama,
        EditPredictionPromptFormat::StarCoder,
        EditPredictionPromptFormat::DeepseekCoder,
        EditPredictionPromptFormat::Qwen,
        EditPredictionPromptFormat::CodeGemma,
        EditPredictionPromptFormat::Codestral,
        EditPredictionPromptFormat::Glm,
    ];

    #[test]
    fn fim_stop_tokens_respect_openai_stop_limit() {
        for format in FIM_PROMPT_FORMATS {
            let stop_tokens = fim_stop_tokens(format);
            assert!(!stop_tokens.is_empty(), "{format:?} has no stop tokens");
            assert!(
                stop_tokens.len() <= 4,
                "{format:?} has {} stop tokens, but OpenAI allows at most 4",
                stop_tokens.len()
            );
        }
    }

    #[test]
    fn fim_stop_tokens_match_prompt_markers() {
        let end_of_text_tokens = ["<|endoftext|>", "<|file_separator|>"];
        for format in FIM_PROMPT_FORMATS {
            let prompt = format_fim_prompt(format, "a", "b");
            for token in fim_stop_tokens(format) {
                if end_of_text_tokens.contains(token) {
                    continue;
                }
                assert!(
                    prompt.contains(token),
                    "stop token {token:?} for {format:?} does not appear in prompt {prompt:?}"
                );
            }
        }
    }

    #[test]
    fn infer_prompt_format_returns_none_for_unsupported_models() {
        assert_eq!(infer_prompt_format("llama3:70b"), None);
        assert_eq!(infer_prompt_format("phi3:mini"), None);
        assert_eq!(infer_prompt_format("nomic-embed-text"), None);
    }
}
