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

        let completion: Arc<str> = clean_fim_completion(&response_text, &prefix, &suffix).into();
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

fn clean_fim_completion(response: &str, prefix: &str, suffix: &str) -> String {
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

    let mut cut = false;

    if remove_repeated_prefix_lines(&mut result, prefix) {
        cut = true;
    }

    if let Some(start) = repeated_completion_line_start(&result) {
        result.truncate(start);
        cut = true;
    }

    if let Some(start) = repeated_suffix_start(&result, suffix) {
        result.truncate(start);
        cut = true;
    }

    if cut {
        trim_whitespace_at_cut(&mut result, suffix);
    }

    truncate_exact_suffix_overlap(&mut result, suffix);

    result
}

/// How much of a repeated line must match before it is treated as a
/// repetition loop rather than a coincidentally similar line of code.
const MIN_REPEATED_LINE_CHARACTERS: usize = 16;

/// How many lines of the suffix an echo is compared against.
///
/// Overruns reproduce at most the next function or two, so a bounded window
/// keeps the comparison cheap even when the suffix spans hundreds of lines.
const ECHO_SUFFIX_LINE_WINDOW: usize = 24;

/// How much of a line must match before it is treated as a reproduced suffix
/// line rather than a coincidence.
const MIN_ECHO_CHARACTERS: usize = 12;

/// How many leading suffix lines a completion may skip before it starts
/// reproducing them. Models sometimes omit a closing delimiter that they then
/// continue past.
const MAX_ECHO_SUFFIX_SKIPS: usize = 2;

/// Removes lines that exactly copy the last line of the text before the
/// cursor.
///
/// Small models frequently copy the line they were just typing, which would
/// duplicate text already in the buffer. Only exact copies of a substantial
/// line are removed, so two similarly shaped statements are left alone.
fn remove_repeated_prefix_lines(completion: &mut String, prefix: &str) -> bool {
    let Some(prefix_line) = prefix
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
    else {
        return false;
    };
    if prefix_line.chars().count() < MIN_REPEATED_LINE_CHARACTERS {
        return false;
    }

    let mut removed = false;
    loop {
        let ranges = line_ranges(completion);
        let Some(index) = ranges
            .iter()
            .position(|range| completion[range.clone()].trim() == prefix_line)
        else {
            break;
        };

        let start = ranges[index].start;
        let end = ranges
            .get(index + 1)
            .map_or(completion.len(), |next| next.start);
        completion.replace_range(start..end, "");
        removed = true;
    }

    removed
}

/// Returns the byte offset of the first line that repeats the non-blank line
/// before it.
fn repeated_completion_line_start(completion: &str) -> Option<usize> {
    let mut previous_line: Option<&str> = None;

    for line_range in line_ranges(completion) {
        let line = completion[line_range.clone()].trim();
        if line.is_empty() {
            continue;
        }
        if line.chars().count() >= MIN_REPEATED_LINE_CHARACTERS && previous_line == Some(line) {
            return Some(line_range.start);
        }
        previous_line = Some(line);
    }

    None
}

/// Returns the byte offset where `completion` starts reproducing `suffix`.
///
/// A small local model does not always reproduce the suffix verbatim: it may
/// drop a method receiver or be cut off mid-line by the output limit. Comparing
/// whole lines by their longest common substring catches those partial echoes,
/// while [`truncate_exact_suffix_overlap`] only catches verbatim ones.
fn repeated_suffix_start(completion: &str, suffix: &str) -> Option<usize> {
    let completion_lines = line_ranges(completion);
    let suffix_lines = line_ranges(suffix)
        .into_iter()
        .take(ECHO_SUFFIX_LINE_WINDOW)
        .collect::<Vec<_>>();

    for (line_index, line_range) in completion_lines.iter().enumerate() {
        if completion[line_range.clone()].trim().is_empty() {
            continue;
        }
        if line_echoes_suffix(
            completion,
            &completion_lines[line_index..],
            suffix,
            &suffix_lines,
        ) {
            return Some(line_range.start);
        }
    }

    None
}

/// Whether every non-blank line from `completion_lines` onward reproduces the
/// beginning of the suffix, in order.
fn line_echoes_suffix(
    completion: &str,
    completion_lines: &[std::ops::Range<usize>],
    suffix: &str,
    suffix_lines: &[std::ops::Range<usize>],
) -> bool {
    let mut suffix_index = 0;
    let mut matched_before = false;
    let mut skipped_suffix_lines = 0;
    let mut matched_lines = 0;
    let mut total_matched_characters = 0;
    let mut longest_match = 0;

    for completion_line in completion_lines {
        let completion_text = &completion[completion_line.clone()];
        if completion_text.trim().is_empty() {
            continue;
        }

        let mut matched_characters = None;
        while suffix_index < suffix_lines.len() {
            let suffix_text = &suffix[suffix_lines[suffix_index].clone()];
            suffix_index += 1;
            if suffix_text.trim().is_empty() {
                continue;
            }

            if let Some(characters) = echoed_line_length(completion_text, suffix_text) {
                matched_characters = Some(characters);
                break;
            }

            // Only tolerate skipped lines before the echo starts. Once it has
            // started, a missing line means this alignment is a coincidence.
            if matched_before || skipped_suffix_lines >= MAX_ECHO_SUFFIX_SKIPS {
                return false;
            }
            skipped_suffix_lines += 1;
        }

        let Some(matched_characters) = matched_characters else {
            // Running out of suffix lines after matching a prefix of it means
            // the completion reproduced the suffix and then kept going.
            return matched_before;
        };

        matched_before = true;
        matched_lines += 1;
        total_matched_characters += matched_characters;
        longest_match = longest_match.max(matched_characters);
    }

    let matched_entire_suffix = suffix_index >= suffix_lines.len();
    matched_lines > 0
        && (longest_match >= MIN_ECHO_CHARACTERS
            || total_matched_characters >= MIN_ECHO_CHARACTERS
            || (matched_entire_suffix && matched_lines >= 2))
}

/// Returns the number of characters shared by two lines if they are similar
/// enough to be the same reproduced line.
fn echoed_line_length(completion_line: &str, suffix_line: &str) -> Option<usize> {
    let completion_line = completion_line.trim();
    let suffix_line = suffix_line.trim();
    if completion_line.is_empty() || suffix_line.is_empty() {
        return None;
    }
    if completion_line == suffix_line {
        return Some(completion_line.chars().count());
    }

    let shorter = completion_line
        .chars()
        .count()
        .min(suffix_line.chars().count());
    if shorter < MIN_ECHO_CHARACTERS {
        return None;
    }

    let common = longest_common_substring_length(completion_line, suffix_line);
    // Require most of the shorter line to match. Two lines that merely share a
    // common phrase (for example a recurring error-handling statement) should
    // not be treated as a reproduced line.
    if common < MIN_ECHO_CHARACTERS || common * 10 < shorter * 6 {
        return None;
    }

    Some(common)
}

fn longest_common_substring_length(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = vec![0; right.len() + 1];
    let mut current = vec![0; right.len() + 1];
    let mut longest = 0;

    for left_character in &left {
        for (right_index, right_character) in right.iter().enumerate() {
            if left_character == right_character {
                current[right_index + 1] = previous[right_index] + 1;
                longest = longest.max(current[right_index + 1]);
            } else {
                current[right_index + 1] = 0;
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }

    longest
}

/// Removes the whitespace left at the cut when a removed echo or repetition
/// replaces the usual newline that separates it from the text after the cursor.
fn trim_whitespace_at_cut(completion: &mut String, suffix: &str) {
    let trimmed_length = completion.trim_end().len();
    let had_trailing_newline = completion[trimmed_length..].contains('\n');
    completion.truncate(trimmed_length);

    // If the suffix starts with its own newline, that newline already
    // separates the kept text from what follows. Otherwise the newline that
    // ended the last kept line has to be restored.
    if had_trailing_newline && !suffix.starts_with('\n') {
        completion.push('\n');
    }
}

/// Splits `text` into the byte ranges of its lines, excluding each newline.
fn line_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            ranges.push(start..index);
            start = index + 1;
        }
    }
    ranges.push(start..text.len());
    ranges
}

fn truncate_exact_suffix_overlap(completion: &mut String, suffix: &str) {
    // A short overlap such as `}` or `};` can be a legitimate part of the
    // middle, so only trim overlaps that are several bytes long and contain
    // something other than indentation.
    const MIN_OVERLAP_BYTES: usize = 4;

    let max_overlap = completion.len().min(suffix.len());
    for overlap in (MIN_OVERLAP_BYTES..=max_overlap).rev() {
        if !completion.is_char_boundary(completion.len() - overlap)
            || !suffix.is_char_boundary(overlap)
        {
            continue;
        }

        let completion_tail = &completion[completion.len() - overlap..];
        if completion_tail != &suffix[..overlap] {
            continue;
        }

        if completion_tail
            .chars()
            .any(|character| !character.is_whitespace())
        {
            completion.truncate(completion.len() - overlap);
        }
        return;
    }
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

    #[test]
    fn clean_fim_completion_removes_an_echoed_suffix() {
        assert_eq!(
            clean_fim_completion(
                "    if n <= 1:\n        return n\n\nprint(fib(10))\n",
                "",
                "\nprint(fib(10))\n"
            ),
            // The suffix starts with a newline, so the trailing one is not
            // repeated before the existing `print` call.
            "    if n <= 1:\n        return n"
        );

        // A generation cut off mid-suffix still leaves a partial echo.
        assert_eq!(
            clean_fim_completion("    return value;\n    let ", "", "    let y = 2;\n"),
            "    return value;\n"
        );
    }

    #[test]
    fn clean_fim_completion_removes_an_echoed_following_function() {
        let suffix = "\n}\nfunc (a *app) majorIncidentAlertKeyFromGroups(ctx context.Context, signal majorIncidentSignal) string {\n\trelated := uniqueSortedStrings(signal.RelatedTickets)\n\tif len(related) == 0 {\n\t\treturn \"\"\n\t}\n\treturn strings.Join(related, \",\")\n}\n";
        let body = "\n\treturn signal.Confidence == majorIncidentConfidenceHigh || signal.Confidence == majorIncidentConfidenceVeryHigh\n";

        for echoed_tail in [
            // A verbatim echo of the closing brace and the next function.
            "}\n\nfunc majorIncidentAlertKeyFromGroups(ctx context.Context, signal majorIncidentSignal) string {\n\trelated := uniqueSortedStrings(signal.RelatedTickets)\n",
            // The model omitted the closing brace but still reproduced the
            // following function.
            "\nfunc majorIncidentAlertKeyFromGroups(ctx context.Context, signal majorIncidentSignal) string {\n\trelated := uniqueSortedStrings(signal.RelatedTickets)\n",
            // The model dropped the method receiver and was cut off mid-line.
            "}\n\nfunc majorIncidentAlertKeyFromGroups(ctx context.Context, signal major",
        ] {
            assert_eq!(
                clean_fim_completion(&format!("{body}{echoed_tail}"), "", suffix),
                body.trim_end(),
                "tail: {echoed_tail:?}"
            );
        }
    }

    #[test]
    fn clean_fim_completion_keeps_a_separating_newline_when_the_suffix_shares_the_line() {
        assert_eq!(
            clean_fim_completion(
                "\n\tdo_work()\n}\nfunc next() {}\n",
                "",
                "}\nfunc next() {}\n"
            ),
            "\n\tdo_work()\n"
        );
    }

    #[test]
    fn clean_fim_completion_removes_a_repeated_line() {
        let repeated = "    'Fixed a bug in the quest system where the quest system was not working properly.',";
        let completion = format!("\n{repeated}\n{repeated}\n{repeated}\n");

        assert_eq!(
            clean_fim_completion(&completion, "", "\n  ],\n"),
            format!("\n{repeated}")
        );
    }

    #[test]
    fn clean_fim_completion_removes_a_line_repeated_from_the_prefix() {
        let repeated = "    'Fixed a bug in the quest system where the quest system was not working properly.',";
        let prefix = format!("{{\n  items: [\n{repeated}\n");
        let completion = format!("\n{repeated}\n    'A new item.',\n");

        assert_eq!(
            clean_fim_completion(&completion, &prefix, "\n  ],\n"),
            "\n    'A new item.',"
        );
    }

    #[test]
    fn clean_fim_completion_removes_an_echo_that_continues_past_the_suffix() {
        assert_eq!(
            clean_fim_completion(
                "\n    'A new item.',\n  ],\n},\n{\n  title: \"Interface &\n",
                "",
                "  ],\n},\n{\n"
            ),
            "\n    'A new item.',\n"
        );
    }

    #[test]
    fn clean_fim_completion_keeps_only_the_new_array_item() {
        let previous_item = "    'Fixed a bug in the quest system where the quest system was not working properly. You can now start quests again after a while.',";
        let prefix = format!(
            "{{\n  title: \"Interface & Gameplay\",\n  items: [\n    'Controllers are now supported.',\n{previous_item}\n"
        );
        let suffix = "  ],\n},\n{\n";
        let new_item = "    'Added a new feature to the game: \"Quests\". You can now start quests again after a while.',";
        let completion =
            format!("\n{new_item}\n{previous_item}\n  ],\n}},\n{{\n  title: \"Interface &");

        assert_eq!(
            clean_fim_completion(&completion, &prefix, suffix),
            format!("\n{new_item}\n")
        );
    }

    #[test]
    fn clean_fim_completion_keeps_independent_endings() {
        assert_eq!(
            clean_fim_completion("    let x = 1;\n", "", "    \n}\n"),
            "    let x = 1;\n"
        );
        assert_eq!(
            clean_fim_completion("    let x = 1;\n", "", ""),
            "    let x = 1;\n"
        );
        // A lone closing delimiter can legitimately close a nested block.
        assert_eq!(clean_fim_completion("}\n", "", "}\n"), "}\n");
    }
}
