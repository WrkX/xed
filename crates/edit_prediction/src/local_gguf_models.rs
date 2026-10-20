//! Model management for the on-device `local_gguf` edit prediction provider.
//!
//! This module lets a user install a GGUF model without a terminal: pick the
//! recommended model and download it, or import a `.gguf` file they already
//! have. Downloads go to app-managed storage under the platform data directory,
//! are written to a temporary file, verified against a pinned SHA-256, and only
//! then moved into place.
//!
//! The configured model paths live in `edit_predictions.local_gguf.model_path`,
//! so the inference provider ([`crate::local_gguf_provider`]) keeps reading the
//! same setting and reloads when it changes. Nothing here runs at startup and
//! nothing touches the network until the user clicks **Download**.

use anyhow::{Context as _, Result};
use fs::{Fs, RemoveOptions};
use futures::{AsyncReadExt as _, AsyncWriteExt as _, future::Either, io::BufReader};
use gpui::{
    App, Context, Entity, Global, PathPromptOptions, Render, SharedString, Task, TaskExt, Window,
    http_client::HttpClient,
};
use language::language_settings::{LocalGgufEditPredictionSettings, all_language_settings};
use language::{EditPredictionPromptFormat, ZetaVersion};
use sha2::{Digest as _, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use ui::{
    Button, ButtonStyle, Color, ContextMenu, DropdownMenu, DropdownStyle, Label, LabelSize,
    ProgressBar, prelude::*,
};

use crate::local_gguf_provider;

/// A pinned GGUF model artifact.
///
/// The revision is an immutable commit SHA and the checksum is verified after
/// download, so a shipped build always installs the exact bytes that were
/// tested. Never pin a mutable branch such as `main`.
#[derive(Clone, Copy, Debug)]
pub struct LocalGgufModelPin {
    /// Stable identifier used for persisted state.
    pub id: &'static str,
    /// Human-readable model name shown in settings.
    pub display_name: &'static str,
    pub parameters: &'static str,
    pub prompt_format: EditPredictionPromptFormat,
    /// Short tradeoff description shown in the model picker.
    pub summary: &'static str,
    /// Hugging Face repository the artifact comes from.
    pub repo: &'static str,
    /// Immutable revision (commit SHA) the artifact is downloaded from.
    pub revision: &'static str,
    /// File name inside the repository.
    pub file_name: &'static str,
    /// Exact size of the artifact in bytes.
    pub size_bytes: u64,
    /// Lowercase hex SHA-256 of the artifact.
    pub sha256: &'static str,
    /// License of the model weights.
    pub license: &'static str,
    /// Direct download URL, pinned to `revision`.
    pub url: &'static str,
}

impl LocalGgufModelPin {
    /// The file name of the partially downloaded artifact.
    fn partial_file_name(&self) -> String {
        format!("{}.part", self.file_name)
    }
}

/// The recommended on-device model.
///
/// The docs originally pin `Qwen2.5-Coder-1.5B-Base`, but the official Base GGUF
/// repository (`Qwen/Qwen2.5-Coder-1.5B-Base-GGUF`) is not reachable (every
/// request returns HTTP 401) while the sibling `-Instruct` GGUF works on the same
/// host. Packages 2 and 3 were validated against this Instruct Q4_K_M artifact,
/// and it shares the architecture, tokenizer and FIM special tokens with Base, so
/// it is the deliberately pinned default. See `predictions/06-risks-and-spikes.md`
/// (finding 7) and the Package 4 status log.
pub const RECOMMENDED_MODEL: LocalGgufModelPin = LocalGgufModelPin {
    id: "qwen2.5-coder-1.5b-instruct-q4_k_m",
    display_name: "Qwen2.5-Coder 1.5B",
    parameters: "1.5B",
    prompt_format: EditPredictionPromptFormat::Qwen,
    summary: "Simple, fast code completion",
    repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
    revision: "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
    file_name: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    size_bytes: 1_117_320_768,
    sha256: "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
    license: "Apache-2.0",
    url: "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/f86cb2c1fa58255f8052cc32aeede1b7482d4361/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
};

/// The smallest model, for machines where latency matters most.
const SMALL_MODEL: LocalGgufModelPin = LocalGgufModelPin {
    id: "qwen2.5-coder-0.5b-instruct-q4_k_m",
    display_name: "Qwen2.5-Coder 0.5B",
    parameters: "0.5B",
    prompt_format: EditPredictionPromptFormat::Qwen,
    summary: "Extremely lightweight autocomplete",
    repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
    revision: "ebb2015119c907b064c512bf053e945850b5875f",
    file_name: "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
    size_bytes: 491_400_064,
    sha256: "1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32",
    license: "Apache-2.0",
    url: "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/ebb2015119c907b064c512bf053e945850b5875f/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
};

const SWEEP_MODEL: LocalGgufModelPin = LocalGgufModelPin {
    id: "sweep-next-edit-1.5b-q8_0",
    display_name: "Sweep Next-Edit 1.5B",
    parameters: "1.5B",
    prompt_format: EditPredictionPromptFormat::Sweep,
    summary: "Predicting actual code edits",
    repo: "sweepai/sweep-next-edit-1.5B",
    revision: "409016591c6c1a94f545f22328a85a3516118f34",
    file_name: "sweep-next-edit-1.5b.q8_0.v2.gguf",
    size_bytes: 1_537_269_856,
    sha256: "1321ea5e5d7529e60f9770c6a0b3a965f89542d16cf4ae51bab267f6a88150da",
    license: "Apache-2.0",
    url: "https://huggingface.co/sweepai/sweep-next-edit-1.5B/resolve/409016591c6c1a94f545f22328a85a3516118f34/sweep-next-edit-1.5b.q8_0.v2.gguf",
};

const ZETA_MODEL: LocalGgufModelPin = LocalGgufModelPin {
    id: "zeta-2.1-q4_k_m",
    display_name: "Zeta 2.1",
    parameters: "~8B",
    prompt_format: EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2_1),
    summary: "More sophisticated Zed-style predictions",
    repo: "mradermacher/zeta-2.1-GGUF",
    revision: "8c2528af755e2a640232eb25e7991cd812e04b07",
    file_name: "zeta-2.1.Q4_K_M.gguf",
    size_bytes: 5_071_410_080,
    sha256: "f031e71d3a467f23457964f7168b9158189a91231efc28cc8b91e6ea473dd71b",
    license: "Apache-2.0",
    url: "https://huggingface.co/mradermacher/zeta-2.1-GGUF/resolve/8c2528af755e2a640232eb25e7991cd812e04b07/zeta-2.1.Q4_K_M.gguf",
};

/// The models offered in settings, ordered from smallest to largest.
///
/// Every entry pins an immutable revision, an exact byte size and a SHA-256, so a
/// shipped build installs exactly the bytes that were tested. Only models whose
/// GGUF repository is licensed permissively are listed: larger Qwen2.5-Coder sizes
/// whose GGUF repo is not (for example the 3B repo, which declares `qwen-research`)
/// are deliberately left out of the shipped catalog.
pub const MODEL_CATALOG: [LocalGgufModelPin; 4] =
    [SMALL_MODEL, RECOMMENDED_MODEL, SWEEP_MODEL, ZETA_MODEL];

pub fn prompt_format_for_model_path(path: Option<&Path>) -> EditPredictionPromptFormat {
    let Some(path) = path else {
        return EditPredictionPromptFormat::Qwen;
    };

    pin_for_file_name(path)
        .map(|pin| pin.prompt_format)
        .or_else(|| infer_prompt_format_from_file_name(path))
        .unwrap_or(EditPredictionPromptFormat::Qwen)
}

/// Infers the prompt format of an imported model from its file name.
///
/// Models installed from [`MODEL_CATALOG`] are matched by exact file name. Users
/// can also import any GGUF file, so a model whose file name mentions its family
/// (for example `codestral-22b-v0.1.Q4_K_M.gguf`) gets the matching FIM markers
/// instead of being treated as a Qwen model by default.
fn infer_prompt_format_from_file_name(path: &Path) -> Option<EditPredictionPromptFormat> {
    let file_name = path.file_stem()?.to_str()?.to_ascii_lowercase();

    Some(if file_name.contains("sweep-next-edit") {
        EditPredictionPromptFormat::Sweep
    } else if file_name.contains("zeta2.1") || file_name.contains("zeta-2.1") {
        EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2_1)
    } else if file_name.contains("zeta2") || file_name.contains("zeta-2") {
        EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2)
    } else if file_name.contains("codellama") || file_name.contains("code-llama") {
        EditPredictionPromptFormat::CodeLlama
    } else if file_name.contains("starcoder") {
        EditPredictionPromptFormat::StarCoder
    } else if file_name.contains("deepseek-coder") {
        EditPredictionPromptFormat::DeepseekCoder
    } else if file_name.contains("codestral") {
        EditPredictionPromptFormat::Codestral
    } else if file_name.contains("codegemma") {
        EditPredictionPromptFormat::CodeGemma
    } else if file_name.contains("glm") {
        EditPredictionPromptFormat::Glm
    } else if file_name.contains("qwen") {
        EditPredictionPromptFormat::Qwen
    } else {
        return None;
    })
}

/// The output-token budget that suits a prompt format.
///
/// Next-edit models rewrite an entire window of code, so they need far more
/// output than a fill-in-the-middle model, which only completes the gap at the
/// cursor. A truncated next-edit rewrite is discarded by the provider.
///
/// This is the value written when a model is selected, not a hard limit: the
/// `max_output_tokens` setting is honored as-is for every request.
pub(crate) fn max_output_tokens_for_format(format: EditPredictionPromptFormat) -> u32 {
    match format {
        EditPredictionPromptFormat::Sweep | EditPredictionPromptFormat::Zeta(_) => 512,
        _ => 256,
    }
}

/// The catalog entry whose file name matches `path`, if any.
///
/// Used to preselect a model that is already installed, so the picker reflects
/// what the provider is actually using.
fn pin_for_file_name(path: &Path) -> Option<LocalGgufModelPin> {
    let file_name = path.file_name()?.to_str()?;
    MODEL_CATALOG
        .into_iter()
        .find(|pin| pin.file_name == file_name)
}

/// The magic bytes at the start of every GGUF file.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// An in-flight download.
#[derive(Clone, Copy, Debug)]
struct ActiveDownload {
    pin: LocalGgufModelPin,
    received: u64,
    total: u64,
}

/// The state of the "test model" self-check.
#[derive(Clone, Debug)]
enum SelfTestState {
    /// A self-test completion is running.
    Running,
    /// The model loaded and generated; the string is a short preview.
    Passed(SharedString),
    /// The model could not load or generate; the string is the error.
    Failed(SharedString),
}

/// How long a self-test may run before it is reported as a timeout.
///
/// A single warm completion takes well under a second on the M1 target; the
/// generous bound only catches a wedged worker or an enormous model.
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(30);

/// A stateful, UI-facing manager for the on-device model.
///
/// It is registered as a global entity (see [`LocalGgufModelManager::init_global`])
/// so both the settings page and the status-bar button can read its state. It
/// only ever mutates `edit_predictions.local_gguf.model_path`; the decision of
/// how to load that path stays in [`crate::local_gguf_provider`].
pub struct LocalGgufModelManager {
    /// Where downloaded models are stored.
    models_dir: PathBuf,
    /// The catalog entry the Download button will fetch.
    selected_model: LocalGgufModelPin,
    /// Set while a download is running.
    download: Option<ActiveDownload>,
    /// The most recent actionable error, if any.
    error: Option<SharedString>,
    /// Keeps the in-flight download task alive; dropping it cancels the download.
    download_task: Option<Task<()>>,
    /// The state of the most recent model self-test, if one has been run.
    self_test: Option<SelfTestState>,
    /// Keeps the in-flight self-test task alive; dropping it cancels the check.
    self_test_task: Option<Task<()>>,
}

struct GlobalLocalGgufModelManager(Entity<LocalGgufModelManager>);

impl Global for GlobalLocalGgufModelManager {}

impl LocalGgufModelManager {
    /// Registers the manager as a global entity. Called once from
    /// [`crate::init`].
    pub fn init_global(cx: &mut App) {
        // Preselect whatever model is already configured, so the picker reflects
        // the model the provider is actually using.
        let selected_model = configured_model_path(cx)
            .as_deref()
            .and_then(pin_for_file_name)
            .unwrap_or(RECOMMENDED_MODEL);
        let manager = cx.new(|_cx| Self::new(default_models_dir(), selected_model));
        cx.set_global(GlobalLocalGgufModelManager(manager));
    }

    /// Returns the global manager, if one has been registered.
    pub fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalLocalGgufModelManager>()
            .map(|global| global.0.clone())
    }

    fn new(models_dir: PathBuf, selected_model: LocalGgufModelPin) -> Self {
        Self {
            models_dir,
            selected_model,
            download: None,
            error: None,
            download_task: None,
            self_test: None,
            self_test_task: None,
        }
    }

    /// A one-line description of the current model state, for the status-bar
    /// tooltip.
    pub fn summary(&self, cx: &App) -> SharedString {
        if let Some(download) = &self.download {
            return format!("Downloading {}…", download.pin.display_name).into();
        }
        if let Some(error) = &self.error {
            return format!("On-device model problem: {error}").into();
        }
        match configured_model_path(cx) {
            Some(path) if path.is_file() => {
                format!("On-device inference ({})", path.display()).into()
            }
            Some(path) => format!("Model missing: {}", path.display()).into(),
            None => "No local model configured — download one in the Local GGUF settings".into(),
        }
    }

    /// Downloads the selected catalog model, then points the provider at it.
    fn start_download(&mut self, cx: &mut Context<Self>) {
        if self.download.is_some() {
            return;
        }
        if !local_gguf::is_supported() {
            self.error = Some("On-device inference is not available in this build.".into());
            cx.notify();
            return;
        }

        let pin = self.selected_model;
        let destination = self.models_dir.join(pin.file_name);
        self.error = None;
        self.download = Some(ActiveDownload {
            pin,
            received: 0,
            total: pin.size_bytes,
        });

        let http_client = cx.http_client();
        self.download_task = Some(cx.spawn(async move |this, cx| {
            let progress_entity = this.clone();
            let mut progress_cx = cx.clone();
            let result = download_model(
                &http_client,
                pin.url,
                pin.size_bytes,
                pin.sha256,
                &destination,
                move |received, total| {
                    if let Err(error) = progress_entity.update(&mut progress_cx, |this, cx| {
                        if let Some(download) = this.download.as_mut() {
                            download.received = received;
                            download.total = total;
                        }
                        cx.notify();
                    }) {
                        log::debug!("local gguf model manager dropped during download: {error:?}");
                    }
                },
            )
            .await;

            if let Err(error) = this.update(cx, |this, cx| {
                this.download = None;
                match result {
                    Ok(()) => {
                        this.error = None;
                        Self::set_model_path(Some(destination.to_string_lossy().into_owned()), cx);
                    }
                    Err(error) => {
                        this.error = Some(format!("{error:#}").into());
                    }
                }
                cx.notify();
            }) {
                log::debug!("local gguf model manager dropped before download finished: {error:?}");
            }
        }));
    }

    /// Cancels an in-flight download and removes the partial file.
    fn cancel_download(&mut self, cx: &mut Context<Self>) {
        let Some(download) = self.download.take() else {
            return;
        };
        // Dropping the task cancels the download future.
        self.download_task = None;
        self.error = None;

        let partial = self.models_dir.join(download.pin.partial_file_name());
        let fs = <dyn Fs>::global(cx);
        cx.background_spawn(async move {
            fs.remove_file(
                &partial,
                RemoveOptions {
                    recursive: false,
                    ignore_if_not_exists: true,
                },
            )
            .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    /// Opens a file picker and points the provider at the chosen GGUF file.
    fn choose_local_file(&mut self, cx: &mut Context<Self>) {
        let selected = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Choose a GGUF model".into()),
        });

        cx.spawn(async move |this, cx| {
            let path = match selected.await {
                Ok(Ok(Some(paths))) => match paths.into_iter().next() {
                    Some(path) => path,
                    None => return,
                },
                // The user cancelled the picker.
                Ok(Ok(None)) => return,
                Ok(Err(error)) => {
                    log::warn!("failed to open the GGUF file picker: {error}");
                    if let Err(update_error) = this.update(cx, |this, cx| {
                        this.error = Some("Could not open the file picker.".into());
                        cx.notify();
                    }) {
                        log::debug!("local gguf model manager dropped: {update_error:?}");
                    }
                    return;
                }
                Err(error) => {
                    log::warn!("GGUF file picker was cancelled unexpectedly: {error}");
                    return;
                }
            };

            if let Err(error) = validate_gguf_file(&path).await {
                if let Err(update_error) = this.update(cx, |this, cx| {
                    this.error = Some(format!("{error:#}").into());
                    cx.notify();
                }) {
                    log::debug!("local gguf model manager dropped: {update_error:?}");
                }
                return;
            }

            if let Err(error) = this.update(cx, |this, cx| {
                this.error = None;
                Self::set_model_path(Some(path.to_string_lossy().into_owned()), cx);
                cx.notify();
            }) {
                log::debug!("local gguf model manager dropped: {error:?}");
            }
        })
        .detach();
    }

    /// Clears the configured model and, for downloaded models, deletes the file.
    ///
    /// Imported files outside the app-managed directory are left on disk: only
    /// the reference to them is removed.
    fn remove_model(&mut self, cx: &mut Context<Self>) {
        // Release the resident model first so the engine frees its memory and
        // reloads from disk on the next request.
        local_gguf_provider::unload();

        self.download = None;
        self.download_task = None;
        self.error = None;
        self.self_test = None;
        self.self_test_task = None;

        let cached = configured_model_path(cx).filter(|path| path.starts_with(&self.models_dir));
        Self::set_model_path(None, cx);

        if let Some(path) = cached {
            let fs = <dyn Fs>::global(cx);
            cx.background_spawn(async move {
                fs.remove_file(
                    &path,
                    RemoveOptions {
                        recursive: false,
                        ignore_if_not_exists: true,
                    },
                )
                .await
            })
            .detach_and_log_err(cx);
        }

        cx.notify();
    }

    /// Runs one bounded completion to confirm the configured model actually loads
    /// and generates on this machine.
    ///
    /// Load and Metal errors otherwise only reach the log on the prediction path,
    /// so this gives the settings page a way to surface them inline.
    fn start_self_test(&mut self, cx: &mut Context<Self>) {
        if self.download.is_some() || matches!(self.self_test, Some(SelfTestState::Running)) {
            return;
        }

        let settings = configured_settings(cx);
        if settings.model_path.is_none() {
            self.self_test = Some(SelfTestState::Failed(
                "No model is configured to test.".into(),
            ));
            cx.notify();
            return;
        }

        self.self_test = Some(SelfTestState::Running);
        cx.notify();

        self.self_test_task = Some(cx.spawn(async move |this, cx| {
            let self_test = local_gguf_provider::self_test(&settings);
            let timeout = cx.background_executor().timer(SELF_TEST_TIMEOUT);
            let outcome =
                match futures::future::select(Box::pin(self_test), Box::pin(timeout)).await {
                    Either::Left((result, _)) => result.map_err(|error| format!("{error:#}")),
                    Either::Right((_, _)) => Err("the model self-test timed out".to_string()),
                };

            if let Err(error) = this.update(cx, |this, cx| {
                this.self_test = Some(match outcome {
                    Ok(text) => {
                        let preview: String = text.trim().chars().take(60).collect();
                        SelfTestState::Passed(format!("Model self-test passed: {preview:?}").into())
                    }
                    Err(error) => SelfTestState::Failed(error.into()),
                });
                cx.notify();
            }) {
                log::debug!("local gguf model manager dropped during self-test: {error:?}");
            }
        }));
    }

    /// Persists the configured model path.
    fn set_model_path(value: Option<String>, cx: &mut Context<Self>) {
        let fs = <dyn Fs>::global(cx);
        settings::update_settings_file(fs, cx, move |settings, _cx| {
            let local_settings = settings
                .project
                .all_languages
                .edit_predictions
                .get_or_insert_default()
                .local_gguf
                .get_or_insert_default();
            let new_format = value
                .as_deref()
                .map(|path| prompt_format_for_model_path(Some(Path::new(path))));
            let previous_format = local_settings
                .model_path
                .as_deref()
                .map(|path| prompt_format_for_model_path(Some(Path::new(path))));

            // Write the budget that suits the inferred prompt format when the
            // format changes, so an imported next-edit model (which rewrites a
            // whole window) gets its larger budget and a FIM model does not
            // inherit a stale one. A budget the user tuned for the same format
            // is preserved.
            if let Some(format) = new_format {
                if new_format != previous_format || local_settings.max_output_tokens.is_none() {
                    local_settings.max_output_tokens = Some(max_output_tokens_for_format(format));
                }
            }
            local_settings.model_path = value;
        });
    }
}

impl Render for LocalGgufModelManager {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let configured = configured_model_path(cx);
        let supported = local_gguf::is_supported();
        let downloading = self.download.is_some();

        let model_ready = configured.as_ref().is_some_and(|path| path.is_file());

        // The picker chooses what *Download model* fetches. Each entry is a pinned
        // catalog model; the entity is captured so a choice can update the manager.
        let selected = self.selected_model;
        let selected_id = selected.id;
        let manager = cx.entity();
        let model_menu = ContextMenu::build(window, cx, move |mut menu, _window, _cx| {
            for pin in MODEL_CATALOG {
                menu = menu.toggleable_entry(
                    pin.display_name,
                    pin.id == selected_id,
                    IconPosition::Start,
                    None,
                    {
                        let manager = manager.clone();
                        move |_window, cx| {
                            manager.update(cx, |this, cx| {
                                this.selected_model = pin;
                                this.self_test = None;
                                this.error = None;
                                let path = this.models_dir.join(pin.file_name);
                                if path.is_file() {
                                    Self::set_model_path(
                                        Some(path.to_string_lossy().into_owned()),
                                        cx,
                                    );
                                }
                                cx.notify();
                            });
                        }
                    },
                );
            }
            menu
        });

        let mut container = v_flex()
            .id("local-gguf-model-manager")
            .min_w_0()
            .gap_2()
            .pt_4();

        if !supported {
            container = container.child(
                Label::new(
                    "On-device inference is not available on this platform; the model cannot be used here.",
                )
                .size(LabelSize::Small)
                .color(Color::Warning),
            );
        }

        container = container.child(
            v_flex()
                .gap_0p5()
                .child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(Label::new("Model"))
                        .child(
                            DropdownMenu::new(
                                "local-gguf-model-dropdown",
                                selected.display_name,
                                model_menu,
                            )
                            .tab_index(0)
                            .disabled(downloading)
                            .style(DropdownStyle::Outlined),
                        ),
                )
                .child(
                    Label::new(format!(
                        "{} parameters • {} download",
                        selected.parameters,
                        format_bytes(selected.size_bytes)
                    ))
                    .size(LabelSize::Small),
                )
                .child(Label::new(selected.summary).size(LabelSize::Small))
                .child(
                    Label::new(format!(
                        "{} • {} @ {}",
                        selected.license,
                        selected.repo,
                        selected.revision.get(..12).unwrap_or(selected.revision)
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                ),
        );

        if let Some(download) = &self.download {
            let total = download.total.max(download.received).max(1);
            container = container.child(
                v_flex()
                    .gap_1()
                    .child(ProgressBar::new(
                        "local-gguf-download-progress",
                        download.received as f32,
                        total as f32,
                        cx,
                    ))
                    .child(
                        Label::new(format!(
                            "Downloading {}… {} of {}",
                            download.pin.display_name,
                            format_bytes(download.received),
                            format_bytes(download.total)
                        ))
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    ),
            );
        } else if let Some(error) = &self.error {
            container = container.child(
                Label::new(error.clone())
                    .size(LabelSize::Small)
                    .color(Color::Error),
            );
        } else {
            match &configured {
                Some(path) if path.is_file() => {
                    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                    container = container.child(
                        v_flex()
                            .gap_0p5()
                            .child(
                                Label::new(format!("Model installed — {}", format_bytes(size)))
                                    .size(LabelSize::Small)
                                    .color(Color::Success),
                            )
                            .child(
                                Label::new(path.display().to_string())
                                    .size(LabelSize::Small)
                                    .color(Color::Muted),
                            ),
                    );
                }
                Some(path) => {
                    container = container.child(
                        Label::new(format!("Model file not found: {}", path.display()))
                            .size(LabelSize::Small)
                            .color(Color::Error),
                    );
                }
                None => {
                    container = container.child(
                        Label::new(
                            "No model installed yet. Download a model above, or choose a GGUF file you already have.",
                        )
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    );
                }
            }
        }

        if let Some(self_test) = &self.self_test {
            let (message, color) = match self_test {
                SelfTestState::Running => ("Testing the model…".to_string(), Color::Muted),
                SelfTestState::Passed(message) => (message.to_string(), Color::Success),
                SelfTestState::Failed(message) => (message.to_string(), Color::Error),
            };
            container = container.child(Label::new(message).size(LabelSize::Small).color(color));
        }

        let download_button = Button::new("local-gguf-download", "Download model")
            .style(ButtonStyle::Filled)
            .disabled(!supported || downloading)
            .on_click(cx.listener(|this, _, _, cx| this.start_download(cx)));

        let choose_button = Button::new("local-gguf-choose", "Choose a local .gguf file…")
            .style(ButtonStyle::Outlined)
            .disabled(!supported || downloading)
            .on_click(cx.listener(|this, _, _, cx| this.choose_local_file(cx)));

        let test_button = Button::new("local-gguf-self-test", "Test model")
            .style(ButtonStyle::Outlined)
            .disabled(
                !supported || downloading || matches!(self.self_test, Some(SelfTestState::Running)),
            )
            .on_click(cx.listener(|this, _, _, cx| this.start_self_test(cx)));

        let cancel_button = Button::new("local-gguf-cancel", "Cancel")
            .style(ButtonStyle::Outlined)
            .on_click(cx.listener(|this, _, _, cx| this.cancel_download(cx)));

        let remove_button = Button::new("local-gguf-remove", "Remove model")
            .style(ButtonStyle::Outlined)
            .on_click(cx.listener(|this, _, _, cx| this.remove_model(cx)));

        container.child(
            h_flex()
                .gap_2()
                .when(downloading, |this| this.child(cancel_button))
                .when(!downloading, |this| {
                    this.child(download_button)
                        .child(choose_button)
                        .when(model_ready, |this| this.child(test_button))
                        .when(model_ready || self.error.is_some(), |this| {
                            this.child(remove_button)
                        })
                }),
        )
    }
}

/// Reads the configured model path (already tilde-expanded by settings).
fn configured_model_path(cx: &App) -> Option<PathBuf> {
    all_language_settings(None, cx)
        .edit_predictions
        .local_gguf
        .model_path
        .clone()
}

/// Reads the full Local GGUF settings, for code that needs more than the path.
fn configured_settings(cx: &App) -> LocalGgufEditPredictionSettings {
    all_language_settings(None, cx)
        .edit_predictions
        .local_gguf
        .clone()
}

/// The app-managed directory where downloaded models are stored.
fn default_models_dir() -> PathBuf {
    paths::models_dir().join("gguf")
}

/// The path of the partial file used while downloading `destination`.
fn partial_path(destination: &Path) -> PathBuf {
    let file_name = destination
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    let mut partial = file_name;
    partial.push(".part");
    destination.with_file_name(partial)
}

/// Compares two hex SHA-256 digests, ignoring case.
pub fn sha256_hex_matches(actual: &str, expected: &str) -> bool {
    actual.eq_ignore_ascii_case(expected)
}

/// Formats a byte count using binary units.
pub fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * MIB;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GB", bytes / GIB)
    } else {
        format!("{:.0} MB", bytes / MIB)
    }
}

/// Refuses a download that cannot fit, leaving headroom for the KV cache and
/// Metal compute buffers, which together can exceed the model size.
fn ensure_enough_disk_space(available: u64, download_size: u64) -> Result<()> {
    const HEADROOM_PERCENT: u64 = 20;
    let needed = download_size.saturating_add(download_size / 100 * HEADROOM_PERCENT);
    anyhow::ensure!(
        available >= needed,
        "not enough disk space to download the model: need about {}, only {} available",
        format_bytes(needed),
        format_bytes(available),
    );
    Ok(())
}

/// Checks that `path` is a readable file starting with the GGUF magic bytes.
///
/// This is a fast structural check for imported files; a full checksum or load
/// validation happens when the engine loads the model.
async fn validate_gguf_file(path: &Path) -> Result<()> {
    let metadata = async_fs::metadata(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(metadata.is_file(), "{} is not a file", path.display());
    anyhow::ensure!(
        metadata.len() > GGUF_MAGIC.len() as u64,
        "{} is not a valid GGUF file (too small)",
        path.display()
    );

    let mut file = async_fs::File::open(path)
        .await
        .with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; GGUF_MAGIC.len()];
    file.read_exact(&mut magic)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    anyhow::ensure!(
        magic == GGUF_MAGIC,
        "{} is not a GGUF file (unexpected magic bytes)",
        path.display()
    );
    Ok(())
}

/// Downloads `url` to `destination`, verifying size and SHA-256 before the file
/// is moved into place.
///
/// The download is written to a sibling `.part` file first, so a failure or an
/// interrupted transfer never leaves a half-written model that looks usable.
async fn download_model(
    http_client: &Arc<dyn HttpClient>,
    url: &str,
    expected_size: u64,
    expected_sha256: &str,
    destination: &Path,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<()> {
    let parent = destination
        .parent()
        .context("model destination has no parent directory")?;
    async_fs::create_dir_all(parent)
        .await
        .with_context(|| format!("creating {}", parent.display()))?;

    if let Ok(available) = fs2::available_space(parent) {
        ensure_enough_disk_space(available, expected_size)?;
    }

    let partial = partial_path(destination);

    let result = async {
        let mut file = async_fs::File::create(&partial)
            .await
            .with_context(|| format!("creating {}", partial.display()))?;

        let mut response = http_client
            .get(url, Default::default(), true)
            .await
            .with_context(|| format!("requesting {url}"))?;
        let status = response.status();
        anyhow::ensure!(
            status.is_success(),
            "model download failed with HTTP status {status}"
        );

        let mut body = BufReader::new(response.body_mut());
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 128 * 1024];
        let mut received: u64 = 0;
        loop {
            let read = body
                .read(&mut buffer)
                .await
                .context("reading the model download")?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            file.write_all(&buffer[..read])
                .await
                .context("writing the model download")?;
            received += read as u64;
            on_progress(received, expected_size.max(received));
        }
        file.flush().await.context("flushing the model download")?;

        anyhow::ensure!(
            received == expected_size,
            "model download was truncated: got {received} of {expected_size} bytes"
        );
        let digest = format!("{:x}", hasher.finalize());
        anyhow::ensure!(
            sha256_hex_matches(&digest, expected_sha256),
            "model download checksum mismatch: expected {expected_sha256}, got {digest}"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;

    match result {
        Ok(()) => {
            // Replace any existing file first so the rename is atomic and also
            // works on platforms where renaming over a file fails.
            match async_fs::remove_file(destination).await {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("removing the existing model at {}", destination.display())
                    });
                }
            }
            async_fs::rename(&partial, destination)
                .await
                .with_context(|| {
                    format!("moving the model into place at {}", destination.display())
                })?;
            Ok(())
        }
        Err(error) => {
            if let Err(cleanup_error) = async_fs::remove_file(&partial).await {
                log::warn!(
                    "failed to remove partial model download {}: {cleanup_error}",
                    partial.display()
                );
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::http_client::{AsyncBody, FakeHttpClient, Response};

    fn body_response(status: u16, body: Vec<u8>) -> Response<AsyncBody> {
        Response::builder()
            .status(status)
            .body(AsyncBody::from(body))
            .expect("failed to build test response")
    }

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    /// Builds a fake client that returns `payload` for every request.
    fn client_returning(payload: Vec<u8>) -> Arc<dyn HttpClient> {
        let client = FakeHttpClient::create(move |_req| {
            let response = body_response(200, payload.clone());
            async move { Ok(response) }
        });
        client
    }

    #[test]
    fn sha256_hex_matches_ignores_case() {
        assert!(sha256_hex_matches("ABCDEF", "abcdef"));
        assert!(!sha256_hex_matches("abcdef", "abcde0"));
    }

    #[test]
    fn format_bytes_uses_binary_units() {
        assert_eq!(format_bytes(0), "0 MB");
        assert_eq!(format_bytes(1_117_320_768), "1.04 GB");
    }

    #[test]
    fn ensure_enough_disk_space_leaves_headroom() {
        let size = 1000;
        assert!(ensure_enough_disk_space(1200, size).is_ok());
        assert!(ensure_enough_disk_space(1000, size).is_err());
    }

    #[test]
    fn download_rejects_a_bad_checksum() {
        let dir = temp_dir();
        let destination = dir.path().join("model.gguf");
        let payload = b"GGUF-not-really".to_vec();
        let size = payload.len() as u64;
        let http_client = client_returning(payload);

        let result = futures::executor::block_on(download_model(
            &http_client,
            "https://example.test/model.gguf",
            size,
            "0000000000000000000000000000000000000000000000000000000000000000",
            &destination,
            |_, _| {},
        ));

        assert!(
            result.is_err(),
            "a checksum mismatch must fail the download"
        );
        assert!(!destination.exists(), "no model should be moved into place");
        assert!(
            !partial_path(&destination).exists(),
            "the partial file should be cleaned up after a failure"
        );
    }

    #[test]
    fn download_rejects_a_truncated_response() {
        let dir = temp_dir();
        let destination = dir.path().join("model.gguf");
        let payload = b"GGUF-truncated".to_vec();
        let size = payload.len() as u64;
        let http_client = client_returning(payload);

        // The response body is shorter than the expected size, as if the
        // connection dropped mid-transfer.
        let result = futures::executor::block_on(download_model(
            &http_client,
            "https://example.test/model.gguf",
            size + 1024,
            "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
            &destination,
            |_, _| {},
        ));

        assert!(result.is_err(), "a truncated download must fail");
        assert!(!destination.exists());
        assert!(!partial_path(&destination).exists());
    }

    #[test]
    fn download_writes_and_verifies_a_matching_file() {
        let dir = temp_dir();
        let destination = dir.path().join("model.gguf");
        let payload = b"GGUF-payload".to_vec();
        let expected_sha = {
            let mut hasher = Sha256::new();
            hasher.update(&payload);
            format!("{:x}", hasher.finalize())
        };
        let size = payload.len() as u64;
        let http_client = client_returning(payload.clone());

        let progress = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(u64, u64)>::new()));
        let progress_clone = progress.clone();
        let result = futures::executor::block_on(download_model(
            &http_client,
            "https://example.test/model.gguf",
            size,
            &expected_sha,
            &destination,
            move |received, total| {
                progress_clone
                    .lock()
                    .expect("progress lock poisoned")
                    .push((received, total));
            },
        ));

        assert!(result.is_ok(), "{result:?}");
        assert_eq!(
            std::fs::read(&destination).expect("failed to read the downloaded file"),
            payload,
            "the verified file should be moved into place"
        );
        assert!(!partial_path(&destination).exists());
        assert!(
            !progress.lock().expect("progress lock poisoned").is_empty(),
            "progress should be reported during the download"
        );
    }

    #[test]
    fn validate_gguf_rejects_a_corrupt_file() {
        let dir = temp_dir();
        let path = dir.path().join("not-a-model.gguf");
        std::fs::write(&path, b"NOTGGUF....").expect("failed to write the test file");

        let result = futures::executor::block_on(validate_gguf_file(&path));
        assert!(
            result.is_err(),
            "a file without the GGUF magic must be rejected"
        );
    }

    #[test]
    fn validate_gguf_accepts_a_gguf_header() {
        let dir = temp_dir();
        let path = dir.path().join("model.gguf");
        let mut bytes = GGUF_MAGIC.to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, &bytes).expect("failed to write the test file");

        assert!(futures::executor::block_on(validate_gguf_file(&path)).is_ok());
    }

    #[test]
    fn partial_path_appends_suffix() {
        let path = Path::new("/models/model.gguf");
        assert_eq!(partial_path(path), Path::new("/models/model.gguf.part"));
    }

    #[test]
    fn model_catalog_pins_are_unique_and_well_formed() {
        assert!(
            MODEL_CATALOG
                .iter()
                .any(|pin| pin.id == RECOMMENDED_MODEL.id),
            "the recommended model must be part of the catalog"
        );

        for pin in MODEL_CATALOG {
            assert!(pin.size_bytes > 0, "{} has no size", pin.id);
            assert_eq!(pin.sha256.len(), 64, "{} has a malformed sha256", pin.id);
            assert!(
                pin.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} has a non-hex sha256",
                pin.id
            );
            assert!(
                pin.url.ends_with(pin.file_name),
                "{} url must end with its file name",
                pin.id
            );
            assert!(
                pin.url.contains(pin.revision),
                "{} url must be pinned to its revision",
                pin.id
            );
            assert!(!pin.summary.is_empty(), "{} has no summary", pin.id);
        }

        for (index, pin) in MODEL_CATALOG.iter().enumerate() {
            for other in &MODEL_CATALOG[index + 1..] {
                assert_ne!(pin.id, other.id, "duplicate id {}", pin.id);
                assert_ne!(
                    pin.file_name, other.file_name,
                    "duplicate file {}",
                    pin.file_name
                );
            }
        }
    }

    #[test]
    fn pin_for_file_name_matches_installed_models() {
        for pin in MODEL_CATALOG {
            let path = Path::new("/models/gguf").join(pin.file_name);
            assert_eq!(pin_for_file_name(&path).map(|pin| pin.id), Some(pin.id));
            assert_eq!(prompt_format_for_model_path(Some(&path)), pin.prompt_format);
        }
        assert_eq!(
            prompt_format_for_model_path(None),
            EditPredictionPromptFormat::Qwen
        );
        assert!(pin_for_file_name(Path::new("/models/gguf/custom.gguf")).is_none());
        assert!(pin_for_file_name(Path::new("/models/gguf")).is_none());
    }

    #[test]
    fn prompt_format_is_inferred_from_imported_file_names() {
        for (file_name, expected) in [
            (
                "qwen2.5-coder-7b-instruct-Q4_K_M.gguf",
                EditPredictionPromptFormat::Qwen,
            ),
            (
                "codestral-22b-v0.1.Q4_K_M.gguf",
                EditPredictionPromptFormat::Codestral,
            ),
            (
                "deepseek-coder-6.7b-instruct-Q5_K_M.gguf",
                EditPredictionPromptFormat::DeepseekCoder,
            ),
            (
                "starcoder2-3b-Q8_0.gguf",
                EditPredictionPromptFormat::StarCoder,
            ),
            (
                "CodeLlama-7B-Q4_K_M.gguf",
                EditPredictionPromptFormat::CodeLlama,
            ),
            (
                "codegemma-7b-Q4_K_M.gguf",
                EditPredictionPromptFormat::CodeGemma,
            ),
            ("glm-4-9b-Q4_K_M.gguf", EditPredictionPromptFormat::Glm),
            (
                "zeta-2.1-Q4_K_M.gguf",
                EditPredictionPromptFormat::Zeta(ZetaVersion::Zeta2_1),
            ),
            (
                "my-sweep-next-edit-v1.gguf",
                EditPredictionPromptFormat::Sweep,
            ),
        ] {
            let path = Path::new("/models/gguf").join(file_name);
            assert_eq!(
                prompt_format_for_model_path(Some(&path)),
                expected,
                "{file_name}"
            );
        }

        // Unknown families keep the Qwen default rather than failing to load.
        assert_eq!(
            prompt_format_for_model_path(Some(Path::new("/models/llama-3-8b.gguf"))),
            EditPredictionPromptFormat::Qwen
        );
    }

    #[test]
    fn output_cap_follows_the_inferred_prompt_format() {
        assert_eq!(
            max_output_tokens_for_format(prompt_format_for_model_path(Some(Path::new(
                "/models/imported/my-sweep-next-edit-v1.gguf"
            )))),
            512
        );
        assert_eq!(
            max_output_tokens_for_format(prompt_format_for_model_path(Some(Path::new(
                "/models/imported/zeta-2.1-Q4_K_M.gguf"
            )))),
            512
        );
        assert_eq!(
            max_output_tokens_for_format(prompt_format_for_model_path(Some(Path::new(
                "/models/gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            )))),
            256
        );
    }

    use fs::FakeFs;
    use gpui::{TestAppContext, UpdateGlobal as _};
    use serde_json::json;
    use settings::SettingsStore;

    /// Points the Local GGUF settings at `path`.
    fn set_configured_model_path(path: &Path, cx: &mut TestAppContext) {
        let model_path = path.to_string_lossy().into_owned();
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.all_languages.edit_predictions =
                        Some(settings::EditPredictionSettingsContent {
                            local_gguf: Some(settings::LocalGgufEditPredictionSettingsContent {
                                model_path: Some(model_path),
                                ..Default::default()
                            }),
                            ..Default::default()
                        });
                });
            });
        });
    }

    #[gpui::test]
    async fn self_test_without_a_model_reports_the_problem(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        });

        let manager = cx.new(|_cx| {
            LocalGgufModelManager::new(PathBuf::from("/models/gguf"), RECOMMENDED_MODEL)
        });
        manager.update(cx, |manager, cx| manager.start_self_test(cx));
        cx.run_until_parked();

        assert!(
            matches!(
                manager.read_with(cx, |manager, _| manager.self_test.clone()),
                Some(SelfTestState::Failed(_))
            ),
            "a self-test with no configured model should fail fast without loading anything"
        );
    }

    #[gpui::test]
    async fn remove_model_clears_the_configured_path_and_deletes_the_cached_file(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        });
        let fs = FakeFs::new(cx.executor());
        cx.update(|cx| <dyn fs::Fs>::set_global(fs.clone(), cx));

        let models_dir = PathBuf::from("/models/gguf");
        let model_path = models_dir.join("model.gguf");
        fs.insert_tree(&models_dir, json!({ "model.gguf": "gguf-bytes" }))
            .await;
        set_configured_model_path(&model_path, cx);
        cx.run_until_parked();

        assert_eq!(
            cx.update(|cx| configured_model_path(cx)),
            Some(model_path.clone()),
            "the configured path should be visible before removal"
        );

        let manager =
            cx.new(|_cx| LocalGgufModelManager::new(models_dir.clone(), RECOMMENDED_MODEL));
        manager.update(cx, |manager, cx| manager.remove_model(cx));
        cx.run_until_parked();

        assert_eq!(
            cx.update(|cx| configured_model_path(cx)),
            None,
            "removing the model should clear the configured path"
        );
        assert!(
            !fs.is_file(&model_path).await,
            "the cached model file should be deleted"
        );
    }
}
