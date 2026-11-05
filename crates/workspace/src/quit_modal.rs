use crate::{DismissDecision, ModalView};
use fs::Fs;
use futures::channel::oneshot;
use gpui::{DismissEvent, EventEmitter, FocusHandle, Focusable, Window};
use settings::update_settings_file_with_completion;
use ui::{AlertModal, Checkbox, ElevationIndex, KeyBinding, prelude::*};
use util::ResultExt as _;

pub struct QuitModal {
    focus_handle: FocusHandle,
    dont_ask_again: bool,
    response_tx: Option<oneshot::Sender<bool>>,
}

impl QuitModal {
    pub fn new(response_tx: oneshot::Sender<bool>, cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            dont_ask_again: false,
            response_tx: Some(response_tx),
        }
    }

    fn confirm(&mut self, _: &menu::Confirm, _: &mut Window, cx: &mut Context<Self>) {
        self.finish(true, cx);
    }

    fn cancel(&mut self, _: &menu::Cancel, _: &mut Window, cx: &mut Context<Self>) {
        self.finish(false, cx);
    }

    fn finish(&mut self, should_quit: bool, cx: &mut Context<Self>) {
        let Some(response_tx) = self.response_tx.take() else {
            return;
        };

        if should_quit && self.dont_ask_again {
            let fs = <dyn Fs>::global(cx);
            let settings_written = update_settings_file_with_completion(fs, cx, |settings, _| {
                settings.workspace.confirm_quit = Some(false);
            });
            cx.spawn(async move |this, cx| {
                match settings_written.await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        log::error!("failed to persist `confirm_quit` setting: {error:#}")
                    }
                    Err(error) => {
                        log::error!("failed to persist `confirm_quit` setting: {error}")
                    }
                }
                if response_tx.send(should_quit).is_err() {
                    log::warn!("quit confirmation was closed before the response was delivered");
                }
                this.update(cx, |_, cx| cx.emit(DismissEvent)).log_err();
            })
            .detach();
        } else {
            if response_tx.send(should_quit).is_err() {
                log::warn!("quit confirmation was closed before the response was delivered");
            }
            cx.emit(DismissEvent);
        }
    }
}

impl Focusable for QuitModal {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<DismissEvent> for QuitModal {}

impl ModalView for QuitModal {
    fn fade_out_background(&self) -> bool {
        true
    }

    fn on_before_dismiss(&mut self, _: &mut Window, _: &mut Context<Self>) -> DismissDecision {
        if let Some(response_tx) = self.response_tx.take()
            && response_tx.send(false).is_err()
        {
            log::warn!("quit confirmation was closed before the response was delivered");
        }
        DismissDecision::Dismiss(true)
    }
}

impl Render for QuitModal {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        AlertModal::new("quit-modal")
            .width(rems(26.))
            .key_context("QuitModal")
            .track_focus(&self.focus_handle(cx))
            .on_action(cx.listener(Self::confirm))
            .on_action(cx.listener(Self::cancel))
            .title("Quit Racoon?")
            .child("Are you sure you want to quit?")
            .footer(
                h_flex()
                    .p_3()
                    .gap_2()
                    .justify_between()
                    .child(
                        Checkbox::new(
                            "quit-dont-ask-again",
                            ToggleState::from(self.dont_ask_again),
                        )
                        .label("Don't ask again")
                        .on_click(cx.listener(
                            |this, state: &ToggleState, _, cx| {
                                this.dont_ask_again = state.selected();
                                cx.stop_propagation();
                            },
                        )),
                    )
                    .child(
                        h_flex()
                            .gap_1()
                            .child(
                                Button::new("quit-cancel", "Cancel")
                                    .color(Color::Muted)
                                    .key_binding(
                                        KeyBinding::for_action(&menu::Cancel, cx)
                                            .map(|kb| kb.size(rems_from_px(12_f32))),
                                    )
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.finish(false, cx);
                                        cx.stop_propagation();
                                    })),
                            )
                            .child(
                                Button::new("quit-confirm", "Quit")
                                    .style(ButtonStyle::Filled)
                                    .layer(ElevationIndex::ModalSurface)
                                    .key_binding(
                                        KeyBinding::for_action(&menu::Confirm, cx)
                                            .map(|kb| kb.size(rems_from_px(12_f32))),
                                    )
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.finish(true, cx);
                                        cx.stop_propagation();
                                    })),
                            ),
                    ),
            )
    }
}
