use crate::{
    Copy, CopyAndTrim, CopyPermalinkToLine, Cut, DisplayPoint, DisplaySnapshot, Editor,
    EvaluateSelectedText, FindAllReferences, GoToDeclaration, GoToDefinition, GoToImplementation,
    GoToTypeDefinition, Paste, Rename, RevealInFileManager, RunToCursor, SelectMode,
    SelectionEffects, SelectionExt, ToDisplayPoint, ToggleCodeActions,
    actions::{Format, FormatSelections},
    selections_collection::SelectionsCollection,
};
use collections::HashMap;
use gpui::prelude::FluentBuilder;
use gpui::{Context, DismissEvent, Entity, Focusable as _, Pixels, Point, Subscription, Window};
use multi_buffer::AnchorRangeExt as _;
use project::DisableAiSettings;
use std::ops::Range;
use text::PointUtf16;
use workspace::OpenInTerminal;
use zed_actions::agent::AddSelectionToThread;
use zed_actions::preview::{
    markdown::OpenPreview as OpenMarkdownPreview, svg::OpenPreview as OpenSvgPreview,
};

#[derive(Debug)]
pub enum MenuPosition {
    /// When the editor is scrolled, the context menu stays on the exact
    /// same position on the screen, never disappearing.
    PinnedToScreen(Point<Pixels>),
    /// When the editor is scrolled, the context menu follows the position it is associated with.
    /// Disappears when the position is no longer visible.
    PinnedToEditor {
        source: multi_buffer::Anchor,
        offset: Point<Pixels>,
    },
}

pub struct MouseContextMenu {
    pub(crate) position: MenuPosition,
    pub(crate) context_menu: Entity<ui::ContextMenu>,
    _dismiss_subscription: Subscription,
    _cursor_move_subscription: Subscription,
}

impl std::fmt::Debug for MouseContextMenu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MouseContextMenu")
            .field("position", &self.position)
            .field("context_menu", &self.context_menu)
            .finish()
    }
}

impl MouseContextMenu {
    pub(crate) fn pinned_to_editor(
        editor: &mut Editor,
        source: multi_buffer::Anchor,
        position: Point<Pixels>,
        context_menu: Entity<ui::ContextMenu>,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> Option<Self> {
        let editor_snapshot = editor.snapshot(window, cx);
        let content_origin = editor.last_bounds?.origin
            + Point {
                x: editor.gutter_dimensions.width,
                y: Pixels::ZERO,
            };
        let source_position = editor.to_pixel_point(source, &editor_snapshot, window, cx)?;
        let menu_position = MenuPosition::PinnedToEditor {
            source,
            offset: position - (source_position + content_origin),
        };
        Some(MouseContextMenu::new(
            editor,
            menu_position,
            context_menu,
            window,
            cx,
        ))
    }

    pub(crate) fn new(
        editor: &Editor,
        position: MenuPosition,
        context_menu: Entity<ui::ContextMenu>,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> Self {
        let context_menu_focus = context_menu.focus_handle(cx);

        // Since `ContextMenu` is rendered in a deferred fashion its focus
        // handle is not linked to the Editor's until after the deferred draw
        // callback runs.
        // We need to wait for that to happen before focusing it, so that
        // calling `contains_focused` on the editor's focus handle returns
        // `true` when the `ContextMenu` is focused.
        let focus_handle = context_menu_focus.clone();
        cx.on_next_frame(window, move |_, window, cx| {
            cx.on_next_frame(window, move |_, window, cx| {
                window.focus(&focus_handle, cx);
            });
        });

        let _dismiss_subscription = cx.subscribe_in(&context_menu, window, {
            let context_menu_focus = context_menu_focus.clone();
            move |editor, _, _event: &DismissEvent, window, cx| {
                editor.mouse_context_menu.take();
                if context_menu_focus.contains_focused(window, cx) {
                    window.focus(&editor.focus_handle(cx), cx);
                }
            }
        });

        let selection_init = *editor.selections.newest_anchor();

        let _cursor_move_subscription = cx.subscribe_in(
            &cx.entity(),
            window,
            move |editor, _, event: &crate::EditorEvent, window, cx| {
                let crate::EditorEvent::SelectionsChanged { local: true } = event else {
                    return;
                };
                let display_snapshot = &editor
                    .display_map
                    .update(cx, |display_map, cx| display_map.snapshot(cx));
                let selection_init_range = selection_init.display_range(display_snapshot);
                let selection_now_range = editor
                    .selections
                    .newest_anchor()
                    .display_range(display_snapshot);
                if selection_now_range == selection_init_range {
                    return;
                }
                editor.mouse_context_menu.take();
                if context_menu_focus.contains_focused(window, cx) {
                    window.focus(&editor.focus_handle(cx), cx);
                }
            },
        );

        Self {
            position,
            context_menu,
            _dismiss_subscription,
            _cursor_move_subscription,
        }
    }
}

fn display_ranges<'a>(
    display_map: &'a DisplaySnapshot,
    selections: &'a SelectionsCollection,
) -> impl Iterator<Item = Range<DisplayPoint>> + 'a {
    let pending = selections.pending_anchor();
    selections
        .disjoint_anchors()
        .iter()
        .chain(pending)
        .map(move |s| s.start.to_display_point(display_map)..s.end.to_display_point(display_map))
}

pub fn deploy_context_menu(
    editor: &mut Editor,
    position: Option<Point<Pixels>>,
    point: DisplayPoint,
    window: &mut Window,
    cx: &mut Context<Editor>,
) {
    if !editor.is_focused(window) {
        window.focus(&editor.focus_handle(cx), cx);
    }

    let display_map = editor.display_snapshot(cx);
    let source_anchor = display_map.display_point_to_anchor(point, text::Bias::Right);
    let context_menu = if let Some(custom) = editor.custom_context_menu.take() {
        let menu = custom(editor, point, window, cx);
        editor.custom_context_menu = Some(custom);
        let Some(menu) = menu else {
            return;
        };
        menu
    } else {
        // Don't show context menu for inline editors (only applies to default menu)
        if !editor.mode().is_full() {
            return;
        }

        // Don't show the context menu if there isn't a project associated with this editor
        let Some(project) = editor.project.clone() else {
            return;
        };

        let snapshot = editor.snapshot(window, cx);
        let display_map = editor.display_snapshot(cx);
        let buffer = snapshot.buffer_snapshot();
        let anchor = buffer.anchor_before(point.to_point(&display_map));
        if !display_ranges(&display_map, &editor.selections).any(|r| r.contains(&point)) {
            // Move the cursor to the clicked location so that dispatched actions make sense
            editor.change_selections(SelectionEffects::no_scroll(), window, cx, |s| {
                s.clear_disjoint();
                s.set_pending_anchor_range(anchor..anchor, SelectMode::Character);
            });
        }

        let focus = window.focused(cx);
        let has_reveal_target = editor.target_file(cx).is_some();
        let has_selections = editor
            .selections
            .all::<PointUtf16>(&display_map)
            .into_iter()
            .any(|s| !s.is_empty());
        let has_git_repo =
            buffer
                .anchor_to_buffer_anchor(anchor)
                .is_some_and(|(buffer_anchor, _)| {
                    project
                        .read(cx)
                        .git_store()
                        .read(cx)
                        .repository_and_path_for_buffer_id(buffer_anchor.buffer_id, cx)
                        .is_some()
                });

        let selected_ranges = editor
            .selections
            .disjoint_anchor_ranges()
            .filter(|range| range.start != range.end)
            .collect::<Vec<_>>();
        let has_selection = !selected_ranges.is_empty();
        let (can_stage, can_unstage) = if has_git_repo {
            let target_ranges = if has_selection {
                selected_ranges
            } else {
                // With no selection, act on the line that was right-clicked.
                let clicked_point = point.to_point(&display_map);
                let line_start = language::Point::new(clicked_point.row, 0);
                let line_end =
                    language::Point::new(clicked_point.row + 1, 0).min(buffer.max_point());
                if line_start == line_end {
                    Vec::new()
                } else {
                    vec![buffer.anchor_before(line_start)..buffer.anchor_after(line_end)]
                }
            };

            let mut ranges_by_buffer =
                HashMap::<text::BufferId, Vec<Range<text::Anchor>>>::default();
            for range in &target_ranges {
                for (buffer_snapshot, buffer_range, _) in
                    buffer.range_to_buffer_ranges(range.to_offset(buffer))
                {
                    ranges_by_buffer
                        .entry(buffer_snapshot.remote_id())
                        .or_default()
                        .push(
                            buffer_snapshot.anchor_before(buffer_range.start.0)
                                ..buffer_snapshot.anchor_after(buffer_range.end.0),
                        );
                }
            }

            let multibuffer = editor.buffer().read(cx);
            let mut stage = false;
            let mut unstage = false;
            for (buffer_id, buffer_ranges) in ranges_by_buffer {
                let Some(diff) = multibuffer.diff_for(buffer_id) else {
                    continue;
                };
                let Some(operations) = diff.read(cx).operations() else {
                    continue;
                };
                let Some(buffer_entity) = multibuffer.buffer(buffer_id) else {
                    continue;
                };

                stage |= operations.can_stage_selected_ranges(
                    diff.clone(),
                    Some(buffer_entity.clone()),
                    &buffer_ranges,
                    cx,
                );
                unstage |= operations.can_unstage_selected_ranges(
                    diff,
                    Some(buffer_entity),
                    &buffer_ranges,
                    cx,
                );

                if stage && unstage {
                    break;
                }
            }
            (stage, unstage)
        } else {
            (false, false)
        };

        let evaluate_selection = window.is_action_available(&EvaluateSelectedText, cx);
        let run_to_cursor = window.is_action_available(&RunToCursor, cx);
        let format_selections = window.is_action_available(&FormatSelections, cx);
        let disable_ai = DisableAiSettings::is_ai_disabled_for_buffer(
            editor.buffer.read(cx).as_singleton().as_ref(),
            cx,
        );

        let is_markdown = editor
            .buffer()
            .read(cx)
            .as_singleton()
            .and_then(|buffer| buffer.read(cx).language())
            .is_some_and(|language| language.name().as_ref() == "Markdown");

        let is_svg = editor
            .buffer()
            .read(cx)
            .as_singleton()
            .and_then(|buffer| buffer.read(cx).file())
            .is_some_and(|file| {
                std::path::Path::new(file.file_name(cx))
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("svg"))
            });

        ui::ContextMenu::build(window, cx, |menu, _window, _cx| {
            let builder = menu
                .on_blur_subscription(Subscription::new(|| {}))
                .when(run_to_cursor, |builder| {
                    builder.action("Run to Cursor", Box::new(RunToCursor))
                })
                .when(evaluate_selection && has_selections, |builder| {
                    builder.action("Evaluate Selection", Box::new(EvaluateSelectedText))
                })
                .when(
                    run_to_cursor || (evaluate_selection && has_selections),
                    |builder| builder.separator(),
                )
                .action("Go to Definition", Box::new(GoToDefinition::default()))
                .action("Go to Declaration", Box::new(GoToDeclaration::default()))
                .action(
                    "Go to Type Definition",
                    Box::new(GoToTypeDefinition::default()),
                )
                .action(
                    "Go to Implementation",
                    Box::new(GoToImplementation::default()),
                )
                .action(
                    "Find All References",
                    Box::new(FindAllReferences::default()),
                )
                .action(
                    "Show Incoming Calls",
                    Box::new(zed_actions::ShowIncomingCalls),
                )
                .action(
                    "Show Outgoing Calls",
                    Box::new(zed_actions::ShowOutgoingCalls),
                )
                .separator()
                .action("Rename Symbol", Box::new(Rename))
                .action("Format Buffer", Box::new(Format))
                .when(format_selections, |cx| {
                    cx.action("Format Selections", Box::new(FormatSelections))
                })
                .action(
                    "Show Code Actions",
                    Box::new(ToggleCodeActions {
                        deployed_from: None,
                        quick_launch: false,
                    }),
                )
                .when(!disable_ai && has_selections, |this| {
                    this.action("Add to Agent Thread", Box::new(AddSelectionToThread))
                })
                .separator()
                .action("Cut", Box::new(Cut))
                .action("Copy", Box::new(Copy))
                .action("Copy and Trim", Box::new(CopyAndTrim))
                .action("Paste", Box::new(Paste))
                .separator()
                .action_disabled_when(
                    !has_reveal_target,
                    ui::utils::reveal_in_file_manager_label(false),
                    Box::new(RevealInFileManager),
                )
                .when(is_markdown, |builder| {
                    builder.action("Open Markdown Preview", Box::new(OpenMarkdownPreview))
                })
                .when(is_svg, |builder| {
                    builder.action("Open SVG Preview", Box::new(OpenSvgPreview))
                })
                .action_disabled_when(
                    !has_reveal_target,
                    "Open in Terminal",
                    Box::new(OpenInTerminal),
                )
                .action_disabled_when(
                    !has_git_repo,
                    "Copy Permalink to Line",
                    Box::new(CopyPermalinkToLine),
                )
                .when(can_stage || can_unstage, |builder| {
                    builder
                        .separator()
                        .when(can_stage, |builder| {
                            builder.action(
                                if has_selection {
                                    "Stage Selected Ranges"
                                } else {
                                    "Stage Line"
                                },
                                Box::new(git::StageSelectedRanges),
                            )
                        })
                        .when(can_unstage, |builder| {
                            builder.action(
                                if has_selection {
                                    "Unstage Selected Ranges"
                                } else {
                                    "Unstage Line"
                                },
                                Box::new(git::UnstageSelectedRanges),
                            )
                        })
                })
                .action_disabled_when(
                    !has_git_repo,
                    "View File History",
                    Box::new(git::FileHistory),
                );
            match focus {
                Some(focus) => builder.context(focus),
                None => builder,
            }
        })
    };

    editor.mouse_context_menu = match position {
        Some(position) => MouseContextMenu::pinned_to_editor(
            editor,
            source_anchor,
            position,
            context_menu,
            window,
            cx,
        ),
        None => {
            let character_size = editor.character_dimensions(window, cx);
            let menu_position = MenuPosition::PinnedToEditor {
                source: source_anchor,
                offset: gpui::point(character_size.em_width, character_size.line_height),
            };
            Some(MouseContextMenu::new(
                editor,
                menu_position,
                context_menu,
                window,
                cx,
            ))
        }
    };
    cx.notify();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        editor_tests::init_test,
        test::{
            editor_lsp_test_context::EditorLspTestContext, editor_test_context::EditorTestContext,
        },
    };
    use indoc::indoc;

    #[gpui::test]
    async fn test_mouse_context_menu(cx: &mut gpui::TestAppContext) {
        init_test(cx, |_| {});

        let mut cx = EditorLspTestContext::new_rust(
            lsp::ServerCapabilities {
                hover_provider: Some(lsp::HoverProviderCapability::Simple(true)),
                ..Default::default()
            },
            cx,
        )
        .await;

        cx.set_state(indoc! {"
            fn teˇst() {
                do_work();
            }
        "});
        let point = cx.display_point(indoc! {"
            fn test() {
                do_wˇork();
            }
        "});
        cx.editor(|editor, _window, _app| assert!(editor.mouse_context_menu.is_none()));

        cx.update_editor(|editor, window, cx| {
            deploy_context_menu(editor, Some(Default::default()), point, window, cx);

            // Assert that, even after deploying the editor's mouse context
            // menu, the editor's focus handle still contains the focused
            // element. The pane's tab bar relies on this to determine whether
            // to show the tab bar buttons and there was a small flicker when
            // deploying the mouse context menu that would cause this to not be
            // true, making it so that the buttons would disappear for a couple
            // of frames.
            assert!(editor.focus_handle.contains_focused(window, cx));
        });

        cx.assert_editor_state(indoc! {"
            fn test() {
                do_wˇork();
            }
        "});
        cx.editor(|editor, _window, _app| assert!(editor.mouse_context_menu.is_some()));
    }

    #[gpui::test]
    async fn test_mouse_context_menu_at_pixel_snapped_scroll_position(
        cx: &mut gpui::TestAppContext,
    ) {
        init_test(cx, |_| {});

        let mut cx = EditorTestContext::new(cx).await;
        cx.set_state(&format!("ˇ{}", "aaaaa\n".repeat(100)));
        cx.update(|window, _| window.set_scale_factor(1.25));
        cx.update_editor(|editor, _, cx| {
            editor.set_text_style_refinement(gpui::TextStyleRefinement {
                font_size: Some(gpui::px(14.).into()),
                line_height: Some(gpui::relative(1.3)),
                ..Default::default()
            });
            cx.notify();
        });
        cx.run_until_parked();

        cx.update_editor(|editor, window, cx| {
            assert_eq!(window.scale_factor(), 1.25);
            let line_height = editor
                .style(cx)
                .text
                .line_height_in_pixels(window.rem_size());
            assert_eq!(line_height, gpui::px(18.));
            assert!(window.pixel_snap_f64(f64::from(line_height)) / f64::from(line_height) < 1.);
            editor.set_scroll_position(gpui::point(0., 1.), window, cx);
            assert_eq!(editor.snapshot(window, cx).scroll_position().y, 1.);

            deploy_context_menu(
                editor,
                Some(gpui::point(gpui::px(200.), gpui::px(200.))),
                DisplayPoint::new(crate::display_map::DisplayRow(5), 0),
                window,
                cx,
            );
            assert!(editor.mouse_context_menu.is_some());
        });
        cx.run_until_parked();

        assert!(cx.debug_bounds("MENU_ITEM-Copy").is_some());
    }

    #[gpui::test]
    async fn test_mouse_context_menu_git_range_actions(cx: &mut gpui::TestAppContext) {
        init_test(cx, |_| {});

        let mut cx = EditorTestContext::new(cx).await;

        let deploy_on_line = |cx: &mut EditorTestContext, head: &str, index: &str, state: &str| {
            cx.set_head_text(head);
            cx.set_index_text(index);
            cx.set_state(state);
            cx.run_until_parked();

            let point = cx.display_point(state);
            cx.update_editor(|editor, window, cx| {
                deploy_context_menu(editor, Some(Default::default()), point, window, cx);
            });
            cx.run_until_parked();
        };

        // Unstaged change: only staging applies.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\n",
            "one\ntwo\nthree\n",
            "one\nTWˇO\nthree\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Line").is_some());
        assert!(cx.debug_bounds("MENU_ITEM-Unstage Line").is_none());

        // Fully staged change: only unstaging applies.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\n",
            "one\nTWO\nthree\n",
            "one\nTWˇO\nthree\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Line").is_none());
        assert!(cx.debug_bounds("MENU_ITEM-Unstage Line").is_some());

        // Partially staged hunk: clicking the already-staged line offers
        // unstaging.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\n",
            "one\nTWO\nthree\n",
            "one\nTWˇO\nTHREE\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Line").is_none());
        assert!(cx.debug_bounds("MENU_ITEM-Unstage Line").is_some());

        // Partially staged hunk: clicking the unstaged line offers staging.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\n",
            "one\nTWO\nthree\n",
            "one\nTWO\nTHRˇEE\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Line").is_some());
        assert!(cx.debug_bounds("MENU_ITEM-Unstage Line").is_none());

        // Partially staged hunk: selecting both the staged and unstaged lines
        // offers both actions.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\n",
            "one\nTWO\nthree\n",
            "one\n«TWO\nTHREE\nˇ»",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Selected Ranges").is_some());
        assert!(
            cx.debug_bounds("MENU_ITEM-Unstage Selected Ranges")
                .is_some()
        );

        // Mixed selection spanning a staged hunk and an unstaged hunk: both
        // actions apply.
        deploy_on_line(
            &mut cx,
            "one\ntwo\nthree\nfour\nfive\n",
            "one\ntwo\nthree\nFOUR\nfive\n",
            "one\n«TWO\nthree\nFOUR\nˇ»five\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Selected Ranges").is_some());
        assert!(
            cx.debug_bounds("MENU_ITEM-Unstage Selected Ranges")
                .is_some()
        );

        // Unchanged line: neither action applies.
        deploy_on_line(
            &mut cx,
            "one\nTWO\nthree\n",
            "one\nTWO\nthree\n",
            "one\nTWˇO\nthree\n",
        );
        assert!(cx.debug_bounds("MENU_ITEM-Stage Line").is_none());
        assert!(cx.debug_bounds("MENU_ITEM-Unstage Line").is_none());
    }
}
