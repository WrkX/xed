use crate::{
    Open, OpenMode, PathList, RecentWorkspace, SerializedWorkspaceLocation, Workspace,
    WorkspaceSettings,
    item::{Item, ItemEvent},
    persistence::WorkspaceDb,
};
use git::Clone as GitClone;
use gpui::{
    Action, App, Context, Entity, EventEmitter, FocusHandle, Focusable, FontWeight,
    InteractiveElement, ParentElement, Render, Role, StatefulInteractiveElement, Styled, Task,
    TaskExt, WeakEntity, Window, actions,
};
use menu::{SelectNext, SelectPrevious};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{DefaultOpenBehavior, Settings};
use ui::{ButtonLike, Divider, Vector, VectorName, prelude::*};
use util::ResultExt;
use zed_actions::{OpenOnboarding, OpenRemote, command_palette};

#[derive(PartialEq, Clone, Debug, Deserialize, Serialize, JsonSchema, Action)]
#[action(namespace = welcome)]
#[serde(transparent)]
pub struct OpenRecentProject {
    pub index: usize,
}

actions!(
    zed,
    [
        /// Show the Zed welcome screen
        ShowWelcome
    ]
);

const ACTION_COUNT: usize = 4;

#[derive(IntoElement)]
struct ActionCard {
    label: SharedString,
    icon: IconName,
    action: Box<dyn Action>,
    tab_index: usize,
    focus_handle: FocusHandle,
}

impl ActionCard {
    fn new(
        label: impl Into<SharedString>,
        icon: IconName,
        action: &dyn Action,
        tab_index: usize,
        focus_handle: FocusHandle,
    ) -> Self {
        Self {
            label: label.into(),
            icon,
            action: action.boxed_clone(),
            tab_index,
            focus_handle,
        }
    }
}

impl RenderOnce for ActionCard {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let colors = cx.theme().colors();
        let id = format!("welcome-action-{}", self.tab_index);

        div()
            .id(id)
            .tab_index(self.tab_index as isize)
            .role(Role::Button)
            .aria_label(self.label.clone())
            .w_full()
            .h(rems_from_px(64_f32))
            .rounded_sm()
            .border_1()
            .border_color(colors.border_variant)
            .bg(colors.element_background)
            .text_left()
            .cursor_pointer()
            .hover(|style| style.bg(colors.element_active))
            .focus_visible(|style| style.border_color(colors.border_focused))
            .on_click(move |_, window, cx| {
                self.focus_handle.dispatch_action(&*self.action, window, cx)
            })
            .child(
                v_flex()
                    .size_full()
                    .items_start()
                    .justify_between()
                    .p_2p5()
                    .child(
                        Icon::new(self.icon)
                            .color(Color::Muted)
                            .size(IconSize::Small),
                    )
                    .child(Label::new(self.label).size(LabelSize::Custom(rems_from_px(13_f32)))),
            )
    }
}

#[derive(IntoElement)]
struct RecentProjectRow {
    name: SharedString,
    path: SharedString,
    action: Box<dyn Action>,
    tab_index: usize,
    focus_handle: FocusHandle,
}

impl RecentProjectRow {
    fn new(
        name: impl Into<SharedString>,
        path: impl Into<SharedString>,
        action: &dyn Action,
        tab_index: usize,
        focus_handle: FocusHandle,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            action: action.boxed_clone(),
            tab_index,
            focus_handle,
        }
    }
}

impl RenderOnce for RecentProjectRow {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let id = format!("welcome-recent-project-{}", self.tab_index);

        ButtonLike::new(id)
            .tab_index(self.tab_index as isize)
            .full_width()
            .height(rems_from_px(26_f32).into())
            .size(ButtonSize::None)
            .style(ButtonStyle::Subtle)
            .child(
                h_flex()
                    .w_full()
                    .gap_4()
                    .px_2()
                    .child(
                        h_flex()
                            .min_w_0()
                            .flex_1()
                            .child(Label::new(self.name).size(LabelSize::Small).truncate()),
                    )
                    .child(
                        h_flex().max_w(rems_from_px(210_f32)).flex_none().child(
                            Label::new(self.path)
                                .size(LabelSize::Custom(rems_from_px(11_f32)))
                                .color(Color::Muted)
                                .truncate(),
                        ),
                    ),
            )
            .on_click(move |_, window, cx| {
                self.focus_handle.dispatch_action(&*self.action, window, cx)
            })
    }
}

pub struct WelcomePage {
    workspace: WeakEntity<Workspace>,
    focus_handle: FocusHandle,
    fallback_to_recent_projects: bool,
    recent_workspaces: Option<Vec<RecentWorkspace>>,
}

impl WelcomePage {
    pub fn new(
        workspace: WeakEntity<Workspace>,
        fallback_to_recent_projects: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        cx.on_focus(&focus_handle, window, |_, _, cx| cx.notify())
            .detach();

        if fallback_to_recent_projects {
            let fs = workspace
                .upgrade()
                .map(|ws| ws.read(cx).app_state().fs.clone());
            let db = WorkspaceDb::global(cx);
            cx.spawn_in(window, async move |this: WeakEntity<Self>, cx| {
                let Some(fs) = fs else { return };
                let workspaces = db
                    .recent_project_workspaces(fs.as_ref())
                    .await
                    .log_err()
                    .unwrap_or_default();

                this.update(cx, |this, cx| {
                    this.recent_workspaces = Some(workspaces);
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }

        WelcomePage {
            workspace,
            focus_handle,
            fallback_to_recent_projects,
            recent_workspaces: None,
        }
    }

    fn select_next(&mut self, _: &SelectNext, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_next(cx);
        cx.notify();
    }

    fn select_previous(&mut self, _: &SelectPrevious, window: &mut Window, cx: &mut Context<Self>) {
        window.focus_prev(cx);
        cx.notify();
    }

    fn open_recent_project(
        &mut self,
        action: &OpenRecentProject,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(recent_workspaces) = &self.recent_workspaces {
            if let Some(workspace) = recent_workspaces.get(action.index) {
                let is_local = matches!(workspace.location, SerializedWorkspaceLocation::Local);

                if is_local {
                    let paths = workspace.paths.paths().to_vec();
                    let open_mode = match WorkspaceSettings::get_global(cx).default_open_behavior {
                        DefaultOpenBehavior::ExistingWindow => OpenMode::Activate,
                        DefaultOpenBehavior::NewWindow => OpenMode::NewWindow,
                    };
                    self.workspace
                        .update(cx, |workspace, cx| {
                            workspace
                                .open_workspace_for_paths(open_mode, paths, window, cx)
                                .detach_and_log_err(cx);
                        })
                        .log_err();
                } else {
                    use zed_actions::OpenRecent;
                    window.dispatch_action(OpenRecent::default().boxed_clone(), cx);
                }
            }
        }
    }

    fn render_header(&self) -> impl IntoElement {
        v_flex()
            .gap_1()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(Vector::square(VectorName::ZedLogo, rems_from_px(24_f32)))
                    .child(
                        Label::new("RACOON")
                            .size(LabelSize::Custom(rems_from_px(20_f32)))
                            .weight(FontWeight::BOLD),
                    ),
            )
            .child(
                Label::new("The editor for what's next")
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
    }

    fn render_action_grid(&self) -> impl IntoElement {
        let open_remote = OpenRemote::default();

        div()
            .w_full()
            .grid()
            .grid_cols(2)
            .gap_2()
            .child(ActionCard::new(
                "Open project",
                IconName::FolderOpen,
                &Open::DEFAULT,
                0,
                self.focus_handle.clone(),
            ))
            .child(ActionCard::new(
                "Clone repo",
                IconName::CloudDownload,
                &GitClone,
                1,
                self.focus_handle.clone(),
            ))
            .child(ActionCard::new(
                "Connect via SSH",
                IconName::Server,
                &open_remote,
                2,
                self.focus_handle.clone(),
            ))
            .child(ActionCard::new(
                "Open command palette",
                IconName::ListCollapse,
                &command_palette::Toggle,
                3,
                self.focus_handle.clone(),
            ))
    }

    fn render_recent_project_section(
        &self,
        recent_projects: Vec<impl IntoElement>,
    ) -> impl IntoElement {
        v_flex()
            .w_full()
            .gap_1()
            .child(
                Label::new("Recent projects")
                    .size(LabelSize::XSmall)
                    .color(Color::Muted)
                    .mb_1(),
            )
            .children(recent_projects)
    }

    fn render_recent_project(
        &self,
        project_index: usize,
        tab_index: usize,
        location: &SerializedWorkspaceLocation,
        paths: &PathList,
    ) -> impl IntoElement {
        let name = project_name(paths);
        let path = match location {
            SerializedWorkspaceLocation::Local => project_path(paths),
            SerializedWorkspaceLocation::Remote(options) => options.display_name(),
        };

        RecentProjectRow::new(
            name,
            path,
            &OpenRecentProject {
                index: project_index,
            },
            tab_index,
            self.focus_handle.clone(),
        )
    }
}

impl Render for WelcomePage {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let recent_projects = self
            .recent_workspaces
            .as_ref()
            .into_iter()
            .flatten()
            .take(5)
            .enumerate()
            .map(|(index, workspace)| {
                self.render_recent_project(
                    index,
                    ACTION_COUNT + index,
                    &workspace.location,
                    &workspace.identity_paths,
                )
            })
            .collect::<Vec<_>>();

        let showing_recent_projects =
            self.fallback_to_recent_projects && !recent_projects.is_empty();
        let next_tab_index = ACTION_COUNT + recent_projects.len();

        v_flex()
            .key_context("Welcome")
            .track_focus(&self.focus_handle(cx))
            .on_action(cx.listener(Self::select_previous))
            .on_action(cx.listener(Self::select_next))
            .on_action(cx.listener(Self::open_recent_project))
            .size_full()
            .bg(cx.theme().colors().editor_background)
            .child(
                v_flex()
                    .id("welcome-content")
                    .flex_1()
                    .w_full()
                    .items_center()
                    .justify_center()
                    .overflow_y_scroll()
                    .p_8()
                    .child(
                        v_flex()
                            .w(rems_from_px(400_f32))
                            .gap_6()
                            .child(self.render_header())
                            .child(self.render_action_grid())
                            .when(showing_recent_projects, |this| {
                                this.child(self.render_recent_project_section(recent_projects))
                            })
                            .when(!self.fallback_to_recent_projects, |this| {
                                this.child(
                                    v_flex().gap_4().child(Divider::horizontal()).child(
                                        Button::new("welcome-exit", "Return to Onboarding")
                                            .tab_index(next_tab_index as isize)
                                            .full_width()
                                            .label_size(LabelSize::XSmall)
                                            .on_click(|_, window, cx| {
                                                window.dispatch_action(
                                                    OpenOnboarding.boxed_clone(),
                                                    cx,
                                                );
                                            }),
                                    ),
                                )
                            }),
                    ),
            )
    }
}

impl EventEmitter<ItemEvent> for WelcomePage {}

impl Focusable for WelcomePage {
    fn focus_handle(&self, _: &App) -> gpui::FocusHandle {
        self.focus_handle.clone()
    }
}

impl Item for WelcomePage {
    type Event = ItemEvent;

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Welcome".into()
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        Some("New Welcome Page Opened")
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn to_item_events(event: &Self::Event, f: &mut dyn FnMut(crate::item::ItemEvent)) {
        f(*event)
    }
}

impl crate::SerializableItem for WelcomePage {
    fn serialized_item_kind() -> &'static str {
        "WelcomePage"
    }

    fn cleanup(
        workspace_id: crate::WorkspaceId,
        alive_items: Vec<crate::ItemId>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<gpui::Result<()>> {
        crate::delete_unloaded_items(
            alive_items,
            workspace_id,
            "welcome_pages",
            &persistence::WelcomePagesDb::global(cx),
            cx,
        )
    }

    fn deserialize(
        _project: Entity<project::Project>,
        workspace: gpui::WeakEntity<Workspace>,
        workspace_id: crate::WorkspaceId,
        item_id: crate::ItemId,
        window: &mut Window,
        cx: &mut App,
    ) -> Task<gpui::Result<Entity<Self>>> {
        if persistence::WelcomePagesDb::global(cx)
            .get_welcome_page(item_id, workspace_id)
            .ok()
            .is_some_and(|is_open| is_open)
        {
            Task::ready(Ok(
                cx.new(|cx| WelcomePage::new(workspace, false, window, cx))
            ))
        } else {
            Task::ready(Err(anyhow::anyhow!("No welcome page to deserialize")))
        }
    }

    fn serialize(
        &mut self,
        workspace: &mut Workspace,
        item_id: crate::ItemId,
        _closing: bool,
        cx: &mut Context<Self>,
    ) -> Option<Task<gpui::Result<()>>> {
        let workspace_id = workspace.database_id()?;
        let db = persistence::WelcomePagesDb::global(cx);
        Some(cx.background_spawn(
            async move { db.save_welcome_page(item_id, workspace_id, true).await },
        ))
    }

    fn should_serialize(&self, event: &Self::Event) -> bool {
        event == &ItemEvent::UpdateTab
    }
}

mod persistence {
    use crate::WorkspaceDb;
    use db::{
        query,
        sqlez::{domain::Domain, thread_safe_connection::ThreadSafeConnection},
        sqlez_macros::sql,
    };

    pub struct WelcomePagesDb(ThreadSafeConnection);

    impl Domain for WelcomePagesDb {
        const NAME: &str = stringify!(WelcomePagesDb);

        const MIGRATIONS: &[&str] = (&[sql!(
                    CREATE TABLE welcome_pages (
                        workspace_id INTEGER,
                        item_id INTEGER UNIQUE,
                        is_open INTEGER DEFAULT FALSE,

                        PRIMARY KEY(workspace_id, item_id),
                        FOREIGN KEY(workspace_id) REFERENCES workspaces(workspace_id)
                        ON DELETE CASCADE
                    ) STRICT;
        )]);
    }

    db::static_connection!(WelcomePagesDb, [WorkspaceDb]);

    impl WelcomePagesDb {
        query! {
            pub async fn save_welcome_page(
                item_id: crate::ItemId,
                workspace_id: crate::WorkspaceId,
                is_open: bool
            ) -> Result<()> {
                INSERT OR REPLACE INTO welcome_pages(item_id, workspace_id, is_open)
                VALUES (?, ?, ?)
            }
        }

        query! {
            pub fn get_welcome_page(
                item_id: crate::ItemId,
                workspace_id: crate::WorkspaceId
            ) -> Result<bool> {
                SELECT is_open
                FROM welcome_pages
                WHERE item_id = ? AND workspace_id = ?
            }
        }
    }
}

fn project_name(paths: &PathList) -> String {
    let joined = paths
        .paths()
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    if joined.is_empty() {
        "Untitled".to_string()
    } else {
        joined
    }
}

fn project_path(paths: &PathList) -> String {
    let Some(path) = paths.ordered_paths().next() else {
        return String::new();
    };
    let directory = path.parent().unwrap_or(path);

    match dirs::home_dir().and_then(|home| {
        directory
            .strip_prefix(home)
            .ok()
            .map(|relative| relative.to_path_buf())
    }) {
        Some(relative) if relative.as_os_str().is_empty() => "~".to_string(),
        Some(relative) => format!("~/{}", relative.display()),
        None => directory.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_name_empty() {
        let paths = PathList::new::<&str>(&[]);
        assert_eq!(project_name(&paths), "Untitled");
    }

    #[test]
    fn test_project_name_single() {
        let paths = PathList::new(&["/home/user/my-project"]);
        assert_eq!(project_name(&paths), "my-project");
    }

    #[test]
    fn test_project_name_multiple() {
        // PathList sorts lexicographically, so filenames appear in alpha order
        let paths = PathList::new(&["/home/user/zed", "/home/user/api"]);
        assert_eq!(project_name(&paths), "api, zed");
    }

    #[test]
    fn test_project_name_root_path_filtered() {
        // A bare root "/" has no file_name(), falls back to "Untitled"
        let paths = PathList::new(&["/"]);
        assert_eq!(project_name(&paths), "Untitled");
    }
}
