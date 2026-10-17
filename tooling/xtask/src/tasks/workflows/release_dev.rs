use super::{runners, steps, steps::named, vars};
use crate::tasks::workflows::{
    release::download_workflow_artifacts,
    run_bundling::{bundle_linux, bundle_windows},
    runners::{Arch, ReleaseChannel},
    steps::{CommonPermissionSets, FluentBuilder, NamedJob, dependant_job},
    vars::WorkflowInput,
};

use gh_workflow::{ctx::Context, *};
use indoc::indoc;

/// Tag of the rolling release that receives the dev-channel artifacts.
const DEV_RELEASE_TAG: &str = "dev";

/// Generates the release_dev.yml workflow.
///
/// Builds the `dev` channel for Linux and Windows and attaches the resulting
/// artifacts to a rolling GitHub prerelease. The workflow depends on no other
/// workflow, so unrelated CI failures cannot block a dev build.
///
/// The publish job runs even if some (or all) of the bundles fail, so that
/// whatever artifacts were produced still make it into the release. It refuses
/// to create a release with no artifacts at all.
pub fn release_dev() -> Workflow {
    let tag = WorkflowInput::string("tag", Some(DEV_RELEASE_TAG.to_string()))
        .description("Tag of the release to create or update. Assets are overwritten on re-run.");

    const DEV: Option<ReleaseChannel> = Some(ReleaseChannel::Dev);

    let bundles = vec![
        bundle_linux(Arch::X86_64, DEV, false, &[]),
        bundle_linux(Arch::AARCH64, DEV, false, &[]),
        bundle_windows(Arch::X86_64, DEV, &[]),
        bundle_windows(Arch::AARCH64, DEV, &[]),
    ];

    let publish = publish_release(&bundles.iter().collect::<Vec<_>>(), &tag);

    named::workflow()
        .with_minimal_permissions()
        .on(Event::default()
            .workflow_dispatch(WorkflowDispatch::default().add_input(tag.name, tag.input())))
        .concurrency(
            Concurrency::default()
                .group("release-dev")
                .cancel_in_progress(true),
        )
        .add_env(("CARGO_TERM_COLOR", "always"))
        .add_env(("RUST_BACKTRACE", "1"))
        .map(|mut workflow| {
            for job in bundles {
                workflow = workflow.add_job(job.name, job.job);
            }
            workflow
        })
        .add_job(publish.name, publish.job)
}

fn publish_release(bundles: &[&NamedJob], tag: &WorkflowInput) -> NamedJob {
    fn create_or_update_release(tag: &WorkflowInput) -> Step<Run> {
        named::bash(indoc! {r#"
            if ! compgen -G "./artifacts/*/*" > /dev/null; then
                echo "No artifacts were produced; refusing to publish an empty release." >&2
                exit 1
            fi

            if gh release view "$TAG" >/dev/null 2>&1; then
                echo "Release '$TAG' already exists; uploading assets."
            else
                echo "Creating release '$TAG' at $GITHUB_SHA."
                gh release create "$TAG" \
                    --target "$GITHUB_SHA" \
                    --title "$TAG (dev build)" \
                    --notes "Automated dev build of $GITHUB_SHA." \
                    --prerelease
            fi

            gh release upload "$TAG" ./artifacts/*/* --clobber
        "#})
        .add_env(("GH_TOKEN", vars::GITHUB_TOKEN))
        .add_env(("GH_REPO", Context::github().repository().to_string()))
        .add_env(("TAG", tag.to_string()))
    }

    NamedJob {
        name: "publish_release".to_owned(),
        job: dependant_job(bundles)
            .cond(Expression::new("!cancelled()"))
            .runs_on(runners::LINUX_SMALL)
            .permissions(Permissions::default().contents(Level::Write))
            .add_step(download_workflow_artifacts())
            .add_step(steps::script("ls -lR ./artifacts"))
            .add_step(create_or_update_release(tag)),
    }
}
