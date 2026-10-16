#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

//! Links clang's Objective-C availability runtime for the native Metal engine.
//!
//! `ggml-metal-device.m` (vendored by `llama-cpp-sys-2`) uses
//! `@available(macOS 15.0, *)`, so clang emits a call to the
//! `__isPlatformVersionAtLeast` helper when the deployment target is below 15.0.
//! That helper lives in clang's `libclang_rt.osx.a`, which the clang *driver*
//! would normally pull in — but Rust links binaries with `-nodefaultlibs`, so
//! release builds fail with `Undefined symbols: ___isPlatformVersionAtLeast`.
//!
//! The runtime is published with `cargo:rustc-link-lib` (rather than
//! `rustc-link-arg`) so that it reaches the final binary even though `local_gguf`
//! is only a transitive dependency of `zed`: cargo propagates link libraries and
//! search paths from a dependency's build script to the whole crate graph, while
//! `rustc-link-arg` applies only to the crate's own targets.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only Apple silicon compiles ggml's Metal backend.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os != "macos" || target_arch != "aarch64" {
        return;
    }
    if env::var_os("CARGO_FEATURE_METAL").is_none() {
        return;
    }

    let compiler = env::var("CC").unwrap_or_else(|_| "clang".to_string());
    let Ok(output) = Command::new(&compiler)
        .arg("--print-file-name=libclang_rt.osx.a")
        .output()
    else {
        println!(
            "cargo:warning=could not run `{compiler}` to locate libclang_rt.osx.a; \
             the Metal engine may fail to link"
        );
        return;
    };

    let runtime = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    // `--print-file-name` echoes the name back when the file cannot be found, so
    // require an absolute path that actually exists before linking it.
    if !runtime.is_absolute() || !runtime.exists() {
        println!(
            "cargo:warning=could not locate libclang_rt.osx.a via `{compiler}`; \
             the Metal engine may fail to link"
        );
        return;
    }

    if let Some(directory) = runtime.parent() {
        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
