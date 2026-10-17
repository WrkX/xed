// Upstream Zed targets Namespace Cloud (`namespace-profile-*`) and self-hosted
// (`self-*`) runners. Those don't exist on this fork, so jobs assigned to them
// queue forever waiting for a runner that never appears. Run everything on
// GitHub-hosted runners instead.
pub const DANGERBOT: Runner = Runner("ubuntu-latest");

pub const LINUX_SMALL: Runner = Runner("ubuntu-latest");
pub const LINUX_DEFAULT: Runner = LINUX_XL;
pub const LINUX_XL: Runner = Runner("ubuntu-latest");
pub const LINUX_LARGE: Runner = Runner("ubuntu-latest");
pub const LINUX_MEDIUM: Runner = Runner("ubuntu-latest");

pub const LINUX_X86_BUNDLER: Runner = Runner("ubuntu-latest");
// `bundle-linux` cross-compiles nothing: it builds for the host architecture
// and names the archive after `uname -m`, so the aarch64 bundle must run on an
// arm64 runner to produce an aarch64 artifact.
pub const LINUX_ARM_BUNDLER: Runner = Runner("ubuntu-24.04-arm");

pub const LINUX_LARGE_RAM: Runner = Runner("ubuntu-latest");

pub const MAC_DEFAULT: Runner = Runner("macos-latest");
pub const MAC_SMALL: Runner = Runner("macos-latest");
pub const WINDOWS_DEFAULT: Runner = Runner("windows-latest");

pub struct Runner(&'static str);

impl Into<gh_workflow::RunsOn> for Runner {
    fn into(self) -> gh_workflow::RunsOn {
        self.0.into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    AARCH64,
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arch::X86_64 => write!(f, "x86_64"),
            Arch::AARCH64 => write!(f, "aarch64"),
        }
    }
}

impl Arch {
    pub fn linux_bundler(&self) -> Runner {
        match self {
            Arch::X86_64 => LINUX_X86_BUNDLER,
            Arch::AARCH64 => LINUX_ARM_BUNDLER,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    Linux,
    Mac,
}

impl Platform {
    pub fn target_triple(self, arch: Arch) -> &'static str {
        match (self, arch) {
            (Self::Linux, Arch::X86_64) => "x86_64-unknown-linux-musl",
            (Self::Linux, Arch::AARCH64) => "aarch64-unknown-linux-musl",
            (Self::Mac, Arch::X86_64) => "x86_64-apple-darwin",
            (Self::Mac, Arch::AARCH64) => "aarch64-apple-darwin",
            (Self::Windows, Arch::X86_64) => "x86_64-pc-windows-msvc",
            (Self::Windows, Arch::AARCH64) => "aarch64-pc-windows-msvc",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Platform::Windows => write!(f, "windows"),
            Platform::Linux => write!(f, "linux"),
            Platform::Mac => write!(f, "mac"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReleaseChannel {
    Dev,
    Nightly,
}
