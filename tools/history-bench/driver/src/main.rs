//! history-bench: replays the BVH's git history and measures every version on
//! identical inputs, with cached, keyed results so any run can be redone or
//! extended with a single command. See README.md.

mod inputs;
mod report;
mod runner;

use std::path::{Path, PathBuf};
use std::process::Command;

use inputs::{SceneMeta, SCENES};

const USAGE: &str = "\
usage: history-bench <prepare|list|run|report> [options]

  prepare        generate scenes, rays and brute-force references
  list           show the commits that would be measured
  run            prepare, measure every commit not already cached, report
  report         rebuild REPORT.md and results.csv from cached results

options:
  --scenes a,b   scenes to use (default: all available, without the 10M ones)
  --big          also use the 10M-triangle scenes
  --commits a,b  commits to measure (default: every commit touching the BVH)
  --mode M       own: each commit's own build config (default)
                 head: today's profile and -C target-cpu=native for all
  --passes N     timed passes per measurement (default 9)
  --procs N      separate processes per measurement (default 5)
  --rays N       rays per set (default 65536)
  --check N      rays verified against brute force (default 4096)
  --core K       core the adapter is pinned to (default 2)
  --toolchain T  build with `cargo +T` (default: rustup default)
  --cache DIR    cache directory (default: <local app data>/rnpt-history-bench)
  --force        re-measure even if cached";

/// Files whose history defines the commits worth measuring.
const BVH_PATHS: &[&str] = &[
    "crates/rnpt/src/bvh.rs",
    "crates/rnpt/src/bvh_builder.rs",
    "crates/rnpt/src/math.rs",
    "crates/rnpt-bvh",
    "Cargo.toml",
    "Cargo.lock",
    ".cargo",
];

#[derive(Clone, Copy)]
pub enum Mode {
    Own,
    Head,
}

impl Mode {
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Own => "own",
            Mode::Head => "head",
        }
    }
}

pub struct Ctx {
    pub repo: PathBuf,
    pub tool_dir: PathBuf,
    pub cache: PathBuf,
    pub mode: Mode,
    pub passes: usize,
    pub procs: usize,
    pub rays: usize,
    pub check: usize,
    pub core: usize,
    pub toolchain: Option<String>,
    pub rustc: String,
    pub machine: String,
    pub force: bool,
}

impl Ctx {
    pub fn results_dir(&self) -> PathBuf {
        self.tool_dir.join("results").join(&self.machine).join(self.mode.name())
    }

    /// A cargo invocation isolated from the environment this tool runs in.
    pub fn cargo(&self) -> Command {
        let mut cmd = Command::new("cargo");
        if let Some(tc) = &self.toolchain {
            cmd.arg(format!("+{tc}"));
        }
        clean_env(&mut cmd);
        cmd
    }
}

fn clean_env(cmd: &mut Command) {
    for (k, _) in std::env::vars() {
        let leaks = (k.starts_with("CARGO_") && k != "CARGO_HOME")
            || k.starts_with("RUSTC")
            || k == "RUSTFLAGS"
            || k == "RUSTDOCFLAGS"
            || k == "RUSTUP_TOOLCHAIN";
        if leaks {
            cmd.env_remove(k);
        }
    }
}

pub struct CommitInfo {
    pub hash: String,
    pub short: String,
    pub date: String,
    pub subject: String,
}

pub fn git(repo: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git").arg("-C").arg(repo).args(args).output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Commits in chronological order, optionally restricted to `paths`.
pub fn commit_log(repo: &Path, paths: &[&str]) -> Result<Vec<CommitInfo>, String> {
    let mut args = vec!["log", "--reverse", "--date=short", "--format=%H%x09%h%x09%ad%x09%s"];
    if !paths.is_empty() {
        args.push("--");
        args.extend(paths);
    }
    Ok(git(repo, &args)?
        .lines()
        .filter_map(|l| {
            let mut f = l.splitn(4, '\t');
            Some(CommitInfo {
                hash: f.next()?.into(),
                short: f.next()?.into(),
                date: f.next()?.into(),
                subject: f.next().unwrap_or("").into(),
            })
        })
        .collect())
}

fn has_bvh(repo: &Path, hash: &str) -> bool {
    ["crates/rnpt/src/bvh.rs", "crates/rnpt-bvh/Cargo.toml"]
        .iter()
        .any(|p| git(repo, &["cat-file", "-e", &format!("{hash}:{p}")]).is_ok())
}

fn select_commits(ctx: &Ctx, wanted: &Option<Vec<String>>) -> Result<Vec<CommitInfo>, String> {
    match wanted {
        None => Ok(commit_log(&ctx.repo, BVH_PATHS)?.into_iter().filter(|c| has_bvh(&ctx.repo, &c.hash)).collect()),
        Some(list) => {
            let all = commit_log(&ctx.repo, &[])?;
            list.iter()
                .map(|w| {
                    let full = git(&ctx.repo, &["rev-parse", &format!("{w}^{{commit}}")])?.trim().to_string();
                    all.iter()
                        .find(|c| c.hash == full)
                        .map(|c| CommitInfo { hash: c.hash.clone(), short: c.short.clone(), date: c.date.clone(), subject: c.subject.clone() })
                        .ok_or(format!("{w}: not in history"))
                })
                .collect()
        }
    }
}

fn cpu_brand() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        let mut bytes = vec![];
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            #[allow(unused_unsafe)]
            let r = unsafe { __cpuid(leaf) };
            for reg in [r.eax, r.ebx, r.ecx, r.edx] {
                bytes.extend_from_slice(&reg.to_le_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).trim_matches(char::from(0)).trim().to_string()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "unknown-cpu".into()
    }
}

fn slug(s: &str) -> String {
    let s: String = s.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    s.split('-').filter(|w| !w.is_empty() && *w != "processor").collect::<Vec<_>>().join("-")
}

fn default_cache() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("rnpt-history-bench")
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(USAGE)?;

    let tool_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let repo = PathBuf::from(git(&tool_dir, &["rev-parse", "--show-toplevel"])?.trim());

    let mut ctx = Ctx {
        repo,
        tool_dir,
        cache: default_cache(),
        mode: Mode::Own,
        passes: 9,
        procs: 5,
        rays: 65536,
        check: 4096,
        core: 2,
        toolchain: None,
        rustc: String::new(),
        machine: slug(&cpu_brand()),
        force: false,
    };
    let mut scene_names: Option<Vec<String>> = None;
    let mut commits: Option<Vec<String>> = None;
    let mut big = false;

    let list = |v: String| v.split(',').map(str::to_string).collect::<Vec<_>>();
    while let Some(k) = args.next() {
        let mut val = || args.next().ok_or(format!("{k} needs a value"));
        match k.as_str() {
            "--scenes" => scene_names = Some(list(val()?)),
            "--commits" => commits = Some(list(val()?)),
            "--big" => big = true,
            "--force" => ctx.force = true,
            "--mode" => {
                ctx.mode = match val()?.as_str() {
                    "own" => Mode::Own,
                    "head" => Mode::Head,
                    m => return Err(format!("unknown mode {m}")),
                }
            }
            "--passes" => ctx.passes = val()?.parse().map_err(|_| "bad --passes")?,
            "--procs" => ctx.procs = val()?.parse().map_err(|_| "bad --procs")?,
            "--rays" => ctx.rays = val()?.parse().map_err(|_| "bad --rays")?,
            "--check" => ctx.check = val()?.parse().map_err(|_| "bad --check")?,
            "--core" => ctx.core = val()?.parse().map_err(|_| "bad --core")?,
            "--toolchain" => ctx.toolchain = Some(val()?),
            "--cache" => ctx.cache = PathBuf::from(val()?),
            _ => return Err(format!("unknown option {k}\n\n{USAGE}")),
        }
    }

    let mut rustc = Command::new("rustc");
    if let Some(tc) = &ctx.toolchain {
        rustc.arg(format!("+{tc}"));
    }
    clean_env(&mut rustc);
    let out = rustc.arg("-V").output().map_err(|e| format!("rustc: {e}"))?;
    ctx.rustc = String::from_utf8_lossy(&out.stdout).trim().to_string();

    let specs: Vec<_> = SCENES
        .iter()
        .filter(|s| match &scene_names {
            Some(names) => names.iter().any(|n| n == s.name),
            None => big || !s.big,
        })
        .collect();

    let prepare = |ctx: &Ctx| -> Result<Vec<SceneMeta>, String> {
        let mut metas = vec![];
        for spec in &specs {
            match inputs::prepare(ctx, spec)? {
                Some(m) => metas.push(m),
                None => eprintln!("[inputs] {}: source asset missing, skipped", spec.name),
            }
        }
        Ok(metas)
    };

    match command.as_str() {
        "prepare" => {
            for m in prepare(&ctx)? {
                println!("{:<14} {:>10} tris  inputs {}", m.name, m.tris, m.scene_hash);
            }
        }
        "list" => {
            for c in select_commits(&ctx, &commits)? {
                println!("{} {} {}", c.short, c.date, c.subject);
            }
        }
        "run" => {
            let metas = prepare(&ctx)?;
            let selected = select_commits(&ctx, &commits)?;
            eprintln!("machine {}, {}, mode {}, {} commit(s), {} scene(s)", ctx.machine, ctx.rustc, ctx.mode.name(), selected.len(), metas.len());
            let mut failed = vec![];
            for c in &selected {
                if let Err(e) = runner::run_commit(&ctx, &metas, c) {
                    eprintln!("[{}] FAILED: {e}", c.short);
                    failed.push(c.short.clone());
                }
            }
            report::write(&ctx)?;
            if !failed.is_empty() {
                return Err(format!("failed commits: {}", failed.join(", ")));
            }
        }
        "report" => report::write(&ctx)?,
        _ => return Err(USAGE.into()),
    }
    Ok(())
}
