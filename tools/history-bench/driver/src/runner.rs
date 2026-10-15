//! Per-commit pipeline: extract the tree, generate the adapter, build, measure,
//! verify against the reference, and merge into the cached results.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use hb_common::{fnv1a, read_f32s, QueryKind};
use serde::{Deserialize, Serialize};

use crate::inputs::{inputs_dir, SceneMeta, SETS};
use crate::{git, CommitInfo, Ctx, Mode};

const COMMON_SRC: &str = include_str!("../../common/src/lib.rs");
const ERA_A_SRC: &str = include_str!("../../adapters/era_a.rs");
const ERA_B_SRC: &str = include_str!("../../adapters/era_b.rs");

/// Bump when the measurement protocol changes without the sources changing.
const PROTOCOL_VERSION: u32 = 2;

/// A cached result is reused only if every input of the measurement is identical.
#[derive(Serialize, Deserialize, Clone)]
pub struct Entry {
    pub scene: String,
    pub set: String,
    pub key: String,
    pub variant: String,
    pub native_occlusion: bool,
    pub rays: usize,
    pub build_ms: f64,
    /// Every timed pass, all processes concatenated.
    pub pass_ns: Vec<u64>,
    /// Median pass time of each process.
    #[serde(default)]
    pub proc_median_ns: Vec<u64>,
    /// Median of the per-process medians.
    pub median_ns: u64,
    pub mrays: f64,
    /// Slowest and fastest process.
    #[serde(default)]
    pub mrays_min: f64,
    #[serde(default)]
    pub mrays_max: f64,
    /// (slowest - fastest) / median across processes, in percent.
    pub spread_pct: f64,
    pub check: Check,
    pub run: RunInfo,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Check {
    pub n: usize,
    /// Hit on one side, miss on the other.
    pub hit_miss: usize,
    /// Both hit, at a different distance.
    pub t_mismatch: usize,
    pub ok: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RunInfo {
    pub rustc: String,
    pub mode: String,
    pub rustflags: String,
    pub profile: String,
    pub pinned: bool,
    pub core: usize,
    pub measured_at_unix: u64,
}

#[derive(Serialize, Deserialize, Default)]
pub struct CommitResults {
    pub commit: String,
    pub short: String,
    pub date: String,
    pub subject: String,
    pub entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct AdapterOut {
    variant: String,
    native_occlusion: bool,
    pinned: bool,
    results: Vec<AdapterResult>,
}

#[derive(Deserialize)]
struct AdapterResult {
    scene: String,
    set: String,
    rays: usize,
    build_ms: f64,
    pass_ns: Vec<u64>,
}

pub fn results_path(ctx: &Ctx, commit: &CommitInfo) -> PathBuf {
    ctx.results_dir().join(format!("{}.json", commit.short))
}

pub fn load_results(path: &Path) -> Option<CommitResults> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

enum Era {
    A,
    B,
}

pub fn run_commit(ctx: &Ctx, scenes: &[SceneMeta], commit: &CommitInfo) -> Result<(), String> {
    let path = results_path(ctx, commit);
    let mut results = load_results(&path).unwrap_or_default();
    results.commit = commit.hash.clone();
    results.short = commit.short.clone();
    results.date = commit.date.clone();
    results.subject = commit.subject.clone();

    let src = ctx.cache.join("src").join(&commit.short);
    if !src.join(".hb-extracted").exists() {
        extract(ctx, commit, &src)?;
    }
    let era = if src.join("crates/rnpt-bvh/Cargo.toml").exists() { Era::B } else { Era::A };
    let (rustflags, profile) = build_config(ctx, &src)?;
    let adapter_src = match era {
        Era::A => ERA_A_SRC,
        Era::B => ERA_B_SRC,
    };
    let adapter_hash = fnv1a(format!("{PROTOCOL_VERSION}\n{COMMON_SRC}\n{adapter_src}").as_bytes());

    let key = |meta: &SceneMeta, set: &str| {
        let k = format!(
            "{adapter_hash:016x}|{}|{}|{}|{rustflags}|{profile}|{}|{}|{}",
            meta.input_key(set),
            ctx.rustc,
            ctx.mode.name(),
            ctx.passes,
            ctx.check,
            ctx.procs
        );
        format!("{:016x}", fnv1a(k.as_bytes()))
    };

    let mut jobs: Vec<(&SceneMeta, &str)> = vec![];
    for meta in scenes {
        for (set, _) in SETS {
            let fresh = results.entries.iter().any(|e| e.scene == meta.name && e.set == *set && e.key == key(meta, set));
            if ctx.force || !fresh {
                jobs.push((meta, set));
            }
        }
    }
    if jobs.is_empty() {
        eprintln!("[{}] up to date", commit.short);
        return Ok(());
    }

    let adapter_dir = src.join("hb-adapter");
    let features = setup_adapter(ctx, &src, &adapter_dir, &era, &profile, adapter_hash)?;
    let target = ctx.cache.join("target").join(format!("{}-{}", commit.short, ctx.mode.name()));

    eprintln!("[{}] building ({}, features: [{}])", commit.short, ctx.mode.name(), features.join(","));
    let mut build = ctx.cargo();
    build.current_dir(&adapter_dir).args(["build", "--release"]).env("CARGO_TARGET_DIR", &target);
    if !features.is_empty() {
        build.args(["--features", &features.join(",")]);
    }
    if let Mode::Head = ctx.mode {
        build.env("RUSTFLAGS", &rustflags);
    }
    run_quiet(&mut build, "cargo build")?;

    // Each measurement is repeated in separate processes: memory layout varies
    // from one process to the next, and on scenes that do not fit in cache
    // that variation dominates the pass-to-pass noise.
    let exe = target.join("release").join(format!("hb-adapter{}", std::env::consts::EXE_SUFFIX));
    let run_dir = ctx.cache.join("run").join(format!("{}-{}", commit.short, ctx.mode.name()));
    let _ = fs::remove_dir_all(&run_dir);
    let mut outs: Vec<AdapterOut> = vec![];
    for p in 0..ctx.procs {
        let out_dir = run_dir.join(p.to_string());
        fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let mut run = Command::new(&exe);
        run.arg("--input").arg(inputs_dir(ctx)).arg("--out-dir").arg(&out_dir);
        run.args(["--passes", &ctx.passes.to_string(), "--check", &ctx.check.to_string(), "--core", &ctx.core.to_string()]);
        for (meta, set) in &jobs {
            run.args(["--job", &format!("{}:{set}", meta.name)]);
        }
        eprintln!("[{}] measuring {} job(s), process {}/{}", commit.short, jobs.len(), p + 1, ctx.procs);
        let status = run.status().map_err(|e| format!("{}: {e}", exe.display()))?;
        if !status.success() {
            return Err(format!("adapter failed: {status}"));
        }
        let text = fs::read_to_string(out_dir.join("adapter.json")).map_err(|e| e.to_string())?;
        outs.push(serde_json::from_str(&text).map_err(|e| e.to_string())?);
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let median_of = |v: &[u64]| {
        let mut s = v.to_vec();
        s.sort_unstable();
        s[s.len() / 2]
    };
    let out = outs.swap_remove(0);
    for r in out.results {
        let meta = scenes.iter().find(|m| m.name == r.scene).unwrap();
        // Results are deterministic: checking the first process is enough.
        let check = verify(ctx, meta, &r.set, &run_dir.join("0"))?;
        let mut proc_median_ns = vec![median_of(&r.pass_ns)];
        let mut pass_ns = r.pass_ns.clone();
        for o in &outs {
            let other = o.results.iter().find(|x| x.scene == r.scene && x.set == r.set).unwrap();
            proc_median_ns.push(median_of(&other.pass_ns));
            pass_ns.extend(&other.pass_ns);
        }
        let median = median_of(&proc_median_ns);
        let (fastest, slowest) = (*proc_median_ns.iter().min().unwrap(), *proc_median_ns.iter().max().unwrap());
        let spread = (slowest - fastest) as f64 / median as f64 * 100.0;
        let mrays = |ns: u64| r.rays as f64 / (ns as f64 * 1e-9) / 1e6;
        if !check.ok {
            eprintln!("[{}] {}/{}: INCORRECT ({} hit/miss, {} t mismatches out of {})", commit.short, r.scene, r.set, check.hit_miss, check.t_mismatch, check.n);
        }
        let entry = Entry {
            key: key(meta, &r.set),
            variant: out.variant.clone(),
            native_occlusion: out.native_occlusion,
            rays: r.rays,
            build_ms: r.build_ms,
            median_ns: median,
            mrays: mrays(median),
            mrays_min: mrays(slowest),
            mrays_max: mrays(fastest),
            spread_pct: spread,
            proc_median_ns,
            pass_ns,
            check,
            run: RunInfo {
                rustc: ctx.rustc.clone(),
                mode: ctx.mode.name().into(),
                rustflags: rustflags.clone(),
                profile: profile.clone(),
                pinned: out.pinned,
                core: ctx.core,
                measured_at_unix: now,
            },
            scene: r.scene,
            set: r.set,
        };
        results.entries.retain(|e| !(e.scene == entry.scene && e.set == entry.set));
        results.entries.push(entry);
    }
    results.entries.sort_by(|a, b| (&a.scene, &a.set).cmp(&(&b.scene, &b.set)));

    fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(&path, serde_json::to_string_pretty(&results).unwrap()).map_err(|e| e.to_string())?;
    Ok(())
}

fn extract(ctx: &Ctx, commit: &CommitInfo, src: &Path) -> Result<(), String> {
    let _ = fs::remove_dir_all(src);
    fs::create_dir_all(src).map_err(|e| e.to_string())?;
    let tar = ctx.cache.join("src").join(format!("{}.tar", commit.short));
    git(&ctx.repo, &["archive", "--format=tar", "-o", &tar.to_string_lossy(), &commit.hash])?;
    let mut cmd = Command::new("tar");
    cmd.arg("-xf").arg(&tar).arg("-C").arg(src);
    run_quiet(&mut cmd, "tar")?;
    let _ = fs::remove_file(&tar);
    fs::write(src.join(".hb-extracted"), &commit.hash).map_err(|e| e.to_string())
}

/// The `[profile.*]` sections of a root Cargo.toml.
fn profile_sections(cargo_toml: &str) -> String {
    let mut out = String::new();
    let mut inside = false;
    for line in cargo_toml.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            inside = t.starts_with("[profile");
        }
        if inside && !t.is_empty() {
            out.push_str(t);
            out.push('\n');
        }
    }
    out
}

/// (rustflags, profile) used for this commit, as recorded with each result.
fn build_config(ctx: &Ctx, src: &Path) -> Result<(String, String), String> {
    match ctx.mode {
        Mode::Own => {
            let profile = profile_sections(&fs::read_to_string(src.join("Cargo.toml")).map_err(|e| e.to_string())?);
            let config = fs::read_to_string(src.join(".cargo/config.toml")).unwrap_or_default();
            let flags = config.lines().find(|l| l.trim_start().starts_with("rustflags")).map_or("(none)".into(), |l| l.trim().to_string());
            Ok((flags, profile))
        }
        Mode::Head => {
            let profile = profile_sections(&git(&ctx.repo, &["show", "HEAD:Cargo.toml"])?);
            Ok(("-C target-cpu=native".into(), profile))
        }
    }
}

/// Write the adapter crate into the checkout and pick the features that match
/// this commit's API (by trying to compile, most recent API first).
fn setup_adapter(ctx: &Ctx, src: &Path, dir: &Path, era: &Era, profile: &str, adapter_hash: u64) -> Result<Vec<String>, String> {
    let common = ctx.tool_dir.join("common").to_string_lossy().replace('\\', "/");
    let deps = match era {
        Era::A => "rnpt = { path = \"../crates/rnpt\" }\nnalgebra = \"0.33\"",
        Era::B => "rnpt-bvh = { path = \"../crates/rnpt-bvh\" }",
    };
    let manifest = format!(
        "[package]\nname = \"hb-adapter\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
         [dependencies]\nhb-common = {{ path = \"{common}\" }}\n{deps}\n\n\
         [features]\ntextures = []\noccluded = []\n\n[workspace]\n\n{profile}"
    );
    fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    fs::write(dir.join("Cargo.toml"), manifest).map_err(|e| e.to_string())?;
    let source = match era {
        Era::A => ERA_A_SRC,
        Era::B => ERA_B_SRC,
    };
    fs::write(dir.join("src/main.rs"), source).map_err(|e| e.to_string())?;
    // Pin dependencies to the versions the commit itself was built with.
    if src.join("Cargo.lock").exists() {
        fs::copy(src.join("Cargo.lock"), dir.join("Cargo.lock")).map_err(|e| e.to_string())?;
    }

    let candidates: &[&[&str]] = match era {
        Era::B => &[&[]],
        Era::A => &[&["textures", "occluded"], &["textures"], &["occluded"], &[]],
    };
    let cached = dir.join(format!("features-{adapter_hash:016x}.txt"));
    if let Ok(f) = fs::read_to_string(&cached) {
        return Ok(f.split(',').filter(|s| !s.is_empty()).map(String::from).collect());
    }
    let target = ctx.cache.join("target").join("check");
    for features in candidates {
        let mut cmd = ctx.cargo();
        cmd.current_dir(dir).arg("check").env("CARGO_TARGET_DIR", &target);
        if !features.is_empty() {
            cmd.args(["--features", &features.join(",")]);
        }
        if run_quiet(&mut cmd, "cargo check").is_ok() {
            fs::write(&cached, features.join(",")).map_err(|e| e.to_string())?;
            return Ok(features.iter().map(|s| s.to_string()).collect());
        }
    }
    Err("no adapter variant compiles for this commit".into())
}

fn verify(ctx: &Ctx, meta: &SceneMeta, set: &str, out_dir: &Path) -> Result<Check, String> {
    let reference = read_f32s(&inputs_dir(ctx).join(format!("{}.{set}.ref", meta.name)));
    let got = read_f32s(&out_dir.join(format!("{}.{set}.check", meta.name)));
    if got.len() != reference.len() {
        return Err(format!("{}/{set}: check size mismatch", meta.name));
    }
    let occlusion = SETS.iter().any(|(s, k)| *s == set && *k == QueryKind::Occluded);
    let (mut hit_miss, mut t_mismatch) = (0, 0);
    for (&g, &r) in got.iter().zip(&reference) {
        if occlusion {
            hit_miss += (g != r) as usize;
            continue;
        }
        match (g.is_finite(), r.is_finite()) {
            (true, true) if (g - r).abs() > 1e-4 * r.abs() + 1e-5 * meta.diag => t_mismatch += 1,
            (a, b) if a != b => hit_miss += 1,
            _ => {}
        }
    }
    let n = reference.len();
    // Tolerates a handful of grazing rays decided differently by FMA rounding.
    let ok = (hit_miss + t_mismatch) as f64 <= 1e-3 * n as f64;
    Ok(Check { n, hit_miss, t_mismatch, ok })
}

fn run_quiet(cmd: &mut Command, what: &str) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("{what}: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(15).collect();
        Err(format!("{what} failed:\n{}", tail.into_iter().rev().collect::<Vec<_>>().join("\n")))
    }
}
