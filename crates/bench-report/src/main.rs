use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── ANSI colours (gracefully degrade if not a tty) ────────────────────────────
const GREEN:  &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED:    &str = "\x1b[31m";
const RESET:  &str = "\x1b[0m";
const BOLD:   &str = "\x1b[1m";
const DIM:    &str = "\x1b[2m";

fn ansi(code: &str) -> &str {
    if std::env::var_os("NO_COLOR").is_some() { "" } else { code }
}

// ── Data loading ──────────────────────────────────────────────────────────────

type Data = HashMap<(String, String, String), f64>; // (group, impl, size) → Mray/s

fn criterion_dir() -> Option<PathBuf> {
    // Walk up from cwd looking for target/criterion
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join("target").join("criterion");
        if candidate.is_dir() { return Some(candidate); }
        if !dir.pop() { return None; }
    }
}

fn load_data(base: &Path) -> Data {
    let mut data = Data::new();
    visit_dir(base, base, &mut data);
    data
}

fn visit_dir(base: &Path, dir: &Path, data: &mut Data) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_dir(base, &path, data);
        } else if path.file_name().map_or(false, |n| n == "estimates.json") {
            // Only care about the "new" snapshot, not "base" or "change"
            let components: Vec<_> = path.strip_prefix(base).ok()
                .map(|p| p.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect())
                .unwrap_or_default();
            // Expected: [group, impl, size, "new", "estimates.json"]
            if components.len() == 5 && components[3] == "new" {
                let (group, impl_, size) = (&components[0], &components[1], &components[2]);
                if let Some(ns) = parse_mean_ns(&path) {
                    let mrays = 1024.0 / (ns / 1e9) / 1e6;
                    data.insert((group.clone(), impl_.clone(), size.clone()), mrays);
                }
            }
        }
    }
}

fn parse_mean_ns(path: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v["mean"]["point_estimate"].as_f64()
}

// ── Formatting ────────────────────────────────────────────────────────────────

fn fmt_mrays(v: Option<f64>) -> String {
    match v {
        None    => format!("{:>8}", "-"),
        Some(v) => format!("{:>8.1}", v),
    }
}

fn fmt_ratio(rnpt: Option<f64>, reference: Option<f64>) -> String {
    match (rnpt, reference) {
        (Some(r), Some(e)) if e > 0.0 => format!("{:>6.0}%", r / e * 100.0),
        _ => format!("{:>7}", "-"),
    }
}

fn ratio_color(rnpt: Option<f64>, reference: Option<f64>) -> &'static str {
    match (rnpt, reference) {
        (Some(r), Some(e)) if e > 0.0 => {
            let pct = r / e;
            if pct >= 0.85 { ansi(GREEN) } else if pct >= 0.65 { ansi(YELLOW) } else { ansi(RED) }
        }
        _ => "",
    }
}

// ── Table ─────────────────────────────────────────────────────────────────────

const RAY_ORDER:   &[&str] = &["coherent", "incoherent", "shadow"];
const SCENE_ORDER: &[&str] = &["hf", "cluster"];
// (lookup key as written by Criterion on disk, display label)
const SIZE_ORDER:  &[(&str, &str)] = &[("100k", "100k"), ("1m", "1M"), ("10m", "10M")];
const SEP: &str  = "────────────────────────────────────────────────────────────────────";
const SEP2: &str = "════════════════════════════════════════════════════════════════════";

fn get(data: &Data, group: &str, impl_: &str, size: &str) -> Option<f64> {
    data.get(&(group.to_owned(), impl_.to_owned(), size.to_owned())).copied()
}

fn print_table(data: &Data) {
    let header = format!(
        "{bold}{:<20}  {:>5}  {:>8}  {:>8}  {:>7}{reset}",
        "ray/scene", "size", "embree1", "rnpt", "vs e1",
        bold = ansi(BOLD), reset = ansi(RESET),
    );
    println!("{header}");

    let mut prev_ray = "";

    for &ray in RAY_ORDER {
        if !prev_ray.is_empty() {
            println!("{}", ansi(DIM).to_owned() + SEP2 + ansi(RESET));
        }
        prev_ray = ray;

        for &scene in SCENE_ORDER {
            let group = format!("{}_{}", ray, scene);

            let has_any = SIZE_ORDER.iter().any(|(key, _)| {
                get(data, &group, "rnpt", key).is_some()
                    || get(data, &group, "embree", key).is_some()
                    || get(data, &group, "embree1", key).is_some()
            });
            if !has_any { continue; }

            println!("{}", ansi(DIM).to_owned() + SEP + ansi(RESET));

            for (i, &(key, disp)) in SIZE_ORDER.iter().enumerate() {
                let e1 = if ray == "coherent" {
                    get(data, &group, "embree1", key)
                } else {
                    get(data, &group, "embree", key)
                };
                let rnpt = get(data, &group, "rnpt", key);

                if e1.is_none() && rnpt.is_none() { continue; }

                let label = if i == 0 { format!("{}/{}", ray, scene) } else { String::new() };
                let color = ratio_color(rnpt, e1);
                let reset = if color.is_empty() { "" } else { ansi(RESET) };

                println!(
                    "{color}{:<20}  {:>5}  {}  {}  {}{reset}",
                    label, disp,
                    fmt_mrays(e1), fmt_mrays(rnpt),
                    fmt_ratio(rnpt, e1),
                    color = color, reset = reset,
                );
            }
        }
    }
    println!("{}", ansi(DIM).to_owned() + SEP + ansi(RESET));
    println!();
    println!("  {}Mray/s — higher is better.  vs e1 = rnpt ÷ embree (100% = parity).{}", ansi(DIM), ansi(RESET));
    println!("  {}green ≥ 85%   yellow ≥ 65%   red < 65%{}", ansi(DIM), ansi(RESET));
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let base = criterion_dir().unwrap_or_else(|| {
        eprintln!("error: target/criterion not found. Run benchmarks first:");
        eprintln!("  cargo bench -p rnpt-bench --features embree");
        std::process::exit(1);
    });

    let data = load_data(&base);
    if data.is_empty() {
        eprintln!("error: no estimates.json files found in {}", base.display());
        std::process::exit(1);
    }

    print_table(&data);
}
