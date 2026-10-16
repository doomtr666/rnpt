//! REPORT.md and results.csv, rebuilt from the cached results only.

use std::collections::BTreeSet;
use std::fs;

use crate::inputs::SETS;
use crate::runner::{load_results, CommitResults, Entry};
use crate::{commit_log, Ctx};

pub fn write(ctx: &Ctx) -> Result<(), String> {
    let dir = ctx.results_dir();
    let order: Vec<String> = commit_log(&ctx.repo, ctx.rev(), &[])?.into_iter().map(|c| c.hash).collect();

    let mut all: Vec<CommitResults> = fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| load_results(&e.path()))
        .collect();
    all.sort_by_key(|r| order.iter().position(|h| *h == r.commit).unwrap_or(usize::MAX));

    let scenes: BTreeSet<&str> = all.iter().flat_map(|r| r.entries.iter().map(|e| e.scene.as_str())).collect();
    let find = |r: &'_ CommitResults, scene: &str, set: &str| -> Option<Entry> {
        r.entries.iter().find(|e| e.scene == scene && e.set == set).cloned()
    };

    let mut md = format!(
        "# BVH history — {} — build config: {}\n\n\
         Single-thread throughput in Mrays/s: median over processes of the median pass. \
         ✗ = results disagree with the brute-force reference; \
         † = no occlusion query yet, shadow rays use closest-hit.\n",
        ctx.machine,
        ctx.mode.name()
    );
    for scene in &scenes {
        md.push_str(&format!("\n## {scene}\n\n| commit | date | subject |"));
        for (set, _) in SETS {
            md.push_str(&format!(" {set} |"));
        }
        md.push_str(" build ms |\n|---|---|---|");
        for _ in SETS {
            md.push_str("---:|");
        }
        md.push_str("---:|\n");

        for r in &all {
            let entries: Vec<Option<Entry>> = SETS.iter().map(|(set, _)| find(r, scene, set)).collect();
            if entries.iter().all(Option::is_none) {
                continue;
            }
            md.push_str(&format!("| `{}` | {} | {} |", r.short, r.date, r.subject.replace('|', "\\|")));
            for e in &entries {
                md.push_str(&match e {
                    Some(e) => format!(
                        " {:.1}{}{} |",
                        e.mrays,
                        if e.check.ok { "" } else { " ✗" },
                        if e.native_occlusion || e.set != "shadow" { "" } else { " †" }
                    ),
                    None => " – |".into(),
                });
            }
            let build = entries.iter().flatten().next().map_or("–".into(), |e| format!("{:.0}", e.build_ms));
            md.push_str(&format!(" {build} |\n"));
        }
    }

    let mut csv = String::from(
        "commit,date,scene,set,mrays,mrays_min,mrays_max,median_ns,spread_pct,build_ms,check_ok,hit_miss,t_mismatch,native_occlusion,variant,rustc,mode,rustflags\n",
    );
    for r in &all {
        for e in &r.entries {
            csv.push_str(&format!(
                "{},{},{},{},{:.3},{:.3},{:.3},{},{:.2},{:.1},{},{},{},{},{},\"{}\",{},\"{}\"\n",
                r.short, r.date, e.scene, e.set, e.mrays, e.mrays_min, e.mrays_max, e.median_ns, e.spread_pct, e.build_ms, e.check.ok,
                e.check.hit_miss, e.check.t_mismatch, e.native_occlusion, e.variant, e.run.rustc, e.run.mode,
                e.run.rustflags.replace('"', "'")
            ));
        }
    }

    fs::write(dir.join("REPORT.md"), md).map_err(|e| e.to_string())?;
    fs::write(dir.join("results.csv"), csv).map_err(|e| e.to_string())?;
    eprintln!("report: {}", dir.join("REPORT.md").display());
    Ok(())
}
