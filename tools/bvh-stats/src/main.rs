//! Counts what the BVH does, instead of timing it.
//!
//! Replays history-bench's cached scenes and ray sets on the working tree's rnpt-bvh,
//! built with its `stats` feature: nodes visited, triangle chunks tested, useful SIMD
//! lanes, per ray, next to what the SAH cost model predicts for the same tree. Counts do
//! not depend on the machine's load; the time column does, and is only indicative.
//! Results are checked against history-bench's brute-force references.
//!
//!   bvh-stats [--scenes a,b] [--cache DIR] [--json FILE]

use hb_common::*;
use rnpt_bvh::{stats, Geometry, Ray, RayAccelerator, Scene};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

const SCENES: &[&str] = &["hf-100k", "hf-1m", "cluster-100k", "cluster-1m", "sponza", "bistro-ext"];
const SETS: &[&str] = &["primary", "diffuse", "shadow", "ortho"];

fn main() {
    let mut scenes: Vec<String> = SCENES.iter().map(|s| s.to_string()).collect();
    let mut cache = default_cache();
    let mut json_out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| fail(&format!("{k} needs a value")));
        match k.as_str() {
            "--scenes" => scenes = val().split(',').map(str::to_string).collect(),
            "--cache" => cache = PathBuf::from(val()),
            "--json" => json_out = Some(PathBuf::from(val())),
            _ => fail("usage: bvh-stats [--scenes a,b] [--cache DIR] [--json FILE]"),
        }
    }
    let inputs = cache.join("inputs");
    let mut json = String::from("[");

    for name in &scenes {
        let scene_path = inputs.join(format!("{name}.scene"));
        if !scene_path.exists() {
            eprintln!("{name}: not in the cache, skipped (run history-bench prepare)");
            continue;
        }
        let tri_scene = read_scene(&scene_path);
        let diag = read_diag(&inputs.join(format!("{name}.meta.json")));
        let t0 = Instant::now();
        let mut scene = Scene::new();
        scene.attach_geometry(Geometry::triangle_mesh(&tri_scene.verts, &tri_scene.tris));
        scene.commit();
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        let tree = scene.tree_stats().unwrap();

        println!("\n## {name}\n");
        println!(
            "{} triangles in {} chunks ({:.1} % of lanes used), {} inner nodes ({:.1} children each), \
             {} leaves, depth {}, {:.1} MB nodes + {:.1} MB triangles, built in {build_ms:.0} ms",
            tree.triangles,
            tree.chunks,
            100.0 * tree.triangles as f64 / (8 * tree.chunks) as f64,
            tree.inner_nodes,
            tree.used_slots as f64 / tree.inner_nodes as f64,
            tree.leaves,
            tree.max_depth,
            tree.node_bytes as f64 / 1e6,
            tree.chunk_bytes as f64 / 1e6,
        );
        println!(
            "SAH prediction for uniform rays: {:.1} nodes, {:.1} chunks per ray\n",
            tree.sah_nodes, tree.sah_chunks
        );
        println!("| set | nodes/ray | boxes hit/ray | culled/ray | chunks/ray | lanes used | nodes vs SAH | chunks vs SAH | ns/ray* | check |");
        println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");

        let mut sets_json = String::new();
        for set_name in SETS {
            let rays_path = inputs.join(format!("{name}.{set_name}.rays"));
            if !rays_path.exists() {
                continue;
            }
            let set = read_rays(&rays_path);
            let rays: Vec<Ray> = set.rays.iter().map(|r| Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax)).collect();
            let reference = read_f32s(&inputs.join(format!("{name}.{set_name}.ref")));

            stats::take();
            let t = Instant::now();
            let results: Vec<f32> = rays
                .iter()
                .map(|r| match set.kind {
                    QueryKind::Closest => scene.closest_hit(r).map_or(f32::INFINITY, |h| h.t),
                    QueryKind::Occluded => scene.any_hit(r) as u32 as f32,
                })
                .collect();
            let ns = t.elapsed().as_nanos() as f64 / rays.len() as f64;
            let s = stats::take();
            let wrong = mismatches(&results, &reference, set.kind, diag);

            let per_ray = |v: u64| v as f64 / s.rays as f64;
            let lanes = 100.0 * s.tri_lanes as f64 / (8 * s.chunks.max(1)) as f64;
            let check = if wrong as f64 <= 1e-3 * reference.len() as f64 {
                "ok".to_string()
            } else {
                format!("✗ {wrong}/{}", reference.len())
            };
            println!(
                "| {set_name} | {:.1} | {:.1} | {:.1} | {:.1} | {lanes:.1} % | {:.2}× | {:.2}× | {ns:.0} | {check} |",
                per_ray(s.nodes),
                per_ray(s.boxes_hit),
                per_ray(s.culled),
                per_ray(s.chunks),
                per_ray(s.nodes) / tree.sah_nodes,
                per_ray(s.chunks) / tree.sah_chunks,
            );
            let _ = write!(
                sets_json,
                "{}{{\"set\":\"{set_name}\",\"rays\":{},\"nodes\":{},\"boxes_hit\":{},\"culled\":{},\"chunks\":{},\"tri_lanes\":{},\"ns_per_ray\":{ns:.1},\"wrong\":{wrong}}}",
                if sets_json.is_empty() { "" } else { "," },
                s.rays, s.nodes, s.boxes_hit, s.culled, s.chunks, s.tri_lanes
            );
        }
        let _ = write!(
            json,
            "{}{{\"scene\":\"{name}\",\"build_ms\":{build_ms:.1},\"triangles\":{},\"chunks\":{},\"inner_nodes\":{},\"leaves\":{},\"used_slots\":{},\"max_depth\":{},\"node_bytes\":{},\"chunk_bytes\":{},\"sah_nodes\":{},\"sah_chunks\":{},\"sets\":[{sets_json}]}}",
            if json.len() > 1 { "," } else { "" },
            tree.triangles, tree.chunks, tree.inner_nodes, tree.leaves, tree.used_slots, tree.max_depth,
            tree.node_bytes, tree.chunk_bytes, tree.sah_nodes, tree.sah_chunks
        );
    }
    println!("\n*single thread, unpinned, with the counters on: indicative only, unlike the counts.");

    json.push(']');
    if let Some(path) = json_out {
        std::fs::write(&path, json).unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
    }
}

/// Same tolerance as history-bench: hit/miss must agree, distances within 1e-4 relative.
fn mismatches(got: &[f32], reference: &[f32], kind: QueryKind, diag: f32) -> usize {
    got.iter()
        .zip(reference)
        .filter(|&(&g, &r)| match kind {
            QueryKind::Occluded => g != r,
            QueryKind::Closest => match (g.is_finite(), r.is_finite()) {
                (true, true) => (g - r).abs() > 1e-4 * r.abs() + 1e-5 * diag,
                (a, b) => a != b,
            },
        })
        .count()
}

fn read_diag(path: &std::path::Path) -> f32 {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.split("\"diag\":")
        .nth(1)
        .and_then(|rest| rest.split([',', '}']).next())
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1.0)
}

fn default_cache() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("rnpt-history-bench")
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}
