//! Counts what the BVH does, instead of timing it.
//!
//! Replays history-bench's cached scenes and ray sets on the working tree's rnpt-bvh,
//! built with its `stats` feature: nodes visited, triangle chunks tested, useful SIMD
//! lanes, per ray, next to what the SAH cost model predicts for the same tree. Counts do
//! not depend on the machine's load; the time column does, and is only indicative.
//! Results are checked against history-bench's brute-force references.
//!
//! Built without the default `count` feature, it times instead: no counters, pinned to
//! one core, median of 9 passes after a warm-up, like history-bench.
//!
//! With counts on, `--working-set` measures the cache lines rays touch per window of
//! consecutive rays, and `--save-visits DIR` records how often each node is visited (one ray
//! in four). `--relayout DIR` rearranges each tree by those visits before measuring.
//!
//!   bvh-stats [--scenes a,b] [--cache DIR] [--json FILE] [--working-set]
//!             [--save-visits DIR] [--relayout DIR]

use hb_common::*;
#[cfg(feature = "count")]
use rnpt_bvh::stats;
use rnpt_bvh::{Geometry, Ray, RayAccelerator, Scene};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

const SCENES: &[&str] = &["hf-100k", "hf-1m", "cluster-100k", "cluster-1m", "sponza", "bistro-ext"];
const SETS: &[&str] = &["primary", "diffuse", "shadow", "ortho"];

fn main() {
    let mut scenes: Vec<String> = SCENES.iter().map(|s| s.to_string()).collect();
    let mut cache = default_cache();
    let mut json_out: Option<PathBuf> = None;
    let mut working_set = false;
    let mut save_visits: Option<PathBuf> = None;
    let mut relayout: Option<PathBuf> = None;
    // Shadow rays: learn the occluder order on all rays, on even rays (then measure odd ones),
    // or on the first half (then measure the second).
    let mut learn_shadows: Option<String> = None;
    // Learn the shadow order and apply it before measuring every set (closest-hit included).
    let mut learn_first = false;
    let mut separation = false;
    // Regroup triangles into chunks guided by half the rays (even ones), with this prior weight.
    let mut refine_prior: Option<f32> = None;
    // Measure odd rays only: what was learned (from even rays) is then judged on unseen rays.
    let mut measure_odd = false;
    let mut args = std::env::args().skip(1);
    while let Some(k) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| fail(&format!("{k} needs a value")));
        match k.as_str() {
            "--scenes" => scenes = val().split(',').map(str::to_string).collect(),
            "--cache" => cache = PathBuf::from(val()),
            "--json" => json_out = Some(PathBuf::from(val())),
            "--working-set" => working_set = true,
            "--save-visits" => save_visits = Some(PathBuf::from(val())),
            "--relayout" => relayout = Some(PathBuf::from(val())),
            "--learn-shadows" => learn_shadows = Some(val()),
            "--learn-shadows-first" => learn_first = true,
            "--separation" => separation = true,
            "--measure-odd" => measure_odd = true,
            "--refine-by-rays" => refine_prior = Some(val().parse().unwrap_or_else(|_| fail("bad --refine-by-rays"))),
            _ => fail("usage: bvh-stats [--scenes a,b] [--cache DIR] [--json FILE] [--working-set] [--save-visits DIR] [--relayout DIR]"),
        }
    }
    let inputs = cache.join("inputs");
    eprintln!(
        "SIMD: avx2 {}, fma {}, avx512f {}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma"),
        cfg!(target_feature = "avx512f")
    );
    if !cfg!(feature = "count") && !pin_to_core(2) {
        eprintln!("warning: could not pin to a core, timings will be noisier");
    }
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
        if let Some(dir) = &relayout {
            let path = dir.join(format!("{name}.visits"));
            let bytes = std::fs::read(&path).unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
            let visits: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            scene.relayout_by_visits(&visits);
            eprintln!("{name}: relaid out by {}", path.display());
        }
        if let Some(prior) = refine_prior {
            let t = Instant::now();
            let mut samples = Vec::new();
            for set_name in ["primary", "diffuse", "shadow"] {
                let path = inputs.join(format!("{name}.{set_name}.rays"));
                if !path.exists() {
                    continue;
                }
                for r in read_rays(&path).rays.iter().step_by(2) {
                    let ray = Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax);
                    let t_end = scene.closest_hit(&ray).map_or(r.tmax, |h| h.t);
                    samples.push(rnpt_bvh::RaySample { org: r.org, dir: r.dir, tmin: r.tmin, t_end });
                }
            }
            let sample_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let rep = scene.refine_chunks_by_rays(&samples, prior).unwrap();
            eprintln!(
                "{name}: refined by {} rays (prior {prior}, area weight {:.3e}): entries {} -> {} ({:+.1} %), sampling {sample_ms:.0} ms, refining + rebuilding {:.0} ms",
                rep.samples,
                rep.area_weight,
                rep.entries_before,
                rep.entries_after,
                100.0 * (rep.entries_after as f64 / rep.entries_before.max(1) as f64 - 1.0),
                t.elapsed().as_secs_f64() * 1e3
            );
            #[cfg(feature = "count")]
            stats::take();
        }
        if learn_first {
            let set = read_rays(&inputs.join(format!("{name}.shadow.rays")));
            scene.set_learn_occluders(true);
            for r in set.rays.iter().step_by(if measure_odd { 2 } else { 1 }) {
                std::hint::black_box(scene.any_hit(&Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax)));
            }
            scene.apply_occluder_order();
            scene.set_learn_occluders(false);
            #[cfg(feature = "count")]
            stats::take();
        }
        let tree = scene.tree_stats().unwrap();

        println!("\n## {name}\n");
        println!(
            "{} triangles in {} chunks ({:.1} % of lanes used), {} inner nodes ({:.1} children each), \
             {} leaves (≤ {} chunks), depth {}, {:.1} MB nodes + {:.1} MB triangles, built in {build_ms:.0} ms",
            tree.triangles,
            tree.chunks,
            100.0 * tree.triangles as f64 / (8 * tree.chunks) as f64,
            tree.inner_nodes,
            tree.used_slots as f64 / tree.inner_nodes as f64,
            tree.leaves,
            tree.max_leaf_chunks,
            tree.max_depth,
            tree.node_bytes as f64 / 1e6,
            tree.chunk_bytes as f64 / 1e6,
        );
        println!(
            "SAH prediction for uniform rays: {:.1} nodes, {:.1} chunks per ray\n",
            tree.sah_nodes, tree.sah_chunks
        );
        if cfg!(feature = "count") {
            println!("| set | nodes/ray | boxes hit/ray | culled/ray | chunks/ray | lanes used | nodes vs SAH | chunks vs SAH | ns/ray* | check |");
            println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");
        } else {
            println!("| set | ns/ray | Mrays/s | spread | check |");
            println!("|---|---:|---:|---:|---|");
        }

        let mut sets_json = String::new();
        for set_name in SETS {
            let rays_path = inputs.join(format!("{name}.{set_name}.rays"));
            if !rays_path.exists() {
                continue;
            }
            let set = read_rays(&rays_path);
            let mut rays: Vec<Ray> = set.rays.iter().map(|r| Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax)).collect();
            let mut reference = read_f32s(&inputs.join(format!("{name}.{set_name}.ref")));
            if measure_odd {
                rays = set.rays.iter().skip(1).step_by(2).map(|r| Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax)).collect();
                reference = reference.iter().skip(1).step_by(2).copied().collect();
            }
            // One learning pass over the shadow rays, then the learned order made physical,
            // then plain traversal: like a progressive render retracing the same pixels.
            if let (Some(mode), QueryKind::Occluded) = (&learn_shadows, set.kind) {
                let n = rays.len();
                // "skip-interleaved" / "skip-halves": same rays measured, nothing learned (baseline).
                let (skip, split) = match mode.strip_prefix("skip-") {
                    Some(m) => (true, m),
                    None => (false, mode.as_str()),
                };
                let learn = |i: usize| match split {
                    "all" => true,
                    "interleaved" => i % 2 == 0,
                    "halves" => i < n / 2,
                    m => fail(&format!("--learn-shadows {m}: [skip-]all, interleaved or halves")),
                };
                if !skip {
                    scene.set_learn_occluders(true);
                    let t = Instant::now();
                    let mut learned = 0;
                    for (i, r) in rays.iter().enumerate() {
                        if learn(i) {
                            std::hint::black_box(scene.any_hit(r));
                            learned += 1;
                        }
                    }
                    let learn_ms = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    scene.apply_occluder_order();
                    let apply_ms = t.elapsed().as_secs_f64() * 1e3;
                    scene.set_learn_occluders(false);
                    eprintln!(
                        "{name}/{set_name}: learning {learned} rays {learn_ms:.1} ms ({:.0} ns/ray), applying {apply_ms:.1} ms",
                        learn_ms * 1e6 / learned as f64
                    );
                }
                // Measure only rays not learned from (all of them for "all").
                if split != "all" {
                    let keep: Vec<usize> = (0..n).filter(|&i| !learn(i)).collect();
                    reference = keep.iter().filter(|&&i| i < reference.len()).map(|&i| reference[i]).collect();
                    rays = keep.iter().map(|&i| { let r = &set.rays[i]; Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax) }).collect();
                }
                #[cfg(feature = "count")]
                stats::take();
            }

            let run = || -> Vec<f32> {
                rays.iter()
                    .map(|r| match set.kind {
                        QueryKind::Closest => scene.closest_hit(r).map_or(f32::INFINITY, |h| h.t),
                        QueryKind::Occluded => scene.any_hit(r) as u32 as f32,
                    })
                    .collect()
            };
            #[cfg(feature = "count")]
            stats::take();
            let t = Instant::now();
            // With counts on, shadow rays are counted apart when occluded: what learning
            // occluders should speed up, against what it must not slow down.
            #[cfg(feature = "count")]
            let mut occluded = (0u64, 0u64, 0u64);
            #[cfg(feature = "count")]
            let results: Vec<f32> = if set.kind == QueryKind::Occluded {
                let mut before = stats::take();
                rays.iter()
                    .map(|r| {
                        let hit = scene.any_hit(r);
                        let now = stats::take();
                        if hit {
                            occluded.0 += 1;
                            occluded.1 += now.nodes;
                            occluded.2 += now.chunks;
                        }
                        before = add(before, now);
                        hit as u32 as f32
                    })
                    .collect::<Vec<_>>()
                    .tap(|_| restore(before))
            } else {
                run()
            };
            #[cfg(not(feature = "count"))]
            let results = run();
            #[allow(unused_mut)]
            let mut ns = t.elapsed().as_nanos() as f64 / rays.len() as f64;
            let wrong = mismatches(&results, &reference, set.kind, diag);
            let check = if wrong as f64 <= 1e-3 * reference.len() as f64 {
                "ok".to_string()
            } else {
                format!("✗ {wrong}/{}", reference.len())
            };

            #[cfg(feature = "count")]
            {
                let s = stats::take();
                let per_ray = |v: u64| v as f64 / s.rays as f64;
                if occluded.0 > 0 {
                    let (n, s_rays) = (occluded.0 as f64, s.rays as f64);
                    let other = s_rays - n;
                    println!(
                        "| ↳ {set_name}, occluded ({:.1} %) | {:.1} |  |  | {:.1} |  |  |  |  | unoccluded: {:.1} nodes, {:.1} chunks |",
                        100.0 * n / s_rays,
                        occluded.1 as f64 / n,
                        occluded.2 as f64 / n,
                        (s.nodes - occluded.1) as f64 / other,
                        (s.chunks - occluded.2) as f64 / other,
                    );
                }
                let lanes = 100.0 * s.tri_lanes as f64 / (8 * s.chunks.max(1)) as f64;
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
            #[cfg(not(feature = "count"))]
            {
                let mut passes: Vec<f64> = (0..9)
                    .map(|_| {
                        let t = Instant::now();
                        std::hint::black_box(run());
                        t.elapsed().as_nanos() as f64 / rays.len() as f64
                    })
                    .collect();
                passes.sort_by(f64::total_cmp);
                ns = passes[passes.len() / 2];
                let spread = 100.0 * (passes[passes.len() - 1] - passes[0]) / ns;
                println!("| {set_name} | {ns:.1} | {:.2} | {spread:.1} % | {check} |", 1e3 / ns);
                let _ = write!(
                    sets_json,
                    "{}{{\"set\":\"{set_name}\",\"rays\":{},\"ns_per_ray\":{ns:.2},\"spread_pct\":{spread:.2},\"wrong\":{wrong}}}",
                    if sets_json.is_empty() { "" } else { "," },
                    rays.len()
                );
            }
        }
        #[cfg(feature = "count")]
        if working_set || save_visits.is_some() {
            explore(&scene, &inputs, name, save_visits.as_deref());
        }
        #[cfg(feature = "count")]
        if separation {
            separate(&scene, &inputs, name);
        }
        #[cfg(not(feature = "count"))]
        let _ = (working_set, &save_visits, separation);

        let _ = write!(
            json,
            "{}{{\"scene\":\"{name}\",\"build_ms\":{build_ms:.1},\"triangles\":{},\"chunks\":{},\"inner_nodes\":{},\"leaves\":{},\"used_slots\":{},\"max_depth\":{},\"node_bytes\":{},\"chunk_bytes\":{},\"sah_nodes\":{},\"sah_chunks\":{},\"sets\":[{sets_json}]}}",
            if json.len() > 1 { "," } else { "" },
            tree.triangles, tree.chunks, tree.inner_nodes, tree.leaves, tree.used_slots, tree.max_depth,
            tree.node_bytes, tree.chunk_bytes, tree.sah_nodes, tree.sah_chunks
        );
    }
    if cfg!(feature = "count") {
        println!("\n*single thread, unpinned, with the counters on: indicative only, unlike the counts.");
    }

    json.push(']');
    if let Some(path) = json_out {
        std::fs::write(&path, json).unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
    }
}

/// Rays per window: about one 32×32 tile, the unit a render thread works through.
#[cfg(feature = "count")]
const WINDOW: usize = 1024;

/// Working set (distinct cache lines touched per window of consecutive rays, against the
/// 48 KB L1 and 1 MB L2 of a Zen 5 core), and node visits sampled on one ray in four.
#[cfg(feature = "count")]
fn explore(scene: &Scene, inputs: &std::path::Path, name: &str, save: Option<&std::path::Path>) {
    use rnpt_bvh::stats::{trace_start, trace_stop, trace_take, Touch};
    use std::collections::HashSet;
    // Nodes are 80 bytes, chunks 320 (5 lines), both arrays from line boundaries.
    let lines = |t: Touch, out: &mut HashSet<u64>| match t {
        Touch::Node(i) => {
            let a = i as u64 * 80;
            for l in a / 64..=(a + 79) / 64 {
                out.insert(l);
            }
        }
        Touch::Chunk(c) => {
            for l in 0..5 {
                out.insert((1 << 62) | (c as u64 * 5 + l));
            }
        }
        Touch::Answer(_) => {}
    };
    let kb = |n: usize| n as f64 * 64.0 / 1024.0;
    let mut visits = vec![0u32; scene.node_count()];
    println!("
| set | working set per {WINDOW} rays, median | p90 | max | whole set |");
    println!("|---|---:|---:|---:|---:|");
    trace_start();
    for set_name in SETS {
        let rays_path = inputs.join(format!("{name}.{set_name}.rays"));
        if !rays_path.exists() {
            continue;
        }
        let set = read_rays(&rays_path);
        let (mut window, mut whole) = (HashSet::new(), HashSet::new());
        let mut sizes = Vec::new();
        for (i, r) in set.rays.iter().enumerate() {
            let ray = Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax);
            match set.kind {
                QueryKind::Closest => drop(scene.closest_hit(&ray)),
                QueryKind::Occluded => drop(scene.any_hit(&ray)),
            }
            for t in trace_take() {
                if let (Touch::Node(n), 0) = (t, i % 4) {
                    visits[n as usize] += 1;
                }
                lines(t, &mut window);
                lines(t, &mut whole);
            }
            if (i + 1) % WINDOW == 0 {
                sizes.push(window.len());
                window.clear();
            }
        }
        sizes.sort_unstable();
        let at = |q: f64| kb(sizes[((sizes.len() - 1) as f64 * q) as usize]);
        println!(
            "| {set_name} | {:.0} KB | {:.0} KB | {:.0} KB | {:.1} MB |",
            at(0.5),
            at(0.9),
            at(1.0),
            kb(whole.len()) / 1024.0
        );
    }
    trace_stop();
    if let Some(dir) = save {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| fail(&format!("{}: {e}", dir.display())));
        let bytes: Vec<u8> = visits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.visits")), bytes).unwrap_or_else(|e| fail(&e.to_string()));
    }
}

#[cfg(feature = "count")]
fn add(a: rnpt_bvh::stats::TraversalStats, b: rnpt_bvh::stats::TraversalStats) -> rnpt_bvh::stats::TraversalStats {
    rnpt_bvh::stats::TraversalStats {
        rays: a.rays + b.rays,
        nodes: a.nodes + b.nodes,
        boxes_hit: a.boxes_hit + b.boxes_hit,
        culled: a.culled + b.culled,
        chunks: a.chunks + b.chunks,
        tri_lanes: a.tri_lanes + b.tri_lanes,
    }
}

/// Puts accumulated counts back, for the caller's `stats::take`.
#[cfg(feature = "count")]
fn restore(s: rnpt_bvh::stats::TraversalStats) {
    rnpt_bvh::stats::put_back(s);
}

#[cfg(feature = "count")]
trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}
#[cfg(feature = "count")]
impl<T> Tap for T {}

/// How well the tree separates what is useful from what is not. A visit is useful when the
/// node lies on the path from the root to the chunk holding the answer (closest hit, or the
/// occluder found); every other visit is a false positive of a box test. Boxes never miss what
/// they contain, so false positives are the whole story. Rays without an answer (miss,
/// unoccluded shadow ray) are counted apart: their traversal is the proof that nothing is there.
#[cfg(feature = "count")]
fn separate(scene: &Scene, inputs: &std::path::Path, name: &str) {
    use rnpt_bvh::stats::{trace_start, trace_stop, trace_take, Touch};
    let (parent, owner) = scene.parent_maps();
    let mut depth = vec![0usize; parent.len()];
    for n in 1..parent.len() {
        depth[n] = depth[parent[n] as usize] + 1; // parents are laid out before children
    }
    let max_depth = depth.iter().copied().max().unwrap_or(0);
    println!("
| set | rays with answer | nodes/ray | useful | chunks/ray | useful | without answer: nodes, chunks | useful share by depth (1, 2, ...) |");
    println!("|---|---:|---:|---:|---:|---:|---|---|");
    trace_start();
    for set_name in SETS {
        let rays_path = inputs.join(format!("{name}.{set_name}.rays"));
        if !rays_path.exists() {
            continue;
        }
        let set = read_rays(&rays_path);
        let (mut answered, mut nodes, mut useful_nodes, mut chunks) = (0u64, 0u64, 0u64, 0u64);
        let (mut miss_nodes, mut miss_chunks) = (0u64, 0u64);
        let mut by_depth = vec![(0u64, 0u64); max_depth + 1];
        let mut path = Vec::new();
        for r in &set.rays {
            let ray = Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax);
            match set.kind {
                QueryKind::Closest => drop(scene.closest_hit(&ray)),
                QueryKind::Occluded => drop(scene.any_hit(&ray)),
            }
            let touched = trace_take();
            let answer = touched.iter().rev().find_map(|t| if let Touch::Answer(c) = t { Some(*c) } else { None });
            let Some(chunk) = answer else {
                for t in &touched {
                    match t {
                        Touch::Node(_) => miss_nodes += 1,
                        Touch::Chunk(_) => miss_chunks += 1,
                        Touch::Answer(_) => {}
                    }
                }
                continue;
            };
            answered += 1;
            path.clear();
            let mut n = owner[chunk as usize];
            while n != u32::MAX {
                path.push(n);
                n = parent[n as usize];
            }
            for t in &touched {
                match *t {
                    Touch::Node(n) => {
                        let useful = path.contains(&n);
                        nodes += 1;
                        useful_nodes += useful as u64;
                        let d = &mut by_depth[depth[n as usize]];
                        d.0 += 1;
                        d.1 += useful as u64;
                    }
                    Touch::Chunk(_) => chunks += 1,
                    Touch::Answer(_) => {}
                }
            }
        }
        let a = answered.max(1) as f64;
        let m = (set.rays.len() as u64 - answered).max(1) as f64;
        let shares: Vec<String> = by_depth[1..]
            .iter()
            .filter(|d| d.0 > 0)
            .map(|d| format!("{:.0}", 100.0 * d.1 as f64 / d.0 as f64))
            .collect();
        println!(
            "| {set_name} | {:.1} % | {:.1} | {:.0} % | {:.1} | {:.0} % | {:.1}, {:.1} | {} |",
            100.0 * answered as f64 / set.rays.len() as f64,
            nodes as f64 / a,
            100.0 * useful_nodes as f64 / nodes.max(1) as f64,
            chunks as f64 / a,
            100.0 * answered as f64 / chunks.max(1) as f64,
            miss_nodes as f64 / m,
            miss_chunks as f64 / m,
            shares.join(" · ")
        );
    }
    trace_stop();
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
