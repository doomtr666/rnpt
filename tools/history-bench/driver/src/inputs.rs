//! Benchmark inputs: scenes, ray sets and brute-force references.
//!
//! Everything is generated once into the cache and keyed by content hash, so
//! every commit is measured on byte-identical inputs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use hb_common::*;
use rnpt_bvh::RayAccelerator;
use serde::{Deserialize, Serialize};

use crate::Ctx;

/// Bump when a generator changes: invalidates inputs and every cached result.
pub const INPUTS_VERSION: u32 = 1;

pub enum Source {
    Heightfield(usize),
    Cluster(usize),
    Gltf(&'static str),
}

pub struct SceneSpec {
    pub name: &'static str,
    pub source: Source,
    /// Only included with `--big`.
    pub big: bool,
}

pub const SCENES: &[SceneSpec] = &[
    SceneSpec { name: "hf-100k", source: Source::Heightfield(100_000), big: false },
    SceneSpec { name: "hf-1m", source: Source::Heightfield(1_000_000), big: false },
    SceneSpec { name: "hf-10m", source: Source::Heightfield(10_000_000), big: true },
    SceneSpec { name: "cluster-100k", source: Source::Cluster(100_000), big: false },
    SceneSpec { name: "cluster-1m", source: Source::Cluster(1_000_000), big: false },
    SceneSpec { name: "cluster-10m", source: Source::Cluster(10_000_000), big: true },
    SceneSpec { name: "sponza", source: Source::Gltf("assets/sponza.glb"), big: false },
    SceneSpec { name: "bistro-ext", source: Source::Gltf("assets/bistro_ext.glb"), big: false },
];

pub const SETS: &[(&str, QueryKind)] = &[
    ("primary", QueryKind::Closest),
    ("diffuse", QueryKind::Closest),
    ("shadow", QueryKind::Occluded),
    // Parallel rays straight down: the workload of the June benchmark (18cc184).
    ("ortho", QueryKind::Closest),
];

#[derive(Serialize, Deserialize)]
pub struct SceneMeta {
    pub name: String,
    pub inputs_version: u32,
    pub verts: usize,
    pub tris: usize,
    pub diag: f32,
    pub rays: usize,
    pub check: usize,
    pub scene_hash: String,
    pub set_hashes: BTreeMap<String, String>,
}

impl SceneMeta {
    /// Identifies the exact inputs of one (scene, ray set) measurement.
    pub fn input_key(&self, set: &str) -> String {
        format!("{}:{}:{}", self.inputs_version, self.scene_hash, self.set_hashes[set])
    }
}

pub fn inputs_dir(ctx: &Ctx) -> PathBuf {
    ctx.cache.join("inputs")
}

/// Load or generate one scene with its ray sets and references.
/// Returns `None` when the scene's source asset is missing.
pub fn prepare(ctx: &Ctx, spec: &SceneSpec) -> Result<Option<SceneMeta>, String> {
    let dir = inputs_dir(ctx);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let meta_path = dir.join(format!("{}.meta.json", spec.name));

    if let Ok(text) = fs::read_to_string(&meta_path) {
        if let Ok(meta) = serde_json::from_str::<SceneMeta>(&text) {
            let complete = SETS.iter().all(|(s, _)| meta.set_hashes.contains_key(*s));
            if meta.inputs_version == INPUTS_VERSION && meta.rays == ctx.rays && meta.check == ctx.check && complete {
                return Ok(Some(meta));
            }
        }
    }

    let scene = match &spec.source {
        Source::Heightfield(n) => heightfield(*n),
        Source::Cluster(n) => sphere_cluster(*n),
        Source::Gltf(rel) => {
            let path = ctx.repo.join(rel);
            if !path.exists() {
                return Ok(None);
            }
            eprintln!("[inputs] importing {}", path.display());
            match import_gltf(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[inputs] {}: import failed, skipped: {e}", spec.name);
                    return Ok(None);
                }
            }
        }
    };
    eprintln!("[inputs] {}: {} triangles, generating rays", spec.name, scene.tris.len());

    let scene_path = dir.join(format!("{}.scene", spec.name));
    write_scene(&scene_path, &scene);
    let scene_hash = file_hash(&scene_path)?;

    let (sets, diag) = generate_rays(&scene, ctx.rays);
    let mut set_hashes = BTreeMap::new();
    for ((name, _), set) in SETS.iter().zip(&sets) {
        let rays_path = dir.join(format!("{}.{name}.rays", spec.name));
        write_rays(&rays_path, set);
        set_hashes.insert(name.to_string(), file_hash(&rays_path)?);

        eprintln!("[inputs] {}/{name}: brute-force reference on {} rays", spec.name, ctx.check.min(set.rays.len()));
        let reference = brute_force(&scene, set, ctx.check);
        write_f32s(&dir.join(format!("{}.{name}.ref", spec.name)), &reference);
    }

    let meta = SceneMeta {
        name: spec.name.to_string(),
        inputs_version: INPUTS_VERSION,
        verts: scene.verts.len(),
        tris: scene.tris.len(),
        diag,
        rays: ctx.rays,
        check: ctx.check,
        scene_hash,
        set_hashes,
    };
    fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
    Ok(Some(meta))
}

fn file_hash(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{:016x}", fnv1a(&bytes)))
}

// ── Small math ───────────────────────────────────────────────────────────────

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / dot(a, a).sqrt())
}

/// SplitMix64: tiny, deterministic, no dependency whose output could drift.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f32()
    }
}

fn tri_normal(s: &TriScene, t: [u32; 3]) -> V3 {
    let v0 = s.verts[t[0] as usize];
    cross(sub(s.verts[t[1] as usize], v0), sub(s.verts[t[2] as usize], v0))
}

// ── Generated scenes (Y up) ──────────────────────────────────────────────────

fn hnoise(x: f32, z: f32) -> f32 {
    (x * 1.7 + z * 3.1).sin() * 0.40 + (x * 5.3 - z * 2.7).sin() * 0.20 + (x * 11.1 + z * 7.3).sin() * 0.10
}

fn heightfield(target_tris: usize) -> TriScene {
    let n = ((target_tris / 2) as f64).sqrt() as usize + 2;
    let mut s = TriScene { verts: Vec::with_capacity(n * n), tris: Vec::with_capacity(2 * (n - 1) * (n - 1)) };
    for r in 0..n {
        for c in 0..n {
            let x = c as f32 / (n - 1) as f32 * 2.0 - 1.0;
            let z = r as f32 / (n - 1) as f32 * 2.0 - 1.0;
            s.verts.push([x, hnoise(x, z), z]);
        }
    }
    for r in 0..n - 1 {
        for c in 0..n - 1 {
            let i00 = (r * n + c) as u32;
            let i10 = ((r + 1) * n + c) as u32;
            let i01 = (r * n + c + 1) as u32;
            let i11 = ((r + 1) * n + c + 1) as u32;
            s.tris.push([i00, i10, i01]);
            s.tris.push([i10, i11, i01]);
        }
    }
    // Front faces up.
    if tri_normal(&s, s.tris[0])[1] < 0.0 {
        s.tris.iter_mut().for_each(|t| t.swap(1, 2));
    }
    s
}

fn sphere_cluster(target_tris: usize) -> TriScene {
    const SUBDIVS: u32 = 4;
    const TRIS_PER_SPHERE: usize = 20 << (2 * SUBDIVS); // 5120
    let count = (target_tris / TRIS_PER_SPHERE).max(1);
    let mut unit = icosphere(SUBDIVS);
    // Front faces outward.
    let c0 = unit.verts[unit.tris[0][0] as usize];
    if dot(tri_normal(&unit, unit.tris[0]), c0) < 0.0 {
        unit.tris.iter_mut().for_each(|t| t.swap(1, 2));
    }

    let mut rng = Rng(0xbabe_cafe_1234);
    let mut s = TriScene { verts: vec![], tris: vec![] };
    for _ in 0..count {
        let c = [rng.range(-0.8, 0.8), rng.range(-0.8, 0.8), rng.range(-0.8, 0.8)];
        let r = rng.range(0.05, 0.25);
        let base = s.verts.len() as u32;
        s.verts.extend(unit.verts.iter().map(|v| add(c, scale(*v, r))));
        s.tris.extend(unit.tris.iter().map(|t| [base + t[0], base + t[1], base + t[2]]));
    }
    s
}

fn icosphere(subdivs: u32) -> TriScene {
    let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut verts: Vec<V3> = [
        [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ]
    .into_iter()
    .map(normalize)
    .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..subdivs {
        let mut cache: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<V3>| {
            *cache.entry((a.min(b), a.max(b))).or_insert_with(|| {
                verts.push(normalize(scale(add(verts[a as usize], verts[b as usize]), 0.5)));
                verts.len() as u32 - 1
            })
        };
        let mut next = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    TriScene { verts, tris }
}

// ── glTF import (flattened to world space) ───────────────────────────────────

type M4 = [[f32; 4]; 4]; // column-major, as glTF

fn mat_mul(a: &M4, b: &M4) -> M4 {
    let mut m = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            m[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    m
}

fn mat_det3(m: &M4) -> f32 {
    dot([m[0][0], m[0][1], m[0][2]], cross([m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]))
}

fn import_gltf(path: &Path) -> Result<TriScene, String> {
    // Only geometry is read: skip validation, which rejects unsupported extensions
    // such as KHR_lights_punctual.
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice_without_validation(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    drop(bytes);
    let buffers = gltf::import_buffers(&document, path.parent(), blob).map_err(|e| e.to_string())?;
    let scene = document.default_scene().or_else(|| document.scenes().next()).ok_or("no scene in glTF")?;

    fn visit(node: &gltf::Node, parent: &M4, buffers: &[gltf::buffer::Data], s: &mut TriScene) {
        let m = mat_mul(parent, &node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            let flip = mat_det3(&m) < 0.0;
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|b| Some(&buffers[b.index()].0[..]));
                let Some(positions) = reader.read_positions() else { continue };
                let base = s.verts.len() as u32;
                s.verts.extend(positions.map(|p| {
                    [
                        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
                        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
                        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
                    ]
                }));
                let count = s.verts.len() as u32 - base;
                let indices: Vec<u32> = match reader.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..count).collect(),
                };
                for t in indices.chunks_exact(3) {
                    let (a, b, c) = (base + t[0], base + t[1], base + t[2]);
                    s.tris.push(if flip { [a, c, b] } else { [a, b, c] });
                }
            }
        }
        for child in node.children() {
            visit(&child, &m, buffers, s);
        }
    }

    let identity: M4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let mut s = TriScene { verts: vec![], tris: vec![] };
    for node in scene.nodes() {
        visit(&node, &identity, &buffers, &mut s);
    }
    if s.tris.is_empty() {
        return Err("no triangles in glTF".into());
    }
    Ok(s)
}

// ── Ray sets ─────────────────────────────────────────────────────────────────

/// Three workloads per scene, all from one pinhole camera looking at the scene:
/// - `primary`: camera rays in scanline order (coherent),
/// - `diffuse`: cosine-sampled bounces from the primary hits (incoherent),
/// - `shadow`: from the primary hits to an area light above the scene.
///
/// Primary hits are found with today's rnpt-bvh; that only shapes the inputs,
/// which are then frozen in the cache and shared by every commit.
fn generate_rays(scene: &TriScene, n: usize) -> (Vec<RaySet>, f32) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in &scene.verts {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    let center = scale(add(lo, hi), 0.5);
    let diag = dot(sub(hi, lo), sub(hi, lo)).sqrt();

    let side = (n as f64).sqrt() as usize;
    let eye = add(center, scale(normalize([0.55, 0.6, 0.85]), 0.9 * diag));
    let fwd = normalize(sub(center, eye));
    let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let tan = (25.0f32).to_radians().tan();

    let mut primary = Vec::with_capacity(n);
    for j in 0..side {
        for i in 0..side {
            let u = ((i as f32 + 0.5) / side as f32 * 2.0 - 1.0) * tan;
            let v = (1.0 - (j as f32 + 0.5) / side as f32 * 2.0) * tan;
            let dir = normalize(add(fwd, add(scale(right, u), scale(up, v))));
            primary.push(RawRay { org: eye, dir, tmin: 0.0, tmax: f32::INFINITY });
        }
    }
    // Pad to exactly n rays if n is not a perfect square.
    while primary.len() < n {
        primary.push(primary[primary.len() % (side * side)]);
    }

    let mut accel = rnpt_bvh::Scene::new();
    accel.attach_geometry(rnpt_bvh::Geometry::triangle_mesh(&scene.verts, &scene.tris));
    accel.commit();

    // (hit point, front-facing unit normal)
    let hits: Vec<(V3, V3)> = primary
        .iter()
        .filter_map(|r| {
            let h = accel.closest_hit(&rnpt_bvh::Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax))?;
            let p = add(r.org, scale(r.dir, h.t));
            Some((p, normalize(tri_normal(scene, scene.tris[h.prim_id as usize]))))
        })
        .collect();
    assert!(!hits.is_empty(), "camera sees no geometry");

    let eps = 1e-4 * diag;
    let mut rng = Rng(0x5eed_0f_ba11);

    let diffuse = (0..n)
        .map(|k| {
            let (p, nrm) = hits[k % hits.len()];
            let (t, b) = onb(nrm);
            let (r1, r2) = (rng.f32(), rng.f32());
            let phi = 2.0 * std::f32::consts::PI * r1;
            let (sr, cz) = (r2.sqrt(), (1.0 - r2).sqrt());
            let dir = normalize(add(add(scale(t, phi.cos() * sr), scale(b, phi.sin() * sr)), scale(nrm, cz)));
            RawRay { org: add(p, scale(nrm, eps)), dir, tmin: 0.0, tmax: f32::INFINITY }
        })
        .collect();

    let light_y = hi[1] + 0.05 * diag;
    let (hx, hz) = (0.25 * (hi[0] - lo[0]), 0.25 * (hi[2] - lo[2]));
    let shadow = (0..n)
        .map(|k| {
            let (p, nrm) = hits[k % hits.len()];
            let org = add(p, scale(nrm, eps));
            // Like a real NEE, prefer light samples above the surface.
            let mut ray = None;
            for _ in 0..8 {
                let l = [center[0] + rng.range(-hx, hx), light_y, center[2] + rng.range(-hz, hz)];
                let to = sub(l, org);
                let dist = dot(to, to).sqrt();
                let r = RawRay { org, dir: scale(to, 1.0 / dist), tmin: 0.0, tmax: dist };
                ray = Some(r);
                if dot(r.dir, nrm) > 0.0 {
                    break;
                }
            }
            ray.unwrap()
        })
        .collect();

    // Generated last and without the RNG, so adding it left the other sets unchanged.
    let top = hi[1] + 0.1 * diag;
    let ortho = (0..n)
        .map(|k| {
            let (i, j) = ((k % side) as f32 + 0.5, ((k / side) % side) as f32 + 0.5);
            let x = lo[0] + (hi[0] - lo[0]) * i / side as f32;
            let z = lo[2] + (hi[2] - lo[2]) * j / side as f32;
            RawRay { org: [x, top, z], dir: [0.0, -1.0, 0.0], tmin: 0.0, tmax: f32::INFINITY }
        })
        .collect();

    (
        vec![
            RaySet { kind: QueryKind::Closest, rays: primary },
            RaySet { kind: QueryKind::Closest, rays: diffuse },
            RaySet { kind: QueryKind::Occluded, rays: shadow },
            RaySet { kind: QueryKind::Closest, rays: ortho },
        ],
        diag,
    )
}

fn onb(n: V3) -> (V3, V3) {
    let a = if n[0].abs() > 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let t = normalize(cross(a, n));
    (t, cross(n, t))
}

// ── Brute-force reference ────────────────────────────────────────────────────

fn brute_force(scene: &TriScene, set: &RaySet, check: usize) -> Vec<f32> {
    let tris: Vec<[V3; 3]> = scene
        .tris
        .iter()
        .map(|t| [scene.verts[t[0] as usize], scene.verts[t[1] as usize], scene.verts[t[2] as usize]])
        .collect();
    let rays = &set.rays[..check.min(set.rays.len())];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = rays.len().div_ceil(threads).max(1);

    let mut out = vec![0.0f32; rays.len()];
    std::thread::scope(|s| {
        for (rs, os) in rays.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let tris = &tris;
            s.spawn(move || {
                for (r, o) in rs.iter().zip(os) {
                    *o = match set.kind {
                        QueryKind::Closest => tris
                            .iter()
                            .filter_map(|t| intersect_triangle(r, t[0], t[1], t[2]))
                            .fold(f32::INFINITY, f32::min),
                        QueryKind::Occluded => {
                            tris.iter().any(|t| intersect_triangle(r, t[0], t[1], t[2]).is_some()) as u32 as f32
                        }
                    };
                }
            });
        }
    });
    out
}
