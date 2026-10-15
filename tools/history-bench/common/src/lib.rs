//! Code shared by the history-bench driver and the per-commit adapters.
//!
//! This crate is compiled inside every historical checkout of rnpt, so it must
//! stay dependency-free and stable: changing it invalidates every cached result.

use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

pub const SCENE_MAGIC: &[u8; 4] = b"HBSC";
pub const RAYS_MAGIC: &[u8; 4] = b"HBRY";
pub const FORMAT_VERSION: u32 = 1;

// ── Data ─────────────────────────────────────────────────────────────────────

/// A flat, world-space triangle soup.
pub struct TriScene {
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryKind {
    Closest,
    Occluded,
}

#[derive(Clone, Copy, Debug)]
pub struct RawRay {
    pub org: [f32; 3],
    pub dir: [f32; 3],
    pub tmin: f32,
    pub tmax: f32,
}

pub struct RaySet {
    pub kind: QueryKind,
    pub rays: Vec<RawRay>,
}

// ── Binary I/O (little-endian) ───────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.bytes(8).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn header(&mut self, magic: &[u8; 4], path: &Path) {
        let m = self.bytes(4);
        let v = self.u32();
        assert!(m == magic && v == FORMAT_VERSION, "{}: bad header", path.display());
    }
}

pub fn write_scene(path: &Path, scene: &TriScene) {
    let mut out = Vec::with_capacity(24 + scene.verts.len() * 12 + scene.tris.len() * 12);
    out.extend_from_slice(SCENE_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(scene.verts.len() as u64).to_le_bytes());
    out.extend_from_slice(&(scene.tris.len() as u64).to_le_bytes());
    for v in &scene.verts {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    for t in &scene.tris {
        for i in t {
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
    fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

pub fn read_scene(path: &Path) -> TriScene {
    let data = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut r = Reader { data: &data, pos: 0 };
    r.header(SCENE_MAGIC, path);
    let nv = r.u64() as usize;
    let nt = r.u64() as usize;
    let verts = (0..nv).map(|_| [r.f32(), r.f32(), r.f32()]).collect();
    let tris = (0..nt).map(|_| [r.u32(), r.u32(), r.u32()]).collect();
    TriScene { verts, tris }
}

pub fn write_rays(path: &Path, set: &RaySet) {
    let mut out = Vec::with_capacity(20 + set.rays.len() * 32);
    out.extend_from_slice(RAYS_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    let kind: u32 = match set.kind {
        QueryKind::Closest => 0,
        QueryKind::Occluded => 1,
    };
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&(set.rays.len() as u64).to_le_bytes());
    for r in &set.rays {
        for c in r.org.iter().chain(&r.dir).chain([&r.tmin, &r.tmax]) {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

pub fn read_rays(path: &Path) -> RaySet {
    let data = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut r = Reader { data: &data, pos: 0 };
    r.header(RAYS_MAGIC, path);
    let kind = match r.u32() {
        0 => QueryKind::Closest,
        _ => QueryKind::Occluded,
    };
    let n = r.u64() as usize;
    let rays = (0..n)
        .map(|_| RawRay {
            org: [r.f32(), r.f32(), r.f32()],
            dir: [r.f32(), r.f32(), r.f32()],
            tmin: r.f32(),
            tmax: r.f32(),
        })
        .collect();
    RaySet { kind, rays }
}

/// Check results: closest-hit `t` (or +inf on a miss), or 1.0 / 0.0 for occlusion.
pub fn write_f32s(path: &Path, values: &[f32]) {
    let out: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, out).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

pub fn read_f32s(path: &Path) -> Vec<f32> {
    let data = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// FNV-1a 64, used to key cached results on their exact inputs.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── Reference intersection ───────────────────────────────────────────────────

/// Möller–Trumbore with the same conventions as every rnpt version: single-sided
/// (back faces culled, `det < 1e-7`), hits accepted in `[tmin, tmax]`.
#[inline(always)]
pub fn intersect_triangle(r: &RawRay, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Option<f32> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let h = cross(r.dir, e2);
    let det = dot(e1, h);
    if det < 1e-7 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = sub(r.org, v0);
    let u = inv_det * dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = inv_det * dot(r.dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = inv_det * dot(e2, q);
    if t < r.tmin || t > r.tmax {
        return None;
    }
    Some(t)
}

// ── Adapter side ─────────────────────────────────────────────────────────────

/// What an adapter must provide for one historical version of the BVH.
pub trait Accel {
    type Ray;
    fn make_ray(&self, r: &RawRay) -> Self::Ray;
    fn closest(&self, r: &Self::Ray) -> Option<f32>;
    fn occluded(&self, r: &Self::Ray) -> bool;
}

pub struct AdapterInfo {
    pub variant: &'static str,
    /// False when the version had no dedicated occlusion query and shadow rays
    /// go through closest-hit.
    pub native_occlusion: bool,
}

/// Pin the current thread to one core and raise the process priority.
/// Returns whether pinning succeeded.
pub fn pin_to_core(core: usize) -> bool {
    #[cfg(windows)]
    {
        use std::ffi::c_void;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> *mut c_void;
            fn GetCurrentProcess() -> *mut c_void;
            fn SetThreadAffinityMask(thread: *mut c_void, mask: usize) -> usize;
            fn SetPriorityClass(process: *mut c_void, class: u32) -> i32;
        }
        const HIGH_PRIORITY_CLASS: u32 = 0x80;
        unsafe {
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
            SetThreadAffinityMask(GetCurrentThread(), 1usize << core) != 0
        }
    }
    #[cfg(not(windows))]
    {
        let _ = core;
        false
    }
}

struct Args {
    input: String,
    out_dir: String,
    passes: usize,
    check: usize,
    core: usize,
    jobs: Vec<(String, String)>,
}

fn parse_args() -> Args {
    let mut a = Args { input: String::new(), out_dir: String::new(), passes: 9, check: 4096, core: 2, jobs: vec![] };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().expect("missing value");
        match k.as_str() {
            "--input" => a.input = v,
            "--out-dir" => a.out_dir = v,
            "--passes" => a.passes = v.parse().unwrap(),
            "--check" => a.check = v.parse().unwrap(),
            "--core" => a.core = v.parse().unwrap(),
            "--job" => {
                let (s, r) = v.split_once(':').expect("job is scene:set");
                a.jobs.push((s.to_string(), r.to_string()));
            }
            _ => panic!("unknown argument {k}"),
        }
    }
    a
}

/// Entry point of every adapter: builds each requested scene once, then times
/// each ray set (one warm-up pass, then `passes` timed passes over all rays)
/// and dumps the first `check` results for the driver to verify.
pub fn adapter_main<A: Accel>(info: AdapterInfo, build: impl Fn(&TriScene) -> A) {
    let args = parse_args();
    let pinned = pin_to_core(args.core);
    let input = Path::new(&args.input);
    let out_dir = Path::new(&args.out_dir);

    let mut scenes: Vec<&str> = args.jobs.iter().map(|(s, _)| s.as_str()).collect();
    scenes.dedup();

    let mut json = format!(
        "{{\"variant\":\"{}\",\"native_occlusion\":{},\"pinned\":{},\"results\":[",
        info.variant, info.native_occlusion, pinned
    );
    let mut first = true;

    for scene_name in scenes {
        let scene = read_scene(&input.join(format!("{scene_name}.scene")));
        let t0 = Instant::now();
        let accel = build(&scene);
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(scene);

        for (_, set_name) in args.jobs.iter().filter(|(s, _)| s == scene_name) {
            let set = read_rays(&input.join(format!("{scene_name}.{set_name}.rays")));
            let rays: Vec<A::Ray> = set.rays.iter().map(|r| accel.make_ray(r)).collect();

            let run_pass = || match set.kind {
                QueryKind::Closest => {
                    for r in &rays {
                        black_box(accel.closest(black_box(r)));
                    }
                }
                QueryKind::Occluded => {
                    for r in &rays {
                        black_box(accel.occluded(black_box(r)));
                    }
                }
            };

            run_pass();
            let pass_ns: Vec<u128> = (0..args.passes)
                .map(|_| {
                    let t = Instant::now();
                    run_pass();
                    t.elapsed().as_nanos()
                })
                .collect();

            let check: Vec<f32> = rays
                .iter()
                .take(args.check)
                .map(|r| match set.kind {
                    QueryKind::Closest => accel.closest(r).unwrap_or(f32::INFINITY),
                    QueryKind::Occluded => accel.occluded(r) as u32 as f32,
                })
                .collect();
            write_f32s(&out_dir.join(format!("{scene_name}.{set_name}.check")), &check);

            let passes: Vec<String> = pass_ns.iter().map(|p| p.to_string()).collect();
            if !first {
                json.push(',');
            }
            first = false;
            json.push_str(&format!(
                "{{\"scene\":\"{scene_name}\",\"set\":\"{set_name}\",\"rays\":{},\"build_ms\":{build_ms:.3},\"pass_ns\":[{}]}}",
                rays.len(),
                passes.join(",")
            ));
            eprintln!("  {scene_name}/{set_name}: done");
        }
    }
    json.push_str("]}");

    let mut f = fs::File::create(out_dir.join("adapter.json")).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}
