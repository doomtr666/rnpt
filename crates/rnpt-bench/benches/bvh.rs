use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Point3, UnitVector3, Vector2, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rnpt::{Bvh, BvhBuilder, Material, Mesh, Node, Ray, Scene, Transform3, Triangle};

// ── Scene generators ──────────────────────────────────────────────────────────

fn hnoise(x: f32, y: f32) -> f32 {
    (x * 1.7 + y * 3.1).sin() * 0.40
  + (x * 5.3 - y * 2.7).sin() * 0.20
  + (x * 11.1 + y * 7.3).sin() * 0.10
}

/// Connected grid with height variation — good BVH, representative of organic surfaces.
fn make_heightfield(target_tris: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let r = ((target_tris / 2) as f64).sqrt() as usize + 2;
    let c = r;
    let mut verts = Vec::with_capacity(r * c);
    let mut tris  = Vec::with_capacity(2 * (r - 1) * (c - 1));
    for ri in 0..r {
        for ci in 0..c {
            let x = ci as f32 / (c - 1) as f32 * 2.0 - 1.0;
            let y = ri as f32 / (r - 1) as f32 * 2.0 - 1.0;
            verts.push([x, y, hnoise(x, y)]);
        }
    }
    for ri in 0..(r - 1) {
        for ci in 0..(c - 1) {
            let i00 = (ri * c + ci) as u32;
            let i10 = ((ri + 1) * c + ci) as u32;
            let i01 = (ri * c + ci + 1) as u32;
            let i11 = ((ri + 1) * c + ci + 1) as u32;
            tris.push([i00, i10, i01]);
            tris.push([i10, i11, i01]);
        }
    }
    (verts, tris)
}

/// N icospheres (level-4, 5120 tris each) at random positions/radii.
/// Simulates "many objects in a scene" — AABBs overlap between spheres.
fn make_sphere_cluster(target_tris: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    const SUBDIVS: u32 = 4;
    const TRIS_PER_SPHERE: usize = 20 * (1 << (2 * SUBDIVS)); // 5120
    let n = (target_tris / TRIS_PER_SPHERE).max(1);
    let (unit_v, unit_t) = icosphere(SUBDIVS);
    let mut rng = SmallRng::seed_from_u64(0xbabe_cafe_1234);
    let mut verts = Vec::with_capacity(n * unit_v.len());
    let mut tris  = Vec::with_capacity(n * unit_t.len());
    for _ in 0..n {
        let cx = rng.gen_range(-0.8f32..0.8);
        let cy = rng.gen_range(-0.8f32..0.8);
        let cz = rng.gen_range(-0.8f32..0.8);
        let r  = rng.gen_range(0.05f32..0.25);
        let base = verts.len() as u32;
        for v in &unit_v { verts.push([cx + v[0]*r, cy + v[1]*r, cz + v[2]*r]); }
        for t in &unit_t { tris.push([base + t[0], base + t[1], base + t[2]]); }
    }
    (verts, tris)
}

/// Random disconnected triangles in [-1,1]³ — maximum AABB overlap, worst-case BVH.
fn make_random_soup(target_tris: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut rng = SmallRng::seed_from_u64(0xc0ffee_dead_beef);
    let mut verts = Vec::with_capacity(target_tris * 3);
    let mut tris  = Vec::with_capacity(target_tris);
    for i in 0..target_tris {
        for _ in 0..3 {
            verts.push([
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
                rng.gen_range(-1.0f32..1.0),
            ]);
        }
        let b = (i * 3) as u32;
        tris.push([b, b + 1, b + 2]);
    }
    (verts, tris)
}

// ── Icosphere ─────────────────────────────────────────────────────────────────

fn icosphere(subdivs: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
    let mut verts: Vec<[f32; 3]> = [
        [-1., phi, 0.], [1., phi, 0.], [-1., -phi, 0.], [1., -phi, 0.],
        [0., -1.,  phi], [0., 1.,  phi], [0., -1., -phi], [0., 1., -phi],
        [phi, 0., -1.], [phi, 0., 1.], [-phi, 0., -1.], [-phi, 0., 1.],
    ].iter().map(|v| {
        let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/l, v[1]/l, v[2]/l]
    }).collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];
    for _ in 0..subdivs {
        let mut cache: HashMap<(u32,u32),u32> = HashMap::new();
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a,b,c] in &tris {
            let ab = midpoint(&mut verts, &mut cache, *a, *b);
            let bc = midpoint(&mut verts, &mut cache, *b, *c);
            let ca = midpoint(&mut verts, &mut cache, *c, *a);
            next.extend([[*a,ab,ca],[*b,bc,ab],[*c,ca,bc],[ab,bc,ca]]);
        }
        tris = next;
    }
    (verts, tris)
}

fn midpoint(verts: &mut Vec<[f32;3]>, cache: &mut HashMap<(u32,u32),u32>, a: u32, b: u32) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = cache.get(&key) { return i; }
    let [ax,ay,az] = verts[a as usize];
    let [bx,by,bz] = verts[b as usize];
    let (mx,my,mz) = ((ax+bx)*0.5, (ay+by)*0.5, (az+bz)*0.5);
    let l = (mx*mx + my*my + mz*mz).sqrt();
    let i = verts.len() as u32;
    verts.push([mx/l, my/l, mz/l]);
    cache.insert(key, i);
    i
}

// ── rnpt helpers ──────────────────────────────────────────────────────────────

fn build_bvh(verts: &[[f32;3]], tris: &[[u32;3]]) -> Bvh {
    let up = UnitVector3::new_normalize(Vector3::new(0.0f32, 0.0, 1.0));
    let scene = Scene {
        meshes: vec![Mesh {
            vertices:  verts.iter().map(|v| Point3::new(v[0],v[1],v[2])).collect(),
            normals:   vec![up; verts.len()],
            uvs:       vec![Vector2::zeros(); verts.len()],
            colors:    vec![Vector3::new(1.,1.,1.); verts.len()],
            triangles: tris.iter().map(|t| Triangle { v0:t[0], v1:t[1], v2:t[2] }).collect(),
            material:  0,
        }],
        materials: vec![Material {
            albedo: Vector3::new(1.,1.,1.), emissive: Vector3::zeros(),
            albedo_texture: None, emissive_texture: None,
        }],
        textures: vec![], lights: vec![],
        nodes: vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0] }],
        roots: vec![0], cameras: vec![],
    };
    BvhBuilder::new(&scene).build().0
}

// ── Ray generation ────────────────────────────────────────────────────────────

const NUM_RAYS: usize = 1024; // must be divisible by 8 for intersect8 packets
const SEED: u64 = 0xdead_beef_cafe_babe;

fn coherent_rays() -> Vec<Ray> {
    // Orthographic rays: identical direction, origins on a uniform grid over [-1,1]².
    // All rays share the same traversal path down to leaf level — maximum BVH coherence.
    let dir = UnitVector3::new_normalize(Vector3::new(0.0f32, 0.0, -1.0));
    let side = (NUM_RAYS as f32).sqrt() as usize;
    (0..NUM_RAYS).map(|i| {
        let x = (i % side) as f32 / (side - 1) as f32 * 2.0 - 1.0;
        let y = (i / side) as f32 / (side - 1) as f32 * 2.0 - 1.0;
        Ray::new(Point3::new(x, y, 3.0), dir)
    }).collect()
}

fn incoherent_rays() -> Vec<Ray> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    (0..NUM_RAYS).map(|_| {
        let o = Point3::new(
            rng.gen_range(-1.2f32..1.2),
            rng.gen_range(-1.2f32..1.2),
            rng.gen_range(0.5f32..3.0),
        );
        let d = UnitVector3::new_normalize(Vector3::new(
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..1.0),
            rng.gen_range(-1.0f32..0.0),
        ));
        Ray::new(o, d)
    }).collect()
}

fn shadow_rays() -> Vec<Ray> {
    let mut rng = SmallRng::seed_from_u64(SEED ^ 0x1234);
    let light = Point3::new(0.0f32, 0.0, 3.0);
    (0..NUM_RAYS).map(|_| {
        let s = Point3::new(
            rng.gen_range(-0.9f32..0.9),
            rng.gen_range(-0.9f32..0.9),
            rng.gen_range(-0.9f32..0.9),
        );
        let to = light.coords - s.coords;
        let dist = to.norm();
        Ray::new_with_minmax(s, UnitVector3::new_normalize(to), 1e-4, dist - 1e-4)
    }).collect()
}

// ── Embree FFI ────────────────────────────────────────────────────────────────

#[cfg(feature = "embree")]
mod embree {
    use std::ptr;

    pub type RTCDevice   = *mut std::ffi::c_void;
    pub type RTCScene    = *mut std::ffi::c_void;
    pub type RTCGeometry = *mut std::ffi::c_void;

    pub const RTC_GEOMETRY_TYPE_TRIANGLE: u32 = 0;
    pub const RTC_BUFFER_TYPE_INDEX:      u32 = 0;
    pub const RTC_BUFFER_TYPE_VERTEX:     u32 = 1;
    pub const RTC_FORMAT_FLOAT3:          u32 = 0x9003;
    pub const RTC_FORMAT_UINT3:           u32 = 0x5003;

    #[repr(C, align(16))]
    #[derive(Clone, Copy, Default)]
    pub struct RTCRay {
        pub org_x:f32, pub org_y:f32, pub org_z:f32, pub tnear:f32,
        pub dir_x:f32, pub dir_y:f32, pub dir_z:f32, pub time:f32,
        pub tfar:f32,  pub mask:u32,  pub id:u32,    pub flags:u32,
    }

    // RTC_GEOMETRY_INSTANCE_ARRAY is defined → instPrimID[1] present
    // 9 × 4 = 36 bytes → padded to 48 by align(16)
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    pub struct RTCHit {
        pub ng_x:f32, pub ng_y:f32, pub ng_z:f32,
        pub u:f32, pub v:f32,
        pub prim_id:u32, pub geom_id:u32,
        pub inst_id:[u32;1], pub inst_prim_id:[u32;1],
    }
    impl Default for RTCHit {
        fn default() -> Self {
            Self { ng_x:0.,ng_y:0.,ng_z:0., u:0.,v:0.,
                prim_id:u32::MAX, geom_id:u32::MAX,
                inst_id:[u32::MAX], inst_prim_id:[u32::MAX] }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct RTCRayHit { pub ray: RTCRay, pub hit: RTCHit }

    // RTCRay8 / RTCHit8 — RTC_ALIGN(32), SoA layout (8 lanes)
    #[repr(C, align(32))]
    pub struct RTCRay8 {
        pub org_x:[f32;8], pub org_y:[f32;8], pub org_z:[f32;8], pub tnear:[f32;8],
        pub dir_x:[f32;8], pub dir_y:[f32;8], pub dir_z:[f32;8], pub time:[f32;8],
        pub tfar:[f32;8], pub mask:[u32;8], pub id:[u32;8], pub flags:[u32;8],
    }
    // RTC_GEOMETRY_INSTANCE_ARRAY defined + RTC_MAX_INSTANCE_LEVEL_COUNT=1
    // layout: ng[8]×3, u/v[8], prim/geomID[8], instID[1][8], instPrimID[1][8]
    // = 224 bytes → already multiple of 32, no padding
    #[repr(C, align(32))]
    pub struct RTCHit8 {
        pub ng_x:[f32;8], pub ng_y:[f32;8], pub ng_z:[f32;8],
        pub u:[f32;8], pub v:[f32;8],
        pub prim_id:[u32;8], pub geom_id:[u32;8],
        pub inst_id:[[u32;8];1], pub inst_prim_id:[[u32;8];1],
    }
    #[repr(C)]
    pub struct RTCRayHit8 { pub ray: RTCRay8, pub hit: RTCHit8 }

    unsafe extern "C" {
        pub fn rtcNewDevice(cfg: *const std::ffi::c_char) -> RTCDevice;
        pub fn rtcReleaseDevice(d: RTCDevice);
        pub fn rtcNewScene(d: RTCDevice) -> RTCScene;
        pub fn rtcReleaseScene(s: RTCScene);
        pub fn rtcNewGeometry(d: RTCDevice, ty: u32) -> RTCGeometry;
        pub fn rtcReleaseGeometry(g: RTCGeometry);
        pub fn rtcSetNewGeometryBuffer(g:RTCGeometry, ty:u32, slot:u32, fmt:u32,
            stride:usize, count:usize) -> *mut std::ffi::c_void;
        pub fn rtcCommitGeometry(g: RTCGeometry);
        pub fn rtcAttachGeometry(s: RTCScene, g: RTCGeometry) -> u32;
        pub fn rtcCommitScene(s: RTCScene);
        pub fn rtcIntersect1(s:RTCScene, rh:*mut RTCRayHit, args:*const std::ffi::c_void);
        pub fn rtcOccluded1(s:RTCScene, r:*mut RTCRay, args:*const std::ffi::c_void);
        pub fn rtcIntersect8(valid:*const i32, s:RTCScene, rh:*mut RTCRayHit8, args:*const std::ffi::c_void);
    }

    pub struct Device(RTCDevice);
    pub struct EmbreeScene(RTCScene);

    impl Device {
        pub fn new() -> Self {
            let d = unsafe { rtcNewDevice(ptr::null()) };
            assert!(!d.is_null());
            Self(d)
        }
    }
    impl Drop for Device { fn drop(&mut self) { unsafe { rtcReleaseDevice(self.0); } } }

    impl EmbreeScene {
        pub fn build(dev: &Device, verts: &[[f32;3]], tris: &[[u32;3]]) -> Self {
            unsafe {
                let s = rtcNewScene(dev.0);
                let g = rtcNewGeometry(dev.0, RTC_GEOMETRY_TYPE_TRIANGLE);
                let vb = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                    12, verts.len()) as *mut f32;
                std::ptr::copy_nonoverlapping(verts.as_ptr() as *const f32, vb, verts.len()*3);
                let ib = rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                    12, tris.len()) as *mut u32;
                std::ptr::copy_nonoverlapping(tris.as_ptr() as *const u32, ib, tris.len()*3);
                rtcCommitGeometry(g); rtcAttachGeometry(s, g); rtcReleaseGeometry(g);
                rtcCommitScene(s);
                Self(s)
            }
        }
        pub fn intersect1(&self, ray: &rnpt::Ray) -> bool {
            let mut rh = RTCRayHit {
                ray: RTCRay {
                    org_x:ray.origin.x, org_y:ray.origin.y, org_z:ray.origin.z, tnear:ray.tmin,
                    dir_x:ray.direction.x, dir_y:ray.direction.y, dir_z:ray.direction.z,
                    time:0., tfar:ray.tmax, mask:u32::MAX, id:0, flags:0,
                },
                hit: RTCHit::default(),
            };
            unsafe { rtcIntersect1(self.0, &mut rh, ptr::null()); }
            rh.hit.geom_id != u32::MAX
        }
        pub fn occluded1(&self, ray: &rnpt::Ray) -> bool {
            let mut r = RTCRay {
                org_x:ray.origin.x, org_y:ray.origin.y, org_z:ray.origin.z, tnear:ray.tmin,
                dir_x:ray.direction.x, dir_y:ray.direction.y, dir_z:ray.direction.z,
                time:0., tfar:ray.tmax, mask:u32::MAX, id:0, flags:0,
            };
            unsafe { rtcOccluded1(self.0, &mut r, ptr::null()); }
            r.tfar < 0.0
        }

        // Packet intersection: NUM_RAYS must be divisible by 8.
        // Rays are already ordered for coherence (same-direction grid), so each packet
        // of 8 adjacent rays shares the same top-level BVH traversal path.
        pub fn intersect8_batch(&self, rays: &[rnpt::Ray]) -> u32 {
            const FULL: i32 = -1i32; // all bits set = valid lane
            let valid = [FULL; 8];
            let mut hits = 0u32;
            for chunk in rays.chunks_exact(8) {
                let mut rh = RTCRayHit8 {
                    ray: RTCRay8 {
                        org_x: std::array::from_fn(|i| chunk[i].origin.x),
                        org_y: std::array::from_fn(|i| chunk[i].origin.y),
                        org_z: std::array::from_fn(|i| chunk[i].origin.z),
                        tnear: std::array::from_fn(|i| chunk[i].tmin),
                        dir_x: std::array::from_fn(|i| chunk[i].direction.x),
                        dir_y: std::array::from_fn(|i| chunk[i].direction.y),
                        dir_z: std::array::from_fn(|i| chunk[i].direction.z),
                        time:  [0.0; 8],
                        tfar:  std::array::from_fn(|i| chunk[i].tmax),
                        mask:  [u32::MAX; 8],
                        id:    [0; 8],
                        flags: [0; 8],
                    },
                    hit: RTCHit8 {
                        ng_x:[0.;8], ng_y:[0.;8], ng_z:[0.;8],
                        u:[0.;8], v:[0.;8],
                        prim_id:[u32::MAX;8], geom_id:[u32::MAX;8],
                        inst_id:[[u32::MAX;8];1], inst_prim_id:[[u32::MAX;8];1],
                    },
                };
                unsafe { rtcIntersect8(valid.as_ptr(), self.0, &mut rh, ptr::null()); }
                for i in 0..8 { if rh.hit.geom_id[i] != u32::MAX { hits += 1; } }
            }
            hits
        }
    }
    impl Drop for EmbreeScene { fn drop(&mut self) { unsafe { rtcReleaseScene(self.0); } } }
}

// ── Bench runner ──────────────────────────────────────────────────────────────

const SIZES: &[(&str, usize)] = &[
    ("10k",  10_000),
    ("100k", 100_000),
    ("1M",   1_000_000),
];

// Random soup BVH is degenerate (O(N) traversal) — beyond 10k it runs for hours.
const SOUP_SIZES: &[(&str, usize)] = &[("10k", 10_000)];

fn bench_scene(
    c:     &mut Criterion,
    tag:   &str,
    sizes: &[(&str, usize)],
    make:  fn(usize) -> (Vec<[f32;3]>, Vec<[u32;3]>),
) {
    let coherent   = coherent_rays();
    let incoherent = incoherent_rays();
    let shadow     = shadow_rays();

    for &(size_label, target) in sizes {
        let (verts, tris) = make(target);
        let n = NUM_RAYS as u64;

        // coherent — rnpt scalar + embree scalar + embree8 packet
        {
            let mut g = c.benchmark_group(format!("coherent/{tag}"));
            g.throughput(Throughput::Elements(n));
            let bvh = build_bvh(&verts, &tris);
            g.bench_function(BenchmarkId::new("rnpt", size_label), |b| {
                b.iter(|| { for r in &coherent { black_box(bvh.intersect(r)); } })
            });
            #[cfg(feature = "embree")] {
                let dev = embree::Device::new();
                let sc  = embree::EmbreeScene::build(&dev, &verts, &tris);
                g.bench_function(BenchmarkId::new("embree1", size_label), |b| {
                    b.iter(|| { for r in &coherent { black_box(sc.intersect1(r)); } })
                });
                g.bench_function(BenchmarkId::new("embree8", size_label), |b| {
                    b.iter(|| black_box(sc.intersect8_batch(&coherent)))
                });
            }
            g.finish();
        }

        // incoherent — scalar only (packet API brings no benefit for random rays)
        {
            let mut g = c.benchmark_group(format!("incoherent/{tag}"));
            g.throughput(Throughput::Elements(n));
            let bvh = build_bvh(&verts, &tris);
            g.bench_function(BenchmarkId::new("rnpt", size_label), |b| {
                b.iter(|| { for r in &incoherent { black_box(bvh.intersect(r)); } })
            });
            #[cfg(feature = "embree")] {
                let dev = embree::Device::new();
                let sc  = embree::EmbreeScene::build(&dev, &verts, &tris);
                g.bench_function(BenchmarkId::new("embree", size_label), |b| {
                    b.iter(|| { for r in &incoherent { black_box(sc.intersect1(r)); } })
                });
            }
            g.finish();
        }

        // shadow — any-hit, scalar
        {
            let mut g = c.benchmark_group(format!("shadow/{tag}"));
            g.throughput(Throughput::Elements(n));
            let bvh = build_bvh(&verts, &tris);
            g.bench_function(BenchmarkId::new("rnpt", size_label), |b| {
                b.iter(|| { for r in &shadow { black_box(bvh.is_occluded(r)); } })
            });
            #[cfg(feature = "embree")] {
                let dev = embree::Device::new();
                let sc  = embree::EmbreeScene::build(&dev, &verts, &tris);
                g.bench_function(BenchmarkId::new("embree", size_label), |b| {
                    b.iter(|| { for r in &shadow { black_box(sc.occluded1(r)); } })
                });
            }
            g.finish();
        }
    }
}

fn bench_hf     (c: &mut Criterion) { bench_scene(c, "hf",      SIZES,      make_heightfield);    }
fn bench_cluster(c: &mut Criterion) { bench_scene(c, "cluster", SIZES,      make_sphere_cluster); }
fn bench_soup   (c: &mut Criterion) { bench_scene(c, "soup",    SOUP_SIZES, make_random_soup);    }

criterion_group!(benches, bench_hf, bench_cluster, bench_soup);
criterion_main!(benches);
