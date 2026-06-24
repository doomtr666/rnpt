use std::collections::HashMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Point3, UnitVector3, Vector2, Vector3};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rnpt::{Bvh, BvhBuilder, Material, Mesh, Node, Ray, Scene, Transform3, Triangle};

#[cfg(feature = "embree")]
use rnpt_bvh_embree::{Geometry as EGeometry, Ray as BvhRay, RayAccelerator, Scene as EScene};

// ── Scene generators ──────────────────────────────────────────────────────────

fn hnoise(x: f32, y: f32) -> f32 {
    (x * 1.7 + y * 3.1).sin() * 0.40
  + (x * 5.3 - y * 2.7).sin() * 0.20
  + (x * 11.1 + y * 7.3).sin() * 0.10
}

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
            tangents:  vec![],
            triangles: tris.iter().map(|t| Triangle { v0:t[0], v1:t[1], v2:t[2] }).collect(),
            material:  0,
        }],
        materials: vec![Material::default()],
        textures: vec![], lights: vec![],
        nodes: vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0] }],
        roots: vec![0], cameras: vec![],
    };
    BvhBuilder::new(&scene).build().0
}

#[cfg(feature = "embree")]
fn build_embree(verts: &[[f32;3]], tris: &[[u32;3]]) -> EScene {
    let mut s = EScene::new();
    s.attach_geometry(EGeometry::triangle_mesh(verts, tris));
    s.commit();
    s
}

// ── Ray generation ────────────────────────────────────────────────────────────

const NUM_RAYS: usize = 1024;
const SEED: u64 = 0xdead_beef_cafe_babe;

fn coherent_rays() -> Vec<Ray> {
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

#[cfg(feature = "embree")]
fn to_bvh_ray(r: &Ray) -> BvhRay {
    BvhRay::new_with_minmax(
        [r.origin.x, r.origin.y, r.origin.z],
        [r.direction.x, r.direction.y, r.direction.z],
        r.tmin,
        r.tmax,
    )
}

// ── Bench runner ──────────────────────────────────────────────────────────────

const SIZES: &[(&str, usize)] = &[
    ("100k", 100_000),
    ("1M",   1_000_000),
    ("10M",  10_000_000),
];

fn bench_scene(
    c:     &mut Criterion,
    tag:   &str,
    sizes: &[(&'static str, usize)],
    make:  fn(usize) -> (Vec<[f32;3]>, Vec<[u32;3]>),
) {
    let coherent   = coherent_rays();
    let incoherent = incoherent_rays();
    let shadow     = shadow_rays();
    let n = NUM_RAYS as u64;

    // Build all scene sizes upfront. Each size is built once and shared
    // across all three ray-type groups — Criterion groups must not be
    // recreated with the same name or previous size results are orphaned.
    struct Built {
        label: &'static str,
        bvh:   Bvh,
        #[cfg(feature = "embree")]
        sc:    EScene,
    }
    let scenes: Vec<Built> = sizes.iter().map(|&(label, target)| {
        let (verts, tris) = make(target);
        Built {
            label,
            bvh: build_bvh(&verts, &tris),
            #[cfg(feature = "embree")]
            sc:  build_embree(&verts, &tris),
        }
    }).collect();

    {
        let mut g = c.benchmark_group(format!("coherent_{tag}"));
        g.throughput(Throughput::Elements(n));
        for s in &scenes {
            g.bench_function(BenchmarkId::new("rnpt", s.label), |b| {
                b.iter(|| { for r in &coherent { black_box(s.bvh.intersect(r)); } })
            });
            #[cfg(feature = "embree")]
            g.bench_function(BenchmarkId::new("embree1", s.label), |b| {
                b.iter(|| { for r in &coherent { black_box(s.sc.closest_hit(&to_bvh_ray(r))); } })
            });
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group(format!("incoherent_{tag}"));
        g.throughput(Throughput::Elements(n));
        for s in &scenes {
            g.bench_function(BenchmarkId::new("rnpt", s.label), |b| {
                b.iter(|| { for r in &incoherent { black_box(s.bvh.intersect(r)); } })
            });
            #[cfg(feature = "embree")]
            g.bench_function(BenchmarkId::new("embree", s.label), |b| {
                b.iter(|| { for r in &incoherent { black_box(s.sc.closest_hit(&to_bvh_ray(r))); } })
            });
        }
        g.finish();
    }

    {
        let mut g = c.benchmark_group(format!("shadow_{tag}"));
        g.throughput(Throughput::Elements(n));
        for s in &scenes {
            g.bench_function(BenchmarkId::new("rnpt", s.label), |b| {
                b.iter(|| { for r in &shadow { black_box(s.bvh.is_occluded(r)); } })
            });
            #[cfg(feature = "embree")]
            g.bench_function(BenchmarkId::new("embree", s.label), |b| {
                b.iter(|| { for r in &shadow { black_box(s.sc.any_hit(&to_bvh_ray(r))); } })
            });
        }
        g.finish();
    }
}

fn bench_hf     (c: &mut Criterion) { bench_scene(c, "hf",      SIZES, make_heightfield);    }
fn bench_cluster(c: &mut Criterion) { bench_scene(c, "cluster", SIZES, make_sphere_cluster); }

criterion_group!(benches, bench_hf, bench_cluster);
criterion_main!(benches);
