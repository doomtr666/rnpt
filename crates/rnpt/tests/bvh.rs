use nalgebra::{Point3, Transform3, UnitVector3, Vector2, Vector3};
use rnpt::{Bvh, BvhBuilder, Material, Mesh, Node, Ray, Scene, Triangle};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn build_scene(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Bvh {
    let up = UnitVector3::new_normalize(Vector3::z());
    let scene = Scene {
        meshes: vec![Mesh {
            vertices:  verts.iter().map(|v| Point3::new(v[0], v[1], v[2])).collect(),
            normals:   vec![up; verts.len()],
            uvs:       vec![Vector2::zeros(); verts.len()],
            colors:    vec![Vector3::new(1.0, 1.0, 1.0); verts.len()],
            triangles: tris.iter().map(|t| Triangle { v0: t[0], v1: t[1], v2: t[2] }).collect(),
            material:  0,
        }],
        materials: vec![Material {
            albedo:           Vector3::new(1.0, 1.0, 1.0),
            emissive:         Vector3::zeros(),
            albedo_texture:   None,
            emissive_texture: None,
        }],
        textures: vec![],
        lights:   vec![],
        nodes:    vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0] }],
        roots:    vec![0],
        cameras:  vec![],
    };
    BvhBuilder::new(&scene).build().0
}

/// Brute-force closest hit using scalar MT over the triangle_meta list.
fn brute_force_t(bvh: &Bvh, ray: &Ray) -> Option<f32> {
    let mut best: Option<f32> = None;
    for meta in &bvh.triangle_meta {
        let v0 = &bvh.vertices[meta.v0 as usize];
        let v1 = &bvh.vertices[meta.v1 as usize];
        let v2 = &bvh.vertices[meta.v2 as usize];
        if let Some(h) = ray.intersect_triangle(v0, v1, v2) {
            if best.map_or(true, |b| h.t < b) {
                best = Some(h.t);
            }
        }
    }
    best
}

// Canonical single triangle: v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) in z=0 plane.
// CCW winding → front face toward +z.
// Reference ray: origin=(0.25,0.25,1) dir=(0,0,-1) → t=1, u=0.25, v=0.25.
fn one_tri() -> Bvh {
    build_scene(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 2]],
    )
}

fn dir_neg_z() -> UnitVector3<f32> {
    UnitVector3::new_normalize(Vector3::new(0.0, 0.0, -1.0))
}

fn ray_ref() -> Ray {
    Ray::new(Point3::new(0.25, 0.25, 1.0), dir_neg_z())
}

// ── A: basic hit, t/u/v ───────────────────────────────────────────────────────

#[test]
fn hit_basic() {
    let hit = one_tri().intersect(&ray_ref()).expect("should hit");
    assert!(approx(hit.hit.t, 1.0), "t={}", hit.hit.t);
    assert!(approx(hit.hit.u, 0.25), "u={}", hit.hit.u);
    assert!(approx(hit.hit.v, 0.25), "v={}", hit.hit.v);
    assert_eq!(hit.material, 0);
}

// ── B: spatial miss ───────────────────────────────────────────────────────────

#[test]
fn miss_outside() {
    let ray = Ray::new(Point3::new(0.9, 0.9, 1.0), dir_neg_z());
    assert!(one_tri().intersect(&ray).is_none());
}

// ── C: ray pointing away from triangle ────────────────────────────────────────

#[test]
fn miss_behind() {
    let ray = Ray::new(
        Point3::new(0.25, 0.25, -1.0),
        UnitVector3::new_normalize(Vector3::new(0.0, 0.0, -1.0)),
    );
    assert!(one_tri().intersect(&ray).is_none());
}

// ── D: back-face culling ──────────────────────────────────────────────────────

#[test]
fn backface_cull() {
    let ray = Ray::new(
        Point3::new(0.25, 0.25, -1.0),
        UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
    );
    assert!(one_tri().intersect(&ray).is_none(), "back face must be culled");
}

// ── E: tmin clips hit ─────────────────────────────────────────────────────────

#[test]
fn tmin_clips() {
    let ray = Ray::new_with_minmax(Point3::new(0.25, 0.25, 1.0), dir_neg_z(), 2.0, f32::INFINITY);
    assert!(one_tri().intersect(&ray).is_none());
}

// ── F: tmax clips hit ─────────────────────────────────────────────────────────

#[test]
fn tmax_clips() {
    let ray = Ray::new_with_minmax(Point3::new(0.25, 0.25, 1.0), dir_neg_z(), 0.0, 0.5);
    assert!(one_tri().intersect(&ray).is_none());
}

// ── G: tmax includes hit ──────────────────────────────────────────────────────

#[test]
fn tmax_includes() {
    let ray = Ray::new_with_minmax(Point3::new(0.25, 0.25, 1.0), dir_neg_z(), 0.0, 1.5);
    let hit = one_tri().intersect(&ray).expect("should hit within tmax=1.5");
    assert!(approx(hit.hit.t, 1.0));
}

// ── H: tmin prevents self-intersection ───────────────────────────────────────

#[test]
fn self_intersection_guard() {
    let ray = Ray::new_with_minmax(
        Point3::new(0.25, 0.25, 0.0),
        dir_neg_z(),
        1e-4,
        f32::INFINITY,
    );
    assert!(one_tri().intersect(&ray).is_none(), "self-hit at t=0 must be clipped by tmin");
}

// ── I: closest of two triangles ───────────────────────────────────────────────

#[test]
fn closest_of_two() {
    let bvh = build_scene(
        &[
            [0.0, 0.0,  0.0], [1.0, 0.0,  0.0], [0.0, 1.0,  0.0],
            [0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0],
        ],
        &[[0, 1, 2], [3, 4, 5]],
    );
    let hit = bvh.intersect(&ray_ref()).expect("should hit");
    assert!(approx(hit.hit.t, 1.0), "must return near hit (t=1.0), got t={}", hit.hit.t);
}

// ── J: barycentric coordinates at triangle vertices ──────────────────────────

#[test]
fn barycentric_at_vertices() {
    let bvh = one_tri();
    let dir = dir_neg_z();

    let h0 = bvh.intersect(&Ray::new(Point3::new(0.0, 0.0, 1.0), dir)).expect("v0 hit");
    assert!(approx(h0.hit.u, 0.0) && approx(h0.hit.v, 0.0),
        "at v0: u={} v={}", h0.hit.u, h0.hit.v);

    let h1 = bvh.intersect(&Ray::new(Point3::new(1.0, 0.0, 1.0), dir)).expect("v1 hit");
    assert!(approx(h1.hit.u, 1.0) && approx(h1.hit.v, 0.0),
        "at v1: u={} v={}", h1.hit.u, h1.hit.v);

    let h2 = bvh.intersect(&Ray::new(Point3::new(0.0, 1.0, 1.0), dir)).expect("v2 hit");
    assert!(approx(h2.hit.u, 0.0) && approx(h2.hit.v, 1.0),
        "at v2: u={} v={}", h2.hit.u, h2.hit.v);
}

// ── K: is_occluded true ───────────────────────────────────────────────────────

#[test]
fn occluded_true() {
    let ray = Ray::new_with_minmax(Point3::new(0.25, 0.25, 1.0), dir_neg_z(), 1e-4, 10.0);
    assert!(one_tri().is_occluded(&ray));
}

// ── L: is_occluded false on spatial miss ──────────────────────────────────────

#[test]
fn occluded_miss() {
    let ray = Ray::new_with_minmax(Point3::new(0.9, 0.9, 1.0), dir_neg_z(), 1e-4, 10.0);
    assert!(!one_tri().is_occluded(&ray));
}

// ── M: is_occluded respects tmax ──────────────────────────────────────────────

#[test]
fn occluded_tmax() {
    let ray = Ray::new_with_minmax(Point3::new(0.25, 0.25, 1.0), dir_neg_z(), 1e-4, 0.5);
    assert!(!one_tri().is_occluded(&ray));
}

// ── N: diagonal rays from multiple octants ────────────────────────────────────

#[test]
fn octant_directions() {
    let bvh_above = build_scene(
        &[
            [0.1,  0.1, 1.0], [0.9,  0.1, 1.0], [0.1,  0.9, 1.0],
            [-0.9, 0.1, 1.0], [-0.1, 0.1, 1.0], [-0.9, 0.9, 1.0],
            [-0.9,-0.9, 1.0], [-0.1,-0.9, 1.0], [-0.9,-0.1, 1.0],
            [0.1, -0.9, 1.0], [0.9, -0.9, 1.0], [0.1, -0.1, 1.0],
        ],
        &[[0,1,2], [3,4,5], [6,7,8], [9,10,11]],
    );

    for (cx, cy) in [(0.4f32, 0.4), (-0.5, 0.4), (-0.5, -0.5), (0.4, -0.5)] {
        let ray = Ray::new(
            Point3::new(cx, cy, 3.0),
            UnitVector3::new_normalize(Vector3::new(0.0, 0.0, -1.0)),
        );
        let hit = bvh_above
            .intersect(&ray)
            .unwrap_or_else(|| panic!("expected hit for cx={cx} cy={cy} from above"));
        assert!(approx(hit.hit.t, 2.0), "expected t=2.0, got {}", hit.hit.t);
    }

    let bvh_below = build_scene(
        &[
            [0.1,  0.1, -1.0], [0.1,  0.9, -1.0], [0.9,  0.1, -1.0],
            [-0.9, 0.1, -1.0], [-0.9, 0.9, -1.0], [-0.1, 0.1, -1.0],
            [-0.9,-0.9, -1.0], [-0.9,-0.1, -1.0], [-0.1,-0.9, -1.0],
            [0.1, -0.9, -1.0], [0.1, -0.1, -1.0], [0.9, -0.9, -1.0],
        ],
        &[[0,1,2], [3,4,5], [6,7,8], [9,10,11]],
    );

    for (cx, cy) in [(0.4f32, 0.4), (-0.5, 0.4), (-0.5, -0.5), (0.4, -0.5)] {
        let ray = Ray::new(
            Point3::new(cx, cy, -3.0),
            UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        );
        let hit = bvh_below
            .intersect(&ray)
            .unwrap_or_else(|| panic!("expected hit for cx={cx} cy={cy} from below"));
        assert!(approx(hit.hit.t, 2.0), "expected t=2.0, got {}", hit.hit.t);
    }
}

// ── O: large scene vs. brute-force ────────────────────────────────────────────

#[test]
fn large_scene_vs_brute_force() {
    let n = 9usize;
    let mut verts = Vec::new();
    let mut tris  = Vec::new();
    for r in 0..n {
        for c in 0..n {
            let x = c as f32 / (n - 1) as f32 * 2.0 - 1.0;
            let y = r as f32 / (n - 1) as f32 * 2.0 - 1.0;
            let z = (x * 2.3 + y * 1.7).sin() * 0.3;
            verts.push([x, y, z]);
        }
    }
    for r in 0..(n - 1) {
        for c in 0..(n - 1) {
            let i00 = (r * n + c) as u32;
            let i10 = ((r + 1) * n + c) as u32;
            let i01 = (r * n + c + 1) as u32;
            let i11 = ((r + 1) * n + c + 1) as u32;
            tris.push([i00, i10, i01]);
            tris.push([i10, i11, i01]);
        }
    }
    let bvh = build_scene(&verts, &tris);

    let ray_params: &[(f32, f32, f32, f32, f32, f32)] = &[
        ( 0.0,  0.0, 2.0,  0.0,  0.0, -1.0),
        ( 0.5,  0.5, 2.0,  0.0,  0.0, -1.0),
        (-0.5,  0.3, 2.0,  0.0,  0.0, -1.0),
        ( 0.3, -0.6, 2.0,  0.0,  0.0, -1.0),
        ( 0.0,  0.0, 2.0,  0.3,  0.2, -1.0),
        ( 0.0,  0.0, 2.0, -0.4,  0.1, -1.0),
        ( 0.0,  0.0, 2.0,  0.2, -0.3, -1.0),
        ( 0.0,  0.0, 2.0, -0.1, -0.4, -1.0),
    ];

    for &(ox, oy, oz, dx, dy, dz) in ray_params {
        let ray = Ray::new(
            Point3::new(ox, oy, oz),
            UnitVector3::new_normalize(Vector3::new(dx, dy, dz)),
        );
        let bvh_t   = bvh.intersect(&ray).map(|h| h.hit.t);
        let brute_t = brute_force_t(&bvh, &ray);
        match (bvh_t, brute_t) {
            (Some(bt), Some(bf)) => {
                assert!(
                    approx(bt, bf),
                    "t mismatch for ray ({ox},{oy},{oz})→({dx},{dy},{dz}): bvh={bt} brute={bf}"
                );
            }
            (None, None) => {}
            _ => panic!(
                "hit/miss disagreement for ray ({ox},{oy},{oz})→({dx},{dy},{dz}): bvh={bvh_t:?} brute={brute_t:?}"
            ),
        }
    }
}

// ── P: empty scene ────────────────────────────────────────────────────────────

#[test]
fn empty_scene() {
    let bvh = build_scene(&[], &[]);
    let ray = Ray::new(
        Point3::new(0.25, 0.25, 1.0),
        UnitVector3::new_normalize(Vector3::new(0.0, 0.0, -1.0)),
    );
    assert!(bvh.intersect(&ray).is_none(), "empty BVH must return None");
    assert!(!bvh.is_occluded(&ray),        "empty BVH must not occlude");
}
