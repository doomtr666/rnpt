use rnpt_bvh::{Geometry, Ray, RayAccelerator, Scene};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn build_scene(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Scene {
    let mut scene = Scene::new();
    scene.attach_geometry(Geometry::triangle_mesh(verts, tris));
    scene.commit();
    scene
}

fn ray(org: [f32; 3], dir: [f32; 3]) -> Ray {
    let len = (dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]).sqrt();
    Ray::new(org, [dir[0]/len, dir[1]/len, dir[2]/len])
}

fn ray_minmax(org: [f32; 3], dir: [f32; 3], tmin: f32, tmax: f32) -> Ray {
    let len = (dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]).sqrt();
    Ray::new_with_minmax(org, [dir[0]/len, dir[1]/len, dir[2]/len], tmin, tmax)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

/// Scalar Möller-Trumbore, front-face only. Independent reference for brute force.
fn mt_hit(v0: [f32;3], v1: [f32;3], v2: [f32;3], r: &Ray) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let h  = cross(r.dir, e2);
    let det = dot(e1, h);
    if det < EPS { return None; }
    let inv_det = 1.0 / det;
    let s = sub(r.org, v0);
    let u = inv_det * dot(s, h);
    if u < 0.0 || u > 1.0 { return None; }
    let q = cross(s, e1);
    let v = inv_det * dot(r.dir, q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = inv_det * dot(e2, q);
    if t < r.tmin || t > r.tmax { return None; }
    Some(t)
}

/// Brute-force closest hit over explicit verts+tris (independent from BVH).
fn brute_force_t(verts: &[[f32;3]], tris: &[[u32;3]], r: &Ray) -> Option<f32> {
    let mut best: Option<f32> = None;
    for &[i0, i1, i2] in tris {
        if let Some(t) = mt_hit(verts[i0 as usize], verts[i1 as usize], verts[i2 as usize], r) {
            if best.map_or(true, |b| t < b) {
                best = Some(t);
            }
        }
    }
    best
}

// Canonical single triangle: v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) in z=0 plane.
// CCW winding → front face toward +z.
// Reference ray: origin=(0.25,0.25,1) dir=(0,0,-1) → t=1, u=0.25, v=0.25.
fn one_tri() -> Scene {
    build_scene(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 2]],
    )
}

fn ray_ref() -> Ray {
    ray([0.25, 0.25, 1.0], [0.0, 0.0, -1.0])
}

// ── A: basic hit, t/u/v ───────────────────────────────────────────────────────

#[test]
fn hit_basic() {
    let hit = one_tri().closest_hit(&ray_ref()).expect("should hit");
    assert!(approx(hit.t, 1.0), "t={}", hit.t);
    assert!(approx(hit.u, 0.25), "u={}", hit.u);
    assert!(approx(hit.v, 0.25), "v={}", hit.v);
    assert_eq!(hit.prim_id, 0);
    assert_eq!(hit.geom_id, 0);
}

// ── B: spatial miss ───────────────────────────────────────────────────────────

#[test]
fn miss_outside() {
    let r = ray([0.9, 0.9, 1.0], [0.0, 0.0, -1.0]);
    assert!(one_tri().closest_hit(&r).is_none());
}

// ── C: ray pointing away from triangle ────────────────────────────────────────

#[test]
fn miss_behind() {
    let r = ray([0.25, 0.25, -1.0], [0.0, 0.0, -1.0]);
    assert!(one_tri().closest_hit(&r).is_none());
}

// ── D: back-face culling ──────────────────────────────────────────────────────

#[test]
fn backface_cull() {
    let r = ray([0.25, 0.25, -1.0], [0.0, 0.0, 1.0]);
    assert!(one_tri().closest_hit(&r).is_none(), "back face must be culled");
}

// ── E: tmin clips hit ─────────────────────────────────────────────────────────

#[test]
fn tmin_clips() {
    let r = ray_minmax([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 2.0, f32::INFINITY);
    assert!(one_tri().closest_hit(&r).is_none());
}

// ── F: tmax clips hit ─────────────────────────────────────────────────────────

#[test]
fn tmax_clips() {
    let r = ray_minmax([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 0.0, 0.5);
    assert!(one_tri().closest_hit(&r).is_none());
}

// ── G: tmax includes hit ──────────────────────────────────────────────────────

#[test]
fn tmax_includes() {
    let r = ray_minmax([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 0.0, 1.5);
    let hit = one_tri().closest_hit(&r).expect("should hit within tmax=1.5");
    assert!(approx(hit.t, 1.0));
}

// ── H: tmin prevents self-intersection ───────────────────────────────────────

#[test]
fn self_intersection_guard() {
    let r = ray_minmax([0.25, 0.25, 0.0], [0.0, 0.0, -1.0], 1e-4, f32::INFINITY);
    assert!(one_tri().closest_hit(&r).is_none(), "self-hit at t=0 must be clipped by tmin");
}

// ── I: closest of two triangles ───────────────────────────────────────────────

#[test]
fn closest_of_two() {
    let scene = build_scene(
        &[
            [0.0, 0.0,  0.0], [1.0, 0.0,  0.0], [0.0, 1.0,  0.0],
            [0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0],
        ],
        &[[0, 1, 2], [3, 4, 5]],
    );
    let hit = scene.closest_hit(&ray_ref()).expect("should hit");
    assert!(approx(hit.t, 1.0), "must return near hit (t=1.0), got t={}", hit.t);
}

// ── J: barycentric coordinates at triangle vertices ──────────────────────────

#[test]
fn barycentric_at_vertices() {
    let scene = one_tri();
    let dir = [0.0, 0.0, -1.0];

    let h0 = scene.closest_hit(&ray([0.0, 0.0, 1.0], dir)).expect("v0 hit");
    assert!(approx(h0.u, 0.0) && approx(h0.v, 0.0), "at v0: u={} v={}", h0.u, h0.v);

    let h1 = scene.closest_hit(&ray([1.0, 0.0, 1.0], dir)).expect("v1 hit");
    assert!(approx(h1.u, 1.0) && approx(h1.v, 0.0), "at v1: u={} v={}", h1.u, h1.v);

    let h2 = scene.closest_hit(&ray([0.0, 1.0, 1.0], dir)).expect("v2 hit");
    assert!(approx(h2.u, 0.0) && approx(h2.v, 1.0), "at v2: u={} v={}", h2.u, h2.v);
}

// ── K: any_hit true ───────────────────────────────────────────────────────────

#[test]
fn occluded_true() {
    let r = ray_minmax([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 1e-4, 10.0);
    assert!(one_tri().any_hit(&r));
}

// ── L: any_hit false on spatial miss ─────────────────────────────────────────

#[test]
fn occluded_miss() {
    let r = ray_minmax([0.9, 0.9, 1.0], [0.0, 0.0, -1.0], 1e-4, 10.0);
    assert!(!one_tri().any_hit(&r));
}

// ── M: any_hit respects tmax ──────────────────────────────────────────────────

#[test]
fn occluded_tmax() {
    let r = ray_minmax([0.25, 0.25, 1.0], [0.0, 0.0, -1.0], 1e-4, 0.5);
    assert!(!one_tri().any_hit(&r));
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
        let r = ray([cx, cy, 3.0], [0.0, 0.0, -1.0]);
        let hit = bvh_above
            .closest_hit(&r)
            .unwrap_or_else(|| panic!("expected hit for cx={cx} cy={cy} from above"));
        assert!(approx(hit.t, 2.0), "expected t=2.0, got {}", hit.t);
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
        let r = ray([cx, cy, -3.0], [0.0, 0.0, 1.0]);
        let hit = bvh_below
            .closest_hit(&r)
            .unwrap_or_else(|| panic!("expected hit for cx={cx} cy={cy} from below"));
        assert!(approx(hit.t, 2.0), "expected t=2.0, got {}", hit.t);
    }
}

// ── O: large scene vs. brute-force ────────────────────────────────────────────

#[test]
fn large_scene_vs_brute_force() {
    let n = 9usize;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut tris:  Vec<[u32; 3]> = Vec::new();
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
    let scene = build_scene(&verts, &tris);

    let ray_params: &[([f32;3], [f32;3])] = &[
        ([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]),
        ([0.5, 0.5, 2.0], [0.0, 0.0, -1.0]),
        ([-0.5, 0.3, 2.0], [0.0, 0.0, -1.0]),
        ([0.3, -0.6, 2.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, 2.0], [0.3, 0.2, -1.0]),
        ([0.0, 0.0, 2.0], [-0.4, 0.1, -1.0]),
        ([0.0, 0.0, 2.0], [0.2, -0.3, -1.0]),
        ([0.0, 0.0, 2.0], [-0.1, -0.4, -1.0]),
    ];

    for &(org, dir) in ray_params {
        let r      = ray(org, dir);
        let bvh_t  = scene.closest_hit(&r).map(|h| h.t);
        let brute_t = brute_force_t(&verts, &tris, &r);
        match (bvh_t, brute_t) {
            (Some(bt), Some(bf)) => {
                assert!(
                    approx(bt, bf),
                    "t mismatch for org={org:?} dir={dir:?}: bvh={bt} brute={bf}"
                );
            }
            (None, None) => {}
            _ => panic!(
                "hit/miss disagreement for org={org:?} dir={dir:?}: bvh={bvh_t:?} brute={brute_t:?}"
            ),
        }
    }
}

// ── P: empty scene ────────────────────────────────────────────────────────────

#[test]
fn empty_scene() {
    let scene = build_scene(&[], &[]);
    let r = ray([0.25, 0.25, 1.0], [0.0, 0.0, -1.0]);
    assert!(scene.closest_hit(&r).is_none(), "empty scene must return None");
    assert!(!scene.any_hit(&r), "empty scene must not occlude");
}

// ── Q: prim_id tracks original triangle index ─────────────────────────────────

#[test]
fn prim_id_correct() {
    // Two triangles. The second one (prim_id=1) is closer to the ray origin.
    // prim_id must match the input tris[] index regardless of BVH ordering.
    let scene = build_scene(
        &[
            // tri 0 at z=-2 (far)
            [0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0],
            // tri 1 at z=0 (near)
            [0.0, 0.0,  0.0], [1.0, 0.0,  0.0], [0.0, 1.0,  0.0],
        ],
        &[[0, 1, 2], [3, 4, 5]],
    );
    let r = ray([0.25, 0.25, 1.0], [0.0, 0.0, -1.0]);
    let hit = scene.closest_hit(&r).expect("should hit");
    assert_eq!(hit.prim_id, 1, "near triangle is prim_id=1, got {}", hit.prim_id);
}
