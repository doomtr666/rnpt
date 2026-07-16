use crate::{Bvh, BvhBuilder, Material, Mesh, Node, Ray, Scene, Triangle};
use nalgebra::{Point3, Transform3, UnitVector3, Vector2, Vector3};

fn build_bvh(verts: &[[f32; 3]], tris: &[[u32; 3]], material: u32) -> Bvh {
    let up = UnitVector3::new_normalize(Vector3::z());
    let n_mats = (material + 1) as usize;
    let scene = Scene {
        meshes: vec![Mesh {
            vertices:  verts.iter().map(|v| Point3::new(v[0], v[1], v[2])).collect(),
            normals:   vec![up; verts.len()],
            uvs:       vec![Vector2::zeros(); verts.len()],
            colors:    vec![Vector3::new(1.0, 1.0, 1.0); verts.len()],
            tangents:  vec![],
            triangles: tris.iter().map(|t| Triangle { v0: t[0], v1: t[1], v2: t[2] }).collect(),
            material,
        }],
        materials: vec![Material::default(); n_mats],
        textures:  vec![],
        lights:    vec![],
        nodes:     vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0] }],
        roots:     vec![0],
        cameras:   vec![],
    };
    BvhBuilder::new(&scene).build().0
}

fn one_tri(material: u32) -> Bvh {
    build_bvh(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[[0, 1, 2]],
        material,
    )
}

fn ray_down() -> Ray {
    Ray::new(
        Point3::new(0.25, 0.25, 1.0),
        UnitVector3::new_normalize(Vector3::new(0.0, 0.0, -1.0)),
    )
}

// ── rnpt wrapper: metadata propagation ───────────────────────────────────────

#[test]
fn material_index_propagated() {
    let hit = one_tri(2).intersect(&ray_down()).expect("should hit");
    assert_eq!(hit.material, 2);
}

#[test]
fn non_emitter_light_is_sentinel() {
    // Non-emissive meshes must have hit.light == u32::MAX (no area light).
    let hit = one_tri(0).intersect(&ray_down()).expect("should hit");
    assert_eq!(hit.light, u32::MAX);
}

#[test]
fn vertex_indices_in_bounds() {
    let bvh = one_tri(0);
    let hit = bvh.intersect(&ray_down()).expect("should hit");
    let n = bvh.vertices.len() as u32;
    assert!(hit.v0 < n && hit.v1 < n && hit.v2 < n);
}

#[test]
fn scene_bounds_are_finite() {
    let bvh = one_tri(0);
    for v in bvh.bounds_min.iter().chain(bvh.bounds_max.iter()) {
        assert!(v.is_finite());
    }
}
