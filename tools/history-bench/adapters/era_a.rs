//! Adapter for the BVH living inside the `rnpt` crate (2026-06-12 → 2026-06-18).
//!
//! Features, picked by the driver by trying to compile:
//! - `textures`: `Mesh`/`Material`/`Scene` gained uvs, colors and textures (0d2606b).
//! - `occluded`: `Bvh::is_occluded` exists (59e8bde).

use hb_common::*;
use rnpt::{Bvh, BvhBuilder, Material, Mesh, Node, Point3, Ray, Scene, Transform3, Triangle, UnitVector3, Vector3};

/// `BvhBuilder::build` returned `Bvh`, then `(Bvh, Vec<MeshEmitter>)` (19f08c5).
trait IntoBvh {
    fn into_bvh(self) -> Bvh;
}
impl IntoBvh for Bvh {
    fn into_bvh(self) -> Bvh {
        self
    }
}
impl<T> IntoBvh for (Bvh, T) {
    fn into_bvh(self) -> Bvh {
        self.0
    }
}

struct Era(Bvh);

impl Accel for Era {
    type Ray = Ray;

    fn make_ray(&self, r: &RawRay) -> Ray {
        let dir = UnitVector3::new_unchecked(Vector3::new(r.dir[0], r.dir[1], r.dir[2]));
        Ray::new_with_minmax(Point3::new(r.org[0], r.org[1], r.org[2]), dir, r.tmin, r.tmax)
    }

    fn closest(&self, r: &Ray) -> Option<f32> {
        self.0.intersect(r).map(|h| h.hit.t)
    }

    #[cfg(feature = "occluded")]
    fn occluded(&self, r: &Ray) -> bool {
        self.0.is_occluded(r)
    }

    #[cfg(not(feature = "occluded"))]
    fn occluded(&self, r: &Ray) -> bool {
        self.0.intersect(r).is_some()
    }
}

fn build(s: &TriScene) -> Era {
    let n = s.verts.len();
    let up = UnitVector3::new_unchecked(Vector3::new(0.0f32, 1.0, 0.0));
    let mesh = Mesh {
        vertices: s.verts.iter().map(|v| Point3::new(v[0], v[1], v[2])).collect(),
        normals: vec![up; n],
        #[cfg(feature = "textures")]
        uvs: vec![nalgebra::Vector2::zeros(); n],
        #[cfg(feature = "textures")]
        colors: vec![Vector3::new(1.0, 1.0, 1.0); n],
        triangles: s.tris.iter().map(|t| Triangle { v0: t[0], v1: t[1], v2: t[2] }).collect(),
        material: 0,
    };
    let material = Material {
        albedo: Vector3::new(1.0, 1.0, 1.0),
        emissive: Vector3::zeros(),
        #[cfg(feature = "textures")]
        albedo_texture: None,
        #[cfg(feature = "textures")]
        emissive_texture: None,
    };
    let scene = Scene {
        meshes: vec![mesh],
        materials: vec![material],
        #[cfg(feature = "textures")]
        textures: vec![],
        lights: vec![],
        nodes: vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0] }],
        roots: vec![0],
        cameras: vec![],
    };
    Era(BvhBuilder::new(&scene).build().into_bvh())
}

fn main() {
    let variant = match (cfg!(feature = "textures"), cfg!(feature = "occluded")) {
        (false, false) => "a",
        (false, true) => "a+occluded",
        (true, false) => "a+textures",
        (true, true) => "a+textures+occluded",
    };
    adapter_main(AdapterInfo { variant, native_occlusion: cfg!(feature = "occluded") }, build);
}
