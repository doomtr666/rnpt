//! Adapter for the standalone `rnpt-bvh` crate (2026-06-18 → today).

use hb_common::*;
use rnpt_bvh::{Geometry, Ray, RayAccelerator, Scene};

struct Era(Scene);

impl Accel for Era {
    type Ray = Ray;

    fn make_ray(&self, r: &RawRay) -> Ray {
        Ray::new_with_minmax(r.org, r.dir, r.tmin, r.tmax)
    }

    fn closest(&self, r: &Ray) -> Option<f32> {
        self.0.closest_hit(r).map(|h| h.t)
    }

    fn occluded(&self, r: &Ray) -> bool {
        self.0.any_hit(r)
    }
}

fn build(s: &TriScene) -> Era {
    let mut scene = Scene::new();
    scene.attach_geometry(Geometry::triangle_mesh(&s.verts, &s.tris));
    scene.commit();
    Era(scene)
}

fn main() {
    adapter_main(AdapterInfo { variant: "b", native_occlusion: true }, build);
}
