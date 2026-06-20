use crate::{Color, TriangleHit};
use nalgebra::{Point3, UnitVector3, Vector2, Vector4};
use rnpt_bvh::RayAccelerator;

/// Per-triangle metadata — material, light, and vertex indices into `Bvh.vertices`.
pub struct TriangleMeta {
    pub material: u32,
    /// Unified light index if this triangle is an area-light emitter, else `u32::MAX`.
    pub light: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    /// True when this triangle accepts backface hits in the BVH (glass with transmission > 0).
    /// Exit-face detection at shading time uses `geo_normal · wo < 0` instead.
    pub double_sided: bool,
}

pub struct BvhHit {
    pub hit: TriangleHit,
    pub material: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    /// Approximate chunk index (`prim_id / 8`). Kept for ABI compatibility; not used.
    pub chunk_idx: u32,
    /// Unified light index if the hit triangle is an emitter, else `u32::MAX`.
    pub light: u32,
}

pub struct Bvh {
    pub accel:         rnpt_bvh::Scene,
    pub vertices:      Vec<Point3<f32>>,
    pub normals:       Vec<UnitVector3<f32>>,
    pub uvs:           Vec<Vector2<f32>>,
    pub colors:        Vec<Color>,
    /// Per-vertex Mikktspace tangents (xyz = world-space direction, w = bitangent sign ±1).
    /// Zero-initialized (w == 0) when the mesh had no TANGENT attribute.
    pub tangents:      Vec<Vector4<f32>>,
    pub triangle_meta: Vec<TriangleMeta>,
}

impl Bvh {
    pub fn intersect(&self, ray: &crate::Ray) -> Option<BvhHit> {
        let bvh_ray = rnpt_bvh::Ray::new_with_minmax(
            [ray.origin.x, ray.origin.y, ray.origin.z],
            [ray.direction.x, ray.direction.y, ray.direction.z],
            ray.tmin,
            ray.tmax,
        );
        let hit = self.accel.closest_hit(&bvh_ray)?;
        let meta = &self.triangle_meta[hit.prim_id as usize];
        Some(BvhHit {
            hit: TriangleHit { t: hit.t, u: hit.u, v: hit.v },
            material: meta.material,
            v0: meta.v0,
            v1: meta.v1,
            v2: meta.v2,
            chunk_idx: hit.prim_id / 8,
            light: meta.light,
        })
    }

    pub fn is_occluded(&self, ray: &crate::Ray) -> bool {
        let bvh_ray = rnpt_bvh::Ray::new_with_minmax(
            [ray.origin.x, ray.origin.y, ray.origin.z],
            [ray.direction.x, ray.direction.y, ray.direction.z],
            ray.tmin,
            ray.tmax,
        );
        self.accel.any_hit(&bvh_ray)
    }
}
