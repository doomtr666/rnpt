use nalgebra::{Point3, UnitVector3, Vector3};
use wide::f32x8;

use crate::bvh::FlatTriangles;

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AABB {
    pub fn invalid() -> Self {
        Self {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    pub fn extend(&mut self, point: Point3<f32>) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }

    pub fn extend_aabb(&mut self, other: &AABB) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    pub fn center(&self) -> Point3<f32> {
        Point3::from((self.min.coords + self.max.coords) / 2.0)
    }

    pub fn surface_area(&self) -> f32 {
        let extent = self.max - self.min;
        if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
            return 0.0;
        }
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }
}

pub struct TriangleHit {
    pub t: f32,
    pub u: f32,
    pub v: f32,
}

pub struct TriangleHitSimd {
    pub hit: TriangleHit,
    pub lane: usize,
}

/// All bits set in the lanes whose bit is set in `mask` (wide's encoding of `true`).
#[inline(always)]
pub fn double_sided_lanes(mask: u8) -> f32x8 {
    let t = f32::from_bits(u32::MAX);
    f32x8::from(std::array::from_fn::<f32, 8, _>(|i| if mask & (1u8 << i) != 0 { t } else { 0.0 }))
}

#[derive(Clone, Debug)]
pub struct InternalRay {
    pub origin: Point3<f32>,
    pub direction: UnitVector3<f32>,
    pub inv_direction: Vector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

impl InternalRay {
    pub fn new(origin: Point3<f32>, direction: UnitVector3<f32>, tmin: f32, tmax: f32) -> Self {
        let inv_direction = Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
        Self { origin, direction, inv_direction, tmin, tmax }
    }

    /// Slab test, branchless: the reciprocal direction is computed once per ray.
    /// Returns the distance at which the ray enters the box.
    pub fn intersect_aabb(&self, aabb: &AABB, current_t_max: f32) -> Option<f32> {
        let t0 = (aabb.min - self.origin).component_mul(&self.inv_direction);
        let t1 = (aabb.max - self.origin).component_mul(&self.inv_direction);

        let tmin = t0.inf(&t1).max().max(self.tmin);
        let tmax = t0.sup(&t1).min().min(current_t_max);

        if tmax < tmin { None } else { Some(tmin) }
    }

    /// Möller–Trumbore on 8 triangles at once, one per SIMD lane. Back faces are
    /// culled unless the lane is double-sided.
    pub fn intersect_simd_8(&self, soa: &FlatTriangles, current_t_max: f32) -> Option<TriangleHitSimd> {
        let v0_x = f32x8::from(soa.v0_x);
        let v0_y = f32x8::from(soa.v0_y);
        let v0_z = f32x8::from(soa.v0_z);
        let e1_x = f32x8::from(soa.e1_x);
        let e1_y = f32x8::from(soa.e1_y);
        let e1_z = f32x8::from(soa.e1_z);
        let e2_x = f32x8::from(soa.e2_x);
        let e2_y = f32x8::from(soa.e2_y);
        let e2_z = f32x8::from(soa.e2_z);

        let dir_x = f32x8::splat(self.direction.x);
        let dir_y = f32x8::splat(self.direction.y);
        let dir_z = f32x8::splat(self.direction.z);

        // h = dir x e2, det = e1 . h
        let h_x = dir_y * e2_z - dir_z * e2_y;
        let h_y = dir_z * e2_x - dir_x * e2_z;
        let h_z = dir_x * e2_y - dir_y * e2_x;
        let det = e1_x * h_x + e1_y * h_y + e1_z * h_z;

        let epsilon = f32x8::splat(1e-7);
        let ds = double_sided_lanes(soa.double_sided_mask);
        let det_mask = det.simd_ge(epsilon) | (ds & det.simd_le(-epsilon));
        let inv_det = f32x8::splat(1.0) / det;

        // s = origin - v0, u = inv_det * (s . h)
        let s_x = f32x8::splat(self.origin.x) - v0_x;
        let s_y = f32x8::splat(self.origin.y) - v0_y;
        let s_z = f32x8::splat(self.origin.z) - v0_z;
        let u = inv_det * (s_x * h_x + s_y * h_y + s_z * h_z);
        let u_mask = u.simd_ge(f32x8::ZERO) & u.simd_le(f32x8::splat(1.0));

        // q = s x e1, v = inv_det * (dir . q)
        let q_x = s_y * e1_z - s_z * e1_y;
        let q_y = s_z * e1_x - s_x * e1_z;
        let q_z = s_x * e1_y - s_y * e1_x;
        let v = inv_det * (dir_x * q_x + dir_y * q_y + dir_z * q_z);
        let uv_mask = v.simd_ge(f32x8::ZERO) & (u + v).simd_le(f32x8::splat(1.0));

        // t = inv_det * (e2 . q)
        let t = inv_det * (e2_x * q_x + e2_y * q_y + e2_z * q_z);
        let tmin_mask = t.simd_ge(f32x8::splat(self.tmin));
        let tmax_mask = t.simd_le(f32x8::splat(current_t_max));

        let final_mask = det_mask & u_mask & uv_mask & tmin_mask & tmax_mask;
        let mut bitmask = final_mask.to_bitmask();
        if bitmask == 0 {
            return None;
        }

        // Closest valid lane, visiting only the lanes set in the mask.
        let t_arr = t.to_array();
        let mut best_t = current_t_max;
        let mut best_lane = None;
        while bitmask != 0 {
            let lane = bitmask.trailing_zeros() as usize;
            if t_arr[lane] < best_t {
                best_t = t_arr[lane];
                best_lane = Some(lane);
            }
            bitmask &= bitmask - 1;
        }

        best_lane.map(|lane| TriangleHitSimd {
            hit: TriangleHit { t: best_t, u: u.to_array()[lane], v: v.to_array()[lane] },
            lane,
        })
    }
}
