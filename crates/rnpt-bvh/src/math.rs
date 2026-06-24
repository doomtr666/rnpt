use crate::bvh::{Bvh8Node, FlatTriangles};
use nalgebra::{Point3, UnitVector3, Vector3};
use wide::f32x8;

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
    pub lane: u32,
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
        let inv_direction = Vector3::new(
            1.0 / direction.x,
            1.0 / direction.y,
            1.0 / direction.z,
        );
        Self {
            origin,
            direction,
            inv_direction,
            tmin,
            tmax,
        }
    }

    pub fn closest_triangle_simd8(
        &self,
        soa: &FlatTriangles,
        current_t_max: f32,
        double_sided_mask: u8,
    ) -> Option<TriangleHitSimd> {
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

        let h_x = dir_y.mul_add(e2_z, -(dir_z * e2_y));
        let h_y = dir_z.mul_add(e2_x, -(dir_x * e2_z));
        let h_z = dir_x.mul_add(e2_y, -(dir_y * e2_x));
        let det = e1_z.mul_add(h_z, e1_y.mul_add(h_y, e1_x * h_x));

        let epsilon = f32x8::splat(1e-7);
        // For double-sided triangles, also accept backface hits (det <= -epsilon).
        // Self-hit avoidance for these triangles relies on tmin = RAY_EPSILON in the caller.
        let ds = {
            let t = f32::from_bits(u32::MAX); // all bits set = "true" in wide's mask encoding
            f32x8::from(std::array::from_fn::<f32, 8, _>(|i| {
                if double_sided_mask & (1u8 << i) != 0 { t } else { 0.0_f32 }
            }))
        };
        let det_mask = det.simd_ge(epsilon) | (ds & det.simd_le(-epsilon));
        let inv_det = f32x8::splat(1.0) / det;

        let s_x = f32x8::splat(self.origin.x) - v0_x;
        let s_y = f32x8::splat(self.origin.y) - v0_y;
        let s_z = f32x8::splat(self.origin.z) - v0_z;

        let u = inv_det * s_z.mul_add(h_z, s_y.mul_add(h_y, s_x * h_x));
        let u_mask = u.simd_ge(f32x8::ZERO) & u.simd_le(f32x8::splat(1.0));

        let q_x = s_y.mul_add(e1_z, -(s_z * e1_y));
        let q_y = s_z.mul_add(e1_x, -(s_x * e1_z));
        let q_z = s_x.mul_add(e1_y, -(s_y * e1_x));

        let v = inv_det * dir_z.mul_add(q_z, dir_y.mul_add(q_y, dir_x * q_x));
        let uv_mask = v.simd_ge(f32x8::ZERO) & (u + v).simd_le(f32x8::splat(1.0 + 1e-6));

        let t = inv_det * e2_z.mul_add(q_z, e2_y.mul_add(q_y, e2_x * q_x));
        let tmin_mask = t.simd_ge(f32x8::splat(self.tmin));
        let tmax_mask = t.simd_le(f32x8::splat(current_t_max));

        let final_mask = det_mask & u_mask & uv_mask & tmin_mask & tmax_mask;
        if final_mask.to_bitmask() == 0 {
            return None;
        }

        let t_valid = final_mask.blend(t, f32x8::splat(f32::INFINITY));
        let t_arr = t_valid.to_array();
        let mut best_t = current_t_max;
        let mut best_lane = None;
        for (lane, &ti) in t_arr.iter().enumerate() {
            if ti < best_t {
                best_t = ti;
                best_lane = Some(lane);
            }
        }

        best_lane.map(|lane| {
            let u_arr = u.to_array();
            let v_arr = v.to_array();
            TriangleHitSimd {
                hit: TriangleHit { t: best_t, u: u_arr[lane], v: v_arr[lane] },
                lane: lane as u32,
            }
        })
    }

    #[inline(always)]
    pub fn any_triangle_simd8(&self, soa: &FlatTriangles, t_max: f32, double_sided_mask: u8) -> bool {
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

        let h_x = dir_y.mul_add(e2_z, -(dir_z * e2_y));
        let h_y = dir_z.mul_add(e2_x, -(dir_x * e2_z));
        let h_z = dir_x.mul_add(e2_y, -(dir_y * e2_x));
        let det = e1_z.mul_add(h_z, e1_y.mul_add(h_y, e1_x * h_x));
        let ds = {
            let t = f32::from_bits(u32::MAX);
            f32x8::from(std::array::from_fn::<f32, 8, _>(|i| {
                if double_sided_mask & (1u8 << i) != 0 { t } else { 0.0_f32 }
            }))
        };
        let epsilon = f32x8::splat(1e-7);
        let det_mask = det.simd_ge(epsilon) | (ds & det.simd_le(-epsilon));
        let inv_det = f32x8::splat(1.0) / det;

        let s_x = f32x8::splat(self.origin.x) - v0_x;
        let s_y = f32x8::splat(self.origin.y) - v0_y;
        let s_z = f32x8::splat(self.origin.z) - v0_z;

        let u = inv_det * s_z.mul_add(h_z, s_y.mul_add(h_y, s_x * h_x));
        let u_mask = u.simd_ge(f32x8::ZERO) & u.simd_le(f32x8::splat(1.0));

        let q_x = s_y.mul_add(e1_z, -(s_z * e1_y));
        let q_y = s_z.mul_add(e1_x, -(s_x * e1_z));
        let q_z = s_x.mul_add(e1_y, -(s_y * e1_x));

        let v = inv_det * dir_z.mul_add(q_z, dir_y.mul_add(q_y, dir_x * q_x));
        let uv_mask = v.simd_ge(f32x8::ZERO) & (u + v).simd_le(f32x8::splat(1.0 + 1e-6));

        let t = inv_det * e2_z.mul_add(q_z, e2_y.mul_add(q_y, e2_x * q_x));
        let tmin_mask = t.simd_ge(f32x8::splat(self.tmin));
        let tmax_mask = t.simd_le(f32x8::splat(t_max));

        let final_mask = det_mask & u_mask & uv_mask & tmin_mask & tmax_mask;
        final_mask.to_bitmask() != 0
    }

    #[inline(always)]
    pub fn intersect_bvh8(&self, node: &Bvh8Node, t_max: f32) -> (u32, [f32; 8]) {
        let p_min_x = f32x8::from(node.p_min_x);
        let p_min_y = f32x8::from(node.p_min_y);
        let p_min_z = f32x8::from(node.p_min_z);
        let p_max_x = f32x8::from(node.p_max_x);
        let p_max_y = f32x8::from(node.p_max_y);
        let p_max_z = f32x8::from(node.p_max_z);

        let inv_dir_x = f32x8::splat(self.inv_direction.x);
        let inv_dir_y = f32x8::splat(self.inv_direction.y);
        let inv_dir_z = f32x8::splat(self.inv_direction.z);
        let ox = f32x8::splat(self.origin.x);
        let oy = f32x8::splat(self.origin.y);
        let oz = f32x8::splat(self.origin.z);

        // t = (p - origin) * inv_dir — subtraction first avoids NaN for axis-aligned rays
        // (p * inv_dir + neg_o_inv gives ±inf + ±inf = NaN when direction component = 0)
        let t0_x = (p_min_x - ox) * inv_dir_x;
        let t1_x = (p_max_x - ox) * inv_dir_x;
        let tmin_x = t0_x.fast_min(t1_x);
        let tmax_x = t0_x.fast_max(t1_x);

        let t0_y = (p_min_y - oy) * inv_dir_y;
        let t1_y = (p_max_y - oy) * inv_dir_y;
        let tmin_y = tmin_x.fast_max(t0_y.fast_min(t1_y));
        let tmax_y = tmax_x.fast_min(t0_y.fast_max(t1_y));

        let t0_z = (p_min_z - oz) * inv_dir_z;
        let t1_z = (p_max_z - oz) * inv_dir_z;
        let tmin_z = tmin_y
            .fast_max(t0_z.fast_min(t1_z))
            .fast_max(f32x8::splat(self.tmin));
        let tmax_z = tmax_y
            .fast_min(t0_z.fast_max(t1_z))
            .fast_min(f32x8::splat(t_max));

        let mask = tmin_z.simd_le(tmax_z);
        (mask.to_bitmask() as u32, tmin_z.to_array())
    }
}
