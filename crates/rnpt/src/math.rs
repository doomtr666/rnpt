use crate::{Bvh8Node, FlatTriangles};
use nalgebra::{Point3, UnitVector3};
use wide::f32x8;

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AABB {
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

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

    pub fn contains(&self, point: Point3<f32>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    pub fn center(&self) -> Point3<f32> {
        Point3::from((self.min.coords + self.max.coords) / 2.0)
    }

    pub fn extend_aabb(&mut self, other: &AABB) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    pub fn surface_area(&self) -> f32 {
        let extent = self.max - self.min;
        if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
            return 0.0;
        }
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }

    // remap a point to [0, 1] based on the AABB extents
    pub fn normalize(&self, v: Point3<f32>) -> Point3<f32> {
        let extent = self.max - self.min;

        // avoid division by 0
        let ext_x = if extent.x > 0.0 { extent.x } else { 1.0 };
        let ext_y = if extent.y > 0.0 { extent.y } else { 1.0 };
        let ext_z = if extent.z > 0.0 { extent.z } else { 1.0 };

        Point3::new(
            (v.x - self.min.x) / ext_x,
            (v.y - self.min.y) / ext_y,
            (v.z - self.min.z) / ext_z,
        )
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
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: UnitVector3<f32>,
    pub inv_direction: Vector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: UnitVector3<f32>) -> Self {
        Self {
            origin,
            direction,
            inv_direction: Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z),
            tmin: 0.0,
            tmax: f32::INFINITY,
        }
    }

    pub fn new_with_minmax(
        origin: Point3<f32>,
        direction: UnitVector3<f32>,
        tmin: f32,
        tmax: f32,
    ) -> Self {
        Self {
            origin,
            direction,
            inv_direction: Vector3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z),
            tmin,
            tmax,
        }
    }

    /// Fast slab intersection method. Returns the distance `t` to the intersection, or `None` if miss.
    pub fn intersect_aabb(&self, aabb: &AABB, current_t_max: f32) -> Option<f32> {
        let t0 = (aabb.min - self.origin).component_mul(&self.inv_direction);
        let t1 = (aabb.max - self.origin).component_mul(&self.inv_direction);

        let tmin_v = t0.inf(&t1);
        let tmax_v = t0.sup(&t1);

        let tmin = tmin_v.max().max(self.tmin);
        let tmax = tmax_v.min().min(current_t_max);

        if tmax < tmin { None } else { Some(tmin) }
    }

    /// Möller–Trumbore algorithm.
    /// Returns `Some(TriangleHit)` if the ray hits the front face of the triangle
    /// within [ray.tmin, ray.tmax], `None` otherwise.
    pub fn intersect_triangle(
        &self,
        v0: &Point3<f32>,
        v1: &Point3<f32>,
        v2: &Point3<f32>,
    ) -> Option<TriangleHit> {
        const EPSILON: f32 = 1e-7;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let h = self.direction.cross(&edge2);
        let det = edge1.dot(&h);

        // Ray parallel to triangle or triangle back face
        if det < EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;
        let s = self.origin - v0;

        let u = inv_det * s.dot(&h);
        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(&edge1);
        let v = inv_det * self.direction.dot(&q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = inv_det * edge2.dot(&q);
        if t < self.tmin || t > self.tmax {
            return None;
        }

        Some(TriangleHit { t, u, v })
    }

    /// SIMD intersection for 8 triangles at once.
    pub fn closest_triangle_simd8(
        &self,
        soa: &FlatTriangles,
        current_t_max: f32,
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

        // h = dir x e2
        let h_x = dir_y * e2_z - dir_z * e2_y;
        let h_y = dir_z * e2_x - dir_x * e2_z;
        let h_z = dir_x * e2_y - dir_y * e2_x;

        // det = e1 . h
        let det = e1_x * h_x + e1_y * h_y + e1_z * h_z;

        let epsilon = f32x8::splat(1e-7);
        let det_mask = det.simd_ge(epsilon); // culling backfaces
        // Note: wide cmp_* returns a f32x8 where bits are all 1 for true, 0 for false.

        let inv_det = f32x8::splat(1.0) / det;

        let origin_x = f32x8::splat(self.origin.x);
        let origin_y = f32x8::splat(self.origin.y);
        let origin_z = f32x8::splat(self.origin.z);

        // s = origin - v0
        let s_x = origin_x - v0_x;
        let s_y = origin_y - v0_y;
        let s_z = origin_z - v0_z;

        // u = inv_det * (s . h)
        let u = inv_det * (s_x * h_x + s_y * h_y + s_z * h_z);
        let u_mask = u.simd_ge(f32x8::ZERO) & u.simd_le(f32x8::splat(1.0));

        // q = s x e1
        let q_x = s_y * e1_z - s_z * e1_y;
        let q_y = s_z * e1_x - s_x * e1_z;
        let q_z = s_x * e1_y - s_y * e1_x;

        // v = inv_det * (dir . q)
        let v = inv_det * (dir_x * q_x + dir_y * q_y + dir_z * q_z);
        let uv_mask = v.simd_ge(f32x8::ZERO) & (u + v).simd_le(f32x8::splat(1.0));

        // t = inv_det * (e2 . q)
        let t = inv_det * (e2_x * q_x + e2_y * q_y + e2_z * q_z);
        let tmin_mask = t.simd_ge(f32x8::splat(self.tmin));
        let tmax_mask = t.simd_le(f32x8::splat(current_t_max));

        let final_mask = det_mask & u_mask & uv_mask & tmin_mask & tmax_mask;

        let bitmask = final_mask.to_bitmask();
        if bitmask == 0 {
            return None;
        }

        // Missed lanes become +inf so they can never win: no need to skip them.
        let t_valid = final_mask.blend(t, f32x8::splat(f32::INFINITY));

        // Horizontal min: 8 unconditional comparisons, no bit twiddling.
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
            // Extract u and v only for the winning lane!
            let u_arr = u.to_array();
            let v_arr = v.to_array();

            TriangleHitSimd {
                hit: TriangleHit {
                    t: best_t,
                    u: u_arr[lane],
                    v: v_arr[lane],
                },
                lane: lane as u32,
            }
        })
    }

    /// Any-hit test for shadow rays: returns true as soon as any of the 8
    /// triangles is hit within [tmin, t_max]. Same masks as `intersect_simd_8`
    /// (incl. front-face culling, to match the closest-hit render exactly), but
    /// no min/argmin/uv reduction — we only need "is anything there?".
    #[inline(always)]
    pub fn any_triangle_simd8(&self, soa: &FlatTriangles, t_max: f32) -> bool {
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

        let h_x = dir_y * e2_z - dir_z * e2_y;
        let h_y = dir_z * e2_x - dir_x * e2_z;
        let h_z = dir_x * e2_y - dir_y * e2_x;

        let det = e1_x * h_x + e1_y * h_y + e1_z * h_z;
        let det_mask = det.simd_ge(f32x8::splat(1e-7));

        let inv_det = f32x8::splat(1.0) / det;

        let s_x = f32x8::splat(self.origin.x) - v0_x;
        let s_y = f32x8::splat(self.origin.y) - v0_y;
        let s_z = f32x8::splat(self.origin.z) - v0_z;

        let u = inv_det * (s_x * h_x + s_y * h_y + s_z * h_z);
        let u_mask = u.simd_ge(f32x8::ZERO) & u.simd_le(f32x8::splat(1.0));

        let q_x = s_y * e1_z - s_z * e1_y;
        let q_y = s_z * e1_x - s_x * e1_z;
        let q_z = s_x * e1_y - s_y * e1_x;

        let v = inv_det * (dir_x * q_x + dir_y * q_y + dir_z * q_z);
        let uv_mask = v.simd_ge(f32x8::ZERO) & (u + v).simd_le(f32x8::splat(1.0));

        let t = inv_det * (e2_x * q_x + e2_y * q_y + e2_z * q_z);
        let tmin_mask = t.simd_ge(f32x8::splat(self.tmin));
        let tmax_mask = t.simd_le(f32x8::splat(t_max));

        let final_mask = det_mask & u_mask & uv_mask & tmin_mask & tmax_mask;
        final_mask.to_bitmask() != 0
    }

    #[inline(always)]
    pub fn intersect_bvh8(&self, node: &Bvh8Node, t_max: f32) -> (u32, [f32; 8]) {
        use wide::f32x8;

        let p_min_x = f32x8::from(node.p_min_x);
        let p_min_y = f32x8::from(node.p_min_y);
        let p_min_z = f32x8::from(node.p_min_z);

        let p_max_x = f32x8::from(node.p_max_x);
        let p_max_y = f32x8::from(node.p_max_y);
        let p_max_z = f32x8::from(node.p_max_z);

        let origin_x = f32x8::splat(self.origin.x);
        let origin_y = f32x8::splat(self.origin.y);
        let origin_z = f32x8::splat(self.origin.z);

        let inv_dir_x = f32x8::splat(self.inv_direction.x);
        let inv_dir_y = f32x8::splat(self.inv_direction.y);
        let inv_dir_z = f32x8::splat(self.inv_direction.z);

        let t0_x = (p_min_x - origin_x) * inv_dir_x;
        let t1_x = (p_max_x - origin_x) * inv_dir_x;
        let tmin_x = t0_x.fast_min(t1_x);
        let tmax_x = t0_x.fast_max(t1_x);

        let t0_y = (p_min_y - origin_y) * inv_dir_y;
        let t1_y = (p_max_y - origin_y) * inv_dir_y;
        let tmin_y = tmin_x.fast_max(t0_y.fast_min(t1_y));
        let tmax_y = tmax_x.fast_min(t0_y.fast_max(t1_y));

        let t0_z = (p_min_z - origin_z) * inv_dir_z;
        let t1_z = (p_max_z - origin_z) * inv_dir_z;
        let tmin_z = tmin_y
            .fast_max(t0_z.fast_min(t1_z))
            .fast_max(f32x8::splat(self.tmin));
        let tmax_z = tmax_y
            .fast_min(t0_z.fast_max(t1_z))
            .fast_min(f32x8::splat(t_max));

        let mask = tmin_z.simd_le(tmax_z);
        (mask.to_bitmask() as u32, tmin_z.to_array())
    }

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.origin + t * self.direction.into_inner()
    }
}

use nalgebra::Vector3;

// Define the type alias
pub type Color = Vector3<f32>;

// Create an extension trait to add color semantics to Vector3
pub trait ColorExt {
    fn black() -> Self;
    fn white() -> Self;
    fn rgb(r: f32, g: f32, b: f32) -> Self;
    fn r(&self) -> f32;
    fn g(&self) -> f32;
    fn b(&self) -> f32;
}

// Implement the trait for Vector3<f32>
impl ColorExt for Color {
    #[inline]
    fn black() -> Self {
        Self::zeros() // Maps to Vector3::zeros()
    }

    #[inline]
    fn white() -> Self {
        Self::repeat(1.0) // Maps to Vector3::repeat(1.0)
    }

    #[inline]
    fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b)
    }

    #[inline]
    fn r(&self) -> f32 {
        self.x
    }
    #[inline]
    fn g(&self) -> f32 {
        self.y
    }
    #[inline]
    fn b(&self) -> f32 {
        self.z
    }
}
