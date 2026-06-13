use nalgebra::{Point3, UnitVector3};

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

#[derive(Clone, Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: UnitVector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: UnitVector3<f32>) -> Self {
        Self {
            origin,
            direction,
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
            tmin,
            tmax,
        }
    }

    /// Fast slab intersection method. Returns the distance `t` to the intersection, or `None` if miss.
    pub fn intersect_aabb(&self, aabb: &AABB, current_t_max: f32) -> Option<f32> {
        let mut tmin = self.tmin;
        let mut tmax = current_t_max;

        // X axis
        let inv_d = 1.0 / self.direction.x;
        let mut t0 = (aabb.min.x - self.origin.x) * inv_d;
        let mut t1 = (aabb.max.x - self.origin.x) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin { return None; }

        // Y axis
        let inv_d = 1.0 / self.direction.y;
        let mut t0 = (aabb.min.y - self.origin.y) * inv_d;
        let mut t1 = (aabb.max.y - self.origin.y) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin { return None; }

        // Z axis
        let inv_d = 1.0 / self.direction.z;
        let mut t0 = (aabb.min.z - self.origin.z) * inv_d;
        let mut t1 = (aabb.max.z - self.origin.z) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin { return None; }

        Some(tmin)
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

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.origin + t * self.direction.into_inner()
    }
}

use nalgebra::Vector3;

// 1. Define the type alias
pub type Color = Vector3<f32>;

// 2. Create an extension trait to add color semantics to Vector3
pub trait ColorExt {
    fn black() -> Self;
    fn white() -> Self;
    fn rgb(r: f32, g: f32, b: f32) -> Self;
    fn r(&self) -> f32;
    fn g(&self) -> f32;
    fn b(&self) -> f32;
}

// 3. Implement the trait for Vector3<f32>
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
