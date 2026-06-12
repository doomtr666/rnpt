use nalgebra::{Point3, UnitVector3};

pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AABB {
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
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

    /// Slab method (Kay & Kajiya). Returns true if the ray intersects the AABB
    /// within [ray.tmin, ray.tmax].
    pub fn intersect_box(&self, box_min: &Point3<f32>, box_max: &Point3<f32>) -> bool {
        let inv_dir = self.direction.map(|c| 1.0 / c);
        let t1 = (box_min - self.origin).component_mul(&inv_dir);
        let t2 = (box_max - self.origin).component_mul(&inv_dir);

        let tenter = t1.zip_map(&t2, f32::min).max();
        let texit = t1.zip_map(&t2, f32::max).min();

        tenter <= texit && texit >= self.tmin && tenter <= self.tmax
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
