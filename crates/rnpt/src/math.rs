use nalgebra::{Point3, UnitVector3, Vector3};

pub struct TriangleHit {
    pub t: f32,
    pub u: f32,
    pub v: f32,
}

#[derive(Clone, Debug)]
pub struct Ray {
    pub origin:        Point3<f32>,
    pub direction:     UnitVector3<f32>,
    pub inv_direction: Vector3<f32>,
    pub tmin:          f32,
    pub tmax:          f32,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: UnitVector3<f32>) -> Self {
        Self {
            origin,
            direction,
            inv_direction: Vector3::new(
                1.0 / direction.x,
                1.0 / direction.y,
                1.0 / direction.z,
            ),
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
            inv_direction: Vector3::new(
                1.0 / direction.x,
                1.0 / direction.y,
                1.0 / direction.z,
            ),
            tmin,
            tmax,
        }
    }

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

pub type Color = Vector3<f32>;

pub trait ColorExt {
    fn black() -> Self;
    fn white() -> Self;
    fn rgb(r: f32, g: f32, b: f32) -> Self;
    fn r(&self) -> f32;
    fn g(&self) -> f32;
    fn b(&self) -> f32;
}

impl ColorExt for Color {
    #[inline] fn black() -> Self { Self::zeros() }
    #[inline] fn white() -> Self { Self::repeat(1.0) }
    #[inline] fn rgb(r: f32, g: f32, b: f32) -> Self { Self::new(r, g, b) }
    #[inline] fn r(&self) -> f32 { self.x }
    #[inline] fn g(&self) -> f32 { self.y }
    #[inline] fn b(&self) -> f32 { self.z }
}
