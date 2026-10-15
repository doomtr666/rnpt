use nalgebra::{Point3, UnitVector3};

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

    /// Remap a point to [0, 1] based on the AABB extents.
    pub fn normalize(&self, v: Point3<f32>) -> Point3<f32> {
        let extent = self.max - self.min;
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
pub struct InternalRay {
    pub origin: Point3<f32>,
    pub direction: UnitVector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

impl InternalRay {
    pub fn new(origin: Point3<f32>, direction: UnitVector3<f32>, tmin: f32, tmax: f32) -> Self {
        Self { origin, direction, tmin, tmax }
    }

    /// Slab test (Kay & Kajiya). Returns the distance at which the ray enters the box.
    pub fn intersect_aabb(&self, aabb: &AABB, current_t_max: f32) -> Option<f32> {
        let mut tmin = self.tmin;
        let mut tmax = current_t_max;

        let inv_d = 1.0 / self.direction.x;
        let mut t0 = (aabb.min.x - self.origin.x) * inv_d;
        let mut t1 = (aabb.max.x - self.origin.x) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin {
            return None;
        }

        let inv_d = 1.0 / self.direction.y;
        let mut t0 = (aabb.min.y - self.origin.y) * inv_d;
        let mut t1 = (aabb.max.y - self.origin.y) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin {
            return None;
        }

        let inv_d = 1.0 / self.direction.z;
        let mut t0 = (aabb.min.z - self.origin.z) * inv_d;
        let mut t1 = (aabb.max.z - self.origin.z) * inv_d;
        if inv_d < 0.0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = if t0 > tmin { t0 } else { tmin };
        tmax = if t1 < tmax { t1 } else { tmax };
        if tmax < tmin {
            return None;
        }

        Some(tmin)
    }

    /// Möller–Trumbore. Back faces are culled unless the triangle is double-sided.
    pub fn intersect_triangle(
        &self,
        v0: &Point3<f32>,
        v1: &Point3<f32>,
        v2: &Point3<f32>,
        double_sided: bool,
    ) -> Option<TriangleHit> {
        const EPSILON: f32 = 1e-7;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        let h = self.direction.cross(&edge2);
        let det = edge1.dot(&h);

        // Ray parallel to the triangle, or back face of a single-sided triangle.
        if det < EPSILON && !(double_sided && det <= -EPSILON) {
            return None;
        }

        let inv_det = 1.0 / det;
        let s = self.origin - v0;

        let u = inv_det * s.dot(&h);
        if !(0.0..=1.0).contains(&u) {
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
}
