use crate::{Color, Pcg32};
use nalgebra::{UnitVector3, Vector3};
use std::f32::consts::PI;

/// One BRDF importance sample.
pub struct BrdfSample {
    pub wi: Vector3<f32>, // sampled incident (next-bounce) direction, world space
    pub f: Color,         // f_r(wo, wi)
    pub pdf: f32,         // solid-angle pdf of wi
}

/// Surface scattering model. Lambertian for now; Cook-Torrance / GGX will be
/// added as variants without touching the integrator (same eval/sample/pdf API).
/// Directions are world-space; `wo` points toward the previous vertex (camera).
pub enum Brdf {
    Lambertian { albedo: Color },
}

impl Brdf {
    /// `f_r(wo, wi)`. The integrator applies the cosine term and visibility.
    pub fn eval(
        &self,
        _normal: &UnitVector3<f32>,
        _wo: &Vector3<f32>,
        _wi: &Vector3<f32>,
    ) -> Color {
        match self {
            Brdf::Lambertian { albedo } => albedo / PI,
        }
    }

    /// Solid-angle pdf of sampling `wi` (needed for MIS).
    pub fn pdf(&self, normal: &UnitVector3<f32>, _wo: &Vector3<f32>, wi: &Vector3<f32>) -> f32 {
        match self {
            Brdf::Lambertian { .. } => normal.dot(wi).max(0.0) / PI,
        }
    }

    /// Importance-sample an incident direction. `None` if degenerate.
    pub fn sample(
        &self,
        normal: &UnitVector3<f32>,
        _wo: &Vector3<f32>,
        rng: &mut Pcg32,
    ) -> Option<BrdfSample> {
        match self {
            Brdf::Lambertian { albedo } => {
                let (wi, pdf) = sample_cosine_hemisphere(normal, rng.next_f32(), rng.next_f32());
                if pdf <= 0.0 {
                    return None;
                }
                Some(BrdfSample {
                    wi,
                    f: albedo / PI,
                    pdf,
                })
            }
        }
    }
}

/// Cosine-weighted hemisphere sample around `normal` (Malley's method + a stable
/// Frisvad orthonormal basis). Returns `(world direction, pdf = cos/PI)`.
fn sample_cosine_hemisphere(normal: &UnitVector3<f32>, u1: f32, u2: f32) -> (Vector3<f32>, f32) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    // local_z is exactly cos(theta)
    let local = Vector3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).sqrt());

    let n = normal.into_inner();
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let tangent = Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bitangent = Vector3::new(b, sign + n.y * n.y * a, -n.y);

    let world_dir = (tangent * local.x + bitangent * local.y + n * local.z).normalize();
    (world_dir, local.z / PI)
}
