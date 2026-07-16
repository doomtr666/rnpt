use crate::Color;
use nalgebra::{UnitVector3, Vector3};
use std::f32::consts::PI;

// ── Fresnel ───────────────────────────────────────────────────────────────────

/// Schlick approximation for unpolarized dielectric Fresnel.
/// `cos_theta_i` must be in [0, 1] (angle between ray and surface normal).
#[inline]
pub(crate) fn fresnel_dielectric(cos_theta_i: f32, ior: f32) -> f32 {
    let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_theta_i).powi(5)
}

/// Schlick Fresnel for conductors/metals. `f0` is the base reflectance at normal incidence.
#[inline]
pub(crate) fn schlick(f0: Color, cos_h: f32) -> Color {
    let p5 = (1.0 - cos_h).powi(5);
    f0 + (Color::new(1.0, 1.0, 1.0) - f0) * p5
}

// ── GGX microfacet distribution ───────────────────────────────────────────────

/// GGX NDF — D(cos_n_h, α²).
#[inline]
pub(crate) fn ggx_d(cos_n_h: f32, a2: f32) -> f32 {
    let d = (cos_n_h * (a2 - 1.0)).mul_add(cos_n_h, 1.0);
    a2 / (PI * d * d)
}

/// Smith masking function G₁ (uncorrelated Schlick-GGX).
#[inline]
pub(crate) fn smith_g1(cos_theta: f32, a2: f32) -> f32 {
    let denom = cos_theta + ((1.0 - a2) * cos_theta).mul_add(cos_theta, a2).sqrt();
    2.0 * cos_theta / denom
}

/// G₁(o)·G₁(i) / (4·cosₒ·cosᵢ) in one pass — avoids redundant sqrts.
/// smith_g1(x) = 2x / (x + sqrt(a2 + (1-a2)·x²))
/// ⟹ G1(o)·G1(i) / (4·cosₒ·cosᵢ) = 1 / (denom_o · denom_i)
#[inline]
pub(crate) fn smith_g2_over_denom4(cos_o: f32, cos_i: f32, a2: f32) -> f32 {
    let one_minus_a2 = 1.0 - a2;
    let denom_o = cos_o + ((one_minus_a2 * cos_o).mul_add(cos_o, a2)).sqrt();
    let denom_i = cos_i + ((one_minus_a2 * cos_i).mul_add(cos_i, a2)).sqrt();
    1.0 / (denom_o * denom_i)
}

// ── Orthonormal basis ─────────────────────────────────────────────────────────

/// Build (tangent, bitangent) orthonormal to `n` via Frisvad's method.
#[inline]
pub(crate) fn frisvad_onb(n: &Vector3<f32>) -> (Vector3<f32>, Vector3<f32>) {
    let sign = if n.z >= 0.0 { 1.0_f32 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vector3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

// ── Lobe mix ──────────────────────────────────────────────────────────────────

/// Probability of sampling the specular lobe — luminance of the base reflectance F0.
#[inline]
pub(crate) fn spec_prob(f0: Color) -> f32 {
    let lum = 0.2126 * f0.x + 0.7152 * f0.y + 0.0722 * f0.z;
    lum.clamp(0.1, 0.9)
}

// ── Hemisphere sampling ───────────────────────────────────────────────────────

/// Cosine-weighted hemisphere sample around `normal` (Malley's method + Frisvad ONB).
/// Returns `(world direction, pdf = cosθ/π)`.
pub(crate) fn sample_cosine_hemisphere(
    normal: &UnitVector3<f32>,
    u1: f32,
    u2: f32,
) -> (Vector3<f32>, f32) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let local = Vector3::new(r * phi.cos(), r * phi.sin(), (1.0 - u1).sqrt());

    let n = normal.into_inner();
    let (tangent, bitangent) = frisvad_onb(&n);
    let world_dir = (tangent * local.x + bitangent * local.y + n * local.z).normalize();
    (world_dir, local.z / PI)
}
