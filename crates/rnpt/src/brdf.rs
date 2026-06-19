use crate::{Color, Pcg32};
use nalgebra::{UnitVector3, Vector3};
use std::f32::consts::PI;

pub struct BrdfSample {
    pub wi: Vector3<f32>, // sampled incident direction (next bounce), world space
    pub f: Color,         // f_r(wo, wi)
    pub pdf: f32,         // solid-angle pdf of wi
}

/// Surface scattering model. All directions world-space; `wo` toward previous vertex.
pub enum Brdf {
    Lambertian { albedo: Color },
    /// GGX Cook-Torrance with metallic-roughness workflow (glTF spec).
    /// Diffuse + specular lobes sampled jointly via MIS mixture.
    CookTorrance { albedo: Color, metallic: f32, roughness: f32 },
}

// ── Microfacet helpers ────────────────────────────────────────────────────────

/// Schlick Fresnel approximation.
#[inline]
fn schlick(f0: Color, cos_h: f32) -> Color {
    let p5 = (1.0 - cos_h).powi(5);
    f0 + (Color::new(1.0, 1.0, 1.0) - f0) * p5
}

/// GGX NDF — D(cos_n_h, a²).
#[inline]
fn ggx_d(cos_n_h: f32, a2: f32) -> f32 {
    let d = cos_n_h * cos_n_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

/// Smith masking function G₁ (uncorrelated Schlick-GGX).
#[inline]
fn smith_g1(cos_theta: f32, a2: f32) -> f32 {
    let denom = cos_theta + (a2 + (1.0 - a2) * cos_theta * cos_theta).sqrt();
    2.0 * cos_theta / denom
}

/// Builds (tangent, bitangent) orthonormal to `n` via Frisvad.
#[inline]
fn frisvad_onb(n: &Vector3<f32>) -> (Vector3<f32>, Vector3<f32>) {
    let sign = if n.z >= 0.0 { 1.0_f32 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vector3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// Probability of sampling the specular lobe — luminance of the base reflectance F0.
#[inline]
fn spec_prob(f0: Color) -> f32 {
    let lum = 0.2126 * f0.x + 0.7152 * f0.y + 0.0722 * f0.z;
    lum.clamp(0.1, 0.9)
}

// ── BRDF implementation ───────────────────────────────────────────────────────

impl Brdf {
    /// f_r(wo, wi). The integrator applies the cosine term and visibility separately.
    pub fn eval(&self, normal: &UnitVector3<f32>, wo: &Vector3<f32>, wi: &Vector3<f32>) -> Color {
        match self {
            Brdf::Lambertian { albedo } => albedo / PI,

            Brdf::CookTorrance { albedo, metallic, roughness } => {
                let cos_n_i = normal.dot(wi);
                let cos_n_o = normal.dot(wo);
                if cos_n_i <= 0.0 || cos_n_o <= 0.0 {
                    return Color::zeros();
                }

                let h = (*wo + *wi).normalize();
                let cos_n_h = normal.dot(&h).max(0.0);
                let cos_h_o = h.dot(wo).max(0.0);

                // Disney remapping: α = roughness², α² = roughness⁴.
                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;

                // Base reflectance: 4% for dielectrics, albedo for metals.
                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let f = schlick(f0, cos_h_o);
                let d = ggx_d(cos_n_h, a2);
                let g = smith_g1(cos_n_o, a2) * smith_g1(cos_n_i, a2);

                let denom = (4.0 * cos_n_o * cos_n_i).max(1e-7);
                let f_spec = f * (d * g / denom);

                // Diffuse: no energy from the specular Fresnel, none for metals.
                let kd = (Color::new(1.0, 1.0, 1.0) - f) * (1.0 - *metallic);
                let f_diff = kd.component_mul(albedo) / PI;

                f_diff + f_spec
            }
        }
    }

    /// Solid-angle pdf of sampling `wi` from this BRDF given `wo`.
    pub fn pdf(&self, normal: &UnitVector3<f32>, wo: &Vector3<f32>, wi: &Vector3<f32>) -> f32 {
        match self {
            Brdf::Lambertian { .. } => normal.dot(wi).max(0.0) / PI,

            Brdf::CookTorrance { albedo, metallic, roughness } => {
                let cos_n_i = normal.dot(wi);
                if cos_n_i <= 0.0 {
                    return 0.0;
                }

                let h = (*wo + *wi).normalize();
                let cos_n_h = normal.dot(&h).max(0.0);
                let cos_h_o = h.dot(wo).max(0.0);

                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;

                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let p_spec = spec_prob(f0);

                // pdf of wi coming from the specular lobe (NDF sampling + Jacobian).
                let d = ggx_d(cos_n_h, a2);
                let pdf_spec = if cos_h_o > 1e-7 { d * cos_n_h / (4.0 * cos_h_o) } else { 0.0 };
                let pdf_diff = cos_n_i / PI;

                p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff
            }
        }
    }

    /// Importance-sample an incident direction. Returns `None` if degenerate.
    pub fn sample(
        &self,
        normal: &UnitVector3<f32>,
        wo: &Vector3<f32>,
        rng: &mut Pcg32,
    ) -> Option<BrdfSample> {
        match self {
            Brdf::Lambertian { albedo } => {
                let (wi, pdf) = sample_cosine_hemisphere(normal, rng.next_f32(), rng.next_f32());
                if pdf <= 0.0 {
                    return None;
                }
                Some(BrdfSample { wi, f: albedo / PI, pdf })
            }

            Brdf::CookTorrance { albedo, metallic, roughness } => {
                let n = normal.into_inner();
                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;

                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let p_spec = spec_prob(f0);

                let wi = if rng.next_f32() < p_spec {
                    // ── Specular lobe: importance-sample the GGX NDF ──────────
                    let (tangent, bitangent) = frisvad_onb(&n);
                    let u1 = rng.next_f32();
                    let u2 = rng.next_f32();

                    // Sample half-vector from GGX NDF (spherical inversion).
                    let cos_theta_h =
                        ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).sqrt().clamp(0.0, 1.0);
                    let sin_theta_h = (1.0 - cos_theta_h * cos_theta_h).sqrt();
                    let phi = 2.0 * PI * u2;
                    let local_h = Vector3::new(
                        sin_theta_h * phi.cos(),
                        sin_theta_h * phi.sin(),
                        cos_theta_h,
                    );
                    let h = (tangent * local_h.x + bitangent * local_h.y + n * local_h.z)
                        .normalize();

                    let dot_h_wo = h.dot(wo);
                    if dot_h_wo <= 0.0 {
                        return None;
                    }
                    // Reflect wo around h to get wi.
                    let wi = h * (2.0 * dot_h_wo) - *wo;
                    if n.dot(&wi) <= 0.0 {
                        return None; // reflected below the surface
                    }
                    wi.normalize()
                } else {
                    // ── Diffuse lobe: cosine-weighted hemisphere ───────────────
                    let (wi, _) = sample_cosine_hemisphere(normal, rng.next_f32(), rng.next_f32());
                    wi
                };

                // Combined f and pdf (MIS mixture over both lobes).
                let f = self.eval(normal, wo, &wi);
                let pdf = self.pdf(normal, wo, &wi);
                if pdf <= 0.0 {
                    return None;
                }
                Some(BrdfSample { wi, f, pdf })
            }
        }
    }
}

/// Cosine-weighted hemisphere sample around `normal` (Malley's method + stable
/// Frisvad orthonormal basis). Returns `(world direction, pdf = cos/π)`.
fn sample_cosine_hemisphere(normal: &UnitVector3<f32>, u1: f32, u2: f32) -> (Vector3<f32>, f32) {
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
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
