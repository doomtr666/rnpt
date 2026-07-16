use crate::{Color, Pcg32};
use crate::microfacet::{
    frisvad_onb, ggx_d, sample_cosine_hemisphere, schlick, smith_g2_over_denom4, spec_prob,
};
use nalgebra::{UnitVector3, Vector3};
use std::f32::consts::PI;

pub struct BrdfSample {
    pub wi: Vector3<f32>, // sampled incident direction (next bounce), world space
    pub f: Color,         // f_r(wo, wi)
    pub pdf: f32,         // solid-angle pdf of wi
}

/// Surface scattering model for opaque surfaces. All directions in world space;
/// `wo` points toward the previous vertex.
pub enum Brdf {
    Lambertian { albedo: Color },
    /// GGX Cook-Torrance with metallic-roughness workflow (glTF spec).
    /// Diffuse + specular lobes sampled jointly via MIS mixture.
    CookTorrance { albedo: Color, metallic: f32, roughness: f32 },
}

/// Terms that depend only on the material and `wo` — invariant across all `wi`
/// evaluations in a RIS loop. Built once via `Brdf::precompute()`; passed to
/// `eval_with_precomp()` / `eval_and_pdf_with_precomp()` to skip the `denom_o` sqrt
/// per candidate.
pub struct BrdfPrecomputed {
    pub a2: f32,
    pub one_minus_a2: f32,
    pub f0: Color,
    pub p_s: f32,     // spec_prob(f0) — lobe-mix weight for pdf
    pub cos_n_o: f32,
    pub denom_o: f32, // Smith G₁ denominator for wo
}

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

                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;

                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let f = schlick(f0, cos_h_o);
                let d = ggx_d(cos_n_h, a2);
                let g_over_d = smith_g2_over_denom4(cos_n_o, cos_n_i, a2);
                let f_spec = f * (d * g_over_d);

                let kd = (Color::new(1.0, 1.0, 1.0) - f) * (1.0 - *metallic);
                kd.component_mul(albedo) / PI + f_spec
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

                let d = ggx_d(cos_n_h, a2);
                let pdf_spec = if cos_h_o > 1e-7 { d * cos_n_h / (4.0 * cos_h_o) } else { 0.0 };
                let pdf_diff = cos_n_i / PI;

                p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff
            }
        }
    }

    /// Precompute the wo-dependent (and material-constant) terms shared by all
    /// `eval_with_precomp` / `eval_and_pdf_with_precomp` calls in a RIS loop.
    /// Returns `None` for Lambertian (no wo-dependent terms worth caching) or
    /// when `cos(N,wo) ≤ 0` (surface facing away).
    pub fn precompute(
        &self,
        normal: &UnitVector3<f32>,
        wo: &Vector3<f32>,
    ) -> Option<BrdfPrecomputed> {
        match self {
            Brdf::Lambertian { .. } => None,
            Brdf::CookTorrance { albedo, metallic, roughness } => {
                let cos_n_o = normal.dot(wo);
                if cos_n_o <= 0.0 {
                    return None;
                }
                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;
                let one_minus_a2 = 1.0 - a2;
                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let p_s = spec_prob(f0);
                let denom_o =
                    cos_n_o + ((one_minus_a2 * cos_n_o).mul_add(cos_n_o, a2)).sqrt();
                Some(BrdfPrecomputed { a2, one_minus_a2, f0, p_s, cos_n_o, denom_o })
            }
        }
    }

    /// `eval()` using precomputed wo-side terms — skips the `denom_o` sqrt.
    pub fn eval_with_precomp(
        &self,
        normal: &UnitVector3<f32>,
        wo: &Vector3<f32>,
        wi: &Vector3<f32>,
        pre: &BrdfPrecomputed,
    ) -> Color {
        match self {
            Brdf::Lambertian { albedo } => albedo / PI,
            Brdf::CookTorrance { albedo, metallic, .. } => {
                let cos_n_i = normal.dot(wi);
                if cos_n_i <= 0.0 {
                    return Color::zeros();
                }
                let h = (*wo + *wi).normalize();
                let cos_n_h = normal.dot(&h).max(0.0);
                let cos_h_o = h.dot(wo).max(0.0);

                let denom_i =
                    cos_n_i + ((pre.one_minus_a2 * cos_n_i).mul_add(cos_n_i, pre.a2)).sqrt();
                let g_over_d = 1.0 / (pre.denom_o * denom_i);

                let f = schlick(pre.f0, cos_h_o);
                let d = ggx_d(cos_n_h, pre.a2);
                let f_spec = f * (d * g_over_d);

                let kd = (Color::new(1.0, 1.0, 1.0) - f) * (1.0 - *metallic);
                kd.component_mul(albedo) / PI + f_spec
            }
        }
    }

    /// `eval_and_pdf()` using precomputed wo-side terms.
    pub fn eval_and_pdf_with_precomp(
        &self,
        normal: &UnitVector3<f32>,
        wo: &Vector3<f32>,
        wi: &Vector3<f32>,
        pre: &BrdfPrecomputed,
    ) -> (Color, f32) {
        match self {
            Brdf::Lambertian { albedo } => {
                let cos_n_i = normal.dot(wi).max(0.0);
                (albedo / PI, cos_n_i / PI)
            }
            Brdf::CookTorrance { albedo, metallic, .. } => {
                let cos_n_i = normal.dot(wi);
                if cos_n_i <= 0.0 {
                    return (Color::zeros(), 0.0);
                }
                let h = (*wo + *wi).normalize();
                let cos_n_h = normal.dot(&h).max(0.0);
                let cos_h_o = h.dot(wo).max(0.0);

                let denom_i =
                    cos_n_i + ((pre.one_minus_a2 * cos_n_i).mul_add(cos_n_i, pre.a2)).sqrt();
                let g_over_d = 1.0 / (pre.denom_o * denom_i);

                let f = schlick(pre.f0, cos_h_o);
                let d = ggx_d(cos_n_h, pre.a2);
                let f_spec = f * (d * g_over_d);

                let kd = (Color::new(1.0, 1.0, 1.0) - f) * (1.0 - *metallic);
                let brdf_val = kd.component_mul(albedo) / PI + f_spec;

                let pdf_spec = if cos_h_o > 1e-7 { d * cos_n_h / (4.0 * cos_h_o) } else { 0.0 };
                let pdf_val = pre.p_s * pdf_spec + (1.0 - pre.p_s) * cos_n_i / PI;

                (brdf_val, pdf_val)
            }
        }
    }

    /// Combined f_r + pdf in one pass — avoids recomputing the half-vector and GGX
    /// terms when both are needed (e.g. BRDF sampling, RIS loops).
    pub fn eval_and_pdf(
        &self,
        normal: &UnitVector3<f32>,
        wo: &Vector3<f32>,
        wi: &Vector3<f32>,
    ) -> (Color, f32) {
        match self {
            Brdf::Lambertian { albedo } => {
                let cos_n_i = normal.dot(wi).max(0.0);
                (albedo / PI, cos_n_i / PI)
            }
            Brdf::CookTorrance { albedo, metallic, roughness } => {
                let cos_n_i = normal.dot(wi);
                let cos_n_o = normal.dot(wo);
                if cos_n_i <= 0.0 || cos_n_o <= 0.0 {
                    return (Color::zeros(), 0.0);
                }

                let h = (*wo + *wi).normalize();
                let cos_n_h = normal.dot(&h).max(0.0);
                let cos_h_o = h.dot(wo).max(0.0);

                let alpha = (*roughness * *roughness).max(0.001_f32);
                let a2 = alpha * alpha;

                let f0 = Color::new(0.04, 0.04, 0.04) * (1.0 - *metallic) + *albedo * *metallic;
                let f = schlick(f0, cos_h_o);
                let d = ggx_d(cos_n_h, a2);
                let g_over_d = smith_g2_over_denom4(cos_n_o, cos_n_i, a2);
                let f_spec = f * (d * g_over_d);

                let kd = (Color::new(1.0, 1.0, 1.0) - f) * (1.0 - *metallic);
                let brdf_val = kd.component_mul(albedo) / PI + f_spec;

                let p_s = spec_prob(f0);
                let pdf_spec = if cos_h_o > 1e-7 { d * cos_n_h / (4.0 * cos_h_o) } else { 0.0 };
                let pdf_val = p_s * pdf_spec + (1.0 - p_s) * cos_n_i / PI;

                (brdf_val, pdf_val)
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
                    let wi = h * (2.0 * dot_h_wo) - *wo;
                    if n.dot(&wi) <= 0.0 {
                        return None;
                    }
                    wi.normalize()
                } else {
                    // ── Diffuse lobe: cosine-weighted hemisphere ───────────────
                    let (wi, _) = sample_cosine_hemisphere(normal, rng.next_f32(), rng.next_f32());
                    wi
                };

                let (f, pdf) = self.eval_and_pdf(normal, wo, &wi);
                if pdf <= 0.0 {
                    return None;
                }
                Some(BrdfSample { wi, f, pdf })
            }
        }
    }
}
