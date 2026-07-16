use crate::{Color, Pcg32};
use crate::microfacet::{fresnel_dielectric, frisvad_onb, smith_g1};
use nalgebra::Vector3;
use std::f32::consts::PI;

pub struct GlassSample {
    pub wi: Vector3<f32>,
    /// Multiply throughput by this (accounts for F, G, pdf and the
    /// stochastic reflect/transmit choice).
    pub weight: Color,
    pub is_reflect: bool,
}

/// Snell refraction. `wo` points into the incident medium, `n` is in the same
/// hemisphere as `wo`. `eta` = η_i / η_t. Returns `None` on TIR.
fn refract(wo: &Vector3<f32>, n: &Vector3<f32>, eta: f32) -> Option<Vector3<f32>> {
    let cos_i = n.dot(wo).max(0.0);
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
    if sin2_t >= 1.0 {
        return None; // TIR
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    Some((-eta * *wo + (eta * cos_i - cos_t) * *n).normalize())
}

/// Sample the glass BSDF: GGX specular reflection or GGX-scattered transmission.
///
/// Samples a GGX microfacet `h` and chooses reflect/transmit with Fresnel probability F.
/// With this split the Fresnel factor cancels in both weights, leaving:
///   weight = G₂·(h·wo)/(cosᵢ·cosₙₕ) × tint
///
/// **Thin** (`thick=false`): scatter direction `wo − 2(h·wo)h`; roughness spreads the
///   transmitted beam without net Snell bending (η_eff = 1 over the zero-thickness slab).
///
/// **Thick** (`thick=true`): Snell refraction around `h`. TIR at a microfacet reflects
///   instead with the same weight formula.
///
/// `n` must be in the wo hemisphere (caller flips for exit faces).
/// `tint` is applied only to the transmission weight.
/// The glTF `transmission` factor is NOT handled here — the caller decides the lobe probability.
pub fn sample_glass(
    n: &Vector3<f32>,
    wo: &Vector3<f32>,
    roughness: f32,
    ior: f32,
    tint: &Color,
    is_exit: bool,
    thick: bool,
    rng: &mut Pcg32,
) -> Option<GlassSample> {
    let cos_i = n.dot(wo).max(1e-5);
    let alpha = (roughness * roughness).max(0.001_f32);
    let a2 = alpha * alpha;

    let (tangent, bitangent) = frisvad_onb(n);
    let u1 = rng.next_f32();
    let u2 = rng.next_f32();
    let cos_h = ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).sqrt().clamp(0.0, 1.0);
    let sin_h = (1.0 - cos_h * cos_h).sqrt();
    let phi = 2.0 * PI * u2;
    let local_h = Vector3::new(sin_h * phi.cos(), sin_h * phi.sin(), cos_h);
    let h = (tangent * local_h.x + bitangent * local_h.y + n * local_h.z).normalize();

    let dot_h_wo = h.dot(wo);
    if dot_h_wo <= 0.0 {
        return None;
    }
    let cos_n_h = n.dot(&h).max(1e-5);
    let f = fresnel_dielectric(dot_h_wo, ior);

    // Shared GGX weight (Fresnel F cancels between BSDF value and selection probability).
    let w = |cos_n_wi: f32| -> f32 {
        smith_g1(cos_i, a2) * smith_g1(cos_n_wi, a2) * dot_h_wo / (cos_i * cos_n_h)
    };

    if rng.next_f32() < f {
        // ── Specular reflection ──────────────────────────────────────────────
        let wi_r = (h * (2.0 * dot_h_wo) - *wo).normalize();
        let cos_n_wi_r = n.dot(&wi_r);
        if cos_n_wi_r <= 0.0 {
            return None;
        }
        Some(GlassSample {
            wi: wi_r,
            weight: Color::new(1.0, 1.0, 1.0) * w(cos_n_wi_r),
            is_reflect: true,
        })
    } else {
        // ── Transmission ─────────────────────────────────────────────────────
        if thick {
            let eta = if is_exit { ior } else { 1.0 / ior };
            match refract(wo, &h, eta) {
                None => {
                    // TIR at this microfacet: fall back to reflection.
                    // Divide by (1 - f) because we took the transmission branch with
                    // probability (1 - f) but the TIR Fresnel factor is 1.0.
                    let wi_r = (h * (2.0 * dot_h_wo) - *wo).normalize();
                    let cos_n_wi_r = n.dot(&wi_r);
                    if cos_n_wi_r <= 0.0 {
                        return None;
                    }
                    let tir_weight =
                        (Color::new(1.0, 1.0, 1.0) * w(cos_n_wi_r)) / (1.0 - f).max(1e-5);
                    Some(GlassSample { wi: wi_r, weight: tir_weight, is_reflect: true })
                }
                Some(wi_t) => {
                    let cos_n_wi_t = n.dot(&wi_t).abs().max(1e-5);
                    Some(GlassSample { wi: wi_t, weight: *tint * w(cos_n_wi_t), is_reflect: false })
                }
            }
        } else {
            // Thin glass: scatter transmitted beam via the GGX microfacet.
            // wi_t = wo − 2(h·wo)h (same Jacobian as reflection; net bending = 0 on average).
            let wi_t = (*wo - h * (2.0 * dot_h_wo)).normalize();
            if n.dot(&wi_t) >= 0.0 {
                return None; // grazing h pushed wi_t back above surface
            }
            let cos_n_wi_t = n.dot(&wi_t).abs().max(1e-5);
            Some(GlassSample { wi: wi_t, weight: *tint * w(cos_n_wi_t), is_reflect: false })
        }
    }
}
