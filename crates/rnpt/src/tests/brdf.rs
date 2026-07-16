use crate::{Brdf, Color, Pcg32};
use crate::microfacet::{fresnel_dielectric, frisvad_onb};
use nalgebra::{UnitVector3, Vector3};
use std::f32::consts::PI;

// ── helpers ───────────────────────────────────────────────────────────────────

fn wo_45deg() -> Vector3<f32> {
    Vector3::new(0.5_f32, 0.866, 0.0).normalize()
}

fn normal_up() -> UnitVector3<f32> {
    UnitVector3::new_normalize(Vector3::new(0.0, 1.0, 0.0))
}

/// Uniformly samples the upper hemisphere around `normal`.
fn sample_uniform_hemisphere(
    normal: &UnitVector3<f32>,
    rng: &mut Pcg32,
) -> Vector3<f32> {
    let u1 = rng.next_f32();
    let u2 = rng.next_f32();
    let cos_theta = u1;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = 2.0 * PI * u2;
    let local = Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
    let (t, b) = frisvad_onb(&normal.into_inner());
    (t * local.x + b * local.y + normal.into_inner() * local.z).normalize()
}

// ── energy conservation ───────────────────────────────────────────────────────

/// MC estimate of ∫ f_r(wo,wi) cos_i dω_i.
/// Physical BRDFs must return ≤ 1 per channel (conservation) and > 0 (not black).
fn reflectance_estimate(brdf: &Brdf, n_samples: usize) -> Color {
    let normal = normal_up();
    let wo = wo_45deg();
    let mut rng = Pcg32::from_seed_128(12345);
    let mut sum = Color::zeros();
    for _ in 0..n_samples {
        if let Some(bs) = brdf.sample(&normal, &wo, &mut rng) {
            if bs.pdf > 0.0 {
                let cos_i = normal.dot(&bs.wi).max(0.0);
                sum += bs.f * (cos_i / bs.pdf);
            }
        }
    }
    sum / n_samples as f32
}

#[test]
fn lambertian_energy_conservation() {
    let albedo = Color::new(0.8, 0.6, 0.4);
    let brdf = Brdf::Lambertian { albedo };
    let r = reflectance_estimate(&brdf, 20_000);
    // Lambertian reflectance = albedo / π * π = albedo
    for (est, expected) in [(r.x, 0.8), (r.y, 0.6), (r.z, 0.4)] {
        assert!(est <= expected + 0.05, "exceeds albedo: {est} vs {expected}");
        assert!(est >= expected - 0.05, "below albedo: {est} vs {expected}");
    }
}

#[test]
fn cook_torrance_energy_conservation_diffuse() {
    // Purely diffuse (metallic=0, high roughness)
    let brdf = Brdf::CookTorrance { albedo: Color::new(0.8, 0.8, 0.8), metallic: 0.0, roughness: 1.0 };
    let r = reflectance_estimate(&brdf, 20_000);
    for ch in [r.x, r.y, r.z] {
        assert!(ch <= 1.05, "energy not conserved: {ch}");
        assert!(ch >= 0.0, "negative reflectance: {ch}");
    }
}

#[test]
fn cook_torrance_energy_conservation_metallic() {
    // Fully metallic mirror-like
    let brdf = Brdf::CookTorrance { albedo: Color::new(1.0, 0.8, 0.2), metallic: 1.0, roughness: 0.1 };
    let r = reflectance_estimate(&brdf, 20_000);
    for ch in [r.x, r.y, r.z] {
        assert!(ch <= 1.05, "energy not conserved: {ch}");
    }
}

// ── pdf normalization ─────────────────────────────────────────────────────────

/// ∫_Ω pdf(wi) dω ≈ 1.  Estimated via uniform hemisphere sampling:
/// E_uniform[pdf] × 2π ≈ 1.
fn pdf_integral(brdf: &Brdf, n_samples: usize) -> f32 {
    let normal = normal_up();
    let wo = wo_45deg();
    let mut rng = Pcg32::from_seed_128(99999);
    let mut sum = 0.0f32;
    for _ in 0..n_samples {
        let wi = sample_uniform_hemisphere(&normal, &mut rng);
        sum += brdf.pdf(&normal, &wo, &wi);
    }
    sum / n_samples as f32 * 2.0 * PI // E[pdf] * hemisphere solid angle
}

#[test]
fn lambertian_pdf_normalizes() {
    let brdf = Brdf::Lambertian { albedo: Color::new(0.5, 0.5, 0.5) };
    let integral = pdf_integral(&brdf, 50_000);
    assert!((integral - 1.0).abs() < 0.03, "pdf integral: {integral}");
}

#[test]
fn cook_torrance_pdf_normalizes_rough() {
    let brdf = Brdf::CookTorrance { albedo: Color::new(0.5, 0.5, 0.5), metallic: 0.0, roughness: 0.8 };
    let integral = pdf_integral(&brdf, 50_000);
    assert!((integral - 1.0).abs() < 0.05, "pdf integral: {integral}");
}

#[test]
fn cook_torrance_pdf_normalizes_smooth() {
    // Very smooth surfaces concentrate the PDF near the specular lobe — harder to integrate
    // uniformly, so we use more samples and a wider tolerance.
    let brdf = Brdf::CookTorrance { albedo: Color::new(0.5, 0.5, 0.5), metallic: 1.0, roughness: 0.2 };
    let integral = pdf_integral(&brdf, 200_000);
    assert!((integral - 1.0).abs() < 0.1, "pdf integral: {integral}");
}

// ── eval_and_pdf consistency ──────────────────────────────────────────────────

#[test]
fn eval_and_pdf_matches_separate_calls() {
    let brdf = Brdf::CookTorrance { albedo: Color::new(0.6, 0.3, 0.9), metallic: 0.5, roughness: 0.4 };
    let normal = normal_up();
    let wo = wo_45deg();
    let mut rng = Pcg32::from_seed_128(777);

    for _ in 0..100 {
        let wi = sample_uniform_hemisphere(&normal, &mut rng);
        let (f_combined, pdf_combined) = brdf.eval_and_pdf(&normal, &wo, &wi);
        let f_sep  = brdf.eval(&normal, &wo, &wi);
        let pdf_sep = brdf.pdf(&normal, &wo, &wi);

        for (a, b) in [(f_combined.x, f_sep.x), (f_combined.y, f_sep.y), (f_combined.z, f_sep.z)] {
            assert!((a - b).abs() < 1e-5, "eval mismatch: {a} vs {b}");
        }
        assert!((pdf_combined - pdf_sep).abs() < 1e-5, "pdf mismatch: {pdf_combined} vs {pdf_sep}");
    }
}

// ── Fresnel ───────────────────────────────────────────────────────────────────

#[test]
fn fresnel_at_normal_incidence() {
    // F(0°, ior) = ((ior-1)/(ior+1))²
    for ior in [1.33f32, 1.5, 2.0] {
        let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
        let f = fresnel_dielectric(1.0, ior);
        assert!((f - r0).abs() < 1e-6, "ior={ior}: got {f}, expected {r0}");
    }
}

#[test]
fn fresnel_approaches_one_at_grazing() {
    // F(90°, any ior) → 1 (total internal reflection at grazing)
    for ior in [1.33f32, 1.5, 2.0] {
        let f = fresnel_dielectric(0.0, ior);
        assert!((f - 1.0).abs() < 1e-5, "ior={ior}: grazing Fresnel = {f}");
    }
}

#[test]
fn fresnel_monotone_in_angle() {
    // F should be non-decreasing as angle increases from 0 to 90°
    let ior = 1.5f32;
    let mut prev = fresnel_dielectric(1.0, ior);
    for i in 1..=20 {
        let cos_theta = 1.0 - i as f32 / 20.0;
        let f = fresnel_dielectric(cos_theta, ior);
        assert!(f >= prev - 1e-5, "Fresnel decreased: {f} < {prev}");
        prev = f;
    }
}
