use crate::{Color, MeshEmitter, Pcg32, Texture};
use nalgebra::{Point3, Vector3};

use std::f32::consts::PI;

/// One sample of incident light at a shading point, in a common (solid-angle)
/// measure so every light type plugs into the same estimator.
pub struct LightSample {
    pub wi: Vector3<f32>, // shading point -> light (unit)
    pub distance: f32,    // for the shadow-ray tmax (INFINITY for directional)
    pub li: Color,        // incident radiance along wi
    pub pdf: f32,         // solid-angle pdf; 1.0 for delta lights
}

/// A scene light. Punctual variants are delta lights (a BRDF ray can never hit
/// them); `Area` wraps an emissive mesh and is the only non-delta type.
#[derive(Clone, Debug)]
pub enum Light {
    Point {
        position: Point3<f32>,
        color: Color,
        intensity: f32,
    },
    Directional {
        direction: Vector3<f32>, // direction the light travels
        color: Color,
        intensity: f32,
    },
    Spot {
        position: Point3<f32>,
        direction: Vector3<f32>,
        color: Color,
        intensity: f32,
    },
    Area(MeshEmitter),
}

impl Light {
    #[inline]
    pub fn is_area(&self) -> bool {
        matches!(self, Light::Area(_))
    }

    /// Delta lights have a singular distribution and can't be reached by BRDF
    /// sampling — they are excluded from MIS later.
    #[inline]
    pub fn is_delta(&self) -> bool {
        !self.is_area()
    }

    /// Solid-angle pdf that NEE would assign to a point on this light, given the
    /// squared distance and emitter-side cosine. `0` for non-area or back-facing
    /// — used for the MIS weight when a BRDF ray lands on an emitter.
    #[inline]
    pub fn area_pdf(&self, dist2: f32, cos_l: f32) -> f32 {
        match self {
            Light::Area(mesh) if cos_l > 0.0 => dist2 / (mesh.total_area() * cos_l),
            _ => 0.0,
        }
    }

    /// Sample incident radiance at `x`. Returns `None` if the light cannot
    /// contribute (degenerate distance, back-facing emitter, ...).
    pub fn sample_li(
        &self,
        x: &Point3<f32>,
        rng: &mut Pcg32,
        textures: &[Texture],
    ) -> Option<LightSample> {
        match self {
            Light::Point {
                position,
                color,
                intensity,
            }
            | Light::Spot {
                position,
                color,
                intensity,
                ..
            } => {
                // (Spot is treated as a point light for now — glTF cone angles
                // are not imported yet.)
                let to = position - x;
                let d2 = to.norm_squared();
                if d2 <= 0.0 {
                    return None;
                }
                let dist = d2.sqrt();
                Some(LightSample {
                    wi: to / dist,
                    distance: dist,
                    li: color * (intensity / (4.0 * PI * d2)),
                    pdf: 1.0,
                })
            }
            Light::Directional {
                direction,
                color,
                intensity,
            } => Some(LightSample {
                wi: -direction, // toward the light
                distance: f32::INFINITY,
                li: color * *intensity,
                pdf: 1.0,
            }),
            Light::Area(mesh) => {
                let s = mesh.sample(rng, textures);
                let to = s.p - x;
                let d2 = to.norm_squared();
                if d2 <= 0.0 {
                    return None;
                }
                let dist = d2.sqrt();
                let wi = to / dist;
                let cos_l = s.normal.dot(&(-wi));
                if cos_l <= 0.0 {
                    return None; // emitter back-facing the shading point
                }
                // area-measure pdf -> solid-angle pdf
                let pdf = s.pdf_area * d2 / cos_l;
                if pdf <= 0.0 {
                    return None;
                }
                Some(LightSample {
                    wi,
                    distance: dist,
                    li: s.le,
                    pdf,
                })
            }
        }
    }
}

/// Build the unified light list: scene punctual lights + one area light per
/// emissive mesh instance (collected during the BVH flatten).
pub fn build_lights(punctual: &[Light], emitters: Vec<MeshEmitter>) -> Vec<Light> {
    let mut lights = punctual.to_vec();
    lights.extend(emitters.into_iter().map(Light::Area));
    lights
}
