use crate::{AliasTable, Color, Distribution2D, MeshEmitter, Pcg32, Texture};
use nalgebra::{Point3, UnitVector3, Vector2, Vector3};

use std::f32::consts::PI;
use std::sync::Arc;

/// Minimum squared distance used for point/spot falloff: gives the idealized
/// point light a finite radius (`sqrt` ≈ 0.5) so the `1/d²` near-field can't
/// diverge into fireflies. Tunable. The shadow ray still uses the true distance.
const POINT_LIGHT_MIN_DIST2: f32 = 0.25;

/// One sample of incident light at a shading point, in a common (solid-angle)
/// measure so every light type plugs into the same estimator.
pub struct LightSample {
    pub wi: Vector3<f32>, // shading point -> light (unit)
    pub distance: f32,    // for the shadow-ray tmax (INFINITY for directional)
    pub li: Color,        // incident radiance along wi
    pub pdf: f32,         // solid-angle pdf; 1.0 for delta lights
}

/// An equirectangular HDRI used as an infinite environment light.
#[derive(Clone, Debug)]
pub struct EnvLight {
    pixels: Vec<Color>,
    width: usize,
    height: usize,
    dist: Distribution2D, // weight = luminance * sin(theta), for importance sampling
    intensity: f32,
    rotation: f32, // azimuthal offset in radians (rotates the panorama around Y)
}

impl EnvLight {
    pub fn new(pixels: Vec<Color>, width: usize, height: usize, intensity: f32, rotation: f32) -> Self {
        // Importance weights: luminance × sin(theta) (the equirect solid-angle
        // factor — otherwise the compressed poles get oversampled).
        let mut func = vec![0.0f32; width * height];
        for v in 0..height {
            let sin_t = (PI * (v as f32 + 0.5) / height as f32).sin();
            for u in 0..width {
                let c = pixels[v * width + u];
                let lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
                func[v * width + u] = lum * sin_t;
            }
        }
        let dist = Distribution2D::new(&func, width, height);
        Self {
            pixels,
            width,
            height,
            dist,
            intensity,
            rotation,
        }
    }

    /// Direction (unit, y-up) → equirectangular (u, v) in `[0,1)`.
    #[inline]
    fn dir_to_uv(&self, dir: &Vector3<f32>) -> (f32, f32) {
        let theta = dir.y.clamp(-1.0, 1.0).acos(); // [0, PI]
        let phi = dir.z.atan2(dir.x) - self.rotation; // apply rotation
        let u = phi.rem_euclid(2.0 * PI) / (2.0 * PI);
        (u, theta / PI)
    }

    /// Equirectangular (u, v) → direction (unit, y-up).
    #[inline]
    fn uv_to_dir(&self, u: f32, v: f32) -> Vector3<f32> {
        let phi = 2.0 * PI * u + self.rotation; // apply rotation
        let theta = PI * v;
        let sin_t = theta.sin();
        Vector3::new(sin_t * phi.cos(), theta.cos(), sin_t * phi.sin())
    }

    /// Radiance at (u, v) — bilinear sample × intensity.
    /// u wraps (equirectangular is periodic), v clamps (poles).
    #[inline]
    fn radiance_uv(&self, u: f32, v: f32) -> Color {
        let w = self.width as i32;
        let h = self.height as i32;

        let px = u * w as f32 - 0.5;
        let py = v * h as f32 - 0.5;

        let ix = px.floor() as i32;
        let iy = py.floor() as i32;
        let tx = px - px.floor();
        let ty = py - py.floor();

        let x0 = ix.rem_euclid(w) as usize;
        let x1 = (ix + 1).rem_euclid(w) as usize;
        let y0 = iy.clamp(0, h - 1) as usize;
        let y1 = (iy + 1).clamp(0, h - 1) as usize;

        let c00 = self.pixels[y0 * self.width + x0];
        let c10 = self.pixels[y0 * self.width + x1];
        let c01 = self.pixels[y1 * self.width + x0];
        let c11 = self.pixels[y1 * self.width + x1];

        let c0 = c00.lerp(&c10, tx);
        let c1 = c01.lerp(&c11, tx);
        c0.lerp(&c1, ty) * self.intensity
    }

    fn radiance(&self, dir: &Vector3<f32>) -> Color {
        let (u, v) = self.dir_to_uv(dir);
        self.radiance_uv(u, v)
    }

    pub fn average_luminance(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.pixels.iter()
            .map(|p| 0.2126 * p.x + 0.7152 * p.y + 0.0722 * p.z)
            .sum();
        (sum / self.pixels.len() as f32) * self.intensity
    }
}

/// A scene light. Punctual variants (Point/Directional/Spot) are delta lights
/// (a BRDF ray can never hit them). `Area` and `Environment` are non-delta and
/// participate in MIS.
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
    Environment(Arc<EnvLight>),
}

impl Light {
    #[inline]
    pub fn is_area(&self) -> bool {
        matches!(self, Light::Area(_))
    }

    /// Delta lights (punctual) have a singular distribution and can't be reached
    /// by BRDF sampling — they are excluded from MIS (weight 1).
    #[inline]
    pub fn is_delta(&self) -> bool {
        matches!(
            self,
            Light::Point { .. } | Light::Directional { .. } | Light::Spot { .. }
        )
    }

    /// Solid-angle pdf this light assigns to a direction — non-zero only for the
    /// environment, for the MIS weight when a BRDF ray escapes into it.
    pub fn env_pdf(&self, dir: &Vector3<f32>) -> f32 {
        match self {
            Light::Environment(env) => {
                let (u, v) = env.dir_to_uv(dir);
                let sin_t = (PI * v).sin();
                if sin_t <= 0.0 {
                    0.0
                } else {
                    env.dist.pdf(u, v) / (2.0 * PI * PI * sin_t)
                }
            }
            _ => 0.0,
        }
    }

    /// Environment radiance along a direction (for the ray-escape background);
    /// `0` for non-environment lights.
    pub fn radiance(&self, dir: &Vector3<f32>) -> Color {
        match self {
            Light::Environment(env) => env.radiance(dir),
            _ => Color::zeros(),
        }
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
                // Regularize the unphysical 1/d² near-field (finite light radius);
                // the shadow ray below still travels the true distance.
                let d2_falloff = d2.max(POINT_LIGHT_MIN_DIST2);
                Some(LightSample {
                    wi: to / dist,
                    distance: dist,
                    li: color * (intensity / (4.0 * PI * d2_falloff)),
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
                // Phase 1: geometry only — check backface BEFORE the texture lookup.
                let geom = mesh.sample_geom(rng);
                let to = geom.p - x;
                let d2 = to.norm_squared();
                if d2 <= 0.0 {
                    return None;
                }
                let dist = d2.sqrt();
                let wi = to / dist;
                let cos_l = geom.normal.dot(&(-wi));
                if cos_l <= 0.0 {
                    return None; // emitter back-facing the shading point
                }
                // Phase 2: emissive texture — only reached for front-facing samples.
                let le = mesh.le_at(&geom, textures);
                // area-measure pdf -> solid-angle pdf
                let pdf = geom.pdf_area * d2 / cos_l;
                if pdf <= 0.0 {
                    return None;
                }
                Some(LightSample {
                    wi,
                    distance: dist,
                    li: le,
                    pdf,
                })
            }
            Light::Environment(env) => {
                // Importance-sample the HDRI by its energy, then convert the
                // image-space pdf to solid angle (Jacobian 2π²·sinθ).
                let (uv, pdf_uv) = env.dist.sample(rng.next_f32(), rng.next_f32());
                if pdf_uv <= 0.0 {
                    return None;
                }
                let sin_t = (PI * uv.1).sin();
                if sin_t <= 0.0 {
                    return None;
                }
                Some(LightSample {
                    wi: env.uv_to_dir(uv.0, uv.1),
                    distance: f32::INFINITY,
                    li: env.radiance_uv(uv.0, uv.1),
                    pdf: pdf_uv / (2.0 * PI * PI * sin_t),
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

// ---------------------------------------------------------------------------
// ReSTIR per-primitive light list
// ---------------------------------------------------------------------------

/// A single emissive triangle stored inline for O(1) ReSTIR candidate sampling.
#[derive(Clone, Debug)]
pub struct RestirTriangle {
    pub v0: Point3<f32>,
    pub v1: Point3<f32>,
    pub v2: Point3<f32>,
    pub normal: UnitVector3<f32>,
    pub uv0: Vector2<f32>,
    pub uv1: Vector2<f32>,
    pub uv2: Vector2<f32>,
    pub emissive: Color,
    pub emissive_texture: Option<u32>,
    pub area: f32,
}

/// One entry in the ReSTIR candidate list.
/// `Triangle` = one emissive triangle (from a `MeshEmitter`).
/// `Punctual` = delegate to `lights[idx]` (Point / Directional / Spot / Environment).
///   `delta` = true for Point/Directional/Spot (no BRDF-side MIS), false for Environment.
#[derive(Clone, Debug)]
pub enum RestirEntry {
    Triangle(RestirTriangle),
    Punctual { idx: usize, delta: bool },
}

impl RestirEntry {
    /// Delta lights (punctual) never participate in BRDF-side MIS.
    #[inline]
    pub fn is_delta(&self) -> bool {
        match self {
            RestirEntry::Punctual { delta, .. } => *delta,
            RestirEntry::Triangle(_) => false,
        }
    }
}

/// Sample a `LightSample` from one `RestirEntry`.
/// For triangles: uniform barycentric + backface check + deferred texture.
/// For punctual: delegates to `lights[i].sample_li()`.
pub fn sample_restir_entry(
    entry: &RestirEntry,
    x: &Point3<f32>,
    rng: &mut Pcg32,
    textures: &[Texture],
    lights: &[Light],
) -> Option<LightSample> {
    match entry {
        RestirEntry::Triangle(tri) => {
            let mut u = rng.next_f32();
            let mut v = rng.next_f32();
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let w = 1.0 - u - v;
            let p = Point3::from(tri.v0.coords * w + tri.v1.coords * u + tri.v2.coords * v);
            let to = p - x;
            let d2 = to.norm_squared();
            if d2 <= 0.0 {
                return None;
            }
            let dist = d2.sqrt();
            let wi = to / dist;
            let cos_l = tri.normal.dot(&(-wi));
            if cos_l <= 0.0 {
                return None;
            }
            let uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
            let mut le = tri.emissive;
            if let Some(tex_idx) = tri.emissive_texture {
                if (tex_idx as usize) < textures.len() {
                    le = le.component_mul(&textures[tex_idx as usize].sample_bilinear(uv));
                }
            }
            // area PDF -> solid-angle PDF
            let pdf = d2 / (tri.area * cos_l);
            if pdf <= 0.0 {
                return None;
            }
            Some(LightSample { wi, distance: dist, li: le, pdf })
        }
        RestirEntry::Punctual { idx, .. } => lights.get(*idx)?.sample_li(x, rng, textures),
    }
}

#[inline]
fn luminance(c: Color) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Build the per-primitive light list used by ReSTIR candidate sampling.
///
/// Each emissive triangle in every `Light::Area` becomes one `RestirEntry::Triangle`.
/// Punctual and environment lights each become one `RestirEntry::Punctual`.
/// An `AliasTable` is built with weights proportional to emissive power so that
/// bright lights are sampled more often.
pub fn build_restir_lights(lights: &[Light]) -> (Vec<RestirEntry>, AliasTable) {
    let mut entries: Vec<RestirEntry> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    for (i, light) in lights.iter().enumerate() {
        match light {
            Light::Area(mesh) => {
                let base_lum = luminance(mesh.emissive()).max(1e-6);
                for tri in mesh.tris() {
                    let area = tri.area();
                    if area <= 0.0 {
                        continue;
                    }
                    entries.push(RestirEntry::Triangle(RestirTriangle {
                        v0: tri.v0,
                        v1: tri.v1,
                        v2: tri.v2,
                        normal: tri.normal,
                        uv0: tri.uv0,
                        uv1: tri.uv1,
                        uv2: tri.uv2,
                        emissive: mesh.emissive(),
                        emissive_texture: mesh.emissive_texture(),
                        area,
                    }));
                    weights.push(area * base_lum);
                }
            }
            Light::Point { color, intensity, .. } | Light::Spot { color, intensity, .. } => {
                entries.push(RestirEntry::Punctual { idx: i, delta: true });
                weights.push((intensity * luminance(*color)).max(1e-6));
            }
            Light::Directional { color, intensity, .. } => {
                entries.push(RestirEntry::Punctual { idx: i, delta: true });
                weights.push((intensity * luminance(*color)).max(1e-6));
            }
            Light::Environment(env) => {
                entries.push(RestirEntry::Punctual { idx: i, delta: false });
                weights.push(env.average_luminance().max(1e-6));
            }
        }
    }

    if entries.is_empty() {
        return (entries, AliasTable::new(&[1.0]));
    }

    let alias = AliasTable::new(&weights);
    (entries, alias)
}
