use crate::{Brdf, Bvh, Camera, Color, ColorExt, Light, Pcg32, Ray, Scene, evaluate_surface};
use nalgebra::{Point3, Transform3, UnitVector3, Vector3};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
#[repr(align(16))]
pub struct Pixel {
    pub accumulated_radiance: Color,
    pub samples: u32,
}

impl Default for Pixel {
    fn default() -> Self {
        Self {
            accumulated_radiance: Color::black(),
            samples: 0,
        }
    }
}

/// Direct-lighting estimator. All three are unbiased and converge to the same
/// image; `Mis` has the lowest variance (and is the default).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SamplingStrategy {
    /// BRDF sampling only: emission on every hit; NEE for delta lights only
    /// (a BRDF ray can never hit a point/directional light).
    BrdfOnly,
    /// Light sampling only (NEE), emission counted on the camera ray only.
    NeeOnly,
    /// Multiple importance sampling (balance heuristic) of NEE and BRDF.
    Mis,
}

#[derive(Clone)]
pub struct PathTracerConfig {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub scene: Arc<Scene>,
    pub bvh: Arc<Bvh>,
    /// Unified light list: scene punctual lights + one area light per emissive mesh.
    pub lights: Arc<Vec<Light>>,
    pub strategy: SamplingStrategy,
}

/// No hard path-length cap: paths terminate via Russian roulette only (PBRT
/// style, unbiased). `u32::MAX` is just a loop guard.
const MAX_BOUNCES: u32 = u32::MAX;
/// Rays start exactly on the surface (single-sided culling rejects
/// self-intersection); this only backs a shadow ray's `tmax` off the light.
const RAY_EPSILON: f32 = 0.001;
/// Russian roulette only kicks in after this many bounces.
const RR_START_BOUNCE: u32 = 3;
/// Floor on the RR termination probability.
const RR_MIN_Q: f32 = 0.05;

/// Russian roulette: unbiasedly terminate low-energy paths. Returns `true` if
/// the path should stop; otherwise rescales `throughput` to stay unbiased.
fn russian_roulette(throughput: &mut Color, bounce: u32, rng: &mut Pcg32) -> bool {
    if bounce <= RR_START_BOUNCE {
        return false;
    }
    let max_channel = throughput.x.max(throughput.y).max(throughput.z);
    let q = (1.0 - max_channel).max(RR_MIN_Q); // termination probability
    if rng.next_f32() < q {
        return true;
    }
    *throughput /= 1.0 - q;
    false
}

pub struct PathTracer {
    config: PathTracerConfig,
    cam2world: Transform3<f32>,
}

impl PathTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        let cam2world = config.camera.compute_camera_to_world();
        Self { config, cam2world }
    }

    fn init_pixel_rng(&self, x: u32, y: u32, frame_index: u32) -> Pcg32 {
        // Pack everything into a single 128-bit primitive
        // High 64 bits: Spatial coordinates (X, Y)
        // Low 64 bits: Temporal coordinate (Frame)
        let mut packed_seed: u128 =
            ((x as u128) << 96) | ((y as u128) << 64) | (frame_index as u128);

        // Quick 128-bit bit-mixing (MurmurHash3 finalizer style for 128-bit blocks)
        packed_seed ^= packed_seed >> 33;
        packed_seed = packed_seed.wrapping_mul(0xff51afd7ed558ccd_u128);
        packed_seed ^= packed_seed >> 33;
        packed_seed = packed_seed.wrapping_mul(0xc4ceb9fe1a85ec53_u128);
        packed_seed ^= packed_seed >> 33;

        // Pass the single 128-bit block to the RNG
        Pcg32::from_seed_128(packed_seed)
    }

    /// Generate the primary camera ray for pixel (x, y), jittered within the pixel.
    pub fn generate_ray(&self, rng: &mut Pcg32, x: f32, y: f32) -> Ray {
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Normalized screen coordinates (-1 to 1) at the center of the pixel
        let jitter_x = rng.next_f32();
        let jitter_y = rng.next_f32();

        let ndc_x = (2.0 * (x + jitter_x) / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * (y + jitter_y) / height);

        let aspect_ratio = width / height;
        let fov_rad = (self.config.camera.fov * std::f32::consts::PI) / 180.0;
        let tan_half_fov = (fov_rad * 0.5).tan();

        // Ray direction in camera space (the camera looks down its local -Z axis).
        let local_dir = Vector3::new(
            ndc_x * aspect_ratio * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        );

        // transform_vector skips the translation (correct for a direction).
        let ray_dir = self.cam2world.transform_vector(&local_dir).normalize();
        // Ray origin = camera position (the translation part of the matrix).
        let ray_origin = self.cam2world.transform_point(&Point3::origin());

        Ray::new(ray_origin, UnitVector3::new_normalize(ray_dir))
    }
    /// Direct lighting via NEE: one shadow ray per light, summed. Every light
    /// type goes through the same `Light::sample_li` interface and the same
    /// solid-angle estimator `f_r * Li * cos_s / pdf`.
    pub fn compute_direct_lighting(
        &self,
        hit_position: &Point3<f32>,
        normal: &UnitVector3<f32>,
        brdf: &Brdf,
        wo: &Vector3<f32>,
        shadow_rays: &mut u64,
        rng: &mut Pcg32,
    ) -> Color {
        let mut total_direct = Color::zeros();
        let textures = &self.config.scene.textures;

        for light in self.config.lights.iter() {
            // BrdfOnly draws no light samples for area lights (their contribution
            // comes from BRDF rays hitting emitters); delta lights always need NEE
            // since a BRDF ray can never hit them.
            if self.config.strategy == SamplingStrategy::BrdfOnly && light.is_area() {
                continue;
            }

            let Some(s) = light.sample_li(hit_position, rng, textures) else {
                continue;
            };

            let cos_s = normal.dot(&s.wi).max(0.0);
            if cos_s <= 0.0 {
                continue;
            }

            // Shadow ray starts at t=0: single-sided culling rejects the
            // originating triangle (det < 0), so no normal offset is needed.
            let tmax = if s.distance.is_finite() {
                s.distance - RAY_EPSILON
            } else {
                f32::INFINITY
            };
            let shadow_ray = Ray::new_with_minmax(
                *hit_position,
                UnitVector3::new_unchecked(s.wi),
                0.0,
                tmax,
            );

            *shadow_rays += 1;
            if !self.config.bvh.is_occluded(&shadow_ray) {
                let f = brdf.eval(normal, wo, &s.wi);
                // MIS balance-heuristic weight against BRDF sampling, for area
                // lights only (delta lights can't be BRDF-sampled → weight 1).
                let w = if self.config.strategy == SamplingStrategy::Mis && light.is_area() {
                    let p_b = brdf.pdf(normal, wo, &s.wi);
                    s.pdf / (s.pdf + p_b)
                } else {
                    1.0
                };
                total_direct += f.component_mul(&s.li) * (cos_s / s.pdf * w);
            }
        }

        total_direct
    }

    /// Trace one path from the camera and return its radiance estimate. Each
    /// bounce: intersect → evaluate the surface → add emission (in NEE mode only
    /// when the emitter is seen directly) → NEE direct lighting → BRDF-sample the
    /// next direction → Russian roulette. No hard depth cap; RR terminates paths.
    /// `rays`/`shadow_rays` accumulate ray counts for stats.
    fn trace_path(
        &self,
        mut ray: Ray,
        rng: &mut Pcg32,
        rays: &mut u64,
        shadow_rays: &mut u64,
    ) -> Color {
        let mut accumulated_radiance = Color::black();
        let mut throughput = Color::white(); // Current path attenuation factor
        // Solid-angle pdf of the BRDF bounce that reached the current vertex
        // (0 for the camera ray) — needed for the MIS weight on emitter hits.
        let mut bsdf_pdf = 0.0f32;

        for bounce in 0..MAX_BOUNCES {
            // Closest-hit intersection with the scene.
            *rays += 1;
            let Some(hit) = self.config.bvh.intersect(&ray) else {
                let sky_radiance = Color::black(); //scene.sample_sky(&ray);
                accumulated_radiance += throughput.component_mul(&sky_radiance);
                break;
            };

            // Evaluate geometry + material at the hit (normal, albedo, emissive).
            let surf = evaluate_surface(&hit, &ray, &self.config.bvh, &self.config.scene);
            let brdf = surf.brdf();
            let wo = -ray.direction.into_inner(); // toward the previous vertex

            let mut local_radiance = Color::black();

            // Emission, weighted per strategy (see SamplingStrategy):
            //  - BrdfOnly: every hit, full weight.
            //  - NeeOnly:  only the camera ray (bounce 0); GI emission is covered
            //              by NEE at the previous vertex (no double counting).
            //  - Mis:      bounce 0 full; otherwise balance-heuristic weight vs the
            //              NEE that could have sampled this emitter point.
            if surf.emissive != Color::zeros() {
                let emit_w = match self.config.strategy {
                    SamplingStrategy::BrdfOnly => 1.0,
                    SamplingStrategy::NeeOnly => {
                        if bounce == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    SamplingStrategy::Mis => {
                        if bounce == 0 {
                            1.0
                        } else {
                            let cos_l = surf.geo_normal.dot(&wo).max(0.0);
                            let p_l = self
                                .config
                                .lights
                                .get(hit.light as usize)
                                .map_or(0.0, |l| l.area_pdf(hit.hit.t * hit.hit.t, cos_l));
                            let denom = bsdf_pdf + p_l;
                            if denom > 0.0 {
                                bsdf_pdf / denom
                            } else {
                                1.0
                            }
                        }
                    }
                };
                local_radiance += surf.emissive * emit_w;
            }

            // Next Event Estimation: analytic lights + emissive-mesh area lights.
            local_radiance += self.compute_direct_lighting(
                &surf.position,
                &surf.normal,
                &brdf,
                &wo,
                shadow_rays,
                rng,
            );

            // Weight this vertex's radiance by the path throughput so far.
            // (No firefly clamp on purpose: clamping biases by clipping energy
            // near lights — the unbiased fix is MIS on the NEE/BRDF samples.)
            accumulated_radiance += throughput.component_mul(&local_radiance);

            // Bounce Setup: importance-sample the BRDF to continue the path.
            let Some(bs) = brdf.sample(&surf.normal, &wo, rng) else {
                break;
            };
            let cos_theta = surf.normal.dot(&bs.wi).max(0.0);
            if bs.pdf <= 0.0 || cos_theta <= 0.0 {
                break;
            }
            bsdf_pdf = bs.pdf; // carried to the next vertex for its MIS weight

            // Update path throughput: f_r * cos_theta / pdf.
            throughput = throughput.component_mul(&(bs.f * (cos_theta / bs.pdf)));

            // Setup the secondary ray for the next loop iteration. Start at t=0:
            // single-sided culling rejects the originating triangle (det < 0).
            ray = Ray::new(surf.position, UnitVector3::new_unchecked(bs.wi));

            // Russian roulette: drop low-energy paths without bias.
            if russian_roulette(&mut throughput, bounce, rng) {
                break;
            }
        }

        accumulated_radiance
    }

    /// Trace one sample for pixel (x, y), accumulating into `pixel`. Returns
    /// `(rays, shadow_rays)`: closest-hit rays (primary + bounces) and any-hit
    /// shadow rays. Thread-safe as long as threads own disjoint pixels.
    pub fn sample_pixel(&self, x: usize, y: usize, pixel: &mut Pixel) -> (u64, u64) {
        let mut rng = self.init_pixel_rng(x as u32, y as u32, pixel.samples);
        let ray = self.generate_ray(&mut rng, x as f32, y as f32);

        let mut rays = 0;
        let mut shadow_rays = 0;
        let sample_color = self.trace_path(ray, &mut rng, &mut rays, &mut shadow_rays);

        pixel.accumulated_radiance += sample_color;
        pixel.samples += 1;
        (rays, shadow_rays)
    }
}
