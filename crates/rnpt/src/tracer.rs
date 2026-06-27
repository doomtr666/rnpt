use crate::{
    Brdf, Bvh, BvhHit, Camera, Color, ColorExt, Light, Pcg32, Ray, Scene, SurfaceInteraction,
    evaluate_surface, sample_glass,
};
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

/// Direct-lighting estimator. All four are unbiased and converge to the same
/// image; `Mis` has the lowest variance for few lights; `ReStirDi` is best
/// for scenes with many area lights.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SamplingStrategy {
    /// BRDF sampling only: emission on every hit; NEE for delta lights only
    /// (a BRDF ray can never hit a point/directional light).
    BrdfOnly,
    /// Light sampling only (NEE), emission counted on the camera ray only.
    NeeOnly,
    /// Multiple importance sampling (balance heuristic) of NEE and BRDF.
    Mis,
    /// Neural Incident Radiance Cache. Inferences incident radiance at bounce 1.
    Nirc,
    /// Traces only the primary bounce (direct lighting + emission).
    DirectOnly,
}

#[derive(Clone)]
pub struct PathTracerConfig {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub scene: Arc<Scene>,
    pub bvh: Arc<Bvh>,
    /// Unified light list: scene punctual lights + area lights + optional environment.
    pub lights: Arc<Vec<Light>>,
    /// Index of the `Light::Environment` in `lights`, if any (for the ray-escape
    /// background / BRDF-side MIS). `None` → black background.
    pub env: Option<usize>,
    pub strategy: SamplingStrategy,
    /// Active network for NIRC inference.
    pub nirc_network: Option<Arc<crate::nirc::NircMlp>>,
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

    /// Hot-swap the inference network without recreating the PathTracer.
    /// Workers call this to pick up NIRC updates without resetting pixel accumulation.
    pub fn set_nirc_network(&mut self, network: Option<Arc<crate::nirc::NircMlp>>) {
        self.config.nirc_network = network;
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
        strategy: SamplingStrategy,
    ) -> Color {
        let mut total_direct = Color::zeros();
        let textures = &self.config.scene.textures;

        for light in self.config.lights.iter() {
            // BrdfOnly draws no light samples for non-delta lights (area + env;
            // their contribution comes from BRDF rays hitting emitters / escaping
            // into the environment); delta lights always need NEE.
            if strategy == SamplingStrategy::BrdfOnly && !light.is_delta() {
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
            let shadow_ray =
                Ray::new_with_minmax(*hit_position, UnitVector3::new_unchecked(s.wi), 0.0, tmax);

            *shadow_rays += 1;
            if !self.config.bvh.is_occluded(&shadow_ray) {
                let f = brdf.eval(normal, wo, &s.wi);
                // MIS balance-heuristic weight against BRDF sampling, for non-delta
                // lights (area + env); delta lights can't be BRDF-sampled → weight 1.
                let w = if strategy == SamplingStrategy::Mis && !light.is_delta() {
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
        strategy: SamplingStrategy,
        max_depth: u32,
    ) -> Color {
        let mut accumulated_radiance = Color::black();
        let mut throughput = Color::white(); // Current path attenuation factor
        // Solid-angle pdf of the BRDF bounce that reached the current vertex
        // (0 for the camera ray) — needed for the MIS weight on emitter hits.
        let mut bsdf_pdf = 0.0f32;

        for bounce in 0..max_depth {
            // Closest-hit intersection with the scene.
            *rays += 1;
            let Some(hit) = self.config.bvh.intersect(&ray) else {
                self.handle_environment_hit(
                    &ray,
                    bounce,
                    bsdf_pdf,
                    &throughput,
                    &mut accumulated_radiance,
                    strategy,
                );
                break;
            };

            // Evaluate geometry + material at the hit (normal, albedo, emissive).
            let surf = evaluate_surface(&hit, &ray, &self.config.bvh, &self.config.scene);
            let wo = -ray.direction.into_inner(); // toward the previous vertex

            // ── Direct Only (Bounce 1) ───────────────────────────────────────
            if strategy == SamplingStrategy::DirectOnly && bounce == 1 {
                break;
            }

            // ── Dielectric (glass) ───────────────────────────────────────────
            if surf.transmission > 0.0 && rng.next_f32() < surf.transmission {
                if !self.handle_dielectric_bounce(
                    &surf,
                    &hit,
                    &wo,
                    &mut throughput,
                    &mut bsdf_pdf,
                    &mut ray,
                    bounce,
                    rng,
                ) {
                    break;
                }
                continue;
            }

            // ── Opaque ───────────────────────────────────────────────────────
            let brdf = surf.brdf();
            if !self.handle_opaque_bounce(
                &surf,
                &hit,
                &wo,
                &brdf,
                &mut throughput,
                &mut bsdf_pdf,
                &mut accumulated_radiance,
                &mut ray,
                shadow_rays,
                bounce,
                rng,
                strategy,
            ) {
                break;
            }
        }

        accumulated_radiance
    }

    /// Traces a full MIS path from pixel (x, y) and returns one training sample per
    /// opaque bounce >= 1. Russian roulette provides unbiased termination — no depth cap.
    ///
    /// The target at bounce n is the suffix-weighted direct radiance from n onward:
    ///   target[n] = (Σ_{k≥n} T_k · direct_k) / T_n
    /// computed by a backward pass after the full path is traced. Emission is excluded
    /// at every vertex; inference re-adds it with the correct MIS weight.
    pub fn collect_training_samples_for_pixel(
        &self,
        x: usize,
        y: usize,
        rng: &mut Pcg32,
    ) -> Vec<(nalgebra::SVector<f32, { crate::nirc::INPUT_DIM }>, nalgebra::SVector<f32, 3>)> {
        // ── Forward pass ──────────────────────────────────────────────────────────────
        // At each opaque bounce n ≥ 1 we record:
        //   input     = encode(ray.origin = pos_{n-1}, ray.dir = d_{n-1→n})
        //   throughput = T_n  (cumulative, accounts for BRDF weights + RR survival)
        //   direct    = NEE at n (emission excluded)

        struct Vertex {
            input: nalgebra::SVector<f32, { crate::nirc::INPUT_DIM }>,
            throughput: Color,
            direct: Color,
        }

        let mut vertices: Vec<Vertex> = Vec::with_capacity(8);
        let mut ray = self.generate_ray(rng, x as f32, y as f32);
        let mut throughput = Color::white();

        for bounce in 0..MAX_BOUNCES {
            let Some(hit) = self.config.bvh.intersect(&ray) else { break; };
            let surf = evaluate_surface(&hit, &ray, &self.config.bvh, &self.config.scene);
            let wo = -ray.direction.into_inner();

            // Transmissive: NRC never queried at inference for dielectrics; skip and stop.
            if surf.transmission > 0.0 {
                break;
            }

            let brdf = surf.brdf();

            // Record training vertex for bounces ≥ 1 (bounce 0 is never queried at inference).
            if bounce >= 1 {
                let input = crate::nirc::NircMlp::encode_inputs(
                    &ray.origin,
                    &ray.direction.into_inner(),
                    &self.config.bvh.bounds_min,
                    &self.config.bvh.bounds_max,
                );
                let mut dummy = 0u64;
                let direct = self.compute_direct_lighting(
                    &surf.position,
                    &surf.normal,
                    &brdf,
                    &wo,
                    &mut dummy,
                    rng,
                    SamplingStrategy::Mis,
                );
                vertices.push(Vertex { input, throughput, direct });
            }

            // Sample next direction and advance.
            let Some(bs) = brdf.sample(&surf.normal, &wo, rng) else { break; };
            let cos_theta = surf.normal.dot(&bs.wi).max(0.0);
            if bs.pdf <= 0.0 || cos_theta <= 0.0 { break; }

            throughput = throughput.component_mul(&(bs.f * (cos_theta / bs.pdf)));
            if russian_roulette(&mut throughput, bounce, rng) { break; }

            ray = Ray::new(surf.position, nalgebra::UnitVector3::new_unchecked(bs.wi));
        }

        if vertices.is_empty() {
            return Vec::new();
        }

        // ── Backward pass ─────────────────────────────────────────────────────────────
        // running_suffix = Σ_{k≥current} T_k · direct_k  (accumulated from the end)
        // target[n] = running_suffix / T_n
        // This is unbiased: no depth truncation, RR compensates path weights.

        let n = vertices.len();
        let mut samples = Vec::with_capacity(n);
        let mut running_suffix = Color::black();

        for v in vertices.iter().rev() {
            running_suffix += v.throughput.component_mul(&v.direct);
            let t = &v.throughput;
            let target = Color::new(
                if t.x > 1e-6 { (running_suffix.x / t.x).max(0.0) } else { 0.0 },
                if t.y > 1e-6 { (running_suffix.y / t.y).max(0.0) } else { 0.0 },
                if t.z > 1e-6 { (running_suffix.z / t.z).max(0.0) } else { 0.0 },
            );
            samples.push((
                v.input,
                nalgebra::SVector::<f32, 3>::new(target.x, target.y, target.z),
            ));
        }

        samples.reverse();
        samples
    }

    fn handle_environment_hit(
        &self,
        ray: &Ray,
        bounce: u32,
        bsdf_pdf: f32,
        throughput: &Color,
        accumulated_radiance: &mut Color,
        strategy: SamplingStrategy,
    ) {
        // Ray escaped: the environment light (background + BRDF-side MIS).
        // Weight mirrors emitter emission: bounce 0 sees the background
        // directly (NEE can't sample it → w=1); deeper bounces are MIS-
        // weighted vs the env NEE that could have sampled this direction.
        if let Some(env_idx) = self.config.env {
            let env = &self.config.lights[env_idx];
            let dir = ray.direction.into_inner();
            let w = match strategy {
                SamplingStrategy::BrdfOnly | SamplingStrategy::DirectOnly => 1.0,
                SamplingStrategy::NeeOnly => {
                    if bounce == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                SamplingStrategy::Mis | SamplingStrategy::Nirc => {
                    if bounce == 0 {
                        1.0
                    } else {
                        let p_l = env.env_pdf(&dir);
                        let denom = bsdf_pdf + p_l;
                        if denom > 0.0 { bsdf_pdf / denom } else { 1.0 }
                    }
                }
            };
            *accumulated_radiance += throughput.component_mul(&env.radiance(&dir)) * w;
        }
    }

    /// Returns `true` if the path should continue, `false` if it terminates.
    fn handle_dielectric_bounce(
        &self,
        surf: &SurfaceInteraction,
        hit: &BvhHit,
        wo: &Vector3<f32>,
        throughput: &mut Color,
        bsdf_pdf: &mut f32,
        ray: &mut Ray,
        bounce: u32,
        rng: &mut Pcg32,
    ) -> bool {
        let is_exit = surf.geo_normal.dot(wo) < 0.0;
        let thick = surf.thickness_factor > 0.0;
        // Flip n so sample_glass always receives a normal in the wo hemisphere.
        let n = if is_exit {
            -surf.normal.into_inner()
        } else {
            surf.normal.into_inner()
        };

        let tint = if thick {
            if is_exit {
                let t = hit.hit.t;
                let d = surf.attenuation_distance;
                if d.is_finite() && d > 0.0 {
                    let inv_d = t / d;
                    Color::new(
                        surf.attenuation_color.x.powf(inv_d),
                        surf.attenuation_color.y.powf(inv_d),
                        surf.attenuation_color.z.powf(inv_d),
                    )
                } else {
                    Color::white()
                }
            } else {
                surf.albedo
            }
        } else {
            surf.albedo // baseColor tints the surface interface
        };

        let Some(gs) = sample_glass(&n, wo, surf.roughness, surf.ior, &tint, is_exit, thick, rng)
        else {
            return false;
        };

        *throughput = throughput.component_mul(&gs.weight);
        *bsdf_pdf = 1e30;
        *ray = Ray::new_with_minmax(
            surf.position,
            UnitVector3::new_normalize(gs.wi),
            RAY_EPSILON,
            f32::INFINITY,
        );

        if russian_roulette(throughput, bounce, rng) {
            return false;
        }
        true
    }

    /// Returns `true` if the path should continue, `false` if it terminates.
    fn handle_opaque_bounce(
        &self,
        surf: &SurfaceInteraction,
        hit: &BvhHit,
        wo: &Vector3<f32>,
        brdf: &Brdf,
        throughput: &mut Color,
        bsdf_pdf: &mut f32,
        accumulated_radiance: &mut Color,
        ray: &mut Ray,
        shadow_rays: &mut u64,
        bounce: u32,
        rng: &mut Pcg32,
        strategy: SamplingStrategy,
    ) -> bool {
        let emission = if surf.emissive != Color::zeros() {
            let emit_w = match strategy {
                SamplingStrategy::BrdfOnly | SamplingStrategy::DirectOnly => 1.0,
                SamplingStrategy::NeeOnly => {
                    if bounce == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                SamplingStrategy::Mis | SamplingStrategy::Nirc => {
                    if bounce == 0 {
                        1.0
                    } else {
                        let cos_l = surf.geo_normal.dot(wo).max(0.0);
                        let p_l = self
                            .config
                            .lights
                            .get(hit.light as usize)
                            .map_or(0.0, |l| l.area_pdf(hit.hit.t * hit.hit.t, cos_l));
                        let denom = *bsdf_pdf + p_l;
                        if denom > 0.0 { *bsdf_pdf / denom } else { 1.0 }
                    }
                }
            };
            surf.emissive * emit_w
        } else {
            Color::black()
        };

        let direct = if strategy == SamplingStrategy::Nirc && bounce == 1 {
            if let Some(network) = &self.config.nirc_network {
                let input = crate::nirc::NircMlp::encode_inputs(
                    &ray.origin,
                    &ray.direction.into_inner(),
                    &self.config.bvh.bounds_min,
                    &self.config.bvh.bounds_max,
                );
                let pred = network.forward(input);
                Color::new(pred[0].max(0.0), pred[1].max(0.0), pred[2].max(0.0))
            } else {
                Color::black()
            }
        } else {
            self.compute_direct_lighting(
                &surf.position,
                &surf.normal,
                brdf,
                wo,
                shadow_rays,
                rng,
                strategy,
            )
        };

        *accumulated_radiance += throughput.component_mul(&(emission + direct));

        if strategy == SamplingStrategy::Nirc && bounce == 1 {
            return false;
        }

        let Some(bs) = brdf.sample(&surf.normal, wo, rng) else {
            return false;
        };
        let cos_theta = surf.normal.dot(&bs.wi).max(0.0);
        if bs.pdf <= 0.0 || cos_theta <= 0.0 {
            return false;
        }
        *bsdf_pdf = bs.pdf;

        *throughput = throughput.component_mul(&(bs.f * (cos_theta / bs.pdf)));

        *ray = Ray::new(surf.position, UnitVector3::new_unchecked(bs.wi));

        if russian_roulette(throughput, bounce, rng) {
            return false;
        }
        true
    }

    /// Trace one sample for pixel (x, y), accumulating into `pixel`. Returns
    /// `(rays, shadow_rays)`: closest-hit rays (primary + bounces) and any-hit
    /// shadow rays. Thread-safe as long as threads own disjoint pixels.
    pub fn sample_pixel(&self, x: usize, y: usize, pixel: &mut Pixel) -> (u64, u64) {
        let mut rng = self.init_pixel_rng(x as u32, y as u32, pixel.samples);
        let ray = self.generate_ray(&mut rng, x as f32, y as f32);

        let mut rays = 0;
        let mut shadow_rays = 0;
        let sample_color = self.trace_path(
            ray,
            &mut rng,
            &mut rays,
            &mut shadow_rays,
            self.config.strategy,
            MAX_BOUNCES,
        );

        pixel.accumulated_radiance += sample_color;
        pixel.samples += 1;
        (rays, shadow_rays)
    }

    pub fn strategy(&self) -> SamplingStrategy {
        self.config.strategy
    }
}
