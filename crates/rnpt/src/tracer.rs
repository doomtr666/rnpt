use crate::{Brdf, Bvh, Camera, Color, ColorExt, Light, Pcg32, Ray, Reservoir, Scene, evaluate_surface, sample_glass};
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
    /// ReSTIR DI: Reservoir Importance Sampling with temporal reuse.
    /// Generates M=32 light candidates per bounce, selects the best via RIS,
    /// casts 1 shadow ray. Much faster than NEE in many-light scenes.
    ReStirDi,
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
}

/// No hard path-length cap: paths terminate via Russian roulette only (PBRT
/// style, unbiased). `u32::MAX` is just a loop guard.
const MAX_BOUNCES: u32 = u32::MAX;
/// Number of RIS candidates generated per pixel per frame from the lights.
const RESTIR_M: u32 = 32;
/// Number of RIS candidates generated per pixel per frame from the BRDF.
const RESTIR_M_BRDF: u32 = 1;
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
            // BrdfOnly draws no light samples for non-delta lights (area + env;
            // their contribution comes from BRDF rays hitting emitters / escaping
            // into the environment); delta lights always need NEE.
            if self.config.strategy == SamplingStrategy::BrdfOnly && !light.is_delta() {
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
                // MIS balance-heuristic weight against BRDF sampling, for non-delta
                // lights (area + env); delta lights can't be BRDF-sampled → weight 1.
                let w = if matches!(self.config.strategy, SamplingStrategy::Mis | SamplingStrategy::ReStirDi) && !light.is_delta() {
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
                // Ray escaped: the environment light (background + BRDF-side MIS).
                // Weight mirrors emitter emission: bounce 0 sees the background
                // directly (NEE can't sample it → w=1); deeper bounces are MIS-
                // weighted vs the env NEE that could have sampled this direction.
                if let Some(env_idx) = self.config.env {
                    let env = &self.config.lights[env_idx];
                    let dir = ray.direction.into_inner();
                    let w = match self.config.strategy {
                        SamplingStrategy::BrdfOnly => 1.0,
                        SamplingStrategy::NeeOnly => {
                            if bounce == 0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        SamplingStrategy::Mis | SamplingStrategy::ReStirDi => {
                            if bounce == 0 {
                                1.0
                            } else {
                                let p_l = env.env_pdf(&dir);
                                let denom = bsdf_pdf + p_l;
                                if denom > 0.0 {
                                    bsdf_pdf / denom
                                } else {
                                    1.0
                                }
                            }
                        }
                    };
                    accumulated_radiance += throughput.component_mul(&env.radiance(&dir)) * w;
                }
                break;
            };

            // Evaluate geometry + material at the hit (normal, albedo, emissive).
            let surf = evaluate_surface(&hit, &ray, &self.config.bvh, &self.config.scene);
            let wo = -ray.direction.into_inner(); // toward the previous vertex

            // ── Dielectric (glass) ───────────────────────────────────────────
            // glTF model: `transmission` fraction → glass BSDF (GGX reflect or refract);
            // `1 − transmission` fraction → opaque BSDF (diffuse + specular + NEE).
            // Stochastic lobe selection; throughput is divided by the selection probability.
            //
            // Thin  (thickness_factor = 0): GGX-scattered straight-through (eta_eff = 1).
            // Thick (thickness_factor > 0): Snell refraction + Beer-Lambert absorption.
            if surf.transmission > 0.0 && rng.next_f32() < surf.transmission {
                // ── Glass lobe ────────────────────────────────────────────────
                let is_exit = surf.geo_normal.dot(&wo) < 0.0;
                let thick = surf.thickness_factor > 0.0;
                // Flip n so sample_glass always receives a normal in the wo hemisphere.
                let n = if is_exit { -surf.normal.into_inner() } else { surf.normal.into_inner() };

                // Tint table (baseColor applied once at entry; Beer-Lambert at exit):
                //   thin  (front/back) → albedo
                //   thick entry        → albedo
                //   thick exit         → Beer-Lambert only
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

                let Some(gs) = sample_glass(
                    &n, &wo, surf.roughness, surf.ior, &tint, is_exit, thick, rng,
                ) else {
                    break;
                };

                // Do not divide by the glass lobe selection probability (surf.transmission).
                // The unscaled weight correctly preserves the specular reflection 
                // without double-counting it against the opaque branch.
                throughput = throughput.component_mul(&gs.weight);
                // Large pdf → MIS weight ≈ 1 (no NEE through glass).
                bsdf_pdf = 1e30;
                ray = Ray::new_with_minmax(
                    surf.position,
                    UnitVector3::new_normalize(gs.wi),
                    RAY_EPSILON,
                    f32::INFINITY,
                );

                if russian_roulette(&mut throughput, bounce, rng) {
                    break;
                }
                continue;
            }

            // Opaque lobe selected (or transmission = 0).
            // Do not scale throughput to compensate for lobe selection probability,
            // as this elegantly blends the shared specular reflection between branches.

            let brdf = surf.brdf();

            // Emission, weighted per strategy (see SamplingStrategy):
            //  - BrdfOnly: every hit, full weight.
            //  - NeeOnly:  only the camera ray (bounce 0); GI emission is covered
            //              by NEE at the previous vertex (no double counting).
            //  - Mis:      bounce 0 full; otherwise balance-heuristic weight vs the
            //              NEE that could have sampled this emitter point.
            let emission = if surf.emissive != Color::zeros() {
                let emit_w = match self.config.strategy {
                    SamplingStrategy::BrdfOnly => 1.0,
                    SamplingStrategy::NeeOnly => {
                        if bounce == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    SamplingStrategy::Mis | SamplingStrategy::ReStirDi => {
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
                surf.emissive * emit_w
            } else {
                Color::black()
            };

            // Next Event Estimation: analytic lights + emissive-mesh area lights.
            let direct = self.compute_direct_lighting(
                &surf.position,
                &surf.normal,
                &brdf,
                &wo,
                shadow_rays,
                rng,
            );

            // Weight this vertex's radiance by the path throughput so far.
            // (No firefly clamp on purpose — the unbiased fix is MIS.)
            accumulated_radiance += throughput.component_mul(&(emission + direct));

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

    pub fn strategy(&self) -> SamplingStrategy {
        self.config.strategy
    }

    /// ReSTIR DI: RIS candidate selection with temporal reservoir reuse.
    /// Called at bounce 0 instead of `compute_direct_lighting`.
    fn compute_restir_direct(
        &self,
        hit_pos: &Point3<f32>,
        normal: &UnitVector3<f32>,
        brdf: &Brdf,
        wo: &Vector3<f32>,
        reservoir: &mut Reservoir,
        shadow_rays: &mut u64,
        rng: &mut Pcg32,
    ) -> Color {
        let n = self.config.lights.len();
        if n == 0 {
            return Color::zeros();
        }
        let textures = &self.config.scene.textures;

        let mut r = Reservoir::default();

        // Precompute wo-dependent BRDF terms once — shared across all M candidates.
        // For CookTorrance: caches a2, f0, denom_o (contains a sqrt) and p_s.
        let pre = brdf.precompute(normal, wo);

        // MIS Weight Factors
        let m_total = (RESTIR_M + RESTIR_M_BRDF) as f32;
        let w_l_factor = RESTIR_M as f32 / m_total;
        let w_b_factor = RESTIR_M_BRDF as f32 / m_total;

        // RIS: generate M candidates from lights, stream into reservoir.
        // IMPORTANT: every attempted candidate — including invalid ones (cos_s ≤ 0,
        // degenerate pdf, or None from sample_li) — must increment r.m.
        for _ in 0..RESTIR_M {
            let idx = (rng.next_f32() * n as f32) as usize;
            let Some(ls) = self.config.lights[idx.min(n - 1)].sample_li(hit_pos, rng, textures)
            else {
                r.m += 1; // zero-weight candidate still drawn from the source distribution
                continue;
            };
            let cos_s = normal.dot(&ls.wi).max(0.0);
            if cos_s <= 0.0 || ls.pdf <= 0.0 {
                r.m += 1;
                continue;
            }

            // p_hat = lum(brdf * L_i * cos_s) — use precomputed wo-side terms when available.
            let (f, p_b) = match (&pre, self.config.lights[idx.min(n - 1)].is_delta()) {
                (Some(pre), true)  => (brdf.eval_with_precomp(normal, wo, &ls.wi, pre), 0.0_f32),
                (Some(pre), false) => brdf.eval_and_pdf_with_precomp(normal, wo, &ls.wi, pre),
                (None, true)       => (brdf.eval(normal, wo, &ls.wi), 0.0_f32),
                (None, false)      => brdf.eval_and_pdf(normal, wo, &ls.wi),
            };
            let unshadowed = f.component_mul(&ls.li) * cos_s;
            let p_hat = 0.2126 * unshadowed.x + 0.7152 * unshadowed.y + 0.0722 * unshadowed.z;

            let p_l = ls.pdf / n as f32;

            let p_combined = w_l_factor * p_l + w_b_factor * p_b;
            let w = if p_combined > 0.0 { p_hat / p_combined } else { 0.0 };

            let light_pos = if ls.distance.is_finite() {
                hit_pos + ls.wi * ls.distance
            } else {
                Point3::from(ls.wi * 1e15f32)
            };
            r.update(light_pos, ls.li, w, rng);
        }

        // RIS: generate M candidates from BRDF, stream into reservoir.
        for _ in 0..RESTIR_M_BRDF {
            let Some(bs) = brdf.sample(normal, wo, rng) else {
                r.m += 1;
                continue;
            };
            let cos_s = normal.dot(&bs.wi).max(0.0);
            if cos_s <= 0.0 || bs.pdf <= 0.0 {
                r.m += 1;
                continue;
            }

            let mut li = Color::zeros();
            let mut distance = f32::INFINITY;
            let mut p_l = 0.0;

            let brdf_ray = Ray::new(*hit_pos, UnitVector3::new_unchecked(bs.wi));
            if let Some(hit) = self.config.bvh.intersect(&brdf_ray) {
                let surf = evaluate_surface(&hit, &brdf_ray, &self.config.bvh, &self.config.scene);
                if surf.emissive != Color::zeros() {
                    li = surf.emissive;
                    distance = hit.hit.t;
                    let cos_l = surf.geo_normal.dot(&(-bs.wi)).max(0.0);
                    if let Some(light) = self.config.lights.get(hit.light as usize) {
                        p_l = light.area_pdf(distance * distance, cos_l) / n as f32;
                    }
                }
            } else {
                if let Some(env_idx) = self.config.env {
                    let env = &self.config.lights[env_idx];
                    li = env.radiance(&bs.wi);
                    p_l = env.env_pdf(&bs.wi) / n as f32;
                }
            }

            if li == Color::zeros() {
                r.m += 1;
                continue;
            }

            let unshadowed = bs.f.component_mul(&li) * cos_s;
            let p_hat = 0.2126 * unshadowed.x + 0.7152 * unshadowed.y + 0.0722 * unshadowed.z;

            let p_b = bs.pdf;
            let p_combined = w_l_factor * p_l + w_b_factor * p_b;
            let w = if p_combined > 0.0 { p_hat / p_combined } else { 0.0 };

            let light_pos = if distance.is_finite() {
                hit_pos + bs.wi * distance
            } else {
                Point3::from(bs.wi * 1e15f32)
            };
            r.update(light_pos, li, w, rng);
        }

        // Temporal combine: stream previous frame's selected sample.
        // Correct formula (Bitterli 2020): combine_w = p̂_cur(y_prev) * W_prev * m_prev.
        // Recomputing wi from the stored light_pos handles the p̂ mismatch caused by
        // sub-pixel jitter shifting the shading point between frames.
        if reservoir.is_valid() && reservoir.big_w_stored > 0.0 {
            let m_cap = 20 * (RESTIR_M + RESTIR_M_BRDF);
            let capped_prev_m = reservoir.m.min(m_cap);
            let to_prev = reservoir.light_pos.coords - hit_pos.coords;
            let prev_dist = to_prev.norm();
            if prev_dist > 0.0 {
                let wi_prev = to_prev / prev_dist;
                let prev_cos = normal.dot(&wi_prev).max(0.0);
                if prev_cos > 0.0 {
                    let f_prev = match &pre {
                        Some(pre) => brdf.eval_with_precomp(normal, wo, &wi_prev, pre),
                        None => brdf.eval(normal, wo, &wi_prev),
                    };
                    let unshadowed = f_prev.component_mul(&reservoir.li) * prev_cos;
                    let p_hat_cur = 0.2126 * unshadowed.x + 0.7152 * unshadowed.y + 0.0722 * unshadowed.z;

                    // p̂_cur(y_prev) * W_prev * m_prev
                    let combine_w = p_hat_cur * reservoir.big_w_stored * capped_prev_m as f32;
                    let m_before = r.m;
                    r.update(reservoir.light_pos, reservoir.li, combine_w, rng);
                    // update() incremented r.m by 1; set it to the correct combined total.
                    r.m = m_before + capped_prev_m;
                }
            }
        }

        if !r.is_valid() {
            *reservoir = Reservoir::default();
            return Color::zeros();
        }

        // Recompute wi and distance from the stored light position for the current
        // shading point (handles slight jitter shift between frames).
        let to = r.light_pos.coords - hit_pos.coords;
        let dist = to.norm();
        let (wi, tmax) = if dist > 1e15 {
            (to / dist, f32::INFINITY) // infinite light sentinel
        } else {
            (to / dist, dist - RAY_EPSILON)
        };

        let cos_s = normal.dot(&wi).max(0.0);
        let f = match &pre {
            Some(pre) => brdf.eval_with_precomp(normal, wo, &wi, pre),
            None => brdf.eval(normal, wo, &wi),
        };
        let unshadowed = f.component_mul(&r.li) * cos_s;
        let p_hat_y = 0.2126 * unshadowed.x + 0.7152 * unshadowed.y + 0.0722 * unshadowed.z;
        let big_w = r.big_w(p_hat_y);
        // Store before shadow test: the sample stays valid for temporal reuse regardless
        // of visibility (next frame will test visibility again).
        r.big_w_stored = big_w;

        let shadow_ray = Ray::new_with_minmax(*hit_pos, UnitVector3::new_unchecked(wi), 0.0, tmax);
        *shadow_rays += 1;

        let contribution = if p_hat_y > 0.0 && big_w > 0.0 {
            if !self.config.bvh.is_occluded(&shadow_ray) {
                unshadowed * big_w
            } else {
                // IMPORTANT: Bitterli 2020 sets W=0 for occluded candidates.
                // If we temporally reuse a shadowed bright light (like the sun),
                // it causes massive "boiling" artifacts (sticky shadows).
                r.big_w_stored = 0.0;
                Color::zeros()
            }
        } else {
            r.big_w_stored = 0.0;
            Color::zeros()
        };

        // Store updated reservoir for next frame's temporal reuse.
        *reservoir = r;
        contribution
    }

    /// Like `trace_path` but uses ReSTIR DI for the first direct-lighting
    /// vertex and falls back to MIS NEE for all subsequent bounces.
    fn trace_path_restir(
        &self,
        mut ray: Ray,
        rng: &mut Pcg32,
        rays: &mut u64,
        shadow_rays: &mut u64,
        reservoir: &mut Reservoir,
    ) -> Color {
        let mut accumulated_radiance = Color::black();
        let mut throughput = Color::white();
        let mut bsdf_pdf = 0.0f32;
        let mut restir_used = false; // true once ReSTIR was applied at the first opaque surface

        for bounce in 0..MAX_BOUNCES {
            *rays += 1;
            let Some(hit) = self.config.bvh.intersect(&ray) else {
                if let Some(env_idx) = self.config.env {
                    let env = &self.config.lights[env_idx];
                    let dir = ray.direction.into_inner();
                    let w = if bounce == 0 {
                        1.0
                    } else if restir_used {
                        // NeeOnly-style: ReSTIR at bounce 0 already estimated all direct
                        // illumination, including the env. Suppress the BRDF-side env hit.
                        0.0
                    } else {
                        // Bounce 0 was glass → fell back to regular MIS.
                        let p_l = env.env_pdf(&dir);
                        let denom = bsdf_pdf + p_l;
                        if denom > 0.0 { bsdf_pdf / denom } else { 1.0 }
                    };
                    accumulated_radiance += throughput.component_mul(&env.radiance(&dir)) * w;
                }
                break;
            };

            let surf = evaluate_surface(&hit, &ray, &self.config.bvh, &self.config.scene);
            let wo = -ray.direction.into_inner();

            // Glass branch — identical to trace_path.
            if surf.transmission > 0.0 && rng.next_f32() < surf.transmission {
                if bounce == 0 {
                    // Camera ray hit glass first: the reservoir can't be used (the next
                    // opaque surface is in a different position each frame due to refraction
                    // jitter). Clear it so no stale sample leaks into the next frame.
                    *reservoir = Reservoir::default();
                }
                let is_exit = surf.geo_normal.dot(&wo) < 0.0;
                let thick = surf.thickness_factor > 0.0;
                let n = if is_exit { -surf.normal.into_inner() } else { surf.normal.into_inner() };
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
                    surf.albedo
                };
                let Some(gs) = sample_glass(&n, &wo, surf.roughness, surf.ior, &tint, is_exit, thick, rng)
                else {
                    break;
                };
                throughput = throughput.component_mul(&gs.weight);
                bsdf_pdf = 1e30;
                ray = Ray::new_with_minmax(
                    surf.position,
                    UnitVector3::new_normalize(gs.wi),
                    RAY_EPSILON,
                    f32::INFINITY,
                );
                if russian_roulette(&mut throughput, bounce, rng) {
                    break;
                }
                continue;
            }

            let brdf = surf.brdf();

            // Emission weight:
            //   bounce 0         → 1.0 (camera ray hit an emitter directly)
            //   bounce > 0, ReSTIR was used → 0.0 (NeeOnly-style: ReSTIR at bounce 0
            //     already estimated all direct illumination; suppress the BRDF-side hit)
            //   bounce > 0, no ReSTIR (glass at bounce 0) → MIS balance heuristic
            let emission = if surf.emissive != Color::zeros() {
                let emit_w = if bounce == 0 {
                    1.0
                } else if restir_used {
                    0.0
                } else {
                    let cos_l = surf.geo_normal.dot(&wo).max(0.0);
                    let p_l = self
                        .config
                        .lights
                        .get(hit.light as usize)
                        .map_or(0.0, |l| l.area_pdf(hit.hit.t * hit.hit.t, cos_l));
                    let denom = bsdf_pdf + p_l;
                    if denom > 0.0 { bsdf_pdf / denom } else { 1.0 }
                };
                surf.emissive * emit_w
            } else {
                Color::black()
            };

            // Direct lighting: ReSTIR only at the first opaque surface (bounce 0).
            // If bounce 0 was glass, this is never reached at bounce 0 (glass continues),
            // so bounce > 0 always falls through to compute_direct_lighting.
            let direct = if bounce == 0 {
                restir_used = true;
                self.compute_restir_direct(
                    &surf.position,
                    &surf.normal,
                    &brdf,
                    &wo,
                    reservoir,
                    shadow_rays,
                    rng,
                )
            } else {
                self.compute_direct_lighting(
                    &surf.position,
                    &surf.normal,
                    &brdf,
                    &wo,
                    shadow_rays,
                    rng,
                )
            };

            accumulated_radiance += throughput.component_mul(&(emission + direct));

            let Some(bs) = brdf.sample(&surf.normal, &wo, rng) else { break; };
            let cos_theta = surf.normal.dot(&bs.wi).max(0.0);
            if bs.pdf <= 0.0 || cos_theta <= 0.0 {
                break;
            }
            bsdf_pdf = bs.pdf;
            throughput = throughput.component_mul(&(bs.f * (cos_theta / bs.pdf)));
            ray = Ray::new(surf.position, UnitVector3::new_unchecked(bs.wi));

            if russian_roulette(&mut throughput, bounce, rng) {
                break;
            }
        }

        accumulated_radiance
    }

    /// ReSTIR DI variant of `sample_pixel`. Uses a per-pixel reservoir for
    /// temporal reuse of the best direct-lighting candidate across frames.
    pub fn sample_pixel_restir(
        &self,
        x: usize,
        y: usize,
        pixel: &mut Pixel,
        reservoir: &mut Reservoir,
    ) -> (u64, u64) {
        let mut rng = self.init_pixel_rng(x as u32, y as u32, pixel.samples);
        let ray = self.generate_ray(&mut rng, x as f32, y as f32);

        let mut rays = 0;
        let mut shadow_rays = 0;
        let sample_color =
            self.trace_path_restir(ray, &mut rng, &mut rays, &mut shadow_rays, reservoir);

        pixel.accumulated_radiance += sample_color;
        pixel.samples += 1;
        (rays, shadow_rays)
    }
}
