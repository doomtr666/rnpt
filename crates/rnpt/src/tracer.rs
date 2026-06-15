use std::sync::Arc;
use crate::{Bvh, Camera, Color, ColorExt, Pcg32, Ray, Scene};
use nalgebra::{Point3, Transform3, UnitVector3, Vector3};

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

#[derive(Clone)]
pub struct PathTracerConfig {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub scene: Arc<Scene>,
    pub bvh: Arc<Bvh>,
}

const MAX_BOUNCES: u32 = 5;
const RAY_EPSILON: f32 = 0.001;
const LIGHT_ATTENUATION_EPSILON: f32 = 1e-4;

pub struct PathTracer {
    config: PathTracerConfig,
    cam2world: Transform3<f32>,
}

impl PathTracer {
    pub fn new(config: PathTracerConfig) -> Self {
        let cam2world = config.camera.compute_camera_to_world();
        Self { config, cam2world }
    }

    fn trace_ray(&self, ray: &Ray) -> Option<crate::bvh::BvhHit> {
        self.config.bvh.intersect(ray)
    }

    fn interpolate_normal(
        &self,
        n0: &UnitVector3<f32>,
        n1: &UnitVector3<f32>,
        n2: &UnitVector3<f32>,
        u: f32,
        v: f32,
    ) -> UnitVector3<f32> {
        let w = 1.0 - u - v;
        let n = n0.scale(w) + n1.scale(u) + n2.scale(v);
        UnitVector3::new_normalize(n)
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

    // English code comments as per instructions
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

        // Ray direction in camera local space
        // La caméra regarde le long de son axe -Z local
        let local_dir = Vector3::new(
            ndc_x * aspect_ratio * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        );

        // Transform by CameraToWorld matrix
        // transform_vector n'applique pas la translation (ce qu'on veut pour une direction)
        let ray_dir = self.cam2world.transform_vector(&local_dir).normalize();

        // L'origine du rayon est la position de la caméra (la partie translation de la matrice)
        let ray_origin = self.cam2world.transform_point(&Point3::origin());

        Ray::new(ray_origin, UnitVector3::new_normalize(ray_dir))
    }
    /*
        /// Samples the hemisphere uniformly.
        /// Returns the world-space direction and the associated probability density function (PDF).
        fn sample_uniform_hemisphere(
            &self,
            normal: &UnitVector3<f32>,
            u1: f32, // Random number in [0, 1)
            u2: f32, // Random number in [0, 1)
        ) -> (Vector3<f32>, f32) {
            // Generate local coordinates on the hemisphere (Z-up axis)
            let phi = 2.0 * std::f32::consts::PI * u2;
            let local_z = u1; // cos(theta) = u1
            let sin_theta = (1.0 - local_z * local_z).max(0.0).sqrt();

            let local_x = sin_theta * phi.cos();
            let local_y = sin_theta * phi.sin();

            // Build a stable orthonormal basis (TBN) from the world normal (Frisvad's method)
            let n = normal.into_inner();
            let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
            let a = -1.0 / (sign + n.z);
            let b = n.x * n.y * a;

            let tangent = Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
            let bitangent = Vector3::new(b, sign + n.y * n.y * a, -n.y);

            // Transform the local sample direction into world space
            let world_dir = (tangent * local_x + bitangent * local_y + n * local_z).normalize();

            // The PDF for a uniform hemisphere distribution is a constant: 1 / (2 * PI)
            let pdf = 1.0 / (2.0 * std::f32::consts::PI);

            (world_dir, pdf)
        }
    */
    fn sample_cosine_hemisphere(
        &self,
        normal: &UnitVector3<f32>,
        u1: f32, // Random number in [0, 1)
        u2: f32, // Random number in [0, 1)
    ) -> (Vector3<f32>, f32) {
        // Local sampling on a disk, then project up to the hemisphere (Malley's method)
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;

        // In local space, the normal is along the Z axis
        let local_dir = Vector3::new(
            r * phi.cos(),
            r * phi.sin(),
            (1.0 - u1).sqrt(), // local_z is exactly cos(theta)
        );

        // Build a stable orthonormal basis (TBN) from the world normal (Frisvad's method)
        let n = normal.into_inner();
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;

        let tangent = Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let bitangent = Vector3::new(b, sign + n.y * n.y * a, -n.y);

        // Transform the local sample direction into world space
        let world_dir =
            (tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z).normalize();

        // The PDF for a cosine-weighted distribution is cos(theta) / PI
        let cos_theta = local_dir.z;
        let pdf = cos_theta / std::f32::consts::PI;

        (world_dir, pdf)
    }

    /// Computes the direct analytical lighting at a given surface intersection.
    /// This function loops through all lights and checks for occlusion via shadow rays.
    pub fn compute_direct_lighting(
        &self,
        hit_position: &Point3<f32>,
        normal: &UnitVector3<f32>,
        albedo: Color,
        shadow_rays: &mut u64,
        rng: &mut Pcg32,
    ) -> Color {
        let mut total_direct = Color::zeros();

        // Evaluate the Lambertian BRDF (constant for the entire surface loop)
        let brdf = albedo / std::f32::consts::PI;

        // Iterate through all analytical light sources in the scene
        for light in &self.config.scene.lights {
            let hit_to_light = light.position - hit_position;
            let distance_squared = hit_to_light.norm_squared();
            let distance = distance_squared.sqrt();
            let light_dir = UnitVector3::new_normalize(hit_to_light);

            // Compute the geometric cosine term (N . L)
            let cos_theta = normal.dot(&light_dir).max(0.0);

            // Early out if the light source is behind the surface
            if cos_theta <= 0.0 {
                continue;
            }

            // Setup the Shadow Ray to query visibility
            let shadow_ray = Ray::new_with_minmax(
                hit_position + normal.into_inner() * RAY_EPSILON,
                light_dir,
                0.0,
                distance - RAY_EPSILON,
            );

            // Ray visibility query (any-hit, no closest-hit work)
            *shadow_rays += 1;
            if !self.config.bvh.is_occluded(&shadow_ray) {
                let attenuation = 1.0 / (4.0 * std::f32::consts::PI * distance_squared).max(LIGHT_ATTENUATION_EPSILON);
                let incident_radiance = light.color * light.intensity * attenuation;
                total_direct += brdf.component_mul(&incident_radiance) * cos_theta;
            }
        }

        // Area lights: each emissive mesh is one light, sampled every call (NEE).
        let textures = &self.config.scene.textures;
        // Offset the shadow-ray origin off the surface, and measure distance FROM
        // that origin, so `tmax = dist - RAY_EPSILON` is a full eps margin
        // regardless of the surface/light angle. (Measuring from the un-offset
        // hit made the margin collapse to eps*(1-cos_s) → the emitter
        // self-occluded on surfaces facing it head-on, causing strange shadows.)
        let origin = hit_position + normal.into_inner() * RAY_EPSILON;
        for mesh in &self.config.bvh.emitters.meshes {
            let s = mesh.sample(rng, textures);

            let to_light = s.p - origin;
            let d2 = to_light.norm_squared();
            if d2 <= 0.0 {
                continue;
            }
            let dist = d2.sqrt();
            let wi = to_light / dist;

            let cos_s = normal.dot(&wi).max(0.0); // at the shaded surface
            let cos_l = s.normal.dot(&(-wi)).max(0.0); // at the emitter (front face)
            if cos_s <= 0.0 || cos_l <= 0.0 {
                continue;
            }

            let shadow_ray = Ray::new_with_minmax(
                origin,
                UnitVector3::new_unchecked(wi),
                0.0,
                dist - RAY_EPSILON,
            );

            *shadow_rays += 1;
            if !self.config.bvh.is_occluded(&shadow_ray) {
                // Area-measure estimator: cos_l / d2 is the solid-angle->area
                // Jacobian; dividing by pdf_area (= 1/total_area) is unbiased.
                let g = cos_s * cos_l / (d2 * s.pdf_area);
                total_direct += brdf.component_mul(&s.le) * g;
            }
        }

        total_direct
    }

    fn trace_path(
        &self,
        mut ray: Ray,
        rng: &mut Pcg32,
        rays: &mut u64,
        shadow_rays: &mut u64,
    ) -> Color {
        let mut accumulated_radiance = Color::black();
        let mut throughput = Color::white(); // Current path attenuation factor

        for bounce in 0..MAX_BOUNCES {
            // Ray intersection
            *rays += 1;
            let hit_opt = self.trace_ray(&ray);

            let Some(hit) = hit_opt else {
                let sky_radiance = Color::black(); //scene.sample_sky(&ray);
                accumulated_radiance += throughput.component_mul(&sky_radiance);
                break;
            };

            // Fetch surface data
            let mat = &self.config.scene.materials[hit.material as usize];

            let n0 = self.config.bvh.normals[hit.v0 as usize];
            let n1 = self.config.bvh.normals[hit.v1 as usize];
            let n2 = self.config.bvh.normals[hit.v2 as usize];

            let normal = self.interpolate_normal(&n0, &n1, &n2, hit.hit.u, hit.hit.v);

            let hit_position = ray.at(hit.hit.t);

            let has_textures = mat.albedo_texture.is_some() || mat.emissive_texture.is_some();
            let mut hit_uv = nalgebra::Vector2::zeros();
            
            let w = 1.0 - hit.hit.u - hit.hit.v;
            
            let c0 = self.config.bvh.colors[hit.v0 as usize];
            let c1 = self.config.bvh.colors[hit.v1 as usize];
            let c2 = self.config.bvh.colors[hit.v2 as usize];
            let vertex_color = c0 * w + c1 * hit.hit.u + c2 * hit.hit.v;

            if has_textures {
                let uv0 = self.config.bvh.uvs[hit.v0 as usize];
                let uv1 = self.config.bvh.uvs[hit.v1 as usize];
                let uv2 = self.config.bvh.uvs[hit.v2 as usize];
                hit_uv = uv0 * w + uv1 * hit.hit.u + uv2 * hit.hit.v;
            }

            let mut albedo = mat.albedo.component_mul(&vertex_color);
            if let Some(tex_idx) = mat.albedo_texture {
                if tex_idx < self.config.scene.textures.len() as u32 {
                    let tex = &self.config.scene.textures[tex_idx as usize];
                    albedo = albedo.component_mul(&tex.sample_bilinear(hit_uv));
                }
            }

            // Direct Lighting Calculation
            let mut local_radiance = Color::black();

            // Self-emission is only added when the emitter is seen directly (the
            // camera/primary ray). On GI bounces it is omitted: that contribution
            // is already covered by NEE (compute_direct_lighting) at the previous
            // vertex, so adding it again would double-count and explode variance.
            if bounce == 0 {
                let mut local_emissive = mat.emissive;
                if let Some(tex_idx) = mat.emissive_texture {
                    if tex_idx < self.config.scene.textures.len() as u32 {
                        let tex = &self.config.scene.textures[tex_idx as usize];
                        local_emissive = local_emissive.component_mul(&tex.sample_bilinear(hit_uv));
                    }
                }
                local_radiance += local_emissive;
            }

            // Next Event Estimation: analytic lights + emissive-mesh area lights.
            let direct_radiance =
                self.compute_direct_lighting(&hit_position, &normal, albedo, shadow_rays, rng);
            local_radiance += direct_radiance;

            // Add the local contribution of this vertex to the pixel, modulated by previous bounces
            let mut sample_radiance = throughput.component_mul(&local_radiance);
            
            // Clamp extreme fireflies
            let max_radiance = 10.0;
            sample_radiance.x = sample_radiance.x.min(max_radiance);
            sample_radiance.y = sample_radiance.y.min(max_radiance);
            sample_radiance.z = sample_radiance.z.min(max_radiance);

            accumulated_radiance += sample_radiance;

            // Bounce Setup

            // If no cache query, we sample the BRDF to continue the path
            let (next_dir, pdf) =
                self.sample_cosine_hemisphere(&normal, rng.next_f32(), rng.next_f32());

            if pdf <= 0.0 {
                break;
            }

            // Calculate the geometric cosine term for the scattering direction
            let cos_theta = normal.dot(&next_dir).max(0.0);
            if cos_theta <= 0.0 {
                break;
            }

            // Evaluate the Lambertian BRDF
            let brdf = albedo / std::f32::consts::PI;

            // Update the path throughput for the next bounce
            // New Throughput = Old Throughput * (BRDF * cos_theta / PDF)
            let brdf_weight = brdf * cos_theta / pdf;
            throughput = throughput.component_mul(&brdf_weight);

            // Setup the secondary ray for the next loop iteration
            // Offset origin along the normal to prevent self-intersection
            ray = Ray::new(
                hit_position + normal.into_inner() * RAY_EPSILON,
                UnitVector3::new_unchecked(next_dir),
            );

            // Russian Roulette to terminate paths that carry no energy
            if bounce > 3 {
                // Correctly extract the maximum component using f32::max
                let max_throughput = throughput.x.max(throughput.y).max(throughput.z);

                let q = (1.0 - max_throughput).max(0.05);
                if rng.next_f32() < q {
                    break;
                }
                throughput /= 1.0 - q;
            }
        }

        accumulated_radiance
    }

    /// A stateless function that computes a sample for a pixel (x, y)
    /// and accumulates the result into the given mutable pixel reference.
    ///
    /// This function is thread-safe as long as distinct threads operate
    /// on distinct pixel references.
    /// Traces one sample and returns `(rays, shadow_rays)`: closest-hit rays
    /// (primary + bounces) and any-hit shadow rays respectively.
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
