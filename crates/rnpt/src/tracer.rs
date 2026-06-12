use crate::{Camera, Color, ColorExt, Material, Pcg32, Ray, Scene, TriangleHit};
use nalgebra::{Point3, Transform3, UnitVector3, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Pixel {
    pub accumulated_radiance: Color,
    pub samples: u32,
    pub mean_luminance: f32, // Running mean of luminance
    pub m2_luminance: f32,   // Running sum of squares of differences
    pub converged: bool,     // Adaptive sampling early-out flag
}

impl Default for Pixel {
    fn default() -> Self {
        Self {
            accumulated_radiance: Color::black(),
            samples: 0,
            mean_luminance: 0.0,
            m2_luminance: 0.0,
            converged: false,
        }
    }
}

pub struct PathTracerConfig {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub scene: Scene,
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

    fn trace_ray(&self, ray: &Ray) -> Option<(TriangleHit, u32, u32)> {
        let mut closest_hit: Option<(TriangleHit, u32, u32)> = None; // (hit, mesh_index, triangle_index)
        let mut t_max = ray.tmax;

        let mut has_parent = vec![false; self.config.scene.nodes.len()];
        for node in &self.config.scene.nodes {
            for &child in &node.children {
                if (child as usize) < has_parent.len() {
                    has_parent[child as usize] = true;
                }
            }
        }

        let mut stack = Vec::new();
        for (idx, &has_p) in has_parent.iter().enumerate() {
            if !has_p {
                stack.push((idx, Transform3::identity()));
            }
        }

        while let Some((node_idx, parent_transform)) = stack.pop() {
            let node = &self.config.scene.nodes[node_idx];
            let world_transform = parent_transform * node.transform;

            for &mesh_idx in &node.meshes {
                let mesh = &self.config.scene.meshes[mesh_idx as usize];
                for (tri_idx, tri) in mesh.triangles.iter().enumerate() {
                    let v0 = world_transform.transform_point(&mesh.vertices[tri.v0 as usize]);
                    let v1 = world_transform.transform_point(&mesh.vertices[tri.v1 as usize]);
                    let v2 = world_transform.transform_point(&mesh.vertices[tri.v2 as usize]);

                    let mut test_ray = ray.clone();
                    test_ray.tmax = t_max;

                    if let Some(hit) = test_ray.intersect_triangle(&v0, &v1, &v2) {
                        t_max = hit.t;
                        closest_hit = Some((hit, mesh_idx as u32, tri_idx as u32));
                    }
                }
            }

            for &child_idx in &node.children {
                stack.push((child_idx as usize, world_transform));
            }
        }

        closest_hit
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
        // 1. Pack everything into a single 128-bit primitive
        // High 64 bits: Spatial coordinates (X, Y)
        // Low 64 bits: Temporal coordinate (Frame)
        let mut packed_seed: u128 =
            ((x as u128) << 96) | ((y as u128) << 64) | (frame_index as u128);

        // 2. Quick 128-bit bit-mixing (MurmurHash3 finalizer style for 128-bit blocks)
        packed_seed ^= packed_seed >> 33;
        packed_seed = packed_seed.wrapping_mul(0xff51afd7ed558ccd_u128);
        packed_seed ^= packed_seed >> 33;
        packed_seed = packed_seed.wrapping_mul(0xc4ceb9fe1a85ec53_u128);
        packed_seed ^= packed_seed >> 33;

        // 3. Pass the single 128-bit block to the RNG
        Pcg32::from_seed_128(packed_seed)
    }

    // English code comments as per instructions
    pub fn generate_ray(&self, rng: &mut Pcg32, x: f32, y: f32) -> Ray {
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // 1. Coordonnées d'écran normalisées (-1 à 1) au centre du pixel
        let jitter_x = rng.next_f32();
        let jitter_y = rng.next_f32();

        let ndc_x = (2.0 * (x + jitter_x) / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * (y + jitter_y) / height);

        let aspect_ratio = width / height;
        let fov_rad = (self.config.camera.fov * std::f32::consts::PI) / 180.0;
        let tan_half_fov = (fov_rad * 0.5).tan();

        // 2. Direction du rayon dans l'espace local de la caméra
        // La caméra regarde le long de son axe -Z local
        let local_dir = Vector3::new(
            ndc_x * aspect_ratio * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0,
        );

        // 3. Transformation par la matrice CameraToWorld
        // transform_vector n'applique pas la translation (ce qu'on veut pour une direction)
        let ray_dir = self.cam2world.transform_vector(&local_dir).normalize();

        // L'origine du rayon est la position de la caméra (la partie translation de la matrice)
        let ray_origin = self.cam2world.transform_point(&Point3::origin());

        Ray {
            origin: ray_origin,
            direction: UnitVector3::new_normalize(ray_dir),
            tmin: 0.0f32,
            tmax: f32::INFINITY,
        }
    }

    /// Samples the hemisphere uniformly.
    /// Returns the world-space direction and the associated probability density function (PDF).
    fn sample_uniform_hemisphere(
        &self,
        normal: &UnitVector3<f32>,
        u1: f32, // Random number in [0, 1)
        u2: f32, // Random number in [0, 1)
    ) -> (Vector3<f32>, f32) {
        // 1. Generate local coordinates on the hemisphere (Z-up axis)
        let phi = 2.0 * std::f32::consts::PI * u2;
        let local_z = u1; // cos(theta) = u1
        let sin_theta = (1.0 - local_z * local_z).max(0.0).sqrt();

        let local_x = sin_theta * phi.cos();
        let local_y = sin_theta * phi.sin();

        // 2. Build a stable orthonormal basis (TBN) from the world normal (Frisvad's method)
        let n = normal.into_inner();
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;

        let tangent = Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let bitangent = Vector3::new(b, sign + n.y * n.y * a, -n.y);

        // 3. Transform the local sample direction into world space
        let world_dir = (tangent * local_x + bitangent * local_y + n * local_z).normalize();

        // 4. The PDF for a uniform hemisphere distribution is a constant: 1 / (2 * PI)
        let pdf = 1.0 / (2.0 * std::f32::consts::PI);

        (world_dir, pdf)
    }

    fn sample_cosine_hemisphere(
        &self,
        normal: &UnitVector3<f32>,
        u1: f32, // Random number in [0, 1)
        u2: f32, // Random number in [0, 1)
    ) -> (Vector3<f32>, f32) {
        // 1. Local sampling on a disk, then project up to the hemisphere (Malley's method)
        let r = u1.sqrt();
        let phi = 2.0 * std::f32::consts::PI * u2;

        // In local space, the normal is along the Z axis
        let local_dir = Vector3::new(
            r * phi.cos(),
            r * phi.sin(),
            (1.0 - u1).sqrt(), // local_z is exactly cos(theta)
        );

        // 2. Build a stable orthonormal basis (TBN) from the world normal (Frisvad's method)
        let n = normal.into_inner();
        let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + n.z);
        let b = n.x * n.y * a;

        let tangent = Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        let bitangent = Vector3::new(b, sign + n.y * n.y * a, -n.y);

        // 3. Transform the local sample direction into world space
        let world_dir =
            (tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z).normalize();

        // 4. The PDF for a cosine-weighted distribution is cos(theta) / PI
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
        mat: &Material,
    ) -> Color {
        let mut total_direct = Color::zeros();

        // 1. Evaluate the Lambertian BRDF (constant for the entire surface loop)
        let brdf = mat.albedo / std::f32::consts::PI;

        // 2. Iterate through all analytical light sources in the scene
        for light in &self.config.scene.lights {
            let hit_to_light = light.position - hit_position;
            let distance_squared = hit_to_light.norm_squared();
            let distance = distance_squared.sqrt();
            let light_dir = UnitVector3::new_normalize(hit_to_light);

            // 3. Compute the geometric cosine term (N . L)
            let cos_theta = normal.dot(&light_dir).max(0.0);

            // Early out if the light source is behind the surface
            if cos_theta <= 0.0 {
                continue;
            }

            // 4. Setup the Shadow Ray to query visibility
            // Self-intersection prevention: offset origin along the normal by 0.001
            // Overshoot prevention: reduce tmax by 0.001 to avoid hitting the light geometry itself
            let shadow_ray = Ray {
                origin: hit_position + normal.into_inner() * 0.001,
                direction: light_dir,
                tmin: 0.0,
                tmax: distance - 0.001,
            };

            // 5. Ray visibility query
            if self.trace_ray(&shadow_ray).is_none() {
                // 6. Calculate physical distance attenuation (Inverse Square Law)
                // 1e-4 safeguard prevents division by zero if the ray is on top of the light
                //let attenuation = 1.0 / distance_squared.max(1e-4);

                // Distance attenuation matching Blender Cycles (Power distributed over a sphere: 4 * PI * d^2)
                // 1e-4 safeguard prevents division by zero if the ray is on top of the light
                let attenuation = 1.0 / (4.0 * std::f32::consts::PI * distance_squared).max(1e-4);

                // 7. Calculate the incident radiance reaching this specific point
                let incident_radiance = light.color * light.intensity * attenuation;

                // 8. Accumulate the reflected radiance contribution
                // Outgoing Radiance = BRDF * Incident Radiance * cos_theta
                total_direct += brdf.component_mul(&incident_radiance) * cos_theta;
            }
        }

        total_direct
    }

    fn trace_path(&self, mut ray: Ray, rng: &mut Pcg32) -> Color {
        let mut accumulated_radiance = Color::black();
        let mut throughput = Color::white(); // Current path attenuation factor

        for bounce in 0..5 {
            // 1. Scene Intersection Query
            let hit_result = self.trace_ray(&ray);

            // If the ray escapes the scene, sample the sky/environment
            let Some(hit) = hit_result else {
                let sky_radiance = Color::black(); //scene.sample_sky(&ray);
                accumulated_radiance += throughput.component_mul(&sky_radiance);
                break;
            };

            // Fetch surface data
            let mesh = &self.config.scene.meshes[hit.1 as usize];
            let mat = &self.config.scene.materials[mesh.material as usize];
            let tri = &mesh.triangles[hit.2 as usize];

            let n0 = mesh.normals[tri.v0 as usize];
            let n1 = mesh.normals[tri.v1 as usize];
            let n2 = mesh.normals[tri.v2 as usize];

            let normal = self.interpolate_normal(&n0, &n1, &n2, hit.0.u, hit.0.v);

            let hit_position = ray.at(hit.0.t);

            // ---------------------------------------------------------
            // STEP A: LOCAL RADIANCE (Direct view & Analytic Lights)
            // ---------------------------------------------------------
            let mut local_radiance = Color::black();

            // 1. Self-emission (Only counted directly if we "fall" on it)
            //if mat.is_emissive() {
            //    local_radiance += mat.evaluate_emissive(hit.uv);
            //}
            local_radiance += mat.emissive;

            // 2. Next Event Estimation (Analytic lights loop)
            // This calculates the direct lighting at this specific vertex
            local_radiance += self.compute_direct_lighting(&hit_position, &normal, mat);

            // Add the local contribution of this vertex to the pixel, modulated by previous bounces
            accumulated_radiance += throughput.component_mul(&local_radiance);

            // ---------------------------------------------------------
            // STEP B: INDIRECT RADIANCE PREPARATION (Secondary Ray)
            // ---------------------------------------------------------

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
            let brdf = mat.albedo / std::f32::consts::PI;

            // Update the path throughput for the next bounce
            // New Throughput = Old Throughput * (BRDF * cos_theta / PDF)
            let brdf_weight = brdf * cos_theta / pdf;
            throughput = throughput.component_mul(&brdf_weight);

            // Setup the secondary ray for the next loop iteration
            // Offset origin along the normal to prevent self-intersection
            ray = Ray {
                origin: hit_position + normal.into_inner() * 0.001,
                direction: UnitVector3::new_unchecked(next_dir),
                tmin: 0.0,
                tmax: f32::INFINITY,
            };

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
    pub fn sample_pixel(&self, x: usize, y: usize, pixel: &mut Pixel) {
        if pixel.converged {
            return;
        }

        let mut rng = self.init_pixel_rng(x as u32, y as u32, pixel.samples);
        let ray = self.generate_ray(&mut rng, x as f32, y as f32);

        let sample_color = self.trace_path(ray, &mut rng);

        pixel.accumulated_radiance += sample_color;
        pixel.samples += 1;

        // 1. Compute luminance of the current sample (ITU-R BT.709 weights)
        let luminance = sample_color.x * 0.2126 + sample_color.y * 0.7152 + sample_color.z * 0.0722;

        // 2. Online update of variance using Welford's algorithm
        let n = pixel.samples as f32;
        let delta = luminance - pixel.mean_luminance;
        pixel.mean_luminance += delta / n;
        let delta2 = luminance - pixel.mean_luminance;
        pixel.m2_luminance += delta * delta2;

        // 3. Statistical test for convergence (only after a baseline of samples)
        if pixel.samples >= 64 {
            let variance = pixel.m2_luminance / (n - 1.0);
            let std_dev = variance.max(0.0).sqrt();

            // Standard error of the mean using a 95% confidence interval (z = 1.96)
            let error = 1.96 * std_dev / n.sqrt();

            // Target threshold: 2% of the running mean + a small absolute epsilon
            // The epsilon prevents the test from stalling in pure pitch-black shadow zones
            let threshold = 0.02 * pixel.mean_luminance + 0.001;

            if error < threshold {
                pixel.converged = true;
            }
        }
    }
}
