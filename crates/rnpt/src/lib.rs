use nalgebra::{Matrix4, Point3, Transform3, UnitVector3, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
}

#[derive(Clone, Debug)]
pub struct Light {
    pub position: Point3<f32>,
    pub direction: Vector3<f32>,
    pub color: [f32; 3],
    pub intensity: f32,
    pub light_type: LightType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub fov: f32, // FOV in degrees
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            fov: 60.0,
        }
    }
}

impl Camera {
    pub fn compute_camera_to_world(&self) -> Transform3<f32> {
        let cam_z = (self.position - self.target).normalize();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let cam_x = up.cross(&cam_z).normalize();
        let cam_y = cam_z.cross(&cam_x);

        let m = Matrix4::new(
            cam_x.x,
            cam_y.x,
            cam_z.x,
            self.position.x,
            cam_x.y,
            cam_y.y,
            cam_z.y,
            self.position.y,
            cam_x.z,
            cam_y.z,
            cam_z.z,
            self.position.z,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        Transform3::from_matrix_unchecked(m)
    }
}

#[derive(Clone, Debug)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: UnitVector3<f32>,
    pub tmin: f32,
    pub tmax: f32,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: [f32; 3],
    pub emissive: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub triangles: Vec<Triangle>,
    pub material: u32,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub transform: Transform3<f32>,
    pub children: Vec<u32>,
    pub mesh: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub nodes: Vec<Node>,
    pub cameras: Vec<Camera>,
}

#[derive(Clone, Copy, Debug)]
pub struct Pixel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub samples: u32,
}

impl Default for Pixel {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            samples: 0,
        }
    }
}

// English code comments as per instructions
pub fn generate_ray_with_matrix(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    fov_degrees: f32,
    c2w: &Transform3<f32>,
) -> (Point3<f32>, Vector3<f32>) {
    // 1. Coordonnées d'écran normalisées (-1 à 1) au centre du pixel
    let ndc_x = (2.0 * (x + 0.5) / width) - 1.0;
    let ndc_y = 1.0 - (2.0 * (y + 0.5) / height);

    let aspect_ratio = width / height;
    let fov_rad = (fov_degrees * std::f32::consts::PI) / 180.0;
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
    let ray_dir = c2w.transform_vector(&local_dir).normalize();

    // L'origine du rayon est la position de la caméra (la partie translation de la matrice)
    let ray_origin = c2w.transform_point(&Point3::origin());

    (ray_origin, ray_dir)
}

/// A stateless function that computes a sample for a pixel (x, y)
/// and accumulates the result into the given mutable pixel reference.
///
/// This function is thread-safe as long as distinct threads operate
/// on distinct pixel references.
pub fn sample_pixel(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    camera: &Camera,
    pixel: &mut Pixel,
) {
    let u = x as f32 / width as f32;
    let v = y as f32 / height as f32;

    // Vary the red based on coordinates and camera position
    let r = 0.5 + 0.5 * (camera.position.x + u * 6.28).sin() * (camera.position.y + v * 6.28).cos();

    // Minor green/blue variations to react to target and fov
    let g = 0.1 * (camera.target.x + u).cos().abs();
    let b = 0.1 * (camera.fov.to_radians() + v).sin().abs();

    pixel.r += r;
    pixel.g += g;
    pixel.b += b;
    pixel.samples += 1;
}

pub struct PathTracerConfig {
    pub width: usize,
    pub height: usize,
    pub camera: Camera,
    pub scene: Scene,
}

pub struct PathTracer {}
