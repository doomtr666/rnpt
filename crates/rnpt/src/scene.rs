use crate::Light;
use crate::math::Color;
use nalgebra::{Matrix4, Point3, Transform3, UnitVector3, Vector2, Vector3, Vector4};

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

/// All textures are stored as linear-light RGB f32, converted from sRGB at load
/// time (see `asset_importer`). `sample_bilinear` is pure interpolation — no
/// color-space work at sample time. Linear maps (normals, roughness, …) are
/// loaded without sRGB conversion and plug into the same sampler unchanged.
#[derive(Clone, Debug)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Color>, // linear f32, converted once at load time
}

impl Texture {
    pub fn sample_bilinear(&self, uv: Vector2<f32>) -> Color {
        if self.width == 0 || self.height == 0 {
            return Color::new(1.0, 0.0, 1.0);
        }

        let u = uv.x - uv.x.floor();
        let v = uv.y - uv.y.floor();

        let x = u * (self.width as f32 - 1.0);
        let y = v * (self.height as f32 - 1.0);

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let tx = x - x0 as f32;
        let ty = y - y0 as f32;

        let get = |x: u32, y: u32| self.pixels[(y * self.width + x) as usize];

        let c00 = get(x0, y0);
        let c10 = get(x1, y0);
        let c01 = get(x0, y1);
        let c11 = get(x1, y1);

        (c00 * (1.0 - tx) + c10 * tx) * (1.0 - ty) + (c01 * (1.0 - tx) + c11 * tx) * ty
    }
}

#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: Color,
    pub emissive: Color,
    pub albedo_texture: Option<u32>,
    pub emissive_texture: Option<u32>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<u32>, // G = roughness, B = metallic (glTF spec)
    pub normal_texture: Option<u32>,
    pub normal_scale: f32,
    pub alpha_cutoff: Option<f32>, // None = OPAQUE/BLEND, Some(t) = MASK
    pub double_sided: bool,
    /// KHR_materials_transmission: 0 = opaque, 1 = fully transparent.
    pub transmission: f32,
    /// KHR_materials_ior: index of refraction (default 1.5 for glass).
    pub ior: f32,
    /// KHR_materials_volume: 0 = thin surface (no refraction), >0 = volume boundary
    /// with Snell refraction and Beer-Lambert absorption.
    pub thickness_factor: f32,
    /// KHR_materials_volume: distance (world units) at which attenuation_color is reached.
    pub attenuation_distance: f32,
    /// KHR_materials_volume: color of transmitted light at attenuation_distance.
    pub attenuation_color: Color,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: Color::new(1.0, 1.0, 1.0),
            emissive: Color::zeros(),
            albedo_texture: None,
            emissive_texture: None,
            metallic: 0.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            alpha_cutoff: None,
            double_sided: false,
            transmission: 0.0,
            ior: 1.5,
            thickness_factor: 0.0,
            attenuation_distance: f32::INFINITY,
            attenuation_color: Color::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<UnitVector3<f32>>,
    pub uvs: Vec<Vector2<f32>>,
    pub colors: Vec<Color>,
    /// Optional Mikktspace tangents (vec4: xyz = tangent direction, w = bitangent sign ±1).
    /// Empty when the mesh has no TANGENT attribute; TBN falls back to UV-delta computation.
    pub tangents: Vec<Vector4<f32>>,
    pub triangles: Vec<Triangle>,
    pub material: u32,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub transform: Transform3<f32>,
    pub children: Vec<u32>,
    pub meshes: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub lights: Vec<Light>,
    pub nodes: Vec<Node>,
    pub roots: Vec<u32>,
    pub cameras: Vec<Camera>,
}
