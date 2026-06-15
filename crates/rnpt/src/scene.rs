use crate::Light;
use crate::math::Color;
use nalgebra::{Matrix4, Point3, Transform3, UnitVector3, Vector2, Vector3};

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
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[u8; 4]>,
}

static DEGAMMA_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

fn get_degamma_lut() -> &'static [f32; 256] {
    DEGAMMA_LUT.get_or_init(|| {
        let mut lut = [0.0; 256];
        for i in 0..256 {
            lut[i] = (i as f32 / 255.0).powf(2.2);
        }
        lut
    })
}

impl Texture {
    pub fn sample_bilinear(&self, uv: Vector2<f32>) -> Color {
        if self.width == 0 || self.height == 0 {
            return Color::new(1.0, 0.0, 1.0); // Magenta error color
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

        let lut = get_degamma_lut();

        let get_color = |x: u32, y: u32| -> Color {
            let idx = (y * self.width + x) as usize;
            let pixel = self.pixels[idx];
            Color::new(
                lut[pixel[0] as usize],
                lut[pixel[1] as usize],
                lut[pixel[2] as usize],
            )
        };

        let c00 = get_color(x0, y0);
        let c10 = get_color(x1, y0);
        let c01 = get_color(x0, y1);
        let c11 = get_color(x1, y1);

        let top = c00 * (1.0 - tx) + c10 * tx;
        let bottom = c01 * (1.0 - tx) + c11 * tx;

        top * (1.0 - ty) + bottom * ty
    }
}

#[derive(Clone, Debug)]
pub struct Material {
    pub albedo: Color,
    pub emissive: Color,
    pub albedo_texture: Option<u32>,
    pub emissive_texture: Option<u32>,
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
