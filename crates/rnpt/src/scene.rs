use crate::math::Color;
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
    pub color: Color,
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
pub struct Material {
    pub albedo: Color,
    pub emissive: Color,
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
    pub lights: Vec<Light>,
    pub nodes: Vec<Node>,
    pub roots: Vec<u32>,
    pub cameras: Vec<Camera>,
}
