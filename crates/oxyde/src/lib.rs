use nalgebra::Point3;

#[derive(Clone, Debug, PartialEq)]
pub struct CameraConfig {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub fov: f32, // FOV in degrees
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            fov: 60.0,
        }
    }
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
    camera: &CameraConfig,
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
