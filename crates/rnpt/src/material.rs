use crate::{Brdf, Bvh, BvhHit, Color, Ray, Scene};
use nalgebra::{Point3, UnitVector3, Vector2};

/// Geometry + material evaluated at a ray hit. This is the single place where
/// surface shading inputs are produced — normal maps, metallic/roughness, etc.
/// will plug in here, and it builds the BRDF for the integrator.
pub struct SurfaceInteraction {
    pub position: Point3<f32>,
    pub normal: UnitVector3<f32>,     // shading normal (interpolated vertex normals)
    pub geo_normal: UnitVector3<f32>, // geometric normal (triangle winding)
    pub albedo: Color,
    pub emissive: Color,
}

impl SurfaceInteraction {
    /// Build the scattering model for this surface. Lambertian for now; will
    /// dispatch on metallic/roughness once those are added.
    pub fn brdf(&self) -> Brdf {
        Brdf::Lambertian { albedo: self.albedo }
    }
}

/// Interpolate geometry (normal, uv, vertex color) and evaluate the material
/// (albedo + emissive, with textures) at a hit.
pub fn evaluate_surface(hit: &BvhHit, ray: &Ray, bvh: &Bvh, scene: &Scene) -> SurfaceInteraction {
    let mat = &scene.materials[hit.material as usize];
    let (u, v) = (hit.hit.u, hit.hit.v);
    let w = 1.0 - u - v;

    // Shading normal (barycentric-interpolated vertex normals).
    let n0 = bvh.normals[hit.v0 as usize];
    let n1 = bvh.normals[hit.v1 as usize];
    let n2 = bvh.normals[hit.v2 as usize];
    let normal = UnitVector3::new_normalize(n0.scale(w) + n1.scale(u) + n2.scale(v));

    // Geometric normal from the triangle winding (raw, to match the emitter
    // normal used by NEE — needed for the MIS cos_l on BRDF hits).
    let p0 = bvh.vertices[hit.v0 as usize];
    let p1 = bvh.vertices[hit.v1 as usize];
    let p2 = bvh.vertices[hit.v2 as usize];
    let geo_normal = UnitVector3::new_normalize((p1 - p0).cross(&(p2 - p0)));

    let position = ray.at(hit.hit.t);

    // Per-vertex color.
    let c0 = bvh.colors[hit.v0 as usize];
    let c1 = bvh.colors[hit.v1 as usize];
    let c2 = bvh.colors[hit.v2 as usize];
    let vertex_color = c0 * w + c1 * u + c2 * v;

    // UVs (only needed if a texture is bound).
    let has_textures = mat.albedo_texture.is_some() || mat.emissive_texture.is_some();
    let uv = if has_textures {
        let uv0 = bvh.uvs[hit.v0 as usize];
        let uv1 = bvh.uvs[hit.v1 as usize];
        let uv2 = bvh.uvs[hit.v2 as usize];
        uv0 * w + uv1 * u + uv2 * v
    } else {
        Vector2::zeros()
    };

    // Modulate a base color by its (optional) texture.
    let modulate = |tex: Option<u32>, base: Color| -> Color {
        match tex {
            Some(i) if (i as usize) < scene.textures.len() => {
                base.component_mul(&scene.textures[i as usize].sample_bilinear(uv))
            }
            _ => base,
        }
    };

    SurfaceInteraction {
        position,
        normal,
        geo_normal,
        albedo: modulate(mat.albedo_texture, mat.albedo.component_mul(&vertex_color)),
        emissive: modulate(mat.emissive_texture, mat.emissive),
    }
}
