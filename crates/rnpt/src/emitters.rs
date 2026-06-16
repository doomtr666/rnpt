use crate::{Color, Pcg32, Texture};
use nalgebra::{Point3, UnitVector3, Vector2};

/// One emissive triangle in world space.
#[derive(Clone, Debug)]
pub struct EmitterTri {
    pub v0: Point3<f32>,
    pub v1: Point3<f32>,
    pub v2: Point3<f32>,
    pub uv0: Vector2<f32>,
    pub uv1: Vector2<f32>,
    pub uv2: Vector2<f32>,
    pub normal: UnitVector3<f32>, // geometric normal = normalize(cross(e1, e2))
}

/// One emissive mesh instance = one area light. Triangles are sampled
/// area-weighted within the mesh, so the area-measure pdf is constant
/// (`1 / total_area`).
#[derive(Clone, Debug)]
pub struct MeshEmitter {
    tris: Vec<EmitterTri>,
    cdf: Vec<f32>, // normalized cumulative area, len == tris.len(), last == 1.0
    total_area: f32,
    emissive: Color, // material is per-mesh, shared by all triangles
    emissive_texture: Option<u32>,
}

/// Result of sampling a point on an area light.
pub struct EmitterSample {
    pub p: Point3<f32>,
    pub normal: UnitVector3<f32>,
    pub le: Color,
    pub pdf_area: f32,
}

impl MeshEmitter {
    /// Build from non-degenerate world-space triangles. Returns `None` if the
    /// total area is zero (nothing to sample).
    pub fn new(tris: Vec<EmitterTri>, emissive: Color, emissive_texture: Option<u32>) -> Option<Self> {
        if tris.is_empty() {
            return None;
        }
        let mut cdf = Vec::with_capacity(tris.len());
        let mut acc = 0.0f32;
        for t in &tris {
            let e1 = t.v1 - t.v0;
            let e2 = t.v2 - t.v0;
            acc += 0.5 * e1.cross(&e2).norm();
            cdf.push(acc);
        }
        let total_area = acc;
        if total_area <= 0.0 {
            return None;
        }
        for c in &mut cdf {
            *c /= total_area;
        }
        Some(Self {
            tris,
            cdf,
            total_area,
            emissive,
            emissive_texture,
        })
    }

    #[inline]
    pub fn total_area(&self) -> f32 {
        self.total_area
    }

    /// Area-weighted sample of a point on the mesh. `pdf_area = 1 / total_area`
    /// regardless of the chosen triangle (pick ∝ area cancels the per-triangle
    /// uniform density) — this keeps the NEE estimator unbiased.
    pub fn sample(&self, rng: &mut Pcg32, textures: &[Texture]) -> EmitterSample {
        // Pick a triangle proportional to its area.
        let xi = rng.next_f32();
        let i = self
            .cdf
            .partition_point(|&c| c < xi)
            .min(self.tris.len() - 1);
        let tri = &self.tris[i];

        // Uniform barycentric point on the triangle (fold method).
        let mut u = rng.next_f32();
        let mut v = rng.next_f32();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;

        let p = Point3::from(tri.v0.coords * w + tri.v1.coords * u + tri.v2.coords * v);
        let uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;

        let mut le = self.emissive;
        if let Some(tex_idx) = self.emissive_texture {
            if (tex_idx as usize) < textures.len() {
                le = le.component_mul(&textures[tex_idx as usize].sample_bilinear(uv));
            }
        }

        EmitterSample {
            p,
            normal: tri.normal,
            le,
            pdf_area: 1.0 / self.total_area,
        }
    }
}

