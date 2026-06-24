use crate::{AliasTable, Color, Pcg32, Texture};
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

impl EmitterTri {
    #[inline]
    pub fn area(&self) -> f32 {
        0.5 * (self.v1 - self.v0).cross(&(self.v2 - self.v0)).norm()
    }
}

/// One emissive mesh instance = one area light. Triangles are sampled
/// area-weighted within the mesh, so the area-measure pdf is constant
/// (`1 / total_area`).
#[derive(Clone, Debug)]
pub struct MeshEmitter {
    tris: Vec<EmitterTri>,
    alias: AliasTable, // O(1) area-weighted triangle selection (replaces CDF binary search)
    total_area: f32,
    emissive: Color, // material is per-mesh, shared by all triangles
    emissive_texture: Option<u32>,
}

/// Geometric sample on an area light — position and normal only, no texture evaluation.
/// Used for deferred emissive lookup: check backface/distance before paying for bilinear fetch.
pub struct EmitterGeomSample {
    pub p: Point3<f32>,
    pub normal: UnitVector3<f32>,
    pub tri_idx: usize,
    pub u: f32,
    pub v: f32,
    pub pdf_area: f32,
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
        let mut areas = Vec::with_capacity(tris.len());
        let mut total_area = 0.0f32;
        for t in &tris {
            let e1 = t.v1 - t.v0;
            let e2 = t.v2 - t.v0;
            let area = 0.5 * e1.cross(&e2).norm();
            areas.push(area);
            total_area += area;
        }
        if total_area <= 0.0 {
            return None;
        }
        Some(Self {
            tris,
            alias: AliasTable::new(&areas),
            total_area,
            emissive,
            emissive_texture,
        })
    }

    #[inline]
    pub fn total_area(&self) -> f32 {
        self.total_area
    }

    #[inline]
    pub fn tris(&self) -> &[EmitterTri] {
        &self.tris
    }

    #[inline]
    pub fn emissive(&self) -> Color {
        self.emissive
    }

    #[inline]
    pub fn emissive_texture(&self) -> Option<u32> {
        self.emissive_texture
    }

    /// Geometry-only sample: O(1) triangle selection + barycentric, no texture lookup.
    /// Call `le_at()` afterwards — only if the sample passes backface/distance tests.
    pub fn sample_geom(&self, rng: &mut Pcg32) -> EmitterGeomSample {
        let tri_idx = self.alias.sample(rng.next_f32());
        let tri = &self.tris[tri_idx];

        let mut u = rng.next_f32();
        let mut v = rng.next_f32();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        let p = Point3::from(tri.v0.coords * w + tri.v1.coords * u + tri.v2.coords * v);

        EmitterGeomSample { p, normal: tri.normal, tri_idx, u, v, pdf_area: 1.0 / self.total_area }
    }

    /// Evaluate emitted radiance at a previously sampled geometry point.
    pub fn le_at(&self, s: &EmitterGeomSample, textures: &[Texture]) -> Color {
        let tri = &self.tris[s.tri_idx];
        let w = 1.0 - s.u - s.v;
        let mut le = self.emissive;
        if let Some(tex_idx) = self.emissive_texture {
            if (tex_idx as usize) < textures.len() {
                let uv = tri.uv0 * w + tri.uv1 * s.u + tri.uv2 * s.v;
                le = le.component_mul(&textures[tex_idx as usize].sample_bilinear(uv));
            }
        }
        le
    }

    /// Area-weighted sample of a point on the mesh. `pdf_area = 1 / total_area`
    /// regardless of the chosen triangle (pick ∝ area cancels the per-triangle
    /// uniform density) — this keeps the NEE estimator unbiased.
    pub fn sample(&self, rng: &mut Pcg32, textures: &[Texture]) -> EmitterSample {
        let i = self.alias.sample(rng.next_f32());
        let tri = &self.tris[i];

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
