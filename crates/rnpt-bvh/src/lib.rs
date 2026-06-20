mod bvh;
mod builder;
mod math;

use nalgebra::{Point3, UnitVector3, Vector3};

/// Ray for BVH queries. `dir` must be normalized.
pub struct Ray {
    pub org:  [f32; 3],
    pub dir:  [f32; 3],
    pub tmin: f32,
    pub tmax: f32,
}

impl Ray {
    pub fn new(org: [f32; 3], dir: [f32; 3]) -> Self {
        Self { org, dir, tmin: 0.0, tmax: f32::INFINITY }
    }

    pub fn new_with_minmax(org: [f32; 3], dir: [f32; 3], tmin: f32, tmax: f32) -> Self {
        Self { org, dir, tmin, tmax }
    }
}

/// Result of a closest-hit query.
pub struct Hit {
    pub t:       f32,
    pub u:       f32,
    pub v:       f32,
    /// Index of the hit triangle within the `Geometry` it belongs to.
    pub prim_id: u32,
    /// Index of the `Geometry` within the `Scene`.
    pub geom_id: u32,
}

pub trait RayAccelerator {
    fn closest_hit(&self, ray: &Ray) -> Option<Hit>;
    fn any_hit(&self, ray: &Ray) -> bool;
}

/// Triangle mesh geometry. Build with `Geometry::triangle_mesh`.
pub struct Geometry {
    pub(crate) verts: Vec<[f32; 3]>,
    pub(crate) tris:  Vec<[u32; 3]>,
    /// Per-triangle double-sided flag. Empty means all triangles are single-sided.
    /// When set, the triangle accepts hits from both front and back.
    pub(crate) double_sided: Vec<bool>,
}

impl Geometry {
    pub fn triangle_mesh(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Self {
        Self { verts: verts.to_vec(), tris: tris.to_vec(), double_sided: Vec::new() }
    }

    pub fn with_double_sided(mut self, ds: Vec<bool>) -> Self {
        self.double_sided = ds;
        self
    }
}

/// BVH-accelerated scene. Attach geometries, then call `commit` to build.
pub struct Scene {
    geometries: Vec<Geometry>,
    inner: Option<bvh::BvhInner>,
}

impl Scene {
    pub fn new() -> Self {
        Self { geometries: Vec::new(), inner: None }
    }

    /// Attach a geometry and return its `geom_id`.
    pub fn attach_geometry(&mut self, geom: Geometry) -> u32 {
        let id = self.geometries.len() as u32;
        self.geometries.push(geom);
        self.inner = None;
        id
    }

    /// Build (or rebuild) the BVH from all attached geometries.
    pub fn commit(&mut self) {
        self.inner = Some(builder::build(&self.geometries));
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl RayAccelerator for Scene {
    fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        self.inner.as_ref()?.closest_hit(&to_internal(ray))
    }

    fn any_hit(&self, ray: &Ray) -> bool {
        self.inner.as_ref().map_or(false, |b| b.any_hit(&to_internal(ray)))
    }
}

fn to_internal(ray: &Ray) -> math::InternalRay {
    let origin = Point3::new(ray.org[0], ray.org[1], ray.org[2]);
    let dir = Vector3::new(ray.dir[0], ray.dir[1], ray.dir[2]);
    let direction = UnitVector3::new_normalize(dir);
    math::InternalRay::new(origin, direction, ray.tmin, ray.tmax)
}
