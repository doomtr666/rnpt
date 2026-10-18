mod bvh;
mod builder;
mod math;
pub mod stats;
mod refine;

pub use refine::RefineReport;

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
#[derive(Clone)]
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

/// A ray observed during rendering, with the distance of its answer: its closest hit, or the
/// end of its segment when it hit nothing (see `Scene::optimized_by_rays`).
#[derive(Clone, Copy, Debug)]
pub struct RaySample {
    pub org: [f32; 3],
    pub dir: [f32; 3],
    pub tmin: f32,
    pub tmax: f32,
    pub t_end: f32,
    /// An occlusion (shadow) query, rather than a closest-hit one.
    pub shadow: bool,
}

/// BVH-accelerated scene. Attach geometries, then call `commit` to build.
pub struct Scene {
    geometries: Vec<Geometry>,
    inner: Option<bvh::BvhInner>,
    sampling: RaySampling,
    regroup_chunks: bool,
}

/// Rays recorded while rendering, one in `every` (see `Scene::start_ray_sampling`).
#[derive(Default)]
struct RaySampling {
    every: std::sync::atomic::AtomicU32,
    samples: std::sync::Mutex<Vec<RaySample>>,
}

impl Scene {
    pub fn new() -> Self {
        Self { geometries: Vec::new(), inner: None, sampling: RaySampling::default(), regroup_chunks: true }
    }

    /// Whether `commit` re-shares triangles between neighbouring chunks to tighten their
    /// boxes (on by default; off to measure what it brings).
    pub fn set_regroup_chunks(&mut self, on: bool) {
        self.regroup_chunks = on;
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
        self.inner = Some(builder::build(&self.geometries, self.regroup_chunks));
    }

    /// Regroups triangles into chunks so that fewer of the sampled rays enter chunk boxes they
    /// find nothing in, then rebuilds the tree over the new chunks. `prior` weighs the
    /// surface-area term that stands for unobserved rays (1 = as much as the sample).
    pub fn refine_chunks_by_rays(&mut self, samples: &[RaySample], prior: f32) -> Option<RefineReport> {
        let inner = self.inner.as_ref()?;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let (groups, report) = refine::refine(&self.geometries, inner, samples, prior, threads);
        self.inner = Some(builder::build_grouped(&self.geometries, &groups));
        Some(report)
    }

    /// A new scene tuned to the rays a render traces, this one untouched (it can keep serving
    /// queries meanwhile): chunks regrouped by the sampled rays, then each node's children put
    /// in the order where the sampled shadow rays found occluders. Same geometry, same results.
    /// `threads`: how many threads the regrouping may use (few, if rendering goes on meanwhile).
    pub fn optimized_by_rays(&self, samples: &[RaySample], threads: usize) -> Option<(Scene, RefineReport)> {
        let inner = self.inner.as_ref()?;
        let (groups, report) = refine::refine(&self.geometries, inner, samples, 1.0, threads);
        let mut scene = Scene {
            geometries: self.geometries.clone(),
            inner: Some(builder::build_grouped(&self.geometries, &groups)),
            sampling: RaySampling::default(),
            regroup_chunks: self.regroup_chunks,
        };
        scene.learn_occluder_order(samples);
        Some((scene, report))
    }

    /// Replays the sampled shadow rays, counting through which child of each node they find
    /// occluders, then sorts every node's children by those counts (see
    /// `BvhInner::apply_occluder_order`).
    pub fn learn_occluder_order(&mut self, samples: &[RaySample]) {
        let Some(inner) = self.inner.as_mut() else { return };
        inner.learn_occluders = true;
        for s in samples.iter().filter(|s| s.shadow) {
            inner.any_hit(&to_internal(&Ray::new_with_minmax(s.org, s.dir, s.tmin, s.tmax)));
        }
        inner.learn_occluders = false;
        inner.apply_occluder_order();
    }

    /// Starts recording one query in `every`, with the distance of its answer, for
    /// `optimized_by_rays` (0 stops). Shadow queries sampled also run a closest-hit query to
    /// know where their answer is.
    pub fn start_ray_sampling(&self, every: u32) {
        self.sampling.every.store(every, std::sync::atomic::Ordering::Relaxed);
    }

    /// The rays recorded so far, and recording stops.
    pub fn take_ray_samples(&self) -> Vec<RaySample> {
        self.start_ray_sampling(0);
        std::mem::take(&mut *self.sampling.samples.lock().unwrap())
    }

    #[inline(always)]
    fn sample_due(&self) -> bool {
        // A counter per thread: a shared one would bounce between cores on every query.
        thread_local! {
            static QUERIES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let every = self.sampling.every.load(std::sync::atomic::Ordering::Relaxed);
        every != 0
            && QUERIES.with(|q| {
                let n = q.get().wrapping_add(1);
                q.set(n);
                n % every == 0
            })
    }

    fn record_sample(&self, ray: &Ray, t_end: f32, shadow: bool) {
        let s = RaySample { org: ray.org, dir: ray.dir, tmin: ray.tmin, tmax: ray.tmax, t_end, shadow };
        self.sampling.samples.lock().unwrap().push(s);
    }

    /// Parent of every node (`u32::MAX` for the root) and owner node of every chunk.
    pub fn parent_maps(&self) -> (Vec<u32>, Vec<u32>) {
        self.inner.as_ref().map_or((Vec::new(), Vec::new()), |b| b.parent_maps())
    }

    /// Inner nodes of the built tree (indices of `stats::Touch::Node`).
    pub fn node_count(&self) -> usize {
        self.inner.as_ref().map_or(0, |b| b.nodes.len())
    }


    /// Shadow rays learn, per node, which child tends to hold occluders (experiment).
    pub fn set_learn_occluders(&mut self, on: bool) {
        if let Some(b) = self.inner.as_mut() {
            b.learn_occluders = on;
        }
    }

    /// Makes the learned order physical (see `BvhInner::apply_occluder_order`).
    pub fn apply_occluder_order(&mut self) {
        if let Some(b) = self.inner.as_mut() {
            b.apply_occluder_order();
        }
    }

    /// Shape of the built tree and its SAH cost terms; `None` before `commit`.
    pub fn tree_stats(&self) -> Option<stats::TreeStats> {
        self.inner.as_ref().map(|b| b.tree_stats())
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl RayAccelerator for Scene {
    fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        let hit = self.inner.as_ref()?.closest_hit(&to_internal(ray));
        if self.sample_due() {
            self.record_sample(ray, hit.as_ref().map_or(ray.tmax, |h| h.t), false);
        }
        hit
    }

    fn any_hit(&self, ray: &Ray) -> bool {
        let Some(b) = self.inner.as_ref() else { return false };
        let occluded = b.any_hit(&to_internal(ray));
        if self.sample_due() {
            let t_end = if occluded { b.closest_hit(&to_internal(ray)).map_or(ray.tmax, |h| h.t) } else { ray.tmax };
            self.record_sample(ray, t_end, true);
        }
        occluded
    }
}

fn to_internal(ray: &Ray) -> math::InternalRay {
    let origin = Point3::new(ray.org[0], ray.org[1], ray.org[2]);
    let dir = Vector3::new(ray.dir[0], ray.dir[1], ray.dir[2]);
    let direction = UnitVector3::new_normalize(dir);
    math::InternalRay::new(origin, direction, ray.tmin, ray.tmax)
}
