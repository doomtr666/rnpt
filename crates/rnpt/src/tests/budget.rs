use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    build_lights, BvhBuilder, Camera, Color, Material, Mesh, Node, ParallelTracer, PathTracerConfig, Pixel,
    RenderBudget, SamplingStrategy, Scene, Triangle,
};
use nalgebra::{Point3, Transform3, UnitVector3, Vector2, Vector3};

const SIZE: usize = 64;
// 64 x 64 x 512 = 2M rays: several NIRC training steps, and enough for the
// workers to get ahead of the trainer and wait.
const BUDGET: u64 = 512;

/// A floor at z = 0 under an emissive ceiling of the same size: paths bounce
/// between the two, so NIRC gets training samples. The camera sees the floor
/// through the back of the (single-sided) ceiling.
fn lit_floor() -> Scene {
    let quad = |z: f32, h: f32, material: u32, up: bool| {
        let verts = [[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]];
        let tris = if up { [[0, 1, 2], [0, 2, 3]] } else { [[0, 2, 1], [0, 3, 2]] };
        let n = UnitVector3::new_normalize(Vector3::new(0.0, 0.0, if up { 1.0 } else { -1.0 }));
        Mesh {
            vertices: verts.iter().map(|v| Point3::new(v[0], v[1], v[2])).collect(),
            normals: vec![n; 4],
            uvs: vec![Vector2::zeros(); 4],
            colors: vec![Vector3::new(1.0, 1.0, 1.0); 4],
            tangents: vec![],
            triangles: tris.iter().map(|t| Triangle { v0: t[0], v1: t[1], v2: t[2] }).collect(),
            material,
        }
    };
    let light = Material { emissive: Color::new(5.0, 5.0, 5.0), ..Default::default() };
    Scene {
        meshes: vec![quad(0.0, 1.0, 0, true), quad(2.0, 1.0, 1, false)],
        materials: vec![Material::default(), light],
        textures: vec![],
        lights: vec![],
        nodes: vec![Node { transform: Transform3::identity(), children: vec![], meshes: vec![0, 1] }],
        roots: vec![0],
        cameras: vec![],
    }
}

/// Render until every pixel spent its budget (driving NIRC training like the
/// GUI does), and return the pixels.
fn render_with_budget(strategy: SamplingStrategy) -> (Vec<Pixel>, usize) {
    let scene = lit_floor();
    let (bvh, emitters) = BvhBuilder::new(&scene).build();
    let config = PathTracerConfig {
        width: SIZE,
        height: SIZE,
        camera: Camera::default(),
        scene: Arc::new(scene),
        bvh: Arc::new(bvh),
        lights: Arc::new(build_lights(&[], emitters)),
        env: None,
        strategy,
        nirc_network: None,
        budget: RenderBudget { rays_per_pixel: BUDGET, samples_per_pixel: 0 },
        indirect_only: false,
    };

    let tracer = ParallelTracer::new(config.clone());
    let mut pixels = vec![Pixel::default(); SIZE * SIZE];
    let mut trained = 0;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        trained += tracer.train_nirc().is_some() as usize;
        tracer.fetch_pixels(&mut pixels);
        if pixels.iter().all(|p| config.budget.is_exhausted(p)) {
            return (pixels, trained);
        }
        assert!(Instant::now() < deadline, "{strategy:?}: budget never reached (stalled workers?)");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn check(strategy: SamplingStrategy) {
    let (pixels, trained) = render_with_budget(strategy);
    if strategy == SamplingStrategy::Nirc {
        assert!(trained > 0, "NIRC never trained");
    }
    for p in &pixels {
        assert!(p.rays >= BUDGET);
        // The last path (plus a training path) may cross the budget, not much more.
        assert!(p.rays < BUDGET + 256, "{strategy:?}: pixel overshot its budget: {} rays", p.rays);
        assert!(p.accumulated_radiance.x.is_finite());
    }
    // Workers are idle once done: the pixels no longer change.
    std::thread::sleep(Duration::from_millis(50));
}

#[test]
fn mis_stops_at_ray_budget() {
    check(SamplingStrategy::Mis);
}

#[test]
fn nirc_stops_at_ray_budget() {
    check(SamplingStrategy::Nirc);
}
