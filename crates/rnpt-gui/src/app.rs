use eframe::egui;
use std::time::{Duration, Instant};
use crate::tonemap::{TonemapOperator, tonemap_and_convert};
use crate::panels;

pub struct RnptGuiApp {
    pub camera: rnpt::Camera,
    pub resolution: [usize; 2],
    pub exposure: f32,
    pub tonemapper: TonemapOperator,

    pub tracer: Option<rnpt::ParallelTracer>,
    pub last_fps_time: Instant,
    pub rays_since_last_fps: u64,
    pub real_rays_since_last_fps: u64,
    pub shadow_rays_since_last_fps: u64,

    pub local_pixels: Vec<rnpt::Pixel>,
    pub local_width: usize,
    pub local_height: usize,
    pub local_rays_per_sec: f64,
    pub local_real_rays_per_sec: f64,
    pub local_shadow_rays_per_sec: f64,

    pub texture_handle: Option<egui::TextureHandle>,
    pub last_exposure: f32,
    pub last_tonemapper: TonemapOperator,

    pub asset_files: Vec<std::path::PathBuf>,
    pub selected_asset_index: usize,
    pub current_scene: Option<std::sync::Arc<rnpt::Scene>>,
    pub current_bvh: Option<std::sync::Arc<rnpt::Bvh>>,
    pub current_lights: Option<std::sync::Arc<Vec<rnpt::Light>>>,

    pub auto_fit: bool,
    pub resize_timeout: Option<Instant>,

    pub strategy: rnpt::SamplingStrategy,
    pub selected_camera_index: usize,

    pub hdr_files: Vec<std::path::PathBuf>,
    pub selected_env: usize,
    pub env: Option<std::sync::Arc<rnpt::EnvLight>>,
    pub env_raw: Option<(Vec<rnpt::Color>, usize, usize)>,
    pub env_intensity: f32,
    pub env_rotation: f32,

    pub scene_load_stats: Option<crate::asset_importer::SceneLoadStats>,
    pub bvh_build_ms: u64,
    pub bvh_triangle_count: usize,

    pub show_debug: bool,

    pub frame_count: u64,
    pub last_loss: f32,
    pub loss_ema: f32,
    pub rel_error: f32,

    pub probe_texture: Option<egui::TextureHandle>,
}

// ── Construction helpers ──────────────────────────────────────────────────────

pub(crate) fn with_env(
    lights: &std::sync::Arc<Vec<rnpt::Light>>,
    env: &Option<std::sync::Arc<rnpt::EnvLight>>,
) -> (std::sync::Arc<Vec<rnpt::Light>>, Option<usize>) {
    match env {
        Some(e) => {
            let mut v = (**lights).clone();
            v.push(rnpt::Light::Environment(e.clone()));
            let idx = v.len() - 1;
            (std::sync::Arc::new(v), Some(idx))
        }
        None => (lights.clone(), None),
    }
}

fn empty_scene() -> rnpt::Scene {
    rnpt::Scene {
        meshes: Vec::new(),
        materials: Vec::new(),
        textures: Vec::new(),
        lights: Vec::new(),
        nodes: Vec::new(),
        roots: Vec::new(),
        cameras: vec![rnpt::Camera::default()],
    }
}

pub(crate) fn build_bvh_and_lights(
    scene: &rnpt::Scene,
) -> (
    std::sync::Arc<rnpt::Bvh>,
    std::sync::Arc<Vec<rnpt::Light>>,
    u64,
    usize,
) {
    let t0 = Instant::now();
    let (bvh, emitters) = rnpt::BvhBuilder::new(scene).build();
    let build_ms = t0.elapsed().as_millis() as u64;
    let tri_count = bvh.triangle_meta.len();
    let lights = rnpt::build_lights(&scene.lights, emitters);
    (
        std::sync::Arc::new(bvh),
        std::sync::Arc::new(lights),
        build_ms,
        tri_count,
    )
}

// ── RnptGuiApp impl ───────────────────────────────────────────────────────────

impl RnptGuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let width = 800;
        let height = 600;
        let mut camera = rnpt::Camera::default();

        let asset_files = crate::asset_importer::list_assets("assets");
        let selected_asset_index = asset_files
            .iter()
            .position(|p| p.file_name().map_or(false, |n| n == "cornell.glb"))
            .unwrap_or(0);

        let mut current_scene = None;
        let mut current_bvh = None;
        let mut current_lights = None;
        let mut scene_load_stats = None;
        let mut bvh_build_ms = 0u64;
        let mut bvh_triangle_count = 0usize;

        if !asset_files.is_empty() {
            if let Ok((scene, stats)) =
                crate::asset_importer::import_scene(&asset_files[selected_asset_index])
            {
                if !scene.cameras.is_empty() {
                    let first_cam = &scene.cameras[0];
                    camera.position = first_cam.position;
                    camera.target = first_cam.target;
                    camera.fov = first_cam.fov;
                }
                let scene_arc = std::sync::Arc::new(scene);
                let (bvh_arc, lights_arc, bms, tris) = build_bvh_and_lights(&scene_arc);
                scene_load_stats = Some(stats);
                bvh_build_ms = bms;
                bvh_triangle_count = tris;
                current_scene = Some(scene_arc);
                current_bvh = Some(bvh_arc);
                current_lights = Some(lights_arc);
            }
        }

        let scene = current_scene
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(empty_scene()));
        let (bvh, lights) = match (current_bvh.clone(), current_lights.clone()) {
            (Some(b), Some(l)) => (b, l),
            _ => {
                let (b, l, bms, tris) = build_bvh_and_lights(&scene);
                bvh_build_ms = bms;
                bvh_triangle_count = tris;
                (b, l)
            }
        };

        let strategy = rnpt::SamplingStrategy::Nirc;
        let config = rnpt::PathTracerConfig {
            width,
            height,
            camera: camera.clone(),
            scene,
            bvh,
            lights,
            env: None,
            strategy,
            nirc_network: None,
        };

        Self {
            camera,
            resolution: [width, height],
            exposure: 1.0,
            tonemapper: TonemapOperator::Aces,
            tracer: Some(rnpt::ParallelTracer::new(config)),
            last_fps_time: Instant::now(),
            rays_since_last_fps: 0,
            real_rays_since_last_fps: 0,
            shadow_rays_since_last_fps: 0,
            local_pixels: vec![rnpt::Pixel::default(); width * height],
            local_width: width,
            local_height: height,
            local_rays_per_sec: 0.0,
            local_real_rays_per_sec: 0.0,
            local_shadow_rays_per_sec: 0.0,
            texture_handle: None,
            last_exposure: 1.0,
            last_tonemapper: TonemapOperator::Aces,
            asset_files,
            selected_asset_index,
            current_scene,
            current_bvh,
            current_lights,
            auto_fit: true,
            resize_timeout: None,
            strategy,
            selected_camera_index: 0,
            hdr_files: crate::asset_importer::list_hdris("assets"),
            selected_env: 0,
            env: None,
            env_raw: None,
            env_intensity: 1.0,
            env_rotation: 0.0,
            scene_load_stats,
            bvh_build_ms,
            bvh_triangle_count,
            show_debug: false,
            frame_count: 0,
            last_loss: 0.0,
            loss_ema: 0.0,
            rel_error: 0.0,
            probe_texture: None,
        }
    }

    pub(crate) fn trigger_reset(&mut self) {
        let scene = self
            .current_scene
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(empty_scene()));
        let (bvh, lights) = match (self.current_bvh.clone(), self.current_lights.clone()) {
            (Some(b), Some(l)) => (b, l),
            _ => {
                let (b, l, bms, tris) = build_bvh_and_lights(&scene);
                self.bvh_build_ms = bms;
                self.bvh_triangle_count = tris;
                (b, l)
            }
        };
        let (lights, env) = with_env(&lights, &self.env);

        let config = rnpt::PathTracerConfig {
            width: self.resolution[0],
            height: self.resolution[1],
            camera: self.camera.clone(),
            scene,
            bvh,
            lights,
            env,
            strategy: self.strategy,
            nirc_network: None,
        };

        self.tracer = None;
        self.tracer = Some(rnpt::ParallelTracer::new(config));
        self.local_width = self.resolution[0];
        self.local_height = self.resolution[1];
        self.local_pixels = vec![rnpt::Pixel::default(); self.local_width * self.local_height];
        self.frame_count = 0;
        self.last_loss = 0.0;
        self.loss_ema = 0.0;
        self.rel_error = 0.0;
        self.rays_since_last_fps = 0;
        self.real_rays_since_last_fps = 0;
        self.shadow_rays_since_last_fps = 0;
        self.local_rays_per_sec = 0.0;
        self.local_real_rays_per_sec = 0.0;
        self.local_shadow_rays_per_sec = 0.0;
    }

    pub(crate) fn load_env(&mut self) {
        if self.selected_env == 0 {
            self.env = None;
            self.env_raw = None;
            return;
        }
        let path = self.hdr_files[self.selected_env - 1].clone();
        self.env_raw = crate::asset_importer::load_hdr(&path);
        self.rebuild_env();
    }

    pub(crate) fn rebuild_env(&mut self) {
        let rot_rad = self.env_rotation.to_radians();
        self.env = self.env_raw.as_ref().map(|(pixels, w, h)| {
            std::sync::Arc::new(rnpt::EnvLight::new(
                pixels.clone(),
                *w,
                *h,
                self.env_intensity,
                rot_rad,
            ))
        });
    }

    pub(crate) fn update_probe(&mut self, ctx: &egui::Context, px: f32, py: f32) {
        const W: usize = 256;
        const H: usize = 128;
        let Some(tracer) = &self.tracer else { return };
        let Some(rgb) = tracer.render_nirc_probe(px, py, W, H) else { return };

        let exposure = self.exposure;
        let tonemapper = self.tonemapper;
        let raw: Vec<u8> = rgb
            .iter()
            .flat_map(|[r, g, b]| {
                let tonemap = |v: f32| -> u8 {
                    let v = v * exposure;
                    let v = match tonemapper {
                        TonemapOperator::Reinhard => v / (v + 1.0),
                        TonemapOperator::Aces => {
                            let (a, b2, c, d, e) = (2.51f32, 0.03f32, 2.43f32, 0.59f32, 0.14f32);
                            (v * (a * v + b2)) / (v * (c * v + d) + e)
                        }
                    };
                    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
                };
                [tonemap(*r), tonemap(*g), tonemap(*b), 255u8]
            })
            .collect();

        let img = egui::ColorImage::from_rgba_unmultiplied([W, H], &raw);
        if let Some(ref mut t) = self.probe_texture {
            t.set(img, egui::TextureOptions::LINEAR);
        } else {
            self.probe_texture =
                Some(ctx.load_texture("nirc_probe", img, egui::TextureOptions::LINEAR));
        }
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for RnptGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Deferred resize: apply after a short debounce to avoid thrashing.
        if let Some(timeout) = self.resize_timeout {
            if Instant::now() >= timeout {
                self.resize_timeout = None;
                self.trigger_reset();
            }
        }

        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.trigger_reset();
        }

        // Fetch pixels, train NIRC, update FPS counters.
        if let Some(tracer) = &self.tracer {
            if let Some(loss) = tracer.train_nirc() {
                self.last_loss = loss;
                self.loss_ema = if self.loss_ema == 0.0 {
                    loss
                } else {
                    self.loss_ema * 0.95 + loss * 0.05
                };
            }
            if let Some(rel) = tracer.nirc_rel_error() {
                self.rel_error = if self.rel_error == 0.0 {
                    rel
                } else {
                    self.rel_error * 0.95 + rel * 0.05
                };
            }

            tracer.fetch_pixels(&mut self.local_pixels);
            self.rays_since_last_fps += tracer.pop_rays_traced();
            self.real_rays_since_last_fps += tracer.pop_real_rays_traced();
            self.shadow_rays_since_last_fps += tracer.pop_shadow_rays_traced();
            self.frame_count += 1;

            let now = Instant::now();
            let elapsed = now.duration_since(self.last_fps_time).as_secs_f64();
            if elapsed >= 0.5 {
                self.local_rays_per_sec = self.rays_since_last_fps as f64 / elapsed;
                self.local_real_rays_per_sec = self.real_rays_since_last_fps as f64 / elapsed;
                self.local_shadow_rays_per_sec = self.shadow_rays_since_last_fps as f64 / elapsed;
                self.rays_since_last_fps = 0;
                self.real_rays_since_last_fps = 0;
                self.shadow_rays_since_last_fps = 0;
                self.last_fps_time = now;
            }
        }

        ctx.request_repaint();

        // Regenerate the display texture when pixels or post-process settings change.
        let exposure_changed = self.exposure != self.last_exposure;
        let tonemapper_changed = self.tonemapper != self.last_tonemapper;
        if exposure_changed || tonemapper_changed || self.texture_handle.is_none() {
            let mut raw_rgba = vec![0u8; self.local_width * self.local_height * 4];
            tonemap_and_convert(&self.local_pixels, self.exposure, self.tonemapper, &mut raw_rgba);
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [self.local_width, self.local_height],
                &raw_rgba,
            );
            if let Some(ref mut t) = self.texture_handle {
                t.set(img, egui::TextureOptions::LINEAR);
            } else {
                self.texture_handle =
                    Some(ctx.load_texture("render_texture", img, egui::TextureOptions::LINEAR));
            }
            self.last_exposure = self.exposure;
            self.last_tonemapper = self.tonemapper;
        } else if self.tracer.is_some() {
            // Pixels changed but post-process is unchanged — update texture in place.
            let mut raw_rgba = vec![0u8; self.local_width * self.local_height * 4];
            tonemap_and_convert(&self.local_pixels, self.exposure, self.tonemapper, &mut raw_rgba);
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [self.local_width, self.local_height],
                &raw_rgba,
            );
            if let Some(ref mut t) = self.texture_handle {
                t.set(img, egui::TextureOptions::LINEAR);
            }
        }

        // ── Left panel ────────────────────────────────────────────────────────
        egui::SidePanel::left("controls_panel")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("RNPT Controls");
                ui.add_space(10.0);

                panels::scene::show(self, ui);
                panels::camera::show(self, ui);
                panels::render::show(self, ui);
                panels::environment::show(self, ui);
                panels::post::show(self, ui);
                panels::stats::show(self, ui);
            });

        // ── Central viewport ──────────────────────────────────────────────────
        let nav = crate::viewport::show(self, ctx);

        if nav.changed {
            crate::camera_nav::orbit(&mut self.camera, nav.orbit[0], nav.orbit[1]);
            crate::camera_nav::pan(&mut self.camera, nav.pan[0], nav.pan[1]);
            crate::camera_nav::dolly(&mut self.camera, nav.scroll);
            self.trigger_reset();
        }

        // Auto-fit: resize render resolution to match viewport.
        if self.auto_fit {
            if let Some(size) = ctx.input(|i| {
                i.viewport().inner_rect.map(|r| [r.width(), r.height()])
            }) {
                // Subtract the side panel width (approximate — egui reports final layout next frame).
                let w = (size[0] - 270.0).max(64.0) as usize;
                let h = size[1].max(64.0) as usize;
                if w != self.resolution[0] || h != self.resolution[1] {
                    self.resolution = [w, h];
                    self.resize_timeout =
                        Some(Instant::now() + Duration::from_millis(150));
                }
            }
        }
    }
}
